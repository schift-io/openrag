use std::fmt;
use std::str::FromStr;

use async_trait::async_trait;
use openrag_core::OpenRagError;
use openrag_db::RlsTransaction;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;
use uuid::Uuid;

use crate::tags::TagFilter;

/// Dimension of every stored vector space: text (chunks, wiki pages, memory entries), visual
/// (page thumbnails, figures) and concepts. Vectors from a larger Matryoshka model are truncated
/// to this and L2-renormalized.
pub const VECTOR_DIM: usize = 256;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum VectorTable {
    CorpusChunks,
    CorpusPages,
    CorpusPageFigures,
    WikiPages,
    MemoryEntries,
}

impl VectorTable {
    pub const ALL: [Self; 5] = [
        Self::CorpusChunks,
        Self::CorpusPages,
        Self::CorpusPageFigures,
        Self::WikiPages,
        Self::MemoryEntries,
    ];

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::CorpusChunks => "corpus.chunks",
            Self::CorpusPages => "corpus.pages",
            Self::CorpusPageFigures => "corpus.page_figures",
            Self::WikiPages => "wiki.pages",
            Self::MemoryEntries => "memory.entries",
        }
    }

    /// Column holding the table's text embedding. Page thumbnails and figures carry none
    /// (migration 012): they are found through their text and tags only.
    pub const fn vector_column(self) -> Option<&'static str> {
        match self {
            Self::CorpusChunks | Self::WikiPages | Self::MemoryEntries => Some("embedding"),
            Self::CorpusPages | Self::CorpusPageFigures => None,
        }
    }

    pub const fn has_vectors(self) -> bool {
        self.vector_column().is_some()
    }

    /// `vector_column`, or a validation error for a table without vectors.
    pub fn require_vector_column(self) -> Result<&'static str, OpenRagError> {
        self.vector_column().ok_or_else(|| {
            OpenRagError::Validation(format!("{} has no vector column", self.as_str()))
        })
    }

    pub const fn hnsw_index_name(self) -> Option<&'static str> {
        match self {
            Self::CorpusChunks => Some("idx_corpus_chunks_vec_hnsw"),
            Self::WikiPages => Some("idx_wiki_pages_vec_hnsw"),
            Self::MemoryEntries => Some("idx_memory_entries_vec_hnsw"),
            Self::CorpusPages | Self::CorpusPageFigures => None,
        }
    }

    pub const fn has_status_active(self) -> bool {
        matches!(self, Self::WikiPages | Self::MemoryEntries)
    }

    pub const fn has_tags_column(self) -> bool {
        matches!(
            self,
            Self::CorpusChunks | Self::WikiPages | Self::MemoryEntries
        )
    }

    pub const fn has_document_id(self) -> bool {
        matches!(
            self,
            Self::CorpusChunks | Self::CorpusPages | Self::CorpusPageFigures
        )
    }

    pub const fn has_bucket_id(self) -> bool {
        matches!(self, Self::WikiPages | Self::MemoryEntries)
    }
}

impl fmt::Display for VectorTable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for VectorTable {
    type Err = OpenRagError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim() {
            "corpus.chunks" => Ok(Self::CorpusChunks),
            "corpus.pages" => Ok(Self::CorpusPages),
            "corpus.page_figures" => Ok(Self::CorpusPageFigures),
            "wiki.pages" => Ok(Self::WikiPages),
            "memory.entries" => Ok(Self::MemoryEntries),
            other => Err(OpenRagError::Validation(format!(
                "unknown or non-vector table '{other}'"
            ))),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StoreMode {
    #[default]
    Offline,
    Live,
}

impl StoreMode {
    pub fn from_env() -> Self {
        match std::env::var("OPENRAG_MODE") {
            Ok(val) if val.trim().eq_ignore_ascii_case("live") => Self::Live,
            _ => Self::Offline,
        }
    }
}

pub fn validate_vector(vec: &[f32]) -> Result<(), OpenRagError> {
    if vec.len() != VECTOR_DIM {
        return Err(OpenRagError::Validation(format!(
            "dimension_mismatch: expected {VECTOR_DIM}d vector, got {}d",
            vec.len()
        )));
    }
    if vec.iter().any(|v| !v.is_finite()) {
        return Err(OpenRagError::Validation(
            "non_finite: vector contains NaN or Inf".into(),
        ));
    }
    if vec.iter().all(|v| *v == 0.0) {
        return Err(OpenRagError::Validation(
            "zero_vector: all-zero vector is not allowed".into(),
        ));
    }
    Ok(())
}

pub fn l2_normalize(vec: &[f32]) -> Result<Vec<f32>, OpenRagError> {
    validate_vector(vec)?;
    let norm: f32 = vec.iter().map(|v| v * v).sum::<f32>().sqrt();
    if norm == 0.0 || !norm.is_finite() {
        return Err(OpenRagError::Validation(
            "zero_vector: vector norm is zero or non-finite".into(),
        ));
    }
    Ok(vec.iter().map(|v| v / norm).collect())
}

pub fn cosine_similarity(a: &[f32], b: &[f32]) -> Result<f32, OpenRagError> {
    validate_vector(a)?;
    validate_vector(b)?;
    let mut dot = 0.0_f64;
    let mut norm_a = 0.0_f64;
    let mut norm_b = 0.0_f64;
    for (x, y) in a.iter().zip(b.iter()) {
        let xf = f64::from(*x);
        let yf = f64::from(*y);
        dot += xf * yf;
        norm_a += xf * xf;
        norm_b += yf * yf;
    }
    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom == 0.0 || !denom.is_finite() {
        return Ok(0.0);
    }
    Ok((dot / denom).clamp(-1.0, 1.0) as f32)
}

pub fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    hex::encode(digest)
}

/// RFC 2104 HMAC-SHA256 implemented over `sha2::Sha256`.
pub fn hmac_sha256(key: &[u8], data: &[u8]) -> [u8; 32] {
    const BLOCK_SIZE: usize = 64;
    let mut key_block = [0_u8; BLOCK_SIZE];
    if key.len() > BLOCK_SIZE {
        let hashed = Sha256::digest(key);
        key_block[..32].copy_from_slice(&hashed);
    } else {
        key_block[..key.len()].copy_from_slice(key);
    }

    let mut ipad = [0x36_u8; BLOCK_SIZE];
    let mut opad = [0x5c_u8; BLOCK_SIZE];
    for i in 0..BLOCK_SIZE {
        ipad[i] ^= key_block[i];
        opad[i] ^= key_block[i];
    }

    let mut inner = Sha256::new();
    inner.update(ipad);
    inner.update(data);
    let inner_hash = inner.finalize();

    let mut outer = Sha256::new();
    outer.update(opad);
    outer.update(inner_hash);
    let out = outer.finalize();
    let mut result = [0_u8; 32];
    result.copy_from_slice(&out);
    result
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VectorUpsertItem {
    pub id: Uuid,
    pub document_id: Option<Uuid>,
    pub acl_scope_key: String,
    pub vector: Vec<f32>,
    pub text: Option<String>,
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(default)]
    pub metadata: serde_json::Map<String, serde_json::Value>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VectorSearchQuery {
    pub org_id: Uuid,
    pub table: VectorTable,
    pub query_vector: Vec<f32>,
    #[serde(default)]
    pub acl_scope_key: Option<String>,
    #[serde(default)]
    pub bucket_id: Option<Uuid>,
    pub top_k: usize,
    #[serde(default)]
    pub tag_filter: Option<TagFilter>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VectorHit {
    pub id: Uuid,
    pub document_id: Option<Uuid>,
    pub table: VectorTable,
    pub cosine_sim: f32,
    pub distance: f32,
    pub acl_scope_key: String,
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(default)]
    pub metadata: serde_json::Map<String, serde_json::Value>,
}

#[async_trait]
pub trait VectorStoreAdapter: Send + Sync {
    async fn upsert(
        &self,
        org_id: Uuid,
        table: VectorTable,
        items: &[VectorUpsertItem],
    ) -> Result<(), OpenRagError>;

    async fn delete(
        &self,
        org_id: Uuid,
        table: VectorTable,
        ids: &[Uuid],
    ) -> Result<(), OpenRagError>;

    async fn search_rls(
        &self,
        tx: &mut RlsTransaction<'_>,
        query: &VectorSearchQuery,
    ) -> Result<Vec<VectorHit>, OpenRagError>;
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HttpRequest {
    pub method: String,
    pub url: String,
    pub headers: Vec<(String, String)>,
    pub body: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HttpResponse {
    pub status: u16,
    pub headers: Vec<(String, String)>,
    pub body: Vec<u8>,
}

#[async_trait]
pub trait HttpTransport: Send + Sync {
    async fn send(&self, req: HttpRequest) -> Result<HttpResponse, OpenRagError>;
}

/// Minimal real HTTP/1.1 transport over `tokio::net::TcpStream` for local/intranet `http://` endpoints.
#[derive(Debug, Default, Clone)]
pub struct TokioHttpTransport;

#[async_trait]
impl HttpTransport for TokioHttpTransport {
    async fn send(&self, req: HttpRequest) -> Result<HttpResponse, OpenRagError> {
        let rest = req.url.strip_prefix("http://").ok_or_else(|| {
            OpenRagError::ExternalAdapter {
                adapter: "http_transport".into(),
                message: format!(
                    "unsupported URL scheme in '{}' (TokioHttpTransport supports http://; inject TLS transport for https://)",
                    req.url
                ),
            }
        })?;
        let (host_port, path) = match rest.split_once('/') {
            Some((hp, p)) => (hp, format!("/{p}")),
            None => (rest, "/".to_string()),
        };
        let addr = if host_port.contains(':') {
            host_port.to_string()
        } else {
            format!("{host_port}:80")
        };

        let mut stream =
            TcpStream::connect(&addr)
                .await
                .map_err(|e| OpenRagError::ExternalAdapter {
                    adapter: "http_transport".into(),
                    message: format!("TCP connect to '{addr}' failed: {e}"),
                })?;

        let mut wire = format!(
            "{} {} HTTP/1.1\r\nHost: {}\r\nConnection: close\r\nContent-Length: {}\r\n",
            req.method,
            path,
            host_port,
            req.body.len()
        );
        for (k, v) in &req.headers {
            if !k.eq_ignore_ascii_case("host")
                && !k.eq_ignore_ascii_case("content-length")
                && !k.eq_ignore_ascii_case("connection")
            {
                wire.push_str(k);
                wire.push_str(": ");
                wire.push_str(v);
                wire.push_str("\r\n");
            }
        }
        wire.push_str("\r\n");

        stream
            .write_all(wire.as_bytes())
            .await
            .map_err(|e| OpenRagError::ExternalAdapter {
                adapter: "http_transport".into(),
                message: format!("failed to write HTTP headers: {e}"),
            })?;
        if !req.body.is_empty() {
            stream
                .write_all(&req.body)
                .await
                .map_err(|e| OpenRagError::ExternalAdapter {
                    adapter: "http_transport".into(),
                    message: format!("failed to write HTTP body: {e}"),
                })?;
        }

        let mut buf = Vec::new();
        stream
            .read_to_end(&mut buf)
            .await
            .map_err(|e| OpenRagError::ExternalAdapter {
                adapter: "http_transport".into(),
                message: format!("failed to read HTTP response: {e}"),
            })?;

        let split_pos = buf
            .windows(4)
            .position(|w| w == b"\r\n\r\n")
            .ok_or_else(|| OpenRagError::ExternalAdapter {
                adapter: "http_transport".into(),
                message: "malformed HTTP response (missing CRLFCRLF)".into(),
            })?;
        let head = String::from_utf8_lossy(&buf[..split_pos]);
        let body = buf[split_pos + 4..].to_vec();
        let mut lines = head.lines();
        let status_line = lines.next().unwrap_or("");
        let status: u16 = status_line
            .split_whitespace()
            .nth(1)
            .and_then(|s| s.parse().ok())
            .ok_or_else(|| OpenRagError::ExternalAdapter {
                adapter: "http_transport".into(),
                message: format!("invalid HTTP status line '{status_line}'"),
            })?;

        let mut headers = Vec::new();
        for line in lines {
            if let Some((k, v)) = line.split_once(':') {
                headers.push((k.trim().to_string(), v.trim().to_string()));
            }
        }

        Ok(HttpResponse {
            status,
            headers,
            body,
        })
    }
}
