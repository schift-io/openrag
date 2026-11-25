//! Text embedder backed by an OpenAI-compatible `/v1/embeddings` endpoint.
//!
//! Selected when `OPENRAG_TEXT_EMBED_URL` is set; otherwise the deterministic hash embedder
//! stays in place. Vectors are truncated to the schema's 256 dimensions and L2-renormalized
//! client-side, which is how Matryoshka (MRL) models such as `google/embeddinggemma-2` are
//! meant to be shortened, so it works against any server whether or not it honors
//! `dimensions`.
//!
//! | variable | meaning |
//! |---|---|
//! | `OPENRAG_TEXT_EMBED_URL` | full endpoint URL, e.g. `http://127.0.0.1:8090/v1/embeddings` |
//! | `OPENRAG_TEXT_EMBED_MODEL` | model id sent as `model` (required with the URL) |
//! | `OPENRAG_TEXT_EMBED_API_KEY` | optional bearer token |
//! | `OPENRAG_TEXT_EMBED_QUERY_PREFIX` | prefix for search queries (asymmetric models) |
//! | `OPENRAG_TEXT_EMBED_DOCUMENT_PREFIX` | prefix for stored chunks, pages, wiki and memory text |
//!
//! Prefixes default to the task instructions published on the model card for known models.
//!
//! PII is masked only toward AI. A server outside this network (see [`is_internal_endpoint`]) is
//! an outside AI service, so document text sent to it has detected PII replaced with
//! `[PII:<TAG>]` first. Queries go as typed, like names typed into a question for an LLM.

use std::net::{IpAddr, ToSocketAddrs};
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use openrag_core::OpenRagError;
use openrag_memory::TextEmbedder;
use openrag_pii::PiiEngine;
use openrag_store::{l2_normalize, VECTOR_DIM};
use serde_json::{json, Value};
use tokio::sync::RwLock;

#[derive(Debug, Clone)]
pub struct HttpTextEmbedder {
    client: reqwest::Client,
    url: String,
    model: String,
    api_key: Option<String>,
    query_prefix: String,
    document_prefix: String,
    label: String,
    /// Set when document text is masked before it is sent (server outside this network).
    pii_mask: Option<Arc<RwLock<PiiEngine>>>,
}

/// Whether an embeddings URL points inside this network: a loopback, private, link-local or
/// shared (100.64/10) address, a single-label host (a compose service such as `local-embed`), a
/// `.localhost` / `.local` / `.internal` / `.lan` / `.home.arpa` name, or a name that resolves only
/// to such addresses. Anything else, including a name that does not resolve, is outside.
pub fn is_internal_endpoint(url: &str) -> bool {
    let Ok(u) = reqwest::Url::parse(url) else {
        return false;
    };
    let Some(host) = u.host_str() else {
        return false;
    };
    let host = host.trim_start_matches('[').trim_end_matches(']').trim_end_matches('.').to_ascii_lowercase();
    if let Ok(ip) = host.parse::<IpAddr>() {
        return internal_ip(ip);
    }
    if host == "localhost" || !host.contains('.') {
        return true;
    }
    const SUFFIXES: [&str; 5] = [".localhost", ".local", ".internal", ".lan", ".home.arpa"];
    if SUFFIXES.iter().any(|s| host.ends_with(s)) {
        return true;
    }
    let port = u.port_or_known_default().unwrap_or(443);
    match (host.as_str(), port).to_socket_addrs() {
        Ok(addrs) => {
            let addrs: Vec<_> = addrs.collect();
            !addrs.is_empty() && addrs.iter().all(|a| internal_ip(a.ip()))
        }
        Err(_) => false,
    }
}

fn internal_ip(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(v4) => {
            let [a, b, ..] = v4.octets();
            v4.is_loopback() || v4.is_private() || v4.is_link_local() || (a == 100 && (b & 0xC0) == 64)
        }
        IpAddr::V6(v6) => {
            let first = v6.segments()[0];
            v6.is_loopback()
                || (first & 0xfe00) == 0xfc00
                || (first & 0xffc0) == 0xfe80
                || v6.to_ipv4_mapped().is_some_and(|v4| internal_ip(IpAddr::V4(v4)))
        }
    }
}

/// Replaces detected PII with `[PII:<TAG>]`; of overlapping spans the earliest (then longest) wins.
fn mask_pii(engine: &PiiEngine, text: &str) -> String {
    let mut spans = engine.detect(text);
    spans.sort_by_key(|s| (s.start_byte, std::cmp::Reverse(s.end_byte)));
    let mut out = String::with_capacity(text.len());
    let mut at = 0usize;
    for s in spans {
        if s.start_byte < at
            || s.end_byte <= s.start_byte
            || !text.is_char_boundary(s.start_byte)
            || !text.is_char_boundary(s.end_byte)
        {
            continue;
        }
        out.push_str(&text[at..s.start_byte]);
        out.push_str("[PII:");
        out.push_str(s.category.token_tag());
        out.push(']');
        at = s.end_byte;
    }
    out.push_str(&text[at..]);
    out
}

/// Task prefixes from the model card (config_sentence_transformers.json, `query` / `document`).
fn known_prefixes(model: &str) -> Option<(&'static str, &'static str)> {
    let m = model.to_ascii_lowercase();
    if m.starts_with("google/embeddinggemma") {
        return Some(("task: search result | query: ", "title: {title} | text: "));
    }
    None
}

const MAX_TITLE_CHARS: usize = 120;

/// Normalizes a document title for embedding prompts: trims, collapses whitespace runs to a
/// single space, replaces `|` with `/` so it cannot break `title: … | text: ` delimiters, and
/// truncates to at most 120 Unicode scalar values.
fn clean_title(raw: &str) -> Option<String> {
    let mut cleaned = String::with_capacity(raw.len().min(MAX_TITLE_CHARS * 4));
    let mut char_count = 0usize;
    let mut pending_space = false;
    for ch in raw.trim().chars() {
        if ch.is_whitespace() {
            pending_space = true;
            continue;
        }
        if pending_space {
            if char_count >= MAX_TITLE_CHARS {
                break;
            }
            cleaned.push(' ');
            char_count += 1;
            pending_space = false;
        }
        if char_count >= MAX_TITLE_CHARS {
            break;
        }
        cleaned.push(if ch == '|' { '/' } else { ch });
        char_count += 1;
    }
    let cleaned = cleaned.trim_end();
    if cleaned.is_empty() {
        None
    } else {
        Some(cleaned.to_string())
    }
}

fn format_document_input(prefix: &str, title: Option<&str>, text: &str) -> String {
    if prefix.contains("{title}") {
        let rendered_prefix = prefix.replace("{title}", title.unwrap_or("none"));
        format!("{rendered_prefix}{text}")
    } else if let Some(title) = title {
        format!("{prefix}{title}\n{text}")
    } else {
        format!("{prefix}{text}")
    }
}

fn env_opt(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|v| !v.trim().is_empty())
}

impl HttpTextEmbedder {
    pub fn new(
        url: impl Into<String>,
        model: impl Into<String>,
        api_key: Option<String>,
        query_prefix: Option<String>,
        document_prefix: Option<String>,
    ) -> Result<Self, OpenRagError> {
        let url = url.into().trim().to_string();
        let model = model.into().trim().to_string();
        if !(url.starts_with("http://") || url.starts_with("https://")) {
            return Err(OpenRagError::Config(format!(
                "OPENRAG_TEXT_EMBED_URL must be an http(s) URL, got '{url}'"
            )));
        }
        if model.is_empty() {
            return Err(OpenRagError::Config(
                "OPENRAG_TEXT_EMBED_MODEL is required when OPENRAG_TEXT_EMBED_URL is set".into(),
            ));
        }
        let (q_default, d_default) = known_prefixes(&model).unwrap_or(("", ""));
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(60))
            .build()
            .map_err(|e| OpenRagError::Config(format!("text embedder http client: {e}")))?;
        Ok(Self {
            client,
            label: format!("{model}@{VECTOR_DIM}d"),
            url,
            model,
            api_key,
            query_prefix: query_prefix.unwrap_or_else(|| q_default.to_string()),
            document_prefix: document_prefix.unwrap_or_else(|| d_default.to_string()),
            pii_mask: None,
        })
    }

    /// Masks PII in document text before sending it when the server is outside this network.
    pub fn mask_pii_if_external(self, engine: Arc<RwLock<PiiEngine>>) -> Self {
        if is_internal_endpoint(&self.url) {
            return self;
        }
        tracing::info!(
            url = %self.url,
            "embeddings server is outside this network: PII in document text is masked before it is sent"
        );
        self.with_pii_masking(engine)
    }

    fn with_pii_masking(mut self, engine: Arc<RwLock<PiiEngine>>) -> Self {
        self.pii_mask = Some(engine);
        self
    }

    pub fn masks_pii(&self) -> bool {
        self.pii_mask.is_some()
    }

    /// `Ok(None)` when `OPENRAG_TEXT_EMBED_URL` is unset.
    pub fn from_env() -> Result<Option<Self>, OpenRagError> {
        let Some(url) = env_opt("OPENRAG_TEXT_EMBED_URL") else {
            return Ok(None);
        };
        Self::new(
            url,
            env_opt("OPENRAG_TEXT_EMBED_MODEL").unwrap_or_default(),
            env_opt("OPENRAG_TEXT_EMBED_API_KEY"),
            // Empty means unset (compose passes the variables through), so the model's own
            // prefixes apply.
            std::env::var("OPENRAG_TEXT_EMBED_QUERY_PREFIX").ok().filter(|p| !p.is_empty()),
            std::env::var("OPENRAG_TEXT_EMBED_DOCUMENT_PREFIX").ok().filter(|p| !p.is_empty()),
        )
        .map(Some)
    }

    async fn embed_input(&self, input: String) -> Result<Vec<f32>, OpenRagError> {
        let mut req = self.client.post(&self.url).json(&json!({
            "model": self.model,
            "input": [input],
            "dimensions": VECTOR_DIM,
        }));
        if let Some(key) = &self.api_key {
            req = req.bearer_auth(key);
        }
        let resp = req
            .send()
            .await
            .map_err(|e| OpenRagError::Internal(format!("text embedder request failed: {e}")))?;
        let status = resp.status();
        let body: Value = resp
            .json()
            .await
            .map_err(|e| OpenRagError::Internal(format!("text embedder returned invalid JSON: {e}")))?;
        if !status.is_success() {
            let msg = body.pointer("/error/message").and_then(Value::as_str).unwrap_or("");
            return Err(OpenRagError::Internal(format!("text embedder returned {status}: {msg}")));
        }
        parse_embedding(&body)
    }
}

/// Reads `data[0].embedding`, keeps the leading 256 dimensions and renormalizes.
pub fn parse_embedding(body: &Value) -> Result<Vec<f32>, OpenRagError> {
    let raw = body
        .pointer("/data/0/embedding")
        .and_then(Value::as_array)
        .ok_or_else(|| OpenRagError::Internal("text embedder response has no data[0].embedding".into()))?;
    if raw.len() < VECTOR_DIM {
        return Err(OpenRagError::Internal(format!(
            "text embedder returned {} dimensions, need at least {VECTOR_DIM}",
            raw.len()
        )));
    }
    let mut out = Vec::with_capacity(VECTOR_DIM);
    for v in raw.iter().take(VECTOR_DIM) {
        let f = v
            .as_f64()
            .filter(|f| f.is_finite())
            .ok_or_else(|| OpenRagError::Internal("text embedder returned a non-finite value".into()))?;
        out.push(f as f32);
    }
    l2_normalize(&out)
}

#[async_trait]
impl TextEmbedder for HttpTextEmbedder {
    fn model_name(&self) -> &str {
        &self.label
    }

    async fn embed_text(&self, text: &str) -> Result<Vec<f32>, OpenRagError> {
        self.embed_document(None, text).await
    }

    async fn embed_document(
        &self,
        title: Option<&str>,
        text: &str,
    ) -> Result<Vec<f32>, OpenRagError> {
        let (masked_title, masked_text);
        let (raw_title, body) = match &self.pii_mask {
            Some(engine) => {
                // Mask before truncating the title so a PII span straddling the cut cannot leak.
                let guard = engine.read().await;
                masked_title = title
                    .filter(|t| !t.trim().is_empty())
                    .map(|t| mask_pii(&guard, t));
                masked_text = mask_pii(&guard, text);
                (masked_title.as_deref(), masked_text.as_str())
            }
            None => (title, text),
        };
        let cleaned_title = raw_title.and_then(clean_title);
        let input = format_document_input(&self.document_prefix, cleaned_title.as_deref(), body);
        self.embed_input(input).await
    }

    async fn embed_query(&self, text: &str) -> Result<Vec<f32>, OpenRagError> {
        self.embed_input(format!("{}{text}", self.query_prefix)).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn truncates_to_schema_dim_and_renormalizes() -> Result<(), OpenRagError> {
        let full: Vec<f64> = (0..768).map(|i| if i < VECTOR_DIM { 1.0 } else { 5.0 }).collect();
        let v = parse_embedding(&json!({ "data": [{ "embedding": full }] }))?;
        assert_eq!(v.len(), VECTOR_DIM);
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5);
        assert!((v[0] - v[127]).abs() < 1e-6, "only the leading dimensions are kept");
        Ok(())
    }

    #[test]
    fn rejects_short_or_missing_vectors() {
        assert!(parse_embedding(&json!({ "data": [{ "embedding": [0.1, 0.2] }] })).is_err());
        assert!(parse_embedding(&json!({ "data": [] })).is_err());
        assert!(parse_embedding(&json!({ "data": [{ "embedding": [null] }] })).is_err());
    }

    #[test]
    fn known_model_gets_card_prefixes_and_env_overrides_them() -> Result<(), OpenRagError> {
        let e = HttpTextEmbedder::new("http://127.0.0.1:1/v1/embeddings", "google/embeddinggemma-2", None, None, None)?;
        assert_eq!(e.query_prefix, "task: search result | query: ");
        assert_eq!(e.document_prefix, "title: {title} | text: ");
        assert_eq!(e.model_name(), "google/embeddinggemma-2@256d");
        let o = HttpTextEmbedder::new("http://x/v1/embeddings", "google/embeddinggemma-2", None, Some(String::new()), None)?;
        assert_eq!(o.query_prefix, "");
        assert!(HttpTextEmbedder::new("ftp://x", "m", None, None, None).is_err());
        assert!(HttpTextEmbedder::new("http://x", " ", None, None, None).is_err());
        Ok(())
    }

    #[tokio::test]
    async fn sends_role_prefix_model_dimensions_and_bearer() -> Result<(), Box<dyn std::error::Error>> {
        use std::sync::{Arc, Mutex};
        type Seen = Arc<Mutex<Vec<(Option<String>, Value)>>>;
        let seen: Seen = Arc::default();
        let rec = Arc::clone(&seen);
        let app = axum::Router::new().route(
            "/v1/embeddings",
            axum::routing::post(move |headers: axum::http::HeaderMap, axum::Json(body): axum::Json<Value>| {
                let rec = Arc::clone(&rec);
                async move {
                    let auth = headers
                        .get("authorization")
                        .and_then(|v| v.to_str().ok())
                        .map(str::to_string);
                    if let Ok(mut g) = rec.lock() {
                        g.push((auth, body));
                    }
                    axum::Json(json!({ "data": [{ "embedding": vec![0.5_f32; 768] }] }))
                }
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
        let addr = listener.local_addr()?;
        tokio::spawn(async move { axum::serve(listener, app).await });

        let e = HttpTextEmbedder::new(
            format!("http://{addr}/v1/embeddings"),
            "google/embeddinggemma-2",
            Some("k-123".into()),
            None,
            None,
        )?;
        assert_eq!(e.embed_query("출장비 한도").await?.len(), VECTOR_DIM);
        assert_eq!(e.embed_text("출장비는 1일 75,000원").await?.len(), VECTOR_DIM);
        let got = seen.lock().map(|g| g.clone()).unwrap_or_default();
        assert_eq!(got.len(), 2);
        assert_eq!(got[0].0.as_deref(), Some("Bearer k-123"));
        assert_eq!(got[0].1["model"], "google/embeddinggemma-2");
        assert_eq!(got[0].1["dimensions"], VECTOR_DIM);
        assert_eq!(got[0].1["input"][0], "task: search result | query: 출장비 한도");
        assert_eq!(got[1].1["input"][0], "title: none | text: 출장비는 1일 75,000원");
        Ok(())
    }

    #[test]
    fn internal_endpoints_are_told_apart() {
        for url in [
            "http://local-embed:8082/v1/embeddings",
            "http://localhost:8082/v1/embeddings",
            "http://127.0.0.1:8090/v1/embeddings",
            "http://10.0.0.5/v1/embeddings",
            "http://172.16.3.4/v1/embeddings",
            "http://192.168.1.20/v1/embeddings",
            "http://100.64.0.9/v1/embeddings",
            "http://[::1]:8082/v1/embeddings",
            "http://[fd00::12]/v1/embeddings",
            "http://embed.local/v1/embeddings",
            "https://vllm.corp.internal/v1/embeddings",
        ] {
            assert!(is_internal_endpoint(url), "{url}");
        }
        for url in [
            "https://api.openai.com/v1/embeddings",
            "http://8.8.8.8/v1/embeddings",
            "http://[2001:4860:4860::8888]/v1/embeddings",
            "https://does-not-resolve.invalid/v1/embeddings",
            "not a url",
        ] {
            assert!(!is_internal_endpoint(url), "{url}");
        }
        let engine = Arc::new(RwLock::new(PiiEngine::default()));
        let local = HttpTextEmbedder::new("http://local-embed:8082/v1/embeddings", "m", None, None, None);
        assert!(local.is_ok_and(|e| !e.mask_pii_if_external(Arc::clone(&engine)).masks_pii()));
        let outside = HttpTextEmbedder::new("http://8.8.8.8/v1/embeddings", "m", None, None, None);
        assert!(outside.is_ok_and(|e| e.mask_pii_if_external(engine).masks_pii()));
    }

    #[tokio::test]
    async fn masked_embedder_sends_documents_without_pii_and_queries_as_typed(
    ) -> Result<(), Box<dyn std::error::Error>> {
        use std::sync::Mutex;
        let seen: Arc<Mutex<Vec<Value>>> = Arc::default();
        let rec = Arc::clone(&seen);
        let app = axum::Router::new().route(
            "/v1/embeddings",
            axum::routing::post(move |axum::Json(body): axum::Json<Value>| {
                let rec = Arc::clone(&rec);
                async move {
                    if let Ok(mut g) = rec.lock() {
                        g.push(body);
                    }
                    axum::Json(json!({ "data": [{ "embedding": vec![0.5_f32; 768] }] }))
                }
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
        let addr = listener.local_addr()?;
        tokio::spawn(async move { axum::serve(listener, app).await });

        let e = HttpTextEmbedder::new(format!("http://{addr}/v1/embeddings"), "m", None, None, None)?
            .with_pii_masking(Arc::new(RwLock::new(PiiEngine::default())));
        e.embed_text("연락처 010-2345-6789, 결제 계좌 123456-01-234567, 메일 a.kim@corp.example").await?;
        e.embed_query("010-2345-6789 누구 번호야?").await?;
        let got = seen.lock().map(|g| g.clone()).unwrap_or_default();
        let doc = got[0]["input"][0].as_str().unwrap_or_default();
        for raw in ["010-2345-6789", "123456-01-234567", "a.kim@corp.example"] {
            assert!(!doc.contains(raw), "{raw} leaked: {doc}");
        }
        assert!(doc.contains("[PII:PHONE]") && doc.contains("[PII:BANK]") && doc.contains("[PII:EMAIL]"), "{doc}");
        assert_eq!(got[1]["input"][0], "010-2345-6789 누구 번호야?");
        Ok(())
    }

    #[tokio::test]
    async fn upstream_error_fails_closed() -> Result<(), Box<dyn std::error::Error>> {
        let app = axum::Router::new().route(
            "/v1/embeddings",
            axum::routing::post(|| async {
                (
                    axum::http::StatusCode::SERVICE_UNAVAILABLE,
                    axum::Json(json!({ "error": { "message": "model loading" } })),
                )
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
        let addr = listener.local_addr()?;
        tokio::spawn(async move { axum::serve(listener, app).await });
        let e = HttpTextEmbedder::new(format!("http://{addr}/v1/embeddings"), "m", None, None, None)?;
        let err = e.embed_text("x").await.err().map(|e| e.to_string()).unwrap_or_default();
        assert!(err.contains("503") && err.contains("model loading"), "{err}");
        Ok(())
    }

    async fn start_mock_embed_server() -> Result<(String, Arc<std::sync::Mutex<Vec<Value>>>), Box<dyn std::error::Error>> {
        let seen: Arc<std::sync::Mutex<Vec<Value>>> = Arc::default();
        let rec = Arc::clone(&seen);
        let app = axum::Router::new().route(
            "/v1/embeddings",
            axum::routing::post(move |axum::Json(body): axum::Json<Value>| {
                let rec = Arc::clone(&rec);
                async move {
                    if let Ok(mut g) = rec.lock() {
                        g.push(body);
                    }
                    axum::Json(json!({ "data": [{ "embedding": vec![0.5_f32; 768] }] }))
                }
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
        let addr = listener.local_addr()?;
        tokio::spawn(async move { axum::serve(listener, app).await });
        Ok((format!("http://{addr}/v1/embeddings"), seen))
    }

    #[tokio::test]
    async fn embed_document_substitutes_title_and_cleans_delimiters() -> Result<(), Box<dyn std::error::Error>> {
        let (url, seen) = start_mock_embed_server().await?;
        let e = HttpTextEmbedder::new(url, "google/embeddinggemma-2", None, None, None)?;

        e.embed_document(Some("출장비 규정"), "일비 75,000원").await?;
        e.embed_document(Some("  2026년\n출장비\t규정 | 개정판  "), "본문").await?;
        let long_title = "가".repeat(150);
        e.embed_document(Some(&long_title), "본문").await?;
        e.embed_document(None, "제목 없음").await?;
        e.embed_document(Some("   \n\t "), "빈 제목").await?;

        let got = seen.lock().map(|g| g.clone()).unwrap_or_default();
        assert_eq!(got.len(), 5);
        assert_eq!(got[0]["input"][0], "title: 출장비 규정 | text: 일비 75,000원");
        assert_eq!(got[1]["input"][0], "title: 2026년 출장비 규정 / 개정판 | text: 본문");
        let expected_long = format!("title: {} | text: 본문", "가".repeat(120));
        assert_eq!(got[2]["input"][0], expected_long);
        assert_eq!(got[3]["input"][0], "title: none | text: 제목 없음");
        assert_eq!(got[4]["input"][0], "title: none | text: 빈 제목");
        Ok(())
    }

    #[tokio::test]
    async fn embed_document_masks_pii_in_title_when_enabled() -> Result<(), Box<dyn std::error::Error>> {
        let (url, seen) = start_mock_embed_server().await?;
        let e = HttpTextEmbedder::new(url, "google/embeddinggemma-2", None, None, None)?
            .with_pii_masking(Arc::new(RwLock::new(PiiEngine::default())));

        e.embed_document(Some("고객 010-2345-6789 상담록"), "연락처 010-9876-5432").await?;

        let got = seen.lock().map(|g| g.clone()).unwrap_or_default();
        assert_eq!(got.len(), 1);
        let input = got[0]["input"][0].as_str().unwrap_or_default();
        assert!(!input.contains("010-2345-6789"), "title phone leaked: {input}");
        assert!(!input.contains("010-9876-5432"), "body phone leaked: {input}");
        assert_eq!(input, "title: 고객 [PII:PHONE] 상담록 | text: 연락처 [PII:PHONE]");
        Ok(())
    }

    #[tokio::test]
    async fn embed_document_prepends_title_for_custom_prefix_without_placeholder(
    ) -> Result<(), Box<dyn std::error::Error>> {
        let (url, seen) = start_mock_embed_server().await?;
        let e = HttpTextEmbedder::new(
            url,
            "intfloat/multilingual-e5-large",
            None,
            Some("query: ".into()),
            Some("passage: ".into()),
        )?;

        e.embed_document(Some("출장비 규정"), "일비 75,000원").await?;
        e.embed_document(None, "제목 없음").await?;
        e.embed_document(Some("   "), "빈 제목").await?;

        let got = seen.lock().map(|g| g.clone()).unwrap_or_default();
        assert_eq!(got.len(), 3);
        assert_eq!(got[0]["input"][0], "passage: 출장비 규정\n일비 75,000원");
        assert_eq!(got[1]["input"][0], "passage: 제목 없음");
        assert_eq!(got[2]["input"][0], "passage: 빈 제목");
        Ok(())
    }
}
