use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use openrag_core::{ExternalMirrorBackend, OpenRagError};
use openrag_db::RlsTransaction;
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use sqlx::{Postgres, QueryBuilder, Row};
use tokio::sync::RwLock;
use uuid::Uuid;

use crate::tags::{
    expand_tag_filter_in_memory, expand_tag_filter_rls, push_tag_filter_predicates,
    ExpandedTagFilter, TagFilter,
};
use crate::types::{
    cosine_similarity, validate_vector, HttpRequest, HttpTransport, StoreMode, VectorHit,
    VectorSearchQuery, VectorStoreAdapter, VectorTable, VectorUpsertItem, VECTOR_DIM,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OssBackendKind {
    Qdrant,
    Faiss,
    Usearch,
}

impl OssBackendKind {
    pub fn from_mirror_backend(backend: ExternalMirrorBackend) -> Result<Self, OpenRagError> {
        match backend {
            ExternalMirrorBackend::Qdrant => Ok(Self::Qdrant),
            ExternalMirrorBackend::Faiss => Ok(Self::Faiss),
            ExternalMirrorBackend::Usearch => Ok(Self::Usearch),
            ExternalMirrorBackend::None | ExternalMirrorBackend::S3Vectors => {
                Err(OpenRagError::Validation(format!(
                    "backend '{backend:?}' is not an OSS vector index backend"
                )))
            }
        }
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Qdrant => "qdrant",
            Self::Faiss => "faiss",
            Self::Usearch => "usearch",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OssVectorPoint {
    pub id: Uuid,
    pub org_id: Uuid,
    pub document_id: Option<Uuid>,
    pub acl_scope_key: String,
    pub vector: Vec<f32>,
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(default)]
    pub payload: Map<String, Value>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OssHit {
    pub id: Uuid,
    pub document_id: Option<Uuid>,
    pub cosine_sim: f32,
    pub distance: f32,
    pub acl_scope_key: String,
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(default)]
    pub payload: Map<String, Value>,
}

#[async_trait]
pub trait OssVectorBackend: Send + Sync {
    fn kind(&self) -> OssBackendKind;

    async fn ensure_collection(
        &self,
        org_id: Uuid,
        table: VectorTable,
        dimension: usize,
    ) -> Result<(), OpenRagError>;

    async fn upsert_points(
        &self,
        org_id: Uuid,
        table: VectorTable,
        points: &[OssVectorPoint],
    ) -> Result<(), OpenRagError>;

    async fn delete_points(
        &self,
        org_id: Uuid,
        table: VectorTable,
        ids: &[Uuid],
    ) -> Result<(), OpenRagError>;

    async fn search_points(
        &self,
        org_id: Uuid,
        table: VectorTable,
        vector: &[f32],
        top_k: usize,
        acl_scope_key: Option<&str>,
        tag_filter: Option<&TagFilter>,
    ) -> Result<Vec<OssHit>, OpenRagError>;

    async fn search_points_expanded(
        &self,
        org_id: Uuid,
        table: VectorTable,
        vector: &[f32],
        top_k: usize,
        acl_scope_key: Option<&str>,
        expanded_tags: Option<&ExpandedTagFilter>,
    ) -> Result<Vec<OssHit>, OpenRagError> {
        let tf_opt = expanded_tags.map(|ef| TagFilter {
            tags_any: ef.tags_any.clone(),
            tags_all: ef
                .tags_all_groups
                .iter()
                .filter_map(|g| g.first().cloned())
                .collect(),
            tags_exclude: ef.tags_exclude.clone(),
            expand_tag_aliases: false,
        });
        self.search_points(org_id, table, vector, top_k, acl_scope_key, tf_opt.as_ref())
            .await
    }
}

type OssCollectionsMap = HashMap<(Uuid, String), HashMap<Uuid, OssVectorPoint>>;

#[derive(Debug, Clone)]
pub struct InMemoryOssVectorStore {
    kind: OssBackendKind,
    collections: Arc<RwLock<OssCollectionsMap>>,
}

impl Default for InMemoryOssVectorStore {
    fn default() -> Self {
        Self::new(OssBackendKind::Qdrant)
    }
}

impl InMemoryOssVectorStore {
    pub fn new(kind: OssBackendKind) -> Self {
        Self {
            kind,
            collections: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    async fn search_points_internal(
        &self,
        org_id: Uuid,
        table: VectorTable,
        vector: &[f32],
        top_k: usize,
        acl_scope_key: Option<&str>,
        expanded_tags: Option<&ExpandedTagFilter>,
    ) -> Result<Vec<OssHit>, OpenRagError> {
        validate_vector(vector)?;
        if top_k == 0 {
            return Ok(Vec::new());
        }

        let guard = self.collections.read().await;
        let Some(map) = guard.get(&(org_id, table.as_str().to_string())) else {
            return Ok(Vec::new());
        };

        let mut hits = Vec::new();
        for pt in map.values() {
            if pt.org_id != org_id {
                continue;
            }
            if let Some(sk) = acl_scope_key {
                if pt.acl_scope_key != sk {
                    continue;
                }
            }
            if let Some(ef) = expanded_tags {
                if !ef.matches_tags(&pt.tags) {
                    continue;
                }
            }
            let sim = cosine_similarity(vector, &pt.vector)?;
            hits.push(OssHit {
                id: pt.id,
                document_id: pt.document_id,
                cosine_sim: sim,
                distance: 1.0 - sim,
                acl_scope_key: pt.acl_scope_key.clone(),
                tags: pt.tags.clone(),
                payload: pt.payload.clone(),
            });
        }
        hits.sort_by(|a, b| {
            b.cosine_sim
                .partial_cmp(&a.cosine_sim)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.id.cmp(&b.id))
        });
        hits.truncate(top_k);
        Ok(hits)
    }
}

#[async_trait]
impl OssVectorBackend for InMemoryOssVectorStore {
    fn kind(&self) -> OssBackendKind {
        self.kind
    }

    async fn ensure_collection(
        &self,
        org_id: Uuid,
        table: VectorTable,
        dimension: usize,
    ) -> Result<(), OpenRagError> {
        if dimension != VECTOR_DIM {
            return Err(OpenRagError::Validation(format!(
                "dimension_mismatch: OSS vector collection requires {VECTOR_DIM}d, got {dimension}"
            )));
        }
        let mut guard = self.collections.write().await;
        guard
            .entry((org_id, table.as_str().to_string()))
            .or_default();
        Ok(())
    }

    async fn upsert_points(
        &self,
        org_id: Uuid,
        table: VectorTable,
        points: &[OssVectorPoint],
    ) -> Result<(), OpenRagError> {
        for pt in points {
            if pt.org_id != org_id {
                return Err(OpenRagError::Forbidden(
                    "cross-org point upsert rejected by OSS vector adapter".into(),
                ));
            }
            validate_vector(&pt.vector)?;
        }
        let mut guard = self.collections.write().await;
        let map = guard
            .entry((org_id, table.as_str().to_string()))
            .or_default();
        for pt in points {
            map.insert(pt.id, pt.clone());
        }
        Ok(())
    }

    async fn delete_points(
        &self,
        org_id: Uuid,
        table: VectorTable,
        ids: &[Uuid],
    ) -> Result<(), OpenRagError> {
        let mut guard = self.collections.write().await;
        if let Some(map) = guard.get_mut(&(org_id, table.as_str().to_string())) {
            for id in ids {
                map.remove(id);
            }
        }
        Ok(())
    }

    async fn search_points(
        &self,
        org_id: Uuid,
        table: VectorTable,
        vector: &[f32],
        top_k: usize,
        acl_scope_key: Option<&str>,
        tag_filter: Option<&TagFilter>,
    ) -> Result<Vec<OssHit>, OpenRagError> {
        let expanded = tag_filter.map(|tf| expand_tag_filter_in_memory(tf, |_| Vec::new()));
        self.search_points_internal(org_id, table, vector, top_k, acl_scope_key, expanded.as_ref())
            .await
    }

    async fn search_points_expanded(
        &self,
        org_id: Uuid,
        table: VectorTable,
        vector: &[f32],
        top_k: usize,
        acl_scope_key: Option<&str>,
        expanded_tags: Option<&ExpandedTagFilter>,
    ) -> Result<Vec<OssHit>, OpenRagError> {
        self.search_points_internal(org_id, table, vector, top_k, acl_scope_key, expanded_tags)
            .await
    }
}

pub struct QdrantHttpBackend {
    base_url: String,
    api_key: Option<String>,
    transport: Arc<dyn HttpTransport>,
}

impl QdrantHttpBackend {
    pub fn new(
        base_url: impl Into<String>,
        api_key: Option<String>,
        transport: Arc<dyn HttpTransport>,
    ) -> Self {
        Self {
            base_url: base_url.into().trim_end_matches('/').to_string(),
            api_key,
            transport,
        }
    }

    fn collection_name(org_id: Uuid, table: VectorTable) -> String {
        format!(
            "openrag_{}_{}",
            org_id.simple(),
            table.as_str().replace('.', "_")
        )
    }

    async fn send_json(
        &self,
        method: &str,
        path: &str,
        body_val: Value,
    ) -> Result<Value, OpenRagError> {
        let body = serde_json::to_vec(&body_val).map_err(|e| OpenRagError::ExternalAdapter {
            adapter: "qdrant".into(),
            message: format!("failed to serialize Qdrant request: {e}"),
        })?;
        let mut headers = vec![("Content-Type".into(), "application/json".into())];
        if let Some(key) = &self.api_key {
            headers.push(("api-key".into(), key.clone()));
        }
        let resp = self
            .transport
            .send(HttpRequest {
                method: method.into(),
                url: format!("{}{path}", self.base_url),
                headers,
                body,
            })
            .await?;
        if resp.status >= 400 {
            return Err(OpenRagError::ExternalAdapter {
                adapter: "qdrant".into(),
                message: format!(
                    "Qdrant HTTP {} on {path}: {}",
                    resp.status,
                    String::from_utf8_lossy(&resp.body)
                ),
            });
        }
        if resp.body.is_empty() {
            return Ok(json!({}));
        }
        serde_json::from_slice(&resp.body).map_err(|e| OpenRagError::ExternalAdapter {
            adapter: "qdrant".into(),
            message: format!("invalid Qdrant JSON response: {e}"),
        })
    }
}

impl QdrantHttpBackend {
    async fn search_points_internal(
        &self,
        org_id: Uuid,
        table: VectorTable,
        vector: &[f32],
        top_k: usize,
        acl_scope_key: Option<&str>,
        expanded_tags: Option<&ExpandedTagFilter>,
    ) -> Result<Vec<OssHit>, OpenRagError> {
        validate_vector(vector)?;
        if top_k == 0 {
            return Ok(Vec::new());
        }
        let coll = Self::collection_name(org_id, table);
        let mut must = vec![json!({
            "key": "org_id",
            "match": { "value": org_id.to_string() }
        })];
        if let Some(sk) = acl_scope_key {
            must.push(json!({
                "key": "acl_scope_key",
                "match": { "value": sk }
            }));
        }
        let mut must_not = Vec::new();
        if let Some(ef) = expanded_tags {
            if !ef.tags_any.is_empty() {
                must.push(json!({
                    "key": "tags",
                    "match": { "any": ef.tags_any }
                }));
            }
            for group in &ef.tags_all_groups {
                if !group.is_empty() {
                    must.push(json!({
                        "key": "tags",
                        "match": { "any": group }
                    }));
                }
            }
            if !ef.tags_exclude.is_empty() {
                must_not.push(json!({
                    "key": "tags",
                    "match": { "any": ef.tags_exclude }
                }));
            }
        }
        let mut filter_obj = Map::new();
        filter_obj.insert("must".into(), Value::Array(must));
        if !must_not.is_empty() {
            filter_obj.insert("must_not".into(), Value::Array(must_not));
        }

        let resp = self
            .send_json(
                "POST",
                &format!("/collections/{coll}/points/search"),
                json!({
                    "vector": vector,
                    "limit": top_k,
                    "with_payload": true,
                    "filter": Value::Object(filter_obj)
                }),
            )
            .await?;
        let results = resp
            .get("result")
            .and_then(|v| v.as_array())
            .cloned()
            .unwrap_or_default();
        let mut out = Vec::with_capacity(results.len());
        for entry in results {
            let Some(id_str) = entry.get("id").and_then(|v| v.as_str()) else {
                continue;
            };
            let Ok(id) = Uuid::parse_str(id_str) else {
                continue;
            };
            let score = entry.get("score").and_then(|v| v.as_f64()).unwrap_or(0.0) as f32;
            let payload = entry
                .get("payload")
                .and_then(|v| v.as_object())
                .cloned()
                .unwrap_or_default();
            let acl_scope = payload
                .get("acl_scope_key")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let document_id = payload
                .get("document_id")
                .and_then(|v| v.as_str())
                .and_then(|s| Uuid::parse_str(s).ok());
            let tags = payload
                .get("tags")
                .and_then(|v| v.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|x| x.as_str().map(ToString::to_string))
                        .collect()
                })
                .unwrap_or_default();
            out.push(OssHit {
                id,
                document_id,
                cosine_sim: score,
                distance: 1.0 - score,
                acl_scope_key: acl_scope,
                tags,
                payload,
            });
        }
        Ok(out)
    }
}

#[async_trait]
impl OssVectorBackend for QdrantHttpBackend {
    fn kind(&self) -> OssBackendKind {
        OssBackendKind::Qdrant
    }

    async fn ensure_collection(
        &self,
        org_id: Uuid,
        table: VectorTable,
        dimension: usize,
    ) -> Result<(), OpenRagError> {
        let coll = Self::collection_name(org_id, table);
        let _ = self
            .send_json(
                "PUT",
                &format!("/collections/{coll}"),
                json!({
                    "vectors": {
                        "size": dimension,
                        "distance": "Cosine"
                    }
                }),
            )
            .await?;
        Ok(())
    }

    async fn upsert_points(
        &self,
        org_id: Uuid,
        table: VectorTable,
        points: &[OssVectorPoint],
    ) -> Result<(), OpenRagError> {
        for pt in points {
            validate_vector(&pt.vector)?;
        }
        let coll = Self::collection_name(org_id, table);
        let serialized: Vec<Value> = points
            .iter()
            .map(|pt| {
                let mut p = pt.payload.clone();
                p.insert("org_id".into(), Value::String(org_id.to_string()));
                p.insert("acl_scope_key".into(), Value::String(pt.acl_scope_key.clone()));
                p.insert(
                    "tags".into(),
                    Value::Array(pt.tags.iter().map(|t| Value::String(t.clone())).collect()),
                );
                if let Some(doc_id) = pt.document_id {
                    p.insert("document_id".into(), Value::String(doc_id.to_string()));
                }
                json!({
                    "id": pt.id.to_string(),
                    "vector": pt.vector,
                    "payload": p
                })
            })
            .collect();
        let _ = self
            .send_json(
                "PUT",
                &format!("/collections/{coll}/points?wait=true"),
                json!({ "points": serialized }),
            )
            .await?;
        Ok(())
    }

    async fn delete_points(
        &self,
        org_id: Uuid,
        table: VectorTable,
        ids: &[Uuid],
    ) -> Result<(), OpenRagError> {
        let coll = Self::collection_name(org_id, table);
        let id_strs: Vec<String> = ids.iter().map(ToString::to_string).collect();
        let _ = self
            .send_json(
                "POST",
                &format!("/collections/{coll}/points/delete?wait=true"),
                json!({ "points": id_strs }),
            )
            .await?;
        Ok(())
    }

    async fn search_points(
        &self,
        org_id: Uuid,
        table: VectorTable,
        vector: &[f32],
        top_k: usize,
        acl_scope_key: Option<&str>,
        tag_filter: Option<&TagFilter>,
    ) -> Result<Vec<OssHit>, OpenRagError> {
        let expanded = tag_filter.map(|tf| expand_tag_filter_in_memory(tf, |_| Vec::new()));
        self.search_points_internal(org_id, table, vector, top_k, acl_scope_key, expanded.as_ref())
            .await
    }

    async fn search_points_expanded(
        &self,
        org_id: Uuid,
        table: VectorTable,
        vector: &[f32],
        top_k: usize,
        acl_scope_key: Option<&str>,
        expanded_tags: Option<&ExpandedTagFilter>,
    ) -> Result<Vec<OssHit>, OpenRagError> {
        self.search_points_internal(org_id, table, vector, top_k, acl_scope_key, expanded_tags)
            .await
    }
}

pub struct OssVectorAdapter {
    backend: Arc<dyn OssVectorBackend>,
    mode: StoreMode,
}

impl OssVectorAdapter {
    pub fn new_offline(kind: OssBackendKind) -> Self {
        Self {
            backend: Arc::new(InMemoryOssVectorStore::new(kind)),
            mode: StoreMode::Offline,
        }
    }

    pub fn with_backend(backend: Arc<dyn OssVectorBackend>, mode: StoreMode) -> Self {
        Self { backend, mode }
    }

    pub fn kind(&self) -> OssBackendKind {
        self.backend.kind()
    }

    pub fn mode(&self) -> StoreMode {
        self.mode
    }

    pub fn backend(&self) -> &Arc<dyn OssVectorBackend> {
        &self.backend
    }
}

#[async_trait]
impl VectorStoreAdapter for OssVectorAdapter {
    async fn upsert(
        &self,
        org_id: Uuid,
        table: VectorTable,
        items: &[VectorUpsertItem],
    ) -> Result<(), OpenRagError> {
        self.backend
            .ensure_collection(org_id, table, VECTOR_DIM)
            .await?;
        let points: Vec<OssVectorPoint> = items
            .iter()
            .map(|item| OssVectorPoint {
                id: item.id,
                org_id,
                document_id: item.document_id,
                acl_scope_key: item.acl_scope_key.clone(),
                vector: item.vector.clone(),
                tags: item.tags.clone(),
                payload: item.metadata.clone(),
            })
            .collect();
        self.backend.upsert_points(org_id, table, &points).await
    }

    async fn delete(
        &self,
        org_id: Uuid,
        table: VectorTable,
        ids: &[Uuid],
    ) -> Result<(), OpenRagError> {
        self.backend.delete_points(org_id, table, ids).await
    }

    async fn search_rls(
        &self,
        tx: &mut RlsTransaction<'_>,
        query: &VectorSearchQuery,
    ) -> Result<Vec<VectorHit>, OpenRagError> {
        validate_vector(&query.query_vector)?;
        if query.top_k == 0 {
            return Ok(Vec::new());
        }

        let expanded_tags = if let Some(tf) = &query.tag_filter {
            if query.table.has_tags_column() {
                Some(expand_tag_filter_rls(tx, query.org_id, tf).await?)
            } else {
                None
            }
        } else {
            None
        };

        let has_pg_postfilter = query.bucket_id.is_some() || expanded_tags.is_some();
        let candidate_limit = if has_pg_postfilter {
            query.top_k.saturating_mul(16).max(64)
        } else {
            query.top_k.saturating_mul(4)
        };

        let raw_hits = self
            .backend
            .search_points_expanded(
                query.org_id,
                query.table,
                &query.query_vector,
                candidate_limit,
                query.acl_scope_key.as_deref(),
                expanded_tags.as_ref(),
            )
            .await?;
        if raw_hits.is_empty() {
            return Ok(Vec::new());
        }

        let candidate_ids: Vec<Uuid> = raw_hits.iter().map(|h| h.id).collect();
        let mut qb: QueryBuilder<'_, Postgres> = QueryBuilder::new("SELECT t.id, ");
        if query.table.has_tags_column() {
            qb.push("t.tags");
        } else {
            qb.push("ARRAY[]::text[] AS tags");
        }
        qb.push(" FROM ");
        qb.push(query.table.as_str());
        qb.push(" AS t");
        if matches!(
            query.table,
            VectorTable::CorpusChunks | VectorTable::CorpusPages | VectorTable::CorpusPageFigures
        ) && query.bucket_id.is_some()
        {
            qb.push(" JOIN corpus.documents AS d ON d.id = t.document_id AND d.org_id = t.org_id");
        }
        qb.push(" WHERE t.org_id = ");
        qb.push_bind(query.org_id);
        if query.table.has_status_active() {
            qb.push(" AND t.status = 'active'");
        }
        if let Some(bid) = query.bucket_id {
            match query.table {
                VectorTable::CorpusChunks
                | VectorTable::CorpusPages
                | VectorTable::CorpusPageFigures => {
                    qb.push(" AND d.bucket_id = ");
                    qb.push_bind(bid);
                }
                VectorTable::WikiPages | VectorTable::MemoryEntries => {
                    qb.push(" AND t.bucket_id = ");
                    qb.push_bind(bid);
                }
            }
        }
        qb.push(" AND t.id = ANY(");
        qb.push_bind(&candidate_ids);
        qb.push("::uuid[])");
        if let Some(tf) = &expanded_tags {
            push_tag_filter_predicates(&mut qb, "t.tags", tf);
        }

        let rows = qb
            .build()
            .fetch_all(&mut **tx.tx_mut())
            .await
            .map_err(|e| OpenRagError::Database(format!("OSS RLS intersection failed: {e}")))?;

        let mut visible_tags: HashMap<Uuid, Vec<String>> = HashMap::with_capacity(rows.len());
        for row in rows {
            let id: Uuid = row
                .try_get("id")
                .map_err(|e| OpenRagError::Database(format!("invalid OSS RLS row id: {e}")))?;
            let tags: Vec<String> = row
                .try_get("tags")
                .map_err(|e| OpenRagError::Database(format!("invalid OSS RLS row tags: {e}")))?;
            visible_tags.insert(id, tags);
        }

        let mut out = Vec::new();
        for hit in raw_hits {
            let Some(pg_tags) = visible_tags.get(&hit.id) else {
                continue;
            };
            let tags = if !pg_tags.is_empty() {
                pg_tags.clone()
            } else {
                hit.tags
            };
            out.push(VectorHit {
                id: hit.id,
                document_id: hit.document_id,
                table: query.table,
                cosine_sim: hit.cosine_sim,
                distance: hit.distance,
                acl_scope_key: hit.acl_scope_key,
                tags,
                metadata: hit.payload,
            });
            if out.len() >= query.top_k {
                break;
            }
        }
        Ok(out)
    }
}
