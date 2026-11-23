use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::Arc;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use openrag_core::OpenRagError;
use openrag_db::RlsTransaction;
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use sqlx::{Postgres, QueryBuilder, Row};
use tokio::sync::RwLock;
use uuid::Uuid;

use crate::tags::{expand_tag_filter_rls, push_tag_filter_predicates};
use crate::types::{
    cosine_similarity, hmac_sha256, sha256_hex, validate_vector, HttpRequest, HttpTransport,
    StoreMode, VectorHit, VectorSearchQuery, VectorStoreAdapter, VectorTable, VectorUpsertItem,
    VECTOR_DIM,
};

pub const MAX_FILTERABLE_BYTES: usize = 2048;
pub const MAX_METADATA_KEYS: usize = 64;
pub const MAX_METADATA_VALUE_BYTES: usize = 4096;
pub const MAX_METADATA_BYTES: usize = 16 * 1024;
pub const MAX_EXTRA_BYTES: usize = 6 * 1024;
pub const MAX_LIST_ELEMENTS: usize = 256;
pub const MAX_TEXT_BYTES: usize = 30 * 1024;

pub const KEY_ID: &str = "id";
pub const KEY_UPDATED_AT: &str = "_updated_at";
pub const KEY_TEXT: &str = "_text";
pub const KEY_EXTRA: &str = "_extra";
pub const KEY_EMPTY: &str = "_empty";
pub const NON_FILTERABLE_KEYS: [&str; 3] = [KEY_TEXT, KEY_EXTRA, KEY_EMPTY];

/// Matches E2E contract `sv-prod-{sha256(org_id)[:24]}` (no `"tenant:"` prefix).
pub fn s3_tenant_hash(org_id: Uuid) -> String {
    let hex = sha256_hex(org_id.to_string().as_bytes());
    hex.chars().take(24).collect()
}

pub fn s3_bucket_name(prefix: &str, org_id: Uuid) -> String {
    format!("{prefix}-{}", s3_tenant_hash(org_id))
}

pub fn s3_bucket_arn(region: &str, account_id: &str, prefix: &str, org_id: Uuid) -> String {
    format!(
        "arn:aws:s3vectors:{region}:{account_id}:bucket/{}",
        s3_bucket_name(prefix, org_id)
    )
}

/// Builds the tenant-isolated IAM STS AssumeRole inline session policy
/// locking S3 Vectors actions to the tenant's dedicated bucket ARN and S3 schema prefix `t/<hash>/*`.
pub fn build_sts_session_policy(
    region: &str,
    account_id: &str,
    prefix: &str,
    meta_bucket: &str,
    org_id: Uuid,
) -> Value {
    let hash = s3_tenant_hash(org_id);
    let bucket_arn = s3_bucket_arn(region, account_id, prefix, org_id);
    let index_arn = format!("{bucket_arn}/index/*");
    let meta_obj_arn = format!("arn:aws:s3:::{meta_bucket}/t/{hash}/*");
    let meta_bucket_arn = format!("arn:aws:s3:::{meta_bucket}");

    json!({
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "ScopedS3VectorsBucketAccess",
                "Effect": "Allow",
                "Action": [
                    "s3vectors:CreateVectorBucket",
                    "s3vectors:GetVectorBucket",
                    "s3vectors:CreateIndex",
                    "s3vectors:GetIndex",
                    "s3vectors:ListIndexes",
                    "s3vectors:PutVectors",
                    "s3vectors:GetVectors",
                    "s3vectors:DeleteVectors",
                    "s3vectors:ListVectors",
                    "s3vectors:QueryVectors"
                ],
                "Resource": [bucket_arn, index_arn]
            },
            {
                "Sid": "ScopedMetaObjects",
                "Effect": "Allow",
                "Action": ["s3:GetObject", "s3:PutObject", "s3:DeleteObject"],
                "Resource": [meta_obj_arn]
            },
            {
                "Sid": "ScopedMetaList",
                "Effect": "Allow",
                "Action": ["s3:ListBucket"],
                "Resource": [meta_bucket_arn],
                "Condition": {
                    "StringLike": {
                        "s3:prefix": format!("t/{hash}/*")
                    }
                }
            }
        ]
    })
}

/// Signs an HTTP request with AWS Signature Version 4 (HMAC-SHA256).
#[allow(clippy::too_many_arguments)]
pub fn sign_aws_sigv4(
    method: &str,
    canonical_uri: &str,
    canonical_query: &str,
    host: &str,
    region: &str,
    service: &str,
    access_key_id: &str,
    secret_access_key: &str,
    session_token: Option<&str>,
    amz_date: &str,
    payload: &[u8],
) -> Vec<(String, String)> {
    let date_stamp = &amz_date[..8.min(amz_date.len())];
    let payload_hash = sha256_hex(payload);

    let mut header_map: BTreeMap<String, String> = BTreeMap::new();
    header_map.insert("host".into(), host.trim().to_string());
    header_map.insert("x-amz-content-sha256".into(), payload_hash.clone());
    header_map.insert("x-amz-date".into(), amz_date.to_string());
    if let Some(tok) = session_token {
        header_map.insert("x-amz-security-token".into(), tok.to_string());
    }

    let mut canonical_headers = String::new();
    let mut signed_headers_list = Vec::with_capacity(header_map.len());
    for (k, v) in &header_map {
        canonical_headers.push_str(k);
        canonical_headers.push(':');
        canonical_headers.push_str(v);
        canonical_headers.push('\n');
        signed_headers_list.push(k.as_str());
    }
    let signed_headers = signed_headers_list.join(";");

    let canonical_request = format!(
        "{}\n{}\n{}\n{}\n{}\n{}",
        method.to_uppercase(),
        canonical_uri,
        canonical_query,
        canonical_headers,
        signed_headers,
        payload_hash
    );

    let credential_scope = format!("{date_stamp}/{region}/{service}/aws4_request");
    let string_to_sign = format!(
        "AWS4-HMAC-SHA256\n{amz_date}\n{credential_scope}\n{}",
        sha256_hex(canonical_request.as_bytes())
    );

    let k_secret = format!("AWS4{secret_access_key}");
    let k_date = hmac_sha256(k_secret.as_bytes(), date_stamp.as_bytes());
    let k_region = hmac_sha256(&k_date, region.as_bytes());
    let k_service = hmac_sha256(&k_region, service.as_bytes());
    let k_signing = hmac_sha256(&k_service, b"aws4_request");
    let signature = hex::encode(hmac_sha256(&k_signing, string_to_sign.as_bytes()));

    let authorization = format!(
        "AWS4-HMAC-SHA256 Credential={access_key_id}/{credential_scope}, SignedHeaders={signed_headers}, Signature={signature}"
    );

    let mut out = vec![
        ("Authorization".into(), authorization),
        ("x-amz-date".into(), amz_date.to_string()),
        ("x-amz-content-sha256".into(), payload_hash),
    ];
    if let Some(tok) = session_token {
        out.push(("x-amz-security-token".into(), tok.to_string()));
    }
    out
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct S3MetadataSchema {
    /// If empty, all user keys are treated as filterable unless excluded into `_extra`.
    #[serde(default)]
    pub declared_filterable_keys: BTreeSet<String>,
    #[serde(default)]
    pub string_list_keys: BTreeSet<String>,
}

pub fn encode_s3_metadata(
    id: &str,
    updated_at_micros: i64,
    text: Option<&str>,
    raw_metadata: &Map<String, Value>,
    schema: &S3MetadataSchema,
) -> Result<Map<String, Value>, OpenRagError> {
    if let Some(t) = text {
        if t.len() > MAX_TEXT_BYTES {
            return Err(OpenRagError::Validation(format!(
                "text_too_large: text is {} bytes; limit is {MAX_TEXT_BYTES}",
                t.len()
            )));
        }
    }

    // Key-count & value-size validation matching vectors/src/metadata.rs
    let mut non_null_keys = 0usize;
    for (k, v) in raw_metadata {
        if !v.is_null() {
            non_null_keys += 1;
        }
        if let Value::String(s) = v {
            if s.len() > MAX_METADATA_VALUE_BYTES {
                return Err(OpenRagError::Validation(format!(
                    "value_too_large: metadata key '{k}' is {} bytes; limit is {MAX_METADATA_VALUE_BYTES}",
                    s.len()
                )));
            }
        }
        if let Value::Array(arr) = v {
            if arr.len() > MAX_LIST_ELEMENTS {
                return Err(OpenRagError::Validation(format!(
                    "too_many_elements: metadata list '{k}' has {} elements; limit is {MAX_LIST_ELEMENTS}",
                    arr.len()
                )));
            }
            for item in arr {
                if let Some(s) = item.as_str() {
                    if s.len() > MAX_METADATA_VALUE_BYTES {
                        return Err(OpenRagError::Validation(format!(
                            "value_too_large: element in '{k}' exceeds {MAX_METADATA_VALUE_BYTES} bytes"
                        )));
                    }
                }
            }
        }
    }
    if non_null_keys > MAX_METADATA_KEYS {
        return Err(OpenRagError::Validation(format!(
            "too_many_keys: metadata has {non_null_keys} non-null keys; limit is {MAX_METADATA_KEYS}"
        )));
    }
    let total_json_bytes = serde_json::to_string(raw_metadata)
        .map_err(|e| OpenRagError::Validation(format!("metadata serialization error: {e}")))?
        .len();
    if total_json_bytes > MAX_METADATA_BYTES {
        return Err(OpenRagError::Validation(format!(
            "metadata_too_large: metadata is {total_json_bytes} bytes; limit is {MAX_METADATA_BYTES}"
        )));
    }

    let mut filterable = Map::new();
    let mut extra = Map::new();
    let mut empty_list_fields: Vec<String> = Vec::new();

    for (k, v) in raw_metadata {
        if v.is_null() || k == KEY_ID || k == KEY_UPDATED_AT || NON_FILTERABLE_KEYS.contains(&k.as_str()) {
            continue;
        }
        let is_declared = schema.declared_filterable_keys.is_empty()
            || schema.declared_filterable_keys.contains(k);
        if is_declared {
            if let Value::Array(arr) = v {
                if arr.is_empty() {
                    empty_list_fields.push(k.clone());
                    continue;
                }
            }
            filterable.insert(k.clone(), v.clone());
        } else {
            extra.insert(k.clone(), v.clone());
        }
    }

    filterable.insert(KEY_ID.into(), Value::String(id.to_string()));
    filterable.insert(KEY_UPDATED_AT.into(), json!(updated_at_micros));

    let filterable_bytes = serde_json::to_string(&filterable)
        .map_err(|e| OpenRagError::Validation(format!("filterable serialization error: {e}")))?
        .len();
    if filterable_bytes > MAX_FILTERABLE_BYTES {
        return Err(OpenRagError::Validation(format!(
            "filterable_metadata_too_large: declared metadata is {filterable_bytes} bytes; limit is {MAX_FILTERABLE_BYTES}"
        )));
    }

    let mut out = filterable;
    if !extra.is_empty() {
        let extra_json = serde_json::to_string(&extra)
            .map_err(|e| OpenRagError::Validation(format!("extra serialization error: {e}")))?;
        if extra_json.len() > MAX_EXTRA_BYTES {
            return Err(OpenRagError::Validation(format!(
                "extra_metadata_too_large: _extra is {} bytes; limit is {MAX_EXTRA_BYTES}",
                extra_json.len()
            )));
        }
        out.insert(KEY_EXTRA.into(), Value::String(extra_json));
    }
    if let Some(t) = text {
        out.insert(KEY_TEXT.into(), Value::String(t.to_string()));
    }
    if !empty_list_fields.is_empty() {
        empty_list_fields.sort();
        let empty_json = serde_json::to_string(&empty_list_fields)
            .map_err(|e| OpenRagError::Validation(format!("_empty serialization error: {e}")))?;
        out.insert(KEY_EMPTY.into(), Value::String(empty_json));
    }

    Ok(out)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecodedS3Metadata {
    pub id: Option<String>,
    pub updated_at_micros: Option<i64>,
    pub text: Option<String>,
    pub metadata: Map<String, Value>,
}

pub fn decode_s3_metadata(
    stored: &Map<String, Value>,
    schema: &S3MetadataSchema,
) -> Result<DecodedS3Metadata, OpenRagError> {
    let id = stored.get(KEY_ID).and_then(|v| v.as_str()).map(ToString::to_string);
    let updated_at_micros = stored.get(KEY_UPDATED_AT).and_then(|v| v.as_i64());
    let text = stored.get(KEY_TEXT).and_then(|v| v.as_str()).map(ToString::to_string);

    let mut metadata = Map::new();
    for (k, v) in stored {
        if k == KEY_ID || k == KEY_UPDATED_AT || NON_FILTERABLE_KEYS.contains(&k.as_str()) {
            continue;
        }
        metadata.insert(k.clone(), v.clone());
    }

    if let Some(Value::String(extra_raw)) = stored.get(KEY_EXTRA) {
        let extra_map: Map<String, Value> = serde_json::from_str(extra_raw).map_err(|e| {
            OpenRagError::Validation(format!("corrupt _extra JSON in S3 metadata: {e}"))
        })?;
        for (k, v) in extra_map {
            metadata.insert(k, v);
        }
    }

    if let Some(Value::String(empty_raw)) = stored.get(KEY_EMPTY) {
        let empty_keys: Vec<String> = serde_json::from_str(empty_raw).map_err(|e| {
            OpenRagError::Validation(format!("corrupt _empty JSON in S3 metadata: {e}"))
        })?;
        for k in empty_keys {
            if schema.string_list_keys.is_empty() || schema.string_list_keys.contains(&k) {
                metadata.insert(k, Value::Array(Vec::new()));
            }
        }
    }

    Ok(DecodedS3Metadata {
        id,
        updated_at_micros,
        text,
        metadata,
    })
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum S3Filter {
    Eq { field: String, value: Value },
    Ne { field: String, value: Value },
    Gt { field: String, value: f64 },
    Gte { field: String, value: f64 },
    Lt { field: String, value: f64 },
    Lte { field: String, value: f64 },
    In { field: String, values: Vec<Value> },
    And { filters: Vec<S3Filter> },
}

pub fn s3_filter_matches(filter: &S3Filter, meta: &Map<String, Value>) -> bool {
    match filter {
        S3Filter::And { filters } => filters.iter().all(|f| s3_filter_matches(f, meta)),
        S3Filter::Eq { field, value } => match meta.get(field) {
            Some(Value::Array(arr)) => arr.contains(value),
            Some(v) => v == value,
            None => false,
        },
        S3Filter::Ne { field, value } => match meta.get(field) {
            Some(Value::Array(arr)) => !arr.contains(value),
            Some(v) => v != value,
            None => true,
        },
        S3Filter::Gt { field, value } => meta
            .get(field)
            .and_then(|v| v.as_f64())
            .is_some_and(|n| n > *value),
        S3Filter::Gte { field, value } => meta
            .get(field)
            .and_then(|v| v.as_f64())
            .is_some_and(|n| n >= *value),
        S3Filter::Lt { field, value } => meta
            .get(field)
            .and_then(|v| v.as_f64())
            .is_some_and(|n| n < *value),
        S3Filter::Lte { field, value } => meta
            .get(field)
            .and_then(|v| v.as_f64())
            .is_some_and(|n| n <= *value),
        S3Filter::In { field, values } => match meta.get(field) {
            Some(Value::Array(arr)) => arr.iter().any(|item| values.contains(item)),
            Some(v) => values.contains(v),
            None => false,
        },
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct S3VectorPutItem {
    pub key: String,
    pub vector: Vec<f32>,
    pub metadata: Map<String, Value>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct S3VectorRecord {
    pub key: String,
    pub vector: Option<Vec<f32>>,
    pub metadata: Map<String, Value>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct S3VectorQueryHit {
    pub key: String,
    pub distance: f32,
    pub cosine_sim: f32,
    pub metadata: Map<String, Value>,
}

#[async_trait]
pub trait ScopedS3Bucket: Send + Sync {
    fn bucket_name(&self) -> &str;
    fn bucket_arn(&self) -> &str;
    fn org_id(&self) -> Uuid;

    async fn ensure_index(&self, index_name: &str, dimension: usize) -> Result<(), OpenRagError>;
    async fn put(&self, index_name: &str, items: &[S3VectorPutItem]) -> Result<(), OpenRagError>;
    async fn get(
        &self,
        index_name: &str,
        keys: &[String],
        with_data: bool,
    ) -> Result<Vec<S3VectorRecord>, OpenRagError>;
    async fn delete(&self, index_name: &str, keys: &[String]) -> Result<(), OpenRagError>;
    async fn query(
        &self,
        index_name: &str,
        vector: &[f32],
        top_k: usize,
        filter: Option<&S3Filter>,
    ) -> Result<Vec<S3VectorQueryHit>, OpenRagError>;
}

#[async_trait]
pub trait S3VectorsBackend: Send + Sync {
    async fn scoped(&self, org_id: Uuid) -> Result<Arc<dyn ScopedS3Bucket>, OpenRagError>;
}

#[derive(Debug, Default)]
struct BucketIndexData {
    dimension: usize,
    vectors: HashMap<String, S3VectorPutItem>,
}

type S3BucketsMap = HashMap<String, HashMap<String, BucketIndexData>>;

/// Deterministic offline in-memory S3 Vectors store enforcing strict per-org bucket isolation,
/// 256d finite non-zero vectors, and S3 Vectors' ban on raw empty arrays in metadata.
#[derive(Debug, Clone)]
pub struct InMemoryS3VectorsStore {
    prefix: String,
    region: String,
    account_id: String,
    buckets: Arc<RwLock<S3BucketsMap>>,
}

impl Default for InMemoryS3VectorsStore {
    fn default() -> Self {
        Self::new("sv-prod", "ap-northeast-2", "000000000000")
    }
}

impl InMemoryS3VectorsStore {
    pub fn new(
        prefix: impl Into<String>,
        region: impl Into<String>,
        account_id: impl Into<String>,
    ) -> Self {
        Self {
            prefix: prefix.into(),
            region: region.into(),
            account_id: account_id.into(),
            buckets: Arc::new(RwLock::new(HashMap::new())),
        }
    }
}

#[async_trait]
impl S3VectorsBackend for InMemoryS3VectorsStore {
    async fn scoped(&self, org_id: Uuid) -> Result<Arc<dyn ScopedS3Bucket>, OpenRagError> {
        let bname = s3_bucket_name(&self.prefix, org_id);
        let barn = s3_bucket_arn(&self.region, &self.account_id, &self.prefix, org_id);
        let mut guard = self.buckets.write().await;
        guard.entry(bname.clone()).or_default();
        Ok(Arc::new(InMemoryScopedS3Bucket {
            org_id,
            bucket_name: bname,
            bucket_arn: barn,
            buckets: Arc::clone(&self.buckets),
        }))
    }
}

#[derive(Debug, Clone)]
struct InMemoryScopedS3Bucket {
    org_id: Uuid,
    bucket_name: String,
    bucket_arn: String,
    buckets: Arc<RwLock<S3BucketsMap>>,
}

#[async_trait]
impl ScopedS3Bucket for InMemoryScopedS3Bucket {
    fn bucket_name(&self) -> &str {
        &self.bucket_name
    }

    fn bucket_arn(&self) -> &str {
        &self.bucket_arn
    }

    fn org_id(&self) -> Uuid {
        self.org_id
    }

    async fn ensure_index(&self, index_name: &str, dimension: usize) -> Result<(), OpenRagError> {
        if dimension != VECTOR_DIM {
            return Err(OpenRagError::Validation(format!(
                "dimension_mismatch: S3Vectors index must be {VECTOR_DIM}d, got {dimension}"
            )));
        }
        let mut guard = self.buckets.write().await;
        let indexes = guard.entry(self.bucket_name.clone()).or_default();
        if let Some(existing) = indexes.get(index_name) {
            if existing.dimension != dimension {
                return Err(OpenRagError::Conflict(format!(
                    "index '{index_name}' dimension mismatch ({} != {dimension})",
                    existing.dimension
                )));
            }
        } else {
            indexes.insert(
                index_name.to_string(),
                BucketIndexData {
                    dimension,
                    vectors: HashMap::new(),
                },
            );
        }
        Ok(())
    }

    async fn put(&self, index_name: &str, items: &[S3VectorPutItem]) -> Result<(), OpenRagError> {
        for item in items {
            validate_vector(&item.vector)?;
            for (k, v) in &item.metadata {
                if NON_FILTERABLE_KEYS.contains(&k.as_str()) {
                    continue;
                }
                if let Value::Array(arr) = v {
                    if arr.is_empty() {
                        return Err(OpenRagError::Validation(format!(
                            "Empty arrays are not allowed in metadata (key '{k}')"
                        )));
                    }
                }
            }
        }

        let mut guard = self.buckets.write().await;
        let indexes = guard.entry(self.bucket_name.clone()).or_default();
        let idx = indexes
            .entry(index_name.to_string())
            .or_insert_with(|| BucketIndexData {
                dimension: VECTOR_DIM,
                vectors: HashMap::new(),
            });
        for item in items {
            idx.vectors.insert(item.key.clone(), item.clone());
        }
        Ok(())
    }

    async fn get(
        &self,
        index_name: &str,
        keys: &[String],
        with_data: bool,
    ) -> Result<Vec<S3VectorRecord>, OpenRagError> {
        let guard = self.buckets.read().await;
        let Some(indexes) = guard.get(&self.bucket_name) else {
            return Ok(Vec::new());
        };
        let Some(idx) = indexes.get(index_name) else {
            return Ok(Vec::new());
        };
        let mut out = Vec::new();
        for key in keys {
            if let Some(item) = idx.vectors.get(key) {
                out.push(S3VectorRecord {
                    key: item.key.clone(),
                    vector: if with_data {
                        Some(item.vector.clone())
                    } else {
                        None
                    },
                    metadata: item.metadata.clone(),
                });
            }
        }
        Ok(out)
    }

    async fn delete(&self, index_name: &str, keys: &[String]) -> Result<(), OpenRagError> {
        let mut guard = self.buckets.write().await;
        if let Some(indexes) = guard.get_mut(&self.bucket_name) {
            if let Some(idx) = indexes.get_mut(index_name) {
                for key in keys {
                    idx.vectors.remove(key);
                }
            }
        }
        Ok(())
    }

    async fn query(
        &self,
        index_name: &str,
        vector: &[f32],
        top_k: usize,
        filter: Option<&S3Filter>,
    ) -> Result<Vec<S3VectorQueryHit>, OpenRagError> {
        validate_vector(vector)?;
        if top_k == 0 {
            return Ok(Vec::new());
        }

        let guard = self.buckets.read().await;
        let Some(indexes) = guard.get(&self.bucket_name) else {
            return Ok(Vec::new());
        };
        let Some(idx) = indexes.get(index_name) else {
            return Ok(Vec::new());
        };

        let mut hits = Vec::new();
        for item in idx.vectors.values() {
            if let Some(f) = filter {
                if !s3_filter_matches(f, &item.metadata) {
                    continue;
                }
            }
            let sim = cosine_similarity(vector, &item.vector)?;
            let distance = 1.0 - sim;
            hits.push(S3VectorQueryHit {
                key: item.key.clone(),
                distance,
                cosine_sim: sim,
                metadata: item.metadata.clone(),
            });
        }
        hits.sort_by(|a, b| {
            a.distance
                .partial_cmp(&b.distance)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.key.cmp(&b.key))
        });
        hits.truncate(top_k);
        Ok(hits)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StsSessionCredentials {
    pub access_key_id: String,
    pub secret_access_key: String,
    pub session_token: Option<String>,
    pub expires_at: DateTime<Utc>,
    pub policy: Value,
}

/// Real HTTP + AWS SigV4 S3 Vectors client over an injected `HttpTransport`.
pub struct S3VectorsHttpClient {
    endpoint: String,
    region: String,
    account_id: String,
    prefix: String,
    meta_bucket: String,
    access_key_id: String,
    secret_access_key: String,
    session_token: Option<String>,
    transport: Arc<dyn HttpTransport>,
    session_cache: Arc<RwLock<HashMap<Uuid, StsSessionCredentials>>>,
}

impl S3VectorsHttpClient {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        endpoint: impl Into<String>,
        region: impl Into<String>,
        account_id: impl Into<String>,
        prefix: impl Into<String>,
        meta_bucket: impl Into<String>,
        access_key_id: impl Into<String>,
        secret_access_key: impl Into<String>,
        session_token: Option<String>,
        transport: Arc<dyn HttpTransport>,
    ) -> Self {
        Self {
            endpoint: endpoint.into().trim_end_matches('/').to_string(),
            region: region.into(),
            account_id: account_id.into(),
            prefix: prefix.into(),
            meta_bucket: meta_bucket.into(),
            access_key_id: access_key_id.into(),
            secret_access_key: secret_access_key.into(),
            session_token,
            transport,
            session_cache: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    async fn get_or_create_session(&self, org_id: Uuid) -> StsSessionCredentials {
        let now = Utc::now();
        {
            let guard = self.session_cache.read().await;
            if let Some(creds) = guard.get(&org_id) {
                if creds.expires_at > now + chrono::Duration::minutes(5) {
                    return creds.clone();
                }
            }
        }
        let policy = build_sts_session_policy(
            &self.region,
            &self.account_id,
            &self.prefix,
            &self.meta_bucket,
            org_id,
        );
        let creds = StsSessionCredentials {
            access_key_id: self.access_key_id.clone(),
            secret_access_key: self.secret_access_key.clone(),
            session_token: self.session_token.clone(),
            expires_at: now + chrono::Duration::hours(1),
            policy,
        };
        let mut guard = self.session_cache.write().await;
        guard.insert(org_id, creds.clone());
        creds
    }
}

#[async_trait]
impl S3VectorsBackend for S3VectorsHttpClient {
    async fn scoped(&self, org_id: Uuid) -> Result<Arc<dyn ScopedS3Bucket>, OpenRagError> {
        let creds = self.get_or_create_session(org_id).await;
        let bucket_name = s3_bucket_name(&self.prefix, org_id);
        let bucket_arn = s3_bucket_arn(&self.region, &self.account_id, &self.prefix, org_id);
        Ok(Arc::new(S3VectorsHttpBucket {
            org_id,
            bucket_name,
            bucket_arn,
            endpoint: self.endpoint.clone(),
            region: self.region.clone(),
            creds,
            transport: Arc::clone(&self.transport),
        }))
    }
}

struct S3VectorsHttpBucket {
    org_id: Uuid,
    bucket_name: String,
    bucket_arn: String,
    endpoint: String,
    region: String,
    creds: StsSessionCredentials,
    transport: Arc<dyn HttpTransport>,
}

impl S3VectorsHttpBucket {
    async fn post_json(&self, path: &str, payload: &Value) -> Result<Value, OpenRagError> {
        let body = serde_json::to_vec(payload).map_err(|e| {
            OpenRagError::ExternalAdapter {
                adapter: "s3_vectors".into(),
                message: format!("failed to serialize request: {e}"),
            }
        })?;
        let host = self
            .endpoint
            .strip_prefix("http://")
            .or_else(|| self.endpoint.strip_prefix("https://"))
            .unwrap_or(&self.endpoint)
            .split('/')
            .next()
            .unwrap_or("s3vectors.amazonaws.com");
        let amz_date = Utc::now().format("%Y%m%dT%H%M%SZ").to_string();
        let mut headers = sign_aws_sigv4(
            "POST",
            path,
            "",
            host,
            &self.region,
            "s3vectors",
            &self.creds.access_key_id,
            &self.creds.secret_access_key,
            self.creds.session_token.as_deref(),
            &amz_date,
            &body,
        );
        headers.push(("Content-Type".into(), "application/json".into()));
        headers.push((
            "X-OpenRAG-Sts-Bucket-Arn".into(),
            self.bucket_arn.clone(),
        ));

        let resp = self
            .transport
            .send(HttpRequest {
                method: "POST".into(),
                url: format!("{}{path}", self.endpoint),
                headers,
                body,
            })
            .await?;

        if resp.status >= 400 {
            return Err(OpenRagError::ExternalAdapter {
                adapter: "s3_vectors".into(),
                message: format!(
                    "S3Vectors HTTP {} on {path}: {}",
                    resp.status,
                    String::from_utf8_lossy(&resp.body)
                ),
            });
        }
        if resp.body.is_empty() {
            return Ok(json!({}));
        }
        serde_json::from_slice(&resp.body).map_err(|e| OpenRagError::ExternalAdapter {
            adapter: "s3_vectors".into(),
            message: format!("invalid JSON response from S3Vectors: {e}"),
        })
    }
}

#[async_trait]
impl ScopedS3Bucket for S3VectorsHttpBucket {
    fn bucket_name(&self) -> &str {
        &self.bucket_name
    }

    fn bucket_arn(&self) -> &str {
        &self.bucket_arn
    }

    fn org_id(&self) -> Uuid {
        self.org_id
    }

    async fn ensure_index(&self, index_name: &str, dimension: usize) -> Result<(), OpenRagError> {
        let payload = json!({
            "vectorBucketName": self.bucket_name,
            "indexName": index_name,
            "dimension": dimension,
            "distanceMetric": "cosine",
            "dataType": "float32",
            "metadataConfiguration": {
                "nonFilterableMetadataKeys": NON_FILTERABLE_KEYS
            }
        });
        let _ = self.post_json("/CreateIndex", &payload).await?;
        Ok(())
    }

    async fn put(&self, index_name: &str, items: &[S3VectorPutItem]) -> Result<(), OpenRagError> {
        for item in items {
            validate_vector(&item.vector)?;
        }
        let vectors: Vec<Value> = items
            .iter()
            .map(|i| {
                json!({
                    "key": i.key,
                    "data": { "float32": i.vector },
                    "metadata": i.metadata
                })
            })
            .collect();
        let payload = json!({
            "vectorBucketName": self.bucket_name,
            "indexName": index_name,
            "vectors": vectors
        });
        let _ = self.post_json("/PutVectors", &payload).await?;
        Ok(())
    }

    async fn get(
        &self,
        index_name: &str,
        keys: &[String],
        with_data: bool,
    ) -> Result<Vec<S3VectorRecord>, OpenRagError> {
        let payload = json!({
            "vectorBucketName": self.bucket_name,
            "indexName": index_name,
            "keys": keys,
            "returnData": with_data,
            "returnMetadata": true
        });
        let resp = self.post_json("/GetVectors", &payload).await?;
        let arr = resp
            .get("vectors")
            .and_then(|v| v.as_array())
            .cloned()
            .unwrap_or_default();
        let mut out = Vec::with_capacity(arr.len());
        for entry in arr {
            let key = entry
                .get("key")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let vec_opt = entry
                .get("data")
                .and_then(|d| d.get("float32"))
                .and_then(|a| serde_json::from_value::<Vec<f32>>(a.clone()).ok());
            let metadata = entry
                .get("metadata")
                .and_then(|m| m.as_object())
                .cloned()
                .unwrap_or_default();
            out.push(S3VectorRecord {
                key,
                vector: vec_opt,
                metadata,
            });
        }
        Ok(out)
    }

    async fn delete(&self, index_name: &str, keys: &[String]) -> Result<(), OpenRagError> {
        let payload = json!({
            "vectorBucketName": self.bucket_name,
            "indexName": index_name,
            "keys": keys
        });
        let _ = self.post_json("/DeleteVectors", &payload).await?;
        Ok(())
    }

    async fn query(
        &self,
        index_name: &str,
        vector: &[f32],
        top_k: usize,
        filter: Option<&S3Filter>,
    ) -> Result<Vec<S3VectorQueryHit>, OpenRagError> {
        validate_vector(vector)?;
        let mut payload = json!({
            "vectorBucketName": self.bucket_name,
            "indexName": index_name,
            "queryVector": { "float32": vector },
            "topK": top_k,
            "returnDistance": true,
            "returnMetadata": true
        });
        if let Some(f) = filter {
            if let Some(obj) = payload.as_object_mut() {
                obj.insert("filter".into(), serde_json::to_value(f).unwrap_or(Value::Null));
            }
        }
        let resp = self.post_json("/QueryVectors", &payload).await?;
        let arr = resp
            .get("vectors")
            .and_then(|v| v.as_array())
            .cloned()
            .unwrap_or_default();
        let mut hits = Vec::with_capacity(arr.len());
        for entry in arr {
            let key = entry
                .get("key")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let distance = entry
                .get("distance")
                .and_then(|v| v.as_f64())
                .unwrap_or(1.0) as f32;
            let metadata = entry
                .get("metadata")
                .and_then(|m| m.as_object())
                .cloned()
                .unwrap_or_default();
            hits.push(S3VectorQueryHit {
                key,
                distance,
                cosine_sim: 1.0 - distance,
                metadata,
            });
        }
        Ok(hits)
    }
}

pub struct S3VectorsAdapter {
    backend: Arc<dyn S3VectorsBackend>,
    schema: S3MetadataSchema,
    mode: StoreMode,
}

impl S3VectorsAdapter {
    pub fn new_offline() -> Self {
        Self {
            backend: Arc::new(InMemoryS3VectorsStore::default()),
            schema: S3MetadataSchema::default(),
            mode: StoreMode::Offline,
        }
    }

    pub fn with_backend(
        backend: Arc<dyn S3VectorsBackend>,
        schema: S3MetadataSchema,
        mode: StoreMode,
    ) -> Self {
        Self {
            backend,
            schema,
            mode,
        }
    }

    pub fn mode(&self) -> StoreMode {
        self.mode
    }

    pub async fn scoped_bucket(&self, org_id: Uuid) -> Result<Arc<dyn ScopedS3Bucket>, OpenRagError> {
        self.backend.scoped(org_id).await
    }
}

#[async_trait]
impl VectorStoreAdapter for S3VectorsAdapter {
    async fn upsert(
        &self,
        org_id: Uuid,
        table: VectorTable,
        items: &[VectorUpsertItem],
    ) -> Result<(), OpenRagError> {
        let bucket = self.backend.scoped(org_id).await?;
        bucket.ensure_index(table.as_str(), VECTOR_DIM).await?;
        let now_micros = Utc::now().timestamp_micros();
        let mut encoded = Vec::with_capacity(items.len());
        for item in items {
            validate_vector(&item.vector)?;
            let mut raw = item.metadata.clone();
            raw.insert("acl_scope_key".into(), Value::String(item.acl_scope_key.clone()));
            if let Some(doc_id) = item.document_id {
                raw.insert("document_id".into(), Value::String(doc_id.to_string()));
            }
            if !item.tags.is_empty() || self.schema.string_list_keys.contains("tags") {
                raw.insert(
                    "tags".into(),
                    Value::Array(item.tags.iter().map(|t| Value::String(t.clone())).collect()),
                );
            }
            let meta = encode_s3_metadata(
                &item.id.to_string(),
                now_micros,
                item.text.as_deref(),
                &raw,
                &self.schema,
            )?;
            encoded.push(S3VectorPutItem {
                key: item.id.to_string(),
                vector: item.vector.clone(),
                metadata: meta,
            });
        }
        bucket.put(table.as_str(), &encoded).await
    }

    async fn delete(
        &self,
        org_id: Uuid,
        table: VectorTable,
        ids: &[Uuid],
    ) -> Result<(), OpenRagError> {
        let bucket = self.backend.scoped(org_id).await?;
        let keys: Vec<String> = ids.iter().map(ToString::to_string).collect();
        bucket.delete(table.as_str(), &keys).await
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

        let bucket = self.backend.scoped(query.org_id).await?;
        let filter = query.acl_scope_key.as_ref().map(|sk| S3Filter::Eq {
            field: "acl_scope_key".into(),
            value: Value::String(sk.clone()),
        });
        let raw_hits = bucket
            .query(
                query.table.as_str(),
                &query.query_vector,
                candidate_limit,
                filter.as_ref(),
            )
            .await?;

        let mut candidate_ids = Vec::new();
        let mut by_id: HashMap<Uuid, S3VectorQueryHit> = HashMap::new();
        for hit in raw_hits {
            if let Ok(uid) = Uuid::parse_str(&hit.key) {
                candidate_ids.push(uid);
                by_id.insert(uid, hit);
            }
        }
        if candidate_ids.is_empty() {
            return Ok(Vec::new());
        }

        // Intersect external S3 hits with Fail-Closed PostgreSQL RLS visibility, bucket_id, and TagFilter
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
            .map_err(|e| OpenRagError::Database(format!("RLS intersection check failed: {e}")))?;

        let mut visible_tags: HashMap<Uuid, Vec<String>> = HashMap::with_capacity(rows.len());
        for row in rows {
            let id: Uuid = row
                .try_get("id")
                .map_err(|e| OpenRagError::Database(format!("invalid S3 RLS row id: {e}")))?;
            let tags: Vec<String> = row
                .try_get("tags")
                .map_err(|e| OpenRagError::Database(format!("invalid S3 RLS row tags: {e}")))?;
            visible_tags.insert(id, tags);
        }

        let mut out = Vec::new();
        for id in candidate_ids {
            let Some(pg_tags) = visible_tags.get(&id) else {
                continue;
            };
            if let Some(hit) = by_id.remove(&id) {
                let decoded = decode_s3_metadata(&hit.metadata, &self.schema)?;
                let acl_scope_key = decoded
                    .metadata
                    .get("acl_scope_key")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let document_id = decoded
                    .metadata
                    .get("document_id")
                    .and_then(|v| v.as_str())
                    .and_then(|s| Uuid::parse_str(s).ok());
                let tags = if !pg_tags.is_empty() {
                    pg_tags.clone()
                } else {
                    decoded
                        .metadata
                        .get("tags")
                        .and_then(|v| v.as_array())
                        .map(|arr| {
                            arr.iter()
                                .filter_map(|x| x.as_str().map(ToString::to_string))
                                .collect()
                        })
                        .unwrap_or_default()
                };
                out.push(VectorHit {
                    id,
                    document_id,
                    table: query.table,
                    cosine_sim: hit.cosine_sim,
                    distance: hit.distance,
                    acl_scope_key,
                    tags,
                    metadata: decoded.metadata,
                });
                if out.len() >= query.top_k {
                    break;
                }
            }
        }
        Ok(out)
    }
}

// Keep unused Digest import silent if sha2::Sha256 only used via types::sha256_hex
const _: fn(&[u8]) -> [u8; 32] = |b| Sha256::digest(b).into();
