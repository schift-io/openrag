use std::collections::BTreeMap;
use std::convert::Infallible;
use std::sync::Arc;

use axum::body::Bytes;
use axum::extract::{FromRequest, Multipart, Path, Query, Request, State};
use axum::http::header::CONTENT_TYPE;
use axum::http::StatusCode;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{delete, get, post, put};
use axum::{Json, Router};
use chrono::{DateTime, Utc};
use futures_util::stream;
use openrag_core::{
    AccessScope, Arn, ArnResourceType, AuthLevel, ChunkerKind, ExternalMirrorBackend, FusionMode,
    GuardAction, GuardDecision, GuardSurface, OpenRagError, PrincipalContext, Sensitivity,
};
use openrag_db::{DbPool, RlsTransaction};
use openrag_eval::{AbQueryInput, AbSearchEvaluator, AbSearchRequest, AbVariantConfig};
use openrag_memory::{
    compact_term, extract_parenthetical_pairs, normalize_term, parse_frontmatter,
    Concept, ConceptAlias, ConceptCategory, ConceptGraph, ConceptSearchMatch, ConceptSource,
    DeterministicTextEmbedder, TextEmbedder, ResolvedConcept, WikiCompiler, WikiEditorType, WikiStatus,
};
use openrag_parse::{
    resolve_ocr_policy, sha256_hex_bytes, sha256_hex_str, DocumentFormat, OcrTask,
    ParsePolicyCtx, ParsePersister, ParsedDocument, ParsedPage, PersistParsedDocumentInput,
};
use openrag_pii::{
    GuardEngine, GuardPolicyRuntime, GuardRepo, InMemoryConceptAliasLookup, PiiEngine,
};
use openrag_retrieve::{RetrieveSpace, SearchRequest};
use openrag_store::{
    add_tags_rls, encode_s3_metadata, normalize_tag, remove_tags_rls, validate_vector,
    DefaultOutboxExecutor, EnqueueOutboxItem, OutboxEnqueuer, OutboxOp, OutboxWorker,
    S3MetadataSchema, TagFilter, TagTable, VectorTable,
};
use pgvector::Vector;
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use sqlx::Row;
use uuid::Uuid;

use crate::auth::{cli_code_exchange, enforce_api_key_arn_scope, AuthPrincipal};
use crate::chunking::chunk_page;
use crate::error::ApiError;
use crate::llm_router::{neutralize_literal_pii_tokens, run_chat_completion, LlmCompletionInput};
use crate::state::{load_active_guard_policy, load_concept_lookup, AppState};

pub fn core_routes() -> Router<Arc<AppState>> {
    Router::new()
        .route("/health", get(health_handler))
        .route("/v1/auth/cli/code-exchange", post(cli_code_exchange))
        .route("/v2/buckets", get(list_buckets_handler).post(create_bucket_handler))
        .route("/v2/buckets/{id}", get(get_bucket_handler))
        .route(
            "/v2/buckets/{id}/documents",
            get(crate::api_docs::list_documents_handler).post(upload_or_create_document_handler),
        )
        .route(
            "/v2/buckets/{id}/documents/{doc_id}",
            delete(delete_document_handler),
        )
        .route("/v2/buckets/{id}/search", post(search_bucket_handler))
        .route("/v1/buckets/{bucket}/documents/sync", put(sync_documents_handler))
        .route("/v1/chat/completions", post(chat_completions_handler))
        .route("/v1/eval/ab-search", post(ab_search_handler))
        .route("/v1/concepts", get(list_concepts_handler).post(create_concept_handler))
        .route("/v1/concepts/link", post(link_concepts_handler))
        .route("/v1/concepts/resolve", post(resolve_concepts_handler))
        .route(
            "/v1/concepts/{id}",
            get(get_concept_handler).delete(delete_concept_handler),
        )
        .route(
            "/v1/concepts/{id}/aliases/{alias_id}",
            delete(delete_concept_alias_handler),
        )
        .route(
            "/v2/buckets/{bucket_id}/documents/{doc_id}/tags",
            get(get_document_tags_handler)
                .post(add_document_tags_handler)
                .delete(remove_document_tags_handler),
        )
        .route(
            "/v1/wiki/pages/{page_id}/tags",
            post(add_wiki_tags_handler).delete(remove_wiki_tags_handler),
        )
        .route(
            "/v1/memories/{memory_id}/tags",
            post(add_memory_tags_handler).delete(remove_memory_tags_handler),
        )
}

#[derive(Debug, Clone, Serialize)]
pub struct BucketRecord {
    pub id: Uuid,
    pub bucket_id: Uuid,
    pub org_id: Uuid,
    pub dept_id: Uuid,
    pub owner_principal_id: Uuid,
    pub slug: String,
    pub name: String,
    pub access_scope: String,
    pub allowed_group_ids: Vec<Uuid>,
    pub denied_group_ids: Vec<Uuid>,
    pub min_auth_level: i16,
    pub arn: String,
    pub created_at: DateTime<Utc>,
}

pub async fn resolve_bucket_rls(
    tx: &mut RlsTransaction<'_>,
    org_id: Uuid,
    bucket_ref: &str,
) -> Result<BucketRecord, OpenRagError> {
    let trimmed = bucket_ref.trim();
    if trimmed.is_empty() {
        return Err(OpenRagError::NotFound("bucket identifier must not be empty".into()));
    }
    let parsed_uuid = Uuid::parse_str(trimmed).ok();
    let row = sqlx::query(
        "SELECT id, org_id, dept_id, owner_principal_id, slug, name, access_scope,
                allowed_group_ids, denied_group_ids, min_auth_level, arn, created_at
         FROM corpus.buckets
         WHERE org_id = $1 AND (($2::uuid IS NOT NULL AND id = $2) OR slug = $3)
         LIMIT 1",
    )
    .bind(org_id)
    .bind(parsed_uuid)
    .bind(trimmed)
    .fetch_optional(&mut **tx.tx_mut())
    .await
    .map_err(|e| OpenRagError::Database(format!("failed querying corpus.buckets: {e}")))?
    .ok_or_else(|| OpenRagError::NotFound(format!("bucket '{trimmed}' not found")))?;

    row_to_bucket(&row)
}

pub(crate) async fn resolve_org_and_dept_paths(
    state: &AppState,
    org_id: Uuid,
    dept_id: Uuid,
) -> Result<(String, String), OpenRagError> {
    let worker = state.require_worker_pool()?;
    let org_slug: String = sqlx::query_scalar("SELECT slug FROM iam.orgs WHERE id = $1")
        .bind(org_id)
        .fetch_optional(worker.inner())
        .await
        .map_err(|e| OpenRagError::Database(format!("failed querying iam.orgs: {e}")))?
        .ok_or_else(|| OpenRagError::NotFound(format!("org {org_id} not found")))?;

    let dept_path: String =
        sqlx::query_scalar("SELECT arn_path FROM iam.departments WHERE org_id = $1 AND id = $2")
            .bind(org_id)
            .bind(dept_id)
            .fetch_optional(worker.inner())
            .await
            .map_err(|e| OpenRagError::Database(format!("failed querying iam.departments: {e}")))?
            .ok_or_else(|| OpenRagError::NotFound(format!("department {dept_id} not found")))?;

    Ok((org_slug, dept_path))
}

async fn health_handler(State(state): State<Arc<AppState>>) -> Response {
    let mut db_connected = false;
    if let Some(pool) = state.app_pool.as_ref().or(state.worker_pool.as_ref()) {
        match pool.health_check().await {
            Ok(()) => db_connected = true,
            Err(e) => {
                return (
                    StatusCode::SERVICE_UNAVAILABLE,
                    Json(json!({
                        "status": "error",
                        "mode": state.mode.as_str(),
                        "db_connected": false,
                        "error": e.to_string(),
                    })),
                )
                    .into_response();
            }
        }
    }

    (
        StatusCode::OK,
        Json(json!({
            "status": "ok",
            "mode": state.mode.as_str(),
            "db_connected": db_connected,
        })),
    )
        .into_response()
}

#[derive(Debug, Deserialize)]
struct CreateBucketRequest {
    slug: String,
    name: Option<String>,
    #[serde(default, alias = "scope")]
    access_scope: Option<AccessScope>,
    min_auth_level: Option<i16>,
    dept_id: Option<Uuid>,
    #[serde(default)]
    allowed_group_ids: Vec<Uuid>,
    #[serde(default)]
    denied_group_ids: Vec<Uuid>,
    pipeline_config_id: Option<Uuid>,
}

async fn list_buckets_handler(
    State(state): State<Arc<AppState>>,
    AuthPrincipal(principal): AuthPrincipal,
) -> Result<Json<Value>, ApiError> {
    let app_pool = state.require_app_pool().map_err(ApiError::from)?;
    let mut tx = app_pool
        .begin_rls_tx(&principal)
        .await
        .map_err(ApiError::from)?;

    let rows = sqlx::query(
        "SELECT id, org_id, dept_id, owner_principal_id, slug, name, access_scope,
                allowed_group_ids, denied_group_ids, min_auth_level, arn, created_at
         FROM corpus.buckets
         WHERE org_id = $1
         ORDER BY created_at DESC",
    )
    .bind(principal.org_id)
    .fetch_all(&mut **tx.tx_mut())
    .await
    .map_err(|e| ApiError::from(OpenRagError::Database(format!("list corpus.buckets: {e}"))))?;

    let mut buckets = Vec::with_capacity(rows.len());
    for row in &rows {
        buckets.push(row_to_bucket(row).map_err(ApiError::from)?);
    }
    tx.commit()
        .await
        .map_err(|e| ApiError::from(OpenRagError::Database(format!("commit list_buckets: {e}"))))?;

    Ok(Json(json!({ "buckets": buckets })))
}

async fn create_bucket_handler(
    State(state): State<Arc<AppState>>,
    AuthPrincipal(principal): AuthPrincipal,
    Json(req): Json<CreateBucketRequest>,
) -> Result<(StatusCode, Json<BucketRecord>), ApiError> {
    let slug = req.slug.trim();
    if slug.is_empty() {
        return Err(ApiError::bad_request("invalid_slug", "bucket slug must not be empty"));
    }
    let name = req
        .name
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .unwrap_or(slug)
        .to_string();

    let access_scope = req.access_scope.unwrap_or(AccessScope::Dept);
    let min_level_i16 = req.min_auth_level.unwrap_or(1);
    let min_auth_level = AuthLevel::try_from(min_level_i16).map_err(ApiError::from)?;

    let worker_pool = state.require_worker_pool().map_err(ApiError::from)?;
    let dept_id = match req.dept_id.or_else(|| principal.dept_ids.first().copied()) {
        Some(id) => id,
        None => sqlx::query_scalar::<_, Uuid>(
            "SELECT id FROM iam.departments WHERE org_id = $1 ORDER BY created_at ASC LIMIT 1",
        )
        .bind(principal.org_id)
        .fetch_optional(worker_pool.inner())
        .await
        .map_err(|e| ApiError::from(OpenRagError::Database(format!("lookup department: {e}"))))?
        .ok_or_else(|| ApiError::bad_request("missing_department", "organization has no departments"))?,
    };

    let (org_slug, dept_path) = resolve_org_and_dept_paths(&state, principal.org_id, dept_id)
        .await
        .map_err(ApiError::from)?;

    let bucket_id = Uuid::now_v7();
    let arn = Arn::new(
        &org_slug,
        &dept_path,
        access_scope,
        ArnResourceType::Bucket,
        min_auth_level,
        bucket_id.to_string(),
    )
    .map_err(ApiError::from)?
    .to_canonical();

    enforce_api_key_arn_scope(&principal, &arn).map_err(ApiError::from)?;

    let app_pool = state.require_app_pool().map_err(ApiError::from)?;
    let mut tx = app_pool
        .begin_rls_tx(&principal)
        .await
        .map_err(ApiError::from)?;

    let row = sqlx::query(
        "INSERT INTO corpus.buckets (
            id, org_id, dept_id, owner_principal_id, slug, name,
            access_scope, allowed_group_ids, denied_group_ids, min_auth_level,
            arn, pipeline_config_id
         ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
         RETURNING id, org_id, dept_id, owner_principal_id, slug, name,
                   access_scope, allowed_group_ids, denied_group_ids, min_auth_level,
                   arn, created_at",
    )
    .bind(bucket_id)
    .bind(principal.org_id)
    .bind(dept_id)
    .bind(principal.principal_id)
    .bind(slug)
    .bind(&name)
    .bind(access_scope.as_str())
    .bind(&req.allowed_group_ids)
    .bind(&req.denied_group_ids)
    .bind(min_auth_level.as_i16())
    .bind(&arn)
    .bind(req.pipeline_config_id)
    .fetch_one(&mut **tx.tx_mut())
    .await
    .map_err(map_sqlx_err("insert corpus.buckets"))?;

    let created = row_to_bucket(&row).map_err(ApiError::from)?;
    tx.commit()
        .await
        .map_err(|e| ApiError::from(OpenRagError::Database(format!("commit create_bucket: {e}"))))?;

    Ok((StatusCode::CREATED, Json(created)))
}

async fn get_bucket_handler(
    State(state): State<Arc<AppState>>,
    AuthPrincipal(principal): AuthPrincipal,
    Path(id): Path<String>,
) -> Result<Json<BucketRecord>, ApiError> {
    let app_pool = state.require_app_pool().map_err(ApiError::from)?;
    let mut tx = app_pool
        .begin_rls_tx(&principal)
        .await
        .map_err(ApiError::from)?;
    let bucket = resolve_bucket_rls(&mut tx, principal.org_id, &id)
        .await
        .map_err(ApiError::from)?;
    enforce_api_key_arn_scope(&principal, &bucket.arn).map_err(ApiError::from)?;
    tx.commit()
        .await
        .map_err(|e| ApiError::from(OpenRagError::Database(format!("commit get_bucket: {e}"))))?;
    Ok(Json(bucket))
}


#[derive(Debug, Default, Deserialize)]
struct CreateDocumentJsonRequest {
    title: Option<String>,
    text: Option<String>,
    body_md: Option<String>,
    #[serde(default, alias = "scope")]
    access_scope: Option<AccessScope>,
    min_auth_level: Option<i16>,
    #[serde(default)]
    allowed_group_ids: Vec<Uuid>,
    #[serde(default)]
    denied_group_ids: Vec<Uuid>,
    sensitivity: Option<Sensitivity>,
    #[serde(default, alias = "mode")]
    ocr_mode: Option<Sensitivity>,
    #[serde(default)]
    batch: bool,
    #[serde(default)]
    tags: Vec<String>,
    external_backend: Option<String>,
    filterable_metadata: Option<Map<String, Value>>,
    revision: Option<i32>,
    valid_from: Option<DateTime<Utc>>,
    valid_to: Option<DateTime<Utc>>,
    source_uri: Option<String>,
    slug: Option<String>,
    external_key: Option<String>,
}

pub(crate) struct IngestPayload {
    pub(crate) file_bytes: Option<Vec<u8>>,
    pub(crate) filename: Option<String>,
    pub(crate) mime_type: Option<String>,
    pub(crate) title: Option<String>,
    pub(crate) text: Option<String>,
    pub(crate) access_scope: Option<AccessScope>,
    pub(crate) min_auth_level: Option<i16>,
    pub(crate) allowed_group_ids: Vec<Uuid>,
    pub(crate) denied_group_ids: Vec<Uuid>,
    pub(crate) sensitivity: Option<Sensitivity>,
    pub(crate) ocr_mode: Option<Sensitivity>,
    pub(crate) batch: bool,
    pub(crate) tags: Vec<String>,
    pub(crate) external_backend: Option<String>,
    pub(crate) filterable_metadata: Option<Map<String, Value>>,
    pub(crate) revision: Option<i32>,
    pub(crate) valid_from: Option<DateTime<Utc>>,
    pub(crate) valid_to: Option<DateTime<Utc>>,
    pub(crate) source_uri: Option<String>,
    pub(crate) slug: Option<String>,
    pub(crate) external_key: Option<String>,
    pub(crate) bucket_ref: Option<String>,
}

async fn upload_or_create_document_handler(
    State(state): State<Arc<AppState>>,
    AuthPrincipal(principal): AuthPrincipal,
    Path(id): Path<String>,
    req: Request,
) -> Result<(StatusCode, Json<Value>), ApiError> {
    let content_type = req
        .headers()
        .get(CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("application/json")
        .to_string();

    let payload = if content_type.to_ascii_lowercase().starts_with("multipart/form-data") {
        let multipart = Multipart::from_request(req, &state)
            .await
            .map_err(|e| ApiError::bad_request("invalid_multipart", format!("multipart error: {e}")))?;
        parse_multipart_ingest(multipart).await?
    } else {
        let body = Bytes::from_request(req, &state)
            .await
            .map_err(|e| ApiError::bad_request("invalid_body", format!("failed reading body: {e}")))?;
        let req_json: CreateDocumentJsonRequest = serde_json::from_slice(&body)
            .map_err(|e| ApiError::bad_request("invalid_json", format!("malformed JSON: {e}")))?;
        IngestPayload {
            file_bytes: None,
            filename: None,
            mime_type: Some("text/plain".into()),
            title: req_json.title,
            text: req_json.text.or(req_json.body_md),
            access_scope: req_json.access_scope,
            min_auth_level: req_json.min_auth_level,
            allowed_group_ids: req_json.allowed_group_ids,
            denied_group_ids: req_json.denied_group_ids,
            sensitivity: req_json.sensitivity,
            ocr_mode: req_json.ocr_mode,
            batch: req_json.batch,
            tags: req_json.tags,
            external_backend: req_json.external_backend,
            filterable_metadata: req_json.filterable_metadata,
            revision: req_json.revision,
            valid_from: req_json.valid_from,
            valid_to: req_json.valid_to,
            source_uri: req_json.source_uri,
            slug: req_json.slug,
            external_key: req_json.external_key,
            bucket_ref: None,
        }
    };

    let app_pool = state.require_app_pool().map_err(ApiError::from)?;
    let bucket = {
        let mut tx = app_pool
            .begin_rls_tx(&principal)
            .await
            .map_err(ApiError::from)?;
        let b = resolve_bucket_rls(&mut tx, principal.org_id, &id)
            .await
            .map_err(ApiError::from)?;
        enforce_api_key_arn_scope(&principal, &b.arn).map_err(ApiError::from)?;
        tx.commit().await.map_err(|e| {
            ApiError::from(OpenRagError::Database(format!("commit bucket resolve: {e}")))
        })?;
        b
    };

    let result = ingest_document_into_bucket(&state, &principal, &bucket, payload)
        .await
        .map_err(ApiError::from)?;

    Ok((ingest_status(&result), Json(result)))
}

pub(crate) async fn parse_multipart_ingest(mut multipart: Multipart) -> Result<IngestPayload, ApiError> {

    let mut payload = IngestPayload {
        file_bytes: None,
        filename: None,
        mime_type: None,
        title: None,
        text: None,
        access_scope: None,
        min_auth_level: None,
        allowed_group_ids: Vec::new(),
        denied_group_ids: Vec::new(),
        sensitivity: None,
        ocr_mode: None,
        batch: false,
        tags: Vec::new(),
        external_backend: None,
        filterable_metadata: None,
        revision: None,
        valid_from: None,
        valid_to: None,
        source_uri: None,
        slug: None,
        external_key: None,
        bucket_ref: None,
    };

    while let Some(field) = multipart
        .next_field()
        .await
        .map_err(|e| ApiError::bad_request("invalid_multipart", format!("multipart error: {e}")))?
    {
        let name = field.name().unwrap_or("").to_string();
        let field_filename = field.file_name().map(ToString::to_string);
        let field_mime = field.content_type().map(ToString::to_string);
        let data = field
            .bytes()
            .await
            .map_err(|e| ApiError::bad_request("invalid_multipart", format!("read field: {e}")))?;

        match name.as_str() {
            "file" | "document" => {
                payload.filename = field_filename;
                payload.mime_type = field_mime;
                payload.file_bytes = Some(data.to_vec());
            }
            "title" => payload.title = non_empty_utf8(&data),
            "text" | "body_md" => payload.text = Some(String::from_utf8_lossy(&data).to_string()),
            "scope" | "access_scope" => {
                if let Some(s) = non_empty_utf8(&data) {
                    payload.access_scope = Some(s.parse().map_err(ApiError::from)?);
                }
            }
            "min_auth_level" => {
                if let Some(s) = non_empty_utf8(&data) {
                    let lvl: i16 = s.parse().map_err(|_| {
                        ApiError::bad_request("invalid_auth_level", "min_auth_level must be integer")
                    })?;
                    payload.min_auth_level = Some(lvl);
                }
            }
            "allowed_group_ids" => {
                payload
                    .allowed_group_ids
                    .extend(parse_uuid_list_field(&data)?);
            }
            "denied_group_ids" => {
                payload
                    .denied_group_ids
                    .extend(parse_uuid_list_field(&data)?);
            }
            "sensitivity" => {
                if let Some(s) = non_empty_utf8(&data) {
                    payload.sensitivity = Some(parse_sensitivity(&s)?);
                }
            }
            "mode" | "ocr_mode" => {
                if let Some(s) = non_empty_utf8(&data) {
                    payload.ocr_mode = Some(parse_sensitivity(&s)?);
                }
            }
            "batch" => {
                if let Some(s) = non_empty_utf8(&data) {
                    payload.batch = matches!(s.to_ascii_lowercase().as_str(), "true" | "1" | "yes");
                }
            }
            "tags" => {
                payload.tags.extend(parse_string_list_field(&data));
            }
            "external_backend" => payload.external_backend = non_empty_utf8(&data),
            "filterable_metadata" => {
                if let Some(s) = non_empty_utf8(&data) {
                    let map: Map<String, Value> = serde_json::from_str(&s).map_err(|e| {
                        ApiError::bad_request(
                            "invalid_metadata",
                            format!("filterable_metadata must be JSON object: {e}"),
                        )
                    })?;
                    payload.filterable_metadata = Some(map);
                }
            }
            "revision" => {
                if let Some(s) = non_empty_utf8(&data) {
                    let rev: i32 = s.parse().map_err(|_| {
                        ApiError::bad_request("invalid_revision", "revision must be integer")
                    })?;
                    payload.revision = Some(rev);
                }
            }
            "valid_from" => {
                if let Some(s) = non_empty_utf8(&data) {
                    payload.valid_from = Some(parse_datetime(&s)?);
                }
            }
            "valid_to" => {
                if let Some(s) = non_empty_utf8(&data) {
                    payload.valid_to = Some(parse_datetime(&s)?);
                }
            }
            "source_uri" => payload.source_uri = non_empty_utf8(&data),
            "slug" => payload.slug = non_empty_utf8(&data),
            "external_key" => payload.external_key = non_empty_utf8(&data),
            "bucket" | "bucket_id" => payload.bucket_ref = non_empty_utf8(&data),
            _ => {}
        }
    }

    Ok(payload)
}

pub(crate) async fn ingest_document_into_bucket(
    state: &AppState,
    principal: &PrincipalContext,
    bucket: &BucketRecord,
    payload: IngestPayload,
) -> Result<Value, OpenRagError> {
    let pipeline = state.pipeline_config.read().await.clone();
    let pii_engine = state.pii_engine.read().await.clone();

    let access_scope = payload
        .access_scope
        .unwrap_or_else(|| bucket.access_scope.parse().unwrap_or(AccessScope::Dept));

    let default_doc_level = (bucket.min_auth_level.max(1) as u8).min(principal.auth_level.as_u8());
    let doc_level = match payload.min_auth_level {
        Some(lvl) => AuthLevel::try_from(lvl)?,
        None => AuthLevel::try_from(default_doc_level)?,
    };

    let mut sensitivity = payload.sensitivity.unwrap_or(Sensitivity::Normal);
    let parse_ctx = ParsePolicyCtx {
        auth_level: doc_level,
        sensitivity,
        requested_ocr_mode: payload.ocr_mode,
        batch: payload.batch,
        ocr_task: OcrTask::Markdown,
        tenant_key: Some(principal.org_id.to_string()),
    };

    let mut ocr_decision = resolve_ocr_policy(&pipeline.parse, &parse_ctx)?;
    if doc_level.as_u8() >= 8 {
        if ocr_decision.effective_mode == Sensitivity::Share {
            ocr_decision.effective_mode = Sensitivity::Zdr;
            ocr_decision.upgraded_from_share = true;
        }
        if sensitivity == Sensitivity::Share {
            sensitivity = Sensitivity::Private;
        }
    }

    // Re-ingesting what is already there changes nothing: the same content with the same access
    // settings returns the document in force. The same external_key with new content becomes
    // that document's next revision, and the previous one goes out of force.
    let incoming_sha = match (&payload.file_bytes, &payload.text) {
        (Some(b), _) if !b.is_empty() => Some(sha256_hex_bytes(b)),
        (None, Some(t)) => Some(sha256_hex_bytes(t.as_bytes())),
        _ => None,
    };
    let doc_key = payload
        .external_key
        .as_deref()
        .map(str::trim)
        .filter(|k| !k.is_empty())
        .map(str::to_string);
    // A changed file with the same external_key replaces the in-force copy: the old document is
    // closed (valid_to) and the new one records `supersedes`. The revision number is left to the
    // caller, because retrieval boosts higher revisions across all hits, not per document.
    let mut superseded: Option<Uuid> = None;
    if let Some(sha) = incoming_sha.as_deref() {
        let app_pool = state.require_app_pool()?;
        let mut tx = app_pool.begin_rls_tx(principal).await?;
        let same = sqlx::query(
            "SELECT id, title, revision FROM corpus.documents
             WHERE org_id = $1 AND bucket_id = $2 AND content_sha256 = $3
               AND access_scope = $4 AND min_auth_level = $5
               AND allowed_group_ids = $6 AND denied_group_ids = $7
               AND status IN ('indexed', 'pending')
               AND (valid_to IS NULL OR valid_to > now())
             ORDER BY created_at DESC
             LIMIT 1",
        )
        .bind(principal.org_id)
        .bind(bucket.id)
        .bind(sha)
        .bind(access_scope.as_str())
        .bind(i16::from(doc_level.as_u8()))
        .bind(&payload.allowed_group_ids)
        .bind(&payload.denied_group_ids)
        .fetch_optional(&mut **tx.tx_mut())
        .await
        .map_err(|e| OpenRagError::Database(format!("lookup identical document: {e}")))?;
        if let Some(row) = same {
            let id: Uuid = row.try_get("id").map_err(|e| OpenRagError::Database(e.to_string()))?;
            let title: String = row.try_get("title").map_err(|e| OpenRagError::Database(e.to_string()))?;
            let revision: i32 = row.try_get("revision").map_err(|e| OpenRagError::Database(e.to_string()))?;
            tx.commit()
                .await
                .map_err(|e| OpenRagError::Database(format!("commit identical lookup: {e}")))?;
            return Ok(json!({
                "id": id,
                "document_id": id,
                "bucket_id": bucket.id,
                "title": title,
                "revision": revision,
                "min_auth_level": doc_level.as_u8(),
                "access_scope": access_scope.as_str(),
                "unchanged": true,
            }));
        }
        if let Some(key) = doc_key.as_deref() {
            let prior = sqlx::query(
                "SELECT id FROM corpus.documents
                 WHERE org_id = $1 AND bucket_id = $2 AND metadata->>'external_key' = $3
                   AND status IN ('indexed', 'pending')
                   AND (valid_to IS NULL OR valid_to > now())
                 ORDER BY created_at DESC
                 LIMIT 1",
            )
            .bind(principal.org_id)
            .bind(bucket.id)
            .bind(key)
            .fetch_optional(&mut **tx.tx_mut())
            .await
            .map_err(|e| OpenRagError::Database(format!("lookup document by external_key: {e}")))?;
            if let Some(row) = prior {
                let id: Uuid = row.try_get("id").map_err(|e| OpenRagError::Database(e.to_string()))?;
                superseded = Some(id);
            }
        }
        tx.commit()
            .await
            .map_err(|e| OpenRagError::Database(format!("commit document lookup: {e}")))?;
    }

    let (mut parsed, source_bytes_opt) = if let Some(bytes) = payload.file_bytes {
        if bytes.is_empty() {
            return Err(OpenRagError::Validation("cannot ingest empty 0-byte file".into()));
        }
        let parsed_doc = state
            .parser
            .parse(
                &bytes,
                payload.mime_type.as_deref(),
                payload.filename.as_deref(),
                &pipeline.parse,
                &parse_ctx,
            )
            .await?;
        (parsed_doc, Some(bytes))
    } else if let Some(ref text_val) = payload.text {
        let title_str = payload
            .title
            .as_deref()
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .unwrap_or("Untitled Document")
            .to_string();
        let aliases = extract_parenthetical_pairs(text_val);
        let page = ParsedPage {
            page_no: 1,
            raw_markdown: text_val.clone(),
            tables_html: Vec::new(),
            formulas: Vec::new(),
            fields: Vec::new(),
            blocks: Vec::new(),
            extracted_aliases: aliases.clone(),
            page_image_rgb: None,
            content_sha256: sha256_hex_str(text_val),
        };
        let parsed_doc = ParsedDocument {
            title: title_str,
            format: DocumentFormat::PlainText,
            mime_type: payload
                .mime_type
                .clone()
                .unwrap_or_else(|| "text/plain".to_string()),
            content_sha256: sha256_hex_bytes(text_val.as_bytes()),
            pages: vec![page],
            full_markdown: text_val.clone(),
            extracted_aliases: aliases,
            ocr_policy: Some(ocr_decision.clone()),
            parser_engine: "json_text".to_string(),
        };
        (parsed_doc, None)
    } else {
        return Err(OpenRagError::Validation(
            "document upload requires multipart 'file' or JSON 'text'".into(),
        ));
    };

    // Layer 1: Neutralize any caller-supplied [PII:<TAG>:<HEX>] literals BEFORE thumbnail/guard/persist/seal
    parsed.title = neutralize_literal_pii_tokens(&parsed.title);
    parsed.full_markdown = neutralize_literal_pii_tokens(&parsed.full_markdown);
    for page in &mut parsed.pages {
        page.raw_markdown = neutralize_literal_pii_tokens(&page.raw_markdown);
    }

    let thumbs = if pipeline.thumbnail.enabled {
        state
            .thumbnailer
            .render(&parsed, source_bytes_opt.as_deref(), &pipeline.thumbnail)?
    } else {
        Vec::new()
    };

    let worker_pool = state.require_worker_pool()?;
    let app_pool = state.require_app_pool()?;

    // Validate S3 Vectors metadata caps BEFORE any DB insert
    let uses_s3_vectors = payload.external_backend.as_deref() == Some("s3_vectors")
        || pipeline.store.external_mirror == ExternalMirrorBackend::S3Vectors
        || has_s3_vectors_binding(worker_pool.inner(), principal.org_id).await?;
    if uses_s3_vectors {
        let raw_meta = payload.filterable_metadata.clone().unwrap_or_default();
        let now_micros = Utc::now().timestamp_micros();
        let preview_id = Uuid::now_v7().to_string();
        let _ = encode_s3_metadata(
            &preview_id,
            now_micros,
            Some(&parsed.full_markdown),
            &raw_meta,
            &S3MetadataSchema::default(),
        )?;
    }

    // Ingestion Gate Guard BEFORE DB insert
    let policy = load_active_guard_policy(Some(worker_pool), principal.org_id, &pipeline).await?;
    let concept_lookup = load_concept_lookup(
        Some(worker_pool),
        &state.in_memory_concepts,
        principal.org_id,
    )
    .await?;

    let guard_decision = GuardEngine::evaluate(
        &parsed.full_markdown,
        GuardSurface::IngestionGate,
        &policy,
        &pii_engine,
        Some(&concept_lookup),
        None,
    )?;

    if !guard_decision.matched_rules.is_empty() {
        let mut gtx = app_pool.begin_rls_tx(principal).await?;
        let _ = GuardRepo::log_guard_event_rls(
            &mut gtx,
            principal.org_id,
            policy.id,
            principal.principal_id,
            GuardSurface::IngestionGate,
            &parsed.full_markdown,
            &guard_decision,
        )
        .await?;
        gtx.commit()
            .await
            .map_err(|e| OpenRagError::Database(format!("commit ingestion guard event: {e}")))?;
    }

    if guard_decision.action == GuardAction::Block {
        let rule_ids: Vec<Uuid> = guard_decision
            .matched_rules
            .iter()
            .filter_map(|m| m.rule_id)
            .collect();
        let cats: Vec<String> = guard_decision
            .matched_rules
            .iter()
            .map(|m| m.pattern_or_category.clone())
            .collect();
        return Err(OpenRagError::GuardBlocked {
            surface: GuardSurface::IngestionGate,
            reason: format!(
                "blocked by {} guard rule(s) on ingestion_gate",
                guard_decision.matched_rules.len()
            ),
            matched_rule_ids: rule_ids,
            matched_categories: cats,
        });
    }

    let (org_slug, dept_path) =
        resolve_org_and_dept_paths(state, principal.org_id, bucket.dept_id).await?;

    let norm_tags = normalize_tags_dedup(&payload.tags);
    let doc_title = neutralize_literal_pii_tokens(
        &payload
            .title
            .as_deref()
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(ToString::to_string)
            .unwrap_or_else(|| parsed.title.clone()),
    );
    let source_uri = payload
        .source_uri
        .clone()
        .or_else(|| payload.external_key.clone())
        .or_else(|| payload.slug.clone())
        .or_else(|| payload.filename.clone());

    let persist_input = PersistParsedDocumentInput {
        org_id: principal.org_id,
        org_slug: org_slug.clone(),
        dept_id: bucket.dept_id,
        dept_path: dept_path.clone(),
        bucket_id: bucket.id,
        owner_principal_id: principal.principal_id,
        access_scope,
        allowed_group_ids: payload.allowed_group_ids.clone(),
        denied_group_ids: payload.denied_group_ids.clone(),
        min_auth_level: doc_level,
        sensitivity,
        title: doc_title.clone(),
        source_uri: source_uri.clone(),
    };

    let mut tx = app_pool.begin_rls_tx(principal).await?;
    let persisted = ParsePersister::persist_parsed_document(
        &mut tx,
        &persist_input,
        &parsed,
        &thumbs,
        state.blob_store.as_ref(),
    )
    .await?;

    let revision = payload.revision.unwrap_or(1);
    let valid_from = payload.valid_from.unwrap_or_else(Utc::now);
    if let Some(old_id) = superseded {
        sqlx::query(
            "UPDATE corpus.documents
             SET valid_to = $1, updated_at = now()
             WHERE org_id = $2 AND id = $3 AND (valid_to IS NULL OR valid_to > $1)",
        )
        .bind(valid_from)
        .bind(principal.org_id)
        .bind(old_id)
        .execute(&mut **tx.tx_mut())
        .await
        .map_err(|e| OpenRagError::Database(format!("close previous revision: {e}")))?;
    }
    let doc_extra_meta = json!({
        "ocr_mode": ocr_decision.effective_mode.as_str(),
        "slug": payload.slug,
        "external_key": payload.external_key,
        "filterable_metadata": payload.filterable_metadata,
        "external_backend": payload.external_backend,
    });
    sqlx::query(
        "UPDATE corpus.documents
         SET tags = $1,
             revision = $2,
             valid_from = $3,
             valid_to = $4,
             metadata = metadata || $5
         WHERE org_id = $6 AND id = $7",
    )
    .bind(&norm_tags)
    .bind(revision)
    .bind(valid_from)
    .bind(payload.valid_to)
    .bind(doc_extra_meta)
    .bind(principal.org_id)
    .bind(persisted.document_id)
    .execute(&mut **tx.tx_mut())
    .await
    .map_err(|e| OpenRagError::Database(format!("update corpus.documents metadata: {e}")))?;

    let acl_scope_key = format!(
        "{}:{}:{}:L{}",
        principal.org_id,
        access_scope.as_str(),
        bucket.dept_id,
        doc_level.as_u8()
    );

    let mut chunk_ids = Vec::new();
    let mut global_chunk_idx: i32 = 0;

    let mask_entire_doc = guard_decision.action == GuardAction::Mask;
    for (page_idx, page) in parsed.pages.iter().enumerate() {
        let page_id = persisted.page_ids.get(page_idx).copied();
        let page_raw_text = if mask_entire_doc {
            if parsed.pages.len() == 1 {
                guard_decision
                    .sanitized_text
                    .clone()
                    .unwrap_or_else(|| page.raw_markdown.clone())
            } else {
                let page_guard = GuardEngine::evaluate(
                    &page.raw_markdown,
                    GuardSurface::IngestionGate,
                    &policy,
                    &pii_engine,
                    Some(&concept_lookup),
                    None,
                )?;
                if page_guard.action == GuardAction::Block {
                    if !page_guard.matched_rules.is_empty() {
                        let mut gtx = app_pool.begin_rls_tx(principal).await?;
                        let _ = GuardRepo::log_guard_event_rls(
                            &mut gtx,
                            principal.org_id,
                            policy.id,
                            principal.principal_id,
                            GuardSurface::IngestionGate,
                            &page.raw_markdown,
                            &page_guard,
                        )
                        .await?;
                        gtx.commit().await.map_err(|e| {
                            OpenRagError::Database(format!("commit ingestion guard event: {e}"))
                        })?;
                    }
                    let rule_ids: Vec<Uuid> = page_guard
                        .matched_rules
                        .iter()
                        .filter_map(|m| m.rule_id)
                        .collect();
                    let cats: Vec<String> = page_guard
                        .matched_rules
                        .iter()
                        .map(|m| m.pattern_or_category.clone())
                        .collect();
                    return Err(OpenRagError::GuardBlocked {
                        surface: GuardSurface::IngestionGate,
                        reason: format!(
                            "blocked by {} guard rule(s) on ingestion_gate",
                            page_guard.matched_rules.len()
                        ),
                        matched_rule_ids: rule_ids,
                        matched_categories: cats,
                    });
                }
                page_guard
                    .sanitized_text
                    .unwrap_or_else(|| page.raw_markdown.clone())
            }
        } else {
            page.raw_markdown.clone()
        };

        // PII is kept: stored text, BM25 and embeddings stay local. It is masked only on the
        // way to an AI (LLM provider, MCP client); see llm_router::seal_text_for_llm.
        let final_page_redacted = page_raw_text.clone();
        if final_page_redacted != page.raw_markdown {
            if let Some(pid) = page_id {
                sqlx::query(
                    "UPDATE corpus.pages SET redacted_markdown = $1 WHERE org_id = $2 AND id = $3",
                )
                .bind(&final_page_redacted)
                .bind(principal.org_id)
                .bind(pid)
                .execute(&mut **tx.tx_mut())
                .await
                .map_err(|e| {
                    OpenRagError::Database(format!("update corpus.pages redacted_markdown: {e}"))
                })?;
            }
        }

        let table_html = if page.tables_html.is_empty() {
            None
        } else {
            Some(page.tables_html.join("\n"))
        };

        let page_chunks = chunk_page(
            pipeline.store.chunker,
            &page_raw_text,
            1000,
            &doc_title,
            state.text_embedder.as_ref(),
        )
        .await?;
        let embed_title = pipeline
            .store
            .embed_document_title
            .then_some(doc_title.as_str());
        for chunk_text in page_chunks {
            let emb = state
                .text_embedder
                .embed_document(embed_title, &chunk_text)
                .await?;
            let text_hash = sha256_hex_str(&chunk_text);
            let chunk_id = Uuid::now_v7();
            let chunk_arn = Arn::new(
                &org_slug,
                &dept_path,
                access_scope,
                ArnResourceType::Chunk,
                doc_level,
                chunk_id.to_string(),
            )?
            .to_canonical();

            sqlx::query(
                "INSERT INTO corpus.chunks (
                    id, org_id, document_id, page_id, dept_id, owner_principal_id,
                    access_scope, allowed_group_ids, denied_group_ids, min_auth_level,
                    arn, acl_scope_key, chunk_idx, redacted_text, bm25_text,
                    table_html, text_hash, embedding, embedding_model,
                    valid_from, valid_to, tags
                 ) VALUES (
                    $1, $2, $3, $4, $5, $6,
                    $7, $8, $9, $10,
                    $11, $12, $13, $14, $15,
                    $16, $17, $18, $19,
                    $20, $21, $22
                 )",
            )
            .bind(chunk_id)
            .bind(principal.org_id)
            .bind(persisted.document_id)
            .bind(page_id)
            .bind(bucket.dept_id)
            .bind(principal.principal_id)
            .bind(access_scope.as_str())
            .bind(&payload.allowed_group_ids)
            .bind(&payload.denied_group_ids)
            .bind(doc_level.as_i16())
            .bind(&chunk_arn)
            .bind(&acl_scope_key)
            .bind(global_chunk_idx)
            .bind(&chunk_text)
            .bind(&chunk_text)
            .bind(&table_html)
            .bind(&text_hash)
            .bind(Vector::from(emb))
            .bind(state.text_embedder.model_name())
            .bind(valid_from)
            .bind(payload.valid_to)
            .bind(&norm_tags)
            .execute(&mut **tx.tx_mut())
            .await
            .map_err(|e| OpenRagError::Database(format!("insert corpus.chunks: {e}")))?;

            let _ = OutboxEnqueuer::enqueue_rls(
                &mut tx,
                &EnqueueOutboxItem {
                    org_id: principal.org_id,
                    target_table: VectorTable::CorpusChunks,
                    target_id: chunk_id,
                    op: OutboxOp::UpsertEmbed,
                    content_sha256: Some(text_hash),
                    priority: 4,
                    payload: json!({ "document_id": persisted.document_id, "chunk_idx": global_chunk_idx }),
                    idempotency_key: None,
                    max_attempts: None,
                },
                false,
            )
            .await?;

            chunk_ids.push(chunk_id);
            global_chunk_idx += 1;
        }
    }

    let extracted_pairs = ConceptGraph::ingest_parenthetical_pairs_rls(
        &mut tx,
        principal.org_id,
        Some(principal.principal_id),
        &parsed.full_markdown,
    )
    .await?;

    tx.commit()
        .await
        .map_err(|e| OpenRagError::Database(format!("commit document ingestion tx: {e}")))?;

    let _ = OutboxWorker::default()
        .drain_for_org(
            worker_pool,
            Some(principal.org_id),
            64,
            &DefaultOutboxExecutor::default(),
        )
        .await?;

    let pages_json: Vec<Value> = persisted
        .page_ids
        .iter()
        .enumerate()
        .map(|(i, pid)| {
            json!({
                "id": pid,
                "page_id": pid,
                "page_no": i + 1,
                "thumb_uri": persisted.thumb_uris.get(i),
            })
        })
        .collect();

    Ok(json!({
        "id": persisted.document_id,
        "document_id": persisted.document_id,
        "bucket_id": bucket.id,
        "arn": persisted.document_arn,
        "title": doc_title,
        "page_id": persisted.page_ids.first(),
        "pages": pages_json,
        "chunks": chunk_ids,
        "figure_ids": persisted.figure_ids,
        "thumb_uris": persisted.thumb_uris,
        "tags": norm_tags,
        "ocr_mode": ocr_decision.effective_mode.as_str(),
        "mode": ocr_decision.effective_mode.as_str(),
        "sensitivity": sensitivity.as_str(),
        "min_auth_level": doc_level.as_u8(),
        "access_scope": access_scope.as_str(),
        "revision": revision,
        "supersedes": superseded,
        "extracted_aliases": extracted_pairs,
    }))
}

/// 201 for a new document or revision, 200 when the same content was already there.
pub(crate) fn ingest_status(result: &Value) -> StatusCode {
    if result.get("unchanged").and_then(Value::as_bool) == Some(true) {
        StatusCode::OK
    } else {
        StatusCode::CREATED
    }
}

async fn delete_document_handler(
    State(state): State<Arc<AppState>>,
    AuthPrincipal(principal): AuthPrincipal,
    Path((id, doc_id)): Path<(String, Uuid)>,
) -> Result<Json<Value>, ApiError> {
    let app_pool = state.require_app_pool().map_err(ApiError::from)?;
    let worker_pool = state.require_worker_pool().map_err(ApiError::from)?;
    let mut tx = app_pool
        .begin_rls_tx(&principal)
        .await
        .map_err(ApiError::from)?;
    let bucket = resolve_bucket_rls(&mut tx, principal.org_id, &id)
        .await
        .map_err(ApiError::from)?;
    enforce_api_key_arn_scope(&principal, &bucket.arn).map_err(ApiError::from)?;

    let doc_exists: Option<Uuid> = sqlx::query_scalar(
        "SELECT id FROM corpus.documents WHERE org_id = $1 AND bucket_id = $2 AND id = $3",
    )
    .bind(principal.org_id)
    .bind(bucket.id)
    .bind(doc_id)
    .fetch_optional(&mut **tx.tx_mut())
    .await
    .map_err(|e| ApiError::from(OpenRagError::Database(format!("check corpus.documents: {e}"))))?;

    let Some(did) = doc_exists else {
        return Err(ApiError::not_found(format!("document {doc_id} not found")));
    };

    let chunk_ids: Vec<Uuid> = sqlx::query_scalar(
        "SELECT id FROM corpus.chunks WHERE org_id = $1 AND document_id = $2",
    )
    .bind(principal.org_id)
    .bind(did)
    .fetch_all(&mut **tx.tx_mut())
    .await
    .map_err(|e| ApiError::from(OpenRagError::Database(format!("fetch doc chunks: {e}"))))?;

    for cid in chunk_ids {
        let _ = OutboxEnqueuer::enqueue_rls(
            &mut tx,
            &EnqueueOutboxItem {
                org_id: principal.org_id,
                target_table: VectorTable::CorpusChunks,
                target_id: cid,
                op: OutboxOp::DeleteVector,
                content_sha256: None,
                priority: 3,
                payload: json!({ "document_id": did }),
                idempotency_key: None,
                max_attempts: None,
            },
            false,
        )
        .await
        .map_err(ApiError::from)?;
    }

    let deleted_id: Option<Uuid> = sqlx::query_scalar(
        "DELETE FROM corpus.documents WHERE org_id = $1 AND bucket_id = $2 AND id = $3 RETURNING id",
    )
    .bind(principal.org_id)
    .bind(bucket.id)
    .bind(did)
    .fetch_optional(&mut **tx.tx_mut())
    .await
    .map_err(|e| ApiError::from(OpenRagError::Database(format!("delete corpus.documents: {e}"))))?;

    let Some(did) = deleted_id else {
        return Err(ApiError::not_found(format!("document {doc_id} not found")));
    };

    tx.commit()
        .await
        .map_err(|e| ApiError::from(OpenRagError::Database(format!("commit delete_document: {e}"))))?;

    let _ = OutboxWorker::default()
        .drain_for_org(
            worker_pool,
            Some(principal.org_id),
            64,
            &DefaultOutboxExecutor::default(),
        )
        .await
        .map_err(ApiError::from)?;

    Ok(Json(json!({
        "deleted": true,
        "id": did,
        "document_id": did,
    })))
}

#[derive(Debug, Deserialize)]
struct BucketSearchHttpRequest {
    #[serde(default, alias = "message")]
    query: String,
    top_k: Option<usize>,
    fusion: Option<FusionMode>,
    rrf_k: Option<u32>,
    alpha: Option<f32>,
    rerank: Option<bool>,
    spaces: Option<Vec<RetrieveSpace>>,
    #[serde(default)]
    latest_revision_only: bool,
    latest_revision_boost: Option<f32>,
    admission_threshold: Option<f32>,
    query_vector: Option<Vec<f32>>,
    #[serde(default)]
    tags_any: Vec<String>,
    #[serde(default)]
    tags_all: Vec<String>,
    #[serde(default)]
    tags_exclude: Vec<String>,
    expand_tag_aliases: Option<bool>,
    facet_limit: Option<i64>,
    /// Reference time for validity filtering/labels (RFC 3339). Default: now.
    as_of: Option<DateTime<Utc>>,
    /// HyDE passage written by the caller (see `SearchRequest::hypothetical_answer`).
    hypothetical_answer: Option<String>,
}

async fn search_bucket_handler(
    State(state): State<Arc<AppState>>,
    AuthPrincipal(principal): AuthPrincipal,
    Path(id): Path<String>,
    Json(req): Json<BucketSearchHttpRequest>,
) -> Result<Json<Value>, ApiError> {
    let app_pool = state.require_app_pool().map_err(ApiError::from)?;
    let bucket = {
        let mut tx = app_pool
            .begin_rls_tx(&principal)
            .await
            .map_err(ApiError::from)?;
        let b = resolve_bucket_rls(&mut tx, principal.org_id, &id)
            .await
            .map_err(ApiError::from)?;
        enforce_api_key_arn_scope(&principal, &b.arn).map_err(ApiError::from)?;
        tx.commit().await.map_err(|e| {
            ApiError::from(OpenRagError::Database(format!("commit bucket lookup: {e}")))
        })?;
        b
    };

    let tag_filter = if req.tags_any.is_empty()
        && req.tags_all.is_empty()
        && req.tags_exclude.is_empty()
        && req.expand_tag_aliases.is_none()
    {
        None
    } else {
        Some(TagFilter {
            tags_any: req.tags_any,
            tags_all: req.tags_all,
            tags_exclude: req.tags_exclude,
            expand_tag_aliases: req.expand_tag_aliases.unwrap_or(false),
        })
    };

    let search_req = SearchRequest {
        query: req.query,
        query_vector: req.query_vector,
        bucket_id: Some(bucket.id),
        spaces: req.spaces,
        fusion: req.fusion,
        rrf_k: req.rrf_k,
        alpha: req.alpha,
        top_k: req.top_k,
        rerank: req.rerank,
        latest_revision_boost: req.latest_revision_boost,
        latest_revision_only: req.latest_revision_only,
        admission_threshold: req.admission_threshold,
        tag_filter,
        facet_limit: req.facet_limit.unwrap_or(50),
        as_of: req.as_of,
        hypothetical_answer: req.hypothetical_answer,
    };

    let retriever = state
        .pg_retriever(principal.org_id)
        .await
        .map_err(ApiError::from)?;
    let resp = retriever
        .search(&principal, &search_req)
        .await
        .map_err(ApiError::from)?;

    let mut hits: Vec<Value> = resp.hits.iter().map(|h| serde_json::to_value(h).unwrap_or(Value::Null)).collect();
    crate::api_docs::add_sources(&state, principal.org_id, &mut hits).await?;
    Ok(Json(json!({
        "results": &hits,
        "hits": &hits,
        "clean_query": &resp.clean_query,
        "cleaned_query": &resp.clean_query,
        "parsed_inline_tags": &resp.parsed_inline_tags,
        "parsed_excluded_tags": &resp.parsed_excluded_tags,
        "expanded_terms": &resp.expanded_terms,
        "tag_facets": &resp.tag_facets,
        "facets": {
            "tags": &resp.tag_facets,
        },
        "admission": &resp.admission,
        "max_score": resp.max_score,
        "top_score": resp.top_score,
        "top_auth_level": resp.top_auth_level,
    })))
}

#[derive(Debug, Deserialize)]
struct SyncDocumentsRequest {
    documents: Vec<SyncDocumentItem>,
}

#[derive(Debug, Deserialize)]
struct SyncDocumentItem {
    #[serde(default)]
    kind: Option<String>,
    slug: Option<String>,
    external_key: Option<String>,
    path: Option<String>,
    body_md: Option<String>,
    content: Option<String>,
    text: Option<String>,
    pages: Option<Vec<String>>,
    body_sha256: Option<String>,
    content_sha256: Option<String>,
    base_sha256: Option<String>,
    base_version: Option<i32>,
    deleted: Option<bool>,
    title: Option<String>,
    #[serde(default, alias = "scope")]
    access_scope: Option<AccessScope>,
    min_auth_level: Option<i16>,
    #[serde(default)]
    tags: Vec<String>,
}

async fn resolve_sync_item_is_wiki(
    tx: &mut RlsTransaction<'_>,
    org_id: Uuid,
    bucket_id: Uuid,
    item: &SyncDocumentItem,
) -> Result<bool, OpenRagError> {
    if let Some(ref kind_raw) = item.kind {
        return match kind_raw.trim().to_ascii_lowercase().as_str() {
            "wiki" => Ok(true),
            "corpus" => Ok(false),
            other => Err(OpenRagError::Validation(format!(
                "unknown sync item kind '{other}' (expected 'wiki' or 'corpus')"
            ))),
        };
    }

    let heuristic_wiki = item.body_md.is_some()
        || item.content.is_some()
        || (item.pages.is_none()
            && item.text.is_none()
            && item
                .path
                .as_deref()
                .is_some_and(|p| p.to_ascii_lowercase().ends_with(".md")));
    if heuristic_wiki {
        return Ok(true);
    }

    if item.deleted == Some(true) && item.pages.is_none() && item.text.is_none() {
        let corpus_key = item
            .external_key
            .as_deref()
            .or(item.slug.as_deref())
            .or(item.path.as_deref())
            .map(str::trim)
            .filter(|s| !s.is_empty());

        let has_corpus: bool = if let Some(key) = corpus_key {
            sqlx::query_scalar(
                "SELECT EXISTS(
                    SELECT 1 FROM corpus.documents
                    WHERE org_id = $1 AND bucket_id = $2
                      AND (source_uri = $3 OR metadata->>'external_key' = $3 OR metadata->>'slug' = $3 OR metadata->>'path' = $3)
                )",
            )
            .bind(org_id)
            .bind(bucket_id)
            .bind(key)
            .fetch_one(&mut **tx.tx_mut())
            .await
            .map_err(|e| OpenRagError::Database(format!("check sync corpus.documents: {e}")))?
        } else {
            false
        };

        if !has_corpus {
            let raw_wiki_slug = item
                .slug
                .as_deref()
                .or(item.external_key.as_deref())
                .or(item.path.as_deref())
                .unwrap_or("");
            let wiki_slug = slugify_simple(raw_wiki_slug);
            if !wiki_slug.is_empty() {
                let has_wiki: bool = sqlx::query_scalar(
                    "SELECT EXISTS(
                        SELECT 1 FROM wiki.pages
                        WHERE org_id = $1 AND bucket_id = $2 AND slug = $3 AND status <> 'archived'
                    )",
                )
                .bind(org_id)
                .bind(bucket_id)
                .bind(&wiki_slug)
                .fetch_one(&mut **tx.tx_mut())
                .await
                .map_err(|e| OpenRagError::Database(format!("check sync wiki.pages: {e}")))?;
                if has_wiki {
                    return Ok(true);
                }
            }
        }
    }

    Ok(false)
}

async fn sync_documents_handler(
    State(state): State<Arc<AppState>>,
    AuthPrincipal(principal): AuthPrincipal,
    Path(bucket_ref): Path<String>,
    Json(req): Json<SyncDocumentsRequest>,
) -> Result<Json<Value>, ApiError> {
    let app_pool = state.require_app_pool().map_err(ApiError::from)?;
    let worker_pool = state.require_worker_pool().map_err(ApiError::from)?;

    let pipeline = state.pipeline_config.read().await.clone();
    let pii_engine = state.pii_engine.read().await.clone();
    let policy = load_active_guard_policy(Some(worker_pool), principal.org_id, &pipeline)
        .await
        .map_err(ApiError::from)?;
    let concept_lookup = load_concept_lookup(
        Some(worker_pool),
        &state.in_memory_concepts,
        principal.org_id,
    )
    .await
    .map_err(ApiError::from)?;
    let uses_s3_vectors = pipeline.store.external_mirror == ExternalMirrorBackend::S3Vectors
        || has_s3_vectors_binding(worker_pool.inner(), principal.org_id)
            .await
            .map_err(ApiError::from)?;

    let mut tx = app_pool
        .begin_rls_tx(&principal)
        .await
        .map_err(ApiError::from)?;
    let bucket = resolve_bucket_rls(&mut tx, principal.org_id, &bucket_ref)
        .await
        .map_err(ApiError::from)?;
    enforce_api_key_arn_scope(&principal, &bucket.arn).map_err(ApiError::from)?;

    let (org_slug, dept_path) = resolve_org_and_dept_paths(&state, principal.org_id, bucket.dept_id)
        .await
        .map_err(ApiError::from)?;

    let mut pending_wiki_archives: Vec<(Uuid, String)> = Vec::new();
    let mut ctx = SyncContext {
        state: &state,
        app_pool,
        principal: &principal,
        bucket: &bucket,
        org_slug: &org_slug,
        dept_path: &dept_path,
        pii_engine: &pii_engine,
        policy: &policy,
        concept_lookup: &concept_lookup,
        uses_s3_vectors,
        embed_document_title: pipeline.store.embed_document_title,
        chunker: pipeline.store.chunker,
        pending_wiki_archives: &mut pending_wiki_archives,
    };

    let mut updated_count: usize = 0;
    let mut delta_updated_count: usize = 0;
    let mut upserted_chunks_count: usize = 0;
    let mut skipped_count: usize = 0;
    let mut deleted_count: usize = 0;
    let mut enqueued_any = false;
    let mut synced_items = Vec::with_capacity(req.documents.len());

    for item in req.documents {
        let is_wiki = resolve_sync_item_is_wiki(&mut tx, principal.org_id, bucket.id, &item)
            .await
            .map_err(ApiError::from)?;

        if is_wiki {
            let outcome = sync_wiki_item(&mut ctx, &mut tx, &item)
                .await
                .map_err(ApiError::from)?;
            if outcome.enqueued {
                enqueued_any = true;
            }
            match outcome.status.as_str() {
                "skipped" => skipped_count += 1,
                "deleted" => deleted_count += 1,
                _ => updated_count += 1,
            }
            synced_items.push(outcome.into_json());
        } else {
            let outcome = sync_corpus_item(&mut ctx, &mut tx, &item)
                .await
                .map_err(ApiError::from)?;
            if outcome.enqueued {
                enqueued_any = true;
            }
            if outcome.deleted {
                deleted_count += 1;
            } else if outcome.skipped {
                skipped_count += 1;
            } else {
                updated_count += 1;
                delta_updated_count += outcome.delta_pages;
                upserted_chunks_count += outcome.upserted_chunks;
            }
            synced_items.push(outcome.summary);
        }
    }

    for (wiki_page_id, wiki_slug) in &pending_wiki_archives {
        let _ = OutboxEnqueuer::enqueue_rls(
            &mut tx,
            &EnqueueOutboxItem {
                org_id: principal.org_id,
                target_table: VectorTable::WikiPages,
                target_id: *wiki_page_id,
                op: OutboxOp::DeleteVector,
                content_sha256: None,
                priority: 3,
                payload: json!({ "slug": wiki_slug }),
                idempotency_key: None,
                max_attempts: None,
            },
            false,
        )
        .await
        .map_err(ApiError::from)?;
        enqueued_any = true;
    }

    tx.commit()
        .await
        .map_err(|e| ApiError::from(OpenRagError::Database(format!("commit sync_documents: {e}"))))?;

    for (wiki_page_id, _) in pending_wiki_archives {
        sqlx::query(
            "UPDATE wiki.pages SET status = 'archived', updated_at = now() WHERE org_id = $1 AND id = $2",
        )
        .bind(principal.org_id)
        .bind(wiki_page_id)
        .execute(worker_pool.inner())
        .await
        .map_err(|e| ApiError::from(OpenRagError::Database(format!("archive wiki.pages: {e}"))))?;
    }

    if enqueued_any {
        let _ = OutboxWorker::default()
            .drain_for_org(
                worker_pool,
                Some(principal.org_id),
                128,
                &DefaultOutboxExecutor::default(),
            )
            .await
            .map_err(ApiError::from)?;
    }

    Ok(Json(json!({
        "updated": updated_count,
        "delta_updated": delta_updated_count,
        "upserted_chunks": upserted_chunks_count,
        "skipped": skipped_count,
        "deleted": deleted_count,
        "documents": synced_items,
    })))
}

struct SyncContext<'a> {
    state: &'a AppState,
    app_pool: &'a DbPool,
    principal: &'a PrincipalContext,
    bucket: &'a BucketRecord,
    org_slug: &'a str,
    dept_path: &'a str,
    pii_engine: &'a PiiEngine,
    policy: &'a GuardPolicyRuntime,
    concept_lookup: &'a InMemoryConceptAliasLookup,
    uses_s3_vectors: bool,
    embed_document_title: bool,
    chunker: ChunkerKind,
    pending_wiki_archives: &'a mut Vec<(Uuid, String)>,
}

async fn persist_sync_guard_event(
    ctx: &SyncContext<'_>,
    tx: &mut RlsTransaction<'_>,
    raw_text: &str,
    decision: &GuardDecision,
) -> Result<(), OpenRagError> {
    if decision.matched_rules.is_empty() {
        return Ok(());
    }
    if decision.action == GuardAction::Block {
        let mut gtx = ctx.app_pool.begin_rls_tx(ctx.principal).await?;
        let _ = GuardRepo::log_guard_event_rls(
            &mut gtx,
            ctx.principal.org_id,
            ctx.policy.id,
            ctx.principal.principal_id,
            GuardSurface::IngestionGate,
            raw_text,
            decision,
        )
        .await?;
        gtx.commit()
            .await
            .map_err(|e| OpenRagError::Database(format!("commit sync guard block event: {e}")))?;
    } else {
        let _ = GuardRepo::log_guard_event_rls(
            tx,
            ctx.principal.org_id,
            ctx.policy.id,
            ctx.principal.principal_id,
            GuardSurface::IngestionGate,
            raw_text,
            decision,
        )
        .await?;
    }
    Ok(())
}

fn build_ingestion_guard_blocked_err(decision: &GuardDecision) -> OpenRagError {
    let rule_ids: Vec<Uuid> = decision
        .matched_rules
        .iter()
        .filter_map(|m| m.rule_id)
        .collect();
    let cats: Vec<String> = decision
        .matched_rules
        .iter()
        .map(|m| m.pattern_or_category.clone())
        .collect();
    OpenRagError::GuardBlocked {
        surface: GuardSurface::IngestionGate,
        reason: format!(
            "blocked by {} guard rule(s) on ingestion_gate",
            decision.matched_rules.len()
        ),
        matched_rule_ids: rule_ids,
        matched_categories: cats,
    }
}

struct WikiSyncOutcome {
    id: Option<Uuid>,
    slug: String,
    status: String,
    enqueued: bool,
}

impl WikiSyncOutcome {
    fn into_json(self) -> Value {
        json!({
            "kind": "wiki",
            "id": self.id,
            "slug": self.slug,
            "status": self.status,
        })
    }
}

async fn sync_wiki_item(
    ctx: &mut SyncContext<'_>,
    tx: &mut RlsTransaction<'_>,
    item: &SyncDocumentItem,
) -> Result<WikiSyncOutcome, OpenRagError> {
    let state = ctx.state;
    let principal = ctx.principal;
    let bucket = ctx.bucket;
    let org_slug = ctx.org_slug;
    let dept_path = ctx.dept_path;

    let raw_content = item
        .content
        .as_deref()
        .or(item.body_md.as_deref())
        .unwrap_or("");

    let fm = if item.content.is_some() {
        parse_frontmatter(raw_content)?
    } else {
        openrag_memory::ParsedFrontmatter {
            body_md: raw_content.to_string(),
            ..Default::default()
        }
    };

    let raw_slug = item
        .slug
        .as_deref()
        .or(fm.slug.as_deref())
        .or(item.external_key.as_deref())
        .or(item.path.as_deref())
        .unwrap_or("");
    let slug = slugify_simple(raw_slug);
    if slug.is_empty() {
        return Err(OpenRagError::Validation(
            "wiki sync item requires slug, external_key, or path".into(),
        ));
    }

    if item.deleted == Some(true) {
        let existing =
            WikiCompiler::get_page_by_slug_rls(tx, principal.org_id, Some(bucket.id), &slug)
                .await?
                .filter(|page| page.status != WikiStatus::Archived);
        if let Some(page) = existing {
            // Verify fail-closed RLS write authorization on both wiki.pages and parent bucket
            // before transitioning status='archived' post-commit via worker_pool (since
            // rls_wiki_read enforces status='active' on the post-update tuple for openrag_app).
            let authorized_id: Option<Uuid> = sqlx::query_scalar(
                "UPDATE wiki.pages SET updated_at = now() WHERE org_id = $1 AND id = $2 RETURNING id",
            )
            .bind(principal.org_id)
            .bind(page.id)
            .fetch_optional(&mut **tx.tx_mut())
            .await
            .map_err(|e| OpenRagError::Database(format!("authorize archive wiki.pages: {e}")))?;
            if authorized_id.is_none() {
                return Err(OpenRagError::Forbidden(format!(
                    "RLS denied write access to archive wiki page '{}'",
                    page.slug
                )));
            }
            if !ctx.pending_wiki_archives.iter().any(|(id, _)| *id == page.id) {
                ctx.pending_wiki_archives.push((page.id, slug.clone()));
            }
            return Ok(WikiSyncOutcome {
                id: Some(page.id),
                slug,
                status: "deleted".into(),
                enqueued: false,
            });
        }
        return Ok(WikiSyncOutcome {
            id: None,
            slug,
            status: "skipped".into(),
            enqueued: false,
        });
    }

    if fm.body_md.trim().is_empty() {
        return Err(OpenRagError::Validation(
            "wiki sync body_md/content must not be empty".into(),
        ));
    }

    let existing =
        WikiCompiler::get_page_by_slug_rls(tx, principal.org_id, Some(bucket.id), &slug).await?;
    if let Some(cur) = &existing {
        if let Some(ref expected_sha) = item.base_sha256 {
            if cur.body_sha256 != *expected_sha {
                return Err(OpenRagError::Conflict(format!(
                    "stale base_sha256 conflict on wiki page '{slug}'"
                )));
            }
        }
        if let Some(expected_ver) = item.base_version {
            if cur.version != expected_ver {
                return Err(OpenRagError::Conflict(format!(
                    "stale base_version conflict on wiki page '{slug}'"
                )));
            }
        }
    }

    // Layer 1: Neutralize any caller-supplied [PII:<TAG>:<HEX>] literals BEFORE Guard/Seal/Wiki upsert
    let sanitized_body_md = neutralize_literal_pii_tokens(&fm.body_md);

    let guard_dec = GuardEngine::evaluate(
        &sanitized_body_md,
        GuardSurface::IngestionGate,
        ctx.policy,
        ctx.pii_engine,
        Some(ctx.concept_lookup),
        None,
    )?;
    persist_sync_guard_event(ctx, tx, &sanitized_body_md, &guard_dec).await?;
    if guard_dec.action == GuardAction::Block {
        return Err(build_ingestion_guard_blocked_err(&guard_dec));
    }
    let guarded_body = if guard_dec.action == GuardAction::Mask {
        guard_dec
            .sanitized_text
            .unwrap_or_else(|| sanitized_body_md.clone())
    } else {
        sanitized_body_md
    };
    let effective_body = guarded_body;

    let title = neutralize_literal_pii_tokens(
        item.title
            .as_deref()
            .or(fm.title.as_deref())
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .unwrap_or(&slug),
    );

    let access_scope = item
        .access_scope
        .or(fm.scope)
        .unwrap_or_else(|| bucket.access_scope.parse().unwrap_or(AccessScope::Dept));
    let min_auth_level = match item.min_auth_level.or(fm.min_auth_level.map(i16::from)) {
        Some(lvl) => AuthLevel::try_from(lvl)?,
        None => AuthLevel::try_from((bucket.min_auth_level.max(1) as u8).min(principal.auth_level.as_u8()))?,
    };

    let mut merged_tags = fm.tags.clone();
    merged_tags.extend(item.tags.clone());
    let norm_tags = normalize_tags_dedup(&merged_tags);

    let (page, changed) = WikiCompiler::upsert_page_rls(
        tx,
        principal,
        Some(bucket.id),
        bucket.dept_id,
        org_slug,
        dept_path,
        access_scope,
        min_auth_level,
        &slug,
        &title,
        &effective_body,
        &norm_tags,
        WikiEditorType::Sync,
        Some("http sync"),
        state.text_embedder.as_ref(),
    )
    .await?;

    ctx.pending_wiki_archives.retain(|(id, _)| *id != page.id);

    if changed {
        if !fm.aliases.is_empty() {
            for alias in &fm.aliases {
                if normalize_term(alias) != normalize_term(&title) {
                    let _ = ConceptGraph::link_terms_rls(
                        tx,
                        principal.org_id,
                        Some(principal.principal_id),
                        &title,
                        alias,
                        ConceptCategory::Domain,
                        ConceptSource::WikiFrontmatter,
                    )
                    .await?;
                }
            }
        }
        let _ = ConceptGraph::ingest_parenthetical_pairs_rls(
            tx,
            principal.org_id,
            Some(principal.principal_id),
            &effective_body,
        )
        .await?;
    }

    Ok(WikiSyncOutcome {
        id: Some(page.id),
        slug: page.slug,
        status: if changed {
            "updated".into()
        } else {
            "skipped".into()
        },
        enqueued: changed,
    })
}

struct CorpusSyncOutcome {
    deleted: bool,
    skipped: bool,
    delta_pages: usize,
    upserted_chunks: usize,
    enqueued: bool,
    summary: Value,
}

async fn enqueue_delete_vectors_for_chunks(
    tx: &mut RlsTransaction<'_>,
    org_id: Uuid,
    doc_id: Uuid,
    chunk_ids: &[Uuid],
) -> Result<bool, OpenRagError> {
    let mut enqueued = false;
    for cid in chunk_ids {
        let _ = OutboxEnqueuer::enqueue_rls(
            tx,
            &EnqueueOutboxItem {
                org_id,
                target_table: VectorTable::CorpusChunks,
                target_id: *cid,
                op: OutboxOp::DeleteVector,
                content_sha256: None,
                priority: 3,
                payload: json!({ "document_id": doc_id }),
                idempotency_key: None,
                max_attempts: None,
            },
            false,
        )
        .await?;
        enqueued = true;
    }
    Ok(enqueued)
}

async fn guard_and_seal_sync_page(
    ctx: &SyncContext<'_>,
    tx: &mut RlsTransaction<'_>,
    raw_page_text: &str,
    guard_dec: &GuardDecision,
) -> Result<(String, String), OpenRagError> {
    let sanitized_raw = neutralize_literal_pii_tokens(raw_page_text);
    persist_sync_guard_event(ctx, tx, &sanitized_raw, guard_dec).await?;
    let guarded_page_text = if guard_dec.action == GuardAction::Mask {
        guard_dec
            .sanitized_text
            .clone()
            .unwrap_or_else(|| sanitized_raw.clone())
    } else {
        sanitized_raw
    };
    let final_page_redacted = guarded_page_text.clone();
    Ok((guarded_page_text, final_page_redacted))
}

async fn sync_corpus_item(
    ctx: &mut SyncContext<'_>,
    tx: &mut RlsTransaction<'_>,
    item: &SyncDocumentItem,
) -> Result<CorpusSyncOutcome, OpenRagError> {
    let state = ctx.state;
    let principal = ctx.principal;
    let bucket = ctx.bucket;
    let org_slug = ctx.org_slug;
    let dept_path = ctx.dept_path;

    let key = item
        .external_key
        .as_deref()
        .or(item.slug.as_deref())
        .or(item.path.as_deref())
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .ok_or_else(|| {
            OpenRagError::Validation(
                "corpus sync item requires external_key, slug, or path".into(),
            )
        })?
        .to_string();

    let existing_row = sqlx::query(
        "SELECT id, content_sha256, revision, access_scope, min_auth_level, title, tags
         FROM corpus.documents
         WHERE org_id = $1 AND bucket_id = $2
           AND (source_uri = $3 OR metadata->>'external_key' = $3 OR metadata->>'slug' = $3 OR metadata->>'path' = $3)
         ORDER BY created_at DESC
         LIMIT 1",
    )
    .bind(principal.org_id)
    .bind(bucket.id)
    .bind(&key)
    .fetch_optional(&mut **tx.tx_mut())
    .await
    .map_err(|e| OpenRagError::Database(format!("lookup sync corpus.documents: {e}")))?;

    if item.deleted == Some(true) {
        if let Some(row) = existing_row {
            let doc_id: Uuid = row
                .try_get("id")
                .map_err(|e| OpenRagError::Database(e.to_string()))?;
            let chunk_ids: Vec<Uuid> = sqlx::query_scalar(
                "SELECT id FROM corpus.chunks WHERE org_id = $1 AND document_id = $2",
            )
            .bind(principal.org_id)
            .bind(doc_id)
            .fetch_all(&mut **tx.tx_mut())
            .await
            .map_err(|e| OpenRagError::Database(format!("load synced chunk ids: {e}")))?;

            let enq_chunks =
                enqueue_delete_vectors_for_chunks(tx, principal.org_id, doc_id, &chunk_ids)
                    .await?;

            sqlx::query("DELETE FROM corpus.documents WHERE org_id = $1 AND id = $2")
                .bind(principal.org_id)
                .bind(doc_id)
                .execute(&mut **tx.tx_mut())
                .await
                .map_err(|e| OpenRagError::Database(format!("delete synced doc: {e}")))?;
            return Ok(CorpusSyncOutcome {
                deleted: true,
                skipped: false,
                delta_pages: 0,
                upserted_chunks: 0,
                enqueued: enq_chunks,
                summary: json!({ "kind": "corpus", "id": doc_id, "key": key, "status": "deleted" }),
            });
        }
        return Ok(CorpusSyncOutcome {
            deleted: false,
            skipped: true,
            delta_pages: 0,
            upserted_chunks: 0,
            enqueued: false,
            summary: json!({ "kind": "corpus", "key": key, "status": "skipped" }),
        });
    }

    // Layer 1: Neutralize any caller-supplied [PII:<TAG>:<HEX>] literals BEFORE Guard/Seal/SHA/Storage
    let pages_input: Vec<String> = if let Some(ref pages) = item.pages {
        pages
            .iter()
            .map(|p| neutralize_literal_pii_tokens(p))
            .collect()
    } else if let Some(ref text) = item.text {
        vec![neutralize_literal_pii_tokens(text)]
    } else {
        Vec::new()
    };

    if ctx.uses_s3_vectors {
        let now_micros = Utc::now().timestamp_micros();
        for page_text in &pages_input {
            let preview_id = Uuid::now_v7().to_string();
            let _ = encode_s3_metadata(
                &preview_id,
                now_micros,
                Some(page_text),
                &Map::new(),
                &S3MetadataSchema::default(),
            )?;
        }
    }

    let mut page_guard_decisions = Vec::with_capacity(pages_input.len());
    for raw_page_text in &pages_input {
        let guard_dec = GuardEngine::evaluate(
            raw_page_text,
            GuardSurface::IngestionGate,
            ctx.policy,
            ctx.pii_engine,
            Some(ctx.concept_lookup),
            None,
        )?;
        if guard_dec.action == GuardAction::Block {
            persist_sync_guard_event(ctx, tx, raw_page_text, &guard_dec).await?;
            return Err(build_ingestion_guard_blocked_err(&guard_dec));
        }
        page_guard_decisions.push(guard_dec);
    }

    let computed_sha = item
        .content_sha256
        .clone()
        .or_else(|| item.body_sha256.clone())
        .unwrap_or_else(|| sha256_hex_str(&pages_input.join("\n---PAGE---\n")));

    let title = neutralize_literal_pii_tokens(
        item.title
            .as_deref()
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .unwrap_or(&key),
    );

    let access_scope = item
        .access_scope
        .unwrap_or_else(|| bucket.access_scope.parse().unwrap_or(AccessScope::Dept));
    let doc_level = match item.min_auth_level {
        Some(lvl) => AuthLevel::try_from(lvl)?,
        None => AuthLevel::try_from((bucket.min_auth_level.max(1) as u8).min(principal.auth_level.as_u8()))?,
    };
    let norm_tags = normalize_tags_dedup(&item.tags);
    let acl_scope_key = format!(
        "{}:{}:{}:L{}",
        principal.org_id,
        access_scope.as_str(),
        bucket.dept_id,
        doc_level.as_u8()
    );

    if let Some(row) = existing_row {
        let doc_id: Uuid = row
            .try_get("id")
            .map_err(|e| OpenRagError::Database(e.to_string()))?;
        let cur_sha: String = row
            .try_get("content_sha256")
            .map_err(|e| OpenRagError::Database(e.to_string()))?;
        let cur_rev: i32 = row
            .try_get("revision")
            .map_err(|e| OpenRagError::Database(e.to_string()))?;

        if let Some(ref base_sha) = item.base_sha256 {
            if cur_sha != *base_sha {
                return Err(OpenRagError::Conflict(format!(
                    "stale base_sha256 conflict on document '{key}'"
                )));
            }
        }
        if let Some(base_ver) = item.base_version {
            if cur_rev != base_ver {
                return Err(OpenRagError::Conflict(format!(
                    "stale base_version conflict on document '{key}'"
                )));
            }
        }

        if cur_sha == computed_sha {
            return Ok(CorpusSyncOutcome {
                deleted: false,
                skipped: true,
                delta_pages: 0,
                upserted_chunks: 0,
                enqueued: false,
                summary: json!({ "kind": "corpus", "id": doc_id, "key": key, "status": "skipped" }),
            });
        }

        let existing_pages = sqlx::query(
            "SELECT id, page_no, content_sha256
             FROM corpus.pages
             WHERE org_id = $1 AND document_id = $2
             ORDER BY page_no ASC",
        )
        .bind(principal.org_id)
        .bind(doc_id)
        .fetch_all(&mut **tx.tx_mut())
        .await
        .map_err(|e| OpenRagError::Database(format!("load existing corpus.pages: {e}")))?;

        let mut page_by_no: BTreeMap<i32, (Uuid, Option<String>)> = BTreeMap::new();
        for pr in existing_pages {
            let pid: Uuid = pr
                .try_get("id")
                .map_err(|e| OpenRagError::Database(e.to_string()))?;
            let pno: i32 = pr
                .try_get("page_no")
                .map_err(|e| OpenRagError::Database(e.to_string()))?;
            let psha: Option<String> = pr
                .try_get("content_sha256")
                .map_err(|e| OpenRagError::Database(e.to_string()))?;
            page_by_no.insert(pno, (pid, psha));
        }

        let max_chunk_idx: Option<i32> = sqlx::query_scalar(
            "SELECT MAX(chunk_idx) FROM corpus.chunks WHERE org_id = $1 AND document_id = $2",
        )
        .bind(principal.org_id)
        .bind(doc_id)
        .fetch_one(&mut **tx.tx_mut())
        .await
        .map_err(|e| OpenRagError::Database(format!("max chunk_idx: {e}")))?;
        let mut next_chunk_idx = max_chunk_idx.unwrap_or(-1) + 1;

        let mut delta_pages = 0usize;
        let mut upserted_chunks = 0usize;
        let mut deleted_any = false;

        for (idx, raw_page_text) in pages_input.iter().enumerate() {
            let page_no = (idx + 1) as i32;
            let page_sha = sha256_hex_str(raw_page_text);
            if let Some((pid, Some(old_sha))) = page_by_no.get(&page_no) {
                if *old_sha == page_sha {
                    continue;
                }
                let (guarded_page_text, final_page_redacted) =
                    guard_and_seal_sync_page(ctx, tx, raw_page_text, &page_guard_decisions[idx])
                        .await?;
                sqlx::query(
                    "UPDATE corpus.pages
                     SET raw_markdown = $1, redacted_markdown = $2, content_sha256 = $3
                     WHERE org_id = $4 AND id = $5",
                )
                .bind(raw_page_text)
                .bind(&final_page_redacted)
                .bind(&page_sha)
                .bind(principal.org_id)
                .bind(*pid)
                .execute(&mut **tx.tx_mut())
                .await
                .map_err(|e| OpenRagError::Database(format!("update delta page: {e}")))?;

                let old_chunk_ids: Vec<Uuid> = sqlx::query_scalar(
                    "SELECT id FROM corpus.chunks WHERE org_id = $1 AND document_id = $2 AND page_id = $3",
                )
                .bind(principal.org_id)
                .bind(doc_id)
                .bind(*pid)
                .fetch_all(&mut **tx.tx_mut())
                .await
                .map_err(|e| OpenRagError::Database(format!("load stale page chunks: {e}")))?;

                if enqueue_delete_vectors_for_chunks(tx, principal.org_id, doc_id, &old_chunk_ids)
                    .await?
                {
                    deleted_any = true;
                }

                sqlx::query(
                    "DELETE FROM corpus.chunks WHERE org_id = $1 AND document_id = $2 AND page_id = $3",
                )
                .bind(principal.org_id)
                .bind(doc_id)
                .bind(*pid)
                .execute(&mut **tx.tx_mut())
                .await
                .map_err(|e| OpenRagError::Database(format!("delete stale page chunks: {e}")))?;

                let inserted = insert_sync_page_chunks(
                    tx,
                    SyncChunkInsertCtx {
                        principal,
                        bucket,
                        org_slug,
                        dept_path,
                        doc_id,
                        page_id: *pid,
                        access_scope,
                        doc_level,
                        acl_scope_key: &acl_scope_key,
                        title: &title,
                        tags: &norm_tags,
                        embed_model: state.text_embedder.model_name(),
                        embedder: state.text_embedder.as_ref(),
                        embed_document_title: ctx.embed_document_title,
                        chunker: ctx.chunker,
                    },
                    &guarded_page_text,
                    &mut next_chunk_idx,
                )
                .await?;
                delta_pages += 1;
                upserted_chunks += inserted;
            } else {
                let (guarded_page_text, final_page_redacted) =
                    guard_and_seal_sync_page(ctx, tx, raw_page_text, &page_guard_decisions[idx])
                        .await?;
                let page_id = insert_sync_page(
                    tx,
                    &SyncPageCtx {
                        principal,
                        bucket,
                        org_slug,
                        dept_path,
                        doc_id,
                        access_scope,
                        doc_level,
                        acl_scope_key: &acl_scope_key,
                    },
                    page_no,
                    raw_page_text,
                    &final_page_redacted,
                    &page_sha,
                )
                .await?;
                let inserted = insert_sync_page_chunks(
                    tx,
                    SyncChunkInsertCtx {
                        principal,
                        bucket,
                        org_slug,
                        dept_path,
                        doc_id,
                        page_id,
                        access_scope,
                        doc_level,
                        acl_scope_key: &acl_scope_key,
                        title: &title,
                        tags: &norm_tags,
                        embed_model: state.text_embedder.model_name(),
                        embedder: state.text_embedder.as_ref(),
                        embed_document_title: ctx.embed_document_title,
                        chunker: ctx.chunker,
                    },
                    &guarded_page_text,
                    &mut next_chunk_idx,
                )
                .await?;
                delta_pages += 1;
                upserted_chunks += inserted;
            }
        }

        // Remove surplus pages (and their chunks) if document shrunk
        let new_len = pages_input.len() as i32;
        let trailing_page_ids: Vec<Uuid> = sqlx::query_scalar(
            "SELECT id FROM corpus.pages WHERE org_id = $1 AND document_id = $2 AND page_no > $3",
        )
        .bind(principal.org_id)
        .bind(doc_id)
        .bind(new_len)
        .fetch_all(&mut **tx.tx_mut())
        .await
        .map_err(|e| OpenRagError::Database(format!("load trailing delta pages: {e}")))?;

        if !trailing_page_ids.is_empty() {
            let trailing_chunk_ids: Vec<Uuid> = sqlx::query_scalar(
                "SELECT id FROM corpus.chunks
                 WHERE org_id = $1 AND document_id = $2 AND page_id = ANY($3::uuid[])",
            )
            .bind(principal.org_id)
            .bind(doc_id)
            .bind(&trailing_page_ids)
            .fetch_all(&mut **tx.tx_mut())
            .await
            .map_err(|e| OpenRagError::Database(format!("load trailing delta chunks: {e}")))?;

            if enqueue_delete_vectors_for_chunks(tx, principal.org_id, doc_id, &trailing_chunk_ids)
                .await?
            {
                deleted_any = true;
            }
            // Pages hold no vectors to delete; removing the trailing pages is still a change.
            if !trailing_page_ids.is_empty() {
                deleted_any = true;
            }

            if !trailing_chunk_ids.is_empty() {
                sqlx::query(
                    "DELETE FROM corpus.chunks
                     WHERE org_id = $1 AND document_id = $2 AND id = ANY($3::uuid[])",
                )
                .bind(principal.org_id)
                .bind(doc_id)
                .bind(&trailing_chunk_ids)
                .execute(&mut **tx.tx_mut())
                .await
                .map_err(|e| {
                    OpenRagError::Database(format!("delete trailing delta chunks: {e}"))
                })?;
            }

            sqlx::query(
                "DELETE FROM corpus.pages
                 WHERE org_id = $1 AND document_id = $2 AND id = ANY($3::uuid[])",
            )
            .bind(principal.org_id)
            .bind(doc_id)
            .bind(&trailing_page_ids)
            .execute(&mut **tx.tx_mut())
            .await
            .map_err(|e| OpenRagError::Database(format!("delete trailing delta pages: {e}")))?;
        }

        sqlx::query(
            "UPDATE corpus.documents
             SET title = $1,
                 content_sha256 = $2,
                 revision = revision + 1,
                 tags = CASE WHEN cardinality($3::text[]) > 0 THEN $3 ELSE tags END,
                 updated_at = now()
             WHERE org_id = $4 AND id = $5",
        )
        .bind(&title)
        .bind(&computed_sha)
        .bind(&norm_tags)
        .bind(principal.org_id)
        .bind(doc_id)
        .execute(&mut **tx.tx_mut())
        .await
        .map_err(|e| OpenRagError::Database(format!("update corpus.documents sha: {e}")))?;

        return Ok(CorpusSyncOutcome {
            deleted: false,
            skipped: delta_pages == 0 && upserted_chunks == 0 && !deleted_any,
            delta_pages,
            upserted_chunks,
            enqueued: upserted_chunks > 0 || delta_pages > 0 || deleted_any,
            summary: json!({
                "kind": "corpus",
                "id": doc_id,
                "key": key,
                "status": if delta_pages == 0 && !deleted_any { "skipped" } else { "updated" },
                "delta_pages": delta_pages,
                "upserted_chunks": upserted_chunks,
            }),
        });
    }

    // Insert brand-new corpus document from sync
    let doc_id = Uuid::now_v7();
    let doc_arn = Arn::new(
        org_slug,
        dept_path,
        access_scope,
        ArnResourceType::Document,
        doc_level,
        doc_id.to_string(),
    )?
    .to_canonical();

    let meta = json!({
        "external_key": item.external_key.as_deref().unwrap_or(&key),
        "slug": item.slug,
        "path": item.path,
        "page_count": pages_input.len(),
    });

    sqlx::query(
        "INSERT INTO corpus.documents (
            id, org_id, bucket_id, dept_id, owner_principal_id,
            access_scope, allowed_group_ids, denied_group_ids, min_auth_level,
            arn, acl_scope_key, title, source_uri, mime_type, content_sha256,
            sensitivity, status, metadata, tags
         ) VALUES (
            $1, $2, $3, $4, $5,
            $6, ARRAY[]::uuid[], ARRAY[]::uuid[], $7,
            $8, $9, $10, $11, 'text/plain', $12,
            'normal', 'indexed', $13, $14
         )",
    )
    .bind(doc_id)
    .bind(principal.org_id)
    .bind(bucket.id)
    .bind(bucket.dept_id)
    .bind(principal.principal_id)
    .bind(access_scope.as_str())
    .bind(doc_level.as_i16())
    .bind(&doc_arn)
    .bind(&acl_scope_key)
    .bind(&title)
    .bind(&key)
    .bind(&computed_sha)
    .bind(meta)
    .bind(&norm_tags)
    .execute(&mut **tx.tx_mut())
    .await
    .map_err(|e| OpenRagError::Database(format!("insert synced corpus.documents: {e}")))?;

    let mut next_chunk_idx = 0i32;
    let mut upserted_chunks = 0usize;
    for (idx, raw_page_text) in pages_input.iter().enumerate() {
        let page_no = (idx + 1) as i32;
        let page_sha = sha256_hex_str(raw_page_text);
        let (guarded_page_text, final_page_redacted) =
            guard_and_seal_sync_page(ctx, tx, raw_page_text, &page_guard_decisions[idx]).await?;
        let page_id = insert_sync_page(
            tx,
            &SyncPageCtx {
                principal,
                bucket,
                org_slug,
                dept_path,
                doc_id,
                access_scope,
                doc_level,
                acl_scope_key: &acl_scope_key,
            },
            page_no,
            raw_page_text,
            &final_page_redacted,
            &page_sha,
        )
        .await?;
        upserted_chunks += insert_sync_page_chunks(
            tx,
            SyncChunkInsertCtx {
                principal,
                bucket,
                org_slug,
                dept_path,
                doc_id,
                page_id,
                access_scope,
                doc_level,
                acl_scope_key: &acl_scope_key,
                title: &title,
                tags: &norm_tags,
                embed_model: state.text_embedder.model_name(),
                embedder: state.text_embedder.as_ref(),
                embed_document_title: ctx.embed_document_title,
                chunker: ctx.chunker,
            },
            &guarded_page_text,
            &mut next_chunk_idx,
        )
        .await?;
    }

    Ok(CorpusSyncOutcome {
        deleted: false,
        skipped: false,
        delta_pages: pages_input.len(),
        upserted_chunks,
        enqueued: upserted_chunks > 0 || !pages_input.is_empty(),
        summary: json!({
            "kind": "corpus",
            "id": doc_id,
            "key": key,
            "status": "created",
            "delta_pages": pages_input.len(),
            "upserted_chunks": upserted_chunks,
        }),
    })
}

struct SyncPageCtx<'a> {
    principal: &'a PrincipalContext,
    bucket: &'a BucketRecord,
    org_slug: &'a str,
    dept_path: &'a str,
    doc_id: Uuid,
    access_scope: AccessScope,
    doc_level: AuthLevel,
    acl_scope_key: &'a str,
}

struct SyncChunkInsertCtx<'a> {
    principal: &'a PrincipalContext,
    bucket: &'a BucketRecord,
    org_slug: &'a str,
    dept_path: &'a str,
    doc_id: Uuid,
    page_id: Uuid,
    access_scope: AccessScope,
    doc_level: AuthLevel,
    acl_scope_key: &'a str,
    title: &'a str,
    tags: &'a [String],
    embed_model: &'a str,
    embedder: &'a dyn TextEmbedder,
    embed_document_title: bool,
    chunker: ChunkerKind,
}

async fn insert_sync_page(
    tx: &mut RlsTransaction<'_>,
    ctx: &SyncPageCtx<'_>,
    page_no: i32,
    raw_page_text: &str,
    redacted_page_text: &str,
    page_sha: &str,
) -> Result<Uuid, OpenRagError> {
    let page_id = Uuid::now_v7();
    let page_arn = Arn::new(
        ctx.org_slug,
        ctx.dept_path,
        ctx.access_scope,
        ArnResourceType::Page,
        ctx.doc_level,
        page_id.to_string(),
    )?
    .to_canonical();
    sqlx::query(
        "INSERT INTO corpus.pages (
            id, org_id, document_id, dept_id, owner_principal_id,
            access_scope, allowed_group_ids, denied_group_ids, min_auth_level,
            arn, acl_scope_key, page_no, raw_markdown, redacted_markdown,
            content_sha256
         ) VALUES (
            $1, $2, $3, $4, $5,
            $6, ARRAY[]::uuid[], ARRAY[]::uuid[], $7,
            $8, $9, $10, $11, $12,
            $13
         )",
    )
    .bind(page_id)
    .bind(ctx.principal.org_id)
    .bind(ctx.doc_id)
    .bind(ctx.bucket.dept_id)
    .bind(ctx.principal.principal_id)
    .bind(ctx.access_scope.as_str())
    .bind(ctx.doc_level.as_i16())
    .bind(&page_arn)
    .bind(ctx.acl_scope_key)
    .bind(page_no)
    .bind(raw_page_text)
    .bind(redacted_page_text)
    .bind(page_sha)
    .execute(&mut **tx.tx_mut())
    .await
    .map_err(|e| OpenRagError::Database(format!("insert sync corpus.pages: {e}")))?;

    Ok(page_id)
}

async fn insert_sync_page_chunks(
    tx: &mut RlsTransaction<'_>,
    ctx: SyncChunkInsertCtx<'_>,
    guarded_page_text: &str,
    next_chunk_idx: &mut i32,
) -> Result<usize, OpenRagError> {
    let sanitized_page_text = neutralize_literal_pii_tokens(guarded_page_text);
    let chunks = chunk_page(
        ctx.chunker,
        &sanitized_page_text,
        1000,
        ctx.title,
        ctx.embedder,
    )
    .await?;
    let embed_title = ctx.embed_document_title.then_some(ctx.title);
    let mut inserted = 0usize;
    for chunk_text in chunks {

        let chunk_id = Uuid::now_v7();
        let chunk_arn = Arn::new(
            ctx.org_slug,
            ctx.dept_path,
            ctx.access_scope,
            ArnResourceType::Chunk,
            ctx.doc_level,
            chunk_id.to_string(),
        )?
        .to_canonical();
        let text_hash = sha256_hex_str(&chunk_text);
        let emb = ctx.embedder.embed_document(embed_title, &chunk_text).await?;

        sqlx::query(
            "INSERT INTO corpus.chunks (
                id, org_id, document_id, page_id, dept_id, owner_principal_id,
                access_scope, allowed_group_ids, denied_group_ids, min_auth_level,
                arn, acl_scope_key, chunk_idx, redacted_text, bm25_text,
                text_hash, embedding, embedding_model, tags
             ) VALUES (
                $1, $2, $3, $4, $5, $6,
                $7, ARRAY[]::uuid[], ARRAY[]::uuid[], $8,
                $9, $10, $11, $12, $12,
                $13, $14, $15, $16
             )",
        )
        .bind(chunk_id)
        .bind(ctx.principal.org_id)
        .bind(ctx.doc_id)
        .bind(ctx.page_id)
        .bind(ctx.bucket.dept_id)
        .bind(ctx.principal.principal_id)
        .bind(ctx.access_scope.as_str())
        .bind(ctx.doc_level.as_i16())
        .bind(&chunk_arn)
        .bind(ctx.acl_scope_key)
        .bind(*next_chunk_idx)
        .bind(&chunk_text)
        .bind(&text_hash)
        .bind(Vector::from(emb))
        .bind(ctx.embed_model)
        .bind(ctx.tags)
        .execute(&mut **tx.tx_mut())
        .await
        .map_err(|e| OpenRagError::Database(format!("insert sync corpus.chunks: {e}")))?;

        let _ = OutboxEnqueuer::enqueue_rls(
            tx,
            &EnqueueOutboxItem {
                org_id: ctx.principal.org_id,
                target_table: VectorTable::CorpusChunks,
                target_id: chunk_id,
                op: OutboxOp::UpsertEmbed,
                content_sha256: Some(text_hash),
                priority: 4,
                payload: json!({ "document_id": ctx.doc_id, "chunk_idx": *next_chunk_idx }),
                idempotency_key: None,
                max_attempts: None,
            },
            false,
        )
        .await?;

        *next_chunk_idx += 1;
        inserted += 1;
    }
    Ok(inserted)
}

#[derive(Debug, Deserialize)]
struct ChatCompletionMessage {
    role: String,
    content: String,
}

#[derive(Debug, Deserialize)]
struct ChatCompletionsRequest {
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    messages: Option<Vec<ChatCompletionMessage>>,
    query: Option<String>,
    message: Option<String>,
    #[serde(default, alias = "bucket")]
    bucket_id: Option<String>,
    include_citations: Option<bool>,
    #[serde(default, alias = "unmask")]
    unmask_pii: bool,
    #[serde(default)]
    allow_unmasked_pii: bool,
    /// Keep `[PII:…]` tokens in the answer (set by MCP `rag_ask`, whose caller is an AI).
    #[serde(default)]
    keep_pii_tokens: bool,
    admission_threshold: Option<f32>,
    top_k: Option<usize>,
    #[serde(default)]
    stream: bool,
    /// Ground the answer only on documents in force at `as_of` (default now).
    #[serde(default)]
    latest_revision_only: bool,
    #[serde(default)]
    as_of: Option<DateTime<Utc>>,
}

async fn chat_completions_handler(
    State(state): State<Arc<AppState>>,
    AuthPrincipal(principal): AuthPrincipal,
    Json(req): Json<ChatCompletionsRequest>,
) -> Result<Response, ApiError> {
    let user_prompt = match &req.messages {
        Some(msgs) => {
            if msgs.is_empty() {
                return Err(ApiError::bad_request(
                    "empty_messages",
                    "messages must contain at least one message",
                ));
            }
            msgs.iter()
                .rev()
                .find(|m| m.role.eq_ignore_ascii_case("user"))
                .or_else(|| msgs.last())
                .map(|m| m.content.trim().to_string())
                .filter(|s| !s.is_empty())
                .ok_or_else(|| {
                    ApiError::bad_request("empty_prompt", "user message content must not be empty")
                })?
        }
        None => req
            .query
            .as_deref()
            .or(req.message.as_deref())
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(ToString::to_string)
            .ok_or_else(|| {
                ApiError::bad_request("empty_messages", "messages or query is required")
            })?,
    };

    let requested_model = req
        .model
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .unwrap_or("openrag-auto");

    let res = run_chat_completion(LlmCompletionInput {
        state: &state,
        principal: &principal,
        requested_model,
        user_prompt: &user_prompt,
        bucket_ref: req.bucket_id.as_deref(),
        spaces: None,
        include_citations: req.include_citations.unwrap_or(true),
        unmask_pii: req.unmask_pii,
        allow_unmasked_pii: req.allow_unmasked_pii,
        keep_pii_tokens: req.keep_pii_tokens,
        admission_threshold: req.admission_threshold,
        top_k: req.top_k,
        latest_revision_only: req.latest_revision_only,
        as_of: req.as_of,
    })
    .await
    .map_err(ApiError::from)?;

    let created_ts = Utc::now().timestamp();
    if req.stream {
        let mut events: Vec<Result<Event, Infallible>> = Vec::new();
        let role_chunk = json!({
            "id": res.id,
            "object": "chat.completion.chunk",
            "created": created_ts,
            "model": res.model,
            "choices": [{
                "index": 0,
                "delta": { "role": "assistant" },
                "finish_reason": Value::Null,
            }],
        });
        events.push(Ok(Event::default().data(role_chunk.to_string())));

        let content_chunk = json!({
            "id": res.id,
            "object": "chat.completion.chunk",
            "created": created_ts,
            "model": res.model,
            "choices": [{
                "index": 0,
                "delta": { "content": res.answer },
                "finish_reason": Value::Null,
            }],
        });
        events.push(Ok(Event::default().data(content_chunk.to_string())));

        let stop_chunk = json!({
            "id": res.id,
            "object": "chat.completion.chunk",
            "created": created_ts,
            "model": res.model,
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": "stop",
            }],
            "citations": res.citations,
            "sources": res.citations,
            "admission": res.admission,
        });
        events.push(Ok(Event::default().data(stop_chunk.to_string())));
        events.push(Ok(Event::default().data("[DONE]")));

        let sse = Sse::new(stream::iter(events)).keep_alive(KeepAlive::default());
        return Ok(sse.into_response());
    }

    Ok(Json(json!({
        "id": res.id,
        "object": "chat.completion",
        "created": created_ts,
        "model": res.model,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": res.answer,
            },
            "finish_reason": "stop",
        }],
        "usage": res.usage,
        "citations": res.citations,
        "sources": res.citations,
        "admission": res.admission,
        "provider": res.provider_name,
        "is_on_premise": res.is_on_premise,
        "pii_masked": res.pii_masked,
        "max_context_auth_level": res.max_context_auth_level,
        "warnings": res.warnings,
    }))
    .into_response())
}

#[derive(Debug, Deserialize)]
struct AbSearchHttpRequest {
    #[serde(default, alias = "bucket")]
    bucket_id: Option<String>,
    golden_set_id: Option<Uuid>,
    #[serde(default, alias = "top_k")]
    k: Option<usize>,
    #[serde(default)]
    queries: Vec<AbQueryInput>,
    #[serde(default)]
    variants: BTreeMap<String, AbVariantConfig>,
}

async fn ab_search_handler(
    State(state): State<Arc<AppState>>,
    AuthPrincipal(principal): AuthPrincipal,
    Json(req): Json<AbSearchHttpRequest>,
) -> Result<Json<Value>, ApiError> {
    let app_pool = state.require_app_pool().map_err(ApiError::from)?;
    let resolved_bucket_id = match req.bucket_id.as_deref().map(str::trim).filter(|s| !s.is_empty()) {
        Some(raw) => {
            let mut tx = app_pool
                .begin_rls_tx(&principal)
                .await
                .map_err(ApiError::from)?;
            let b = resolve_bucket_rls(&mut tx, principal.org_id, raw)
                .await
                .map_err(ApiError::from)?;
            enforce_api_key_arn_scope(&principal, &b.arn).map_err(ApiError::from)?;
            tx.commit().await.map_err(|e| {
                ApiError::from(OpenRagError::Database(format!("commit ab_search bucket: {e}")))
            })?;
            Some(b.id)
        }
        None => None,
    };

    let ab_req = AbSearchRequest {
        bucket_id: resolved_bucket_id,
        golden_set_id: req.golden_set_id,
        k: req.k,
        queries: req.queries,
        variants: req.variants,
    };

    let retriever = state
        .pg_retriever(principal.org_id)
        .await
        .map_err(ApiError::from)?;
    let eval_store = state.pg_eval_store().map_err(ApiError::from)?;
    let pipeline = state.pipeline_config.read().await.clone();

    let report = AbSearchEvaluator::evaluate(
        &retriever,
        Some(&eval_store),
        &principal,
        &ab_req,
        Some(&pipeline),
    )
    .await
    .map_err(ApiError::from)?;

    let value = serde_json::to_value(&report)
        .map_err(|e| ApiError::from(OpenRagError::Internal(format!("serialize ab report: {e}"))))?;
    Ok(Json(value))
}

#[derive(Debug, Deserialize)]
pub(crate) struct ListConceptsQuery {
    pub(crate) query: Option<String>,
    pub(crate) lang: Option<String>,
}

pub(crate) async fn list_concepts_handler(
    State(state): State<Arc<AppState>>,
    AuthPrincipal(principal): AuthPrincipal,
    Query(q): Query<ListConceptsQuery>,
) -> Result<Json<Value>, ApiError> {
    let app_pool = state.require_app_pool().map_err(ApiError::from)?;
    let mut tx = app_pool
        .begin_rls_tx(&principal)
        .await
        .map_err(ApiError::from)?;
    let all = ConceptGraph::list_concepts_rls(&mut tx, principal.org_id)
        .await
        .map_err(ApiError::from)?;
    tx.commit().await.map_err(|e| {
        ApiError::from(OpenRagError::Database(format!("commit list_concepts: {e}")))
    })?;

    let query_norm = q.query.as_deref().map(normalize_term).unwrap_or_default();
    let query_comp = q.query.as_deref().map(compact_term).unwrap_or_default();
    let lang_filter = q
        .lang
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_ascii_lowercase);

    let mut filtered = Vec::new();
    for rc in all {
        if let Some(ref lang) = lang_filter {
            let has_lang = rc
                .aliases
                .iter()
                .any(|a| a.lang.eq_ignore_ascii_case(lang));
            if !has_lang {
                continue;
            }
        }
        if !query_norm.is_empty() {
            let matches_concept = rc.concept.canonical_key.contains(&query_norm)
                || normalize_term(&rc.concept.display_name).contains(&query_norm)
                || (!query_comp.is_empty() && compact_term(&rc.concept.canonical_key).contains(&query_comp));
            let matches_alias = rc.aliases.iter().any(|a| {
                a.normalized_alias.contains(&query_norm)
                    || (!query_comp.is_empty() && a.compact_alias.contains(&query_comp))
            });
            if !matches_concept && !matches_alias {
                continue;
            }
        }
        filtered.push(resolved_concept_to_json(&rc));
    }

    Ok(Json(json!({ "concepts": filtered })))
}

#[derive(Debug, Deserialize)]
pub(crate) struct CreateConceptRequest {
    #[serde(alias = "canonical_key", alias = "display_name")]
    pub(crate) canonical_name: String,
    pub(crate) category: Option<ConceptCategory>,
    pub(crate) description: Option<String>,
    #[serde(default)]
    pub(crate) aliases: Vec<String>,
    pub(crate) embedding: Option<Vec<f32>>,
    pub(crate) wiki_page_id: Option<Uuid>,
}

pub(crate) async fn create_concept_handler(
    State(state): State<Arc<AppState>>,
    AuthPrincipal(principal): AuthPrincipal,
    Json(req): Json<CreateConceptRequest>,
) -> Result<(StatusCode, Json<Value>), ApiError> {
    let norm_key = normalize_term(&req.canonical_name);
    if norm_key.is_empty() {
        return Err(ApiError::bad_request(
            "invalid_canonical_name",
            "canonical_name must not be empty",
        ));
    }

    let emb_vec = match req.embedding {
        Some(v) => {
            validate_vector(&v).map_err(ApiError::from)?;
            v
        }
        None => DeterministicTextEmbedder::embed_sync(&req.canonical_name).map_err(ApiError::from)?,
    };

    let category = req.category.unwrap_or(ConceptCategory::Domain);
    let display_name = req.canonical_name.trim().to_string();

    let app_pool = state.require_app_pool().map_err(ApiError::from)?;
    let mut tx = app_pool
        .begin_rls_tx(&principal)
        .await
        .map_err(ApiError::from)?;

    let row = sqlx::query(
        "INSERT INTO wiki.concepts (
            org_id, canonical_key, display_name, category, description,
            wiki_page_id, embedding, created_by
         ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
         ON CONFLICT (org_id, canonical_key) DO UPDATE SET
            display_name = EXCLUDED.display_name,
            description = COALESCE(EXCLUDED.description, wiki.concepts.description),
            embedding = EXCLUDED.embedding,
            updated_at = now()
         RETURNING id, org_id, canonical_key, display_name, category, description,
                   wiki_page_id, created_by, created_at, updated_at",
    )
    .bind(principal.org_id)
    .bind(&norm_key)
    .bind(&display_name)
    .bind(category.as_str())
    .bind(&req.description)
    .bind(req.wiki_page_id)
    .bind(Vector::from(emb_vec))
    .bind(Some(principal.principal_id))
    .fetch_one(&mut **tx.tx_mut())
    .await
    .map_err(map_sqlx_err("insert wiki.concepts"))?;

    let concept = row_to_concept(&row).map_err(ApiError::from)?;

    let mut all_aliases = vec![display_name.clone()];
    all_aliases.extend(req.aliases.clone());
    for raw in all_aliases {
        let trimmed = raw.trim();
        let norm = normalize_term(trimmed);
        let comp = compact_term(trimmed);
        if norm.is_empty() {
            continue;
        }
        let lang = detect_lang_hint(trimmed);
        sqlx::query(
            "INSERT INTO wiki.concept_aliases (
                org_id, concept_id, alias_text, normalized_alias, compact_alias,
                lang, source, created_by
             ) VALUES ($1, $2, $3, $4, $5, $6, 'manual', $7)
             ON CONFLICT (org_id, concept_id, normalized_alias) DO NOTHING",
        )
        .bind(principal.org_id)
        .bind(concept.id)
        .bind(trimmed)
        .bind(&norm)
        .bind(&comp)
        .bind(lang)
        .bind(Some(principal.principal_id))
        .execute(&mut **tx.tx_mut())
        .await
        .map_err(map_sqlx_err("insert wiki.concept_aliases"))?;
    }

    let resolved = ConceptGraph::resolve_rls(&mut tx, principal.org_id, &norm_key)
        .await
        .map_err(ApiError::from)?
        .ok_or_else(|| ApiError::from(OpenRagError::Internal("concept missing after create".into())))?;

    tx.commit().await.map_err(|e| {
        ApiError::from(OpenRagError::Database(format!("commit create_concept: {e}")))
    })?;

    Ok((StatusCode::CREATED, Json(resolved_concept_to_json(&resolved))))
}

pub(crate) async fn get_concept_handler(
    State(state): State<Arc<AppState>>,
    AuthPrincipal(principal): AuthPrincipal,
    Path(id): Path<Uuid>,
) -> Result<Json<Value>, ApiError> {
    let app_pool = state.require_app_pool().map_err(ApiError::from)?;
    let mut tx = app_pool
        .begin_rls_tx(&principal)
        .await
        .map_err(ApiError::from)?;

    let row = sqlx::query(
        "SELECT id, org_id, canonical_key, display_name, category, description,
                wiki_page_id, created_by, created_at, updated_at
         FROM wiki.concepts
         WHERE org_id = $1 AND id = $2",
    )
    .bind(principal.org_id)
    .bind(id)
    .fetch_optional(&mut **tx.tx_mut())
    .await
    .map_err(|e| ApiError::from(OpenRagError::Database(format!("fetch wiki.concepts: {e}"))))?
    .ok_or_else(|| ApiError::not_found(format!("concept {id} not found")))?;

    let concept = row_to_concept(&row).map_err(ApiError::from)?;
    let aliases = fetch_aliases_for_concept(&mut tx, principal.org_id, concept.id)
        .await
        .map_err(ApiError::from)?;
    tx.commit().await.map_err(|e| {
        ApiError::from(OpenRagError::Database(format!("commit get_concept: {e}")))
    })?;

    Ok(Json(resolved_concept_to_json(&ResolvedConcept {
        concept,
        aliases,
    })))
}

#[derive(Debug, Deserialize)]
pub(crate) struct LinkConceptsRequest {
    pub(crate) source: String,
    pub(crate) target: String,
    pub(crate) canonical_name: Option<String>,
    pub(crate) category: Option<ConceptCategory>,
    pub(crate) concept_id: Option<Uuid>,
}

pub(crate) async fn link_concepts_handler(
    State(state): State<Arc<AppState>>,
    AuthPrincipal(principal): AuthPrincipal,
    Json(req): Json<LinkConceptsRequest>,
) -> Result<(StatusCode, Json<Value>), ApiError> {
    let norm_src = normalize_term(&req.source);
    let norm_tgt = normalize_term(&req.target);
    if norm_src.is_empty() || norm_tgt.is_empty() {
        return Err(ApiError::bad_request(
            "invalid_concept_term",
            "source and target terms must not be empty",
        ));
    }
    if norm_src == norm_tgt {
        return Err(ApiError::bad_request(
            "self_link_rejected",
            "cannot link a term to itself",
        ));
    }

    let app_pool = state.require_app_pool().map_err(ApiError::from)?;
    let mut tx = app_pool
        .begin_rls_tx(&principal)
        .await
        .map_err(ApiError::from)?;

    let category = req.category.unwrap_or(ConceptCategory::Domain);
    let mut resolved = if let Some(cid) = req.concept_id {
        let concept_row = sqlx::query(
            "SELECT id, org_id, canonical_key, display_name, category, description,
                    wiki_page_id, created_by, created_at, updated_at
             FROM wiki.concepts
             WHERE org_id = $1 AND id = $2",
        )
        .bind(principal.org_id)
        .bind(cid)
        .fetch_optional(&mut **tx.tx_mut())
        .await
        .map_err(|e| ApiError::from(OpenRagError::Database(format!("verify concept_id: {e}"))))?
        .ok_or_else(|| ApiError::not_found(format!("concept {cid} not found")))?;

        let concept = row_to_concept(&concept_row).map_err(ApiError::from)?;
        for raw_term in [&req.source, &req.target] {
            let trimmed = raw_term.trim();
            let norm = normalize_term(trimmed);
            let comp = compact_term(trimmed);
            if !norm.is_empty() {
                sqlx::query(
                    "INSERT INTO wiki.concept_aliases (
                        org_id, concept_id, alias_text, normalized_alias, compact_alias,
                        lang, source, created_by
                     ) VALUES ($1, $2, $3, $4, $5, $6, 'manual', $7)
                     ON CONFLICT (org_id, concept_id, normalized_alias) DO NOTHING",
                )
                .bind(principal.org_id)
                .bind(cid)
                .bind(trimmed)
                .bind(&norm)
                .bind(&comp)
                .bind(detect_lang_hint(trimmed))
                .bind(Some(principal.principal_id))
                .execute(&mut **tx.tx_mut())
                .await
                .map_err(map_sqlx_err("insert wiki.concept_aliases"))?;
            }
        }
        let aliases = fetch_aliases_for_concept(&mut tx, principal.org_id, cid)
            .await
            .map_err(ApiError::from)?;
        ResolvedConcept { concept, aliases }
    } else {
        ConceptGraph::link_terms_rls(
            &mut tx,
            principal.org_id,
            Some(principal.principal_id),
            &req.source,
            &req.target,
            category,
            ConceptSource::Manual,
        )
        .await
        .map_err(ApiError::from)?
    };

    if let Some(canon) = req
        .canonical_name
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
    {
        let norm_canon = normalize_term(canon);
        let comp_canon = compact_term(canon);
        if !norm_canon.is_empty() {
            sqlx::query(
                "INSERT INTO wiki.concept_aliases (
                    org_id, concept_id, alias_text, normalized_alias, compact_alias,
                    lang, source, created_by
                 ) VALUES ($1, $2, $3, $4, $5, $6, 'manual', $7)
                 ON CONFLICT (org_id, concept_id, normalized_alias) DO NOTHING",
            )
            .bind(principal.org_id)
            .bind(resolved.concept.id)
            .bind(canon)
            .bind(&norm_canon)
            .bind(&comp_canon)
            .bind(detect_lang_hint(canon))
            .bind(Some(principal.principal_id))
            .execute(&mut **tx.tx_mut())
            .await
            .map_err(map_sqlx_err("insert canonical alias"))?;

            resolved.aliases = fetch_aliases_for_concept(&mut tx, principal.org_id, resolved.concept.id)
                .await
                .map_err(ApiError::from)?;
        }
    }

    tx.commit().await.map_err(|e| {
        ApiError::from(OpenRagError::Database(format!("commit link_concepts: {e}")))
    })?;

    Ok((StatusCode::CREATED, Json(resolved_concept_to_json(&resolved))))
}

#[derive(Debug, Deserialize)]
pub(crate) struct ResolveConceptRequest {
    pub(crate) term: String,
    #[serde(default)]
    pub(crate) use_semantic: bool,
    pub(crate) embedding: Option<Vec<f32>>,
    pub(crate) threshold: Option<f32>,
    pub(crate) top_k: Option<usize>,
}

pub(crate) async fn resolve_concepts_handler(
    State(state): State<Arc<AppState>>,
    AuthPrincipal(principal): AuthPrincipal,
    Json(req): Json<ResolveConceptRequest>,
) -> Result<Json<Value>, ApiError> {
    if let Some(ref emb) = req.embedding {
        validate_vector(emb).map_err(ApiError::from)?;
    }
    let threshold = req.threshold.unwrap_or(0.85);
    if !threshold.is_finite() || !(-1.0..=1.0).contains(&threshold) {
        return Err(ApiError::bad_request(
            "invalid_threshold",
            "threshold must be within -1.0..=1.0",
        ));
    }

    let app_pool = state.require_app_pool().map_err(ApiError::from)?;
    let mut tx = app_pool
        .begin_rls_tx(&principal)
        .await
        .map_err(ApiError::from)?;

    let exact = ConceptGraph::resolve_rls(&mut tx, principal.org_id, &req.term)
        .await
        .map_err(ApiError::from)?;

    if let Some(rc) = exact {
        tx.commit().await.map_err(|e| {
            ApiError::from(OpenRagError::Database(format!("commit resolve_concept: {e}")))
        })?;
        let mut val = resolved_concept_to_json(&rc);
        if let Some(obj) = val.as_object_mut() {
            obj.insert("matched".into(), json!(true));
            obj.insert("found".into(), json!(true));
            obj.insert("matches".into(), json!([]));
        }
        return Ok(Json(val));
    }

    let mut semantic_matches: Vec<ConceptSearchMatch> = Vec::new();
    let mut best_resolved: Option<ResolvedConcept> = None;
    if req.use_semantic || req.embedding.is_some() {
        let query_vec = match req.embedding {
            Some(v) => v,
            None => DeterministicTextEmbedder::embed_sync(&req.term).map_err(ApiError::from)?,
        };
        let limit = i64::try_from(req.top_k.unwrap_or(5).clamp(1, 50)).unwrap_or(5);
        semantic_matches = ConceptGraph::find_similar_by_vector_rls(
            &mut tx,
            principal.org_id,
            &query_vec,
            threshold,
            limit,
        )
        .await
        .map_err(ApiError::from)?;
        if let Some(first) = semantic_matches.first() {
            let aliases = fetch_aliases_for_concept(&mut tx, principal.org_id, first.concept.id)
                .await
                .map_err(ApiError::from)?;
            best_resolved = Some(ResolvedConcept {
                concept: first.concept.clone(),
                aliases,
            });
        }
    }

    tx.commit().await.map_err(|e| {
        ApiError::from(OpenRagError::Database(format!("commit resolve_concept: {e}")))
    })?;

    if let Some(rc) = best_resolved {
        let mut val = resolved_concept_to_json(&rc);
        if let Some(obj) = val.as_object_mut() {
            obj.insert("matched".into(), json!(true));
            obj.insert("found".into(), json!(true));
            obj.insert("matches".into(), json!(semantic_matches));
        }
        return Ok(Json(val));
    }

    Ok(Json(json!({
        "matched": false,
        "found": false,
        "concept_id": Value::Null,
        "canonical_id": Value::Null,
        "concept": Value::Null,
        "aliases": [],
        "matches": [],
    })))
}

pub(crate) async fn delete_concept_handler(
    State(state): State<Arc<AppState>>,
    AuthPrincipal(principal): AuthPrincipal,
    Path(id): Path<Uuid>,
) -> Result<Json<Value>, ApiError> {
    let app_pool = state.require_app_pool().map_err(ApiError::from)?;
    let worker_pool = state.require_worker_pool().map_err(ApiError::from)?;
    let mut tx = app_pool
        .begin_rls_tx(&principal)
        .await
        .map_err(ApiError::from)?;

    let deleted: Option<Uuid> = sqlx::query_scalar(
        "DELETE FROM wiki.concepts WHERE org_id = $1 AND id = $2 RETURNING id",
    )
    .bind(principal.org_id)
    .bind(id)
    .fetch_optional(&mut **tx.tx_mut())
    .await
    .map_err(|e| ApiError::from(OpenRagError::Database(format!("delete wiki.concepts: {e}"))))?;

    tx.commit().await.map_err(|e| {
        ApiError::from(OpenRagError::Database(format!("commit delete_concept: {e}")))
    })?;

    if let Some(cid) = deleted {
        return Ok(Json(json!({ "deleted": true, "id": cid })));
    }

    let exists_in_org: bool = sqlx::query_scalar(
        "SELECT EXISTS(SELECT 1 FROM wiki.concepts WHERE org_id = $1 AND id = $2)",
    )
    .bind(principal.org_id)
    .bind(id)
    .fetch_one(worker_pool.inner())
    .await
    .map_err(|e| ApiError::from(OpenRagError::Database(format!("check concept existence: {e}"))))?;

    if exists_in_org {
        Err(ApiError::forbidden(
            "forbidden",
            "only concept creator or admin may delete concept",
        ))
    } else {
        Err(ApiError::not_found(format!("concept {id} not found")))
    }
}

pub(crate) async fn delete_concept_alias_handler(
    State(state): State<Arc<AppState>>,
    AuthPrincipal(principal): AuthPrincipal,
    Path((concept_id, alias_id)): Path<(Uuid, Uuid)>,
) -> Result<Json<Value>, ApiError> {
    let app_pool = state.require_app_pool().map_err(ApiError::from)?;
    let worker_pool = state.require_worker_pool().map_err(ApiError::from)?;
    let mut tx = app_pool
        .begin_rls_tx(&principal)
        .await
        .map_err(ApiError::from)?;

    let deleted: Option<Uuid> = sqlx::query_scalar(
        "DELETE FROM wiki.concept_aliases
         WHERE org_id = $1 AND concept_id = $2 AND id = $3
         RETURNING id",
    )
    .bind(principal.org_id)
    .bind(concept_id)
    .bind(alias_id)
    .fetch_optional(&mut **tx.tx_mut())
    .await
    .map_err(|e| {
        ApiError::from(OpenRagError::Database(format!(
            "delete wiki.concept_aliases: {e}"
        )))
    })?;

    tx.commit().await.map_err(|e| {
        ApiError::from(OpenRagError::Database(format!("commit delete_alias: {e}")))
    })?;

    if let Some(aid) = deleted {
        return Ok(Json(json!({
            "deleted": true,
            "id": aid,
            "concept_id": concept_id,
        })));
    }

    let exists_in_org: bool = sqlx::query_scalar(
        "SELECT EXISTS(SELECT 1 FROM wiki.concept_aliases WHERE org_id = $1 AND concept_id = $2 AND id = $3)",
    )
    .bind(principal.org_id)
    .bind(concept_id)
    .bind(alias_id)
    .fetch_one(worker_pool.inner())
    .await
    .map_err(|e| ApiError::from(OpenRagError::Database(format!("check alias existence: {e}"))))?;

    if exists_in_org {
        Err(ApiError::forbidden(
            "forbidden",
            "only alias creator or admin may delete alias",
        ))
    } else {
        Err(ApiError::not_found(format!(
            "alias {alias_id} not found on concept {concept_id}"
        )))
    }
}

#[derive(Debug, Deserialize)]
struct TagsMutationRequest {
    tags: Vec<String>,
}

async fn get_document_tags_handler(
    State(state): State<Arc<AppState>>,
    AuthPrincipal(principal): AuthPrincipal,
    Path((bucket_ref, doc_id)): Path<(String, Uuid)>,
) -> Result<Json<Value>, ApiError> {
    let app_pool = state.require_app_pool().map_err(ApiError::from)?;
    let mut tx = app_pool
        .begin_rls_tx(&principal)
        .await
        .map_err(ApiError::from)?;
    let bucket = resolve_bucket_rls(&mut tx, principal.org_id, &bucket_ref)
        .await
        .map_err(ApiError::from)?;
    enforce_api_key_arn_scope(&principal, &bucket.arn).map_err(ApiError::from)?;

    let tags: Option<Vec<String>> = sqlx::query_scalar(
        "SELECT tags FROM corpus.documents WHERE org_id = $1 AND bucket_id = $2 AND id = $3",
    )
    .bind(principal.org_id)
    .bind(bucket.id)
    .bind(doc_id)
    .fetch_optional(&mut **tx.tx_mut())
    .await
    .map_err(|e| ApiError::from(OpenRagError::Database(format!("fetch document tags: {e}"))))?;

    let tags = tags.ok_or_else(|| ApiError::not_found(format!("document {doc_id} not found")))?;
    tx.commit().await.map_err(|e| {
        ApiError::from(OpenRagError::Database(format!("commit get_doc_tags: {e}")))
    })?;

    Ok(Json(json!({
        "document_id": doc_id,
        "bucket_id": bucket.id,
        "tags": tags,
    })))
}

async fn add_document_tags_handler(
    State(state): State<Arc<AppState>>,
    AuthPrincipal(principal): AuthPrincipal,
    Path((bucket_ref, doc_id)): Path<(String, Uuid)>,
    Json(req): Json<TagsMutationRequest>,
) -> Result<Json<Value>, ApiError> {
    let app_pool = state.require_app_pool().map_err(ApiError::from)?;
    let mut tx = app_pool
        .begin_rls_tx(&principal)
        .await
        .map_err(ApiError::from)?;
    let bucket = resolve_bucket_rls(&mut tx, principal.org_id, &bucket_ref)
        .await
        .map_err(ApiError::from)?;
    enforce_api_key_arn_scope(&principal, &bucket.arn).map_err(ApiError::from)?;

    let exists: Option<Uuid> = sqlx::query_scalar(
        "SELECT id FROM corpus.documents WHERE org_id = $1 AND bucket_id = $2 AND id = $3",
    )
    .bind(principal.org_id)
    .bind(bucket.id)
    .bind(doc_id)
    .fetch_optional(&mut **tx.tx_mut())
    .await
    .map_err(|e| ApiError::from(OpenRagError::Database(format!("check document in bucket: {e}"))))?;
    if exists.is_none() {
        return Err(ApiError::not_found(format!("document {doc_id} not found")));
    }

    let tags = match add_tags_rls(
        &mut tx,
        principal.org_id,
        TagTable::CorpusDocuments,
        doc_id,
        &req.tags,
        true,
    )
    .await
    {
        Ok(t) => t,
        Err(OpenRagError::NotFound(_)) => {
            return Err(ApiError::forbidden(
                "forbidden",
                format!("insufficient write permission on document {doc_id}"),
            ));
        }
        Err(err) => return Err(ApiError::from(err)),
    };
    tx.commit().await.map_err(|e| {
        ApiError::from(OpenRagError::Database(format!("commit add_doc_tags: {e}")))
    })?;

    Ok(Json(json!({
        "document_id": doc_id,
        "bucket_id": bucket.id,
        "tags": tags,
    })))
}

async fn remove_document_tags_handler(
    State(state): State<Arc<AppState>>,
    AuthPrincipal(principal): AuthPrincipal,
    Path((bucket_ref, doc_id)): Path<(String, Uuid)>,
    Json(req): Json<TagsMutationRequest>,
) -> Result<Json<Value>, ApiError> {
    let app_pool = state.require_app_pool().map_err(ApiError::from)?;
    let mut tx = app_pool
        .begin_rls_tx(&principal)
        .await
        .map_err(ApiError::from)?;
    let bucket = resolve_bucket_rls(&mut tx, principal.org_id, &bucket_ref)
        .await
        .map_err(ApiError::from)?;
    enforce_api_key_arn_scope(&principal, &bucket.arn).map_err(ApiError::from)?;

    let exists: Option<Uuid> = sqlx::query_scalar(
        "SELECT id FROM corpus.documents WHERE org_id = $1 AND bucket_id = $2 AND id = $3",
    )
    .bind(principal.org_id)
    .bind(bucket.id)
    .bind(doc_id)
    .fetch_optional(&mut **tx.tx_mut())
    .await
    .map_err(|e| ApiError::from(OpenRagError::Database(format!("check document in bucket: {e}"))))?;
    if exists.is_none() {
        return Err(ApiError::not_found(format!("document {doc_id} not found")));
    }

    let tags = match remove_tags_rls(
        &mut tx,
        principal.org_id,
        TagTable::CorpusDocuments,
        doc_id,
        &req.tags,
        true,
    )
    .await
    {
        Ok(t) => t,
        Err(OpenRagError::NotFound(_)) => {
            return Err(ApiError::forbidden(
                "forbidden",
                format!("insufficient write permission on document {doc_id}"),
            ));
        }
        Err(err) => return Err(ApiError::from(err)),
    };
    tx.commit().await.map_err(|e| {
        ApiError::from(OpenRagError::Database(format!("commit remove_doc_tags: {e}")))
    })?;

    Ok(Json(json!({
        "document_id": doc_id,
        "bucket_id": bucket.id,
        "tags": tags,
    })))
}

async fn add_wiki_tags_handler(
    State(state): State<Arc<AppState>>,
    AuthPrincipal(principal): AuthPrincipal,
    Path(page_id): Path<Uuid>,
    Json(req): Json<TagsMutationRequest>,
) -> Result<Json<Value>, ApiError> {
    let app_pool = state.require_app_pool().map_err(ApiError::from)?;
    let mut tx = app_pool
        .begin_rls_tx(&principal)
        .await
        .map_err(ApiError::from)?;

    let active_exists: Option<Uuid> = sqlx::query_scalar(
        "SELECT id FROM wiki.pages WHERE org_id = $1 AND id = $2 AND status = 'active'",
    )
    .bind(principal.org_id)
    .bind(page_id)
    .fetch_optional(&mut **tx.tx_mut())
    .await
    .map_err(|e| ApiError::from(OpenRagError::Database(format!("check active wiki page: {e}"))))?;
    if active_exists.is_none() {
        return Err(ApiError::not_found(format!("wiki page {page_id} not found")));
    }

    let tags = match add_tags_rls(
        &mut tx,
        principal.org_id,
        TagTable::WikiPages,
        page_id,
        &req.tags,
        false,
    )
    .await
    {
        Ok(t) => t,
        Err(OpenRagError::NotFound(_)) => {
            return Err(ApiError::forbidden(
                "forbidden",
                format!("insufficient write permission on wiki page {page_id}"),
            ));
        }
        Err(err) => return Err(ApiError::from(err)),
    };
    tx.commit().await.map_err(|e| {
        ApiError::from(OpenRagError::Database(format!("commit add_wiki_tags: {e}")))
    })?;

    Ok(Json(json!({ "page_id": page_id, "tags": tags })))
}

async fn remove_wiki_tags_handler(
    State(state): State<Arc<AppState>>,
    AuthPrincipal(principal): AuthPrincipal,
    Path(page_id): Path<Uuid>,
    Json(req): Json<TagsMutationRequest>,
) -> Result<Json<Value>, ApiError> {
    let app_pool = state.require_app_pool().map_err(ApiError::from)?;
    let mut tx = app_pool
        .begin_rls_tx(&principal)
        .await
        .map_err(ApiError::from)?;

    let active_exists: Option<Uuid> = sqlx::query_scalar(
        "SELECT id FROM wiki.pages WHERE org_id = $1 AND id = $2 AND status = 'active'",
    )
    .bind(principal.org_id)
    .bind(page_id)
    .fetch_optional(&mut **tx.tx_mut())
    .await
    .map_err(|e| ApiError::from(OpenRagError::Database(format!("check active wiki page: {e}"))))?;
    if active_exists.is_none() {
        return Err(ApiError::not_found(format!("wiki page {page_id} not found")));
    }

    let tags = match remove_tags_rls(
        &mut tx,
        principal.org_id,
        TagTable::WikiPages,
        page_id,
        &req.tags,
        false,
    )
    .await
    {
        Ok(t) => t,
        Err(OpenRagError::NotFound(_)) => {
            return Err(ApiError::forbidden(
                "forbidden",
                format!("insufficient write permission on wiki page {page_id}"),
            ));
        }
        Err(err) => return Err(ApiError::from(err)),
    };
    tx.commit().await.map_err(|e| {
        ApiError::from(OpenRagError::Database(format!("commit remove_wiki_tags: {e}")))
    })?;

    Ok(Json(json!({ "page_id": page_id, "tags": tags })))
}

async fn add_memory_tags_handler(
    State(state): State<Arc<AppState>>,
    AuthPrincipal(principal): AuthPrincipal,
    Path(memory_id): Path<Uuid>,
    Json(req): Json<TagsMutationRequest>,
) -> Result<Json<Value>, ApiError> {
    let app_pool = state.require_app_pool().map_err(ApiError::from)?;
    let mut tx = app_pool
        .begin_rls_tx(&principal)
        .await
        .map_err(ApiError::from)?;

    let active_exists: Option<Uuid> = sqlx::query_scalar(
        "SELECT id FROM memory.entries WHERE org_id = $1 AND id = $2 AND status = 'active'",
    )
    .bind(principal.org_id)
    .bind(memory_id)
    .fetch_optional(&mut **tx.tx_mut())
    .await
    .map_err(|e| ApiError::from(OpenRagError::Database(format!("check active memory entry: {e}"))))?;
    if active_exists.is_none() {
        return Err(ApiError::not_found(format!("memory entry {memory_id} not found")));
    }

    let tags = match add_tags_rls(
        &mut tx,
        principal.org_id,
        TagTable::MemoryEntries,
        memory_id,
        &req.tags,
        false,
    )
    .await
    {
        Ok(t) => t,
        Err(OpenRagError::NotFound(_)) => {
            return Err(ApiError::forbidden(
                "forbidden",
                format!("insufficient write permission on memory entry {memory_id}"),
            ));
        }
        Err(err) => return Err(ApiError::from(err)),
    };
    tx.commit().await.map_err(|e| {
        ApiError::from(OpenRagError::Database(format!("commit add_memory_tags: {e}")))
    })?;

    Ok(Json(json!({ "memory_id": memory_id, "tags": tags })))
}

async fn remove_memory_tags_handler(
    State(state): State<Arc<AppState>>,
    AuthPrincipal(principal): AuthPrincipal,
    Path(memory_id): Path<Uuid>,
    Json(req): Json<TagsMutationRequest>,
) -> Result<Json<Value>, ApiError> {
    let app_pool = state.require_app_pool().map_err(ApiError::from)?;
    let mut tx = app_pool
        .begin_rls_tx(&principal)
        .await
        .map_err(ApiError::from)?;

    let active_exists: Option<Uuid> = sqlx::query_scalar(
        "SELECT id FROM memory.entries WHERE org_id = $1 AND id = $2 AND status = 'active'",
    )
    .bind(principal.org_id)
    .bind(memory_id)
    .fetch_optional(&mut **tx.tx_mut())
    .await
    .map_err(|e| ApiError::from(OpenRagError::Database(format!("check active memory entry: {e}"))))?;
    if active_exists.is_none() {
        return Err(ApiError::not_found(format!("memory entry {memory_id} not found")));
    }

    let tags = match remove_tags_rls(
        &mut tx,
        principal.org_id,
        TagTable::MemoryEntries,
        memory_id,
        &req.tags,
        false,
    )
    .await
    {
        Ok(t) => t,
        Err(OpenRagError::NotFound(_)) => {
            return Err(ApiError::forbidden(
                "forbidden",
                format!("insufficient write permission on memory entry {memory_id}"),
            ));
        }
        Err(err) => return Err(ApiError::from(err)),
    };
    tx.commit().await.map_err(|e| {
        ApiError::from(OpenRagError::Database(format!("commit remove_memory_tags: {e}")))
    })?;

    Ok(Json(json!({ "memory_id": memory_id, "tags": tags })))
}

// ============================================================================
// Helpers
// ============================================================================

async fn has_s3_vectors_binding(
    worker_pool: &sqlx::PgPool,
    org_id: Uuid,
) -> Result<bool, OpenRagError> {
    let exists: bool = sqlx::query_scalar(
        "SELECT EXISTS(
            SELECT 1 FROM vectors.index_bindings
            WHERE org_id = $1
              AND external_backend = 's3_vectors'
              AND table_name IN ('corpus.chunks', 'corpus.pages', 'corpus.page_figures')
        )",
    )
    .bind(org_id)
    .fetch_one(worker_pool)
    .await
    .map_err(|e| OpenRagError::Database(format!("check s3_vectors binding: {e}")))?;
    Ok(exists)
}

pub(crate) fn normalize_tags_dedup(raw: &[String]) -> Vec<String> {
    let mut out = Vec::new();
    for t in raw {
        let norm = normalize_tag(t);
        if !norm.is_empty() && !out.contains(&norm) {
            out.push(norm);
        }
    }
    out.sort();
    out
}

pub(crate) fn slugify_simple(raw: &str) -> String {
    let trimmed = raw.trim().trim_end_matches(".md");
    let mut out = String::new();
    let mut prev_dash = false;
    for ch in trimmed.chars() {
        if ch.is_alphanumeric() {
            for lc in ch.to_lowercase() {
                out.push(lc);
            }
            prev_dash = false;
        } else if (ch == '-' || ch == '_' || ch.is_whitespace() || ch == '/')
            && !prev_dash
            && !out.is_empty()
        {
            out.push('-');
            prev_dash = true;
        }
    }
    out.trim_matches('-').to_string()
}

pub(crate) fn detect_lang_hint(term: &str) -> &'static str {
    let has_hangul = term.chars().any(|c| ('\u{AC00}'..='\u{D7A3}').contains(&c));
    let has_ascii = term.chars().any(|c| c.is_ascii_alphabetic());
    match (has_hangul, has_ascii) {
        (true, false) => "ko",
        (false, true) => "en",
        (true, true) => "mul",
        (false, false) => "und",
    }
}

fn non_empty_utf8(bytes: &[u8]) -> Option<String> {
    let s = String::from_utf8_lossy(bytes).trim().to_string();
    if s.is_empty() {
        None
    } else {
        Some(s)
    }
}

fn parse_sensitivity(raw: &str) -> Result<Sensitivity, ApiError> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "share" => Ok(Sensitivity::Share),
        "private" => Ok(Sensitivity::Private),
        "zdr" => Ok(Sensitivity::Zdr),
        "normal" => Ok(Sensitivity::Normal),
        "confidential" => Ok(Sensitivity::Confidential),
        "secret" => Ok(Sensitivity::Secret),
        other => Err(ApiError::bad_request(
            "invalid_sensitivity",
            format!("unknown sensitivity/mode '{other}'"),
        )),
    }
}

fn parse_string_list_field(bytes: &[u8]) -> Vec<String> {
    let raw = String::from_utf8_lossy(bytes);
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Vec::new();
    }
    if trimmed.starts_with('[') {
        if let Ok(arr) = serde_json::from_str::<Vec<String>>(trimmed) {
            return arr;
        }
    }
    trimmed
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(ToString::to_string)
        .collect()
}

fn parse_uuid_list_field(bytes: &[u8]) -> Result<Vec<Uuid>, ApiError> {
    let items = parse_string_list_field(bytes);
    let mut out = Vec::with_capacity(items.len());
    for s in items {
        let u = Uuid::parse_str(&s).map_err(|e| {
            ApiError::bad_request("invalid_uuid", format!("invalid UUID '{s}': {e}"))
        })?;
        out.push(u);
    }
    Ok(out)
}

fn parse_datetime(raw: &str) -> Result<DateTime<Utc>, ApiError> {
    DateTime::parse_from_rfc3339(raw)
        .map(|dt| dt.with_timezone(&Utc))
        .map_err(|e| {
            ApiError::bad_request("invalid_datetime", format!("invalid RFC3339 timestamp: {e}"))
        })
}

pub(crate) fn resolved_concept_to_json(rc: &ResolvedConcept) -> Value {
    json!({
        "id": rc.concept.id,
        "concept_id": rc.concept.id,
        "canonical_id": rc.concept.id,
        "org_id": rc.concept.org_id,
        "canonical_key": rc.concept.canonical_key,
        "canonical_name": rc.concept.display_name,
        "display_name": rc.concept.display_name,
        "category": rc.concept.category,
        "description": rc.concept.description,
        "wiki_page_id": rc.concept.wiki_page_id,
        "created_by": rc.concept.created_by,
        "created_at": rc.concept.created_at,
        "updated_at": rc.concept.updated_at,
        "concept": rc.concept,
        "aliases": rc.aliases,
    })
}

async fn fetch_aliases_for_concept(
    tx: &mut RlsTransaction<'_>,
    org_id: Uuid,
    concept_id: Uuid,
) -> Result<Vec<ConceptAlias>, OpenRagError> {
    let rows = sqlx::query(
        "SELECT id, org_id, concept_id, alias_text, normalized_alias, compact_alias,
                lang, source, confidence, created_by, created_at
         FROM wiki.concept_aliases
         WHERE org_id = $1 AND concept_id = $2
         ORDER BY normalized_alias ASC",
    )
    .bind(org_id)
    .bind(concept_id)
    .fetch_all(&mut **tx.tx_mut())
    .await
    .map_err(|e| OpenRagError::Database(format!("load concept_aliases: {e}")))?;

    let mut out = Vec::with_capacity(rows.len());
    for r in rows {
        let src_str: String = r
            .try_get("source")
            .map_err(|e| OpenRagError::Database(e.to_string()))?;
        out.push(ConceptAlias {
            id: r.try_get("id").map_err(|e| OpenRagError::Database(e.to_string()))?,
            org_id: r.try_get("org_id").map_err(|e| OpenRagError::Database(e.to_string()))?,
            concept_id: r
                .try_get("concept_id")
                .map_err(|e| OpenRagError::Database(e.to_string()))?,
            alias_text: r
                .try_get("alias_text")
                .map_err(|e| OpenRagError::Database(e.to_string()))?,
            normalized_alias: r
                .try_get("normalized_alias")
                .map_err(|e| OpenRagError::Database(e.to_string()))?,
            compact_alias: r
                .try_get("compact_alias")
                .map_err(|e| OpenRagError::Database(e.to_string()))?,
            lang: r.try_get("lang").map_err(|e| OpenRagError::Database(e.to_string()))?,
            source: src_str.parse()?,
            confidence: r
                .try_get("confidence")
                .map_err(|e| OpenRagError::Database(e.to_string()))?,
            created_by: r
                .try_get("created_by")
                .map_err(|e| OpenRagError::Database(e.to_string()))?,
            created_at: r
                .try_get("created_at")
                .map_err(|e| OpenRagError::Database(e.to_string()))?,
        });
    }
    Ok(out)
}

fn row_to_concept(row: &sqlx::postgres::PgRow) -> Result<Concept, OpenRagError> {
    let cat_str: String = row
        .try_get("category")
        .map_err(|e| OpenRagError::Database(e.to_string()))?;
    Ok(Concept {
        id: row.try_get("id").map_err(|e| OpenRagError::Database(e.to_string()))?,
        org_id: row.try_get("org_id").map_err(|e| OpenRagError::Database(e.to_string()))?,
        canonical_key: row
            .try_get("canonical_key")
            .map_err(|e| OpenRagError::Database(e.to_string()))?,
        display_name: row
            .try_get("display_name")
            .map_err(|e| OpenRagError::Database(e.to_string()))?,
        category: cat_str.parse()?,
        description: row
            .try_get("description")
            .map_err(|e| OpenRagError::Database(e.to_string()))?,
        wiki_page_id: row
            .try_get("wiki_page_id")
            .map_err(|e| OpenRagError::Database(e.to_string()))?,
        created_by: row
            .try_get("created_by")
            .map_err(|e| OpenRagError::Database(e.to_string()))?,
        created_at: row
            .try_get("created_at")
            .map_err(|e| OpenRagError::Database(e.to_string()))?,
        updated_at: row
            .try_get("updated_at")
            .map_err(|e| OpenRagError::Database(e.to_string()))?,
    })
}

pub(crate) fn row_to_bucket(row: &sqlx::postgres::PgRow) -> Result<BucketRecord, OpenRagError> {
    let id: Uuid = row.try_get("id").map_err(|e| OpenRagError::Database(e.to_string()))?;
    Ok(BucketRecord {
        id,
        bucket_id: id,
        org_id: row.try_get("org_id").map_err(|e| OpenRagError::Database(e.to_string()))?,
        dept_id: row.try_get("dept_id").map_err(|e| OpenRagError::Database(e.to_string()))?,
        owner_principal_id: row
            .try_get("owner_principal_id")
            .map_err(|e| OpenRagError::Database(e.to_string()))?,
        slug: row.try_get("slug").map_err(|e| OpenRagError::Database(e.to_string()))?,
        name: row.try_get("name").map_err(|e| OpenRagError::Database(e.to_string()))?,
        access_scope: row
            .try_get("access_scope")
            .map_err(|e| OpenRagError::Database(e.to_string()))?,
        allowed_group_ids: row
            .try_get("allowed_group_ids")
            .map_err(|e| OpenRagError::Database(e.to_string()))?,
        denied_group_ids: row
            .try_get("denied_group_ids")
            .map_err(|e| OpenRagError::Database(e.to_string()))?,
        min_auth_level: row
            .try_get("min_auth_level")
            .map_err(|e| OpenRagError::Database(e.to_string()))?,
        arn: row.try_get("arn").map_err(|e| OpenRagError::Database(e.to_string()))?,
        created_at: row
            .try_get("created_at")
            .map_err(|e| OpenRagError::Database(e.to_string()))?,
    })
}

pub(crate) fn db_col_err(e: sqlx::Error) -> ApiError {
    ApiError::from(OpenRagError::Database(e.to_string()))
}

pub(crate) fn map_sqlx_err(context: &'static str) -> impl FnOnce(sqlx::Error) -> ApiError {
    move |err| {
        if let Some(db_err) = err.as_database_error() {
            match db_err.code().as_deref() {
                Some("23505") => {
                    return ApiError::conflict(format!("{context}: {}", db_err.message()));
                }
                Some("42501") => {
                    return ApiError::forbidden(
                        "rls_denied",
                        format!("{context}: {}", db_err.message()),
                    );
                }
                Some("23503") => {
                    return ApiError::not_found(format!("{context}: {}", db_err.message()));
                }
                _ => {}
            }
        }
        ApiError::from(OpenRagError::Database(format!("{context}: {err}")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chunking::split_text_chunks;

    #[test]
    fn layer1_neutralize_handles_chunks_and_frontmatter_titles() {
        let injected = "LAUNDER-DOC-999 [PII:EMAIL:6c64d00d] and [PII:RRN_KR:1234abcd]";
        let sanitized = neutralize_literal_pii_tokens(injected);
        assert_eq!(
            sanitized,
            "LAUNDER-DOC-999 [PII-LITERAL:EMAIL:6c64d00d] and [PII-LITERAL:RRN_KR:1234abcd]"
        );
        let chunks = split_text_chunks(&sanitized, 1000, "Title [PII-LITERAL:EMAIL:6c64d00d]");
        assert_eq!(chunks.len(), 1);
        assert!(!chunks[0].contains("[PII:EMAIL:"));
        assert!(!chunks[0].contains("[PII:RRN_KR:"));

        // Empty body fallback to title must also keep neutralized title
        let empty_chunks = split_text_chunks(
            "   ",
            1000,
            &neutralize_literal_pii_tokens("LaunderTitle [PII:EMAIL:deadbeef]"),
        );
        assert_eq!(empty_chunks, vec!["LaunderTitle [PII-LITERAL:EMAIL:deadbeef]"]);
    }
}
