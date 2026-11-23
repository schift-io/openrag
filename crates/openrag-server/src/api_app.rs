use std::str::FromStr;
use std::sync::Arc;

use axum::extract::{FromRequest, Multipart, Path, Query, Request, State};
use axum::http::header::CONTENT_TYPE;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{delete, get, post};
use axum::{Json, Router};
use chrono::{DateTime, Utc};
use openrag_core::{
    AccessScope, Arn, ArnResourceType, AuthLevel, MemoryBackendKind, OpenRagError,
    PrincipalContext,
};
use openrag_eval::{FeedbackManager, FeedbackVerdict, SubmitFeedbackInput};
use openrag_memory::{
    ActorType, ApplyPatchInput, MemoryEntry, MemoryRouter, PatchOp, PromoteMemoryInput,
    PromoteTarget,
};
use openrag_pii::LocalThumbnailStore;
use openrag_pii::ThumbnailStore as PiiThumbnailStore;
use openrag_retrieve::RetrieveSpace;
use openrag_store::{normalize_tag, TagFilter};
use serde::Deserialize;
use serde_json::{json, Value};
use sqlx::Row;
use uuid::Uuid;

use crate::api_core::{
    create_concept_handler, db_col_err, delete_concept_alias_handler, delete_concept_handler,
    get_concept_handler, ingest_document_into_bucket, link_concepts_handler, list_concepts_handler,
    map_sqlx_err, parse_multipart_ingest, resolve_bucket_rls, resolve_concepts_handler,
    resolve_org_and_dept_paths, row_to_bucket, BucketRecord,
};
use crate::auth::{enforce_api_key_arn_scope, AuthPrincipal};
use crate::error::ApiError;
use crate::llm_router::{neutralize_literal_pii_tokens, run_chat_completion, LlmCompletionInput};
use crate::state::AppState;

pub fn app_routes() -> Router<Arc<AppState>> {
    Router::new()
        .route("/api/app/chat", post(app_chat_handler))
        .route("/api/app/upload", post(app_upload_handler))
        .route(
            "/api/app/pages/{page_id}/thumbnail",
            get(app_page_thumbnail_handler),
        )
        .route("/api/app/feedback", post(app_feedback_handler))
        .route("/api/app/wiki", get(app_wiki_list_handler))
        .route("/api/app/wiki/{page_id}", get(app_wiki_get_handler))
        .route("/api/app/memory", get(app_memory_list_handler))
        .route("/api/app/memory/patch", post(app_memory_patch_handler))
        .route("/api/app/memory/retract", post(app_memory_retract_handler))
        .route("/api/app/memory/promote", post(app_memory_promote_handler))
        .route("/api/app/connect", get(app_connect_handler))
        .route(
            "/api/app/concepts",
            get(list_concepts_handler).post(create_concept_handler),
        )
        .route("/api/app/concepts/link", post(link_concepts_handler))
        .route("/api/app/concepts/resolve", post(resolve_concepts_handler))
        .route(
            "/api/app/concepts/{id}",
            get(get_concept_handler).delete(delete_concept_handler),
        )
        .route(
            "/api/app/concepts/{id}/aliases/{alias_id}",
            delete(delete_concept_alias_handler),
        )
}

#[derive(Debug, Deserialize)]
struct AppChatMessage {
    role: String,
    content: String,
}

#[derive(Debug, Deserialize)]
struct AppChatRequest {
    message: Option<String>,
    query: Option<String>,
    messages: Option<Vec<AppChatMessage>>,
    #[serde(default, alias = "bucket")]
    bucket_id: Option<String>,
    model: Option<String>,
    #[serde(default, alias = "unmask")]
    unmask_pii: bool,
    #[serde(default)]
    allow_unmasked_pii: bool,
    top_k: Option<usize>,
    spaces: Option<Vec<RetrieveSpace>>,
    admission_threshold: Option<f32>,
    include_citations: Option<bool>,
}

async fn app_chat_handler(
    State(state): State<Arc<AppState>>,
    AuthPrincipal(principal): AuthPrincipal,
    Json(req): Json<AppChatRequest>,
) -> Result<Json<Value>, ApiError> {
    if let Some(ref spaces) = req.spaces {
        if spaces.is_empty() {
            return Err(ApiError::bad_request(
                "invalid_spaces",
                "spaces must not be empty when specified",
            ));
        }
    }

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
                ApiError::bad_request("empty_prompt", "message or query is required")
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
        spaces: req.spaces,
        include_citations: req.include_citations.unwrap_or(true),
        unmask_pii: req.unmask_pii,
        allow_unmasked_pii: req.allow_unmasked_pii,
        admission_threshold: req.admission_threshold,
        top_k: req.top_k,
        latest_revision_only: false,
        as_of: None,
    })
    .await
    .map_err(ApiError::from)?;

    let created_ts = Utc::now().timestamp();
    Ok(Json(json!({
        "id": res.id,
        "object": "chat.completion",
        "created": created_ts,
        "model": res.model,
        "answer": res.answer,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": res.answer,
            },
            "finish_reason": "stop",
        }],
        "citations": res.citations,
        "sources": res.citations,
        "admission": res.admission,
        "usage": res.usage,
        "provider": res.provider_name,
        "is_on_premise": res.is_on_premise,
        "pii_masked": res.pii_masked,
        "max_context_auth_level": res.max_context_auth_level,
        "warnings": res.warnings,
    })))
}

async fn app_upload_handler(
    State(state): State<Arc<AppState>>,
    AuthPrincipal(principal): AuthPrincipal,
    req: Request,
) -> Result<(StatusCode, Json<Value>), ApiError> {
    let multipart = Multipart::from_request(req, &state)
        .await
        .map_err(|_| ApiError::bad_request("invalid_upload", "upload form data is invalid"))?;
    let mut payload = parse_multipart_ingest(multipart).await?;

    let has_file = payload.file_bytes.as_ref().is_some_and(|b| !b.is_empty());
    let has_text = payload
        .text
        .as_deref()
        .is_some_and(|t| !t.trim().is_empty());
    if !has_file && !has_text {
        return Err(ApiError::bad_request(
            "missing_file",
            "upload requires a non-empty file",
        ));
    }

    let app_pool = state.require_app_pool().map_err(ApiError::from)?;
    let worker_pool = state.require_worker_pool().map_err(ApiError::from)?;

    let bucket = resolve_or_create_app_bucket(
        &state,
        app_pool,
        worker_pool,
        &principal,
        payload.bucket_ref.as_deref(),
        payload.access_scope,
    )
    .await?;

    let effective_scope = payload
        .access_scope
        .unwrap_or_else(|| bucket.access_scope.parse().unwrap_or(AccessScope::Dept));
    if payload.allowed_group_ids.is_empty()
        && effective_scope == AccessScope::Group
        && !bucket.allowed_group_ids.is_empty()
    {
        payload.allowed_group_ids = bucket.allowed_group_ids.clone();
    }

    let result = ingest_document_into_bucket(&state, &principal, &bucket, payload)
        .await
        .map_err(ApiError::from)?;

    Ok((StatusCode::CREATED, Json(result)))
}

async fn resolve_or_create_app_bucket(
    state: &AppState,
    app_pool: &openrag_db::DbPool,
    worker_pool: &openrag_db::DbPool,
    principal: &PrincipalContext,
    bucket_ref: Option<&str>,
    requested_scope: Option<AccessScope>,
) -> Result<BucketRecord, ApiError> {
    let mut tx = app_pool
        .begin_rls_tx(principal)
        .await
        .map_err(ApiError::from)?;

    if let Some(b_ref) = bucket_ref.map(str::trim).filter(|s| !s.is_empty()) {
        let b = resolve_bucket_rls(&mut tx, principal.org_id, b_ref)
            .await
            .map_err(ApiError::from)?;
        enforce_api_key_arn_scope(principal, &b.arn).map_err(ApiError::from)?;
        tx.commit()
            .await
            .map_err(|e| ApiError::from(OpenRagError::Database(format!("commit bucket resolve: {e}"))))?;
        return Ok(b);
    }

    let existing_row = sqlx::query(
        "SELECT id, org_id, dept_id, owner_principal_id, slug, name, access_scope,
                allowed_group_ids, denied_group_ids, min_auth_level, arn, created_at
         FROM corpus.buckets
         WHERE org_id = $1
           AND iam.rls_can_access(org_id, dept_id, owner_principal_id, access_scope,
                                  allowed_group_ids, arn, min_auth_level, denied_group_ids,
                                  'write', 'bucket', id)
         ORDER BY created_at ASC
         LIMIT 1",
    )
    .bind(principal.org_id)
    .fetch_optional(&mut **tx.tx_mut())
    .await
    .map_err(|e| ApiError::from(OpenRagError::Database(format!("find writable bucket: {e}"))))?;

    if let Some(row) = existing_row {
        let b = row_to_bucket(&row).map_err(ApiError::from)?;
        enforce_api_key_arn_scope(principal, &b.arn).map_err(ApiError::from)?;
        tx.commit()
            .await
            .map_err(|e| ApiError::from(OpenRagError::Database(format!("commit bucket find: {e}"))))?;
        return Ok(b);
    }

    let (dept_id, bucket_scope) = match principal.dept_ids.first().copied() {
        Some(id) => (id, requested_scope.unwrap_or(AccessScope::Dept)),
        None => {
            let fallback_dept: Uuid = sqlx::query_scalar(
                "SELECT id FROM iam.departments WHERE org_id = $1 ORDER BY created_at ASC LIMIT 1",
            )
            .bind(principal.org_id)
            .fetch_optional(worker_pool.inner())
            .await
            .map_err(|e| ApiError::from(OpenRagError::Database(format!("lookup department: {e}"))))?
            .ok_or_else(|| {
                ApiError::bad_request("missing_department", "organization has no departments")
            })?;
            (fallback_dept, AccessScope::Private)
        }
    };

    let (org_slug, dept_path) = resolve_org_and_dept_paths(state, principal.org_id, dept_id)
        .await
        .map_err(ApiError::from)?;

    let bucket_id = Uuid::now_v7();
    let short_id = &bucket_id.simple().to_string()[..8];
    let slug = format!("app-default-{short_id}");
    let bucket_arn = Arn::new(
        &org_slug,
        &dept_path,
        bucket_scope,
        ArnResourceType::Bucket,
        AuthLevel::MIN,
        bucket_id.to_string(),
    )
    .map_err(ApiError::from)?
    .to_canonical();

    enforce_api_key_arn_scope(principal, &bucket_arn).map_err(ApiError::from)?;

    let row = sqlx::query(
        "INSERT INTO corpus.buckets (
            id, org_id, dept_id, owner_principal_id, slug, name,
            access_scope, allowed_group_ids, denied_group_ids, min_auth_level, arn
         ) VALUES ($1, $2, $3, $4, $5, 'Default Upload Space', $6, $7, $8, 1, $9)
         RETURNING id, org_id, dept_id, owner_principal_id, slug, name,
                   access_scope, allowed_group_ids, denied_group_ids, min_auth_level,
                   arn, created_at",
    )
    .bind(bucket_id)
    .bind(principal.org_id)
    .bind(dept_id)
    .bind(principal.principal_id)
    .bind(&slug)
    .bind(bucket_scope.as_str())
    .bind(Vec::<Uuid>::new())
    .bind(Vec::<Uuid>::new())
    .bind(&bucket_arn)
    .fetch_one(&mut **tx.tx_mut())
    .await
    .map_err(map_sqlx_err("insert default app bucket"))?;

    let created = row_to_bucket(&row).map_err(ApiError::from)?;
    tx.commit()
        .await
        .map_err(|e| ApiError::from(OpenRagError::Database(format!("commit default bucket: {e}"))))?;
    Ok(created)
}

async fn app_page_thumbnail_handler(
    State(state): State<Arc<AppState>>,
    AuthPrincipal(principal): AuthPrincipal,
    Path(page_id): Path<Uuid>,
) -> Result<Response, ApiError> {
    let app_pool = state.require_app_pool().map_err(ApiError::from)?;
    let worker_pool = state.require_worker_pool().map_err(ApiError::from)?;

    let mut tx = app_pool
        .begin_rls_tx(&principal)
        .await
        .map_err(ApiError::from)?;

    let row = sqlx::query(
        "SELECT thumb_uri, redacted_thumb_uri, arn
         FROM corpus.pages
         WHERE org_id = $1 AND id = $2",
    )
    .bind(principal.org_id)
    .bind(page_id)
    .fetch_optional(&mut **tx.tx_mut())
    .await
    .map_err(|e| ApiError::from(OpenRagError::Database(format!("query page thumbnail: {e}"))))?;

    tx.commit()
        .await
        .map_err(|e| ApiError::from(OpenRagError::Database(format!("commit thumbnail tx: {e}"))))?;

    let Some(r) = row else {
        let exists: bool = sqlx::query_scalar(
            "SELECT EXISTS(SELECT 1 FROM corpus.pages WHERE org_id = $1 AND id = $2)",
        )
        .bind(principal.org_id)
        .bind(page_id)
        .fetch_one(worker_pool.inner())
        .await
        .map_err(|e| ApiError::from(OpenRagError::Database(format!("probe page existence: {e}"))))?;

        return if exists {
            Err(ApiError::forbidden(
                "forbidden",
                "access denied to page thumbnail",
            ))
        } else {
            Err(ApiError::not_found(format!("page {page_id} not found")))
        };
    };

    let arn: String = r.try_get("arn").map_err(db_col_err)?;
    enforce_api_key_arn_scope(&principal, &arn).map_err(ApiError::from)?;

    let thumb_uri: Option<String> = r.try_get("thumb_uri").map_err(db_col_err)?;
    let redacted_uri: Option<String> = r.try_get("redacted_thumb_uri").map_err(db_col_err)?;
    let target_uri = redacted_uri
        .as_deref()
        .or(thumb_uri.as_deref())
        .ok_or_else(|| ApiError::not_found(format!("thumbnail for page {page_id} not found")))?;

    let bytes = match state.blob_store.get_bytes(target_uri) {
        Ok(b) => b,
        Err(err) => {
            if target_uri.starts_with("mem://") {
                return Err(ApiError::from(err));
            }
            LocalThumbnailStore::new()
                .read_bytes(target_uri)
                .map_err(ApiError::from)?
        }
    };

    let mime = detect_image_mime(&bytes);
    Ok((StatusCode::OK, [(CONTENT_TYPE, mime)], bytes).into_response())
}

fn detect_image_mime(bytes: &[u8]) -> &'static str {
    if bytes.len() >= 12 && &bytes[0..4] == b"RIFF" && &bytes[8..12] == b"WEBP" {
        "image/webp"
    } else if bytes.starts_with(b"\x89PNG\r\n\x1a\n") {
        "image/png"
    } else if bytes.starts_with(&[0xFF, 0xD8, 0xFF]) {
        "image/jpeg"
    } else {
        "application/octet-stream"
    }
}

#[derive(Debug, Deserialize)]
struct AppFeedbackRequest {
    #[serde(alias = "query")]
    query_text: String,
    verdict: String,
    #[serde(default, alias = "answer")]
    response_text: Option<String>,
    #[serde(default)]
    cited_chunk_ids: Vec<Uuid>,
    #[serde(default)]
    expected_doc_ids: Vec<Uuid>,
    #[serde(default)]
    selected_doc_id: Option<Uuid>,
    #[serde(default)]
    selected_page_id: Option<Uuid>,
    #[serde(default)]
    comment: Option<String>,
}

async fn app_feedback_handler(
    State(state): State<Arc<AppState>>,
    AuthPrincipal(principal): AuthPrincipal,
    Json(req): Json<AppFeedbackRequest>,
) -> Result<(StatusCode, Json<Value>), ApiError> {
    let verdict = FeedbackVerdict::from_str(&req.verdict).map_err(ApiError::from)?;
    let mut expected_doc_ids = req.expected_doc_ids;
    if let Some(doc_id) = req.selected_doc_id {
        if !expected_doc_ids.contains(&doc_id) {
            expected_doc_ids.push(doc_id);
        }
    }
    let mut cited_chunk_ids = req.cited_chunk_ids;
    if cited_chunk_ids.is_empty() {
        if let Some(page_id) = req.selected_page_id {
            cited_chunk_ids.push(page_id);
        }
    }
    let input = SubmitFeedbackInput {
        query_text: req.query_text,
        response_text: req.response_text,
        cited_chunk_ids,
        expected_doc_ids,
        verdict,
        comment: req.comment,
    };
    input.validate().map_err(ApiError::from)?;

    let store = state.pg_eval_store().map_err(ApiError::from)?;
    let fb = FeedbackManager::new(Arc::new(store))
        .submit_feedback_rls(&principal, input)
        .await
        .map_err(ApiError::from)?;

    Ok((
        StatusCode::CREATED,
        Json(json!({
            "id": fb.id,
            "feedback_id": fb.id,
            "org_id": fb.org_id,
            "principal_id": fb.principal_id,
            "query_text": fb.query_text,
            "response_text": fb.response_text,
            "cited_chunk_ids": fb.cited_chunk_ids,
            "verdict": fb.verdict,
            "comment": fb.comment,
            "promoted_golden_item_id": fb.promoted_golden_item_id,
            "created_at": fb.created_at,
        })),
    ))
}

#[derive(Debug, Deserialize)]
struct AppWikiQuery {
    tag: Option<String>,
    q: Option<String>,
    limit: Option<i64>,
}

async fn app_wiki_list_handler(
    State(state): State<Arc<AppState>>,
    AuthPrincipal(principal): AuthPrincipal,
    Query(q): Query<AppWikiQuery>,
) -> Result<Json<Value>, ApiError> {
    let app_pool = state.require_app_pool().map_err(ApiError::from)?;
    let limit = q.limit.unwrap_or(100).clamp(1, 500);
    let tag_filter = q
        .tag
        .as_deref()
        .map(normalize_tag)
        .filter(|s| !s.is_empty());
    let text_filter = q
        .q
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(ToString::to_string);

    let mut tx = app_pool
        .begin_rls_tx(&principal)
        .await
        .map_err(ApiError::from)?;

    let rows = sqlx::query(
        "SELECT id, org_id, bucket_id, dept_id, owner_principal_id, access_scope,
                allowed_group_ids, denied_group_ids, min_auth_level, arn, slug, title,
                body_md, tags, status, version, created_at, updated_at
         FROM wiki.pages
         WHERE org_id = $1
           AND status = 'active'
           AND ($2::text IS NULL OR $2 = ANY(tags))
           AND ($3::text IS NULL OR title ILIKE '%' || $3 || '%' OR body_md ILIKE '%' || $3 || '%')
         ORDER BY updated_at DESC
         LIMIT $4",
    )
    .bind(principal.org_id)
    .bind(tag_filter.as_deref())
    .bind(text_filter.as_deref())
    .bind(limit)
    .fetch_all(&mut **tx.tx_mut())
    .await
    .map_err(|e| ApiError::from(OpenRagError::Database(format!("list wiki.pages: {e}"))))?;

    tx.commit()
        .await
        .map_err(|e| ApiError::from(OpenRagError::Database(format!("commit wiki list: {e}"))))?;

    let mut pages = Vec::with_capacity(rows.len());
    for row in &rows {
        let arn: String = row.try_get("arn").map_err(db_col_err)?;
        if enforce_api_key_arn_scope(&principal, &arn).is_err() {
            continue;
        }
        pages.push(wiki_row_to_json(row)?);
    }

    Ok(Json(json!({ "pages": pages, "items": pages })))
}

async fn app_wiki_get_handler(
    State(state): State<Arc<AppState>>,
    AuthPrincipal(principal): AuthPrincipal,
    Path(page_id): Path<Uuid>,
) -> Result<Json<Value>, ApiError> {
    let app_pool = state.require_app_pool().map_err(ApiError::from)?;
    let worker_pool = state.require_worker_pool().map_err(ApiError::from)?;

    let mut tx = app_pool
        .begin_rls_tx(&principal)
        .await
        .map_err(ApiError::from)?;

    let row = sqlx::query(
        "SELECT id, org_id, bucket_id, dept_id, owner_principal_id, access_scope,
                allowed_group_ids, denied_group_ids, min_auth_level, arn, slug, title,
                body_md, tags, status, version, created_at, updated_at
         FROM wiki.pages
         WHERE org_id = $1 AND id = $2 AND status = 'active'",
    )
    .bind(principal.org_id)
    .bind(page_id)
    .fetch_optional(&mut **tx.tx_mut())
    .await
    .map_err(|e| ApiError::from(OpenRagError::Database(format!("get wiki.pages: {e}"))))?;

    tx.commit()
        .await
        .map_err(|e| ApiError::from(OpenRagError::Database(format!("commit wiki get: {e}"))))?;

    let Some(r) = row else {
        let exists: bool = sqlx::query_scalar(
            "SELECT EXISTS(SELECT 1 FROM wiki.pages WHERE org_id = $1 AND id = $2 AND status = 'active')",
        )
        .bind(principal.org_id)
        .bind(page_id)
        .fetch_one(worker_pool.inner())
        .await
        .map_err(|e| ApiError::from(OpenRagError::Database(format!("probe wiki existence: {e}"))))?;

        return if exists {
            Err(ApiError::forbidden(
                "forbidden",
                "access denied to wiki page",
            ))
        } else {
            Err(ApiError::not_found(format!("wiki page {page_id} not found")))
        };
    };

    let arn: String = r.try_get("arn").map_err(db_col_err)?;
    enforce_api_key_arn_scope(&principal, &arn).map_err(ApiError::from)?;
    Ok(Json(wiki_row_to_json(&r)?))
}

fn wiki_row_to_json(row: &sqlx::postgres::PgRow) -> Result<Value, ApiError> {
    let id: Uuid = row.try_get("id").map_err(db_col_err)?;
    let org_id: Uuid = row.try_get("org_id").map_err(db_col_err)?;
    let bucket_id: Option<Uuid> = row.try_get("bucket_id").map_err(db_col_err)?;
    let dept_id: Uuid = row.try_get("dept_id").map_err(db_col_err)?;
    let owner_principal_id: Uuid = row.try_get("owner_principal_id").map_err(db_col_err)?;
    let access_scope: String = row.try_get("access_scope").map_err(db_col_err)?;
    let allowed_group_ids: Vec<Uuid> = row.try_get("allowed_group_ids").map_err(db_col_err)?;
    let denied_group_ids: Vec<Uuid> = row.try_get("denied_group_ids").map_err(db_col_err)?;
    let min_auth_level: i16 = row.try_get("min_auth_level").map_err(db_col_err)?;
    let arn: String = row.try_get("arn").map_err(db_col_err)?;
    let slug: String = row.try_get("slug").map_err(db_col_err)?;
    let title: String = row.try_get("title").map_err(db_col_err)?;
    let body_md: String = row.try_get("body_md").map_err(db_col_err)?;
    let tags: Vec<String> = row.try_get("tags").map_err(db_col_err)?;
    let status: String = row.try_get("status").map_err(db_col_err)?;
    let version: i32 = row.try_get("version").map_err(db_col_err)?;
    let created_at: DateTime<Utc> = row.try_get("created_at").map_err(db_col_err)?;
    let updated_at: DateTime<Utc> = row.try_get("updated_at").map_err(db_col_err)?;

    Ok(json!({
        "id": id,
        "page_id": id,
        "org_id": org_id,
        "bucket_id": bucket_id,
        "dept_id": dept_id,
        "owner_principal_id": owner_principal_id,
        "access_scope": access_scope,
        "allowed_group_ids": allowed_group_ids,
        "denied_group_ids": denied_group_ids,
        "min_auth_level": min_auth_level,
        "arn": arn,
        "slug": slug,
        "title": title,
        "body_md": body_md,
        "tags": tags,
        "status": status,
        "version": version,
        "created_at": created_at,
        "updated_at": updated_at,
    }))
}

#[derive(Debug, Deserialize)]
struct AppMemoryQuery {
    q: Option<String>,
    tag: Option<String>,
    limit: Option<i64>,
    backend: Option<MemoryBackendKind>,
}

async fn app_memory_list_handler(
    State(state): State<Arc<AppState>>,
    AuthPrincipal(principal): AuthPrincipal,
    Query(q): Query<AppMemoryQuery>,
) -> Result<Json<Value>, ApiError> {
    let router = state.active_memory_router().await.map_err(ApiError::from)?;
    let limit = q.limit.unwrap_or(100).clamp(1, 500);
    let tag_filter = q
        .tag
        .as_deref()
        .map(normalize_tag)
        .filter(|t| !t.is_empty())
        .map(|t| TagFilter {
            tags_any: vec![t],
            tags_all: Vec::new(),
            tags_exclude: Vec::new(),
            expand_tag_aliases: true,
        });

    let entries = if let Some(query_str) = q.q.as_deref().map(str::trim).filter(|s| !s.is_empty()) {
        router
            .search(
                &principal,
                query_str,
                tag_filter.as_ref(),
                limit as usize,
                q.backend,
            )
            .await
            .map_err(ApiError::from)?
    } else {
        router
            .list_entries(&principal, tag_filter.as_ref(), limit, q.backend)
            .await
            .map_err(ApiError::from)?
    };

    let items: Vec<Value> = entries
        .into_iter()
        .filter(|e| enforce_api_key_arn_scope(&principal, &e.arn).is_ok())
        .map(|e| memory_entry_to_json(&e))
        .collect();

    let backend_name = q
        .backend
        .unwrap_or_else(|| router.primary_kind())
        .as_str();

    Ok(Json(json!({
        "entries": items,
        "items": items,
        "memories": items,
        "backend": backend_name,
    })))
}

fn memory_entry_to_json(entry: &MemoryEntry) -> Value {
    json!({
        "id": entry.id,
        "memory_id": entry.id,
        "org_id": entry.org_id,
        "bucket_id": entry.bucket_id,
        "dept_id": entry.dept_id,
        "owner_principal_id": entry.owner_principal_id,
        "access_scope": entry.access_scope,
        "allowed_group_ids": entry.allowed_group_ids,
        "denied_group_ids": entry.denied_group_ids,
        "min_auth_level": entry.min_auth_level,
        "arn": entry.arn,
        "acl_scope_key": entry.acl_scope_key,
        "memory_kind": entry.memory_kind,
        "title": entry.title,
        "effective_text": entry.effective_text,
        "tags": entry.tags,
        "source_episode_id": entry.source_episode_id,
        "source_document_id": entry.source_document_id,
        "promoted_wiki_id": entry.promoted_wiki_id,
        "confidence": entry.confidence,
        "status": entry.status,
        "superseded_by": entry.superseded_by,
        "version": entry.version,
        "valid_from": entry.valid_from,
        "valid_to": entry.valid_to,
        "created_at": entry.created_at,
        "updated_at": entry.updated_at,
    })
}

async fn probe_memory_access(
    state: &AppState,
    router: &MemoryRouter,
    principal: &PrincipalContext,
    memory_id: Uuid,
    backend_override: Option<MemoryBackendKind>,
) -> Result<MemoryEntry, ApiError> {
    if let Some(entry) = router
        .get_entry(principal, memory_id, backend_override)
        .await
        .map_err(ApiError::from)?
    {
        enforce_api_key_arn_scope(principal, &entry.arn).map_err(ApiError::from)?;
        return Ok(entry);
    }

    if let Some(worker_pool) = state.worker_pool.as_ref() {
        let status_opt: Option<String> = sqlx::query_scalar(
            "SELECT status FROM memory.entries WHERE org_id = $1 AND id = $2",
        )
        .bind(principal.org_id)
        .bind(memory_id)
        .fetch_optional(worker_pool.inner())
        .await
        .map_err(|e| ApiError::from(OpenRagError::Database(format!("probe memory.entries: {e}"))))?;

        return match status_opt.as_deref() {
            None => Err(ApiError::not_found(format!(
                "memory entry {memory_id} not found"
            ))),
            Some("active") => Err(ApiError::forbidden(
                "forbidden",
                "access denied to memory entry",
            )),
            Some(other) => Err(ApiError::conflict(format!(
                "memory entry {memory_id} is {other} and cannot be modified"
            ))),
        };
    }

    Err(ApiError::not_found(format!(
        "memory entry {memory_id} not found"
    )))
}

#[derive(Debug, Deserialize)]
struct AppMemoryPatchRequest {
    #[serde(alias = "entry_id")]
    memory_id: Uuid,
    #[serde(alias = "patch_op")]
    op: String,
    #[serde(
        default,
        alias = "new_text",
        alias = "content",
        alias = "new_body",
        alias = "new_body_text",
        alias = "body"
    )]
    text: Option<String>,
    #[serde(default)]
    reason: Option<String>,
    #[serde(default)]
    hard_purge: bool,
    #[serde(default)]
    backend: Option<MemoryBackendKind>,
}

async fn app_memory_patch_handler(
    State(state): State<Arc<AppState>>,
    AuthPrincipal(principal): AuthPrincipal,
    Json(req): Json<AppMemoryPatchRequest>,
) -> Result<Json<Value>, ApiError> {
    let op = PatchOp::from_str(&req.op).map_err(ApiError::from)?;
    if matches!(op, PatchOp::PromoteScope | PatchOp::PromoteWiki) {
        return Err(ApiError::bad_request(
            "invalid_patch_op",
            "use /api/app/memory/promote for promotion operations",
        ));
    }

    let router = state.active_memory_router().await.map_err(ApiError::from)?;
    let _entry = probe_memory_access(&state, &router, &principal, req.memory_id, req.backend).await?;

    let neutralized_text = req
        .text
        .as_deref()
        .map(neutralize_literal_pii_tokens);

    let patch = router
        .apply_patch(
            &principal,
            ApplyPatchInput {
                memory_entry_id: req.memory_id,
                patch_op: op,
                new_text: neutralized_text,
                reason: req.reason,
                actor_type: ActorType::Human,
                hard_purge: req.hard_purge,
            },
            req.backend,
        )
        .await
        .map_err(ApiError::from)?;

    Ok(Json(json!({
        "id": patch.id,
        "patch_id": patch.id,
        "memory_id": patch.memory_entry_id,
        "memory_entry_id": patch.memory_entry_id,
        "patch_op": patch.patch_op,
        "prev_version": patch.prev_version,
        "new_version": patch.new_version,
        "prev_text": patch.prev_text,
        "new_text": patch.new_text,
        "effective_text": patch.new_text,
        "reason": patch.reason,
        "created_at": patch.created_at,
    })))
}

#[derive(Debug, Deserialize)]
struct AppMemoryRetractRequest {
    #[serde(alias = "entry_id")]
    memory_id: Uuid,
    #[serde(default)]
    reason: Option<String>,
    #[serde(default)]
    backend: Option<MemoryBackendKind>,
}

async fn app_memory_retract_handler(
    State(state): State<Arc<AppState>>,
    AuthPrincipal(principal): AuthPrincipal,
    Json(req): Json<AppMemoryRetractRequest>,
) -> Result<Json<Value>, ApiError> {
    let router = state.active_memory_router().await.map_err(ApiError::from)?;
    let _entry = probe_memory_access(&state, &router, &principal, req.memory_id, req.backend).await?;

    let patch = router
        .apply_patch(
            &principal,
            ApplyPatchInput {
                memory_entry_id: req.memory_id,
                patch_op: PatchOp::Retract,
                new_text: None,
                reason: req.reason,
                actor_type: ActorType::Human,
                hard_purge: false,
            },
            req.backend,
        )
        .await
        .map_err(ApiError::from)?;

    Ok(Json(json!({
        "id": patch.id,
        "patch_id": patch.id,
        "memory_id": patch.memory_entry_id,
        "status": "retracted",
        "prev_version": patch.prev_version,
        "new_version": patch.new_version,
        "created_at": patch.created_at,
    })))
}

#[derive(Debug, Deserialize)]
struct AppMemoryPromoteRequest {
    #[serde(alias = "entry_id")]
    memory_id: Uuid,
    #[serde(alias = "target", alias = "target_scope")]
    to: String,
    #[serde(default, alias = "bucket_id")]
    bucket: Option<String>,
    #[serde(default)]
    target_group_ids: Vec<Uuid>,
    #[serde(default)]
    min_auth_level: Option<i16>,
    #[serde(default)]
    wiki_slug: Option<String>,
    #[serde(default)]
    wiki_title: Option<String>,
    #[serde(default)]
    reason: Option<String>,
    #[serde(default)]
    backend: Option<MemoryBackendKind>,
}

async fn app_memory_promote_handler(
    State(state): State<Arc<AppState>>,
    AuthPrincipal(principal): AuthPrincipal,
    Json(req): Json<AppMemoryPromoteRequest>,
) -> Result<Json<Value>, ApiError> {
    let target = PromoteTarget::from_str(&req.to).map_err(ApiError::from)?;
    let min_auth_level = match req.min_auth_level {
        Some(lvl) => Some(AuthLevel::try_from(lvl).map_err(ApiError::from)?),
        None => None,
    };

    let router = state.active_memory_router().await.map_err(ApiError::from)?;
    let entry = probe_memory_access(&state, &router, &principal, req.memory_id, req.backend).await?;

    let (org_slug, dept_path) = if state.worker_pool.is_some() {
        resolve_org_and_dept_paths(&state, principal.org_id, entry.dept_id)
            .await
            .map_err(ApiError::from)?
    } else {
        (principal.org_id.to_string(), entry.dept_id.to_string())
    };

    let target_bucket_id = if target == PromoteTarget::Wiki {
        let Some(app_pool) = state.app_pool.as_ref() else {
            return Err(ApiError::from(OpenRagError::Config(
                "PostgreSQL app pool required for wiki promotion".into(),
            )));
        };
        let mut tx = app_pool
            .begin_rls_tx(&principal)
            .await
            .map_err(ApiError::from)?;

        let bucket_id = if let Some(b_ref) = req.bucket.as_deref().map(str::trim).filter(|s| !s.is_empty())
        {
            match resolve_bucket_rls(&mut tx, principal.org_id, b_ref).await {
                Ok(b) => {
                    enforce_api_key_arn_scope(&principal, &b.arn).map_err(ApiError::from)?;
                    b.id
                }
                Err(OpenRagError::NotFound(msg)) => {
                    if let Some(worker_pool) = state.worker_pool.as_ref() {
                        let parsed_uuid = Uuid::parse_str(b_ref).ok();
                        let exists: Option<i32> = sqlx::query_scalar(
                            "SELECT 1 FROM corpus.buckets
                             WHERE org_id = $1 AND (($2::uuid IS NOT NULL AND id = $2) OR slug = $3)
                             LIMIT 1",
                        )
                        .bind(principal.org_id)
                        .bind(parsed_uuid)
                        .bind(b_ref)
                        .fetch_optional(worker_pool.inner())
                        .await
                        .map_err(|e| {
                            ApiError::from(OpenRagError::Database(format!(
                                "probe corpus.buckets existence: {e}"
                            )))
                        })?;
                        if exists.is_some() {
                            return Err(ApiError::forbidden(
                                "forbidden",
                                "access denied to target bucket",
                            ));
                        }
                    }
                    return Err(ApiError::not_found(msg));
                }
                Err(e) => return Err(ApiError::from(e)),
            }
        } else {
            let row = sqlx::query(
                "SELECT id, arn
                 FROM corpus.buckets
                 WHERE org_id = $1
                   AND iam.rls_can_access(org_id, dept_id, owner_principal_id, access_scope,
                                          allowed_group_ids, arn, min_auth_level, denied_group_ids,
                                          'write', 'bucket', id)
                 ORDER BY created_at ASC
                 LIMIT 1",
            )
            .bind(principal.org_id)
            .fetch_optional(&mut **tx.tx_mut())
            .await
            .map_err(|e| {
                ApiError::from(OpenRagError::Database(format!(
                    "find writable wiki bucket: {e}"
                )))
            })?
            .ok_or_else(|| {
                ApiError::forbidden(
                    "forbidden",
                    "no writable bucket available for wiki promotion",
                )
            })?;
            let id: Uuid = row.try_get("id").map_err(db_col_err)?;
            let arn: String = row.try_get("arn").map_err(db_col_err)?;
            enforce_api_key_arn_scope(&principal, &arn).map_err(ApiError::from)?;
            id
        };
        tx.commit().await.map_err(|e| {
            ApiError::from(OpenRagError::Database(format!("commit wiki bucket resolve: {e}")))
        })?;
        Some(bucket_id)
    } else {
        None
    };

    let neutralized_title = req
        .wiki_title
        .as_deref()
        .or(entry.title.as_deref())
        .map(neutralize_literal_pii_tokens);

    let res = router
        .promote(
            &principal,
            PromoteMemoryInput {
                memory_entry_id: req.memory_id,
                target,
                target_bucket_id,
                target_group_ids: req.target_group_ids,
                min_auth_level,
                wiki_slug: req.wiki_slug,
                wiki_title: neutralized_title,
                org_slug,
                dept_path,
                reason: req.reason,
            },
            req.backend,
        )
        .await
        .map_err(ApiError::from)?;

    if let (Some(wiki_id), Some(app_pool)) = (res.promoted_wiki_id, state.app_pool.as_ref()) {
        let neutralized_body = neutralize_literal_pii_tokens(&res.entry.effective_text);
        if neutralized_body != res.entry.effective_text {
            let body_sha = openrag_db::sha256_hex(&neutralized_body);
            let emb = state
                .text_embedder
                .embed_text(&neutralized_body)
                .await
                .map_err(ApiError::from)?;
            let mut tx = app_pool
                .begin_rls_tx(&principal)
                .await
                .map_err(ApiError::from)?;
            sqlx::query(
                "UPDATE wiki.pages
                 SET body_md = $1, bm25_text = $1, body_sha256 = $2, embedding = $3
                 WHERE org_id = $4 AND id = $5",
            )
            .bind(&neutralized_body)
            .bind(&body_sha)
            .bind(pgvector::Vector::from(emb))
            .bind(principal.org_id)
            .bind(wiki_id)
            .execute(&mut **tx.tx_mut())
            .await
            .map_err(|e| {
                ApiError::from(OpenRagError::Database(format!(
                    "neutralize promoted wiki page: {e}"
                )))
            })?;
            sqlx::query(
                "UPDATE wiki.revisions
                 SET body_md = $1, body_sha256 = $2
                 WHERE org_id = $3 AND page_id = $4",
            )
            .bind(&neutralized_body)
            .bind(&body_sha)
            .bind(principal.org_id)
            .bind(wiki_id)
            .execute(&mut **tx.tx_mut())
            .await
            .map_err(|e| {
                ApiError::from(OpenRagError::Database(format!(
                    "neutralize promoted wiki revision: {e}"
                )))
            })?;
            tx.commit().await.map_err(|e| {
                ApiError::from(OpenRagError::Database(format!(
                    "commit wiki neutralization: {e}"
                )))
            })?;
            if let Some(worker_pool) = state.worker_pool.as_ref() {
                sqlx::query(
                    "UPDATE indexing.outbox
                     SET content_sha256 = $1
                     WHERE org_id = $2 AND target_table = 'wiki.pages' AND target_id = $3",
                )
                .bind(&body_sha)
                .bind(principal.org_id)
                .bind(wiki_id)
                .execute(worker_pool.inner())
                .await
                .map_err(|e| {
                    ApiError::from(OpenRagError::Database(format!(
                        "neutralize promoted wiki outbox: {e}"
                    )))
                })?;
            }
        }
    }

    Ok(Json(json!({
        "id": res.entry.id,
        "memory_id": res.entry.id,
        "entry": memory_entry_to_json(&res.entry),
        "patch": res.patch,
        "promoted_wiki_id": res.promoted_wiki_id,
        "wiki_page_id": res.promoted_wiki_id,
        "page_id": res.promoted_wiki_id,
    })))
}

async fn app_connect_handler(
    AuthPrincipal(principal): AuthPrincipal,
) -> Result<Json<Value>, ApiError> {
    let base_url = std::env::var("OPENRAG_PUBLIC_URL")
        .ok()
        .map(|s| s.trim().trim_end_matches('/').to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "http://127.0.0.1:8080".to_string());
    let mcp_http_url = format!("{base_url}/mcp");

    let claude_desktop_json = json!({
        "mcpServers": {
            "openrag": {
                "command": "openrag-mcp",
                "args": [],
                "env": {
                    "OPENRAG_API_URL": base_url,
                    "OPENRAG_API_KEY": "<YOUR_API_KEY>"
                }
            }
        }
    });

    let claude_code_http = json!({
        "mcpServers": {
            "openrag": {
                "type": "http",
                "url": mcp_http_url,
                "headers": {
                    "Authorization": "Bearer <YOUR_API_KEY>"
                }
            }
        }
    });

    let cursor_json = json!({
        "mcpServers": {
            "openrag-stdio": {
                "command": "openrag",
                "args": ["mcp", "--stdio"],
                "env": {
                    "OPENRAG_API_KEY": "<YOUR_API_KEY>"
                }
            },
            "openrag-http": {
                "url": mcp_http_url,
                "headers": {
                    "Authorization": "Bearer <YOUR_API_KEY>"
                }
            }
        }
    });

    let codex_toml = format!(
        "[mcp_servers.openrag]\ncommand = \"openrag\"\nargs = [\"mcp\", \"--stdio\"]\n\n[mcp_servers.openrag_http]\nurl = \"{mcp_http_url}\"\nbearer_token_env_var = \"OPENRAG_API_KEY\""
    );

    Ok(Json(json!({
        "mcp_http_path": "/mcp",
        "mcp_http_url": mcp_http_url,
        "mcp_stdio_command": "openrag mcp --stdio",
        "mcp_stdio_binary": "openrag-mcp",
        "cli_exchange_endpoint": "/v1/auth/cli/code-exchange",
        "principal": {
            "org_id": principal.org_id,
            "principal_id": principal.principal_id,
            "auth_level": principal.auth_level.as_u8(),
            "can_unmask_pii": principal.can_unmask_pii,
            "is_org_admin": principal.is_org_admin,
            "scoped_arns": principal.scoped_arns,
        },
        "clients": {
            "claude_desktop": {
                "display_name": "Claude Desktop",
                "config_file": "claude_desktop_config.json",
                "stdio": claude_desktop_json,
                "http_url": mcp_http_url,
            },
            "claude_code": {
                "display_name": "Claude Code",
                "cli_command": "claude mcp add openrag -- openrag mcp --stdio",
                "config_file": ".mcp.json",
                "http": claude_code_http,
            },
            "cursor": {
                "display_name": "Cursor",
                "config_file": ".cursor/mcp.json",
                "config": cursor_json,
            },
            "codex": {
                "display_name": "Codex",
                "config_file": "~/.codex/config.toml",
                "toml": codex_toml,
                "stdio_command": "openrag mcp --stdio",
                "http_url": mcp_http_url,
            },
        },
    })))
}
