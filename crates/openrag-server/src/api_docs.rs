//! Documents as a list, spreadsheets read on demand, and the watched-folder status.
//!
//! - `GET  /v2/buckets/{id}/documents` — documents in a bucket the caller may read (RLS); the
//!   route itself is declared with the upload route in `api_core`.
//! - `GET  /v2/documents/{id}/sheet` — rows of a spreadsheet, read from its watched file.
//! - `GET  /v2/watch`, `POST /v2/watch/scan` — what the folder watcher is doing (org admins).

use std::collections::HashMap;
use std::sync::Arc;

use axum::extract::{Path, Query, State};
use axum::http::StatusCode;
use axum::routing::{get, post};
use axum::{Json, Router};
use chrono::{DateTime, Utc};
use openrag_core::{OpenRagError, PrincipalContext};
use openrag_parse::DocumentFormat;
use serde::Deserialize;
use serde_json::{json, Value};
use sqlx::Row;
use uuid::Uuid;

use crate::api_core::resolve_bucket_rls;
use crate::auth::{enforce_api_key_arn_scope, AuthPrincipal};
use crate::error::ApiError;
use crate::state::AppState;
use crate::watch;

pub fn docs_routes() -> Router<Arc<AppState>> {
    Router::new()
        .route("/v2/documents/{id}/sheet", get(read_sheet_handler))
        .route("/v2/watch", get(watch_status_handler))
        .route("/v2/watch/scan", post(watch_scan_handler))
}

fn db_err(what: &'static str) -> impl Fn(sqlx::Error) -> ApiError {
    move |e| ApiError::from(OpenRagError::Database(format!("{what}: {e}")))
}

fn require_admin(principal: &PrincipalContext) -> Result<(), ApiError> {
    if principal.is_org_admin {
        Ok(())
    } else {
        Err(ApiError::forbidden("admin_required", "only organization admins can see the documents folder status"))
    }
}

/// `Some(Xlsx | Csv)` for spreadsheets, by MIME type.
fn spreadsheet_format(mime: &str) -> Option<DocumentFormat> {
    let m = mime.to_ascii_lowercase();
    if m.contains("spreadsheetml.sheet") {
        Some(DocumentFormat::Xlsx)
    } else if m.starts_with("text/csv") {
        Some(DocumentFormat::Csv)
    } else {
        None
    }
}

/// Where a document came from, as shown to people and AI clients.
pub(crate) fn source_json(mime: &str, source_uri: Option<&str>, key: Option<&str>, host: Option<&str>, rel: Option<&str>) -> Value {
    let path = host.or(key).or(source_uri);
    let mut out = json!({ "path": path, "folder_path": rel });
    if spreadsheet_format(mime).is_some() {
        out["kind"] = json!("spreadsheet");
        out["read_with"] = json!("rag_read_sheet");
    }
    out
}

/// `source` for each hit of a search response: original path, path inside the folder, and how
/// to read spreadsheets. Hits that are not document chunks are left as they are.
pub(crate) async fn add_sources(state: &AppState, org: Uuid, hits: &mut [Value]) -> Result<(), ApiError> {
    let ids: Vec<Uuid> = hits
        .iter()
        .filter_map(|h| h.get("document_id").and_then(Value::as_str).and_then(|s| Uuid::parse_str(s).ok()))
        .collect();
    if ids.is_empty() {
        return Ok(());
    }
    let worker = state.require_worker_pool().map_err(ApiError::from)?;
    let rows = sqlx::query(
        "SELECT id, mime_type, source_uri, metadata->>'external_key' AS key,
                metadata->>'watch_host_path' AS host, metadata->>'watch_rel_path' AS rel
         FROM corpus.documents WHERE org_id = $1 AND id = ANY($2)",
    )
    .bind(org)
    .bind(&ids)
    .fetch_all(worker.inner())
    .await
    .map_err(db_err("look up hit sources"))?;
    let mut by_id: HashMap<Uuid, Value> = HashMap::new();
    for r in rows {
        let id: Uuid = r.try_get("id").map_err(db_err("id"))?;
        let mime: Option<String> = r.try_get("mime_type").map_err(db_err("mime_type"))?;
        let get = |c: &'static str| r.try_get::<Option<String>, _>(c).map_err(db_err(c));
        let (uri, key, host, rel) = (get("source_uri")?, get("key")?, get("host")?, get("rel")?);
        by_id.insert(id, source_json(mime.as_deref().unwrap_or(""), uri.as_deref(), key.as_deref(), host.as_deref(), rel.as_deref()));
    }
    for h in hits.iter_mut() {
        let id = h.get("document_id").and_then(Value::as_str).and_then(|s| Uuid::parse_str(s).ok());
        if let (Some(src), Some(obj)) = (id.and_then(|i| by_id.get(&i)), h.as_object_mut()) {
            obj.insert("source".into(), src.clone());
        }
    }
    Ok(())
}

#[derive(Debug, Deserialize)]
pub(crate) struct ListQuery {
    q: Option<String>,
    include_closed: Option<bool>,
    limit: Option<i64>,
    offset: Option<i64>,
}

fn like_pattern(q: &str) -> String {
    let escaped = q.trim().replace('\\', "\\\\").replace('%', "\\%").replace('_', "\\_");
    format!("%{escaped}%")
}

pub(crate) async fn list_documents_handler(
    State(state): State<Arc<AppState>>,
    AuthPrincipal(principal): AuthPrincipal,
    Path(id): Path<String>,
    Query(q): Query<ListQuery>,
) -> Result<Json<Value>, ApiError> {
    let limit = q.limit.unwrap_or(50).clamp(1, 500);
    let offset = q.offset.unwrap_or(0).max(0);
    let include_closed = q.include_closed.unwrap_or(false);
    let pattern = q.q.as_deref().map(str::trim).filter(|s| !s.is_empty()).map(like_pattern);

    let app_pool = state.require_app_pool().map_err(ApiError::from)?;
    let mut tx = app_pool.begin_rls_tx(&principal).await.map_err(ApiError::from)?;
    let bucket = resolve_bucket_rls(&mut tx, principal.org_id, &id).await.map_err(ApiError::from)?;
    enforce_api_key_arn_scope(&principal, &bucket.arn).map_err(ApiError::from)?;

    const FILTER: &str = "d.org_id = $1 AND d.bucket_id = $2
        AND ($3::text IS NULL OR d.title ILIKE $3 ESCAPE '\\' OR coalesce(d.metadata->>'watch_rel_path', d.metadata->>'external_key', '') ILIKE $3 ESCAPE '\\')
        AND ($4 OR d.valid_to IS NULL OR d.valid_to > now())";
    let total: i64 = sqlx::query_scalar(&format!("SELECT count(*) FROM corpus.documents d WHERE {FILTER}"))
        .bind(principal.org_id)
        .bind(bucket.id)
        .bind(&pattern)
        .bind(include_closed)
        .fetch_one(&mut **tx.tx_mut())
        .await
        .map_err(db_err("count documents"))?;
    let rows = sqlx::query(&format!(
        "SELECT d.id, d.bucket_id, d.title, d.mime_type, d.source_uri, d.revision, d.status, d.valid_from, d.valid_to,
                d.sensitivity, d.access_scope, d.min_auth_level, d.tags,
                d.created_at, d.updated_at, d.metadata->>'external_key' AS key,
                d.metadata->>'watch_host_path' AS host, d.metadata->>'watch_rel_path' AS rel,
                (SELECT count(*) FROM corpus.chunks c WHERE c.document_id = d.id) AS chunks
         FROM corpus.documents d WHERE {FILTER}
         ORDER BY d.created_at DESC
         LIMIT $5 OFFSET $6"
    ))
    .bind(principal.org_id)
    .bind(bucket.id)
    .bind(&pattern)
    .bind(include_closed)
    .bind(limit)
    .bind(offset)
    .fetch_all(&mut **tx.tx_mut())
    .await
    .map_err(db_err("list documents"))?;
    tx.commit().await.map_err(ApiError::from)?;

    let mut docs = Vec::with_capacity(rows.len());
    for r in rows {
        let get = |c: &'static str| r.try_get::<Option<String>, _>(c).map_err(db_err(c));
        let mime = get("mime_type")?.unwrap_or_default();
        let valid_to: Option<DateTime<Utc>> = r.try_get("valid_to").map_err(db_err("valid_to"))?;
        let in_force = valid_to.is_none_or(|t| t > Utc::now());
        let doc_id = r.try_get::<Uuid, _>("id").map_err(db_err("id"))?;
        docs.push(json!({
            "id": doc_id,
            "document_id": doc_id,
            "bucket_id": r.try_get::<Uuid, _>("bucket_id").map_err(db_err("bucket_id"))?,
            "title": get("title")?,
            "source_uri": get("source_uri")?,
            "mime_type": mime,
            "sensitivity": get("sensitivity")?,
            "access_scope": get("access_scope")?,
            "min_auth_level": r.try_get::<i16, _>("min_auth_level").map_err(db_err("min_auth_level"))?,
            "tags": r.try_get::<Vec<String>, _>("tags").map_err(db_err("tags"))?,
            "source": source_json(&mime, get("source_uri")?.as_deref(), get("key")?.as_deref(), get("host")?.as_deref(), get("rel")?.as_deref()),
            "revision": r.try_get::<i32, _>("revision").map_err(db_err("revision"))?,
            "status": get("status")?,
            "in_force": in_force,
            "valid_from": r.try_get::<Option<DateTime<Utc>>, _>("valid_from").map_err(db_err("valid_from"))?,
            "valid_to": valid_to,
            "chunks": r.try_get::<i64, _>("chunks").map_err(db_err("chunks"))?,
            "created_at": r.try_get::<DateTime<Utc>, _>("created_at").map_err(db_err("created_at"))?,
            "updated_at": r.try_get::<DateTime<Utc>, _>("updated_at").map_err(db_err("updated_at"))?,
        }));
    }
    Ok(Json(json!({
        "bucket": bucket.slug,
        "bucket_id": bucket.id,
        "total": total,
        "offset": offset,
        "limit": limit,
        "documents": docs,
    })))
}

#[derive(Debug, Deserialize)]
struct SheetQuery {
    sheet: Option<String>,
    offset: Option<usize>,
    limit: Option<usize>,
}

fn md_table(header: &[String], rows: &[Vec<String>]) -> String {
    let width = rows.iter().map(Vec::len).chain([header.len()]).max().unwrap_or(0);
    let line = |cells: &[String]| {
        let mut c: Vec<String> = cells.iter().map(|x| x.replace('|', "\\|").replace('\n', " ")).collect();
        c.resize(width, String::new());
        format!("| {} |", c.join(" | "))
    };
    let mut out = Vec::with_capacity(rows.len() + 2);
    if !header.is_empty() {
        out.push(line(header));
        out.push(format!("|{}|", vec![" --- "; width].join("|")));
    }
    out.extend(rows.iter().map(|r| line(r)));
    out.join("\n")
}

async fn read_sheet_handler(
    State(state): State<Arc<AppState>>,
    AuthPrincipal(principal): AuthPrincipal,
    Path(id): Path<Uuid>,
    Query(q): Query<SheetQuery>,
) -> Result<Json<Value>, ApiError> {
    let app_pool = state.require_app_pool().map_err(ApiError::from)?;
    let mut tx = app_pool.begin_rls_tx(&principal).await.map_err(ApiError::from)?;
    let row = sqlx::query(
        "SELECT d.title, d.mime_type, d.content_sha256, d.metadata->>'external_key' AS key,
                d.metadata->>'source' AS src, b.arn
         FROM corpus.documents d JOIN corpus.buckets b ON b.id = d.bucket_id
         WHERE d.org_id = $1 AND d.id = $2",
    )
    .bind(principal.org_id)
    .bind(id)
    .fetch_optional(&mut **tx.tx_mut())
    .await
    .map_err(db_err("look up document"))?
    .ok_or_else(|| ApiError::not_found(format!("document {id} not found")))?;
    tx.commit().await.map_err(ApiError::from)?;
    let arn: String = row.try_get("arn").map_err(db_err("arn"))?;
    enforce_api_key_arn_scope(&principal, &arn).map_err(ApiError::from)?;

    let get = |c: &'static str| row.try_get::<Option<String>, _>(c).map_err(db_err(c));
    let title = get("title")?.unwrap_or_default();
    let format = spreadsheet_format(&get("mime_type")?.unwrap_or_default())
        .ok_or_else(|| ApiError::bad_request("not_a_spreadsheet", "this document is not an Excel or CSV file; read it with rag_get_page"))?;
    let key = get("key")?.unwrap_or_default();
    let path = (get("src")?.as_deref() == Some("watch"))
        .then(|| watch::watched_file(&key))
        .flatten()
        .ok_or_else(|| {
            ApiError::bad_request(
                "original_not_kept",
                "the server keeps no copy of this file: only spreadsheets from a watched documents folder can be read row by row",
            )
        })?;
    let bytes = tokio::fs::read(&path)
        .await
        .map_err(|e| ApiError::not_found(format!("the original file {} cannot be read: {e}", path.display())))?;
    let changed = openrag_parse::sha256_hex_bytes(&bytes) != get("content_sha256")?.unwrap_or_default();

    let offset = q.offset.unwrap_or(0);
    let limit = q.limit.unwrap_or(50).clamp(1, 500);
    let name = path.file_name().and_then(|n| n.to_str()).map(ToString::to_string);
    let sheet = q.sheet.clone();
    let slice = tokio::task::spawn_blocking(move || {
        openrag_parse::read_sheet(&bytes, format, name.as_deref(), sheet.as_deref(), offset, limit)
    })
    .await
    .map_err(|e| ApiError::from(OpenRagError::Internal(format!("read sheet: {e}"))))?
    .map_err(ApiError::from)?;

    let next = (slice.offset + slice.rows.len() < slice.total_rows).then(|| slice.offset + slice.rows.len());
    Ok(Json(json!({
        "document_id": id,
        "title": title,
        "path": watch::host_path_of(&key).unwrap_or(key),
        "sheets": slice.sheets,
        "sheet": slice.sheet,
        "columns": slice.header,
        "offset": slice.offset,
        "total_rows": slice.total_rows,
        "next_offset": next,
        "rows": slice.rows,
        "text": md_table(&slice.header, &slice.rows),
        "changed_since_indexed": changed,
    })))
}

async fn watch_status_handler(AuthPrincipal(principal): AuthPrincipal) -> Result<Json<Value>, ApiError> {
    require_admin(&principal)?;
    Ok(Json(match watch::status() {
        Some(r) => {
            let mut v = serde_json::to_value(&r).map_err(|e| ApiError::from(OpenRagError::Internal(e.to_string())))?;
            v["watching"] = json!(true);
            v
        }
        None => json!({
            "watching": false,
            "message": "no documents folder is watched: set OPENRAG_DOCS_DIRS (compose watches OPENRAG_DOCS_DIR at /docs)",
        }),
    }))
}

async fn watch_scan_handler(AuthPrincipal(principal): AuthPrincipal) -> Result<(StatusCode, Json<Value>), ApiError> {
    require_admin(&principal)?;
    if watch::request_scan() {
        Ok((StatusCode::ACCEPTED, Json(json!({ "scan": "requested" }))))
    } else {
        Err(ApiError::bad_request("not_watching", "no documents folder is watched"))
    }
}
