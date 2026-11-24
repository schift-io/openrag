//! Team members and their API keys, beside the create routes in `api_studio`.
//!
//! - `GET  /api/studio/iam/principals` — members of the caller's organization (org admins).
//! - `GET  /api/studio/iam/api-keys` — keys: every key in the organization for org admins, the
//!   caller's own keys otherwise. Never the key itself, only its prefix.
//! - `POST /api/studio/iam/api-keys/{id}/revoke` — stops a key at once (org admins, or the key's
//!   owner). Revoked keys stay listed with `revoked_at`.

use std::sync::Arc;

use axum::extract::{Path, Query, State};
use axum::routing::post;
use axum::{Json, Router};
use chrono::{DateTime, Utc};
use openrag_core::OpenRagError;
use serde::Deserialize;
use serde_json::{json, Value};
use sqlx::Row;
use uuid::Uuid;

use crate::auth::AuthPrincipal;
use crate::error::ApiError;
use crate::state::AppState;

pub fn member_routes() -> Router<Arc<AppState>> {
    Router::new().route("/api/studio/iam/api-keys/{id}/revoke", post(revoke_key_handler))
}

fn db_err(what: &'static str) -> impl Fn(sqlx::Error) -> ApiError {
    move |e| ApiError::from(OpenRagError::Database(format!("{what}: {e}")))
}

pub(crate) async fn list_principals_handler(
    State(state): State<Arc<AppState>>,
    AuthPrincipal(principal): AuthPrincipal,
) -> Result<Json<Value>, ApiError> {
    if !principal.is_org_admin {
        return Err(ApiError::forbidden("admin_required", "only organization admins can list members"));
    }
    let worker = state.require_worker_pool().map_err(ApiError::from)?;
    let rows = sqlx::query(
        "SELECT p.id, p.email, p.display_name, p.auth_level, p.is_org_admin, p.status, p.created_at,
                coalesce(array_agg(DISTINCT d.arn_path) FILTER (WHERE d.arn_path IS NOT NULL), '{}') AS depts,
                coalesce(array_agg(DISTINCT d.id) FILTER (WHERE d.id IS NOT NULL), '{}') AS dept_ids,
                (SELECT count(*) FROM iam.api_keys k
                  WHERE k.org_id = p.org_id AND k.principal_id = p.id AND k.revoked_at IS NULL
                    AND (k.expires_at IS NULL OR k.expires_at > now())) AS active_keys
         FROM iam.principals p
         LEFT JOIN iam.dept_memberships m ON m.org_id = p.org_id AND m.principal_id = p.id
         LEFT JOIN iam.departments d ON d.org_id = m.org_id AND d.id = m.dept_id
         WHERE p.org_id = $1
         GROUP BY p.id
         ORDER BY p.created_at",
    )
    .bind(principal.org_id)
    .fetch_all(worker.inner())
    .await
    .map_err(db_err("list principals"))?;
    let mut out = Vec::with_capacity(rows.len());
    for r in rows {
        out.push(json!({
            "id": r.try_get::<Uuid, _>("id").map_err(db_err("id"))?,
            "email": r.try_get::<String, _>("email").map_err(db_err("email"))?,
            "display_name": r.try_get::<String, _>("display_name").map_err(db_err("display_name"))?,
            "auth_level": r.try_get::<i16, _>("auth_level").map_err(db_err("auth_level"))?,
            "is_org_admin": r.try_get::<bool, _>("is_org_admin").map_err(db_err("is_org_admin"))?,
            "status": r.try_get::<String, _>("status").map_err(db_err("status"))?,
            "departments": r.try_get::<Vec<String>, _>("depts").map_err(db_err("depts"))?,
            "dept_ids": r.try_get::<Vec<Uuid>, _>("dept_ids").map_err(db_err("dept_ids"))?,
            "active_keys": r.try_get::<i64, _>("active_keys").map_err(db_err("active_keys"))?,
            "created_at": r.try_get::<DateTime<Utc>, _>("created_at").map_err(db_err("created_at"))?,
        }));
    }
    Ok(Json(json!({ "principals": out })))
}

#[derive(Debug, Deserialize)]
pub(crate) struct KeyQuery {
    principal_id: Option<Uuid>,
    include_revoked: Option<bool>,
}

pub(crate) async fn list_keys_handler(
    State(state): State<Arc<AppState>>,
    AuthPrincipal(principal): AuthPrincipal,
    Query(q): Query<KeyQuery>,
) -> Result<Json<Value>, ApiError> {
    // Members other than admins see their own keys only.
    let owner = if principal.is_org_admin { q.principal_id } else { Some(principal.principal_id) };
    let worker = state.require_worker_pool().map_err(ApiError::from)?;
    let rows = sqlx::query(
        "SELECT k.id, k.name, k.key_prefix, k.principal_id, p.email, p.display_name, k.max_auth_level,
                k.created_at, k.expires_at, k.revoked_at
         FROM iam.api_keys k JOIN iam.principals p ON p.org_id = k.org_id AND p.id = k.principal_id
         WHERE k.org_id = $1 AND ($2::uuid IS NULL OR k.principal_id = $2) AND ($3 OR k.revoked_at IS NULL)
         ORDER BY k.created_at",
    )
    .bind(principal.org_id)
    .bind(owner)
    .bind(q.include_revoked.unwrap_or(false))
    .fetch_all(worker.inner())
    .await
    .map_err(db_err("list api keys"))?;
    let mut out = Vec::with_capacity(rows.len());
    for r in rows {
        let expires: Option<DateTime<Utc>> = r.try_get("expires_at").map_err(db_err("expires_at"))?;
        let revoked: Option<DateTime<Utc>> = r.try_get("revoked_at").map_err(db_err("revoked_at"))?;
        let state = match (revoked, expires) {
            (Some(_), _) => "revoked",
            (None, Some(t)) if t <= Utc::now() => "expired",
            _ => "active",
        };
        out.push(json!({
            "id": r.try_get::<Uuid, _>("id").map_err(db_err("id"))?,
            "name": r.try_get::<String, _>("name").map_err(db_err("name"))?,
            "key_prefix": r.try_get::<String, _>("key_prefix").map_err(db_err("key_prefix"))?,
            "principal_id": r.try_get::<Uuid, _>("principal_id").map_err(db_err("principal_id"))?,
            "email": r.try_get::<String, _>("email").map_err(db_err("email"))?,
            "display_name": r.try_get::<String, _>("display_name").map_err(db_err("display_name"))?,
            "max_auth_level": r.try_get::<Option<i16>, _>("max_auth_level").map_err(db_err("max_auth_level"))?,
            "state": state,
            "created_at": r.try_get::<DateTime<Utc>, _>("created_at").map_err(db_err("created_at"))?,
            "expires_at": expires,
            "revoked_at": revoked,
        }));
    }
    Ok(Json(json!({ "api_keys": out })))
}

async fn revoke_key_handler(
    State(state): State<Arc<AppState>>,
    AuthPrincipal(principal): AuthPrincipal,
    Path(id): Path<Uuid>,
) -> Result<Json<Value>, ApiError> {
    let worker = state.require_worker_pool().map_err(ApiError::from)?;
    let row = sqlx::query("SELECT principal_id, revoked_at FROM iam.api_keys WHERE org_id = $1 AND id = $2")
        .bind(principal.org_id)
        .bind(id)
        .fetch_optional(worker.inner())
        .await
        .map_err(db_err("look up api key"))?
        .ok_or_else(|| ApiError::not_found(format!("API key {id} not found")))?;
    let owner: Uuid = row.try_get("principal_id").map_err(db_err("principal_id"))?;
    if !principal.is_org_admin && owner != principal.principal_id {
        return Err(ApiError::forbidden("forbidden", "only organization admins can revoke someone else's key"));
    }
    let already: Option<DateTime<Utc>> = row.try_get("revoked_at").map_err(db_err("revoked_at"))?;
    let revoked_at: DateTime<Utc> = match already {
        Some(t) => t,
        None => sqlx::query_scalar(
            "UPDATE iam.api_keys SET revoked_at = now() WHERE org_id = $1 AND id = $2 RETURNING revoked_at",
        )
        .bind(principal.org_id)
        .bind(id)
        .fetch_one(worker.inner())
        .await
        .map_err(db_err("revoke api key"))?,
    };
    Ok(Json(json!({ "id": id, "revoked": true, "revoked_at": revoked_at, "already_revoked": already.is_some() })))
}
