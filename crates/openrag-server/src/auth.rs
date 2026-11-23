use std::sync::Arc;

use axum::extract::{FromRequestParts, State};
use axum::http::header::AUTHORIZATION;
use axum::http::request::Parts;
use axum::http::StatusCode;
use axum::Json;
use chrono::{DateTime, Duration, Utc};
use openrag_core::{arn_pattern_matches, AuthLevel, OpenRagError, PrincipalContext};
use openrag_db::{sha256_hex, DbPool};
use ring::rand::{SecureRandom, SystemRandom};
use serde::{Deserialize, Serialize};
use sqlx::Row;
use uuid::Uuid;

use crate::error::ApiError;
use crate::llm_router::ensure_default_providers;
use crate::state::AppState;

#[derive(Debug, Clone)]
pub struct AuthPrincipal(pub PrincipalContext);

/// Pre-authenticated caller attached as a request *extension* by in-process callers
/// (MCP tool dispatch, `openrag` CLI). Extensions cannot be set from the network, so
/// this never widens the remote attack surface; remote requests still need `Bearer`.
#[derive(Debug, Clone)]
pub(crate) struct TrustedPrincipal(pub(crate) PrincipalContext);

impl FromRequestParts<Arc<AppState>> for AuthPrincipal {
    type Rejection = ApiError;

    async fn from_request_parts(
        parts: &mut Parts,
        state: &Arc<AppState>,
    ) -> Result<Self, Self::Rejection> {
        if let Some(TrustedPrincipal(ctx)) = parts.extensions.get::<TrustedPrincipal>() {
            let mut ctx = ctx.clone();
            ctx.bypass_rls = false;
            return Ok(Self(ctx));
        }
        let auth_header = parts
            .headers
            .get(AUTHORIZATION)
            .and_then(|v| v.to_str().ok())
            .ok_or_else(|| ApiError::unauthorized("missing Authorization header"))?;

        let token = extract_bearer_token(auth_header)
            .ok_or_else(|| ApiError::unauthorized("invalid Authorization scheme; expected Bearer"))?;

        let worker_pool = state
            .worker_pool
            .as_ref()
            .ok_or_else(|| ApiError::unauthorized("authentication database not configured"))?;

        let ctx = authenticate_bearer_token(worker_pool, token)
            .await
            .map_err(ApiError::from)?;
        Ok(Self(ctx))
    }
}

pub fn extract_bearer_token(header_val: &str) -> Option<&str> {
    let trimmed = header_val.trim();
    if trimmed.len() < 7 {
        return None;
    }
    let (scheme, rest) = trimmed.split_at(7);
    if !scheme.eq_ignore_ascii_case("bearer ") {
        return None;
    }
    let token = rest.trim();
    if token.is_empty() {
        return None;
    }
    Some(token)
}

pub async fn authenticate_bearer_token(
    worker_pool: &DbPool,
    token: &str,
) -> Result<PrincipalContext, OpenRagError> {
    let key_hash = sha256_hex(token);
    let key_row = sqlx::query(
        "SELECT id, org_id, principal_id, scoped_arns, scoped_group_ids, max_auth_level,
                expires_at, revoked_at
         FROM iam.api_keys
         WHERE key_hash = $1",
    )
    .bind(&key_hash)
    .fetch_optional(worker_pool.inner())
    .await
    .map_err(|e| OpenRagError::Database(format!("failed querying iam.api_keys: {e}")))?
    .ok_or_else(|| OpenRagError::Unauthorized("invalid API key".into()))?;

    let revoked_at: Option<DateTime<Utc>> = key_row
        .try_get("revoked_at")
        .map_err(|e| OpenRagError::Database(format!("api_keys.revoked_at: {e}")))?;
    if revoked_at.is_some() {
        return Err(OpenRagError::Unauthorized("API key has been revoked".into()));
    }

    let expires_at: Option<DateTime<Utc>> = key_row
        .try_get("expires_at")
        .map_err(|e| OpenRagError::Database(format!("api_keys.expires_at: {e}")))?;
    if let Some(exp) = expires_at {
        if exp <= Utc::now() {
            return Err(OpenRagError::Unauthorized("API key has expired".into()));
        }
    }

    let key_id: Uuid = key_row
        .try_get("id")
        .map_err(|e| OpenRagError::Database(format!("api_keys.id: {e}")))?;
    let org_id: Uuid = key_row
        .try_get("org_id")
        .map_err(|e| OpenRagError::Database(format!("api_keys.org_id: {e}")))?;
    let principal_id: Uuid = key_row
        .try_get("principal_id")
        .map_err(|e| OpenRagError::Database(format!("api_keys.principal_id: {e}")))?;
    let scoped_arns: Vec<String> = key_row
        .try_get("scoped_arns")
        .map_err(|e| OpenRagError::Database(format!("api_keys.scoped_arns: {e}")))?;
    let scoped_group_ids: Vec<Uuid> = key_row
        .try_get("scoped_group_ids")
        .map_err(|e| OpenRagError::Database(format!("api_keys.scoped_group_ids: {e}")))?;
    let max_auth_level: Option<i16> = key_row
        .try_get("max_auth_level")
        .map_err(|e| OpenRagError::Database(format!("api_keys.max_auth_level: {e}")))?;

    let mut ctx =
        load_principal_context(worker_pool, org_id, principal_id, max_auth_level, "API key principal not found")
            .await?;
    ctx.api_key_id = Some(key_id);
    ctx.scoped_arns = scoped_arns;
    ctx.scoped_group_ids = scoped_group_ids;
    ctx.bypass_rls = false;
    Ok(ctx)
}

/// Loads a principal's live IAM context (level, departments, groups, admin flags).
/// `max_auth_level` caps the effective level (API-key ceiling). Inactive or missing
/// principals fail closed with `Unauthorized`.
pub async fn load_principal_context(
    worker_pool: &DbPool,
    org_id: Uuid,
    principal_id: Uuid,
    max_auth_level: Option<i16>,
    not_found_msg: &str,
) -> Result<PrincipalContext, OpenRagError> {
    let principal_row = sqlx::query(
        "SELECT auth_level, can_unmask_pii, is_org_admin, status
         FROM iam.principals
         WHERE org_id = $1 AND id = $2",
    )
    .bind(org_id)
    .bind(principal_id)
    .fetch_optional(worker_pool.inner())
    .await
    .map_err(|e| OpenRagError::Database(format!("failed querying iam.principals: {e}")))?
    .ok_or_else(|| OpenRagError::Unauthorized(not_found_msg.into()))?;

    let status: String = principal_row
        .try_get("status")
        .map_err(|e| OpenRagError::Database(format!("principals.status: {e}")))?;
    if status != "active" {
        return Err(OpenRagError::Unauthorized("principal is not active".into()));
    }

    let principal_level: i16 = principal_row
        .try_get("auth_level")
        .map_err(|e| OpenRagError::Database(format!("principals.auth_level: {e}")))?;
    let can_unmask_pii: bool = principal_row
        .try_get("can_unmask_pii")
        .map_err(|e| OpenRagError::Database(format!("principals.can_unmask_pii: {e}")))?;
    let is_org_admin: bool = principal_row
        .try_get("is_org_admin")
        .map_err(|e| OpenRagError::Database(format!("principals.is_org_admin: {e}")))?;

    let effective_level_i16 = match max_auth_level {
        Some(cap) => principal_level.min(cap),
        None => principal_level,
    };
    let auth_level = AuthLevel::try_from(effective_level_i16)?;

    let dept_rows = sqlx::query(
        "SELECT dept_id, dept_role
         FROM iam.dept_memberships
         WHERE org_id = $1 AND principal_id = $2",
    )
    .bind(org_id)
    .bind(principal_id)
    .fetch_all(worker_pool.inner())
    .await
    .map_err(|e| OpenRagError::Database(format!("failed querying iam.dept_memberships: {e}")))?;

    let mut dept_ids = Vec::with_capacity(dept_rows.len());
    let mut dept_admin_ids = Vec::new();
    for dr in dept_rows {
        let dept_id: Uuid = dr
            .try_get("dept_id")
            .map_err(|e| OpenRagError::Database(format!("dept_memberships.dept_id: {e}")))?;
        let role: String = dr
            .try_get("dept_role")
            .map_err(|e| OpenRagError::Database(format!("dept_memberships.dept_role: {e}")))?;
        dept_ids.push(dept_id);
        if role == "dept_admin" {
            dept_admin_ids.push(dept_id);
        }
    }

    let group_ids: Vec<Uuid> = sqlx::query_scalar(
        "SELECT group_id
         FROM iam.group_members
         WHERE org_id = $1 AND principal_id = $2",
    )
    .bind(org_id)
    .bind(principal_id)
    .fetch_all(worker_pool.inner())
    .await
    .map_err(|e| OpenRagError::Database(format!("failed querying iam.group_members: {e}")))?;

    let mut ctx = PrincipalContext::new(org_id, principal_id, auth_level);
    ctx.dept_ids = dept_ids;
    ctx.dept_admin_ids = dept_admin_ids;
    ctx.group_ids = group_ids;
    ctx.can_unmask_pii = can_unmask_pii;
    ctx.is_org_admin = is_org_admin;
    ctx.bypass_rls = false;
    Ok(ctx)
}

pub fn enforce_api_key_arn_scope(
    principal: &PrincipalContext,
    arn: &str,
) -> Result<(), OpenRagError> {
    if principal.scoped_arns.is_empty() {
        return Ok(());
    }
    if principal
        .scoped_arns
        .iter()
        .any(|pat| arn_pattern_matches(pat, arn))
    {
        Ok(())
    } else {
        Err(OpenRagError::Forbidden(format!(
            "API key scoped_arns does not permit access to {arn}"
        )))
    }
}

#[derive(Debug, Deserialize)]
pub struct CliCodeExchangeRequest {
    pub code: String,
    pub org_id: Option<Uuid>,
    pub principal_id: Option<Uuid>,
    pub name: Option<String>,
    pub max_auth_level: Option<i16>,
    #[serde(default)]
    pub scoped_arns: Vec<String>,
    #[serde(default)]
    pub scoped_group_ids: Vec<Uuid>,
    pub ttl_seconds: Option<i64>,
}

#[derive(Debug, Serialize)]
pub struct CliCodeExchangeResponse {
    pub api_key: String,
    pub token: String,
    pub key_id: Uuid,
    pub key_prefix: String,
    pub org_id: Uuid,
    pub principal_id: Uuid,
    pub max_auth_level: Option<i16>,
    pub scoped_arns: Vec<String>,
    pub scoped_group_ids: Vec<Uuid>,
    pub expires_at: Option<DateTime<Utc>>,
}

/// Ceilings carried by a one-time CLI code. At exchange time they can only be narrowed.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct CliCodeCaps {
    pub max_auth_level: Option<i16>,
    #[serde(default)]
    pub scoped_arns: Vec<String>,
    #[serde(default)]
    pub scoped_group_ids: Vec<Uuid>,
}

#[derive(Debug, Clone, Serialize)]
pub struct IssuedCliCode {
    pub code: String,
    pub code_id: Uuid,
    pub org_id: Uuid,
    pub principal_id: Uuid,
    pub issued_by: Option<Uuid>,
    pub expires_at: DateTime<Utc>,
    pub max_auth_level: Option<i16>,
    pub scoped_arns: Vec<String>,
    pub scoped_group_ids: Vec<Uuid>,
    pub exchange_endpoint: &'static str,
}

pub const CLI_CODE_DEFAULT_TTL_SECS: i64 = 600;
pub const CLI_CODE_MAX_TTL_SECS: i64 = 86_400;

fn random_token(prefix: &str) -> Result<String, OpenRagError> {
    let mut rand_bytes = [0u8; 24];
    SystemRandom::new()
        .fill(&mut rand_bytes)
        .map_err(|_| OpenRagError::Internal("CSPRNG failure".into()))?;
    Ok(format!("{prefix}{}", hex::encode(rand_bytes)))
}

/// Issues a single-use, short-lived code for `principal_id`. `issued_by` is the org admin
/// who issued it, or `None` for the local database operator (`openrag auth code`).
pub async fn issue_cli_code(
    worker_pool: &DbPool,
    org_id: Uuid,
    principal_id: Uuid,
    issued_by: Option<Uuid>,
    caps: CliCodeCaps,
    ttl_seconds: Option<i64>,
) -> Result<IssuedCliCode, OpenRagError> {
    let ttl = ttl_seconds.unwrap_or(CLI_CODE_DEFAULT_TTL_SECS);
    if !(1..=CLI_CODE_MAX_TTL_SECS).contains(&ttl) {
        return Err(OpenRagError::Validation(format!(
            "ttl_seconds must be between 1 and {CLI_CODE_MAX_TTL_SECS}"
        )));
    }
    if let Some(level) = caps.max_auth_level {
        if !(1..=10).contains(&level) {
            return Err(OpenRagError::Validation(format!(
                "max_auth_level must be between 1 and 10, got {level}"
            )));
        }
    }
    let status: Option<String> =
        sqlx::query_scalar("SELECT status FROM iam.principals WHERE org_id = $1 AND id = $2")
            .bind(org_id)
            .bind(principal_id)
            .fetch_optional(worker_pool.inner())
            .await
            .map_err(|e| OpenRagError::Database(format!("principal lookup: {e}")))?;
    if status.as_deref() != Some("active") {
        return Err(OpenRagError::NotFound(
            "principal does not exist or is inactive in this org".into(),
        ));
    }

    let code = random_token("oragc_")?;
    let expires_at = Utc::now() + Duration::seconds(ttl);
    let code_id: Uuid = sqlx::query_scalar(
        "INSERT INTO iam.cli_auth_codes (
            org_id, principal_id, issued_by, code_hash,
            max_auth_level, scoped_arns, scoped_group_ids, expires_at
         ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
         RETURNING id",
    )
    .bind(org_id)
    .bind(principal_id)
    .bind(issued_by)
    .bind(sha256_hex(&code))
    .bind(caps.max_auth_level)
    .bind(&caps.scoped_arns)
    .bind(&caps.scoped_group_ids)
    .bind(expires_at)
    .fetch_one(worker_pool.inner())
    .await
    .map_err(|e| OpenRagError::Database(format!("insert iam.cli_auth_codes: {e}")))?;

    Ok(IssuedCliCode {
        code,
        code_id,
        org_id,
        principal_id,
        issued_by,
        expires_at,
        max_auth_level: caps.max_auth_level,
        scoped_arns: caps.scoped_arns,
        scoped_group_ids: caps.scoped_group_ids,
        exchange_endpoint: "/v1/auth/cli/code-exchange",
    })
}

/// Narrows a code ceiling with the exchange request. `None` from the request keeps the code's.
fn narrow_list<T: PartialEq + Clone>(
    code: &[T],
    requested: &[T],
    what: &str,
) -> Result<Vec<T>, ApiError> {
    if requested.is_empty() {
        return Ok(code.to_vec());
    }
    if !code.is_empty() && requested.iter().any(|r| !code.contains(r)) {
        return Err(ApiError::forbidden(
            "scope_exceeds_code",
            format!("requested {what} exceed the authorization code's scope"),
        ));
    }
    Ok(requested.to_vec())
}

/// Exchanges a one-time code (issued by an org admin or the local operator) for an API key.
/// Unknown, used and expired codes all fail with the same `401`.
pub async fn cli_code_exchange(
    State(state): State<Arc<AppState>>,
    Json(req): Json<CliCodeExchangeRequest>,
) -> Result<(StatusCode, Json<CliCodeExchangeResponse>), ApiError> {
    let code = req.code.trim();
    if code.is_empty() {
        return Err(ApiError::bad_request(
            "invalid_code",
            "authorization code must not be empty",
        ));
    }
    if let Some(level) = req.max_auth_level {
        if !(1..=10).contains(&level) {
            return Err(ApiError::bad_request(
                "invalid_auth_level",
                format!("max_auth_level must be between 1 and 10, got {level}"),
            ));
        }
    }
    let expires_at = match req.ttl_seconds {
        Some(ttl) if ttl <= 0 => {
            return Err(ApiError::bad_request(
                "invalid_ttl",
                "ttl_seconds must be positive",
            ));
        }
        Some(ttl) => Some(Utc::now() + Duration::seconds(ttl)),
        None => None,
    };

    let worker_pool = state
        .worker_pool
        .as_ref()
        .ok_or_else(|| ApiError::unauthorized("authentication database not configured"))?;
    let invalid = || ApiError::unauthorized("invalid or expired authorization code");

    let mut tx = worker_pool
        .inner()
        .begin()
        .await
        .map_err(|e| ApiError::from(OpenRagError::Database(format!("begin code exchange: {e}"))))?;
    let row = sqlx::query(
        "SELECT id, org_id, principal_id, max_auth_level, scoped_arns, scoped_group_ids,
                expires_at, used_at
         FROM iam.cli_auth_codes
         WHERE code_hash = $1
         FOR UPDATE",
    )
    .bind(sha256_hex(code))
    .fetch_optional(&mut *tx)
    .await
    .map_err(|e| ApiError::from(OpenRagError::Database(format!("code lookup: {e}"))))?
    .ok_or_else(invalid)?;

    let col = |e: sqlx::Error| ApiError::from(OpenRagError::Database(format!("cli_auth_codes: {e}")));
    let code_id: Uuid = row.try_get("id").map_err(col)?;
    let org_id: Uuid = row.try_get("org_id").map_err(col)?;
    let principal_id: Uuid = row.try_get("principal_id").map_err(col)?;
    let code_level: Option<i16> = row.try_get("max_auth_level").map_err(col)?;
    let code_arns: Vec<String> = row.try_get("scoped_arns").map_err(col)?;
    let code_groups: Vec<Uuid> = row.try_get("scoped_group_ids").map_err(col)?;
    let code_expires: DateTime<Utc> = row.try_get("expires_at").map_err(col)?;
    let used_at: Option<DateTime<Utc>> = row.try_get("used_at").map_err(col)?;
    if used_at.is_some() || code_expires <= Utc::now() {
        return Err(invalid());
    }
    if req.org_id.is_some_and(|o| o != org_id) || req.principal_id.is_some_and(|p| p != principal_id) {
        return Err(invalid());
    }

    let principal_status: Option<String> =
        sqlx::query_scalar("SELECT status FROM iam.principals WHERE org_id = $1 AND id = $2")
            .bind(org_id)
            .bind(principal_id)
            .fetch_optional(&mut *tx)
            .await
            .map_err(|e| ApiError::from(OpenRagError::Database(format!("principal lookup: {e}"))))?;
    if principal_status.as_deref() != Some("active") {
        return Err(ApiError::unauthorized("principal does not exist or is inactive"));
    }

    let max_auth_level = match (code_level, req.max_auth_level) {
        (Some(c), Some(r)) => Some(c.min(r)),
        (c, r) => c.or(r),
    };
    let scoped_arns = narrow_list(&code_arns, &req.scoped_arns, "scoped_arns")?;
    let scoped_group_ids = narrow_list(&code_groups, &req.scoped_group_ids, "scoped_group_ids")?;

    let raw_key = random_token("orag_").map_err(ApiError::from)?;
    let key_prefix = raw_key.chars().take(12).collect::<String>();
    let key_hash = sha256_hex(&raw_key);
    let key_name = req
        .name
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(ToString::to_string)
        .unwrap_or_else(|| format!("cli-{}", &key_prefix));

    let key_id: Uuid = sqlx::query_scalar(
        "INSERT INTO iam.api_keys (
            org_id, principal_id, name, key_prefix, key_hash,
            scoped_arns, scoped_group_ids, max_auth_level, expires_at
         ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
         RETURNING id",
    )
    .bind(org_id)
    .bind(principal_id)
    .bind(&key_name)
    .bind(&key_prefix)
    .bind(&key_hash)
    .bind(&scoped_arns)
    .bind(&scoped_group_ids)
    .bind(max_auth_level)
    .bind(expires_at)
    .fetch_one(&mut *tx)
    .await
    .map_err(|e| ApiError::from(OpenRagError::Database(format!("insert iam.api_keys: {e}"))))?;

    sqlx::query("UPDATE iam.cli_auth_codes SET used_at = now(), used_key_id = $2 WHERE id = $1")
        .bind(code_id)
        .bind(key_id)
        .execute(&mut *tx)
        .await
        .map_err(|e| ApiError::from(OpenRagError::Database(format!("consume code: {e}"))))?;
    tx.commit()
        .await
        .map_err(|e| ApiError::from(OpenRagError::Database(format!("commit code exchange: {e}"))))?;

    ensure_default_providers(worker_pool, org_id)
        .await
        .map_err(ApiError::from)?;

    Ok((
        StatusCode::CREATED,
        Json(CliCodeExchangeResponse {
            api_key: raw_key.clone(),
            token: raw_key,
            key_id,
            key_prefix,
            org_id,
            principal_id,
            max_auth_level,
            scoped_arns,
            scoped_group_ids,
            expires_at,
        }),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_bearer_token_handles_case_and_rejects_malformed() {
        assert_eq!(extract_bearer_token("Bearer orag_abc123"), Some("orag_abc123"));
        assert_eq!(extract_bearer_token("bearer   tok_xyz  "), Some("tok_xyz"));
        assert_eq!(extract_bearer_token("Basic abc"), None);
        assert_eq!(extract_bearer_token("Bearer "), None);
        assert_eq!(extract_bearer_token(""), None);
    }

    #[test]
    fn enforce_api_key_arn_scope_allows_empty_and_matching_patterns() {
        let org_id = Uuid::new_v4();
        let pid = Uuid::new_v4();
        let mut ctx = PrincipalContext::new(org_id, pid, AuthLevel::MAX);
        assert!(enforce_api_key_arn_scope(&ctx, "arn:openrag:acme:dept2:dept:bucket:L3/1").is_ok());

        ctx.scoped_arns = vec!["arn:openrag:acme:dept1:*".to_string()];
        assert!(enforce_api_key_arn_scope(&ctx, "arn:openrag:acme:dept1:dept:bucket:L3/1").is_ok());
        assert!(enforce_api_key_arn_scope(&ctx, "arn:openrag:acme:dept2:dept:bucket:L3/1").is_err());
    }
}
