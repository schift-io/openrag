use openrag_core::{OpenRagError, PrincipalContext};
use sqlx::{PgPool, Postgres, Transaction};
use uuid::Uuid;

pub fn encode_uuid_array(ids: &[Uuid]) -> String {
    if ids.is_empty() {
        return "{}".to_string();
    }
    let mut out = String::with_capacity(ids.len() * 37 + 2);
    out.push('{');
    for (idx, id) in ids.iter().enumerate() {
        if idx > 0 {
            out.push(',');
        }
        out.push_str(&id.to_string());
    }
    out.push('}');
    out
}

pub fn encode_text_array(items: &[String]) -> String {
    if items.is_empty() {
        return "{}".to_string();
    }
    let mut out = String::from("{");
    for (idx, item) in items.iter().enumerate() {
        if idx > 0 {
            out.push(',');
        }
        out.push('"');
        for ch in item.chars() {
            if ch == '\\' || ch == '"' {
                out.push('\\');
            }
            out.push(ch);
        }
        out.push('"');
    }
    out.push('}');
    out
}

pub struct RlsTransaction<'a> {
    tx: Transaction<'a, Postgres>,
}

impl<'a> AsMut<Transaction<'a, Postgres>> for RlsTransaction<'a> {
    fn as_mut(&mut self) -> &mut Transaction<'a, Postgres> {
        &mut self.tx
    }
}

impl<'a> RlsTransaction<'a> {
    pub fn tx_mut(&mut self) -> &mut Transaction<'a, Postgres> {
        &mut self.tx
    }

    pub async fn commit(self) -> Result<(), OpenRagError> {
        self.tx
            .commit()
            .await
            .map_err(|e| OpenRagError::Database(format!("failed to commit RLS transaction: {e}")))
    }

    pub async fn rollback(self) -> Result<(), OpenRagError> {
        self.tx.rollback().await.map_err(|e| {
            OpenRagError::Database(format!("failed to rollback RLS transaction: {e}"))
        })
    }
}

/// Begins a PostgreSQL transaction and binds all session GUCs transaction-locally
/// via parameterized `SELECT set_config($1, $2, true)` (never `SET LOCAL = $1`).
/// `app.bypass_rls` is unconditionally forced to `'false'`.
pub async fn begin_rls_tx<'a>(
    pool: &'a PgPool,
    ctx: &PrincipalContext,
) -> Result<RlsTransaction<'a>, OpenRagError> {
    let mut tx = pool.begin().await.map_err(|e| {
        OpenRagError::Database(format!("failed to begin RLS transaction: {e}"))
    })?;

    let org_id = ctx.org_id.to_string();
    let principal_id = ctx.principal_id.to_string();
    let auth_level = ctx.auth_level.as_u8().to_string();
    let dept_ids = encode_uuid_array(&ctx.dept_ids);
    let dept_admin_ids = encode_uuid_array(&ctx.dept_admin_ids);
    let group_ids = encode_uuid_array(&ctx.group_ids);
    let scoped_group_ids = encode_uuid_array(&ctx.scoped_group_ids);
    let api_key_id = ctx
        .api_key_id
        .map_or_else(String::new, |id| id.to_string());
    let scoped_arns = encode_text_array(&ctx.scoped_arns);

    sqlx::query(
        "SELECT
            set_config('app.org_id', $1, true),
            set_config('app.principal_id', $2, true),
            set_config('app.auth_level', $3, true),
            set_config('app.dept_ids', $4, true),
            set_config('app.dept_admin_ids', $5, true),
            set_config('app.group_ids', $6, true),
            set_config('app.scoped_group_ids', $7, true),
            set_config('app.api_key_id', $8, true),
            set_config('app.scoped_arns', $9, true),
            set_config('app.bypass_rls', 'false', true)",
    )
    .bind(&org_id)
    .bind(&principal_id)
    .bind(&auth_level)
    .bind(&dept_ids)
    .bind(&dept_admin_ids)
    .bind(&group_ids)
    .bind(&scoped_group_ids)
    .bind(&api_key_id)
    .bind(&scoped_arns)
    .execute(&mut *tx)
    .await
    .map_err(|e| OpenRagError::Database(format!("failed to bind RLS session context: {e}")))?;

    Ok(RlsTransaction { tx })
}
