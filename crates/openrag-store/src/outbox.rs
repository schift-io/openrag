use std::collections::{BTreeMap, HashMap};
use std::fmt;
use std::str::FromStr;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use openrag_core::OpenRagError;
use openrag_db::{DbPool, RlsTransaction};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sqlx::Row;
use uuid::Uuid;

use crate::types::VectorTable;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OutboxOp {
    UpsertEmbed,
    DeleteVector,
    RecompileWiki,
    SyncExternalIndex,
    RedactReindex,
}

impl OutboxOp {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::UpsertEmbed => "upsert_embed",
            Self::DeleteVector => "delete_vector",
            Self::RecompileWiki => "recompile_wiki",
            Self::SyncExternalIndex => "sync_external_index",
            Self::RedactReindex => "redact_reindex",
        }
    }
}

impl fmt::Display for OutboxOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for OutboxOp {
    type Err = OpenRagError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim() {
            "upsert_embed" => Ok(Self::UpsertEmbed),
            "delete_vector" => Ok(Self::DeleteVector),
            "recompile_wiki" => Ok(Self::RecompileWiki),
            "sync_external_index" => Ok(Self::SyncExternalIndex),
            "redact_reindex" => Ok(Self::RedactReindex),
            other => Err(OpenRagError::Validation(format!(
                "invalid outbox op '{other}'"
            ))),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OutboxStatus {
    Pending,
    Processing,
    Done,
    Quarantined,
}

impl FromStr for OutboxStatus {
    type Err = OpenRagError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim() {
            "pending" => Ok(Self::Pending),
            "processing" => Ok(Self::Processing),
            "done" => Ok(Self::Done),
            "quarantined" => Ok(Self::Quarantined),
            other => Err(OpenRagError::Database(format!(
                "invalid outbox status '{other}'"
            ))),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OutboxRow {
    pub id: i64,
    pub org_id: Uuid,
    pub target_table: VectorTable,
    pub target_id: Uuid,
    pub op: OutboxOp,
    pub content_sha256: Option<String>,
    pub priority: i16,
    pub payload: Value,
    pub status: OutboxStatus,
    pub attempts: i32,
    pub max_attempts: i32,
    pub last_error: Option<String>,
    pub idempotency_key: Option<String>,
    pub available_at: DateTime<Utc>,
    pub processed_at: Option<DateTime<Utc>>,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EnqueueOutboxItem {
    pub org_id: Uuid,
    pub target_table: VectorTable,
    pub target_id: Uuid,
    pub op: OutboxOp,
    pub content_sha256: Option<String>,
    pub priority: i16,
    pub payload: Value,
    pub idempotency_key: Option<String>,
    pub max_attempts: Option<i32>,
}

impl EnqueueOutboxItem {
    pub fn validate(&self) -> Result<(), OpenRagError> {
        if !(1..=10).contains(&self.priority) {
            return Err(OpenRagError::Validation(format!(
                "outbox priority must be within 1..=10, got {}",
                self.priority
            )));
        }
        if let Some(ma) = self.max_attempts {
            if ma <= 0 {
                return Err(OpenRagError::Validation(format!(
                    "outbox max_attempts must be >= 1, got {ma}"
                )));
            }
        }
        Ok(())
    }
}

pub struct OutboxEnqueuer;

impl OutboxEnqueuer {
    /// Enqueues an outbox job inside the caller's `RlsTransaction`.
    /// When `coalesce_idempotency` is true and an active `idempotency_key` already exists,
    /// returns `Ok(None)` instead of `OpenRagError::Conflict`.
    pub async fn enqueue_rls(
        tx: &mut RlsTransaction<'_>,
        item: &EnqueueOutboxItem,
        coalesce_idempotency: bool,
    ) -> Result<Option<i64>, OpenRagError> {
        item.validate()?;
        let max_attempts = item.max_attempts.unwrap_or(3);

        let res = sqlx::query_scalar::<_, i64>(
            "INSERT INTO indexing.outbox
             (org_id, target_table, target_id, op, content_sha256, priority, payload, max_attempts, idempotency_key)
             VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
             RETURNING id",
        )
        .bind(item.org_id)
        .bind(item.target_table.as_str())
        .bind(item.target_id)
        .bind(item.op.as_str())
        .bind(&item.content_sha256)
        .bind(item.priority)
        .bind(&item.payload)
        .bind(max_attempts)
        .bind(&item.idempotency_key)
        .fetch_one(&mut **tx.tx_mut())
        .await;

        match res {
            Ok(id) => Ok(Some(id)),
            Err(sqlx::Error::Database(db_err)) if db_err.code().as_deref() == Some("23505") => {
                if coalesce_idempotency {
                    Ok(None)
                } else {
                    Err(OpenRagError::Conflict(format!(
                        "duplicate active outbox idempotency_key: {}",
                        item.idempotency_key.as_deref().unwrap_or("")
                    )))
                }
            }
            Err(sqlx::Error::Database(db_err)) if db_err.code().as_deref() == Some("42501") => {
                Err(OpenRagError::Forbidden(
                    "RLS denied outbox enqueue for target resource".into(),
                ))
            }
            Err(e) => Err(OpenRagError::Database(format!(
                "failed to enqueue outbox item: {e}"
            ))),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OutboxExecOutcome {
    Processed { backend_name: String },
    SkippedNoop { backend_name: String },
    TransientError { detail: String },
    NonRetryableError { status_code: u16, detail: String },
}

#[async_trait]
pub trait OutboxExecutor: Send + Sync {
    async fn execute(&self, item: &OutboxRow) -> OutboxExecOutcome;
}

#[derive(Debug, Default, Clone)]
pub struct DefaultOutboxExecutor {
    pub backend_name: String,
}

impl DefaultOutboxExecutor {
    pub fn new(backend_name: impl Into<String>) -> Self {
        Self {
            backend_name: backend_name.into(),
        }
    }
}

#[async_trait]
impl OutboxExecutor for DefaultOutboxExecutor {
    async fn execute(&self, item: &OutboxRow) -> OutboxExecOutcome {
        if let Some(code) = item
            .payload
            .get("simulate_error_code")
            .and_then(|v| v.as_u64())
        {
            let status_code = code as u16;
            if status_code == 403 || status_code == 422 || (400..500).contains(&status_code) {
                return OutboxExecOutcome::NonRetryableError {
                    status_code,
                    detail: format!("simulated non-retryable HTTP {status_code}"),
                };
            }
            return OutboxExecOutcome::TransientError {
                detail: format!("simulated HTTP {status_code}"),
            };
        }
        if item
            .payload
            .get("simulate_transient_error")
            .and_then(|v| v.as_bool())
            == Some(true)
        {
            return OutboxExecOutcome::TransientError {
                detail: "simulated transient indexing failure".into(),
            };
        }
        if matches!(item.op, OutboxOp::UpsertEmbed | OutboxOp::SyncExternalIndex) {
            if let (Some(sha), Some(prev)) = (
                item.content_sha256.as_deref(),
                item.payload.get("prev_sha256").and_then(|v| v.as_str()),
            ) {
                if !sha.is_empty() && sha == prev {
                    return OutboxExecOutcome::SkippedNoop {
                        backend_name: self.backend_name.clone(),
                    };
                }
            }
        }
        OutboxExecOutcome::Processed {
            backend_name: self.backend_name.clone(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WatermarkRecord {
    pub org_id: Uuid,
    pub target_table: String,
    pub backend_name: String,
    pub last_outbox_id: i64,
    pub indexed_count: i64,
    pub last_synced_at: DateTime<Utc>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QuarantineRecord {
    pub id: Uuid,
    pub outbox_id: Option<i64>,
    pub org_id: Uuid,
    pub target_table: String,
    pub target_id: Uuid,
    pub op: String,
    pub error_class: String,
    pub error_detail: String,
    pub payload: Value,
    pub resolved_at: Option<DateTime<Utc>>,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OutboxStatusSummary {
    pub pending: i64,
    pub processing: i64,
    pub done: i64,
    pub quarantined: i64,
    pub watermarks: Vec<WatermarkRecord>,
    pub unresolved_quarantine: Vec<QuarantineRecord>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct DrainReport {
    pub claimed: usize,
    pub processed: usize,
    pub skipped_noop: usize,
    pub retried: usize,
    pub quarantined: usize,
    pub claimed_ids: Vec<i64>,
    pub errors: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeltaPlan {
    pub upsert_ids: Vec<Uuid>,
    pub delete_ids: Vec<Uuid>,
    pub skipped_ids: Vec<Uuid>,
}

#[derive(Debug, Clone)]
pub struct OutboxWorker {
    /// Zero backoff by default so immediate `openrag index drain --limit 1` calls re-claim promptly.
    pub retry_backoff_secs: i64,
    pub default_backend_name: String,
}

impl Default for OutboxWorker {
    fn default() -> Self {
        Self {
            retry_backoff_secs: 0,
            default_backend_name: "pg_pinecone".into(),
        }
    }
}

impl OutboxWorker {
    pub fn check_document_delta(
        prev_hashes: &HashMap<Uuid, String>,
        new_hashes: &HashMap<Uuid, String>,
    ) -> DeltaPlan {
        let mut upsert_ids = Vec::new();
        let mut delete_ids = Vec::new();
        let mut skipped_ids = Vec::new();

        for (id, new_hash) in new_hashes {
            match prev_hashes.get(id) {
                Some(old_hash) if old_hash == new_hash => skipped_ids.push(*id),
                _ => upsert_ids.push(*id),
            }
        }
        for old_id in prev_hashes.keys() {
            if !new_hashes.contains_key(old_id) {
                delete_ids.push(*old_id);
            }
        }
        upsert_ids.sort();
        delete_ids.sort();
        skipped_ids.sort();
        DeltaPlan {
            upsert_ids,
            delete_ids,
            skipped_ids,
        }
    }

    pub async fn drain(
        &self,
        worker_pool: &DbPool,
        limit: usize,
        executor: &dyn OutboxExecutor,
    ) -> Result<DrainReport, OpenRagError> {
        self.drain_for_org(worker_pool, None, limit, executor).await
    }

    pub async fn drain_for_org(
        &self,
        worker_pool: &DbPool,
        org_id: Option<Uuid>,
        limit: usize,
        executor: &dyn OutboxExecutor,
    ) -> Result<DrainReport, OpenRagError> {
        let mut report = DrainReport::default();
        if limit == 0 {
            return Ok(report);
        }

        let mut claim_tx = worker_pool.inner().begin().await.map_err(|e| {
            OpenRagError::Database(format!("failed to begin outbox claim tx: {e}"))
        })?;
        let rows = sqlx::query(
            "WITH next_batch AS (
                 SELECT id
                 FROM indexing.outbox
                 WHERE status = 'pending'
                   AND available_at <= now()
                   AND ($2::uuid IS NULL OR org_id = $2)
                 ORDER BY priority ASC, available_at ASC, id ASC
                 LIMIT $1
                 FOR UPDATE SKIP LOCKED
             )
             UPDATE indexing.outbox AS o
             SET status = 'processing'
             FROM next_batch
             WHERE o.id = next_batch.id
             RETURNING o.id, o.org_id, o.target_table, o.target_id, o.op, o.content_sha256,
                       o.priority, o.payload, o.status, o.attempts, o.max_attempts,
                       o.last_error, o.idempotency_key, o.available_at, o.processed_at, o.created_at",
        )
        .bind(limit as i64)
        .bind(org_id)
        .fetch_all(&mut *claim_tx)
        .await
        .map_err(|e| OpenRagError::Database(format!("failed to claim outbox rows: {e}")))?;

        let mut claimed_items = Vec::with_capacity(rows.len());
        for row in &rows {
            claimed_items.push(row_to_outbox(row)?);
        }
        claim_tx.commit().await.map_err(|e| {
            OpenRagError::Database(format!("failed to commit outbox claim tx: {e}"))
        })?;

        // Re-order in claimed priority order because UPDATE ... FROM RETURNING order isn't guaranteed
        claimed_items.sort_by(|a, b| {
            a.priority
                .cmp(&b.priority)
                .then_with(|| a.available_at.cmp(&b.available_at))
                .then_with(|| a.id.cmp(&b.id))
        });

        report.claimed = claimed_items.len();
        for item in claimed_items {
            report.claimed_ids.push(item.id);
            let outcome = if self.is_db_sha256_noop(worker_pool, &item).await? {
                OutboxExecOutcome::SkippedNoop {
                    backend_name: self.default_backend_name.clone(),
                }
            } else {
                executor.execute(&item).await
            };

            match outcome {
                OutboxExecOutcome::Processed { backend_name } => {
                    self.mark_done_and_watermark(worker_pool, &item, &backend_name, true)
                        .await?;
                    report.processed += 1;
                }
                OutboxExecOutcome::SkippedNoop { backend_name } => {
                    self.mark_done_and_watermark(worker_pool, &item, &backend_name, false)
                        .await?;
                    report.skipped_noop += 1;
                }
                OutboxExecOutcome::TransientError { detail } => {
                    let quarantined = self
                        .handle_failure(worker_pool, &item, "transient_error", &detail, false)
                        .await?;
                    report.errors.push(detail);
                    if quarantined {
                        report.quarantined += 1;
                    } else {
                        report.retried += 1;
                    }
                }
                OutboxExecOutcome::NonRetryableError {
                    status_code,
                    detail,
                } => {
                    let err_class = format!("http_{status_code}");
                    let _ = self
                        .handle_failure(worker_pool, &item, &err_class, &detail, true)
                        .await?;
                    report.errors.push(detail);
                    report.quarantined += 1;
                }
            }
        }

        Ok(report)
    }

    async fn is_db_sha256_noop(
        &self,
        worker_pool: &DbPool,
        item: &OutboxRow,
    ) -> Result<bool, OpenRagError> {
        if !matches!(item.op, OutboxOp::UpsertEmbed | OutboxOp::SyncExternalIndex) {
            return Ok(false);
        }
        if item
            .payload
            .get("force_reindex")
            .and_then(|v| v.as_bool())
            == Some(true)
        {
            return Ok(false);
        }
        if item.payload.get("simulate_error_code").is_some()
            || item
                .payload
                .get("simulate_transient_error")
                .and_then(|v| v.as_bool())
                == Some(true)
        {
            return Ok(false);
        }
        let Some(sha) = item.content_sha256.as_deref() else {
            return Ok(false);
        };
        if sha.trim().is_empty() {
            return Ok(false);
        }

        let already_synced: bool = sqlx::query_scalar(
            "SELECT EXISTS (
                 SELECT 1
                 FROM indexing.outbox AS prev
                 WHERE prev.org_id = $1
                   AND prev.target_table = $2
                   AND prev.target_id = $3
                   AND prev.id < $4
                   AND prev.status = 'done'
                   AND prev.op IN ('upsert_embed', 'sync_external_index')
                   AND prev.content_sha256 = $5
                   AND NOT EXISTS (
                       SELECT 1
                       FROM indexing.outbox AS newer
                       WHERE newer.org_id = $1
                         AND newer.target_table = $2
                         AND newer.target_id = $3
                         AND newer.id > prev.id
                         AND newer.id < $4
                         AND newer.status = 'done'
                         AND (newer.op IN ('delete_vector', 'redact_reindex')
                              OR (newer.content_sha256 IS DISTINCT FROM $5))
                   )
             )",
        )
        .bind(item.org_id)
        .bind(item.target_table.as_str())
        .bind(item.target_id)
        .bind(item.id)
        .bind(sha)
        .fetch_one(worker_pool.inner())
        .await
        .map_err(|e| OpenRagError::Database(format!("SHA-256 No-Op check failed: {e}")))?;

        Ok(already_synced)
    }

    async fn mark_done_and_watermark(
        &self,
        worker_pool: &DbPool,
        item: &OutboxRow,
        backend_name: &str,
        increment_indexed_count: bool,
    ) -> Result<(), OpenRagError> {
        let mut tx = worker_pool.inner().begin().await.map_err(|e| {
            OpenRagError::Database(format!("failed to begin done/watermark tx: {e}"))
        })?;

        sqlx::query(
            "UPDATE indexing.outbox
             SET status = 'done', processed_at = now(), last_error = NULL
             WHERE id = $1",
        )
        .bind(item.id)
        .execute(&mut *tx)
        .await
        .map_err(|e| OpenRagError::Database(format!("failed to mark outbox done: {e}")))?;

        let delta: i64 = if increment_indexed_count { 1 } else { 0 };
        sqlx::query(
            "INSERT INTO indexing.watermarks
             (org_id, target_table, backend_name, last_outbox_id, indexed_count, last_synced_at)
             VALUES ($1, $2, $3, $4, $5, now())
             ON CONFLICT (org_id, target_table, backend_name)
             DO UPDATE SET
                 last_outbox_id = GREATEST(indexing.watermarks.last_outbox_id, EXCLUDED.last_outbox_id),
                 indexed_count  = indexing.watermarks.indexed_count + EXCLUDED.indexed_count,
                 last_synced_at = now()",
        )
        .bind(item.org_id)
        .bind(item.target_table.as_str())
        .bind(backend_name)
        .bind(item.id)
        .bind(delta)
        .execute(&mut *tx)
        .await
        .map_err(|e| OpenRagError::Database(format!("failed to update watermark: {e}")))?;

        tx.commit().await.map_err(|e| {
            OpenRagError::Database(format!("failed to commit done/watermark tx: {e}"))
        })?;
        Ok(())
    }

    async fn handle_failure(
        &self,
        worker_pool: &DbPool,
        item: &OutboxRow,
        error_class: &str,
        detail: &str,
        non_retryable: bool,
    ) -> Result<bool, OpenRagError> {
        let new_attempts = item.attempts + 1;
        let should_quarantine = non_retryable || new_attempts >= item.max_attempts;

        let mut tx = worker_pool.inner().begin().await.map_err(|e| {
            OpenRagError::Database(format!("failed to begin outbox failure tx: {e}"))
        })?;

        if should_quarantine {
            sqlx::query(
                "UPDATE indexing.outbox
                 SET status = 'quarantined', attempts = $1, last_error = $2
                 WHERE id = $3",
            )
            .bind(new_attempts)
            .bind(detail)
            .bind(item.id)
            .execute(&mut *tx)
            .await
            .map_err(|e| {
                OpenRagError::Database(format!("failed to quarantine outbox row: {e}"))
            })?;

            sqlx::query(
                "INSERT INTO indexing.quarantine
                 (outbox_id, org_id, target_table, target_id, op, error_class, error_detail, payload)
                 VALUES ($1, $2, $3, $4, $5, $6, $7, $8)",
            )
            .bind(item.id)
            .bind(item.org_id)
            .bind(item.target_table.as_str())
            .bind(item.target_id)
            .bind(item.op.as_str())
            .bind(error_class)
            .bind(detail)
            .bind(&item.payload)
            .execute(&mut *tx)
            .await
            .map_err(|e| {
                OpenRagError::Database(format!("failed to insert indexing.quarantine row: {e}"))
            })?;
        } else {
            sqlx::query(
                "UPDATE indexing.outbox
                 SET status = 'pending',
                     attempts = $1,
                     last_error = $2,
                     available_at = now() + ($3 * interval '1 second')
                 WHERE id = $4",
            )
            .bind(new_attempts)
            .bind(detail)
            .bind(self.retry_backoff_secs * i64::from(new_attempts))
            .bind(item.id)
            .execute(&mut *tx)
            .await
            .map_err(|e| OpenRagError::Database(format!("failed to reschedule outbox row: {e}")))?;
        }

        tx.commit().await.map_err(|e| {
            OpenRagError::Database(format!("failed to commit outbox failure tx: {e}"))
        })?;
        Ok(should_quarantine)
    }

    pub async fn status(
        &self,
        worker_pool: &DbPool,
        org_id: Option<Uuid>,
    ) -> Result<OutboxStatusSummary, OpenRagError> {
        let count_rows = sqlx::query(
            "SELECT status, count(*)::bigint AS cnt
             FROM indexing.outbox
             WHERE ($1::uuid IS NULL OR org_id = $1)
             GROUP BY status",
        )
        .bind(org_id)
        .fetch_all(worker_pool.inner())
        .await
        .map_err(|e| OpenRagError::Database(format!("failed to count outbox statuses: {e}")))?;

        let mut counts: BTreeMap<String, i64> = BTreeMap::new();
        for r in count_rows {
            let st: String = r
                .try_get("status")
                .map_err(|e| OpenRagError::Database(format!("invalid status row: {e}")))?;
            let cnt: i64 = r
                .try_get("cnt")
                .map_err(|e| OpenRagError::Database(format!("invalid cnt row: {e}")))?;
            counts.insert(st, cnt);
        }

        let wm_rows = sqlx::query(
            "SELECT org_id, target_table, backend_name, last_outbox_id, indexed_count, last_synced_at
             FROM indexing.watermarks
             WHERE ($1::uuid IS NULL OR org_id = $1)
             ORDER BY target_table ASC, backend_name ASC",
        )
        .bind(org_id)
        .fetch_all(worker_pool.inner())
        .await
        .map_err(|e| OpenRagError::Database(format!("failed to list watermarks: {e}")))?;

        let mut watermarks = Vec::with_capacity(wm_rows.len());
        for r in wm_rows {
            watermarks.push(WatermarkRecord {
                org_id: r
                    .try_get("org_id")
                    .map_err(|e| OpenRagError::Database(format!("invalid wm org_id: {e}")))?,
                target_table: r
                    .try_get("target_table")
                    .map_err(|e| OpenRagError::Database(format!("invalid wm target_table: {e}")))?,
                backend_name: r
                    .try_get("backend_name")
                    .map_err(|e| OpenRagError::Database(format!("invalid wm backend_name: {e}")))?,
                last_outbox_id: r
                    .try_get("last_outbox_id")
                    .map_err(|e| OpenRagError::Database(format!("invalid wm last_outbox_id: {e}")))?,
                indexed_count: r
                    .try_get("indexed_count")
                    .map_err(|e| OpenRagError::Database(format!("invalid wm indexed_count: {e}")))?,
                last_synced_at: r
                    .try_get("last_synced_at")
                    .map_err(|e| OpenRagError::Database(format!("invalid wm last_synced_at: {e}")))?,
            });
        }

        let q_rows = sqlx::query(
            "SELECT id, outbox_id, org_id, target_table, target_id, op, error_class, error_detail, payload, resolved_at, created_at
             FROM indexing.quarantine
             WHERE resolved_at IS NULL AND ($1::uuid IS NULL OR org_id = $1)
             ORDER BY created_at ASC",
        )
        .bind(org_id)
        .fetch_all(worker_pool.inner())
        .await
        .map_err(|e| OpenRagError::Database(format!("failed to list quarantine: {e}")))?;

        let mut unresolved_quarantine = Vec::with_capacity(q_rows.len());
        for r in q_rows {
            unresolved_quarantine.push(row_to_quarantine(&r)?);
        }

        Ok(OutboxStatusSummary {
            pending: *counts.get("pending").unwrap_or(&0),
            processing: *counts.get("processing").unwrap_or(&0),
            done: *counts.get("done").unwrap_or(&0),
            quarantined: *counts.get("quarantined").unwrap_or(&0),
            watermarks,
            unresolved_quarantine,
        })
    }

    pub async fn retry_quarantine(
        &self,
        worker_pool: &DbPool,
        quarantine_id: Option<Uuid>,
        retry_all: bool,
        org_id: Option<Uuid>,
    ) -> Result<usize, OpenRagError> {
        if quarantine_id.is_none() && !retry_all {
            return Err(OpenRagError::Validation(
                "must specify quarantine_id or retry_all=true".into(),
            ));
        }

        let mut tx = worker_pool.inner().begin().await.map_err(|e| {
            OpenRagError::Database(format!("failed to begin quarantine retry tx: {e}"))
        })?;

        let targets = if let Some(qid) = quarantine_id {
            let row = sqlx::query(
                "SELECT id, outbox_id, org_id, target_table, target_id, op, error_class, error_detail, payload, resolved_at, created_at
                 FROM indexing.quarantine
                 WHERE id = $1 AND ($2::uuid IS NULL OR org_id = $2)
                 FOR UPDATE",
            )
            .bind(qid)
            .bind(org_id)
            .fetch_optional(&mut *tx)
            .await
            .map_err(|e| OpenRagError::Database(format!("failed to load quarantine row: {e}")))?;

            let Some(r) = row else {
                return Err(OpenRagError::NotFound(format!(
                    "quarantine record {qid} not found"
                )));
            };
            let q = row_to_quarantine(&r)?;
            if q.resolved_at.is_some() {
                return Err(OpenRagError::Conflict(format!(
                    "quarantine record {qid} is already resolved"
                )));
            }
            vec![q]
        } else {
            let rows = sqlx::query(
                "SELECT id, outbox_id, org_id, target_table, target_id, op, error_class, error_detail, payload, resolved_at, created_at
                 FROM indexing.quarantine
                 WHERE resolved_at IS NULL AND ($1::uuid IS NULL OR org_id = $1)
                 ORDER BY created_at ASC
                 FOR UPDATE",
            )
            .bind(org_id)
            .fetch_all(&mut *tx)
            .await
            .map_err(|e| OpenRagError::Database(format!("failed to load quarantine rows: {e}")))?;
            rows.iter().map(row_to_quarantine).collect::<Result<Vec<_>, _>>()?
        };

        let mut retried = 0usize;
        for q in targets {
            if let Some(oid) = q.outbox_id {
                sqlx::query(
                    "UPDATE indexing.outbox
                     SET status = 'pending',
                         attempts = 0,
                         available_at = now(),
                         last_error = NULL,
                         payload = payload - 'simulate_transient_error' - 'simulate_error_code'
                     WHERE id = $1 AND status = 'quarantined'",
                )
                .bind(oid)
                .execute(&mut *tx)
                .await
                .map_err(|e| {
                    OpenRagError::Database(format!("failed to reset quarantined outbox row: {e}"))
                })?;
            } else {
                let new_oid: i64 = sqlx::query_scalar(
                    "INSERT INTO indexing.outbox
                     (org_id, target_table, target_id, op, priority, payload, status, attempts, max_attempts)
                     VALUES ($1, $2, $3, $4, 5, $5 - 'simulate_transient_error' - 'simulate_error_code', 'pending', 0, 3)
                     RETURNING id",
                )
                .bind(q.org_id)
                .bind(&q.target_table)
                .bind(q.target_id)
                .bind(&q.op)
                .bind(&q.payload)
                .fetch_one(&mut *tx)
                .await
                .map_err(|e| {
                    OpenRagError::Database(format!("failed to re-enqueue quarantined item: {e}"))
                })?;

                sqlx::query("UPDATE indexing.quarantine SET outbox_id = $1 WHERE id = $2")
                    .bind(new_oid)
                    .bind(q.id)
                    .execute(&mut *tx)
                    .await
                    .map_err(|e| {
                        OpenRagError::Database(format!("failed to link new outbox_id: {e}"))
                    })?;
            }

            sqlx::query(
                "UPDATE indexing.quarantine SET resolved_at = now() WHERE id = $1 AND resolved_at IS NULL",
            )
            .bind(q.id)
            .execute(&mut *tx)
            .await
            .map_err(|e| {
                OpenRagError::Database(format!("failed to resolve quarantine record: {e}"))
            })?;
            retried += 1;
        }

        tx.commit().await.map_err(|e| {
            OpenRagError::Database(format!("failed to commit quarantine retry tx: {e}"))
        })?;
        Ok(retried)
    }
}

fn row_to_outbox(row: &sqlx::postgres::PgRow) -> Result<OutboxRow, OpenRagError> {
    let table_str: String = row
        .try_get("target_table")
        .map_err(|e| OpenRagError::Database(format!("invalid target_table: {e}")))?;
    let op_str: String = row
        .try_get("op")
        .map_err(|e| OpenRagError::Database(format!("invalid op: {e}")))?;
    let status_str: String = row
        .try_get("status")
        .map_err(|e| OpenRagError::Database(format!("invalid status: {e}")))?;

    Ok(OutboxRow {
        id: row
            .try_get("id")
            .map_err(|e| OpenRagError::Database(format!("invalid id: {e}")))?,
        org_id: row
            .try_get("org_id")
            .map_err(|e| OpenRagError::Database(format!("invalid org_id: {e}")))?,
        target_table: VectorTable::from_str(&table_str)?,
        target_id: row
            .try_get("target_id")
            .map_err(|e| OpenRagError::Database(format!("invalid target_id: {e}")))?,
        op: OutboxOp::from_str(&op_str)?,
        content_sha256: row
            .try_get("content_sha256")
            .map_err(|e| OpenRagError::Database(format!("invalid content_sha256: {e}")))?,
        priority: row
            .try_get("priority")
            .map_err(|e| OpenRagError::Database(format!("invalid priority: {e}")))?,
        payload: row
            .try_get("payload")
            .map_err(|e| OpenRagError::Database(format!("invalid payload: {e}")))?,
        status: OutboxStatus::from_str(&status_str)?,
        attempts: row
            .try_get("attempts")
            .map_err(|e| OpenRagError::Database(format!("invalid attempts: {e}")))?,
        max_attempts: row
            .try_get("max_attempts")
            .map_err(|e| OpenRagError::Database(format!("invalid max_attempts: {e}")))?,
        last_error: row
            .try_get("last_error")
            .map_err(|e| OpenRagError::Database(format!("invalid last_error: {e}")))?,
        idempotency_key: row
            .try_get("idempotency_key")
            .map_err(|e| OpenRagError::Database(format!("invalid idempotency_key: {e}")))?,
        available_at: row
            .try_get("available_at")
            .map_err(|e| OpenRagError::Database(format!("invalid available_at: {e}")))?,
        processed_at: row
            .try_get("processed_at")
            .map_err(|e| OpenRagError::Database(format!("invalid processed_at: {e}")))?,
        created_at: row
            .try_get("created_at")
            .map_err(|e| OpenRagError::Database(format!("invalid created_at: {e}")))?,
    })
}

fn row_to_quarantine(row: &sqlx::postgres::PgRow) -> Result<QuarantineRecord, OpenRagError> {
    Ok(QuarantineRecord {
        id: row
            .try_get("id")
            .map_err(|e| OpenRagError::Database(format!("invalid quarantine id: {e}")))?,
        outbox_id: row
            .try_get("outbox_id")
            .map_err(|e| OpenRagError::Database(format!("invalid quarantine outbox_id: {e}")))?,
        org_id: row
            .try_get("org_id")
            .map_err(|e| OpenRagError::Database(format!("invalid quarantine org_id: {e}")))?,
        target_table: row
            .try_get("target_table")
            .map_err(|e| OpenRagError::Database(format!("invalid quarantine target_table: {e}")))?,
        target_id: row
            .try_get("target_id")
            .map_err(|e| OpenRagError::Database(format!("invalid quarantine target_id: {e}")))?,
        op: row
            .try_get("op")
            .map_err(|e| OpenRagError::Database(format!("invalid quarantine op: {e}")))?,
        error_class: row
            .try_get("error_class")
            .map_err(|e| OpenRagError::Database(format!("invalid quarantine error_class: {e}")))?,
        error_detail: row
            .try_get("error_detail")
            .map_err(|e| OpenRagError::Database(format!("invalid quarantine error_detail: {e}")))?,
        payload: row
            .try_get("payload")
            .map_err(|e| OpenRagError::Database(format!("invalid quarantine payload: {e}")))?,
        resolved_at: row
            .try_get("resolved_at")
            .map_err(|e| OpenRagError::Database(format!("invalid quarantine resolved_at: {e}")))?,
        created_at: row
            .try_get("created_at")
            .map_err(|e| OpenRagError::Database(format!("invalid quarantine created_at: {e}")))?,
    })
}
