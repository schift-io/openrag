use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use chrono::Utc;
use openrag_core::{AuthLevel, OpenRagError, OptimizationStrategy, PrincipalContext};
use openrag_db::DbPool;
use sqlx::{postgres::PgRow, Row};
use uuid::Uuid;

use crate::types::{
    BenchItemResultRecord, BenchRunRecord, BenchTrack, CreateGoldenItemInput,
    CreateGoldenSetInput, GoldenAxis, GoldenItem, GoldenSet, OptimizationStatus,
    OptimizationTrialRecord, PipelineConfigRecord, PromoteFeedbackInput, PromoteFeedbackResult,
    RegisterPipelineInput, ReviewStatus, SearchFeedback, SubmitFeedbackInput,
};

#[derive(Debug, Clone)]
pub struct NewBenchRun {
    pub org_id: Uuid,
    pub golden_set_id: Uuid,
    pub pipeline_config_id: Option<Uuid>,
    pub trial_id: Option<Uuid>,
    pub track: BenchTrack,
    pub metrics: serde_json::Value,
    pub pii_leak_count: i32,
    pub arn_violation_count: i32,
}

#[derive(Debug, Clone)]
pub struct NewBenchItemResult {
    pub golden_item_id: Uuid,
    pub retrieved_doc_ids: Vec<Uuid>,
    pub latency_ms: i32,
    pub score: f32,
    pub pii_leaked: bool,
    pub arn_violated: bool,
    pub details: serde_json::Value,
}

#[derive(Debug, Clone)]
pub struct NewOptimizationTrial {
    pub org_id: Uuid,
    pub golden_set_id: Uuid,
    pub search_space_yaml: String,
    pub strategy: OptimizationStrategy,
}

#[derive(Debug, Clone)]
pub struct CompleteOptimizationTrial {
    pub best_config_yaml: Option<String>,
    pub best_score: Option<f32>,
    pub security_gate_pass: bool,
    pub status: OptimizationStatus,
}

#[async_trait]
pub trait EvalStore: Send + Sync {
    async fn create_golden_set(
        &self,
        org_id: Uuid,
        input: &CreateGoldenSetInput,
    ) -> Result<GoldenSet, OpenRagError>;

    async fn get_golden_set(&self, org_id: Uuid, id: Uuid) -> Result<GoldenSet, OpenRagError>;

    async fn get_golden_set_by_name(
        &self,
        org_id: Uuid,
        name: &str,
    ) -> Result<GoldenSet, OpenRagError>;

    async fn list_golden_sets(&self, org_id: Uuid) -> Result<Vec<GoldenSet>, OpenRagError>;

    async fn add_golden_item(
        &self,
        org_id: Uuid,
        golden_set_id: Uuid,
        input: &CreateGoldenItemInput,
    ) -> Result<GoldenItem, OpenRagError>;

    async fn get_golden_item(&self, org_id: Uuid, item_id: Uuid)
        -> Result<GoldenItem, OpenRagError>;

    async fn list_golden_items(
        &self,
        org_id: Uuid,
        golden_set_id: Uuid,
        include_rejected: bool,
    ) -> Result<Vec<GoldenItem>, OpenRagError>;

    async fn update_review_status(
        &self,
        org_id: Uuid,
        item_id: Uuid,
        status: ReviewStatus,
    ) -> Result<GoldenItem, OpenRagError>;

    async fn submit_feedback_rls(
        &self,
        ctx: &PrincipalContext,
        input: &SubmitFeedbackInput,
    ) -> Result<SearchFeedback, OpenRagError>;

    async fn list_feedback_rls(
        &self,
        ctx: &PrincipalContext,
    ) -> Result<Vec<SearchFeedback>, OpenRagError>;

    async fn promote_feedback(
        &self,
        org_id: Uuid,
        actor: &PrincipalContext,
        feedback_id: Uuid,
        input: &PromoteFeedbackInput,
    ) -> Result<PromoteFeedbackResult, OpenRagError>;

    async fn resolve_principal(
        &self,
        org_id: Uuid,
        principal_id: Uuid,
    ) -> Result<Option<PrincipalContext>, OpenRagError>;

    async fn insert_bench_run(
        &self,
        run: &NewBenchRun,
        items: &[NewBenchItemResult],
    ) -> Result<BenchRunRecord, OpenRagError>;

    async fn get_bench_run(
        &self,
        org_id: Uuid,
        bench_run_id: Uuid,
    ) -> Result<BenchRunRecord, OpenRagError>;

    async fn list_bench_runs(
        &self,
        org_id: Uuid,
        golden_set_id: Option<Uuid>,
    ) -> Result<Vec<BenchRunRecord>, OpenRagError>;

    async fn list_bench_item_results(
        &self,
        org_id: Uuid,
        bench_run_id: Uuid,
    ) -> Result<Vec<BenchItemResultRecord>, OpenRagError>;

    async fn insert_optimization_trial(
        &self,
        trial: &NewOptimizationTrial,
    ) -> Result<OptimizationTrialRecord, OpenRagError>;

    async fn complete_optimization_trial(
        &self,
        org_id: Uuid,
        trial_id: Uuid,
        update: &CompleteOptimizationTrial,
    ) -> Result<OptimizationTrialRecord, OpenRagError>;

    async fn list_optimization_trials(
        &self,
        org_id: Uuid,
        golden_set_id: Option<Uuid>,
    ) -> Result<Vec<OptimizationTrialRecord>, OpenRagError>;

    async fn register_pipeline_config(
        &self,
        org_id: Uuid,
        input: &RegisterPipelineInput,
    ) -> Result<PipelineConfigRecord, OpenRagError>;

    async fn get_pipeline_config(
        &self,
        org_id: Uuid,
        config_id: Uuid,
    ) -> Result<PipelineConfigRecord, OpenRagError>;

    async fn list_pipeline_configs(
        &self,
        org_id: Uuid,
    ) -> Result<Vec<PipelineConfigRecord>, OpenRagError>;

    async fn latest_bench_run_for_config(
        &self,
        org_id: Uuid,
        pipeline_config_id: Uuid,
    ) -> Result<Option<BenchRunRecord>, OpenRagError>;

    async fn latest_bench_runs_by_track_for_config(
        &self,
        org_id: Uuid,
        pipeline_config_id: Uuid,
    ) -> Result<Vec<BenchRunRecord>, OpenRagError>;

    async fn mark_pipeline_promoted(
        &self,
        org_id: Uuid,
        config_id: Uuid,
        bucket_ref: Option<&str>,
    ) -> Result<PipelineConfigRecord, OpenRagError>;
}

#[derive(Clone, Debug)]
pub struct PgEvalStore {
    app_pool: DbPool,
    worker_pool: DbPool,
}

impl PgEvalStore {
    pub fn new(app_pool: DbPool, worker_pool: DbPool) -> Self {
        Self {
            app_pool,
            worker_pool,
        }
    }

    pub fn app_pool(&self) -> &DbPool {
        &self.app_pool
    }

    pub fn worker_pool(&self) -> &DbPool {
        &self.worker_pool
    }
}

fn map_pg_err(context: &str, err: sqlx::Error) -> OpenRagError {
    if let Some(db_err) = err.as_database_error() {
        if db_err.code().as_deref() == Some("23505") {
            return OpenRagError::Conflict(format!("{context}: {}", db_err.message()));
        }
        if db_err.code().as_deref() == Some("23503") {
            return OpenRagError::NotFound(format!("{context}: {}", db_err.message()));
        }
    }
    OpenRagError::Database(format!("{context}: {err}"))
}

fn row_to_golden_set(row: &PgRow) -> Result<GoldenSet, OpenRagError> {
    let axis_str: String = row
        .try_get("axis")
        .map_err(|e| OpenRagError::Database(format!("golden_set.axis: {e}")))?;
    Ok(GoldenSet {
        id: row
            .try_get("id")
            .map_err(|e| OpenRagError::Database(format!("golden_set.id: {e}")))?,
        org_id: row
            .try_get("org_id")
            .map_err(|e| OpenRagError::Database(format!("golden_set.org_id: {e}")))?,
        name: row
            .try_get("name")
            .map_err(|e| OpenRagError::Database(format!("golden_set.name: {e}")))?,
        axis: axis_str.parse()?,
        description: row
            .try_get("description")
            .map_err(|e| OpenRagError::Database(format!("golden_set.description: {e}")))?,
        version: row
            .try_get("version")
            .map_err(|e| OpenRagError::Database(format!("golden_set.version: {e}")))?,
        created_at: row
            .try_get("created_at")
            .map_err(|e| OpenRagError::Database(format!("golden_set.created_at: {e}")))?,
    })
}

fn row_to_golden_item(row: &PgRow) -> Result<GoldenItem, OpenRagError> {
    let axis_str: String = row
        .try_get("axis")
        .map_err(|e| OpenRagError::Database(format!("golden_item.axis: {e}")))?;
    let status_str: String = row
        .try_get("review_status")
        .map_err(|e| OpenRagError::Database(format!("golden_item.review_status: {e}")))?;
    Ok(GoldenItem {
        id: row
            .try_get("id")
            .map_err(|e| OpenRagError::Database(format!("golden_item.id: {e}")))?,
        org_id: row
            .try_get("org_id")
            .map_err(|e| OpenRagError::Database(format!("golden_item.org_id: {e}")))?,
        golden_set_id: row
            .try_get("golden_set_id")
            .map_err(|e| OpenRagError::Database(format!("golden_item.golden_set_id: {e}")))?,
        axis: axis_str.parse()?,
        query_text: row
            .try_get("query_text")
            .map_err(|e| OpenRagError::Database(format!("golden_item.query_text: {e}")))?,
        test_principal_id: row
            .try_get("test_principal_id")
            .map_err(|e| OpenRagError::Database(format!("golden_item.test_principal_id: {e}")))?,
        expected_doc_ids: row
            .try_get("expected_doc_ids")
            .map_err(|e| OpenRagError::Database(format!("golden_item.expected_doc_ids: {e}")))?,
        forbidden_doc_ids: row
            .try_get("forbidden_doc_ids")
            .map_err(|e| OpenRagError::Database(format!("golden_item.forbidden_doc_ids: {e}")))?,
        expected_answer: row
            .try_get("expected_answer")
            .map_err(|e| OpenRagError::Database(format!("golden_item.expected_answer: {e}")))?,
        expected_pii_tokens: row
            .try_get("expected_pii_tokens")
            .map_err(|e| OpenRagError::Database(format!("golden_item.expected_pii_tokens: {e}")))?,
        expected_edges: row
            .try_get("expected_edges")
            .map_err(|e| OpenRagError::Database(format!("golden_item.expected_edges: {e}")))?,
        review_status: status_str.parse()?,
        created_at: row
            .try_get("created_at")
            .map_err(|e| OpenRagError::Database(format!("golden_item.created_at: {e}")))?,
    })
}

fn row_to_feedback(row: &PgRow) -> Result<SearchFeedback, OpenRagError> {
    let verdict_str: String = row
        .try_get("verdict")
        .map_err(|e| OpenRagError::Database(format!("search_feedback.verdict: {e}")))?;
    Ok(SearchFeedback {
        id: row
            .try_get("id")
            .map_err(|e| OpenRagError::Database(format!("search_feedback.id: {e}")))?,
        org_id: row
            .try_get("org_id")
            .map_err(|e| OpenRagError::Database(format!("search_feedback.org_id: {e}")))?,
        principal_id: row
            .try_get("principal_id")
            .map_err(|e| OpenRagError::Database(format!("search_feedback.principal_id: {e}")))?,
        query_text: row
            .try_get("query_text")
            .map_err(|e| OpenRagError::Database(format!("search_feedback.query_text: {e}")))?,
        response_text: row
            .try_get("response_text")
            .map_err(|e| OpenRagError::Database(format!("search_feedback.response_text: {e}")))?,
        cited_chunk_ids: row
            .try_get("cited_chunk_ids")
            .map_err(|e| OpenRagError::Database(format!("search_feedback.cited_chunk_ids: {e}")))?,
        verdict: verdict_str.parse()?,
        comment: row
            .try_get("comment")
            .map_err(|e| OpenRagError::Database(format!("search_feedback.comment: {e}")))?,
        promoted_golden_item_id: row.try_get("promoted_golden_item_id").map_err(|e| {
            OpenRagError::Database(format!("search_feedback.promoted_golden_item_id: {e}"))
        })?,
        created_at: row
            .try_get("created_at")
            .map_err(|e| OpenRagError::Database(format!("search_feedback.created_at: {e}")))?,
    })
}

fn row_to_bench_run(row: &PgRow) -> Result<BenchRunRecord, OpenRagError> {
    let track_str: String = row
        .try_get("track")
        .map_err(|e| OpenRagError::Database(format!("bench_run.track: {e}")))?;
    Ok(BenchRunRecord {
        id: row
            .try_get("id")
            .map_err(|e| OpenRagError::Database(format!("bench_run.id: {e}")))?,
        org_id: row
            .try_get("org_id")
            .map_err(|e| OpenRagError::Database(format!("bench_run.org_id: {e}")))?,
        golden_set_id: row
            .try_get("golden_set_id")
            .map_err(|e| OpenRagError::Database(format!("bench_run.golden_set_id: {e}")))?,
        pipeline_config_id: row
            .try_get("pipeline_config_id")
            .map_err(|e| OpenRagError::Database(format!("bench_run.pipeline_config_id: {e}")))?,
        trial_id: row
            .try_get("trial_id")
            .map_err(|e| OpenRagError::Database(format!("bench_run.trial_id: {e}")))?,
        track: track_str.parse()?,
        metrics: row
            .try_get("metrics")
            .map_err(|e| OpenRagError::Database(format!("bench_run.metrics: {e}")))?,
        pii_leak_count: row
            .try_get("pii_leak_count")
            .map_err(|e| OpenRagError::Database(format!("bench_run.pii_leak_count: {e}")))?,
        arn_violation_count: row
            .try_get("arn_violation_count")
            .map_err(|e| OpenRagError::Database(format!("bench_run.arn_violation_count: {e}")))?,
        security_gate_pass: row
            .try_get("security_gate_pass")
            .map_err(|e| OpenRagError::Database(format!("bench_run.security_gate_pass: {e}")))?,
        created_at: row
            .try_get("created_at")
            .map_err(|e| OpenRagError::Database(format!("bench_run.created_at: {e}")))?,
    })
}

fn row_to_bench_item_result(row: &PgRow) -> Result<BenchItemResultRecord, OpenRagError> {
    Ok(BenchItemResultRecord {
        id: row
            .try_get("id")
            .map_err(|e| OpenRagError::Database(format!("bench_item_result.id: {e}")))?,
        bench_run_id: row
            .try_get("bench_run_id")
            .map_err(|e| OpenRagError::Database(format!("bench_item_result.bench_run_id: {e}")))?,
        golden_item_id: row.try_get("golden_item_id").map_err(|e| {
            OpenRagError::Database(format!("bench_item_result.golden_item_id: {e}"))
        })?,
        retrieved_doc_ids: row.try_get("retrieved_doc_ids").map_err(|e| {
            OpenRagError::Database(format!("bench_item_result.retrieved_doc_ids: {e}"))
        })?,
        latency_ms: row
            .try_get("latency_ms")
            .map_err(|e| OpenRagError::Database(format!("bench_item_result.latency_ms: {e}")))?,
        score: row
            .try_get("score")
            .map_err(|e| OpenRagError::Database(format!("bench_item_result.score: {e}")))?,
        pii_leaked: row
            .try_get("pii_leaked")
            .map_err(|e| OpenRagError::Database(format!("bench_item_result.pii_leaked: {e}")))?,
        arn_violated: row
            .try_get("arn_violated")
            .map_err(|e| OpenRagError::Database(format!("bench_item_result.arn_violated: {e}")))?,
        details: row
            .try_get("details")
            .map_err(|e| OpenRagError::Database(format!("bench_item_result.details: {e}")))?,
    })
}

fn parse_strategy(s: &str) -> Result<OptimizationStrategy, OpenRagError> {
    match s.trim() {
        "greedy_node" => Ok(OptimizationStrategy::GreedyNode),
        "grid" => Ok(OptimizationStrategy::Grid),
        "ab_pair" => Ok(OptimizationStrategy::AbPair),
        other => Err(OpenRagError::Database(format!(
            "invalid optimization strategy '{other}'"
        ))),
    }
}

fn strategy_str(strategy: OptimizationStrategy) -> &'static str {
    match strategy {
        OptimizationStrategy::GreedyNode => "greedy_node",
        OptimizationStrategy::Grid => "grid",
        OptimizationStrategy::AbPair => "ab_pair",
    }
}

fn row_to_trial(row: &PgRow) -> Result<OptimizationTrialRecord, OpenRagError> {
    let strat_str: String = row
        .try_get("strategy")
        .map_err(|e| OpenRagError::Database(format!("trial.strategy: {e}")))?;
    let status_str: String = row
        .try_get("status")
        .map_err(|e| OpenRagError::Database(format!("trial.status: {e}")))?;
    Ok(OptimizationTrialRecord {
        id: row
            .try_get("id")
            .map_err(|e| OpenRagError::Database(format!("trial.id: {e}")))?,
        org_id: row
            .try_get("org_id")
            .map_err(|e| OpenRagError::Database(format!("trial.org_id: {e}")))?,
        golden_set_id: row
            .try_get("golden_set_id")
            .map_err(|e| OpenRagError::Database(format!("trial.golden_set_id: {e}")))?,
        search_space_yaml: row
            .try_get("search_space_yaml")
            .map_err(|e| OpenRagError::Database(format!("trial.search_space_yaml: {e}")))?,
        strategy: parse_strategy(&strat_str)?,
        best_config_yaml: row
            .try_get("best_config_yaml")
            .map_err(|e| OpenRagError::Database(format!("trial.best_config_yaml: {e}")))?,
        best_score: row
            .try_get("best_score")
            .map_err(|e| OpenRagError::Database(format!("trial.best_score: {e}")))?,
        security_gate_pass: row
            .try_get("security_gate_pass")
            .map_err(|e| OpenRagError::Database(format!("trial.security_gate_pass: {e}")))?,
        status: status_str.parse()?,
        created_at: row
            .try_get("created_at")
            .map_err(|e| OpenRagError::Database(format!("trial.created_at: {e}")))?,
        completed_at: row
            .try_get("completed_at")
            .map_err(|e| OpenRagError::Database(format!("trial.completed_at: {e}")))?,
    })
}

fn row_to_pipeline_config(row: &PgRow) -> Result<PipelineConfigRecord, OpenRagError> {
    let profile_str: String = row
        .try_get("profile")
        .map_err(|e| OpenRagError::Database(format!("pipeline_config.profile: {e}")))?;
    Ok(PipelineConfigRecord {
        id: row
            .try_get("id")
            .map_err(|e| OpenRagError::Database(format!("pipeline_config.id: {e}")))?,
        org_id: row
            .try_get("org_id")
            .map_err(|e| OpenRagError::Database(format!("pipeline_config.org_id: {e}")))?,
        name: row
            .try_get("name")
            .map_err(|e| OpenRagError::Database(format!("pipeline_config.name: {e}")))?,
        profile: profile_str.parse()?,
        pipeline_yaml: row
            .try_get("pipeline_yaml")
            .map_err(|e| OpenRagError::Database(format!("pipeline_config.pipeline_yaml: {e}")))?,
        config_json: row
            .try_get("config_json")
            .map_err(|e| OpenRagError::Database(format!("pipeline_config.config_json: {e}")))?,
        is_promoted: row
            .try_get("is_promoted")
            .map_err(|e| OpenRagError::Database(format!("pipeline_config.is_promoted: {e}")))?,
        promoted_at: row
            .try_get("promoted_at")
            .map_err(|e| OpenRagError::Database(format!("pipeline_config.promoted_at: {e}")))?,
        created_by: row
            .try_get("created_by")
            .map_err(|e| OpenRagError::Database(format!("pipeline_config.created_by: {e}")))?,
        created_at: row
            .try_get("created_at")
            .map_err(|e| OpenRagError::Database(format!("pipeline_config.created_at: {e}")))?,
    })
}

#[async_trait]
impl EvalStore for PgEvalStore {
    async fn create_golden_set(
        &self,
        org_id: Uuid,
        input: &CreateGoldenSetInput,
    ) -> Result<GoldenSet, OpenRagError> {
        input.validate()?;
        let row = sqlx::query(
            "INSERT INTO eval.golden_sets (org_id, name, axis, description, version)
             VALUES ($1, $2, $3, $4, $5)
             RETURNING id, org_id, name, axis, description, version, created_at",
        )
        .bind(org_id)
        .bind(input.name.trim())
        .bind(input.axis.as_str())
        .bind(input.description.as_deref())
        .bind(input.version.unwrap_or(1))
        .fetch_one(self.worker_pool.inner())
        .await
        .map_err(|e| map_pg_err("create golden_set", e))?;
        row_to_golden_set(&row)
    }

    async fn get_golden_set(&self, org_id: Uuid, id: Uuid) -> Result<GoldenSet, OpenRagError> {
        let row = sqlx::query(
            "SELECT id, org_id, name, axis, description, version, created_at
             FROM eval.golden_sets
             WHERE org_id = $1 AND id = $2",
        )
        .bind(org_id)
        .bind(id)
        .fetch_optional(self.worker_pool.inner())
        .await
        .map_err(|e| map_pg_err("get golden_set", e))?
        .ok_or_else(|| OpenRagError::NotFound(format!("golden set {id} not found")))?;
        row_to_golden_set(&row)
    }

    async fn get_golden_set_by_name(
        &self,
        org_id: Uuid,
        name: &str,
    ) -> Result<GoldenSet, OpenRagError> {
        let row = sqlx::query(
            "SELECT id, org_id, name, axis, description, version, created_at
             FROM eval.golden_sets
             WHERE org_id = $1 AND name = $2",
        )
        .bind(org_id)
        .bind(name.trim())
        .fetch_optional(self.worker_pool.inner())
        .await
        .map_err(|e| map_pg_err("get golden_set by name", e))?
        .ok_or_else(|| OpenRagError::NotFound(format!("golden set '{name}' not found")))?;
        row_to_golden_set(&row)
    }

    async fn list_golden_sets(&self, org_id: Uuid) -> Result<Vec<GoldenSet>, OpenRagError> {
        let rows = sqlx::query(
            "SELECT id, org_id, name, axis, description, version, created_at
             FROM eval.golden_sets
             WHERE org_id = $1
             ORDER BY created_at DESC, name ASC",
        )
        .bind(org_id)
        .fetch_all(self.worker_pool.inner())
        .await
        .map_err(|e| map_pg_err("list golden_sets", e))?;
        rows.iter().map(row_to_golden_set).collect()
    }

    async fn add_golden_item(
        &self,
        org_id: Uuid,
        golden_set_id: Uuid,
        input: &CreateGoldenItemInput,
    ) -> Result<GoldenItem, OpenRagError> {
        input.validate()?;
        let _gs = self.get_golden_set(org_id, golden_set_id).await?;
        let status = input.review_status.unwrap_or(ReviewStatus::Approved);
        let row = sqlx::query(
            "INSERT INTO eval.golden_items (
                org_id, golden_set_id, axis, query_text, test_principal_id,
                expected_doc_ids, forbidden_doc_ids, expected_answer,
                expected_pii_tokens, expected_edges, review_status
             ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
             RETURNING id, org_id, golden_set_id, axis, query_text, test_principal_id,
                       expected_doc_ids, forbidden_doc_ids, expected_answer,
                       expected_pii_tokens, expected_edges, review_status, created_at",
        )
        .bind(org_id)
        .bind(golden_set_id)
        .bind(input.axis.as_str())
        .bind(input.query_text.trim())
        .bind(input.test_principal_id)
        .bind(&input.expected_doc_ids)
        .bind(&input.forbidden_doc_ids)
        .bind(input.expected_answer.as_deref())
        .bind(&input.expected_pii_tokens)
        .bind(&input.expected_edges)
        .bind(status.as_str())
        .fetch_one(self.worker_pool.inner())
        .await
        .map_err(|e| map_pg_err("insert golden_item", e))?;
        row_to_golden_item(&row)
    }

    async fn get_golden_item(
        &self,
        org_id: Uuid,
        item_id: Uuid,
    ) -> Result<GoldenItem, OpenRagError> {
        let row = sqlx::query(
            "SELECT id, org_id, golden_set_id, axis, query_text, test_principal_id,
                    expected_doc_ids, forbidden_doc_ids, expected_answer,
                    expected_pii_tokens, expected_edges, review_status, created_at
             FROM eval.golden_items
             WHERE org_id = $1 AND id = $2",
        )
        .bind(org_id)
        .bind(item_id)
        .fetch_optional(self.worker_pool.inner())
        .await
        .map_err(|e| map_pg_err("get golden_item", e))?
        .ok_or_else(|| OpenRagError::NotFound(format!("golden item {item_id} not found")))?;
        row_to_golden_item(&row)
    }

    async fn list_golden_items(
        &self,
        org_id: Uuid,
        golden_set_id: Uuid,
        include_rejected: bool,
    ) -> Result<Vec<GoldenItem>, OpenRagError> {
        let rows = if include_rejected {
            sqlx::query(
                "SELECT id, org_id, golden_set_id, axis, query_text, test_principal_id,
                        expected_doc_ids, forbidden_doc_ids, expected_answer,
                        expected_pii_tokens, expected_edges, review_status, created_at
                 FROM eval.golden_items
                 WHERE org_id = $1 AND golden_set_id = $2
                 ORDER BY created_at ASC, id ASC",
            )
            .bind(org_id)
            .bind(golden_set_id)
            .fetch_all(self.worker_pool.inner())
            .await
            .map_err(|e| map_pg_err("list golden_items", e))?
        } else {
            sqlx::query(
                "SELECT id, org_id, golden_set_id, axis, query_text, test_principal_id,
                        expected_doc_ids, forbidden_doc_ids, expected_answer,
                        expected_pii_tokens, expected_edges, review_status, created_at
                 FROM eval.golden_items
                 WHERE org_id = $1 AND golden_set_id = $2 AND review_status <> 'rejected'
                 ORDER BY created_at ASC, id ASC",
            )
            .bind(org_id)
            .bind(golden_set_id)
            .fetch_all(self.worker_pool.inner())
            .await
            .map_err(|e| map_pg_err("list golden_items", e))?
        };
        rows.iter().map(row_to_golden_item).collect()
    }

    async fn update_review_status(
        &self,
        org_id: Uuid,
        item_id: Uuid,
        status: ReviewStatus,
    ) -> Result<GoldenItem, OpenRagError> {
        let row = sqlx::query(
            "UPDATE eval.golden_items
             SET review_status = $3
             WHERE org_id = $1 AND id = $2
             RETURNING id, org_id, golden_set_id, axis, query_text, test_principal_id,
                       expected_doc_ids, forbidden_doc_ids, expected_answer,
                       expected_pii_tokens, expected_edges, review_status, created_at",
        )
        .bind(org_id)
        .bind(item_id)
        .bind(status.as_str())
        .fetch_optional(self.worker_pool.inner())
        .await
        .map_err(|e| map_pg_err("update golden_item review_status", e))?
        .ok_or_else(|| OpenRagError::NotFound(format!("golden item {item_id} not found")))?;
        row_to_golden_item(&row)
    }

    async fn submit_feedback_rls(
        &self,
        ctx: &PrincipalContext,
        input: &SubmitFeedbackInput,
    ) -> Result<SearchFeedback, OpenRagError> {
        input.validate()?;
        let mut tx = self.app_pool.begin_rls_tx(ctx).await?;
        let row = sqlx::query(
            "INSERT INTO eval.search_feedback (
                org_id, principal_id, query_text, response_text, cited_chunk_ids, verdict, comment
             ) VALUES ($1, $2, $3, $4, $5, $6, $7)
             RETURNING id, org_id, principal_id, query_text, response_text, cited_chunk_ids,
                       verdict, comment, promoted_golden_item_id, created_at",
        )
        .bind(ctx.org_id)
        .bind(ctx.principal_id)
        .bind(input.query_text.trim())
        .bind(input.response_text.as_deref())
        .bind(&input.cited_chunk_ids)
        .bind(input.verdict.as_str())
        .bind(input.comment.as_deref())
        .fetch_one(&mut **tx.tx_mut())
        .await
        .map_err(|e| map_pg_err("insert eval.search_feedback under RLS", e))?;
        tx.commit().await?;
        row_to_feedback(&row)
    }

    async fn list_feedback_rls(
        &self,
        ctx: &PrincipalContext,
    ) -> Result<Vec<SearchFeedback>, OpenRagError> {
        let mut tx = self.app_pool.begin_rls_tx(ctx).await?;
        let rows = sqlx::query(
            "SELECT id, org_id, principal_id, query_text, response_text, cited_chunk_ids,
                    verdict, comment, promoted_golden_item_id, created_at
             FROM eval.search_feedback
             WHERE org_id = $1
             ORDER BY created_at DESC",
        )
        .bind(ctx.org_id)
        .fetch_all(&mut **tx.tx_mut())
        .await
        .map_err(|e| map_pg_err("list eval.search_feedback under RLS", e))?;
        tx.commit().await?;
        rows.iter().map(row_to_feedback).collect()
    }

    async fn promote_feedback(
        &self,
        org_id: Uuid,
        actor: &PrincipalContext,
        feedback_id: Uuid,
        input: &PromoteFeedbackInput,
    ) -> Result<PromoteFeedbackResult, OpenRagError> {
        if actor.org_id != org_id || !actor.is_org_admin {
            return Err(OpenRagError::Forbidden(
                "org_admin privilege in target org required to promote feedback".into(),
            ));
        }
        let axis = input.axis.unwrap_or(GoldenAxis::Retrieval);
        if !axis.is_item_axis() {
            return Err(OpenRagError::Validation(
                "promoted golden item axis cannot be 'multi'".into(),
            ));
        }

        let mut tx = self
            .worker_pool
            .inner()
            .begin()
            .await
            .map_err(|e| OpenRagError::Database(format!("begin promote_feedback tx: {e}")))?;

        let fb_row = sqlx::query(
            "SELECT id, org_id, principal_id, query_text, response_text, cited_chunk_ids,
                    verdict, comment, promoted_golden_item_id, created_at
             FROM eval.search_feedback
             WHERE id = $1 AND org_id = $2
             FOR UPDATE",
        )
        .bind(feedback_id)
        .bind(org_id)
        .fetch_optional(&mut *tx)
        .await
        .map_err(|e| map_pg_err("lock search_feedback", e))?
        .ok_or_else(|| OpenRagError::NotFound(format!("search feedback {feedback_id} not found")))?;
        let feedback = row_to_feedback(&fb_row)?;

        let gs_exists: Option<Uuid> = sqlx::query_scalar(
            "SELECT id FROM eval.golden_sets WHERE id = $1 AND org_id = $2",
        )
        .bind(input.golden_set_id)
        .bind(org_id)
        .fetch_optional(&mut *tx)
        .await
        .map_err(|e| map_pg_err("verify golden_set for promotion", e))?;
        if gs_exists.is_none() {
            return Err(OpenRagError::NotFound(format!(
                "golden set {} not found in org",
                input.golden_set_id
            )));
        }

        if let Some(existing_item_id) = feedback.promoted_golden_item_id {
            let item_row = sqlx::query(
                "SELECT id, org_id, golden_set_id, axis, query_text, test_principal_id,
                        expected_doc_ids, forbidden_doc_ids, expected_answer,
                        expected_pii_tokens, expected_edges, review_status, created_at
                 FROM eval.golden_items
                 WHERE id = $1 AND org_id = $2",
            )
            .bind(existing_item_id)
            .bind(org_id)
            .fetch_optional(&mut *tx)
            .await
            .map_err(|e| map_pg_err("fetch promoted golden_item", e))?
            .ok_or_else(|| {
                OpenRagError::NotFound(format!("promoted golden item {existing_item_id} missing"))
            })?;
            let item = row_to_golden_item(&item_row)?;
            if item.golden_set_id != input.golden_set_id {
                return Err(OpenRagError::Conflict(format!(
                    "feedback {feedback_id} already promoted to golden set {}",
                    item.golden_set_id
                )));
            }
            tx.commit()
                .await
                .map_err(|e| OpenRagError::Database(format!("commit promote tx: {e}")))?;
            return Ok(PromoteFeedbackResult {
                feedback,
                item,
                already_promoted: true,
            });
        }

        let expected_docs = if input.expected_doc_ids.is_empty()
            && input.forbidden_doc_ids.is_empty()
            && input.expected_answer.is_none()
            && input.expected_pii_tokens.is_empty()
            && input.expected_edges.is_none()
        {
            feedback.cited_chunk_ids.clone()
        } else {
            input.expected_doc_ids.clone()
        };
        let expected_answer = input
            .expected_answer
            .clone()
            .or_else(|| feedback.response_text.clone());
        let review_status = input.review_status.unwrap_or(ReviewStatus::Approved);

        let item_row = sqlx::query(
            "INSERT INTO eval.golden_items (
                org_id, golden_set_id, axis, query_text, test_principal_id,
                expected_doc_ids, forbidden_doc_ids, expected_answer,
                expected_pii_tokens, expected_edges, review_status
             ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
             RETURNING id, org_id, golden_set_id, axis, query_text, test_principal_id,
                       expected_doc_ids, forbidden_doc_ids, expected_answer,
                       expected_pii_tokens, expected_edges, review_status, created_at",
        )
        .bind(org_id)
        .bind(input.golden_set_id)
        .bind(axis.as_str())
        .bind(feedback.query_text.trim())
        .bind(input.test_principal_id.or(Some(feedback.principal_id)))
        .bind(&expected_docs)
        .bind(&input.forbidden_doc_ids)
        .bind(expected_answer.as_deref())
        .bind(&input.expected_pii_tokens)
        .bind(&input.expected_edges)
        .bind(review_status.as_str())
        .fetch_one(&mut *tx)
        .await
        .map_err(|e| map_pg_err("insert promoted golden_item", e))?;
        let item = row_to_golden_item(&item_row)?;

        let updated_fb_row = sqlx::query(
            "UPDATE eval.search_feedback
             SET promoted_golden_item_id = $1
             WHERE id = $2 AND org_id = $3 AND promoted_golden_item_id IS NULL
             RETURNING id, org_id, principal_id, query_text, response_text, cited_chunk_ids,
                       verdict, comment, promoted_golden_item_id, created_at",
        )
        .bind(item.id)
        .bind(feedback_id)
        .bind(org_id)
        .fetch_optional(&mut *tx)
        .await
        .map_err(|e| map_pg_err("link promoted_golden_item_id", e))?
        .ok_or_else(|| {
            OpenRagError::Conflict(format!("feedback {feedback_id} was already promoted"))
        })?;
        let updated_feedback = row_to_feedback(&updated_fb_row)?;

        tx.commit()
            .await
            .map_err(|e| OpenRagError::Database(format!("commit promote_feedback tx: {e}")))?;

        Ok(PromoteFeedbackResult {
            feedback: updated_feedback,
            item,
            already_promoted: false,
        })
    }

    async fn resolve_principal(
        &self,
        org_id: Uuid,
        principal_id: Uuid,
    ) -> Result<Option<PrincipalContext>, OpenRagError> {
        let row = sqlx::query(
            "SELECT id, org_id, auth_level, can_unmask_pii, is_org_admin
             FROM iam.principals
             WHERE org_id = $1 AND id = $2 AND status = 'active'",
        )
        .bind(org_id)
        .bind(principal_id)
        .fetch_optional(self.worker_pool.inner())
        .await
        .map_err(|e| map_pg_err("resolve iam.principals", e))?;

        let Some(row) = row else {
            return Ok(None);
        };

        let raw_level: i16 = row
            .try_get("auth_level")
            .map_err(|e| OpenRagError::Database(format!("principal.auth_level: {e}")))?;
        let auth_level = AuthLevel::try_from(raw_level)?;
        let can_unmask_pii: bool = row
            .try_get("can_unmask_pii")
            .map_err(|e| OpenRagError::Database(format!("principal.can_unmask_pii: {e}")))?;
        let is_org_admin: bool = row
            .try_get("is_org_admin")
            .map_err(|e| OpenRagError::Database(format!("principal.is_org_admin: {e}")))?;

        let dept_rows = sqlx::query(
            "SELECT dept_id, dept_role
             FROM iam.dept_memberships
             WHERE org_id = $1 AND principal_id = $2",
        )
        .bind(org_id)
        .bind(principal_id)
        .fetch_all(self.worker_pool.inner())
        .await
        .map_err(|e| map_pg_err("resolve dept_memberships", e))?;

        let mut dept_ids = Vec::with_capacity(dept_rows.len());
        let mut dept_admin_ids = Vec::new();
        for dr in dept_rows {
            let dept_id: Uuid = dr
                .try_get("dept_id")
                .map_err(|e| OpenRagError::Database(format!("dept_membership.dept_id: {e}")))?;
            let dept_role: String = dr
                .try_get("dept_role")
                .map_err(|e| OpenRagError::Database(format!("dept_membership.dept_role: {e}")))?;
            dept_ids.push(dept_id);
            if dept_role == "dept_admin" {
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
        .fetch_all(self.worker_pool.inner())
        .await
        .map_err(|e| map_pg_err("resolve group_members", e))?;

        let mut ctx = PrincipalContext::new(org_id, principal_id, auth_level);
        ctx.dept_ids = dept_ids;
        ctx.dept_admin_ids = dept_admin_ids;
        ctx.group_ids = group_ids;
        ctx.can_unmask_pii = can_unmask_pii;
        ctx.is_org_admin = is_org_admin;
        Ok(Some(ctx))
    }

    async fn insert_bench_run(
        &self,
        run: &NewBenchRun,
        items: &[NewBenchItemResult],
    ) -> Result<BenchRunRecord, OpenRagError> {
        let mut tx = self
            .worker_pool
            .inner()
            .begin()
            .await
            .map_err(|e| OpenRagError::Database(format!("begin insert_bench_run tx: {e}")))?;

        // CRITICAL: security_gate_pass is GENERATED ALWAYS — NEVER write it in INSERT/UPDATE.
        let run_row = sqlx::query(
            "INSERT INTO eval.bench_runs (
                org_id, golden_set_id, pipeline_config_id, trial_id,
                track, metrics, pii_leak_count, arn_violation_count
             ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
             RETURNING id, org_id, golden_set_id, pipeline_config_id, trial_id,
                       track, metrics, pii_leak_count, arn_violation_count,
                       security_gate_pass, created_at",
        )
        .bind(run.org_id)
        .bind(run.golden_set_id)
        .bind(run.pipeline_config_id)
        .bind(run.trial_id)
        .bind(run.track.as_str())
        .bind(&run.metrics)
        .bind(run.pii_leak_count)
        .bind(run.arn_violation_count)
        .fetch_one(&mut *tx)
        .await
        .map_err(|e| map_pg_err("insert eval.bench_runs", e))?;
        let record = row_to_bench_run(&run_row)?;

        for item in items {
            sqlx::query(
                "INSERT INTO eval.bench_item_results (
                    bench_run_id, golden_item_id, retrieved_doc_ids,
                    latency_ms, score, pii_leaked, arn_violated, details
                 ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)",
            )
            .bind(record.id)
            .bind(item.golden_item_id)
            .bind(&item.retrieved_doc_ids)
            .bind(item.latency_ms)
            .bind(item.score)
            .bind(item.pii_leaked)
            .bind(item.arn_violated)
            .bind(&item.details)
            .execute(&mut *tx)
            .await
            .map_err(|e| map_pg_err("insert eval.bench_item_results", e))?;
        }

        tx.commit()
            .await
            .map_err(|e| OpenRagError::Database(format!("commit insert_bench_run tx: {e}")))?;

        Ok(record)
    }

    async fn get_bench_run(
        &self,
        org_id: Uuid,
        bench_run_id: Uuid,
    ) -> Result<BenchRunRecord, OpenRagError> {
        let row = sqlx::query(
            "SELECT id, org_id, golden_set_id, pipeline_config_id, trial_id,
                    track, metrics, pii_leak_count, arn_violation_count,
                    security_gate_pass, created_at
             FROM eval.bench_runs
             WHERE org_id = $1 AND id = $2",
        )
        .bind(org_id)
        .bind(bench_run_id)
        .fetch_optional(self.worker_pool.inner())
        .await
        .map_err(|e| map_pg_err("get bench_run", e))?
        .ok_or_else(|| OpenRagError::NotFound(format!("bench run {bench_run_id} not found")))?;
        row_to_bench_run(&row)
    }

    async fn list_bench_runs(
        &self,
        org_id: Uuid,
        golden_set_id: Option<Uuid>,
    ) -> Result<Vec<BenchRunRecord>, OpenRagError> {
        let rows = match golden_set_id {
            Some(gs_id) => sqlx::query(
                "SELECT id, org_id, golden_set_id, pipeline_config_id, trial_id,
                        track, metrics, pii_leak_count, arn_violation_count,
                        security_gate_pass, created_at
                 FROM eval.bench_runs
                 WHERE org_id = $1 AND golden_set_id = $2
                 ORDER BY created_at DESC",
            )
            .bind(org_id)
            .bind(gs_id)
            .fetch_all(self.worker_pool.inner())
            .await
            .map_err(|e| map_pg_err("list bench_runs", e))?,
            None => sqlx::query(
                "SELECT id, org_id, golden_set_id, pipeline_config_id, trial_id,
                        track, metrics, pii_leak_count, arn_violation_count,
                        security_gate_pass, created_at
                 FROM eval.bench_runs
                 WHERE org_id = $1
                 ORDER BY created_at DESC",
            )
            .bind(org_id)
            .fetch_all(self.worker_pool.inner())
            .await
            .map_err(|e| map_pg_err("list bench_runs", e))?,
        };
        rows.iter().map(row_to_bench_run).collect()
    }

    async fn list_bench_item_results(
        &self,
        org_id: Uuid,
        bench_run_id: Uuid,
    ) -> Result<Vec<BenchItemResultRecord>, OpenRagError> {
        let rows = sqlx::query(
            "SELECT bir.id, bir.bench_run_id, bir.golden_item_id, bir.retrieved_doc_ids,
                    bir.latency_ms, bir.score, bir.pii_leaked, bir.arn_violated, bir.details
             FROM eval.bench_item_results AS bir
             JOIN eval.bench_runs AS br ON br.id = bir.bench_run_id
             WHERE br.org_id = $1 AND bir.bench_run_id = $2
             ORDER BY bir.id ASC",
        )
        .bind(org_id)
        .bind(bench_run_id)
        .fetch_all(self.worker_pool.inner())
        .await
        .map_err(|e| map_pg_err("list bench_item_results", e))?;
        rows.iter().map(row_to_bench_item_result).collect()
    }

    async fn insert_optimization_trial(
        &self,
        trial: &NewOptimizationTrial,
    ) -> Result<OptimizationTrialRecord, OpenRagError> {
        let row = sqlx::query(
            "INSERT INTO eval.optimization_trials (
                org_id, golden_set_id, search_space_yaml, strategy, status
             ) VALUES ($1, $2, $3, $4, 'running')
             RETURNING id, org_id, golden_set_id, search_space_yaml, strategy,
                       best_config_yaml, best_score, security_gate_pass, status,
                       created_at, completed_at",
        )
        .bind(trial.org_id)
        .bind(trial.golden_set_id)
        .bind(&trial.search_space_yaml)
        .bind(strategy_str(trial.strategy))
        .fetch_one(self.worker_pool.inner())
        .await
        .map_err(|e| map_pg_err("insert optimization_trial", e))?;
        row_to_trial(&row)
    }

    async fn complete_optimization_trial(
        &self,
        org_id: Uuid,
        trial_id: Uuid,
        update: &CompleteOptimizationTrial,
    ) -> Result<OptimizationTrialRecord, OpenRagError> {
        let row = sqlx::query(
            "UPDATE eval.optimization_trials
             SET best_config_yaml = $3,
                 best_score = $4,
                 security_gate_pass = $5,
                 status = $6,
                 completed_at = now()
             WHERE org_id = $1 AND id = $2
             RETURNING id, org_id, golden_set_id, search_space_yaml, strategy,
                       best_config_yaml, best_score, security_gate_pass, status,
                       created_at, completed_at",
        )
        .bind(org_id)
        .bind(trial_id)
        .bind(update.best_config_yaml.as_deref())
        .bind(update.best_score)
        .bind(update.security_gate_pass)
        .bind(update.status.as_str())
        .fetch_optional(self.worker_pool.inner())
        .await
        .map_err(|e| map_pg_err("complete optimization_trial", e))?
        .ok_or_else(|| {
            OpenRagError::NotFound(format!("optimization trial {trial_id} not found"))
        })?;
        row_to_trial(&row)
    }

    async fn list_optimization_trials(
        &self,
        org_id: Uuid,
        golden_set_id: Option<Uuid>,
    ) -> Result<Vec<OptimizationTrialRecord>, OpenRagError> {
        let rows = match golden_set_id {
            Some(gs_id) => sqlx::query(
                "SELECT id, org_id, golden_set_id, search_space_yaml, strategy,
                        best_config_yaml, best_score, security_gate_pass, status,
                        created_at, completed_at
                 FROM eval.optimization_trials
                 WHERE org_id = $1 AND golden_set_id = $2
                 ORDER BY created_at DESC",
            )
            .bind(org_id)
            .bind(gs_id)
            .fetch_all(self.worker_pool.inner())
            .await
            .map_err(|e| map_pg_err("list optimization_trials", e))?,
            None => sqlx::query(
                "SELECT id, org_id, golden_set_id, search_space_yaml, strategy,
                        best_config_yaml, best_score, security_gate_pass, status,
                        created_at, completed_at
                 FROM eval.optimization_trials
                 WHERE org_id = $1
                 ORDER BY created_at DESC",
            )
            .bind(org_id)
            .fetch_all(self.worker_pool.inner())
            .await
            .map_err(|e| map_pg_err("list optimization_trials", e))?,
        };
        rows.iter().map(row_to_trial).collect()
    }

    async fn register_pipeline_config(
        &self,
        org_id: Uuid,
        input: &RegisterPipelineInput,
    ) -> Result<PipelineConfigRecord, OpenRagError> {
        if input.name.trim().is_empty() {
            return Err(OpenRagError::Validation(
                "pipeline config name must not be empty".into(),
            ));
        }
        let row = sqlx::query(
            "INSERT INTO eval.pipeline_configs (
                org_id, name, profile, pipeline_yaml, config_json, created_by
             ) VALUES ($1, $2, $3, $4, $5, $6)
             RETURNING id, org_id, name, profile, pipeline_yaml, config_json,
                       is_promoted, promoted_at, created_by, created_at",
        )
        .bind(org_id)
        .bind(input.name.trim())
        .bind(input.profile.as_str())
        .bind(&input.pipeline_yaml)
        .bind(&input.config_json)
        .bind(input.created_by)
        .fetch_one(self.worker_pool.inner())
        .await
        .map_err(|e| map_pg_err("insert eval.pipeline_configs", e))?;
        row_to_pipeline_config(&row)
    }

    async fn get_pipeline_config(
        &self,
        org_id: Uuid,
        config_id: Uuid,
    ) -> Result<PipelineConfigRecord, OpenRagError> {
        let row = sqlx::query(
            "SELECT id, org_id, name, profile, pipeline_yaml, config_json,
                    is_promoted, promoted_at, created_by, created_at
             FROM eval.pipeline_configs
             WHERE org_id = $1 AND id = $2",
        )
        .bind(org_id)
        .bind(config_id)
        .fetch_optional(self.worker_pool.inner())
        .await
        .map_err(|e| map_pg_err("get eval.pipeline_configs", e))?
        .ok_or_else(|| {
            OpenRagError::NotFound(format!("pipeline config {config_id} not found"))
        })?;
        row_to_pipeline_config(&row)
    }

    async fn list_pipeline_configs(
        &self,
        org_id: Uuid,
    ) -> Result<Vec<PipelineConfigRecord>, OpenRagError> {
        let rows = sqlx::query(
            "SELECT id, org_id, name, profile, pipeline_yaml, config_json,
                    is_promoted, promoted_at, created_by, created_at
             FROM eval.pipeline_configs
             WHERE org_id = $1
             ORDER BY created_at DESC",
        )
        .bind(org_id)
        .fetch_all(self.worker_pool.inner())
        .await
        .map_err(|e| map_pg_err("list eval.pipeline_configs", e))?;
        rows.iter().map(row_to_pipeline_config).collect()
    }

    async fn latest_bench_run_for_config(
        &self,
        org_id: Uuid,
        pipeline_config_id: Uuid,
    ) -> Result<Option<BenchRunRecord>, OpenRagError> {
        let row = sqlx::query(
            "SELECT id, org_id, golden_set_id, pipeline_config_id, trial_id,
                    track, metrics, pii_leak_count, arn_violation_count,
                    security_gate_pass, created_at
             FROM eval.bench_runs
             WHERE org_id = $1 AND pipeline_config_id = $2
             ORDER BY created_at DESC, security_gate_pass ASC, id DESC
             LIMIT 1",
        )
        .bind(org_id)
        .bind(pipeline_config_id)
        .fetch_optional(self.worker_pool.inner())
        .await
        .map_err(|e| map_pg_err("latest bench_run for pipeline_config", e))?;
        row.as_ref().map(row_to_bench_run).transpose()
    }

    async fn latest_bench_runs_by_track_for_config(
        &self,
        org_id: Uuid,
        pipeline_config_id: Uuid,
    ) -> Result<Vec<BenchRunRecord>, OpenRagError> {
        let rows = sqlx::query(
            "SELECT DISTINCT ON (track)
                    id, org_id, golden_set_id, pipeline_config_id, trial_id,
                    track, metrics, pii_leak_count, arn_violation_count,
                    security_gate_pass, created_at
             FROM eval.bench_runs
             WHERE org_id = $1 AND pipeline_config_id = $2
             ORDER BY track, created_at DESC, security_gate_pass ASC, id DESC",
        )
        .bind(org_id)
        .bind(pipeline_config_id)
        .fetch_all(self.worker_pool.inner())
        .await
        .map_err(|e| map_pg_err("latest bench_runs by track for pipeline_config", e))?;
        rows.iter().map(row_to_bench_run).collect()
    }

    async fn mark_pipeline_promoted(
        &self,
        org_id: Uuid,
        config_id: Uuid,
        bucket_ref: Option<&str>,
    ) -> Result<PipelineConfigRecord, OpenRagError> {
        let mut tx = self
            .worker_pool
            .inner()
            .begin()
            .await
            .map_err(|e| OpenRagError::Database(format!("begin mark_pipeline_promoted tx: {e}")))?;

        let row = sqlx::query(
            "UPDATE eval.pipeline_configs
             SET is_promoted = true, promoted_at = now()
             WHERE org_id = $1 AND id = $2
             RETURNING id, org_id, name, profile, pipeline_yaml, config_json,
                       is_promoted, promoted_at, created_by, created_at",
        )
        .bind(org_id)
        .bind(config_id)
        .fetch_optional(&mut *tx)
        .await
        .map_err(|e| map_pg_err("promote eval.pipeline_configs", e))?
        .ok_or_else(|| {
            OpenRagError::NotFound(format!("pipeline config {config_id} not found"))
        })?;
        let record = row_to_pipeline_config(&row)?;

        if let Some(raw_bucket) = bucket_ref {
            let trimmed = raw_bucket.trim();
            if !trimmed.is_empty() {
                let updated = if let Ok(bucket_id) = Uuid::parse_str(trimmed) {
                    sqlx::query(
                        "UPDATE corpus.buckets
                         SET pipeline_config_id = $3
                         WHERE org_id = $1 AND id = $2",
                    )
                    .bind(org_id)
                    .bind(bucket_id)
                    .bind(config_id)
                    .execute(&mut *tx)
                    .await
                    .map_err(|e| map_pg_err("bind pipeline_config to bucket by id", e))?
                } else {
                    sqlx::query(
                        "UPDATE corpus.buckets
                         SET pipeline_config_id = $3
                         WHERE org_id = $1 AND slug = $2",
                    )
                    .bind(org_id)
                    .bind(trimmed)
                    .bind(config_id)
                    .execute(&mut *tx)
                    .await
                    .map_err(|e| map_pg_err("bind pipeline_config to bucket by slug", e))?
                };
                if updated.rows_affected() == 0 {
                    return Err(OpenRagError::NotFound(format!(
                        "bucket '{trimmed}' not found in org"
                    )));
                }
            }
        }

        tx.commit().await.map_err(|e| {
            OpenRagError::Database(format!("commit mark_pipeline_promoted tx: {e}"))
        })?;

        Ok(record)
    }
}

#[derive(Debug, Default)]
struct InMemoryEvalState {
    golden_sets: HashMap<Uuid, GoldenSet>,
    golden_items: HashMap<Uuid, GoldenItem>,
    feedbacks: HashMap<Uuid, SearchFeedback>,
    principals: HashMap<(Uuid, Uuid), PrincipalContext>,
    bench_runs: HashMap<Uuid, BenchRunRecord>,
    bench_run_seq: HashMap<Uuid, u64>,
    next_seq: u64,
    bench_item_results: Vec<BenchItemResultRecord>,
    trials: HashMap<Uuid, OptimizationTrialRecord>,
    pipeline_configs: HashMap<Uuid, PipelineConfigRecord>,
    bucket_bindings: HashMap<(Uuid, String), Uuid>,
}

#[derive(Debug, Clone, Default)]
pub struct InMemoryEvalStore {
    state: Arc<Mutex<InMemoryEvalState>>,
}

impl InMemoryEvalStore {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert_principal(&self, ctx: PrincipalContext) -> Result<(), OpenRagError> {
        let mut guard = self
            .state
            .lock()
            .map_err(|_| OpenRagError::Internal("InMemoryEvalStore lock poisoned".into()))?;
        guard.principals.insert((ctx.org_id, ctx.principal_id), ctx);
        Ok(())
    }

    pub fn get_bucket_binding(
        &self,
        org_id: Uuid,
        bucket_ref: &str,
    ) -> Result<Option<Uuid>, OpenRagError> {
        let guard = self
            .state
            .lock()
            .map_err(|_| OpenRagError::Internal("InMemoryEvalStore lock poisoned".into()))?;
        Ok(guard
            .bucket_bindings
            .get(&(org_id, bucket_ref.trim().to_string()))
            .copied())
    }
}

#[async_trait]
impl EvalStore for InMemoryEvalStore {
    async fn create_golden_set(
        &self,
        org_id: Uuid,
        input: &CreateGoldenSetInput,
    ) -> Result<GoldenSet, OpenRagError> {
        input.validate()?;
        let name = input.name.trim().to_string();
        let mut guard = self
            .state
            .lock()
            .map_err(|_| OpenRagError::Internal("InMemoryEvalStore lock poisoned".into()))?;
        if guard
            .golden_sets
            .values()
            .any(|gs| gs.org_id == org_id && gs.name == name)
        {
            return Err(OpenRagError::Conflict(format!(
                "golden set '{name}' already exists in org"
            )));
        }
        let gs = GoldenSet {
            id: Uuid::new_v4(),
            org_id,
            name,
            axis: input.axis,
            description: input.description.clone(),
            version: input.version.unwrap_or(1),
            created_at: Utc::now(),
        };
        guard.golden_sets.insert(gs.id, gs.clone());
        Ok(gs)
    }

    async fn get_golden_set(&self, org_id: Uuid, id: Uuid) -> Result<GoldenSet, OpenRagError> {
        let guard = self
            .state
            .lock()
            .map_err(|_| OpenRagError::Internal("InMemoryEvalStore lock poisoned".into()))?;
        guard
            .golden_sets
            .get(&id)
            .filter(|gs| gs.org_id == org_id)
            .cloned()
            .ok_or_else(|| OpenRagError::NotFound(format!("golden set {id} not found")))
    }

    async fn get_golden_set_by_name(
        &self,
        org_id: Uuid,
        name: &str,
    ) -> Result<GoldenSet, OpenRagError> {
        let trimmed = name.trim();
        let guard = self
            .state
            .lock()
            .map_err(|_| OpenRagError::Internal("InMemoryEvalStore lock poisoned".into()))?;
        guard
            .golden_sets
            .values()
            .find(|gs| gs.org_id == org_id && gs.name == trimmed)
            .cloned()
            .ok_or_else(|| OpenRagError::NotFound(format!("golden set '{name}' not found")))
    }

    async fn list_golden_sets(&self, org_id: Uuid) -> Result<Vec<GoldenSet>, OpenRagError> {
        let guard = self
            .state
            .lock()
            .map_err(|_| OpenRagError::Internal("InMemoryEvalStore lock poisoned".into()))?;
        let mut out: Vec<GoldenSet> = guard
            .golden_sets
            .values()
            .filter(|gs| gs.org_id == org_id)
            .cloned()
            .collect();
        out.sort_by(|a, b| b.created_at.cmp(&a.created_at));
        Ok(out)
    }

    async fn add_golden_item(
        &self,
        org_id: Uuid,
        golden_set_id: Uuid,
        input: &CreateGoldenItemInput,
    ) -> Result<GoldenItem, OpenRagError> {
        input.validate()?;
        let mut guard = self
            .state
            .lock()
            .map_err(|_| OpenRagError::Internal("InMemoryEvalStore lock poisoned".into()))?;
        if guard
            .golden_sets
            .get(&golden_set_id)
            .is_none_or(|gs| gs.org_id != org_id)
        {
            return Err(OpenRagError::NotFound(format!(
                "golden set {golden_set_id} not found"
            )));
        }
        let item = GoldenItem {
            id: Uuid::new_v4(),
            org_id,
            golden_set_id,
            axis: input.axis,
            query_text: input.query_text.trim().to_string(),
            test_principal_id: input.test_principal_id,
            expected_doc_ids: input.expected_doc_ids.clone(),
            forbidden_doc_ids: input.forbidden_doc_ids.clone(),
            expected_answer: input.expected_answer.clone(),
            expected_pii_tokens: input.expected_pii_tokens.clone(),
            expected_edges: input.expected_edges.clone(),
            review_status: input.review_status.unwrap_or(ReviewStatus::Approved),
            created_at: Utc::now(),
        };
        guard.golden_items.insert(item.id, item.clone());
        Ok(item)
    }

    async fn get_golden_item(
        &self,
        org_id: Uuid,
        item_id: Uuid,
    ) -> Result<GoldenItem, OpenRagError> {
        let guard = self
            .state
            .lock()
            .map_err(|_| OpenRagError::Internal("InMemoryEvalStore lock poisoned".into()))?;
        guard
            .golden_items
            .get(&item_id)
            .filter(|i| i.org_id == org_id)
            .cloned()
            .ok_or_else(|| OpenRagError::NotFound(format!("golden item {item_id} not found")))
    }

    async fn list_golden_items(
        &self,
        org_id: Uuid,
        golden_set_id: Uuid,
        include_rejected: bool,
    ) -> Result<Vec<GoldenItem>, OpenRagError> {
        let guard = self
            .state
            .lock()
            .map_err(|_| OpenRagError::Internal("InMemoryEvalStore lock poisoned".into()))?;
        let mut out: Vec<GoldenItem> = guard
            .golden_items
            .values()
            .filter(|item| {
                item.org_id == org_id
                    && item.golden_set_id == golden_set_id
                    && (include_rejected || item.review_status != ReviewStatus::Rejected)
            })
            .cloned()
            .collect();
        out.sort_by(|a, b| a.created_at.cmp(&b.created_at).then_with(|| a.id.cmp(&b.id)));
        Ok(out)
    }

    async fn update_review_status(
        &self,
        org_id: Uuid,
        item_id: Uuid,
        status: ReviewStatus,
    ) -> Result<GoldenItem, OpenRagError> {
        let mut guard = self
            .state
            .lock()
            .map_err(|_| OpenRagError::Internal("InMemoryEvalStore lock poisoned".into()))?;
        let item = guard
            .golden_items
            .get_mut(&item_id)
            .filter(|i| i.org_id == org_id)
            .ok_or_else(|| OpenRagError::NotFound(format!("golden item {item_id} not found")))?;
        item.review_status = status;
        Ok(item.clone())
    }

    async fn submit_feedback_rls(
        &self,
        ctx: &PrincipalContext,
        input: &SubmitFeedbackInput,
    ) -> Result<SearchFeedback, OpenRagError> {
        input.validate()?;
        let mut guard = self
            .state
            .lock()
            .map_err(|_| OpenRagError::Internal("InMemoryEvalStore lock poisoned".into()))?;
        let fb = SearchFeedback {
            id: Uuid::new_v4(),
            org_id: ctx.org_id,
            principal_id: ctx.principal_id,
            query_text: input.query_text.trim().to_string(),
            response_text: input.response_text.clone(),
            cited_chunk_ids: input.cited_chunk_ids.clone(),
            verdict: input.verdict,
            comment: input.comment.clone(),
            promoted_golden_item_id: None,
            created_at: Utc::now(),
        };
        guard.feedbacks.insert(fb.id, fb.clone());
        Ok(fb)
    }

    async fn list_feedback_rls(
        &self,
        ctx: &PrincipalContext,
    ) -> Result<Vec<SearchFeedback>, OpenRagError> {
        let guard = self
            .state
            .lock()
            .map_err(|_| OpenRagError::Internal("InMemoryEvalStore lock poisoned".into()))?;
        let mut out: Vec<SearchFeedback> = guard
            .feedbacks
            .values()
            .filter(|fb| {
                fb.org_id == ctx.org_id && (fb.principal_id == ctx.principal_id || ctx.is_org_admin)
            })
            .cloned()
            .collect();
        out.sort_by(|a, b| b.created_at.cmp(&a.created_at));
        Ok(out)
    }

    async fn promote_feedback(
        &self,
        org_id: Uuid,
        actor: &PrincipalContext,
        feedback_id: Uuid,
        input: &PromoteFeedbackInput,
    ) -> Result<PromoteFeedbackResult, OpenRagError> {
        if actor.org_id != org_id || !actor.is_org_admin {
            return Err(OpenRagError::Forbidden(
                "org_admin privilege in target org required to promote feedback".into(),
            ));
        }
        let axis = input.axis.unwrap_or(GoldenAxis::Retrieval);
        if !axis.is_item_axis() {
            return Err(OpenRagError::Validation(
                "promoted golden item axis cannot be 'multi'".into(),
            ));
        }

        let mut guard = self
            .state
            .lock()
            .map_err(|_| OpenRagError::Internal("InMemoryEvalStore lock poisoned".into()))?;

        if guard
            .golden_sets
            .get(&input.golden_set_id)
            .is_none_or(|gs| gs.org_id != org_id)
        {
            return Err(OpenRagError::NotFound(format!(
                "golden set {} not found",
                input.golden_set_id
            )));
        }

        let existing_fb = guard
            .feedbacks
            .get(&feedback_id)
            .filter(|fb| fb.org_id == org_id)
            .cloned()
            .ok_or_else(|| {
                OpenRagError::NotFound(format!("search feedback {feedback_id} not found"))
            })?;

        if let Some(existing_item_id) = existing_fb.promoted_golden_item_id {
            let item = guard
                .golden_items
                .get(&existing_item_id)
                .filter(|i| i.org_id == org_id)
                .cloned()
                .ok_or_else(|| {
                    OpenRagError::NotFound(format!("promoted item {existing_item_id} missing"))
                })?;
            if item.golden_set_id != input.golden_set_id {
                return Err(OpenRagError::Conflict(format!(
                    "feedback {feedback_id} already promoted to golden set {}",
                    item.golden_set_id
                )));
            }
            return Ok(PromoteFeedbackResult {
                feedback: existing_fb,
                item,
                already_promoted: true,
            });
        }

        let expected_docs = if input.expected_doc_ids.is_empty()
            && input.forbidden_doc_ids.is_empty()
            && input.expected_answer.is_none()
            && input.expected_pii_tokens.is_empty()
            && input.expected_edges.is_none()
        {
            existing_fb.cited_chunk_ids.clone()
        } else {
            input.expected_doc_ids.clone()
        };

        let item = GoldenItem {
            id: Uuid::new_v4(),
            org_id,
            golden_set_id: input.golden_set_id,
            axis,
            query_text: existing_fb.query_text.clone(),
            test_principal_id: input.test_principal_id.or(Some(existing_fb.principal_id)),
            expected_doc_ids: expected_docs,
            forbidden_doc_ids: input.forbidden_doc_ids.clone(),
            expected_answer: input
                .expected_answer
                .clone()
                .or_else(|| existing_fb.response_text.clone()),
            expected_pii_tokens: input.expected_pii_tokens.clone(),
            expected_edges: input.expected_edges.clone(),
            review_status: input.review_status.unwrap_or(ReviewStatus::Approved),
            created_at: Utc::now(),
        };
        guard.golden_items.insert(item.id, item.clone());

        let fb_mut = guard
            .feedbacks
            .get_mut(&feedback_id)
            .ok_or_else(|| OpenRagError::NotFound("feedback disappeared".into()))?;
        fb_mut.promoted_golden_item_id = Some(item.id);
        let updated_fb = fb_mut.clone();

        Ok(PromoteFeedbackResult {
            feedback: updated_fb,
            item,
            already_promoted: false,
        })
    }

    async fn resolve_principal(
        &self,
        org_id: Uuid,
        principal_id: Uuid,
    ) -> Result<Option<PrincipalContext>, OpenRagError> {
        let guard = self
            .state
            .lock()
            .map_err(|_| OpenRagError::Internal("InMemoryEvalStore lock poisoned".into()))?;
        Ok(guard.principals.get(&(org_id, principal_id)).cloned())
    }

    async fn insert_bench_run(
        &self,
        run: &NewBenchRun,
        items: &[NewBenchItemResult],
    ) -> Result<BenchRunRecord, OpenRagError> {
        let mut guard = self
            .state
            .lock()
            .map_err(|_| OpenRagError::Internal("InMemoryEvalStore lock poisoned".into()))?;
        let record = BenchRunRecord {
            id: Uuid::new_v4(),
            org_id: run.org_id,
            golden_set_id: run.golden_set_id,
            pipeline_config_id: run.pipeline_config_id,
            trial_id: run.trial_id,
            track: run.track,
            metrics: run.metrics.clone(),
            pii_leak_count: run.pii_leak_count,
            arn_violation_count: run.arn_violation_count,
            security_gate_pass: run.pii_leak_count == 0 && run.arn_violation_count == 0,
            created_at: Utc::now(),
        };
        guard.next_seq += 1;
        let seq = guard.next_seq;
        guard.bench_runs.insert(record.id, record.clone());
        guard.bench_run_seq.insert(record.id, seq);
        for item in items {
            guard.bench_item_results.push(BenchItemResultRecord {
                id: Uuid::new_v4(),
                bench_run_id: record.id,
                golden_item_id: item.golden_item_id,
                retrieved_doc_ids: item.retrieved_doc_ids.clone(),
                latency_ms: item.latency_ms,
                score: item.score,
                pii_leaked: item.pii_leaked,
                arn_violated: item.arn_violated,
                details: item.details.clone(),
            });
        }
        Ok(record)
    }

    async fn get_bench_run(
        &self,
        org_id: Uuid,
        bench_run_id: Uuid,
    ) -> Result<BenchRunRecord, OpenRagError> {
        let guard = self
            .state
            .lock()
            .map_err(|_| OpenRagError::Internal("InMemoryEvalStore lock poisoned".into()))?;
        guard
            .bench_runs
            .get(&bench_run_id)
            .filter(|r| r.org_id == org_id)
            .cloned()
            .ok_or_else(|| OpenRagError::NotFound(format!("bench run {bench_run_id} not found")))
    }

    async fn list_bench_runs(
        &self,
        org_id: Uuid,
        golden_set_id: Option<Uuid>,
    ) -> Result<Vec<BenchRunRecord>, OpenRagError> {
        let guard = self
            .state
            .lock()
            .map_err(|_| OpenRagError::Internal("InMemoryEvalStore lock poisoned".into()))?;
        let mut out: Vec<BenchRunRecord> = guard
            .bench_runs
            .values()
            .filter(|r| {
                r.org_id == org_id && golden_set_id.is_none_or(|gs| r.golden_set_id == gs)
            })
            .cloned()
            .collect();
        out.sort_by(|a, b| b.created_at.cmp(&a.created_at));
        Ok(out)
    }

    async fn list_bench_item_results(
        &self,
        org_id: Uuid,
        bench_run_id: Uuid,
    ) -> Result<Vec<BenchItemResultRecord>, OpenRagError> {
        let guard = self
            .state
            .lock()
            .map_err(|_| OpenRagError::Internal("InMemoryEvalStore lock poisoned".into()))?;
        if guard
            .bench_runs
            .get(&bench_run_id)
            .is_none_or(|r| r.org_id != org_id)
        {
            return Err(OpenRagError::NotFound(format!(
                "bench run {bench_run_id} not found"
            )));
        }
        Ok(guard
            .bench_item_results
            .iter()
            .filter(|ir| ir.bench_run_id == bench_run_id)
            .cloned()
            .collect())
    }

    async fn insert_optimization_trial(
        &self,
        trial: &NewOptimizationTrial,
    ) -> Result<OptimizationTrialRecord, OpenRagError> {
        let mut guard = self
            .state
            .lock()
            .map_err(|_| OpenRagError::Internal("InMemoryEvalStore lock poisoned".into()))?;
        let record = OptimizationTrialRecord {
            id: Uuid::new_v4(),
            org_id: trial.org_id,
            golden_set_id: trial.golden_set_id,
            search_space_yaml: trial.search_space_yaml.clone(),
            strategy: trial.strategy,
            best_config_yaml: None,
            best_score: None,
            security_gate_pass: false,
            status: OptimizationStatus::Running,
            created_at: Utc::now(),
            completed_at: None,
        };
        guard.trials.insert(record.id, record.clone());
        Ok(record)
    }

    async fn complete_optimization_trial(
        &self,
        org_id: Uuid,
        trial_id: Uuid,
        update: &CompleteOptimizationTrial,
    ) -> Result<OptimizationTrialRecord, OpenRagError> {
        let mut guard = self
            .state
            .lock()
            .map_err(|_| OpenRagError::Internal("InMemoryEvalStore lock poisoned".into()))?;
        let rec = guard
            .trials
            .get_mut(&trial_id)
            .filter(|t| t.org_id == org_id)
            .ok_or_else(|| {
                OpenRagError::NotFound(format!("optimization trial {trial_id} not found"))
            })?;
        rec.best_config_yaml = update.best_config_yaml.clone();
        rec.best_score = update.best_score;
        rec.security_gate_pass = update.security_gate_pass;
        rec.status = update.status;
        rec.completed_at = Some(Utc::now());
        Ok(rec.clone())
    }

    async fn list_optimization_trials(
        &self,
        org_id: Uuid,
        golden_set_id: Option<Uuid>,
    ) -> Result<Vec<OptimizationTrialRecord>, OpenRagError> {
        let guard = self
            .state
            .lock()
            .map_err(|_| OpenRagError::Internal("InMemoryEvalStore lock poisoned".into()))?;
        let mut out: Vec<OptimizationTrialRecord> = guard
            .trials
            .values()
            .filter(|t| {
                t.org_id == org_id && golden_set_id.is_none_or(|gs| t.golden_set_id == gs)
            })
            .cloned()
            .collect();
        out.sort_by(|a, b| b.created_at.cmp(&a.created_at));
        Ok(out)
    }

    async fn register_pipeline_config(
        &self,
        org_id: Uuid,
        input: &RegisterPipelineInput,
    ) -> Result<PipelineConfigRecord, OpenRagError> {
        if input.name.trim().is_empty() {
            return Err(OpenRagError::Validation(
                "pipeline config name must not be empty".into(),
            ));
        }
        let mut guard = self
            .state
            .lock()
            .map_err(|_| OpenRagError::Internal("InMemoryEvalStore lock poisoned".into()))?;
        let name = input.name.trim().to_string();
        if guard
            .pipeline_configs
            .values()
            .any(|p| p.org_id == org_id && p.name == name)
        {
            return Err(OpenRagError::Conflict(format!(
                "pipeline config '{name}' already exists"
            )));
        }
        let rec = PipelineConfigRecord {
            id: Uuid::new_v4(),
            org_id,
            name,
            profile: input.profile,
            pipeline_yaml: input.pipeline_yaml.clone(),
            config_json: input.config_json.clone(),
            is_promoted: false,
            promoted_at: None,
            created_by: input.created_by,
            created_at: Utc::now(),
        };
        guard.pipeline_configs.insert(rec.id, rec.clone());
        Ok(rec)
    }

    async fn get_pipeline_config(
        &self,
        org_id: Uuid,
        config_id: Uuid,
    ) -> Result<PipelineConfigRecord, OpenRagError> {
        let guard = self
            .state
            .lock()
            .map_err(|_| OpenRagError::Internal("InMemoryEvalStore lock poisoned".into()))?;
        guard
            .pipeline_configs
            .get(&config_id)
            .filter(|p| p.org_id == org_id)
            .cloned()
            .ok_or_else(|| {
                OpenRagError::NotFound(format!("pipeline config {config_id} not found"))
            })
    }

    async fn list_pipeline_configs(
        &self,
        org_id: Uuid,
    ) -> Result<Vec<PipelineConfigRecord>, OpenRagError> {
        let guard = self
            .state
            .lock()
            .map_err(|_| OpenRagError::Internal("InMemoryEvalStore lock poisoned".into()))?;
        let mut out: Vec<PipelineConfigRecord> = guard
            .pipeline_configs
            .values()
            .filter(|p| p.org_id == org_id)
            .cloned()
            .collect();
        out.sort_by(|a, b| b.created_at.cmp(&a.created_at));
        Ok(out)
    }

    async fn latest_bench_run_for_config(
        &self,
        org_id: Uuid,
        pipeline_config_id: Uuid,
    ) -> Result<Option<BenchRunRecord>, OpenRagError> {
        let guard = self
            .state
            .lock()
            .map_err(|_| OpenRagError::Internal("InMemoryEvalStore lock poisoned".into()))?;
        let latest = guard
            .bench_runs
            .values()
            .filter(|r| r.org_id == org_id && r.pipeline_config_id == Some(pipeline_config_id))
            .max_by_key(|r| {
                let seq = guard.bench_run_seq.get(&r.id).copied().unwrap_or(0);
                (r.created_at, seq, !r.security_gate_pass, r.id)
            })
            .cloned();
        Ok(latest)
    }

    async fn latest_bench_runs_by_track_for_config(
        &self,
        org_id: Uuid,
        pipeline_config_id: Uuid,
    ) -> Result<Vec<BenchRunRecord>, OpenRagError> {
        let guard = self
            .state
            .lock()
            .map_err(|_| OpenRagError::Internal("InMemoryEvalStore lock poisoned".into()))?;
        let mut out = Vec::new();
        for track in BenchTrack::ALL {
            if let Some(latest) = guard
                .bench_runs
                .values()
                .filter(|r| {
                    r.org_id == org_id
                        && r.pipeline_config_id == Some(pipeline_config_id)
                        && r.track == track
                })
                .max_by_key(|r| {
                    let seq = guard.bench_run_seq.get(&r.id).copied().unwrap_or(0);
                    (r.created_at, seq, !r.security_gate_pass, r.id)
                })
                .cloned()
            {
                out.push(latest);
            }
        }
        Ok(out)
    }

    async fn mark_pipeline_promoted(
        &self,
        org_id: Uuid,
        config_id: Uuid,
        bucket_ref: Option<&str>,
    ) -> Result<PipelineConfigRecord, OpenRagError> {
        let mut guard = self
            .state
            .lock()
            .map_err(|_| OpenRagError::Internal("InMemoryEvalStore lock poisoned".into()))?;
        let rec = guard
            .pipeline_configs
            .get_mut(&config_id)
            .filter(|p| p.org_id == org_id)
            .ok_or_else(|| {
                OpenRagError::NotFound(format!("pipeline config {config_id} not found"))
            })?;
        rec.is_promoted = true;
        rec.promoted_at = Some(Utc::now());
        let cloned = rec.clone();
        if let Some(bucket) = bucket_ref {
            let trimmed = bucket.trim();
            if !trimmed.is_empty() {
                guard
                    .bucket_bindings
                    .insert((org_id, trimmed.to_string()), config_id);
            }
        }
        Ok(cloned)
    }
}

#[derive(Clone)]
pub struct GoldenSetManager<S: EvalStore> {
    store: Arc<S>,
}

impl<S: EvalStore> GoldenSetManager<S> {
    pub fn new(store: Arc<S>) -> Self {
        Self { store }
    }

    pub fn store(&self) -> &Arc<S> {
        &self.store
    }

    pub async fn create_golden_set(
        &self,
        org_id: Uuid,
        input: CreateGoldenSetInput,
    ) -> Result<GoldenSet, OpenRagError> {
        self.store.create_golden_set(org_id, &input).await
    }

    pub async fn get_golden_set(&self, org_id: Uuid, id: Uuid) -> Result<GoldenSet, OpenRagError> {
        self.store.get_golden_set(org_id, id).await
    }

    pub async fn get_golden_set_by_name(
        &self,
        org_id: Uuid,
        name: &str,
    ) -> Result<GoldenSet, OpenRagError> {
        self.store.get_golden_set_by_name(org_id, name).await
    }

    pub async fn list_golden_sets(&self, org_id: Uuid) -> Result<Vec<GoldenSet>, OpenRagError> {
        self.store.list_golden_sets(org_id).await
    }

    pub async fn add_golden_item(
        &self,
        org_id: Uuid,
        golden_set_id: Uuid,
        input: CreateGoldenItemInput,
    ) -> Result<GoldenItem, OpenRagError> {
        self.store
            .add_golden_item(org_id, golden_set_id, &input)
            .await
    }

    pub async fn get_golden_item(
        &self,
        org_id: Uuid,
        item_id: Uuid,
    ) -> Result<GoldenItem, OpenRagError> {
        self.store.get_golden_item(org_id, item_id).await
    }

    pub async fn list_golden_items(
        &self,
        org_id: Uuid,
        golden_set_id: Uuid,
        include_rejected: bool,
    ) -> Result<Vec<GoldenItem>, OpenRagError> {
        self.store
            .list_golden_items(org_id, golden_set_id, include_rejected)
            .await
    }

    pub async fn update_review_status(
        &self,
        org_id: Uuid,
        item_id: Uuid,
        status: ReviewStatus,
    ) -> Result<GoldenItem, OpenRagError> {
        self.store
            .update_review_status(org_id, item_id, status)
            .await
    }
}

#[derive(Clone)]
pub struct FeedbackManager<S: EvalStore> {
    store: Arc<S>,
}

pub type FeedbackPromoter<S> = FeedbackManager<S>;

impl<S: EvalStore> FeedbackManager<S> {
    pub fn new(store: Arc<S>) -> Self {
        Self { store }
    }

    pub async fn submit_feedback_rls(
        &self,
        ctx: &PrincipalContext,
        input: SubmitFeedbackInput,
    ) -> Result<SearchFeedback, OpenRagError> {
        self.store.submit_feedback_rls(ctx, &input).await
    }

    pub async fn list_feedback_rls(
        &self,
        ctx: &PrincipalContext,
    ) -> Result<Vec<SearchFeedback>, OpenRagError> {
        self.store.list_feedback_rls(ctx).await
    }

    pub async fn promote_feedback(
        &self,
        org_id: Uuid,
        actor: &PrincipalContext,
        feedback_id: Uuid,
        input: PromoteFeedbackInput,
    ) -> Result<PromoteFeedbackResult, OpenRagError> {
        self.store
            .promote_feedback(org_id, actor, feedback_id, &input)
            .await
    }
}
