use std::collections::BTreeMap;
use std::str::FromStr;
use std::sync::Arc;

use axum::extract::{Path, Query, State};
use axum::http::StatusCode;
use axum::routing::{delete, get, patch, post, put};
use axum::{Json, Router};
use chrono::{DateTime, Utc};
use openrag_core::{
    AccessScope, Arn, ArnResourceType, AuthLevel, GuardAction, GuardRuleConfig, GuardRuleKind,
    GuardSurface, IamAction, MemoryConfig, OpenRagError, OptimizationStrategy, PgPineconeEngine,
    PipelineConfig, PipelineProfile, PrincipalContext, Sensitivity,
};
use openrag_db::DbPool;
use openrag_eval::{
    parse_pipeline_yaml_with_profile, AutoRagOptimizer, BenchRunRequest, BenchRunner,
    CreateGoldenItemInput, CreateGoldenSetInput, FeedbackManager, GoldenSetManager,
    OptimizeRequest, PipelineProfileKind, PipelinePromoter, PromoteFeedbackInput,
    PromotePipelineInput, ReviewStatus,
};
use openrag_iam::{
    GroupDefinition, GroupMembership, IamEvaluator, IamPolicy, PolicyEffect, PolicyStatement,
    ResourceDescriptor, ResourceGroupBinding,
};
use openrag_parse::{resolve_ocr_policy, OcrTask, ParsePolicyCtx, ThumbnailBlobStore};
use openrag_pii::{
    CascadeRedactor, DetectorProfileInput, GuardPolicyRuntime, GuardRepo, GuardRuleRuntime,
    LocalThumbnailStore, PiiEngine, PiiProfileRepo, RedactionJobRequest, RedactionMode,
    RedactionTargetScope, RedactionTextEmbedder, ThumbnailStore as PiiThumbnailStore,
};
use openrag_retrieve::SearchRequest;
use openrag_store::{
    DefaultOutboxExecutor, OutboxWorker, PgPineconeManager, StoreMode, VectorTable,
};
use regex::Regex;
use ring::rand::{SecureRandom, SystemRandom};
use serde::Deserialize;
use serde_json::{json, Value};
use sqlx::Row;
use uuid::Uuid;

use crate::api_core::{
    create_concept_handler, db_col_err, delete_concept_alias_handler, delete_concept_handler,
    get_concept_handler, link_concepts_handler, list_concepts_handler, map_sqlx_err,
    resolve_concepts_handler, CreateConceptRequest, LinkConceptsRequest,
};
use crate::auth::{issue_cli_code, AuthPrincipal, CliCodeCaps};
use crate::error::ApiError;
use crate::llm_router::{ensure_default_providers, list_providers};
use crate::state::{
    default_pipeline_config, ensure_default_guard_policy, load_active_guard_policy,
    load_concept_lookup, non_empty_env, AppState, OpenRagMode,
};

pub fn studio_routes() -> Router<Arc<AppState>> {
    Router::new()
        .route(
            "/api/studio/pipeline",
            get(studio_pipeline_get_handler).put(studio_pipeline_update_handler),
        )
        .route("/api/studio/pipeline/test", post(studio_pipeline_test_handler))
        .route(
            "/api/studio/index/engines",
            get(studio_index_engines_get_handler).put(studio_index_engines_set_handler),
        )
        .route("/api/studio/index/outbox", get(studio_index_outbox_handler))
        .route(
            "/api/studio/index/quarantine/{id}/retry",
            post(studio_quarantine_retry_handler),
        )
        .route("/api/studio/iam/orgs", post(studio_iam_create_org_handler))
        .route(
            "/api/studio/iam/principals",
            post(studio_iam_create_principal_handler).get(crate::api_members::list_principals_handler),
        )
        .route(
            "/api/studio/iam/api-keys",
            post(studio_iam_create_api_key_handler).get(crate::api_members::list_keys_handler),
        )
        .route("/api/studio/iam/cli-codes", post(studio_iam_create_cli_code_handler))
        .route(
            "/api/studio/iam/groups",
            get(studio_iam_list_groups_handler).post(studio_iam_create_group_handler),
        )
        .route(
            "/api/studio/iam/groups/{id}/members",
            post(studio_iam_add_group_member_handler),
        )
        .route(
            "/api/studio/iam/groups/{id}/members/{principal_id}",
            delete(studio_iam_remove_group_member_handler),
        )
        .route(
            "/api/studio/iam/resource-bindings",
            post(studio_iam_create_resource_binding_handler),
        )
        .route("/api/studio/iam/policies", post(studio_iam_create_policy_handler))
        .route("/api/studio/iam/simulate", post(studio_iam_simulate_handler))
        .route(
            "/api/studio/pii/profiles",
            get(studio_pii_profiles_get_handler).put(studio_pii_profiles_update_handler),
        )
        .route(
            "/api/studio/pii/guard-rules",
            get(studio_pii_guard_rules_list_handler).post(studio_pii_guard_rules_create_handler),
        )
        .route(
            "/api/studio/pii/guard-rules/{id}",
            put(studio_pii_guard_rules_update_handler)
                .delete(studio_pii_guard_rules_delete_handler),
        )
        .route(
            "/api/studio/pii/guard-policies",
            get(studio_pii_guard_policies_list_handler)
                .put(studio_pii_guard_policies_update_default_handler),
        )
        .route(
            "/api/studio/pii/guard-policy",
            get(studio_pii_guard_policies_list_handler)
                .put(studio_pii_guard_policies_update_default_handler),
        )
        .route(
            "/api/studio/pii/guard-policies/{id}",
            put(studio_pii_guard_policies_update_by_id_handler),
        )
        .route(
            "/api/studio/pii/guard-events",
            get(studio_pii_guard_events_handler),
        )
        .route("/api/studio/pii/guard-test", post(studio_pii_guard_test_handler))
        .route(
            "/api/studio/pii/redactions",
            get(studio_pii_redactions_list_handler).post(studio_pii_redactions_create_handler),
        )
        .route(
            "/api/studio/pii/audit-logs",
            get(studio_pii_audit_logs_handler),
        )
        .route(
            "/api/studio/gateway/providers",
            get(studio_gateway_providers_handler),
        )
        .route(
            "/api/studio/gateway/routing-rules",
            post(studio_gateway_routing_rules_create_handler),
        )
        .route(
            "/api/studio/gateway/logs",
            get(studio_gateway_logs_handler),
        )
        .route(
            "/api/studio/eval/golden-sets",
            get(studio_eval_golden_sets_list_handler)
                .post(studio_eval_golden_sets_create_handler),
        )
        .route(
            "/api/studio/eval/golden-sets/{id}/items",
            get(studio_eval_golden_items_list_handler)
                .post(studio_eval_golden_items_create_handler),
        )
        .route(
            "/api/studio/eval/golden-items/{id}",
            patch(studio_eval_golden_item_patch_handler),
        )
        .route(
            "/api/studio/eval/feedback/{id}/promote",
            post(studio_eval_feedback_promote_handler),
        )
        .route("/api/studio/eval/bench", post(studio_eval_bench_handler))
        .route("/api/studio/eval/optimize", post(studio_eval_optimize_handler))
        .route(
            "/api/studio/eval/leaderboard",
            get(studio_eval_leaderboard_handler),
        )
        .route("/api/studio/eval/report", get(studio_eval_report_handler))
        .route("/api/studio/eval/promote", post(studio_eval_promote_handler))
        .route(
            "/api/studio/wiki/conflicts",
            get(studio_wiki_conflicts_handler),
        )
        .route(
            "/api/studio/plugins/connectors",
            get(studio_plugins_connectors_handler),
        )
        .route(
            "/api/studio/concepts",
            get(list_concepts_handler).post(studio_concepts_create_handler),
        )
        .route(
            "/api/studio/concepts/link",
            post(studio_concepts_link_handler),
        )
        .route(
            "/api/studio/concepts/resolve",
            post(resolve_concepts_handler),
        )
        .route(
            "/api/studio/concepts/{id}",
            get(get_concept_handler).delete(studio_concepts_delete_handler),
        )
        .route(
            "/api/studio/concepts/{id}/aliases/{alias_id}",
            delete(studio_concepts_delete_alias_handler),
        )
        .route(
            "/api/studio/knowledge/tags",
            get(studio_knowledge_tags_handler),
        )
}

#[derive(Debug, Deserialize)]
struct CreateCliCodeRequest {
    principal_id: Uuid,
    ttl_seconds: Option<i64>,
    max_auth_level: Option<i16>,
    #[serde(default)]
    scoped_arns: Vec<String>,
    #[serde(default)]
    scoped_group_ids: Vec<Uuid>,
}

/// Org admins issue one-time CLI codes for principals of their own org. The code can never
/// be broader than the issuing key: its level is capped at the issuer's effective level and
/// an ARN-scoped issuer can only hand out codes inside that scope.
async fn studio_iam_create_cli_code_handler(
    State(state): State<Arc<AppState>>,
    AuthPrincipal(principal): AuthPrincipal,
    Json(req): Json<CreateCliCodeRequest>,
) -> Result<(StatusCode, Json<Value>), ApiError> {
    require_org_admin(&principal)?;
    let worker_pool = state.require_worker_pool().map_err(ApiError::from)?;

    let issuer_level = principal.auth_level.as_i16();
    let max_auth_level = Some(req.max_auth_level.unwrap_or(issuer_level).min(issuer_level));
    let scoped_arns = if principal.scoped_arns.is_empty() {
        req.scoped_arns
    } else if req.scoped_arns.is_empty() {
        principal.scoped_arns.clone()
    } else if req.scoped_arns.iter().all(|a| principal.scoped_arns.contains(a)) {
        req.scoped_arns
    } else {
        return Err(ApiError::forbidden(
            "scope_exceeds_issuer",
            "scoped_arns must stay within the issuing key's scope",
        ));
    };
    let scoped_group_ids = if principal.scoped_group_ids.is_empty() {
        req.scoped_group_ids
    } else if req.scoped_group_ids.is_empty() {
        principal.scoped_group_ids.clone()
    } else if req
        .scoped_group_ids
        .iter()
        .all(|g| principal.scoped_group_ids.contains(g))
    {
        req.scoped_group_ids
    } else {
        return Err(ApiError::forbidden(
            "scope_exceeds_issuer",
            "scoped_group_ids must stay within the issuing key's scope",
        ));
    };

    let issued = issue_cli_code(
        worker_pool,
        principal.org_id,
        req.principal_id,
        Some(principal.principal_id),
        CliCodeCaps {
            max_auth_level,
            scoped_arns,
            scoped_group_ids,
        },
        req.ttl_seconds,
    )
    .await
    .map_err(ApiError::from)?;
    let body = serde_json::to_value(&issued)
        .map_err(|e| ApiError::from(OpenRagError::Internal(format!("serialize cli code: {e}"))))?;
    Ok((StatusCode::CREATED, Json(body)))
}

fn require_org_admin(principal: &PrincipalContext) -> Result<(), ApiError> {
    if !principal.is_org_admin {
        return Err(ApiError::forbidden(
            "admin_required",
            "org_admin role required for studio mutation",
        ));
    }
    Ok(())
}

// ============================================================================
// Thumbnail & Embedder Adapters for CascadeRedactor
// ============================================================================

pub struct BlobThumbnailStoreAdapter(pub Arc<dyn ThumbnailBlobStore>);

/// URIs only the configured blob store can serve; anything else is a local file path.
pub fn is_store_uri(uri: &str) -> bool {
    uri.starts_with("mem://") || uri.starts_with("s3://")
}

#[async_trait::async_trait]
impl PiiThumbnailStore for BlobThumbnailStoreAdapter {
    async fn read_bytes(&self, uri: &str) -> Result<Vec<u8>, OpenRagError> {
        match self.0.get_bytes(uri).await {
            Ok(bytes) => Ok(bytes),
            Err(err) if is_store_uri(uri) => Err(err),
            Err(_) => LocalThumbnailStore::new().read_bytes(uri).await,
        }
    }

    async fn write_bytes(&self, uri: &str, bytes: &[u8]) -> Result<(), OpenRagError> {
        if is_store_uri(uri) {
            self.0.put_uri(uri, bytes).await
        } else {
            LocalThumbnailStore::new().write_bytes(uri, bytes).await
        }
    }
}

/// The server's text embedder, handed to the redactor so scrubbed rows are re-embedded.
pub struct RedactionEmbedderAdapter(pub Arc<dyn openrag_memory::TextEmbedder>);

#[async_trait::async_trait]
impl RedactionTextEmbedder for RedactionEmbedderAdapter {
    async fn embed_text(&self, text: &str) -> Result<Vec<f32>, OpenRagError> {
        self.0.embed_text(text).await
    }
}

// ============================================================================
// A. Pipeline Console (/api/studio/pipeline*)
// ============================================================================

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct PartialMemoryPipelineDoc {
    version: Option<u32>,
    profile: Option<PipelineProfileKind>,
    memory: Option<MemoryConfig>,
}

pub fn parse_studio_pipeline_yaml(yaml: &str) -> Result<(PipelineConfig, PipelineProfileKind), OpenRagError> {
    match parse_pipeline_yaml_with_profile(yaml) {
        Ok(res) => Ok(res),
        Err(orig_err) => {
            if let Ok(doc) = serde_yaml_ng::from_str::<PartialMemoryPipelineDoc>(yaml) {
                if let Some(mem) = doc.memory {
                    let mut cfg = default_pipeline_config();
                    if let Some(v) = doc.version {
                        cfg.version = v;
                    }
                    let profile_kind = doc.profile.unwrap_or(PipelineProfileKind::Standard);
                    cfg.profile = match profile_kind {
                        PipelineProfileKind::Airgap => PipelineProfile::Airgap,
                        PipelineProfileKind::Standard | PipelineProfileKind::Custom => {
                            PipelineProfile::Standard
                        }
                    };
                    cfg.memory = mem;
                    cfg.validate()?;
                    return Ok((cfg, profile_kind));
                }
            }
            Err(orig_err)
        }
    }
}

async fn studio_pipeline_get_handler(
    State(state): State<Arc<AppState>>,
    AuthPrincipal(_principal): AuthPrincipal,
) -> Result<Json<Value>, ApiError> {
    let cfg = state.pipeline_config.read().await.clone();
    let yaml_str = serde_yaml_ng::to_string(&cfg)
        .map_err(|e| ApiError::from(OpenRagError::Internal(format!("serialize pipeline YAML: {e}"))))?;

    Ok(Json(json!({
        "version": cfg.version,
        "profile": cfg.profile,
        "pipeline_yaml": yaml_str,
        "config": cfg,
        "memory": cfg.memory,
    })))
}

#[derive(Debug, Deserialize)]
struct UpdatePipelineRequest {
    #[serde(alias = "yaml")]
    pipeline_yaml: String,
    name: Option<String>,
}

async fn studio_pipeline_update_handler(
    State(state): State<Arc<AppState>>,
    AuthPrincipal(principal): AuthPrincipal,
    Json(req): Json<UpdatePipelineRequest>,
) -> Result<Json<Value>, ApiError> {
    require_org_admin(&principal)?;

    let (parsed, profile_kind) =
        parse_studio_pipeline_yaml(&req.pipeline_yaml).map_err(ApiError::from)?;
    parsed.validate().map_err(ApiError::from)?;

    let config_json = serde_json::to_value(&parsed)
        .map_err(|e| ApiError::from(OpenRagError::Internal(format!("serialize pipeline JSON: {e}"))))?;
    let name = req
        .name
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .unwrap_or("active")
        .to_string();

    let saved_id: Option<Uuid> = if let Some(worker_pool) = state.worker_pool.as_ref() {
        let id: Uuid = sqlx::query_scalar(
            "INSERT INTO eval.pipeline_configs (
                org_id, name, profile, pipeline_yaml, config_json, created_by
             ) VALUES ($1, $2, $3, $4, $5, $6)
             ON CONFLICT (org_id, name) DO UPDATE SET
                profile = EXCLUDED.profile,
                pipeline_yaml = EXCLUDED.pipeline_yaml,
                config_json = EXCLUDED.config_json
             RETURNING id",
        )
        .bind(principal.org_id)
        .bind(&name)
        .bind(profile_kind.as_str())
        .bind(&req.pipeline_yaml)
        .bind(&config_json)
        .bind(principal.principal_id)
        .fetch_one(worker_pool.inner())
        .await
        .map_err(map_sqlx_err("upsert eval.pipeline_configs"))?;
        Some(id)
    } else {
        None
    };

    *state.pipeline_config.write().await = parsed.clone();
    *state.pii_engine.write().await = PiiEngine::from_config(&parsed.pii);

    Ok(Json(json!({
        "id": saved_id,
        "pipeline_config_id": saved_id,
        "name": name,
        "profile": profile_kind,
        "pipeline_yaml": req.pipeline_yaml,
        "config": parsed,
        "memory": parsed.memory,
    })))
}

#[derive(Debug, Deserialize)]
struct TestPipelineRequest {
    pipeline_yaml: Option<String>,
    query: Option<String>,
    dewarp: Option<bool>,
    auth_level: Option<i16>,
    #[serde(default, alias = "bucket")]
    bucket_id: Option<Uuid>,
    top_k: Option<usize>,
}

async fn studio_pipeline_test_handler(
    State(state): State<Arc<AppState>>,
    AuthPrincipal(principal): AuthPrincipal,
    Json(req): Json<TestPipelineRequest>,
) -> Result<Json<Value>, ApiError> {
    let cfg = match req.pipeline_yaml.as_deref() {
        Some(yaml) => {
            let (parsed, _) = parse_studio_pipeline_yaml(yaml).map_err(ApiError::from)?;
            parsed.validate().map_err(ApiError::from)?;
            parsed
        }
        None => state.pipeline_config.read().await.clone(),
    };

    let auth_level = match req.auth_level {
        Some(lvl) => AuthLevel::try_from(lvl).map_err(ApiError::from)?,
        None => principal.auth_level,
    };

    let ocr_decision = resolve_ocr_policy(
        &cfg.parse,
        &ParsePolicyCtx {
            auth_level,
            sensitivity: Sensitivity::Normal,
            requested_ocr_mode: None,
            batch: false,
            ocr_task: OcrTask::Markdown,
            tenant_key: Some(principal.org_id.to_string()),
        },
    )
    .map_err(ApiError::from)?;

    let mut decomposed_hits = Vec::new();
    if let Some(query_str) = req.query.as_deref().map(str::trim).filter(|s| !s.is_empty()) {
        let mut s_req = SearchRequest::new(query_str);
        s_req.bucket_id = req.bucket_id;
        let retrieve = crate::state::retrieval_pipeline(cfg.clone(), state.has_text_model).retrieve;
        s_req.top_k = req.top_k.or(Some(retrieve.top_k));
        s_req.fusion = Some(retrieve.fusion);
        s_req.alpha = Some(retrieve.alpha);
        s_req.rrf_k = Some(retrieve.rrf_k);
        s_req.rerank = Some(retrieve.rerank);

        let resp = if state.app_pool.is_some() {
            let retriever = state
                .pg_retriever(principal.org_id)
                .await
                .map_err(ApiError::from)?;
            retriever
                .search(&principal, &s_req)
                .await
                .map_err(ApiError::from)?
        } else {
            state
                .in_memory_retrieve
                .search(&principal, &s_req)
                .await
                .map_err(ApiError::from)?
        };

        for hit in resp.hits {
            decomposed_hits.push(json!({
                "id": hit.id,
                "document_id": hit.document_id,
                "space": hit.space,
                "title": hit.title,
                "vector": hit.vector_score,
                "bm25": hit.bm25_score,
                "score": hit.score_breakdown.fused_score,
                "vector_score": hit.vector_score,
                "bm25_score": hit.bm25_score,
                "fused_score": hit.score_breakdown.fused_score,
            }));
        }
    }

    let first_vec = decomposed_hits
        .first()
        .and_then(|h| h.get("vector"))
        .and_then(Value::as_f64)
        .unwrap_or(0.0);
    let first_bm25 = decomposed_hits
        .first()
        .and_then(|h| h.get("bm25"))
        .and_then(Value::as_f64)
        .unwrap_or(0.0);
    let first_score = decomposed_hits
        .first()
        .and_then(|h| h.get("score"))
        .and_then(Value::as_f64)
        .unwrap_or(0.0);

    Ok(Json(json!({
        "valid": true,
        "ocr_mode": ocr_decision.effective_mode.as_str(),
        "mode": ocr_decision.effective_mode.as_str(),
        "execution_target": ocr_decision.execution_target,
        "dewarp": req.dewarp.unwrap_or(cfg.parse.dewarp_enabled),
        "auth_level": auth_level.as_u8(),
        "memory": cfg.memory,
        "vector": first_vec,
        "bm25": first_bm25,
        "score": first_score,
        "score_decomposition": {
            "fusion": cfg.retrieve.fusion,
            "alpha": cfg.retrieve.alpha,
            "rrf_k": cfg.retrieve.rrf_k,
            "vector": first_vec,
            "bm25": first_bm25,
            "score": first_score,
        },
        "hits": decomposed_hits,
    })))
}

// ============================================================================
// B. Index / Outbox / Quarantine Console (/api/studio/index/*)
// ============================================================================

async fn studio_index_engines_get_handler(
    State(state): State<Arc<AppState>>,
    AuthPrincipal(principal): AuthPrincipal,
) -> Result<Json<Value>, ApiError> {
    let worker_pool = state.require_worker_pool().map_err(ApiError::from)?;
    let caps = worker_pool.capabilities().await.map_err(ApiError::from)?;
    let mut bindings = PgPineconeManager::list_bindings(worker_pool, principal.org_id)
        .await
        .map_err(ApiError::from)?;

    if bindings.is_empty() {
        let default_engine = state.pipeline_config.read().await.store.pg_pinecone_engine;
        let now = Utc::now();
        for table in VectorTable::ALL {
            let Some(vector_column) = table.vector_column() else {
                continue;
            };
            bindings.push(openrag_store::IndexBinding {
                id: Uuid::nil(),
                org_id: principal.org_id,
                table_name: table.as_str().to_string(),
                vector_column: vector_column.to_string(),
                pg_pinecone_engine: default_engine,
                external_backend: openrag_core::ExternalMirrorBackend::None,
                external_config: json!({}),
                dimension: openrag_store::VECTOR_DIM as i16,
                filter_column: Some("acl_scope_key".to_string()),
                status: "ready".to_string(),
                updated_at: now,
            });
        }
    }

    Ok(Json(json!({
        "engines": bindings,
        "bindings": bindings,
        "items": bindings,
        "capabilities": caps,
    })))
}

#[derive(Debug, Deserialize)]
struct SetIndexEngineRequest {
    #[serde(alias = "table")]
    table_name: String,
    engine: PgPineconeEngine,
}

async fn studio_index_engines_set_handler(
    State(state): State<Arc<AppState>>,
    AuthPrincipal(principal): AuthPrincipal,
    Json(req): Json<SetIndexEngineRequest>,
) -> Result<Json<Value>, ApiError> {
    require_org_admin(&principal)?;

    let table = VectorTable::from_str(&req.table_name).map_err(ApiError::from)?;
    let worker_pool = state.require_worker_pool().map_err(ApiError::from)?;
    let pinecone_key = non_empty_env("PINECONE_API_KEY");
    let store_mode = if state.mode == OpenRagMode::Live {
        StoreMode::Live
    } else {
        StoreMode::Offline
    };

    let binding = PgPineconeManager::set_engine(
        worker_pool,
        principal.org_id,
        table,
        req.engine,
        store_mode,
        pinecone_key.as_deref(),
    )
    .await
    .map_err(ApiError::from)?;

    if table == VectorTable::CorpusChunks {
        state.pipeline_config.write().await.store.pg_pinecone_engine = req.engine;
    }

    let mode_label = if req.engine == PgPineconeEngine::Remote && pinecone_key.is_none() {
        "fallback"
    } else {
        match req.engine {
            PgPineconeEngine::Native => "native",
            PgPineconeEngine::Pgvector => "pgvector",
            PgPineconeEngine::Remote => "remote",
        }
    };

    Ok(Json(json!({
        "binding": binding,
        "table_name": binding.table_name,
        "engine": binding.pg_pinecone_engine,
        "mode": mode_label,
    })))
}

async fn studio_index_outbox_handler(
    State(state): State<Arc<AppState>>,
    AuthPrincipal(principal): AuthPrincipal,
) -> Result<Json<Value>, ApiError> {
    require_org_admin(&principal)?;
    let worker_pool = state.require_worker_pool().map_err(ApiError::from)?;
    let summary = OutboxWorker::default()
        .status(worker_pool, Some(principal.org_id))
        .await
        .map_err(ApiError::from)?;

    let outbox_rows = sqlx::query(
        "SELECT id, org_id, target_table, target_id, op, status, attempts, max_attempts, last_error, created_at, processed_at
         FROM indexing.outbox
         WHERE org_id = $1
         ORDER BY created_at DESC, id DESC
         LIMIT 100",
    )
    .bind(principal.org_id)
    .fetch_all(worker_pool.inner())
    .await
    .map_err(|e| ApiError::from(OpenRagError::Database(format!("list indexing.outbox: {e}"))))?;

    let mut outbox = Vec::with_capacity(outbox_rows.len());
    for r in outbox_rows {
        outbox.push(json!({
            "id": r.try_get::<i64, _>("id").map_err(db_col_err)?,
            "org_id": r.try_get::<Uuid, _>("org_id").map_err(db_col_err)?,
            "target_table": r.try_get::<String, _>("target_table").map_err(db_col_err)?,
            "target_id": r.try_get::<Uuid, _>("target_id").map_err(db_col_err)?,
            "op": r.try_get::<String, _>("op").map_err(db_col_err)?,
            "status": r.try_get::<String, _>("status").map_err(db_col_err)?,
            "attempts": r.try_get::<i32, _>("attempts").map_err(db_col_err)?,
            "max_attempts": r.try_get::<i32, _>("max_attempts").map_err(db_col_err)?,
            "last_error": r.try_get::<Option<String>, _>("last_error").map_err(db_col_err)?,
            "created_at": r.try_get::<DateTime<Utc>, _>("created_at").map_err(db_col_err)?,
            "processed_at": r.try_get::<Option<DateTime<Utc>>, _>("processed_at").map_err(db_col_err)?,
        }));
    }

    let q_rows = sqlx::query(
        "SELECT id, outbox_id, org_id, target_table, target_id, op, error_class, error_detail, resolved_at, created_at
         FROM indexing.quarantine
         WHERE org_id = $1
         ORDER BY created_at DESC
         LIMIT 100",
    )
    .bind(principal.org_id)
    .fetch_all(worker_pool.inner())
    .await
    .map_err(|e| ApiError::from(OpenRagError::Database(format!("list indexing.quarantine: {e}"))))?;

    let mut quarantine = Vec::with_capacity(q_rows.len());
    for r in q_rows {
        quarantine.push(json!({
            "id": r.try_get::<Uuid, _>("id").map_err(db_col_err)?,
            "outbox_id": r.try_get::<Option<i64>, _>("outbox_id").map_err(db_col_err)?,
            "org_id": r.try_get::<Uuid, _>("org_id").map_err(db_col_err)?,
            "target_table": r.try_get::<String, _>("target_table").map_err(db_col_err)?,
            "target_id": r.try_get::<Uuid, _>("target_id").map_err(db_col_err)?,
            "op": r.try_get::<String, _>("op").map_err(db_col_err)?,
            "error_class": r.try_get::<String, _>("error_class").map_err(db_col_err)?,
            "error_detail": r.try_get::<String, _>("error_detail").map_err(db_col_err)?,
            "resolved_at": r.try_get::<Option<DateTime<Utc>>, _>("resolved_at").map_err(db_col_err)?,
            "created_at": r.try_get::<DateTime<Utc>, _>("created_at").map_err(db_col_err)?,
        }));
    }

    Ok(Json(json!({
        "summary": summary,
        "outbox": outbox,
        "quarantine": quarantine,
        "watermarks": summary.watermarks,
    })))
}

async fn studio_quarantine_retry_handler(
    State(state): State<Arc<AppState>>,
    AuthPrincipal(principal): AuthPrincipal,
    Path(id): Path<Uuid>,
) -> Result<Json<Value>, ApiError> {
    require_org_admin(&principal)?;

    let worker_pool = state.require_worker_pool().map_err(ApiError::from)?;
    let retried = OutboxWorker::default()
        .retry_quarantine(worker_pool, Some(id), false, Some(principal.org_id))
        .await
        .map_err(ApiError::from)?;

    let drained = OutboxWorker::default()
        .drain_for_org(
            worker_pool,
            Some(principal.org_id),
            64,
            &DefaultOutboxExecutor::default(),
        )
        .await
        .map_err(ApiError::from)?;

    Ok(Json(json!({
        "retried": retried,
        "quarantine_id": id,
        "drained": drained,
    })))
}

// ============================================================================
// C. IAM & ARN Simulator Console (/api/studio/iam/*)
// ============================================================================

#[derive(Debug, Deserialize)]
struct CreateOrgRequest {
    id: Option<Uuid>,
    slug: String,
    name: String,
}

async fn studio_iam_create_org_handler(
    State(state): State<Arc<AppState>>,
    AuthPrincipal(principal): AuthPrincipal,
    Json(req): Json<CreateOrgRequest>,
) -> Result<(StatusCode, Json<Value>), ApiError> {
    require_org_admin(&principal)?;

    let slug = req.slug.trim();
    let name = req.name.trim();
    if slug.is_empty() || name.is_empty() {
        return Err(ApiError::bad_request(
            "invalid_org",
            "slug and name must not be empty",
        ));
    }

    let worker_pool = state.require_worker_pool().map_err(ApiError::from)?;
    let org_id = req.id.unwrap_or_else(Uuid::now_v7);

    let row = sqlx::query(
        "INSERT INTO iam.orgs (id, slug, name)
         VALUES ($1, $2, $3)
         RETURNING id, slug, name, created_at",
    )
    .bind(org_id)
    .bind(slug)
    .bind(name)
    .fetch_one(worker_pool.inner())
    .await
    .map_err(map_sqlx_err("insert iam.orgs"))?;

    let id: Uuid = row.try_get("id").map_err(db_col_err)?;
    Ok((
        StatusCode::CREATED,
        Json(json!({
            "id": id,
            "org_id": id,
            "slug": row.try_get::<String, _>("slug").map_err(db_col_err)?,
            "name": row.try_get::<String, _>("name").map_err(db_col_err)?,
            "created_at": row.try_get::<DateTime<Utc>, _>("created_at").map_err(db_col_err)?,
        })),
    ))
}

#[derive(Debug, Deserialize)]
struct CreatePrincipalRequest {
    id: Option<Uuid>,
    org_id: Option<Uuid>,
    email: String,
    display_name: String,
    auth_level: Option<i16>,
    #[serde(default)]
    can_unmask_pii: bool,
    #[serde(default)]
    is_org_admin: bool,
    #[serde(default)]
    dept_ids: Vec<Uuid>,
    #[serde(default)]
    dept_admin_ids: Vec<Uuid>,
    #[serde(default)]
    group_ids: Vec<Uuid>,
}

async fn studio_iam_create_principal_handler(
    State(state): State<Arc<AppState>>,
    AuthPrincipal(principal): AuthPrincipal,
    Json(req): Json<CreatePrincipalRequest>,
) -> Result<(StatusCode, Json<Value>), ApiError> {
    require_org_admin(&principal)?;

    let email = req.email.trim();
    let display_name = req.display_name.trim();
    if email.is_empty() || display_name.is_empty() {
        return Err(ApiError::bad_request(
            "invalid_principal",
            "email and display_name must not be empty",
        ));
    }

    let level = AuthLevel::try_from(req.auth_level.unwrap_or(3)).map_err(ApiError::from)?;
    let org_id = req.org_id.unwrap_or(principal.org_id);
    if org_id != principal.org_id {
        return Err(ApiError::forbidden(
            "forbidden",
            "cannot manage principals or API keys in another organization",
        ));
    }
    let pid = req.id.unwrap_or_else(Uuid::now_v7);

    let worker_pool = state.require_worker_pool().map_err(ApiError::from)?;
    let mut tx = worker_pool
        .inner()
        .begin()
        .await
        .map_err(|e| ApiError::from(OpenRagError::Database(format!("begin principal tx: {e}"))))?;

    let row = sqlx::query(
        "INSERT INTO iam.principals (
            id, org_id, email, display_name, auth_level, can_unmask_pii, is_org_admin
         ) VALUES ($1, $2, $3, $4, $5, $6, $7)
         RETURNING id, org_id, email, display_name, auth_level, can_unmask_pii, is_org_admin, status, created_at",
    )
    .bind(pid)
    .bind(org_id)
    .bind(email)
    .bind(display_name)
    .bind(level.as_i16())
    .bind(req.can_unmask_pii)
    .bind(req.is_org_admin)
    .fetch_one(&mut *tx)
    .await
    .map_err(map_sqlx_err("insert iam.principals"))?;

    for did in &req.dept_ids {
        sqlx::query(
            "INSERT INTO iam.dept_memberships (org_id, dept_id, principal_id, dept_role)
             VALUES ($1, $2, $3, 'member')
             ON CONFLICT (dept_id, principal_id) DO NOTHING",
        )
        .bind(org_id)
        .bind(did)
        .bind(pid)
        .execute(&mut *tx)
        .await
        .map_err(map_sqlx_err("insert iam.dept_memberships"))?;
    }

    for did in &req.dept_admin_ids {
        sqlx::query(
            "INSERT INTO iam.dept_memberships (org_id, dept_id, principal_id, dept_role)
             VALUES ($1, $2, $3, 'dept_admin')
             ON CONFLICT (dept_id, principal_id) DO UPDATE SET dept_role = 'dept_admin'",
        )
        .bind(org_id)
        .bind(did)
        .bind(pid)
        .execute(&mut *tx)
        .await
        .map_err(map_sqlx_err("insert dept_admin membership"))?;
    }

    for gid in &req.group_ids {
        sqlx::query(
            "INSERT INTO iam.group_members (org_id, group_id, principal_id)
             VALUES ($1, $2, $3)
             ON CONFLICT (group_id, principal_id) DO NOTHING",
        )
        .bind(org_id)
        .bind(gid)
        .bind(pid)
        .execute(&mut *tx)
        .await
        .map_err(map_sqlx_err("insert iam.group_members"))?;
    }

    tx.commit()
        .await
        .map_err(|e| ApiError::from(OpenRagError::Database(format!("commit principal tx: {e}"))))?;

    let created_id: Uuid = row.try_get("id").map_err(db_col_err)?;
    Ok((
        StatusCode::CREATED,
        Json(json!({
            "id": created_id,
            "principal_id": created_id,
            "org_id": row.try_get::<Uuid, _>("org_id").map_err(db_col_err)?,
            "email": row.try_get::<String, _>("email").map_err(db_col_err)?,
            "display_name": row.try_get::<String, _>("display_name").map_err(db_col_err)?,
            "auth_level": row.try_get::<i16, _>("auth_level").map_err(db_col_err)?,
            "can_unmask_pii": row.try_get::<bool, _>("can_unmask_pii").map_err(db_col_err)?,
            "is_org_admin": row.try_get::<bool, _>("is_org_admin").map_err(db_col_err)?,
            "status": row.try_get::<String, _>("status").map_err(db_col_err)?,
            "created_at": row.try_get::<DateTime<Utc>, _>("created_at").map_err(db_col_err)?,
        })),
    ))
}

#[derive(Debug, Deserialize)]
struct CreateApiKeyRequest {
    org_id: Option<Uuid>,
    principal_id: Uuid,
    name: Option<String>,
    max_auth_level: Option<i16>,
    #[serde(default)]
    scoped_group_ids: Vec<Uuid>,
    #[serde(default)]
    scoped_arns: Vec<String>,
    #[serde(default)]
    allowed_actions: Vec<String>,
}

async fn studio_iam_create_api_key_handler(
    State(state): State<Arc<AppState>>,
    AuthPrincipal(principal): AuthPrincipal,
    Json(req): Json<CreateApiKeyRequest>,
) -> Result<(StatusCode, Json<Value>), ApiError> {
    require_org_admin(&principal)?;

    let org_id = req.org_id.unwrap_or(principal.org_id);
    if org_id != principal.org_id {
        return Err(ApiError::forbidden(
            "forbidden",
            "cannot manage principals or API keys in another organization",
        ));
    }
    if let Some(lvl) = req.max_auth_level {
        let _ = AuthLevel::try_from(lvl).map_err(ApiError::from)?;
    }

    let worker_pool = state.require_worker_pool().map_err(ApiError::from)?;
    let principal_in_org: Option<i32> = sqlx::query_scalar(
        "SELECT 1 FROM iam.principals WHERE id = $1 AND org_id = $2",
    )
    .bind(req.principal_id)
    .bind(principal.org_id)
    .fetch_optional(worker_pool.inner())
    .await
    .map_err(|e| {
        ApiError::from(OpenRagError::Database(format!(
            "verify principal org membership: {e}"
        )))
    })?;
    if principal_in_org.is_none() {
        return Err(ApiError::forbidden(
            "forbidden",
            "cannot mint API key for a principal outside caller organization",
        ));
    }

    let mut rand_bytes = [0u8; 24];
    SystemRandom::new()
        .fill(&mut rand_bytes)
        .map_err(|_| ApiError::from(OpenRagError::Internal("CSPRNG failure minting key".into())))?;
    let raw_key = format!("orag_{}", hex::encode(rand_bytes));
    let key_prefix: String = raw_key.chars().take(12).collect();
    let key_hash = openrag_db::sha256_hex(&raw_key);
    let key_name = req
        .name
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(ToString::to_string)
        .unwrap_or_else(|| format!("studio-{key_prefix}"));

    let key_id: Uuid = sqlx::query_scalar(
        "INSERT INTO iam.api_keys (
            org_id, principal_id, name, key_prefix, key_hash,
            scoped_arns, scoped_group_ids, max_auth_level
         ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
         RETURNING id",
    )
    .bind(org_id)
    .bind(req.principal_id)
    .bind(&key_name)
    .bind(&key_prefix)
    .bind(&key_hash)
    .bind(&req.scoped_arns)
    .bind(&req.scoped_group_ids)
    .bind(req.max_auth_level)
    .fetch_one(worker_pool.inner())
    .await
    .map_err(map_sqlx_err("insert iam.api_keys"))?;

    Ok((
        StatusCode::CREATED,
        Json(json!({
            "id": key_id,
            "key_id": key_id,
            "token": raw_key,
            "api_key": raw_key,
            "key": raw_key,
            "key_prefix": key_prefix,
            "org_id": org_id,
            "principal_id": req.principal_id,
            "max_auth_level": req.max_auth_level,
            "scoped_group_ids": req.scoped_group_ids,
            "scoped_arns": req.scoped_arns,
            "allowed_actions": req.allowed_actions,
        })),
    ))
}

async fn studio_iam_list_groups_handler(
    State(state): State<Arc<AppState>>,
    AuthPrincipal(principal): AuthPrincipal,
) -> Result<Json<Value>, ApiError> {
    let worker_pool = state.require_worker_pool().map_err(ApiError::from)?;
    let rows = sqlx::query(
        "SELECT id, org_id, dept_id, name, group_type, min_auth_level, max_auth_level,
                allowed_actions, description, created_at
         FROM iam.groups
         WHERE org_id = $1
         ORDER BY created_at DESC",
    )
    .bind(principal.org_id)
    .fetch_all(worker_pool.inner())
    .await
    .map_err(|e| ApiError::from(OpenRagError::Database(format!("list iam.groups: {e}"))))?;

    let mut groups = Vec::with_capacity(rows.len());
    for r in &rows {
        let id: Uuid = r.try_get("id").map_err(db_col_err)?;
        groups.push(json!({
            "id": id,
            "group_id": id,
            "org_id": r.try_get::<Uuid, _>("org_id").map_err(db_col_err)?,
            "dept_id": r.try_get::<Option<Uuid>, _>("dept_id").map_err(db_col_err)?,
            "name": r.try_get::<String, _>("name").map_err(db_col_err)?,
            "group_type": r.try_get::<String, _>("group_type").map_err(db_col_err)?,
            "min_auth_level": r.try_get::<i16, _>("min_auth_level").map_err(db_col_err)?,
            "max_auth_level": r.try_get::<i16, _>("max_auth_level").map_err(db_col_err)?,
            "allowed_actions": r.try_get::<Vec<String>, _>("allowed_actions").map_err(db_col_err)?,
            "description": r.try_get::<Option<String>, _>("description").map_err(db_col_err)?,
            "created_at": r.try_get::<DateTime<Utc>, _>("created_at").map_err(db_col_err)?,
        }));
    }

    Ok(Json(json!({ "groups": groups, "items": groups })))
}

#[derive(Debug, Deserialize)]
struct CreateGroupRequest {
    id: Option<Uuid>,
    org_id: Option<Uuid>,
    slug: Option<String>,
    dept_id: Option<Uuid>,
    name: String,
    group_type: Option<String>,
    min_auth_level: Option<i16>,
    max_auth_level: Option<i16>,
    allowed_actions: Option<Vec<String>>,
    description: Option<String>,
    #[serde(default)]
    member_ids: Vec<Uuid>,
}

async fn studio_iam_create_group_handler(
    State(state): State<Arc<AppState>>,
    AuthPrincipal(principal): AuthPrincipal,
    Json(req): Json<CreateGroupRequest>,
) -> Result<(StatusCode, Json<Value>), ApiError> {
    require_org_admin(&principal)?;

    let name = req.name.trim();
    if name.is_empty() {
        return Err(ApiError::bad_request(
            "invalid_group_name",
            "group name must not be empty",
        ));
    }
    let _ignored_slug = req.slug.as_deref();
    let org_id = req.org_id.unwrap_or(principal.org_id);
    if org_id != principal.org_id {
        return Err(ApiError::forbidden(
            "forbidden",
            "cannot create IAM group in another organization",
        ));
    }

    let group_type = req
        .group_type
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .unwrap_or("permission_group");
    if !matches!(
        group_type,
        "permission_group" | "project_tf" | "functional" | "clearance"
    ) {
        return Err(ApiError::bad_request(
            "invalid_group_type",
            format!("invalid group_type '{group_type}'"),
        ));
    }

    let min_level = AuthLevel::try_from(req.min_auth_level.unwrap_or(1)).map_err(ApiError::from)?;
    let max_level =
        AuthLevel::try_from(req.max_auth_level.unwrap_or(10)).map_err(ApiError::from)?;
    if min_level > max_level {
        return Err(ApiError::bad_request(
            "invalid_auth_level_range",
            "min_auth_level must be <= max_auth_level",
        ));
    }

    let allowed_actions = req
        .allowed_actions
        .unwrap_or_else(|| vec!["read".to_string(), "search".to_string()]);
    for act in &allowed_actions {
        let _ = IamAction::from_str(act).map_err(ApiError::from)?;
    }

    let group_id = req.id.unwrap_or_else(Uuid::now_v7);
    let worker_pool = state.require_worker_pool().map_err(ApiError::from)?;
    let mut tx = worker_pool
        .inner()
        .begin()
        .await
        .map_err(|e| ApiError::from(OpenRagError::Database(format!("begin group tx: {e}"))))?;

    let row = sqlx::query(
        "INSERT INTO iam.groups (
            id, org_id, dept_id, name, group_type, min_auth_level, max_auth_level,
            allowed_actions, description
         ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
         RETURNING id, org_id, dept_id, name, group_type, min_auth_level, max_auth_level,
                   allowed_actions, description, created_at",
    )
    .bind(group_id)
    .bind(org_id)
    .bind(req.dept_id)
    .bind(name)
    .bind(group_type)
    .bind(min_level.as_i16())
    .bind(max_level.as_i16())
    .bind(&allowed_actions)
    .bind(&req.description)
    .fetch_one(&mut *tx)
    .await
    .map_err(map_sqlx_err("insert iam.groups"))?;

    let created_id: Uuid = row.try_get("id").map_err(db_col_err)?;
    for member_id in &req.member_ids {
        sqlx::query(
            "INSERT INTO iam.group_members (org_id, group_id, principal_id)
             VALUES ($1, $2, $3)
             ON CONFLICT (group_id, principal_id) DO NOTHING",
        )
        .bind(org_id)
        .bind(created_id)
        .bind(member_id)
        .execute(&mut *tx)
        .await
        .map_err(map_sqlx_err("insert iam.group_members"))?;
    }

    tx.commit()
        .await
        .map_err(|e| ApiError::from(OpenRagError::Database(format!("commit group tx: {e}"))))?;

    Ok((
        StatusCode::CREATED,
        Json(json!({
            "id": created_id,
            "group_id": created_id,
            "org_id": row.try_get::<Uuid, _>("org_id").map_err(db_col_err)?,
            "dept_id": row.try_get::<Option<Uuid>, _>("dept_id").map_err(db_col_err)?,
            "name": row.try_get::<String, _>("name").map_err(db_col_err)?,
            "group_type": row.try_get::<String, _>("group_type").map_err(db_col_err)?,
            "min_auth_level": row.try_get::<i16, _>("min_auth_level").map_err(db_col_err)?,
            "max_auth_level": row.try_get::<i16, _>("max_auth_level").map_err(db_col_err)?,
            "allowed_actions": row.try_get::<Vec<String>, _>("allowed_actions").map_err(db_col_err)?,
            "description": row.try_get::<Option<String>, _>("description").map_err(db_col_err)?,
            "member_ids": req.member_ids,
            "created_at": row.try_get::<DateTime<Utc>, _>("created_at").map_err(db_col_err)?,
        })),
    ))
}

#[derive(Debug, Deserialize)]
struct AddGroupMemberRequest {
    principal_id: Uuid,
    member_role: Option<String>,
    action_overrides: Option<Vec<String>>,
    level_override: Option<i16>,
}

async fn studio_iam_add_group_member_handler(
    State(state): State<Arc<AppState>>,
    AuthPrincipal(principal): AuthPrincipal,
    Path(group_id): Path<Uuid>,
    Json(req): Json<AddGroupMemberRequest>,
) -> Result<(StatusCode, Json<Value>), ApiError> {
    require_org_admin(&principal)?;

    let role = req.member_role.as_deref().unwrap_or("member");
    if !matches!(role, "admin" | "member") {
        return Err(ApiError::bad_request(
            "invalid_member_role",
            "member_role must be admin or member",
        ));
    }
    if let Some(lvl) = req.level_override {
        let _ = AuthLevel::try_from(lvl).map_err(ApiError::from)?;
    }
    if let Some(ref acts) = req.action_overrides {
        for act in acts {
            let _ = IamAction::from_str(act).map_err(ApiError::from)?;
        }
    }

    let worker_pool = state.require_worker_pool().map_err(ApiError::from)?;
    let exists: bool = sqlx::query_scalar(
        "SELECT EXISTS(SELECT 1 FROM iam.groups WHERE org_id = $1 AND id = $2)",
    )
    .bind(principal.org_id)
    .bind(group_id)
    .fetch_one(worker_pool.inner())
    .await
    .map_err(|e| ApiError::from(OpenRagError::Database(format!("check iam.groups: {e}"))))?;

    if !exists {
        return Err(ApiError::not_found(format!("group {group_id} not found")));
    }

    sqlx::query(
        "INSERT INTO iam.group_members (
            org_id, group_id, principal_id, member_role, action_overrides, level_override
         ) VALUES ($1, $2, $3, $4, $5, $6)
         ON CONFLICT (group_id, principal_id) DO UPDATE SET
            member_role = EXCLUDED.member_role,
            action_overrides = EXCLUDED.action_overrides,
            level_override = EXCLUDED.level_override",
    )
    .bind(principal.org_id)
    .bind(group_id)
    .bind(req.principal_id)
    .bind(role)
    .bind(&req.action_overrides)
    .bind(req.level_override)
    .execute(worker_pool.inner())
    .await
    .map_err(map_sqlx_err("upsert iam.group_members"))?;

    Ok((
        StatusCode::CREATED,
        Json(json!({
            "org_id": principal.org_id,
            "group_id": group_id,
            "principal_id": req.principal_id,
            "member_role": role,
        })),
    ))
}

async fn studio_iam_remove_group_member_handler(
    State(state): State<Arc<AppState>>,
    AuthPrincipal(principal): AuthPrincipal,
    Path((group_id, member_id)): Path<(Uuid, Uuid)>,
) -> Result<Json<Value>, ApiError> {
    require_org_admin(&principal)?;

    let worker_pool = state.require_worker_pool().map_err(ApiError::from)?;
    let res = sqlx::query(
        "DELETE FROM iam.group_members
         WHERE org_id = $1 AND group_id = $2 AND principal_id = $3",
    )
    .bind(principal.org_id)
    .bind(group_id)
    .bind(member_id)
    .execute(worker_pool.inner())
    .await
    .map_err(|e| ApiError::from(OpenRagError::Database(format!("delete iam.group_members: {e}"))))?;

    Ok(Json(json!({
        "deleted": res.rows_affected() > 0,
        "group_id": group_id,
        "principal_id": member_id,
    })))
}

#[derive(Debug, Deserialize)]
struct CreateResourceBindingRequest {
    org_id: Option<Uuid>,
    resource_type: String,
    resource_id: Uuid,
    #[serde(default, alias = "arn")]
    resource_arn: Option<String>,
    group_id: Uuid,
    effect: String,
    actions: Option<Vec<String>>,
    min_auth_level: Option<i16>,
}

async fn studio_iam_create_resource_binding_handler(
    State(state): State<Arc<AppState>>,
    AuthPrincipal(principal): AuthPrincipal,
    Json(req): Json<CreateResourceBindingRequest>,
) -> Result<(StatusCode, Json<Value>), ApiError> {
    require_org_admin(&principal)?;

    let org_id = req.org_id.unwrap_or(principal.org_id);
    if org_id != principal.org_id {
        return Err(ApiError::forbidden(
            "forbidden",
            "cannot bind group in another organization",
        ));
    }

    let rtype = req.resource_type.trim();
    if !matches!(
        rtype,
        "bucket" | "document" | "page" | "figure" | "chunk" | "memory" | "episode" | "wiki"
    ) {
        return Err(ApiError::bad_request(
            "invalid_resource_type",
            format!("invalid resource_type '{rtype}'"),
        ));
    }

    let effect = req.effect.trim().to_ascii_lowercase();
    if !matches!(effect.as_str(), "allow" | "deny") {
        return Err(ApiError::bad_request(
            "invalid_effect",
            format!("effect must be allow or deny, got '{effect}'"),
        ));
    }

    if let Some(lvl) = req.min_auth_level {
        let _ = AuthLevel::try_from(lvl).map_err(ApiError::from)?;
    }

    let actions = req
        .actions
        .unwrap_or_else(|| vec!["read".to_string(), "search".to_string()]);
    for act in &actions {
        let _ = IamAction::from_str(act).map_err(ApiError::from)?;
    }

    if let Some(ref raw_arn) = req.resource_arn {
        let _ = Arn::parse(raw_arn).map_err(ApiError::from)?;
    }

    let worker_pool = state.require_worker_pool().map_err(ApiError::from)?;
    let id: Uuid = sqlx::query_scalar(
        "INSERT INTO iam.resource_group_bindings (
            org_id, resource_type, resource_id, group_id, effect, actions, min_auth_level
         ) VALUES ($1, $2, $3, $4, $5, $6, $7)
         ON CONFLICT (resource_type, resource_id, group_id, effect) DO UPDATE SET
            actions = EXCLUDED.actions,
            min_auth_level = EXCLUDED.min_auth_level
         RETURNING id",
    )
    .bind(org_id)
    .bind(rtype)
    .bind(req.resource_id)
    .bind(req.group_id)
    .bind(&effect)
    .bind(&actions)
    .bind(req.min_auth_level)
    .fetch_one(worker_pool.inner())
    .await
    .map_err(map_sqlx_err("upsert iam.resource_group_bindings"))?;

    Ok((
        StatusCode::CREATED,
        Json(json!({
            "id": id,
            "binding_id": id,
            "org_id": org_id,
            "resource_type": rtype,
            "resource_id": req.resource_id,
            "group_id": req.group_id,
            "effect": effect,
            "actions": actions,
            "min_auth_level": req.min_auth_level,
        })),
    ))
}

#[derive(Debug, Deserialize)]
struct PolicyDocumentBody {
    statements: Vec<PolicyStatement>,
}

#[derive(Debug, Deserialize)]
struct PolicyAttachmentInput {
    target_type: String,
    target_id: Uuid,
    min_auth_level_override: Option<i16>,
}

#[derive(Debug, Deserialize)]
struct CreatePolicyRequest {
    id: Option<Uuid>,
    org_id: Option<Uuid>,
    name: String,
    description: Option<String>,
    document: PolicyDocumentBody,
    #[serde(default)]
    attachments: Vec<PolicyAttachmentInput>,
}

async fn studio_iam_create_policy_handler(
    State(state): State<Arc<AppState>>,
    AuthPrincipal(principal): AuthPrincipal,
    Json(req): Json<CreatePolicyRequest>,
) -> Result<(StatusCode, Json<Value>), ApiError> {
    require_org_admin(&principal)?;

    let name = req.name.trim();
    if name.is_empty() {
        return Err(ApiError::bad_request(
            "invalid_policy_name",
            "policy name must not be empty",
        ));
    }
    let org_id = req.org_id.unwrap_or(principal.org_id);
    if org_id != principal.org_id {
        return Err(ApiError::forbidden(
            "forbidden",
            "cannot create IAM policy in another organization",
        ));
    }

    for att in &req.attachments {
        if !matches!(
            att.target_type.as_str(),
            "principal" | "dept" | "group" | "api_key"
        ) {
            return Err(ApiError::bad_request(
                "invalid_attachment_target_type",
                format!("invalid target_type '{}'", att.target_type),
            ));
        }
        if let Some(lvl) = att.min_auth_level_override {
            let _ = AuthLevel::try_from(lvl).map_err(ApiError::from)?;
        }
    }

    let doc_json = json!({ "statements": req.document.statements });
    let policy_id = req.id.unwrap_or_else(Uuid::now_v7);

    let worker_pool = state.require_worker_pool().map_err(ApiError::from)?;
    let mut tx = worker_pool
        .inner()
        .begin()
        .await
        .map_err(|e| ApiError::from(OpenRagError::Database(format!("begin policy tx: {e}"))))?;

    let created_id: Uuid = sqlx::query_scalar(
        "INSERT INTO iam.policies (id, org_id, name, description, document)
         VALUES ($1, $2, $3, $4, $5)
         RETURNING id",
    )
    .bind(policy_id)
    .bind(org_id)
    .bind(name)
    .bind(&req.description)
    .bind(&doc_json)
    .fetch_one(&mut *tx)
    .await
    .map_err(map_sqlx_err("insert iam.policies"))?;

    for att in &req.attachments {
        sqlx::query(
            "INSERT INTO iam.policy_attachments (
                org_id, policy_id, target_type, target_id, min_auth_level_override
             ) VALUES ($1, $2, $3, $4, $5)
             ON CONFLICT (policy_id, target_type, target_id) DO UPDATE SET
                min_auth_level_override = EXCLUDED.min_auth_level_override",
        )
        .bind(org_id)
        .bind(created_id)
        .bind(&att.target_type)
        .bind(att.target_id)
        .bind(att.min_auth_level_override)
        .execute(&mut *tx)
        .await
        .map_err(map_sqlx_err("insert iam.policy_attachments"))?;
    }

    tx.commit()
        .await
        .map_err(|e| ApiError::from(OpenRagError::Database(format!("commit policy tx: {e}"))))?;

    Ok((
        StatusCode::CREATED,
        Json(json!({
            "id": created_id,
            "policy_id": created_id,
            "org_id": org_id,
            "name": name,
            "document": doc_json,
        })),
    ))
}

#[derive(Debug, Deserialize)]
struct SimulateIamRequest {
    org_id: Option<Uuid>,
    principal_id: Uuid,
    action: String,
    auth_level: Option<i16>,
    can_unmask_pii: Option<bool>,
    dept_ids: Option<Vec<Uuid>>,
    dept_admin_ids: Option<Vec<Uuid>>,
    group_ids: Option<Vec<Uuid>>,
    #[serde(default, alias = "arn")]
    resource_arn: Option<String>,
    resource_type: Option<String>,
    resource_id: Option<Uuid>,
    dept_id: Option<Uuid>,
    owner_principal_id: Option<Uuid>,
    access_scope: Option<AccessScope>,
    allowed_group_ids: Option<Vec<Uuid>>,
    denied_group_ids: Option<Vec<Uuid>>,
    min_auth_level: Option<i16>,
}

struct HydratedResourceMeta {
    resource_type: ArnResourceType,
    resource_id: Uuid,
    dept_id: Uuid,
    owner_principal_id: Uuid,
    access_scope: AccessScope,
    allowed_group_ids: Vec<Uuid>,
    denied_group_ids: Vec<Uuid>,
    min_auth_level: AuthLevel,
    arn: Arn,
}

async fn hydrate_resource_from_db(
    worker_pool: &DbPool,
    org_id: Uuid,
    rtype_str: &str,
    resource_id: Uuid,
) -> Result<Option<HydratedResourceMeta>, OpenRagError> {
    let table_sql = match rtype_str {
        "bucket" => Some((
            ArnResourceType::Bucket,
            "SELECT dept_id, owner_principal_id, access_scope, allowed_group_ids, denied_group_ids, min_auth_level, arn FROM corpus.buckets WHERE org_id = $1 AND id = $2",
        )),
        "doc" | "document" => Some((
            ArnResourceType::Document,
            "SELECT dept_id, owner_principal_id, access_scope, allowed_group_ids, denied_group_ids, min_auth_level, arn FROM corpus.documents WHERE org_id = $1 AND id = $2",
        )),
        "page" => Some((
            ArnResourceType::Page,
            "SELECT dept_id, owner_principal_id, access_scope, allowed_group_ids, denied_group_ids, min_auth_level, arn FROM corpus.pages WHERE org_id = $1 AND id = $2",
        )),
        "figure" => Some((
            ArnResourceType::Figure,
            "SELECT dept_id, owner_principal_id, access_scope, allowed_group_ids, denied_group_ids, min_auth_level, arn FROM corpus.page_figures WHERE org_id = $1 AND id = $2",
        )),
        "chunk" => Some((
            ArnResourceType::Chunk,
            "SELECT dept_id, owner_principal_id, access_scope, allowed_group_ids, denied_group_ids, min_auth_level, arn FROM corpus.chunks WHERE org_id = $1 AND id = $2",
        )),
        "memory" => Some((
            ArnResourceType::Memory,
            "SELECT dept_id, owner_principal_id, access_scope, allowed_group_ids, denied_group_ids, min_auth_level, arn FROM memory.entries WHERE org_id = $1 AND id = $2",
        )),
        "episode" => Some((
            ArnResourceType::Episode,
            "SELECT dept_id, owner_principal_id, access_scope, allowed_group_ids, denied_group_ids, min_auth_level, arn FROM memory.episodes WHERE org_id = $1 AND id = $2",
        )),
        "wiki" => Some((
            ArnResourceType::Wiki,
            "SELECT dept_id, owner_principal_id, access_scope, allowed_group_ids, denied_group_ids, min_auth_level, arn FROM wiki.pages WHERE org_id = $1 AND id = $2",
        )),
        _ => None,
    };

    let Some((arn_rtype, sql)) = table_sql else {
        return Ok(None);
    };

    let row = sqlx::query(sql)
        .bind(org_id)
        .bind(resource_id)
        .fetch_optional(worker_pool.inner())
        .await
        .map_err(|e| OpenRagError::Database(format!("hydrate simulate resource: {e}")))?;

    let Some(r) = row else {
        return Ok(None);
    };

    let scope_str: String = r
        .try_get("access_scope")
        .map_err(|e| OpenRagError::Database(e.to_string()))?;
    let min_lvl_i16: i16 = r
        .try_get("min_auth_level")
        .map_err(|e| OpenRagError::Database(e.to_string()))?;
    let arn_str: String = r
        .try_get("arn")
        .map_err(|e| OpenRagError::Database(e.to_string()))?;

    Ok(Some(HydratedResourceMeta {
        resource_type: arn_rtype,
        resource_id,
        dept_id: r
            .try_get("dept_id")
            .map_err(|e| OpenRagError::Database(e.to_string()))?,
        owner_principal_id: r
            .try_get("owner_principal_id")
            .map_err(|e| OpenRagError::Database(e.to_string()))?,
        access_scope: scope_str.parse()?,
        allowed_group_ids: r
            .try_get("allowed_group_ids")
            .map_err(|e| OpenRagError::Database(e.to_string()))?,
        denied_group_ids: r
            .try_get("denied_group_ids")
            .map_err(|e| OpenRagError::Database(e.to_string()))?,
        min_auth_level: AuthLevel::try_from(min_lvl_i16)?,
        arn: Arn::parse(&arn_str)?,
    }))
}

async fn studio_iam_simulate_handler(
    State(state): State<Arc<AppState>>,
    AuthPrincipal(caller): AuthPrincipal,
    Json(req): Json<SimulateIamRequest>,
) -> Result<Json<Value>, ApiError> {
    let action = IamAction::from_str(&req.action).map_err(ApiError::from)?;
    let parsed_arn_opt = match req.resource_arn.as_deref() {
        Some(raw) => Some(Arn::parse(raw).map_err(ApiError::from)?),
        None => None,
    };

    let org_id = req.org_id.unwrap_or(caller.org_id);
    if org_id != caller.org_id {
        return Ok(Json(json!({
            "allowed": false,
            "reason": "principal_not_found_or_cross_org",
            "matched_denies": [],
            "matched_allows": [],
            "effective_level": 0,
            "effective_min_level": 10,
            "effective_actions": [],
            "trace": [],
        })));
    }

    let worker_pool = state.require_worker_pool().map_err(ApiError::from)?;

    let p_row = sqlx::query(
        "SELECT auth_level, can_unmask_pii, is_org_admin, status
         FROM iam.principals
         WHERE org_id = $1 AND id = $2",
    )
    .bind(org_id)
    .bind(req.principal_id)
    .fetch_optional(worker_pool.inner())
    .await
    .map_err(|e| ApiError::from(OpenRagError::Database(format!("lookup simulate principal: {e}"))))?;

    let Some(pr) = p_row else {
        return Ok(Json(json!({
            "allowed": false,
            "reason": "principal_not_found_or_cross_org",
            "matched_denies": [],
            "matched_allows": [],
            "effective_level": 0,
            "effective_min_level": 10,
            "effective_actions": [],
            "trace": [],
        })));
    };
    let p_status: String = pr.try_get("status").map_err(db_col_err)?;
    if p_status != "active" {
        return Ok(Json(json!({
            "allowed": false,
            "reason": "principal_inactive",
            "matched_denies": [],
            "matched_allows": [],
            "effective_level": 1,
            "effective_min_level": 1,
            "effective_actions": [],
            "trace": [],
        })));
    }

    let db_auth_level_i16: i16 = pr.try_get("auth_level").map_err(db_col_err)?;
    let db_can_unmask: bool = pr.try_get("can_unmask_pii").map_err(db_col_err)?;
    let db_is_admin: bool = pr.try_get("is_org_admin").map_err(db_col_err)?;

    let eff_auth_level = match req.auth_level {
        Some(lvl) => AuthLevel::try_from(lvl).map_err(ApiError::from)?,
        None => AuthLevel::try_from(db_auth_level_i16).map_err(ApiError::from)?,
    };

    let dept_rows = sqlx::query(
        "SELECT dept_id, dept_role
         FROM iam.dept_memberships
         WHERE org_id = $1 AND principal_id = $2",
    )
    .bind(org_id)
    .bind(req.principal_id)
    .fetch_all(worker_pool.inner())
    .await
    .map_err(|e| ApiError::from(OpenRagError::Database(format!("load dept_memberships: {e}"))))?;

    let mut db_dept_ids = Vec::with_capacity(dept_rows.len());
    let mut db_dept_admin_ids = Vec::new();
    for dr in dept_rows {
        let did: Uuid = dr.try_get("dept_id").map_err(db_col_err)?;
        let role: String = dr.try_get("dept_role").map_err(db_col_err)?;
        db_dept_ids.push(did);
        if role == "dept_admin" {
            db_dept_admin_ids.push(did);
        }
    }

    let db_group_ids: Vec<Uuid> = sqlx::query_scalar(
        "SELECT group_id FROM iam.group_members WHERE org_id = $1 AND principal_id = $2",
    )
    .bind(org_id)
    .bind(req.principal_id)
    .fetch_all(worker_pool.inner())
    .await
    .map_err(|e| ApiError::from(OpenRagError::Database(format!("load group_members: {e}"))))?;

    let mut sim_principal = PrincipalContext::new(org_id, req.principal_id, eff_auth_level);
    sim_principal.dept_ids = req.dept_ids.clone().unwrap_or(db_dept_ids);
    sim_principal.dept_admin_ids = req.dept_admin_ids.clone().unwrap_or(db_dept_admin_ids);
    let mut merged_groups = db_group_ids;
    if let Some(ref extra_groups) = req.group_ids {
        for gid in extra_groups {
            if !merged_groups.contains(gid) {
                merged_groups.push(*gid);
            }
        }
    }
    sim_principal.group_ids = merged_groups;
    sim_principal.can_unmask_pii = req.can_unmask_pii.unwrap_or(db_can_unmask);
    sim_principal.is_org_admin = db_is_admin;

    let hydrated = if let (Some(rtype), Some(rid)) =
        (req.resource_type.as_deref(), req.resource_id)
    {
        hydrate_resource_from_db(worker_pool, org_id, rtype, rid)
            .await
            .map_err(ApiError::from)?
    } else {
        None
    };

    let resource_type = hydrated
        .as_ref()
        .map(|h| h.resource_type)
        .or_else(|| parsed_arn_opt.as_ref().map(|a| a.resource_type))
        .or_else(|| {
            req.resource_type
                .as_deref()
                .and_then(|s| s.parse::<ArnResourceType>().ok())
        })
        .unwrap_or(ArnResourceType::Document);

    let resource_id = req
        .resource_id
        .or_else(|| hydrated.as_ref().map(|h| h.resource_id))
        .or_else(|| {
            parsed_arn_opt
                .as_ref()
                .and_then(|a| Uuid::parse_str(&a.resource_id).ok())
        })
        .unwrap_or_else(Uuid::nil);

    let access_scope = req
        .access_scope
        .or_else(|| hydrated.as_ref().map(|h| h.access_scope))
        .or_else(|| parsed_arn_opt.as_ref().map(|a| a.scope))
        .unwrap_or(AccessScope::Dept);

    let min_auth_level = match req.min_auth_level {
        Some(lvl) => AuthLevel::try_from(lvl).map_err(ApiError::from)?,
        None => hydrated
            .as_ref()
            .map(|h| h.min_auth_level)
            .or_else(|| parsed_arn_opt.as_ref().map(|a| a.auth_level))
            .unwrap_or(AuthLevel::MIN),
    };

    let dept_id = req
        .dept_id
        .or_else(|| hydrated.as_ref().map(|h| h.dept_id))
        .unwrap_or(Uuid::nil());
    let owner_principal_id = req
        .owner_principal_id
        .or_else(|| hydrated.as_ref().map(|h| h.owner_principal_id))
        .unwrap_or(Uuid::nil());
    let allowed_group_ids = req
        .allowed_group_ids
        .or_else(|| hydrated.as_ref().map(|h| h.allowed_group_ids.clone()))
        .unwrap_or_default();
    let denied_group_ids = req
        .denied_group_ids
        .or_else(|| hydrated.as_ref().map(|h| h.denied_group_ids.clone()))
        .unwrap_or_default();

    let arn = match parsed_arn_opt {
        Some(a) => a,
        None => match hydrated {
            Some(ref h) => h.arn.clone(),
            None => Arn::new(
                "org",
                "general",
                access_scope,
                resource_type,
                min_auth_level,
                resource_id.to_string(),
            )
            .map_err(ApiError::from)?,
        },
    };

    let resource_desc = ResourceDescriptor {
        org_id,
        resource_type,
        resource_id,
        dept_id,
        owner_principal_id,
        access_scope,
        allowed_group_ids,
        denied_group_ids,
        min_auth_level,
        arn,
    };

    let group_rows = sqlx::query(
        "SELECT id, org_id, name, min_auth_level, max_auth_level, allowed_actions
         FROM iam.groups
         WHERE org_id = $1",
    )
    .bind(org_id)
    .fetch_all(worker_pool.inner())
    .await
    .map_err(|e| ApiError::from(OpenRagError::Database(format!("load simulate groups: {e}"))))?;

    let mut groups = Vec::with_capacity(group_rows.len());
    for gr in group_rows {
        let min_l: i16 = gr.try_get("min_auth_level").map_err(db_col_err)?;
        let max_l: i16 = gr.try_get("max_auth_level").map_err(db_col_err)?;
        let raw_acts: Vec<String> = gr.try_get("allowed_actions").map_err(db_col_err)?;
        let allowed_actions = raw_acts
            .iter()
            .filter_map(|a| IamAction::from_str(a).ok())
            .collect();
        groups.push(GroupDefinition {
            id: gr.try_get("id").map_err(db_col_err)?,
            org_id: gr.try_get("org_id").map_err(db_col_err)?,
            name: gr.try_get("name").map_err(db_col_err)?,
            min_auth_level: AuthLevel::try_from(min_l).map_err(ApiError::from)?,
            max_auth_level: AuthLevel::try_from(max_l).map_err(ApiError::from)?,
            allowed_actions,
        });
    }

    let gm_rows = sqlx::query(
        "SELECT group_id, principal_id, action_overrides, level_override
         FROM iam.group_members
         WHERE org_id = $1 AND principal_id = $2",
    )
    .bind(org_id)
    .bind(req.principal_id)
    .fetch_all(worker_pool.inner())
    .await
    .map_err(|e| ApiError::from(OpenRagError::Database(format!("load simulate memberships: {e}"))))?;

    let mut memberships = Vec::with_capacity(gm_rows.len());
    for mr in gm_rows {
        let acts_opt: Option<Vec<String>> = mr.try_get("action_overrides").map_err(db_col_err)?;
        let action_overrides = acts_opt.map(|list| {
            list.iter()
                .filter_map(|a| IamAction::from_str(a).ok())
                .collect()
        });
        let lvl_opt: Option<i16> = mr.try_get("level_override").map_err(db_col_err)?;
        let level_override = lvl_opt.and_then(|l| AuthLevel::try_from(l).ok());
        memberships.push(GroupMembership {
            group_id: mr.try_get("group_id").map_err(db_col_err)?,
            principal_id: mr.try_get("principal_id").map_err(db_col_err)?,
            action_overrides,
            level_override,
        });
    }
    if let Some(ref extra_groups) = req.group_ids {
        for gid in extra_groups {
            if !memberships
                .iter()
                .any(|m| m.group_id == *gid && m.principal_id == req.principal_id)
            {
                memberships.push(GroupMembership {
                    group_id: *gid,
                    principal_id: req.principal_id,
                    action_overrides: None,
                    level_override: None,
                });
            }
        }
    }

    let rgb_rows = sqlx::query(
        "SELECT resource_type, resource_id, group_id, effect, actions, min_auth_level
         FROM iam.resource_group_bindings
         WHERE org_id = $1 AND resource_type = $2 AND resource_id = $3",
    )
    .bind(org_id)
    .bind(resource_type.as_str())
    .bind(resource_id)
    .fetch_all(worker_pool.inner())
    .await
    .map_err(|e| ApiError::from(OpenRagError::Database(format!("load simulate bindings: {e}"))))?;

    let mut bindings = Vec::with_capacity(rgb_rows.len());
    for br in rgb_rows {
        let rt_str: String = br.try_get("resource_type").map_err(db_col_err)?;
        let eff_str: String = br.try_get("effect").map_err(db_col_err)?;
        let raw_acts: Vec<String> = br.try_get("actions").map_err(db_col_err)?;
        let min_l_opt: Option<i16> = br.try_get("min_auth_level").map_err(db_col_err)?;
        bindings.push(ResourceGroupBinding {
            resource_type: rt_str.parse().map_err(ApiError::from)?,
            resource_id: br.try_get("resource_id").map_err(db_col_err)?,
            group_id: br.try_get("group_id").map_err(db_col_err)?,
            effect: if eff_str == "deny" {
                PolicyEffect::Deny
            } else {
                PolicyEffect::Allow
            },
            actions: raw_acts
                .iter()
                .filter_map(|a| IamAction::from_str(a).ok())
                .collect(),
            min_auth_level: min_l_opt.and_then(|l| AuthLevel::try_from(l).ok()),
        });
    }

    let pol_rows = sqlx::query(
        "SELECT pol.id, pol.name, pol.document, pa.min_auth_level_override
         FROM iam.policy_attachments AS pa
         JOIN iam.policies AS pol ON pol.id = pa.policy_id
         WHERE pa.org_id = $1
           AND pol.org_id = $1
           AND (
               (pa.target_type = 'principal' AND pa.target_id = $2)
               OR (pa.target_type = 'dept' AND pa.target_id = ANY($3::uuid[]))
               OR (pa.target_type = 'group' AND pa.target_id = ANY($4::uuid[]))
           )",
    )
    .bind(org_id)
    .bind(req.principal_id)
    .bind(&sim_principal.dept_ids)
    .bind(&sim_principal.group_ids)
    .fetch_all(worker_pool.inner())
    .await
    .map_err(|e| ApiError::from(OpenRagError::Database(format!("load simulate policies: {e}"))))?;

    let mut policies = Vec::with_capacity(pol_rows.len());
    for pr in pol_rows {
        let doc_val: Value = pr.try_get("document").map_err(db_col_err)?;
        let statements: Vec<PolicyStatement> = doc_val
            .get("statements")
            .and_then(|s| serde_json::from_value(s.clone()).ok())
            .unwrap_or_default();
        let min_ov_opt: Option<i16> = pr.try_get("min_auth_level_override").map_err(db_col_err)?;
        policies.push(IamPolicy {
            id: pr.try_get("id").map_err(db_col_err)?,
            name: pr.try_get("name").map_err(db_col_err)?,
            statements,
            min_auth_level_override: min_ov_opt.and_then(|l| AuthLevel::try_from(l).ok()),
        });
    }

    let decision = IamEvaluator::simulate(
        &sim_principal,
        &resource_desc,
        action,
        &groups,
        &memberships,
        &bindings,
        &policies,
    );

    let eff_min = decision.effective_min_level.as_u8();
    Ok(Json(json!({
        "allowed": decision.allowed,
        "reason": decision.reason,
        "matched_denies": decision.matched_denies,
        "matched_allows": decision.matched_allows,
        "effective_level": eff_min,
        "effective_min_level": eff_min,
        "effective_actions": decision.effective_actions,
        "trace": decision.trace,
    })))
}

// ============================================================================
// D. PII / Safe-Word Guard / Cascade Redaction Console (/api/studio/pii/*)
// ============================================================================

async fn studio_pii_profiles_get_handler(
    State(state): State<Arc<AppState>>,
    AuthPrincipal(principal): AuthPrincipal,
) -> Result<Json<Value>, ApiError> {
    let worker_pool = state.require_worker_pool().map_err(ApiError::from)?;
    if let Some(profile) = PiiProfileRepo::get_active_profile(worker_pool.inner(), principal.org_id)
        .await
        .map_err(ApiError::from)?
    {
        return Ok(Json(json!({
            "id": profile.id,
            "org_id": profile.org_id,
            "name": profile.name,
            "use_schift_v7": profile.use_schift_v7,
            "use_marker_ko_pii": profile.use_marker_ko_pii,
            "use_schift_v6_org": profile.use_schift_v6_org,
            "custom_dict_terms": profile.custom_dict_terms,
            "allowlist_terms": profile.allowlist_terms,
            "is_default": profile.is_default,
            "updated_at": profile.updated_at,
        })));
    }

    let engine = state.pii_engine.read().await.clone();
    Ok(Json(json!({
        "org_id": principal.org_id,
        "name": "default",
        "use_schift_v7": engine.use_schift_v7,
        "use_marker_ko_pii": engine.use_marker_ko_pii,
        "use_schift_v6_org": engine.use_schift_v6_org,
        "custom_dict_terms": engine.custom_dict_terms,
        "allowlist_terms": engine.allowlist_terms,
        "is_default": true,
    })))
}

#[derive(Debug, Deserialize)]
struct UpdatePiiProfileRequest {
    name: Option<String>,
    use_schift_v7: Option<bool>,
    use_marker_ko_pii: Option<bool>,
    use_schift_v6_org: Option<bool>,
    custom_dict_terms: Option<Vec<String>>,
    #[serde(default, alias = "allowlist")]
    allowlist_terms: Option<Vec<String>>,
    is_default: Option<bool>,
}

async fn studio_pii_profiles_update_handler(
    State(state): State<Arc<AppState>>,
    AuthPrincipal(principal): AuthPrincipal,
    Json(req): Json<UpdatePiiProfileRequest>,
) -> Result<Json<Value>, ApiError> {
    require_org_admin(&principal)?;

    let worker_pool = state.require_worker_pool().map_err(ApiError::from)?;
    let existing = PiiProfileRepo::get_active_profile(worker_pool.inner(), principal.org_id)
        .await
        .map_err(ApiError::from)?;
    let current = state.pii_engine.read().await.clone();

    let name = req
        .name
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(ToString::to_string)
        .or_else(|| existing.as_ref().map(|p| p.name.clone()))
        .unwrap_or_else(|| "default".to_string());

    let input = DetectorProfileInput {
        name,
        use_schift_v7: req
            .use_schift_v7
            .unwrap_or_else(|| existing.as_ref().map_or(current.use_schift_v7, |p| p.use_schift_v7)),
        use_marker_ko_pii: req.use_marker_ko_pii.unwrap_or_else(|| {
            existing
                .as_ref()
                .map_or(current.use_marker_ko_pii, |p| p.use_marker_ko_pii)
        }),
        use_schift_v6_org: req.use_schift_v6_org.unwrap_or_else(|| {
            existing
                .as_ref()
                .map_or(current.use_schift_v6_org, |p| p.use_schift_v6_org)
        }),
        custom_dict_terms: req.custom_dict_terms.unwrap_or_else(|| {
            existing
                .as_ref()
                .map_or(current.custom_dict_terms, |p| p.custom_dict_terms.clone())
        }),
        allowlist_terms: req.allowlist_terms.unwrap_or_else(|| {
            existing
                .as_ref()
                .map_or(current.allowlist_terms, |p| p.allowlist_terms.clone())
        }),
        is_default: req.is_default.unwrap_or(true),
    };

    let saved = PiiProfileRepo::upsert_profile(worker_pool.inner(), principal.org_id, &input)
        .await
        .map_err(ApiError::from)?;
    *state.pii_engine.write().await = PiiEngine::from_profile(&saved);

    Ok(Json(json!({
        "id": saved.id,
        "org_id": saved.org_id,
        "name": saved.name,
        "use_schift_v7": saved.use_schift_v7,
        "use_marker_ko_pii": saved.use_marker_ko_pii,
        "use_schift_v6_org": saved.use_schift_v6_org,
        "custom_dict_terms": saved.custom_dict_terms,
        "allowlist_terms": saved.allowlist_terms,
        "is_default": saved.is_default,
        "updated_at": saved.updated_at,
    })))
}

async fn studio_pii_guard_rules_list_handler(
    State(state): State<Arc<AppState>>,
    AuthPrincipal(principal): AuthPrincipal,
) -> Result<Json<Value>, ApiError> {
    let worker_pool = state.require_worker_pool().map_err(ApiError::from)?;
    let pipeline = state.pipeline_config.read().await.clone();
    let policy = ensure_default_guard_policy(worker_pool, principal.org_id, &pipeline)
        .await
        .map_err(ApiError::from)?;

    let rules = GuardRepo::list_rules(worker_pool.inner(), principal.org_id, policy.id)
        .await
        .map_err(ApiError::from)?;

    let items: Vec<Value> = rules
        .iter()
        .map(|r| {
            json!({
                "id": r.id,
                "rule_id": r.id,
                "policy_id": r.policy_id,
                "org_id": r.org_id,
                "kind": r.kind,
                "rule_kind": r.kind,
                "pattern_or_category": r.pattern_or_category,
                "action": r.action,
                "surfaces": r.surfaces,
                "replacement_text": r.replacement_text,
                "priority": r.priority,
                "enabled": r.enabled,
                "created_at": r.created_at,
            })
        })
        .collect();

    Ok(Json(json!({ "rules": items, "items": items })))
}

#[derive(Debug, Deserialize)]
struct CreateGuardRuleRequest {
    policy_id: Option<Uuid>,
    #[serde(alias = "rule_kind")]
    kind: GuardRuleKind,
    pattern_or_category: String,
    action: GuardAction,
    surfaces: Option<Vec<GuardSurface>>,
    replacement_text: Option<String>,
    enabled: Option<bool>,
    priority: Option<i32>,
}

async fn studio_pii_guard_rules_create_handler(
    State(state): State<Arc<AppState>>,
    AuthPrincipal(principal): AuthPrincipal,
    Json(req): Json<CreateGuardRuleRequest>,
) -> Result<(StatusCode, Json<Value>), ApiError> {
    require_org_admin(&principal)?;

    let worker_pool = state.require_worker_pool().map_err(ApiError::from)?;
    let pipeline = state.pipeline_config.read().await.clone();
    let policy_id = match req.policy_id {
        Some(pid) => pid,
        None => {
            ensure_default_guard_policy(worker_pool, principal.org_id, &pipeline)
                .await
                .map_err(ApiError::from)?
                .id
        }
    };

    let surfaces = req.surfaces.unwrap_or_else(|| {
        vec![
            GuardSurface::InboundQuery,
            GuardSurface::LlmEgress,
            GuardSurface::McpEgress,
            GuardSurface::IngestionGate,
        ]
    });

    let cfg = GuardRuleConfig {
        kind: req.kind,
        pattern_or_category: req.pattern_or_category,
        action: req.action,
        surfaces,
        replacement_text: req.replacement_text,
        enabled: req.enabled.unwrap_or(true),
        priority: req.priority.unwrap_or(100),
    };

    let rule = GuardRepo::create_rule(worker_pool.inner(), principal.org_id, policy_id, &cfg)
        .await
        .map_err(ApiError::from)?;

    Ok((
        StatusCode::CREATED,
        Json(json!({
            "id": rule.id,
            "rule_id": rule.id,
            "policy_id": rule.policy_id,
            "org_id": rule.org_id,
            "kind": rule.kind,
            "rule_kind": rule.kind,
            "pattern_or_category": rule.pattern_or_category,
            "action": rule.action,
            "surfaces": rule.surfaces,
            "replacement_text": rule.replacement_text,
            "priority": rule.priority,
            "enabled": rule.enabled,
            "created_at": rule.created_at,
        })),
    ))
}

#[derive(Debug, Deserialize)]
struct UpdateGuardRuleRequest {
    #[serde(default, alias = "rule_kind")]
    kind: Option<GuardRuleKind>,
    pattern_or_category: Option<String>,
    action: Option<GuardAction>,
    surfaces: Option<Vec<GuardSurface>>,
    replacement_text: Option<String>,
    priority: Option<i32>,
    enabled: Option<bool>,
}

async fn studio_pii_guard_rules_update_handler(
    State(state): State<Arc<AppState>>,
    AuthPrincipal(principal): AuthPrincipal,
    Path(rule_id): Path<Uuid>,
    Json(req): Json<UpdateGuardRuleRequest>,
) -> Result<Json<Value>, ApiError> {
    require_org_admin(&principal)?;

    let worker_pool = state.require_worker_pool().map_err(ApiError::from)?;
    let existing = sqlx::query(
        "SELECT id, policy_id, rule_kind, pattern_or_category, action, surfaces,
                replacement_text, priority, enabled
         FROM pii.guard_rules
         WHERE org_id = $1 AND id = $2",
    )
    .bind(principal.org_id)
    .bind(rule_id)
    .fetch_optional(worker_pool.inner())
    .await
    .map_err(|e| ApiError::from(OpenRagError::Database(format!("load pii.guard_rules: {e}"))))?
    .ok_or_else(|| ApiError::not_found(format!("guard rule {rule_id} not found")))?;

    let cur_kind_str: String = existing.try_get("rule_kind").map_err(db_col_err)?;
    let cur_pattern: String = existing.try_get("pattern_or_category").map_err(db_col_err)?;
    let cur_action_str: String = existing.try_get("action").map_err(db_col_err)?;
    let cur_surfaces: Vec<String> = existing.try_get("surfaces").map_err(db_col_err)?;
    let cur_repl: Option<String> = existing.try_get("replacement_text").map_err(db_col_err)?;
    let cur_prio: i32 = existing.try_get("priority").map_err(db_col_err)?;
    let cur_enabled: bool = existing.try_get("enabled").map_err(db_col_err)?;

    let next_kind = match req.kind {
        Some(k) => match k {
            GuardRuleKind::PiiCategory => "pii_category",
            GuardRuleKind::SafeWord => "safe_word",
            GuardRuleKind::Keyword => "keyword",
            GuardRuleKind::Regex => "regex",
        }
        .to_string(),
        None => cur_kind_str,
    };
    let next_pattern = req.pattern_or_category.unwrap_or(cur_pattern);
    if next_pattern.trim().is_empty() {
        return Err(ApiError::bad_request(
            "invalid_pattern",
            "pattern_or_category must not be empty",
        ));
    }
    if next_kind == "regex" {
        Regex::new(&next_pattern).map_err(|e| {
            ApiError::bad_request("invalid_regex", format!("invalid regex '{next_pattern}': {e}"))
        })?;
    }

    let next_action = req
        .action
        .map(|a| a.as_str().to_string())
        .unwrap_or(cur_action_str);
    let next_surfaces = match req.surfaces {
        Some(s) => {
            if s.is_empty() {
                return Err(ApiError::bad_request(
                    "invalid_surfaces",
                    "surfaces must not be empty",
                ));
            }
            s.iter().map(|x| x.as_str().to_string()).collect()
        }
        None => cur_surfaces,
    };
    let next_repl = req.replacement_text.or(cur_repl);
    let next_prio = req.priority.unwrap_or(cur_prio);
    let next_enabled = req.enabled.unwrap_or(cur_enabled);

    let updated = sqlx::query(
        "UPDATE pii.guard_rules
         SET rule_kind = $3,
             pattern_or_category = $4,
             action = $5,
             surfaces = $6,
             replacement_text = $7,
             priority = $8,
             enabled = $9
         WHERE org_id = $1 AND id = $2
         RETURNING id, policy_id, rule_kind, pattern_or_category, action, surfaces,
                   replacement_text, priority, enabled, created_at",
    )
    .bind(principal.org_id)
    .bind(rule_id)
    .bind(&next_kind)
    .bind(&next_pattern)
    .bind(&next_action)
    .bind(&next_surfaces)
    .bind(&next_repl)
    .bind(next_prio)
    .bind(next_enabled)
    .fetch_one(worker_pool.inner())
    .await
    .map_err(map_sqlx_err("update pii.guard_rules"))?;

    let id: Uuid = updated.try_get("id").map_err(db_col_err)?;
    let kind_out: String = updated.try_get("rule_kind").map_err(db_col_err)?;
    Ok(Json(json!({
        "id": id,
        "rule_id": id,
        "policy_id": updated.try_get::<Uuid, _>("policy_id").map_err(db_col_err)?,
        "kind": kind_out,
        "rule_kind": kind_out,
        "pattern_or_category": updated.try_get::<String, _>("pattern_or_category").map_err(db_col_err)?,
        "action": updated.try_get::<String, _>("action").map_err(db_col_err)?,
        "surfaces": updated.try_get::<Vec<String>, _>("surfaces").map_err(db_col_err)?,
        "replacement_text": updated.try_get::<Option<String>, _>("replacement_text").map_err(db_col_err)?,
        "priority": updated.try_get::<i32, _>("priority").map_err(db_col_err)?,
        "enabled": updated.try_get::<bool, _>("enabled").map_err(db_col_err)?,
        "created_at": updated.try_get::<DateTime<Utc>, _>("created_at").map_err(db_col_err)?,
    })))
}

async fn studio_pii_guard_rules_delete_handler(
    State(state): State<Arc<AppState>>,
    AuthPrincipal(principal): AuthPrincipal,
    Path(rule_id): Path<Uuid>,
) -> Result<Json<Value>, ApiError> {
    require_org_admin(&principal)?;

    let worker_pool = state.require_worker_pool().map_err(ApiError::from)?;
    let deleted = GuardRepo::delete_rule(worker_pool.inner(), principal.org_id, rule_id)
        .await
        .map_err(ApiError::from)?;
    if !deleted {
        return Err(ApiError::not_found(format!("guard rule {rule_id} not found")));
    }

    Ok(Json(json!({ "deleted": true, "id": rule_id })))
}

async fn studio_pii_guard_policies_list_handler(
    State(state): State<Arc<AppState>>,
    AuthPrincipal(principal): AuthPrincipal,
) -> Result<Json<Value>, ApiError> {
    let worker_pool = state.require_worker_pool().map_err(ApiError::from)?;
    let pipeline = state.pipeline_config.read().await.clone();
    let _ = ensure_default_guard_policy(worker_pool, principal.org_id, &pipeline)
        .await
        .map_err(ApiError::from)?;

    let rows = sqlx::query(
        "SELECT id, org_id, bucket_id, name, master_enabled, default_action,
                allowlist_terms, surfaces, is_active, updated_at
         FROM pii.guard_policies
         WHERE org_id = $1
         ORDER BY is_active DESC, updated_at DESC",
    )
    .bind(principal.org_id)
    .fetch_all(worker_pool.inner())
    .await
    .map_err(|e| ApiError::from(OpenRagError::Database(format!("list pii.guard_policies: {e}"))))?;

    let mut policies = Vec::with_capacity(rows.len());
    for r in &rows {
        let id: Uuid = r.try_get("id").map_err(db_col_err)?;
        let allowlist: Vec<String> = r.try_get("allowlist_terms").map_err(db_col_err)?;
        policies.push(json!({
            "id": id,
            "policy_id": id,
            "org_id": r.try_get::<Uuid, _>("org_id").map_err(db_col_err)?,
            "bucket_id": r.try_get::<Option<Uuid>, _>("bucket_id").map_err(db_col_err)?,
            "name": r.try_get::<String, _>("name").map_err(db_col_err)?,
            "master_enabled": r.try_get::<bool, _>("master_enabled").map_err(db_col_err)?,
            "default_action": r.try_get::<String, _>("default_action").map_err(db_col_err)?,
            "allowlist_terms": allowlist,
            "allowlist": allowlist,
            "surfaces": r.try_get::<Vec<String>, _>("surfaces").map_err(db_col_err)?,
            "is_active": r.try_get::<bool, _>("is_active").map_err(db_col_err)?,
            "updated_at": r.try_get::<DateTime<Utc>, _>("updated_at").map_err(db_col_err)?,
        }));
    }

    Ok(Json(json!({ "policies": policies, "items": policies })))
}

#[derive(Debug, Deserialize)]
struct UpdateGuardPolicyRequest {
    master_enabled: Option<bool>,
    default_action: Option<GuardAction>,
    #[serde(default, alias = "allowlist")]
    allowlist_terms: Option<Vec<String>>,
    surfaces: Option<Vec<GuardSurface>>,
}

async fn studio_pii_guard_policies_update_default_handler(
    State(state): State<Arc<AppState>>,
    AuthPrincipal(principal): AuthPrincipal,
    Json(req): Json<UpdateGuardPolicyRequest>,
) -> Result<Json<Value>, ApiError> {
    require_org_admin(&principal)?;

    let worker_pool = state.require_worker_pool().map_err(ApiError::from)?;
    let pipeline = state.pipeline_config.read().await.clone();
    let default_policy = ensure_default_guard_policy(worker_pool, principal.org_id, &pipeline)
        .await
        .map_err(ApiError::from)?;
    update_guard_policy_inner(worker_pool, principal.org_id, default_policy.id, req).await
}

async fn studio_pii_guard_policies_update_by_id_handler(
    State(state): State<Arc<AppState>>,
    AuthPrincipal(principal): AuthPrincipal,
    Path(policy_id): Path<Uuid>,
    Json(req): Json<UpdateGuardPolicyRequest>,
) -> Result<Json<Value>, ApiError> {
    require_org_admin(&principal)?;

    let worker_pool = state.require_worker_pool().map_err(ApiError::from)?;
    update_guard_policy_inner(worker_pool, principal.org_id, policy_id, req).await
}

async fn update_guard_policy_inner(
    worker_pool: &DbPool,
    org_id: Uuid,
    policy_id: Uuid,
    req: UpdateGuardPolicyRequest,
) -> Result<Json<Value>, ApiError> {
    let existing = sqlx::query(
        "SELECT master_enabled, default_action, allowlist_terms, surfaces
         FROM pii.guard_policies
         WHERE org_id = $1 AND id = $2",
    )
    .bind(org_id)
    .bind(policy_id)
    .fetch_optional(worker_pool.inner())
    .await
    .map_err(|e| ApiError::from(OpenRagError::Database(format!("load guard policy: {e}"))))?
    .ok_or_else(|| ApiError::not_found(format!("guard policy {policy_id} not found")))?;

    let cur_enabled: bool = existing.try_get("master_enabled").map_err(db_col_err)?;
    let cur_def_str: String = existing.try_get("default_action").map_err(db_col_err)?;
    let cur_allow: Vec<String> = existing.try_get("allowlist_terms").map_err(db_col_err)?;
    let cur_surf_strs: Vec<String> = existing.try_get("surfaces").map_err(db_col_err)?;

    let cur_def = match cur_def_str.as_str() {
        "allow" => GuardAction::Allow,
        "warn" => GuardAction::Warn,
        "mask" => GuardAction::Mask,
        "block" => GuardAction::Block,
        other => {
            return Err(ApiError::from(OpenRagError::Database(format!(
                "invalid guard default_action '{other}'"
            ))))
        }
    };
    let cur_surfs: Vec<GuardSurface> = cur_surf_strs
        .iter()
        .filter_map(|s| match s.as_str() {
            "inbound_query" => Some(GuardSurface::InboundQuery),
            "llm_egress" => Some(GuardSurface::LlmEgress),
            "mcp_egress" => Some(GuardSurface::McpEgress),
            "ingestion_gate" => Some(GuardSurface::IngestionGate),
            _ => None,
        })
        .collect();

    let updated = GuardRepo::update_policy(
        worker_pool.inner(),
        org_id,
        policy_id,
        req.master_enabled.unwrap_or(cur_enabled),
        req.default_action.unwrap_or(cur_def),
        &req.allowlist_terms.unwrap_or(cur_allow),
        &req.surfaces.unwrap_or(cur_surfs),
    )
    .await
    .map_err(ApiError::from)?;

    Ok(Json(json!({
        "id": updated.id,
        "policy_id": updated.id,
        "org_id": updated.org_id,
        "name": updated.name,
        "master_enabled": updated.master_enabled,
        "default_action": updated.default_action,
        "allowlist_terms": updated.allowlist_terms,
        "allowlist": updated.allowlist_terms,
        "surfaces": updated.surfaces,
        "is_active": updated.is_active,
        "updated_at": updated.updated_at,
    })))
}

async fn studio_pii_guard_events_handler(
    State(state): State<Arc<AppState>>,
    AuthPrincipal(principal): AuthPrincipal,
) -> Result<Json<Value>, ApiError> {
    let app_pool = state.require_app_pool().map_err(ApiError::from)?;
    let mut tx = app_pool
        .begin_rls_tx(&principal)
        .await
        .map_err(ApiError::from)?;

    let rows = sqlx::query(
        "SELECT id, org_id, policy_id, principal_id, surface, decision,
                matched_rule_ids, matched_categories, metadata, created_at
         FROM pii.guard_events
         WHERE org_id = $1
         ORDER BY created_at DESC
         LIMIT 100",
    )
    .bind(principal.org_id)
    .fetch_all(&mut **tx.tx_mut())
    .await
    .map_err(|e| ApiError::from(OpenRagError::Database(format!("list pii.guard_events: {e}"))))?;

    tx.commit()
        .await
        .map_err(|e| ApiError::from(OpenRagError::Database(format!("commit guard_events: {e}"))))?;

    let mut events = Vec::with_capacity(rows.len());
    for r in &rows {
        let raw_meta: Value = r.try_get("metadata").map_err(db_col_err)?;
        let safe_meta = json!({
            "surface": raw_meta.get("surface"),
            "decision": raw_meta.get("decision"),
            "match_count": raw_meta.get("match_count"),
            "warnings_count": raw_meta.get("warnings_count"),
            "text_len": raw_meta.get("text_len"),
            "text_sha256_prefix": raw_meta.get("text_sha256_prefix"),
        });
        events.push(json!({
            "id": r.try_get::<Uuid, _>("id").map_err(db_col_err)?,
            "org_id": r.try_get::<Uuid, _>("org_id").map_err(db_col_err)?,
            "policy_id": r.try_get::<Option<Uuid>, _>("policy_id").map_err(db_col_err)?,
            "principal_id": r.try_get::<Option<Uuid>, _>("principal_id").map_err(db_col_err)?,
            "surface": r.try_get::<String, _>("surface").map_err(db_col_err)?,
            "decision": r.try_get::<String, _>("decision").map_err(db_col_err)?,
            "matched_rule_ids": r.try_get::<Vec<Uuid>, _>("matched_rule_ids").map_err(db_col_err)?,
            "matched_categories": r.try_get::<Vec<String>, _>("matched_categories").map_err(db_col_err)?,
            "metadata": safe_meta,
            "created_at": r.try_get::<DateTime<Utc>, _>("created_at").map_err(db_col_err)?,
        }));
    }

    Ok(Json(json!({ "events": events, "items": events })))
}

#[derive(Debug, Deserialize)]
struct GuardTestRequest {
    text: String,
    surface: GuardSurface,
    #[serde(default, alias = "master_enabled")]
    guard_enabled: Option<bool>,
    #[serde(default, alias = "allowlist_terms")]
    allowlist: Option<Vec<String>>,
    #[serde(default)]
    rules: Option<Vec<GuardRuleConfig>>,
}

async fn studio_pii_guard_test_handler(
    State(state): State<Arc<AppState>>,
    AuthPrincipal(principal): AuthPrincipal,
    Json(req): Json<GuardTestRequest>,
) -> Result<Json<Value>, ApiError> {
    let pipeline = state.pipeline_config.read().await.clone();
    let pii_engine = state.pii_engine.read().await.clone();
    let mut policy: GuardPolicyRuntime = load_active_guard_policy(
        state.worker_pool.as_ref(),
        principal.org_id,
        &pipeline,
    )
    .await
    .map_err(ApiError::from)?;

    if let Some(enabled) = req.guard_enabled {
        policy.master_enabled = enabled;
    }
    if let Some(allowlist) = req.allowlist {
        policy.allowlist_terms = allowlist;
    }
    if let Some(rules) = req.rules {
        policy.rules = rules.iter().map(GuardRuleRuntime::from).collect();
    }

    let lookup = load_concept_lookup(
        state.worker_pool.as_ref(),
        &state.in_memory_concepts,
        principal.org_id,
    )
    .await
    .map_err(ApiError::from)?;

    let decision = openrag_pii::GuardEngine::evaluate(
        &req.text,
        req.surface,
        &policy,
        &pii_engine,
        Some(&lookup),
        None,
    )
    .map_err(ApiError::from)?;

    let mut matched_categories: Vec<String> = Vec::new();
    for m in &decision.matched_rules {
        let label = match m.kind {
            GuardRuleKind::PiiCategory => openrag_pii::PiiCategory::parse_alias(&m.pattern_or_category)
                .map_or_else(
                    || m.pattern_or_category.clone(),
                    |c| c.as_canonical_str().to_string(),
                ),
            GuardRuleKind::SafeWord => "safe_word".to_string(),
            GuardRuleKind::Keyword => "keyword".to_string(),
            GuardRuleKind::Regex => "regex".to_string(),
        };
        if !matched_categories.contains(&label) {
            matched_categories.push(label);
        }
    }

    let detector_matches: Vec<Value> = if !policy.master_enabled || pii_engine.all_detectors_disabled()
    {
        Vec::new()
    } else {
        pii_engine
            .detect(&req.text)
            .into_iter()
            .map(|s| {
                json!({
                    "category": s.category.as_canonical_str(),
                    "detector_source": s.detector_source,
                    "start_byte": s.start_byte,
                    "end_byte": s.end_byte,
                    "confidence": s.confidence,
                })
            })
            .collect()
    };

    let sanitized = decision
        .sanitized_text
        .unwrap_or_else(|| req.text.clone());

    Ok(Json(json!({
        "decision": decision.action.as_str(),
        "action": decision.action.as_str(),
        "sanitized_text": sanitized,
        "masked_text": sanitized,
        "warnings": decision.warnings,
        "matched_rules": decision.matched_rules,
        "matched_categories": matched_categories,
        "detector_matches": detector_matches,
    })))
}

async fn studio_pii_redactions_list_handler(
    State(state): State<Arc<AppState>>,
    AuthPrincipal(principal): AuthPrincipal,
) -> Result<Json<Value>, ApiError> {
    require_org_admin(&principal)?;
    let worker_pool = state.require_worker_pool().map_err(ApiError::from)?;
    let rows = sqlx::query(
        "SELECT id, org_id, requested_by, target_scope, target_ref, redaction_mode,
                redact_thumbnails, reason, status, affected_counts, created_at, completed_at
         FROM pii.redaction_jobs
         WHERE org_id = $1
         ORDER BY created_at DESC
         LIMIT 100",
    )
    .bind(principal.org_id)
    .fetch_all(worker_pool.inner())
    .await
    .map_err(|e| ApiError::from(OpenRagError::Database(format!("list pii.redaction_jobs: {e}"))))?;

    let mut jobs = Vec::with_capacity(rows.len());
    for r in &rows {
        let id: Uuid = r.try_get("id").map_err(db_col_err)?;
        jobs.push(json!({
            "id": id,
            "job_id": id,
            "org_id": r.try_get::<Uuid, _>("org_id").map_err(db_col_err)?,
            "requested_by": r.try_get::<Uuid, _>("requested_by").map_err(db_col_err)?,
            "target_scope": r.try_get::<String, _>("target_scope").map_err(db_col_err)?,
            "target_ref": r.try_get::<Value, _>("target_ref").map_err(db_col_err)?,
            "redaction_mode": r.try_get::<String, _>("redaction_mode").map_err(db_col_err)?,
            "redact_thumbnails": r.try_get::<bool, _>("redact_thumbnails").map_err(db_col_err)?,
            "reason": r.try_get::<String, _>("reason").map_err(db_col_err)?,
            "status": r.try_get::<String, _>("status").map_err(db_col_err)?,
            "affected_counts": r.try_get::<Value, _>("affected_counts").map_err(db_col_err)?,
            "created_at": r.try_get::<DateTime<Utc>, _>("created_at").map_err(db_col_err)?,
            "completed_at": r.try_get::<Option<DateTime<Utc>>, _>("completed_at").map_err(db_col_err)?,
        }));
    }

    Ok(Json(json!({ "jobs": jobs, "redactions": jobs, "items": jobs })))
}

#[derive(Debug, Deserialize)]
struct CreateRedactionRequest {
    target_scope: RedactionTargetScope,
    target_ref: Value,
    redaction_mode: RedactionMode,
    redact_thumbnails: Option<bool>,
    reason: String,
}

async fn studio_pii_redactions_create_handler(
    State(state): State<Arc<AppState>>,
    AuthPrincipal(principal): AuthPrincipal,
    Json(req): Json<CreateRedactionRequest>,
) -> Result<(StatusCode, Json<Value>), ApiError> {
    require_org_admin(&principal)?;

    let worker_pool = state.require_worker_pool().map_err(ApiError::from)?;
    let job_req = RedactionJobRequest {
        org_id: principal.org_id,
        target_scope: req.target_scope,
        target_ref: req.target_ref,
        redaction_mode: req.redaction_mode,
        redact_thumbnails: req.redact_thumbnails.unwrap_or(true),
        reason: req.reason,
    };

    let thumb_adapter = BlobThumbnailStoreAdapter(Arc::clone(&state.blob_store));
    let embed_adapter = RedactionEmbedderAdapter(Arc::clone(&state.text_embedder));
    let res = CascadeRedactor::run_job(
        worker_pool.inner(),
        &principal,
        job_req,
        &thumb_adapter,
        &embed_adapter,
        Some(&state.master_kek),
    )
    .await
    .map_err(ApiError::from)?;

    Ok((
        StatusCode::CREATED,
        Json(json!({
            "id": res.job_id,
            "job_id": res.job_id,
            "status": res.status,
            "replacement_token": res.replacement_token,
            "affected_counts": res.affected_counts,
        })),
    ))
}

async fn studio_pii_audit_logs_handler(
    State(state): State<Arc<AppState>>,
    AuthPrincipal(principal): AuthPrincipal,
) -> Result<Json<Value>, ApiError> {
    let app_pool = state.require_app_pool().map_err(ApiError::from)?;
    let mut tx = app_pool
        .begin_rls_tx(&principal)
        .await
        .map_err(ApiError::from)?;

    let rows = sqlx::query(
        "SELECT id, org_id, principal_id, action, resource_arn, vault_entry_id,
                detector_source, allowed, reason, created_at
         FROM pii.audit_logs
         WHERE org_id = $1
         ORDER BY created_at DESC
         LIMIT 200",
    )
    .bind(principal.org_id)
    .fetch_all(&mut **tx.tx_mut())
    .await
    .map_err(|e| ApiError::from(OpenRagError::Database(format!("list pii.audit_logs: {e}"))))?;

    tx.commit()
        .await
        .map_err(|e| ApiError::from(OpenRagError::Database(format!("commit audit_logs: {e}"))))?;

    let mut logs = Vec::with_capacity(rows.len());
    for r in &rows {
        logs.push(json!({
            "id": r.try_get::<Uuid, _>("id").map_err(db_col_err)?,
            "org_id": r.try_get::<Uuid, _>("org_id").map_err(db_col_err)?,
            "principal_id": r.try_get::<Option<Uuid>, _>("principal_id").map_err(db_col_err)?,
            "action": r.try_get::<String, _>("action").map_err(db_col_err)?,
            "resource_arn": r.try_get::<Option<String>, _>("resource_arn").map_err(db_col_err)?,
            "vault_entry_id": r.try_get::<Option<Uuid>, _>("vault_entry_id").map_err(db_col_err)?,
            "detector_source": r.try_get::<Option<String>, _>("detector_source").map_err(db_col_err)?,
            "allowed": r.try_get::<bool, _>("allowed").map_err(db_col_err)?,
            "reason": r.try_get::<Option<String>, _>("reason").map_err(db_col_err)?,
            "created_at": r.try_get::<DateTime<Utc>, _>("created_at").map_err(db_col_err)?,
        }));
    }

    Ok(Json(json!({ "audit_logs": logs, "logs": logs, "items": logs })))
}

// ============================================================================
// E. Gateway Console (/api/studio/gateway/*)
// ============================================================================

async fn studio_gateway_providers_handler(
    State(state): State<Arc<AppState>>,
    AuthPrincipal(principal): AuthPrincipal,
) -> Result<Json<Value>, ApiError> {
    let worker_pool = state.require_worker_pool().map_err(ApiError::from)?;
    ensure_default_providers(worker_pool, principal.org_id)
        .await
        .map_err(ApiError::from)?;
    let providers = list_providers(worker_pool, principal.org_id)
        .await
        .map_err(ApiError::from)?;

    Ok(Json(json!({ "providers": providers, "items": providers })))
}

#[derive(Debug, Deserialize)]
struct CreateRoutingRuleRequest {
    priority: Option<i32>,
    min_auth_level: i16,
    max_auth_level: i16,
    require_pii_masked: Option<bool>,
    target_provider_id: Uuid,
    fallback_provider_id: Option<Uuid>,
    enabled: Option<bool>,
}

async fn studio_gateway_routing_rules_create_handler(
    State(state): State<Arc<AppState>>,
    AuthPrincipal(principal): AuthPrincipal,
    Json(req): Json<CreateRoutingRuleRequest>,
) -> Result<(StatusCode, Json<Value>), ApiError> {
    require_org_admin(&principal)?;

    let min_l = AuthLevel::try_from(req.min_auth_level).map_err(ApiError::from)?;
    let max_l = AuthLevel::try_from(req.max_auth_level).map_err(ApiError::from)?;
    if min_l > max_l {
        return Err(ApiError::bad_request(
            "invalid_auth_level_range",
            "min_auth_level must be <= max_auth_level",
        ));
    }

    let worker_pool = state.require_worker_pool().map_err(ApiError::from)?;
    let priority = req.priority.unwrap_or(100);
    let require_pii_masked = req.require_pii_masked.unwrap_or(true);
    let enabled = req.enabled.unwrap_or(true);

    let id: Uuid = sqlx::query_scalar(
        "INSERT INTO gateway.routing_rules (
            org_id, priority, min_auth_level, max_auth_level,
            require_pii_masked, target_provider_id, fallback_provider_id, enabled
         ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
         RETURNING id",
    )
    .bind(principal.org_id)
    .bind(priority)
    .bind(min_l.as_i16())
    .bind(max_l.as_i16())
    .bind(require_pii_masked)
    .bind(req.target_provider_id)
    .bind(req.fallback_provider_id)
    .bind(enabled)
    .fetch_one(worker_pool.inner())
    .await
    .map_err(map_sqlx_err("insert gateway.routing_rules"))?;

    Ok((
        StatusCode::CREATED,
        Json(json!({
            "id": id,
            "rule_id": id,
            "org_id": principal.org_id,
            "priority": priority,
            "min_auth_level": min_l.as_i16(),
            "max_auth_level": max_l.as_i16(),
            "require_pii_masked": require_pii_masked,
            "target_provider_id": req.target_provider_id,
            "fallback_provider_id": req.fallback_provider_id,
            "enabled": enabled,
        })),
    ))
}

async fn studio_gateway_logs_handler(
    State(state): State<Arc<AppState>>,
    AuthPrincipal(principal): AuthPrincipal,
) -> Result<Json<Value>, ApiError> {
    require_org_admin(&principal)?;
    let worker_pool = state.require_worker_pool().map_err(ApiError::from)?;
    let rows = sqlx::query(
        "SELECT id, org_id, principal_id, provider_id, model_id, max_context_auth_level,
                pii_masked, prompt_tokens, completion_tokens, total_tokens,
                cost_usd::float8 AS cost_usd, latency_ms, status, created_at
         FROM gateway.request_logs
         WHERE org_id = $1
         ORDER BY created_at DESC
         LIMIT 200",
    )
    .bind(principal.org_id)
    .fetch_all(worker_pool.inner())
    .await
    .map_err(|e| ApiError::from(OpenRagError::Database(format!("list gateway.request_logs: {e}"))))?;

    let mut logs = Vec::with_capacity(rows.len());
    for r in &rows {
        logs.push(json!({
            "id": r.try_get::<Uuid, _>("id").map_err(db_col_err)?,
            "org_id": r.try_get::<Uuid, _>("org_id").map_err(db_col_err)?,
            "principal_id": r.try_get::<Option<Uuid>, _>("principal_id").map_err(db_col_err)?,
            "provider_id": r.try_get::<Option<Uuid>, _>("provider_id").map_err(db_col_err)?,
            "model_id": r.try_get::<String, _>("model_id").map_err(db_col_err)?,
            "max_context_auth_level": r.try_get::<Option<i16>, _>("max_context_auth_level").map_err(db_col_err)?,
            "pii_masked": r.try_get::<bool, _>("pii_masked").map_err(db_col_err)?,
            "prompt_tokens": r.try_get::<i32, _>("prompt_tokens").map_err(db_col_err)?,
            "completion_tokens": r.try_get::<i32, _>("completion_tokens").map_err(db_col_err)?,
            "total_tokens": r.try_get::<i32, _>("total_tokens").map_err(db_col_err)?,
            "cost_usd": r.try_get::<f64, _>("cost_usd").map_err(db_col_err)?,
            "latency_ms": r.try_get::<i32, _>("latency_ms").map_err(db_col_err)?,
            "status": r.try_get::<String, _>("status").map_err(db_col_err)?,
            "created_at": r.try_get::<DateTime<Utc>, _>("created_at").map_err(db_col_err)?,
        }));
    }

    Ok(Json(json!({ "logs": logs, "items": logs })))
}

// ============================================================================
// F. Eval & AutoRAG Bench Console (/api/studio/eval/*)
// ============================================================================

async fn studio_eval_golden_sets_list_handler(
    State(state): State<Arc<AppState>>,
    AuthPrincipal(principal): AuthPrincipal,
) -> Result<Json<Value>, ApiError> {
    let store = Arc::new(state.pg_eval_store().map_err(ApiError::from)?);
    let sets = GoldenSetManager::new(store)
        .list_golden_sets(principal.org_id)
        .await
        .map_err(ApiError::from)?;
    Ok(Json(json!({ "golden_sets": sets, "items": sets })))
}

async fn studio_eval_golden_sets_create_handler(
    State(state): State<Arc<AppState>>,
    AuthPrincipal(principal): AuthPrincipal,
    Json(input): Json<CreateGoldenSetInput>,
) -> Result<(StatusCode, Json<Value>), ApiError> {
    require_org_admin(&principal)?;

    let store = Arc::new(state.pg_eval_store().map_err(ApiError::from)?);
    let gs = GoldenSetManager::new(store)
        .create_golden_set(principal.org_id, input)
        .await
        .map_err(ApiError::from)?;

    Ok((
        StatusCode::CREATED,
        Json(json!({
            "id": gs.id,
            "golden_set_id": gs.id,
            "org_id": gs.org_id,
            "name": gs.name,
            "axis": gs.axis,
            "description": gs.description,
            "version": gs.version,
            "created_at": gs.created_at,
        })),
    ))
}

async fn studio_eval_golden_items_list_handler(
    State(state): State<Arc<AppState>>,
    AuthPrincipal(principal): AuthPrincipal,
    Path(golden_set_id): Path<Uuid>,
) -> Result<Json<Value>, ApiError> {
    let store = Arc::new(state.pg_eval_store().map_err(ApiError::from)?);
    let items = GoldenSetManager::new(store)
        .list_golden_items(principal.org_id, golden_set_id, true)
        .await
        .map_err(ApiError::from)?;
    Ok(Json(json!({ "items": items })))
}

async fn studio_eval_golden_items_create_handler(
    State(state): State<Arc<AppState>>,
    AuthPrincipal(principal): AuthPrincipal,
    Path(golden_set_id): Path<Uuid>,
    Json(input): Json<CreateGoldenItemInput>,
) -> Result<(StatusCode, Json<Value>), ApiError> {
    require_org_admin(&principal)?;

    let store = Arc::new(state.pg_eval_store().map_err(ApiError::from)?);
    let item = GoldenSetManager::new(store)
        .add_golden_item(principal.org_id, golden_set_id, input)
        .await
        .map_err(ApiError::from)?;

    Ok((
        StatusCode::CREATED,
        Json(json!({
            "id": item.id,
            "item_id": item.id,
            "org_id": item.org_id,
            "golden_set_id": item.golden_set_id,
            "axis": item.axis,
            "query_text": item.query_text,
            "test_principal_id": item.test_principal_id,
            "expected_doc_ids": item.expected_doc_ids,
            "forbidden_doc_ids": item.forbidden_doc_ids,
            "expected_answer": item.expected_answer,
            "expected_pii_tokens": item.expected_pii_tokens,
            "expected_edges": item.expected_edges,
            "review_status": item.review_status,
            "created_at": item.created_at,
        })),
    ))
}

#[derive(Debug, Deserialize)]
struct PatchGoldenItemRequest {
    review_status: ReviewStatus,
}

async fn studio_eval_golden_item_patch_handler(
    State(state): State<Arc<AppState>>,
    AuthPrincipal(principal): AuthPrincipal,
    Path(item_id): Path<Uuid>,
    Json(req): Json<PatchGoldenItemRequest>,
) -> Result<Json<Value>, ApiError> {
    require_org_admin(&principal)?;

    let store = Arc::new(state.pg_eval_store().map_err(ApiError::from)?);
    let updated = GoldenSetManager::new(store)
        .update_review_status(principal.org_id, item_id, req.review_status)
        .await
        .map_err(ApiError::from)?;

    let val = serde_json::to_value(&updated)
        .map_err(|e| ApiError::from(OpenRagError::Internal(format!("serialize golden_item: {e}"))))?;
    Ok(Json(val))
}

async fn studio_eval_feedback_promote_handler(
    State(state): State<Arc<AppState>>,
    AuthPrincipal(principal): AuthPrincipal,
    Path(feedback_id): Path<Uuid>,
    Json(input): Json<PromoteFeedbackInput>,
) -> Result<Json<Value>, ApiError> {
    require_org_admin(&principal)?;

    let store = Arc::new(state.pg_eval_store().map_err(ApiError::from)?);
    let res = FeedbackManager::new(store)
        .promote_feedback(principal.org_id, &principal, feedback_id, input)
        .await
        .map_err(ApiError::from)?;

    Ok(Json(json!({
        "id": res.item.id,
        "item_id": res.item.id,
        "already_promoted": res.already_promoted,
        "feedback": res.feedback,
        "item": res.item,
    })))
}

async fn studio_eval_bench_handler(
    State(state): State<Arc<AppState>>,
    AuthPrincipal(principal): AuthPrincipal,
    Json(req): Json<BenchRunRequest>,
) -> Result<Json<Value>, ApiError> {
    require_org_admin(&principal)?;

    let store = state.pg_eval_store().map_err(ApiError::from)?;
    let retriever = state
        .pg_retriever(principal.org_id)
        .await
        .map_err(ApiError::from)?;
    let pipeline = state.pipeline_config.read().await.clone();

    let summary = BenchRunner::run_all_tracks(
        &store,
        &retriever,
        &principal,
        &req,
        if req.pipeline_yaml.is_some() {
            None
        } else {
            Some(&pipeline)
        },
    )
    .await
    .map_err(ApiError::from)?;

    let val = serde_json::to_value(&summary)
        .map_err(|e| ApiError::from(OpenRagError::Internal(format!("serialize bench summary: {e}"))))?;
    Ok(Json(val))
}

#[derive(Debug, Deserialize)]
struct StudioOptimizeRequest {
    golden_set_id: Option<Uuid>,
    golden_set: Option<String>,
    bucket_id: Option<Uuid>,
    strategy: Option<OptimizationStrategy>,
    search_space_yaml: Option<String>,
    base_pipeline_yaml: Option<String>,
    require_zero_pii_leak: Option<bool>,
    require_zero_arn_violation: Option<bool>,
}

async fn studio_eval_optimize_handler(
    State(state): State<Arc<AppState>>,
    AuthPrincipal(principal): AuthPrincipal,
    Json(req): Json<StudioOptimizeRequest>,
) -> Result<Json<Value>, ApiError> {
    require_org_admin(&principal)?;

    let search_space_yaml = match req.search_space_yaml {
        Some(y) if !y.trim().is_empty() => y,
        _ => serde_yaml_ng::to_string(&state.search_space_config).map_err(|e| {
            ApiError::from(OpenRagError::Internal(format!(
                "serialize search_space_config: {e}"
            )))
        })?,
    };

    let base_pipeline_yaml = match req.base_pipeline_yaml {
        Some(y) if !y.trim().is_empty() => Some(y),
        _ => {
            let cur = state.pipeline_config.read().await.clone();
            Some(serde_yaml_ng::to_string(&cur).map_err(|e| {
                ApiError::from(OpenRagError::Internal(format!(
                    "serialize base pipeline_config: {e}"
                )))
            })?)
        }
    };

    let opt_req = OptimizeRequest {
        golden_set_id: req.golden_set_id,
        golden_set: req.golden_set,
        bucket_id: req.bucket_id,
        strategy: req.strategy,
        search_space_yaml,
        base_pipeline_yaml,
        require_zero_pii_leak: req.require_zero_pii_leak,
        require_zero_arn_violation: req.require_zero_arn_violation,
        export_best_path: None,
    };

    let store = state.pg_eval_store().map_err(ApiError::from)?;
    let retriever = state
        .pg_retriever(principal.org_id)
        .await
        .map_err(ApiError::from)?;
    let report = AutoRagOptimizer::optimize(&store, &retriever, &principal, &opt_req)
        .await
        .map_err(ApiError::from)?;

    let mut val = serde_json::to_value(&report)
        .map_err(|e| ApiError::from(OpenRagError::Internal(format!("serialize optimize report: {e}"))))?;
    if let Some(obj) = val.as_object_mut() {
        obj.insert(
            "best".into(),
            json!({
                "trial_id": report.trial.id,
                "score": report.best_score,
                "config_yaml": report.best_config_yaml,
                "pipeline_yaml": report.best_config_yaml,
                "security_gates": report.security_gates,
            }),
        );
    }
    Ok(Json(val))
}

async fn studio_eval_leaderboard_handler(
    State(state): State<Arc<AppState>>,
    AuthPrincipal(principal): AuthPrincipal,
) -> Result<Json<Value>, ApiError> {
    let store = state.pg_eval_store().map_err(ApiError::from)?;
    let entries = BenchRunner::leaderboard(&store, principal.org_id)
        .await
        .map_err(ApiError::from)?;
    let security_gate = !entries.is_empty() && entries.iter().all(|e| e.security_gate_pass);
    let top_ndcg = entries.first().map_or(0.0, |e| e.ndcg_at_10);

    Ok(Json(json!({
        "entries": entries,
        "leaderboard": entries,
        "security_gate": security_gate,
        "ndcg": top_ndcg,
    })))
}

#[derive(Debug, Deserialize)]
struct EvalReportQuery {
    golden_set_id: Option<Uuid>,
}

async fn studio_eval_report_handler(
    State(state): State<Arc<AppState>>,
    AuthPrincipal(principal): AuthPrincipal,
    Query(q): Query<EvalReportQuery>,
) -> Result<Json<Value>, ApiError> {
    let store = state.pg_eval_store().map_err(ApiError::from)?;
    let report = BenchRunner::generate_poc_report(&store, principal.org_id, q.golden_set_id)
        .await
        .map_err(ApiError::from)?;
    let val = serde_json::to_value(&report)
        .map_err(|e| ApiError::from(OpenRagError::Internal(format!("serialize eval report: {e}"))))?;
    Ok(Json(val))
}

async fn studio_eval_promote_handler(
    State(state): State<Arc<AppState>>,
    AuthPrincipal(principal): AuthPrincipal,
    Json(input): Json<PromotePipelineInput>,
) -> Result<Json<Value>, ApiError> {
    require_org_admin(&principal)?;

    let store = Arc::new(state.pg_eval_store().map_err(ApiError::from)?);
    let promoted = PipelinePromoter::new(store)
        .promote_config(principal.org_id, &principal, input)
        .await
        .map_err(ApiError::from)?;

    let val = serde_json::to_value(&promoted)
        .map_err(|e| ApiError::from(OpenRagError::Internal(format!("serialize promoted config: {e}"))))?;
    Ok(Json(val))
}

// ============================================================================
// G. Wiki / Plugins / Concepts / Knowledge Ops
// ============================================================================

async fn studio_wiki_conflicts_handler(
    State(state): State<Arc<AppState>>,
    AuthPrincipal(principal): AuthPrincipal,
) -> Result<Json<Value>, ApiError> {
    let app_pool = state.require_app_pool().map_err(ApiError::from)?;
    let mut tx = app_pool
        .begin_rls_tx(&principal)
        .await
        .map_err(ApiError::from)?;

    let rows = sqlx::query(
        "SELECT e.id, e.org_id, e.src_page_id, e.dst_page_id, e.relation_type,
                e.confidence, e.resolved_at, e.created_at,
                sp.title AS src_title, sp.slug AS src_slug,
                dp.title AS dst_title, dp.slug AS dst_slug
         FROM wiki.edges AS e
         JOIN wiki.pages AS sp ON sp.id = e.src_page_id AND sp.org_id = e.org_id
         JOIN wiki.pages AS dp ON dp.id = e.dst_page_id AND dp.org_id = e.org_id
         WHERE e.org_id = $1
           AND e.relation_type IN ('contradicts', 'supersedes')
         ORDER BY e.created_at DESC",
    )
    .bind(principal.org_id)
    .fetch_all(&mut **tx.tx_mut())
    .await
    .map_err(|e| ApiError::from(OpenRagError::Database(format!("list wiki conflicts: {e}"))))?;

    tx.commit()
        .await
        .map_err(|e| ApiError::from(OpenRagError::Database(format!("commit wiki conflicts: {e}"))))?;

    let mut conflicts = Vec::with_capacity(rows.len());
    for r in &rows {
        let id: Uuid = r.try_get("id").map_err(db_col_err)?;
        conflicts.push(json!({
            "id": id,
            "edge_id": id,
            "org_id": r.try_get::<Uuid, _>("org_id").map_err(db_col_err)?,
            "src_page_id": r.try_get::<Uuid, _>("src_page_id").map_err(db_col_err)?,
            "dst_page_id": r.try_get::<Uuid, _>("dst_page_id").map_err(db_col_err)?,
            "relation_type": r.try_get::<String, _>("relation_type").map_err(db_col_err)?,
            "confidence": r.try_get::<f32, _>("confidence").map_err(db_col_err)?,
            "resolved_at": r.try_get::<Option<DateTime<Utc>>, _>("resolved_at").map_err(db_col_err)?,
            "created_at": r.try_get::<DateTime<Utc>, _>("created_at").map_err(db_col_err)?,
            "src_title": r.try_get::<String, _>("src_title").map_err(db_col_err)?,
            "src_slug": r.try_get::<String, _>("src_slug").map_err(db_col_err)?,
            "dst_title": r.try_get::<String, _>("dst_title").map_err(db_col_err)?,
            "dst_slug": r.try_get::<String, _>("dst_slug").map_err(db_col_err)?,
        }));
    }

    Ok(Json(json!({
        "conflicts": conflicts,
        "edges": conflicts,
        "items": conflicts,
    })))
}

async fn studio_plugins_connectors_handler(
    State(state): State<Arc<AppState>>,
    AuthPrincipal(principal): AuthPrincipal,
) -> Result<Json<Value>, ApiError> {
    let worker_pool = state.require_worker_pool().map_err(ApiError::from)?;

    let existing_cnt: i64 = sqlx::query_scalar(
        "SELECT count(*) FROM plugins.connectors WHERE org_id = $1",
    )
    .bind(principal.org_id)
    .fetch_one(worker_pool.inner())
    .await
    .map_err(|e| ApiError::from(OpenRagError::Database(format!("count plugins.connectors: {e}"))))?;

    if existing_cnt == 0 {
        sqlx::query(
            "INSERT INTO plugins.connectors (org_id, connector_type, name, status, last_heartbeat_at)
             VALUES
             ($1, 'kordoc_hook', 'kordoc-mcp-bridge', 'healthy', now()),
             ($1, 'docufinder_sync', 'docufinder-connector', 'healthy', now()),
             ($1, 'schift_index', 'schift_index', 'healthy', now()),
             ($1, 'ai_memory', 'ai-memory-hooks', 'healthy', now())
             ON CONFLICT (org_id, name) DO NOTHING",
        )
        .bind(principal.org_id)
        .execute(worker_pool.inner())
        .await
        .map_err(|e| {
            ApiError::from(OpenRagError::Database(format!(
                "seed default plugins.connectors: {e}"
            )))
        })?;
    }

    let rows = sqlx::query(
        "SELECT id, org_id, connector_type, name, config, status, last_heartbeat_at, last_error, created_at
         FROM plugins.connectors
         WHERE org_id = $1
         ORDER BY created_at ASC",
    )
    .bind(principal.org_id)
    .fetch_all(worker_pool.inner())
    .await
    .map_err(|e| ApiError::from(OpenRagError::Database(format!("list plugins.connectors: {e}"))))?;

    let mut connectors = Vec::with_capacity(rows.len());
    for r in &rows {
        let cfg: Value = r.try_get("config").map_err(db_col_err)?;
        let sanitized_cfg = sanitize_connector_config(&cfg);
        connectors.push(json!({
            "id": r.try_get::<Uuid, _>("id").map_err(db_col_err)?,
            "org_id": r.try_get::<Uuid, _>("org_id").map_err(db_col_err)?,
            "connector_type": r.try_get::<String, _>("connector_type").map_err(db_col_err)?,
            "name": r.try_get::<String, _>("name").map_err(db_col_err)?,
            "config": sanitized_cfg,
            "status": r.try_get::<String, _>("status").map_err(db_col_err)?,
            "last_heartbeat_at": r.try_get::<Option<DateTime<Utc>>, _>("last_heartbeat_at").map_err(db_col_err)?,
            "last_error": r.try_get::<Option<String>, _>("last_error").map_err(db_col_err)?,
            "created_at": r.try_get::<DateTime<Utc>, _>("created_at").map_err(db_col_err)?,
        }));
    }

    Ok(Json(json!({ "connectors": connectors, "items": connectors })))
}

fn sanitize_connector_config(cfg: &Value) -> Value {
    let Some(obj) = cfg.as_object() else {
        return json!({});
    };
    let mut out = serde_json::Map::new();
    for (k, v) in obj {
        let lower = k.to_ascii_lowercase();
        if lower.contains("secret") || lower.contains("token") || lower.contains("password") {
            out.insert(k.clone(), json!("***REDACTED***"));
        } else {
            out.insert(k.clone(), v.clone());
        }
    }
    Value::Object(out)
}

async fn studio_concepts_create_handler(
    State(state): State<Arc<AppState>>,
    AuthPrincipal(principal): AuthPrincipal,
    Json(req): Json<CreateConceptRequest>,
) -> Result<(StatusCode, Json<Value>), ApiError> {
    require_org_admin(&principal)?;
    create_concept_handler(State(state), AuthPrincipal(principal), Json(req)).await
}

async fn studio_concepts_link_handler(
    State(state): State<Arc<AppState>>,
    AuthPrincipal(principal): AuthPrincipal,
    Json(req): Json<LinkConceptsRequest>,
) -> Result<(StatusCode, Json<Value>), ApiError> {
    require_org_admin(&principal)?;
    link_concepts_handler(State(state), AuthPrincipal(principal), Json(req)).await
}

async fn studio_concepts_delete_handler(
    State(state): State<Arc<AppState>>,
    AuthPrincipal(principal): AuthPrincipal,
    Path(id): Path<Uuid>,
) -> Result<Json<Value>, ApiError> {
    require_org_admin(&principal)?;
    delete_concept_handler(State(state), AuthPrincipal(principal), Path(id)).await
}

async fn studio_concepts_delete_alias_handler(
    State(state): State<Arc<AppState>>,
    AuthPrincipal(principal): AuthPrincipal,
    Path(ids): Path<(Uuid, Uuid)>,
) -> Result<Json<Value>, ApiError> {
    require_org_admin(&principal)?;
    delete_concept_alias_handler(State(state), AuthPrincipal(principal), Path(ids)).await
}

async fn studio_knowledge_tags_handler(
    State(state): State<Arc<AppState>>,
    AuthPrincipal(principal): AuthPrincipal,
) -> Result<Json<Value>, ApiError> {
    let app_pool = state.require_app_pool().map_err(ApiError::from)?;
    let mut tx = app_pool
        .begin_rls_tx(&principal)
        .await
        .map_err(ApiError::from)?;

    let rows = sqlx::query(
        "SELECT tag, count(*)::bigint AS count
         FROM (
             SELECT unnest(tags) AS tag
             FROM corpus.documents
             WHERE org_id = $1 AND status <> 'archived'
             UNION ALL
             SELECT unnest(tags) AS tag
             FROM wiki.pages
             WHERE org_id = $1 AND status = 'active'
             UNION ALL
             SELECT unnest(tags) AS tag
             FROM memory.entries
             WHERE org_id = $1 AND status = 'active'
         ) AS all_tags
         WHERE btrim(tag) <> ''
         GROUP BY tag
         ORDER BY count DESC, tag ASC",
    )
    .bind(principal.org_id)
    .fetch_all(&mut **tx.tx_mut())
    .await
    .map_err(|e| ApiError::from(OpenRagError::Database(format!("aggregate knowledge tags: {e}"))))?;

    tx.commit()
        .await
        .map_err(|e| ApiError::from(OpenRagError::Database(format!("commit knowledge tags: {e}"))))?;

    let mut tags = Vec::with_capacity(rows.len());
    let mut facets: BTreeMap<String, i64> = BTreeMap::new();
    for r in &rows {
        let tag: String = r.try_get("tag").map_err(db_col_err)?;
        let count: i64 = r.try_get("count").map_err(db_col_err)?;
        facets.insert(tag.clone(), count);
        tags.push(json!({ "tag": tag, "count": count }));
    }

    Ok(Json(json!({
        "tags": tags,
        "facets": facets,
        "items": tags,
    })))
}
