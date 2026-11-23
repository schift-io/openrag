use std::collections::{HashMap, HashSet};
use std::str::FromStr;
use std::sync::LazyLock;
use std::time::Instant;

use async_trait::async_trait;
use openrag_core::{
    AccessScope, Arn, ArnResourceType, AuthLevel, GuardAction, GuardDecision, GuardSurface,
    IamAction, OpenRagError, PipelineConfig, PipelineProfile, PrincipalContext,
};
use openrag_db::DbPool;
use openrag_iam::{
    GroupDefinition, GroupMembership, IamEvaluator, IamPolicy, PolicyEffect, PolicyStatement,
    ResourceDescriptor, ResourceGroupBinding,
};
use openrag_pii::{
    Capsule, CapsuleIamCtx, ConceptAliasLookup, GuardEngine, GuardPolicyRuntime, GuardRepo,
    MasterKek, PiiEngine, PiiVaultRepo, UnmaskResult,
};
use openrag_retrieve::{EvidenceAdmission, RetrieveSpace, SearchHit, SearchRequest};
use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sqlx::Row;
use uuid::Uuid;

use crate::auth::enforce_api_key_arn_scope;
use crate::state::{
    load_active_guard_policy, load_concept_lookup, non_empty_env, AppState, OpenRagMode,
};

#[async_trait]
pub trait LlmHttpTransport: Send + Sync {
    async fn post_json(
        &self,
        url: &str,
        bearer: Option<&str>,
        body: &Value,
    ) -> Result<(u16, String), OpenRagError>;
}

#[derive(Debug, Clone)]
pub struct ReqwestLlmTransport {
    client: reqwest::Client,
}

impl ReqwestLlmTransport {
    pub fn new(client: reqwest::Client) -> Self {
        Self { client }
    }
}

#[async_trait]
impl LlmHttpTransport for ReqwestLlmTransport {
    async fn post_json(
        &self,
        url: &str,
        bearer: Option<&str>,
        body: &Value,
    ) -> Result<(u16, String), OpenRagError> {
        let mut req = self.client.post(url).json(body);
        if let Some(key) = bearer {
            req = req.bearer_auth(key);
        }
        let resp = req.send().await.map_err(|e| OpenRagError::ExternalAdapter {
            adapter: "llm_http".into(),
            message: format!("HTTP call failed: {e}"),
        })?;
        let status = resp.status().as_u16();
        let text = resp.text().await.unwrap_or_default();
        Ok((status, text))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GatewayProvider {
    pub id: Uuid,
    pub org_id: Uuid,
    pub name: String,
    pub provider_type: String,
    pub base_url: String,
    pub model_id: String,
    pub is_on_premise: bool,
    pub max_auth_level: i16,
    pub allow_unmasked_pii: bool,
    pub config: Value,
    pub enabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GatewayRoutingRule {
    pub id: Uuid,
    pub org_id: Uuid,
    pub priority: i32,
    pub min_auth_level: i16,
    pub max_auth_level: i16,
    pub require_pii_masked: bool,
    pub target_provider_id: Uuid,
    pub fallback_provider_id: Option<Uuid>,
    pub enabled: bool,
}

#[derive(Clone)]
pub struct LlmCompletionInput<'a> {
    pub state: &'a AppState,
    pub principal: &'a PrincipalContext,
    pub requested_model: &'a str,
    pub user_prompt: &'a str,
    pub bucket_ref: Option<&'a str>,
    pub spaces: Option<Vec<RetrieveSpace>>,
    pub include_citations: bool,
    pub unmask_pii: bool,
    pub allow_unmasked_pii: bool,
    pub admission_threshold: Option<f32>,
    pub top_k: Option<usize>,
    /// Retrieve only documents in force at the reference time (`as_of`, default now).
    pub latest_revision_only: bool,
    pub as_of: Option<chrono::DateTime<chrono::Utc>>,
}

#[derive(Debug, Clone, Serialize)]
pub struct CitationItem {
    pub index: usize,
    pub document_id: Option<Uuid>,
    pub page_id: Option<Uuid>,
    pub space: String,
    pub title: Option<String>,
    pub snippet: String,
    pub score: f32,
    pub thumb_uri: Option<String>,
    pub tags: Vec<String>,
    /// Validity at the reference time: current, scheduled, expired, superseded, older_revision.
    pub validity: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct LlmUsage {
    pub prompt_tokens: i32,
    pub completion_tokens: i32,
    pub total_tokens: i32,
}

#[derive(Debug, Clone, Serialize)]
pub struct LlmCompletionResult {
    pub id: String,
    pub model: String,
    pub provider_id: Option<Uuid>,
    pub provider_name: String,
    pub is_on_premise: bool,
    pub max_context_auth_level: u8,
    pub pii_masked: bool,
    pub answer: String,
    pub citations: Vec<CitationItem>,
    pub admission: EvidenceAdmission,
    pub warnings: Vec<String>,
    pub usage: LlmUsage,
}

#[derive(Debug, Clone)]
pub struct PolicyEvalCtx {
    pub max_context_auth_level: u8,
    pub pii_masked: bool,
    pub must_be_on_premise: bool,
    pub requested_model: String,
}

pub async fn ensure_default_providers(
    worker_pool: &DbPool,
    org_id: Uuid,
) -> Result<(), OpenRagError> {
    let count: i64 =
        sqlx::query_scalar("SELECT count(*) FROM gateway.providers WHERE org_id = $1")
            .bind(org_id)
            .fetch_one(worker_pool.inner())
            .await
            .map_err(|e| {
                OpenRagError::Database(format!("failed counting gateway.providers: {e}"))
            })?;

    if count > 0 {
        return Ok(());
    }
    seed_providers(worker_pool, org_id, &configured_provider_seeds()).await
}

/// One `gateway.providers` row to create for an org.
#[derive(Debug, Clone)]
pub struct ProviderSeed {
    pub name: String,
    pub provider_type: String,
    pub base_url: String,
    pub model_id: String,
    pub is_on_premise: bool,
    pub max_auth_level: u8,
    pub allow_unmasked_pii: bool,
    pub config: Value,
}

/// Providers the environment configures. OpenRAG ships no default LLM: without these variables an
/// org gets no provider, `/v1/chat/completions` and `rag_ask` answer 503, and answers come from the
/// MCP client (Claude Code, Codex, ...) calling `rag_search`.
///
/// - `OPENRAG_LOCAL_LLM_URL` + `OPENRAG_LOCAL_LLM_MODEL`: an on-prem OpenAI-compatible chat
///   completions server (vLLM or any compatible one).
/// - `OPENRAG_LLM_CLI=claude|codex` (optional `OPENRAG_LLM_CLI_MODEL`, `OPENRAG_LLM_CLI_COMMAND`):
///   the Claude Code or Codex CLI on this host, already logged in. It sends the prompt to
///   Anthropic or OpenAI, so it counts as an external provider.
pub fn configured_provider_seeds() -> Vec<ProviderSeed> {
    let mut seeds = Vec::new();
    match (
        non_empty_env("OPENRAG_LOCAL_LLM_URL"),
        non_empty_env("OPENRAG_LOCAL_LLM_MODEL"),
    ) {
        (Some(url), Some(model)) => seeds.push(ProviderSeed {
            name: "vllm_local".into(),
            provider_type: "vllm_local".into(),
            base_url: url,
            model_id: model,
            is_on_premise: true,
            max_auth_level: 10,
            allow_unmasked_pii: true,
            config: json!({ "api_key_env": "VLLM_API_KEY" }),
        }),
        (None, None) => {}
        _ => tracing::warn!(
            "OPENRAG_LOCAL_LLM_URL and OPENRAG_LOCAL_LLM_MODEL must be set together; no local LLM provider"
        ),
    }
    if let Some(cli) = non_empty_env("OPENRAG_LLM_CLI") {
        match cli.trim() {
            kind @ ("claude" | "codex") => {
                let mut config = json!({});
                if let Some(command) = non_empty_env("OPENRAG_LLM_CLI_COMMAND") {
                    config["command"] = json!(command);
                }
                seeds.push(ProviderSeed {
                    name: format!("{kind}_cli"),
                    provider_type: format!("{kind}_cli"),
                    base_url: String::new(),
                    model_id: non_empty_env("OPENRAG_LLM_CLI_MODEL").unwrap_or_default(),
                    is_on_premise: false,
                    max_auth_level: 7,
                    allow_unmasked_pii: false,
                    config,
                });
            }
            other => tracing::warn!("OPENRAG_LLM_CLI={other} is not claude or codex; ignored"),
        }
    }
    seeds
}

/// Insert `seeds` for `org_id`, skipping names the org already has.
pub async fn seed_providers(
    worker_pool: &DbPool,
    org_id: Uuid,
    seeds: &[ProviderSeed],
) -> Result<(), OpenRagError> {
    for seed in seeds {
        sqlx::query(
            "INSERT INTO gateway.providers (
                org_id, name, provider_type, base_url, model_id,
                is_on_premise, max_auth_level, allow_unmasked_pii, config, enabled
             ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, true)
             ON CONFLICT (org_id, name) DO NOTHING",
        )
        .bind(org_id)
        .bind(&seed.name)
        .bind(&seed.provider_type)
        .bind(&seed.base_url)
        .bind(&seed.model_id)
        .bind(seed.is_on_premise)
        .bind(i16::from(seed.max_auth_level))
        .bind(seed.allow_unmasked_pii)
        .bind(&seed.config)
        .execute(worker_pool.inner())
        .await
        .map_err(|e| OpenRagError::Database(format!("failed seeding gateway.providers: {e}")))?;
    }
    Ok(())
}

pub async fn list_providers(
    worker_pool: &DbPool,
    org_id: Uuid,
) -> Result<Vec<GatewayProvider>, OpenRagError> {
    let rows = sqlx::query(
        "SELECT id, org_id, name, provider_type, base_url, model_id,
                is_on_premise, max_auth_level, allow_unmasked_pii, config, enabled
         FROM gateway.providers
         WHERE org_id = $1
         ORDER BY is_on_premise ASC, name ASC",
    )
    .bind(org_id)
    .fetch_all(worker_pool.inner())
    .await
    .map_err(|e| OpenRagError::Database(format!("failed listing gateway.providers: {e}")))?;

    rows.iter().map(row_to_provider).collect()
}

pub async fn list_routing_rules(
    worker_pool: &DbPool,
    org_id: Uuid,
) -> Result<Vec<GatewayRoutingRule>, OpenRagError> {
    let rows = sqlx::query(
        "SELECT id, org_id, priority, min_auth_level, max_auth_level,
                require_pii_masked, target_provider_id, fallback_provider_id, enabled
         FROM gateway.routing_rules
         WHERE org_id = $1 AND enabled = true
         ORDER BY priority ASC, id ASC",
    )
    .bind(org_id)
    .fetch_all(worker_pool.inner())
    .await
    .map_err(|e| OpenRagError::Database(format!("failed listing gateway.routing_rules: {e}")))?;

    rows.iter().map(row_to_rule).collect()
}

pub fn select_compliant_provider<'a>(
    providers: &'a [GatewayProvider],
    rules: &[GatewayRoutingRule],
    pipeline: &PipelineConfig,
    ctx: &PolicyEvalCtx,
) -> Option<&'a GatewayProvider> {
    let is_compliant = |p: &GatewayProvider| -> bool {
        p.enabled
            && p.max_auth_level >= i16::from(ctx.max_context_auth_level)
            && (!ctx.must_be_on_premise || p.is_on_premise)
            && (ctx.pii_masked || p.allow_unmasked_pii)
    };

    let mut sorted_rules: Vec<&GatewayRoutingRule> = rules.iter().collect();
    sorted_rules.sort_by_key(|r| (r.priority, r.id));

    for rule in sorted_rules {
        if !rule.enabled {
            continue;
        }
        let lvl = i16::from(ctx.max_context_auth_level);
        if lvl < rule.min_auth_level || lvl > rule.max_auth_level {
            continue;
        }
        if rule.require_pii_masked && !ctx.pii_masked {
            continue;
        }
        if let Some(p) = providers
            .iter()
            .find(|p| p.id == rule.target_provider_id && is_compliant(p))
        {
            return Some(p);
        }
        if let Some(fb_id) = rule.fallback_provider_id {
            if let Some(p) = providers
                .iter()
                .find(|p| p.id == fb_id && is_compliant(p))
            {
                return Some(p);
            }
        }
    }

    let compliant: Vec<&GatewayProvider> = providers.iter().filter(|p| is_compliant(p)).collect();
    if compliant.is_empty() {
        return None;
    }

    if ctx.must_be_on_premise {
        if let Some(p) = compliant
            .iter()
            .copied()
            .find(|p| p.name == pipeline.llm.local_provider || p.provider_type == "vllm_local")
        {
            return Some(p);
        }
        return compliant.into_iter().next();
    }

    let req_model = ctx.requested_model.trim();
    if !req_model.is_empty() && req_model != "openrag-auto" && req_model != "openrag-anything" {
        if let Some(p) = compliant
            .iter()
            .copied()
            .find(|p| p.model_id == req_model || p.name == req_model)
        {
            return Some(p);
        }
    }

    if let Some(p) = compliant
        .iter()
        .copied()
        .find(|p| p.name == pipeline.llm.external_provider)
    {
        return Some(p);
    }

    compliant.into_iter().next()
}

pub async fn run_chat_completion(
    input: LlmCompletionInput<'_>,
) -> Result<LlmCompletionResult, OpenRagError> {
    let started = Instant::now();
    let pipeline = input.state.pipeline_config.read().await.clone();
    let pii_engine = input.state.pii_engine.read().await.clone();
    let policy = load_active_guard_policy(
        input.state.worker_pool.as_ref(),
        input.principal.org_id,
        &pipeline,
    )
    .await?;
    let concept_lookup = load_concept_lookup(
        input.state.worker_pool.as_ref(),
        &input.state.in_memory_concepts,
        input.principal.org_id,
    )
    .await?;

    // Stage 1: InboundQuery Guard
    let inbound_eval = GuardEngine::evaluate(
        input.user_prompt,
        GuardSurface::InboundQuery,
        &policy,
        &pii_engine,
        Some(&concept_lookup),
        None,
    )?;

    if !inbound_eval.matched_rules.is_empty() {
        persist_guard_event(
            input.state,
            input.principal,
            policy.id,
            GuardSurface::InboundQuery,
            input.user_prompt,
            &inbound_eval,
        )
        .await?;
    }

    if inbound_eval.action == GuardAction::Block {
        return Err(build_guard_blocked_err(
            GuardSurface::InboundQuery,
            &inbound_eval,
        ));
    }

    let mut warnings = inbound_eval.warnings.clone();
    let sanitized_prompt = inbound_eval
        .sanitized_text
        .unwrap_or_else(|| input.user_prompt.to_string());

    // Stage 2: Retrieval (with RLS & bucket resolution)
    let (bucket_uuid, bucket_min_level) = match input.bucket_ref {
        Some(raw) if !raw.trim().is_empty() => {
            let (id, level, arn) =
                resolve_bucket_meta(input.state, input.principal, raw.trim()).await?;
            enforce_api_key_arn_scope(input.principal, &arn)?;
            (Some(id), level)
        }
        _ => (None, 1_u8),
    };

    let mut search_req = SearchRequest::new(&sanitized_prompt);
    search_req.bucket_id = bucket_uuid;
    search_req.spaces = input.spaces.clone();
    search_req.top_k = input.top_k;
    search_req.admission_threshold = input.admission_threshold;
    search_req.latest_revision_only = input.latest_revision_only;
    search_req.as_of = input.as_of;

    let search_resp = if input.state.app_pool.is_some() {
        let retriever = input.state.pg_retriever(input.principal.org_id).await?;
        retriever.search(input.principal, &search_req).await?
    } else {
        input
            .state
            .in_memory_retrieve
            .search(input.principal, &search_req)
            .await?
    };

    let mut max_context_auth_level = bucket_min_level;
    for hit in &search_resp.hits {
        if let Some(lvl) = hit.min_auth_level {
            if let Ok(u) = u8::try_from(lvl) {
                max_context_auth_level = max_context_auth_level.max(u.clamp(1, 10));
            }
        }
    }

    // Stage 3: Pre-seal LlmEgress Guard + PII Capsule masking vs unmasked handling
    let mut context_blocks = Vec::with_capacity(search_resp.hits.len());
    for (idx, hit) in search_resp.hits.iter().enumerate() {
        context_blocks.push(format!("[{}] {}", idx + 1, hit.text));
    }
    let assembled_context = context_blocks.join("\n");

    let raw_egress_payload = if assembled_context.is_empty() {
        sanitized_prompt.clone()
    } else {
        format!("{sanitized_prompt}\n{assembled_context}")
    };
    let pre_egress_eval = GuardEngine::evaluate(
        &raw_egress_payload,
        GuardSurface::LlmEgress,
        &policy,
        &pii_engine,
        Some(&concept_lookup),
        None,
    )?;
    if pre_egress_eval.action == GuardAction::Block {
        persist_guard_event(
            input.state,
            input.principal,
            policy.id,
            GuardSurface::LlmEgress,
            &raw_egress_payload,
            &pre_egress_eval,
        )
        .await?;
        return Err(build_guard_blocked_err(
            GuardSurface::LlmEgress,
            &pre_egress_eval,
        ));
    }

    let raw_prompt_spans = pii_engine.detect(&sanitized_prompt);
    let raw_context_spans = pii_engine.detect(&assembled_context);
    let has_raw_pii = !raw_prompt_spans.is_empty() || !raw_context_spans.is_empty();

    let mut effective_hits = search_resp.hits.clone();
    let pre_seal_hit_tokens: Vec<HashSet<String>> = effective_hits
        .iter()
        .map(extract_hit_pii_tokens)
        .collect();
    let (mut effective_prompt, mut effective_context, pii_masked) =
        if pipeline.pii.capsule_enabled && !input.allow_unmasked_pii {
            let masked_prompt = seal_text_for_llm(
                input.state,
                input.principal.org_id,
                &pii_engine,
                &sanitized_prompt,
            )
            .await?;
            for hit in &mut effective_hits {
                hit.text = seal_text_for_llm(
                    input.state,
                    input.principal.org_id,
                    &pii_engine,
                    &hit.text,
                )
                .await?;
                hit.snippet = seal_text_for_llm(
                    input.state,
                    input.principal.org_id,
                    &pii_engine,
                    &hit.snippet,
                )
                .await?;
            }
            let masked_context = rebuild_assembled_context(&effective_hits);
            (masked_prompt, masked_context, true)
        } else {
            let is_masked = !has_raw_pii && !input.allow_unmasked_pii;
            (sanitized_prompt.clone(), assembled_context, is_masked)
        };

    // Stage 4: Post-seal LlmEgress Guard (handles non-PII rules and unsealed Mask rules)
    let egress_payload = if effective_context.is_empty() {
        effective_prompt.clone()
    } else {
        format!("{effective_prompt}\n{effective_context}")
    };
    let egress_eval = GuardEngine::evaluate(
        &egress_payload,
        GuardSurface::LlmEgress,
        &policy,
        &pii_engine,
        Some(&concept_lookup),
        None,
    )?;

    if !egress_eval.matched_rules.is_empty() {
        persist_guard_event(
            input.state,
            input.principal,
            policy.id,
            GuardSurface::LlmEgress,
            &egress_payload,
            &egress_eval,
        )
        .await?;
    } else if !pre_egress_eval.matched_rules.is_empty() {
        persist_guard_event(
            input.state,
            input.principal,
            policy.id,
            GuardSurface::LlmEgress,
            &raw_egress_payload,
            &pre_egress_eval,
        )
        .await?;
    }

    if egress_eval.action == GuardAction::Block {
        return Err(build_guard_blocked_err(
            GuardSurface::LlmEgress,
            &egress_eval,
        ));
    }

    if egress_eval.action == GuardAction::Mask {
        let mask_ctx = EgressMaskCtx {
            policy: &policy,
            pii_engine: &pii_engine,
            concept_lookup: &concept_lookup,
        };
        effective_prompt = apply_egress_mask(&effective_prompt, &mask_ctx)?;
        for hit in &mut effective_hits {
            hit.text = apply_egress_mask(&hit.text, &mask_ctx)?;
            hit.snippet = apply_egress_mask(&hit.snippet, &mask_ctx)?;
        }
        effective_context = rebuild_assembled_context(&effective_hits);
    }

    let mut freshly_sealed_hit_tokens: HashSet<String> = HashSet::new();
    let mut preexisting_hit_tokens: Vec<HitPreexistingTokens> =
        Vec::with_capacity(effective_hits.len());
    for (idx, hit) in effective_hits.iter().enumerate() {
        let post_tokens = extract_hit_pii_tokens(hit);
        let pre_tokens = pre_seal_hit_tokens.get(idx);
        let mut hit_preexisting = HashSet::new();
        for tok in post_tokens {
            if pre_tokens.is_some_and(|pre| pre.contains(&tok)) {
                hit_preexisting.insert(tok);
            } else {
                freshly_sealed_hit_tokens.insert(tok);
            }
        }
        if !hit_preexisting.is_empty() {
            preexisting_hit_tokens.push(HitPreexistingTokens {
                space: hit.space,
                id: hit.id,
                page_id: hit.page_id,
                tokens: hit_preexisting,
            });
        }
    }

    let authorized_hit_tokens = if input.unmask_pii {
        resolve_authorized_hit_tokens(
            input.state.worker_pool.as_ref(),
            input.principal.org_id,
            &effective_hits,
            &freshly_sealed_hit_tokens,
            &preexisting_hit_tokens,
        )
        .await?
    } else {
        HashSet::new()
    };

    for w in &pre_egress_eval.warnings {
        if !warnings.contains(w) {
            warnings.push(w.clone());
        }
    }
    for w in &egress_eval.warnings {
        if !warnings.contains(w) {
            warnings.push(w.clone());
        }
    }

    // Stage 5: Policy-enforced provider routing
    let must_be_on_premise = pipeline.profile == PipelineProfile::Airgap
        || max_context_auth_level >= pipeline.llm.local_only_min_auth_level.as_u8()
        || (!pii_masked && pipeline.llm.require_pii_masked_for_external)
        || input.unmask_pii;

    let eval_ctx = PolicyEvalCtx {
        max_context_auth_level,
        pii_masked,
        must_be_on_premise,
        requested_model: input.requested_model.to_string(),
    };

    let selected_provider = if let Some(worker_pool) = input.state.worker_pool.as_ref() {
        ensure_default_providers(worker_pool, input.principal.org_id).await?;
        let providers = list_providers(worker_pool, input.principal.org_id).await?;
        let rules = list_routing_rules(worker_pool, input.principal.org_id).await?;
        let Some(provider) = select_compliant_provider(&providers, &rules, &pipeline, &eval_ctx)
        else {
            let any_enabled = providers.iter().any(|p| p.enabled);
            if any_enabled {
                return Err(OpenRagError::Forbidden(format!(
                    "LLM routing policy rejected request: no enabled provider satisfies max_context_auth_level={}, must_be_on_premise={}, pii_masked={}",
                    max_context_auth_level, must_be_on_premise, pii_masked
                )));
            }
            return Err(OpenRagError::ExternalAdapter {
                adapter: "llm_gateway".into(),
                message: "503 Service Unavailable: no enabled LLM providers configured".into(),
            });
        };
        Some(provider.clone())
    } else {
        None
    };

    let (provider_id, provider_name, provider_type, model_id, is_on_premise, base_url, cfg_json) =
        match &selected_provider {
            Some(p) => (
                Some(p.id),
                p.name.clone(),
                p.provider_type.clone(),
                p.model_id.clone(),
                p.is_on_premise,
                p.base_url.clone(),
                p.config.clone(),
            ),
            None => (
                None,
                if must_be_on_premise {
                    pipeline.llm.local_provider.clone()
                } else {
                    pipeline.llm.external_provider.clone()
                },
                "mock".to_string(),
                if must_be_on_premise {
                    pipeline.llm.local_model.clone()
                } else {
                    pipeline.llm.external_model.clone()
                },
                must_be_on_premise,
                String::new(),
                json!({}),
            ),
        };

    // Stage 6: Execute LLM call (live or deterministic offline mock)
    let citations = if input.include_citations {
        build_citations(&effective_hits)
    } else {
        Vec::new()
    };

    let mut answer = execute_provider_completion(
        input.state,
        &provider_type,
        &base_url,
        &model_id,
        &cfg_json,
        &effective_prompt,
        &effective_hits,
    )
    .await?;

    // Re-check egress guard on generated output
    let post_eval = GuardEngine::evaluate(
        &answer,
        GuardSurface::LlmEgress,
        &policy,
        &pii_engine,
        Some(&concept_lookup),
        None,
    )?;
    if post_eval.action == GuardAction::Block {
        persist_guard_event(
            input.state,
            input.principal,
            policy.id,
            GuardSurface::LlmEgress,
            &answer,
            &post_eval,
        )
        .await?;
        return Err(build_guard_blocked_err(GuardSurface::LlmEgress, &post_eval));
    }
    if let Some(sanitized_out) = post_eval.sanitized_text {
        answer = sanitized_out;
    }

    if input.unmask_pii {
        if effective_hits.is_empty() || authorized_hit_tokens.is_empty() {
            if contains_pii_tokens(&answer) {
                let reason = if effective_hits.is_empty() {
                    "unmask_denied_no_retrieved_hits"
                } else {
                    "unmask_denied_untrusted_token_not_in_hits"
                };
                if let Some(worker_pool) = input.state.worker_pool.as_ref() {
                    let _ = PiiVaultRepo::write_audit_log_worker(
                        worker_pool.inner(),
                        input.principal.org_id,
                        Some(input.principal.principal_id),
                        "pii:unmask",
                        None,
                        None,
                        None,
                        false,
                        Some(reason),
                    )
                    .await;
                }
                warnings.push(format!("pii_unmask_denied: {reason}"));
            }
        } else if let Some(worker_pool) = input.state.worker_pool.as_ref() {
            let unmasked = authorize_and_unmask_answer(UnmaskAnswerInput {
                worker_pool,
                principal: input.principal,
                answer: &answer,
                bucket_uuid,
                hits: &effective_hits,
                authorized_hit_tokens: &authorized_hit_tokens,
                master_kek: &input.state.master_kek,
            })
            .await?;
            if !unmasked.allowed && !unmasked.reason.is_empty() {
                warnings.push(format!("pii_unmask_denied: {}", unmasked.reason));
            }
            answer = unmasked.text;
        } else {
            let protected = protect_unauthorized_pii_tokens(&answer, &authorized_hit_tokens);
            let unmasked = if contains_pii_tokens(&protected.text) {
                Capsule::unmask_in_memory(
                    &protected.text,
                    input.principal,
                    None,
                    &input.state.master_kek,
                    &input.state.in_memory_vault,
                )?
            } else {
                UnmaskResult {
                    text: protected.text.clone(),
                    unmasked_count: 0,
                    denied_count: 0,
                    allowed: true,
                    reason: String::new(),
                }
            };
            if !protected.protected_tokens.is_empty() {
                warnings.push(
                    "pii_unmask_denied: unmask_denied_untrusted_token_not_in_hits".to_string(),
                );
            } else if !unmasked.allowed && !unmasked.reason.is_empty() {
                warnings.push(format!("pii_unmask_denied: {}", unmasked.reason));
            }
            answer = restore_protected_tokens(&unmasked.text, &protected.protected_tokens);
        }
    }

    let prompt_tokens = estimate_tokens(&effective_prompt) + estimate_tokens(&effective_context);
    let completion_tokens = estimate_tokens(&answer);
    let total_tokens = prompt_tokens + completion_tokens;
    let latency_ms = started.elapsed().as_millis().min(i32::MAX as u128) as i32;

    if let Some(worker_pool) = input.state.worker_pool.as_ref() {
        sqlx::query(
            "INSERT INTO gateway.request_logs (
                org_id, principal_id, provider_id, model_id,
                max_context_auth_level, pii_masked,
                prompt_tokens, completion_tokens, total_tokens,
                cost_usd, latency_ms, status
             ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, 0, $10, 'ok')",
        )
        .bind(input.principal.org_id)
        .bind(input.principal.principal_id)
        .bind(provider_id)
        .bind(&model_id)
        .bind(i16::from(max_context_auth_level))
        .bind(pii_masked)
        .bind(prompt_tokens)
        .bind(completion_tokens)
        .bind(total_tokens)
        .bind(latency_ms)
        .execute(worker_pool.inner())
        .await
        .map_err(|e| OpenRagError::Database(format!("failed logging gateway.request_logs: {e}")))?;
    }

    Ok(LlmCompletionResult {
        id: format!("chatcmpl-{}", Uuid::new_v4().simple()),
        model: model_id,
        provider_id,
        provider_name,
        is_on_premise,
        max_context_auth_level,
        pii_masked,
        answer,
        citations,
        admission: search_resp.admission,
        warnings,
        usage: LlmUsage {
            prompt_tokens,
            completion_tokens,
            total_tokens,
        },
    })
}

async fn resolve_bucket_meta(
    state: &AppState,
    principal: &PrincipalContext,
    bucket_ref: &str,
) -> Result<(Uuid, u8, String), OpenRagError> {
    let app_pool = state.require_app_pool()?;
    let mut tx = app_pool.begin_rls_tx(principal).await?;
    let parsed_uuid = Uuid::parse_str(bucket_ref).ok();
    let row = sqlx::query(
        "SELECT id, min_auth_level, arn
         FROM corpus.buckets
         WHERE org_id = $1 AND ($2::uuid IS NOT NULL AND id = $2 OR slug = $3)
         LIMIT 1",
    )
    .bind(principal.org_id)
    .bind(parsed_uuid)
    .bind(bucket_ref)
    .fetch_optional(&mut **tx.tx_mut())
    .await
    .map_err(|e| OpenRagError::Database(format!("failed resolving bucket: {e}")))?
    .ok_or_else(|| OpenRagError::NotFound(format!("bucket '{bucket_ref}' not found")))?;

    let id: Uuid = row
        .try_get("id")
        .map_err(|e| OpenRagError::Database(e.to_string()))?;
    let min_level: i16 = row
        .try_get("min_auth_level")
        .map_err(|e| OpenRagError::Database(e.to_string()))?;
    let arn: String = row
        .try_get("arn")
        .map_err(|e| OpenRagError::Database(e.to_string()))?;
    tx.commit()
        .await
        .map_err(|e| OpenRagError::Database(format!("commit bucket lookup: {e}")))?;
    Ok((id, u8::try_from(min_level).unwrap_or(1).clamp(1, 10), arn))
}

async fn seal_text_for_llm(
    state: &AppState,
    org_id: Uuid,
    pii_engine: &PiiEngine,
    text: &str,
) -> Result<String, OpenRagError> {
    if text.is_empty() {
        return Ok(String::new());
    }
    let spans = pii_engine.detect(text);
    if spans.is_empty() {
        return Ok(text.to_string());
    }
    if let Some(worker_pool) = state.worker_pool.as_ref() {
        let sealed = Capsule::mask_and_seal_pg(
            worker_pool.inner(),
            text,
            &spans,
            org_id,
            &state.master_kek,
        )
        .await?;
        Ok(sealed.masked_text)
    } else {
        let sealed = Capsule::mask_and_seal_in_memory(
            text,
            &spans,
            org_id,
            &state.master_kek,
            &state.in_memory_vault,
        )?;
        Ok(sealed.masked_text)
    }
}

pub(crate) async fn persist_guard_event(
    state: &AppState,
    principal: &PrincipalContext,
    policy_id: Option<Uuid>,
    surface: GuardSurface,
    input_text: &str,
    decision: &GuardDecision,
) -> Result<(), OpenRagError> {
    let Some(app_pool) = state.app_pool.as_ref() else {
        return Ok(());
    };
    let mut tx = app_pool.begin_rls_tx(principal).await?;
    let _ = GuardRepo::log_guard_event_rls(
        &mut tx,
        principal.org_id,
        policy_id,
        principal.principal_id,
        surface,
        input_text,
        decision,
    )
    .await?;
    tx.commit()
        .await
        .map_err(|e| OpenRagError::Database(format!("commit guard_events tx: {e}")))?;
    Ok(())
}

pub(crate) fn build_guard_blocked_err(
    surface: GuardSurface,
    decision: &GuardDecision,
) -> OpenRagError {
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
        surface,
        reason: format!(
            "blocked by {} guard rule(s) on {}",
            decision.matched_rules.len(),
            surface.as_str()
        ),
        matched_rule_ids: rule_ids,
        matched_categories: cats,
    }
}

async fn execute_provider_completion(
    state: &AppState,
    provider_type: &str,
    base_url: &str,
    model_id: &str,
    cfg_json: &Value,
    prompt: &str,
    hits: &[SearchHit],
) -> Result<String, OpenRagError> {
    let airgap = state.pipeline_config.read().await.profile == PipelineProfile::Airgap;
    if matches!(provider_type, "claude_cli" | "codex_cli") && state.mode != OpenRagMode::Offline {
        if airgap {
            return Err(OpenRagError::Forbidden(format!(
                "{provider_type} sends the prompt to Anthropic / OpenAI; not available in the air-gapped profile"
            )));
        }
        return crate::llm_cli::run_cli_completion(
            provider_type,
            model_id,
            cfg_json,
            &grounded_system_prompt(hits),
            prompt,
        )
        .await;
    }

    let api_key_env = cfg_json
        .get("api_key_env")
        .and_then(|v| v.as_str())
        .unwrap_or(match provider_type {
            "openrouter" => "OPENROUTER_API_KEY",
            "openai" => "OPENAI_API_KEY",
            "vllm_local" => "VLLM_API_KEY",
            _ => "",
        });
    let api_key = if api_key_env.is_empty() {
        None
    } else {
        non_empty_env(api_key_env)
    };

    let should_call_http = provider_type != "mock"
        && !base_url.trim().is_empty()
        && match state.mode {
            OpenRagMode::Offline => false,
            OpenRagMode::Live => true,
            OpenRagMode::Auto => {
                if provider_type == "vllm_local" {
                    non_empty_env("OPENRAG_VLLM_LIVE").as_deref() == Some("1")
                } else {
                    api_key.is_some()
                }
            }
        };

    if should_call_http {
        if state.mode == OpenRagMode::Live
            && matches!(provider_type, "openrouter" | "openai")
            && api_key.is_none()
        {
            return Err(OpenRagError::ExternalAdapter {
                adapter: provider_type.to_string(),
                message: format!(
                    "missing API key env '{api_key_env}' for live {provider_type} call"
                ),
            });
        }
        let sys_content = grounded_system_prompt(hits);
        let trimmed_base = base_url.trim_end_matches('/');
        let url = if trimmed_base.ends_with("/chat/completions") {
            trimmed_base.to_string()
        } else {
            format!("{trimmed_base}/chat/completions")
        };
        let req_body = json!({
            "model": model_id,
            "messages": [
                { "role": "system", "content": sys_content },
                { "role": "user", "content": prompt }
            ],
            "temperature": 0.1
        });
        if airgap {
            crate::egress::check(&*state.pipeline_config.read().await, &url)?;
        }
        let (status, body_text) = state
            .llm_transport
            .post_json(&url, api_key.as_deref(), &req_body)
            .await
            .map_err(|e| match e {
                OpenRagError::ExternalAdapter { message, .. } => OpenRagError::ExternalAdapter {
                    adapter: provider_type.to_string(),
                    message,
                },
                other => other,
            })?;
        if !(200..300).contains(&status) {
            return Err(OpenRagError::ExternalAdapter {
                adapter: provider_type.to_string(),
                message: format!("HTTP {status}: {body_text}"),
            });
        }
        let payload: Value =
            serde_json::from_str(&body_text).map_err(|e| OpenRagError::ExternalAdapter {
                adapter: provider_type.to_string(),
                message: format!("invalid JSON response: {e}"),
            })?;
        if let Some(content) = payload
            .get("choices")
            .and_then(|c| c.get(0))
            .and_then(|c0| c0.get("message"))
            .and_then(|m| m.get("content"))
            .and_then(|t| t.as_str())
        {
            return Ok(content.to_string());
        }
        return Err(OpenRagError::ExternalAdapter {
            adapter: provider_type.to_string(),
            message: "missing choices[0].message.content in provider response".into(),
        });
    }

    // Deterministic offline grounded synthesis
    if hits.is_empty() {
        return Ok(format!(
            "Grounded response ({model_id}): No matching indexed documents were found for \"{prompt}\"."
        ));
    }

    let mut lines = Vec::with_capacity(hits.len().min(4) + 1);
    lines.push(format!("Based on retrieved evidence ({model_id}):"));
    for (idx, hit) in hits.iter().take(4).enumerate() {
        let label = hit.title.as_deref().unwrap_or("Document");
        let excerpt: String = hit.snippet.chars().take(240).collect();
        lines.push(format!("[{}] {}: {}", idx + 1, label, excerpt));
    }
    Ok(lines.join("\n"))
}

/// System prompt carrying the top retrieved passages the answer must stay within.
fn grounded_system_prompt(hits: &[SearchHit]) -> String {
    let context_summary: String = hits
        .iter()
        .take(5)
        .enumerate()
        .map(|(i, h)| format!("[{}] {}", i + 1, h.text))
        .collect::<Vec<_>>()
        .join("\n\n");
    if context_summary.is_empty() {
        "You are OpenRAG enterprise assistant. Answer concisely and accurately.".to_string()
    } else {
        format!(
            "You are OpenRAG enterprise assistant. Answer using only the provided context:\n\n{context_summary}"
        )
    }
}

fn rebuild_assembled_context(hits: &[SearchHit]) -> String {
    let mut blocks = Vec::with_capacity(hits.len());
    for (idx, hit) in hits.iter().enumerate() {
        blocks.push(format!("[{}] {}", idx + 1, hit.text));
    }
    blocks.join("\n")
}

struct EgressMaskCtx<'a> {
    policy: &'a GuardPolicyRuntime,
    pii_engine: &'a PiiEngine,
    concept_lookup: &'a dyn ConceptAliasLookup,
}

fn apply_egress_mask(text: &str, ctx: &EgressMaskCtx<'_>) -> Result<String, OpenRagError> {
    if text.is_empty() {
        return Ok(String::new());
    }
    let decision = GuardEngine::evaluate(
        text,
        GuardSurface::LlmEgress,
        ctx.policy,
        ctx.pii_engine,
        Some(ctx.concept_lookup),
        None,
    )?;
    if decision.action == GuardAction::Mask {
        Ok(decision
            .sanitized_text
            .unwrap_or_else(|| text.to_string()))
    } else {
        Ok(text.to_string())
    }
}

struct IamUnmaskBundle {
    groups: Vec<GroupDefinition>,
    memberships: Vec<GroupMembership>,
    bindings: Vec<ResourceGroupBinding>,
    policies: Vec<IamPolicy>,
}

struct LoadedDescriptors {
    chunks: HashMap<Uuid, (ResourceDescriptor, Uuid)>,
    pages: HashMap<Uuid, (ResourceDescriptor, Uuid)>,
    documents: HashMap<Uuid, (ResourceDescriptor, Uuid)>,
    wikis: HashMap<Uuid, (ResourceDescriptor, Option<Uuid>)>,
    memories: HashMap<Uuid, (ResourceDescriptor, Option<Uuid>)>,
    buckets: HashMap<Uuid, ResourceDescriptor>,
}

fn pii_token_regex() -> Option<&'static Regex> {
    static RE: LazyLock<Option<Regex>> =
        LazyLock::new(|| Regex::new(r"(?i)\[PII:[A-Z0-9_]+:[0-9a-f]{4,32}\]").ok());
    RE.as_ref()
}

fn pii_token_capture_regex() -> Option<&'static Regex> {
    static RE: LazyLock<Option<Regex>> =
        LazyLock::new(|| Regex::new(r"(?i)\[PII:([A-Z0-9_]+:[0-9a-f]{4,32})\]").ok());
    RE.as_ref()
}

pub fn neutralize_literal_pii_tokens(text: &str) -> String {
    let Some(re) = pii_token_capture_regex() else {
        return text.to_string();
    };
    re.replace_all(text, "[PII-LITERAL:$1]").into_owned()
}

fn contains_pii_tokens(text: &str) -> bool {
    pii_token_regex().is_some_and(|re| re.is_match(text))
}

fn count_pii_tokens(text: &str) -> usize {
    pii_token_regex().map_or(0, |re| re.find_iter(text).count())
}

fn extract_hit_pii_tokens(hit: &SearchHit) -> HashSet<String> {
    let mut tokens = HashSet::new();
    let Some(re) = pii_token_regex() else {
        return tokens;
    };
    for m in re.find_iter(&hit.text) {
        tokens.insert(m.as_str().to_string());
    }
    for m in re.find_iter(&hit.snippet) {
        tokens.insert(m.as_str().to_string());
    }
    tokens
}

#[derive(Debug, Clone)]
struct HitPreexistingTokens {
    space: RetrieveSpace,
    id: Uuid,
    page_id: Option<Uuid>,
    tokens: HashSet<String>,
}

async fn resolve_authorized_hit_tokens(
    worker_pool: Option<&DbPool>,
    org_id: Uuid,
    effective_hits: &[SearchHit],
    freshly_sealed_hit_tokens: &HashSet<String>,
    preexisting_hit_tokens: &[HitPreexistingTokens],
) -> Result<HashSet<String>, OpenRagError> {
    let Some(worker_pool) = worker_pool else {
        return Ok(collect_hit_pii_tokens(effective_hits));
    };

    let mut authorized = freshly_sealed_hit_tokens.clone();
    let mut chunk_ids: Vec<Uuid> = Vec::new();
    let mut page_ids: Vec<Uuid> = Vec::new();

    for entry in preexisting_hit_tokens {
        match entry.space {
            RetrieveSpace::Chunks => {
                if !chunk_ids.contains(&entry.id) {
                    chunk_ids.push(entry.id);
                }
            }
            RetrieveSpace::Thumb => {
                let pid = entry.page_id.unwrap_or(entry.id);
                if !page_ids.contains(&pid) {
                    page_ids.push(pid);
                }
            }
            RetrieveSpace::Wiki | RetrieveSpace::Memory => {
                for tok in &entry.tokens {
                    authorized.insert(tok.clone());
                }
            }
        }
    }

    let mut chunk_bound_tokens: HashMap<Uuid, HashSet<String>> = HashMap::new();
    if !chunk_ids.is_empty() {
        let rows = sqlx::query(
            "SELECT cs.chunk_id, ve.token
             FROM pii.chunk_spans AS cs
             JOIN pii.vault_entries AS ve
               ON ve.id = cs.vault_entry_id AND ve.org_id = cs.org_id
             WHERE cs.org_id = $1 AND cs.chunk_id = ANY($2::uuid[])",
        )
        .bind(org_id)
        .bind(&chunk_ids)
        .fetch_all(worker_pool.inner())
        .await
        .map_err(|e| OpenRagError::Database(format!("load chunk token provenance: {e}")))?;
        for r in rows {
            let chunk_id: Uuid = r
                .try_get("chunk_id")
                .map_err(|e| OpenRagError::Database(e.to_string()))?;
            let token: String = r
                .try_get("token")
                .map_err(|e| OpenRagError::Database(e.to_string()))?;
            chunk_bound_tokens.entry(chunk_id).or_default().insert(token);
        }
    }

    let mut page_bound_tokens: HashMap<Uuid, HashSet<String>> = HashMap::new();
    if !page_ids.is_empty() {
        let rows = sqlx::query(
            "SELECT cs.page_id, ve.token
             FROM pii.chunk_spans AS cs
             JOIN pii.vault_entries AS ve
               ON ve.id = cs.vault_entry_id AND ve.org_id = cs.org_id
             WHERE cs.org_id = $1 AND cs.page_id = ANY($2::uuid[])",
        )
        .bind(org_id)
        .bind(&page_ids)
        .fetch_all(worker_pool.inner())
        .await
        .map_err(|e| OpenRagError::Database(format!("load page token provenance: {e}")))?;
        for r in rows {
            let page_id_opt: Option<Uuid> = r
                .try_get("page_id")
                .map_err(|e| OpenRagError::Database(e.to_string()))?;
            let token: String = r
                .try_get("token")
                .map_err(|e| OpenRagError::Database(e.to_string()))?;
            if let Some(pid) = page_id_opt {
                page_bound_tokens.entry(pid).or_default().insert(token);
            }
        }
    }

    for entry in preexisting_hit_tokens {
        match entry.space {
            RetrieveSpace::Chunks => {
                if let Some(bound) = chunk_bound_tokens.get(&entry.id) {
                    for tok in &entry.tokens {
                        if bound.contains(tok) {
                            authorized.insert(tok.clone());
                        }
                    }
                }
            }
            RetrieveSpace::Thumb => {
                let pid = entry.page_id.unwrap_or(entry.id);
                if let Some(bound) = page_bound_tokens.get(&pid) {
                    for tok in &entry.tokens {
                        if bound.contains(tok) {
                            authorized.insert(tok.clone());
                        }
                    }
                }
            }
            RetrieveSpace::Wiki | RetrieveSpace::Memory => {}
        }
    }

    Ok(authorized)
}

fn collect_hit_pii_tokens(hits: &[SearchHit]) -> HashSet<String> {
    let mut tokens = HashSet::new();
    for hit in hits {
        tokens.extend(extract_hit_pii_tokens(hit));
    }
    tokens
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ProtectedTokenEntry {
    placeholder: String,
    original: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ProtectedPiiText {
    text: String,
    protected_tokens: Vec<ProtectedTokenEntry>,
}

fn protect_unauthorized_pii_tokens(
    text: &str,
    authorized_tokens: &HashSet<String>,
) -> ProtectedPiiText {
    let Some(re) = pii_token_regex() else {
        return ProtectedPiiText {
            text: text.to_string(),
            protected_tokens: Vec::new(),
        };
    };

    let mut nonce = Uuid::new_v4().simple().to_string();
    while text.contains(&nonce) {
        nonce = Uuid::new_v4().simple().to_string();
    }

    let mut out = String::with_capacity(text.len());
    let mut last = 0usize;
    let mut protected_tokens = Vec::new();

    for m in re.find_iter(text) {
        let tok = m.as_str();
        if authorized_tokens.contains(tok) {
            continue;
        }
        out.push_str(&text[last..m.start()]);
        let placeholder = format!(
            "<<OPENRAG_PII_PROTECTED_{nonce}_{}>>",
            protected_tokens.len()
        );
        out.push_str(&placeholder);
        protected_tokens.push(ProtectedTokenEntry {
            placeholder,
            original: tok.to_string(),
        });
        last = m.end();
    }
    out.push_str(&text[last..]);

    ProtectedPiiText {
        text: out,
        protected_tokens,
    }
}

fn restore_protected_tokens(text: &str, protected_tokens: &[ProtectedTokenEntry]) -> String {
    if protected_tokens.is_empty() {
        return text.to_string();
    }
    let mut restored = text.to_string();
    for entry in protected_tokens {
        restored = restored.replace(&entry.placeholder, &entry.original);
    }
    restored
}

struct UnmaskAnswerInput<'a> {
    worker_pool: &'a DbPool,
    principal: &'a PrincipalContext,
    answer: &'a str,
    bucket_uuid: Option<Uuid>,
    hits: &'a [SearchHit],
    authorized_hit_tokens: &'a HashSet<String>,
    master_kek: &'a MasterKek,
}

async fn authorize_and_unmask_answer(
    input: UnmaskAnswerInput<'_>,
) -> Result<UnmaskResult, OpenRagError> {
    let UnmaskAnswerInput {
        worker_pool,
        principal,
        answer,
        bucket_uuid,
        hits,
        authorized_hit_tokens,
        master_kek,
    } = input;

    if hits.is_empty() || authorized_hit_tokens.is_empty() {
        let denied_count = count_pii_tokens(answer);
        if denied_count == 0 {
            return Ok(UnmaskResult {
                text: answer.to_string(),
                unmasked_count: 0,
                denied_count: 0,
                allowed: true,
                reason: String::new(),
            });
        }
        let reason = if hits.is_empty() {
            "unmask_denied_no_retrieved_hits"
        } else {
            "unmask_denied_untrusted_token_not_in_hits"
        };
        let _ = PiiVaultRepo::write_audit_log_worker(
            worker_pool.inner(),
            principal.org_id,
            Some(principal.principal_id),
            "pii:unmask",
            None,
            None,
            None,
            false,
            Some(reason),
        )
        .await;
        return Ok(UnmaskResult {
            text: answer.to_string(),
            unmasked_count: 0,
            denied_count,
            allowed: false,
            reason: reason.to_string(),
        });
    }

    let descriptors =
        load_resource_descriptors(worker_pool.inner(), principal.org_id, bucket_uuid, hits).await?;

    let mut resource_ids: HashSet<Uuid> = HashSet::new();
    for id in descriptors.chunks.keys() {
        resource_ids.insert(*id);
    }
    for id in descriptors.pages.keys() {
        resource_ids.insert(*id);
    }
    for id in descriptors.documents.keys() {
        resource_ids.insert(*id);
    }
    for id in descriptors.wikis.keys() {
        resource_ids.insert(*id);
    }
    for id in descriptors.memories.keys() {
        resource_ids.insert(*id);
    }
    for id in descriptors.buckets.keys() {
        resource_ids.insert(*id);
    }
    let all_resource_ids: Vec<Uuid> = resource_ids.into_iter().collect();

    let iam =
        load_iam_unmask_bundle(worker_pool.inner(), principal, &all_resource_ids).await?;

    let mut first_allowed: Option<&ResourceDescriptor> = None;
    let mut first_denied: Option<&ResourceDescriptor> = None;
    let mut missing_hit_id: Option<Uuid> = None;

    for hit in hits {
        let (hit_desc_opt, doc_desc_opt, hit_bucket_opt) = match hit.space {
            RetrieveSpace::Chunks => {
                let ch = descriptors.chunks.get(&hit.id);
                let doc_id = ch.map(|(_, d)| *d).or(hit.document_id);
                let doc_pair = doc_id.and_then(|d| descriptors.documents.get(&d));
                let bkt = doc_pair
                    .and_then(|(_, bid)| descriptors.buckets.get(bid))
                    .or_else(|| bucket_uuid.and_then(|bid| descriptors.buckets.get(&bid)));
                (ch.map(|(r, _)| r), doc_pair.map(|(r, _)| r), bkt)
            }
            RetrieveSpace::Thumb => {
                let pg = descriptors.pages.get(&hit.id);
                let doc_id = pg.map(|(_, d)| *d).or(hit.document_id);
                let doc_pair = doc_id.and_then(|d| descriptors.documents.get(&d));
                let bkt = doc_pair
                    .and_then(|(_, bid)| descriptors.buckets.get(bid))
                    .or_else(|| bucket_uuid.and_then(|bid| descriptors.buckets.get(&bid)));
                (pg.map(|(r, _)| r), doc_pair.map(|(r, _)| r), bkt)
            }
            RetrieveSpace::Wiki => {
                let wk = descriptors.wikis.get(&hit.id);
                let bkt = wk
                    .and_then(|(_, bid_opt)| bid_opt.and_then(|bid| descriptors.buckets.get(&bid)))
                    .or_else(|| bucket_uuid.and_then(|bid| descriptors.buckets.get(&bid)));
                (wk.map(|(r, _)| r), None, bkt)
            }
            RetrieveSpace::Memory => {
                let mem = descriptors.memories.get(&hit.id);
                let bkt = mem
                    .and_then(|(_, bid_opt)| bid_opt.and_then(|bid| descriptors.buckets.get(&bid)))
                    .or_else(|| bucket_uuid.and_then(|bid| descriptors.buckets.get(&bid)));
                (mem.map(|(r, _)| r), None, bkt)
            }
        };

        let Some(hit_desc) = hit_desc_opt else {
            missing_hit_id = Some(hit.id);
            break;
        };

        match evaluate_hit_unmask(principal, hit_desc, doc_desc_opt, hit_bucket_opt, &iam) {
            Ok(allowing_desc) => {
                if first_allowed.is_none() {
                    first_allowed = Some(allowing_desc);
                }
            }
            Err(denying_desc) => {
                first_denied = Some(denying_desc);
                break;
            }
        }
    }

    let synthetic_fallback: ResourceDescriptor;
    let selected_desc: &ResourceDescriptor = if let Some(missing_id) = missing_hit_id {
        synthetic_fallback = build_fail_closed_descriptor(principal, missing_id)?;
        &synthetic_fallback
    } else if let Some(denied) = first_denied {
        denied
    } else if let Some(allowed) = first_allowed {
        allowed
    } else {
        synthetic_fallback = build_org_policy_descriptor(worker_pool.inner(), principal).await?;
        &synthetic_fallback
    };

    let resource_arn = selected_desc.arn.to_canonical();
    let iam_ctx = CapsuleIamCtx {
        resource: selected_desc,
        groups: &iam.groups,
        memberships: &iam.memberships,
        bindings: &iam.bindings,
        policies: &iam.policies,
    };

    if missing_hit_id.is_some() || first_denied.is_some() {
        return Capsule::unmask_pg(
            worker_pool.inner(),
            answer,
            principal,
            Some(&iam_ctx),
            Some(&resource_arn),
            master_kek,
        )
        .await;
    }

    let protected = protect_unauthorized_pii_tokens(answer, authorized_hit_tokens);
    if !protected.protected_tokens.is_empty() {
        let _ = PiiVaultRepo::write_audit_log_worker(
            worker_pool.inner(),
            principal.org_id,
            Some(principal.principal_id),
            "pii:unmask",
            Some(&resource_arn),
            None,
            None,
            false,
            Some("unmask_denied_untrusted_token_not_in_hits"),
        )
        .await;
    }

    let res = if contains_pii_tokens(&protected.text) {
        Capsule::unmask_pg(
            worker_pool.inner(),
            &protected.text,
            principal,
            Some(&iam_ctx),
            Some(&resource_arn),
            master_kek,
        )
        .await?
    } else {
        UnmaskResult {
            text: protected.text.clone(),
            unmasked_count: 0,
            denied_count: 0,
            allowed: true,
            reason: String::new(),
        }
    };

    let restored_text = restore_protected_tokens(&res.text, &protected.protected_tokens);
    if !protected.protected_tokens.is_empty() {
        let total_denied = res.denied_count + protected.protected_tokens.len();
        let reason = if !res.allowed && !res.reason.is_empty() {
            res.reason
        } else {
            "unmask_denied_untrusted_token_not_in_hits".to_string()
        };
        Ok(UnmaskResult {
            text: restored_text,
            unmasked_count: res.unmasked_count,
            denied_count: total_denied,
            allowed: false,
            reason,
        })
    } else {
        Ok(UnmaskResult {
            text: restored_text,
            ..res
        })
    }
}

fn evaluate_hit_unmask<'a>(
    principal: &PrincipalContext,
    hit_desc: &'a ResourceDescriptor,
    doc_desc: Option<&'a ResourceDescriptor>,
    bucket_desc: Option<&'a ResourceDescriptor>,
    iam: &IamUnmaskBundle,
) -> Result<&'a ResourceDescriptor, &'a ResourceDescriptor> {
    let d_hit = IamEvaluator::evaluate(
        principal,
        hit_desc,
        IamAction::PiiUnmask,
        &iam.groups,
        &iam.memberships,
        &iam.bindings,
        &iam.policies,
    );
    let d_doc = doc_desc.map(|d| {
        IamEvaluator::evaluate(
            principal,
            d,
            IamAction::PiiUnmask,
            &iam.groups,
            &iam.memberships,
            &iam.bindings,
            &iam.policies,
        )
    });
    let d_bucket = bucket_desc.map(|b| {
        IamEvaluator::evaluate(
            principal,
            b,
            IamAction::PiiUnmask,
            &iam.groups,
            &iam.memberships,
            &iam.bindings,
            &iam.policies,
        )
    });

    let is_hard_deny = |dec: &openrag_iam::AccessDecision| -> bool {
        !dec.matched_denies.is_empty()
            || matches!(
                dec.reason.as_str(),
                "api_key_arn_scope_denied"
                    | "principal_cannot_unmask_pii"
                    | "insufficient_auth_level"
                    | "org_mismatch"
            )
    };

    if is_hard_deny(&d_hit) {
        return Err(hit_desc);
    }
    if let (Some(doc_d), Some(dec)) = (doc_desc, d_doc.as_ref()) {
        if is_hard_deny(dec) {
            return Err(doc_d);
        }
    }
    if let (Some(bkt_d), Some(dec)) = (bucket_desc, d_bucket.as_ref()) {
        if is_hard_deny(dec) {
            return Err(bkt_d);
        }
    }

    if hit_desc.access_scope == AccessScope::Private {
        return if d_hit.allowed {
            Ok(hit_desc)
        } else {
            Err(hit_desc)
        };
    }

    if d_hit.allowed {
        return Ok(hit_desc);
    }
    if let (Some(doc_d), Some(dec)) = (doc_desc, d_doc.as_ref()) {
        if dec.allowed {
            return Ok(doc_d);
        }
    }
    if let (Some(bkt_d), Some(dec)) = (bucket_desc, d_bucket.as_ref()) {
        if dec.allowed
            && dec
                .matched_allows
                .iter()
                .any(|m| !m.source.starts_with("scope_"))
        {
            return Ok(bkt_d);
        }
    }

    Err(hit_desc)
}

fn build_fail_closed_descriptor(
    principal: &PrincipalContext,
    resource_id: Uuid,
) -> Result<ResourceDescriptor, OpenRagError> {
    let arn = Arn::new(
        "org",
        "unknown",
        AccessScope::Private,
        ArnResourceType::Chunk,
        AuthLevel::MAX,
        resource_id.to_string(),
    )?;
    Ok(ResourceDescriptor {
        org_id: principal.org_id,
        resource_type: ArnResourceType::Chunk,
        resource_id,
        dept_id: Uuid::nil(),
        owner_principal_id: Uuid::nil(),
        access_scope: AccessScope::Private,
        allowed_group_ids: Vec::new(),
        denied_group_ids: Vec::new(),
        min_auth_level: AuthLevel::MAX,
        arn,
    })
}

async fn build_org_policy_descriptor(
    worker_pool: &sqlx::PgPool,
    principal: &PrincipalContext,
) -> Result<ResourceDescriptor, OpenRagError> {
    let slug: String = sqlx::query_scalar("SELECT slug FROM iam.orgs WHERE id = $1")
        .bind(principal.org_id)
        .fetch_optional(worker_pool)
        .await
        .map_err(|e| OpenRagError::Database(format!("load org slug for unmask: {e}")))?
        .unwrap_or_else(|| "org".to_string());
    let dept_id = principal.dept_ids.first().copied().unwrap_or(principal.org_id);
    let arn = Arn::new(
        &slug,
        "org",
        AccessScope::Org,
        ArnResourceType::Policy,
        AuthLevel::MIN,
        principal.org_id.to_string(),
    )?;
    Ok(ResourceDescriptor {
        org_id: principal.org_id,
        resource_type: ArnResourceType::Policy,
        resource_id: principal.org_id,
        dept_id,
        owner_principal_id: Uuid::nil(),
        access_scope: AccessScope::Org,
        allowed_group_ids: Vec::new(),
        denied_group_ids: Vec::new(),
        min_auth_level: AuthLevel::MIN,
        arn,
    })
}

async fn load_resource_descriptors(
    worker_pool: &sqlx::PgPool,
    org_id: Uuid,
    bucket_uuid: Option<Uuid>,
    hits: &[SearchHit],
) -> Result<LoadedDescriptors, OpenRagError> {
    let mut chunk_ids = Vec::new();
    let mut page_ids = Vec::new();
    let mut wiki_ids = Vec::new();
    let mut memory_ids = Vec::new();
    let mut doc_id_set: HashSet<Uuid> = HashSet::new();

    for hit in hits {
        match hit.space {
            RetrieveSpace::Chunks => {
                chunk_ids.push(hit.id);
                if let Some(did) = hit.document_id {
                    doc_id_set.insert(did);
                }
            }
            RetrieveSpace::Thumb => {
                page_ids.push(hit.id);
                if let Some(did) = hit.document_id {
                    doc_id_set.insert(did);
                }
            }
            RetrieveSpace::Wiki => wiki_ids.push(hit.id),
            RetrieveSpace::Memory => memory_ids.push(hit.id),
        }
    }

    let mut chunks = HashMap::new();
    if !chunk_ids.is_empty() {
        let rows = sqlx::query(
            "SELECT id, org_id, document_id, dept_id, owner_principal_id, access_scope,
                    allowed_group_ids, denied_group_ids, min_auth_level, arn
             FROM corpus.chunks
             WHERE org_id = $1 AND id = ANY($2::uuid[])",
        )
        .bind(org_id)
        .bind(&chunk_ids)
        .fetch_all(worker_pool)
        .await
        .map_err(|e| OpenRagError::Database(format!("load corpus.chunks for unmask: {e}")))?;
        for r in &rows {
            let desc = row_to_resource_descriptor(r, ArnResourceType::Chunk)?;
            let doc_id: Uuid = r
                .try_get("document_id")
                .map_err(|e| OpenRagError::Database(e.to_string()))?;
            doc_id_set.insert(doc_id);
            chunks.insert(desc.resource_id, (desc, doc_id));
        }
    }

    let mut pages = HashMap::new();
    if !page_ids.is_empty() {
        let rows = sqlx::query(
            "SELECT id, org_id, document_id, dept_id, owner_principal_id, access_scope,
                    allowed_group_ids, denied_group_ids, min_auth_level, arn
             FROM corpus.pages
             WHERE org_id = $1 AND id = ANY($2::uuid[])",
        )
        .bind(org_id)
        .bind(&page_ids)
        .fetch_all(worker_pool)
        .await
        .map_err(|e| OpenRagError::Database(format!("load corpus.pages for unmask: {e}")))?;
        for r in &rows {
            let desc = row_to_resource_descriptor(r, ArnResourceType::Page)?;
            let doc_id: Uuid = r
                .try_get("document_id")
                .map_err(|e| OpenRagError::Database(e.to_string()))?;
            doc_id_set.insert(doc_id);
            pages.insert(desc.resource_id, (desc, doc_id));
        }
    }

    let mut bucket_id_set: HashSet<Uuid> = HashSet::new();
    if let Some(bid) = bucket_uuid {
        bucket_id_set.insert(bid);
    }

    let doc_ids: Vec<Uuid> = doc_id_set.into_iter().collect();
    let mut documents = HashMap::new();
    if !doc_ids.is_empty() {
        let rows = sqlx::query(
            "SELECT id, org_id, bucket_id, dept_id, owner_principal_id, access_scope,
                    allowed_group_ids, denied_group_ids, min_auth_level, arn
             FROM corpus.documents
             WHERE org_id = $1 AND id = ANY($2::uuid[])",
        )
        .bind(org_id)
        .bind(&doc_ids)
        .fetch_all(worker_pool)
        .await
        .map_err(|e| OpenRagError::Database(format!("load corpus.documents for unmask: {e}")))?;
        for r in &rows {
            let desc = row_to_resource_descriptor(r, ArnResourceType::Document)?;
            let bucket_id: Uuid = r
                .try_get("bucket_id")
                .map_err(|e| OpenRagError::Database(e.to_string()))?;
            bucket_id_set.insert(bucket_id);
            documents.insert(desc.resource_id, (desc, bucket_id));
        }
    }

    let mut wikis = HashMap::new();
    if !wiki_ids.is_empty() {
        let rows = sqlx::query(
            "SELECT id, org_id, bucket_id, dept_id, owner_principal_id, access_scope,
                    allowed_group_ids, denied_group_ids, min_auth_level, arn
             FROM wiki.pages
             WHERE org_id = $1 AND id = ANY($2::uuid[])",
        )
        .bind(org_id)
        .bind(&wiki_ids)
        .fetch_all(worker_pool)
        .await
        .map_err(|e| OpenRagError::Database(format!("load wiki.pages for unmask: {e}")))?;
        for r in &rows {
            let desc = row_to_resource_descriptor(r, ArnResourceType::Wiki)?;
            let bucket_id: Option<Uuid> = r
                .try_get("bucket_id")
                .map_err(|e| OpenRagError::Database(e.to_string()))?;
            if let Some(bid) = bucket_id {
                bucket_id_set.insert(bid);
            }
            wikis.insert(desc.resource_id, (desc, bucket_id));
        }
    }

    let mut memories = HashMap::new();
    if !memory_ids.is_empty() {
        let rows = sqlx::query(
            "SELECT id, org_id, bucket_id, dept_id, owner_principal_id, access_scope,
                    allowed_group_ids, denied_group_ids, min_auth_level, arn
             FROM memory.entries
             WHERE org_id = $1 AND id = ANY($2::uuid[])",
        )
        .bind(org_id)
        .bind(&memory_ids)
        .fetch_all(worker_pool)
        .await
        .map_err(|e| OpenRagError::Database(format!("load memory.entries for unmask: {e}")))?;
        for r in &rows {
            let desc = row_to_resource_descriptor(r, ArnResourceType::Memory)?;
            let bucket_id: Option<Uuid> = r
                .try_get("bucket_id")
                .map_err(|e| OpenRagError::Database(e.to_string()))?;
            if let Some(bid) = bucket_id {
                bucket_id_set.insert(bid);
            }
            memories.insert(desc.resource_id, (desc, bucket_id));
        }
    }

    let bucket_ids: Vec<Uuid> = bucket_id_set.into_iter().collect();
    let mut buckets = HashMap::new();
    if !bucket_ids.is_empty() {
        let rows = sqlx::query(
            "SELECT id, org_id, dept_id, owner_principal_id, access_scope,
                    allowed_group_ids, denied_group_ids, min_auth_level, arn
             FROM corpus.buckets
             WHERE org_id = $1 AND id = ANY($2::uuid[])",
        )
        .bind(org_id)
        .bind(&bucket_ids)
        .fetch_all(worker_pool)
        .await
        .map_err(|e| OpenRagError::Database(format!("load corpus.buckets for unmask: {e}")))?;
        for r in &rows {
            let desc = row_to_resource_descriptor(r, ArnResourceType::Bucket)?;
            buckets.insert(desc.resource_id, desc);
        }
    }

    Ok(LoadedDescriptors {
        chunks,
        pages,
        documents,
        wikis,
        memories,
        buckets,
    })
}

async fn load_iam_unmask_bundle(
    worker_pool: &sqlx::PgPool,
    principal: &PrincipalContext,
    resource_ids: &[Uuid],
) -> Result<IamUnmaskBundle, OpenRagError> {
    let group_rows = sqlx::query(
        "SELECT id, org_id, name, min_auth_level, max_auth_level, allowed_actions
         FROM iam.groups
         WHERE org_id = $1",
    )
    .bind(principal.org_id)
    .fetch_all(worker_pool)
    .await
    .map_err(|e| OpenRagError::Database(format!("load iam.groups for unmask: {e}")))?;

    let mut groups = Vec::with_capacity(group_rows.len());
    for r in group_rows {
        let min_lvl: i16 = r
            .try_get("min_auth_level")
            .map_err(|e| OpenRagError::Database(e.to_string()))?;
        let max_lvl: i16 = r
            .try_get("max_auth_level")
            .map_err(|e| OpenRagError::Database(e.to_string()))?;
        let raw_actions: Vec<String> = r
            .try_get("allowed_actions")
            .map_err(|e| OpenRagError::Database(e.to_string()))?;
        let allowed_actions = raw_actions
            .iter()
            .filter_map(|s| IamAction::from_str(s).ok())
            .collect();
        groups.push(GroupDefinition {
            id: r.try_get("id").map_err(|e| OpenRagError::Database(e.to_string()))?,
            org_id: r
                .try_get("org_id")
                .map_err(|e| OpenRagError::Database(e.to_string()))?,
            name: r
                .try_get("name")
                .map_err(|e| OpenRagError::Database(e.to_string()))?,
            min_auth_level: AuthLevel::try_from(min_lvl)?,
            max_auth_level: AuthLevel::try_from(max_lvl)?,
            allowed_actions,
        });
    }

    let member_rows = sqlx::query(
        "SELECT group_id, principal_id, action_overrides, level_override
         FROM iam.group_members
         WHERE org_id = $1 AND principal_id = $2",
    )
    .bind(principal.org_id)
    .bind(principal.principal_id)
    .fetch_all(worker_pool)
    .await
    .map_err(|e| OpenRagError::Database(format!("load iam.group_members for unmask: {e}")))?;

    let mut memberships = Vec::with_capacity(member_rows.len());
    let mut deny_group_ids = principal.group_ids.clone();
    for r in member_rows {
        let group_id: Uuid = r
            .try_get("group_id")
            .map_err(|e| OpenRagError::Database(e.to_string()))?;
        let principal_id: Uuid = r
            .try_get("principal_id")
            .map_err(|e| OpenRagError::Database(e.to_string()))?;
        let raw_overrides: Option<Vec<String>> = r
            .try_get("action_overrides")
            .map_err(|e| OpenRagError::Database(e.to_string()))?;
        let lvl_ov: Option<i16> = r
            .try_get("level_override")
            .map_err(|e| OpenRagError::Database(e.to_string()))?;
        if !deny_group_ids.contains(&group_id) {
            deny_group_ids.push(group_id);
        }
        memberships.push(GroupMembership {
            group_id,
            principal_id,
            action_overrides: raw_overrides.map(|list| {
                list.iter()
                    .filter_map(|s| IamAction::from_str(s).ok())
                    .collect()
            }),
            level_override: lvl_ov.and_then(|v| AuthLevel::try_from(v).ok()),
        });
    }

    let mut bindings = Vec::new();
    if !resource_ids.is_empty() {
        let bind_rows = sqlx::query(
            "SELECT resource_type, resource_id, group_id, effect, actions, min_auth_level
             FROM iam.resource_group_bindings
             WHERE org_id = $1 AND resource_id = ANY($2::uuid[])",
        )
        .bind(principal.org_id)
        .bind(resource_ids)
        .fetch_all(worker_pool)
        .await
        .map_err(|e| {
            OpenRagError::Database(format!("load iam.resource_group_bindings for unmask: {e}"))
        })?;
        for r in bind_rows {
            let rt_str: String = r
                .try_get("resource_type")
                .map_err(|e| OpenRagError::Database(e.to_string()))?;
            let eff_str: String = r
                .try_get("effect")
                .map_err(|e| OpenRagError::Database(e.to_string()))?;
            let raw_actions: Vec<String> = r
                .try_get("actions")
                .map_err(|e| OpenRagError::Database(e.to_string()))?;
            let min_lvl: Option<i16> = r
                .try_get("min_auth_level")
                .map_err(|e| OpenRagError::Database(e.to_string()))?;
            let effect = match eff_str.as_str() {
                "allow" => PolicyEffect::Allow,
                "deny" => PolicyEffect::Deny,
                other => {
                    return Err(OpenRagError::Database(format!(
                        "invalid binding effect '{other}'"
                    )));
                }
            };
            bindings.push(ResourceGroupBinding {
                resource_type: ArnResourceType::from_str(&rt_str)?,
                resource_id: r
                    .try_get("resource_id")
                    .map_err(|e| OpenRagError::Database(e.to_string()))?,
                group_id: r
                    .try_get("group_id")
                    .map_err(|e| OpenRagError::Database(e.to_string()))?,
                effect,
                actions: raw_actions
                    .iter()
                    .filter_map(|s| IamAction::from_str(s).ok())
                    .collect(),
                min_auth_level: min_lvl.and_then(|v| AuthLevel::try_from(v).ok()),
            });
        }
    }

    let policy_rows = sqlx::query(
        "SELECT pol.id, pol.name, pol.document, pa.target_type, pa.target_id, pa.min_auth_level_override
         FROM iam.policy_attachments AS pa
         JOIN iam.policies AS pol ON pol.id = pa.policy_id AND pol.org_id = pa.org_id
         WHERE pa.org_id = $1
           AND (
               (pa.target_type = 'principal' AND pa.target_id = $2) OR
               (pa.target_type = 'dept' AND pa.target_id = ANY($3::uuid[])) OR
               (pa.target_type = 'group' AND pa.target_id = ANY($4::uuid[])) OR
               (pa.target_type = 'api_key' AND $5::uuid IS NOT NULL AND pa.target_id = $5)
           )",
    )
    .bind(principal.org_id)
    .bind(principal.principal_id)
    .bind(&principal.dept_ids)
    .bind(&deny_group_ids)
    .bind(principal.api_key_id)
    .fetch_all(worker_pool)
    .await
    .map_err(|e| OpenRagError::Database(format!("load iam.policies for unmask: {e}")))?;

    let mut policies = Vec::with_capacity(policy_rows.len());
    for r in policy_rows {
        let doc_val: Value = r
            .try_get("document")
            .map_err(|e| OpenRagError::Database(e.to_string()))?;
        let target_type: String = r
            .try_get("target_type")
            .map_err(|e| OpenRagError::Database(e.to_string()))?;
        let target_id: Uuid = r
            .try_get("target_id")
            .map_err(|e| OpenRagError::Database(e.to_string()))?;
        let min_ov: Option<i16> = r
            .try_get("min_auth_level_override")
            .map_err(|e| OpenRagError::Database(e.to_string()))?;

        let raw_stmts: Vec<PolicyStatement> = match doc_val.get("statements") {
            Some(arr) => serde_json::from_value(arr.clone()).unwrap_or_default(),
            None => Vec::new(),
        };
        let statements = if target_type == "group"
            && !principal.scoped_group_ids.is_empty()
            && !principal.scoped_group_ids.contains(&target_id)
        {
            raw_stmts
                .into_iter()
                .filter(|s| s.effect == PolicyEffect::Deny)
                .collect()
        } else {
            raw_stmts
        };

        policies.push(IamPolicy {
            id: r.try_get("id").map_err(|e| OpenRagError::Database(e.to_string()))?,
            name: r
                .try_get("name")
                .map_err(|e| OpenRagError::Database(e.to_string()))?,
            statements,
            min_auth_level_override: min_ov.and_then(|v| AuthLevel::try_from(v).ok()),
        });
    }

    Ok(IamUnmaskBundle {
        groups,
        memberships,
        bindings,
        policies,
    })
}

fn row_to_resource_descriptor(
    row: &sqlx::postgres::PgRow,
    resource_type: ArnResourceType,
) -> Result<ResourceDescriptor, OpenRagError> {
    let scope_str: String = row
        .try_get("access_scope")
        .map_err(|e| OpenRagError::Database(e.to_string()))?;
    let min_lvl: i16 = row
        .try_get("min_auth_level")
        .map_err(|e| OpenRagError::Database(e.to_string()))?;
    let arn_str: String = row
        .try_get("arn")
        .map_err(|e| OpenRagError::Database(e.to_string()))?;
    Ok(ResourceDescriptor {
        org_id: row
            .try_get("org_id")
            .map_err(|e| OpenRagError::Database(e.to_string()))?,
        resource_type,
        resource_id: row
            .try_get("id")
            .map_err(|e| OpenRagError::Database(e.to_string()))?,
        dept_id: row
            .try_get("dept_id")
            .map_err(|e| OpenRagError::Database(e.to_string()))?,
        owner_principal_id: row
            .try_get("owner_principal_id")
            .map_err(|e| OpenRagError::Database(e.to_string()))?,
        access_scope: AccessScope::from_str(&scope_str)?,
        allowed_group_ids: row
            .try_get("allowed_group_ids")
            .map_err(|e| OpenRagError::Database(e.to_string()))?,
        denied_group_ids: row
            .try_get("denied_group_ids")
            .map_err(|e| OpenRagError::Database(e.to_string()))?,
        min_auth_level: AuthLevel::try_from(min_lvl)?,
        arn: Arn::parse(&arn_str)?,
    })
}

fn build_citations(hits: &[SearchHit]) -> Vec<CitationItem> {
    hits.iter()
        .enumerate()
        .map(|(idx, h)| CitationItem {
            index: idx + 1,
            document_id: h.document_id,
            page_id: h.page_id,
            space: h.space.as_str().to_string(),
            title: h.title.clone(),
            snippet: h.snippet.clone(),
            score: h.score,
            thumb_uri: h.thumb_uri.clone(),
            tags: h.tags.clone(),
            validity: h.validity.as_str().to_string(),
        })
        .collect()
}

fn estimate_tokens(text: &str) -> i32 {
    let chars = text.chars().count();
    ((chars / 4).max(1)).min(i32::MAX as usize) as i32
}

fn row_to_provider(row: &sqlx::postgres::PgRow) -> Result<GatewayProvider, OpenRagError> {
    Ok(GatewayProvider {
        id: row
            .try_get("id")
            .map_err(|e| OpenRagError::Database(e.to_string()))?,
        org_id: row
            .try_get("org_id")
            .map_err(|e| OpenRagError::Database(e.to_string()))?,
        name: row
            .try_get("name")
            .map_err(|e| OpenRagError::Database(e.to_string()))?,
        provider_type: row
            .try_get("provider_type")
            .map_err(|e| OpenRagError::Database(e.to_string()))?,
        base_url: row
            .try_get("base_url")
            .map_err(|e| OpenRagError::Database(e.to_string()))?,
        model_id: row
            .try_get("model_id")
            .map_err(|e| OpenRagError::Database(e.to_string()))?,
        is_on_premise: row
            .try_get("is_on_premise")
            .map_err(|e| OpenRagError::Database(e.to_string()))?,
        max_auth_level: row
            .try_get("max_auth_level")
            .map_err(|e| OpenRagError::Database(e.to_string()))?,
        allow_unmasked_pii: row
            .try_get("allow_unmasked_pii")
            .map_err(|e| OpenRagError::Database(e.to_string()))?,
        config: row
            .try_get("config")
            .map_err(|e| OpenRagError::Database(e.to_string()))?,
        enabled: row
            .try_get("enabled")
            .map_err(|e| OpenRagError::Database(e.to_string()))?,
    })
}

fn row_to_rule(row: &sqlx::postgres::PgRow) -> Result<GatewayRoutingRule, OpenRagError> {
    Ok(GatewayRoutingRule {
        id: row
            .try_get("id")
            .map_err(|e| OpenRagError::Database(e.to_string()))?,
        org_id: row
            .try_get("org_id")
            .map_err(|e| OpenRagError::Database(e.to_string()))?,
        priority: row
            .try_get("priority")
            .map_err(|e| OpenRagError::Database(e.to_string()))?,
        min_auth_level: row
            .try_get("min_auth_level")
            .map_err(|e| OpenRagError::Database(e.to_string()))?,
        max_auth_level: row
            .try_get("max_auth_level")
            .map_err(|e| OpenRagError::Database(e.to_string()))?,
        require_pii_masked: row
            .try_get("require_pii_masked")
            .map_err(|e| OpenRagError::Database(e.to_string()))?,
        target_provider_id: row
            .try_get("target_provider_id")
            .map_err(|e| OpenRagError::Database(e.to_string()))?,
        fallback_provider_id: row
            .try_get("fallback_provider_id")
            .map_err(|e| OpenRagError::Database(e.to_string()))?,
        enabled: row
            .try_get("enabled")
            .map_err(|e| OpenRagError::Database(e.to_string()))?,
    })
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::state::default_pipeline_config;

    fn sample_providers(org_id: Uuid) -> Vec<GatewayProvider> {
        vec![
            GatewayProvider {
                id: Uuid::new_v4(),
                org_id,
                name: "openrouter".into(),
                provider_type: "openrouter".into(),
                base_url: "https://openrouter.ai/api/v1".into(),
                model_id: "anthropic/claude-3.5-sonnet".into(),
                is_on_premise: false,
                max_auth_level: 7,
                allow_unmasked_pii: false,
                config: json!({}),
                enabled: true,
            },
            GatewayProvider {
                id: Uuid::new_v4(),
                org_id,
                name: "vllm_local".into(),
                provider_type: "vllm_local".into(),
                base_url: "http://127.0.0.1:8000/v1".into(),
                model_id: "Qwen/Qwen2.5-32B-Instruct".into(),
                is_on_premise: true,
                max_auth_level: 10,
                allow_unmasked_pii: true,
                config: json!({}),
                enabled: true,
            },
        ]
    }

    #[test]
    fn routing_forces_l8_to_vllm_local_and_allows_l7_masked_external() {
        let org_id = Uuid::new_v4();
        let providers = sample_providers(org_id);
        let pipeline = default_pipeline_config();

        let l7_ctx = PolicyEvalCtx {
            max_context_auth_level: 7,
            pii_masked: true,
            must_be_on_premise: false,
            requested_model: "anthropic/claude-3.5-sonnet".into(),
        };
        let p7 = select_compliant_provider(&providers, &[], &pipeline, &l7_ctx);
        assert_eq!(p7.map(|p| p.name.as_str()), Some("openrouter"));

        let l8_ctx = PolicyEvalCtx {
            max_context_auth_level: 8,
            pii_masked: true,
            must_be_on_premise: true,
            requested_model: "anthropic/claude-3.5-sonnet".into(),
        };
        let p8 = select_compliant_provider(&providers, &[], &pipeline, &l8_ctx);
        assert_eq!(p8.map(|p| p.name.as_str()), Some("vllm_local"));
    }

    #[test]
    fn routing_blocks_when_only_external_available_for_unmasked_pii_or_l9() {
        let org_id = Uuid::new_v4();
        let mut providers = sample_providers(org_id);
        providers[1].enabled = false;
        let pipeline = default_pipeline_config();

        let unmasked_ctx = PolicyEvalCtx {
            max_context_auth_level: 3,
            pii_masked: false,
            must_be_on_premise: true,
            requested_model: "openrag-auto".into(),
        };
        assert!(select_compliant_provider(&providers, &[], &pipeline, &unmasked_ctx).is_none());

        let l9_ctx = PolicyEvalCtx {
            max_context_auth_level: 9,
            pii_masked: true,
            must_be_on_premise: true,
            requested_model: "openrag-auto".into(),
        };
        assert!(select_compliant_provider(&providers, &[], &pipeline, &l9_ctx).is_none());
    }

    #[test]
    fn routing_rules_respect_priority_and_tiebreak_even_when_unsorted() {
        let org_id = Uuid::new_v4();
        let mut providers = sample_providers(org_id);
        let fast_ext_id = Uuid::new_v4();
        let alt_ext_id = Uuid::new_v4();
        providers.push(GatewayProvider {
            id: fast_ext_id,
            org_id,
            name: "fast-external".into(),
            provider_type: "openai".into(),
            base_url: "https://api.openai.com/v1".into(),
            model_id: "gpt-4o-mini".into(),
            is_on_premise: false,
            max_auth_level: 7,
            allow_unmasked_pii: false,
            config: json!({}),
            enabled: true,
        });
        providers.push(GatewayProvider {
            id: alt_ext_id,
            org_id,
            name: "alt-external".into(),
            provider_type: "openai".into(),
            base_url: "https://api.openai.com/v1".into(),
            model_id: "gpt-4o".into(),
            is_on_premise: false,
            max_auth_level: 7,
            allow_unmasked_pii: false,
            config: json!({}),
            enabled: true,
        });
        let pipeline = default_pipeline_config();

        // Unsorted rules: priority 100 comes before priority 10
        let rules = vec![
            GatewayRoutingRule {
                id: Uuid::from_u128(10),
                org_id,
                priority: 100,
                min_auth_level: 1,
                max_auth_level: 10,
                require_pii_masked: true,
                target_provider_id: alt_ext_id,
                fallback_provider_id: None,
                enabled: true,
            },
            GatewayRoutingRule {
                id: Uuid::from_u128(20),
                org_id,
                priority: 10,
                min_auth_level: 1,
                max_auth_level: 10,
                require_pii_masked: true,
                target_provider_id: fast_ext_id,
                fallback_provider_id: None,
                enabled: true,
            },
        ];

        let ctx = PolicyEvalCtx {
            max_context_auth_level: 4,
            pii_masked: true,
            must_be_on_premise: false,
            requested_model: "openrag-auto".into(),
        };
        let picked = select_compliant_provider(&providers, &rules, &pipeline, &ctx);
        assert_eq!(picked.map(|p| p.id), Some(fast_ext_id));

        // Equal priority breaks deterministically by smaller Uuid id
        let low_id = Uuid::from_u128(1);
        let high_id = Uuid::from_u128(2);
        let tie_rules = vec![
            GatewayRoutingRule {
                id: high_id,
                org_id,
                priority: 50,
                min_auth_level: 1,
                max_auth_level: 10,
                require_pii_masked: true,
                target_provider_id: alt_ext_id,
                fallback_provider_id: None,
                enabled: true,
            },
            GatewayRoutingRule {
                id: low_id,
                org_id,
                priority: 50,
                min_auth_level: 1,
                max_auth_level: 10,
                require_pii_masked: true,
                target_provider_id: fast_ext_id,
                fallback_provider_id: None,
                enabled: true,
            },
        ];
        let tie_picked = select_compliant_provider(&providers, &tie_rules, &pipeline, &ctx);
        assert_eq!(tie_picked.map(|p| p.id), Some(fast_ext_id));
    }

    #[test]
    fn routing_rules_fallback_when_target_disabled_or_noncompliant() {
        let org_id = Uuid::new_v4();
        let mut providers = sample_providers(org_id);
        let l5_ext_id = Uuid::new_v4();
        let vllm_id = providers[1].id;
        providers.push(GatewayProvider {
            id: l5_ext_id,
            org_id,
            name: "l5-limited".into(),
            provider_type: "openai".into(),
            base_url: "https://api.openai.com/v1".into(),
            model_id: "gpt-4o".into(),
            is_on_premise: false,
            max_auth_level: 5,
            allow_unmasked_pii: false,
            config: json!({}),
            enabled: true,
        });
        let pipeline = default_pipeline_config();

        let rules = vec![GatewayRoutingRule {
            id: Uuid::new_v4(),
            org_id,
            priority: 1,
            min_auth_level: 1,
            max_auth_level: 10,
            require_pii_masked: true,
            target_provider_id: l5_ext_id,
            fallback_provider_id: Some(vllm_id),
            enabled: true,
        }];

        // Context L6 exceeds target max_auth_level=5 -> falls back to vllm_local
        let ctx = PolicyEvalCtx {
            max_context_auth_level: 6,
            pii_masked: true,
            must_be_on_premise: false,
            requested_model: "openrag-auto".into(),
        };
        let picked = select_compliant_provider(&providers, &rules, &pipeline, &ctx);
        assert_eq!(picked.map(|p| p.id), Some(vllm_id));

        // Disabled target also falls back to fallback_provider_id
        providers[2].enabled = false;
        let ctx_l3 = PolicyEvalCtx {
            max_context_auth_level: 3,
            pii_masked: true,
            must_be_on_premise: false,
            requested_model: "openrag-auto".into(),
        };
        let picked2 = select_compliant_provider(&providers, &rules, &pipeline, &ctx_l3);
        assert_eq!(picked2.map(|p| p.id), Some(vllm_id));
    }

    #[test]
    fn routing_rules_require_pii_masked_and_must_be_on_premise() {
        let org_id = Uuid::new_v4();
        let mut providers = sample_providers(org_id);
        let openrouter_id = providers[0].id;
        let vllm_id = providers[1].id;
        let ext2_id = Uuid::new_v4();
        providers.push(GatewayProvider {
            id: ext2_id,
            org_id,
            name: "openai-unmasked".into(),
            provider_type: "openai".into(),
            base_url: "https://api.openai.com/v1".into(),
            model_id: "gpt-4o".into(),
            is_on_premise: false,
            max_auth_level: 7,
            allow_unmasked_pii: true,
            config: json!({}),
            enabled: true,
        });
        let pipeline = default_pipeline_config();

        let rules = vec![GatewayRoutingRule {
            id: Uuid::new_v4(),
            org_id,
            priority: 1,
            min_auth_level: 1,
            max_auth_level: 10,
            require_pii_masked: true,
            target_provider_id: openrouter_id,
            fallback_provider_id: Some(ext2_id),
            enabled: true,
        }];

        // Unmasked PII skips rule with require_pii_masked=true and must_be_on_premise selects vllm_local
        let ctx_unmasked = PolicyEvalCtx {
            max_context_auth_level: 3,
            pii_masked: false,
            must_be_on_premise: true,
            requested_model: "openrag-auto".into(),
        };
        let picked = select_compliant_provider(&providers, &rules, &pipeline, &ctx_unmasked);
        assert_eq!(picked.map(|p| p.id), Some(vllm_id));

        // Even with pii_masked=true, must_be_on_premise rejects both external target and external fallback
        let ctx_onprem = PolicyEvalCtx {
            max_context_auth_level: 8,
            pii_masked: true,
            must_be_on_premise: true,
            requested_model: "openrag-auto".into(),
        };
        let picked_onprem = select_compliant_provider(&providers, &rules, &pipeline, &ctx_onprem);
        assert_eq!(picked_onprem.map(|p| p.id), Some(vllm_id));
    }

    #[derive(Debug)]
    struct RecordedHttpCall {
        url: String,
        bearer: Option<String>,
        body: Value,
    }

    struct MockLlmTransport {
        calls: std::sync::Mutex<Vec<RecordedHttpCall>>,
        outcome: Result<(u16, String), String>,
    }

    impl MockLlmTransport {
        fn ok(content: &str) -> Arc<Self> {
            Arc::new(Self {
                calls: std::sync::Mutex::new(Vec::new()),
                outcome: Ok((
                    200,
                    json!({"choices":[{"message":{"content": content}}]}).to_string(),
                )),
            })
        }

        fn raw_status(status: u16, body: &str) -> Arc<Self> {
            Arc::new(Self {
                calls: std::sync::Mutex::new(Vec::new()),
                outcome: Ok((status, body.to_string())),
            })
        }

        fn transport_err(msg: &str) -> Arc<Self> {
            Arc::new(Self {
                calls: std::sync::Mutex::new(Vec::new()),
                outcome: Err(msg.to_string()),
            })
        }
    }

    #[async_trait]
    impl LlmHttpTransport for MockLlmTransport {
        async fn post_json(
            &self,
            url: &str,
            bearer: Option<&str>,
            body: &Value,
        ) -> Result<(u16, String), OpenRagError> {
            if let Ok(mut guard) = self.calls.lock() {
                guard.push(RecordedHttpCall {
                    url: url.to_string(),
                    bearer: bearer.map(ToString::to_string),
                    body: body.clone(),
                });
            }
            match &self.outcome {
                Ok((status, txt)) => Ok((*status, txt.clone())),
                Err(msg) => Err(OpenRagError::ExternalAdapter {
                    adapter: "mock_transport".into(),
                    message: msg.clone(),
                }),
            }
        }
    }

    fn make_hit(idx: usize) -> SearchHit {
        SearchHit {
            id: Uuid::new_v4(),
            document_id: Some(Uuid::new_v4()),
            page_id: Some(Uuid::new_v4()),
            space: RetrieveSpace::Chunks,
            title: Some(format!("Doc {idx}")),
            snippet: format!("snippet-{idx}"),
            text: format!("hit-body-{idx}"),
            score: 0.9,
            vector_score: 0.8,
            bm25_score: 0.7,
            score_breakdown: Default::default(),
            tags: Vec::new(),
            min_auth_level: Some(1),
            revision: Some(1),
            valid_from: None,
            valid_to: None,
            thumb_uri: None,
            bbox_norm_1000: None,
            acl_scope_key: "scope".into(),
            validity: Default::default(),
        }
    }

    #[tokio::test]
    async fn execute_provider_completion_live_and_auto_with_mock_transport() -> anyhow::Result<()> {
        let base_state = AppState::new_offline_for_test()?;
        let hits: Vec<SearchHit> = (1..=7).map(make_hit).collect();

        // 1. Live openrouter with API key -> sends Bearer auth, limits system context to 5 hits
        let env_or = "OPENRAG_TEST_OR_KEY_1";
        std::env::set_var(env_or, "sk-or-test-123");
        let t_or = MockLlmTransport::ok("OR_OK");
        let state_or = base_state.with_mode_and_transport(OpenRagMode::Live, t_or.clone());
        let ans_or = execute_provider_completion(
            &state_or,
            "openrouter",
            "https://openrouter.ai/api/v1/",
            "anthropic/claude-3.5-sonnet",
            &json!({"api_key_env": env_or}),
            "summarize",
            &hits,
        )
        .await?;
        assert_eq!(ans_or, "OR_OK");
        {
            let calls = t_or
                .calls
                .lock()
                .map_err(|_| anyhow::anyhow!("mutex poisoned"))?;
            assert_eq!(calls.len(), 1);
            assert_eq!(calls[0].url, "https://openrouter.ai/api/v1/chat/completions");
            assert_eq!(calls[0].bearer.as_deref(), Some("sk-or-test-123"));
            let sys = calls[0].body["messages"][0]["content"]
                .as_str()
                .unwrap_or("");
            assert!(sys.contains("hit-body-5"));
            assert!(!sys.contains("hit-body-6"));
        }

        // 2. Live openai when base_url already ends with /chat/completions -> no double suffix
        let env_oai = "OPENRAG_TEST_OAI_KEY_1";
        std::env::set_var(env_oai, "sk-oai-test-456");
        let t_oai = MockLlmTransport::ok("OAI_OK");
        let state_oai = base_state.with_mode_and_transport(OpenRagMode::Live, t_oai.clone());
        let ans_oai = execute_provider_completion(
            &state_oai,
            "openai",
            "https://api.openai.com/v1/chat/completions",
            "gpt-4o",
            &json!({"api_key_env": env_oai}),
            "hello",
            &[],
        )
        .await?;
        assert_eq!(ans_oai, "OAI_OK");
        {
            let calls = t_oai
                .calls
                .lock()
                .map_err(|_| anyhow::anyhow!("mutex poisoned"))?;
            assert_eq!(calls[0].url, "https://api.openai.com/v1/chat/completions");
        }

        // 3. Live vllm_local without API key -> omits Bearer header
        let t_vllm = MockLlmTransport::ok("VLLM_OK");
        let state_vllm = base_state.with_mode_and_transport(OpenRagMode::Live, t_vllm.clone());
        let ans_vllm = execute_provider_completion(
            &state_vllm,
            "vllm_local",
            "http://127.0.0.1:8000/v1",
            "Qwen/Qwen2.5-32B-Instruct",
            &json!({"api_key_env": "OPENRAG_TEST_NONEXISTENT_VLLM_KEY"}),
            "hello vllm",
            &[],
        )
        .await?;
        assert_eq!(ans_vllm, "VLLM_OK");
        {
            let calls = t_vllm
                .calls
                .lock()
                .map_err(|_| anyhow::anyhow!("mutex poisoned"))?;
            assert_eq!(calls[0].bearer, None);
        }

        // 4. Live openrouter/openai with missing API key fails fast without HTTP call
        let t_missing = MockLlmTransport::ok("SHOULD_NOT_CALL");
        let state_missing =
            base_state.with_mode_and_transport(OpenRagMode::Live, t_missing.clone());
        let err_missing = execute_provider_completion(
            &state_missing,
            "openrouter",
            "https://openrouter.ai/api/v1",
            "anthropic/claude-3.5-sonnet",
            &json!({"api_key_env": "OPENRAG_TEST_MISSING_OR_KEY_XYZ"}),
            "hello",
            &[],
        )
        .await;
        assert!(matches!(err_missing, Err(OpenRagError::ExternalAdapter { .. })));
        {
            let calls = t_missing
                .calls
                .lock()
                .map_err(|_| anyhow::anyhow!("mutex poisoned"))?;
            assert!(calls.is_empty());
        }

        // 5. Auto mode with API key present invokes HTTP transport
        let t_auto = MockLlmTransport::ok("AUTO_OK");
        let state_auto = base_state.with_mode_and_transport(OpenRagMode::Auto, t_auto.clone());
        let ans_auto = execute_provider_completion(
            &state_auto,
            "openrouter",
            "https://openrouter.ai/api/v1",
            "anthropic/claude-3.5-sonnet",
            &json!({"api_key_env": env_or}),
            "auto test",
            &[],
        )
        .await?;
        assert_eq!(ans_auto, "AUTO_OK");

        Ok(())
    }

    #[tokio::test]
    async fn execute_provider_completion_maps_http_and_json_errors_to_external_adapter(
    ) -> anyhow::Result<()> {
        let base_state = AppState::new_offline_for_test()?;
        let env_k = "OPENRAG_TEST_ERR_KEY";
        std::env::set_var(env_k, "sk-err");

        for status in [401_u16, 429, 500] {
            let t = MockLlmTransport::raw_status(status, "upstream error");
            let state = base_state.with_mode_and_transport(OpenRagMode::Live, t);
            let res = execute_provider_completion(
                &state,
                "openai",
                "https://api.openai.com/v1",
                "gpt-4o",
                &json!({"api_key_env": env_k}),
                "err",
                &[],
            )
            .await;
            assert!(
                matches!(res, Err(OpenRagError::ExternalAdapter { .. })),
                "expected ExternalAdapter for HTTP {status}"
            );
        }

        for bad_body in ["not json", r#"{"choices":[]}"#, r#"{"choices":[{"message":{}}]}"#] {
            let t = MockLlmTransport::raw_status(200, bad_body);
            let state = base_state.with_mode_and_transport(OpenRagMode::Live, t);
            let res = execute_provider_completion(
                &state,
                "vllm_local",
                "http://127.0.0.1:8000/v1",
                "qwen",
                &json!({}),
                "err",
                &[],
            )
            .await;
            assert!(matches!(res, Err(OpenRagError::ExternalAdapter { .. })));
        }

        let t_net = MockLlmTransport::transport_err("connection refused");
        let state_net = base_state.with_mode_and_transport(OpenRagMode::Live, t_net);
        let res_net = execute_provider_completion(
            &state_net,
            "vllm_local",
            "http://127.0.0.1:8000/v1",
            "qwen",
            &json!({}),
            "err",
            &[],
        )
        .await;
        assert!(matches!(res_net, Err(OpenRagError::ExternalAdapter { .. })));

        Ok(())
    }

    #[tokio::test]
    async fn execute_provider_completion_offline_never_calls_http_transport() -> anyhow::Result<()>
    {
        let base_state = AppState::new_offline_for_test()?;
        let env_k = "OPENRAG_TEST_OFFLINE_KEY";
        std::env::set_var(env_k, "sk-offline");
        let t = MockLlmTransport::ok("SHOULD_NOT_BE_CALLED");
        let state = base_state.with_mode_and_transport(OpenRagMode::Offline, t.clone());

        let ans = execute_provider_completion(
            &state,
            "openrouter",
            "https://openrouter.ai/api/v1",
            "anthropic/claude-3.5-sonnet",
            &json!({"api_key_env": env_k}),
            "offline check",
            &[make_hit(1)],
        )
        .await?;
        assert!(ans.contains("Based on retrieved evidence"));
        let calls = t
            .calls
            .lock()
            .map_err(|_| anyhow::anyhow!("mutex poisoned"))?;
        assert!(calls.is_empty());
        Ok(())
    }

    #[tokio::test]
    async fn llm_egress_masks_prompt_and_hits_when_allow_unmasked_pii_true() -> anyhow::Result<()>
    {
        let state = AppState::new_offline_for_test()?;
        let org_id = Uuid::new_v4();
        let dept_id = Uuid::new_v4();
        let principal_id = Uuid::new_v4();
        {
            let mut cfg = state.pipeline_config.write().await;
            cfg.pii.guard.rules.push(openrag_core::GuardRuleConfig {
                kind: openrag_core::GuardRuleKind::PiiCategory,
                pattern_or_category: "email".into(),
                action: GuardAction::Mask,
                surfaces: vec![GuardSurface::LlmEgress],
                replacement_text: None,
                enabled: true,
                priority: 5,
            });
            cfg.pii.guard.rules.push(openrag_core::GuardRuleConfig {
                kind: openrag_core::GuardRuleKind::SafeWord,
                pattern_or_category: "ALPHA-MASK-TOKEN".into(),
                action: GuardAction::Mask,
                surfaces: vec![GuardSurface::LlmEgress],
                replacement_text: Some("[SAFE_MASKED]".into()),
                enabled: true,
                priority: 6,
            });
        }

        let mut principal =
            PrincipalContext::new(org_id, principal_id, AuthLevel::new(8)?);
        principal.dept_ids = vec![dept_id];
        principal.can_unmask_pii = true;

        let doc_id = Uuid::new_v4();
        state
            .in_memory_retrieve
            .backend()
            .insert(openrag_retrieve::InMemoryDocRecord {
                id: Uuid::new_v4(),
                document_id: Some(doc_id),
                page_id: None,
                org_id,
                bucket_id: None,
                dept_id,
                owner_principal_id: principal_id,
                access_scope: AccessScope::Dept,
                allowed_group_ids: Vec::new(),
                denied_group_ids: Vec::new(),
                min_auth_level: 3,
                space: RetrieveSpace::Chunks,
                title: Some("Egress Mask Doc".into()),
                text: "Contact bob@acme.test about ALPHA-MASK-TOKEN schedule".into(),
                tags: Vec::new(),
                document_tags: None,
                revision: 1,
                valid_from: chrono::Utc::now(),
                valid_to: None,
                is_archived: false,
                is_superseded: false,
                thumb_uri: None,
                bbox_norm_1000: None,
                acl_scope_key: "scope".into(),
                embedding: None,
            })
            .await;

        let res = run_chat_completion(LlmCompletionInput {
            state: &state,
            principal: &principal,
            requested_model: "openrag-auto",
            user_prompt: "schedule for alice@acme.test",
            bucket_ref: None,
            spaces: None,
            include_citations: true,
            unmask_pii: false,
            allow_unmasked_pii: true,
            admission_threshold: Some(0.0),
            top_k: Some(5),
            latest_revision_only: false,
            as_of: None,
        })
        .await?;

        assert!(!res.answer.contains("bob@acme.test"));
        assert!(!res.answer.contains("ALPHA-MASK-TOKEN"));
        assert!(res.answer.contains("[REDACTED:EMAIL]") || res.answer.contains("[SAFE_MASKED]"));
        assert!(!res.citations.is_empty());
        assert!(!res.citations[0].snippet.contains("bob@acme.test"));
        assert!(!res.citations[0].snippet.contains("ALPHA-MASK-TOKEN"));
        Ok(())
    }

    #[test]
    fn evaluate_hit_unmask_blocks_dept_admin_bucket_fallback_on_group_hit() -> anyhow::Result<()> {
        let org_id = Uuid::new_v4();
        let dept_id = Uuid::new_v4();
        let owner_id = Uuid::new_v4();
        let dept_admin_id = Uuid::new_v4();
        let group_id = Uuid::new_v4();
        let doc_id = Uuid::new_v4();
        let chunk_id = Uuid::new_v4();
        let bucket_id = Uuid::new_v4();

        let mut dept_admin =
            PrincipalContext::new(org_id, dept_admin_id, AuthLevel::new(8)?);
        dept_admin.dept_ids = vec![dept_id];
        dept_admin.dept_admin_ids = vec![dept_id];
        dept_admin.group_ids = vec![group_id];
        dept_admin.can_unmask_pii = true;

        let bucket_desc = ResourceDescriptor {
            org_id,
            resource_type: ArnResourceType::Bucket,
            resource_id: bucket_id,
            dept_id,
            owner_principal_id: owner_id,
            access_scope: AccessScope::Dept,
            allowed_group_ids: Vec::new(),
            denied_group_ids: Vec::new(),
            min_auth_level: AuthLevel::new(1)?,
            arn: Arn::new(
                "org",
                "dept1",
                AccessScope::Dept,
                ArnResourceType::Bucket,
                AuthLevel::new(1)?,
                bucket_id.to_string(),
            )?,
        };
        let doc_desc = ResourceDescriptor {
            org_id,
            resource_type: ArnResourceType::Document,
            resource_id: doc_id,
            dept_id,
            owner_principal_id: owner_id,
            access_scope: AccessScope::Group,
            allowed_group_ids: Vec::new(),
            denied_group_ids: Vec::new(),
            min_auth_level: AuthLevel::new(3)?,
            arn: Arn::new(
                "org",
                "dept1",
                AccessScope::Group,
                ArnResourceType::Document,
                AuthLevel::new(3)?,
                doc_id.to_string(),
            )?,
        };
        let hit_desc = ResourceDescriptor {
            org_id,
            resource_type: ArnResourceType::Chunk,
            resource_id: chunk_id,
            dept_id,
            owner_principal_id: owner_id,
            access_scope: AccessScope::Group,
            allowed_group_ids: Vec::new(),
            denied_group_ids: Vec::new(),
            min_auth_level: AuthLevel::new(3)?,
            arn: Arn::new(
                "org",
                "dept1",
                AccessScope::Group,
                ArnResourceType::Chunk,
                AuthLevel::new(3)?,
                chunk_id.to_string(),
            )?,
        };

        let iam = IamUnmaskBundle {
            groups: vec![GroupDefinition {
                id: group_id,
                org_id,
                name: "tf-read-only".into(),
                allowed_actions: vec![
                    IamAction::Read,
                    IamAction::Search,
                    IamAction::PiiUnmask,
                ],
                min_auth_level: AuthLevel::MIN,
                max_auth_level: AuthLevel::MAX,
            }],
            memberships: vec![GroupMembership {
                group_id,
                principal_id: dept_admin_id,
                action_overrides: None,
                level_override: None,
            }],
            bindings: vec![
                ResourceGroupBinding {
                    resource_type: ArnResourceType::Document,
                    resource_id: doc_id,
                    group_id,
                    effect: PolicyEffect::Allow,
                    actions: vec![IamAction::Read, IamAction::Search],
                    min_auth_level: None,
                },
                ResourceGroupBinding {
                    resource_type: ArnResourceType::Chunk,
                    resource_id: chunk_id,
                    group_id,
                    effect: PolicyEffect::Allow,
                    actions: vec![IamAction::Read, IamAction::Search],
                    min_auth_level: None,
                },
            ],
            policies: Vec::new(),
        };

        let res = evaluate_hit_unmask(
            &dept_admin,
            &hit_desc,
            Some(&doc_desc),
            Some(&bucket_desc),
            &iam,
        );
        assert!(
            res.is_err(),
            "dept_admin bucket scope_dept_admin must not override missing pii:unmask on group-scoped hit"
        );
        Ok(())
    }

    #[test]
    fn evaluate_hit_unmask_blocks_org_admin_bucket_fallback_on_group_hit() -> anyhow::Result<()> {
        let org_id = Uuid::new_v4();
        let dept_id = Uuid::new_v4();
        let owner_id = Uuid::new_v4();
        let org_admin_id = Uuid::new_v4();
        let group_id = Uuid::new_v4();
        let doc_id = Uuid::new_v4();
        let chunk_id = Uuid::new_v4();
        let bucket_id = Uuid::new_v4();

        let mut org_admin =
            PrincipalContext::new(org_id, org_admin_id, AuthLevel::new(8)?);
        org_admin.dept_ids = vec![dept_id];
        org_admin.group_ids = vec![group_id];
        org_admin.can_unmask_pii = true;
        org_admin.is_org_admin = true;

        let bucket_desc = ResourceDescriptor {
            org_id,
            resource_type: ArnResourceType::Bucket,
            resource_id: bucket_id,
            dept_id,
            owner_principal_id: owner_id,
            access_scope: AccessScope::Org,
            allowed_group_ids: Vec::new(),
            denied_group_ids: Vec::new(),
            min_auth_level: AuthLevel::new(1)?,
            arn: Arn::new(
                "org",
                "dept1",
                AccessScope::Org,
                ArnResourceType::Bucket,
                AuthLevel::new(1)?,
                bucket_id.to_string(),
            )?,
        };
        let doc_desc = ResourceDescriptor {
            org_id,
            resource_type: ArnResourceType::Document,
            resource_id: doc_id,
            dept_id,
            owner_principal_id: owner_id,
            access_scope: AccessScope::Group,
            allowed_group_ids: vec![group_id],
            denied_group_ids: Vec::new(),
            min_auth_level: AuthLevel::new(1)?,
            arn: Arn::new(
                "org",
                "dept1",
                AccessScope::Group,
                ArnResourceType::Document,
                AuthLevel::new(1)?,
                doc_id.to_string(),
            )?,
        };
        let hit_desc = ResourceDescriptor {
            org_id,
            resource_type: ArnResourceType::Chunk,
            resource_id: chunk_id,
            dept_id,
            owner_principal_id: owner_id,
            access_scope: AccessScope::Group,
            allowed_group_ids: vec![group_id],
            denied_group_ids: Vec::new(),
            min_auth_level: AuthLevel::new(1)?,
            arn: Arn::new(
                "org",
                "dept1",
                AccessScope::Group,
                ArnResourceType::Chunk,
                AuthLevel::new(1)?,
                chunk_id.to_string(),
            )?,
        };

        let iam = IamUnmaskBundle {
            groups: vec![GroupDefinition {
                id: group_id,
                org_id,
                name: "grp-ro".into(),
                allowed_actions: vec![IamAction::Read, IamAction::Search],
                min_auth_level: AuthLevel::MIN,
                max_auth_level: AuthLevel::MAX,
            }],
            memberships: vec![GroupMembership {
                group_id,
                principal_id: org_admin_id,
                action_overrides: None,
                level_override: None,
            }],
            bindings: Vec::new(),
            policies: Vec::new(),
        };

        let res = evaluate_hit_unmask(
            &org_admin,
            &hit_desc,
            Some(&doc_desc),
            Some(&bucket_desc),
            &iam,
        );
        assert!(
            res.is_err(),
            "org_admin bucket scope_org_admin must not override missing pii:unmask on group-scoped hit"
        );
        Ok(())
    }

    #[test]
    fn evaluate_hit_unmask_blocks_org_admin_bucket_fallback_on_dept_hit() -> anyhow::Result<()> {
        let org_id = Uuid::new_v4();
        let dept_a = Uuid::new_v4();
        let dept_b = Uuid::new_v4();
        let owner_id = Uuid::new_v4();
        let org_admin_id = Uuid::new_v4();
        let dept_b_admin_id = Uuid::new_v4();
        let dept_a_admin_id = Uuid::new_v4();
        let doc_id = Uuid::new_v4();
        let chunk_id = Uuid::new_v4();
        let org_bucket_id = Uuid::new_v4();
        let dept_b_bucket_id = Uuid::new_v4();
        let org_doc_id = Uuid::new_v4();
        let org_chunk_id = Uuid::new_v4();

        let mut org_admin =
            PrincipalContext::new(org_id, org_admin_id, AuthLevel::new(8)?);
        org_admin.dept_ids = vec![dept_a];
        org_admin.can_unmask_pii = true;
        org_admin.is_org_admin = true;

        let mut dept_b_admin =
            PrincipalContext::new(org_id, dept_b_admin_id, AuthLevel::new(8)?);
        dept_b_admin.dept_ids = vec![dept_a, dept_b];
        dept_b_admin.dept_admin_ids = vec![dept_b];
        dept_b_admin.can_unmask_pii = true;

        let mut dept_a_admin =
            PrincipalContext::new(org_id, dept_a_admin_id, AuthLevel::new(8)?);
        dept_a_admin.dept_ids = vec![dept_a];
        dept_a_admin.dept_admin_ids = vec![dept_a];
        dept_a_admin.can_unmask_pii = true;

        let org_bucket_desc = ResourceDescriptor {
            org_id,
            resource_type: ArnResourceType::Bucket,
            resource_id: org_bucket_id,
            dept_id: dept_a,
            owner_principal_id: owner_id,
            access_scope: AccessScope::Org,
            allowed_group_ids: Vec::new(),
            denied_group_ids: Vec::new(),
            min_auth_level: AuthLevel::new(1)?,
            arn: Arn::new(
                "org",
                "dept_a",
                AccessScope::Org,
                ArnResourceType::Bucket,
                AuthLevel::new(1)?,
                org_bucket_id.to_string(),
            )?,
        };
        let dept_b_bucket_desc = ResourceDescriptor {
            org_id,
            resource_type: ArnResourceType::Bucket,
            resource_id: dept_b_bucket_id,
            dept_id: dept_b,
            owner_principal_id: owner_id,
            access_scope: AccessScope::Dept,
            allowed_group_ids: Vec::new(),
            denied_group_ids: Vec::new(),
            min_auth_level: AuthLevel::new(1)?,
            arn: Arn::new(
                "org",
                "dept_b",
                AccessScope::Dept,
                ArnResourceType::Bucket,
                AuthLevel::new(1)?,
                dept_b_bucket_id.to_string(),
            )?,
        };
        let dept_a_doc_desc = ResourceDescriptor {
            org_id,
            resource_type: ArnResourceType::Document,
            resource_id: doc_id,
            dept_id: dept_a,
            owner_principal_id: owner_id,
            access_scope: AccessScope::Dept,
            allowed_group_ids: Vec::new(),
            denied_group_ids: Vec::new(),
            min_auth_level: AuthLevel::new(1)?,
            arn: Arn::new(
                "org",
                "dept_a",
                AccessScope::Dept,
                ArnResourceType::Document,
                AuthLevel::new(1)?,
                doc_id.to_string(),
            )?,
        };
        let dept_a_hit_desc = ResourceDescriptor {
            org_id,
            resource_type: ArnResourceType::Chunk,
            resource_id: chunk_id,
            dept_id: dept_a,
            owner_principal_id: owner_id,
            access_scope: AccessScope::Dept,
            allowed_group_ids: Vec::new(),
            denied_group_ids: Vec::new(),
            min_auth_level: AuthLevel::new(1)?,
            arn: Arn::new(
                "org",
                "dept_a",
                AccessScope::Dept,
                ArnResourceType::Chunk,
                AuthLevel::new(1)?,
                chunk_id.to_string(),
            )?,
        };
        let org_doc_desc = ResourceDescriptor {
            org_id,
            resource_type: ArnResourceType::Document,
            resource_id: org_doc_id,
            dept_id: dept_a,
            owner_principal_id: owner_id,
            access_scope: AccessScope::Org,
            allowed_group_ids: Vec::new(),
            denied_group_ids: Vec::new(),
            min_auth_level: AuthLevel::new(1)?,
            arn: Arn::new(
                "org",
                "dept_a",
                AccessScope::Org,
                ArnResourceType::Document,
                AuthLevel::new(1)?,
                org_doc_id.to_string(),
            )?,
        };
        let org_hit_desc = ResourceDescriptor {
            org_id,
            resource_type: ArnResourceType::Chunk,
            resource_id: org_chunk_id,
            dept_id: dept_a,
            owner_principal_id: owner_id,
            access_scope: AccessScope::Org,
            allowed_group_ids: Vec::new(),
            denied_group_ids: Vec::new(),
            min_auth_level: AuthLevel::new(1)?,
            arn: Arn::new(
                "org",
                "dept_a",
                AccessScope::Org,
                ArnResourceType::Chunk,
                AuthLevel::new(1)?,
                org_chunk_id.to_string(),
            )?,
        };

        let iam = IamUnmaskBundle {
            groups: Vec::new(),
            memberships: Vec::new(),
            bindings: Vec::new(),
            policies: Vec::new(),
        };

        // 1. org_admin (non-dept_admin, non-owner) on Dept-scoped hit inside Org-scoped bucket -> denied
        let res_org_admin_on_dept = evaluate_hit_unmask(
            &org_admin,
            &dept_a_hit_desc,
            Some(&dept_a_doc_desc),
            Some(&org_bucket_desc),
            &iam,
        );
        assert!(
            res_org_admin_on_dept.is_err(),
            "org_admin bucket scope_org_admin must not override Dept-scoped hit/doc"
        );

        // 2. dept_admin of dept_b on Dept-scoped hit in dept_a inside Dept-scoped bucket in dept_b -> denied
        let res_dept_b_admin_on_dept_a = evaluate_hit_unmask(
            &dept_b_admin,
            &dept_a_hit_desc,
            Some(&dept_a_doc_desc),
            Some(&dept_b_bucket_desc),
            &iam,
        );
        assert!(
            res_dept_b_admin_on_dept_a.is_err(),
            "dept_admin of dept_b must not override Dept-scoped hit in dept_a via dept_b bucket fallback"
        );

        // 3. Positive controls: dept_admin of dept_a on Dept-scoped hit -> allowed; org_admin on Org-scoped hit -> allowed
        let res_dept_a_admin = evaluate_hit_unmask(
            &dept_a_admin,
            &dept_a_hit_desc,
            Some(&dept_a_doc_desc),
            Some(&org_bucket_desc),
            &iam,
        );
        assert!(
            res_dept_a_admin.is_ok(),
            "dept_admin of dept_a must still unmask Dept-scoped hit in dept_a"
        );
        let res_org_admin_on_org = evaluate_hit_unmask(
            &org_admin,
            &org_hit_desc,
            Some(&org_doc_desc),
            Some(&org_bucket_desc),
            &iam,
        );
        assert!(
            res_org_admin_on_org.is_ok(),
            "org_admin must still unmask Org-scoped hit inside Org-scoped bucket"
        );
        Ok(())
    }

    #[test]
    fn protect_unauthorized_pii_tokens_protects_foreign_and_resists_placeholder_spoofing() {
        let mut authorized = HashSet::new();
        authorized.insert("[PII:EMAIL:aaaa1111]".to_string());

        let spoofed = "<<OPENRAG_PII_PROTECTED_00000000000000000000000000000000_0>>";
        let input = format!(
            "Allowed=[PII:EMAIL:aaaa1111] Injected=[PII:EMAIL:bbbb2222] Phone=[PII:PHONE_KR:cccc3333] Spoof={spoofed}"
        );

        let protected = protect_unauthorized_pii_tokens(&input, &authorized);
        assert_eq!(protected.protected_tokens.len(), 2);
        assert!(protected.text.contains("[PII:EMAIL:aaaa1111]"));
        assert!(!protected.text.contains("[PII:EMAIL:bbbb2222]"));
        assert!(!protected.text.contains("[PII:PHONE_KR:cccc3333]"));
        assert!(protected.text.contains(spoofed));
        assert_eq!(count_pii_tokens(&protected.text), 1);

        let after_capsule = protected
            .text
            .replace("[PII:EMAIL:aaaa1111]", "allowed@acme.test");
        let restored = restore_protected_tokens(&after_capsule, &protected.protected_tokens);
        assert!(restored.contains("Allowed=allowed@acme.test"));
        assert!(restored.contains("Injected=[PII:EMAIL:bbbb2222]"));
        assert!(restored.contains("Phone=[PII:PHONE_KR:cccc3333]"));
        assert!(restored.contains(spoofed));
    }

    #[tokio::test]
    async fn run_chat_completion_unmask_pii_blocks_prompt_echo_oracle_when_hits_empty_and_dummy_hits(
    ) -> anyhow::Result<()> {
        let base_state = AppState::new_offline_for_test()?;
        let org_id = Uuid::new_v4();
        let dept_id = Uuid::new_v4();
        let principal_id = Uuid::new_v4();

        let mut principal = PrincipalContext::new(org_id, principal_id, AuthLevel::new(8)?);
        principal.dept_ids = vec![dept_id];
        principal.can_unmask_pii = true;
        principal.is_org_admin = true;

        // Seal a foreign secret directly into the org's in-memory vault
        let pii_engine = base_state.pii_engine.read().await.clone();
        let secret_text = "Confidential oracle.secret@acme.test";
        let spans = pii_engine.detect(secret_text);
        let sealed = Capsule::mask_and_seal_in_memory(
            secret_text,
            &spans,
            org_id,
            &base_state.master_kek,
            &base_state.in_memory_vault,
        )?;
        let foreign_tok = sealed
            .tokens
            .first()
            .ok_or_else(|| anyhow::anyhow!("expected sealed email token"))?
            .token
            .clone();

        // 1. hits.is_empty() in Offline mode echoes user_prompt -> must NOT unmask foreign_tok
        let res_empty = run_chat_completion(LlmCompletionInput {
            state: &base_state,
            principal: &principal,
            requested_model: "openrag-auto",
            user_prompt: &format!("Explain {foreign_tok}"),
            bucket_ref: None,
            spaces: None,
            include_citations: true,
            unmask_pii: true,
            allow_unmasked_pii: false,
            admission_threshold: Some(0.0),
            top_k: Some(5),
            latest_revision_only: false,
            as_of: None,
        })
        .await?;
        assert!(!res_empty.answer.contains("oracle.secret@acme.test"));
        assert!(res_empty.answer.contains(&foreign_tok));
        assert!(res_empty
            .warnings
            .iter()
            .any(|w| w.contains("unmask_denied_no_retrieved_hits")));

        // 2. Harmless dummy hit (no PII in hits -> authorized_hit_tokens.is_empty())
        base_state
            .in_memory_retrieve
            .backend()
            .insert(openrag_retrieve::InMemoryDocRecord {
                id: Uuid::new_v4(),
                document_id: Some(Uuid::new_v4()),
                page_id: None,
                org_id,
                bucket_id: None,
                dept_id,
                owner_principal_id: principal_id,
                access_scope: AccessScope::Private,
                allowed_group_ids: Vec::new(),
                denied_group_ids: Vec::new(),
                min_auth_level: 1,
                space: RetrieveSpace::Chunks,
                title: Some("Dummy".into()),
                text: "DUMMYORACLE1 harmless text".into(),
                tags: Vec::new(),
                document_tags: None,
                revision: 1,
                valid_from: chrono::Utc::now(),
                valid_to: None,
                is_archived: false,
                is_superseded: false,
                thumb_uri: None,
                bbox_norm_1000: None,
                acl_scope_key: "scope".into(),
                embedding: None,
            })
            .await;

        let res_dummy = run_chat_completion(LlmCompletionInput {
            state: &base_state,
            principal: &principal,
            requested_model: "openrag-auto",
            user_prompt: &format!("DUMMYORACLE1 {foreign_tok}"),
            bucket_ref: None,
            spaces: None,
            include_citations: true,
            unmask_pii: true,
            allow_unmasked_pii: false,
            admission_threshold: Some(0.0),
            top_k: Some(5),
            latest_revision_only: false,
            as_of: None,
        })
        .await?;
        assert!(!res_dummy.answer.contains("oracle.secret@acme.test"));
        assert!(res_dummy.answer.contains("DUMMYORACLE1"));

        // 3. Direct Capsule::unmask_in_memory binding check:
        //    When answer contains both an authorized hit token and an injected foreign token,
        //    only the authorized hit token is unmasked while foreign_tok stays sealed verbatim.
        let own_spans = pii_engine.detect("OwnHit999 own.user@acme.test");
        let own_sealed = Capsule::mask_and_seal_in_memory(
            "OwnHit999 own.user@acme.test",
            &own_spans,
            org_id,
            &base_state.master_kek,
            &base_state.in_memory_vault,
        )?;
        let own_tok = own_sealed
            .tokens
            .first()
            .ok_or_else(|| anyhow::anyhow!("expected own sealed token"))?
            .token
            .clone();

        let mut authorized_hit_tokens = HashSet::new();
        authorized_hit_tokens.insert(own_tok.clone());

        let simulated_llm_answer = format!("Own={own_tok} Foreign={foreign_tok}");
        let protected = protect_unauthorized_pii_tokens(&simulated_llm_answer, &authorized_hit_tokens);
        assert_eq!(protected.protected_tokens.len(), 1);
        let unmasked = Capsule::unmask_in_memory(
            &protected.text,
            &principal,
            None,
            &base_state.master_kek,
            &base_state.in_memory_vault,
        )?;
        let restored = restore_protected_tokens(&unmasked.text, &protected.protected_tokens);
        assert!(restored.contains("Own=own.user@acme.test"));
        assert!(!restored.contains("oracle.secret@acme.test"));
        assert!(restored.contains(&format!("Foreign={foreign_tok}")));

        Ok(())
    }

    #[test]
    fn neutralize_literal_pii_tokens_rewrites_vault_tokens_and_is_idempotent() {
        let input = "Laundered [PII:EMAIL:6c64d00d] and [PII:PHONE_KR:a1b2c3d4ef] keep [SAFE:1234]";
        let once = neutralize_literal_pii_tokens(input);
        assert_eq!(
            once,
            "Laundered [PII-LITERAL:EMAIL:6c64d00d] and [PII-LITERAL:PHONE_KR:a1b2c3d4ef] keep [SAFE:1234]"
        );
        assert!(
            !contains_pii_tokens(&once),
            "neutralized string must not match pii_token_regex(): {once}"
        );
        let twice = neutralize_literal_pii_tokens(&once);
        assert_eq!(twice, once, "neutralize_literal_pii_tokens must be idempotent");
    }

    #[test]
    fn build_fail_closed_descriptor_denies_dept_admin_with_max_auth_level() -> anyhow::Result<()> {
        let org_id = Uuid::new_v4();
        let dept_id = Uuid::new_v4();
        let principal_id = Uuid::new_v4();
        let missing_resource_id = Uuid::new_v4();

        let mut dept_admin = PrincipalContext::new(org_id, principal_id, AuthLevel::MAX);
        dept_admin.dept_ids = vec![dept_id];
        dept_admin.dept_admin_ids = vec![dept_id];
        dept_admin.can_unmask_pii = true;
        dept_admin.is_org_admin = true;

        let desc = build_fail_closed_descriptor(&dept_admin, missing_resource_id)?;
        assert_eq!(desc.dept_id, Uuid::nil());
        assert_eq!(desc.owner_principal_id, Uuid::nil());
        assert_eq!(desc.access_scope, AccessScope::Private);
        assert_eq!(desc.min_auth_level, AuthLevel::MAX);

        let decision = IamEvaluator::evaluate(
            &dept_admin,
            &desc,
            IamAction::PiiUnmask,
            &[],
            &[],
            &[],
            &[],
        );
        assert!(
            !decision.allowed,
            "fail-closed descriptor must deny even dept_admin with AuthLevel::MAX, got {:?}",
            decision
        );
        assert_eq!(decision.reason, "private_scope_denied");
        Ok(())
    }
}
