use std::collections::{HashMap, HashSet};
use std::time::Instant;

use openrag_core::{GuardAction, GuardSurface, OpenRagError, PipelineConfig, PrincipalContext};
use openrag_pii::{
    Capsule, GuardEngine, GuardPolicyRuntime, InMemoryVaultStore, MasterKek, PiiEngine,
};
use openrag_retrieve::{EvidenceAdmission, HybridRetriever, RetrieveBackend, SearchRequest};
use uuid::Uuid;

use crate::golden::{EvalStore, NewBenchItemResult, NewBenchRun};
use crate::metrics::{
    check_arn_violation, check_pii_leak, extract_hit_doc_id, mrr, ndcg_at_k, recall_at_k,
};
use crate::optimizer::{default_pipeline_config, parse_pipeline_yaml_tolerant};
use crate::types::{
    BenchMetrics, BenchReport, BenchRunRequest, BenchRunSummary, BenchTrack, GoldenAxis,
    GoldenItem, GoldenSet, LeaderboardEntry, SecurityGateResult,
};

pub fn apply_egress_capsule(
    raw_text: &str,
    pipeline: &PipelineConfig,
    pii_engine: &PiiEngine,
) -> Result<String, OpenRagError> {
    if !pipeline.pii.capsule_enabled {
        return Ok(raw_text.to_string());
    }
    let spans = pii_engine.detect(raw_text);
    let masked = if spans.is_empty() {
        raw_text.to_string()
    } else {
        let kek = MasterKek::from_bytes([0x5a; 32]);
        let vault = InMemoryVaultStore::new();
        Capsule::mask_and_seal_in_memory(raw_text, &spans, Uuid::nil(), &kek, &vault)?
            .masked_text
    };
    let policy = GuardPolicyRuntime::from_guard_config(Uuid::nil(), &pipeline.pii.guard);
    let guard = GuardEngine::evaluate(
        &masked,
        GuardSurface::LlmEgress,
        &policy,
        pii_engine,
        None,
        None,
    )?;
    match guard.action {
        GuardAction::Block => Ok("[BLOCKED_BY_GUARD]".to_string()),
        GuardAction::Mask => Ok(guard.sanitized_text.unwrap_or(masked)),
        GuardAction::Allow | GuardAction::Warn => Ok(masked),
    }
}

pub fn token_overlap_score(expected: &str, actual: &str) -> f64 {
    let expected_tokens: HashSet<String> = expected
        .split_whitespace()
        .map(|s| {
            s.trim_matches(|c: char| !c.is_alphanumeric())
                .to_ascii_lowercase()
        })
        .filter(|s| !s.is_empty())
        .collect();
    if expected_tokens.is_empty() {
        return 1.0;
    }
    let actual_lower = actual.to_ascii_lowercase();
    let matched = expected_tokens
        .iter()
        .filter(|tok| actual_lower.contains(tok.as_str()))
        .count();
    (matched as f64) / (expected_tokens.len() as f64)
}

pub fn pararel_edge_overlap_score(edges: &serde_json::Value, context: &str) -> f64 {
    let mut terms = Vec::new();
    collect_json_strings(edges, &mut terms);
    let normalized: Vec<String> = terms
        .into_iter()
        .map(|t| t.trim().to_ascii_lowercase())
        .filter(|t| !t.is_empty())
        .collect();
    if normalized.is_empty() {
        return 1.0;
    }
    let ctx_lower = context.to_ascii_lowercase();
    let hits = normalized
        .iter()
        .filter(|t| ctx_lower.contains(t.as_str()))
        .count();
    (hits as f64) / (normalized.len() as f64)
}

fn collect_json_strings(val: &serde_json::Value, out: &mut Vec<String>) {
    match val {
        serde_json::Value::String(s) => out.push(s.clone()),
        serde_json::Value::Array(arr) => {
            for item in arr {
                collect_json_strings(item, out);
            }
        }
        serde_json::Value::Object(map) => {
            for (k, v) in map {
                if k == "src"
                    || k == "dst"
                    || k == "source"
                    || k == "target"
                    || k == "label"
                    || matches!(
                        v,
                        serde_json::Value::Array(_) | serde_json::Value::Object(_)
                    )
                {
                    collect_json_strings(v, out);
                }
            }
        }
        _ => {}
    }
}

pub struct BenchRunner;

impl BenchRunner {
    pub async fn run_all_tracks<S: EvalStore, B: RetrieveBackend>(
        store: &S,
        retriever: &HybridRetriever<B>,
        actor: &PrincipalContext,
        req: &BenchRunRequest,
        pipeline_override: Option<&PipelineConfig>,
    ) -> Result<BenchRunSummary, OpenRagError> {
        let gs = resolve_golden_set(store, actor.org_id, req).await?;
        let items = store.list_golden_items(actor.org_id, gs.id, false).await?;
        if items.is_empty() {
            return Err(OpenRagError::Validation(format!(
                "golden set '{}' has no non-rejected items to evaluate",
                gs.name
            )));
        }

        let pipeline = match (pipeline_override, req.pipeline_yaml.as_deref()) {
            (Some(p), _) => p.clone(),
            (None, Some(yaml)) => parse_pipeline_yaml_tolerant(yaml)?,
            (None, None) => default_pipeline_config(),
        };
        let pii_engine = PiiEngine::from_config(&pipeline.pii);

        let mut principal_cache: HashMap<Uuid, PrincipalContext> = HashMap::new();
        for item in &items {
            if let Some(pid) = item.test_principal_id {
                if let std::collections::hash_map::Entry::Vacant(entry) = principal_cache.entry(pid)
                {
                    let resolved = store
                        .resolve_principal(actor.org_id, pid)
                        .await?
                        .ok_or_else(|| {
                            OpenRagError::NotFound(format!(
                                "test_principal_id {pid} not found in org {}",
                                actor.org_id
                            ))
                        })?;
                    entry.insert(resolved);
                }
            }
        }

        let mut runs = Vec::with_capacity(BenchTrack::ALL.len());
        let mut all_item_results = Vec::new();
        let mut total_pii_leaks = 0_i32;
        let mut total_arn_violations = 0_i32;

        for track in BenchTrack::ALL {
            let outcome = evaluate_track(TrackEvalCtx {
                track,
                retriever,
                actor,
                principal_cache: &principal_cache,
                items: &items,
                bucket_id: req.bucket_id,
                pipeline: &pipeline,
                pii_engine: &pii_engine,
            })
            .await?;

            let metrics_json = serde_json::to_value(&outcome.metrics)
                .map_err(|e| OpenRagError::Internal(format!("serialize bench metrics: {e}")))?;

            let run_record = store
                .insert_bench_run(
                    &NewBenchRun {
                        org_id: actor.org_id,
                        golden_set_id: gs.id,
                        pipeline_config_id: req.pipeline_config_id,
                        trial_id: req.trial_id,
                        track,
                        metrics: metrics_json,
                        pii_leak_count: outcome.pii_leak_count,
                        arn_violation_count: outcome.arn_violation_count,
                    },
                    &outcome.item_results,
                )
                .await?;

            total_pii_leaks += outcome.pii_leak_count;
            total_arn_violations += outcome.arn_violation_count;

            let persisted_items = store
                .list_bench_item_results(actor.org_id, run_record.id)
                .await?;
            all_item_results.extend(persisted_items);
            runs.push(run_record);
        }

        Ok(BenchRunSummary {
            golden_set: gs,
            runs,
            item_results: all_item_results,
            security_gates: SecurityGateResult::new(total_pii_leaks, total_arn_violations),
        })
    }

    pub async fn leaderboard<S: EvalStore>(
        store: &S,
        org_id: Uuid,
    ) -> Result<Vec<LeaderboardEntry>, OpenRagError> {
        let runs = store.list_bench_runs(org_id, None).await?;
        let sets = store.list_golden_sets(org_id).await?;
        let set_names: HashMap<Uuid, String> =
            sets.into_iter().map(|gs| (gs.id, gs.name)).collect();

        let mut entries: Vec<LeaderboardEntry> = runs
            .into_iter()
            .map(|r| {
                let mrr_val = metric_f64(&r.metrics, "mrr");
                let recall_val = metric_f64(&r.metrics, "recall_at_10")
                    .max(metric_f64(&r.metrics, "recall"));
                let ndcg_val =
                    metric_f64(&r.metrics, "ndcg_at_10").max(metric_f64(&r.metrics, "ndcg"));
                let latency_val = metric_f64(&r.metrics, "avg_latency_ms");
                let gs_name = set_names
                    .get(&r.golden_set_id)
                    .cloned()
                    .unwrap_or_else(|| r.golden_set_id.to_string());
                LeaderboardEntry {
                    bench_run_id: r.id,
                    golden_set_id: r.golden_set_id,
                    golden_set_name: gs_name,
                    pipeline_config_id: r.pipeline_config_id,
                    track: r.track,
                    mrr: mrr_val,
                    recall_at_10: recall_val,
                    ndcg_at_10: ndcg_val,
                    avg_latency_ms: latency_val,
                    pii_leak_count: r.pii_leak_count,
                    arn_violation_count: r.arn_violation_count,
                    security_gate_pass: r.security_gate_pass,
                    created_at: r.created_at,
                }
            })
            .collect();

        entries.sort_by(|a, b| {
            b.security_gate_pass
                .cmp(&a.security_gate_pass)
                .then_with(|| b.ndcg_at_10.total_cmp(&a.ndcg_at_10))
                .then_with(|| b.created_at.cmp(&a.created_at))
        });

        Ok(entries)
    }

    pub async fn generate_poc_report<S: EvalStore>(
        store: &S,
        org_id: Uuid,
        golden_set_id: Option<Uuid>,
    ) -> Result<BenchReport, OpenRagError> {
        let golden_set = match golden_set_id {
            Some(id) => Some(store.get_golden_set(org_id, id).await?),
            None => None,
        };
        let mut entries = Self::leaderboard(store, org_id).await?;
        if let Some(gs_id) = golden_set_id {
            entries.retain(|e| e.golden_set_id == gs_id);
        }

        let overall_pass = !entries.is_empty() && entries.iter().all(|e| e.security_gate_pass);
        let mut md = String::from("# OpenRAG PoC Benchmark Report\n\n");
        if let Some(gs) = &golden_set {
            md.push_str(&format!(
                "- Golden Set: `{}` ({})\n- Security Gate Pass: `{}`\n\n",
                gs.name, gs.axis, overall_pass
            ));
        } else {
            md.push_str(&format!(
                "- Org ID: `{org_id}`\n- Overall Security Gate Pass: `{overall_pass}`\n\n"
            ));
        }
        md.push_str("| Golden Set | Track | MRR | Recall@10 | NDCG@10 | Latency (ms) | PII Leaks | ARN Violations | Gate |\n");
        md.push_str("|---|---|---|---|---|---|---|---|---|\n");
        for e in &entries {
            md.push_str(&format!(
                "| {} | {} | {:.4} | {:.4} | {:.4} | {:.2} | {} | {} | {} |\n",
                e.golden_set_name,
                e.track,
                e.mrr,
                e.recall_at_10,
                e.ndcg_at_10,
                e.avg_latency_ms,
                e.pii_leak_count,
                e.arn_violation_count,
                if e.security_gate_pass { "PASS" } else { "FAIL" }
            ));
        }

        Ok(BenchReport {
            org_id,
            golden_set,
            entries,
            overall_security_gate_pass: overall_pass,
            markdown: md,
        })
    }
}

fn metric_f64(val: &serde_json::Value, key: &str) -> f64 {
    val.get(key)
        .and_then(|v| v.as_f64())
        .filter(|n| n.is_finite())
        .unwrap_or(0.0)
}

async fn resolve_golden_set<S: EvalStore>(
    store: &S,
    org_id: Uuid,
    req: &BenchRunRequest,
) -> Result<GoldenSet, OpenRagError> {
    if let Some(id) = req.golden_set_id {
        return store.get_golden_set(org_id, id).await;
    }
    if let Some(raw_name) = req.golden_set.as_deref() {
        let trimmed = raw_name.trim();
        if trimmed.is_empty() {
            return Err(OpenRagError::Validation(
                "golden_set name must not be empty".into(),
            ));
        }
        if let Ok(uuid) = Uuid::parse_str(trimmed) {
            if let Ok(gs) = store.get_golden_set(org_id, uuid).await {
                return Ok(gs);
            }
        }
        return store.get_golden_set_by_name(org_id, trimmed).await;
    }
    Err(OpenRagError::Validation(
        "golden_set_id or golden_set is required".into(),
    ))
}

struct TrackEvalCtx<'a, B: RetrieveBackend> {
    track: BenchTrack,
    retriever: &'a HybridRetriever<B>,
    actor: &'a PrincipalContext,
    principal_cache: &'a HashMap<Uuid, PrincipalContext>,
    items: &'a [GoldenItem],
    bucket_id: Option<Uuid>,
    pipeline: &'a PipelineConfig,
    pii_engine: &'a PiiEngine,
}

struct TrackEvalOutcome {
    metrics: BenchMetrics,
    item_results: Vec<NewBenchItemResult>,
    pii_leak_count: i32,
    arn_violation_count: i32,
}

async fn evaluate_track<B: RetrieveBackend>(
    ctx: TrackEvalCtx<'_, B>,
) -> Result<TrackEvalOutcome, OpenRagError> {
    let mut mrr_sum = 0.0_f64;
    let mut recall_sum = 0.0_f64;
    let mut ndcg_sum = 0.0_f64;
    let mut latency_sum = 0.0_f64;
    let mut pii_leak_count = 0_i32;
    let mut arn_violation_count = 0_i32;
    let mut item_results = Vec::with_capacity(ctx.items.len());

    for item in ctx.items {
        let item_ctx = item
            .test_principal_id
            .and_then(|pid| ctx.principal_cache.get(&pid))
            .unwrap_or(ctx.actor);

        let mut search_req =
            SearchRequest::from_retrieve_config(&item.query_text, &ctx.pipeline.retrieve)?;
        search_req.bucket_id = ctx.bucket_id;

        let started = Instant::now();
        let resp = ctx.retriever.search(item_ctx, &search_req).await?;
        let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
        let latency_i32 = elapsed_ms.round().clamp(0.0, i32::MAX as f64) as i32;

        let retrieved_doc_ids: Vec<Uuid> = resp.hits.iter().map(extract_hit_doc_id).collect();
        let item_mrr = mrr(&retrieved_doc_ids, &item.expected_doc_ids);
        let item_recall = recall_at_k(&retrieved_doc_ids, &item.expected_doc_ids, 10);
        let item_ndcg = ndcg_at_k(&retrieved_doc_ids, &item.expected_doc_ids, 10);

        let arn_violated = check_arn_violation(&retrieved_doc_ids, &item.forbidden_doc_ids, false);
        if arn_violated {
            arn_violation_count += 1;
        }

        let joined_hits: String = resp
            .hits
            .iter()
            .map(|h| h.text.as_str())
            .collect::<Vec<_>>()
            .join("\n");

        let (item_score, pii_leaked) = match ctx.track {
            BenchTrack::Track1Retrieval => {
                let s = if item.expected_doc_ids.is_empty() {
                    if arn_violated {
                        0.0
                    } else {
                        1.0
                    }
                } else {
                    item_ndcg
                };
                (s, false)
            }
            BenchTrack::Track2E2eRag => {
                let egress =
                    apply_egress_capsule(&joined_hits, ctx.pipeline, ctx.pii_engine)?;
                let leaked = check_pii_leak(
                    &egress,
                    &item.expected_pii_tokens,
                    ctx.pipeline,
                    ctx.pii_engine,
                )?;
                let answer_score = if item.axis == GoldenAxis::Pararel {
                    if let Some(edges) = &item.expected_edges {
                        pararel_edge_overlap_score(edges, &joined_hits)
                    } else {
                        item_ndcg
                    }
                } else if let Some(ans) = item.expected_answer.as_deref() {
                    if !ans.trim().is_empty() {
                        token_overlap_score(ans, &joined_hits)
                    } else {
                        item_ndcg
                    }
                } else if !item.expected_doc_ids.is_empty() {
                    item_ndcg
                } else if resp.admission == EvidenceAdmission::Admit {
                    1.0
                } else {
                    0.0
                };
                let s = if leaked || arn_violated {
                    0.0
                } else {
                    answer_score
                };
                (s, leaked)
            }
            BenchTrack::Track3SecurityOcr => {
                let raw_egress_candidate = if (item.axis == GoldenAxis::Pii
                    || !item.expected_pii_tokens.is_empty())
                    && joined_hits.trim().is_empty()
                {
                    item.query_text.clone()
                } else if item.axis == GoldenAxis::Pii || !item.expected_pii_tokens.is_empty() {
                    format!("{joined_hits}\n{}", item.query_text)
                } else {
                    joined_hits.clone()
                };
                let egress =
                    apply_egress_capsule(&raw_egress_candidate, ctx.pipeline, ctx.pii_engine)?;
                let leaked = !ctx.pipeline.pii.capsule_enabled
                    || check_pii_leak(
                        &egress,
                        &item.expected_pii_tokens,
                        ctx.pipeline,
                        ctx.pii_engine,
                    )?;
                let base_score = if item.axis == GoldenAxis::Ocr {
                    if let Some(ans) = item.expected_answer.as_deref() {
                        token_overlap_score(ans, &joined_hits)
                    } else if !item.expected_doc_ids.is_empty() {
                        item_ndcg
                    } else {
                        1.0
                    }
                } else if !item.expected_doc_ids.is_empty() {
                    item_ndcg
                } else {
                    1.0
                };
                let s = if leaked || arn_violated {
                    0.0
                } else {
                    base_score
                };
                (s, leaked)
            }
        };

        if pii_leaked {
            pii_leak_count += 1;
        }

        mrr_sum += item_mrr;
        recall_sum += item_recall;
        ndcg_sum += item_ndcg;
        latency_sum += elapsed_ms;

        item_results.push(NewBenchItemResult {
            golden_item_id: item.id,
            retrieved_doc_ids,
            latency_ms: latency_i32,
            score: item_score as f32,
            pii_leaked,
            arn_violated,
            details: serde_json::json!({
                "track": ctx.track.as_str(),
                "axis": item.axis.as_str(),
                "admission": resp.admission.as_str(),
                "mrr": item_mrr,
                "recall_at_10": item_recall,
                "ndcg_at_10": item_ndcg,
            }),
        });
    }

    let n = ctx.items.len() as f64;
    let recall = recall_sum / n;
    let ndcg = ndcg_sum / n;
    Ok(TrackEvalOutcome {
        metrics: BenchMetrics {
            mrr: mrr_sum / n,
            recall,
            recall_at_10: recall,
            ndcg,
            ndcg_at_10: ndcg,
            avg_latency_ms: latency_sum / n,
            item_count: ctx.items.len(),
        },
        item_results,
        pii_leak_count,
        arn_violation_count,
    })
}
