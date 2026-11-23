use std::collections::{BTreeMap, HashMap};
use std::time::Instant;

use openrag_core::{OpenRagError, PipelineConfig, PrincipalContext};
use openrag_pii::PiiEngine;
use openrag_retrieve::{HybridRetriever, RetrieveBackend, SearchRequest};
use uuid::Uuid;

use crate::golden::EvalStore;
use crate::metrics::{
    check_arn_violation, check_pii_leak, extract_hit_doc_id, mrr, ndcg_at_k, recall_at_k,
};
use crate::optimizer::default_pipeline_config;
use crate::types::{
    AbQueryInput, AbQueryItem, AbSearchReport, AbSearchRequest, AbVariantMetrics,
    SecurityGateResult,
};

pub struct AbSearchEvaluator;

impl AbSearchEvaluator {
    pub async fn evaluate<B: RetrieveBackend>(
        retriever: &HybridRetriever<B>,
        store: Option<&dyn EvalStore>,
        ctx: &PrincipalContext,
        req: &AbSearchRequest,
        pipeline: Option<&PipelineConfig>,
    ) -> Result<AbSearchReport, OpenRagError> {
        let default_k = req.validate()?;
        let cfg = match pipeline {
            Some(p) => p.clone(),
            None => default_pipeline_config(),
        };
        let pii_engine = PiiEngine::from_config(&cfg.pii);
        let resolved_queries = resolve_ab_queries(store, ctx.org_id, req).await?;

        let mut variants_out = BTreeMap::new();
        let mut total_pii_leaks = 0_i32;
        let mut total_arn_violations = 0_i32;

        for (variant_name, variant_cfg) in &req.variants {
            let overrides = variant_cfg.to_retrieve_overrides(default_k);
            let eval_k = variant_cfg.top_k.unwrap_or(default_k);

            let mut mrr_sum = 0.0_f64;
            let mut recall_sum = 0.0_f64;
            let mut ndcg_sum = 0.0_f64;
            let mut latency_sum = 0.0_f64;
            let mut variant_pii_leaks = 0_i32;
            let mut variant_arn_violations = 0_i32;

            for q in &resolved_queries {
                let mut search_req = SearchRequest::new(&q.query);
                search_req.bucket_id = req.bucket_id;
                search_req.latest_revision_only = variant_cfg.latest_revision_only.unwrap_or(false);

                let started = Instant::now();
                let resp = retriever
                    .search_with_overrides(ctx, &search_req, &overrides)
                    .await?;
                let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;

                let ranked_doc_ids: Vec<Uuid> = resp.hits.iter().map(extract_hit_doc_id).collect();
                mrr_sum += mrr(&ranked_doc_ids, &q.relevant_ids);
                recall_sum += recall_at_k(&ranked_doc_ids, &q.relevant_ids, eval_k);
                ndcg_sum += ndcg_at_k(&ranked_doc_ids, &q.relevant_ids, eval_k);
                latency_sum += elapsed_ms;

                if check_arn_violation(&ranked_doc_ids, &q.forbidden_ids, false) {
                    variant_arn_violations += 1;
                }

                let mut query_leaked_pii = false;
                for hit in &resp.hits {
                    if check_pii_leak(&hit.text, &q.expected_pii_tokens, &cfg, &pii_engine)?
                        || check_pii_leak(&hit.snippet, &q.expected_pii_tokens, &cfg, &pii_engine)?
                    {
                        query_leaked_pii = true;
                        break;
                    }
                }
                if query_leaked_pii {
                    variant_pii_leaks += 1;
                }
            }

            let n = resolved_queries.len() as f64;
            let gates = SecurityGateResult::new(variant_pii_leaks, variant_arn_violations);
            total_pii_leaks += variant_pii_leaks;
            total_arn_violations += variant_arn_violations;

            let mean_recall = recall_sum / n;
            let mean_ndcg = ndcg_sum / n;

            variants_out.insert(
                variant_name.clone(),
                AbVariantMetrics {
                    mrr: mrr_sum / n,
                    recall: mean_recall,
                    recall_at_k: mean_recall,
                    ndcg: mean_ndcg,
                    ndcg_at_k: mean_ndcg,
                    avg_latency_ms: latency_sum / n,
                    query_count: resolved_queries.len(),
                    security_gates: gates,
                },
            );
        }

        let winner = variants_out
            .iter()
            .filter(|(_, m)| m.security_gates.pass)
            .max_by(|(name_a, a), (name_b, b)| {
                a.ndcg_at_k
                    .total_cmp(&b.ndcg_at_k)
                    .then_with(|| a.mrr.total_cmp(&b.mrr))
                    .then_with(|| a.recall_at_k.total_cmp(&b.recall_at_k))
                    .then_with(|| name_b.cmp(name_a))
            })
            .map(|(name, _)| name.clone());

        Ok(AbSearchReport {
            variants: variants_out,
            security_gates: SecurityGateResult::new(total_pii_leaks, total_arn_violations),
            winner,
        })
    }
}

struct GoldenQueryExpectations {
    relevant_ids: Vec<Uuid>,
    forbidden_ids: Vec<Uuid>,
    expected_pii_tokens: Vec<String>,
}

async fn resolve_ab_queries(
    store: Option<&dyn EvalStore>,
    org_id: Uuid,
    req: &AbSearchRequest,
) -> Result<Vec<AbQueryItem>, OpenRagError> {
    let mut item_by_query: HashMap<String, GoldenQueryExpectations> = HashMap::new();
    if let Some(gs_id) = req.golden_set_id {
        if let Some(st) = store {
            let _gs = st.get_golden_set(org_id, gs_id).await?;
            let items = st.list_golden_items(org_id, gs_id, false).await?;
            for item in items {
                item_by_query
                    .entry(item.query_text.trim().to_string())
                    .or_insert_with(|| GoldenQueryExpectations {
                        relevant_ids: item.expected_doc_ids,
                        forbidden_ids: item.forbidden_doc_ids,
                        expected_pii_tokens: item.expected_pii_tokens,
                    });
            }
        }
    }

    let mut resolved = Vec::with_capacity(req.queries.len());
    for raw in &req.queries {
        let item = match raw {
            AbQueryInput::Inline(q) => q.clone(),
            AbQueryInput::QueryText(text) => {
                let key = text.trim();
                if let Some(exp) = item_by_query.get(key) {
                    AbQueryItem {
                        query: key.to_string(),
                        relevant_ids: exp.relevant_ids.clone(),
                        forbidden_ids: exp.forbidden_ids.clone(),
                        expected_pii_tokens: exp.expected_pii_tokens.clone(),
                    }
                } else {
                    AbQueryItem {
                        query: key.to_string(),
                        relevant_ids: Vec::new(),
                        forbidden_ids: Vec::new(),
                        expected_pii_tokens: Vec::new(),
                    }
                }
            }
        };
        if item.query.trim().is_empty() {
            return Err(OpenRagError::Validation(
                "ab_search query text must not be empty".into(),
            ));
        }
        resolved.push(item);
    }

    Ok(resolved)
}
