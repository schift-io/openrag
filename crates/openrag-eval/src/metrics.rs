use std::collections::HashSet;

use openrag_core::{GuardAction, GuardSurface, OpenRagError, PipelineConfig};
use openrag_pii::{GuardEngine, GuardPolicyRuntime, PiiEngine};
use openrag_retrieve::SearchHit;
use uuid::Uuid;

pub fn dedup_doc_ids_in_order(ids: &[Uuid]) -> Vec<Uuid> {
    let mut seen = HashSet::with_capacity(ids.len());
    let mut out = Vec::with_capacity(ids.len());
    for &id in ids {
        if seen.insert(id) {
            out.push(id);
        }
    }
    out
}

pub fn extract_hit_doc_id(hit: &SearchHit) -> Uuid {
    hit.document_id.unwrap_or(hit.id)
}

pub fn mrr(ranked_doc_ids: &[Uuid], relevant_ids: &[Uuid]) -> f64 {
    if relevant_ids.is_empty() {
        return 0.0;
    }
    let rel_set: HashSet<Uuid> = relevant_ids.iter().copied().collect();
    let deduped = dedup_doc_ids_in_order(ranked_doc_ids);
    for (idx, doc_id) in deduped.iter().enumerate() {
        if rel_set.contains(doc_id) {
            return 1.0 / ((idx + 1) as f64);
        }
    }
    0.0
}

pub fn recall_at_k(ranked_doc_ids: &[Uuid], relevant_ids: &[Uuid], k: usize) -> f64 {
    if k == 0 || relevant_ids.is_empty() {
        return 0.0;
    }
    let rel_set: HashSet<Uuid> = relevant_ids.iter().copied().collect();
    if rel_set.is_empty() {
        return 0.0;
    }
    let deduped = dedup_doc_ids_in_order(ranked_doc_ids);
    let hits = deduped
        .into_iter()
        .take(k)
        .filter(|id| rel_set.contains(id))
        .count();
    (hits as f64) / (rel_set.len() as f64)
}

pub fn ndcg_at_k(ranked_doc_ids: &[Uuid], relevant_ids: &[Uuid], k: usize) -> f64 {
    if k == 0 || relevant_ids.is_empty() {
        return 0.0;
    }
    let rel_set: HashSet<Uuid> = relevant_ids.iter().copied().collect();
    if rel_set.is_empty() {
        return 0.0;
    }
    let deduped = dedup_doc_ids_in_order(ranked_doc_ids);
    let mut dcg = 0.0_f64;
    for (idx, doc_id) in deduped.into_iter().take(k).enumerate() {
        if rel_set.contains(&doc_id) {
            // 1-based rank is (idx + 1); discount denominator is log2(rank + 1) = log2(idx + 2).
            dcg += 1.0 / ((idx + 2) as f64).log2();
        }
    }
    let ideal_count = k.min(rel_set.len());
    let mut idcg = 0.0_f64;
    for idx in 0..ideal_count {
        idcg += 1.0 / ((idx + 2) as f64).log2();
    }
    if idcg == 0.0 || !idcg.is_finite() || !dcg.is_finite() {
        return 0.0;
    }
    (dcg / idcg).clamp(0.0, 1.0)
}

pub fn check_arn_violation(
    retrieved_doc_ids: &[Uuid],
    forbidden_doc_ids: &[Uuid],
    bypass_rls_test_only: bool,
) -> bool {
    if bypass_rls_test_only {
        return true;
    }
    if forbidden_doc_ids.is_empty() {
        return false;
    }
    let forbidden: HashSet<Uuid> = forbidden_doc_ids.iter().copied().collect();
    retrieved_doc_ids.iter().any(|id| forbidden.contains(id))
}

pub fn check_pii_leak(
    egress_text: &str,
    expected_pii_tokens: &[String],
    pipeline: &PipelineConfig,
    pii_engine: &PiiEngine,
) -> Result<bool, OpenRagError> {
    let spans = pii_engine.detect(egress_text);
    let raw_token_present = expected_pii_tokens
        .iter()
        .map(|t| t.trim())
        .filter(|t| !t.is_empty())
        .any(|t| egress_text.contains(t));

    if !pipeline.pii.capsule_enabled {
        return Ok(!spans.is_empty() || raw_token_present);
    }

    let policy = GuardPolicyRuntime::from_guard_config(Uuid::nil(), &pipeline.pii.guard);
    let guard = GuardEngine::evaluate(
        egress_text,
        GuardSurface::LlmEgress,
        &policy,
        pii_engine,
        None,
        None,
    )?;

    let unmasked_guard_trigger = matches!(guard.action, GuardAction::Mask | GuardAction::Block);
    Ok(!spans.is_empty() || raw_token_present || unmasked_guard_trigger)
}
