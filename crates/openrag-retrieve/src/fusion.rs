use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

use openrag_core::FusionMode;
use openrag_memory::concepts::normalize_term;
use openrag_store::{cosine_similarity, normalize_tag, tokenize_bm25};
use uuid::Uuid;

use crate::types::{
    BackendExecutionOutput, EvidenceAdmission, FusionContext, HydratedCandidate,
    ResolvedRetrieveParams, RetrieveSpace, ScoreBreakdown, SearchHit, SearchResponse,
    ValidityState,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum FacetEntityKey {
    CorpusDoc(Uuid),
    WikiPage(Uuid),
    MemoryEntry(Uuid),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum RevisionSpaceGroup {
    Corpus,
    Wiki,
    Memory,
}

impl RevisionSpaceGroup {
    const fn from_space(space: RetrieveSpace) -> Self {
        match space {
            RetrieveSpace::Chunks | RetrieveSpace::Thumb => Self::Corpus,
            RetrieveSpace::Wiki => Self::Wiki,
            RetrieveSpace::Memory => Self::Memory,
        }
    }
}

struct CandidateAccum<'a> {
    hydrated: &'a HydratedCandidate,
    raw_cosine_sim: Option<f32>,
    vec_rank: Option<usize>,
    raw_bm25: Option<f64>,
    bm25_rank: Option<usize>,
    tag_only_rank: Option<usize>,
}

struct ScoredCandidate {
    hit: SearchHit,
    sort_score: f32,
    entity_key: FacetEntityKey,
    tags_norm: BTreeSet<String>,
}

#[inline]
fn sanitize_unit(val: f32) -> f32 {
    if val.is_finite() {
        val.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

#[inline]
fn finite_or_neg_inf(val: f32) -> f32 {
    if val.is_finite() {
        val
    } else {
        f32::NEG_INFINITY
    }
}

#[inline]
fn sanitize_non_neg_f64(val: f64) -> f64 {
    if val.is_finite() {
        val.max(0.0)
    } else {
        0.0
    }
}

pub fn fuse_and_rank(
    fctx: FusionContext,
    params: &ResolvedRetrieveParams,
    exec: BackendExecutionOutput,
) -> SearchResponse {
    let FusionContext {
        clean_query,
        parsed_inline_tags,
        parsed_excluded_tags,
        tag_only_query,
        facet_limit,
        now,
    } = fctx;

    let is_future = |h: &HydratedCandidate| -> bool { h.valid_from.is_some_and(|vf| vf > now) };
    let is_stale = |h: &HydratedCandidate| -> bool {
        h.is_superseded || h.valid_to.is_some_and(|vt| vt < now) || is_future(h)
    };

    // Pre-compute max active revision per (space group, normalized title) when `latest_revision_only`
    // is enabled, so stale/superseded/future-dated/older-revision hits within the same space group
    // (`Chunks`+`Thumb` share `Corpus`, while `Wiki` and `Memory` stay independent) are excluded
    // BEFORE computing `max_bm25` and 1-based RRF ranks (`BUG-1`, `BUG-3`).
    // Always computed: it also drives the `older_revision` validity label when
    // `latest_revision_only` is off.
    let mut max_rev_by_title_only: HashMap<(RevisionSpaceGroup, String), i32> = HashMap::new();
    {
        let mut seen_keys: HashSet<(RetrieveSpace, Uuid)> = HashSet::new();
        for (space, h) in &exec.vector_hits {
            seen_keys.insert((*space, h.id));
        }
        for (space, h) in &exec.bm25_hits {
            seen_keys.insert((*space, h.id));
        }
        for h in &exec.tag_only_hits {
            seen_keys.insert((h.space, h.id));
        }
        for key in seen_keys {
            let Some(h) = exec.hydrated.get(&key) else {
                continue;
            };
            if !passes_post_hydration_filters(h, exec.expanded_tag_filter.as_ref()) || is_stale(h) {
                continue;
            }
            let rev = h.revision.unwrap_or(1).max(1);
            if let Some(title) = &h.title {
                let norm_t = normalize_term(title);
                if !norm_t.is_empty() {
                    let group = RevisionSpaceGroup::from_space(h.space);
                    let cur = max_rev_by_title_only.entry((group, norm_t)).or_insert(rev);
                    if rev > *cur {
                        *cur = rev;
                    }
                }
            }
        }
    }

    let validity_of = |h: &HydratedCandidate| -> ValidityState {
        if is_future(h) {
            return ValidityState::Scheduled;
        }
        if h.valid_to.is_some_and(|vt| vt < now) {
            return ValidityState::Expired;
        }
        if h.is_superseded {
            return ValidityState::Superseded;
        }
        if let Some(title) = &h.title {
            let norm_t = normalize_term(title);
            let group = RevisionSpaceGroup::from_space(h.space);
            if let Some(&max_t_rev) = max_rev_by_title_only.get(&(group, norm_t)) {
                if h.revision.unwrap_or(1).max(1) < max_t_rev {
                    return ValidityState::OlderRevision;
                }
            }
        }
        ValidityState::Current
    };

    let resolve_surviving = |key: &(RetrieveSpace, Uuid)| -> Option<&HydratedCandidate> {
        let h = exec.hydrated.get(key)?;
        if !passes_post_hydration_filters(h, exec.expanded_tag_filter.as_ref()) {
            return None;
        }
        if params.latest_revision_only {
            if is_stale(h) {
                return None;
            }
            if let Some(title) = &h.title {
                let norm_t = normalize_term(title);
                let group = RevisionSpaceGroup::from_space(h.space);
                if let Some(&max_t_rev) = max_rev_by_title_only.get(&(group, norm_t)) {
                    let rev = h.revision.unwrap_or(1).max(1);
                    if rev < max_t_rev {
                        return None;
                    }
                }
            }
        }
        Some(h)
    };

    // 1. Filter first, then sort and build global 1-based ranks for vector, BM25, and tag-only hits.
    let mut vec_sorted: Vec<_> = exec
        .vector_hits
        .into_iter()
        .filter(|(space, h)| resolve_surviving(&(*space, h.id)).is_some())
        .collect();
    vec_sorted.sort_by(|a, b| {
        finite_or_neg_inf(b.1.cosine_sim)
            .total_cmp(&finite_or_neg_inf(a.1.cosine_sim))
            .then_with(|| a.1.id.cmp(&b.1.id))
            .then_with(|| a.0.cmp(&b.0))
    });

    let mut bm25_sorted: Vec<_> = exec
        .bm25_hits
        .into_iter()
        .filter(|(space, h)| resolve_surviving(&(*space, h.id)).is_some())
        .collect();
    bm25_sorted.sort_by(|a, b| {
        sanitize_non_neg_f64(b.1.score)
            .total_cmp(&sanitize_non_neg_f64(a.1.score))
            .then_with(|| a.1.id.cmp(&b.1.id))
            .then_with(|| a.0.cmp(&b.0))
    });

    let max_bm25 = bm25_sorted
        .iter()
        .map(|(_, h)| sanitize_non_neg_f64(h.score))
        .fold(0.0_f64, f64::max);

    let mut accums: HashMap<(RetrieveSpace, Uuid), CandidateAccum<'_>> = HashMap::new();

    for (idx, (space, vhit)) in vec_sorted.iter().enumerate() {
        let key = (*space, vhit.id);
        let Some(hydrated) = resolve_surviving(&key) else {
            continue;
        };
        let entry = accums.entry(key).or_insert_with(|| CandidateAccum {
            hydrated,
            raw_cosine_sim: None,
            vec_rank: None,
            raw_bm25: None,
            bm25_rank: None,
            tag_only_rank: None,
        });
        let rank = idx + 1;
        if entry.vec_rank.is_none() {
            entry.vec_rank = Some(rank);
            entry.raw_cosine_sim = Some(sanitize_unit(vhit.cosine_sim));
        }
    }

    for (idx, (space, bhit)) in bm25_sorted.iter().enumerate() {
        let key = (*space, bhit.id);
        let Some(hydrated) = resolve_surviving(&key) else {
            continue;
        };
        let entry = accums.entry(key).or_insert_with(|| CandidateAccum {
            hydrated,
            raw_cosine_sim: None,
            vec_rank: None,
            raw_bm25: None,
            bm25_rank: None,
            tag_only_rank: None,
        });
        let rank = idx + 1;
        if entry.bm25_rank.is_none() {
            entry.bm25_rank = Some(rank);
            entry.raw_bm25 = Some(sanitize_non_neg_f64(bhit.score));
        }
    }

    let mut tag_only_rank_counter = 0_usize;
    for thit in &exec.tag_only_hits {
        let key = (thit.space, thit.id);
        let Some(hydrated) = resolve_surviving(&key) else {
            continue;
        };
        tag_only_rank_counter += 1;
        let entry = accums.entry(key).or_insert_with(|| CandidateAccum {
            hydrated,
            raw_cosine_sim: None,
            vec_rank: None,
            raw_bm25: None,
            bm25_rank: None,
            tag_only_rank: None,
        });
        if entry.tag_only_rank.is_none() {
            entry.tag_only_rank = Some(tag_only_rank_counter);
        }
    }

    if accums.is_empty() {
        return SearchResponse::empty(
            clean_query,
            parsed_inline_tags,
            parsed_excluded_tags,
            exec.expanded_terms,
        );
    }

    // 2. Identify active vs expired/superseded/future-dated revisions.
    let mut max_active_rev: i32 = 1;
    let mut any_stale = false;

    for cand in accums.values() {
        if is_stale(cand.hydrated) {
            any_stale = true;
            continue;
        }
        let rev = cand.hydrated.revision.unwrap_or(1).max(1);
        if rev > max_active_rev {
            max_active_rev = rev;
        }
    }

    // Prepare lexical rerank query weights.
    let lex_query = LexicalQueryWeights::new(&clean_query, &exec.expanded_terms);

    let mut scored = Vec::with_capacity(accums.len());
    let rrf_k_f = params.rrf_k as f32;
    let max_rrf = 2.0 / (rrf_k_f + 1.0);

    for cand in accums.into_values() {
        let h = cand.hydrated;
        let stale = is_stale(h);

        let mut norm_vec = sanitize_unit(cand.raw_cosine_sim.unwrap_or(0.0));
        let mut norm_bm25 = if max_bm25 > 0.0 {
            sanitize_unit((sanitize_non_neg_f64(cand.raw_bm25.unwrap_or(0.0)) / max_bm25) as f32)
        } else {
            0.0
        };

        // Tag-only query: any surviving hit matched the requested tag filter (`BUG-5`).
        if tag_only_query {
            norm_bm25 = 1.0;
            norm_vec = norm_vec.max(0.5);
        }

        let exact_cos = if params.rerank {
            h.embedding
                .as_deref()
                .and_then(|emb| cosine_similarity(&exec.query_vector, emb).ok())
                .map(sanitize_unit)
                .unwrap_or(norm_vec)
        } else {
            norm_vec
        };

        let lex_overlap = if tag_only_query {
            1.0
        } else {
            lex_query.score_document(h.title.as_deref(), &h.text)
        };

        let lex_rrf_rank = if tag_only_query {
            cand.bm25_rank.or(cand.tag_only_rank).or(Some(1))
        } else {
            cand.bm25_rank
        };
        let rrf_raw = cand.vec_rank.map_or(0.0, |r| 1.0 / (rrf_k_f + r as f32))
            + lex_rrf_rank.map_or(0.0, |r| 1.0 / (rrf_k_f + r as f32))
            + if !tag_only_query && cand.vec_rank.is_none() && cand.bm25_rank.is_none() {
                cand.tag_only_rank
                    .map_or(0.0, |r| 1.0 / (rrf_k_f + r as f32))
            } else {
                0.0
            };
        let rrf_score = if max_rrf > 0.0 {
            (rrf_raw / max_rrf).clamp(0.0, 1.0)
        } else {
            0.0
        };

        // A space without vectors (page thumbnails) can only be scored lexically.
        let alpha = if h.space.vector_table().has_vectors() {
            params.alpha
        } else {
            1.0
        };
        let convex_score = ((1.0 - alpha) * norm_vec + alpha * norm_bm25).clamp(0.0, 1.0);

        // Calibrate RRF and admission by genuine semantic/lexical/tag evidence strength (`BUG-2`),
        // damping low random-hash cosine noise when there is zero lexical/BM25 match.
        let sem_strength = norm_vec.max(exact_cos);
        let lex_strength = if tag_only_query {
            1.0
        } else if cand.bm25_rank.is_some() {
            norm_bm25.max(lex_overlap)
        } else {
            lex_overlap
        };
        let damped_sem = if lex_strength > 0.0 || sem_strength >= 0.40 {
            sem_strength
        } else {
            sem_strength * (sem_strength / 0.40).clamp(0.0, 1.0)
        };
        let signal_strength = if tag_only_query {
            1.0
        } else {
            damped_sem.max(lex_strength).clamp(0.0, 1.0)
        };

        let mut fused_base = match params.fusion {
            FusionMode::Rrf => (rrf_score * signal_strength).clamp(0.0, 1.0),
            FusionMode::Convex => convex_score,
        };

        let rerank_score = if params.rerank {
            let r_score = (0.65 * exact_cos + 0.35 * lex_overlap).clamp(0.0, 1.0);
            fused_base = (0.65 * fused_base + 0.35 * r_score).clamp(0.0, 1.0);
            r_score
        } else {
            0.0
        };

        let revision_boost = if stale {
            fused_base *= 0.85;
            0.0
        } else if signal_strength < 0.25 {
            // Do not allow revision_boost to elevate off-topic noise above admission_threshold.
            0.0
        } else {
            let rev = h.revision.unwrap_or(1).max(1);
            if max_active_rev > 1 {
                params.latest_revision_boost * ((rev - 1) as f32 / (max_active_rev - 1) as f32)
            } else if any_stale {
                params.latest_revision_boost
            } else {
                0.0
            }
        };

        let sort_score = (fused_base + revision_boost).max(0.0);
        let final_score = sort_score.clamp(0.0, 1.0);

        let entity_key = match h.space {
            RetrieveSpace::Chunks | RetrieveSpace::Thumb => {
                FacetEntityKey::CorpusDoc(h.document_id.unwrap_or(h.id))
            }
            RetrieveSpace::Wiki => FacetEntityKey::WikiPage(h.id),
            RetrieveSpace::Memory => FacetEntityKey::MemoryEntry(h.id),
        };
        let tags_norm: BTreeSet<String> = h
            .tags
            .iter()
            .map(|t| normalize_tag(t))
            .filter(|t| !t.is_empty())
            .collect();

        let hit = SearchHit {
            id: h.id,
            document_id: h.document_id,
            page_id: h.page_id,
            space: h.space,
            title: h.title.clone(),
            snippet: make_snippet(&h.text, 240),
            text: h.text.clone(),
            score: final_score,
            vector_score: norm_vec,
            bm25_score: norm_bm25,
            score_breakdown: ScoreBreakdown {
                vector_score: norm_vec,
                bm25_score: norm_bm25,
                rrf_score,
                convex_score,
                revision_boost,
                rerank_score,
                fused_score: fused_base,
            },
            tags: tags_norm.iter().cloned().collect(),
            min_auth_level: h.min_auth_level,
            revision: h.revision,
            valid_from: h.valid_from,
            valid_to: h.valid_to,
            thumb_uri: h.thumb_uri.clone(),
            bbox_norm_1000: h.bbox_norm_1000,
            acl_scope_key: h.acl_scope_key.clone(),
            validity: validity_of(h),
        };

        scored.push(ScoredCandidate {
            hit,
            sort_score,
            entity_key,
            tags_norm,
        });
    }

    // 3. Sort descending by uncapped sort_score, deterministic tie-break by id.
    scored.sort_by(|a, b| {
        b.sort_score
            .total_cmp(&a.sort_score)
            .then_with(|| a.hit.id.cmp(&b.hit.id))
            .then_with(|| a.hit.space.cmp(&b.hit.space))
    });

    if scored.is_empty() {
        return SearchResponse::empty(
            clean_query,
            parsed_inline_tags,
            parsed_excluded_tags,
            exec.expanded_terms,
        );
    }

    // 4. Aggregate deduplicated RLS-safe tag facets across surviving candidates.
    let tag_facets = compute_dedup_tag_facets(&scored, facet_limit);

    // 5. Truncate to top_k and evaluate EvidenceAdmission.
    scored.truncate(params.top_k);
    let hits: Vec<SearchHit> = scored.into_iter().map(|s| s.hit).collect();

    let top_score = hits.first().map_or(0.0, |h| h.score);
    let max_score = top_score;
    let top_auth_level = hits.iter().filter_map(|h| h.min_auth_level).max();
    let admission = if !hits.is_empty() && max_score >= params.admission_threshold {
        EvidenceAdmission::Admit
    } else {
        EvidenceAdmission::Abstain
    };

    SearchResponse {
        hits,
        clean_query,
        parsed_inline_tags,
        parsed_excluded_tags,
        expanded_terms: exec.expanded_terms,
        tag_facets,
        admission,
        max_score,
        top_score,
        top_auth_level,
    }
}

fn passes_post_hydration_filters(
    h: &HydratedCandidate,
    expanded_tag_filter: Option<&openrag_store::ExpandedTagFilter>,
) -> bool {
    if h.is_archived {
        return false;
    }
    if let Some(tf) = expanded_tag_filter {
        if !tf.is_empty() && !tf.matches_tags(&h.tags) {
            return false;
        }
    }
    true
}

fn compute_dedup_tag_facets(
    candidates: &[ScoredCandidate],
    facet_limit: i64,
) -> BTreeMap<String, i64> {
    if candidates.is_empty() || facet_limit <= 0 {
        return BTreeMap::new();
    }
    let mut entity_tags: HashMap<FacetEntityKey, BTreeSet<String>> = HashMap::new();
    for cand in candidates {
        let entry = entity_tags.entry(cand.entity_key).or_default();
        for tag in &cand.tags_norm {
            entry.insert(tag.clone());
        }
    }
    let mut counts: HashMap<String, i64> = HashMap::new();
    for tags in entity_tags.values() {
        for tag in tags {
            *counts.entry(tag.clone()).or_insert(0) += 1;
        }
    }
    let mut sorted: Vec<(String, i64)> = counts.into_iter().collect();
    sorted.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    sorted.truncate(facet_limit as usize);
    sorted.into_iter().collect()
}

struct LexicalQueryWeights {
    terms: BTreeMap<String, f32>,
    total_weight: f32,
    norm_phrase: String,
}

impl LexicalQueryWeights {
    fn new(clean_query: &str, expanded_terms: &[String]) -> Self {
        let mut terms = BTreeMap::new();
        for tok in tokenize_bm25(clean_query) {
            terms.entry(tok).or_insert(1.0_f32);
        }
        for extra in expanded_terms {
            for tok in tokenize_bm25(extra) {
                terms.entry(tok).or_insert(0.85_f32);
            }
        }
        let total_weight: f32 = terms.values().sum();
        let norm_phrase = normalize_term(clean_query);
        Self {
            terms,
            total_weight,
            norm_phrase,
        }
    }

    fn score_document(&self, title: Option<&str>, text: &str) -> f32 {
        if self.total_weight <= 0.0 {
            return 0.0;
        }
        let combined = match title {
            Some(t) if !t.trim().is_empty() => format!("{t} {text}"),
            _ => text.to_string(),
        };
        let doc_tokens: BTreeSet<String> = tokenize_bm25(&combined).into_iter().collect();
        let mut matched = 0.0_f32;
        for (term, weight) in &self.terms {
            if doc_tokens.contains(term) {
                matched += *weight;
            }
        }
        let mut ratio = (matched / self.total_weight).clamp(0.0, 1.0);
        if self.norm_phrase.chars().count() >= 2 {
            let norm_doc = normalize_term(&combined);
            if norm_doc.contains(&self.norm_phrase) {
                ratio = (ratio + 0.15).clamp(0.0, 1.0);
            }
        }
        ratio
    }
}

pub(crate) fn make_snippet(text: &str, max_chars: usize) -> String {
    let trimmed = text.trim();
    if trimmed.chars().count() <= max_chars {
        trimmed.to_string()
    } else {
        let prefix: String = trimmed.chars().take(max_chars).collect();
        format!("{prefix}...")
    }
}
