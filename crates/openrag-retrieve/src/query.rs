use std::collections::BTreeSet;

use openrag_core::{OpenRagError, RetrieveConfig};
use openrag_memory::concepts::normalize_term;
use openrag_store::{
    normalize_tag, parse_inline_tags, tokenize_bm25, validate_vector, ParsedTagQuery, TagFilter,
};

use crate::types::{
    validate_numeric_params, validate_resolved_values, ResolvedRetrieveParams, RetrieveSpace,
    SearchRequest,
};

const TRAILING_TAG_PUNCT: &[char] = &[',', '.', ';', ':', '!', '?', ')', ']', '}', '"', '\'', '>'];

pub fn parse_spaces(raw_spaces: &[String]) -> Result<Vec<RetrieveSpace>, OpenRagError> {
    if raw_spaces.is_empty() {
        return Err(OpenRagError::Validation("spaces must not be empty".into()));
    }
    let mut out = Vec::with_capacity(raw_spaces.len());
    for raw in raw_spaces {
        let space: RetrieveSpace = raw.parse()?;
        if !out.contains(&space) {
            out.push(space);
        }
    }
    Ok(out)
}

pub fn resolve_retrieve_params(
    req: &SearchRequest,
    defaults: &RetrieveConfig,
) -> Result<ResolvedRetrieveParams, OpenRagError> {
    if let Some(vec) = &req.query_vector {
        validate_vector(vec)?;
    }

    let spaces = match &req.spaces {
        Some(sp) => {
            if sp.is_empty() {
                return Err(OpenRagError::Validation("spaces must not be empty".into()));
            }
            let mut dedup = Vec::with_capacity(sp.len());
            for s in sp {
                if !dedup.contains(s) {
                    dedup.push(*s);
                }
            }
            dedup
        }
        None => validate_resolved_values(
            &defaults.spaces,
            defaults.rrf_k,
            defaults.alpha,
            defaults.top_k,
            defaults.latest_revision_boost,
            defaults.admission_threshold,
        )?,
    };

    let fusion = req.fusion.unwrap_or(defaults.fusion);
    let rrf_k = req.rrf_k.unwrap_or(defaults.rrf_k);
    let alpha = req.alpha.unwrap_or(defaults.alpha);
    let top_k = req.top_k.unwrap_or(defaults.top_k);
    let rerank = req.rerank.unwrap_or(defaults.rerank);
    let latest_revision_boost = req
        .latest_revision_boost
        .unwrap_or(defaults.latest_revision_boost);
    let admission_threshold = req
        .admission_threshold
        .unwrap_or(defaults.admission_threshold);

    validate_numeric_params(
        rrf_k,
        alpha,
        top_k,
        latest_revision_boost,
        admission_threshold,
    )?;

    if req.facet_limit <= 0 {
        return Err(OpenRagError::Validation(format!(
            "facet_limit must be >= 1, got {}",
            req.facet_limit
        )));
    }

    Ok(ResolvedRetrieveParams {
        spaces,
        fusion,
        rrf_k,
        alpha,
        top_k,
        rerank,
        latest_revision_boost,
        latest_revision_only: req.latest_revision_only,
        admission_threshold,
    })
}

/// Parses inline `#tag`, `tag:<name>`, `-#tag`, `-tag:<name>` tokens while:
/// - preserving URL fragments (`https://example.org/docs#section1`)
/// - stripping trailing punctuation (`,`, `.`, `;`, `:`, `!`, `?`, `)`, `]`, `}`, `"`, `'`, `>`)
/// - normalizing tags and dropping empty malformed prefixes (`#`, `tag:`).
pub fn sanitize_and_parse_inline_tags(query: &str) -> ParsedTagQuery {
    let mut sanitized_tokens = Vec::with_capacity(8);
    for token in query.split_whitespace() {
        if let Some((prefix, raw_body)) = split_tag_token_prefix(token) {
            let trimmed_body = raw_body.trim_end_matches(TRAILING_TAG_PUNCT);
            if normalize_tag(trimmed_body).is_empty() {
                // Bare malformed prefix like "#" or "tag:" or "#," — drop from tag extraction
                // while preserving clean query continuity without panicking.
                continue;
            }
            sanitized_tokens.push(format!("{prefix}{trimmed_body}"));
        } else {
            sanitized_tokens.push(token.to_string());
        }
    }
    let sanitized_query = sanitized_tokens.join(" ");
    parse_inline_tags(&sanitized_query)
}

fn split_tag_token_prefix(token: &str) -> Option<(&str, &str)> {
    if let Some(rest) = token.strip_prefix("-#") {
        Some(("-#", rest))
    } else if token
        .as_bytes()
        .get(..5)
        .is_some_and(|p| p.eq_ignore_ascii_case(b"-tag:"))
    {
        Some((&token[..5], &token[5..]))
    } else if let Some(rest) = token.strip_prefix('#') {
        Some(("#", rest))
    } else if token
        .as_bytes()
        .get(..4)
        .is_some_and(|p| p.eq_ignore_ascii_case(b"tag:"))
    {
        Some((&token[..4], &token[4..]))
    } else {
        None
    }
}

pub fn merge_tag_filters(base: Option<&TagFilter>, parsed: &ParsedTagQuery) -> TagFilter {
    let mut merged = base.cloned().unwrap_or_default();
    // Multiple inline tags (#alpha #beta) require AND semantics -> tags_all.
    merged.tags_all.extend(parsed.inline_tags.iter().cloned());
    merged
        .tags_exclude
        .extend(parsed.excluded_tags.iter().cloned());
    merged.normalized()
}

pub fn has_conflicting_tags(filter: &TagFilter) -> bool {
    let filter = filter.normalized();
    if filter.expand_tag_aliases || filter.tags_exclude.is_empty() {
        // When expand_tag_aliases is true, let alias expansion + evaluation handle it,
        // unless direct normalized collision exists in tags_all.
        if !filter.tags_all.is_empty()
            && filter
                .tags_all
                .iter()
                .any(|t| filter.tags_exclude.contains(t))
        {
            return true;
        }
        return false;
    }
    if filter
        .tags_all
        .iter()
        .any(|t| filter.tags_exclude.contains(t))
    {
        return true;
    }
    if !filter.tags_any.is_empty()
        && filter
            .tags_any
            .iter()
            .all(|t| filter.tags_exclude.contains(t))
    {
        return true;
    }
    false
}

const KOREAN_PARTICLES: &[&str] = &[
    "에서", "으로", "을", "를", "은", "는", "이", "가", "의", "에", "로", "와", "과",
];

fn strip_korean_particle(word: &str) -> Option<String> {
    for suffix in KOREAN_PARTICLES {
        if let Some(stem) = word.strip_suffix(suffix) {
            if stem.chars().count() >= 2 {
                return Some(stem.to_string());
            }
        }
    }
    None
}

/// Builds single-word tokens, Korean particle-stripped variants, contiguous 2..5-word shingles,
/// and the full normalized `clean_query` for `ConceptGraph::expand_terms_rls` so both
/// single-word (`apple` <-> `사과`, `"사과를"`) and multi-word (`llmwiki` <-> `세컨드 브레인`,
/// 4+ word aliases) concepts expand cleanly.
pub fn build_concept_lookup_terms(clean_query: &str) -> Vec<String> {
    let norm_full = normalize_term(clean_query);
    if norm_full.is_empty() {
        return Vec::new();
    }

    let raw_words: Vec<String> = norm_full
        .split_whitespace()
        .map(strip_edge_punctuation)
        .filter(|w| !w.is_empty())
        .collect();

    let mut seen = BTreeSet::new();
    let mut terms = Vec::new();
    let mut push = |s: String| {
        if !s.is_empty() && seen.insert(s.clone()) {
            terms.push(s);
        }
    };

    for word in &raw_words {
        push(word.clone());
        if let Some(stem) = strip_korean_particle(word) {
            push(stem);
        }
    }
    let max_win = raw_words.len().min(5);
    for win_size in 2..=max_win {
        for win in raw_words.windows(win_size) {
            push(win.join(" "));
            if let Some(last) = win.last() {
                if let Some(stem) = strip_korean_particle(last) {
                    let mut prefix: Vec<&str> =
                        win[..win.len() - 1].iter().map(String::as_str).collect();
                    prefix.push(stem.as_str());
                    push(prefix.join(" "));
                }
            }
        }
    }
    push(norm_full);

    // Also include BM25 tokens (e.g. Korean clause normalization like "3조") if distinct.
    for tok in tokenize_bm25(clean_query) {
        push(tok);
    }

    terms
}

fn strip_edge_punctuation(token: &str) -> String {
    let trimmed = token.trim_matches(|c: char| !c.is_alphanumeric() && c != '-' && c != '_');
    trimmed.to_string()
}

pub fn query_has_searchable_tokens(clean_query: &str) -> bool {
    !tokenize_bm25(clean_query).is_empty()
}

/// Computes the text used for generating the default `256d` query vector when `query_vector`
/// is omitted. When concept expansion uncovered aliases beyond the original query tokens,
/// includes those newly discovered aliases so cross-lingual vector similarity improves too.
pub fn build_embed_query_text(
    clean_query: &str,
    expanded_terms: &[String],
    tag_filter: &TagFilter,
) -> String {
    if !query_has_searchable_tokens(clean_query) {
        let mut tag_terms = Vec::new();
        tag_terms.extend(tag_filter.tags_all.iter().cloned());
        tag_terms.extend(tag_filter.tags_any.iter().cloned());
        return tag_terms.join(" ");
    }

    let orig_tokens: BTreeSet<String> = tokenize_bm25(clean_query).into_iter().collect();
    let mut extra_terms = Vec::new();
    for term in expanded_terms {
        let term_toks = tokenize_bm25(term);
        if term_toks.iter().any(|t| !orig_tokens.contains(t)) {
            extra_terms.push(term.clone());
        }
    }
    if extra_terms.is_empty() {
        clean_query.trim().to_string()
    } else {
        format!("{} {}", clean_query.trim(), extra_terms.join(" "))
    }
}
