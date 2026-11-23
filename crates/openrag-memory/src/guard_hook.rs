use std::cmp::Reverse;

use async_trait::async_trait;
use openrag_core::{
    GuardAction, GuardConfig, GuardDecision, GuardMatch, GuardRuleKind, GuardSurface, OpenRagError,
};
use regex::Regex;
use unicode_normalization::UnicodeNormalization;
use uuid::Uuid;

use crate::concepts::{compact_term, normalize_term, InMemoryConceptStore};

#[derive(Debug, Clone)]
struct IndexedProjection {
    text: String,
    byte_spans: Vec<(usize, usize)>,
}

fn is_cluster_continuation(ch: char) -> bool {
    matches!(
        ch,
        '\u{0300}'..='\u{036F}'
            | '\u{1160}'..='\u{11FF}'
            | '\u{1AB0}'..='\u{1AFF}'
            | '\u{1DC0}'..='\u{1DFF}'
            | '\u{20D0}'..='\u{20FF}'
            | '\u{D7B0}'..='\u{D7FF}'
            | '\u{FE00}'..='\u{FE0F}'
            | '\u{FE20}'..='\u{FE2F}'
    )
}

fn segment_clusters(text: &str) -> Vec<(usize, usize, &str)> {
    let mut segments = Vec::new();
    let mut iter = text.char_indices().peekable();
    while let Some((start, _)) = iter.next() {
        let mut end = start
            + text[start..]
                .chars()
                .next()
                .map_or(0, char::len_utf8);
        while let Some(&(next_idx, next_ch)) = iter.peek() {
            if is_cluster_continuation(next_ch) {
                end = next_idx + next_ch.len_utf8();
                let _ = iter.next();
            } else {
                break;
            }
        }
        segments.push((start, end, &text[start..end]));
    }
    segments
}

fn build_projections(text: &str) -> (IndexedProjection, IndexedProjection) {
    let segments = segment_clusters(text);

    let mut norm_text = String::with_capacity(text.len());
    let mut norm_spans: Vec<(usize, usize)> = Vec::with_capacity(text.len());
    let mut pending_ws: Option<(usize, usize)> = None;

    let mut comp_text = String::with_capacity(text.len());
    let mut comp_spans: Vec<(usize, usize)> = Vec::with_capacity(text.len());

    let mut buf = [0_u8; 4];

    for (seg_start, seg_end, seg_slice) in segments {
        for nc in seg_slice.nfkc() {
            if nc.is_whitespace() {
                if !norm_text.is_empty() {
                    pending_ws = Some(match pending_ws {
                        Some((ps, _)) => (ps, seg_end),
                        None => (seg_start, seg_end),
                    });
                }
                continue;
            }

            if let Some(ws_span) = pending_ws.take() {
                norm_text.push(' ');
                norm_spans.push(ws_span);
            }

            for lc in nc.to_lowercase() {
                let encoded = lc.encode_utf8(&mut buf);
                norm_text.push_str(encoded);
                for _ in 0..encoded.len() {
                    norm_spans.push((seg_start, seg_end));
                }

                if lc.is_alphanumeric() {
                    comp_text.push_str(encoded);
                    for _ in 0..encoded.len() {
                        comp_spans.push((seg_start, seg_end));
                    }
                }
            }
        }
    }

    (
        IndexedProjection {
            text: norm_text,
            byte_spans: norm_spans,
        },
        IndexedProjection {
            text: comp_text,
            byte_spans: comp_spans,
        },
    )
}

fn find_all_occurrences(haystack: &str, needle: &str) -> Vec<(usize, usize)> {
    if needle.is_empty() || haystack.len() < needle.len() {
        return Vec::new();
    }
    let mut out = Vec::new();
    let mut pos = 0usize;
    while pos <= haystack.len() - needle.len() {
        if let Some(rel) = haystack[pos..].find(needle) {
            let start = pos + rel;
            let end = start + needle.len();
            out.push((start, end));
            let step = haystack[start..]
                .chars()
                .next()
                .map_or(1, char::len_utf8);
            pos = start + step;
        } else {
            break;
        }
    }
    out
}

fn map_projected_span(
    proj: &IndexedProjection,
    start: usize,
    end: usize,
    orig_text: &str,
) -> Option<(usize, usize)> {
    if start >= end || end > proj.byte_spans.len() {
        return None;
    }
    let orig_start = proj.byte_spans.get(start)?.0;
    let orig_end = proj.byte_spans.get(end - 1)?.1;
    if orig_start < orig_end
        && orig_end <= orig_text.len()
        && orig_text.is_char_boundary(orig_start)
        && orig_text.is_char_boundary(orig_end)
    {
        Some((orig_start, orig_end))
    } else {
        None
    }
}

fn is_covered_by_allowlist(allow_intervals: &[(usize, usize)], start: usize, end: usize) -> bool {
    allow_intervals
        .iter()
        .any(|&(as_, ae)| as_ <= start && end <= ae)
}

#[derive(Debug, Clone)]
struct MaskSpan<'a> {
    start: usize,
    end: usize,
    rule_index: usize,
    replacement: &'a str,
}

fn apply_mask_spans(text: &str, mut spans: Vec<MaskSpan<'_>>) -> String {
    if spans.is_empty() {
        return text.to_string();
    }
    spans.sort_by_key(|s| (s.start, Reverse(s.end), s.rule_index));

    let mut merged: Vec<(usize, usize, &str)> = Vec::with_capacity(spans.len());
    for sp in spans {
        if let Some(last) = merged.last_mut() {
            if sp.start < last.1 {
                if sp.end > last.1 {
                    last.1 = sp.end;
                }
                continue;
            }
        }
        merged.push((sp.start, sp.end, sp.replacement));
    }

    let mut out = String::with_capacity(text.len());
    let mut cursor = 0usize;
    for (start, end, repl) in merged {
        if cursor < start {
            out.push_str(&text[cursor..start]);
        }
        out.push_str(repl);
        cursor = end;
    }
    if cursor < text.len() {
        out.push_str(&text[cursor..]);
    }
    out
}

#[async_trait]
pub trait MemoryGuardHook: Send + Sync {
    async fn inspect(
        &self,
        org_id: Uuid,
        text: &str,
        surface: GuardSurface,
    ) -> Result<GuardDecision, OpenRagError>;
}

/// Built-in deterministic Guard hook used by `openrag-memory` (and replaceable by `openrag-pii::GuardEngine` in M4).
/// Supports Safe-Word/Keyword NFKC + compact anti-obfuscation matching, Concept/Alias expansion, Regex rules, and basic PII patterns.
#[derive(Debug, Clone)]
pub struct DefaultMemoryGuard {
    config: GuardConfig,
    concepts: InMemoryConceptStore,
}

impl Default for DefaultMemoryGuard {
    fn default() -> Self {
        Self::new(
            GuardConfig {
                enabled: true,
                default_action: GuardAction::Mask,
                allowlist: Vec::new(),
                rules: Vec::new(),
            },
            InMemoryConceptStore::new(),
        )
    }
}

impl DefaultMemoryGuard {
    pub fn new(config: GuardConfig, concepts: InMemoryConceptStore) -> Self {
        Self { config, concepts }
    }

    pub fn config(&self) -> &GuardConfig {
        &self.config
    }

    pub fn concepts(&self) -> &InMemoryConceptStore {
        &self.concepts
    }
}

#[async_trait]
impl MemoryGuardHook for DefaultMemoryGuard {
    async fn inspect(
        &self,
        org_id: Uuid,
        text: &str,
        surface: GuardSurface,
    ) -> Result<GuardDecision, OpenRagError> {
        if !self.config.enabled {
            return Ok(GuardDecision {
                action: GuardAction::Allow,
                sanitized_text: Some(text.to_string()),
                warnings: Vec::new(),
                matched_rules: Vec::new(),
            });
        }

        let (norm_proj, comp_proj) = build_projections(text);

        let mut allow_intervals: Vec<(usize, usize)> = Vec::new();
        for allow in &self.config.allowlist {
            let na = normalize_term(allow);
            if !na.is_empty() {
                for (s, e) in find_all_occurrences(&norm_proj.text, &na) {
                    if let Some(span) = map_projected_span(&norm_proj, s, e, text) {
                        allow_intervals.push(span);
                    }
                }
            }
            let ca = compact_term(allow);
            if !ca.is_empty() {
                for (s, e) in find_all_occurrences(&comp_proj.text, &ca) {
                    if let Some(span) = map_projected_span(&comp_proj, s, e, text) {
                        allow_intervals.push(span);
                    }
                }
            }
        }

        let mut highest_action = GuardAction::Allow;
        let mut matched_rules = Vec::new();
        let mut warnings = Vec::new();
        let mut mask_spans: Vec<MaskSpan<'_>> = Vec::new();

        for (idx, rule) in self.config.rules.iter().enumerate() {
            if !rule.enabled || !rule.surfaces.contains(&surface) {
                continue;
            }
            let mut matched = false;
            match rule.kind {
                GuardRuleKind::SafeWord | GuardRuleKind::Keyword => {
                    let expanded = self
                        .concepts
                        .expand_terms(org_id, std::slice::from_ref(&rule.pattern_or_category))
                        .await;
                    let repl = rule.replacement_text.as_deref().unwrap_or("[MASKED]");
                    let mut unlocalizable = false;

                    for candidate in expanded {
                        let nc = normalize_term(&candidate);
                        if !nc.is_empty() {
                            for (s, e) in find_all_occurrences(&norm_proj.text, &nc) {
                                if let Some((os, oe)) = map_projected_span(&norm_proj, s, e, text) {
                                    if !is_covered_by_allowlist(&allow_intervals, os, oe) {
                                        matched = true;
                                        if rule.action == GuardAction::Mask {
                                            mask_spans.push(MaskSpan {
                                                start: os,
                                                end: oe,
                                                rule_index: idx,
                                                replacement: repl,
                                            });
                                        }
                                    }
                                } else {
                                    matched = true;
                                    unlocalizable = true;
                                }
                            }
                        }

                        let cc = compact_term(&candidate);
                        if !cc.is_empty() {
                            for (s, e) in find_all_occurrences(&comp_proj.text, &cc) {
                                if let Some((os, oe)) = map_projected_span(&comp_proj, s, e, text) {
                                    if !is_covered_by_allowlist(&allow_intervals, os, oe) {
                                        matched = true;
                                        if rule.action == GuardAction::Mask {
                                            mask_spans.push(MaskSpan {
                                                start: os,
                                                end: oe,
                                                rule_index: idx,
                                                replacement: repl,
                                            });
                                        }
                                    }
                                } else {
                                    matched = true;
                                    unlocalizable = true;
                                }
                            }
                        }
                    }

                    if matched && rule.action == GuardAction::Mask && unlocalizable {
                        return Err(OpenRagError::GuardBlocked {
                            surface,
                            reason: format!(
                                "failed closed: unable to localize matched span for rule '{}'",
                                rule.pattern_or_category
                            ),
                            matched_rule_ids: Vec::new(),
                            matched_categories: vec![rule.pattern_or_category.clone()],
                        });
                    }
                }
                GuardRuleKind::Regex => {
                    if let Ok(re) = Regex::new(&rule.pattern_or_category) {
                        let repl = rule.replacement_text.as_deref().unwrap_or("[MASKED]");
                        for m in re.find_iter(text) {
                            if !is_covered_by_allowlist(&allow_intervals, m.start(), m.end()) {
                                matched = true;
                                if rule.action == GuardAction::Mask {
                                    mask_spans.push(MaskSpan {
                                        start: m.start(),
                                        end: m.end(),
                                        rule_index: idx,
                                        replacement: repl,
                                    });
                                }
                            }
                        }
                    }
                }
                GuardRuleKind::PiiCategory => {
                    if let Some(pat) = builtin_pii_regex(&rule.pattern_or_category) {
                        let repl = rule
                            .replacement_text
                            .as_deref()
                            .unwrap_or("[REDACTED_PII]");
                        for m in pat.find_iter(text) {
                            if !is_covered_by_allowlist(&allow_intervals, m.start(), m.end()) {
                                matched = true;
                                if rule.action == GuardAction::Mask {
                                    mask_spans.push(MaskSpan {
                                        start: m.start(),
                                        end: m.end(),
                                        rule_index: idx,
                                        replacement: repl,
                                    });
                                }
                            }
                        }
                    }
                }
            }

            if matched {
                if rule.action > highest_action {
                    highest_action = rule.action;
                }
                if rule.action == GuardAction::Warn {
                    warnings.push(format!(
                        "guard warning ({}): matched {}",
                        surface.as_str(),
                        rule.pattern_or_category
                    ));
                }
                matched_rules.push(GuardMatch {
                    rule_index: idx,
                    rule_id: None,
                    kind: rule.kind,
                    pattern_or_category: rule.pattern_or_category.clone(),
                    action: rule.action,
                });
            }
        }

        if highest_action == GuardAction::Block {
            let categories: Vec<String> = matched_rules
                .iter()
                .map(|m| m.pattern_or_category.clone())
                .collect();
            return Err(OpenRagError::GuardBlocked {
                surface,
                reason: format!("blocked by guard rule(s): {}", categories.join(", ")),
                matched_rule_ids: Vec::new(),
                matched_categories: categories,
            });
        }

        let sanitized = apply_mask_spans(text, mask_spans);

        Ok(GuardDecision {
            action: highest_action,
            sanitized_text: Some(sanitized),
            warnings,
            matched_rules,
        })
    }
}

fn builtin_pii_regex(category: &str) -> Option<Regex> {
    match category.trim() {
        "rrn_kr" => Regex::new(r"\b\d{6}[-–]\d{7}\b").ok(),
        "phone_kr" => Regex::new(r"\b01[016789][-\s]?\d{3,4}[-\s]?\d{4}\b").ok(),
        "email" => Regex::new(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b").ok(),
        _ => None,
    }
}
