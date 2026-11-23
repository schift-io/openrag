use std::collections::{HashMap, HashSet};

use openrag_core::OpenRagError;
use sqlx::{PgPool, Row};
use unicode_normalization::UnicodeNormalization;
use uuid::Uuid;

pub trait ConceptAliasLookup: Send + Sync {
    fn expand_aliases(&self, term: &str) -> Vec<String>;
}

pub trait SemanticEmbedder128: Send + Sync {
    fn embed_128d(&self, text: &str) -> [f32; 128];
}

pub fn is_zero_width(ch: char) -> bool {
    matches!(
        ch,
        '\u{200B}' | '\u{200C}' | '\u{200D}' | '\u{FEFF}' | '\u{00AD}' | '\u{2060}'
    )
}

pub fn is_obfuscation_separator(ch: char) -> bool {
    ch.is_whitespace()
        || matches!(
            ch,
            '-' | '_' | '.' | '/' | '·' | ':' | ',' | '‐' | '‑' | '–' | '—' | '−' | '•' | '*'
        )
}

pub fn normalize_nfkc_lower(input: &str) -> String {
    let mut filtered = String::with_capacity(input.len());
    for ch in input.chars() {
        if is_stripped_for_projection(ch, false) {
            continue;
        }
        filtered.push(ch);
    }
    let mut out = String::with_capacity(filtered.len());
    for ch in filtered.nfkc() {
        if is_zero_width(ch) {
            continue;
        }
        for lc in ch.to_lowercase() {
            out.push(lc);
        }
    }
    out
}

pub fn normalize_compact(input: &str) -> String {
    let mut filtered = String::with_capacity(input.len());
    for ch in input.chars() {
        if is_stripped_for_projection(ch, true) {
            continue;
        }
        filtered.push(ch);
    }
    let mut out = String::with_capacity(filtered.len());
    for ch in filtered.nfkc() {
        if is_zero_width(ch) || is_obfuscation_separator(ch) {
            continue;
        }
        for lc in ch.to_lowercase() {
            out.push(lc);
        }
    }
    out
}

/// Indexed view of an input string mapping NFKC-normalized & compact-normalized character runs
/// back to exact UTF-8 byte offsets `[orig_start..orig_end]` in the original `&str`.
#[derive(Debug, Clone)]
pub struct IndexedText {
    pub original: String,
    pub nfkc_lower: String,
    pub nfkc_spans: Vec<(usize, usize)>,
    pub compact: String,
    pub compact_spans: Vec<(usize, usize)>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum HangulClusterState {
    None,
    Leading,
    Vowel,
    Trailing,
}

fn is_hangul_leading_jamo(ch: char) -> bool {
    matches!(ch, '\u{1100}'..='\u{115F}' | '\u{A960}'..='\u{A97F}')
}

fn is_hangul_vowel_jamo(ch: char) -> bool {
    matches!(ch, '\u{1160}'..='\u{11A7}' | '\u{D7B0}'..='\u{D7C6}')
}

fn is_hangul_trailing_jamo(ch: char) -> bool {
    matches!(ch, '\u{11A8}'..='\u{11FF}' | '\u{D7CB}'..='\u{D7FB}')
}

fn is_precomposed_hangul_lv(ch: char) -> bool {
    ('\u{AC00}'..='\u{D7A3}').contains(&ch) && ((ch as u32 - 0xAC00) % 28 == 0)
}

fn is_variation_selector(ch: char) -> bool {
    matches!(ch, '\u{FE00}'..='\u{FE0F}' | '\u{E0100}'..='\u{E01EF}')
}

struct Cluster {
    text: String,
    orig_start: usize,
    orig_end: usize,
}

fn is_stripped_for_projection(ch: char, strip_separators: bool) -> bool {
    if is_zero_width(ch) || (strip_separators && is_obfuscation_separator(ch)) {
        return true;
    }
    if ch.is_ascii() {
        return false;
    }
    ch.nfkc()
        .all(|c| is_zero_width(c) || (strip_separators && is_obfuscation_separator(c)))
}

fn build_projection(input: &str, strip_separators: bool) -> (String, Vec<(usize, usize)>) {
    let mut clusters: Vec<Cluster> = Vec::new();
    let mut hangul_state = HangulClusterState::None;

    for (orig_start, orig_ch) in input.char_indices() {
        if is_stripped_for_projection(orig_ch, strip_separators) {
            continue;
        }
        let orig_end = orig_start + orig_ch.len_utf8();
        let ccc = unicode_normalization::char::canonical_combining_class(orig_ch);
        let attach_to_prev = !clusters.is_empty()
            && (ccc != 0
                || is_variation_selector(orig_ch)
                || (hangul_state == HangulClusterState::Leading
                    && is_hangul_vowel_jamo(orig_ch))
                || (hangul_state == HangulClusterState::Vowel
                    && is_hangul_trailing_jamo(orig_ch)));

        if attach_to_prev {
            if let Some(last) = clusters.last_mut() {
                last.text.push(orig_ch);
                last.orig_end = orig_end;
            }
            if is_hangul_vowel_jamo(orig_ch) && hangul_state == HangulClusterState::Leading {
                hangul_state = HangulClusterState::Vowel;
            } else if is_hangul_trailing_jamo(orig_ch)
                && hangul_state == HangulClusterState::Vowel
            {
                hangul_state = HangulClusterState::Trailing;
            }
        } else {
            let mut text = String::with_capacity(orig_ch.len_utf8());
            text.push(orig_ch);
            clusters.push(Cluster {
                text,
                orig_start,
                orig_end,
            });
            hangul_state = if is_hangul_leading_jamo(orig_ch) {
                HangulClusterState::Leading
            } else if is_precomposed_hangul_lv(orig_ch) {
                HangulClusterState::Vowel
            } else {
                HangulClusterState::None
            };
        }
    }

    let mut projected = String::with_capacity(input.len());
    let mut spans: Vec<(usize, usize)> = Vec::with_capacity(input.len());

    for cluster in clusters {
        for norm_ch in cluster.text.nfkc() {
            if is_zero_width(norm_ch) || (strip_separators && is_obfuscation_separator(norm_ch)) {
                continue;
            }
            for lc in norm_ch.to_lowercase() {
                let lc_len = lc.len_utf8();
                for _ in 0..lc_len {
                    spans.push((cluster.orig_start, cluster.orig_end));
                }
                projected.push(lc);
            }
        }
    }

    (projected, spans)
}

impl IndexedText {
    pub fn new(input: &str) -> Self {
        let (nfkc_lower, nfkc_spans) = build_projection(input, false);
        let (compact, compact_spans) = build_projection(input, true);

        Self {
            original: input.to_string(),
            nfkc_lower,
            nfkc_spans,
            compact,
            compact_spans,
        }
    }

    /// Finds all matches of `needle` across both NFKC-lowercased and compact
    /// (whitespace/punctuation/zero-width stripped) projections, returning original UTF-8 byte ranges.
    pub fn find_term_orig_spans(&self, needle: &str) -> Vec<(usize, usize)> {
        let mut ranges: Vec<(usize, usize)> = Vec::new();

        let norm_needle = normalize_nfkc_lower(needle);
        if !norm_needle.is_empty() {
            for (byte_idx, matched) in self.nfkc_lower.match_indices(&norm_needle) {
                let end_idx = byte_idx + matched.len();
                if let (Some(&(orig_s, _)), Some(&(_, orig_e))) = (
                    self.nfkc_spans.get(byte_idx),
                    self.nfkc_spans.get(end_idx.saturating_sub(1)),
                ) {
                    if orig_s < orig_e {
                        ranges.push((orig_s, orig_e));
                    }
                }
            }
        }

        let compact_needle = normalize_compact(needle);
        if !compact_needle.is_empty() {
            for (byte_idx, matched) in self.compact.match_indices(&compact_needle) {
                let end_idx = byte_idx + matched.len();
                if let (Some(&(orig_s, _)), Some(&(_, orig_e))) = (
                    self.compact_spans.get(byte_idx),
                    self.compact_spans.get(end_idx.saturating_sub(1)),
                ) {
                    if orig_s < orig_e {
                        ranges.push((orig_s, orig_e));
                    }
                }
            }
        }

        ranges.sort_unstable();
        ranges.dedup();
        ranges
    }
}

#[derive(Debug, Clone, Default)]
pub struct InMemoryConceptAliasLookup {
    /// Maps normalized compact key -> all surface aliases in that equivalence group.
    groups: HashMap<String, Vec<String>>,
}

impl InMemoryConceptAliasLookup {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn link_terms(&mut self, terms: &[&str]) {
        let mut merged: Vec<String> = Vec::new();
        let mut seen_norm: HashSet<String> = HashSet::new();

        for &t in terms {
            let trimmed = t.trim();
            if trimmed.is_empty() {
                continue;
            }
            let key = normalize_compact(trimmed);
            if let Some(existing) = self.groups.get(&key) {
                for item in existing {
                    let nk = normalize_compact(item);
                    if seen_norm.insert(nk) {
                        merged.push(item.clone());
                    }
                }
            }
            if seen_norm.insert(key) {
                merged.push(trimmed.to_string());
            }
        }

        for item in &merged {
            let key = normalize_compact(item);
            self.groups.insert(key, merged.clone());
        }
    }

    pub async fn load_from_pg(pool: &PgPool, org_id: Uuid) -> Result<Self, OpenRagError> {
        let rows = sqlx::query(
            "SELECT concept_id, alias_text
             FROM wiki.concept_aliases
             WHERE org_id = $1
             ORDER BY created_at ASC",
        )
        .bind(org_id)
        .fetch_all(pool)
        .await
        .map_err(|e| OpenRagError::Database(format!("failed loading wiki.concept_aliases: {e}")))?;

        let mut by_concept: HashMap<Uuid, Vec<String>> = HashMap::new();
        for row in rows {
            let cid: Uuid = row
                .try_get("concept_id")
                .map_err(|e| OpenRagError::Database(e.to_string()))?;
            let alias: String = row
                .try_get("alias_text")
                .map_err(|e| OpenRagError::Database(e.to_string()))?;
            by_concept.entry(cid).or_default().push(alias);
        }

        let mut lookup = Self::new();
        for (_cid, aliases) in by_concept {
            let refs: Vec<&str> = aliases.iter().map(String::as_str).collect();
            lookup.link_terms(&refs);
        }
        Ok(lookup)
    }
}

impl ConceptAliasLookup for InMemoryConceptAliasLookup {
    fn expand_aliases(&self, term: &str) -> Vec<String> {
        let key = normalize_compact(term);
        if let Some(group) = self.groups.get(&key) {
            if !group.is_empty() {
                return group.clone();
            }
        }
        vec![term.to_string()]
    }
}

/// Deterministic offline 128d semantic embedder using character 1..=3 gram hashing + L2 norm.
#[derive(Debug, Clone, Default)]
pub struct DeterministicSemanticEmbedder128;

impl DeterministicSemanticEmbedder128 {
    pub const fn new() -> Self {
        Self
    }
}

impl SemanticEmbedder128 for DeterministicSemanticEmbedder128 {
    fn embed_128d(&self, text: &str) -> [f32; 128] {
        let mut vec = [0.0f32; 128];
        let norm = normalize_compact(text);
        let chars: Vec<char> = norm.chars().collect();
        if chars.is_empty() {
            vec[0] = 1.0;
            return vec;
        }
        for n in 1usize..=3usize {
            if chars.len() < n {
                continue;
            }
            for win in chars.windows(n) {
                let mut h: u64 = 0xcbf29ce484222325;
                for &c in win {
                    h ^= c as u64;
                    h = h.wrapping_mul(0x100000001b3);
                }
                let idx = (h as usize) % 128;
                let sign = if (h >> 63) == 0 { 1.0f32 } else { -1.0f32 };
                vec[idx] += sign * (n as f32);
            }
        }
        let norm_sq: f64 = vec.iter().map(|&v| (v as f64) * (v as f64)).sum();
        let len = norm_sq.sqrt();
        if len > 0.0 && len.is_finite() {
            for v in &mut vec {
                *v = ((*v as f64) / len) as f32;
            }
        } else {
            vec[0] = 1.0;
        }
        vec
    }
}

pub fn cosine_similarity_128d(a: &[f32; 128], b: &[f32; 128]) -> f32 {
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    for i in 0..128 {
        let av = a[i] as f64;
        let bv = b[i] as f64;
        dot += av * bv;
        na += av * av;
        nb += bv * bv;
    }
    let denom = na.sqrt() * nb.sqrt();
    if denom <= 0.0 || !denom.is_finite() {
        0.0
    } else {
        (dot / denom) as f32
    }
}
