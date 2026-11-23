use std::collections::HashSet;
use std::sync::LazyLock;

use regex::Regex;

/// Extracts parenthetical cross-lingual / acronym concept pairs of the form
/// `한글(English)` or `한글（English）`, e.g. `제안요청서(RFP)`, `사과(apple)`,
/// `인공지능(Artificial Intelligence)`.
///
/// Filters out pure numbers (`(2026)`), citations (`(p.3)`, `(pp. 1-3)`),
/// and parentheticals lacking ASCII letters.
pub fn extract_parenthetical_aliases(text: &str) -> Vec<(String, String)> {
    static PAIR_RE: LazyLock<Option<Regex>> = LazyLock::new(|| {
        Regex::new(r"([^\s\(\)\[\]\{\}<>《》「」『』.,;:!?/\\|&*+=~\u{FF08}\u{FF09}]{1,40})[\(\u{FF08}]([^\)\u{FF09}\r\n]{2,80})[\)\u{FF09}]").ok()
    });
    static CITATION_RE: LazyLock<Option<Regex>> = LazyLock::new(|| {
        Regex::new(r"^(?:pp?\.?\s*\d+[\d\s,\-–—]*|fig\.?\s*\d+|table\s*\d+|v\d+(?:\.\d+)*)$").ok()
    });

    let Some(re) = PAIR_RE.as_ref() else {
        return Vec::new();
    };

    let mut out = Vec::new();
    let mut seen: HashSet<(String, String)> = HashSet::new();

    for caps in re.captures_iter(text) {
        let Some(left_m) = caps.get(1) else {
            continue;
        };
        let Some(right_m) = caps.get(2) else {
            continue;
        };

        let left = trim_korean_term(left_m.as_str());
        let right = right_m.as_str().trim();

        if left.is_empty() || right.len() < 2 {
            continue;
        }
        if !contains_hangul_syllable(left) {
            continue;
        }
        if !right.chars().any(|c| c.is_ascii_alphabetic()) {
            continue;
        }
        let right_lower = right.to_ascii_lowercase();
        if let Some(cit_re) = CITATION_RE.as_ref() {
            if cit_re.is_match(&right_lower) {
                continue;
            }
        }

        let pair = (left.to_string(), right.to_string());
        if seen.insert(pair.clone()) {
            out.push(pair);
        }
    }

    out
}

fn contains_hangul_syllable(s: &str) -> bool {
    s.chars().any(|c| ('\u{AC00}'..='\u{D7A3}').contains(&c))
}

fn trim_korean_term(raw: &str) -> &str {
    raw.trim_matches(|c: char| {
        c.is_whitespace() || matches!(c, '"' | '\'' | '“' | '”' | '‘' | '’' | '·' | ',' | ':')
    })
}
