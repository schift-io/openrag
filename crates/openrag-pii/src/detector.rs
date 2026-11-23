use std::sync::LazyLock;

use chrono::{DateTime, Utc};
use openrag_core::{OpenRagError, PiiConfig};
use regex::Regex;
use serde::{Deserialize, Serialize};
use sqlx::{PgPool, Row};
use uuid::Uuid;

use crate::category::PiiCategory;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DetectorProfile {
    pub id: Uuid,
    pub org_id: Uuid,
    pub name: String,
    pub use_schift_v7: bool,
    pub use_marker_ko_pii: bool,
    pub use_schift_v6_org: bool,
    pub custom_dict_terms: Vec<String>,
    pub allowlist_terms: Vec<String>,
    pub is_default: bool,
    pub updated_at: DateTime<Utc>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DetectorProfileInput {
    pub name: String,
    pub use_schift_v7: bool,
    pub use_marker_ko_pii: bool,
    pub use_schift_v6_org: bool,
    pub custom_dict_terms: Vec<String>,
    pub allowlist_terms: Vec<String>,
    pub is_default: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PiiSpan {
    pub category: PiiCategory,
    pub start_byte: usize,
    pub end_byte: usize,
    pub matched_text: String,
    pub detector_source: String,
    pub confidence: f32,
    pub detector_trace: serde_json::Value,
}

#[derive(Debug, Clone)]
pub struct PiiEngine {
    pub use_schift_v7: bool,
    pub use_marker_ko_pii: bool,
    pub use_schift_v6_org: bool,
    pub custom_dict_terms: Vec<String>,
    pub allowlist_terms: Vec<String>,
}

impl Default for PiiEngine {
    fn default() -> Self {
        Self {
            use_schift_v7: true,
            use_marker_ko_pii: false,
            use_schift_v6_org: false,
            custom_dict_terms: Vec::new(),
            allowlist_terms: Vec::new(),
        }
    }
}

impl PiiEngine {
    pub fn from_config(cfg: &PiiConfig) -> Self {
        Self {
            use_schift_v7: cfg.schift_v7,
            use_marker_ko_pii: cfg.marker_ko_pii,
            use_schift_v6_org: cfg.schift_v6,
            custom_dict_terms: cfg.custom_dict_terms.clone(),
            allowlist_terms: cfg.allowlist_terms.clone(),
        }
    }

    pub fn from_profile(p: &DetectorProfile) -> Self {
        Self {
            use_schift_v7: p.use_schift_v7,
            use_marker_ko_pii: p.use_marker_ko_pii,
            use_schift_v6_org: p.use_schift_v6_org,
            custom_dict_terms: p.custom_dict_terms.clone(),
            allowlist_terms: p.allowlist_terms.clone(),
        }
    }

    pub fn all_detectors_disabled(&self) -> bool {
        !self.use_schift_v7
            && !self.use_marker_ko_pii
            && !self.use_schift_v6_org
            && self
                .custom_dict_terms
                .iter()
                .all(|t| t.trim().is_empty())
    }

    pub fn detect(&self, text: &str) -> Vec<PiiSpan> {
        if text.trim().is_empty() || self.all_detectors_disabled() {
            return Vec::new();
        }

        let allowlist_spans = collect_allowlist_spans(text, &self.allowlist_terms);
        let mut candidates: Vec<(u8, PiiSpan)> = Vec::new();

        // 1. Custom dictionary terms (highest priority on equal/longer match)
        for term in &self.custom_dict_terms {
            let trimmed = term.trim();
            if trimmed.is_empty() {
                continue;
            }
            for (start, matched) in text.match_indices(trimmed) {
                candidates.push((
                    0,
                    PiiSpan {
                        category: PiiCategory::CustomDict,
                        start_byte: start,
                        end_byte: start + matched.len(),
                        matched_text: matched.to_string(),
                        detector_source: "custom_dict".to_string(),
                        confidence: 1.0,
                        detector_trace: serde_json::json!({
                            "rule": "custom_dict_exact",
                            "term": trimmed,
                        }),
                    },
                ));
            }
        }

        let ml_source = if self.use_schift_v7 {
            Some("schift-ko-pii-v7")
        } else if self.use_marker_ko_pii {
            Some("marker-ko-pii")
        } else if self.use_schift_v6_org {
            Some("schift-ko-pii-v6")
        } else {
            None
        };

        if let Some(primary_src) = ml_source {
            // Korean RRN (hyphenated)
            detect_rrn(text, primary_src, &mut candidates);
            // Unhyphenated 13-digit RRN when marker-ko-pii + explicit RRN label
            if self.use_marker_ko_pii {
                detect_unhyphenated_rrn(text, "marker-ko-pii", &mut candidates);
            }
            // Korean Phone
            detect_phone(text, primary_src, &mut candidates);
            // Email
            detect_email(text, primary_src, &mut candidates);
            // Credit Card (Luhn validated)
            detect_credit_card(text, primary_src, &mut candidates);
            // Passport
            detect_passport(text, primary_src, &mut candidates);
            // Bank Account (context-anchored)
            detect_bank_account(text, primary_src, &mut candidates);
            // Secret keys
            detect_secret_keys(text, primary_src, &mut candidates);

            // Korean Person Names & Addresses (enabled by v7 / marker_ko_pii / v6)
            let ner_src = if self.use_schift_v7 {
                "schift-ko-pii-v7"
            } else if self.use_marker_ko_pii {
                "marker-ko-pii"
            } else {
                "schift-ko-pii-v6"
            };
            detect_korean_person_names(text, ner_src, &mut candidates);
            detect_korean_addresses(text, ner_src, &mut candidates);
        }

        // Longest-span overlap resolution: start ASC, length DESC, priority ASC
        candidates.sort_by(|(pa, a), (pb, b)| {
            a.start_byte
                .cmp(&b.start_byte)
                .then_with(|| {
                    (b.end_byte - b.start_byte).cmp(&(a.end_byte - a.start_byte))
                })
                .then_with(|| pa.cmp(pb))
        });

        let mut selected: Vec<PiiSpan> = Vec::new();
        for (_, span) in candidates {
            if is_covered_by_allowlist(span.start_byte, span.end_byte, &allowlist_spans) {
                continue;
            }
            if let Some(last) = selected.last() {
                if span.start_byte < last.end_byte {
                    continue;
                }
            }
            selected.push(span);
        }

        selected
    }
}

fn collect_allowlist_spans(text: &str, allowlist_terms: &[String]) -> Vec<(usize, usize)> {
    let mut spans = Vec::new();
    for term in allowlist_terms {
        let trimmed = term.trim();
        if trimmed.is_empty() {
            continue;
        }
        for (start, matched) in text.match_indices(trimmed) {
            spans.push((start, start + matched.len()));
        }
    }
    spans
}

fn is_covered_by_allowlist(start: usize, end: usize, allowlist_spans: &[(usize, usize)]) -> bool {
    allowlist_spans
        .iter()
        .any(|&(as_, ae)| start >= as_ && end <= ae)
}

fn has_ascii_digit_boundary(text: &str, start: usize, end: usize) -> bool {
    let before_digit = text[..start]
        .chars()
        .next_back()
        .is_some_and(|c| c.is_ascii_digit());
    let after_digit = text[end..]
        .chars()
        .next()
        .is_some_and(|c| c.is_ascii_digit());
    !before_digit && !after_digit
}

pub fn rrn_mod11_checksum_valid(digits: &[u8; 13]) -> bool {
    const WEIGHTS: [u32; 12] = [2, 3, 4, 5, 6, 7, 8, 9, 2, 3, 4, 5];
    let mut sum = 0u32;
    for (i, &w) in WEIGHTS.iter().enumerate() {
        sum += (digits[i] as u32) * w;
    }
    let check = (11 - (sum % 11)) % 10;
    check == (digits[12] as u32)
}

pub fn validate_korean_rrn_digits(digits: &[u8; 13]) -> Option<bool> {
    let yy = (digits[0] as u32) * 10 + (digits[1] as u32);
    let mm = (digits[2] as u32) * 10 + (digits[3] as u32);
    let dd = (digits[4] as u32) * 10 + (digits[5] as u32);
    let gender = digits[6];

    let century: i32 = match gender {
        1 | 2 | 5 | 6 => 1900,
        3 | 4 | 7 | 8 => 2000,
        _ => return None,
    };
    if !(1..=12).contains(&mm) || dd == 0 {
        return None;
    }
    let year = century + (yy as i32);
    let max_day = match mm {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 => {
            let leap = (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0);
            if leap { 29 } else { 28 }
        }
        _ => return None,
    };
    if dd > max_day {
        return None;
    }

    Some(rrn_mod11_checksum_valid(digits))
}

fn parse_13_digits(raw: &str) -> Option<[u8; 13]> {
    let mut out = [0u8; 13];
    let mut idx = 0usize;
    for b in raw.bytes() {
        if b.is_ascii_digit() {
            if idx >= 13 {
                return None;
            }
            out[idx] = b - b'0';
            idx += 1;
        }
    }
    if idx == 13 { Some(out) } else { None }
}

fn detect_rrn(text: &str, source: &str, out: &mut Vec<(u8, PiiSpan)>) {
    static RE: LazyLock<Option<Regex>> =
        LazyLock::new(|| Regex::new(r"[0-9]{6}-[1-8][0-9]{6}").ok());
    let Some(re) = RE.as_ref() else {
        return;
    };
    for m in re.find_iter(text) {
        if !has_ascii_digit_boundary(text, m.start(), m.end()) {
            continue;
        }
        let Some(digits) = parse_13_digits(m.as_str()) else {
            continue;
        };
        let Some(checksum_valid) = validate_korean_rrn_digits(&digits) else {
            continue;
        };
        let confidence = if checksum_valid { 0.99 } else { 0.88 };
        out.push((
            1,
            PiiSpan {
                category: PiiCategory::RrnKr,
                start_byte: m.start(),
                end_byte: m.end(),
                matched_text: m.as_str().to_string(),
                detector_source: source.to_string(),
                confidence,
                detector_trace: serde_json::json!({
                    "rule": "kr_rrn",
                    "date_valid": true,
                    "gender_valid": true,
                    "checksum_valid": checksum_valid,
                }),
            },
        ));
    }
}

fn detect_unhyphenated_rrn(text: &str, source: &str, out: &mut Vec<(u8, PiiSpan)>) {
    static RE: LazyLock<Option<Regex>> = LazyLock::new(|| {
        Regex::new(r"(?:주민등록번호|주민번호|RRN)\s*[:\s]*([0-9]{6}[1-8][0-9]{6})").ok()
    });
    let Some(re) = RE.as_ref() else {
        return;
    };
    for caps in re.captures_iter(text) {
        let Some(m) = caps.get(1) else {
            continue;
        };
        if !has_ascii_digit_boundary(text, m.start(), m.end()) {
            continue;
        }
        let Some(digits) = parse_13_digits(m.as_str()) else {
            continue;
        };
        let Some(checksum_valid) = validate_korean_rrn_digits(&digits) else {
            continue;
        };
        out.push((
            1,
            PiiSpan {
                category: PiiCategory::RrnKr,
                start_byte: m.start(),
                end_byte: m.end(),
                matched_text: m.as_str().to_string(),
                detector_source: source.to_string(),
                confidence: if checksum_valid { 0.96 } else { 0.84 },
                detector_trace: serde_json::json!({
                    "rule": "kr_rrn_unhyphenated",
                    "date_valid": true,
                    "gender_valid": true,
                    "checksum_valid": checksum_valid,
                }),
            },
        ));
    }
}

fn detect_phone(text: &str, source: &str, out: &mut Vec<(u8, PiiSpan)>) {
    static RE_HYPHEN: LazyLock<Option<Regex>> = LazyLock::new(|| {
        Regex::new(r"(?:01[016789]|02|0[3-6][1-5]|070)-[0-9]{3,4}-[0-9]{4}").ok()
    });
    static RE_PLAIN: LazyLock<Option<Regex>> =
        LazyLock::new(|| Regex::new(r"01[016789][0-9]{7,8}").ok());

    if let Some(re) = RE_HYPHEN.as_ref() {
        for m in re.find_iter(text) {
            if !has_ascii_digit_boundary(text, m.start(), m.end()) {
                continue;
            }
            out.push((
                2,
                PiiSpan {
                    category: PiiCategory::PhoneKr,
                    start_byte: m.start(),
                    end_byte: m.end(),
                    matched_text: m.as_str().to_string(),
                    detector_source: source.to_string(),
                    confidence: 0.96,
                    detector_trace: serde_json::json!({"rule": "kr_phone_hyphen"}),
                },
            ));
        }
    }
    if let Some(re) = RE_PLAIN.as_ref() {
        for m in re.find_iter(text) {
            if !has_ascii_digit_boundary(text, m.start(), m.end()) {
                continue;
            }
            out.push((
                3,
                PiiSpan {
                    category: PiiCategory::PhoneKr,
                    start_byte: m.start(),
                    end_byte: m.end(),
                    matched_text: m.as_str().to_string(),
                    detector_source: source.to_string(),
                    confidence: 0.85,
                    detector_trace: serde_json::json!({"rule": "kr_phone_compact"}),
                },
            ));
        }
    }
}

fn detect_email(text: &str, source: &str, out: &mut Vec<(u8, PiiSpan)>) {
    static RE: LazyLock<Option<Regex>> = LazyLock::new(|| {
        Regex::new(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}").ok()
    });
    let Some(re) = RE.as_ref() else {
        return;
    };
    for m in re.find_iter(text) {
        out.push((
            2,
            PiiSpan {
                category: PiiCategory::Email,
                start_byte: m.start(),
                end_byte: m.end(),
                matched_text: m.as_str().to_string(),
                detector_source: source.to_string(),
                confidence: 0.98,
                detector_trace: serde_json::json!({"rule": "email"}),
            },
        ));
    }
}

pub fn luhn_valid(digits: &[u8]) -> bool {
    if digits.len() < 13 || digits.len() > 19 {
        return false;
    }
    let mut sum = 0u32;
    let mut double = false;
    for &d in digits.iter().rev() {
        let mut val = d as u32;
        if double {
            val *= 2;
            if val > 9 {
                val -= 9;
            }
        }
        sum += val;
        double = !double;
    }
    sum % 10 == 0
}

fn detect_credit_card(text: &str, source: &str, out: &mut Vec<(u8, PiiSpan)>) {
    static RE: LazyLock<Option<Regex>> = LazyLock::new(|| {
        Regex::new(r"[0-9]{4}(?:[\s\-]?[0-9]{4}){2,3}(?:[\s\-]?[0-9]{1,3})?").ok()
    });
    let Some(re) = RE.as_ref() else {
        return;
    };
    for m in re.find_iter(text) {
        if !has_ascii_digit_boundary(text, m.start(), m.end()) {
            continue;
        }
        let digits: Vec<u8> = m
            .as_str()
            .bytes()
            .filter(|b| b.is_ascii_digit())
            .map(|b| b - b'0')
            .collect();
        if !luhn_valid(&digits) {
            continue;
        }
        out.push((
            2,
            PiiSpan {
                category: PiiCategory::CreditCard,
                start_byte: m.start(),
                end_byte: m.end(),
                matched_text: m.as_str().to_string(),
                detector_source: source.to_string(),
                confidence: 0.95,
                detector_trace: serde_json::json!({"rule": "credit_card_luhn", "luhn_valid": true}),
            },
        ));
    }
}

fn detect_passport(text: &str, source: &str, out: &mut Vec<(u8, PiiSpan)>) {
    static RE: LazyLock<Option<Regex>> = LazyLock::new(|| {
        Regex::new(r"(?:[MSRDG][0-9]{8}|[MS][0-9]{3}[A-Z][0-9]{4})").ok()
    });
    let Some(re) = RE.as_ref() else {
        return;
    };
    for m in re.find_iter(text) {
        let before_alnum = text[..m.start()]
            .chars()
            .next_back()
            .is_some_and(|c| c.is_ascii_alphanumeric());
        let after_alnum = text[m.end()..]
            .chars()
            .next()
            .is_some_and(|c| c.is_ascii_alphanumeric());
        if before_alnum || after_alnum {
            continue;
        }
        out.push((
            3,
            PiiSpan {
                category: PiiCategory::Passport,
                start_byte: m.start(),
                end_byte: m.end(),
                matched_text: m.as_str().to_string(),
                detector_source: source.to_string(),
                confidence: 0.88,
                detector_trace: serde_json::json!({"rule": "kr_passport"}),
            },
        ));
    }
}

fn detect_bank_account(text: &str, source: &str, out: &mut Vec<(u8, PiiSpan)>) {
    static RE: LazyLock<Option<Regex>> = LazyLock::new(|| {
        Regex::new(r"(?:계좌|입금|은행|account)[^\r\n0-9]{0,16}([0-9]{2,4}-[0-9]{2,6}-[0-9]{4,7})")
            .ok()
    });
    let Some(re) = RE.as_ref() else {
        return;
    };
    for caps in re.captures_iter(text) {
        let Some(m) = caps.get(1) else {
            continue;
        };
        if !has_ascii_digit_boundary(text, m.start(), m.end()) {
            continue;
        }
        out.push((
            3,
            PiiSpan {
                category: PiiCategory::BankAccount,
                start_byte: m.start(),
                end_byte: m.end(),
                matched_text: m.as_str().to_string(),
                detector_source: source.to_string(),
                confidence: 0.87,
                detector_trace: serde_json::json!({"rule": "kr_bank_account_ctx"}),
            },
        ));
    }
}

fn detect_secret_keys(text: &str, source: &str, out: &mut Vec<(u8, PiiSpan)>) {
    static RE: LazyLock<Option<Regex>> = LazyLock::new(|| {
        Regex::new(r"(?i)(?:sk-[a-z0-9\-_]{16,}|AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{20,})").ok()
    });
    let Some(re) = RE.as_ref() else {
        return;
    };
    for m in re.find_iter(text) {
        out.push((
            1,
            PiiSpan {
                category: PiiCategory::SecretKey,
                start_byte: m.start(),
                end_byte: m.end(),
                matched_text: m.as_str().to_string(),
                detector_source: source.to_string(),
                confidence: 0.99,
                detector_trace: serde_json::json!({"rule": "secret_key_prefix"}),
            },
        ));
    }
}

fn detect_korean_person_names(text: &str, source: &str, out: &mut Vec<(u8, PiiSpan)>) {
    // Capture ONLY the Korean name after explicit label (`성명:`, `담당자:`, `이름:`, `신청인:`, `수신:`)
    static RE_LABELED: LazyLock<Option<Regex>> = LazyLock::new(|| {
        Regex::new(r"(?:성명|담당자|이름|작성자|신청인|수신)\s*[:：]?\s*([가-힣]{2,4})").ok()
    });
    if let Some(re) = RE_LABELED.as_ref() {
        for caps in re.captures_iter(text) {
            let Some(m) = caps.get(1) else {
                continue;
            };
            out.push((
                4,
                PiiSpan {
                    category: PiiCategory::PersonName,
                    start_byte: m.start(),
                    end_byte: m.end(),
                    matched_text: m.as_str().to_string(),
                    detector_source: source.to_string(),
                    confidence: 0.90,
                    detector_trace: serde_json::json!({"rule": "kr_name_labeled"}),
                },
            ));
        }
    }

    // Honorific/title-anchored common Korean surname + 2 given-name syllables
    static RE_HONORIFIC: LazyLock<Option<Regex>> = LazyLock::new(|| {
        Regex::new(r"([김이박최정강조윤장임한오서신권황안송류홍전고문양손배조백허유남심노정하곽성차주우구신임전민유류][가-힣]{1,2})(?:님|씨|과장|부장|팀장|대리|연구원|주무관)").ok()
    });
    if let Some(re) = RE_HONORIFIC.as_ref() {
        for caps in re.captures_iter(text) {
            let Some(m) = caps.get(1) else {
                continue;
            };
            let before_hangul = text[..m.start()]
                .chars()
                .next_back()
                .is_some_and(|c| ('\u{AC00}'..='\u{D7A3}').contains(&c));
            if before_hangul {
                continue;
            }
            out.push((
                5,
                PiiSpan {
                    category: PiiCategory::PersonName,
                    start_byte: m.start(),
                    end_byte: m.end(),
                    matched_text: m.as_str().to_string(),
                    detector_source: source.to_string(),
                    confidence: 0.82,
                    detector_trace: serde_json::json!({"rule": "kr_name_honorific"}),
                },
            ));
        }
    }

    // Full name, a space, then a job title: "김민준 과장은", "이서연 대리". Only three-syllable
    // names (surname + two), so two-syllable words ("신입 사원", "전체 사원") stay out; words
    // ending like a unit name ("전산실 과장", "구매팀 차장") stay out; the title must end the
    // word or take a particle.
    static RE_SPACED_TITLE: LazyLock<Option<Regex>> = LazyLock::new(|| {
        Regex::new(r"([김이박최정강조윤장임한오서신권황안송류홍전고문양손배백허유남심노하곽성차주우구민][가-힣]{2}) (사원|주임|대리|과장|차장|부장|팀장|실장|본부장|이사|상무|전무|대표|선임|책임|수석|연구원|주무관|교수|박사)").ok()
    });
    if let Some(re) = RE_SPACED_TITLE.as_ref() {
        for caps in re.captures_iter(text) {
            let (Some(m), Some(title)) = (caps.get(1), caps.get(2)) else {
                continue;
            };
            let before_hangul = text[..m.start()]
                .chars()
                .next_back()
                .is_some_and(|c| ('\u{AC00}'..='\u{D7A3}').contains(&c));
            let unit_suffix = m
                .as_str()
                .chars()
                .next_back()
                .is_some_and(|c| "팀실부과처국청소사원단회".contains(c));
            let mut after = text[title.end()..].chars();
            let title_ends_word = match after.next() {
                None => true,
                Some(c) if !('\u{AC00}'..='\u{D7A3}').contains(&c) => true,
                Some(c) if "은는이가을를의께도와과님".contains(c) => true,
                Some('에') => after.next() == Some('게'),
                Some(_) => false,
            };
            if before_hangul || unit_suffix || !title_ends_word {
                continue;
            }
            out.push((
                5,
                PiiSpan {
                    category: PiiCategory::PersonName,
                    start_byte: m.start(),
                    end_byte: m.end(),
                    matched_text: m.as_str().to_string(),
                    detector_source: source.to_string(),
                    confidence: 0.80,
                    detector_trace: serde_json::json!({"rule": "kr_name_spaced_title"}),
                },
            ));
        }
    }
}

fn detect_korean_addresses(text: &str, source: &str, out: &mut Vec<(u8, PiiSpan)>) {
    static RE: LazyLock<Option<Regex>> = LazyLock::new(|| {
        Regex::new(r"(?:서울특별시|부산광역시|대구광역시|인천광역시|광주광역시|대전광역시|울산광역시|세종특별자치시|[가-힣]{1,4}도)\s+[가-힣0-9]+(?:시|군|구)\s+[가-힣0-9\s\-]+?(?:로|길|동|읍|면)(?:\s+[0-9\-]+)?(?:\s+[가-힣0-9]+(?:호|층|동))?").ok()
    });
    let Some(re) = RE.as_ref() else {
        return;
    };
    for m in re.find_iter(text) {
        out.push((
            3,
            PiiSpan {
                category: PiiCategory::AddressKr,
                start_byte: m.start(),
                end_byte: m.end(),
                matched_text: m.as_str().to_string(),
                detector_source: source.to_string(),
                confidence: 0.88,
                detector_trace: serde_json::json!({"rule": "kr_address"}),
            },
        ));
    }
}

/// Privileged/worker repository for `pii.detector_profiles`.
pub struct PiiProfileRepo;

impl PiiProfileRepo {
    pub async fn upsert_profile(
        worker_pool: &PgPool,
        org_id: Uuid,
        input: &DetectorProfileInput,
    ) -> Result<DetectorProfile, OpenRagError> {
        let mut tx = worker_pool.begin().await.map_err(|e| {
            OpenRagError::Database(format!("failed starting detector_profile tx: {e}"))
        })?;

        if input.is_default {
            sqlx::query(
                "UPDATE pii.detector_profiles SET is_default = false WHERE org_id = $1 AND name <> $2",
            )
            .bind(org_id)
            .bind(&input.name)
            .execute(&mut *tx)
            .await
            .map_err(|e| {
                OpenRagError::Database(format!("failed clearing previous default profile: {e}"))
            })?;
        }

        let row = sqlx::query(
            "INSERT INTO pii.detector_profiles (
                org_id, name, use_schift_v7, use_marker_ko_pii, use_schift_v6_org,
                custom_dict_terms, allowlist_terms, is_default, updated_at
             ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, now())
             ON CONFLICT (org_id, name) DO UPDATE SET
                use_schift_v7 = EXCLUDED.use_schift_v7,
                use_marker_ko_pii = EXCLUDED.use_marker_ko_pii,
                use_schift_v6_org = EXCLUDED.use_schift_v6_org,
                custom_dict_terms = EXCLUDED.custom_dict_terms,
                allowlist_terms = EXCLUDED.allowlist_terms,
                is_default = EXCLUDED.is_default,
                updated_at = now()
             RETURNING id, org_id, name, use_schift_v7, use_marker_ko_pii, use_schift_v6_org,
                       custom_dict_terms, allowlist_terms, is_default, updated_at",
        )
        .bind(org_id)
        .bind(&input.name)
        .bind(input.use_schift_v7)
        .bind(input.use_marker_ko_pii)
        .bind(input.use_schift_v6_org)
        .bind(&input.custom_dict_terms)
        .bind(&input.allowlist_terms)
        .bind(input.is_default)
        .fetch_one(&mut *tx)
        .await
        .map_err(|e| {
            OpenRagError::Database(format!("failed upserting pii.detector_profiles: {e}"))
        })?;

        tx.commit().await.map_err(|e| {
            OpenRagError::Database(format!("failed committing detector_profile tx: {e}"))
        })?;

        row_to_profile(&row)
    }

    pub async fn get_active_profile(
        worker_pool: &PgPool,
        org_id: Uuid,
    ) -> Result<Option<DetectorProfile>, OpenRagError> {
        let row = sqlx::query(
            "SELECT id, org_id, name, use_schift_v7, use_marker_ko_pii, use_schift_v6_org,
                    custom_dict_terms, allowlist_terms, is_default, updated_at
             FROM pii.detector_profiles
             WHERE org_id = $1
             ORDER BY is_default DESC, updated_at DESC
             LIMIT 1",
        )
        .bind(org_id)
        .fetch_optional(worker_pool)
        .await
        .map_err(|e| {
            OpenRagError::Database(format!("failed querying pii.detector_profiles: {e}"))
        })?;

        row.as_ref().map(row_to_profile).transpose()
    }
}

fn row_to_profile(row: &sqlx::postgres::PgRow) -> Result<DetectorProfile, OpenRagError> {
    Ok(DetectorProfile {
        id: row
            .try_get("id")
            .map_err(|e| OpenRagError::Database(format!("invalid profile id: {e}")))?,
        org_id: row
            .try_get("org_id")
            .map_err(|e| OpenRagError::Database(format!("invalid profile org_id: {e}")))?,
        name: row
            .try_get("name")
            .map_err(|e| OpenRagError::Database(format!("invalid profile name: {e}")))?,
        use_schift_v7: row
            .try_get("use_schift_v7")
            .map_err(|e| OpenRagError::Database(format!("invalid use_schift_v7: {e}")))?,
        use_marker_ko_pii: row
            .try_get("use_marker_ko_pii")
            .map_err(|e| OpenRagError::Database(format!("invalid use_marker_ko_pii: {e}")))?,
        use_schift_v6_org: row
            .try_get("use_schift_v6_org")
            .map_err(|e| OpenRagError::Database(format!("invalid use_schift_v6_org: {e}")))?,
        custom_dict_terms: row
            .try_get("custom_dict_terms")
            .map_err(|e| OpenRagError::Database(format!("invalid custom_dict_terms: {e}")))?,
        allowlist_terms: row
            .try_get("allowlist_terms")
            .map_err(|e| OpenRagError::Database(format!("invalid allowlist_terms: {e}")))?,
        is_default: row
            .try_get("is_default")
            .map_err(|e| OpenRagError::Database(format!("invalid is_default: {e}")))?,
        updated_at: row
            .try_get("updated_at")
            .map_err(|e| OpenRagError::Database(format!("invalid updated_at: {e}")))?,
    })
}
