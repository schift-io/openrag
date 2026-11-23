use std::collections::HashSet;

use chrono::{DateTime, Utc};
use openrag_core::{
    GuardAction, GuardConfig, GuardDecision, GuardMatch, GuardRuleConfig, GuardRuleKind,
    GuardSurface, OpenRagError,
};
use openrag_db::RlsTransaction;
use regex::Regex;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use sqlx::{PgPool, Row};
use uuid::Uuid;

use crate::category::PiiCategory;
use crate::concepts::{
    cosine_similarity_128d, normalize_compact, ConceptAliasLookup, InMemoryConceptAliasLookup,
    IndexedText, SemanticEmbedder128,
};
use crate::detector::{PiiEngine, PiiSpan};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GuardRuleRuntime {
    pub id: Option<Uuid>,
    pub kind: GuardRuleKind,
    pub pattern_or_category: String,
    pub action: GuardAction,
    pub surfaces: Vec<GuardSurface>,
    pub replacement_text: Option<String>,
    pub enabled: bool,
    pub priority: i32,
}

impl From<&GuardRuleConfig> for GuardRuleRuntime {
    fn from(r: &GuardRuleConfig) -> Self {
        Self {
            id: None,
            kind: r.kind,
            pattern_or_category: r.pattern_or_category.clone(),
            action: r.action,
            surfaces: r.surfaces.clone(),
            replacement_text: r.replacement_text.clone(),
            enabled: r.enabled,
            priority: r.priority,
        }
    }
}

#[derive(Debug, Clone)]
pub struct GuardPolicyRuntime {
    pub id: Option<Uuid>,
    pub org_id: Uuid,
    pub master_enabled: bool,
    pub default_action: GuardAction,
    pub allowlist_terms: Vec<String>,
    pub rules: Vec<GuardRuleRuntime>,
    pub semantic_threshold: Option<f32>,
}

impl GuardPolicyRuntime {
    pub fn from_guard_config(org_id: Uuid, cfg: &GuardConfig) -> Self {
        Self {
            id: None,
            org_id,
            master_enabled: cfg.enabled,
            default_action: cfg.default_action,
            allowlist_terms: cfg.allowlist.clone(),
            rules: cfg.rules.iter().map(GuardRuleRuntime::from).collect(),
            semantic_threshold: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GuardPolicyRow {
    pub id: Uuid,
    pub org_id: Uuid,
    pub bucket_id: Option<Uuid>,
    pub name: String,
    pub master_enabled: bool,
    pub default_action: GuardAction,
    pub allowlist_terms: Vec<String>,
    pub surfaces: Vec<GuardSurface>,
    pub is_active: bool,
    pub updated_at: DateTime<Utc>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GuardRuleRow {
    pub id: Uuid,
    pub policy_id: Uuid,
    pub org_id: Uuid,
    pub kind: GuardRuleKind,
    pub pattern_or_category: String,
    pub action: GuardAction,
    pub surfaces: Vec<GuardSurface>,
    pub replacement_text: Option<String>,
    pub priority: i32,
    pub enabled: bool,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone)]
struct SpanReplacement {
    start_byte: usize,
    end_byte: usize,
    action: GuardAction,
    priority: i32,
    replacement: String,
}

pub struct GuardEngine;

impl GuardEngine {
    pub fn evaluate(
        text: &str,
        surface: GuardSurface,
        policy: &GuardPolicyRuntime,
        pii_engine: &PiiEngine,
        concept_lookup: Option<&dyn ConceptAliasLookup>,
        semantic_embedder: Option<&dyn SemanticEmbedder128>,
    ) -> Result<GuardDecision, OpenRagError> {
        if !policy.master_enabled {
            return Ok(GuardDecision {
                action: GuardAction::Allow,
                sanitized_text: Some(text.to_string()),
                warnings: Vec::new(),
                matched_rules: Vec::new(),
            });
        }

        let indexed = IndexedText::new(text);
        let allowlist_spans = collect_guard_allowlist_spans(&indexed, &policy.allowlist_terms);

        let default_lookup = InMemoryConceptAliasLookup::new();
        let lookup = concept_lookup.unwrap_or(&default_lookup);

        let mut active_rules: Vec<(usize, &GuardRuleRuntime)> = policy
            .rules
            .iter()
            .enumerate()
            .filter(|(_, r)| r.enabled && r.surfaces.contains(&surface))
            .collect();
        active_rules.sort_by(|(ia, a), (ib, b)| a.priority.cmp(&b.priority).then_with(|| ia.cmp(ib)));

        let needs_pii = active_rules
            .iter()
            .any(|(_, r)| r.kind == GuardRuleKind::PiiCategory);
        let pii_spans: Vec<PiiSpan> = if needs_pii {
            pii_engine
                .detect(text)
                .into_iter()
                .filter(|s| !covered_by_allowlist(s.start_byte, s.end_byte, &allowlist_spans))
                .collect()
        } else {
            Vec::new()
        };

        let mut matched_rules: Vec<GuardMatch> = Vec::new();
        let mut warnings: Vec<String> = Vec::new();
        let mut replacements: Vec<SpanReplacement> = Vec::new();

        for (rule_idx, rule) in active_rules {
            let mut hit_spans: Vec<(usize, usize)> = Vec::new();
            let mut semantic_hit = false;

            match rule.kind {
                GuardRuleKind::PiiCategory => {
                    let target_cat = PiiCategory::parse_alias(&rule.pattern_or_category);
                    for s in &pii_spans {
                        let cat_matches = match target_cat {
                            Some(cat) => s.category == cat,
                            None => {
                                s.category.as_canonical_str().eq_ignore_ascii_case(&rule.pattern_or_category)
                            }
                        };
                        if cat_matches {
                            hit_spans.push((s.start_byte, s.end_byte));
                        }
                    }
                }
                GuardRuleKind::SafeWord | GuardRuleKind::Keyword => {
                    let aliases = lookup.expand_aliases(&rule.pattern_or_category);
                    for alias in &aliases {
                        for (s, e) in indexed.find_term_orig_spans(alias) {
                            if !covered_by_allowlist(s, e, &allowlist_spans) {
                                hit_spans.push((s, e));
                            }
                        }
                    }

                    if hit_spans.is_empty() {
                        if let (Some(thresh), Some(emb)) =
                            (policy.semantic_threshold, semantic_embedder)
                        {
                            let semantic_text = strip_allowlist_spans(text, &allowlist_spans);
                            if !normalize_compact(&semantic_text).is_empty() {
                                let text_vec = emb.embed_128d(&semantic_text);
                                for alias in &aliases {
                                    let alias_vec = emb.embed_128d(alias);
                                    if cosine_similarity_128d(&text_vec, &alias_vec) >= thresh {
                                        semantic_hit = true;
                                        break;
                                    }
                                }
                            }
                        }
                    }
                }
                GuardRuleKind::Regex => {
                    let re = Regex::new(&rule.pattern_or_category).map_err(|e| {
                        OpenRagError::Validation(format!(
                            "invalid guard regex '{}': {e}",
                            rule.pattern_or_category
                        ))
                    })?;
                    for m in re.find_iter(text) {
                        if !covered_by_allowlist(m.start(), m.end(), &allowlist_spans) {
                            hit_spans.push((m.start(), m.end()));
                        }
                    }
                }
            }

            if hit_spans.is_empty() && !semantic_hit {
                continue;
            }

            matched_rules.push(GuardMatch {
                rule_index: rule_idx,
                rule_id: rule.id,
                kind: rule.kind,
                pattern_or_category: rule.pattern_or_category.clone(),
                action: rule.action,
            });

            if rule.action == GuardAction::Warn {
                warnings.push(format!(
                    "guard warning on {}: rule '{}' ({:?}) matched",
                    surface, rule.pattern_or_category, rule.kind
                ));
            } else if rule.action == GuardAction::Mask {
                let repl = rule.replacement_text.clone().unwrap_or_else(|| {
                    if rule.kind == GuardRuleKind::PiiCategory {
                        let tag = PiiCategory::parse_alias(&rule.pattern_or_category)
                            .map_or("PII", |c| c.token_tag());
                        format!("[REDACTED:{tag}]")
                    } else {
                        "[REDACTED]".to_string()
                    }
                });

                if hit_spans.is_empty() && semantic_hit {
                    replacements.push(SpanReplacement {
                        start_byte: 0,
                        end_byte: text.len(),
                        action: rule.action,
                        priority: rule.priority,
                        replacement: repl,
                    });
                } else {
                    for (s, e) in hit_spans {
                        replacements.push(SpanReplacement {
                            start_byte: s,
                            end_byte: e,
                            action: rule.action,
                            priority: rule.priority,
                            replacement: repl.clone(),
                        });
                    }
                }
            }
        }

        let decision_action = matched_rules
            .iter()
            .map(|m| m.action)
            .max()
            .unwrap_or(GuardAction::Allow);

        let sanitized_text = match decision_action {
            GuardAction::Block => None,
            GuardAction::Mask => Some(apply_non_overlapping_replacements(text, &mut replacements)?),
            GuardAction::Warn | GuardAction::Allow => Some(text.to_string()),
        };

        Ok(GuardDecision {
            action: decision_action,
            sanitized_text,
            warnings,
            matched_rules,
        })
    }

    pub fn enforce(
        text: &str,
        surface: GuardSurface,
        policy: &GuardPolicyRuntime,
        pii_engine: &PiiEngine,
        concept_lookup: Option<&dyn ConceptAliasLookup>,
        semantic_embedder: Option<&dyn SemanticEmbedder128>,
    ) -> Result<GuardDecision, OpenRagError> {
        let decision = Self::evaluate(
            text,
            surface,
            policy,
            pii_engine,
            concept_lookup,
            semantic_embedder,
        )?;
        if decision.action == GuardAction::Block {
            let matched_rule_ids: Vec<Uuid> = decision
                .matched_rules
                .iter()
                .filter_map(|m| m.rule_id)
                .collect();
            let mut cats: Vec<String> = Vec::new();
            let mut seen = HashSet::new();
            for m in &decision.matched_rules {
                if seen.insert(m.pattern_or_category.clone()) {
                    cats.push(m.pattern_or_category.clone());
                }
            }
            return Err(OpenRagError::GuardBlocked {
                surface,
                reason: format!(
                    "blocked by {} guard rule(s) on {}",
                    decision.matched_rules.len(),
                    surface
                ),
                matched_rule_ids,
                matched_categories: cats,
            });
        }
        Ok(decision)
    }
}

fn collect_guard_allowlist_spans(indexed: &IndexedText, terms: &[String]) -> Vec<(usize, usize)> {
    let mut spans = Vec::new();
    for term in terms {
        let trimmed = term.trim();
        if trimmed.is_empty() {
            continue;
        }
        for (s, m) in indexed.original.match_indices(trimmed) {
            spans.push((s, s + m.len()));
        }
        spans.extend(indexed.find_term_orig_spans(trimmed));
    }
    spans.sort_unstable();
    spans.dedup();
    spans
}

fn covered_by_allowlist(start: usize, end: usize, allowlist_spans: &[(usize, usize)]) -> bool {
    allowlist_spans
        .iter()
        .any(|&(as_, ae)| start >= as_ && end <= ae)
}

fn strip_allowlist_spans(text: &str, allowlist_spans: &[(usize, usize)]) -> String {
    if allowlist_spans.is_empty() {
        return text.to_string();
    }
    let mut merged: Vec<(usize, usize)> = Vec::new();
    for &(s, e) in allowlist_spans {
        let s = s.min(text.len());
        let e = e.min(text.len());
        if s >= e || !text.is_char_boundary(s) || !text.is_char_boundary(e) {
            continue;
        }
        if let Some(last) = merged.last_mut() {
            if s <= last.1 {
                last.1 = last.1.max(e);
                continue;
            }
        }
        merged.push((s, e));
    }
    let mut out = String::with_capacity(text.len());
    let mut cursor = 0usize;
    for (s, e) in merged {
        if cursor < s {
            out.push_str(&text[cursor..s]);
        }
        if !out.is_empty() && !out.ends_with(' ') {
            out.push(' ');
        }
        cursor = e;
    }
    if cursor < text.len() {
        out.push_str(&text[cursor..]);
    }
    out
}

#[allow(dead_code)]
fn is_text_covered_entirely_by_allowlist(text: &str, allowlist_spans: &[(usize, usize)]) -> bool {
    normalize_compact(&strip_allowlist_spans(text, allowlist_spans)).is_empty()
}

fn apply_non_overlapping_replacements(
    text: &str,
    replacements: &mut [SpanReplacement],
) -> Result<String, OpenRagError> {
    replacements.sort_by(|a, b| {
        a.start_byte
            .cmp(&b.start_byte)
            .then_with(|| (b.end_byte - b.start_byte).cmp(&(a.end_byte - a.start_byte)))
            .then_with(|| b.action.cmp(&a.action))
            .then_with(|| a.priority.cmp(&b.priority))
    });

    let mut out = String::with_capacity(text.len());
    let mut cursor = 0usize;
    for r in replacements {
        if r.start_byte < cursor || r.end_byte > text.len() {
            continue;
        }
        if !text.is_char_boundary(r.start_byte) || !text.is_char_boundary(r.end_byte) {
            return Err(OpenRagError::Internal(
                "guard replacement offsets do not align with UTF-8 char boundaries".into(),
            ));
        }
        out.push_str(&text[cursor..r.start_byte]);
        out.push_str(&r.replacement);
        cursor = r.end_byte;
    }
    out.push_str(&text[cursor..]);
    Ok(out)
}

fn parse_guard_action(s: &str) -> Result<GuardAction, OpenRagError> {
    match s {
        "allow" => Ok(GuardAction::Allow),
        "warn" => Ok(GuardAction::Warn),
        "mask" => Ok(GuardAction::Mask),
        "block" => Ok(GuardAction::Block),
        other => Err(OpenRagError::Database(format!(
            "invalid guard action '{other}'"
        ))),
    }
}

fn parse_guard_surface(s: &str) -> Result<GuardSurface, OpenRagError> {
    match s {
        "inbound_query" => Ok(GuardSurface::InboundQuery),
        "llm_egress" => Ok(GuardSurface::LlmEgress),
        "mcp_egress" => Ok(GuardSurface::McpEgress),
        "ingestion_gate" => Ok(GuardSurface::IngestionGate),
        other => Err(OpenRagError::Database(format!(
            "invalid guard surface '{other}'"
        ))),
    }
}

fn parse_guard_rule_kind(s: &str) -> Result<GuardRuleKind, OpenRagError> {
    match s {
        "pii_category" => Ok(GuardRuleKind::PiiCategory),
        "safe_word" => Ok(GuardRuleKind::SafeWord),
        "keyword" => Ok(GuardRuleKind::Keyword),
        "regex" => Ok(GuardRuleKind::Regex),
        other => Err(OpenRagError::Database(format!(
            "invalid guard rule_kind '{other}'"
        ))),
    }
}

fn guard_rule_kind_str(k: GuardRuleKind) -> &'static str {
    match k {
        GuardRuleKind::PiiCategory => "pii_category",
        GuardRuleKind::SafeWord => "safe_word",
        GuardRuleKind::Keyword => "keyword",
        GuardRuleKind::Regex => "regex",
    }
}

/// Repository for `pii.guard_policies`, `pii.guard_rules`, and `pii.guard_events`.
pub struct GuardRepo;

impl GuardRepo {
    pub async fn get_or_create_active_policy(
        worker_pool: &PgPool,
        org_id: Uuid,
        name: &str,
    ) -> Result<GuardPolicyRow, OpenRagError> {
        let existing = sqlx::query(
            "SELECT id, org_id, bucket_id, name, master_enabled, default_action,
                    allowlist_terms, surfaces, is_active, updated_at
             FROM pii.guard_policies
             WHERE org_id = $1 AND name = $2",
        )
        .bind(org_id)
        .bind(name)
        .fetch_optional(worker_pool)
        .await
        .map_err(|e| OpenRagError::Database(format!("failed querying pii.guard_policies: {e}")))?;

        if let Some(row) = existing {
            return row_to_policy(&row);
        }

        let row = sqlx::query(
            "INSERT INTO pii.guard_policies (org_id, name)
             VALUES ($1, $2)
             ON CONFLICT (org_id, name) DO UPDATE SET updated_at = now()
             RETURNING id, org_id, bucket_id, name, master_enabled, default_action,
                       allowlist_terms, surfaces, is_active, updated_at",
        )
        .bind(org_id)
        .bind(name)
        .fetch_one(worker_pool)
        .await
        .map_err(|e| OpenRagError::Database(format!("failed creating pii.guard_policies: {e}")))?;

        row_to_policy(&row)
    }

    pub async fn update_policy(
        worker_pool: &PgPool,
        org_id: Uuid,
        policy_id: Uuid,
        master_enabled: bool,
        default_action: GuardAction,
        allowlist_terms: &[String],
        surfaces: &[GuardSurface],
    ) -> Result<GuardPolicyRow, OpenRagError> {
        let surface_strs: Vec<String> = surfaces.iter().map(|s| s.as_str().to_string()).collect();
        let row = sqlx::query(
            "UPDATE pii.guard_policies
             SET master_enabled = $3,
                 default_action = $4,
                 allowlist_terms = $5,
                 surfaces = $6,
                 updated_at = now()
             WHERE org_id = $1 AND id = $2
             RETURNING id, org_id, bucket_id, name, master_enabled, default_action,
                       allowlist_terms, surfaces, is_active, updated_at",
        )
        .bind(org_id)
        .bind(policy_id)
        .bind(master_enabled)
        .bind(default_action.as_str())
        .bind(allowlist_terms)
        .bind(&surface_strs)
        .fetch_optional(worker_pool)
        .await
        .map_err(|e| OpenRagError::Database(format!("failed updating pii.guard_policies: {e}")))?
        .ok_or_else(|| OpenRagError::NotFound(format!("guard policy {policy_id} not found")))?;

        row_to_policy(&row)
    }

    pub async fn create_rule(
        worker_pool: &PgPool,
        org_id: Uuid,
        policy_id: Uuid,
        input: &GuardRuleConfig,
    ) -> Result<GuardRuleRow, OpenRagError> {
        if input.pattern_or_category.trim().is_empty() {
            return Err(OpenRagError::Validation(
                "guard rule pattern_or_category must not be empty".into(),
            ));
        }
        if input.surfaces.is_empty() {
            return Err(OpenRagError::Validation(
                "guard rule surfaces must not be empty".into(),
            ));
        }
        if input.kind == GuardRuleKind::Regex {
            Regex::new(&input.pattern_or_category).map_err(|e| {
                OpenRagError::Validation(format!(
                    "invalid guard regex '{}': {e}",
                    input.pattern_or_category
                ))
            })?;
        }

        let surface_strs: Vec<String> = input
            .surfaces
            .iter()
            .map(|s| s.as_str().to_string())
            .collect();

        let row = sqlx::query(
            "INSERT INTO pii.guard_rules (
                policy_id, org_id, rule_kind, pattern_or_category, action,
                surfaces, replacement_text, priority, enabled
             ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
             RETURNING id, policy_id, org_id, rule_kind, pattern_or_category, action,
                       surfaces, replacement_text, priority, enabled, created_at",
        )
        .bind(policy_id)
        .bind(org_id)
        .bind(guard_rule_kind_str(input.kind))
        .bind(&input.pattern_or_category)
        .bind(input.action.as_str())
        .bind(&surface_strs)
        .bind(&input.replacement_text)
        .bind(input.priority)
        .bind(input.enabled)
        .fetch_one(worker_pool)
        .await
        .map_err(|e| OpenRagError::Database(format!("failed inserting pii.guard_rules: {e}")))?;

        row_to_rule(&row)
    }

    pub async fn delete_rule(
        worker_pool: &PgPool,
        org_id: Uuid,
        rule_id: Uuid,
    ) -> Result<bool, OpenRagError> {
        let res = sqlx::query("DELETE FROM pii.guard_rules WHERE org_id = $1 AND id = $2")
            .bind(org_id)
            .bind(rule_id)
            .execute(worker_pool)
            .await
            .map_err(|e| OpenRagError::Database(format!("failed deleting pii.guard_rules: {e}")))?;
        Ok(res.rows_affected() > 0)
    }

    pub async fn list_rules(
        worker_pool: &PgPool,
        org_id: Uuid,
        policy_id: Uuid,
    ) -> Result<Vec<GuardRuleRow>, OpenRagError> {
        let rows = sqlx::query(
            "SELECT id, policy_id, org_id, rule_kind, pattern_or_category, action,
                    surfaces, replacement_text, priority, enabled, created_at
             FROM pii.guard_rules
             WHERE org_id = $1 AND policy_id = $2
             ORDER BY priority ASC, created_at ASC",
        )
        .bind(org_id)
        .bind(policy_id)
        .fetch_all(worker_pool)
        .await
        .map_err(|e| OpenRagError::Database(format!("failed listing pii.guard_rules: {e}")))?;

        rows.iter().map(row_to_rule).collect()
    }

    /// Builds sanitized metadata JSONB that NEVER includes raw input text or blocked secret/PII strings.
    pub fn build_safe_event_metadata(
        surface: GuardSurface,
        input_text: &str,
        decision: &GuardDecision,
    ) -> serde_json::Value {
        let digest = Sha256::digest(input_text.as_bytes());
        let sha_prefix = hex::encode(&digest[..8]);
        serde_json::json!({
            "surface": surface.as_str(),
            "decision": decision.action.as_str(),
            "match_count": decision.matched_rules.len(),
            "warnings_count": decision.warnings.len(),
            "text_len": input_text.len(),
            "text_sha256_prefix": sha_prefix,
        })
    }

    pub async fn log_guard_event_rls(
        tx: &mut RlsTransaction<'_>,
        org_id: Uuid,
        policy_id: Option<Uuid>,
        principal_id: Uuid,
        surface: GuardSurface,
        input_text: &str,
        decision: &GuardDecision,
    ) -> Result<Uuid, OpenRagError> {
        let rule_ids: Vec<Uuid> = decision
            .matched_rules
            .iter()
            .filter_map(|m| m.rule_id)
            .collect();
        let categories: Vec<String> = decision
            .matched_rules
            .iter()
            .map(|m| match m.kind {
                GuardRuleKind::PiiCategory => PiiCategory::parse_alias(&m.pattern_or_category)
                    .map_or_else(
                        || m.pattern_or_category.clone(),
                        |c| c.as_canonical_str().to_string(),
                    ),
                GuardRuleKind::SafeWord => "safe_word".to_string(),
                GuardRuleKind::Keyword => "keyword".to_string(),
                GuardRuleKind::Regex => "regex".to_string(),
            })
            .collect();
        let metadata = Self::build_safe_event_metadata(surface, input_text, decision);

        let id: Uuid = sqlx::query_scalar(
            "INSERT INTO pii.guard_events (
                org_id, policy_id, principal_id, surface, decision,
                matched_rule_ids, matched_categories, metadata
             ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
             RETURNING id",
        )
        .bind(org_id)
        .bind(policy_id)
        .bind(principal_id)
        .bind(surface.as_str())
        .bind(decision.action.as_str())
        .bind(&rule_ids)
        .bind(&categories)
        .bind(metadata)
        .fetch_one(&mut **tx.tx_mut())
        .await
        .map_err(|e| OpenRagError::Database(format!("failed inserting pii.guard_events: {e}")))?;

        Ok(id)
    }
}

fn row_to_policy(row: &sqlx::postgres::PgRow) -> Result<GuardPolicyRow, OpenRagError> {
    let def_str: String = row
        .try_get("default_action")
        .map_err(|e| OpenRagError::Database(e.to_string()))?;
    let surf_strs: Vec<String> = row
        .try_get("surfaces")
        .map_err(|e| OpenRagError::Database(e.to_string()))?;
    let surfaces = surf_strs
        .iter()
        .map(|s| parse_guard_surface(s))
        .collect::<Result<Vec<_>, _>>()?;

    Ok(GuardPolicyRow {
        id: row.try_get("id").map_err(|e| OpenRagError::Database(e.to_string()))?,
        org_id: row.try_get("org_id").map_err(|e| OpenRagError::Database(e.to_string()))?,
        bucket_id: row.try_get("bucket_id").map_err(|e| OpenRagError::Database(e.to_string()))?,
        name: row.try_get("name").map_err(|e| OpenRagError::Database(e.to_string()))?,
        master_enabled: row
            .try_get("master_enabled")
            .map_err(|e| OpenRagError::Database(e.to_string()))?,
        default_action: parse_guard_action(&def_str)?,
        allowlist_terms: row
            .try_get("allowlist_terms")
            .map_err(|e| OpenRagError::Database(e.to_string()))?,
        surfaces,
        is_active: row.try_get("is_active").map_err(|e| OpenRagError::Database(e.to_string()))?,
        updated_at: row
            .try_get("updated_at")
            .map_err(|e| OpenRagError::Database(e.to_string()))?,
    })
}

fn row_to_rule(row: &sqlx::postgres::PgRow) -> Result<GuardRuleRow, OpenRagError> {
    let kind_str: String = row
        .try_get("rule_kind")
        .map_err(|e| OpenRagError::Database(e.to_string()))?;
    let act_str: String = row
        .try_get("action")
        .map_err(|e| OpenRagError::Database(e.to_string()))?;
    let surf_strs: Vec<String> = row
        .try_get("surfaces")
        .map_err(|e| OpenRagError::Database(e.to_string()))?;
    let surfaces = surf_strs
        .iter()
        .map(|s| parse_guard_surface(s))
        .collect::<Result<Vec<_>, _>>()?;

    Ok(GuardRuleRow {
        id: row.try_get("id").map_err(|e| OpenRagError::Database(e.to_string()))?,
        policy_id: row
            .try_get("policy_id")
            .map_err(|e| OpenRagError::Database(e.to_string()))?,
        org_id: row.try_get("org_id").map_err(|e| OpenRagError::Database(e.to_string()))?,
        kind: parse_guard_rule_kind(&kind_str)?,
        pattern_or_category: row
            .try_get("pattern_or_category")
            .map_err(|e| OpenRagError::Database(e.to_string()))?,
        action: parse_guard_action(&act_str)?,
        surfaces,
        replacement_text: row
            .try_get("replacement_text")
            .map_err(|e| OpenRagError::Database(e.to_string()))?,
        priority: row.try_get("priority").map_err(|e| OpenRagError::Database(e.to_string()))?,
        enabled: row.try_get("enabled").map_err(|e| OpenRagError::Database(e.to_string()))?,
        created_at: row
            .try_get("created_at")
            .map_err(|e| OpenRagError::Database(e.to_string()))?,
    })
}
