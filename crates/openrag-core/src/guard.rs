use std::fmt;

use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Strictness order: `Block > Mask > Warn > Allow`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GuardAction {
    Allow,
    Warn,
    Mask,
    Block,
}

impl GuardAction {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Allow => "allow",
            Self::Warn => "warn",
            Self::Mask => "mask",
            Self::Block => "block",
        }
    }
}

impl fmt::Display for GuardAction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GuardSurface {
    InboundQuery,
    LlmEgress,
    McpEgress,
    IngestionGate,
}

impl GuardSurface {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::InboundQuery => "inbound_query",
            Self::LlmEgress => "llm_egress",
            Self::McpEgress => "mcp_egress",
            Self::IngestionGate => "ingestion_gate",
        }
    }
}

impl fmt::Display for GuardSurface {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GuardRuleKind {
    PiiCategory,
    SafeWord,
    Keyword,
    Regex,
}

const fn default_true() -> bool {
    true
}

const fn default_rule_priority() -> i32 {
    100
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GuardRuleConfig {
    #[serde(alias = "rule_kind")]
    pub kind: GuardRuleKind,
    pub pattern_or_category: String,
    pub action: GuardAction,
    pub surfaces: Vec<GuardSurface>,
    #[serde(default)]
    pub replacement_text: Option<String>,
    #[serde(default = "default_true")]
    pub enabled: bool,
    #[serde(default = "default_rule_priority")]
    pub priority: i32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GuardConfig {
    pub enabled: bool,
    pub default_action: GuardAction,
    #[serde(default)]
    pub allowlist: Vec<String>,
    #[serde(default)]
    pub rules: Vec<GuardRuleConfig>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GuardMatch {
    pub rule_index: usize,
    #[serde(default)]
    pub rule_id: Option<Uuid>,
    #[serde(alias = "rule_kind")]
    pub kind: GuardRuleKind,
    pub pattern_or_category: String,
    pub action: GuardAction,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GuardDecision {
    pub action: GuardAction,
    pub sanitized_text: Option<String>,
    pub warnings: Vec<String>,
    pub matched_rules: Vec<GuardMatch>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn guard_rule_and_match_accept_kind_and_rule_kind_aliases() {
        let with_kind = r#"{
            "kind": "safe_word",
            "pattern_or_category": "PROJECT-ORION",
            "action": "block",
            "surfaces": ["inbound_query"]
        }"#;
        let with_rule_kind = r#"{
            "rule_kind": "safe_word",
            "pattern_or_category": "PROJECT-ORION",
            "action": "block",
            "surfaces": ["inbound_query"]
        }"#;
        let parsed_kind: Result<GuardRuleConfig, _> = serde_json::from_str(with_kind);
        let parsed_rule_kind: Result<GuardRuleConfig, _> = serde_json::from_str(with_rule_kind);
        assert!(parsed_kind.is_ok());
        assert!(parsed_rule_kind.is_ok());
        assert_eq!(parsed_kind.ok(), parsed_rule_kind.ok());

        // deny_unknown_fields must remain enforced
        let with_unknown = r#"{
            "rule_kind": "safe_word",
            "pattern_or_category": "PROJECT-ORION",
            "action": "block",
            "surfaces": ["inbound_query"],
            "unknown_extra": 1
        }"#;
        let parsed_unknown: Result<GuardRuleConfig, _> = serde_json::from_str(with_unknown);
        assert!(parsed_unknown.is_err());

        let match_with_rule_kind = r#"{
            "rule_index": 0,
            "rule_id": null,
            "rule_kind": "pii_category",
            "pattern_or_category": "RRN",
            "action": "mask"
        }"#;
        let parsed_match: Result<GuardMatch, _> = serde_json::from_str(match_with_rule_kind);
        assert!(parsed_match.is_ok());
    }
}
