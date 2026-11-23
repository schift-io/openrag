use std::collections::BTreeMap;
use std::fmt;
use std::str::FromStr;

use chrono::{DateTime, Utc};
use openrag_core::{
    FusionMode, OpenRagError, OptimizationStrategy, PipelineProfile, SecurityGateConfig,
};
use openrag_retrieve::{RetrieveOverrides, MAX_TOP_K};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GoldenAxis {
    Retrieval,
    Ocr,
    Pii,
    Pararel,
    Multi,
}

impl GoldenAxis {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Retrieval => "retrieval",
            Self::Ocr => "ocr",
            Self::Pii => "pii",
            Self::Pararel => "pararel",
            Self::Multi => "multi",
        }
    }

    pub const fn is_item_axis(self) -> bool {
        !matches!(self, Self::Multi)
    }
}

impl fmt::Display for GoldenAxis {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for GoldenAxis {
    type Err = OpenRagError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "retrieval" => Ok(Self::Retrieval),
            "ocr" => Ok(Self::Ocr),
            "pii" => Ok(Self::Pii),
            "pararel" => Ok(Self::Pararel),
            "multi" => Ok(Self::Multi),
            other => Err(OpenRagError::Validation(format!(
                "invalid golden axis '{other}'; expected retrieval, ocr, pii, pararel, or multi"
            ))),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReviewStatus {
    Pending,
    #[default]
    Approved,
    Rejected,
}

impl ReviewStatus {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Pending => "pending",
            Self::Approved => "approved",
            Self::Rejected => "rejected",
        }
    }
}

impl fmt::Display for ReviewStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for ReviewStatus {
    type Err = OpenRagError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "pending" => Ok(Self::Pending),
            "approved" => Ok(Self::Approved),
            "rejected" => Ok(Self::Rejected),
            other => Err(OpenRagError::Validation(format!(
                "invalid review_status '{other}'; expected pending, approved, or rejected"
            ))),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FeedbackVerdict {
    Up,
    Down,
    WrongEvidence,
    MissingCitation,
}

impl FeedbackVerdict {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Up => "up",
            Self::Down => "down",
            Self::WrongEvidence => "wrong_evidence",
            Self::MissingCitation => "missing_citation",
        }
    }
}

impl fmt::Display for FeedbackVerdict {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for FeedbackVerdict {
    type Err = OpenRagError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "up" => Ok(Self::Up),
            "down" => Ok(Self::Down),
            "wrong_evidence" => Ok(Self::WrongEvidence),
            "missing_citation" => Ok(Self::MissingCitation),
            other => Err(OpenRagError::Validation(format!(
                "invalid feedback verdict '{other}'; expected up, down, wrong_evidence, or missing_citation"
            ))),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BenchTrack {
    #[serde(rename = "track1_retrieval")]
    Track1Retrieval,
    #[serde(rename = "track2_e2e_rag")]
    Track2E2eRag,
    #[serde(rename = "track3_security_ocr")]
    Track3SecurityOcr,
}

impl BenchTrack {
    pub const ALL: [Self; 3] = [
        Self::Track1Retrieval,
        Self::Track2E2eRag,
        Self::Track3SecurityOcr,
    ];

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Track1Retrieval => "track1_retrieval",
            Self::Track2E2eRag => "track2_e2e_rag",
            Self::Track3SecurityOcr => "track3_security_ocr",
        }
    }
}

impl fmt::Display for BenchTrack {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for BenchTrack {
    type Err = OpenRagError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "track1_retrieval" => Ok(Self::Track1Retrieval),
            "track2_e2e_rag" => Ok(Self::Track2E2eRag),
            "track3_security_ocr" => Ok(Self::Track3SecurityOcr),
            other => Err(OpenRagError::Validation(format!(
                "invalid bench track '{other}'"
            ))),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OptimizationStatus {
    Running,
    Completed,
    BlockedSecurityGate,
    Failed,
}

impl OptimizationStatus {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Running => "running",
            Self::Completed => "completed",
            Self::BlockedSecurityGate => "blocked_security_gate",
            Self::Failed => "failed",
        }
    }
}

impl fmt::Display for OptimizationStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for OptimizationStatus {
    type Err = OpenRagError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "running" => Ok(Self::Running),
            "completed" => Ok(Self::Completed),
            "blocked_security_gate" => Ok(Self::BlockedSecurityGate),
            "failed" => Ok(Self::Failed),
            other => Err(OpenRagError::Validation(format!(
                "invalid optimization status '{other}'"
            ))),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PipelineProfileKind {
    Standard,
    Airgap,
    Custom,
}

impl PipelineProfileKind {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Standard => "standard",
            Self::Airgap => "airgap",
            Self::Custom => "custom",
        }
    }
}

impl From<PipelineProfile> for PipelineProfileKind {
    fn from(value: PipelineProfile) -> Self {
        match value {
            PipelineProfile::Standard => Self::Standard,
            PipelineProfile::Airgap => Self::Airgap,
        }
    }
}

impl fmt::Display for PipelineProfileKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for PipelineProfileKind {
    type Err = OpenRagError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "standard" => Ok(Self::Standard),
            "airgap" => Ok(Self::Airgap),
            "custom" => Ok(Self::Custom),
            other => Err(OpenRagError::Validation(format!(
                "invalid pipeline profile '{other}'; expected standard, airgap, or custom"
            ))),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GoldenSet {
    pub id: Uuid,
    pub org_id: Uuid,
    pub name: String,
    pub axis: GoldenAxis,
    pub description: Option<String>,
    pub version: i32,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CreateGoldenSetInput {
    pub name: String,
    pub axis: GoldenAxis,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub version: Option<i32>,
}

impl CreateGoldenSetInput {
    pub fn validate(&self) -> Result<(), OpenRagError> {
        if self.name.trim().is_empty() {
            return Err(OpenRagError::Validation(
                "golden set name must not be empty".into(),
            ));
        }
        if let Some(v) = self.version {
            if v <= 0 {
                return Err(OpenRagError::Validation(format!(
                    "golden set version must be >= 1, got {v}"
                )));
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GoldenItem {
    pub id: Uuid,
    pub org_id: Uuid,
    pub golden_set_id: Uuid,
    pub axis: GoldenAxis,
    pub query_text: String,
    pub test_principal_id: Option<Uuid>,
    #[serde(default)]
    pub expected_doc_ids: Vec<Uuid>,
    #[serde(default)]
    pub forbidden_doc_ids: Vec<Uuid>,
    pub expected_answer: Option<String>,
    #[serde(default)]
    pub expected_pii_tokens: Vec<String>,
    pub expected_edges: Option<serde_json::Value>,
    pub review_status: ReviewStatus,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CreateGoldenItemInput {
    pub axis: GoldenAxis,
    pub query_text: String,
    #[serde(default)]
    pub test_principal_id: Option<Uuid>,
    #[serde(default)]
    pub expected_doc_ids: Vec<Uuid>,
    #[serde(default)]
    pub forbidden_doc_ids: Vec<Uuid>,
    #[serde(default)]
    pub expected_answer: Option<String>,
    #[serde(default)]
    pub expected_pii_tokens: Vec<String>,
    #[serde(default)]
    pub expected_edges: Option<serde_json::Value>,
    #[serde(default)]
    pub review_status: Option<ReviewStatus>,
}

fn has_non_empty_edges(edges: Option<&serde_json::Value>) -> bool {
    let Some(val) = edges else {
        return false;
    };
    match val {
        serde_json::Value::Null => false,
        serde_json::Value::Array(arr) => !arr.is_empty(),
        serde_json::Value::Object(obj) => !obj.is_empty(),
        serde_json::Value::String(s) => !s.trim().is_empty(),
        _ => true,
    }
}

impl CreateGoldenItemInput {
    pub fn validate(&self) -> Result<(), OpenRagError> {
        if !self.axis.is_item_axis() {
            return Err(OpenRagError::Validation(
                "golden item axis cannot be 'multi'; expected retrieval, ocr, pii, or pararel"
                    .into(),
            ));
        }
        if self.query_text.trim().is_empty() {
            return Err(OpenRagError::Validation(
                "golden item query_text must not be empty".into(),
            ));
        }
        let has_answer = self
            .expected_answer
            .as_deref()
            .is_some_and(|s| !s.trim().is_empty());
        let has_pii = self
            .expected_pii_tokens
            .iter()
            .any(|t| !t.trim().is_empty());
        let has_edges = has_non_empty_edges(self.expected_edges.as_ref());

        if self.expected_doc_ids.is_empty()
            && self.forbidden_doc_ids.is_empty()
            && !has_answer
            && !has_pii
            && !has_edges
        {
            return Err(OpenRagError::Validation(
                "golden item must define at least one expected target (expected_doc_ids, forbidden_doc_ids, expected_answer, expected_pii_tokens, or expected_edges)".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SearchFeedback {
    pub id: Uuid,
    pub org_id: Uuid,
    pub principal_id: Uuid,
    pub query_text: String,
    pub response_text: Option<String>,
    #[serde(default)]
    pub cited_chunk_ids: Vec<Uuid>,
    pub verdict: FeedbackVerdict,
    pub comment: Option<String>,
    pub promoted_golden_item_id: Option<Uuid>,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SubmitFeedbackInput {
    pub query_text: String,
    #[serde(default)]
    pub response_text: Option<String>,
    #[serde(default)]
    pub cited_chunk_ids: Vec<Uuid>,
    #[serde(default)]
    pub expected_doc_ids: Vec<Uuid>,
    pub verdict: FeedbackVerdict,
    #[serde(default)]
    pub comment: Option<String>,
}

impl SubmitFeedbackInput {
    pub fn validate(&self) -> Result<(), OpenRagError> {
        if self.query_text.trim().is_empty() {
            return Err(OpenRagError::Validation(
                "feedback query_text must not be empty".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PromoteFeedbackInput {
    pub golden_set_id: Uuid,
    #[serde(default)]
    pub axis: Option<GoldenAxis>,
    #[serde(default)]
    pub test_principal_id: Option<Uuid>,
    #[serde(default)]
    pub expected_doc_ids: Vec<Uuid>,
    #[serde(default)]
    pub forbidden_doc_ids: Vec<Uuid>,
    #[serde(default)]
    pub expected_answer: Option<String>,
    #[serde(default)]
    pub expected_pii_tokens: Vec<String>,
    #[serde(default)]
    pub expected_edges: Option<serde_json::Value>,
    #[serde(default)]
    pub review_status: Option<ReviewStatus>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PromoteFeedbackResult {
    pub feedback: SearchFeedback,
    pub item: GoldenItem,
    pub already_promoted: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct SecurityGateResult {
    pub pii_leak_count: i32,
    pub arn_violation_count: i32,
    pub pass: bool,
}

impl SecurityGateResult {
    pub const fn new(pii_leak_count: i32, arn_violation_count: i32) -> Self {
        Self {
            pii_leak_count,
            arn_violation_count,
            pass: pii_leak_count == 0 && arn_violation_count == 0,
        }
    }

    pub const fn passes_policy(&self, policy: &SecurityGateConfig) -> bool {
        let pii_ok = !policy.require_zero_pii_leak || self.pii_leak_count == 0;
        let arn_ok = !policy.require_zero_arn_violation || self.arn_violation_count == 0;
        pii_ok && arn_ok
    }
}

#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct BenchMetrics {
    pub mrr: f64,
    pub recall: f64,
    pub recall_at_10: f64,
    pub ndcg: f64,
    pub ndcg_at_10: f64,
    pub avg_latency_ms: f64,
    pub item_count: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BenchRunRecord {
    pub id: Uuid,
    pub org_id: Uuid,
    pub golden_set_id: Uuid,
    pub pipeline_config_id: Option<Uuid>,
    pub trial_id: Option<Uuid>,
    pub track: BenchTrack,
    pub metrics: serde_json::Value,
    pub pii_leak_count: i32,
    pub arn_violation_count: i32,
    pub security_gate_pass: bool,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BenchItemResultRecord {
    pub id: Uuid,
    pub bench_run_id: Uuid,
    pub golden_item_id: Uuid,
    #[serde(default)]
    pub retrieved_doc_ids: Vec<Uuid>,
    pub latency_ms: i32,
    pub score: f32,
    pub pii_leaked: bool,
    pub arn_violated: bool,
    pub details: serde_json::Value,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BenchRunRequest {
    #[serde(default)]
    pub golden_set_id: Option<Uuid>,
    #[serde(default)]
    pub golden_set: Option<String>,
    #[serde(default)]
    pub bucket_id: Option<Uuid>,
    #[serde(default)]
    pub pipeline_config_id: Option<Uuid>,
    #[serde(default)]
    pub trial_id: Option<Uuid>,
    #[serde(default)]
    pub pipeline_yaml: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BenchRunSummary {
    pub golden_set: GoldenSet,
    pub runs: Vec<BenchRunRecord>,
    pub item_results: Vec<BenchItemResultRecord>,
    pub security_gates: SecurityGateResult,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LeaderboardEntry {
    pub bench_run_id: Uuid,
    pub golden_set_id: Uuid,
    pub golden_set_name: String,
    pub pipeline_config_id: Option<Uuid>,
    pub track: BenchTrack,
    pub mrr: f64,
    pub recall_at_10: f64,
    pub ndcg_at_10: f64,
    pub avg_latency_ms: f64,
    pub pii_leak_count: i32,
    pub arn_violation_count: i32,
    pub security_gate_pass: bool,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BenchReport {
    pub org_id: Uuid,
    pub golden_set: Option<GoldenSet>,
    pub entries: Vec<LeaderboardEntry>,
    pub overall_security_gate_pass: bool,
    pub markdown: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AbQueryItem {
    #[serde(alias = "query_text")]
    pub query: String,
    #[serde(default, alias = "expected_doc_ids")]
    pub relevant_ids: Vec<Uuid>,
    #[serde(default, alias = "forbidden_doc_ids")]
    pub forbidden_ids: Vec<Uuid>,
    #[serde(default)]
    pub expected_pii_tokens: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum AbQueryInput {
    Inline(AbQueryItem),
    QueryText(String),
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct AbVariantConfig {
    #[serde(default)]
    pub fusion: Option<FusionMode>,
    #[serde(default)]
    pub alpha: Option<f32>,
    #[serde(default)]
    pub rerank: Option<bool>,
    #[serde(default, alias = "k")]
    pub top_k: Option<usize>,
    #[serde(default)]
    pub spaces: Option<Vec<String>>,
    #[serde(default)]
    pub latest_revision_only: Option<bool>,
    #[serde(default)]
    pub latest_revision_boost: Option<f32>,
    #[serde(default)]
    pub rrf_k: Option<u32>,
}

impl AbVariantConfig {
    pub fn to_retrieve_overrides(&self, default_k: usize) -> RetrieveOverrides {
        RetrieveOverrides {
            spaces: self.spaces.clone(),
            fusion: self.fusion,
            rrf_k: self.rrf_k,
            alpha: self.alpha,
            top_k: self.top_k.or(Some(default_k)),
            rerank: self.rerank,
            latest_revision_boost: self.latest_revision_boost,
            admission_threshold: None,
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct AbSearchRequest {
    #[serde(default, alias = "bucket")]
    pub bucket_id: Option<Uuid>,
    #[serde(default)]
    pub golden_set_id: Option<Uuid>,
    #[serde(default, alias = "top_k")]
    pub k: Option<usize>,
    #[serde(default)]
    pub queries: Vec<AbQueryInput>,
    #[serde(default)]
    pub variants: BTreeMap<String, AbVariantConfig>,
}

impl AbSearchRequest {
    pub fn validate(&self) -> Result<usize, OpenRagError> {
        let k = self.k.unwrap_or(10);
        if k == 0 || k > MAX_TOP_K {
            return Err(OpenRagError::Validation(format!(
                "ab_search k must be within 1..={MAX_TOP_K}, got {k}"
            )));
        }
        if self.queries.is_empty() {
            return Err(OpenRagError::Validation(
                "ab_search queries must not be empty".into(),
            ));
        }
        if self.variants.is_empty() {
            return Err(OpenRagError::Validation(
                "ab_search variants must not be empty".into(),
            ));
        }
        for key in self.variants.keys() {
            let trimmed = key.trim();
            if trimmed.is_empty() {
                return Err(OpenRagError::Validation(
                    "ab_search variant name must not be empty".into(),
                ));
            }
            if matches!(trimmed, "security_gates" | "winner" | "query_count") {
                return Err(OpenRagError::Validation(format!(
                    "ab_search variant name '{key}' is reserved"
                )));
            }
        }
        Ok(k)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AbVariantMetrics {
    pub mrr: f64,
    pub recall: f64,
    pub recall_at_k: f64,
    pub ndcg: f64,
    pub ndcg_at_k: f64,
    pub avg_latency_ms: f64,
    pub query_count: usize,
    pub security_gates: SecurityGateResult,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AbSearchReport {
    #[serde(flatten)]
    pub variants: BTreeMap<String, AbVariantMetrics>,
    pub security_gates: SecurityGateResult,
    pub winner: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OptimizationTrialRecord {
    pub id: Uuid,
    pub org_id: Uuid,
    pub golden_set_id: Uuid,
    pub search_space_yaml: String,
    pub strategy: OptimizationStrategy,
    pub best_config_yaml: Option<String>,
    pub best_score: Option<f32>,
    pub security_gate_pass: bool,
    pub status: OptimizationStatus,
    pub created_at: DateTime<Utc>,
    pub completed_at: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct OptimizeRequest {
    #[serde(default)]
    pub golden_set_id: Option<Uuid>,
    #[serde(default)]
    pub golden_set: Option<String>,
    #[serde(default)]
    pub bucket_id: Option<Uuid>,
    #[serde(default)]
    pub strategy: Option<OptimizationStrategy>,
    pub search_space_yaml: String,
    #[serde(default)]
    pub base_pipeline_yaml: Option<String>,
    #[serde(default)]
    pub require_zero_pii_leak: Option<bool>,
    #[serde(default)]
    pub require_zero_arn_violation: Option<bool>,
    #[serde(default)]
    pub export_best_path: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CandidateEvaluation {
    pub description: String,
    pub score: f64,
    pub mrr: f64,
    pub recall_at_10: f64,
    pub ndcg_at_10: f64,
    pub security_gates: SecurityGateResult,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OptimizeReport {
    pub trial: OptimizationTrialRecord,
    pub best_config_yaml: Option<String>,
    pub best_score: Option<f32>,
    pub security_gates: SecurityGateResult,
    pub evaluated_candidates: usize,
    pub history: Vec<CandidateEvaluation>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PipelineConfigRecord {
    pub id: Uuid,
    pub org_id: Uuid,
    pub name: String,
    pub profile: PipelineProfileKind,
    pub pipeline_yaml: String,
    pub config_json: serde_json::Value,
    pub is_promoted: bool,
    pub promoted_at: Option<DateTime<Utc>>,
    pub created_by: Option<Uuid>,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegisterPipelineInput {
    pub name: String,
    pub profile: PipelineProfileKind,
    pub pipeline_yaml: String,
    pub config_json: serde_json::Value,
    #[serde(default)]
    pub created_by: Option<Uuid>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PromotePipelineInput {
    #[serde(default)]
    pub pipeline_config_id: Option<Uuid>,
    #[serde(default)]
    pub bench_run_id: Option<Uuid>,
    #[serde(default)]
    pub bucket_ref: Option<String>,
}
