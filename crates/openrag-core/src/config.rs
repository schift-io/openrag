use std::collections::BTreeMap;
use std::path::Path;

use regex::Regex;
use serde::{Deserialize, Serialize};

use crate::access::{AuthLevel, Sensitivity};
use crate::error::OpenRagError;
use crate::guard::{GuardConfig, GuardRuleKind};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PipelineProfile {
    Standard,
    Airgap,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OcrModeRule {
    pub min_auth_level: AuthLevel,
    pub required_mode: Sensitivity,
}

fn default_pdf_engine() -> String {
    "kordoc".to_string()
}

const fn default_kordoc_ocr() -> bool {
    true
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ParseConfig {
    pub hwp_engine: String,
    #[serde(default = "default_pdf_engine")]
    pub pdf_engine: String,
    #[serde(default = "default_kordoc_ocr")]
    pub kordoc_ocr: bool,
    pub ocr_engine: String,
    pub dewarp_enabled: bool,
    pub pdf_native_enabled: bool,
    pub default_ocr_mode: Sensitivity,
    #[serde(default)]
    pub ocr_mode_by_auth_level: Vec<OcrModeRule>,
    pub reject_zdr_with_batch: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
/// Page thumbnails and figure crops, kept only for display; they are not embedded.
pub struct ThumbnailConfig {
    /// Render and store them at ingest. Off by default: nothing in the first release shows
    /// them, and they are the only thing written to the object store.
    #[serde(default)]
    pub enabled: bool,
    pub format: String,
    pub max_edge_px: u32,
    pub figure_max_edge_px: u32,
    pub figure_max_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PiiConfig {
    pub schift_v7: bool,
    pub marker_ko_pii: bool,
    pub schift_v6: bool,
    #[serde(default)]
    pub custom_dict_terms: Vec<String>,
    #[serde(default)]
    pub allowlist_terms: Vec<String>,
    pub capsule_enabled: bool,
    pub guard: GuardConfig,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryBackendKind {
    #[default]
    OpenragPg,
    Kumiho,
}

impl MemoryBackendKind {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::OpenragPg => "openrag_pg",
            Self::Kumiho => "kumiho",
        }
    }
}

const fn default_memory_offline_fallback() -> bool {
    true
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MemoryConfig {
    #[serde(default)]
    pub backend: MemoryBackendKind,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mirror_to: Option<MemoryBackendKind>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kumiho_url: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kumiho_api_key_env: Option<String>,
    #[serde(default = "default_memory_offline_fallback")]
    pub offline_fallback: bool,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            backend: MemoryBackendKind::OpenragPg,
            mirror_to: None,
            kumiho_url: None,
            kumiho_api_key_env: None,
            offline_fallback: true,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PgPineconeEngine {
    Native,
    Pgvector,
    Remote,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExternalMirrorBackend {
    None,
    S3Vectors,
    Qdrant,
    Faiss,
    Usearch,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ChunkerKind {
    Fixed,
    #[default]
    Structure,
    Semantic,
}

const fn default_embed_document_title() -> bool {
    false
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StoreConfig {
    pub pg_pinecone_engine: PgPineconeEngine,
    pub dim: u16,
    pub bm25_engine: String,
    pub bm25_fallback: String,
    pub external_mirror: ExternalMirrorBackend,
    pub outbox_batch_size: u32,
    pub outbox_max_attempts: u32,
    #[serde(default)]
    pub pg_pinecone_search: PgPineconeSearchConfig,
    /// Include the document title when embedding corpus chunks (default off).
    #[serde(default = "default_embed_document_title")]
    pub embed_document_title: bool,
    /// Page chunking strategy (`fixed`, `structure`, or `semantic`).
    #[serde(default)]
    pub chunker: ChunkerKind,
}

/// pg_pinecone (native engine) search settings, set on each search transaction with `SET LOCAL`.
/// 0 leaves pg_pinecone's own value (the probe count its index build calibrated).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct PgPineconeSearchConfig {
    /// `pinecone.scan_fraction`: share of IVF lists probed per query, 0.0..=1.0.
    pub scan_fraction: f32,
    /// `pinecone.rescore_multiplier`: candidates re-ranked with exact distance, in hundreds, 0..=1000.
    pub rescore_multiplier: u32,
}

impl Default for PgPineconeSearchConfig {
    /// Measured 2026-11-24 on 100k rows (Ko-StrategyQA + MIRACL ko, EG2@256) through the HTTP
    /// search path: nDCG@10 0.7109 → 0.7147 (exact scan 0.7159) at the same p50 latency.
    fn default() -> Self {
        Self { scan_fraction: 0.5, rescore_multiplier: 4 }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FusionMode {
    Rrf,
    Convex,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RetrieveConfig {
    pub spaces: Vec<String>,
    pub fusion: FusionMode,
    pub rrf_k: u32,
    pub alpha: f32,
    pub top_k: usize,
    pub rerank: bool,
    pub latest_revision_boost: f32,
    pub admission_threshold: f32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LlmConfig {
    pub external_provider: String,
    pub external_model: String,
    pub local_provider: String,
    pub local_model: String,
    pub local_only_min_auth_level: AuthLevel,
    pub require_pii_masked_for_external: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PipelineConfig {
    pub version: u32,
    pub profile: PipelineProfile,
    pub parse: ParseConfig,
    pub thumbnail: ThumbnailConfig,
    pub pii: PiiConfig,
    #[serde(default)]
    pub memory: MemoryConfig,
    pub store: StoreConfig,
    pub retrieve: RetrieveConfig,
    pub llm: LlmConfig,
    /// Air-gapped profile only: extra hosts the server may call (see [`crate::egress`]).
    #[serde(default)]
    pub egress: crate::egress::EgressConfig,
}

impl PipelineConfig {
    pub fn from_yaml_str(yaml: &str) -> Result<Self, OpenRagError> {
        let cfg: Self = serde_yaml_ng::from_str(yaml)
            .map_err(|e| OpenRagError::Config(format!("failed to parse pipeline YAML: {e}")))?;
        cfg.validate()?;
        Ok(cfg)
    }

    pub fn from_yaml_file(path: &Path) -> Result<Self, OpenRagError> {
        let raw = std::fs::read_to_string(path).map_err(|e| {
            OpenRagError::Config(format!(
                "failed to read pipeline config '{}': {e}",
                path.display()
            ))
        })?;
        Self::from_yaml_str(&raw)
    }

    pub fn validate(&self) -> Result<(), OpenRagError> {
        if self.version == 0 {
            return Err(OpenRagError::Config("pipeline version must be >= 1".into()));
        }
        self.egress.targets()?;
        if self.store.dim != 256 {
            return Err(OpenRagError::Config(format!(
                "store.dim must be 256, got {}",
                self.store.dim
            )));
        }
        let search = self.store.pg_pinecone_search;
        if !search.scan_fraction.is_finite() || !(0.0..=1.0).contains(&search.scan_fraction) {
            return Err(OpenRagError::Config(format!(
                "store.pg_pinecone_search.scan_fraction must be within 0.0..=1.0, got {}",
                search.scan_fraction
            )));
        }
        if search.rescore_multiplier > 1000 {
            return Err(OpenRagError::Config(format!(
                "store.pg_pinecone_search.rescore_multiplier must be within 0..=1000, got {}",
                search.rescore_multiplier
            )));
        }
        if self.thumbnail.max_edge_px == 0
            || self.thumbnail.figure_max_edge_px == 0
            || self.thumbnail.figure_max_bytes == 0
        {
            return Err(OpenRagError::Config(
                "thumbnail.max_edge_px, figure_max_edge_px, and figure_max_bytes must be >= 1"
                    .into(),
            ));
        }
        if let Some(url) = &self.memory.kumiho_url {
            if url.trim().is_empty() {
                return Err(OpenRagError::Config(
                    "memory.kumiho_url must not be empty when specified".into(),
                ));
            }
        }
        if let Some(env_name) = &self.memory.kumiho_api_key_env {
            if env_name.trim().is_empty() {
                return Err(OpenRagError::Config(
                    "memory.kumiho_api_key_env must not be empty when specified".into(),
                ));
            }
        }
        if self.memory.backend == MemoryBackendKind::OpenragPg
            && self.memory.mirror_to == Some(MemoryBackendKind::OpenragPg)
        {
            return Err(OpenRagError::Config(
                "memory.mirror_to cannot be 'openrag_pg' when memory.backend is already 'openrag_pg'"
                    .into(),
            ));
        }
        if self.store.outbox_batch_size == 0 || self.store.outbox_max_attempts == 0 {
            return Err(OpenRagError::Config(
                "outbox_batch_size and outbox_max_attempts must be >= 1".into(),
            ));
        }
        if self.retrieve.spaces.is_empty() || self.retrieve.top_k == 0 || self.retrieve.rrf_k == 0 {
            return Err(OpenRagError::Config(
                "retrieve.spaces, top_k, and rrf_k must be non-empty / > 0".into(),
            ));
        }
        if !self.retrieve.alpha.is_finite() || !(0.0..=1.0).contains(&self.retrieve.alpha) {
            return Err(OpenRagError::Config(format!(
                "retrieve.alpha must be finite and in 0.0..=1.0, got {}",
                self.retrieve.alpha
            )));
        }
        if !self.retrieve.admission_threshold.is_finite()
            || !(0.0..=1.0).contains(&self.retrieve.admission_threshold)
        {
            return Err(OpenRagError::Config(format!(
                "retrieve.admission_threshold must be finite and in 0.0..=1.0, got {}",
                self.retrieve.admission_threshold
            )));
        }
        if !self.retrieve.latest_revision_boost.is_finite()
            || self.retrieve.latest_revision_boost < 0.0
        {
            return Err(OpenRagError::Config(format!(
                "retrieve.latest_revision_boost must be finite and >= 0.0, got {}",
                self.retrieve.latest_revision_boost
            )));
        }
        for (idx, rule) in self.pii.guard.rules.iter().enumerate() {
            if rule.pattern_or_category.trim().is_empty() {
                return Err(OpenRagError::Config(format!(
                    "pii.guard.rules[{idx}].pattern_or_category must not be empty"
                )));
            }
            if rule.surfaces.is_empty() {
                return Err(OpenRagError::Config(format!(
                    "pii.guard.rules[{idx}].surfaces must not be empty"
                )));
            }
            if rule.kind == GuardRuleKind::Regex {
                Regex::new(&rule.pattern_or_category).map_err(|e| {
                    OpenRagError::Config(format!(
                        "invalid regex in pii.guard.rules[{idx}] ('{}'): {e}",
                        rule.pattern_or_category
                    ))
                })?;
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OptimizationStrategy {
    GreedyNode,
    Grid,
    AbPair,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SecurityGateConfig {
    pub require_zero_pii_leak: bool,
    pub require_zero_arn_violation: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SearchSpaceConfig {
    pub version: u32,
    pub golden_set: String,
    pub strategy: OptimizationStrategy,
    pub metric: String,
    pub security_gates: SecurityGateConfig,
    pub nodes: BTreeMap<String, Vec<serde_yaml_ng::Value>>,
}

impl SearchSpaceConfig {
    pub fn from_yaml_str(yaml: &str) -> Result<Self, OpenRagError> {
        let cfg: Self = serde_yaml_ng::from_str(yaml)
            .map_err(|e| OpenRagError::Config(format!("failed to parse search_space YAML: {e}")))?;
        cfg.validate()?;
        Ok(cfg)
    }

    pub fn from_yaml_file(path: &Path) -> Result<Self, OpenRagError> {
        let raw = std::fs::read_to_string(path).map_err(|e| {
            OpenRagError::Config(format!(
                "failed to read search space config '{}': {e}",
                path.display()
            ))
        })?;
        Self::from_yaml_str(&raw)
    }

    pub fn validate(&self) -> Result<(), OpenRagError> {
        if self.version == 0 || self.golden_set.trim().is_empty() || self.metric.trim().is_empty() {
            return Err(OpenRagError::Config(
                "search_space version, golden_set, and metric must be valid".into(),
            ));
        }
        if self.nodes.is_empty() {
            return Err(OpenRagError::Config(
                "search_space.nodes must contain at least one candidate axis".into(),
            ));
        }
        for (name, vals) in &self.nodes {
            if vals.is_empty() {
                return Err(OpenRagError::Config(format!(
                    "search_space.nodes['{name}'] must not be empty"
                )));
            }
        }
        Ok(())
    }
}
