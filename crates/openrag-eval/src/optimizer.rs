use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;

use openrag_core::{
    AuthLevel, ChunkerKind, ExternalMirrorBackend, FusionMode, GuardAction, GuardConfig,
    GuardRuleConfig, GuardRuleKind, GuardSurface, LlmConfig, MemoryBackendKind, MemoryConfig,
    OcrModeRule, OpenRagError, OptimizationStrategy, ParseConfig, PgPineconeEngine, PipelineConfig,
    PipelineProfile, PiiConfig, PrincipalContext, RetrieveConfig, SecurityGateConfig, Sensitivity,
    StoreConfig, ThumbnailConfig,
};
use openrag_pii::PiiEngine;
use openrag_retrieve::{HybridRetriever, RetrieveBackend, RetrieveOverrides, SearchRequest};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::bench::apply_egress_capsule;
use crate::golden::{
    CompleteOptimizationTrial, EvalStore, NewOptimizationTrial,
};
use crate::metrics::{
    check_arn_violation, check_pii_leak, extract_hit_doc_id, mrr, ndcg_at_k, recall_at_k,
};
use crate::types::{
    CandidateEvaluation, GoldenAxis, GoldenItem, GoldenSet, OptimizationStatus, OptimizeReport,
    OptimizeRequest, PipelineConfigRecord, PipelineProfileKind, PromotePipelineInput,
    RegisterPipelineInput, SecurityGateResult,
};

pub fn default_pipeline_config() -> PipelineConfig {
    PipelineConfig {
        version: 1,
        profile: PipelineProfile::Standard,
        parse: ParseConfig {
            hwp_engine: "kordoc".into(),
            pdf_engine: "kordoc".into(),
            kordoc_ocr: true,
            ocr_engine: "schift_ocr".into(),
            dewarp_enabled: true,
            pdf_native_enabled: true,
            default_ocr_mode: Sensitivity::Share,
            ocr_mode_by_auth_level: vec![
                OcrModeRule {
                    min_auth_level: AuthLevel::MAX,
                    required_mode: Sensitivity::Zdr,
                },
                OcrModeRule {
                    min_auth_level: AuthLevel::MIN,
                    required_mode: Sensitivity::Share,
                },
            ],
            reject_zdr_with_batch: true,
        },
        thumbnail: ThumbnailConfig {
            enabled: false,
            format: "webp".into(),
            max_edge_px: 1600,
            figure_max_edge_px: 1280,
            figure_max_bytes: 307_200,
        },
        pii: PiiConfig {
            schift_v7: true,
            marker_ko_pii: false,
            schift_v6: false,
            custom_dict_terms: Vec::new(),
            allowlist_terms: Vec::new(),
            capsule_enabled: true,
            guard: GuardConfig {
                enabled: true,
                default_action: GuardAction::Mask,
                allowlist: Vec::new(),
                rules: vec![
                    GuardRuleConfig {
                        kind: GuardRuleKind::PiiCategory,
                        pattern_or_category: "rrn_kr".into(),
                        action: GuardAction::Block,
                        surfaces: vec![GuardSurface::LlmEgress, GuardSurface::McpEgress],
                        replacement_text: None,
                        enabled: true,
                        priority: 10,
                    },
                    GuardRuleConfig {
                        kind: GuardRuleKind::PiiCategory,
                        pattern_or_category: "phone_kr".into(),
                        action: GuardAction::Mask,
                        surfaces: vec![
                            GuardSurface::InboundQuery,
                            GuardSurface::LlmEgress,
                            GuardSurface::McpEgress,
                            GuardSurface::IngestionGate,
                        ],
                        replacement_text: None,
                        enabled: true,
                        priority: 20,
                    },
                ],
            },
        },
        memory: MemoryConfig {
            backend: MemoryBackendKind::OpenragPg,
            mirror_to: None,
            kumiho_url: None,
            kumiho_api_key_env: None,
            offline_fallback: true,
        },
        store: StoreConfig {
            pg_pinecone_engine: PgPineconeEngine::Native,
            dim: 256,
            bm25_engine: "pg_search".into(),
            bm25_fallback: "pg_trgm".into(),
            external_mirror: ExternalMirrorBackend::None,
            outbox_batch_size: 64,
            outbox_max_attempts: 3,
            pg_pinecone_search: Default::default(),
            embed_document_title: false,
            chunker: ChunkerKind::Structure,
        },
        retrieve: RetrieveConfig {
            spaces: vec![
                "chunks".into(),
                "wiki".into(),
                "memory".into(),
            ],
            fusion: FusionMode::Convex,
            rrf_k: 60,
            alpha: 0.0,
            top_k: 12,
            rerank: true,
            latest_revision_boost: 0.15,
            admission_threshold: 0.35,
        },
        llm: LlmConfig {
            external_provider: "openrouter".into(),
            external_model: "anthropic/claude-3.5-sonnet".into(),
            local_provider: "vllm_local".into(),
            local_model: "Qwen/Qwen2.5-32B-Instruct".into(),
            local_only_min_auth_level: AuthLevel::try_from(8_u8).unwrap_or(AuthLevel::MAX),
            require_pii_masked_for_external: true,
        },
        egress: Default::default(),
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NormalizedSearchSpace {
    pub version: u32,
    pub golden_set: String,
    pub strategy: OptimizationStrategy,
    pub metric: String,
    pub security_gates: SecurityGateConfig,
    pub nodes: BTreeMap<String, Vec<serde_yaml_ng::Value>>,
}

fn default_metric() -> String {
    "ndcg_at_10".to_string()
}

const fn default_security_gates() -> SecurityGateConfig {
    SecurityGateConfig {
        require_zero_pii_leak: true,
        require_zero_arn_violation: true,
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawSearchSpaceDoc {
    version: u32,
    #[serde(default)]
    golden_set: String,
    strategy: OptimizationStrategy,
    #[serde(default = "default_metric")]
    metric: String,
    #[serde(default = "default_security_gates")]
    security_gates: SecurityGateConfig,
    nodes: BTreeMap<String, Vec<serde_yaml_ng::Value>>,
}

pub fn normalize_search_space_node_key(raw: &str) -> String {
    match raw.trim() {
        "retrieve.fusion" | "retrieve_fusion" => "retrieve.fusion".to_string(),
        "retrieve.alpha" | "retrieve_alpha" => "retrieve.alpha".to_string(),
        "retrieve.top_k" | "retrieve_top_k" => "retrieve.top_k".to_string(),
        "retrieve.rerank" | "retrieve_rerank" => "retrieve.rerank".to_string(),
        "retrieve.latest_revision_boost" | "retrieve_latest_revision_boost" => {
            "retrieve.latest_revision_boost".to_string()
        }
        "retrieve.rrf_k" | "retrieve_rrf_k" => "retrieve.rrf_k".to_string(),
        "retrieve.admission_threshold" | "retrieve_admission_threshold" => {
            "retrieve.admission_threshold".to_string()
        }
        "retrieve.spaces" | "retrieve_spaces" => "retrieve.spaces".to_string(),
        "parse" | "parse.ocr_engine" | "parse_ocr_engine" => "parse.ocr_engine".to_string(),
        "parse.dewarp_enabled" | "parse_dewarp_enabled" => "parse.dewarp_enabled".to_string(),
        "store.pg_pinecone_engine" | "store_pg_pinecone_engine" => {
            "store.pg_pinecone_engine".to_string()
        }
        "llm.model" | "llm_model" => "llm.model".to_string(),
        "pii.capsule_enabled" | "pii_capsule_enabled" => "pii.capsule_enabled".to_string(),
        "iam.bypass_rls_test_only" | "iam_bypass_rls_test_only" => {
            "iam.bypass_rls_test_only".to_string()
        }
        "chunking" => "chunking".to_string(),
        other => other.replace('_', "."),
    }
}

pub fn parse_search_space_yaml(yaml: &str) -> Result<NormalizedSearchSpace, OpenRagError> {
    let raw: RawSearchSpaceDoc = serde_yaml_ng::from_str(yaml).map_err(|e| {
        OpenRagError::Validation(format!("invalid search_space YAML: {e}"))
    })?;

    if raw.version == 0 {
        return Err(OpenRagError::Validation(
            "search_space.version must be >= 1".into(),
        ));
    }
    if raw.metric.trim().is_empty() {
        return Err(OpenRagError::Validation(
            "search_space.metric must not be empty".into(),
        ));
    }
    if raw.nodes.is_empty() {
        return Err(OpenRagError::Validation(
            "search_space.nodes must contain at least one candidate axis".into(),
        ));
    }

    let mut normalized_nodes: BTreeMap<String, Vec<serde_yaml_ng::Value>> = BTreeMap::new();
    for (raw_key, values) in raw.nodes {
        if values.is_empty() {
            return Err(OpenRagError::Validation(format!(
                "search_space.nodes['{raw_key}'] must not be empty"
            )));
        }
        let canon = normalize_search_space_node_key(&raw_key);
        if normalized_nodes.contains_key(&canon) {
            return Err(OpenRagError::Validation(format!(
                "duplicate search_space node key after normalization: '{canon}'"
            )));
        }
        normalized_nodes.insert(canon, values);
    }

    Ok(NormalizedSearchSpace {
        version: raw.version,
        golden_set: raw.golden_set,
        strategy: raw.strategy,
        metric: raw.metric,
        security_gates: raw.security_gates,
        nodes: normalized_nodes,
    })
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum BoostValue {
    Float(f32),
    Bool(bool),
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum OcrModeMapOrList {
    List(Vec<OcrModeRule>),
    ThresholdMap {
        #[serde(default)]
        share_max_level: Option<u8>,
        #[serde(default)]
        private_min_level: Option<u8>,
        #[serde(default)]
        zdr_min_level: Option<u8>,
    },
}

#[derive(Debug, Deserialize)]
struct TolerantParseConfig {
    #[serde(alias = "hwp")]
    hwp_engine: Option<String>,
    #[serde(alias = "pdf")]
    pdf_engine: Option<String>,
    kordoc_ocr: Option<bool>,
    #[serde(alias = "ocr")]
    ocr_engine: Option<String>,
    #[serde(alias = "dewarp")]
    dewarp_enabled: Option<bool>,
    pdf_native_enabled: Option<bool>,
    default_ocr_mode: Option<Sensitivity>,
    ocr_mode_by_auth_level: Option<OcrModeMapOrList>,
    reject_zdr_with_batch: Option<bool>,
}

#[derive(Debug, Deserialize)]
struct TolerantThumbnailConfig {
    enabled: Option<bool>,
    format: Option<String>,
    max_edge_px: Option<u32>,
    figure_max_edge_px: Option<u32>,
    figure_max_bytes: Option<u64>,
}

#[derive(Debug, Deserialize)]
struct TolerantPiiConfig {
    schift_v7: Option<bool>,
    marker_ko_pii: Option<bool>,
    schift_v6: Option<bool>,
    #[serde(alias = "custom_dict")]
    custom_dict_terms: Option<Vec<String>>,
    #[serde(alias = "allowlist")]
    allowlist_terms: Option<Vec<String>>,
    capsule_enabled: Option<bool>,
    guard: Option<GuardConfig>,
}

#[derive(Debug, Deserialize)]
struct TolerantStoreConfig {
    pg_pinecone_engine: Option<PgPineconeEngine>,
    dim: Option<u16>,
    #[serde(alias = "bm25")]
    bm25_engine: Option<String>,
    bm25_fallback: Option<String>,
    external_mirror: Option<ExternalMirrorBackend>,
    outbox_batch_size: Option<u32>,
    outbox_max_attempts: Option<u32>,
    pg_pinecone_search: Option<openrag_core::PgPineconeSearchConfig>,
    embed_document_title: Option<bool>,
    chunker: Option<ChunkerKind>,
}

#[derive(Debug, Deserialize)]
struct TolerantRetrieveConfig {
    spaces: Option<Vec<String>>,
    fusion: Option<FusionMode>,
    rrf_k: Option<u32>,
    alpha: Option<f32>,
    top_k: Option<usize>,
    rerank: Option<bool>,
    latest_revision_boost: Option<BoostValue>,
    admission_threshold: Option<f32>,
}

#[derive(Debug, Deserialize)]
struct TolerantLlmConfig {
    external_provider: Option<String>,
    external_model: Option<String>,
    local_provider: Option<String>,
    local_model: Option<String>,
    local_only_min_auth_level: Option<AuthLevel>,
    require_pii_masked_for_external: Option<bool>,
}

#[derive(Debug, Deserialize)]
struct TolerantPipelineDoc {
    version: Option<u32>,
    profile: Option<PipelineProfileKind>,
    parse: Option<TolerantParseConfig>,
    thumbnail: Option<TolerantThumbnailConfig>,
    pii: Option<TolerantPiiConfig>,
    memory: Option<MemoryConfig>,
    store: Option<TolerantStoreConfig>,
    retrieve: Option<TolerantRetrieveConfig>,
    llm: Option<TolerantLlmConfig>,
}

pub fn parse_pipeline_yaml_with_profile(
    yaml: &str,
) -> Result<(PipelineConfig, PipelineProfileKind), OpenRagError> {
    if let Ok(strict) = PipelineConfig::from_yaml_str(yaml) {
        let profile = PipelineProfileKind::from(strict.profile);
        return Ok((strict, profile));
    }

    let doc: TolerantPipelineDoc = serde_yaml_ng::from_str(yaml).map_err(|e| {
        OpenRagError::Config(format!("failed to parse pipeline YAML: {e}"))
    })?;

    if doc.parse.is_none()
        && doc.thumbnail.is_none()
        && doc.pii.is_none()
        && doc.store.is_none()
        && doc.retrieve.is_none()
        && doc.llm.is_none()
    {
        return Err(OpenRagError::Config(
            "pipeline YAML must contain pipeline configuration sections".into(),
        ));
    }

    let mut cfg = default_pipeline_config();
    if let Some(v) = doc.version {
        cfg.version = v;
    }

    let profile_kind = doc.profile.unwrap_or(PipelineProfileKind::Standard);
    cfg.profile = match profile_kind {
        PipelineProfileKind::Airgap => PipelineProfile::Airgap,
        PipelineProfileKind::Standard | PipelineProfileKind::Custom => PipelineProfile::Standard,
    };

    if let Some(p) = doc.parse {
        if let Some(v) = p.hwp_engine {
            cfg.parse.hwp_engine = v;
        }
        if let Some(v) = p.pdf_engine {
            cfg.parse.pdf_engine = v;
        }
        if let Some(v) = p.kordoc_ocr {
            cfg.parse.kordoc_ocr = v;
        }
        if let Some(v) = p.ocr_engine {
            cfg.parse.ocr_engine = v;
        }
        if let Some(v) = p.dewarp_enabled {
            cfg.parse.dewarp_enabled = v;
        }
        if let Some(v) = p.pdf_native_enabled {
            cfg.parse.pdf_native_enabled = v;
        }
        if let Some(v) = p.default_ocr_mode {
            cfg.parse.default_ocr_mode = v;
        }
        if let Some(v) = p.reject_zdr_with_batch {
            cfg.parse.reject_zdr_with_batch = v;
        }
        if let Some(mode_map) = p.ocr_mode_by_auth_level {
            cfg.parse.ocr_mode_by_auth_level = match mode_map {
                OcrModeMapOrList::List(list) => list,
                OcrModeMapOrList::ThresholdMap {
                    share_max_level,
                    private_min_level,
                    zdr_min_level,
                } => {
                    let mut rules = Vec::new();
                    if let Some(lvl) = zdr_min_level {
                        rules.push(OcrModeRule {
                            min_auth_level: AuthLevel::new(lvl)?,
                            required_mode: Sensitivity::Zdr,
                        });
                    }
                    if let Some(lvl) = private_min_level {
                        rules.push(OcrModeRule {
                            min_auth_level: AuthLevel::new(lvl)?,
                            required_mode: Sensitivity::Private,
                        });
                    }
                    if let Some(lvl) = share_max_level {
                        let _ = lvl;
                        rules.push(OcrModeRule {
                            min_auth_level: AuthLevel::MIN,
                            required_mode: Sensitivity::Share,
                        });
                    }
                    rules
                }
            };
        }
    }

    if let Some(t) = doc.thumbnail {
        if let Some(v) = t.format {
            cfg.thumbnail.format = v;
        }
        if let Some(v) = t.max_edge_px {
            cfg.thumbnail.max_edge_px = v;
        }
        if let Some(v) = t.figure_max_edge_px {
            cfg.thumbnail.figure_max_edge_px = v;
        }
        if let Some(v) = t.enabled {
            cfg.thumbnail.enabled = v;
        }
        if let Some(v) = t.figure_max_bytes {
            cfg.thumbnail.figure_max_bytes = v;
        }
    }

    if let Some(pii) = doc.pii {
        if let Some(v) = pii.schift_v7 {
            cfg.pii.schift_v7 = v;
        }
        if let Some(v) = pii.marker_ko_pii {
            cfg.pii.marker_ko_pii = v;
        }
        if let Some(v) = pii.schift_v6 {
            cfg.pii.schift_v6 = v;
        }
        if let Some(v) = pii.custom_dict_terms {
            cfg.pii.custom_dict_terms = v;
        }
        if let Some(v) = pii.allowlist_terms {
            cfg.pii.allowlist_terms = v;
        }
        if let Some(v) = pii.capsule_enabled {
            cfg.pii.capsule_enabled = v;
        }
        if let Some(v) = pii.guard {
            cfg.pii.guard = v;
        }
    }

    if let Some(mem) = doc.memory {
        cfg.memory = mem;
    }

    if let Some(st) = doc.store {
        if let Some(v) = st.pg_pinecone_engine {
            cfg.store.pg_pinecone_engine = v;
        }
        if let Some(v) = st.dim {
            cfg.store.dim = v;
        }
        if let Some(v) = st.bm25_engine {
            cfg.store.bm25_engine = v;
        }
        if let Some(v) = st.bm25_fallback {
            cfg.store.bm25_fallback = v;
        }
        if let Some(v) = st.external_mirror {
            cfg.store.external_mirror = v;
        }
        if let Some(v) = st.outbox_batch_size {
            cfg.store.outbox_batch_size = v;
        }
        if let Some(v) = st.outbox_max_attempts {
            cfg.store.outbox_max_attempts = v;
        }
        if let Some(v) = st.pg_pinecone_search {
            cfg.store.pg_pinecone_search = v;
        }
        if let Some(v) = st.embed_document_title {
            cfg.store.embed_document_title = v;
        }
        if let Some(v) = st.chunker {
            cfg.store.chunker = v;
        }
    }

    if let Some(ret) = doc.retrieve {
        if let Some(v) = ret.spaces {
            cfg.retrieve.spaces = v;
        }
        if let Some(v) = ret.fusion {
            cfg.retrieve.fusion = v;
        }
        if let Some(v) = ret.rrf_k {
            cfg.retrieve.rrf_k = v;
        }
        if let Some(v) = ret.alpha {
            cfg.retrieve.alpha = v;
        }
        if let Some(v) = ret.top_k {
            cfg.retrieve.top_k = v;
        }
        if let Some(v) = ret.rerank {
            cfg.retrieve.rerank = v;
        }
        if let Some(v) = ret.latest_revision_boost {
            cfg.retrieve.latest_revision_boost = match v {
                BoostValue::Float(f) => f,
                BoostValue::Bool(true) => 0.15,
                BoostValue::Bool(false) => 0.0,
            };
        }
        if let Some(v) = ret.admission_threshold {
            cfg.retrieve.admission_threshold = v;
        }
    }

    if let Some(llm) = doc.llm {
        if let Some(v) = llm.external_provider {
            cfg.llm.external_provider = v;
        }
        if let Some(v) = llm.external_model {
            cfg.llm.external_model = v;
        }
        if let Some(v) = llm.local_provider {
            cfg.llm.local_provider = v;
        }
        if let Some(v) = llm.local_model {
            cfg.llm.local_model = v;
        }
        if let Some(v) = llm.local_only_min_auth_level {
            cfg.llm.local_only_min_auth_level = v;
        }
        if let Some(v) = llm.require_pii_masked_for_external {
            cfg.llm.require_pii_masked_for_external = v;
        }
    }

    cfg.validate()?;
    Ok((cfg, profile_kind))
}

pub fn parse_pipeline_yaml_tolerant(yaml: &str) -> Result<PipelineConfig, OpenRagError> {
    parse_pipeline_yaml_with_profile(yaml).map(|(cfg, _)| cfg)
}

#[derive(Debug, Clone)]
struct CandidateState {
    pipeline: PipelineConfig,
    bypass_rls_test_only: bool,
    assignments: BTreeMap<String, String>,
}

impl CandidateState {
    fn from_pipeline(pipeline: PipelineConfig) -> Self {
        Self {
            pipeline,
            bypass_rls_test_only: false,
            assignments: BTreeMap::new(),
        }
    }

    fn description(&self) -> String {
        if self.assignments.is_empty() {
            return "baseline".to_string();
        }
        self.assignments
            .iter()
            .map(|(k, v)| format!("{k}={v}"))
            .collect::<Vec<_>>()
            .join(", ")
    }

    fn apply_node(
        &mut self,
        canonical_key: &str,
        value: &serde_yaml_ng::Value,
    ) -> Result<(), OpenRagError> {
        self.assignments
            .insert(canonical_key.to_string(), yaml_value_label(value));

        match canonical_key {
            "retrieve.fusion" => {
                let s = value.as_str().ok_or_else(|| {
                    OpenRagError::Validation("retrieve.fusion candidate must be a string".into())
                })?;
                self.pipeline.retrieve.fusion = match s.trim().to_ascii_lowercase().as_str() {
                    "rrf" => FusionMode::Rrf,
                    "convex" => FusionMode::Convex,
                    other => {
                        return Err(OpenRagError::Validation(format!(
                            "unknown fusion mode '{other}'"
                        )))
                    }
                };
            }
            "retrieve.alpha" => {
                let f = yaml_as_f64(value, "retrieve.alpha")? as f32;
                if !f.is_finite() || !(0.0..=1.0).contains(&f) {
                    return Err(OpenRagError::Validation(format!(
                        "retrieve.alpha must be in 0.0..=1.0, got {f}"
                    )));
                }
                self.pipeline.retrieve.alpha = f;
            }
            "retrieve.top_k" => {
                let k = yaml_as_usize(value, "retrieve.top_k")?;
                if k == 0 || k > 200 {
                    return Err(OpenRagError::Validation(format!(
                        "retrieve.top_k must be in 1..=200, got {k}"
                    )));
                }
                self.pipeline.retrieve.top_k = k;
            }
            "retrieve.rerank" => {
                self.pipeline.retrieve.rerank = yaml_as_bool(value, "retrieve.rerank")?;
            }
            "retrieve.latest_revision_boost" => {
                if let Some(b) = value.as_bool() {
                    self.pipeline.retrieve.latest_revision_boost = if b { 0.15 } else { 0.0 };
                } else {
                    let f = yaml_as_f64(value, "retrieve.latest_revision_boost")? as f32;
                    if !f.is_finite() || f < 0.0 {
                        return Err(OpenRagError::Validation(format!(
                            "retrieve.latest_revision_boost must be >= 0.0, got {f}"
                        )));
                    }
                    self.pipeline.retrieve.latest_revision_boost = f;
                }
            }
            "retrieve.rrf_k" => {
                let rrf_k = yaml_as_usize(value, "retrieve.rrf_k")?;
                if rrf_k == 0 {
                    return Err(OpenRagError::Validation(
                        "retrieve.rrf_k must be >= 1".into(),
                    ));
                }
                self.pipeline.retrieve.rrf_k = rrf_k as u32;
            }
            "retrieve.admission_threshold" => {
                let t = yaml_as_f64(value, "retrieve.admission_threshold")? as f32;
                if !t.is_finite() || !(0.0..=1.0).contains(&t) {
                    return Err(OpenRagError::Validation(format!(
                        "retrieve.admission_threshold must be in 0.0..=1.0, got {t}"
                    )));
                }
                self.pipeline.retrieve.admission_threshold = t;
            }
            "retrieve.spaces" => {
                let seq = value.as_sequence().ok_or_else(|| {
                    OpenRagError::Validation("retrieve.spaces candidate must be a sequence".into())
                })?;
                let mut spaces = Vec::with_capacity(seq.len());
                for item in seq {
                    let s = item.as_str().ok_or_else(|| {
                        OpenRagError::Validation("retrieve.spaces entries must be strings".into())
                    })?;
                    spaces.push(s.to_string());
                }
                self.pipeline.retrieve.spaces = spaces;
            }
            "parse.ocr_engine" => {
                let s = value.as_str().ok_or_else(|| {
                    OpenRagError::Validation("parse.ocr_engine candidate must be a string".into())
                })?;
                self.pipeline.parse.ocr_engine = s.to_string();
            }
            "parse.dewarp_enabled" => {
                self.pipeline.parse.dewarp_enabled = yaml_as_bool(value, "parse.dewarp_enabled")?;
            }
            "store.pg_pinecone_engine" => {
                let s = value.as_str().ok_or_else(|| {
                    OpenRagError::Validation(
                        "store.pg_pinecone_engine candidate must be a string".into(),
                    )
                })?;
                self.pipeline.store.pg_pinecone_engine = match s.trim().to_ascii_lowercase().as_str()
                {
                    "native" => PgPineconeEngine::Native,
                    "pgvector" => PgPineconeEngine::Pgvector,
                    "remote" => PgPineconeEngine::Remote,
                    other => {
                        return Err(OpenRagError::Validation(format!(
                            "unknown pg_pinecone_engine '{other}'"
                        )))
                    }
                };
            }
            "llm.model" => {
                let s = value.as_str().ok_or_else(|| {
                    OpenRagError::Validation("llm.model candidate must be a string".into())
                })?;
                if s.to_ascii_lowercase().contains("local") || s.contains("Qwen") {
                    self.pipeline.llm.local_model = s.to_string();
                } else {
                    self.pipeline.llm.external_model = s.to_string();
                }
            }
            "pii.capsule_enabled" => {
                self.pipeline.pii.capsule_enabled = yaml_as_bool(value, "pii.capsule_enabled")?;
            }
            "iam.bypass_rls_test_only" => {
                self.bypass_rls_test_only = yaml_as_bool(value, "iam.bypass_rls_test_only")?;
            }
            "chunking" => {}
            _ => {}
        }
        Ok(())
    }
}

fn yaml_value_label(v: &serde_yaml_ng::Value) -> String {
    match v {
        serde_yaml_ng::Value::Bool(b) => b.to_string(),
        serde_yaml_ng::Value::Number(n) => n.to_string(),
        serde_yaml_ng::Value::String(s) => s.clone(),
        other => format!("{other:?}"),
    }
}

fn yaml_as_bool(v: &serde_yaml_ng::Value, field: &str) -> Result<bool, OpenRagError> {
    v.as_bool()
        .ok_or_else(|| OpenRagError::Validation(format!("{field} candidate must be a boolean")))
}

fn yaml_as_f64(v: &serde_yaml_ng::Value, field: &str) -> Result<f64, OpenRagError> {
    v.as_f64()
        .ok_or_else(|| OpenRagError::Validation(format!("{field} candidate must be a number")))
}

fn yaml_as_usize(v: &serde_yaml_ng::Value, field: &str) -> Result<usize, OpenRagError> {
    let u = v.as_u64().ok_or_else(|| {
        OpenRagError::Validation(format!("{field} candidate must be a non-negative integer"))
    })?;
    usize::try_from(u)
        .map_err(|_| OpenRagError::Validation(format!("{field} integer out of range")))
}

pub struct AutoRagOptimizer;

impl AutoRagOptimizer {
    pub async fn optimize<S: EvalStore, B: RetrieveBackend>(
        store: &S,
        retriever: &HybridRetriever<B>,
        actor: &PrincipalContext,
        req: &OptimizeRequest,
    ) -> Result<OptimizeReport, OpenRagError> {
        let space = parse_search_space_yaml(&req.search_space_yaml)?;
        let strategy = req.strategy.unwrap_or(space.strategy);
        let gate_policy = SecurityGateConfig {
            require_zero_pii_leak: req
                .require_zero_pii_leak
                .unwrap_or(space.security_gates.require_zero_pii_leak),
            require_zero_arn_violation: req
                .require_zero_arn_violation
                .unwrap_or(space.security_gates.require_zero_arn_violation),
        };

        let base_pipeline = match req.base_pipeline_yaml.as_deref() {
            Some(yaml) => parse_pipeline_yaml_tolerant(yaml)?,
            None => default_pipeline_config(),
        };

        let gs = resolve_opt_golden_set(store, actor.org_id, req, &space.golden_set).await?;
        let items = store.list_golden_items(actor.org_id, gs.id, false).await?;
        if items.is_empty() {
            return Err(OpenRagError::Validation(format!(
                "golden set '{}' has no non-rejected items for optimization",
                gs.name
            )));
        }

        let mut principal_cache: BTreeMap<Uuid, PrincipalContext> = BTreeMap::new();
        for item in &items {
            if let Some(pid) = item.test_principal_id {
                if let std::collections::btree_map::Entry::Vacant(entry) =
                    principal_cache.entry(pid)
                {
                    let resolved = store
                        .resolve_principal(actor.org_id, pid)
                        .await?
                        .ok_or_else(|| {
                            OpenRagError::NotFound(format!(
                                "test_principal_id {pid} not found in org {}",
                                actor.org_id
                            ))
                        })?;
                    entry.insert(resolved);
                }
            }
        }

        let trial = store
            .insert_optimization_trial(&NewOptimizationTrial {
                org_id: actor.org_id,
                golden_set_id: gs.id,
                search_space_yaml: req.search_space_yaml.clone(),
                strategy,
            })
            .await?;

        let eval_ctx = OptimizeEvalCtx {
            retriever,
            actor,
            principal_cache: &principal_cache,
            items: &items,
            bucket_id: req.bucket_id,
            metric: &space.metric,
        };

        let mut history: Vec<CandidateEvaluation> = Vec::new();
        let mut best_passing: Option<(CandidateState, CandidateEvaluation)> = None;

        let search_res = match strategy {
            OptimizationStrategy::GreedyNode => {
                run_greedy_node(
                    &eval_ctx,
                    &base_pipeline,
                    &space.nodes,
                    &gate_policy,
                    &mut history,
                    &mut best_passing,
                )
                .await
            }
            OptimizationStrategy::Grid => {
                run_grid(
                    &eval_ctx,
                    &base_pipeline,
                    &space.nodes,
                    &gate_policy,
                    &mut history,
                    &mut best_passing,
                )
                .await
            }
            OptimizationStrategy::AbPair => {
                run_ab_pair(
                    &eval_ctx,
                    &base_pipeline,
                    &space.nodes,
                    &gate_policy,
                    &mut history,
                    &mut best_passing,
                )
                .await
            }
        };

        if let Err(err) = search_res {
            let _ = store
                .complete_optimization_trial(
                    actor.org_id,
                    trial.id,
                    &CompleteOptimizationTrial {
                        best_config_yaml: None,
                        best_score: None,
                        security_gate_pass: false,
                        status: OptimizationStatus::Failed,
                    },
                )
                .await;
            return Err(err);
        }

        let Some((best_state, best_eval)) = best_passing else {
            let worst_gate = history
                .iter()
                .map(|h| h.security_gates)
                .find(|g| !g.passes_policy(&gate_policy))
                .unwrap_or_else(|| SecurityGateResult::new(1, 0));

            let _completed = store
                .complete_optimization_trial(
                    actor.org_id,
                    trial.id,
                    &CompleteOptimizationTrial {
                        best_config_yaml: None,
                        best_score: None,
                        security_gate_pass: false,
                        status: OptimizationStatus::BlockedSecurityGate,
                    },
                )
                .await?;

            return Err(OpenRagError::Forbidden(format!(
                "AutoRAG optimization blocked by security gates (pii_leak_count={}, arn_violation_count={}); no clean candidate passed",
                worst_gate.pii_leak_count, worst_gate.arn_violation_count
            )));
        };

        let best_yaml = serde_yaml_ng::to_string(&best_state.pipeline)
            .map_err(|e| OpenRagError::Internal(format!("serialize best pipeline YAML: {e}")))?;

        if let Some(export_path) = req.export_best_path.as_deref() {
            write_atomic_yaml(Path::new(export_path), &best_yaml)?;
        }

        let completed = store
            .complete_optimization_trial(
                actor.org_id,
                trial.id,
                &CompleteOptimizationTrial {
                    best_config_yaml: Some(best_yaml.clone()),
                    best_score: Some(best_eval.score as f32),
                    security_gate_pass: true,
                    status: OptimizationStatus::Completed,
                },
            )
            .await?;

        Ok(OptimizeReport {
            trial: completed,
            best_config_yaml: Some(best_yaml),
            best_score: Some(best_eval.score as f32),
            security_gates: best_eval.security_gates,
            evaluated_candidates: history.len(),
            history,
        })
    }
}

struct OptimizeEvalCtx<'a, B: RetrieveBackend> {
    retriever: &'a HybridRetriever<B>,
    actor: &'a PrincipalContext,
    principal_cache: &'a BTreeMap<Uuid, PrincipalContext>,
    items: &'a [GoldenItem],
    bucket_id: Option<Uuid>,
    metric: &'a str,
}

async fn evaluate_candidate<B: RetrieveBackend>(
    ctx: &OptimizeEvalCtx<'_, B>,
    candidate: &CandidateState,
) -> Result<CandidateEvaluation, OpenRagError> {
    let pii_engine = PiiEngine::from_config(&candidate.pipeline.pii);
    let overrides = RetrieveOverrides {
        spaces: Some(candidate.pipeline.retrieve.spaces.clone()),
        fusion: Some(candidate.pipeline.retrieve.fusion),
        rrf_k: Some(candidate.pipeline.retrieve.rrf_k),
        alpha: Some(candidate.pipeline.retrieve.alpha),
        top_k: Some(candidate.pipeline.retrieve.top_k),
        rerank: Some(candidate.pipeline.retrieve.rerank),
        latest_revision_boost: Some(candidate.pipeline.retrieve.latest_revision_boost),
        admission_threshold: Some(candidate.pipeline.retrieve.admission_threshold),
    };

    let mut mrr_sum = 0.0_f64;
    let mut recall_sum = 0.0_f64;
    let mut ndcg_sum = 0.0_f64;
    let mut pii_leak_count = 0_i32;
    let mut arn_violation_count = 0_i32;

    for item in ctx.items {
        let item_ctx = item
            .test_principal_id
            .and_then(|pid| ctx.principal_cache.get(&pid))
            .unwrap_or(ctx.actor);

        let mut search_req = SearchRequest::new(&item.query_text);
        search_req.bucket_id = ctx.bucket_id;

        let resp = ctx
            .retriever
            .search_with_overrides(item_ctx, &search_req, &overrides)
            .await?;

        let retrieved_doc_ids: Vec<Uuid> = resp.hits.iter().map(extract_hit_doc_id).collect();
        mrr_sum += mrr(&retrieved_doc_ids, &item.expected_doc_ids);
        recall_sum += recall_at_k(&retrieved_doc_ids, &item.expected_doc_ids, 10);
        ndcg_sum += ndcg_at_k(&retrieved_doc_ids, &item.expected_doc_ids, 10);

        if check_arn_violation(
            &retrieved_doc_ids,
            &item.forbidden_doc_ids,
            candidate.bypass_rls_test_only,
        ) {
            arn_violation_count += 1;
        }

        let joined_hits: String = resp
            .hits
            .iter()
            .map(|h| h.text.as_str())
            .collect::<Vec<_>>()
            .join("\n");
        let raw_egress = if item.axis == GoldenAxis::Pii || !item.expected_pii_tokens.is_empty() {
            if joined_hits.trim().is_empty() {
                item.query_text.clone()
            } else {
                format!("{joined_hits}\n{}", item.query_text)
            }
        } else {
            joined_hits
        };

        let egress = apply_egress_capsule(&raw_egress, &candidate.pipeline, &pii_engine)?;
        if !candidate.pipeline.pii.capsule_enabled
            || check_pii_leak(
                &egress,
                &item.expected_pii_tokens,
                &candidate.pipeline,
                &pii_engine,
            )?
        {
            pii_leak_count += 1;
        }
    }

    let n = ctx.items.len() as f64;
    let mean_mrr = mrr_sum / n;
    let mean_recall = recall_sum / n;
    let mean_ndcg = ndcg_sum / n;
    let score = match ctx.metric.trim().to_ascii_lowercase().as_str() {
        "mrr" => mean_mrr,
        "recall" | "recall_at_10" | "recall@10" => mean_recall,
        _ => mean_ndcg,
    };

    Ok(CandidateEvaluation {
        description: candidate.description(),
        score,
        mrr: mean_mrr,
        recall_at_10: mean_recall,
        ndcg_at_10: mean_ndcg,
        security_gates: SecurityGateResult::new(pii_leak_count, arn_violation_count),
    })
}

fn update_best(
    best_passing: &mut Option<(CandidateState, CandidateEvaluation)>,
    gate_policy: &SecurityGateConfig,
    state: &CandidateState,
    eval: &CandidateEvaluation,
) {
    if !eval.security_gates.passes_policy(gate_policy) {
        return;
    }
    let replace = match best_passing {
        None => true,
        Some((_, cur_best)) => eval.score > cur_best.score,
    };
    if replace {
        *best_passing = Some((state.clone(), eval.clone()));
    }
}

async fn run_greedy_node<B: RetrieveBackend>(
    ctx: &OptimizeEvalCtx<'_, B>,
    base: &PipelineConfig,
    nodes: &BTreeMap<String, Vec<serde_yaml_ng::Value>>,
    gate_policy: &SecurityGateConfig,
    history: &mut Vec<CandidateEvaluation>,
    best_passing: &mut Option<(CandidateState, CandidateEvaluation)>,
) -> Result<(), OpenRagError> {
    // Initialize every axis in `nodes` to its first specified candidate value so that
    // constrained search spaces (e.g. `pii.capsule_enabled: [false]`) cannot silently
    // inherit an unlisted default from `base` on subsequent node sweeps.
    let mut current = CandidateState::from_pipeline(base.clone());
    for (node_key, values) in nodes {
        if let Some(first_val) = values.first() {
            current.apply_node(node_key, first_val)?;
        }
    }

    let initial_eval = evaluate_candidate(ctx, &current).await?;
    history.push(initial_eval.clone());
    update_best(best_passing, gate_policy, &current, &initial_eval);

    for (node_key, values) in nodes {
        let mut best_for_node: Option<(CandidateState, CandidateEvaluation)> = None;
        for val in values {
            let mut trial_state = current.clone();
            trial_state.apply_node(node_key, val)?;
            let eval = evaluate_candidate(ctx, &trial_state).await?;
            history.push(eval.clone());
            update_best(best_passing, gate_policy, &trial_state, &eval);
            if eval.security_gates.passes_policy(gate_policy) {
                let better = best_for_node
                    .as_ref()
                    .is_none_or(|(_, cur)| eval.score > cur.score);
                if better {
                    best_for_node = Some((trial_state, eval));
                }
            }
        }
        if let Some((chosen, _)) = best_for_node {
            current = chosen;
        }
    }
    Ok(())
}

async fn run_grid<B: RetrieveBackend>(
    ctx: &OptimizeEvalCtx<'_, B>,
    base: &PipelineConfig,
    nodes: &BTreeMap<String, Vec<serde_yaml_ng::Value>>,
    gate_policy: &SecurityGateConfig,
    history: &mut Vec<CandidateEvaluation>,
    best_passing: &mut Option<(CandidateState, CandidateEvaluation)>,
) -> Result<(), OpenRagError> {
    const MAX_GRID_COMBINATIONS: usize = 4096;
    let mut states = vec![CandidateState::from_pipeline(base.clone())];

    for (node_key, values) in nodes {
        let mut next_states = Vec::new();
        for st in &states {
            for val in values {
                if next_states.len() >= MAX_GRID_COMBINATIONS {
                    break;
                }
                let mut cloned = st.clone();
                cloned.apply_node(node_key, val)?;
                next_states.push(cloned);
            }
            if next_states.len() >= MAX_GRID_COMBINATIONS {
                break;
            }
        }
        states = next_states;
    }

    for st in states {
        let eval = evaluate_candidate(ctx, &st).await?;
        history.push(eval.clone());
        update_best(best_passing, gate_policy, &st, &eval);
    }
    Ok(())
}

async fn run_ab_pair<B: RetrieveBackend>(
    ctx: &OptimizeEvalCtx<'_, B>,
    base: &PipelineConfig,
    nodes: &BTreeMap<String, Vec<serde_yaml_ng::Value>>,
    gate_policy: &SecurityGateConfig,
    history: &mut Vec<CandidateEvaluation>,
    best_passing: &mut Option<(CandidateState, CandidateEvaluation)>,
) -> Result<(), OpenRagError> {
    let mut baseline = CandidateState::from_pipeline(base.clone());
    for (node_key, values) in nodes {
        if let Some(first_val) = values.first() {
            baseline.apply_node(node_key, first_val)?;
        }
    }
    let base_eval = evaluate_candidate(ctx, &baseline).await?;
    history.push(base_eval.clone());
    update_best(best_passing, gate_policy, &baseline, &base_eval);

    for (node_key, values) in nodes {
        for val in values {
            let mut variant = baseline.clone();
            variant.apply_node(node_key, val)?;
            let eval = evaluate_candidate(ctx, &variant).await?;
            history.push(eval.clone());
            update_best(best_passing, gate_policy, &variant, &eval);
        }
    }
    Ok(())
}

async fn resolve_opt_golden_set<S: EvalStore>(
    store: &S,
    org_id: Uuid,
    req: &OptimizeRequest,
    space_golden_set: &str,
) -> Result<GoldenSet, OpenRagError> {
    if let Some(id) = req.golden_set_id {
        return store.get_golden_set(org_id, id).await;
    }
    let name = req
        .golden_set
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| space_golden_set.trim());
    if name.is_empty() {
        return Err(OpenRagError::Validation(
            "golden_set name or golden_set_id is required".into(),
        ));
    }
    if let Ok(uuid) = Uuid::parse_str(name) {
        if let Ok(gs) = store.get_golden_set(org_id, uuid).await {
            return Ok(gs);
        }
    }
    store.get_golden_set_by_name(org_id, name).await
}

fn write_atomic_yaml(path: &Path, yaml: &str) -> Result<(), OpenRagError> {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    if !parent.as_os_str().is_empty() {
        std::fs::create_dir_all(parent).map_err(|e| {
            OpenRagError::Internal(format!(
                "failed to create directory '{}': {e}",
                parent.display()
            ))
        })?;
    }
    let tmp_name = format!(".openrag-best-{}.tmp", Uuid::new_v4());
    let tmp_path = parent.join(tmp_name);
    std::fs::write(&tmp_path, yaml).map_err(|e| {
        OpenRagError::Internal(format!(
            "failed writing temp pipeline YAML '{}': {e}",
            tmp_path.display()
        ))
    })?;
    std::fs::rename(&tmp_path, path).map_err(|e| {
        let _ = std::fs::remove_file(&tmp_path);
        OpenRagError::Internal(format!(
            "failed renaming temp pipeline YAML to '{}': {e}",
            path.display()
        ))
    })?;
    Ok(())
}

#[derive(Clone)]
pub struct PipelinePromoter<S: EvalStore> {
    store: Arc<S>,
}

impl<S: EvalStore> PipelinePromoter<S> {
    pub fn new(store: Arc<S>) -> Self {
        Self { store }
    }

    pub async fn register_config(
        &self,
        org_id: Uuid,
        name: &str,
        pipeline_yaml: &str,
        created_by: Option<Uuid>,
    ) -> Result<PipelineConfigRecord, OpenRagError> {
        let (parsed, profile) = parse_pipeline_yaml_with_profile(pipeline_yaml)?;
        let config_json = serde_json::to_value(&parsed)
            .map_err(|e| OpenRagError::Internal(format!("serialize pipeline config JSON: {e}")))?;
        self.store
            .register_pipeline_config(
                org_id,
                &RegisterPipelineInput {
                    name: name.to_string(),
                    profile,
                    pipeline_yaml: pipeline_yaml.to_string(),
                    config_json,
                    created_by,
                },
            )
            .await
    }

    pub async fn promote_config(
        &self,
        org_id: Uuid,
        actor: &PrincipalContext,
        input: PromotePipelineInput,
    ) -> Result<PipelineConfigRecord, OpenRagError> {
        if actor.org_id != org_id || !actor.is_org_admin {
            return Err(OpenRagError::Forbidden(
                "org_admin privilege required to promote pipeline config".into(),
            ));
        }

        let config_id = input.pipeline_config_id.ok_or_else(|| {
            OpenRagError::Validation("pipeline_config_id is required".into())
        })?;
        let record = self.store.get_pipeline_config(org_id, config_id).await?;

        verify_pipeline_record_safety(&record)?;

        if let Some(run_id) = input.bench_run_id {
            let run = self.store.get_bench_run(org_id, run_id).await?;
            if let Some(pid) = run.pipeline_config_id {
                if pid != config_id {
                    return Err(OpenRagError::Validation(format!(
                        "bench run {run_id} belongs to pipeline config {pid}, not {config_id}"
                    )));
                }
            }
            if !run.security_gate_pass || run.pii_leak_count > 0 || run.arn_violation_count > 0 {
                return Err(OpenRagError::Forbidden(format!(
                    "cannot promote pipeline config: bench run {run_id} failed security gate (pii_leak_count={}, arn_violation_count={})",
                    run.pii_leak_count, run.arn_violation_count
                )));
            }
        }

        let latest_runs = self
            .store
            .latest_bench_runs_by_track_for_config(org_id, config_id)
            .await?;
        for latest_run in latest_runs {
            if !latest_run.security_gate_pass
                || latest_run.pii_leak_count > 0
                || latest_run.arn_violation_count > 0
            {
                return Err(OpenRagError::Forbidden(format!(
                    "cannot promote pipeline config: latest {} bench run {} failed security gate (pii_leak_count={}, arn_violation_count={})",
                    latest_run.track,
                    latest_run.id,
                    latest_run.pii_leak_count,
                    latest_run.arn_violation_count
                )));
            }
        }

        self.store
            .mark_pipeline_promoted(org_id, config_id, input.bucket_ref.as_deref())
            .await
    }

    pub async fn promote_from_yaml(
        &self,
        org_id: Uuid,
        actor: &PrincipalContext,
        name: &str,
        pipeline_yaml: &str,
        bucket_ref: Option<&str>,
    ) -> Result<PipelineConfigRecord, OpenRagError> {
        if actor.org_id != org_id || !actor.is_org_admin {
            return Err(OpenRagError::Forbidden(
                "org_admin privilege required to promote pipeline config".into(),
            ));
        }
        let (parsed, profile) = parse_pipeline_yaml_with_profile(pipeline_yaml)?;
        if !parsed.pii.capsule_enabled {
            return Err(OpenRagError::Forbidden(
                "cannot promote pipeline config with pii.capsule_enabled = false".into(),
            ));
        }
        let config_json = serde_json::to_value(&parsed)
            .map_err(|e| OpenRagError::Internal(format!("serialize pipeline config JSON: {e}")))?;

        let record = self
            .store
            .register_pipeline_config(
                org_id,
                &RegisterPipelineInput {
                    name: name.to_string(),
                    profile,
                    pipeline_yaml: pipeline_yaml.to_string(),
                    config_json,
                    created_by: Some(actor.principal_id),
                },
            )
            .await?;

        self.store
            .mark_pipeline_promoted(org_id, record.id, bucket_ref)
            .await
    }
}

fn verify_pipeline_record_safety(record: &PipelineConfigRecord) -> Result<(), OpenRagError> {
    if record
        .config_json
        .pointer("/pii/capsule_enabled")
        .and_then(|v| v.as_bool())
        == Some(false)
    {
        return Err(OpenRagError::Forbidden(
            "cannot promote pipeline config with pii.capsule_enabled = false".into(),
        ));
    }

    if let Some(dim) = record
        .config_json
        .pointer("/store/dim")
        .and_then(|v| v.as_u64())
    {
        if dim != 256 {
            return Err(OpenRagError::Validation(format!(
                "cannot promote pipeline config with store.dim != 256 (got {dim})"
            )));
        }
    }

    if let Ok(raw_yaml) = serde_yaml_ng::from_str::<serde_yaml_ng::Value>(&record.pipeline_yaml) {
        if let Some(pii_map) = raw_yaml.get("pii").and_then(|v| v.as_mapping()) {
            if pii_map
                .get(serde_yaml_ng::Value::String("capsule_enabled".into()))
                .and_then(|v| v.as_bool())
                == Some(false)
            {
                return Err(OpenRagError::Forbidden(
                    "cannot promote pipeline config with pii.capsule_enabled = false".into(),
                ));
            }
        }
        if let Some(store_map) = raw_yaml.get("store").and_then(|v| v.as_mapping()) {
            if let Some(dim) = store_map
                .get(serde_yaml_ng::Value::String("dim".into()))
                .and_then(|v| v.as_u64())
            {
                if dim != 256 {
                    return Err(OpenRagError::Validation(format!(
                        "cannot promote pipeline config with store.dim != 256 (got {dim})"
                    )));
                }
            }
        }
    }

    Ok(())
}
