use std::collections::{BTreeMap, HashMap};
use std::fmt;
use std::str::FromStr;

use chrono::{DateTime, Utc};
use openrag_core::{BBoxNorm1000, FusionMode, OpenRagError, RetrieveConfig};
use openrag_store::{Bm25Hit, ExpandedTagFilter, TagFilter, VectorHit, VectorTable};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

pub const MAX_TOP_K: usize = 200;
pub const DEFAULT_FACET_LIMIT: i64 = 50;

const fn default_facet_limit() -> i64 {
    DEFAULT_FACET_LIMIT
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Ord, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RetrieveSpace {
    Chunks,
    Thumb,
    Wiki,
    Memory,
}

impl RetrieveSpace {
    pub const ALL: [Self; 4] = [Self::Chunks, Self::Thumb, Self::Wiki, Self::Memory];

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Chunks => "chunks",
            Self::Thumb => "thumb",
            Self::Wiki => "wiki",
            Self::Memory => "memory",
        }
    }

    pub const fn vector_table(self) -> VectorTable {
        match self {
            Self::Chunks => VectorTable::CorpusChunks,
            Self::Thumb => VectorTable::CorpusPages,
            Self::Wiki => VectorTable::WikiPages,
            Self::Memory => VectorTable::MemoryEntries,
        }
    }
}

impl fmt::Display for RetrieveSpace {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for RetrieveSpace {
    type Err = OpenRagError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "chunks" => Ok(Self::Chunks),
            "thumb" => Ok(Self::Thumb),
            "wiki" => Ok(Self::Wiki),
            "memory" => Ok(Self::Memory),
            other => Err(OpenRagError::Validation(format!(
                "unknown retrieval space '{other}'; expected one of: chunks, thumb, wiki, memory"
            ))),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceAdmission {
    Admit,
    Abstain,
}

pub type AdmissionVerdict = EvidenceAdmission;

impl EvidenceAdmission {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Admit => "admit",
            Self::Abstain => "abstain",
        }
    }
}

impl fmt::Display for EvidenceAdmission {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct ScoreBreakdown {
    pub vector_score: f32,
    pub bm25_score: f32,
    pub rrf_score: f32,
    pub convex_score: f32,
    pub revision_boost: f32,
    pub rerank_score: f32,
    pub fused_score: f32,
}

/// Validity of a hit at the request's reference time (`as_of`, default: now).
/// `Current` is only claimed when nothing says otherwise: not yet effective, expired,
/// superseded and older-revision hits are labeled as such.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ValidityState {
    #[default]
    Current,
    /// `valid_from` is after the reference time (not yet in force).
    Scheduled,
    /// `valid_to` is before the reference time.
    Expired,
    /// Explicitly superseded (wiki/memory supersede links reflect the present state).
    Superseded,
    /// A newer valid revision with the same title exists in the same space group.
    OlderRevision,
}

impl ValidityState {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Current => "current",
            Self::Scheduled => "scheduled",
            Self::Expired => "expired",
            Self::Superseded => "superseded",
            Self::OlderRevision => "older_revision",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SearchHit {
    pub id: Uuid,
    pub document_id: Option<Uuid>,
    pub page_id: Option<Uuid>,
    pub space: RetrieveSpace,
    pub title: Option<String>,
    pub snippet: String,
    pub text: String,
    pub score: f32,
    pub vector_score: f32,
    pub bm25_score: f32,
    pub score_breakdown: ScoreBreakdown,
    #[serde(default)]
    pub tags: Vec<String>,
    pub min_auth_level: Option<i16>,
    pub revision: Option<i32>,
    pub valid_from: Option<DateTime<Utc>>,
    pub valid_to: Option<DateTime<Utc>>,
    pub thumb_uri: Option<String>,
    pub bbox_norm_1000: Option<BBoxNorm1000>,
    pub acl_scope_key: String,
    /// Validity at the request's reference time.
    #[serde(default)]
    pub validity: ValidityState,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SearchRequest {
    pub query: String,
    #[serde(default)]
    pub query_vector: Option<Vec<f32>>,
    #[serde(default)]
    pub bucket_id: Option<Uuid>,
    #[serde(default)]
    pub spaces: Option<Vec<RetrieveSpace>>,
    #[serde(default)]
    pub fusion: Option<FusionMode>,
    #[serde(default)]
    pub rrf_k: Option<u32>,
    #[serde(default)]
    pub alpha: Option<f32>,
    #[serde(default)]
    pub top_k: Option<usize>,
    #[serde(default)]
    pub rerank: Option<bool>,
    #[serde(default)]
    pub latest_revision_boost: Option<f32>,
    #[serde(default)]
    pub latest_revision_only: bool,
    #[serde(default)]
    pub admission_threshold: Option<f32>,
    #[serde(default)]
    pub tag_filter: Option<TagFilter>,
    #[serde(default = "default_facet_limit")]
    pub facet_limit: i64,
    /// Reference time for validity (`valid_from`/`valid_to`, revisions). `None` = now.
    /// With `latest_revision_only`, only what was in force at this instant survives.
    #[serde(default)]
    pub as_of: Option<DateTime<Utc>>,
}

impl Default for SearchRequest {
    fn default() -> Self {
        Self {
            query: String::new(),
            query_vector: None,
            bucket_id: None,
            spaces: None,
            fusion: None,
            rrf_k: None,
            alpha: None,
            top_k: None,
            rerank: None,
            latest_revision_boost: None,
            latest_revision_only: false,
            admission_threshold: None,
            tag_filter: None,
            facet_limit: DEFAULT_FACET_LIMIT,
            as_of: None,
        }
    }
}

impl SearchRequest {
    pub fn new(query: impl Into<String>) -> Self {
        Self {
            query: query.into(),
            ..Self::default()
        }
    }

    pub fn from_retrieve_config(
        query: impl Into<String>,
        cfg: &RetrieveConfig,
    ) -> Result<Self, OpenRagError> {
        let mut spaces = Vec::with_capacity(cfg.spaces.len());
        for raw in &cfg.spaces {
            let parsed: RetrieveSpace = raw.parse()?;
            if !spaces.contains(&parsed) {
                spaces.push(parsed);
            }
        }
        Ok(Self {
            query: query.into(),
            spaces: Some(spaces),
            fusion: Some(cfg.fusion),
            rrf_k: Some(cfg.rrf_k),
            alpha: Some(cfg.alpha),
            top_k: Some(cfg.top_k),
            rerank: Some(cfg.rerank),
            latest_revision_boost: Some(cfg.latest_revision_boost),
            admission_threshold: Some(cfg.admission_threshold),
            ..Self::default()
        })
    }
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct RetrieveOverrides {
    #[serde(default)]
    pub spaces: Option<Vec<String>>,
    #[serde(default)]
    pub fusion: Option<FusionMode>,
    #[serde(default)]
    pub rrf_k: Option<u32>,
    #[serde(default)]
    pub alpha: Option<f32>,
    #[serde(default)]
    pub top_k: Option<usize>,
    #[serde(default)]
    pub rerank: Option<bool>,
    #[serde(default)]
    pub latest_revision_boost: Option<f32>,
    #[serde(default)]
    pub admission_threshold: Option<f32>,
}

impl RetrieveOverrides {
    pub fn apply_to_config(&self, base: &RetrieveConfig) -> Result<RetrieveConfig, OpenRagError> {
        let mut merged = base.clone();
        if let Some(spaces) = &self.spaces {
            merged.spaces = spaces.clone();
        }
        if let Some(fusion) = self.fusion {
            merged.fusion = fusion;
        }
        if let Some(rrf_k) = self.rrf_k {
            merged.rrf_k = rrf_k;
        }
        if let Some(alpha) = self.alpha {
            merged.alpha = alpha;
        }
        if let Some(top_k) = self.top_k {
            merged.top_k = top_k;
        }
        if let Some(rerank) = self.rerank {
            merged.rerank = rerank;
        }
        if let Some(boost) = self.latest_revision_boost {
            merged.latest_revision_boost = boost;
        }
        if let Some(thresh) = self.admission_threshold {
            merged.admission_threshold = thresh;
        }
        validate_resolved_values(
            &merged.spaces,
            merged.rrf_k,
            merged.alpha,
            merged.top_k,
            merged.latest_revision_boost,
            merged.admission_threshold,
        )?;
        Ok(merged)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResolvedRetrieveParams {
    pub spaces: Vec<RetrieveSpace>,
    pub fusion: FusionMode,
    pub rrf_k: u32,
    pub alpha: f32,
    pub top_k: usize,
    pub rerank: bool,
    pub latest_revision_boost: f32,
    pub latest_revision_only: bool,
    pub admission_threshold: f32,
}

pub(crate) fn validate_resolved_values(
    space_strs: &[String],
    rrf_k: u32,
    alpha: f32,
    top_k: usize,
    latest_revision_boost: f32,
    admission_threshold: f32,
) -> Result<Vec<RetrieveSpace>, OpenRagError> {
    if space_strs.is_empty() {
        return Err(OpenRagError::Validation("spaces must not be empty".into()));
    }
    let mut spaces = Vec::with_capacity(space_strs.len());
    for raw in space_strs {
        let s: RetrieveSpace = raw.parse()?;
        if !spaces.contains(&s) {
            spaces.push(s);
        }
    }
    validate_numeric_params(
        rrf_k,
        alpha,
        top_k,
        latest_revision_boost,
        admission_threshold,
    )?;
    Ok(spaces)
}

pub(crate) fn validate_numeric_params(
    rrf_k: u32,
    alpha: f32,
    top_k: usize,
    latest_revision_boost: f32,
    admission_threshold: f32,
) -> Result<(), OpenRagError> {
    if rrf_k == 0 {
        return Err(OpenRagError::Validation("rrf_k must be >= 1".into()));
    }
    if !alpha.is_finite() || !(0.0..=1.0).contains(&alpha) {
        return Err(OpenRagError::Validation(format!(
            "alpha must be finite and within 0.0..=1.0, got {alpha}"
        )));
    }
    if top_k == 0 || top_k > MAX_TOP_K {
        return Err(OpenRagError::Validation(format!(
            "top_k must be within 1..={MAX_TOP_K}, got {top_k}"
        )));
    }
    if !latest_revision_boost.is_finite() || latest_revision_boost < 0.0 {
        return Err(OpenRagError::Validation(format!(
            "latest_revision_boost must be finite and >= 0.0, got {latest_revision_boost}"
        )));
    }
    if !admission_threshold.is_finite() || !(0.0..=1.0).contains(&admission_threshold) {
        return Err(OpenRagError::Validation(format!(
            "admission_threshold must be finite and within 0.0..=1.0, got {admission_threshold}"
        )));
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SearchResponse {
    pub hits: Vec<SearchHit>,
    pub clean_query: String,
    #[serde(default)]
    pub parsed_inline_tags: Vec<String>,
    #[serde(default)]
    pub parsed_excluded_tags: Vec<String>,
    #[serde(default)]
    pub expanded_terms: Vec<String>,
    #[serde(default)]
    pub tag_facets: BTreeMap<String, i64>,
    pub admission: EvidenceAdmission,
    pub max_score: f32,
    pub top_score: f32,
    pub top_auth_level: Option<i16>,
}

impl SearchResponse {
    pub fn empty(
        clean_query: String,
        parsed_inline_tags: Vec<String>,
        parsed_excluded_tags: Vec<String>,
        expanded_terms: Vec<String>,
    ) -> Self {
        Self {
            hits: Vec::new(),
            clean_query,
            parsed_inline_tags,
            parsed_excluded_tags,
            expanded_terms,
            tag_facets: BTreeMap::new(),
            admission: EvidenceAdmission::Abstain,
            max_score: 0.0,
            top_score: 0.0,
            top_auth_level: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct HydratedCandidate {
    pub id: Uuid,
    pub document_id: Option<Uuid>,
    pub page_id: Option<Uuid>,
    pub space: RetrieveSpace,
    pub title: Option<String>,
    pub text: String,
    pub tags: Vec<String>,
    pub min_auth_level: Option<i16>,
    pub revision: Option<i32>,
    pub valid_from: Option<DateTime<Utc>>,
    pub valid_to: Option<DateTime<Utc>>,
    pub is_archived: bool,
    pub is_superseded: bool,
    pub thumb_uri: Option<String>,
    pub bbox_norm_1000: Option<BBoxNorm1000>,
    pub acl_scope_key: String,
    pub embedding: Option<Vec<f32>>,
}

#[derive(Debug, Clone, Default)]
pub struct BackendExecutionInput {
    pub clean_query: String,
    pub explicit_query_vector: Option<Vec<f32>>,
    pub bucket_id: Option<Uuid>,
    pub spaces: Vec<RetrieveSpace>,
    pub effective_tag_filter: TagFilter,
    pub candidate_limit_per_space: usize,
    pub include_embeddings: bool,
    pub tag_only_query: bool,
    /// Skip the BM25 leg of spaces with vectors: set when fusion is convex with alpha 0, where BM25
    /// cannot change their scores. Spaces without vectors (Thumb) always keep it.
    pub skip_lexical: bool,
}

#[derive(Debug, Clone)]
pub struct TagOnlyHit {
    pub space: RetrieveSpace,
    pub id: Uuid,
    pub document_id: Option<Uuid>,
    pub acl_scope_key: String,
    pub tags: Vec<String>,
}

#[derive(Debug, Clone, Default)]
pub struct BackendExecutionOutput {
    pub vector_hits: Vec<(RetrieveSpace, VectorHit)>,
    pub bm25_hits: Vec<(RetrieveSpace, Bm25Hit)>,
    pub tag_only_hits: Vec<TagOnlyHit>,
    pub expanded_terms: Vec<String>,
    pub expanded_tag_filter: Option<ExpandedTagFilter>,
    pub query_vector: Vec<f32>,
    pub hydrated: HashMap<(RetrieveSpace, Uuid), HydratedCandidate>,
}

#[derive(Debug, Clone)]
pub struct FusionContext {
    pub clean_query: String,
    pub parsed_inline_tags: Vec<String>,
    pub parsed_excluded_tags: Vec<String>,
    pub tag_only_query: bool,
    pub facet_limit: i64,
    pub now: DateTime<Utc>,
}
