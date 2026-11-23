use std::fmt;
use std::str::FromStr;

use chrono::{DateTime, Utc};
use openrag_core::{AccessScope, AuthLevel, OpenRagError};
use openrag_store::{TagFacet, TagFilter};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use uuid::Uuid;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryKind {
    Fact,
    Decision,
    Procedure,
    Preference,
    Correction,
    DailyLog,
}

impl MemoryKind {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Fact => "fact",
            Self::Decision => "decision",
            Self::Procedure => "procedure",
            Self::Preference => "preference",
            Self::Correction => "correction",
            Self::DailyLog => "daily_log",
        }
    }
}

impl fmt::Display for MemoryKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for MemoryKind {
    type Err = OpenRagError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim() {
            "fact" => Ok(Self::Fact),
            "decision" => Ok(Self::Decision),
            "procedure" => Ok(Self::Procedure),
            "preference" => Ok(Self::Preference),
            "correction" => Ok(Self::Correction),
            "daily_log" => Ok(Self::DailyLog),
            other => Err(OpenRagError::Validation(format!(
                "invalid memory_kind '{other}'"
            ))),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryStatus {
    Active,
    Superseded,
    Retracted,
    Redacted,
    Archived,
}

impl MemoryStatus {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Active => "active",
            Self::Superseded => "superseded",
            Self::Retracted => "retracted",
            Self::Redacted => "redacted",
            Self::Archived => "archived",
        }
    }
}

impl FromStr for MemoryStatus {
    type Err = OpenRagError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim() {
            "active" => Ok(Self::Active),
            "superseded" => Ok(Self::Superseded),
            "retracted" => Ok(Self::Retracted),
            "redacted" => Ok(Self::Redacted),
            "archived" => Ok(Self::Archived),
            other => Err(OpenRagError::Validation(format!(
                "invalid memory status '{other}'"
            ))),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PatchOp {
    Amend,
    Supersede,
    Retract,
    Redact,
    PromoteScope,
    PromoteWiki,
}

impl PatchOp {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Amend => "amend",
            Self::Supersede => "supersede",
            Self::Retract => "retract",
            Self::Redact => "redact",
            Self::PromoteScope => "promote_scope",
            Self::PromoteWiki => "promote_wiki",
        }
    }
}

impl FromStr for PatchOp {
    type Err = OpenRagError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim() {
            "amend" => Ok(Self::Amend),
            "supersede" => Ok(Self::Supersede),
            "retract" => Ok(Self::Retract),
            "redact" => Ok(Self::Redact),
            "promote_scope" => Ok(Self::PromoteScope),
            "promote_wiki" => Ok(Self::PromoteWiki),
            other => Err(OpenRagError::Validation(format!(
                "invalid patch_op '{other}'"
            ))),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ActorType {
    Human,
    Agent,
    Redactor,
    Compiler,
}

impl ActorType {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Human => "human",
            Self::Agent => "agent",
            Self::Redactor => "redactor",
            Self::Compiler => "compiler",
        }
    }
}

impl FromStr for ActorType {
    type Err = OpenRagError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim() {
            "human" => Ok(Self::Human),
            "agent" => Ok(Self::Agent),
            "redactor" => Ok(Self::Redactor),
            "compiler" => Ok(Self::Compiler),
            other => Err(OpenRagError::Validation(format!(
                "invalid actor_type '{other}'"
            ))),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SourceHarness {
    ChatUi,
    ClaudeCode,
    Codex,
    Cursor,
    Kordoc,
    Docufinder,
}

impl SourceHarness {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ChatUi => "chat_ui",
            Self::ClaudeCode => "claude_code",
            Self::Codex => "codex",
            Self::Cursor => "cursor",
            Self::Kordoc => "kordoc",
            Self::Docufinder => "docufinder",
        }
    }
}

impl FromStr for SourceHarness {
    type Err = OpenRagError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim() {
            "chat_ui" => Ok(Self::ChatUi),
            "claude_code" => Ok(Self::ClaudeCode),
            "codex" => Ok(Self::Codex),
            "cursor" => Ok(Self::Cursor),
            "kordoc" => Ok(Self::Kordoc),
            "docufinder" => Ok(Self::Docufinder),
            other => Err(OpenRagError::Validation(format!(
                "invalid source_harness '{other}'"
            ))),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryEdgeType {
    DerivedFrom,
    Supports,
    Contradicts,
    Supersedes,
}

impl MemoryEdgeType {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::DerivedFrom => "derived_from",
            Self::Supports => "supports",
            Self::Contradicts => "contradicts",
            Self::Supersedes => "supersedes",
        }
    }
}

impl fmt::Display for MemoryEdgeType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for MemoryEdgeType {
    type Err = OpenRagError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim() {
            "derived_from" => Ok(Self::DerivedFrom),
            "supports" => Ok(Self::Supports),
            "contradicts" => Ok(Self::Contradicts),
            "supersedes" => Ok(Self::Supersedes),
            other => Err(OpenRagError::Validation(format!(
                "invalid memory edge_type '{other}'; expected derived_from|supports|contradicts|supersedes"
            ))),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WikiRelationType {
    Equivalent,
    Entails,
    Contradicts,
    Supersedes,
    Refines,
    Related,
    References,
    ParentOf,
}

impl WikiRelationType {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Equivalent => "equivalent",
            Self::Entails => "entails",
            Self::Contradicts => "contradicts",
            Self::Supersedes => "supersedes",
            Self::Refines => "refines",
            Self::Related => "related",
            Self::References => "references",
            Self::ParentOf => "parent_of",
        }
    }
}

impl FromStr for WikiRelationType {
    type Err = OpenRagError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim() {
            "equivalent" => Ok(Self::Equivalent),
            "entails" => Ok(Self::Entails),
            "contradicts" => Ok(Self::Contradicts),
            "supersedes" => Ok(Self::Supersedes),
            "refines" => Ok(Self::Refines),
            "related" => Ok(Self::Related),
            "references" => Ok(Self::References),
            "parent_of" => Ok(Self::ParentOf),
            other => Err(OpenRagError::Validation(format!(
                "invalid wiki relation_type '{other}'"
            ))),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WikiStatus {
    Active,
    Superseded,
    Archived,
    Redacted,
}

impl WikiStatus {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Active => "active",
            Self::Superseded => "superseded",
            Self::Archived => "archived",
            Self::Redacted => "redacted",
        }
    }
}

impl FromStr for WikiStatus {
    type Err = OpenRagError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim() {
            "active" => Ok(Self::Active),
            "superseded" => Ok(Self::Superseded),
            "archived" => Ok(Self::Archived),
            "redacted" => Ok(Self::Redacted),
            other => Err(OpenRagError::Validation(format!(
                "invalid wiki status '{other}'"
            ))),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WikiEditorType {
    Human,
    Agent,
    Compiler,
    Redactor,
    Sync,
}

impl WikiEditorType {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Human => "human",
            Self::Agent => "agent",
            Self::Compiler => "compiler",
            Self::Redactor => "redactor",
            Self::Sync => "sync",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WikiSourceType {
    Chunk,
    Memory,
    Document,
}

impl WikiSourceType {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Chunk => "chunk",
            Self::Memory => "memory",
            Self::Document => "document",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConceptCategory {
    #[default]
    General,
    Domain,
    Entity,
    Acronym,
    SafewordAlias,
}

impl ConceptCategory {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::General => "general",
            Self::Domain => "domain",
            Self::Entity => "entity",
            Self::Acronym => "acronym",
            Self::SafewordAlias => "safeword_alias",
        }
    }
}

impl FromStr for ConceptCategory {
    type Err = OpenRagError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim() {
            "general" => Ok(Self::General),
            "domain" => Ok(Self::Domain),
            "entity" => Ok(Self::Entity),
            "acronym" => Ok(Self::Acronym),
            "safeword_alias" => Ok(Self::SafewordAlias),
            other => Err(OpenRagError::Validation(format!(
                "invalid concept category '{other}'"
            ))),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConceptSource {
    #[default]
    Manual,
    IngestParenthetical,
    WikiFrontmatter,
    SemanticAuto,
    Seed,
}

impl ConceptSource {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Manual => "manual",
            Self::IngestParenthetical => "ingest_parenthetical",
            Self::WikiFrontmatter => "wiki_frontmatter",
            Self::SemanticAuto => "semantic_auto",
            Self::Seed => "seed",
        }
    }
}

impl FromStr for ConceptSource {
    type Err = OpenRagError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim() {
            "manual" => Ok(Self::Manual),
            "ingest_parenthetical" => Ok(Self::IngestParenthetical),
            "wiki_frontmatter" => Ok(Self::WikiFrontmatter),
            "semantic_auto" => Ok(Self::SemanticAuto),
            "seed" => Ok(Self::Seed),
            other => Err(OpenRagError::Validation(format!(
                "invalid concept source '{other}'"
            ))),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MemoryEpisode {
    pub id: Uuid,
    pub org_id: Uuid,
    pub dept_id: Uuid,
    pub owner_principal_id: Uuid,
    pub access_scope: AccessScope,
    pub allowed_group_ids: Vec<Uuid>,
    pub denied_group_ids: Vec<Uuid>,
    pub min_auth_level: AuthLevel,
    pub arn: String,
    pub source_harness: SourceHarness,
    pub session_key: Option<String>,
    pub summary_md: String,
    pub raw_redacted_json: Option<Value>,
    pub content_sha256: String,
    pub redaction_applied: bool,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MemoryEntry {
    pub id: Uuid,
    pub org_id: Uuid,
    pub bucket_id: Option<Uuid>,
    pub dept_id: Uuid,
    pub owner_principal_id: Uuid,
    pub access_scope: AccessScope,
    pub allowed_group_ids: Vec<Uuid>,
    pub denied_group_ids: Vec<Uuid>,
    pub min_auth_level: AuthLevel,
    pub arn: String,
    pub acl_scope_key: String,
    pub memory_kind: MemoryKind,
    pub title: Option<String>,
    pub effective_text: String,
    pub bm25_text: String,
    pub content_sha256: String,
    pub tags: Vec<String>,
    pub source_episode_id: Option<Uuid>,
    pub source_document_id: Option<Uuid>,
    pub promoted_wiki_id: Option<Uuid>,
    pub confidence: f32,
    pub status: MemoryStatus,
    pub superseded_by: Option<Uuid>,
    pub version: i32,
    pub valid_from: DateTime<Utc>,
    pub valid_to: Option<DateTime<Utc>>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemoryPatch {
    pub id: Uuid,
    pub org_id: Uuid,
    pub memory_entry_id: Uuid,
    pub patch_op: PatchOp,
    pub prev_version: i32,
    pub new_version: i32,
    pub prev_text: Option<String>,
    pub new_text: Option<String>,
    pub prev_scope: Option<String>,
    pub new_scope: Option<String>,
    pub reason: Option<String>,
    pub actor_type: ActorType,
    pub actor_id: String,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MemoryEdge {
    pub id: Uuid,
    pub org_id: Uuid,
    pub src_memory_id: Uuid,
    pub dst_memory_id: Uuid,
    pub relation_type: MemoryEdgeType,
    pub confidence: f32,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RecordEpisodeInput {
    pub dept_id: Uuid,
    pub access_scope: AccessScope,
    #[serde(default)]
    pub allowed_group_ids: Vec<Uuid>,
    #[serde(default)]
    pub denied_group_ids: Vec<Uuid>,
    pub min_auth_level: AuthLevel,
    pub org_slug: String,
    pub dept_path: String,
    pub source_harness: SourceHarness,
    pub session_key: Option<String>,
    pub summary_md: String,
    pub raw_redacted_json: Option<Value>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UpsertMemoryInput {
    pub bucket_id: Option<Uuid>,
    pub dept_id: Uuid,
    pub access_scope: AccessScope,
    #[serde(default)]
    pub allowed_group_ids: Vec<Uuid>,
    #[serde(default)]
    pub denied_group_ids: Vec<Uuid>,
    pub min_auth_level: AuthLevel,
    pub org_slug: String,
    pub dept_path: String,
    pub memory_kind: MemoryKind,
    pub title: Option<String>,
    pub text: String,
    #[serde(default)]
    pub tags: Vec<String>,
    pub source_episode_id: Option<Uuid>,
    pub source_document_id: Option<Uuid>,
    pub confidence: Option<f32>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ApplyPatchInput {
    pub memory_entry_id: Uuid,
    pub patch_op: PatchOp,
    pub new_text: Option<String>,
    pub reason: Option<String>,
    pub actor_type: ActorType,
    pub hard_purge: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PromoteTarget {
    Private,
    Dept,
    Group,
    Org,
    Wiki,
}

impl FromStr for PromoteTarget {
    type Err = OpenRagError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim() {
            "private" => Ok(Self::Private),
            "dept" => Ok(Self::Dept),
            "group" => Ok(Self::Group),
            "org" => Ok(Self::Org),
            "wiki" => Ok(Self::Wiki),
            other => Err(OpenRagError::Validation(format!(
                "invalid promote target '{other}'; expected private|dept|group|org|wiki"
            ))),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PromoteMemoryInput {
    pub memory_entry_id: Uuid,
    pub target: PromoteTarget,
    pub target_bucket_id: Option<Uuid>,
    #[serde(default)]
    pub target_group_ids: Vec<Uuid>,
    pub min_auth_level: Option<AuthLevel>,
    pub wiki_slug: Option<String>,
    pub wiki_title: Option<String>,
    pub org_slug: String,
    pub dept_path: String,
    pub reason: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PromoteResult {
    pub entry: MemoryEntry,
    pub patch: MemoryPatch,
    pub promoted_wiki_id: Option<Uuid>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EngageRequest {
    pub query: String,
    pub scope: Option<AccessScope>,
    pub top_k: usize,
    #[serde(default)]
    pub tag_filter: Option<TagFilter>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EngageHit {
    pub entry: MemoryEntry,
    pub score: f32,
    pub edges: Vec<MemoryEdge>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EngageResponse {
    pub memories: Vec<EngageHit>,
    pub expanded_terms: Vec<String>,
    pub tag_facets: Vec<TagFacet>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReflectRequest {
    pub dept_id: Uuid,
    pub org_slug: String,
    pub dept_path: String,
    pub content: String,
    pub title: Option<String>,
    pub memory_kind: Option<MemoryKind>,
    pub episode_id: Option<Uuid>,
    pub access_scope: AccessScope,
    #[serde(default)]
    pub allowed_group_ids: Vec<Uuid>,
    #[serde(default)]
    pub denied_group_ids: Vec<Uuid>,
    pub min_auth_level: AuthLevel,
    #[serde(default)]
    pub tags: Vec<String>,
    pub edge_type: Option<MemoryEdgeType>,
    pub target_memory_id: Option<Uuid>,
    #[serde(default)]
    pub extract_concepts: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReflectResponse {
    pub entry: MemoryEntry,
    pub episode_id: Uuid,
    pub edge: Option<MemoryEdge>,
    pub extracted_concepts: Vec<(String, String)>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WikiPage {
    pub id: Uuid,
    pub org_id: Uuid,
    pub bucket_id: Option<Uuid>,
    pub dept_id: Uuid,
    pub owner_principal_id: Uuid,
    pub access_scope: AccessScope,
    pub allowed_group_ids: Vec<Uuid>,
    pub denied_group_ids: Vec<Uuid>,
    pub min_auth_level: AuthLevel,
    pub arn: String,
    pub acl_scope_key: String,
    pub slug: String,
    pub title: String,
    pub body_md: String,
    pub bm25_text: String,
    pub body_sha256: String,
    pub tags: Vec<String>,
    pub status: WikiStatus,
    pub version: i32,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WikiRevision {
    pub id: Uuid,
    pub org_id: Uuid,
    pub page_id: Uuid,
    pub version: i32,
    pub body_md: String,
    pub body_sha256: String,
    pub editor_type: WikiEditorType,
    pub editor_id: String,
    pub change_note: Option<String>,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WikiEdge {
    pub id: Uuid,
    pub org_id: Uuid,
    pub src_page_id: Uuid,
    pub dst_page_id: Uuid,
    pub relation_type: WikiRelationType,
    pub confidence: f32,
    pub resolved_at: Option<DateTime<Utc>>,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WikiPageSource {
    pub org_id: Uuid,
    pub page_id: Uuid,
    pub source_type: WikiSourceType,
    pub source_id: Uuid,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WikiSyncChange {
    pub id: Uuid,
    pub org_id: Uuid,
    pub bucket_id: Option<Uuid>,
    pub page_id: Option<Uuid>,
    pub file_path: String,
    pub direction: String,
    pub body_sha256: String,
    pub status: String,
    pub synced_at: DateTime<Utc>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WikiSearchHit {
    pub page: WikiPage,
    pub score: f32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SyncDocInput {
    pub file_path: String,
    pub slug: String,
    pub title: String,
    pub body_md: String,
    #[serde(default)]
    pub tags: Vec<String>,
    pub base_sha256: Option<String>,
    pub base_version: Option<i32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SyncDocOutcome {
    Created,
    Updated,
    SkippedNoop,
    Conflict,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SyncDocResult {
    pub slug: String,
    pub page_id: Option<Uuid>,
    pub outcome: SyncDocOutcome,
    pub body_sha256: String,
    pub version: Option<i32>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct WikiImportReport {
    pub imported: usize,
    pub updated: usize,
    pub skipped_noop: usize,
    pub conflicts: usize,
    pub edges_created: usize,
    pub results: Vec<SyncDocResult>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Concept {
    pub id: Uuid,
    pub org_id: Uuid,
    pub canonical_key: String,
    pub display_name: String,
    pub category: ConceptCategory,
    pub description: Option<String>,
    pub wiki_page_id: Option<Uuid>,
    pub created_by: Option<Uuid>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConceptAlias {
    pub id: Uuid,
    pub org_id: Uuid,
    pub concept_id: Uuid,
    pub alias_text: String,
    pub normalized_alias: String,
    pub compact_alias: String,
    pub lang: String,
    pub source: ConceptSource,
    pub confidence: f32,
    pub created_by: Option<Uuid>,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResolvedConcept {
    pub concept: Concept,
    pub aliases: Vec<ConceptAlias>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConceptSearchMatch {
    pub concept: Concept,
    pub cosine_sim: f32,
}
