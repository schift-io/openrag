pub mod backend;
pub mod concepts;
pub mod embed;
pub mod guard_hook;
pub mod kumiho_backend;
pub mod pg_backend;
pub mod types;
pub mod wiki;

pub use backend::{MemoryBackend, MemoryRouter};
pub use concepts::{
    compact_term, extract_parenthetical_pairs, normalize_term, ConceptGraph,
    InMemoryConceptStore,
};
pub use embed::{DeterministicTextEmbedder, TextEmbedder};
pub use guard_hook::{DefaultMemoryGuard, MemoryGuardHook};
pub use kumiho_backend::KumihoMemoryBackend;
pub use pg_backend::PostgresMemoryBackend;
pub use types::{
    ActorType, ApplyPatchInput, Concept, ConceptAlias, ConceptCategory, ConceptSearchMatch,
    ConceptSource, EngageHit, EngageRequest, EngageResponse, MemoryEdge, MemoryEdgeType,
    MemoryEntry, MemoryEpisode, MemoryKind, MemoryPatch, MemoryStatus, PatchOp,
    PromoteMemoryInput, PromoteResult, PromoteTarget, RecordEpisodeInput, ReflectRequest,
    ReflectResponse, ResolvedConcept, SourceHarness, SyncDocInput, SyncDocOutcome, SyncDocResult,
    UpsertMemoryInput, WikiEdge, WikiEditorType, WikiImportReport, WikiPage, WikiPageSource,
    WikiRelationType, WikiRevision, WikiSearchHit, WikiSourceType, WikiStatus, WikiSyncChange,
};
pub use wiki::{
    extract_wikilinks, parse_frontmatter, render_frontmatter_markdown, MarkdownSync,
    ParsedFrontmatter, WikiCompiler,
};
