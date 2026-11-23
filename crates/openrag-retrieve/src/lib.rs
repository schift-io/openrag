pub mod backend;
pub mod fusion;
pub mod query;
pub mod retriever;
pub mod types;

pub use backend::{InMemoryDocRecord, InMemoryRetrieveBackend, PgRetrieveBackend, RetrieveBackend};
pub use fusion::fuse_and_rank;
pub use query::{
    build_concept_lookup_terms, build_embed_query_text, has_conflicting_tags, merge_tag_filters,
    parse_spaces, query_has_searchable_tokens, resolve_retrieve_params,
    sanitize_and_parse_inline_tags,
};
pub use retriever::{
    default_retrieve_config, HybridRetriever, InMemoryHybridRetriever, PgHybridRetriever,
};
pub use types::{
    AdmissionVerdict, BackendExecutionInput, BackendExecutionOutput, EvidenceAdmission,
    FusionContext, HydratedCandidate, ResolvedRetrieveParams, RetrieveOverrides, RetrieveSpace,
    ScoreBreakdown, SearchHit, SearchRequest, SearchResponse, TagOnlyHit, ValidityState,
    DEFAULT_FACET_LIMIT, MAX_TOP_K,
};
