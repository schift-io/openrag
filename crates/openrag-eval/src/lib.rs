#![deny(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::todo,
    clippy::dbg_macro
)]

pub mod ab_search;
pub mod bench;
pub mod golden;
pub mod metrics;
pub mod optimizer;
pub mod types;

pub use ab_search::AbSearchEvaluator;
pub use bench::{
    apply_egress_capsule, pararel_edge_overlap_score, token_overlap_score, BenchRunner,
};
pub use golden::{
    CompleteOptimizationTrial, EvalStore, FeedbackManager, FeedbackPromoter, GoldenSetManager,
    InMemoryEvalStore, NewBenchItemResult, NewBenchRun, NewOptimizationTrial, PgEvalStore,
};
pub use metrics::{
    check_arn_violation, check_pii_leak, dedup_doc_ids_in_order, extract_hit_doc_id, mrr,
    ndcg_at_k, recall_at_k,
};
pub use optimizer::{
    default_pipeline_config, normalize_search_space_node_key, parse_pipeline_yaml_tolerant,
    parse_pipeline_yaml_with_profile, parse_search_space_yaml, AutoRagOptimizer,
    NormalizedSearchSpace, PipelinePromoter,
};
pub use types::{
    AbQueryInput, AbQueryItem, AbSearchReport, AbSearchRequest, AbVariantConfig, AbVariantMetrics,
    BenchItemResultRecord, BenchMetrics, BenchReport, BenchRunRecord, BenchRunRequest,
    BenchRunSummary, BenchTrack, CandidateEvaluation, CreateGoldenItemInput, CreateGoldenSetInput,
    FeedbackVerdict, GoldenAxis, GoldenItem, GoldenSet, LeaderboardEntry, OptimizationStatus,
    OptimizationTrialRecord, OptimizeReport, OptimizeRequest, PipelineConfigRecord,
    PipelineProfileKind, PromoteFeedbackInput, PromoteFeedbackResult, PromotePipelineInput,
    RegisterPipelineInput, ReviewStatus, SearchFeedback, SecurityGateResult, SubmitFeedbackInput,
};
