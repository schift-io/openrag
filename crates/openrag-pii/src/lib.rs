#![deny(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::todo,
    clippy::dbg_macro
)]

pub mod capsule;
pub mod category;
pub mod concepts;
pub mod detector;
pub mod guard;
pub mod redactor;
pub mod thumbnail;
pub mod vault;

pub use capsule::{Capsule, CapsuleIamCtx, InMemoryVaultStore, SealedText, SealedTokenRecord, UnmaskResult};
pub use category::PiiCategory;
pub use concepts::{
    cosine_similarity_128d, normalize_compact, normalize_nfkc_lower, ConceptAliasLookup,
    DeterministicSemanticEmbedder128, InMemoryConceptAliasLookup, IndexedText, SemanticEmbedder128,
};
pub use detector::{
    luhn_valid, rrn_mod11_checksum_valid, validate_korean_rrn_digits, DetectorProfile,
    DetectorProfileInput, PiiEngine, PiiProfileRepo, PiiSpan,
};
pub use guard::{GuardEngine, GuardPolicyRow, GuardPolicyRuntime, GuardRepo, GuardRuleRow, GuardRuleRuntime};
pub use redactor::{
    AffectedCounts, CascadeRedactor, RedactionJobRequest, RedactionJobResult, RedactionMode,
    RedactionTargetScope, RedactionTextEmbedder,
};
pub use thumbnail::{mask_image_bboxes, LocalThumbnailStore, ThumbnailStore};
pub use vault::{MasterKek, PiiVaultRepo, StoredChunkSpanInput, VaultCrypto, VaultEntryRow};
