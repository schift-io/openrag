pub mod bm25;
pub mod engine;
pub mod oss_vector;
pub mod outbox;
pub mod s3_vectors;
pub mod tags;
pub mod types;

pub use bm25::{tokenize_bm25, Bm25DocInput, Bm25Hit, Bm25Index, Bm25Query, Bm25Store, BM25_B, BM25_K1};
pub use engine::{IndexBinding, PgPineconeManager, RemoteVectorQuerier};
pub use oss_vector::{
    InMemoryOssVectorStore, OssBackendKind, OssHit, OssVectorAdapter, OssVectorBackend,
    OssVectorPoint, QdrantHttpBackend,
};
pub use outbox::{
    DeltaPlan, DrainReport, EnqueueOutboxItem, OutboxEnqueuer, OutboxExecOutcome, OutboxExecutor,
    OutboxOp, OutboxRow, OutboxStatus, OutboxStatusSummary, OutboxWorker, QuarantineRecord,
    WatermarkRecord, DefaultOutboxExecutor,
};
pub use s3_vectors::{
    build_sts_session_policy, decode_s3_metadata, encode_s3_metadata, s3_bucket_arn,
    s3_bucket_name, s3_filter_matches, s3_tenant_hash, sign_aws_sigv4, InMemoryS3VectorsStore,
    S3Filter, S3MetadataSchema, S3VectorPutItem, S3VectorQueryHit, S3VectorRecord,
    S3VectorsAdapter, S3VectorsBackend, S3VectorsHttpClient, ScopedS3Bucket, KEY_EMPTY, KEY_EXTRA,
    KEY_ID, KEY_TEXT, KEY_UPDATED_AT, MAX_EXTRA_BYTES, MAX_FILTERABLE_BYTES, MAX_LIST_ELEMENTS,
    MAX_METADATA_BYTES, MAX_METADATA_KEYS, MAX_METADATA_VALUE_BYTES, MAX_TEXT_BYTES,
    NON_FILTERABLE_KEYS,
};
pub use tags::{
    add_tags_rls, compute_tag_facets_rls, expand_tag_filter_in_memory, expand_tag_filter_rls,
    normalize_tag, parse_inline_tags, push_tag_filter_predicates, remove_tags_rls,
    resolve_tag_aliases_rls, ExpandedTagFilter, ParsedTagQuery, TagFacet, TagFilter, TagTable,
};
pub use types::{
    cosine_similarity, hmac_sha256, l2_normalize, sha256_hex, validate_vector, HttpRequest,
    HttpResponse, HttpTransport, StoreMode, TokioHttpTransport, VectorHit, VectorSearchQuery,
    VectorStoreAdapter, VectorTable, VectorUpsertItem, VECTOR_DIM,
};
