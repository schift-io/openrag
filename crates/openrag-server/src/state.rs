use std::collections::BTreeMap;
use std::path::Path;
use std::str::FromStr;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use openrag_core::{
    AuthLevel, ExternalMirrorBackend, FusionMode, GuardAction, GuardConfig, GuardDecision,
    GuardRuleConfig, GuardRuleKind, GuardSurface, LlmConfig, MemoryConfig, OcrModeRule,
    OpenRagError, OptimizationStrategy, ParseConfig, PgPineconeEngine, PiiConfig, PipelineConfig,
    PipelineProfile, RetrieveConfig, SearchSpaceConfig, SecurityGateConfig, Sensitivity,
    StoreConfig, ThumbnailConfig,
};
use openrag_db::DbPool;
use openrag_eval::{InMemoryEvalStore, PgEvalStore};
use openrag_memory::{
    DeterministicTextEmbedder, TextEmbedder, InMemoryConceptStore, KumihoMemoryBackend,
    MemoryBackend, MemoryGuardHook, MemoryRouter, PostgresMemoryBackend,
};
use openrag_parse::{
    DocumentParser, InMemoryBlobStore, LocalFsBlobStore, S3BlobConfig, S3BlobStore, SchiftOcrClient,
    ThumbnailBlobStore,
    Thumbnailer,
};
use openrag_pii::{
    GuardEngine, GuardPolicyRuntime, GuardRepo, GuardRuleRuntime, InMemoryConceptAliasLookup,
    InMemoryVaultStore, MasterKek, PiiEngine,
};
use crate::llm_router::{LlmHttpTransport, ReqwestLlmTransport};
use openrag_retrieve::{
    HybridRetriever, InMemoryHybridRetriever, InMemoryRetrieveBackend, PgHybridRetriever,
    PgRetrieveBackend,
};
use sha2::{Digest, Sha256};
use tokio::sync::RwLock;
use uuid::Uuid;


#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpenRagMode {
    Offline,
    Live,
    Auto,
}

impl OpenRagMode {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Offline => "offline",
            Self::Live => "live",
            Self::Auto => "auto",
        }
    }
}

impl FromStr for OpenRagMode {
    type Err = OpenRagError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "" | "auto" => Ok(Self::Auto),
            "offline" => Ok(Self::Offline),
            "live" => Ok(Self::Live),
            other => Err(OpenRagError::Config(format!(
                "invalid OPENRAG_MODE '{other}'; expected offline|live|auto"
            ))),
        }
    }
}

pub struct PiiMemoryGuardHook {
    worker_pool: Option<DbPool>,
    pipeline_config: Arc<RwLock<PipelineConfig>>,
    pii_engine: Arc<RwLock<PiiEngine>>,
    in_memory_concepts: InMemoryConceptStore,
}

impl PiiMemoryGuardHook {
    pub fn new(
        worker_pool: Option<DbPool>,
        pipeline_config: Arc<RwLock<PipelineConfig>>,
        pii_engine: Arc<RwLock<PiiEngine>>,
        in_memory_concepts: InMemoryConceptStore,
    ) -> Self {
        Self {
            worker_pool,
            pipeline_config,
            pii_engine,
            in_memory_concepts,
        }
    }
}

#[async_trait]
impl MemoryGuardHook for PiiMemoryGuardHook {
    async fn inspect(
        &self,
        org_id: Uuid,
        text: &str,
        surface: GuardSurface,
    ) -> Result<GuardDecision, OpenRagError> {
        let pipeline = self.pipeline_config.read().await.clone();
        let pii_engine = self.pii_engine.read().await.clone();
        let policy =
            load_active_guard_policy(self.worker_pool.as_ref(), org_id, &pipeline).await?;
        let lookup =
            load_concept_lookup(self.worker_pool.as_ref(), &self.in_memory_concepts, org_id)
                .await?;
        GuardEngine::enforce(text, surface, &policy, &pii_engine, Some(&lookup), None)
    }
}

pub struct AppStateParts {
    mode: OpenRagMode,
    app_pool: Option<DbPool>,
    worker_pool: Option<DbPool>,
    admin_pool: Option<DbPool>,
    default_org_id: Option<Uuid>,
    pipeline: PipelineConfig,
    search_space_config: SearchSpaceConfig,
    master_kek: MasterKek,
    blob_store: Arc<dyn ThumbnailBlobStore>,
}

pub struct AppState {
    pub mode: OpenRagMode,
    pub app_pool: Option<DbPool>,
    pub worker_pool: Option<DbPool>,
    pub admin_pool: Option<DbPool>,
    pub default_org_id: Option<Uuid>,
    pub pipeline_config: Arc<RwLock<PipelineConfig>>,
    pub search_space_config: SearchSpaceConfig,
    pub master_kek: MasterKek,
    pub pii_engine: Arc<RwLock<PiiEngine>>,
    pub in_memory_vault: Arc<InMemoryVaultStore>,
    pub parser: Arc<DocumentParser>,
    pub thumbnailer: Thumbnailer,
    pub text_embedder: Arc<dyn TextEmbedder>,
    /// False when no text embedding model is configured and the hash embedder stands in.
    pub has_text_model: bool,
    pub blob_store: Arc<dyn ThumbnailBlobStore>,
    pub memory_router: Arc<MemoryRouter>,
    pub kumiho_backend: Arc<KumihoMemoryBackend>,
    pub in_memory_retrieve: Arc<InMemoryHybridRetriever>,
    pub in_memory_eval: Arc<InMemoryEvalStore>,
    pub in_memory_concepts: InMemoryConceptStore,
    pub http_client: reqwest::Client,
    pub llm_transport: Arc<dyn LlmHttpTransport>,
}

impl AppState {
    pub async fn from_env() -> Result<Arc<Self>, OpenRagError> {
        let mode_raw = std::env::var("OPENRAG_MODE").unwrap_or_else(|_| "auto".to_string());
        let mode: OpenRagMode = mode_raw.parse()?;

        let app_dsn = non_empty_env("DATABASE_URL");
        let admin_dsn = non_empty_env("ADMIN_DATABASE_URL");
        let worker_dsn = non_empty_env("WORKER_DATABASE_URL")
            .or_else(|| admin_dsn.clone())
            .or_else(|| app_dsn.clone());

        if mode == OpenRagMode::Live && app_dsn.is_none() {
            return Err(OpenRagError::Config(
                "DATABASE_URL is required when OPENRAG_MODE=live".into(),
            ));
        }

        let app_pool = match app_dsn {
            Some(ref dsn) => Some(DbPool::connect(dsn, 16).await?),
            None => None,
        };
        let admin_pool = match admin_dsn {
            Some(ref dsn) => Some(DbPool::connect(dsn, 8).await?),
            None => None,
        };
        let worker_pool = match worker_dsn {
            Some(ref dsn) => Some(DbPool::connect(dsn, 16).await?),
            None => None,
        };

        if let Some(mig_pool) = admin_pool.as_ref().or(worker_pool.as_ref()) {
            ensure_migrations_if_needed(mig_pool).await?;
        }

        let default_org_id = match non_empty_env("OPENRAG_ORG_ID") {
            Some(raw) => Some(Uuid::parse_str(&raw).map_err(|e| {
                OpenRagError::Config(format!("invalid OPENRAG_ORG_ID '{raw}': {e}"))
            })?),
            None => None,
        };

        let pipeline = load_pipeline_config_from_env()?;
        if pipeline.profile == openrag_core::PipelineProfile::Airgap {
            let allowed: Vec<String> = crate::egress::configured_targets()
                .into_iter()
                .chain(pipeline.egress.targets()?)
                .map(|t| match t.port {
                    Some(p) => format!("{}:{p}", t.host),
                    None => t.host,
                })
                .collect();
            tracing::info!(?allowed, "air-gapped profile: outbound HTTP limited to these hosts");
        }
        let search_space_config = load_search_space_config_from_env()?;
        let master_kek = resolve_master_kek()?;

        let blob_store = blob_store_from_env().await?;

        Self::build_with_parts(AppStateParts {
            mode,
            app_pool,
            worker_pool,
            admin_pool,
            default_org_id,
            pipeline,
            search_space_config,
            master_kek,
            blob_store,
        })
    }

    pub fn new_offline_for_test() -> Result<Arc<Self>, OpenRagError> {
        let pipeline = default_pipeline_config();
        let search_space_config = default_search_space_config();
        let master_kek = resolve_master_kek_dev_default();
        Self::build_with_parts(AppStateParts {
            mode: OpenRagMode::Offline,
            app_pool: None,
            worker_pool: None,
            admin_pool: None,
            default_org_id: None,
            pipeline,
            search_space_config,
            master_kek,
            blob_store: Arc::new(InMemoryBlobStore::new()),
        })
    }

    fn build_with_parts(parts: AppStateParts) -> Result<Arc<Self>, OpenRagError> {
        let AppStateParts {
            mode,
            app_pool,
            worker_pool,
            admin_pool,
            default_org_id,
            pipeline,
            search_space_config,
            master_kek,
            blob_store,
        } = parts;
        let pii_engine = Arc::new(RwLock::new(PiiEngine::from_config(&pipeline.pii)));
        let pipeline_config = Arc::new(RwLock::new(pipeline.clone()));
        let in_mem_retrieve_backend = InMemoryRetrieveBackend::new();
        let in_memory_concepts = in_mem_retrieve_backend.concepts().as_ref().clone();
        let text_embedder: Arc<dyn TextEmbedder> = match crate::text_embed::HttpTextEmbedder::from_env()? {
            Some(http) => Arc::new(http),
            None => Arc::new(DeterministicTextEmbedder::default()),
        };
        let has_text_model = text_embedder_is_model(text_embedder.as_ref());
        if !has_text_model && is_vector_only(&pipeline.retrieve) {
            tracing::warn!(
                "OPENRAG_TEXT_EMBED_URL is unset: hash vectors carry no meaning, so vector-only \
                 retrieval (convex, alpha 0) falls back to rrf with BM25"
            );
        }

        let guard_hook: Arc<dyn MemoryGuardHook> = Arc::new(PiiMemoryGuardHook::new(
            worker_pool.clone(),
            Arc::clone(&pipeline_config),
            Arc::clone(&pii_engine),
            in_memory_concepts.clone(),
        ));

        let pg_backend: Option<Arc<dyn MemoryBackend>> = app_pool.as_ref().map(|pool| {
            let b: Arc<dyn MemoryBackend> = Arc::new(PostgresMemoryBackend::with_components(
                pool.clone(),
                Arc::clone(&text_embedder),
                Arc::clone(&guard_hook),
            ));
            b
        });

        let kumiho_api_key = pipeline
            .memory
            .kumiho_api_key_env
            .as_deref()
            .and_then(non_empty_env);
        let kumiho_backend = Arc::new(KumihoMemoryBackend::with_config(
            pipeline.memory.kumiho_url.clone(),
            kumiho_api_key,
            pipeline.memory.offline_fallback,
            None,
            Arc::clone(&text_embedder),
            Arc::clone(&guard_hook),
            in_memory_concepts.clone(),
        ));
        let kumiho_dyn: Arc<dyn MemoryBackend> = kumiho_backend.clone();

        let effective_memory_cfg = if pg_backend.is_none()
            && pipeline.memory.backend == openrag_core::MemoryBackendKind::OpenragPg
            && pipeline.memory.offline_fallback
            && mode != OpenRagMode::Live
        {
            let mut m = pipeline.memory.clone();
            m.backend = openrag_core::MemoryBackendKind::Kumiho;
            m.mirror_to = None;
            m
        } else {
            pipeline.memory.clone()
        };

        let memory_router = Arc::new(MemoryRouter::from_backends(
            &effective_memory_cfg,
            pg_backend,
            Some(kumiho_dyn),
        )?);

        let in_memory_retrieve = Arc::new(HybridRetriever::from_pipeline_config(
            in_mem_retrieve_backend,
            &retrieval_pipeline(pipeline.clone(), has_text_model),
        )?);

        let http_client = reqwest::Client::builder()
            .timeout(Duration::from_secs(45))
            .build()
            .map_err(|e| OpenRagError::Internal(format!("failed building reqwest client: {e}")))?;
        let llm_transport: Arc<dyn LlmHttpTransport> =
            Arc::new(ReqwestLlmTransport::new(http_client.clone()));

        Ok(Arc::new(Self {
            mode,
            app_pool,
            worker_pool,
            admin_pool,
            default_org_id,
            pipeline_config,
            search_space_config,
            master_kek,
            pii_engine,
            in_memory_vault: Arc::new(InMemoryVaultStore::new()),
            parser: Arc::new(document_parser_from_env()?),
            thumbnailer: Thumbnailer::new(),
            text_embedder,
            has_text_model,
            blob_store,
            memory_router,
            kumiho_backend,
            in_memory_retrieve,
            in_memory_eval: Arc::new(InMemoryEvalStore::new()),
            in_memory_concepts,
            http_client,
            llm_transport,
        }))
    }

    pub fn with_mode_and_transport(
        &self,
        mode: OpenRagMode,
        llm_transport: Arc<dyn LlmHttpTransport>,
    ) -> Arc<Self> {
        Arc::new(Self {
            mode,
            app_pool: self.app_pool.clone(),
            worker_pool: self.worker_pool.clone(),
            admin_pool: self.admin_pool.clone(),
            default_org_id: self.default_org_id,
            pipeline_config: Arc::clone(&self.pipeline_config),
            search_space_config: self.search_space_config.clone(),
            master_kek: self.master_kek.clone(),
            pii_engine: Arc::clone(&self.pii_engine),
            in_memory_vault: Arc::clone(&self.in_memory_vault),
            parser: Arc::clone(&self.parser),
            thumbnailer: self.thumbnailer.clone(),
            text_embedder: Arc::clone(&self.text_embedder),
            has_text_model: self.has_text_model,
            blob_store: Arc::clone(&self.blob_store),
            memory_router: Arc::clone(&self.memory_router),
            kumiho_backend: Arc::clone(&self.kumiho_backend),
            in_memory_retrieve: Arc::clone(&self.in_memory_retrieve),
            in_memory_eval: Arc::clone(&self.in_memory_eval),
            in_memory_concepts: self.in_memory_concepts.clone(),
            http_client: self.http_client.clone(),
            llm_transport,
        })
    }

    pub fn require_worker_pool(&self) -> Result<&DbPool, OpenRagError> {
        self.worker_pool
            .as_ref()
            .ok_or_else(|| OpenRagError::Config("PostgreSQL worker pool is not configured".into()))
    }

    pub fn require_app_pool(&self) -> Result<&DbPool, OpenRagError> {
        self.app_pool
            .as_ref()
            .ok_or_else(|| OpenRagError::Config("PostgreSQL app pool is not configured".into()))
    }

    pub async fn pg_retriever(&self, org_id: Uuid) -> Result<PgHybridRetriever, OpenRagError> {
        let app_pool = self.require_app_pool()?.clone();
        let pipeline = retrieval_pipeline(self.pipeline_config.read().await.clone(), self.has_text_model);
        let mut backend = PgRetrieveBackend::new(app_pool).with_embedder(Arc::clone(&self.text_embedder));
        if let Some(worker) = self.worker_pool.as_ref() {
            backend = backend.load_bindings_from_worker(worker, org_id).await?;
        }
        HybridRetriever::from_pipeline_config(backend, &pipeline)
    }

    pub fn pg_eval_store(&self) -> Result<PgEvalStore, OpenRagError> {
        let app = self.require_app_pool()?.clone();
        let worker = self.require_worker_pool()?.clone();
        Ok(PgEvalStore::new(app, worker))
    }

    pub async fn active_memory_router(&self) -> Result<MemoryRouter, OpenRagError> {
        let mem_cfg = self.pipeline_config.read().await.memory.clone();
        let pg_dyn = self
            .memory_router
            .backend(Some(openrag_core::MemoryBackendKind::OpenragPg))
            .ok()
            .cloned();
        let kumiho_dyn: Arc<dyn MemoryBackend> = self.kumiho_backend.clone();
        if pg_dyn.is_none()
            && mem_cfg.backend == openrag_core::MemoryBackendKind::OpenragPg
            && mem_cfg.offline_fallback
            && self.mode != OpenRagMode::Live
        {
            let mut effective = mem_cfg;
            effective.backend = openrag_core::MemoryBackendKind::Kumiho;
            effective.mirror_to = None;
            return MemoryRouter::from_backends(&effective, None, Some(kumiho_dyn));
        }
        MemoryRouter::from_backends(&mem_cfg, pg_dyn, Some(kumiho_dyn))
    }
}

/// OCR goes to a Schift OCR endpoint when `OPENRAG_SCHIFT_OCR_URL` is set: Schift's service, an
/// on-prem Schift, or any server speaking its `/v1/run` API (key in `SCHIFT_OCR_API_KEY`).
/// Unset, images fall back to local foreground-block detection.
/// Thumbnail/figure storage: an S3 bucket when `OPENRAG_S3_BUCKET` is set (MinIO next to OpenRAG,
/// or AWS S3 and other S3-compatible services), else files under `OPENRAG_BLOB_DIR`, else memory.
async fn blob_store_from_env() -> Result<Arc<dyn ThumbnailBlobStore>, OpenRagError> {
    if let Some(bucket) = non_empty_env("OPENRAG_S3_BUCKET") {
        let require = |name: &str| {
            non_empty_env(name).ok_or_else(|| {
                OpenRagError::Config(format!("OPENRAG_S3_BUCKET is set, so {name} is required"))
            })
        };
        let path_style = match non_empty_env("OPENRAG_S3_URL_STYLE").as_deref() {
            None | Some("path") => true,
            Some("virtual") => false,
            Some(other) => {
                return Err(OpenRagError::Config(format!(
                    "OPENRAG_S3_URL_STYLE must be path or virtual, got '{other}'"
                )))
            }
        };
        let store = S3BlobStore::new(S3BlobConfig {
            endpoint: require("OPENRAG_S3_ENDPOINT")?,
            bucket,
            region: non_empty_env("OPENRAG_S3_REGION").unwrap_or_else(|| "us-east-1".into()),
            access_key_id: require("OPENRAG_S3_ACCESS_KEY_ID")?,
            secret_access_key: require("OPENRAG_S3_SECRET_ACCESS_KEY")?,
            path_style,
        })?;
        store.ensure_bucket().await?;
        tracing::info!(bucket = store.bucket_name(), "thumbnails go to S3");
        return Ok(Arc::new(store));
    }
    Ok(match non_empty_env("OPENRAG_BLOB_DIR") {
        Some(dir) => Arc::new(LocalFsBlobStore::new(Path::new(&dir))?),
        None => Arc::new(InMemoryBlobStore::new()),
    })
}

fn document_parser_from_env() -> Result<DocumentParser, OpenRagError> {
    let mut parser = DocumentParser::new();
    if let Some(url) = non_empty_env("OPENRAG_SCHIFT_OCR_URL") {
        parser.ocr_client = Some(SchiftOcrClient::new(url, non_empty_env("SCHIFT_OCR_API_KEY"))?);
    }
    Ok(parser)
}

pub fn non_empty_env(name: &str) -> Option<String> {
    std::env::var(name)
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

pub async fn ensure_migrations_if_needed(pool: &DbPool) -> Result<bool, OpenRagError> {
    let (already_migrated, has_cli_codes, has_128d_vectors, has_page_vectors, has_cli_llm_types): (
        bool,
        bool,
        bool,
        bool,
        bool,
    ) = sqlx::query_as(
        "SELECT to_regclass('iam.orgs') IS NOT NULL AND to_regclass('wiki.concepts') IS NOT NULL,
                to_regclass('iam.cli_auth_codes') IS NOT NULL,
                EXISTS (SELECT 1 FROM pg_attribute
                        WHERE attrelid = to_regclass('corpus.chunks')
                          AND attname = 'embedding' AND NOT attisdropped
                          AND format_type(atttypid, atttypmod) = 'vector(128)'),
                EXISTS (SELECT 1 FROM pg_attribute
                        WHERE attrelid = to_regclass('corpus.pages')
                          AND attname = 'thumb_embedding' AND NOT attisdropped),
                EXISTS (SELECT 1 FROM pg_constraint
                        WHERE conrelid = to_regclass('gateway.providers')
                          AND conname = 'providers_provider_type_check'
                          AND pg_get_constraintdef(oid) LIKE '%claude_cli%')",
    )
    .fetch_one(pool.inner())
    .await
    .map_err(|e| OpenRagError::Database(format!("failed probing migration state: {e}")))?;

    if !already_migrated {
        let _ = pool.run_migrations().await?;
        return Ok(true);
    }
    let mut applied = false;
    if !has_cli_codes {
        // 009 is idempotent, so it can be applied directly on psql-migrated databases
        // that predate it (no openrag_schema_migrations bookkeeping there).
        apply_embedded_migration(pool, "009").await?;
        applied = true;
    }
    // 010 only rewrites guard rules that still hold the original seeded sk- pattern, so
    // re-running it on every start is a no-op once existing orgs have been brought in line.
    applied |= apply_embedded_migration(pool, "010").await? > 0;
    if has_128d_vectors {
        // 011 widens every vector column to 256d and clears the 128d vectors it cannot keep.
        apply_embedded_migration(pool, "011").await?;
        applied = true;
    }
    if has_page_vectors {
        // 012 drops the page thumbnail / figure vector columns: those are no longer embedded.
        apply_embedded_migration(pool, "012").await?;
        applied = true;
    }
    if !has_cli_llm_types {
        // 013 lets gateway.providers hold the claude_cli / codex_cli provider types.
        apply_embedded_migration(pool, "013").await?;
        applied = true;
    }
    Ok(applied)
}

async fn apply_embedded_migration(pool: &DbPool, version: &str) -> Result<u64, OpenRagError> {
    let spec = openrag_db::EMBEDDED_MIGRATIONS
        .iter()
        .find(|m| m.version == version)
        .ok_or_else(|| OpenRagError::Internal(format!("embedded migration {version} missing")))?;
    let res = sqlx::raw_sql(spec.sql)
        .execute(pool.inner())
        .await
        .map_err(|e| OpenRagError::Database(format!("failed applying migration {version}: {e}")))?;
    Ok(res.rows_affected())
}

pub fn resolve_master_kek() -> Result<MasterKek, OpenRagError> {
    if non_empty_env("OPENRAG_PII_KEK_B64").is_some() || non_empty_env("OPENRAG_PII_KEK_HEX").is_some()
    {
        return MasterKek::from_env();
    }
    Ok(resolve_master_kek_dev_default())
}

pub fn resolve_master_kek_dev_default() -> MasterKek {
    let digest = Sha256::digest(b"openrag-local-dev-kek-v1");
    let mut arr = [0u8; 32];
    arr.copy_from_slice(&digest[..32]);
    MasterKek::from_bytes(arr)
}

pub fn default_pipeline_config() -> PipelineConfig {
    PipelineConfig {
        version: 1,
        profile: PipelineProfile::Standard,
        parse: ParseConfig {
            hwp_engine: "kordoc".into(),
            ocr_engine: "schift_ocr".into(),
            dewarp_enabled: true,
            pdf_native_enabled: true,
            default_ocr_mode: Sensitivity::Share,
            ocr_mode_by_auth_level: vec![
                OcrModeRule {
                    min_auth_level: AuthLevel::new(8).unwrap_or(AuthLevel::MAX),
                    required_mode: Sensitivity::Zdr,
                },
                OcrModeRule {
                    min_auth_level: AuthLevel::new(5).unwrap_or(AuthLevel::MAX),
                    required_mode: Sensitivity::Private,
                },
            ],
            reject_zdr_with_batch: true,
        },
        thumbnail: ThumbnailConfig {
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
        memory: MemoryConfig::default(),
        store: StoreConfig {
            pg_pinecone_engine: PgPineconeEngine::Native,
            dim: 256,
            bm25_engine: "pg_search".into(),
            bm25_fallback: "pg_trgm".into(),
            external_mirror: ExternalMirrorBackend::None,
            outbox_batch_size: 64,
            outbox_max_attempts: 3,
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
            local_only_min_auth_level: AuthLevel::new(8).unwrap_or(AuthLevel::MAX),
            require_pii_masked_for_external: true,
        },
        egress: Default::default(),
    }
}

pub fn default_search_space_config() -> SearchSpaceConfig {
    let mut nodes = BTreeMap::new();
    nodes.insert(
        "retrieve_fusion".into(),
        vec![
            serde_yaml_ng::Value::String("rrf".into()),
            serde_yaml_ng::Value::String("convex".into()),
        ],
    );
    nodes.insert(
        "retrieve_alpha".into(),
        vec![
            serde_yaml_ng::Value::from(0.2_f64),
            serde_yaml_ng::Value::from(0.3_f64),
            serde_yaml_ng::Value::from(0.5_f64),
        ],
    );
    SearchSpaceConfig {
        version: 1,
        golden_set: "gov_v1".into(),
        strategy: OptimizationStrategy::GreedyNode,
        metric: "ndcg_at_10".into(),
        security_gates: SecurityGateConfig {
            require_zero_pii_leak: true,
            require_zero_arn_violation: true,
        },
        nodes,
    }
}

fn text_embedder_is_model(embedder: &dyn TextEmbedder) -> bool {
    embedder.model_name() != DeterministicTextEmbedder::default().model_name()
}

fn is_vector_only(retrieve: &RetrieveConfig) -> bool {
    retrieve.fusion == FusionMode::Convex && retrieve.alpha <= 0.0
}

/// The retrieval settings actually used. Vector-only retrieval needs a text embedding model; with the
/// hash fallback (`OPENRAG_TEXT_EMBED_URL` unset) the vectors carry no meaning, so BM25 has to carry
/// the ranking and fusion becomes rrf.
pub fn retrieval_pipeline(mut pipeline: PipelineConfig, has_text_model: bool) -> PipelineConfig {
    if !has_text_model && is_vector_only(&pipeline.retrieve) {
        pipeline.retrieve.fusion = FusionMode::Rrf;
    }
    pipeline
}

/// `configs/pipeline.default.yaml`, compiled in so a binary run outside the checkout
/// still carries the full default guard rules (sk- keys, safe-words, PII masking).
pub const EMBEDDED_DEFAULT_PIPELINE_YAML: &str = include_str!("../../../configs/pipeline.default.yaml");

fn load_pipeline_config_from_env() -> Result<PipelineConfig, OpenRagError> {
    if let Some(path) = non_empty_env("OPENRAG_PIPELINE_CONFIG") {
        return PipelineConfig::from_yaml_file(Path::new(&path));
    }
    let default_path = Path::new("configs/pipeline.default.yaml");
    if default_path.is_file() {
        return PipelineConfig::from_yaml_file(default_path);
    }
    PipelineConfig::from_yaml_str(EMBEDDED_DEFAULT_PIPELINE_YAML)
}

fn load_search_space_config_from_env() -> Result<SearchSpaceConfig, OpenRagError> {
    if let Some(path) = non_empty_env("OPENRAG_SEARCH_SPACE_CONFIG") {
        return SearchSpaceConfig::from_yaml_file(Path::new(&path));
    }
    let default_path = Path::new("configs/search_space.example.yaml");
    if default_path.is_file() {
        return SearchSpaceConfig::from_yaml_file(default_path);
    }
    Ok(default_search_space_config())
}

fn parse_guard_action_str(raw: &str) -> Result<GuardAction, OpenRagError> {
    match raw {
        "allow" => Ok(GuardAction::Allow),
        "warn" => Ok(GuardAction::Warn),
        "mask" => Ok(GuardAction::Mask),
        "block" => Ok(GuardAction::Block),
        other => Err(OpenRagError::Database(format!(
            "invalid guard action '{other}'"
        ))),
    }
}

fn parse_guard_surface_str(raw: &str) -> Result<GuardSurface, OpenRagError> {
    match raw {
        "inbound_query" => Ok(GuardSurface::InboundQuery),
        "llm_egress" => Ok(GuardSurface::LlmEgress),
        "mcp_egress" => Ok(GuardSurface::McpEgress),
        "ingestion_gate" => Ok(GuardSurface::IngestionGate),
        other => Err(OpenRagError::Database(format!(
            "invalid guard surface '{other}'"
        ))),
    }
}

fn parse_guard_policy_row(
    row: &sqlx::postgres::PgRow,
) -> Result<openrag_pii::GuardPolicyRow, OpenRagError> {
    use sqlx::Row;

    let def_str: String = row
        .try_get("default_action")
        .map_err(|e| OpenRagError::Database(e.to_string()))?;
    let surf_strs: Vec<String> = row
        .try_get("surfaces")
        .map_err(|e| OpenRagError::Database(e.to_string()))?;
    let surfaces = surf_strs
        .iter()
        .map(|s| parse_guard_surface_str(s))
        .collect::<Result<Vec<_>, _>>()?;

    Ok(openrag_pii::GuardPolicyRow {
        id: row
            .try_get("id")
            .map_err(|e| OpenRagError::Database(e.to_string()))?,
        org_id: row
            .try_get("org_id")
            .map_err(|e| OpenRagError::Database(e.to_string()))?,
        bucket_id: row
            .try_get("bucket_id")
            .map_err(|e| OpenRagError::Database(e.to_string()))?,
        name: row
            .try_get("name")
            .map_err(|e| OpenRagError::Database(e.to_string()))?,
        master_enabled: row
            .try_get("master_enabled")
            .map_err(|e| OpenRagError::Database(e.to_string()))?,
        default_action: parse_guard_action_str(&def_str)?,
        allowlist_terms: row
            .try_get("allowlist_terms")
            .map_err(|e| OpenRagError::Database(e.to_string()))?,
        surfaces,
        is_active: row
            .try_get("is_active")
            .map_err(|e| OpenRagError::Database(e.to_string()))?,
        updated_at: row
            .try_get("updated_at")
            .map_err(|e| OpenRagError::Database(e.to_string()))?,
    })
}

pub async fn ensure_default_guard_policy(
    worker_pool: &DbPool,
    org_id: Uuid,
    pipeline: &PipelineConfig,
) -> Result<openrag_pii::GuardPolicyRow, OpenRagError> {
    let mut tx = worker_pool
        .inner()
        .begin()
        .await
        .map_err(|e| OpenRagError::Database(format!("begin guard policy tx: {e}")))?;

    let lock_key = format!("guard_policy_seed:{org_id}");
    sqlx::query("SELECT pg_advisory_xact_lock(hashtextextended($1, 0))")
        .bind(&lock_key)
        .execute(&mut *tx)
        .await
        .map_err(|e| OpenRagError::Database(format!("acquire guard policy lock: {e}")))?;

    let existing_row = sqlx::query(
        "SELECT id, org_id, bucket_id, name, master_enabled, default_action,
                allowlist_terms, surfaces, is_active, updated_at
         FROM pii.guard_policies
         WHERE org_id = $1
         ORDER BY is_active DESC, (name = 'default') DESC, updated_at DESC
         LIMIT 1",
    )
    .bind(org_id)
    .fetch_optional(&mut *tx)
    .await
    .map_err(|e| OpenRagError::Database(format!("query pii.guard_policies: {e}")))?;

    if let Some(row) = existing_row {
        let policy = parse_guard_policy_row(&row)?;
        tx.commit()
            .await
            .map_err(|e| OpenRagError::Database(format!("commit guard policy lookup tx: {e}")))?;
        return Ok(policy);
    }

    let surfaces = [
        GuardSurface::InboundQuery,
        GuardSurface::LlmEgress,
        GuardSurface::McpEgress,
        GuardSurface::IngestionGate,
    ];
    let surface_strs: Vec<String> = surfaces.iter().map(|s| s.as_str().to_string()).collect();

    let policy_row = sqlx::query(
        "INSERT INTO pii.guard_policies (
            org_id, name, master_enabled, default_action, allowlist_terms, surfaces, is_active
         ) VALUES ($1, 'default', $2, $3, $4, $5, true)
         ON CONFLICT (org_id, name) DO UPDATE SET updated_at = now()
         RETURNING id, org_id, bucket_id, name, master_enabled, default_action,
                   allowlist_terms, surfaces, is_active, updated_at",
    )
    .bind(org_id)
    .bind(pipeline.pii.guard.enabled)
    .bind(pipeline.pii.guard.default_action.as_str())
    .bind(&pipeline.pii.guard.allowlist)
    .bind(&surface_strs)
    .fetch_one(&mut *tx)
    .await
    .map_err(|e| OpenRagError::Database(format!("upsert default pii.guard_policies: {e}")))?;

    let policy = parse_guard_policy_row(&policy_row)?;

    let existing_rules: i64 = sqlx::query_scalar(
        "SELECT count(*) FROM pii.guard_rules WHERE org_id = $1 AND policy_id = $2",
    )
    .bind(org_id)
    .bind(policy.id)
    .fetch_one(&mut *tx)
    .await
    .map_err(|e| OpenRagError::Database(format!("count pii.guard_rules: {e}")))?;

    if existing_rules == 0 {
        for rule in &pipeline.pii.guard.rules {
            if rule.pattern_or_category.trim().is_empty() || rule.surfaces.is_empty() {
                continue;
            }
            if rule.kind == GuardRuleKind::Regex {
                let _ = regex::Regex::new(&rule.pattern_or_category).map_err(|e| {
                    OpenRagError::Validation(format!(
                        "invalid guard regex '{}': {e}",
                        rule.pattern_or_category
                    ))
                })?;
            }
            let kind_str = match rule.kind {
                GuardRuleKind::PiiCategory => "pii_category",
                GuardRuleKind::SafeWord => "safe_word",
                GuardRuleKind::Keyword => "keyword",
                GuardRuleKind::Regex => "regex",
            };
            let rule_surfaces: Vec<String> =
                rule.surfaces.iter().map(|s| s.as_str().to_string()).collect();
            sqlx::query(
                "INSERT INTO pii.guard_rules (
                    policy_id, org_id, rule_kind, pattern_or_category, action,
                    surfaces, replacement_text, priority, enabled
                 ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)",
            )
            .bind(policy.id)
            .bind(org_id)
            .bind(kind_str)
            .bind(&rule.pattern_or_category)
            .bind(rule.action.as_str())
            .bind(&rule_surfaces)
            .bind(&rule.replacement_text)
            .bind(rule.priority)
            .bind(rule.enabled)
            .execute(&mut *tx)
            .await
            .map_err(|e| OpenRagError::Database(format!("insert default pii.guard_rules: {e}")))?;
        }
    }

    tx.commit()
        .await
        .map_err(|e| OpenRagError::Database(format!("commit guard policy seed tx: {e}")))?;

    Ok(policy)
}

pub async fn load_active_guard_policy(
    worker_pool: Option<&DbPool>,
    org_id: Uuid,
    pipeline: &PipelineConfig,
) -> Result<GuardPolicyRuntime, OpenRagError> {
    let Some(pool) = worker_pool else {
        return Ok(GuardPolicyRuntime::from_guard_config(org_id, &pipeline.pii.guard));
    };

    let total_count: i64 = sqlx::query_scalar(
        "SELECT count(*) FROM pii.guard_policies WHERE org_id = $1",
    )
    .bind(org_id)
    .fetch_one(pool.inner())
    .await
    .map_err(|e| OpenRagError::Database(format!("failed counting pii.guard_policies: {e}")))?;

    let active_row = if total_count == 0 {
        let p = ensure_default_guard_policy(pool, org_id, pipeline).await?;
        Some((p.id, p.master_enabled, p.default_action, p.allowlist_terms))
    } else {
        use sqlx::Row;
        let row = sqlx::query(
            "SELECT id, master_enabled, default_action, allowlist_terms
             FROM pii.guard_policies
             WHERE org_id = $1 AND is_active = true
             ORDER BY updated_at DESC
             LIMIT 1",
        )
        .bind(org_id)
        .fetch_optional(pool.inner())
        .await
        .map_err(|e| OpenRagError::Database(format!("failed querying active guard policy: {e}")))?;

        match row {
            Some(r) => {
                let id: Uuid = r
                    .try_get("id")
                    .map_err(|e| OpenRagError::Database(e.to_string()))?;
                let master_enabled: bool = r
                    .try_get("master_enabled")
                    .map_err(|e| OpenRagError::Database(e.to_string()))?;
                let def_str: String = r
                    .try_get("default_action")
                    .map_err(|e| OpenRagError::Database(e.to_string()))?;
                let default_action = match def_str.as_str() {
                    "allow" => GuardAction::Allow,
                    "warn" => GuardAction::Warn,
                    "mask" => GuardAction::Mask,
                    "block" => GuardAction::Block,
                    other => {
                        return Err(OpenRagError::Database(format!(
                            "invalid guard default_action '{other}'"
                        )));
                    }
                };
                let allowlist_terms: Vec<String> = r
                    .try_get("allowlist_terms")
                    .map_err(|e| OpenRagError::Database(e.to_string()))?;
                Some((id, master_enabled, default_action, allowlist_terms))
            }
            None => None,
        }
    };

    let Some((policy_id, master_enabled, default_action, allowlist_terms)) = active_row else {
        return Ok(GuardPolicyRuntime {
            id: None,
            org_id,
            master_enabled: false,
            default_action: GuardAction::Allow,
            allowlist_terms: Vec::new(),
            rules: Vec::new(),
            semantic_threshold: None,
        });
    };

    let db_rules = GuardRepo::list_rules(pool.inner(), org_id, policy_id).await?;
    let rules = db_rules
        .into_iter()
        .map(|r| GuardRuleRuntime {
            id: Some(r.id),
            kind: r.kind,
            pattern_or_category: r.pattern_or_category,
            action: r.action,
            surfaces: r.surfaces,
            replacement_text: r.replacement_text,
            enabled: r.enabled,
            priority: r.priority,
        })
        .collect();

    Ok(GuardPolicyRuntime {
        id: Some(policy_id),
        org_id,
        master_enabled,
        default_action,
        allowlist_terms,
        rules,
        semantic_threshold: None,
    })
}

pub async fn load_concept_lookup(
    worker_pool: Option<&DbPool>,
    in_memory_concepts: &InMemoryConceptStore,
    org_id: Uuid,
) -> Result<InMemoryConceptAliasLookup, OpenRagError> {
    if let Some(pool) = worker_pool {
        return InMemoryConceptAliasLookup::load_from_pg(pool.inner(), org_id).await;
    }
    let list = in_memory_concepts.list_concepts(org_id).await;
    let mut lookup = InMemoryConceptAliasLookup::new();
    for item in list {
        let mut terms: Vec<String> = Vec::with_capacity(item.aliases.len() + 2);
        terms.push(item.concept.canonical_key);
        terms.push(item.concept.display_name);
        for al in item.aliases {
            terms.push(al.alias_text);
        }
        let refs: Vec<&str> = terms.iter().map(String::as_str).collect();
        lookup.link_terms(&refs);
    }
    Ok(lookup)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedded_default_pipeline_parses_and_carries_guard_rules() {
        let cfg = PipelineConfig::from_yaml_str(EMBEDDED_DEFAULT_PIPELINE_YAML);
        assert!(cfg.is_ok(), "{cfg:?}");
        for needle in ["PROJECT-BLACKBOX", "CONFIDENTIAL-DRAFT", "phone_kr", r"\bsk-"] {
            assert!(
                EMBEDDED_DEFAULT_PIPELINE_YAML.contains(needle),
                "embedded default must carry {needle}"
            );
        }
    }
}
