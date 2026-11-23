use std::collections::BTreeMap;

use chrono::Utc;
use openrag_core::{FusionMode, OpenRagError, PipelineConfig, PrincipalContext, RetrieveConfig};
use openrag_db::DbPool;
use uuid::Uuid;

use crate::backend::{InMemoryRetrieveBackend, PgRetrieveBackend, RetrieveBackend};
use crate::fusion::fuse_and_rank;
use crate::query::{
    has_conflicting_tags, merge_tag_filters, query_has_searchable_tokens, resolve_retrieve_params,
    sanitize_and_parse_inline_tags,
};
use crate::types::{
    validate_resolved_values, BackendExecutionInput, FusionContext, RetrieveOverrides,
    RetrieveSpace, SearchRequest, SearchResponse,
};

pub fn default_retrieve_config() -> RetrieveConfig {
    RetrieveConfig {
        spaces: vec!["chunks".into(), "wiki".into(), "memory".into()],
        fusion: FusionMode::Convex,
        rrf_k: 60,
        alpha: 0.0,
        top_k: 12,
        rerank: true,
        latest_revision_boost: 0.15,
        admission_threshold: 0.35,
    }
}

pub struct HybridRetriever<B: RetrieveBackend> {
    backend: B,
    defaults: RetrieveConfig,
}

pub type PgHybridRetriever = HybridRetriever<PgRetrieveBackend>;
pub type InMemoryHybridRetriever = HybridRetriever<InMemoryRetrieveBackend>;

impl PgHybridRetriever {
    pub fn new_pg(app_pool: DbPool) -> Self {
        Self::new(PgRetrieveBackend::new(app_pool))
    }
}

impl<B: RetrieveBackend> HybridRetriever<B> {
    pub fn new(backend: B) -> Self {
        Self {
            backend,
            defaults: default_retrieve_config(),
        }
    }

    pub fn with_config(backend: B, defaults: RetrieveConfig) -> Result<Self, OpenRagError> {
        validate_resolved_values(
            &defaults.spaces,
            defaults.rrf_k,
            defaults.alpha,
            defaults.top_k,
            defaults.latest_revision_boost,
            defaults.admission_threshold,
        )?;
        Ok(Self { backend, defaults })
    }

    pub fn from_pipeline_config(
        backend: B,
        pipeline: &PipelineConfig,
    ) -> Result<Self, OpenRagError> {
        pipeline.validate()?;
        Self::with_config(backend, pipeline.retrieve.clone())
    }

    pub fn default_config(&self) -> &RetrieveConfig {
        &self.defaults
    }

    pub fn backend(&self) -> &B {
        &self.backend
    }

    pub async fn search(
        &self,
        ctx: &PrincipalContext,
        req: &SearchRequest,
    ) -> Result<SearchResponse, OpenRagError> {
        self.search_with_config(ctx, req, &self.defaults).await
    }

    pub async fn search_with_overrides(
        &self,
        ctx: &PrincipalContext,
        req: &SearchRequest,
        overrides: &RetrieveOverrides,
    ) -> Result<SearchResponse, OpenRagError> {
        let merged = overrides.apply_to_config(&self.defaults)?;
        self.search_with_config(ctx, req, &merged).await
    }

    async fn search_with_config(
        &self,
        ctx: &PrincipalContext,
        req: &SearchRequest,
        cfg: &RetrieveConfig,
    ) -> Result<SearchResponse, OpenRagError> {
        let params = resolve_retrieve_params(req, cfg)?;
        let parsed = sanitize_and_parse_inline_tags(&req.query);
        let effective_tag_filter = merge_tag_filters(req.tag_filter.as_ref(), &parsed);

        let has_tokens = query_has_searchable_tokens(&parsed.clean_query);
        let has_tags = !effective_tag_filter.is_empty();
        let has_vec = req.query_vector.is_some();

        // Empty / punctuation-only query with no tags and no explicit vector: graceful empty result.
        if !has_tokens && !has_tags && !has_vec {
            return Ok(SearchResponse::empty(
                parsed.clean_query,
                parsed.inline_tags,
                parsed.excluded_tags,
                Vec::new(),
            ));
        }

        // Conflicting tag filter (e.g. tags_all=["hr"], tags_exclude=["hr"]) -> deterministic empty.
        if has_conflicting_tags(&effective_tag_filter) {
            return Ok(SearchResponse::empty(
                parsed.clean_query,
                parsed.inline_tags,
                parsed.excluded_tags,
                Vec::new(),
            ));
        }

        let tag_only_query = !has_tokens && has_tags && !has_vec;
        let candidate_limit_per_space = params.top_k.saturating_mul(4).clamp(40, 500);

        let exec_input = BackendExecutionInput {
            clean_query: parsed.clean_query.clone(),
            explicit_query_vector: req.query_vector.clone(),
            bucket_id: req.bucket_id,
            spaces: params.spaces.clone(),
            effective_tag_filter,
            candidate_limit_per_space,
            include_embeddings: params.rerank,
            tag_only_query,
            skip_lexical: params.fusion == FusionMode::Convex && params.alpha <= 0.0,
        };

        let exec_output = self.backend.execute_plan(ctx, &exec_input).await?;

        Ok(fuse_and_rank(
            FusionContext {
                clean_query: parsed.clean_query,
                parsed_inline_tags: parsed.inline_tags,
                parsed_excluded_tags: parsed.excluded_tags,
                tag_only_query,
                facet_limit: req.facet_limit,
                now: req.as_of.unwrap_or_else(Utc::now),
            },
            &params,
            exec_output,
        ))
    }

    pub async fn bucket_tag_facets(
        &self,
        ctx: &PrincipalContext,
        spaces: Option<&[RetrieveSpace]>,
        bucket_id: Option<Uuid>,
        limit: i64,
    ) -> Result<BTreeMap<String, i64>, OpenRagError> {
        let resolved_spaces = match spaces {
            Some(sp) => {
                if sp.is_empty() {
                    return Err(OpenRagError::Validation("spaces must not be empty".into()));
                }
                sp.to_vec()
            }
            None => {
                let mut out = Vec::with_capacity(self.defaults.spaces.len());
                for raw in &self.defaults.spaces {
                    out.push(raw.parse()?);
                }
                out
            }
        };
        if limit <= 0 {
            return Err(OpenRagError::Validation(format!(
                "facet_limit must be >= 1, got {limit}"
            )));
        }
        self.backend
            .compute_bucket_facets(ctx, &resolved_spaces, bucket_id, limit)
            .await
    }
}
