use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use openrag_core::{MemoryBackendKind, MemoryConfig, OpenRagError, PrincipalContext};
use openrag_store::TagFilter;
use uuid::Uuid;

use crate::types::{
    ApplyPatchInput, EngageRequest, EngageResponse, MemoryEdge, MemoryEdgeType, MemoryEntry,
    MemoryEpisode, MemoryPatch, PatchOp, PromoteMemoryInput, PromoteResult, RecordEpisodeInput,
    ReflectRequest, ReflectResponse, UpsertMemoryInput,
};

#[async_trait]
pub trait MemoryBackend: Send + Sync {
    fn kind(&self) -> MemoryBackendKind;

    async fn record_episode(
        &self,
        ctx: &PrincipalContext,
        input: RecordEpisodeInput,
    ) -> Result<MemoryEpisode, OpenRagError>;

    async fn upsert_entry(
        &self,
        ctx: &PrincipalContext,
        input: UpsertMemoryInput,
    ) -> Result<MemoryEntry, OpenRagError>;

    async fn get_entry(
        &self,
        ctx: &PrincipalContext,
        entry_id: Uuid,
    ) -> Result<Option<MemoryEntry>, OpenRagError>;

    async fn list_entries(
        &self,
        ctx: &PrincipalContext,
        tag_filter: Option<&TagFilter>,
        limit: i64,
    ) -> Result<Vec<MemoryEntry>, OpenRagError>;

    async fn apply_patch(
        &self,
        ctx: &PrincipalContext,
        input: ApplyPatchInput,
    ) -> Result<MemoryPatch, OpenRagError>;

    async fn promote(
        &self,
        ctx: &PrincipalContext,
        input: PromoteMemoryInput,
    ) -> Result<PromoteResult, OpenRagError>;

    async fn search(
        &self,
        ctx: &PrincipalContext,
        query: &str,
        tag_filter: Option<&TagFilter>,
        top_k: usize,
    ) -> Result<Vec<MemoryEntry>, OpenRagError>;

    async fn engage(
        &self,
        ctx: &PrincipalContext,
        req: EngageRequest,
    ) -> Result<EngageResponse, OpenRagError>;

    async fn reflect(
        &self,
        ctx: &PrincipalContext,
        req: ReflectRequest,
    ) -> Result<ReflectResponse, OpenRagError>;

    async fn add_edge(
        &self,
        ctx: &PrincipalContext,
        src_memory_id: Uuid,
        dst_memory_id: Uuid,
        relation_type: MemoryEdgeType,
        confidence: f32,
    ) -> Result<MemoryEdge, OpenRagError>;

    async fn list_edges(
        &self,
        ctx: &PrincipalContext,
        memory_id: Uuid,
    ) -> Result<Vec<MemoryEdge>, OpenRagError>;

    async fn record_episode_with_id(
        &self,
        ctx: &PrincipalContext,
        episode_id: Option<Uuid>,
        input: RecordEpisodeInput,
    ) -> Result<MemoryEpisode, OpenRagError> {
        let _ = episode_id;
        self.record_episode(ctx, input).await
    }

    async fn upsert_entry_with_id(
        &self,
        ctx: &PrincipalContext,
        entry_id: Option<Uuid>,
        input: UpsertMemoryInput,
    ) -> Result<MemoryEntry, OpenRagError> {
        let _ = entry_id;
        self.upsert_entry(ctx, input).await
    }

    async fn apply_patch_with_new_id(
        &self,
        ctx: &PrincipalContext,
        input: ApplyPatchInput,
        new_entry_id: Option<Uuid>,
    ) -> Result<MemoryPatch, OpenRagError> {
        let _ = new_entry_id;
        self.apply_patch(ctx, input).await
    }

    async fn reflect_with_ids(
        &self,
        ctx: &PrincipalContext,
        entry_id: Option<Uuid>,
        episode_id: Option<Uuid>,
        req: ReflectRequest,
    ) -> Result<ReflectResponse, OpenRagError> {
        let _ = (entry_id, episode_id);
        self.reflect(ctx, req).await
    }
}

/// Pluggable router over registered `MemoryBackend` implementations (`openrag_pg` | `kumiho`)
/// with optional `mirror_to` dual-write replication and per-call backend override.
#[derive(Clone)]
pub struct MemoryRouter {
    primary_kind: MemoryBackendKind,
    mirror_kind: Option<MemoryBackendKind>,
    backends: HashMap<MemoryBackendKind, Arc<dyn MemoryBackend>>,
}

impl MemoryRouter {
    pub fn from_backends(
        config: &MemoryConfig,
        pg_backend: Option<Arc<dyn MemoryBackend>>,
        kumiho_backend: Option<Arc<dyn MemoryBackend>>,
    ) -> Result<Self, OpenRagError> {
        let mut backends: HashMap<MemoryBackendKind, Arc<dyn MemoryBackend>> = HashMap::new();
        if let Some(pg) = pg_backend {
            backends.insert(MemoryBackendKind::OpenragPg, pg);
        }
        if let Some(km) = kumiho_backend {
            backends.insert(MemoryBackendKind::Kumiho, km);
        }
        if !backends.contains_key(&config.backend) {
            return Err(OpenRagError::Config(format!(
                "configured memory.backend '{}' is not registered",
                config.backend.as_str()
            )));
        }
        if let Some(mirror) = config.mirror_to {
            if !backends.contains_key(&mirror) {
                return Err(OpenRagError::Config(format!(
                    "configured memory.mirror_to '{}' is not registered",
                    mirror.as_str()
                )));
            }
        }
        Ok(Self {
            primary_kind: config.backend,
            mirror_kind: config.mirror_to.filter(|m| *m != config.backend),
            backends,
        })
    }

    pub fn primary_kind(&self) -> MemoryBackendKind {
        self.primary_kind
    }

    pub fn mirror_kind(&self) -> Option<MemoryBackendKind> {
        self.mirror_kind
    }

    pub fn backend(&self, override_kind: Option<MemoryBackendKind>) -> Result<&Arc<dyn MemoryBackend>, OpenRagError> {
        let target = override_kind.unwrap_or(self.primary_kind);
        self.backends.get(&target).ok_or_else(|| {
            OpenRagError::Config(format!(
                "memory backend '{}' is not available",
                target.as_str()
            ))
        })
    }

    fn mirror_backend(&self, override_kind: Option<MemoryBackendKind>) -> Option<&Arc<dyn MemoryBackend>> {
        let active = override_kind.unwrap_or(self.primary_kind);
        let mirror = self.mirror_kind?;
        if mirror == active {
            return None;
        }
        self.backends.get(&mirror)
    }

    pub async fn record_episode(
        &self,
        ctx: &PrincipalContext,
        input: RecordEpisodeInput,
        override_kind: Option<MemoryBackendKind>,
    ) -> Result<MemoryEpisode, OpenRagError> {
        let primary = self.backend(override_kind)?;
        let res = primary.record_episode(ctx, input.clone()).await?;
        if let Some(mirror) = self.mirror_backend(override_kind) {
            let _ = mirror
                .record_episode_with_id(ctx, Some(res.id), input)
                .await?;
        }
        Ok(res)
    }

    pub async fn upsert_entry(
        &self,
        ctx: &PrincipalContext,
        input: UpsertMemoryInput,
        override_kind: Option<MemoryBackendKind>,
    ) -> Result<MemoryEntry, OpenRagError> {
        let primary = self.backend(override_kind)?;
        let res = primary.upsert_entry(ctx, input.clone()).await?;
        if let Some(mirror) = self.mirror_backend(override_kind) {
            let _ = mirror
                .upsert_entry_with_id(ctx, Some(res.id), input)
                .await?;
        }
        Ok(res)
    }

    pub async fn get_entry(
        &self,
        ctx: &PrincipalContext,
        entry_id: Uuid,
        override_kind: Option<MemoryBackendKind>,
    ) -> Result<Option<MemoryEntry>, OpenRagError> {
        self.backend(override_kind)?.get_entry(ctx, entry_id).await
    }

    pub async fn list_entries(
        &self,
        ctx: &PrincipalContext,
        tag_filter: Option<&TagFilter>,
        limit: i64,
        override_kind: Option<MemoryBackendKind>,
    ) -> Result<Vec<MemoryEntry>, OpenRagError> {
        self.backend(override_kind)?
            .list_entries(ctx, tag_filter, limit)
            .await
    }

    pub async fn search(
        &self,
        ctx: &PrincipalContext,
        query: &str,
        tag_filter: Option<&TagFilter>,
        top_k: usize,
        override_kind: Option<MemoryBackendKind>,
    ) -> Result<Vec<MemoryEntry>, OpenRagError> {
        self.backend(override_kind)?
            .search(ctx, query, tag_filter, top_k)
            .await
    }

    pub async fn engage(
        &self,
        ctx: &PrincipalContext,
        req: EngageRequest,
        override_kind: Option<MemoryBackendKind>,
    ) -> Result<EngageResponse, OpenRagError> {
        self.backend(override_kind)?.engage(ctx, req).await
    }

    pub async fn reflect(
        &self,
        ctx: &PrincipalContext,
        req: ReflectRequest,
        override_kind: Option<MemoryBackendKind>,
    ) -> Result<ReflectResponse, OpenRagError> {
        let primary = self.backend(override_kind)?;
        let primary_had_episode = req.episode_id.is_some();
        let res = primary.reflect(ctx, req.clone()).await?;
        if let Some(mirror) = self.mirror_backend(override_kind) {
            let mirror_ep_id = if primary_had_episode {
                None
            } else {
                Some(res.episode_id)
            };
            let _ = mirror
                .reflect_with_ids(ctx, Some(res.entry.id), mirror_ep_id, req)
                .await?;
        }
        Ok(res)
    }

    pub async fn apply_patch(
        &self,
        ctx: &PrincipalContext,
        input: ApplyPatchInput,
        override_kind: Option<MemoryBackendKind>,
    ) -> Result<MemoryPatch, OpenRagError> {
        let primary = self.backend(override_kind)?;
        let forced_new_id = (input.patch_op == PatchOp::Supersede).then(Uuid::new_v4);
        let res = primary
            .apply_patch_with_new_id(ctx, input.clone(), forced_new_id)
            .await?;
        if let Some(mirror) = self.mirror_backend(override_kind) {
            let _ = mirror
                .apply_patch_with_new_id(ctx, input, forced_new_id)
                .await?;
        }
        Ok(res)
    }

    pub async fn promote(
        &self,
        ctx: &PrincipalContext,
        input: PromoteMemoryInput,
        override_kind: Option<MemoryBackendKind>,
    ) -> Result<PromoteResult, OpenRagError> {
        let primary = self.backend(override_kind)?;
        let res = primary.promote(ctx, input.clone()).await?;
        if let Some(mirror) = self.mirror_backend(override_kind) {
            let _ = mirror.promote(ctx, input).await?;
        }
        Ok(res)
    }

    pub async fn add_edge(
        &self,
        ctx: &PrincipalContext,
        src_memory_id: Uuid,
        dst_memory_id: Uuid,
        relation_type: MemoryEdgeType,
        confidence: f32,
        override_kind: Option<MemoryBackendKind>,
    ) -> Result<MemoryEdge, OpenRagError> {
        let primary = self.backend(override_kind)?;
        let res = primary
            .add_edge(ctx, src_memory_id, dst_memory_id, relation_type, confidence)
            .await?;
        if let Some(mirror) = self.mirror_backend(override_kind) {
            let _ = mirror
                .add_edge(ctx, src_memory_id, dst_memory_id, relation_type, confidence)
                .await?;
        }
        Ok(res)
    }

    pub async fn list_edges(
        &self,
        ctx: &PrincipalContext,
        memory_id: Uuid,
        override_kind: Option<MemoryBackendKind>,
    ) -> Result<Vec<MemoryEdge>, OpenRagError> {
        self.backend(override_kind)?
            .list_edges(ctx, memory_id)
            .await
    }
}
