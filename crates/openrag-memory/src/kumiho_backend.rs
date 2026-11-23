use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use async_trait::async_trait;
use chrono::Utc;
use openrag_core::{
    AccessScope, Arn, ArnResourceType, GuardAction, GuardSurface, IamAction, MemoryBackendKind,
    OpenRagError, PrincipalContext,
};
use openrag_iam::{
    GroupDefinition, GroupMembership, IamEvaluator, IamPolicy, ResourceDescriptor,
    ResourceGroupBinding,
};
use openrag_store::{
    cosine_similarity, expand_tag_filter_in_memory, normalize_tag, parse_inline_tags, sha256_hex,
    ExpandedTagFilter, HttpRequest, HttpTransport, TagFacet, TagFilter,
};
use serde_json::{json, Value};
use tokio::sync::RwLock;
use uuid::Uuid;

use crate::concepts::{extract_parenthetical_pairs, InMemoryConceptStore};
use crate::embed::{DeterministicTextEmbedder, TextEmbedder};
use crate::guard_hook::{DefaultMemoryGuard, MemoryGuardHook};
use crate::pg_backend::normalize_tags;
use crate::types::{
    ActorType, ApplyPatchInput, ConceptCategory, ConceptSource, EngageHit, EngageRequest,
    EngageResponse, MemoryEdge, MemoryEdgeType, MemoryEntry, MemoryEpisode, MemoryKind,
    MemoryPatch, MemoryStatus, PatchOp, PromoteMemoryInput, PromoteResult, PromoteTarget,
    RecordEpisodeInput, ReflectRequest, ReflectResponse, SourceHarness, UpsertMemoryInput,
};
use crate::MemoryBackend;

#[derive(Debug, Default)]
struct KumihoGraphState {
    episodes: HashMap<Uuid, MemoryEpisode>,
    entries: HashMap<Uuid, MemoryEntry>,
    vectors: HashMap<Uuid, Vec<f32>>,
    patches: Vec<MemoryPatch>,
    edges: Vec<MemoryEdge>,
    groups: Vec<GroupDefinition>,
    memberships: Vec<GroupMembership>,
    bindings: Vec<ResourceGroupBinding>,
    policies: Vec<IamPolicy>,
}

/// Pluggable Kumiho-compatible cognitive property-graph memory backend (`kumiho` mode).
/// Supports live HTTP/MCP transport (`kumiho_url` + bearer key) as well as a genuine
/// deterministic in-memory property graph enforcing 3-axis IAM, Guard, CCLG patches, and edges.
pub struct KumihoMemoryBackend {
    kumiho_url: Option<String>,
    api_key: Option<String>,
    offline_fallback: bool,
    transport: Option<Arc<dyn HttpTransport>>,
    embedder: Arc<dyn TextEmbedder>,
    guard: Arc<dyn MemoryGuardHook>,
    concepts: InMemoryConceptStore,
    state: Arc<RwLock<KumihoGraphState>>,
}

impl Default for KumihoMemoryBackend {
    fn default() -> Self {
        Self::new_offline()
    }
}

impl KumihoMemoryBackend {
    pub fn new_offline() -> Self {
        let concepts = InMemoryConceptStore::new();
        Self {
            kumiho_url: None,
            api_key: None,
            offline_fallback: true,
            transport: None,
            embedder: Arc::new(DeterministicTextEmbedder::default()),
            guard: Arc::new(DefaultMemoryGuard::default()),
            concepts,
            state: Arc::new(RwLock::new(KumihoGraphState::default())),
        }
    }

    pub fn with_config(
        kumiho_url: Option<String>,
        api_key: Option<String>,
        offline_fallback: bool,
        transport: Option<Arc<dyn HttpTransport>>,
        embedder: Arc<dyn TextEmbedder>,
        guard: Arc<dyn MemoryGuardHook>,
        concepts: InMemoryConceptStore,
    ) -> Self {
        Self {
            kumiho_url: kumiho_url.map(|u| u.trim_end_matches('/').to_string()),
            api_key,
            offline_fallback,
            transport,
            embedder,
            guard,
            concepts,
            state: Arc::new(RwLock::new(KumihoGraphState::default())),
        }
    }

    pub async fn seed_iam_context(
        &self,
        groups: Vec<GroupDefinition>,
        memberships: Vec<GroupMembership>,
        bindings: Vec<ResourceGroupBinding>,
        policies: Vec<IamPolicy>,
    ) {
        let mut guard = self.state.write().await;
        guard.groups = groups;
        guard.memberships = memberships;
        guard.bindings = bindings;
        guard.policies = policies;
    }

    pub async fn get_episode(&self, id: Uuid) -> Option<MemoryEpisode> {
        let guard = self.state.read().await;
        guard.episodes.get(&id).cloned()
    }

    pub async fn list_patches_for_entry(&self, entry_id: Uuid) -> Vec<MemoryPatch> {
        let guard = self.state.read().await;
        guard
            .patches
            .iter()
            .filter(|p| p.memory_entry_id == entry_id)
            .cloned()
            .collect()
    }

    fn can_access_entry(
        state: &KumihoGraphState,
        ctx: &PrincipalContext,
        entry: &MemoryEntry,
        action: IamAction,
    ) -> bool {
        let Ok(arn) = Arn::parse(&entry.arn) else {
            return false;
        };
        let desc = ResourceDescriptor {
            org_id: entry.org_id,
            resource_type: ArnResourceType::Memory,
            resource_id: entry.id,
            dept_id: entry.dept_id,
            owner_principal_id: entry.owner_principal_id,
            access_scope: entry.access_scope,
            allowed_group_ids: entry.allowed_group_ids.clone(),
            denied_group_ids: entry.denied_group_ids.clone(),
            min_auth_level: entry.min_auth_level,
            arn,
        };
        IamEvaluator::evaluate(
            ctx,
            &desc,
            action,
            &state.groups,
            &state.memberships,
            &state.bindings,
            &state.policies,
        )
        .allowed
    }

    fn can_access_episode(
        state: &KumihoGraphState,
        ctx: &PrincipalContext,
        ep: &MemoryEpisode,
        action: IamAction,
    ) -> bool {
        let Ok(arn) = Arn::parse(&ep.arn) else {
            return false;
        };
        let desc = ResourceDescriptor {
            org_id: ep.org_id,
            resource_type: ArnResourceType::Episode,
            resource_id: ep.id,
            dept_id: ep.dept_id,
            owner_principal_id: ep.owner_principal_id,
            access_scope: ep.access_scope,
            allowed_group_ids: ep.allowed_group_ids.clone(),
            denied_group_ids: ep.denied_group_ids.clone(),
            min_auth_level: ep.min_auth_level,
            arn,
        };
        IamEvaluator::evaluate(
            ctx,
            &desc,
            action,
            &state.groups,
            &state.memberships,
            &state.bindings,
            &state.policies,
        )
        .allowed
    }

    fn edge_visible_to(
        state: &KumihoGraphState,
        ctx: &PrincipalContext,
        edge: &MemoryEdge,
    ) -> bool {
        if edge.org_id != ctx.org_id {
            return false;
        }
        let Some(src) = state.entries.get(&edge.src_memory_id) else {
            return false;
        };
        let Some(dst) = state.entries.get(&edge.dst_memory_id) else {
            return false;
        };
        let src_ok = src.org_id == ctx.org_id
            && src.status == MemoryStatus::Active
            && Self::can_access_entry(state, ctx, src, IamAction::Read);
        let dst_ok = dst.org_id == ctx.org_id
            && (dst.status == MemoryStatus::Active
                || (edge.relation_type == MemoryEdgeType::Supersedes
                    && dst.status == MemoryStatus::Superseded))
            && Self::can_access_entry(state, ctx, dst, IamAction::Read);
        src_ok && dst_ok
    }

    fn validate_reflect_targets(
        state: &KumihoGraphState,
        ctx: &PrincipalContext,
        req: &ReflectRequest,
    ) -> Result<(), OpenRagError> {
        if let Some(eid) = req.episode_id {
            let Some(ep) = state.episodes.get(&eid) else {
                return Err(OpenRagError::NotFound(format!("episode {eid} not found")));
            };
            if ep.org_id != ctx.org_id {
                return Err(OpenRagError::NotFound(format!("episode {eid} not found")));
            }
            if !Self::can_access_episode(state, ctx, ep, IamAction::Read) {
                return Err(OpenRagError::Forbidden(
                    "caller lacks read access to episode".into(),
                ));
            }
        }

        if let (Some(etype), Some(target_id)) = (req.edge_type, req.target_memory_id) {
            let Some(target) = state.entries.get(&target_id) else {
                return Err(OpenRagError::NotFound(format!(
                    "target memory {target_id} not found"
                )));
            };
            if target.org_id != ctx.org_id {
                return Err(OpenRagError::NotFound(format!(
                    "target memory {target_id} not found"
                )));
            }
            let required_action = if etype == MemoryEdgeType::Supersedes {
                IamAction::Write
            } else {
                IamAction::Read
            };
            if !Self::can_access_entry(state, ctx, target, required_action) {
                return Err(OpenRagError::Forbidden(format!(
                    "caller lacks {} access on target memory {target_id}",
                    if etype == MemoryEdgeType::Supersedes {
                        "write"
                    } else {
                        "read"
                    }
                )));
            }
            if target.status != MemoryStatus::Active {
                return Err(OpenRagError::Conflict(format!(
                    "target memory {target_id} is not active"
                )));
            }
        }
        Ok(())
    }

    async fn try_remote_rpc(
        &self,
        method: &str,
        params: Value,
    ) -> Result<Option<Value>, OpenRagError> {
        let (Some(url), Some(transport)) = (&self.kumiho_url, &self.transport) else {
            if !self.offline_fallback {
                return Err(OpenRagError::ExternalAdapter {
                    adapter: "kumiho".into(),
                    message: "kumiho_url/transport not configured and offline_fallback=false".into(),
                });
            }
            return Ok(None);
        };

        let body = serde_json::to_vec(&json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": format!("tools/call"),
            "params": {
                "name": method,
                "arguments": params
            }
        }))
        .map_err(|e| OpenRagError::ExternalAdapter {
            adapter: "kumiho".into(),
            message: format!("failed to serialize kumiho RPC: {e}"),
        })?;

        let mut headers = vec![("Content-Type".into(), "application/json".into())];
        if let Some(key) = &self.api_key {
            headers.push(("Authorization".into(), format!("Bearer {key}")));
        }

        match transport
            .send(HttpRequest {
                method: "POST".into(),
                url: format!("{url}/mcp"),
                headers,
                body,
            })
            .await
        {
            Ok(resp) if resp.status < 400 => {
                let parsed: Value = serde_json::from_slice(&resp.body).unwrap_or(Value::Null);
                Ok(Some(parsed))
            }
            Ok(resp) if !self.offline_fallback => Err(OpenRagError::ExternalAdapter {
                adapter: "kumiho".into(),
                message: format!(
                    "kumiho HTTP {} on {method}: {}",
                    resp.status,
                    String::from_utf8_lossy(&resp.body)
                ),
            }),
            Err(e) if !self.offline_fallback => Err(e),
            _ => Ok(None),
        }
    }
    async fn record_episode_impl(
        &self,
        ctx: &PrincipalContext,
        forced_ep_id: Option<Uuid>,
        input: RecordEpisodeInput,
    ) -> Result<MemoryEpisode, OpenRagError> {
        let ep_id = forced_ep_id.unwrap_or_else(Uuid::new_v4);
        let arn = Arn::new(
            &input.org_slug,
            &input.dept_path,
            input.access_scope,
            ArnResourceType::Episode,
            input.min_auth_level,
            ep_id.to_string(),
        )?
        .to_canonical();
        let mut ep = MemoryEpisode {
            id: ep_id,
            org_id: ctx.org_id,
            dept_id: input.dept_id,
            owner_principal_id: ctx.principal_id,
            access_scope: input.access_scope,
            allowed_group_ids: input.allowed_group_ids,
            denied_group_ids: input.denied_group_ids,
            min_auth_level: input.min_auth_level,
            arn,
            source_harness: input.source_harness,
            session_key: input.session_key,
            summary_md: input.summary_md,
            raw_redacted_json: input.raw_redacted_json,
            content_sha256: String::new(),
            redaction_applied: true,
            created_at: Utc::now(),
        };
        ep.content_sha256 = sha256_hex(ep.summary_md.as_bytes());
        let mut guard = self.state.write().await;
        guard.episodes.insert(ep.id, ep.clone());
        Ok(ep)
    }

    async fn upsert_entry_impl(
        &self,
        ctx: &PrincipalContext,
        forced_entry_id: Option<Uuid>,
        mut input: UpsertMemoryInput,
    ) -> Result<MemoryEntry, OpenRagError> {
        let dec = self
            .guard
            .inspect(ctx.org_id, &input.text, GuardSurface::IngestionGate)
            .await?;
        if dec.action == GuardAction::Mask {
            if let Some(sanitized) = dec.sanitized_text {
                input.text = sanitized;
            }
        }
        let id = forced_entry_id.unwrap_or_else(Uuid::new_v4);
        let now = Utc::now();
        let arn = Arn::new(
            &input.org_slug,
            &input.dept_path,
            input.access_scope,
            ArnResourceType::Memory,
            input.min_auth_level,
            id.to_string(),
        )?
        .to_canonical();
        let acl_scope_key = format!(
            "{}:{}:L{}",
            input.dept_path,
            input.access_scope.as_str(),
            input.min_auth_level.as_u8()
        );
        let sha = sha256_hex(input.text.as_bytes());
        let vec = self.embedder.embed_text(&input.text).await?;
        let entry = MemoryEntry {
            id,
            org_id: ctx.org_id,
            bucket_id: input.bucket_id,
            dept_id: input.dept_id,
            owner_principal_id: ctx.principal_id,
            access_scope: input.access_scope,
            allowed_group_ids: input.allowed_group_ids,
            denied_group_ids: input.denied_group_ids,
            min_auth_level: input.min_auth_level,
            arn,
            acl_scope_key,
            memory_kind: input.memory_kind,
            title: input.title,
            effective_text: input.text.clone(),
            bm25_text: input.text,
            content_sha256: sha,
            tags: normalize_tags(&input.tags),
            source_episode_id: input.source_episode_id,
            source_document_id: input.source_document_id,
            promoted_wiki_id: None,
            confidence: input.confidence.unwrap_or(0.8),
            status: MemoryStatus::Active,
            superseded_by: None,
            version: 1,
            valid_from: now,
            valid_to: None,
            created_at: now,
            updated_at: now,
        };
        let mut guard = self.state.write().await;
        if let Some(eid) = entry.source_episode_id {
            let Some(ep) = guard.episodes.get(&eid) else {
                return Err(OpenRagError::NotFound(format!("episode {eid} not found")));
            };
            if ep.org_id != ctx.org_id {
                return Err(OpenRagError::NotFound(format!("episode {eid} not found")));
            }
            if !Self::can_access_episode(&guard, ctx, ep, IamAction::Read) {
                return Err(OpenRagError::Forbidden(
                    "caller lacks read access to source_episode_id".into(),
                ));
            }
        }
        guard.entries.insert(entry.id, entry.clone());
        guard.vectors.insert(entry.id, vec);
        Ok(entry)
    }

    async fn reflect_impl(
        &self,
        ctx: &PrincipalContext,
        forced_entry_id: Option<Uuid>,
        forced_episode_id: Option<Uuid>,
        mut req: ReflectRequest,
    ) -> Result<ReflectResponse, OpenRagError> {
        if req.edge_type.is_some() != req.target_memory_id.is_some() {
            return Err(OpenRagError::Validation(
                "reflect requires both edge_type and target_memory_id".into(),
            ));
        }

        let dec = self
            .guard
            .inspect(ctx.org_id, &req.content, GuardSurface::IngestionGate)
            .await?;
        if dec.action == GuardAction::Mask {
            if let Some(sanitized) = dec.sanitized_text {
                req.content = sanitized;
            }
        }

        let _ = self
            .try_remote_rpc(
                "memory_reflect",
                json!({
                    "org_id": ctx.org_id,
                    "principal_id": ctx.principal_id,
                    "content": req.content,
                    "scope": req.access_scope.as_str()
                }),
            )
            .await?;

        // Pre-validate episode_id and target_memory_id BEFORE mutating any state or concepts
        {
            let read_guard = self.state.read().await;
            Self::validate_reflect_targets(&read_guard, ctx, &req)?;
        }

        let episode_id = req
            .episode_id
            .or(forced_episode_id)
            .unwrap_or_else(Uuid::new_v4);
        let new_episode = if req.episode_id.is_none() {
            let ep_arn = Arn::new(
                &req.org_slug,
                &req.dept_path,
                req.access_scope,
                ArnResourceType::Episode,
                req.min_auth_level,
                episode_id.to_string(),
            )?
            .to_canonical();
            Some(MemoryEpisode {
                id: episode_id,
                org_id: ctx.org_id,
                dept_id: req.dept_id,
                owner_principal_id: ctx.principal_id,
                access_scope: req.access_scope,
                allowed_group_ids: req.allowed_group_ids.clone(),
                denied_group_ids: req.denied_group_ids.clone(),
                min_auth_level: req.min_auth_level,
                arn: ep_arn,
                source_harness: SourceHarness::ChatUi,
                session_key: None,
                summary_md: req.content.clone(),
                raw_redacted_json: None,
                content_sha256: sha256_hex(req.content.as_bytes()),
                redaction_applied: true,
                created_at: Utc::now(),
            })
        } else {
            None
        };

        let entry_id = forced_entry_id.unwrap_or_else(Uuid::new_v4);
        let entry_arn = Arn::new(
            &req.org_slug,
            &req.dept_path,
            req.access_scope,
            ArnResourceType::Memory,
            req.min_auth_level,
            entry_id.to_string(),
        )?
        .to_canonical();
        let acl_scope_key = format!(
            "{}:{}:L{}",
            req.dept_path,
            req.access_scope.as_str(),
            req.min_auth_level.as_u8()
        );
        let sha = sha256_hex(req.content.as_bytes());
        let vec = self.embedder.embed_text(&req.content).await?;
        let now = Utc::now();
        let entry = MemoryEntry {
            id: entry_id,
            org_id: ctx.org_id,
            bucket_id: None,
            dept_id: req.dept_id,
            owner_principal_id: ctx.principal_id,
            access_scope: req.access_scope,
            allowed_group_ids: req.allowed_group_ids.clone(),
            denied_group_ids: req.denied_group_ids.clone(),
            min_auth_level: req.min_auth_level,
            arn: entry_arn,
            acl_scope_key,
            memory_kind: req.memory_kind.unwrap_or(MemoryKind::Fact),
            title: req.title.clone(),
            effective_text: req.content.clone(),
            bm25_text: req.content.clone(),
            content_sha256: sha,
            tags: normalize_tags(&req.tags),
            source_episode_id: Some(episode_id),
            source_document_id: None,
            promoted_wiki_id: None,
            confidence: 0.85,
            status: MemoryStatus::Active,
            superseded_by: None,
            version: 1,
            valid_from: now,
            valid_to: None,
            created_at: now,
            updated_at: now,
        };

        // Atomically re-validate and apply all state mutations under a single write lock
        let edge = {
            let mut guard = self.state.write().await;
            Self::validate_reflect_targets(&guard, ctx, &req)?;

            if let Some(ep) = new_episode {
                guard.episodes.insert(ep.id, ep);
            }
            guard.entries.insert(entry.id, entry.clone());
            guard.vectors.insert(entry.id, vec);

            if let (Some(etype), Some(target_id)) = (req.edge_type, req.target_memory_id) {
                if etype == MemoryEdgeType::Supersedes {
                    if let Some(t_mut) = guard.entries.get_mut(&target_id) {
                        let prev_ver = t_mut.version;
                        let prev_text = t_mut.effective_text.clone();
                        t_mut.status = MemoryStatus::Superseded;
                        t_mut.valid_to = Some(now);
                        t_mut.superseded_by = Some(entry.id);
                        t_mut.version = prev_ver + 1;
                        t_mut.updated_at = now;
                        guard.patches.push(MemoryPatch {
                            id: Uuid::new_v4(),
                            org_id: ctx.org_id,
                            memory_entry_id: target_id,
                            patch_op: PatchOp::Supersede,
                            prev_version: prev_ver,
                            new_version: prev_ver + 1,
                            prev_text: Some(prev_text),
                            new_text: Some(entry.effective_text.clone()),
                            prev_scope: None,
                            new_scope: None,
                            reason: Some("superseded by reflect".into()),
                            actor_type: ActorType::Agent,
                            actor_id: ctx.principal_id.to_string(),
                            created_at: now,
                        });
                    }
                }
                let edge = MemoryEdge {
                    id: Uuid::new_v4(),
                    org_id: ctx.org_id,
                    src_memory_id: entry.id,
                    dst_memory_id: target_id,
                    relation_type: etype,
                    confidence: 1.0,
                    created_at: now,
                };
                guard.edges.push(edge.clone());
                Some(edge)
            } else {
                None
            }
        };

        let extracted_concepts = if req.extract_concepts {
            let pairs = extract_parenthetical_pairs(&req.content);
            for (a, b) in &pairs {
                let _ = self
                    .concepts
                    .link_terms(
                        ctx.org_id,
                        Some(ctx.principal_id),
                        a,
                        b,
                        ConceptCategory::Domain,
                        ConceptSource::IngestParenthetical,
                    )
                    .await?;
            }
            pairs
        } else {
            Vec::new()
        };

        Ok(ReflectResponse {
            entry,
            episode_id,
            edge,
            extracted_concepts,
        })
    }
    async fn expand_tag_filter(&self, org_id: Uuid, tf: &TagFilter) -> ExpandedTagFilter {
        let mut alias_map: HashMap<String, Vec<String>> = HashMap::new();
        if tf.expand_tag_aliases {
            for raw in tf
                .tags_any
                .iter()
                .chain(tf.tags_all.iter())
                .chain(tf.tags_exclude.iter())
            {
                let norm = normalize_tag(raw);
                if !norm.is_empty() && !alias_map.contains_key(&norm) {
                    let exp = self
                        .concepts
                        .expand_terms(org_id, std::slice::from_ref(&norm))
                        .await;
                    alias_map.insert(norm, exp);
                }
            }
        }
        expand_tag_filter_in_memory(tf, |k| alias_map.get(k).cloned().unwrap_or_default())
    }

    async fn apply_patch_impl(
        &self,
        ctx: &PrincipalContext,
        input: ApplyPatchInput,
        forced_new_entry_id: Option<Uuid>,
    ) -> Result<MemoryPatch, OpenRagError> {
        let sanitized_and_vec = if matches!(input.patch_op, PatchOp::Amend | PatchOp::Supersede) {
            let raw_text = input.new_text.as_deref().ok_or_else(|| {
                OpenRagError::Validation("amend/supersede requires non-empty new_text".into())
            })?;
            if raw_text.trim().is_empty() {
                return Err(OpenRagError::Validation(
                    "new_text must not be empty".into(),
                ));
            }
            let dec = self
                .guard
                .inspect(ctx.org_id, raw_text, GuardSurface::IngestionGate)
                .await?;
            let sanitized = if dec.action == GuardAction::Mask {
                dec.sanitized_text.unwrap_or_else(|| raw_text.to_string())
            } else {
                raw_text.to_string()
            };
            let vec = self.embedder.embed_text(&sanitized).await?;
            Some((sanitized, vec))
        } else {
            None
        };

        let mut guard = self.state.write().await;
        let existing = guard
            .entries
            .get(&input.memory_entry_id)
            .cloned()
            .ok_or_else(|| {
                OpenRagError::NotFound(format!("memory entry {} not found", input.memory_entry_id))
            })?;
        if existing.org_id != ctx.org_id {
            return Err(OpenRagError::NotFound(format!(
                "memory entry {} not found",
                input.memory_entry_id
            )));
        }
        if !Self::can_access_entry(&guard, ctx, &existing, IamAction::Write) {
            return Err(OpenRagError::Forbidden(
                "caller lacks write access to memory entry".into(),
            ));
        }
        if existing.status != MemoryStatus::Active {
            return Err(OpenRagError::Conflict(format!(
                "cannot patch non-active memory entry {}",
                existing.id
            )));
        }

        let now = Utc::now();
        let next_ver = existing.version + 1;
        let actor_id = ctx.principal_id.to_string();

        let patch = match input.patch_op {
            PatchOp::Amend => {
                let (text, vec) = sanitized_and_vec.ok_or_else(|| {
                    OpenRagError::Internal("missing sanitized amend payload".into())
                })?;
                let sha = sha256_hex(text.as_bytes());
                if let Some(ent) = guard.entries.get_mut(&existing.id) {
                    ent.effective_text = text.clone();
                    ent.bm25_text = text.clone();
                    ent.content_sha256 = sha;
                    ent.version = next_ver;
                    ent.updated_at = now;
                }
                guard.vectors.insert(existing.id, vec);
                MemoryPatch {
                    id: Uuid::new_v4(),
                    org_id: ctx.org_id,
                    memory_entry_id: existing.id,
                    patch_op: PatchOp::Amend,
                    prev_version: existing.version,
                    new_version: next_ver,
                    prev_text: Some(existing.effective_text),
                    new_text: Some(text),
                    prev_scope: None,
                    new_scope: None,
                    reason: input.reason,
                    actor_type: input.actor_type,
                    actor_id,
                    created_at: now,
                }
            }
            PatchOp::Supersede => {
                let (text, vec) = sanitized_and_vec.ok_or_else(|| {
                    OpenRagError::Internal("missing sanitized supersede payload".into())
                })?;
                let new_id = forced_new_entry_id.unwrap_or_else(Uuid::new_v4);
                let sha = sha256_hex(text.as_bytes());
                let parsed_arn = Arn::parse(&existing.arn)?;
                let new_arn = Arn::new(
                    parsed_arn.org,
                    parsed_arn.dept_path,
                    existing.access_scope,
                    ArnResourceType::Memory,
                    existing.min_auth_level,
                    new_id.to_string(),
                )?
                .to_canonical();
                let new_entry = MemoryEntry {
                    id: new_id,
                    arn: new_arn,
                    effective_text: text.clone(),
                    bm25_text: text.clone(),
                    content_sha256: sha,
                    version: 1,
                    valid_from: now,
                    valid_to: None,
                    created_at: now,
                    updated_at: now,
                    status: MemoryStatus::Active,
                    superseded_by: None,
                    ..existing.clone()
                };
                if let Some(old) = guard.entries.get_mut(&existing.id) {
                    old.status = MemoryStatus::Superseded;
                    old.valid_to = Some(now);
                    old.superseded_by = Some(new_id);
                    old.version = next_ver;
                    old.updated_at = now;
                }
                guard.entries.insert(new_id, new_entry);
                guard.vectors.insert(new_id, vec);
                guard.edges.push(MemoryEdge {
                    id: Uuid::new_v4(),
                    org_id: ctx.org_id,
                    src_memory_id: new_id,
                    dst_memory_id: existing.id,
                    relation_type: MemoryEdgeType::Supersedes,
                    confidence: 1.0,
                    created_at: now,
                });
                MemoryPatch {
                    id: Uuid::new_v4(),
                    org_id: ctx.org_id,
                    memory_entry_id: existing.id,
                    patch_op: PatchOp::Supersede,
                    prev_version: existing.version,
                    new_version: next_ver,
                    prev_text: Some(existing.effective_text),
                    new_text: Some(text),
                    prev_scope: None,
                    new_scope: None,
                    reason: input.reason,
                    actor_type: input.actor_type,
                    actor_id,
                    created_at: now,
                }
            }
            PatchOp::Retract => {
                if let Some(old) = guard.entries.get_mut(&existing.id) {
                    old.status = MemoryStatus::Retracted;
                    old.valid_to = Some(now);
                    old.version = next_ver;
                    old.updated_at = now;
                }
                MemoryPatch {
                    id: Uuid::new_v4(),
                    org_id: ctx.org_id,
                    memory_entry_id: existing.id,
                    patch_op: PatchOp::Retract,
                    prev_version: existing.version,
                    new_version: next_ver,
                    prev_text: Some(existing.effective_text),
                    new_text: None,
                    prev_scope: None,
                    new_scope: None,
                    reason: input.reason,
                    actor_type: input.actor_type,
                    actor_id,
                    created_at: now,
                }
            }
            PatchOp::Redact => {
                if input.hard_purge {
                    for p in &mut guard.patches {
                        if p.memory_entry_id == existing.id {
                            p.prev_text = None;
                            p.new_text = None;
                        }
                    }
                }
                if let Some(old) = guard.entries.get_mut(&existing.id) {
                    old.status = MemoryStatus::Redacted;
                    old.effective_text = "[REDACTED]".into();
                    old.bm25_text = String::new();
                    old.valid_to = Some(now);
                    old.version = next_ver;
                    old.updated_at = now;
                }
                MemoryPatch {
                    id: Uuid::new_v4(),
                    org_id: ctx.org_id,
                    memory_entry_id: existing.id,
                    patch_op: PatchOp::Redact,
                    prev_version: existing.version,
                    new_version: next_ver,
                    prev_text: if input.hard_purge {
                        None
                    } else {
                        Some(existing.effective_text)
                    },
                    new_text: if input.hard_purge {
                        None
                    } else {
                        Some("[REDACTED]".into())
                    },
                    prev_scope: None,
                    new_scope: None,
                    reason: input.reason,
                    actor_type: input.actor_type,
                    actor_id,
                    created_at: now,
                }
            }
            PatchOp::PromoteScope | PatchOp::PromoteWiki => {
                return Err(OpenRagError::Validation(
                    "use MemoryBackend::promote() for promote ops".into(),
                ));
            }
        };

        guard.patches.push(patch.clone());
        Ok(patch)
    }
}

#[async_trait]
impl MemoryBackend for KumihoMemoryBackend {
    fn kind(&self) -> MemoryBackendKind {
        MemoryBackendKind::Kumiho
    }

    async fn record_episode(
        &self,
        ctx: &PrincipalContext,
        input: RecordEpisodeInput,
    ) -> Result<MemoryEpisode, OpenRagError> {
        self.record_episode_impl(ctx, None, input).await
    }

    async fn record_episode_with_id(
        &self,
        ctx: &PrincipalContext,
        episode_id: Option<Uuid>,
        input: RecordEpisodeInput,
    ) -> Result<MemoryEpisode, OpenRagError> {
        self.record_episode_impl(ctx, episode_id, input).await
    }

    async fn upsert_entry(
        &self,
        ctx: &PrincipalContext,
        input: UpsertMemoryInput,
    ) -> Result<MemoryEntry, OpenRagError> {
        self.upsert_entry_impl(ctx, None, input).await
    }

    async fn upsert_entry_with_id(
        &self,
        ctx: &PrincipalContext,
        entry_id: Option<Uuid>,
        input: UpsertMemoryInput,
    ) -> Result<MemoryEntry, OpenRagError> {
        self.upsert_entry_impl(ctx, entry_id, input).await
    }

    async fn get_entry(
        &self,
        ctx: &PrincipalContext,
        entry_id: Uuid,
    ) -> Result<Option<MemoryEntry>, OpenRagError> {
        let guard = self.state.read().await;
        let Some(entry) = guard.entries.get(&entry_id) else {
            return Ok(None);
        };
        if entry.org_id != ctx.org_id || entry.status != MemoryStatus::Active {
            return Ok(None);
        }
        if !Self::can_access_entry(&guard, ctx, entry, IamAction::Read) {
            return Ok(None);
        }
        Ok(Some(entry.clone()))
    }

    async fn list_entries(
        &self,
        ctx: &PrincipalContext,
        tag_filter: Option<&TagFilter>,
        limit: i64,
    ) -> Result<Vec<MemoryEntry>, OpenRagError> {
        if limit <= 0 {
            return Ok(Vec::new());
        }
        let expanded_tags = if let Some(tf) = tag_filter {
            Some(self.expand_tag_filter(ctx.org_id, tf).await)
        } else {
            None
        };

        let guard = self.state.read().await;
        let mut out: Vec<MemoryEntry> = guard
            .entries
            .values()
            .filter(|e| {
                e.org_id == ctx.org_id
                    && e.status == MemoryStatus::Active
                    && Self::can_access_entry(&guard, ctx, e, IamAction::Read)
                    && expanded_tags
                        .as_ref()
                        .is_none_or(|ef| ef.matches_tags(&e.tags))
            })
            .cloned()
            .collect();
        out.sort_by(|a, b| b.updated_at.cmp(&a.updated_at));
        out.truncate(limit as usize);
        Ok(out)
    }

    async fn apply_patch(
        &self,
        ctx: &PrincipalContext,
        input: ApplyPatchInput,
    ) -> Result<MemoryPatch, OpenRagError> {
        self.apply_patch_impl(ctx, input, None).await
    }

    async fn apply_patch_with_new_id(
        &self,
        ctx: &PrincipalContext,
        input: ApplyPatchInput,
        new_entry_id: Option<Uuid>,
    ) -> Result<MemoryPatch, OpenRagError> {
        self.apply_patch_impl(ctx, input, new_entry_id).await
    }

    async fn promote(
        &self,
        ctx: &PrincipalContext,
        input: PromoteMemoryInput,
    ) -> Result<PromoteResult, OpenRagError> {
        let mut guard = self.state.write().await;
        let existing = guard
            .entries
            .get(&input.memory_entry_id)
            .cloned()
            .ok_or_else(|| {
                OpenRagError::NotFound(format!("memory entry {} not found", input.memory_entry_id))
            })?;
        if existing.org_id != ctx.org_id {
            return Err(OpenRagError::NotFound(format!(
                "memory entry {} not found",
                input.memory_entry_id
            )));
        }
        if !Self::can_access_entry(&guard, ctx, &existing, IamAction::Write) {
            return Err(OpenRagError::Forbidden(
                "caller lacks write access to memory entry".into(),
            ));
        }
        if existing.status != MemoryStatus::Active {
            return Err(OpenRagError::Conflict(
                "cannot promote non-active memory entry".into(),
            ));
        }

        if let Some(lvl) = input.min_auth_level {
            if lvl > ctx.auth_level {
                return Err(OpenRagError::Forbidden(format!(
                    "caller auth_level {} cannot raise min_auth_level to {}",
                    ctx.auth_level, lvl
                )));
            }
        }

        let cur_rank = match existing.access_scope {
            AccessScope::Private => 0,
            AccessScope::Dept => 1,
            AccessScope::Group => 2,
            AccessScope::Org => 3,
        };
        let dst_rank = match input.target {
            PromoteTarget::Private => 0,
            PromoteTarget::Dept => 1,
            PromoteTarget::Group => 2,
            PromoteTarget::Org => 3,
            PromoteTarget::Wiki => 4,
        };
        if dst_rank <= cur_rank {
            return Err(OpenRagError::Validation(
                "promote target must be higher than current access_scope".into(),
            ));
        }
        if matches!(input.target, PromoteTarget::Group)
            && input.target_group_ids.is_empty()
            && existing.allowed_group_ids.is_empty()
        {
            return Err(OpenRagError::Validation(
                "promoting to 'group' scope requires non-empty target_group_ids".into(),
            ));
        }

        let now = Utc::now();
        let next_ver = existing.version + 1;
        let actor_id = ctx.principal_id.to_string();

        let (updated, patch, wiki_id) = match input.target {
            PromoteTarget::Wiki => {
                let wid = Uuid::new_v4();
                if let Some(ent) = guard.entries.get_mut(&existing.id) {
                    ent.promoted_wiki_id = Some(wid);
                    ent.version = next_ver;
                    ent.updated_at = now;
                }
                let updated = guard
                    .entries
                    .get(&existing.id)
                    .cloned()
                    .ok_or_else(|| OpenRagError::Internal("missing entry".into()))?;
                let patch = MemoryPatch {
                    id: Uuid::new_v4(),
                    org_id: ctx.org_id,
                    memory_entry_id: existing.id,
                    patch_op: PatchOp::PromoteWiki,
                    prev_version: existing.version,
                    new_version: next_ver,
                    prev_text: None,
                    new_text: input.wiki_slug,
                    prev_scope: Some(existing.access_scope.as_str().into()),
                    new_scope: Some("wiki".into()),
                    reason: input.reason,
                    actor_type: ActorType::Human,
                    actor_id,
                    created_at: now,
                };
                (updated, patch, Some(wid))
            }
            other => {
                let new_scope = match other {
                    PromoteTarget::Private => AccessScope::Private,
                    PromoteTarget::Dept => AccessScope::Dept,
                    PromoteTarget::Group => AccessScope::Group,
                    PromoteTarget::Org | PromoteTarget::Wiki => AccessScope::Org,
                };
                let new_arn = Arn::new(
                    &input.org_slug,
                    &input.dept_path,
                    new_scope,
                    ArnResourceType::Memory,
                    input.min_auth_level.unwrap_or(existing.min_auth_level),
                    existing.id.to_string(),
                )?
                .to_canonical();
                if let Some(ent) = guard.entries.get_mut(&existing.id) {
                    ent.access_scope = new_scope;
                    ent.arn = new_arn;
                    if !input.target_group_ids.is_empty() {
                        ent.allowed_group_ids = input.target_group_ids;
                    }
                    if let Some(lvl) = input.min_auth_level {
                        ent.min_auth_level = lvl;
                    }
                    ent.version = next_ver;
                    ent.updated_at = now;
                }
                let updated = guard
                    .entries
                    .get(&existing.id)
                    .cloned()
                    .ok_or_else(|| OpenRagError::Internal("missing entry".into()))?;
                let patch = MemoryPatch {
                    id: Uuid::new_v4(),
                    org_id: ctx.org_id,
                    memory_entry_id: existing.id,
                    patch_op: PatchOp::PromoteScope,
                    prev_version: existing.version,
                    new_version: next_ver,
                    prev_text: None,
                    new_text: None,
                    prev_scope: Some(existing.access_scope.as_str().into()),
                    new_scope: Some(new_scope.as_str().into()),
                    reason: input.reason,
                    actor_type: ActorType::Human,
                    actor_id,
                    created_at: now,
                };
                (updated, patch, None)
            }
        };

        guard.patches.push(patch.clone());
        Ok(PromoteResult {
            entry: updated,
            patch,
            promoted_wiki_id: wiki_id,
        })
    }

    async fn search(
        &self,
        ctx: &PrincipalContext,
        query: &str,
        tag_filter: Option<&TagFilter>,
        top_k: usize,
    ) -> Result<Vec<MemoryEntry>, OpenRagError> {
        let resp = self
            .engage(
                ctx,
                EngageRequest {
                    query: query.to_string(),
                    scope: None,
                    top_k,
                    tag_filter: tag_filter.cloned(),
                },
            )
            .await?;
        Ok(resp.memories.into_iter().map(|h| h.entry).collect())
    }

    async fn engage(
        &self,
        ctx: &PrincipalContext,
        req: EngageRequest,
    ) -> Result<EngageResponse, OpenRagError> {
        if req.top_k == 0 {
            return Ok(EngageResponse {
                memories: Vec::new(),
                expanded_terms: Vec::new(),
                tag_facets: Vec::new(),
            });
        }
        let _ = self
            .guard
            .inspect(ctx.org_id, &req.query, GuardSurface::InboundQuery)
            .await?;

        let _ = self
            .try_remote_rpc(
                "memory_engage",
                json!({
                    "org_id": ctx.org_id,
                    "principal_id": ctx.principal_id,
                    "query": req.query,
                    "top_k": req.top_k
                }),
            )
            .await?;

        let parsed = parse_inline_tags(&req.query);
        let mut combined_tf = req.tag_filter.clone().unwrap_or_default();
        for t in parsed.inline_tags {
            if !combined_tf.tags_any.contains(&t) {
                combined_tf.tags_any.push(t);
            }
        }
        for t in parsed.excluded_tags {
            if !combined_tf.tags_exclude.contains(&t) {
                combined_tf.tags_exclude.push(t);
            }
        }

        let raw_terms: Vec<String> = parsed
            .clean_query
            .split_whitespace()
            .map(ToString::to_string)
            .collect();
        let expanded_terms = self.concepts.expand_terms(ctx.org_id, &raw_terms).await;
        let expanded_tags = if combined_tf.is_empty() {
            None
        } else {
            Some(self.expand_tag_filter(ctx.org_id, &combined_tf).await)
        };

        let query_str = if expanded_terms.is_empty() {
            req.query.clone()
        } else {
            expanded_terms.join(" ")
        };
        let qvec = self.embedder.embed_query(&query_str).await?;

        let guard = self.state.read().await;
        let lower_terms: Vec<String> = expanded_terms.iter().map(|s| s.to_lowercase()).collect();
        let mut facet_counts: BTreeMap<String, i64> = BTreeMap::new();
        let mut hits = Vec::new();

        for entry in guard.entries.values() {
            if entry.status != MemoryStatus::Active {
                continue;
            }
            if !Self::can_access_entry(&guard, ctx, entry, IamAction::Read) {
                continue;
            }
            for t in &entry.tags {
                *facet_counts.entry(t.clone()).or_insert(0) += 1;
            }
            if let Some(sc) = req.scope {
                if entry.access_scope != sc {
                    continue;
                }
            }
            if let Some(ef) = &expanded_tags {
                if !ef.matches_tags(&entry.tags) {
                    continue;
                }
            }

            let vec_sim = match guard.vectors.get(&entry.id) {
                Some(ev) => cosine_similarity(&qvec, ev)?,
                None => 0.0,
            };
            let lower_text = entry.effective_text.to_lowercase();
            let lex_hits = lower_terms
                .iter()
                .filter(|t| !t.is_empty() && lower_text.contains(t.as_str()))
                .count();
            let lex_boost = if lower_terms.is_empty() {
                0.0
            } else {
                (lex_hits as f32 / lower_terms.len() as f32) * 0.5
            };

            let mut sanitized_entry = entry.clone();
            match self
                .guard
                .inspect(ctx.org_id, &sanitized_entry.effective_text, GuardSurface::McpEgress)
                .await
            {
                Ok(dec) => {
                    if dec.action == GuardAction::Mask {
                        if let Some(s) = dec.sanitized_text {
                            sanitized_entry.effective_text = s;
                        }
                    }
                }
                Err(OpenRagError::GuardBlocked { .. }) => continue,
                Err(e) => return Err(e),
            }

            let edges: Vec<MemoryEdge> = guard
                .edges
                .iter()
                .filter(|edge| {
                    (edge.src_memory_id == entry.id || edge.dst_memory_id == entry.id)
                        && Self::edge_visible_to(&guard, ctx, edge)
                })
                .cloned()
                .collect();

            hits.push(EngageHit {
                entry: sanitized_entry,
                score: vec_sim + lex_boost,
                edges,
            });
        }

        hits.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| b.entry.updated_at.cmp(&a.entry.updated_at))
        });
        hits.truncate(req.top_k);

        let mut tag_facets: Vec<TagFacet> = facet_counts
            .into_iter()
            .map(|(tag, count)| TagFacet { tag, count })
            .collect();
        tag_facets.sort_by(|a, b| b.count.cmp(&a.count).then_with(|| a.tag.cmp(&b.tag)));

        Ok(EngageResponse {
            memories: hits,
            expanded_terms,
            tag_facets,
        })
    }

    async fn reflect(
        &self,
        ctx: &PrincipalContext,
        req: ReflectRequest,
    ) -> Result<ReflectResponse, OpenRagError> {
        self.reflect_impl(ctx, None, None, req).await
    }

    async fn reflect_with_ids(
        &self,
        ctx: &PrincipalContext,
        entry_id: Option<Uuid>,
        episode_id: Option<Uuid>,
        req: ReflectRequest,
    ) -> Result<ReflectResponse, OpenRagError> {
        self.reflect_impl(ctx, entry_id, episode_id, req).await
    }

    async fn add_edge(
        &self,
        ctx: &PrincipalContext,
        src_memory_id: Uuid,
        dst_memory_id: Uuid,
        relation_type: MemoryEdgeType,
        confidence: f32,
    ) -> Result<MemoryEdge, OpenRagError> {
        if !confidence.is_finite() || !(0.0..=1.0).contains(&confidence) {
            return Err(OpenRagError::Validation(format!(
                "edge confidence must be finite and within [0.0, 1.0], got {confidence}"
            )));
        }
        if relation_type == MemoryEdgeType::Supersedes && src_memory_id == dst_memory_id {
            return Err(OpenRagError::Validation(
                "memory entry cannot supersede itself".into(),
            ));
        }

        let mut guard = self.state.write().await;
        let src = guard
            .entries
            .get(&src_memory_id)
            .filter(|e| e.org_id == ctx.org_id)
            .cloned()
            .ok_or_else(|| OpenRagError::NotFound(format!("src memory {src_memory_id} not found")))?;
        let dst = guard
            .entries
            .get(&dst_memory_id)
            .filter(|e| e.org_id == ctx.org_id)
            .cloned()
            .ok_or_else(|| OpenRagError::NotFound(format!("dst memory {dst_memory_id} not found")))?;
        if !Self::can_access_entry(&guard, ctx, &src, IamAction::Write)
            || !Self::can_access_entry(&guard, ctx, &dst, IamAction::Read)
        {
            return Err(OpenRagError::Forbidden(
                "caller lacks write(src)/read(dst) for memory edge".into(),
            ));
        }
        if src.status != MemoryStatus::Active {
            return Err(OpenRagError::Conflict(format!(
                "src memory {src_memory_id} is not active"
            )));
        }
        let dst_status_ok = dst.status == MemoryStatus::Active
            || (relation_type == MemoryEdgeType::Supersedes
                && dst.status == MemoryStatus::Superseded);
        if !dst_status_ok {
            return Err(OpenRagError::Conflict(format!(
                "dst memory {dst_memory_id} is not active"
            )));
        }

        if let Some(existing) = guard.edges.iter_mut().find(|e| {
            e.org_id == ctx.org_id
                && e.src_memory_id == src_memory_id
                && e.dst_memory_id == dst_memory_id
                && e.relation_type == relation_type
        }) {
            existing.confidence = confidence;
            return Ok(existing.clone());
        }

        let edge = MemoryEdge {
            id: Uuid::new_v4(),
            org_id: ctx.org_id,
            src_memory_id,
            dst_memory_id,
            relation_type,
            confidence,
            created_at: Utc::now(),
        };
        guard.edges.push(edge.clone());
        Ok(edge)
    }

    async fn list_edges(
        &self,
        ctx: &PrincipalContext,
        memory_id: Uuid,
    ) -> Result<Vec<MemoryEdge>, OpenRagError> {
        let guard = self.state.read().await;
        let out = guard
            .edges
            .iter()
            .filter(|e| {
                (e.src_memory_id == memory_id || e.dst_memory_id == memory_id)
                    && Self::edge_visible_to(&guard, ctx, e)
            })
            .cloned()
            .collect();
        Ok(out)
    }
}
