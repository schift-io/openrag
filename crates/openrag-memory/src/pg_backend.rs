use std::collections::{BTreeSet, HashMap};
use std::sync::Arc;

use async_trait::async_trait;
use openrag_core::{
    AccessScope, Arn, ArnResourceType, AuthLevel, GuardAction, GuardSurface, MemoryBackendKind,
    OpenRagError, PrincipalContext,
};
use openrag_db::{DbPool, RlsTransaction};
use openrag_store::{
    compute_tag_facets_rls, cosine_similarity, expand_tag_filter_rls, normalize_tag,
    parse_inline_tags, push_tag_filter_predicates, sha256_hex, EnqueueOutboxItem,
    OutboxEnqueuer, OutboxOp, TagFilter, TagTable, VectorTable,
};
use pgvector::Vector;
use serde_json::json;
use sqlx::{Postgres, QueryBuilder, Row};
use uuid::Uuid;

use crate::backend::MemoryBackend;
use crate::concepts::ConceptGraph;
use crate::embed::{DeterministicTextEmbedder, TextEmbedder};
use crate::guard_hook::{DefaultMemoryGuard, MemoryGuardHook};
use crate::types::{
    ActorType, ApplyPatchInput, EngageHit, EngageRequest, EngageResponse, MemoryEdge,
    MemoryEdgeType, MemoryEntry, MemoryEpisode, MemoryKind, MemoryPatch, MemoryStatus, PatchOp,
    PromoteMemoryInput, PromoteResult, PromoteTarget, RecordEpisodeInput, ReflectRequest,
    ReflectResponse, UpsertMemoryInput, WikiEditorType, WikiSourceType,
};

pub struct PostgresMemoryBackend {
    pool: DbPool,
    embedder: Arc<dyn TextEmbedder>,
    guard: Arc<dyn MemoryGuardHook>,
}

impl PostgresMemoryBackend {
    pub fn new(pool: DbPool) -> Self {
        Self {
            pool,
            embedder: Arc::new(DeterministicTextEmbedder::default()),
            guard: Arc::new(DefaultMemoryGuard::default()),
        }
    }

    pub fn with_components(
        pool: DbPool,
        embedder: Arc<dyn TextEmbedder>,
        guard: Arc<dyn MemoryGuardHook>,
    ) -> Self {
        Self {
            pool,
            embedder,
            guard,
        }
    }

    fn scope_rank(scope: AccessScope) -> u8 {
        match scope {
            AccessScope::Private => 0,
            AccessScope::Dept => 1,
            AccessScope::Group => 2,
            AccessScope::Org => 3,
        }
    }

    fn target_rank(target: PromoteTarget) -> u8 {
        match target {
            PromoteTarget::Private => 0,
            PromoteTarget::Dept => 1,
            PromoteTarget::Group => 2,
            PromoteTarget::Org => 3,
            PromoteTarget::Wiki => 4,
        }
    }

    async fn fetch_active_entry_rls(
        tx: &mut RlsTransaction<'_>,
        org_id: Uuid,
        entry_id: Uuid,
    ) -> Result<Option<MemoryEntry>, OpenRagError> {
        let row = sqlx::query(
            "SELECT id, org_id, bucket_id, dept_id, owner_principal_id, access_scope,
                    allowed_group_ids, denied_group_ids, min_auth_level, arn, acl_scope_key,
                    memory_kind, title, effective_text, bm25_text, content_sha256, tags,
                    source_episode_id, source_document_id, promoted_wiki_id, confidence,
                    status, superseded_by, version, valid_from, valid_to, created_at, updated_at
             FROM memory.entries
             WHERE org_id = $1 AND id = $2",
        )
        .bind(org_id)
        .bind(entry_id)
        .fetch_optional(&mut **tx.tx_mut())
        .await
        .map_err(|e| OpenRagError::Database(format!("failed to fetch memory.entry: {e}")))?;
        row.as_ref().map(row_to_entry).transpose()
    }

    async fn insert_entry_rls(
        &self,
        tx: &mut RlsTransaction<'_>,
        ctx: &PrincipalContext,
        forced_entry_id: Option<Uuid>,
        input: &UpsertMemoryInput,
    ) -> Result<MemoryEntry, OpenRagError> {
        let entry_id = forced_entry_id.unwrap_or_else(Uuid::new_v4);
        let arn = Arn::new(
            &input.org_slug,
            &input.dept_path,
            input.access_scope,
            ArnResourceType::Memory,
            input.min_auth_level,
            entry_id.to_string(),
        )?
        .to_canonical();
        let acl_scope_key = format!(
            "{}:{}:L{}",
            input.dept_path,
            input.access_scope.as_str(),
            input.min_auth_level.as_u8()
        );
        let content_sha256 = sha256_hex(input.text.as_bytes());
        let emb = self.embedder.embed_text(&input.text).await?;
        let norm_tags = normalize_tags(&input.tags);
        let confidence = input.confidence.unwrap_or(0.8);

        let row = sqlx::query(
            "INSERT INTO memory.entries
             (id, org_id, bucket_id, dept_id, owner_principal_id, access_scope,
              allowed_group_ids, denied_group_ids, min_auth_level, arn, acl_scope_key,
              memory_kind, title, effective_text, bm25_text, content_sha256, embedding,
              embedding_model, tags, source_episode_id, source_document_id, confidence)
             VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22)
             RETURNING id, org_id, bucket_id, dept_id, owner_principal_id, access_scope,
                       allowed_group_ids, denied_group_ids, min_auth_level, arn, acl_scope_key,
                       memory_kind, title, effective_text, bm25_text, content_sha256, tags,
                       source_episode_id, source_document_id, promoted_wiki_id, confidence,
                       status, superseded_by, version, valid_from, valid_to, created_at, updated_at",
        )
        .bind(entry_id)
        .bind(ctx.org_id)
        .bind(input.bucket_id)
        .bind(input.dept_id)
        .bind(ctx.principal_id)
        .bind(input.access_scope.as_str())
        .bind(&input.allowed_group_ids)
        .bind(&input.denied_group_ids)
        .bind(input.min_auth_level.as_i16())
        .bind(&arn)
        .bind(&acl_scope_key)
        .bind(input.memory_kind.as_str())
        .bind(&input.title)
        .bind(&input.text)
        .bind(&input.text)
        .bind(&content_sha256)
        .bind(Vector::from(emb))
        .bind(self.embedder.model_name())
        .bind(&norm_tags)
        .bind(input.source_episode_id)
        .bind(input.source_document_id)
        .bind(confidence)
        .fetch_one(&mut **tx.tx_mut())
        .await
        .map_err(|e| map_pg_rls_err(e, "insert memory.entries"))?;

        let entry = row_to_entry(&row)?;

        OutboxEnqueuer::enqueue_rls(
            tx,
            &EnqueueOutboxItem {
                org_id: ctx.org_id,
                target_table: VectorTable::MemoryEntries,
                target_id: entry.id,
                op: OutboxOp::UpsertEmbed,
                content_sha256: Some(entry.content_sha256.clone()),
                priority: 5,
                payload: json!({ "memory_kind": entry.memory_kind.as_str() }),
                idempotency_key: None,
                max_attempts: None,
            },
            false,
        )
        .await?;

        Ok(entry)
    }

    #[allow(clippy::too_many_arguments)]
    async fn insert_patch_rls(
        tx: &mut RlsTransaction<'_>,
        org_id: Uuid,
        entry_id: Uuid,
        op: PatchOp,
        prev_version: i32,
        new_version: i32,
        prev_text: Option<&str>,
        new_text: Option<&str>,
        prev_scope: Option<&str>,
        new_scope: Option<&str>,
        reason: Option<&str>,
        actor_type: ActorType,
        actor_id: &str,
    ) -> Result<MemoryPatch, OpenRagError> {
        let row = sqlx::query(
            "INSERT INTO memory.patches
             (org_id, memory_entry_id, patch_op, prev_version, new_version,
              prev_text, new_text, prev_scope, new_scope, reason, actor_type, actor_id)
             VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
             RETURNING id, org_id, memory_entry_id, patch_op, prev_version, new_version,
                       prev_text, new_text, prev_scope, new_scope, reason, actor_type, actor_id, created_at",
        )
        .bind(org_id)
        .bind(entry_id)
        .bind(op.as_str())
        .bind(prev_version)
        .bind(new_version)
        .bind(prev_text)
        .bind(new_text)
        .bind(prev_scope)
        .bind(new_scope)
        .bind(reason)
        .bind(actor_type.as_str())
        .bind(actor_id)
        .fetch_one(&mut **tx.tx_mut())
        .await
        .map_err(|e| map_pg_rls_err(e, "insert memory.patches"))?;

        row_to_patch(&row)
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
        let sha = sha256_hex(input.summary_md.as_bytes());

        let mut tx = self.pool.begin_rls_tx(ctx).await?;
        let row = sqlx::query(
            "INSERT INTO memory.episodes
             (id, org_id, dept_id, owner_principal_id, access_scope, allowed_group_ids,
              denied_group_ids, min_auth_level, arn, source_harness, session_key,
              summary_md, raw_redacted_json, content_sha256, redaction_applied)
             VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, true)
             RETURNING id, org_id, dept_id, owner_principal_id, access_scope, allowed_group_ids,
                       denied_group_ids, min_auth_level, arn, source_harness, session_key,
                       summary_md, raw_redacted_json, content_sha256, redaction_applied, created_at",
        )
        .bind(ep_id)
        .bind(ctx.org_id)
        .bind(input.dept_id)
        .bind(ctx.principal_id)
        .bind(input.access_scope.as_str())
        .bind(&input.allowed_group_ids)
        .bind(&input.denied_group_ids)
        .bind(input.min_auth_level.as_i16())
        .bind(&arn)
        .bind(input.source_harness.as_str())
        .bind(&input.session_key)
        .bind(&input.summary_md)
        .bind(&input.raw_redacted_json)
        .bind(&sha)
        .fetch_one(&mut **tx.tx_mut())
        .await
        .map_err(|e| map_pg_rls_err(e, "insert memory.episodes"))?;

        let ep = row_to_episode(&row)?;
        tx.commit().await?;
        Ok(ep)
    }

    async fn upsert_entry_impl(
        &self,
        ctx: &PrincipalContext,
        forced_entry_id: Option<Uuid>,
        mut input: UpsertMemoryInput,
    ) -> Result<MemoryEntry, OpenRagError> {
        let decision = self
            .guard
            .inspect(ctx.org_id, &input.text, GuardSurface::IngestionGate)
            .await?;
        if decision.action == GuardAction::Mask {
            if let Some(sanitized) = decision.sanitized_text {
                input.text = sanitized;
            }
        }
        let mut tx = self.pool.begin_rls_tx(ctx).await?;
        let entry = self
            .insert_entry_rls(&mut tx, ctx, forced_entry_id, &input)
            .await?;
        tx.commit().await?;
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
                "reflect requires both edge_type and target_memory_id when linking an edge".into(),
            ));
        }

        // Guard BEFORE opening DB transaction -> zero writes if blocked
        let guard_dec = self
            .guard
            .inspect(ctx.org_id, &req.content, GuardSurface::IngestionGate)
            .await?;
        if guard_dec.action == GuardAction::Mask {
            if let Some(sanitized) = guard_dec.sanitized_text {
                req.content = sanitized;
            }
        }

        let mut tx = self.pool.begin_rls_tx(ctx).await?;

        let episode_id = if let Some(eid) = req.episode_id {
            eid
        } else {
            let ep_id = forced_episode_id.unwrap_or_else(Uuid::new_v4);
            let ep_arn = Arn::new(
                &req.org_slug,
                &req.dept_path,
                req.access_scope,
                ArnResourceType::Episode,
                req.min_auth_level,
                ep_id.to_string(),
            )?
            .to_canonical();
            let ep_sha = sha256_hex(req.content.as_bytes());
            sqlx::query(
                "INSERT INTO memory.episodes
                 (id, org_id, dept_id, owner_principal_id, access_scope, allowed_group_ids,
                  denied_group_ids, min_auth_level, arn, source_harness, summary_md, content_sha256)
                 VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, 'chat_ui', $10, $11)",
            )
            .bind(ep_id)
            .bind(ctx.org_id)
            .bind(req.dept_id)
            .bind(ctx.principal_id)
            .bind(req.access_scope.as_str())
            .bind(&req.allowed_group_ids)
            .bind(&req.denied_group_ids)
            .bind(req.min_auth_level.as_i16())
            .bind(&ep_arn)
            .bind(&req.content)
            .bind(&ep_sha)
            .execute(&mut **tx.tx_mut())
            .await
            .map_err(|e| map_pg_rls_err(e, "insert reflect episode"))?;
            ep_id
        };

        let extracted_concepts = if req.extract_concepts {
            ConceptGraph::ingest_parenthetical_pairs_rls(
                &mut tx,
                ctx.org_id,
                Some(ctx.principal_id),
                &req.content,
            )
            .await?
        } else {
            Vec::new()
        };

        let entry = self
            .insert_entry_rls(
                &mut tx,
                ctx,
                forced_entry_id,
                &UpsertMemoryInput {
                    bucket_id: None,
                    dept_id: req.dept_id,
                    access_scope: req.access_scope,
                    allowed_group_ids: req.allowed_group_ids.clone(),
                    denied_group_ids: req.denied_group_ids.clone(),
                    min_auth_level: req.min_auth_level,
                    org_slug: req.org_slug.clone(),
                    dept_path: req.dept_path.clone(),
                    memory_kind: req.memory_kind.unwrap_or(MemoryKind::Fact),
                    title: req.title.clone(),
                    text: req.content.clone(),
                    tags: req.tags.clone(),
                    source_episode_id: Some(episode_id),
                    source_document_id: None,
                    confidence: Some(0.85),
                },
            )
            .await?;

        let edge = if let (Some(etype), Some(target_id)) = (req.edge_type, req.target_memory_id) {
            let sup_target = if etype == MemoryEdgeType::Supersedes {
                let target_opt = Self::fetch_active_entry_rls(&mut tx, ctx.org_id, target_id).await?;
                let target = target_opt.ok_or_else(|| {
                    OpenRagError::NotFound(format!("target memory {target_id} not found or inaccessible"))
                })?;
                let actor_id = ctx.principal_id.to_string();
                Self::insert_patch_rls(
                    &mut tx,
                    ctx.org_id,
                    target.id,
                    PatchOp::Supersede,
                    target.version,
                    target.version + 1,
                    Some(&target.effective_text),
                    Some(&entry.effective_text),
                    None,
                    None,
                    Some("superseded by reflect"),
                    ActorType::Agent,
                    &actor_id,
                )
                .await?;
                Some(target)
            } else {
                None
            };

            let erow = sqlx::query(
                "INSERT INTO memory.edges (org_id, src_memory_id, dst_memory_id, relation_type, confidence)
                 VALUES ($1, $2, $3, $4, 1.0)
                 ON CONFLICT (org_id, src_memory_id, dst_memory_id, relation_type)
                 DO UPDATE SET confidence = EXCLUDED.confidence
                 RETURNING id, org_id, src_memory_id, dst_memory_id, relation_type, confidence, created_at",
            )
            .bind(ctx.org_id)
            .bind(entry.id)
            .bind(target_id)
            .bind(etype.as_str())
            .fetch_one(&mut **tx.tx_mut())
            .await
            .map_err(|e| map_pg_rls_err(e, "insert memory.edges in reflect"))?;

            if let Some(target) = sup_target {
                sqlx::query(
                    "SELECT iam.memory_transition_status($1, $2, 'superseded', $3, $4)",
                )
                .bind(ctx.org_id)
                .bind(target.id)
                .bind(entry.id)
                .bind(target.version + 1)
                .execute(&mut **tx.tx_mut())
                .await
                .map_err(|e| map_pg_rls_err(e, "supersede target in reflect"))?;
            }
            Some(row_to_edge(&erow)?)
        } else {
            None
        };

        tx.commit().await?;
        Ok(ReflectResponse {
            entry,
            episode_id,
            edge,
            extracted_concepts,
        })
    }

    async fn apply_patch_impl(
        &self,
        ctx: &PrincipalContext,
        input: ApplyPatchInput,
        forced_new_entry_id: Option<Uuid>,
    ) -> Result<MemoryPatch, OpenRagError> {
        let mut tx = self.pool.begin_rls_tx(ctx).await?;
        let Some(entry) = Self::fetch_active_entry_rls(&mut tx, ctx.org_id, input.memory_entry_id).await? else {
            return Err(OpenRagError::Conflict(format!(
                "memory entry {} is not active (retracted/superseded/redacted) or inaccessible",
                input.memory_entry_id
            )));
        };
        if entry.status != MemoryStatus::Active {
            return Err(OpenRagError::Conflict(format!(
                "cannot patch non-active memory entry {}",
                entry.id
            )));
        }

        let actor_id = ctx.principal_id.to_string();
        let patch = match input.patch_op {
            PatchOp::Amend => {
                let raw_text = input.new_text.as_deref().ok_or_else(|| {
                    OpenRagError::Validation("amend requires non-empty new_text".into())
                })?;
                if raw_text.trim().is_empty() {
                    return Err(OpenRagError::Validation(
                        "amend new_text must not be empty".into(),
                    ));
                }
                let guard_dec = self
                    .guard
                    .inspect(ctx.org_id, raw_text, GuardSurface::IngestionGate)
                    .await?;
                let next_text = if guard_dec.action == GuardAction::Mask {
                    guard_dec.sanitized_text.unwrap_or_else(|| raw_text.to_string())
                } else {
                    raw_text.to_string()
                };
                let next_sha = sha256_hex(next_text.as_bytes());
                let next_emb = self.embedder.embed_text(&next_text).await?;
                let next_ver = entry.version + 1;

                sqlx::query(
                    "UPDATE memory.entries
                     SET effective_text = $1, bm25_text = $2, content_sha256 = $3,
                         embedding = $4, version = $5, updated_at = now()
                     WHERE org_id = $6 AND id = $7",
                )
                .bind(&next_text)
                .bind(&next_text)
                .bind(&next_sha)
                .bind(Vector::from(next_emb))
                .bind(next_ver)
                .bind(ctx.org_id)
                .bind(entry.id)
                .execute(&mut **tx.tx_mut())
                .await
                .map_err(|e| map_pg_rls_err(e, "amend memory.entries"))?;

                let p = Self::insert_patch_rls(
                    &mut tx,
                    ctx.org_id,
                    entry.id,
                    PatchOp::Amend,
                    entry.version,
                    next_ver,
                    Some(&entry.effective_text),
                    Some(&next_text),
                    None,
                    None,
                    input.reason.as_deref(),
                    input.actor_type,
                    &actor_id,
                )
                .await?;

                OutboxEnqueuer::enqueue_rls(
                    &mut tx,
                    &EnqueueOutboxItem {
                        org_id: ctx.org_id,
                        target_table: VectorTable::MemoryEntries,
                        target_id: entry.id,
                        op: OutboxOp::UpsertEmbed,
                        content_sha256: Some(next_sha),
                        priority: 3,
                        payload: json!({ "patch_op": "amend" }),
                        idempotency_key: None,
                        max_attempts: None,
                    },
                    false,
                )
                .await?;
                p
            }
            PatchOp::Supersede => {
                let raw_text = input.new_text.as_deref().ok_or_else(|| {
                    OpenRagError::Validation("supersede requires non-empty new_text".into())
                })?;
                if raw_text.trim().is_empty() {
                    return Err(OpenRagError::Validation(
                        "supersede new_text must not be empty".into(),
                    ));
                }
                let guard_dec = self
                    .guard
                    .inspect(ctx.org_id, raw_text, GuardSurface::IngestionGate)
                    .await?;
                let next_text = if guard_dec.action == GuardAction::Mask {
                    guard_dec.sanitized_text.unwrap_or_else(|| raw_text.to_string())
                } else {
                    raw_text.to_string()
                };

                let new_id = forced_new_entry_id.unwrap_or_else(Uuid::new_v4);
                let new_sha = sha256_hex(next_text.as_bytes());
                let new_emb = self.embedder.embed_text(&next_text).await?;
                let parsed_old_arn = Arn::parse(&entry.arn)?;
                let new_arn = Arn::new(
                    parsed_old_arn.org,
                    parsed_old_arn.dept_path,
                    entry.access_scope,
                    ArnResourceType::Memory,
                    entry.min_auth_level,
                    new_id.to_string(),
                )?
                .to_canonical();

                // Insert new active replacement entry first
                sqlx::query(
                    "INSERT INTO memory.entries
                     (id, org_id, bucket_id, dept_id, owner_principal_id, access_scope,
                      allowed_group_ids, denied_group_ids, min_auth_level, arn, acl_scope_key,
                      memory_kind, title, effective_text, bm25_text, content_sha256, embedding,
                      embedding_model, tags, source_episode_id, source_document_id, confidence)
                     VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22)",
                )
                .bind(new_id)
                .bind(ctx.org_id)
                .bind(entry.bucket_id)
                .bind(entry.dept_id)
                .bind(entry.owner_principal_id)
                .bind(entry.access_scope.as_str())
                .bind(&entry.allowed_group_ids)
                .bind(&entry.denied_group_ids)
                .bind(entry.min_auth_level.as_i16())
                .bind(&new_arn)
                .bind(&entry.acl_scope_key)
                .bind(entry.memory_kind.as_str())
                .bind(&entry.title)
                .bind(&next_text)
                .bind(&next_text)
                .bind(&new_sha)
                .bind(Vector::from(new_emb))
                .bind(self.embedder.model_name())
                .bind(&entry.tags)
                .bind(entry.source_episode_id)
                .bind(entry.source_document_id)
                .bind(entry.confidence)
                .execute(&mut **tx.tx_mut())
                .await
                .map_err(|e| map_pg_rls_err(e, "insert superseding memory.entry"))?;

                // Record patch on old entry BEFORE setting status='superseded' (so patches INSERT RLS passes)
                let next_ver = entry.version + 1;
                let p = Self::insert_patch_rls(
                    &mut tx,
                    ctx.org_id,
                    entry.id,
                    PatchOp::Supersede,
                    entry.version,
                    next_ver,
                    Some(&entry.effective_text),
                    Some(&next_text),
                    None,
                    None,
                    input.reason.as_deref(),
                    input.actor_type,
                    &actor_id,
                )
                .await?;

                sqlx::query(
                    "INSERT INTO memory.edges (org_id, src_memory_id, dst_memory_id, relation_type, confidence)
                     VALUES ($1, $2, $3, 'supersedes', 1.0)
                     ON CONFLICT (org_id, src_memory_id, dst_memory_id, relation_type) DO NOTHING",
                )
                .bind(ctx.org_id)
                .bind(new_id)
                .bind(entry.id)
                .execute(&mut **tx.tx_mut())
                .await
                .map_err(|e| map_pg_rls_err(e, "insert supersedes edge"))?;

                OutboxEnqueuer::enqueue_rls(
                    &mut tx,
                    &EnqueueOutboxItem {
                        org_id: ctx.org_id,
                        target_table: VectorTable::MemoryEntries,
                        target_id: entry.id,
                        op: OutboxOp::DeleteVector,
                        content_sha256: None,
                        priority: 2,
                        payload: json!({ "superseded_by": new_id }),
                        idempotency_key: None,
                        max_attempts: None,
                    },
                    false,
                )
                .await?;
                OutboxEnqueuer::enqueue_rls(
                    &mut tx,
                    &EnqueueOutboxItem {
                        org_id: ctx.org_id,
                        target_table: VectorTable::MemoryEntries,
                        target_id: new_id,
                        op: OutboxOp::UpsertEmbed,
                        content_sha256: Some(new_sha),
                        priority: 3,
                        payload: json!({ "supersedes": entry.id }),
                        idempotency_key: None,
                        max_attempts: None,
                    },
                    false,
                )
                .await?;

                sqlx::query(
                    "SELECT iam.memory_transition_status($1, $2, 'superseded', $3, $4)",
                )
                .bind(ctx.org_id)
                .bind(entry.id)
                .bind(new_id)
                .bind(next_ver)
                .execute(&mut **tx.tx_mut())
                .await
                .map_err(|e| map_pg_rls_err(e, "mark memory.entries superseded"))?;
                p
            }
            PatchOp::Retract => {
                let next_ver = entry.version + 1;
                let p = Self::insert_patch_rls(
                    &mut tx,
                    ctx.org_id,
                    entry.id,
                    PatchOp::Retract,
                    entry.version,
                    next_ver,
                    Some(&entry.effective_text),
                    None,
                    None,
                    None,
                    input.reason.as_deref(),
                    input.actor_type,
                    &actor_id,
                )
                .await?;

                OutboxEnqueuer::enqueue_rls(
                    &mut tx,
                    &EnqueueOutboxItem {
                        org_id: ctx.org_id,
                        target_table: VectorTable::MemoryEntries,
                        target_id: entry.id,
                        op: OutboxOp::DeleteVector,
                        content_sha256: None,
                        priority: 2,
                        payload: json!({ "patch_op": "retract" }),
                        idempotency_key: None,
                        max_attempts: None,
                    },
                    false,
                )
                .await?;

                sqlx::query(
                    "SELECT iam.memory_transition_status($1, $2, 'retracted', NULL, $3)",
                )
                .bind(ctx.org_id)
                .bind(entry.id)
                .bind(next_ver)
                .execute(&mut **tx.tx_mut())
                .await
                .map_err(|e| map_pg_rls_err(e, "retract memory.entries"))?;
                p
            }
            PatchOp::Redact => {
                let next_ver = entry.version + 1;
                if input.hard_purge {
                    sqlx::query(
                        "UPDATE memory.patches
                         SET prev_text = NULL, new_text = NULL
                         WHERE org_id = $1 AND memory_entry_id = $2",
                    )
                    .bind(ctx.org_id)
                    .bind(entry.id)
                    .execute(&mut **tx.tx_mut())
                    .await
                    .map_err(|e| map_pg_rls_err(e, "hard_purge memory.patches"))?;
                }
                let p = Self::insert_patch_rls(
                    &mut tx,
                    ctx.org_id,
                    entry.id,
                    PatchOp::Redact,
                    entry.version,
                    next_ver,
                    if input.hard_purge {
                        None
                    } else {
                        Some(&entry.effective_text)
                    },
                    if input.hard_purge {
                        None
                    } else {
                        Some("[REDACTED]")
                    },
                    None,
                    None,
                    input.reason.as_deref(),
                    input.actor_type,
                    &actor_id,
                )
                .await?;

                let redacted_sha = sha256_hex(b"[REDACTED]");
                OutboxEnqueuer::enqueue_rls(
                    &mut tx,
                    &EnqueueOutboxItem {
                        org_id: ctx.org_id,
                        target_table: VectorTable::MemoryEntries,
                        target_id: entry.id,
                        op: OutboxOp::RedactReindex,
                        content_sha256: Some(redacted_sha.clone()),
                        priority: 1,
                        payload: json!({ "hard_purge": input.hard_purge }),
                        idempotency_key: None,
                        max_attempts: None,
                    },
                    false,
                )
                .await?;

                sqlx::query(
                    "SELECT iam.memory_transition_status($1, $2, 'redacted', NULL, $3, '[REDACTED]', '', $4)",
                )
                .bind(ctx.org_id)
                .bind(entry.id)
                .bind(next_ver)
                .bind(&redacted_sha)
                .execute(&mut **tx.tx_mut())
                .await
                .map_err(|e| map_pg_rls_err(e, "redact memory.entries"))?;
                p
            }
            PatchOp::PromoteScope | PatchOp::PromoteWiki => {
                return Err(OpenRagError::Validation(
                    "use MemoryBackend::promote() for promote_scope / promote_wiki".into(),
                ));
            }
        };

        tx.commit().await?;
        Ok(patch)
    }
}

#[async_trait]
impl MemoryBackend for PostgresMemoryBackend {
    fn kind(&self) -> MemoryBackendKind {
        MemoryBackendKind::OpenragPg
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
        let mut tx = self.pool.begin_rls_tx(ctx).await?;
        let res = Self::fetch_active_entry_rls(&mut tx, ctx.org_id, entry_id).await?;
        tx.rollback().await?;
        Ok(res)
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
        let mut tx = self.pool.begin_rls_tx(ctx).await?;
        let expanded_tags = if let Some(tf) = tag_filter {
            Some(expand_tag_filter_rls(&mut tx, ctx.org_id, tf).await?)
        } else {
            None
        };

        let mut qb: QueryBuilder<'_, Postgres> = QueryBuilder::new(
            "SELECT id, org_id, bucket_id, dept_id, owner_principal_id, access_scope,
                    allowed_group_ids, denied_group_ids, min_auth_level, arn, acl_scope_key,
                    memory_kind, title, effective_text, bm25_text, content_sha256, tags,
                    source_episode_id, source_document_id, promoted_wiki_id, confidence,
                    status, superseded_by, version, valid_from, valid_to, created_at, updated_at
             FROM memory.entries
             WHERE org_id = ",
        );
        qb.push_bind(ctx.org_id);
        qb.push(" AND status = 'active'");
        if let Some(tf) = &expanded_tags {
            push_tag_filter_predicates(&mut qb, "tags", tf);
        }
        qb.push(" ORDER BY updated_at DESC, id ASC LIMIT ");
        qb.push_bind(limit);

        let rows = qb
            .build()
            .fetch_all(&mut **tx.tx_mut())
            .await
            .map_err(|e| OpenRagError::Database(format!("failed to list memory entries: {e}")))?;
        let entries = rows.iter().map(row_to_entry).collect::<Result<Vec<_>, _>>()?;
        tx.rollback().await?;
        Ok(entries)
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
        let mut tx = self.pool.begin_rls_tx(ctx).await?;
        let Some(entry) = Self::fetch_active_entry_rls(&mut tx, ctx.org_id, input.memory_entry_id).await? else {
            return Err(OpenRagError::Conflict(format!(
                "memory entry {} is not active or inaccessible",
                input.memory_entry_id
            )));
        };

        let cur_rank = Self::scope_rank(entry.access_scope);
        let dst_rank = Self::target_rank(input.target);
        if dst_rank <= cur_rank {
            return Err(OpenRagError::Validation(format!(
                "cannot demote or no-op promote memory from '{}' to '{:?}'",
                entry.access_scope, input.target
            )));
        }

        let actor_id = ctx.principal_id.to_string();
        let next_ver = entry.version + 1;

        match input.target {
            PromoteTarget::Private
            | PromoteTarget::Dept
            | PromoteTarget::Group
            | PromoteTarget::Org => {
                let new_scope = match input.target {
                    PromoteTarget::Private => AccessScope::Private,
                    PromoteTarget::Dept => AccessScope::Dept,
                    PromoteTarget::Group => AccessScope::Group,
                    PromoteTarget::Org => AccessScope::Org,
                    PromoteTarget::Wiki => AccessScope::Org,
                };
                if new_scope == AccessScope::Group && input.target_group_ids.is_empty() && entry.allowed_group_ids.is_empty() {
                    return Err(OpenRagError::Validation(
                        "promoting to 'group' scope requires non-empty target_group_ids".into(),
                    ));
                }
                let next_groups = if input.target_group_ids.is_empty() {
                    entry.allowed_group_ids.clone()
                } else {
                    input.target_group_ids.clone()
                };
                let next_level = input.min_auth_level.unwrap_or(entry.min_auth_level);
                let next_arn = Arn::new(
                    &input.org_slug,
                    &input.dept_path,
                    new_scope,
                    ArnResourceType::Memory,
                    next_level,
                    entry.id.to_string(),
                )?
                .to_canonical();
                let next_scope_key = format!(
                    "{}:{}:L{}",
                    input.dept_path,
                    new_scope.as_str(),
                    next_level.as_u8()
                );

                let row = sqlx::query(
                    "UPDATE memory.entries
                     SET access_scope = $1,
                         allowed_group_ids = $2,
                         min_auth_level = $3,
                         arn = $4,
                         acl_scope_key = $5,
                         version = $6,
                         updated_at = now()
                     WHERE org_id = $7 AND id = $8
                     RETURNING id, org_id, bucket_id, dept_id, owner_principal_id, access_scope,
                               allowed_group_ids, denied_group_ids, min_auth_level, arn, acl_scope_key,
                               memory_kind, title, effective_text, bm25_text, content_sha256, tags,
                               source_episode_id, source_document_id, promoted_wiki_id, confidence,
                               status, superseded_by, version, valid_from, valid_to, created_at, updated_at",
                )
                .bind(new_scope.as_str())
                .bind(&next_groups)
                .bind(next_level.as_i16())
                .bind(&next_arn)
                .bind(&next_scope_key)
                .bind(next_ver)
                .bind(ctx.org_id)
                .bind(entry.id)
                .fetch_one(&mut **tx.tx_mut())
                .await
                .map_err(|e| map_pg_rls_err(e, "promote memory scope"))?;
                let updated = row_to_entry(&row)?;

                let patch = Self::insert_patch_rls(
                    &mut tx,
                    ctx.org_id,
                    entry.id,
                    PatchOp::PromoteScope,
                    entry.version,
                    next_ver,
                    None,
                    None,
                    Some(entry.access_scope.as_str()),
                    Some(new_scope.as_str()),
                    input.reason.as_deref(),
                    ActorType::Human,
                    &actor_id,
                )
                .await?;

                OutboxEnqueuer::enqueue_rls(
                    &mut tx,
                    &EnqueueOutboxItem {
                        org_id: ctx.org_id,
                        target_table: VectorTable::MemoryEntries,
                        target_id: entry.id,
                        op: OutboxOp::UpsertEmbed,
                        content_sha256: Some(updated.content_sha256.clone()),
                        priority: 4,
                        payload: json!({ "promoted_to": new_scope.as_str() }),
                        idempotency_key: None,
                        max_attempts: None,
                    },
                    false,
                )
                .await?;

                tx.commit().await?;
                Ok(PromoteResult {
                    entry: updated,
                    patch,
                    promoted_wiki_id: None,
                })
            }
            PromoteTarget::Wiki => {
                let bucket_id = input.target_bucket_id.ok_or_else(|| {
                    OpenRagError::Validation(
                        "promote to 'wiki' requires target_bucket_id".into(),
                    )
                })?;

                let bucket_row = sqlx::query(
                    "SELECT b.id, b.dept_id, b.access_scope, b.allowed_group_ids, b.denied_group_ids,
                            b.min_auth_level,
                            iam.rls_can_access(b.org_id, b.dept_id, b.owner_principal_id, b.access_scope,
                                               b.allowed_group_ids, b.arn, b.min_auth_level, b.denied_group_ids,
                                               'write', 'bucket', b.id) AS can_write
                     FROM corpus.buckets AS b
                     WHERE b.org_id = $1 AND b.id = $2",
                )
                .bind(ctx.org_id)
                .bind(bucket_id)
                .fetch_optional(&mut **tx.tx_mut())
                .await
                .map_err(|e| OpenRagError::Database(format!("failed to query target bucket: {e}")))?;

                let Some(b_row) = bucket_row else {
                    return Err(OpenRagError::Forbidden(format!(
                        "bucket {bucket_id} not found or inaccessible"
                    )));
                };
                let can_write: bool = b_row
                    .try_get("can_write")
                    .map_err(|e| OpenRagError::Database(format!("invalid can_write: {e}")))?;
                if !can_write {
                    return Err(OpenRagError::Forbidden(format!(
                        "caller lacks write authorization on bucket {bucket_id}"
                    )));
                }
                let b_scope_str: String = b_row
                    .try_get("access_scope")
                    .map_err(|e| OpenRagError::Database(format!("invalid bucket scope: {e}")))?;
                let b_scope: AccessScope = b_scope_str.parse()?;
                let b_dept_id: Uuid = b_row
                    .try_get("dept_id")
                    .map_err(|e| OpenRagError::Database(format!("invalid bucket dept_id: {e}")))?;
                let b_allowed: Vec<Uuid> = b_row
                    .try_get("allowed_group_ids")
                    .map_err(|e| OpenRagError::Database(format!("invalid bucket allowed_groups: {e}")))?;
                let b_denied: Vec<Uuid> = b_row
                    .try_get("denied_group_ids")
                    .map_err(|e| OpenRagError::Database(format!("invalid bucket denied_groups: {e}")))?;
                let b_min_level_i16: i16 = b_row
                    .try_get("min_auth_level")
                    .map_err(|e| OpenRagError::Database(format!("invalid bucket min_auth_level: {e}")))?;
                let b_min_level = AuthLevel::try_from(b_min_level_i16)?;

                if ctx.auth_level < b_min_level {
                    return Err(OpenRagError::Forbidden(format!(
                        "caller auth_level {} < bucket min_auth_level {}",
                        ctx.auth_level, b_min_level
                    )));
                }

                let wiki_id = Uuid::new_v4();
                let wiki_level = input
                    .min_auth_level
                    .unwrap_or(entry.min_auth_level)
                    .max(b_min_level);
                let wiki_arn = Arn::new(
                    &input.org_slug,
                    &input.dept_path,
                    b_scope,
                    ArnResourceType::Wiki,
                    wiki_level,
                    wiki_id.to_string(),
                )?
                .to_canonical();
                let wiki_scope_key = format!(
                    "{}:{}:L{}",
                    input.dept_path,
                    b_scope.as_str(),
                    wiki_level.as_u8()
                );
                let slug = input
                    .wiki_slug
                    .clone()
                    .unwrap_or_else(|| format!("memory-{}", entry.id.simple()));
                let title = input
                    .wiki_title
                    .clone()
                    .or_else(|| entry.title.clone())
                    .unwrap_or_else(|| format!("Wiki: {slug}"));
                let body_sha = sha256_hex(entry.effective_text.as_bytes());
                let emb = self.embedder.embed_text(&entry.effective_text).await?;

                sqlx::query(
                    "INSERT INTO wiki.pages
                     (id, org_id, bucket_id, dept_id, owner_principal_id, access_scope,
                      allowed_group_ids, denied_group_ids, min_auth_level, arn, acl_scope_key,
                      slug, title, body_md, bm25_text, body_sha256, embedding, tags)
                     VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18)",
                )
                .bind(wiki_id)
                .bind(ctx.org_id)
                .bind(bucket_id)
                .bind(b_dept_id)
                .bind(ctx.principal_id)
                .bind(b_scope.as_str())
                .bind(&b_allowed)
                .bind(&b_denied)
                .bind(wiki_level.as_i16())
                .bind(&wiki_arn)
                .bind(&wiki_scope_key)
                .bind(&slug)
                .bind(&title)
                .bind(&entry.effective_text)
                .bind(&entry.effective_text)
                .bind(&body_sha)
                .bind(Vector::from(emb))
                .bind(&entry.tags)
                .execute(&mut **tx.tx_mut())
                .await
                .map_err(|e| map_pg_rls_err(e, "insert promoted wiki.page"))?;

                sqlx::query(
                    "INSERT INTO wiki.revisions
                     (org_id, page_id, version, body_md, body_sha256, editor_type, editor_id, change_note)
                     VALUES ($1, $2, 1, $3, $4, $5, $6, $7)",
                )
                .bind(ctx.org_id)
                .bind(wiki_id)
                .bind(&entry.effective_text)
                .bind(&body_sha)
                .bind(WikiEditorType::Compiler.as_str())
                .bind(&actor_id)
                .bind(input.reason.as_deref().unwrap_or("promoted from memory"))
                .execute(&mut **tx.tx_mut())
                .await
                .map_err(|e| map_pg_rls_err(e, "insert wiki.revisions"))?;

                sqlx::query(
                    "INSERT INTO wiki.page_sources (org_id, page_id, source_type, source_id)
                     VALUES ($1, $2, $3, $4)
                     ON CONFLICT (page_id, source_type, source_id) DO NOTHING",
                )
                .bind(ctx.org_id)
                .bind(wiki_id)
                .bind(WikiSourceType::Memory.as_str())
                .bind(entry.id)
                .execute(&mut **tx.tx_mut())
                .await
                .map_err(|e| map_pg_rls_err(e, "insert wiki.page_sources"))?;

                let row = sqlx::query(
                    "UPDATE memory.entries
                     SET promoted_wiki_id = $1, version = $2, updated_at = now()
                     WHERE org_id = $3 AND id = $4
                     RETURNING id, org_id, bucket_id, dept_id, owner_principal_id, access_scope,
                               allowed_group_ids, denied_group_ids, min_auth_level, arn, acl_scope_key,
                               memory_kind, title, effective_text, bm25_text, content_sha256, tags,
                               source_episode_id, source_document_id, promoted_wiki_id, confidence,
                               status, superseded_by, version, valid_from, valid_to, created_at, updated_at",
                )
                .bind(wiki_id)
                .bind(next_ver)
                .bind(ctx.org_id)
                .bind(entry.id)
                .fetch_one(&mut **tx.tx_mut())
                .await
                .map_err(|e| map_pg_rls_err(e, "update memory.entries promoted_wiki_id"))?;
                let updated = row_to_entry(&row)?;

                let patch = Self::insert_patch_rls(
                    &mut tx,
                    ctx.org_id,
                    entry.id,
                    PatchOp::PromoteWiki,
                    entry.version,
                    next_ver,
                    None,
                    Some(&slug),
                    Some(entry.access_scope.as_str()),
                    Some("wiki"),
                    input.reason.as_deref(),
                    ActorType::Human,
                    &actor_id,
                )
                .await?;

                OutboxEnqueuer::enqueue_rls(
                    &mut tx,
                    &EnqueueOutboxItem {
                        org_id: ctx.org_id,
                        target_table: VectorTable::WikiPages,
                        target_id: wiki_id,
                        op: OutboxOp::UpsertEmbed,
                        content_sha256: Some(body_sha),
                        priority: 3,
                        payload: json!({ "promoted_from_memory": entry.id }),
                        idempotency_key: None,
                        max_attempts: None,
                    },
                    false,
                )
                .await?;

                tx.commit().await?;
                Ok(PromoteResult {
                    entry: updated,
                    patch,
                    promoted_wiki_id: Some(wiki_id),
                })
            }
        }
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

        let mut tx = self.pool.begin_rls_tx(ctx).await?;
        let raw_terms: Vec<String> = parsed
            .clean_query
            .split_whitespace()
            .map(ToString::to_string)
            .collect();
        let expanded_terms = ConceptGraph::expand_terms_rls(&mut tx, ctx.org_id, &raw_terms).await?;
        let expanded_tags = if combined_tf.is_empty() {
            None
        } else {
            Some(expand_tag_filter_rls(&mut tx, ctx.org_id, &combined_tf).await?)
        };

        let query_for_embed = if expanded_terms.is_empty() {
            req.query.clone()
        } else {
            expanded_terms.join(" ")
        };
        let qvec = Vector::from(self.embedder.embed_query(&query_for_embed).await?);

        let mut qb: QueryBuilder<'_, Postgres> = QueryBuilder::new(
            "SELECT id, org_id, bucket_id, dept_id, owner_principal_id, access_scope,
                    allowed_group_ids, denied_group_ids, min_auth_level, arn, acl_scope_key,
                    memory_kind, title, effective_text, bm25_text, content_sha256, tags,
                    source_episode_id, source_document_id, promoted_wiki_id, confidence,
                    status, superseded_by, version, valid_from, valid_to, created_at, updated_at,
                    CASE WHEN embedding IS NOT NULL THEN (1.0 - (embedding <=> ",
        );
        qb.push_bind(qvec);
        qb.push("))::real ELSE 0.0::real END AS vec_sim FROM memory.entries WHERE org_id = ");
        qb.push_bind(ctx.org_id);
        qb.push(" AND status = 'active'");
        if let Some(scope) = req.scope {
            qb.push(" AND access_scope = ");
            qb.push_bind(scope.as_str());
        }
        if let Some(tf) = &expanded_tags {
            push_tag_filter_predicates(&mut qb, "tags", tf);
        }
        qb.push(" ORDER BY vec_sim DESC, updated_at DESC LIMIT ");
        qb.push_bind((req.top_k * 4) as i64);

        let rows = qb
            .build()
            .fetch_all(&mut **tx.tx_mut())
            .await
            .map_err(|e| OpenRagError::Database(format!("memory engage query failed: {e}")))?;

        let mut scored: Vec<(MemoryEntry, f32)> = Vec::with_capacity(rows.len());
        let lower_terms: Vec<String> = expanded_terms.iter().map(|s| s.to_lowercase()).collect();
        for row in &rows {
            let mut entry = row_to_entry(row)?;
            let vec_sim: f32 = row.try_get("vec_sim").unwrap_or(0.0);
            let lower_text = entry.effective_text.to_lowercase();
            let term_hits = lower_terms
                .iter()
                .filter(|t| !t.is_empty() && lower_text.contains(t.as_str()))
                .count();
            let lex_boost = if lower_terms.is_empty() {
                0.0
            } else {
                (term_hits as f32 / lower_terms.len() as f32) * 0.5
            };

            // Enforce egress guard on returned memory text
            match self
                .guard
                .inspect(ctx.org_id, &entry.effective_text, GuardSurface::McpEgress)
                .await
            {
                Ok(dec) => {
                    if dec.action == GuardAction::Mask {
                        if let Some(sanitized) = dec.sanitized_text {
                            entry.effective_text = sanitized;
                        }
                    }
                    scored.push((entry, vec_sim + lex_boost));
                }
                Err(OpenRagError::GuardBlocked { .. }) => continue,
                Err(e) => return Err(e),
            }
        }

        scored.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| b.0.updated_at.cmp(&a.0.updated_at))
        });
        scored.truncate(req.top_k);

        let hit_ids: Vec<Uuid> = scored.iter().map(|(e, _)| e.id).collect();
        let mut edges_by_mem: HashMap<Uuid, Vec<MemoryEdge>> = HashMap::new();
        if !hit_ids.is_empty() {
            let edge_rows = sqlx::query(
                "SELECT id, org_id, src_memory_id, dst_memory_id, relation_type, confidence, created_at
                 FROM memory.edges
                 WHERE org_id = $1 AND (src_memory_id = ANY($2::uuid[]) OR dst_memory_id = ANY($2::uuid[]))
                 ORDER BY created_at DESC",
            )
            .bind(ctx.org_id)
            .bind(&hit_ids)
            .fetch_all(&mut **tx.tx_mut())
            .await
            .map_err(|e| OpenRagError::Database(format!("failed to fetch memory edges: {e}")))?;

            for er in edge_rows {
                let edge = row_to_edge(&er)?;
                edges_by_mem
                    .entry(edge.src_memory_id)
                    .or_default()
                    .push(edge.clone());
                if edge.dst_memory_id != edge.src_memory_id {
                    edges_by_mem
                        .entry(edge.dst_memory_id)
                        .or_default()
                        .push(edge);
                }
            }
        }

        let tag_facets = compute_tag_facets_rls(
            &mut tx,
            ctx.org_id,
            &[TagTable::MemoryEntries],
            None,
            20,
        )
        .await?;
        tx.rollback().await?;

        let memories = scored
            .into_iter()
            .map(|(entry, score)| {
                let edges = edges_by_mem.remove(&entry.id).unwrap_or_default();
                EngageHit {
                    entry,
                    score,
                    edges,
                }
            })
            .collect();

        Ok(EngageResponse {
            memories,
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
        let mut tx = self.pool.begin_rls_tx(ctx).await?;
        let erow = sqlx::query(
            "INSERT INTO memory.edges (org_id, src_memory_id, dst_memory_id, relation_type, confidence)
             VALUES ($1, $2, $3, $4, $5)
             ON CONFLICT (org_id, src_memory_id, dst_memory_id, relation_type)
             DO UPDATE SET confidence = EXCLUDED.confidence
             RETURNING id, org_id, src_memory_id, dst_memory_id, relation_type, confidence, created_at",
        )
        .bind(ctx.org_id)
        .bind(src_memory_id)
        .bind(dst_memory_id)
        .bind(relation_type.as_str())
        .bind(confidence)
        .fetch_one(&mut **tx.tx_mut())
        .await
        .map_err(|e| map_pg_rls_err(e, "insert memory.edges"))?;
        let edge = row_to_edge(&erow)?;
        tx.commit().await?;
        Ok(edge)
    }

    async fn list_edges(
        &self,
        ctx: &PrincipalContext,
        memory_id: Uuid,
    ) -> Result<Vec<MemoryEdge>, OpenRagError> {
        let mut tx = self.pool.begin_rls_tx(ctx).await?;
        let rows = sqlx::query(
            "SELECT id, org_id, src_memory_id, dst_memory_id, relation_type, confidence, created_at
             FROM memory.edges
             WHERE org_id = $1 AND (src_memory_id = $2 OR dst_memory_id = $2)
             ORDER BY created_at ASC",
        )
        .bind(ctx.org_id)
        .bind(memory_id)
        .fetch_all(&mut **tx.tx_mut())
        .await
        .map_err(|e| OpenRagError::Database(format!("failed to list memory.edges: {e}")))?;
        let edges = rows.iter().map(row_to_edge).collect::<Result<Vec<_>, _>>()?;
        tx.rollback().await?;
        Ok(edges)
    }
}

pub(crate) fn normalize_tags(tags: &[String]) -> Vec<String> {
    let mut set = BTreeSet::new();
    for t in tags {
        let n = normalize_tag(t);
        if !n.is_empty() {
            set.insert(n);
        }
    }
    set.into_iter().collect()
}

fn map_pg_rls_err(e: sqlx::Error, op: &str) -> OpenRagError {
    if let sqlx::Error::Database(ref db_err) = e {
        if db_err.code().as_deref() == Some("42501") {
            return OpenRagError::Forbidden(format!("RLS denied {op}: {}", db_err.message()));
        }
        if db_err.code().as_deref() == Some("23505") {
            return OpenRagError::Conflict(format!("unique conflict on {op}: {}", db_err.message()));
        }
    }
    OpenRagError::Database(format!("{op} failed: {e}"))
}

fn row_to_episode(row: &sqlx::postgres::PgRow) -> Result<MemoryEpisode, OpenRagError> {
    let scope_str: String = row
        .try_get("access_scope")
        .map_err(|e| OpenRagError::Database(format!("invalid access_scope: {e}")))?;
    let level_i16: i16 = row
        .try_get("min_auth_level")
        .map_err(|e| OpenRagError::Database(format!("invalid min_auth_level: {e}")))?;
    let harness_str: String = row
        .try_get("source_harness")
        .map_err(|e| OpenRagError::Database(format!("invalid source_harness: {e}")))?;
    Ok(MemoryEpisode {
        id: row.try_get("id").map_err(|e| OpenRagError::Database(e.to_string()))?,
        org_id: row.try_get("org_id").map_err(|e| OpenRagError::Database(e.to_string()))?,
        dept_id: row.try_get("dept_id").map_err(|e| OpenRagError::Database(e.to_string()))?,
        owner_principal_id: row
            .try_get("owner_principal_id")
            .map_err(|e| OpenRagError::Database(e.to_string()))?,
        access_scope: scope_str.parse()?,
        allowed_group_ids: row
            .try_get("allowed_group_ids")
            .map_err(|e| OpenRagError::Database(e.to_string()))?,
        denied_group_ids: row
            .try_get("denied_group_ids")
            .map_err(|e| OpenRagError::Database(e.to_string()))?,
        min_auth_level: AuthLevel::try_from(level_i16)?,
        arn: row.try_get("arn").map_err(|e| OpenRagError::Database(e.to_string()))?,
        source_harness: harness_str.parse()?,
        session_key: row
            .try_get("session_key")
            .map_err(|e| OpenRagError::Database(e.to_string()))?,
        summary_md: row
            .try_get("summary_md")
            .map_err(|e| OpenRagError::Database(e.to_string()))?,
        raw_redacted_json: row
            .try_get("raw_redacted_json")
            .map_err(|e| OpenRagError::Database(e.to_string()))?,
        content_sha256: row
            .try_get("content_sha256")
            .map_err(|e| OpenRagError::Database(e.to_string()))?,
        redaction_applied: row
            .try_get("redaction_applied")
            .map_err(|e| OpenRagError::Database(e.to_string()))?,
        created_at: row
            .try_get("created_at")
            .map_err(|e| OpenRagError::Database(e.to_string()))?,
    })
}

fn row_to_entry(row: &sqlx::postgres::PgRow) -> Result<MemoryEntry, OpenRagError> {
    let scope_str: String = row
        .try_get("access_scope")
        .map_err(|e| OpenRagError::Database(e.to_string()))?;
    let level_i16: i16 = row
        .try_get("min_auth_level")
        .map_err(|e| OpenRagError::Database(e.to_string()))?;
    let kind_str: String = row
        .try_get("memory_kind")
        .map_err(|e| OpenRagError::Database(e.to_string()))?;
    let status_str: String = row
        .try_get("status")
        .map_err(|e| OpenRagError::Database(e.to_string()))?;
    Ok(MemoryEntry {
        id: row.try_get("id").map_err(|e| OpenRagError::Database(e.to_string()))?,
        org_id: row.try_get("org_id").map_err(|e| OpenRagError::Database(e.to_string()))?,
        bucket_id: row
            .try_get("bucket_id")
            .map_err(|e| OpenRagError::Database(e.to_string()))?,
        dept_id: row.try_get("dept_id").map_err(|e| OpenRagError::Database(e.to_string()))?,
        owner_principal_id: row
            .try_get("owner_principal_id")
            .map_err(|e| OpenRagError::Database(e.to_string()))?,
        access_scope: scope_str.parse()?,
        allowed_group_ids: row
            .try_get("allowed_group_ids")
            .map_err(|e| OpenRagError::Database(e.to_string()))?,
        denied_group_ids: row
            .try_get("denied_group_ids")
            .map_err(|e| OpenRagError::Database(e.to_string()))?,
        min_auth_level: AuthLevel::try_from(level_i16)?,
        arn: row.try_get("arn").map_err(|e| OpenRagError::Database(e.to_string()))?,
        acl_scope_key: row
            .try_get("acl_scope_key")
            .map_err(|e| OpenRagError::Database(e.to_string()))?,
        memory_kind: kind_str.parse()?,
        title: row.try_get("title").map_err(|e| OpenRagError::Database(e.to_string()))?,
        effective_text: row
            .try_get("effective_text")
            .map_err(|e| OpenRagError::Database(e.to_string()))?,
        bm25_text: row
            .try_get("bm25_text")
            .map_err(|e| OpenRagError::Database(e.to_string()))?,
        content_sha256: row
            .try_get("content_sha256")
            .map_err(|e| OpenRagError::Database(e.to_string()))?,
        tags: row.try_get("tags").map_err(|e| OpenRagError::Database(e.to_string()))?,
        source_episode_id: row
            .try_get("source_episode_id")
            .map_err(|e| OpenRagError::Database(e.to_string()))?,
        source_document_id: row
            .try_get("source_document_id")
            .map_err(|e| OpenRagError::Database(e.to_string()))?,
        promoted_wiki_id: row
            .try_get("promoted_wiki_id")
            .map_err(|e| OpenRagError::Database(e.to_string()))?,
        confidence: row
            .try_get("confidence")
            .map_err(|e| OpenRagError::Database(e.to_string()))?,
        status: status_str.parse()?,
        superseded_by: row
            .try_get("superseded_by")
            .map_err(|e| OpenRagError::Database(e.to_string()))?,
        version: row
            .try_get("version")
            .map_err(|e| OpenRagError::Database(e.to_string()))?,
        valid_from: row
            .try_get("valid_from")
            .map_err(|e| OpenRagError::Database(e.to_string()))?,
        valid_to: row
            .try_get("valid_to")
            .map_err(|e| OpenRagError::Database(e.to_string()))?,
        created_at: row
            .try_get("created_at")
            .map_err(|e| OpenRagError::Database(e.to_string()))?,
        updated_at: row
            .try_get("updated_at")
            .map_err(|e| OpenRagError::Database(e.to_string()))?,
    })
}

fn row_to_patch(row: &sqlx::postgres::PgRow) -> Result<MemoryPatch, OpenRagError> {
    let op_str: String = row
        .try_get("patch_op")
        .map_err(|e| OpenRagError::Database(e.to_string()))?;
    let actor_str: String = row
        .try_get("actor_type")
        .map_err(|e| OpenRagError::Database(e.to_string()))?;
    Ok(MemoryPatch {
        id: row.try_get("id").map_err(|e| OpenRagError::Database(e.to_string()))?,
        org_id: row.try_get("org_id").map_err(|e| OpenRagError::Database(e.to_string()))?,
        memory_entry_id: row
            .try_get("memory_entry_id")
            .map_err(|e| OpenRagError::Database(e.to_string()))?,
        patch_op: op_str.parse()?,
        prev_version: row
            .try_get("prev_version")
            .map_err(|e| OpenRagError::Database(e.to_string()))?,
        new_version: row
            .try_get("new_version")
            .map_err(|e| OpenRagError::Database(e.to_string()))?,
        prev_text: row
            .try_get("prev_text")
            .map_err(|e| OpenRagError::Database(e.to_string()))?,
        new_text: row
            .try_get("new_text")
            .map_err(|e| OpenRagError::Database(e.to_string()))?,
        prev_scope: row
            .try_get("prev_scope")
            .map_err(|e| OpenRagError::Database(e.to_string()))?,
        new_scope: row
            .try_get("new_scope")
            .map_err(|e| OpenRagError::Database(e.to_string()))?,
        reason: row.try_get("reason").map_err(|e| OpenRagError::Database(e.to_string()))?,
        actor_type: actor_str.parse()?,
        actor_id: row
            .try_get("actor_id")
            .map_err(|e| OpenRagError::Database(e.to_string()))?,
        created_at: row
            .try_get("created_at")
            .map_err(|e| OpenRagError::Database(e.to_string()))?,
    })
}

fn row_to_edge(row: &sqlx::postgres::PgRow) -> Result<MemoryEdge, OpenRagError> {
    let rel_str: String = row
        .try_get("relation_type")
        .map_err(|e| OpenRagError::Database(e.to_string()))?;
    Ok(MemoryEdge {
        id: row.try_get("id").map_err(|e| OpenRagError::Database(e.to_string()))?,
        org_id: row.try_get("org_id").map_err(|e| OpenRagError::Database(e.to_string()))?,
        src_memory_id: row
            .try_get("src_memory_id")
            .map_err(|e| OpenRagError::Database(e.to_string()))?,
        dst_memory_id: row
            .try_get("dst_memory_id")
            .map_err(|e| OpenRagError::Database(e.to_string()))?,
        relation_type: rel_str.parse()?,
        confidence: row
            .try_get("confidence")
            .map_err(|e| OpenRagError::Database(e.to_string()))?,
        created_at: row
            .try_get("created_at")
            .map_err(|e| OpenRagError::Database(e.to_string()))?,
    })
}

// Keep unused helper quiet if cosine_similarity only used in kumiho
const _: fn(&[f32], &[f32]) -> Result<f32, OpenRagError> = cosine_similarity;
