use std::collections::{HashMap, HashSet};

use async_trait::async_trait;
use openrag_core::{BBoxNorm1000, OpenRagError, PrincipalContext};
use pgvector::Vector;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use sqlx::{PgPool, Postgres, Row, Transaction};
use uuid::Uuid;

use crate::thumbnail::{mask_image_bboxes, LocalThumbnailStore, ThumbnailStore};
use crate::vault::{MasterKek, VaultCrypto};

/// The text embedder ingestion uses. A scrubbed chunk, memory entry or wiki page is re-embedded
/// from its scrubbed text inside the redaction transaction, so its vector is always derived from
/// the text stored next to it and a failed embedding rolls the whole job back.
#[async_trait]
pub trait RedactionTextEmbedder: Send + Sync {
    async fn embed_text(&self, text: &str) -> Result<Vec<f32>, OpenRagError>;
    /// Embeds a scrubbed corpus chunk alongside its optional document title.
    async fn embed_document(
        &self,
        _title: Option<&str>,
        text: &str,
    ) -> Result<Vec<f32>, OpenRagError> {
        self.embed_text(text).await
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RedactionTargetScope {
    Document,
    Chunk,
    Memory,
    Wiki,
    PatternOrg,
    PatternDept,
    VaultEntry,
}

impl RedactionTargetScope {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Document => "document",
            Self::Chunk => "chunk",
            Self::Memory => "memory",
            Self::Wiki => "wiki",
            Self::PatternOrg => "pattern_org",
            Self::PatternDept => "pattern_dept",
            Self::VaultEntry => "vault_entry",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RedactionMode {
    MaskKeepVault,
    HardPurge,
    TombstoneDelete,
}

impl RedactionMode {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::MaskKeepVault => "mask_keep_vault",
            Self::HardPurge => "hard_purge",
            Self::TombstoneDelete => "tombstone_delete",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RedactionJobRequest {
    pub org_id: Uuid,
    pub target_scope: RedactionTargetScope,
    pub target_ref: serde_json::Value,
    pub redaction_mode: RedactionMode,
    pub redact_thumbnails: bool,
    pub reason: String,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct AffectedCounts {
    pub chunks: u64,
    pub pages: u64,
    pub documents: u64,
    pub memories: u64,
    pub memory_patches_cleared: u64,
    pub memory_episodes: u64,
    pub wiki_pages: u64,
    pub wiki_revisions: u64,
    pub vault_entries_purged: u64,
    pub thumbnails_masked: u64,
    pub outbox_enqueued: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RedactionJobResult {
    pub job_id: Uuid,
    pub status: String,
    pub replacement_token: String,
    pub affected_counts: AffectedCounts,
}

pub struct CascadeRedactor;

impl CascadeRedactor {
    pub fn authorize_caller(
        caller: &PrincipalContext,
        req: &RedactionJobRequest,
    ) -> Result<(), OpenRagError> {
        if caller.org_id != req.org_id {
            return Err(OpenRagError::Forbidden(
                "cannot run cascade redaction across organizations".into(),
            ));
        }
        if caller.is_org_admin {
            return Ok(());
        }
        match req.target_scope {
            RedactionTargetScope::PatternOrg | RedactionTargetScope::VaultEntry => {
                Err(OpenRagError::Forbidden(
                    "org-wide or vault redaction requires org_admin".into(),
                ))
            }
            RedactionTargetScope::PatternDept => {
                let dept_id = parse_uuid_field(&req.target_ref, "dept_id")?;
                if caller.dept_admin_ids.contains(&dept_id) {
                    Ok(())
                } else {
                    Err(OpenRagError::Forbidden(
                        "dept pattern redaction requires dept_admin of target dept".into(),
                    ))
                }
            }
            _ => {
                if !caller.dept_admin_ids.is_empty() {
                    Ok(())
                } else {
                    Err(OpenRagError::Forbidden(
                        "non-admin members are forbidden from running cascade redaction".into(),
                    ))
                }
            }
        }
    }

    pub async fn run_job(
        worker_pool: &PgPool,
        caller: &PrincipalContext,
        req: RedactionJobRequest,
        thumb_store: &dyn ThumbnailStore,
        text_embedder: &dyn RedactionTextEmbedder,
        kek: Option<&MasterKek>,
        embed_document_title: bool,
    ) -> Result<RedactionJobResult, OpenRagError> {
        Self::authorize_caller(caller, &req)?;

        if req.reason.trim().is_empty() {
            return Err(OpenRagError::Validation(
                "redaction job reason must not be empty".into(),
            ));
        }

        let job_id: Uuid = sqlx::query_scalar(
            "INSERT INTO pii.redaction_jobs (
                org_id, requested_by, target_scope, target_ref,
                redaction_mode, redact_thumbnails, reason, status
             ) VALUES ($1, $2, $3, $4, $5, $6, $7, 'running')
             RETURNING id",
        )
        .bind(req.org_id)
        .bind(caller.principal_id)
        .bind(req.target_scope.as_str())
        .bind(&req.target_ref)
        .bind(req.redaction_mode.as_str())
        .bind(req.redact_thumbnails)
        .bind(&req.reason)
        .fetch_one(worker_pool)
        .await
        .map_err(|e| OpenRagError::Database(format!("failed inserting pii.redaction_jobs: {e}")))?;

        let replacement = format!("[REDACTED:{job_id}]");
        let mut counts = AffectedCounts::default();

        let mut tx = worker_pool.begin().await.map_err(|e| {
            OpenRagError::Database(format!("failed starting cascade redaction tx: {e}"))
        })?;

        let res = Self::execute_tx(
            &mut tx,
            job_id,
            caller,
            &req,
            &replacement,
            &mut counts,
            thumb_store,
            text_embedder,
            kek,
            embed_document_title,
        )
        .await;

        if let Err(err) = res {
            let _ = tx.rollback().await;
            let _ = sqlx::query(
                "UPDATE pii.redaction_jobs SET status = 'failed', completed_at = now() WHERE id = $1",
            )
            .bind(job_id)
            .execute(worker_pool)
            .await;
            return Err(err);
        }

        let counts_json = serde_json::to_value(&counts)
            .map_err(|e| OpenRagError::Internal(format!("failed serializing affected_counts: {e}")))?;

        sqlx::query(
            "UPDATE pii.redaction_jobs
             SET status = 'completed', affected_counts = $2, completed_at = now()
             WHERE id = $1",
        )
        .bind(job_id)
        .bind(counts_json)
        .execute(&mut *tx)
        .await
        .map_err(|e| OpenRagError::Database(format!("failed finalizing pii.redaction_jobs: {e}")))?;

        tx.commit().await.map_err(|e| {
            OpenRagError::Database(format!("failed committing cascade redaction tx: {e}"))
        })?;

        Ok(RedactionJobResult {
            job_id,
            status: "completed".to_string(),
            replacement_token: replacement,
            affected_counts: counts,
        })
    }

    #[allow(clippy::too_many_arguments)]
    async fn execute_tx(
        tx: &mut Transaction<'_, Postgres>,
        job_id: Uuid,
        caller: &PrincipalContext,
        req: &RedactionJobRequest,
        replacement: &str,
        counts: &mut AffectedCounts,
        thumb_store: &dyn ThumbnailStore,
        text_embedder: &dyn RedactionTextEmbedder,
        kek: Option<&MasterKek>,
        embed_document_title: bool,
    ) -> Result<(), OpenRagError> {
        let org_id = req.org_id;
        let is_hard_purge = req.redaction_mode == RedactionMode::HardPurge;
        let is_tombstone = req.redaction_mode == RedactionMode::TombstoneDelete;
        let outbox_op = if is_tombstone {
            "delete_vector"
        } else {
            "redact_reindex"
        };

        let explicit_bbox = parse_optional_bbox(&req.target_ref)?;
        let pattern = req
            .target_ref
            .get("pattern")
            .and_then(|v| v.as_str())
            .map(str::to_string);

        let mut doc_ids: Vec<Uuid> = Vec::new();
        let mut chunk_ids: Vec<Uuid> = Vec::new();
        let mut page_ids: Vec<Uuid> = Vec::new();
        let mut memory_ids: Vec<Uuid> = Vec::new();
        let mut episode_ids: Vec<Uuid> = Vec::new();
        let mut wiki_ids: Vec<Uuid> = Vec::new();
        let mut vault_ids: Vec<Uuid> = Vec::new();

        match req.target_scope {
            RedactionTargetScope::Document => {
                let doc_id = parse_uuid_field(&req.target_ref, "document_id")?;
                // Verify dept_admin authorization for document dept if not org_admin
                if !caller.is_org_admin {
                    let dept_id: Option<Uuid> = sqlx::query_scalar(
                        "SELECT dept_id FROM corpus.documents WHERE org_id = $1 AND id = $2",
                    )
                    .bind(org_id)
                    .bind(doc_id)
                    .fetch_optional(&mut **tx)
                    .await
                    .map_err(|e| OpenRagError::Database(e.to_string()))?;
                    let Some(d) = dept_id else {
                        return Err(OpenRagError::Forbidden(
                            "target document not found or not accessible to dept_admin".into(),
                        ));
                    };
                    if !caller.dept_admin_ids.contains(&d) {
                        return Err(OpenRagError::Forbidden(
                            "caller is not dept_admin for target document department".into(),
                        ));
                    }
                }
                doc_ids.push(doc_id);
                page_ids = sqlx::query_scalar(
                    "SELECT id FROM corpus.pages WHERE org_id = $1 AND document_id = $2",
                )
                .bind(org_id)
                .bind(doc_id)
                .fetch_all(&mut **tx)
                .await
                .map_err(|e| OpenRagError::Database(e.to_string()))?;

                chunk_ids = sqlx::query_scalar(
                    "SELECT id FROM corpus.chunks WHERE org_id = $1 AND document_id = $2",
                )
                .bind(org_id)
                .bind(doc_id)
                .fetch_all(&mut **tx)
                .await
                .map_err(|e| OpenRagError::Database(e.to_string()))?;

                memory_ids = if caller.is_org_admin {
                    sqlx::query_scalar(
                        "SELECT id FROM memory.entries WHERE org_id = $1 AND source_document_id = $2",
                    )
                    .bind(org_id)
                    .bind(doc_id)
                    .fetch_all(&mut **tx)
                    .await
                } else {
                    sqlx::query_scalar(
                        "SELECT id FROM memory.entries
                         WHERE org_id = $1 AND source_document_id = $2 AND dept_id = ANY($3)",
                    )
                    .bind(org_id)
                    .bind(doc_id)
                    .bind(&caller.dept_admin_ids)
                    .fetch_all(&mut **tx)
                    .await
                }
                .map_err(|e| OpenRagError::Database(e.to_string()))?;

                wiki_ids = if caller.is_org_admin {
                    sqlx::query_scalar(
                        "SELECT DISTINCT page_id FROM wiki.page_sources
                         WHERE org_id = $1 AND source_type = 'document' AND source_id = $2",
                    )
                    .bind(org_id)
                    .bind(doc_id)
                    .fetch_all(&mut **tx)
                    .await
                } else {
                    sqlx::query_scalar(
                        "SELECT DISTINCT ps.page_id
                         FROM wiki.page_sources ps
                         JOIN wiki.pages wp ON wp.id = ps.page_id AND wp.org_id = ps.org_id
                         WHERE ps.org_id = $1 AND ps.source_type = 'document' AND ps.source_id = $2
                           AND wp.dept_id = ANY($3)",
                    )
                    .bind(org_id)
                    .bind(doc_id)
                    .bind(&caller.dept_admin_ids)
                    .fetch_all(&mut **tx)
                    .await
                }
                .map_err(|e| OpenRagError::Database(e.to_string()))?;

                if !chunk_ids.is_empty() {
                    vault_ids = sqlx::query_scalar(
                        "SELECT DISTINCT vault_entry_id FROM pii.chunk_spans
                         WHERE org_id = $1 AND chunk_id = ANY($2) AND vault_entry_id IS NOT NULL",
                    )
                    .bind(org_id)
                    .bind(&chunk_ids)
                    .fetch_all(&mut **tx)
                    .await
                    .map_err(|e| OpenRagError::Database(e.to_string()))?;
                }
            }
            RedactionTargetScope::Chunk => {
                let cid = parse_uuid_field(&req.target_ref, "chunk_id")?;
                let chunk_row = sqlx::query(
                    "SELECT dept_id, page_id FROM corpus.chunks WHERE org_id = $1 AND id = $2",
                )
                .bind(org_id)
                .bind(cid)
                .fetch_optional(&mut **tx)
                .await
                .map_err(|e| OpenRagError::Database(e.to_string()))?;

                if !caller.is_org_admin {
                    let Some(ref row) = chunk_row else {
                        return Err(OpenRagError::Forbidden(
                            "target chunk not found or not accessible to dept_admin".into(),
                        ));
                    };
                    let dept_id: Uuid = row
                        .try_get("dept_id")
                        .map_err(|e| OpenRagError::Database(e.to_string()))?;
                    if !caller.dept_admin_ids.contains(&dept_id) {
                        return Err(OpenRagError::Forbidden(
                            "caller is not dept_admin for target chunk department".into(),
                        ));
                    }
                }

                chunk_ids.push(cid);
                if let Some(row) = chunk_row {
                    let pid: Option<Uuid> = row
                        .try_get("page_id")
                        .map_err(|e| OpenRagError::Database(e.to_string()))?;
                    if let Some(p) = pid {
                        page_ids.push(p);
                    }
                }
                vault_ids = sqlx::query_scalar(
                    "SELECT DISTINCT vault_entry_id FROM pii.chunk_spans
                     WHERE org_id = $1 AND chunk_id = $2 AND vault_entry_id IS NOT NULL",
                )
                .bind(org_id)
                .bind(cid)
                .fetch_all(&mut **tx)
                .await
                .map_err(|e| OpenRagError::Database(e.to_string()))?;
            }
            RedactionTargetScope::Memory => {
                let mid = parse_uuid_field(&req.target_ref, "memory_entry_id")?;
                if !caller.is_org_admin {
                    let dept_id: Option<Uuid> = sqlx::query_scalar(
                        "SELECT dept_id FROM memory.entries WHERE org_id = $1 AND id = $2",
                    )
                    .bind(org_id)
                    .bind(mid)
                    .fetch_optional(&mut **tx)
                    .await
                    .map_err(|e| OpenRagError::Database(e.to_string()))?;
                    let Some(d) = dept_id else {
                        return Err(OpenRagError::Forbidden(
                            "target memory entry not found or not accessible to dept_admin".into(),
                        ));
                    };
                    if !caller.dept_admin_ids.contains(&d) {
                        return Err(OpenRagError::Forbidden(
                            "caller is not dept_admin for target memory department".into(),
                        ));
                    }
                }
                memory_ids.push(mid);
            }
            RedactionTargetScope::Wiki => {
                let wid = parse_uuid_field(&req.target_ref, "wiki_page_id")?;
                if !caller.is_org_admin {
                    let dept_id: Option<Uuid> = sqlx::query_scalar(
                        "SELECT dept_id FROM wiki.pages WHERE org_id = $1 AND id = $2",
                    )
                    .bind(org_id)
                    .bind(wid)
                    .fetch_optional(&mut **tx)
                    .await
                    .map_err(|e| OpenRagError::Database(e.to_string()))?;
                    let Some(d) = dept_id else {
                        return Err(OpenRagError::Forbidden(
                            "target wiki page not found or not accessible to dept_admin".into(),
                        ));
                    };
                    if !caller.dept_admin_ids.contains(&d) {
                        return Err(OpenRagError::Forbidden(
                            "caller is not dept_admin for target wiki page department".into(),
                        ));
                    }
                }
                wiki_ids.push(wid);
            }
            RedactionTargetScope::VaultEntry => {
                if let Ok(vid) = parse_uuid_field(&req.target_ref, "vault_entry_id") {
                    vault_ids.push(vid);
                } else if let Some(tok) = req.target_ref.get("token").and_then(|v| v.as_str()) {
                    let found: Option<Uuid> = sqlx::query_scalar(
                        "SELECT id FROM pii.vault_entries WHERE org_id = $1 AND token = $2",
                    )
                    .bind(org_id)
                    .bind(tok)
                    .fetch_optional(&mut **tx)
                    .await
                    .map_err(|e| OpenRagError::Database(e.to_string()))?;
                    if let Some(id) = found {
                        vault_ids.push(id);
                    }
                }
                if !vault_ids.is_empty() {
                    chunk_ids = sqlx::query_scalar(
                        "SELECT DISTINCT chunk_id FROM pii.chunk_spans
                         WHERE org_id = $1 AND vault_entry_id = ANY($2)",
                    )
                    .bind(org_id)
                    .bind(&vault_ids)
                    .fetch_all(&mut **tx)
                    .await
                    .map_err(|e| OpenRagError::Database(e.to_string()))?;
                }
            }
            RedactionTargetScope::PatternOrg | RedactionTargetScope::PatternDept => {
                let pat = pattern.as_deref().ok_or_else(|| {
                    OpenRagError::Validation("pattern scope requires target_ref.pattern".into())
                })?;
                if pat.is_empty() {
                    return Err(OpenRagError::Validation(
                        "pattern scope requires non-empty target_ref.pattern".into(),
                    ));
                }
                let dept_filter = if req.target_scope == RedactionTargetScope::PatternDept {
                    Some(parse_uuid_field(&req.target_ref, "dept_id")?)
                } else {
                    None
                };

                chunk_ids = if let Some(d) = dept_filter {
                    sqlx::query_scalar(
                        "SELECT id FROM corpus.chunks
                         WHERE org_id = $1 AND dept_id = $2
                           AND (redacted_text LIKE '%' || $3 || '%' OR bm25_text LIKE '%' || $3 || '%')",
                    )
                    .bind(org_id)
                    .bind(d)
                    .bind(pat)
                    .fetch_all(&mut **tx)
                    .await
                } else {
                    sqlx::query_scalar(
                        "SELECT id FROM corpus.chunks
                         WHERE org_id = $1
                           AND (redacted_text LIKE '%' || $2 || '%' OR bm25_text LIKE '%' || $2 || '%')",
                    )
                    .bind(org_id)
                    .bind(pat)
                    .fetch_all(&mut **tx)
                    .await
                }
                .map_err(|e| OpenRagError::Database(e.to_string()))?;

                page_ids = if let Some(d) = dept_filter {
                    sqlx::query_scalar(
                        "SELECT id FROM corpus.pages
                         WHERE org_id = $1 AND dept_id = $2
                           AND (COALESCE(redacted_markdown,'') LIKE '%' || $3 || '%' OR COALESCE(raw_markdown,'') LIKE '%' || $3 || '%')",
                    )
                    .bind(org_id)
                    .bind(d)
                    .bind(pat)
                    .fetch_all(&mut **tx)
                    .await
                } else {
                    sqlx::query_scalar(
                        "SELECT id FROM corpus.pages
                         WHERE org_id = $1
                           AND (COALESCE(redacted_markdown,'') LIKE '%' || $2 || '%' OR COALESCE(raw_markdown,'') LIKE '%' || $2 || '%')",
                    )
                    .bind(org_id)
                    .bind(pat)
                    .fetch_all(&mut **tx)
                    .await
                }
                .map_err(|e| OpenRagError::Database(e.to_string()))?;

                memory_ids = if let Some(d) = dept_filter {
                    sqlx::query_scalar(
                        "SELECT id FROM memory.entries
                         WHERE org_id = $1 AND dept_id = $2
                           AND (effective_text LIKE '%' || $3 || '%' OR bm25_text LIKE '%' || $3 || '%')",
                    )
                    .bind(org_id)
                    .bind(d)
                    .bind(pat)
                    .fetch_all(&mut **tx)
                    .await
                } else {
                    sqlx::query_scalar(
                        "SELECT id FROM memory.entries
                         WHERE org_id = $1
                           AND (effective_text LIKE '%' || $2 || '%' OR bm25_text LIKE '%' || $2 || '%')",
                    )
                    .bind(org_id)
                    .bind(pat)
                    .fetch_all(&mut **tx)
                    .await
                }
                .map_err(|e| OpenRagError::Database(e.to_string()))?;

                episode_ids = if let Some(d) = dept_filter {
                    sqlx::query_scalar(
                        "SELECT id FROM memory.episodes
                         WHERE org_id = $1 AND dept_id = $2
                           AND COALESCE(summary_md,'') LIKE '%' || $3 || '%'",
                    )
                    .bind(org_id)
                    .bind(d)
                    .bind(pat)
                    .fetch_all(&mut **tx)
                    .await
                } else {
                    sqlx::query_scalar(
                        "SELECT id FROM memory.episodes
                         WHERE org_id = $1
                           AND COALESCE(summary_md,'') LIKE '%' || $2 || '%'",
                    )
                    .bind(org_id)
                    .bind(pat)
                    .fetch_all(&mut **tx)
                    .await
                }
                .map_err(|e| OpenRagError::Database(e.to_string()))?;

                wiki_ids = if let Some(d) = dept_filter {
                    sqlx::query_scalar(
                        "SELECT id FROM wiki.pages
                         WHERE org_id = $1 AND dept_id = $2
                           AND (body_md LIKE '%' || $3 || '%' OR bm25_text LIKE '%' || $3 || '%')",
                    )
                    .bind(org_id)
                    .bind(d)
                    .bind(pat)
                    .fetch_all(&mut **tx)
                    .await
                } else {
                    sqlx::query_scalar(
                        "SELECT id FROM wiki.pages
                         WHERE org_id = $1
                           AND (body_md LIKE '%' || $2 || '%' OR bm25_text LIKE '%' || $2 || '%')",
                    )
                    .bind(org_id)
                    .bind(pat)
                    .fetch_all(&mut **tx)
                    .await
                }
                .map_err(|e| OpenRagError::Database(e.to_string()))?;

                if !chunk_ids.is_empty() {
                    let span_vids: Vec<Uuid> = sqlx::query_scalar(
                        "SELECT DISTINCT vault_entry_id FROM pii.chunk_spans
                         WHERE org_id = $1 AND chunk_id = ANY($2) AND vault_entry_id IS NOT NULL",
                    )
                    .bind(org_id)
                    .bind(&chunk_ids)
                    .fetch_all(&mut **tx)
                    .await
                    .map_err(|e| OpenRagError::Database(e.to_string()))?;
                    vault_ids.extend(span_vids);
                }

                // Match vault_entries containing `pat` when KEK is supplied
                if let Some(master_kek) = kek {
                    let vrows = if let Some(d) = dept_filter {
                        sqlx::query(
                            "SELECT DISTINCT ve.id, ve.token, ve.ciphertext, ve.nonce, vk.wrapped_dek
                             FROM pii.vault_entries ve
                             JOIN pii.vault_keys vk ON vk.id = ve.vault_key_id AND vk.org_id = ve.org_id
                             JOIN pii.chunk_spans cs ON cs.org_id = ve.org_id AND cs.vault_entry_id = ve.id
                             JOIN corpus.chunks c ON c.org_id = cs.org_id AND c.id = cs.chunk_id
                             WHERE ve.org_id = $1 AND ve.purged_at IS NULL AND c.dept_id = $2",
                        )
                        .bind(org_id)
                        .bind(d)
                        .fetch_all(&mut **tx)
                        .await
                    } else {
                        sqlx::query(
                            "SELECT ve.id, ve.token, ve.ciphertext, ve.nonce, vk.wrapped_dek
                             FROM pii.vault_entries ve
                             JOIN pii.vault_keys vk ON vk.id = ve.vault_key_id AND vk.org_id = ve.org_id
                             WHERE ve.org_id = $1 AND ve.purged_at IS NULL",
                        )
                        .bind(org_id)
                        .fetch_all(&mut **tx)
                        .await
                    }
                    .map_err(|e| OpenRagError::Database(e.to_string()))?;

                    for vr in vrows {
                        let vid: Uuid = vr.try_get("id").map_err(|e| OpenRagError::Database(e.to_string()))?;
                        let tok: String = vr.try_get("token").map_err(|e| OpenRagError::Database(e.to_string()))?;
                        let ct: Vec<u8> = vr.try_get("ciphertext").map_err(|e| OpenRagError::Database(e.to_string()))?;
                        let nonce: Vec<u8> = vr.try_get("nonce").map_err(|e| OpenRagError::Database(e.to_string()))?;
                        let wdek: Vec<u8> = vr.try_get("wrapped_dek").map_err(|e| OpenRagError::Database(e.to_string()))?;
                        if let Ok(dek) = VaultCrypto::unwrap_dek(master_kek, org_id, &wdek) {
                            if let Ok(plain) = VaultCrypto::decrypt_value(&dek, org_id, &tok, &ct, &nonce) {
                                if plain.contains(pat) {
                                    vault_ids.push(vid);
                                }
                            }
                        }
                    }
                }
            }
        }

        // Deduplicate IDs
        dedup_uuids(&mut doc_ids);
        dedup_uuids(&mut chunk_ids);
        dedup_uuids(&mut page_ids);
        dedup_uuids(&mut memory_ids);
        dedup_uuids(&mut episode_ids);
        dedup_uuids(&mut wiki_ids);
        dedup_uuids(&mut vault_ids);

        // 1. Scrub corpus.chunks
        for cid in &chunk_ids {
            let row = sqlx::query(
                "SELECT c.redacted_text, c.bm25_text, c.table_html, d.title AS doc_title
                 FROM corpus.chunks c
                 LEFT JOIN corpus.documents d ON d.org_id = c.org_id AND d.id = c.document_id
                 WHERE c.org_id = $1 AND c.id = $2",
            )
            .bind(org_id)
            .bind(cid)
            .fetch_optional(&mut **tx)
            .await
            .map_err(|e| OpenRagError::Database(e.to_string()))?;

            let Some(row) = row else {
                continue;
            };
            let old_red: String = row.try_get("redacted_text").map_err(|e| OpenRagError::Database(e.to_string()))?;
            let old_bm25: String = row.try_get("bm25_text").map_err(|e| OpenRagError::Database(e.to_string()))?;
            let old_tbl: Option<String> = row.try_get("table_html").map_err(|e| OpenRagError::Database(e.to_string()))?;
            let doc_title: Option<String> = row.try_get("doc_title").map_err(|e| OpenRagError::Database(e.to_string()))?;

            let new_red = scrub_text(&old_red, pattern.as_deref(), replacement);
            let new_bm25 = scrub_text(&old_bm25, pattern.as_deref(), replacement);
            let new_tbl = old_tbl.map(|t| scrub_text(&t, pattern.as_deref(), replacement));
            let new_hash = sha256_hex_str(&new_red);
            // Match ingestion's contextual embedding so redacted chunks stay in the same vector space.
            let embed_title = embed_document_title.then_some(doc_title.as_deref()).flatten();
            let new_emb = Vector::from(text_embedder.embed_document(embed_title, &new_red).await?);

            sqlx::query(
                "UPDATE corpus.chunks
                 SET redacted_text = $3, bm25_text = $4, table_html = $5, text_hash = $6, embedding = $7
                 WHERE org_id = $1 AND id = $2",
            )
            .bind(org_id)
            .bind(cid)
            .bind(&new_red)
            .bind(&new_bm25)
            .bind(&new_tbl)
            .bind(&new_hash)
            .bind(new_emb)
            .execute(&mut **tx)
            .await
            .map_err(|e| OpenRagError::Database(e.to_string()))?;

            insert_redaction_event(tx, job_id, org_id, "corpus.chunks", *cid, "text_masked").await?;
            enqueue_outbox(tx, org_id, "corpus.chunks", *cid, outbox_op, Some(&new_hash)).await?;
            counts.chunks += 1;
            counts.outbox_enqueued += 1;
        }

        // 2. Scrub corpus.pages + optional black-box BBox thumbnail masking + 256d re-embed
        let chunk_bboxes_by_page = load_page_bboxes(tx, org_id, &page_ids).await?;

        for pid in &page_ids {
            let row = sqlx::query(
                "SELECT raw_markdown, redacted_markdown, thumb_uri, redacted_thumb_uri, content_sha256
                 FROM corpus.pages
                 WHERE org_id = $1 AND id = $2",
            )
            .bind(org_id)
            .bind(pid)
            .fetch_optional(&mut **tx)
            .await
            .map_err(|e| OpenRagError::Database(e.to_string()))?;

            let Some(row) = row else {
                continue;
            };
            let old_raw: Option<String> = row.try_get("raw_markdown").map_err(|e| OpenRagError::Database(e.to_string()))?;
            let old_red: Option<String> = row.try_get("redacted_markdown").map_err(|e| OpenRagError::Database(e.to_string()))?;
            let thumb_uri: Option<String> = row.try_get("thumb_uri").map_err(|e| OpenRagError::Database(e.to_string()))?;
            let redacted_thumb_uri: Option<String> = row.try_get("redacted_thumb_uri").map_err(|e| OpenRagError::Database(e.to_string()))?;

            let new_red = old_red.map(|t| scrub_text(&t, pattern.as_deref(), replacement));
            let new_raw = if is_hard_purge || is_tombstone {
                old_raw.map(|t| scrub_text(&t, pattern.as_deref(), replacement))
            } else {
                old_raw
            };
            let new_sha = new_red
                .as_deref()
                .map(sha256_hex_str)
                .unwrap_or_else(|| sha256_hex_str(replacement));

            sqlx::query(
                "UPDATE corpus.pages
                 SET raw_markdown = $3, redacted_markdown = $4, content_sha256 = $5
                 WHERE org_id = $1 AND id = $2",
            )
            .bind(org_id)
            .bind(pid)
            .bind(&new_raw)
            .bind(&new_red)
            .bind(&new_sha)
            .execute(&mut **tx)
            .await
            .map_err(|e| OpenRagError::Database(e.to_string()))?;

            insert_redaction_event(tx, job_id, org_id, "corpus.pages", *pid, "text_masked").await?;
            counts.pages += 1;

            if req.redact_thumbnails {
                let source_uri = redacted_thumb_uri.as_deref().or(thumb_uri.as_deref());
                if let Some(src_uri) = source_uri {
                    let mut bboxes = Vec::new();
                    if let Some(b) = explicit_bbox {
                        bboxes.push(b);
                    } else if let Some(db_boxes) = chunk_bboxes_by_page.get(pid) {
                        bboxes.extend(db_boxes.iter().copied());
                    }
                    if bboxes.is_empty() && req.target_scope == RedactionTargetScope::Document {
                        bboxes.push(BBoxNorm1000::new(0, 0, 999, 999)?);
                    }

                    if !bboxes.is_empty() {
                        let orig_bytes = thumb_store.read_bytes(src_uri).await?;
                        let masked_bytes = mask_image_bboxes(&orig_bytes, &bboxes)?;
                        let out_uri = if is_hard_purge {
                            if let Some(orig) = thumb_uri.as_deref() {
                                thumb_store.write_bytes(orig, &masked_bytes).await?;
                            }
                            redacted_thumb_uri
                                .clone()
                                .unwrap_or_else(|| LocalThumbnailStore::derive_redacted_uri(src_uri))
                        } else {
                            redacted_thumb_uri
                                .clone()
                                .unwrap_or_else(|| LocalThumbnailStore::derive_redacted_uri(src_uri))
                        };
                        thumb_store.write_bytes(&out_uri, &masked_bytes).await?;

                        sqlx::query(
                            "UPDATE corpus.pages
                             SET redacted_thumb_uri = $3
                             WHERE org_id = $1 AND id = $2",
                        )
                        .bind(org_id)
                        .bind(pid)
                        .bind(&out_uri)
                        .execute(&mut **tx)
                        .await
                        .map_err(|e| OpenRagError::Database(e.to_string()))?;

                        insert_redaction_event(
                            tx,
                            job_id,
                            org_id,
                            "corpus.pages",
                            *pid,
                            "thumb_bbox_blacked",
                        )
                        .await?;
                        counts.thumbnails_masked += 1;
                    }
                }
            }

            counts.outbox_enqueued += 1;
        }

        // 3. Update corpus.documents status
        for did in &doc_ids {
            let res = sqlx::query(
                "UPDATE corpus.documents SET status = 'redacted', updated_at = now()
                 WHERE org_id = $1 AND id = $2",
            )
            .bind(org_id)
            .bind(did)
            .execute(&mut **tx)
            .await
            .map_err(|e| OpenRagError::Database(e.to_string()))?;
            if res.rows_affected() > 0 {
                insert_redaction_event(tx, job_id, org_id, "corpus.documents", *did, "status_redacted")
                    .await?;
                counts.documents += 1;
            }
        }

        // 4. Scrub memory.entries & memory.patches (clear prev_text = NULL on hard_purge)
        for mid in &memory_ids {
            let row = sqlx::query(
                "SELECT effective_text, bm25_text, version, access_scope
                 FROM memory.entries
                 WHERE org_id = $1 AND id = $2",
            )
            .bind(org_id)
            .bind(mid)
            .fetch_optional(&mut **tx)
            .await
            .map_err(|e| OpenRagError::Database(e.to_string()))?;

            let Some(row) = row else {
                continue;
            };
            let old_text: String = row.try_get("effective_text").map_err(|e| OpenRagError::Database(e.to_string()))?;
            let old_bm25: String = row.try_get("bm25_text").map_err(|e| OpenRagError::Database(e.to_string()))?;
            let version: i32 = row.try_get("version").map_err(|e| OpenRagError::Database(e.to_string()))?;
            let scope: String = row.try_get("access_scope").map_err(|e| OpenRagError::Database(e.to_string()))?;

            let new_text = scrub_text(&old_text, pattern.as_deref(), replacement);
            let new_bm25 = scrub_text(&old_bm25, pattern.as_deref(), replacement);
            let new_sha = sha256_hex_str(&new_text);
            let new_ver = version + 1;
            let new_emb = Vector::from(text_embedder.embed_text(&new_text).await?);

            sqlx::query(
                "UPDATE memory.entries
                 SET effective_text = $3, bm25_text = $4, content_sha256 = $5,
                     version = $6, status = 'redacted', updated_at = now(), embedding = $7
                 WHERE org_id = $1 AND id = $2",
            )
            .bind(org_id)
            .bind(mid)
            .bind(&new_text)
            .bind(&new_bm25)
            .bind(&new_sha)
            .bind(new_ver)
            .bind(new_emb)
            .execute(&mut **tx)
            .await
            .map_err(|e| OpenRagError::Database(e.to_string()))?;

            if is_hard_purge {
                let cleared = sqlx::query(
                    "UPDATE memory.patches
                     SET prev_text = NULL,
                         new_text = CASE WHEN new_text IS NOT NULL THEN $3 ELSE NULL END
                     WHERE org_id = $1 AND memory_entry_id = $2",
                )
                .bind(org_id)
                .bind(mid)
                .bind(replacement)
                .execute(&mut **tx)
                .await
                .map_err(|e| OpenRagError::Database(e.to_string()))?;
                counts.memory_patches_cleared += cleared.rows_affected();
            }

            sqlx::query(
                "INSERT INTO memory.patches (
                    org_id, memory_entry_id, patch_op, prev_version, new_version,
                    prev_text, new_text, prev_scope, new_scope, reason, actor_type, actor_id
                 ) VALUES ($1, $2, 'redact', $3, $4, $5, $6, $7, $7, $8, 'redactor', $9)",
            )
            .bind(org_id)
            .bind(mid)
            .bind(version)
            .bind(new_ver)
            .bind(if is_hard_purge { None::<&str> } else { Some(old_text.as_str()) })
            .bind(&new_text)
            .bind(&scope)
            .bind(&req.reason)
            .bind(caller.principal_id.to_string())
            .execute(&mut **tx)
            .await
            .map_err(|e| OpenRagError::Database(e.to_string()))?;

            insert_redaction_event(tx, job_id, org_id, "memory.entries", *mid, "memory_redacted")
                .await?;
            enqueue_outbox(tx, org_id, "memory.entries", *mid, outbox_op, Some(&new_sha)).await?;
            counts.memories += 1;
            counts.outbox_enqueued += 1;
        }

        // 5. Scrub memory.episodes
        for eid in &episode_ids {
            let row = sqlx::query(
                "SELECT summary_md FROM memory.episodes WHERE org_id = $1 AND id = $2",
            )
            .bind(org_id)
            .bind(eid)
            .fetch_optional(&mut **tx)
            .await
            .map_err(|e| OpenRagError::Database(e.to_string()))?;
            if let Some(row) = row {
                let old_md: Option<String> = row.try_get("summary_md").map_err(|e| OpenRagError::Database(e.to_string()))?;
                let new_md = old_md.map(|t| scrub_text(&t, pattern.as_deref(), replacement));
                let new_sha = sha256_hex_str(new_md.as_deref().unwrap_or(replacement));
                sqlx::query(
                    "UPDATE memory.episodes
                     SET summary_md = $3,
                         raw_redacted_json = CASE WHEN $4 THEN '{}'::jsonb ELSE raw_redacted_json END,
                         content_sha256 = $5,
                         redaction_applied = true
                     WHERE org_id = $1 AND id = $2",
                )
                .bind(org_id)
                .bind(eid)
                .bind(&new_md)
                .bind(is_hard_purge)
                .bind(&new_sha)
                .execute(&mut **tx)
                .await
                .map_err(|e| OpenRagError::Database(e.to_string()))?;
                insert_redaction_event(tx, job_id, org_id, "memory.episodes", *eid, "episode_redacted")
                    .await?;
                counts.memory_episodes += 1;
            }
        }

        // 6. Scrub wiki.pages & wiki.revisions
        for wid in &wiki_ids {
            let row = sqlx::query(
                "SELECT title, body_md, bm25_text, version FROM wiki.pages WHERE org_id = $1 AND id = $2",
            )
            .bind(org_id)
            .bind(wid)
            .fetch_optional(&mut **tx)
            .await
            .map_err(|e| OpenRagError::Database(e.to_string()))?;

            let Some(row) = row else {
                continue;
            };
            let title: String = row.try_get("title").map_err(|e| OpenRagError::Database(e.to_string()))?;
            let old_body: String = row.try_get("body_md").map_err(|e| OpenRagError::Database(e.to_string()))?;
            let old_bm25: String = row.try_get("bm25_text").map_err(|e| OpenRagError::Database(e.to_string()))?;
            let version: i32 = row.try_get("version").map_err(|e| OpenRagError::Database(e.to_string()))?;

            let new_body = scrub_text(&old_body, pattern.as_deref(), replacement);
            let new_bm25 = scrub_text(&old_bm25, pattern.as_deref(), replacement);
            let new_sha = sha256_hex_str(&new_body);
            let new_ver = version + 1;
            // Same input as the wiki write path (openrag-memory wiki.rs): title, newline, body.
            let new_emb = Vector::from(text_embedder.embed_text(&format!("{title}\n{new_body}")).await?);

            sqlx::query(
                "UPDATE wiki.pages
                 SET body_md = $3, bm25_text = $4, body_sha256 = $5,
                     version = $6, status = 'redacted', updated_at = now(), embedding = $7
                 WHERE org_id = $1 AND id = $2",
            )
            .bind(org_id)
            .bind(wid)
            .bind(&new_body)
            .bind(&new_bm25)
            .bind(&new_sha)
            .bind(new_ver)
            .bind(new_emb)
            .execute(&mut **tx)
            .await
            .map_err(|e| OpenRagError::Database(e.to_string()))?;

            if is_hard_purge {
                let rev_upd = sqlx::query(
                    "UPDATE wiki.revisions
                     SET body_md = $3, body_sha256 = $4
                     WHERE org_id = $1 AND page_id = $2",
                )
                .bind(org_id)
                .bind(wid)
                .bind(replacement)
                .bind(sha256_hex_str(replacement))
                .execute(&mut **tx)
                .await
                .map_err(|e| OpenRagError::Database(e.to_string()))?;
                counts.wiki_revisions += rev_upd.rows_affected();
            }

            sqlx::query(
                "INSERT INTO wiki.revisions (
                    org_id, page_id, version, body_md, body_sha256,
                    editor_type, editor_id, change_note
                 ) VALUES ($1, $2, $3, $4, $5, 'redactor', $6, $7)",
            )
            .bind(org_id)
            .bind(wid)
            .bind(new_ver)
            .bind(&new_body)
            .bind(&new_sha)
            .bind(caller.principal_id.to_string())
            .bind(&req.reason)
            .execute(&mut **tx)
            .await
            .map_err(|e| OpenRagError::Database(e.to_string()))?;

            insert_redaction_event(tx, job_id, org_id, "wiki.pages", *wid, "wiki_redacted").await?;
            enqueue_outbox(tx, org_id, "wiki.pages", *wid, outbox_op, Some(&new_sha)).await?;
            counts.wiki_pages += 1;
            counts.outbox_enqueued += 1;
        }

        // 7. Purge pii.vault_entries on HardPurge / TombstoneDelete
        if is_hard_purge || is_tombstone {
            for vid in &vault_ids {
                let res = sqlx::query(
                    "UPDATE pii.vault_entries
                     SET purged_at = now(),
                         ciphertext = '\\x00'::bytea,
                         nonce = '\\x00'::bytea
                     WHERE org_id = $1 AND id = $2 AND purged_at IS NULL",
                )
                .bind(org_id)
                .bind(vid)
                .execute(&mut **tx)
                .await
                .map_err(|e| OpenRagError::Database(e.to_string()))?;
                if res.rows_affected() > 0 {
                    insert_redaction_event(
                        tx,
                        job_id,
                        org_id,
                        "pii.vault_entries",
                        *vid,
                        "vault_purged",
                    )
                    .await?;
                    counts.vault_entries_purged += 1;
                }
            }
        }

        Ok(())
    }
}

async fn load_page_bboxes(
    tx: &mut Transaction<'_, Postgres>,
    org_id: Uuid,
    page_ids: &[Uuid],
) -> Result<HashMap<Uuid, Vec<BBoxNorm1000>>, OpenRagError> {
    let mut map: HashMap<Uuid, Vec<BBoxNorm1000>> = HashMap::new();
    if page_ids.is_empty() {
        return Ok(map);
    }

    let rows = sqlx::query(
        "SELECT page_id, bbox_norm_1000 FROM pii.chunk_spans
         WHERE org_id = $1 AND page_id = ANY($2) AND bbox_norm_1000 IS NOT NULL
         UNION ALL
         SELECT page_id, bbox_norm_1000 FROM corpus.chunks
         WHERE org_id = $1 AND page_id = ANY($2) AND bbox_norm_1000 IS NOT NULL
         UNION ALL
         SELECT page_id, bbox_norm_1000 FROM corpus.page_figures
         WHERE org_id = $1 AND page_id = ANY($2) AND bbox_norm_1000 IS NOT NULL",
    )
    .bind(org_id)
    .bind(page_ids)
    .fetch_all(&mut **tx)
    .await
    .map_err(|e| OpenRagError::Database(format!("failed loading page bboxes: {e}")))?;

    for row in rows {
        let pid: Option<Uuid> = row.try_get("page_id").map_err(|e| OpenRagError::Database(e.to_string()))?;
        let arr: Option<Vec<i32>> = row.try_get("bbox_norm_1000").map_err(|e| OpenRagError::Database(e.to_string()))?;
        if let (Some(pid), Some(coords)) = (pid, arr) {
            if coords.len() == 4 {
                if let Ok(bbox) = BBoxNorm1000::new(coords[0], coords[1], coords[2], coords[3]) {
                    map.entry(pid).or_default().push(bbox);
                }
            }
        }
    }
    Ok(map)
}

async fn insert_redaction_event(
    tx: &mut Transaction<'_, Postgres>,
    job_id: Uuid,
    org_id: Uuid,
    target_table: &str,
    target_id: Uuid,
    action_taken: &str,
) -> Result<(), OpenRagError> {
    sqlx::query(
        "INSERT INTO pii.redaction_events (
            redaction_job_id, org_id, target_table, target_id, action_taken
         ) VALUES ($1, $2, $3, $4, $5)",
    )
    .bind(job_id)
    .bind(org_id)
    .bind(target_table)
    .bind(target_id)
    .bind(action_taken)
    .execute(&mut **tx)
    .await
    .map_err(|e| OpenRagError::Database(format!("failed inserting pii.redaction_events: {e}")))?;
    Ok(())
}

async fn enqueue_outbox(
    tx: &mut Transaction<'_, Postgres>,
    org_id: Uuid,
    target_table: &str,
    target_id: Uuid,
    op: &str,
    content_sha256: Option<&str>,
) -> Result<(), OpenRagError> {
    sqlx::query(
        "INSERT INTO indexing.outbox (
            org_id, target_table, target_id, op, content_sha256, priority, payload
         ) VALUES ($1, $2, $3, $4, $5, 1, '{}'::jsonb)",
    )
    .bind(org_id)
    .bind(target_table)
    .bind(target_id)
    .bind(op)
    .bind(content_sha256)
    .execute(&mut **tx)
    .await
    .map_err(|e| OpenRagError::Database(format!("failed enqueuing indexing.outbox: {e}")))?;
    Ok(())
}

fn scrub_text(text: &str, pattern: Option<&str>, replacement: &str) -> String {
    match pattern {
        Some(pat) if !pat.is_empty() => text.replace(pat, replacement),
        _ => replacement.to_string(),
    }
}

fn sha256_hex_str(s: &str) -> String {
    let digest = Sha256::digest(s.as_bytes());
    hex::encode(digest)
}

fn parse_uuid_field(obj: &serde_json::Value, key: &str) -> Result<Uuid, OpenRagError> {
    let raw = obj
        .get(key)
        .and_then(|v| v.as_str())
        .ok_or_else(|| OpenRagError::Validation(format!("target_ref missing UUID field '{key}'")))?;
    Uuid::parse_str(raw)
        .map_err(|e| OpenRagError::Validation(format!("invalid UUID in target_ref.{key}: {e}")))
}

fn parse_optional_bbox(obj: &serde_json::Value) -> Result<Option<BBoxNorm1000>, OpenRagError> {
    let Some(val) = obj.get("bbox_norm_1000") else {
        return Ok(None);
    };
    let arr = val.as_array().ok_or_else(|| {
        OpenRagError::Validation("target_ref.bbox_norm_1000 must be an array of 4 ints".into())
    })?;
    if arr.len() != 4 {
        return Err(OpenRagError::Validation(
            "target_ref.bbox_norm_1000 must have exactly 4 elements".into(),
        ));
    }
    let x0 = arr[0]
        .as_i64()
        .ok_or_else(|| OpenRagError::Validation("bbox[0] must be int".into()))? as i32;
    let y0 = arr[1]
        .as_i64()
        .ok_or_else(|| OpenRagError::Validation("bbox[1] must be int".into()))? as i32;
    let x1 = arr[2]
        .as_i64()
        .ok_or_else(|| OpenRagError::Validation("bbox[2] must be int".into()))? as i32;
    let y1 = arr[3]
        .as_i64()
        .ok_or_else(|| OpenRagError::Validation("bbox[3] must be int".into()))? as i32;
    Ok(Some(BBoxNorm1000::new(x0, y0, x1, y1)?))
}

fn dedup_uuids(items: &mut Vec<Uuid>) {
    let mut seen = HashSet::new();
    items.retain(|id| seen.insert(*id));
}
