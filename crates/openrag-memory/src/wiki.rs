use std::collections::BTreeSet;
use std::fs;
use std::path::{Component, Path, PathBuf};

use openrag_core::{AccessScope, Arn, ArnResourceType, AuthLevel, OpenRagError, PrincipalContext};
use openrag_db::RlsTransaction;
use openrag_store::{
    expand_tag_filter_rls, normalize_tag, push_tag_filter_predicates, sha256_hex,
    EnqueueOutboxItem, OutboxEnqueuer, OutboxOp, TagFilter, VectorTable,
};
use pgvector::Vector;
use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::json;
use sqlx::{Postgres, QueryBuilder, Row};
use uuid::Uuid;

use crate::concepts::ConceptGraph;
use crate::embed::TextEmbedder;
use crate::types::{
    ConceptCategory, ConceptSource, SyncDocInput, SyncDocOutcome, SyncDocResult, WikiEdge,
    WikiEditorType, WikiImportReport, WikiPage, WikiRelationType, WikiSearchHit,
};

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ParsedFrontmatter {
    pub title: Option<String>,
    pub slug: Option<String>,
    pub scope: Option<AccessScope>,
    pub min_auth_level: Option<u8>,
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(default)]
    pub aliases: Vec<String>,
    pub supersedes: Option<String>,
    pub body_md: String,
}

pub fn parse_frontmatter(raw: &str) -> Result<ParsedFrontmatter, OpenRagError> {
    let trimmed = raw.trim_start_matches('\u{feff}');
    let Some(after_start) = trimmed
        .strip_prefix("---\n")
        .or_else(|| trimmed.strip_prefix("---\r\n"))
    else {
        return Ok(ParsedFrontmatter {
            body_md: raw.to_string(),
            ..Default::default()
        });
    };

    let end_idx = after_start
        .find("\n---\n")
        .or_else(|| after_start.find("\r\n---\r\n"))
        .or_else(|| after_start.find("\n---\r\n"))
        .or_else(|| after_start.find("\r\n---\n"))
        .or_else(|| {
            if after_start.ends_with("\n---") {
                Some(after_start.len() - 4)
            } else if after_start.ends_with("\r\n---") {
                Some(after_start.len() - 5)
            } else {
                None
            }
        });
    let Some(idx) = end_idx else {
        return Ok(ParsedFrontmatter {
            body_md: raw.to_string(),
            ..Default::default()
        });
    };

    let yaml_part = &after_start[..idx];
    let rest = &after_start[idx..];
    let body = rest
        .strip_prefix("\n---\n")
        .or_else(|| rest.strip_prefix("\r\n---\r\n"))
        .or_else(|| rest.strip_prefix("\n---\r\n"))
        .or_else(|| rest.strip_prefix("\r\n---\n"))
        .or_else(|| rest.strip_prefix("\n---"))
        .or_else(|| rest.strip_prefix("\r\n---"))
        .unwrap_or(rest)
        .trim_start_matches(['\r', '\n'])
        .trim_end()
        .to_string();

    #[derive(Deserialize, Default)]
    struct FmYaml {
        title: Option<String>,
        slug: Option<String>,
        scope: Option<String>,
        min_auth_level: Option<u8>,
        #[serde(default)]
        tags: Vec<String>,
        #[serde(default)]
        aliases: Vec<String>,
        supersedes: Option<String>,
    }

    let parsed: FmYaml = serde_yaml_ng::from_str(yaml_part).map_err(|e| {
        OpenRagError::Validation(format!("invalid wiki YAML frontmatter: {e}"))
    })?;
    let scope = parsed
        .scope
        .as_deref()
        .map(str::parse)
        .transpose()?;

    Ok(ParsedFrontmatter {
        title: parsed.title,
        slug: parsed.slug,
        scope,
        min_auth_level: parsed.min_auth_level,
        tags: normalize_tags_vec(&parsed.tags),
        aliases: parsed.aliases,
        supersedes: parsed.supersedes,
        body_md: body,
    })
}

pub fn extract_wikilinks(body_md: &str) -> Vec<String> {
    let Ok(re) = Regex::new(r"\[\[([^\]\|]+)(?:\|[^\]]+)?\]\]") else {
        return Vec::new();
    };
    let mut out = Vec::new();
    let mut seen = BTreeSet::new();
    for cap in re.captures_iter(body_md) {
        if let Some(m) = cap.get(1) {
            let slug = slugify(m.as_str());
            if !slug.is_empty() && seen.insert(slug.clone()) {
                out.push(slug);
            }
        }
    }
    out
}

pub fn render_frontmatter_markdown(page: &WikiPage) -> Result<String, OpenRagError> {
    #[derive(Serialize)]
    struct FmOut<'a> {
        title: &'a str,
        slug: &'a str,
        scope: &'a str,
        min_auth_level: u8,
        version: i32,
        tags: &'a [String],
    }

    let fm = FmOut {
        title: &page.title,
        slug: &page.slug,
        scope: page.access_scope.as_str(),
        min_auth_level: page.min_auth_level.as_u8(),
        version: page.version,
        tags: &page.tags,
    };
    let yaml = serde_yaml_ng::to_string(&fm).map_err(|e| {
        OpenRagError::Internal(format!("failed to serialize wiki frontmatter: {e}"))
    })?;
    Ok(format!("---\n{yaml}---\n\n{}\n", page.body_md.trim_end()))
}

pub fn slugify(raw: &str) -> String {
    let trimmed = raw.trim().trim_end_matches(".md");
    let mut out = String::new();
    let mut prev_dash = false;
    for ch in trimmed.chars() {
        if ch.is_alphanumeric() {
            for lc in ch.to_lowercase() {
                out.push(lc);
            }
            prev_dash = false;
        } else if (ch == '-' || ch == '_' || ch.is_whitespace()) && !prev_dash && !out.is_empty() {
            out.push('-');
            prev_dash = true;
        }
    }
    out.trim_matches('-').to_string()
}

fn normalize_tags_vec(tags: &[String]) -> Vec<String> {
    let mut set = BTreeSet::new();
    for t in tags {
        let n = normalize_tag(t);
        if !n.is_empty() {
            set.insert(n);
        }
    }
    set.into_iter().collect()
}

pub struct WikiCompiler;

impl WikiCompiler {
    #[allow(clippy::too_many_arguments)]
    pub async fn upsert_page_rls(
        tx: &mut RlsTransaction<'_>,
        ctx: &PrincipalContext,
        bucket_id: Option<Uuid>,
        dept_id: Uuid,
        org_slug: &str,
        dept_path: &str,
        access_scope: AccessScope,
        min_auth_level: AuthLevel,
        slug: &str,
        title: &str,
        body_md: &str,
        tags: &[String],
        editor_type: WikiEditorType,
        change_note: Option<&str>,
        embedder: &dyn TextEmbedder,
    ) -> Result<(WikiPage, bool), OpenRagError> {
        let norm_slug = slugify(slug);
        if norm_slug.is_empty() {
            return Err(OpenRagError::Validation("wiki slug must not be empty".into()));
        }
        let body_md = body_md.trim_end();
        let body_sha = sha256_hex(body_md.as_bytes());
        let norm_tags = normalize_tags_vec(tags);
        let actor_id = ctx.principal_id.to_string();

        let existing = Self::get_page_by_slug_rls(tx, ctx.org_id, bucket_id, &norm_slug).await?;
        if let Some(cur) = existing {
            if cur.body_sha256 == body_sha && cur.title == title && cur.tags == norm_tags {
                return Ok((cur, false));
            }
            let emb = embedder.embed_text(&format!("{title}\n{body_md}")).await?;
            let next_ver = cur.version + 1;
            let row = sqlx::query(
                "UPDATE wiki.pages
                 SET title = $1,
                     body_md = $2,
                     bm25_text = $3,
                     body_sha256 = $4,
                     embedding = $5,
                     tags = $6,
                     version = $7,
                     updated_at = now()
                 WHERE org_id = $8 AND id = $9
                 RETURNING id, org_id, bucket_id, dept_id, owner_principal_id, access_scope,
                           allowed_group_ids, denied_group_ids, min_auth_level, arn, acl_scope_key,
                           slug, title, body_md, bm25_text, body_sha256, tags, status, version,
                           created_at, updated_at",
            )
            .bind(title)
            .bind(body_md)
            .bind(format!("{title}\n{body_md}"))
            .bind(&body_sha)
            .bind(Vector::from(emb))
            .bind(&norm_tags)
            .bind(next_ver)
            .bind(ctx.org_id)
            .bind(cur.id)
            .fetch_one(&mut **tx.tx_mut())
            .await
            .map_err(|e| map_pg_err(e, "update wiki.pages"))?;
            let updated = row_to_wiki_page(&row)?;

            sqlx::query(
                "INSERT INTO wiki.revisions
                 (org_id, page_id, version, body_md, body_sha256, editor_type, editor_id, change_note)
                 VALUES ($1, $2, $3, $4, $5, $6, $7, $8)",
            )
            .bind(ctx.org_id)
            .bind(updated.id)
            .bind(next_ver)
            .bind(body_md)
            .bind(&body_sha)
            .bind(editor_type.as_str())
            .bind(&actor_id)
            .bind(change_note)
            .execute(&mut **tx.tx_mut())
            .await
            .map_err(|e| map_pg_err(e, "insert wiki.revisions"))?;

            OutboxEnqueuer::enqueue_rls(
                tx,
                &EnqueueOutboxItem {
                    org_id: ctx.org_id,
                    target_table: VectorTable::WikiPages,
                    target_id: updated.id,
                    op: OutboxOp::UpsertEmbed,
                    content_sha256: Some(body_sha),
                    priority: 4,
                    payload: json!({ "slug": updated.slug, "version": updated.version }),
                    idempotency_key: None,
                    max_attempts: None,
                },
                false,
            )
            .await?;

            return Ok((updated, true));
        }

        let page_id = Uuid::new_v4();
        let arn = Arn::new(
            org_slug,
            dept_path,
            access_scope,
            ArnResourceType::Wiki,
            min_auth_level,
            page_id.to_string(),
        )?
        .to_canonical();
        let acl_scope_key = format!(
            "{dept_path}:{}:L{}",
            access_scope.as_str(),
            min_auth_level.as_u8()
        );
        let emb = embedder.embed_text(&format!("{title}\n{body_md}")).await?;

        let row = sqlx::query(
            "INSERT INTO wiki.pages
             (id, org_id, bucket_id, dept_id, owner_principal_id, access_scope, min_auth_level,
              arn, acl_scope_key, slug, title, body_md, bm25_text, body_sha256, embedding, tags)
             VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16)
             RETURNING id, org_id, bucket_id, dept_id, owner_principal_id, access_scope,
                       allowed_group_ids, denied_group_ids, min_auth_level, arn, acl_scope_key,
                       slug, title, body_md, bm25_text, body_sha256, tags, status, version,
                       created_at, updated_at",
        )
        .bind(page_id)
        .bind(ctx.org_id)
        .bind(bucket_id)
        .bind(dept_id)
        .bind(ctx.principal_id)
        .bind(access_scope.as_str())
        .bind(min_auth_level.as_i16())
        .bind(&arn)
        .bind(&acl_scope_key)
        .bind(&norm_slug)
        .bind(title)
        .bind(body_md)
        .bind(format!("{title}\n{body_md}"))
        .bind(&body_sha)
        .bind(Vector::from(emb))
        .bind(&norm_tags)
        .fetch_one(&mut **tx.tx_mut())
        .await
        .map_err(|e| map_pg_err(e, "insert wiki.pages"))?;
        let page = row_to_wiki_page(&row)?;

        sqlx::query(
            "INSERT INTO wiki.revisions
             (org_id, page_id, version, body_md, body_sha256, editor_type, editor_id, change_note)
             VALUES ($1, $2, 1, $3, $4, $5, $6, $7)",
        )
        .bind(ctx.org_id)
        .bind(page.id)
        .bind(body_md)
        .bind(&body_sha)
        .bind(editor_type.as_str())
        .bind(&actor_id)
        .bind(change_note)
        .execute(&mut **tx.tx_mut())
        .await
        .map_err(|e| map_pg_err(e, "insert wiki.revisions"))?;

        OutboxEnqueuer::enqueue_rls(
            tx,
            &EnqueueOutboxItem {
                org_id: ctx.org_id,
                target_table: VectorTable::WikiPages,
                target_id: page.id,
                op: OutboxOp::UpsertEmbed,
                content_sha256: Some(body_sha),
                priority: 4,
                payload: json!({ "slug": page.slug, "version": 1 }),
                idempotency_key: None,
                max_attempts: None,
            },
            false,
        )
        .await?;

        Ok((page, true))
    }

    pub async fn get_page_by_id_rls(
        tx: &mut RlsTransaction<'_>,
        org_id: Uuid,
        page_id: Uuid,
    ) -> Result<Option<WikiPage>, OpenRagError> {
        let row = sqlx::query(
            "SELECT id, org_id, bucket_id, dept_id, owner_principal_id, access_scope,
                    allowed_group_ids, denied_group_ids, min_auth_level, arn, acl_scope_key,
                    slug, title, body_md, bm25_text, body_sha256, tags, status, version,
                    created_at, updated_at
             FROM wiki.pages
             WHERE org_id = $1 AND id = $2",
        )
        .bind(org_id)
        .bind(page_id)
        .fetch_optional(&mut **tx.tx_mut())
        .await
        .map_err(|e| OpenRagError::Database(format!("failed to fetch wiki page: {e}")))?;
        row.as_ref().map(row_to_wiki_page).transpose()
    }

    pub async fn get_page_by_slug_rls(
        tx: &mut RlsTransaction<'_>,
        org_id: Uuid,
        bucket_id: Option<Uuid>,
        slug: &str,
    ) -> Result<Option<WikiPage>, OpenRagError> {
        let row = sqlx::query(
            "SELECT id, org_id, bucket_id, dept_id, owner_principal_id, access_scope,
                    allowed_group_ids, denied_group_ids, min_auth_level, arn, acl_scope_key,
                    slug, title, body_md, bm25_text, body_sha256, tags, status, version,
                    created_at, updated_at
             FROM wiki.pages
             WHERE org_id = $1 AND slug = $2 AND ($3::uuid IS NULL OR bucket_id = $3)
             LIMIT 1",
        )
        .bind(org_id)
        .bind(slug)
        .bind(bucket_id)
        .fetch_optional(&mut **tx.tx_mut())
        .await
        .map_err(|e| OpenRagError::Database(format!("failed to fetch wiki page by slug: {e}")))?;
        row.as_ref().map(row_to_wiki_page).transpose()
    }

    pub async fn search_pages_rls(
        tx: &mut RlsTransaction<'_>,
        org_id: Uuid,
        query: &str,
        tag_filter: Option<&TagFilter>,
        top_k: usize,
        embedder: &dyn TextEmbedder,
    ) -> Result<Vec<WikiSearchHit>, OpenRagError> {
        if top_k == 0 {
            return Ok(Vec::new());
        }
        let raw_terms: Vec<String> = query.split_whitespace().map(ToString::to_string).collect();
        let expanded_terms = ConceptGraph::expand_terms_rls(tx, org_id, &raw_terms).await?;
        let expanded_tags = if let Some(tf) = tag_filter {
            Some(expand_tag_filter_rls(tx, org_id, tf).await?)
        } else {
            None
        };

        let embed_input = if expanded_terms.is_empty() {
            query.to_string()
        } else {
            expanded_terms.join(" ")
        };
        let qvec = Vector::from(embedder.embed_query(&embed_input).await?);

        let mut qb: QueryBuilder<'_, Postgres> = QueryBuilder::new(
            "SELECT id, org_id, bucket_id, dept_id, owner_principal_id, access_scope,
                    allowed_group_ids, denied_group_ids, min_auth_level, arn, acl_scope_key,
                    slug, title, body_md, bm25_text, body_sha256, tags, status, version,
                    created_at, updated_at,
                    CASE WHEN embedding IS NOT NULL THEN (1.0 - (embedding <=> ",
        );
        qb.push_bind(qvec);
        qb.push("))::real ELSE 0.0::real END AS vec_sim FROM wiki.pages WHERE org_id = ");
        qb.push_bind(org_id);
        qb.push(" AND status = 'active'");
        if let Some(tf) = &expanded_tags {
            push_tag_filter_predicates(&mut qb, "tags", tf);
        }
        qb.push(" ORDER BY vec_sim DESC, updated_at DESC LIMIT ");
        qb.push_bind((top_k * 4) as i64);

        let rows = qb
            .build()
            .fetch_all(&mut **tx.tx_mut())
            .await
            .map_err(|e| OpenRagError::Database(format!("wiki search failed: {e}")))?;

        let lower_terms: Vec<String> = expanded_terms.iter().map(|s| s.to_lowercase()).collect();
        let mut hits = Vec::with_capacity(rows.len());
        for row in &rows {
            let page = row_to_wiki_page(row)?;
            let vec_sim: f32 = row.try_get("vec_sim").unwrap_or(0.0);
            let lower_body = page.bm25_text.to_lowercase();
            let lex_hits = lower_terms
                .iter()
                .filter(|t| !t.is_empty() && lower_body.contains(t.as_str()))
                .count();
            let lex_boost = if lower_terms.is_empty() {
                0.0
            } else {
                (lex_hits as f32 / lower_terms.len() as f32) * 0.5
            };
            hits.push(WikiSearchHit {
                page,
                score: vec_sim + lex_boost,
            });
        }
        hits.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        hits.truncate(top_k);
        Ok(hits)
    }

    pub async fn link_edge_rls(
        tx: &mut RlsTransaction<'_>,
        org_id: Uuid,
        src_page_id: Uuid,
        dst_page_id: Uuid,
        relation_type: WikiRelationType,
        confidence: f32,
    ) -> Result<WikiEdge, OpenRagError> {
        let erow = sqlx::query(
            "INSERT INTO wiki.edges (org_id, src_page_id, dst_page_id, relation_type, confidence)
             VALUES ($1, $2, $3, $4, $5)
             RETURNING id, org_id, src_page_id, dst_page_id, relation_type, confidence, resolved_at, created_at",
        )
        .bind(org_id)
        .bind(src_page_id)
        .bind(dst_page_id)
        .bind(relation_type.as_str())
        .bind(confidence)
        .fetch_one(&mut **tx.tx_mut())
        .await
        .map_err(|e| map_pg_err(e, "insert wiki.edges"))?;
        row_to_wiki_edge(&erow)
    }

    pub async fn list_edges_rls(
        tx: &mut RlsTransaction<'_>,
        org_id: Uuid,
        page_id: Uuid,
    ) -> Result<Vec<WikiEdge>, OpenRagError> {
        let rows = sqlx::query(
            "SELECT id, org_id, src_page_id, dst_page_id, relation_type, confidence, resolved_at, created_at
             FROM wiki.edges
             WHERE org_id = $1 AND (src_page_id = $2 OR dst_page_id = $2)
             ORDER BY created_at ASC",
        )
        .bind(org_id)
        .bind(page_id)
        .fetch_all(&mut **tx.tx_mut())
        .await
        .map_err(|e| OpenRagError::Database(format!("failed to list wiki.edges: {e}")))?;
        rows.iter().map(row_to_wiki_edge).collect()
    }

    pub async fn list_conflicts_rls(
        tx: &mut RlsTransaction<'_>,
        org_id: Uuid,
    ) -> Result<Vec<WikiEdge>, OpenRagError> {
        let rows = sqlx::query(
            "SELECT id, org_id, src_page_id, dst_page_id, relation_type, confidence, resolved_at, created_at
             FROM wiki.edges
             WHERE org_id = $1 AND relation_type = 'contradicts' AND resolved_at IS NULL
             ORDER BY created_at ASC",
        )
        .bind(org_id)
        .fetch_all(&mut **tx.tx_mut())
        .await
        .map_err(|e| OpenRagError::Database(format!("failed to list wiki conflicts: {e}")))?;
        rows.iter().map(row_to_wiki_edge).collect()
    }

    pub async fn resolve_conflict_rls(
        tx: &mut RlsTransaction<'_>,
        org_id: Uuid,
        edge_id: Uuid,
    ) -> Result<WikiEdge, OpenRagError> {
        let row = sqlx::query(
            "UPDATE wiki.edges
             SET resolved_at = now()
             WHERE org_id = $1 AND id = $2
             RETURNING id, org_id, src_page_id, dst_page_id, relation_type, confidence, resolved_at, created_at",
        )
        .bind(org_id)
        .bind(edge_id)
        .fetch_optional(&mut **tx.tx_mut())
        .await
        .map_err(|e| map_pg_err(e, "resolve wiki conflict"))?
        .ok_or_else(|| OpenRagError::NotFound(format!("wiki edge {edge_id} not found")))?;
        row_to_wiki_edge(&row)
    }
}

pub struct MarkdownSync;

impl MarkdownSync {
    pub fn validate_safe_relative_path(raw: &str) -> Result<(), OpenRagError> {
        let path = Path::new(raw);
        if path.is_absolute() {
            return Err(OpenRagError::Validation(format!(
                "path traversal rejected: absolute path '{raw}' is not allowed"
            )));
        }
        for comp in path.components() {
            match comp {
                Component::ParentDir | Component::RootDir | Component::Prefix(_) => {
                    return Err(OpenRagError::Validation(format!(
                        "path traversal rejected in '{raw}'"
                    )));
                }
                Component::Normal(_) | Component::CurDir => {}
            }
        }
        Ok(())
    }

    fn collect_safe_markdown_files(
        canon_root: &Path,
        dir: &Path,
        out: &mut Vec<PathBuf>,
    ) -> Result<(), OpenRagError> {
        let entries = fs::read_dir(dir).map_err(|e| {
            OpenRagError::Validation(format!("failed to read directory '{}': {e}", dir.display()))
        })?;
        for entry in entries {
            let entry = entry.map_err(|e| {
                OpenRagError::Validation(format!("failed to read directory entry: {e}"))
            })?;
            let path = entry.path();
            let meta = fs::symlink_metadata(&path).map_err(|e| {
                OpenRagError::Validation(format!("failed to stat '{}': {e}", path.display()))
            })?;
            if meta.file_type().is_symlink() {
                return Err(OpenRagError::Validation(format!(
                    "symlink traversal rejected: '{}'",
                    path.display()
                )));
            }
            let canon = path.canonicalize().map_err(|e| {
                OpenRagError::Validation(format!("failed to canonicalize '{}': {e}", path.display()))
            })?;
            if !canon.starts_with(canon_root) {
                return Err(OpenRagError::Validation(format!(
                    "path traversal outside vault root rejected: '{}'",
                    path.display()
                )));
            }
            if meta.is_dir() {
                Self::collect_safe_markdown_files(canon_root, &canon, out)?;
            } else if canon.extension().and_then(|s| s.to_str()) == Some("md") {
                out.push(canon);
            }
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub async fn import_markdown_dir_rls(
        tx: &mut RlsTransaction<'_>,
        ctx: &PrincipalContext,
        vault_root: &Path,
        bucket_id: Option<Uuid>,
        dept_id: Uuid,
        org_slug: &str,
        dept_path: &str,
        default_scope: AccessScope,
        default_level: AuthLevel,
        embedder: &dyn TextEmbedder,
    ) -> Result<WikiImportReport, OpenRagError> {
        let canon_root = vault_root.canonicalize().map_err(|e| {
            OpenRagError::Validation(format!(
                "invalid vault directory '{}': {e}",
                vault_root.display()
            ))
        })?;
        let mut files = Vec::new();
        Self::collect_safe_markdown_files(&canon_root, &canon_root, &mut files)?;
        files.sort();

        let mut report = WikiImportReport::default();
        let mut link_requests: Vec<(Uuid, String, WikiRelationType)> = Vec::new();

        for file in files {
            let rel_path = file
                .strip_prefix(&canon_root)
                .map_err(|e| OpenRagError::Validation(format!("strip_prefix failed: {e}")))?
                .to_string_lossy()
                .to_string();
            let raw = fs::read_to_string(&file).map_err(|e| {
                OpenRagError::Validation(format!("failed to read '{}': {e}", file.display()))
            })?;
            let fm = parse_frontmatter(&raw)?;
            let stem = file
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("page");
            let slug = fm
                .slug
                .as_deref()
                .map(slugify)
                .filter(|s| !s.is_empty())
                .unwrap_or_else(|| slugify(stem));
            let title = fm.title.clone().unwrap_or_else(|| stem.to_string());
            let scope = fm.scope.unwrap_or(default_scope);
            let level = fm
                .min_auth_level
                .map(AuthLevel::new)
                .transpose()?
                .unwrap_or(default_level);
            let body_sha = sha256_hex(fm.body_md.trim_end().as_bytes());

            let existed =
                WikiCompiler::get_page_by_slug_rls(tx, ctx.org_id, bucket_id, &slug).await?;
            let (page, changed) = WikiCompiler::upsert_page_rls(
                tx,
                ctx,
                bucket_id,
                dept_id,
                org_slug,
                dept_path,
                scope,
                level,
                &slug,
                &title,
                &fm.body_md,
                &fm.tags,
                WikiEditorType::Sync,
                Some("markdown import"),
                embedder,
            )
            .await?;

            // Link aliases from YAML frontmatter into ConceptGraph
            for alias in &fm.aliases {
                if !alias.trim().is_empty() {
                    let _ = ConceptGraph::link_terms_rls(
                        tx,
                        ctx.org_id,
                        Some(ctx.principal_id),
                        &title,
                        alias,
                        ConceptCategory::Domain,
                        ConceptSource::WikiFrontmatter,
                    )
                    .await?;
                }
            }

            if let Some(sup_slug) = fm.supersedes {
                link_requests.push((page.id, slugify(&sup_slug), WikiRelationType::Supersedes));
            }
            for target_slug in extract_wikilinks(&fm.body_md) {
                link_requests.push((page.id, target_slug, WikiRelationType::References));
            }

            sqlx::query(
                "INSERT INTO wiki.sync_changes
                 (org_id, bucket_id, page_id, file_path, direction, body_sha256, status)
                 VALUES ($1, $2, $3, $4, 'import', $5, 'synced')",
            )
            .bind(ctx.org_id)
            .bind(bucket_id)
            .bind(page.id)
            .bind(&rel_path)
            .bind(&body_sha)
            .execute(&mut **tx.tx_mut())
            .await
            .map_err(|e| map_pg_err(e, "insert wiki.sync_changes import"))?;

            let outcome = match (existed.is_some(), changed) {
                (false, true) => {
                    report.imported += 1;
                    SyncDocOutcome::Created
                }
                (true, true) => {
                    report.updated += 1;
                    SyncDocOutcome::Updated
                }
                (_, false) => {
                    report.skipped_noop += 1;
                    SyncDocOutcome::SkippedNoop
                }
            };
            report.results.push(SyncDocResult {
                slug: page.slug.clone(),
                page_id: Some(page.id),
                outcome,
                body_sha256: body_sha,
                version: Some(page.version),
            });
        }

        for (src_id, dst_slug, rel_type) in link_requests {
            if let Some(dst) =
                WikiCompiler::get_page_by_slug_rls(tx, ctx.org_id, bucket_id, &dst_slug).await?
            {
                if dst.id != src_id {
                    let _ = WikiCompiler::link_edge_rls(
                        tx,
                        ctx.org_id,
                        src_id,
                        dst.id,
                        rel_type,
                        1.0,
                    )
                    .await?;
                    report.edges_created += 1;
                }
            }
        }

        Ok(report)
    }

    pub async fn export_markdown_dir_rls(
        tx: &mut RlsTransaction<'_>,
        org_id: Uuid,
        bucket_id: Option<Uuid>,
        out_dir: &Path,
    ) -> Result<Vec<PathBuf>, OpenRagError> {
        fs::create_dir_all(out_dir).map_err(|e| {
            OpenRagError::Validation(format!("failed to create export dir '{}': {e}", out_dir.display()))
        })?;
        let canon_out = out_dir.canonicalize().map_err(|e| {
            OpenRagError::Validation(format!("failed to canonicalize export dir: {e}"))
        })?;

        let rows = sqlx::query(
            "SELECT id, org_id, bucket_id, dept_id, owner_principal_id, access_scope,
                    allowed_group_ids, denied_group_ids, min_auth_level, arn, acl_scope_key,
                    slug, title, body_md, bm25_text, body_sha256, tags, status, version,
                    created_at, updated_at
             FROM wiki.pages
             WHERE org_id = $1 AND status = 'active' AND ($2::uuid IS NULL OR bucket_id = $2)
             ORDER BY slug ASC",
        )
        .bind(org_id)
        .bind(bucket_id)
        .fetch_all(&mut **tx.tx_mut())
        .await
        .map_err(|e| OpenRagError::Database(format!("failed to list wiki pages for export: {e}")))?;

        let mut written = Vec::with_capacity(rows.len());
        for row in rows {
            let page = row_to_wiki_page(&row)?;
            let safe_slug = slugify(&page.slug);
            if safe_slug.is_empty() {
                continue;
            }
            let filename = format!("{safe_slug}.md");
            Self::validate_safe_relative_path(&filename)?;
            let dest = canon_out.join(&filename);
            let md = render_frontmatter_markdown(&page)?;
            fs::write(&dest, md.as_bytes()).map_err(|e| {
                OpenRagError::Internal(format!("failed to write '{}': {e}", dest.display()))
            })?;

            sqlx::query(
                "INSERT INTO wiki.sync_changes
                 (org_id, bucket_id, page_id, file_path, direction, body_sha256, status)
                 VALUES ($1, $2, $3, $4, 'export', $5, 'synced')",
            )
            .bind(org_id)
            .bind(page.bucket_id)
            .bind(page.id)
            .bind(&filename)
            .bind(&page.body_sha256)
            .execute(&mut **tx.tx_mut())
            .await
            .map_err(|e| map_pg_err(e, "insert wiki.sync_changes export"))?;

            written.push(dest);
        }
        Ok(written)
    }

    #[allow(clippy::too_many_arguments)]
    pub async fn sync_documents_rls(
        tx: &mut RlsTransaction<'_>,
        ctx: &PrincipalContext,
        bucket_id: Option<Uuid>,
        dept_id: Uuid,
        org_slug: &str,
        dept_path: &str,
        access_scope: AccessScope,
        min_auth_level: AuthLevel,
        docs: &[SyncDocInput],
        embedder: &dyn TextEmbedder,
    ) -> Result<WikiImportReport, OpenRagError> {
        let mut report = WikiImportReport::default();

        for doc in docs {
            Self::validate_safe_relative_path(&doc.file_path)?;
            let slug = slugify(&doc.slug);
            let norm_body = doc.body_md.trim_end();
            let incoming_sha = sha256_hex(norm_body.as_bytes());
            let existing =
                WikiCompiler::get_page_by_slug_rls(tx, ctx.org_id, bucket_id, &slug).await?;

            if let Some(cur) = existing {
                if cur.body_sha256 == incoming_sha {
                    report.skipped_noop += 1;
                    report.results.push(SyncDocResult {
                        slug,
                        page_id: Some(cur.id),
                        outcome: SyncDocOutcome::SkippedNoop,
                        body_sha256: incoming_sha,
                        version: Some(cur.version),
                    });
                    continue;
                }

                let sha_conflict = doc
                    .base_sha256
                    .as_deref()
                    .is_some_and(|b| b != cur.body_sha256);
                let ver_conflict = doc
                    .base_version
                    .is_some_and(|bv| bv != cur.version);
                if sha_conflict || ver_conflict {
                    sqlx::query(
                        "INSERT INTO wiki.sync_changes
                         (org_id, bucket_id, page_id, file_path, direction, body_sha256, status)
                         VALUES ($1, $2, $3, $4, 'import', $5, 'conflict')",
                    )
                    .bind(ctx.org_id)
                    .bind(bucket_id)
                    .bind(cur.id)
                    .bind(&doc.file_path)
                    .bind(&incoming_sha)
                    .execute(&mut **tx.tx_mut())
                    .await
                    .map_err(|e| map_pg_err(e, "insert wiki.sync_changes conflict"))?;

                    report.conflicts += 1;
                    report.results.push(SyncDocResult {
                        slug,
                        page_id: Some(cur.id),
                        outcome: SyncDocOutcome::Conflict,
                        body_sha256: incoming_sha,
                        version: Some(cur.version),
                    });
                    continue;
                }
            }

            let existed =
                WikiCompiler::get_page_by_slug_rls(tx, ctx.org_id, bucket_id, &slug).await?.is_some();
            let (page, changed) = WikiCompiler::upsert_page_rls(
                tx,
                ctx,
                bucket_id,
                dept_id,
                org_slug,
                dept_path,
                access_scope,
                min_auth_level,
                &slug,
                &doc.title,
                &doc.body_md,
                &doc.tags,
                WikiEditorType::Sync,
                Some("sync_documents"),
                embedder,
            )
            .await?;

            sqlx::query(
                "INSERT INTO wiki.sync_changes
                 (org_id, bucket_id, page_id, file_path, direction, body_sha256, status)
                 VALUES ($1, $2, $3, $4, 'import', $5, 'synced')",
            )
            .bind(ctx.org_id)
            .bind(bucket_id)
            .bind(page.id)
            .bind(&doc.file_path)
            .bind(&incoming_sha)
            .execute(&mut **tx.tx_mut())
            .await
            .map_err(|e| map_pg_err(e, "insert wiki.sync_changes synced"))?;

            let outcome = if !existed && changed {
                report.imported += 1;
                SyncDocOutcome::Created
            } else if changed {
                report.updated += 1;
                SyncDocOutcome::Updated
            } else {
                report.skipped_noop += 1;
                SyncDocOutcome::SkippedNoop
            };
            report.results.push(SyncDocResult {
                slug: page.slug,
                page_id: Some(page.id),
                outcome,
                body_sha256: incoming_sha,
                version: Some(page.version),
            });
        }

        Ok(report)
    }
}

fn map_pg_err(e: sqlx::Error, op: &str) -> OpenRagError {
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

fn row_to_wiki_page(row: &sqlx::postgres::PgRow) -> Result<WikiPage, OpenRagError> {
    let scope_str: String = row
        .try_get("access_scope")
        .map_err(|e| OpenRagError::Database(e.to_string()))?;
    let level_i16: i16 = row
        .try_get("min_auth_level")
        .map_err(|e| OpenRagError::Database(e.to_string()))?;
    let status_str: String = row
        .try_get("status")
        .map_err(|e| OpenRagError::Database(e.to_string()))?;
    Ok(WikiPage {
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
        slug: row.try_get("slug").map_err(|e| OpenRagError::Database(e.to_string()))?,
        title: row.try_get("title").map_err(|e| OpenRagError::Database(e.to_string()))?,
        body_md: row
            .try_get("body_md")
            .map_err(|e| OpenRagError::Database(e.to_string()))?,
        bm25_text: row
            .try_get("bm25_text")
            .map_err(|e| OpenRagError::Database(e.to_string()))?,
        body_sha256: row
            .try_get("body_sha256")
            .map_err(|e| OpenRagError::Database(e.to_string()))?,
        tags: row.try_get("tags").map_err(|e| OpenRagError::Database(e.to_string()))?,
        status: status_str.parse()?,
        version: row
            .try_get("version")
            .map_err(|e| OpenRagError::Database(e.to_string()))?,
        created_at: row
            .try_get("created_at")
            .map_err(|e| OpenRagError::Database(e.to_string()))?,
        updated_at: row
            .try_get("updated_at")
            .map_err(|e| OpenRagError::Database(e.to_string()))?,
    })
}

fn row_to_wiki_edge(row: &sqlx::postgres::PgRow) -> Result<WikiEdge, OpenRagError> {
    let rel_str: String = row
        .try_get("relation_type")
        .map_err(|e| OpenRagError::Database(e.to_string()))?;
    Ok(WikiEdge {
        id: row.try_get("id").map_err(|e| OpenRagError::Database(e.to_string()))?,
        org_id: row.try_get("org_id").map_err(|e| OpenRagError::Database(e.to_string()))?,
        src_page_id: row
            .try_get("src_page_id")
            .map_err(|e| OpenRagError::Database(e.to_string()))?,
        dst_page_id: row
            .try_get("dst_page_id")
            .map_err(|e| OpenRagError::Database(e.to_string()))?,
        relation_type: rel_str.parse()?,
        confidence: row
            .try_get("confidence")
            .map_err(|e| OpenRagError::Database(e.to_string()))?,
        resolved_at: row
            .try_get("resolved_at")
            .map_err(|e| OpenRagError::Database(e.to_string()))?,
        created_at: row
            .try_get("created_at")
            .map_err(|e| OpenRagError::Database(e.to_string()))?,
    })
}
