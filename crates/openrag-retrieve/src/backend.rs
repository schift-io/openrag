use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::sync::Arc;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use openrag_core::{AccessScope, BBoxNorm1000, OpenRagError, PgPineconeSearchConfig, PrincipalContext};
use openrag_db::{DbPool, RlsTransaction};
use openrag_memory::{ConceptGraph, DeterministicTextEmbedder, InMemoryConceptStore, TextEmbedder};
use openrag_store::{
    cosine_similarity, expand_tag_filter_in_memory, expand_tag_filter_rls, normalize_tag,
    push_tag_filter_predicates, tokenize_bm25, Bm25DocInput, Bm25Hit, Bm25Index, Bm25Query,
    Bm25Store, ExpandedTagFilter, IndexBinding, PgPineconeManager, TagTable, VectorHit,
    VectorSearchQuery, VectorTable,
};
use pgvector::Vector;
use sqlx::{Postgres, QueryBuilder, Row};
use tokio::sync::RwLock;
use uuid::Uuid;

use crate::query::{
    build_concept_lookup_terms, build_embed_query_text, query_has_searchable_tokens,
};
use crate::types::{
    BackendExecutionInput, BackendExecutionOutput, HydratedCandidate, RetrieveSpace, TagOnlyHit,
};

#[async_trait]
pub trait RetrieveBackend: Send + Sync {
    async fn execute_plan(
        &self,
        ctx: &PrincipalContext,
        input: &BackendExecutionInput,
    ) -> Result<BackendExecutionOutput, OpenRagError>;

    async fn compute_bucket_facets(
        &self,
        ctx: &PrincipalContext,
        spaces: &[RetrieveSpace],
        bucket_id: Option<Uuid>,
        limit: i64,
    ) -> Result<BTreeMap<String, i64>, OpenRagError>;
}

pub struct PgRetrieveBackend {
    app_pool: DbPool,
    manager: PgPineconeManager,
    embedder: Arc<dyn TextEmbedder>,
    bindings: HashMap<VectorTable, IndexBinding>,
    pinecone_search: Option<PgPineconeSearchConfig>,
}

impl PgRetrieveBackend {
    pub fn new(app_pool: DbPool) -> Self {
        Self {
            app_pool,
            manager: PgPineconeManager::new(),
            embedder: Arc::new(DeterministicTextEmbedder::default()),
            bindings: HashMap::new(),
            pinecone_search: None,
        }
    }

    pub fn with_manager(mut self, manager: PgPineconeManager) -> Self {
        self.manager = manager;
        self
    }

    pub fn with_embedder(mut self, embedder: Arc<dyn TextEmbedder>) -> Self {
        self.embedder = embedder;
        self
    }

    pub fn with_bindings(mut self, bindings: HashMap<VectorTable, IndexBinding>) -> Self {
        self.bindings = bindings;
        self
    }

    /// pg_pinecone search settings for every search transaction (native engine only).
    pub fn with_pg_pinecone_search(mut self, cfg: PgPineconeSearchConfig) -> Self {
        self.pinecone_search = Some(cfg);
        self
    }

    pub async fn load_bindings_from_worker(
        mut self,
        worker_pool: &DbPool,
        org_id: Uuid,
    ) -> Result<Self, OpenRagError> {
        let list = PgPineconeManager::list_bindings(worker_pool, org_id).await?;
        let mut map = HashMap::new();
        for b in list {
            if let Ok(table) = b.table_name.parse::<VectorTable>() {
                map.insert(table, b);
            }
        }
        self.bindings = map;
        Ok(self)
    }
}

#[async_trait]
impl RetrieveBackend for PgRetrieveBackend {
    async fn execute_plan(
        &self,
        ctx: &PrincipalContext,
        input: &BackendExecutionInput,
    ) -> Result<BackendExecutionOutput, OpenRagError> {
        let caps = self.app_pool.capabilities().await?;
        let mut tx = self.app_pool.begin_rls_tx(ctx).await?;
        if let Some(cfg) = self.pinecone_search.filter(|_| caps.has_pg_pinecone) {
            set_pg_pinecone_search(&mut tx, cfg).await?;
        }

        let lookup_terms = build_concept_lookup_terms(&input.clean_query);
        let expanded_terms = if lookup_terms.is_empty() {
            Vec::new()
        } else {
            ConceptGraph::expand_terms_rls(&mut tx, ctx.org_id, &lookup_terms).await?
        };

        let expanded_tag_filter = if input.effective_tag_filter.is_empty() {
            None
        } else {
            Some(expand_tag_filter_rls(&mut tx, ctx.org_id, &input.effective_tag_filter).await?)
        };

        let query_vector = match &input.explicit_query_vector {
            Some(v) => v.clone(),
            None => {
                let embed_text = build_embed_query_text(
                    &input.clean_query,
                    &expanded_terms,
                    &input.effective_tag_filter,
                );
                self.embedder.embed_query(&embed_text).await?
            }
        };

        let mut vector_hits = Vec::new();
        let mut bm25_hits = Vec::new();
        let mut tag_only_hits = Vec::new();
        let has_searchable = query_has_searchable_tokens(&input.clean_query);

        for &space in &input.spaces {
            let table = space.vector_table();
            let binding = self.bindings.get(&table);

            let is_thumb = space == RetrieveSpace::Thumb;
            let is_corpus_space = matches!(space, RetrieveSpace::Chunks | RetrieveSpace::Thumb);
            let space_limit = if is_thumb && expanded_tag_filter.is_some() {
                input
                    .candidate_limit_per_space
                    .saturating_mul(2)
                    .clamp(80, 500)
            } else {
                input.candidate_limit_per_space
            };
            let space_tag_filter = if is_thumb || input.effective_tag_filter.is_empty() {
                None
            } else {
                Some(input.effective_tag_filter.clone())
            };

            let mut space_v_hits = Vec::new();
            let mut space_b_hits = Vec::new();
            let mut seen_vec_ids = HashSet::new();
            let mut seen_bm25_ids = HashSet::new();

            let v_query = VectorSearchQuery {
                org_id: ctx.org_id,
                table,
                query_vector: query_vector.clone(),
                acl_scope_key: None,
                bucket_id: input.bucket_id,
                top_k: space_limit,
                tag_filter: space_tag_filter.clone(),
            };
            let v_hits = if table.has_vectors() {
                self.manager.search_rls(&mut tx, binding, &v_query).await?
            } else {
                Vec::new()
            };
            for h in v_hits {
                if seen_vec_ids.insert(h.id) {
                    space_v_hits.push(h);
                }
            }

            if has_searchable {
                // Convex alpha 0: BM25 cannot change the score of a space with vectors, so it is
                // not queried there. A space without vectors (Thumb) is scored by BM25 alone.
                if !(input.skip_lexical && table.has_vectors()) {
                    let b_query = Bm25Query {
                        org_id: ctx.org_id,
                        table,
                        query_text: input.clean_query.clone(),
                        expanded_terms: expanded_terms.clone(),
                        bucket_id: input.bucket_id,
                        tag_filter: space_tag_filter,
                        limit: space_limit,
                    };
                    let b_hits = Bm25Store::search_rls(&mut tx, &caps, &b_query).await?;
                    for h in b_hits {
                        if seen_bm25_ids.insert(h.id) {
                            space_b_hits.push(h);
                        }
                    }
                }
            } else if input.tag_only_query {
                if let Some(etf) = &expanded_tag_filter {
                    let t_hits = query_tag_only_space_rls(
                        &mut tx,
                        ctx.org_id,
                        space,
                        etf,
                        input.bucket_id,
                        space_limit,
                    )
                    .await?;
                    tag_only_hits.extend(t_hits);
                }
            }

            // BUG-4 & BUG-6: For `Chunks` and `Thumb`, if `Thumb` has an active tag filter,
            // or if the initial candidate pool is saturated or contains archived documents,
            // execute an RLS-guarded SQL query joining `corpus.documents` with `d.status <> 'archived'`
            // (and `d.tags` predicates for `Thumb`) so active documents are never starved.
            if is_corpus_space {
                let need_tagged_thumb = is_thumb && expanded_tag_filter.is_some();
                // A full candidate pool is not starvation by itself: active documents can only
                // be pushed out when archived ones occupy slots. Re-querying on every full pool
                // ran the BM25 scan twice at ordinary top_k and, with pg_search, replaced its
                // ranked hits with the unranked fallback candidates.
                let has_archived = if need_tagged_thumb {
                    false
                } else {
                    space_hits_contain_archived(&mut tx, ctx.org_id, &space_v_hits, &space_b_hits)
                        .await?
                };

                if need_tagged_thumb || has_archived {
                    let active_v = query_corpus_active_vectors_rls(
                        &mut tx,
                        ctx.org_id,
                        space,
                        &query_vector,
                        input.bucket_id,
                        expanded_tag_filter.as_ref(),
                        space_limit,
                    )
                    .await?;
                    for h in active_v {
                        if seen_vec_ids.insert(h.id) {
                            space_v_hits.push(h);
                        }
                    }

                    if has_searchable && !(input.skip_lexical && table.has_vectors()) {
                        let bm25_query = ActiveCorpusBm25Query {
                            org_id: ctx.org_id,
                            space,
                            query_text: &input.clean_query,
                            expanded_terms: &expanded_terms,
                            bucket_id: input.bucket_id,
                            expanded_tag_filter: expanded_tag_filter.as_ref(),
                            limit: space_limit,
                        };
                        let active_b = query_corpus_active_bm25_rls(&mut tx, &bm25_query).await?;
                        if !caps.has_pg_search || !active_b.is_empty() {
                            space_b_hits = active_b;
                        }
                    }
                }
            }

            for h in space_v_hits {
                vector_hits.push((space, h));
            }
            for h in space_b_hits {
                bm25_hits.push((space, h));
            }
        }

        let hydrated = hydrate_pg_candidates(
            &mut tx,
            ctx.org_id,
            &vector_hits,
            &bm25_hits,
            &tag_only_hits,
            input.include_embeddings,
        )
        .await?;

        tx.rollback().await?;

        Ok(BackendExecutionOutput {
            vector_hits,
            bm25_hits,
            tag_only_hits,
            expanded_terms,
            expanded_tag_filter,
            query_vector,
            hydrated,
        })
    }

    async fn compute_bucket_facets(
        &self,
        ctx: &PrincipalContext,
        spaces: &[RetrieveSpace],
        bucket_id: Option<Uuid>,
        limit: i64,
    ) -> Result<BTreeMap<String, i64>, OpenRagError> {
        let mut tables = Vec::new();
        if spaces.contains(&RetrieveSpace::Chunks) || spaces.contains(&RetrieveSpace::Thumb) {
            tables.push(TagTable::CorpusDocuments);
        }
        if spaces.contains(&RetrieveSpace::Wiki) {
            tables.push(TagTable::WikiPages);
        }
        if spaces.contains(&RetrieveSpace::Memory) {
            tables.push(TagTable::MemoryEntries);
        }
        if tables.is_empty() || limit <= 0 {
            return Ok(BTreeMap::new());
        }
        let mut tx = self.app_pool.begin_rls_tx(ctx).await?;
        let facets =
            compute_active_bucket_facets_rls(&mut tx, ctx.org_id, &tables, bucket_id, limit)
                .await?;
        tx.rollback().await?;
        Ok(facets)
    }
}

async fn compute_active_bucket_facets_rls(
    tx: &mut RlsTransaction<'_>,
    org_id: Uuid,
    tables: &[TagTable],
    bucket_id: Option<Uuid>,
    limit: i64,
) -> Result<BTreeMap<String, i64>, OpenRagError> {
    if tables.is_empty() || limit <= 0 {
        return Ok(BTreeMap::new());
    }

    let mut qb: QueryBuilder<'_, Postgres> =
        QueryBuilder::new("SELECT tag, count(*)::bigint AS cnt FROM (");
    for (idx, table) in tables.iter().enumerate() {
        if idx > 0 {
            qb.push(" UNION ALL ");
        }
        match table {
            TagTable::CorpusDocuments => {
                qb.push(
                    "SELECT unnest(d.tags) AS tag FROM corpus.documents AS d WHERE d.org_id = ",
                );
                qb.push_bind(org_id);
                qb.push(" AND d.status <> 'archived'");
                if let Some(bid) = bucket_id {
                    qb.push(" AND d.bucket_id = ");
                    qb.push_bind(bid);
                }
            }
            TagTable::CorpusChunks => {
                qb.push(
                    "SELECT unnest(c.tags) AS tag FROM corpus.chunks AS c
                     JOIN corpus.documents AS cd ON cd.id = c.document_id AND cd.org_id = c.org_id
                     WHERE c.org_id = ",
                );
                qb.push_bind(org_id);
                qb.push(" AND cd.status <> 'archived'");
                if let Some(bid) = bucket_id {
                    qb.push(" AND cd.bucket_id = ");
                    qb.push_bind(bid);
                }
            }
            TagTable::WikiPages => {
                qb.push("SELECT unnest(w.tags) AS tag FROM wiki.pages AS w WHERE w.org_id = ");
                qb.push_bind(org_id);
                qb.push(" AND w.status = 'active'");
                if let Some(bid) = bucket_id {
                    qb.push(" AND w.bucket_id = ");
                    qb.push_bind(bid);
                }
            }
            TagTable::MemoryEntries => {
                qb.push("SELECT unnest(m.tags) AS tag FROM memory.entries AS m WHERE m.org_id = ");
                qb.push_bind(org_id);
                qb.push(" AND m.status = 'active'");
                if let Some(bid) = bucket_id {
                    qb.push(" AND m.bucket_id = ");
                    qb.push_bind(bid);
                }
            }
        }
    }
    qb.push(") AS all_tags GROUP BY tag ORDER BY cnt DESC, tag ASC LIMIT ");
    qb.push_bind(limit);

    let rows = qb
        .build()
        .fetch_all(&mut **tx.tx_mut())
        .await
        .map_err(|e| OpenRagError::Database(format!("failed to compute bucket tag facets: {e}")))?;

    let mut out = BTreeMap::new();
    for row in rows {
        let tag: String = row
            .try_get("tag")
            .map_err(|e| OpenRagError::Database(format!("invalid facet tag row: {e}")))?;
        let count: i64 = row
            .try_get("cnt")
            .map_err(|e| OpenRagError::Database(format!("invalid facet cnt row: {e}")))?;
        out.insert(tag, count);
    }
    Ok(out)
}

/// `SET LOCAL` for pg_pinecone's probe and rescore settings; 0 keeps pg_pinecone's own value.
async fn set_pg_pinecone_search(
    tx: &mut RlsTransaction<'_>,
    cfg: PgPineconeSearchConfig,
) -> Result<(), OpenRagError> {
    let scan = (cfg.scan_fraction > 0.0).then(|| cfg.scan_fraction.to_string());
    let rescore = (cfg.rescore_multiplier > 0).then(|| cfg.rescore_multiplier.to_string());
    if scan.is_none() && rescore.is_none() {
        return Ok(());
    }
    sqlx::query(
        "SELECT CASE WHEN $1::text IS NULL THEN NULL ELSE set_config('pinecone.scan_fraction', $1, true) END,
                CASE WHEN $2::text IS NULL THEN NULL ELSE set_config('pinecone.rescore_multiplier', $2, true) END",
    )
    .bind(scan)
    .bind(rescore)
    .execute(&mut **tx.tx_mut())
    .await
    .map_err(|e| OpenRagError::Database(format!("set pg_pinecone search settings: {e}")))?;
    Ok(())
}

async fn space_hits_contain_archived(
    tx: &mut RlsTransaction<'_>,
    org_id: Uuid,
    v_hits: &[VectorHit],
    b_hits: &[Bm25Hit],
) -> Result<bool, OpenRagError> {
    let mut doc_ids = BTreeSet::new();
    for h in v_hits {
        if let Some(did) = h.document_id {
            doc_ids.insert(did);
        }
    }
    for h in b_hits {
        if let Some(did) = h.document_id {
            doc_ids.insert(did);
        }
    }
    if doc_ids.is_empty() {
        return Ok(false);
    }
    let ids: Vec<Uuid> = doc_ids.into_iter().collect();
    sqlx::query_scalar::<_, bool>(
        "SELECT EXISTS(
            SELECT 1 FROM corpus.documents
            WHERE org_id = $1 AND id = ANY($2::uuid[]) AND status = 'archived'
        )",
    )
    .bind(org_id)
    .bind(&ids)
    .fetch_one(&mut **tx.tx_mut())
    .await
    .map_err(|e| OpenRagError::Database(format!("failed to check archived status: {e}")))
}

async fn query_corpus_active_vectors_rls(
    tx: &mut RlsTransaction<'_>,
    org_id: Uuid,
    space: RetrieveSpace,
    query_vector: &[f32],
    bucket_id: Option<Uuid>,
    expanded_tag_filter: Option<&ExpandedTagFilter>,
    limit: usize,
) -> Result<Vec<VectorHit>, OpenRagError> {
    if limit == 0 {
        return Ok(Vec::new());
    }
    let table = space.vector_table();
    let Some(vec_col) = table.vector_column() else {
        return Ok(Vec::new());
    };
    let qvec = Vector::from(query_vector.to_vec());

    let mut qb: QueryBuilder<'_, Postgres> =
        QueryBuilder::new("SELECT t.id, t.document_id, t.acl_scope_key, ");
    if space == RetrieveSpace::Thumb {
        qb.push("d.tags");
    } else {
        qb.push("t.tags");
    }
    qb.push(", (1.0 - (t.");
    qb.push(vec_col);
    qb.push(" <=> ");
    qb.push_bind(qvec.clone());
    qb.push("))::real AS cosine_sim, (t.");
    qb.push(vec_col);
    qb.push(" <=> ");
    qb.push_bind(qvec.clone());
    qb.push(")::real AS distance FROM ");
    qb.push(table.as_str());
    qb.push(" AS t JOIN corpus.documents AS d ON d.id = t.document_id AND d.org_id = t.org_id WHERE t.org_id = ");
    qb.push_bind(org_id);
    qb.push(" AND d.status <> 'archived' AND t.");
    qb.push(vec_col);
    qb.push(" IS NOT NULL");

    if let Some(bid) = bucket_id {
        qb.push(" AND d.bucket_id = ");
        qb.push_bind(bid);
    }
    if let Some(etf) = expanded_tag_filter {
        let tags_col = if space == RetrieveSpace::Thumb {
            "d.tags"
        } else {
            "t.tags"
        };
        push_tag_filter_predicates(&mut qb, tags_col, etf);
    }
    qb.push(" ORDER BY t.");
    qb.push(vec_col);
    qb.push(" <=> ");
    qb.push_bind(qvec);
    qb.push(" ASC, t.id ASC LIMIT ");
    qb.push_bind(limit as i64);

    let rows = qb
        .build()
        .fetch_all(&mut **tx.tx_mut())
        .await
        .map_err(|e| OpenRagError::Database(format!("active corpus vector query failed: {e}")))?;

    let mut hits = Vec::with_capacity(rows.len());
    for row in rows {
        let id: Uuid = row
            .try_get("id")
            .map_err(|e| OpenRagError::Database(format!("invalid active vec id: {e}")))?;
        let document_id: Option<Uuid> = row
            .try_get("document_id")
            .map_err(|e| OpenRagError::Database(format!("invalid active vec doc_id: {e}")))?;
        let acl_scope_key: String = row
            .try_get("acl_scope_key")
            .map_err(|e| OpenRagError::Database(format!("invalid active vec acl: {e}")))?;
        let tags: Vec<String> = row
            .try_get("tags")
            .map_err(|e| OpenRagError::Database(format!("invalid active vec tags: {e}")))?;
        let cosine_sim: f32 = row
            .try_get("cosine_sim")
            .map_err(|e| OpenRagError::Database(format!("invalid active vec cos: {e}")))?;
        let distance: f32 = row
            .try_get("distance")
            .map_err(|e| OpenRagError::Database(format!("invalid active vec dist: {e}")))?;
        hits.push(VectorHit {
            id,
            document_id,
            table,
            cosine_sim,
            distance,
            acl_scope_key,
            tags,
            metadata: serde_json::Map::new(),
        });
    }
    Ok(hits)
}

struct ActiveCorpusBm25Query<'a> {
    org_id: Uuid,
    space: RetrieveSpace,
    query_text: &'a str,
    expanded_terms: &'a [String],
    bucket_id: Option<Uuid>,
    expanded_tag_filter: Option<&'a ExpandedTagFilter>,
    limit: usize,
}

async fn query_corpus_active_bm25_rls(
    tx: &mut RlsTransaction<'_>,
    q: &ActiveCorpusBm25Query<'_>,
) -> Result<Vec<Bm25Hit>, OpenRagError> {
    if q.limit == 0 {
        return Ok(Vec::new());
    }
    let table = q.space.vector_table();
    let text_col = match q.space {
        RetrieveSpace::Chunks => "t.bm25_text",
        RetrieveSpace::Thumb => "COALESCE(t.redacted_markdown, t.raw_markdown, '')",
        _ => return Ok(Vec::new()),
    };

    let mut all_tokens: Vec<String> = tokenize_bm25(q.query_text);
    for extra in q.expanded_terms {
        all_tokens.extend(tokenize_bm25(extra));
    }
    let mut patterns: Vec<String> = Vec::new();
    for tok in &all_tokens {
        let pat = format!("%{}%", escape_like(tok));
        if !patterns.contains(&pat) {
            patterns.push(pat);
        }
    }
    if patterns.is_empty() {
        return Ok(Vec::new());
    }

    let mut qb: QueryBuilder<'_, Postgres> = QueryBuilder::new("SELECT t.id, t.document_id, ");
    if q.space == RetrieveSpace::Thumb {
        qb.push("d.tags");
    } else {
        qb.push("t.tags");
    }
    qb.push(", ");
    qb.push(text_col);
    qb.push(" AS text_body FROM ");
    qb.push(table.as_str());
    qb.push(" AS t JOIN corpus.documents AS d ON d.id = t.document_id AND d.org_id = t.org_id WHERE t.org_id = ");
    qb.push_bind(q.org_id);
    qb.push(" AND d.status <> 'archived'");

    if let Some(bid) = q.bucket_id {
        qb.push(" AND d.bucket_id = ");
        qb.push_bind(bid);
    }
    if let Some(etf) = q.expanded_tag_filter {
        let tags_col = if q.space == RetrieveSpace::Thumb {
            "d.tags"
        } else {
            "t.tags"
        };
        push_tag_filter_predicates(&mut qb, tags_col, etf);
    }

    qb.push(" AND (");
    qb.push(text_col);
    qb.push(" ILIKE ANY(");
    qb.push_bind(patterns);
    qb.push("::text[]) OR ");
    qb.push(text_col);
    qb.push(" % ");
    qb.push_bind(q.query_text);
    qb.push(") ORDER BY t.id ASC LIMIT 500");

    let rows = qb
        .build()
        .fetch_all(&mut **tx.tx_mut())
        .await
        .map_err(|e| OpenRagError::Database(format!("active corpus bm25 query failed: {e}")))?;

    let mut docs = Vec::with_capacity(rows.len());
    for row in rows {
        let id: Uuid = row
            .try_get("id")
            .map_err(|e| OpenRagError::Database(format!("invalid active bm25 id: {e}")))?;
        let document_id: Option<Uuid> = row
            .try_get("document_id")
            .map_err(|e| OpenRagError::Database(format!("invalid active bm25 doc_id: {e}")))?;
        let tags: Vec<String> = row
            .try_get("tags")
            .map_err(|e| OpenRagError::Database(format!("invalid active bm25 tags: {e}")))?;
        let text_body: String = row
            .try_get("text_body")
            .map_err(|e| OpenRagError::Database(format!("invalid active bm25 text: {e}")))?;
        docs.push(Bm25DocInput {
            id,
            document_id,
            text: text_body,
            tags,
        });
    }

    let index = Bm25Index::from_docs(&docs);
    Ok(index.search_with_expansion(
        table,
        q.query_text,
        q.expanded_terms,
        q.expanded_tag_filter,
        q.limit,
    ))
}

fn escape_like(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            '\\' | '%' | '_' => {
                out.push('\\');
                out.push(ch);
            }
            _ => out.push(ch),
        }
    }
    out
}

async fn query_tag_only_space_rls(
    tx: &mut RlsTransaction<'_>,
    org_id: Uuid,
    space: RetrieveSpace,
    etf: &ExpandedTagFilter,
    bucket_id: Option<Uuid>,
    limit: usize,
) -> Result<Vec<TagOnlyHit>, OpenRagError> {
    if limit == 0 {
        return Ok(Vec::new());
    }
    let mut qb: QueryBuilder<'_, Postgres> = match space {
        RetrieveSpace::Chunks => {
            let mut q = QueryBuilder::new(
                "SELECT t.id, t.document_id, t.acl_scope_key, t.tags
                 FROM corpus.chunks AS t
                 JOIN corpus.documents AS d ON d.id = t.document_id AND d.org_id = t.org_id
                 WHERE t.org_id = ",
            );
            q.push_bind(org_id);
            q.push(" AND d.status <> 'archived'");
            if let Some(bid) = bucket_id {
                q.push(" AND d.bucket_id = ");
                q.push_bind(bid);
            }
            push_tag_filter_predicates(&mut q, "t.tags", etf);
            q
        }
        RetrieveSpace::Thumb => {
            let mut q = QueryBuilder::new(
                "SELECT t.id, t.document_id, t.acl_scope_key, d.tags
                 FROM corpus.pages AS t
                 JOIN corpus.documents AS d ON d.id = t.document_id AND d.org_id = t.org_id
                 WHERE t.org_id = ",
            );
            q.push_bind(org_id);
            q.push(" AND d.status <> 'archived'");
            if let Some(bid) = bucket_id {
                q.push(" AND d.bucket_id = ");
                q.push_bind(bid);
            }
            push_tag_filter_predicates(&mut q, "d.tags", etf);
            q
        }
        RetrieveSpace::Wiki => {
            let mut q = QueryBuilder::new(
                "SELECT t.id, NULL::uuid AS document_id, t.acl_scope_key, t.tags
                 FROM wiki.pages AS t
                 WHERE t.org_id = ",
            );
            q.push_bind(org_id);
            q.push(" AND t.status = 'active'");
            if let Some(bid) = bucket_id {
                q.push(" AND t.bucket_id = ");
                q.push_bind(bid);
            }
            push_tag_filter_predicates(&mut q, "t.tags", etf);
            q
        }
        RetrieveSpace::Memory => {
            let mut q = QueryBuilder::new(
                "SELECT t.id, NULL::uuid AS document_id, t.acl_scope_key, t.tags
                 FROM memory.entries AS t
                 WHERE t.org_id = ",
            );
            q.push_bind(org_id);
            q.push(" AND t.status = 'active'");
            if let Some(bid) = bucket_id {
                q.push(" AND t.bucket_id = ");
                q.push_bind(bid);
            }
            push_tag_filter_predicates(&mut q, "t.tags", etf);
            q
        }
    };
    qb.push(" ORDER BY t.id ASC LIMIT ");
    qb.push_bind(limit as i64);

    let rows = qb
        .build()
        .fetch_all(&mut **tx.tx_mut())
        .await
        .map_err(|e| OpenRagError::Database(format!("tag-only query failed for {space}: {e}")))?;

    let mut out = Vec::with_capacity(rows.len());
    for row in rows {
        let id: Uuid = row
            .try_get("id")
            .map_err(|e| OpenRagError::Database(format!("invalid tag-only id: {e}")))?;
        let document_id: Option<Uuid> = row
            .try_get("document_id")
            .map_err(|e| OpenRagError::Database(format!("invalid tag-only document_id: {e}")))?;
        let acl_scope_key: String = row
            .try_get("acl_scope_key")
            .map_err(|e| OpenRagError::Database(format!("invalid tag-only acl_scope_key: {e}")))?;
        let tags: Vec<String> = row
            .try_get("tags")
            .map_err(|e| OpenRagError::Database(format!("invalid tag-only tags: {e}")))?;
        out.push(TagOnlyHit {
            space,
            id,
            document_id,
            acl_scope_key,
            tags,
        });
    }
    Ok(out)
}

#[derive(Debug, Clone)]
struct DocRowMeta {
    title: String,
    min_auth_level: i16,
    revision: i32,
    valid_from: DateTime<Utc>,
    valid_to: Option<DateTime<Utc>>,
    status: String,
    tags: Vec<String>,
}

async fn hydrate_pg_candidates(
    tx: &mut RlsTransaction<'_>,
    org_id: Uuid,
    vector_hits: &[(RetrieveSpace, VectorHit)],
    bm25_hits: &[(RetrieveSpace, Bm25Hit)],
    tag_only_hits: &[TagOnlyHit],
    include_embeddings: bool,
) -> Result<HashMap<(RetrieveSpace, Uuid), HydratedCandidate>, OpenRagError> {
    let mut chunk_ids_set = BTreeSet::new();
    let mut page_ids_set = BTreeSet::new();
    let mut wiki_ids_set = BTreeSet::new();
    let mut mem_ids_set = BTreeSet::new();
    let mut doc_ids_set = BTreeSet::new();

    for (space, h) in vector_hits {
        match space {
            RetrieveSpace::Chunks => {
                chunk_ids_set.insert(h.id);
                if let Some(did) = h.document_id {
                    doc_ids_set.insert(did);
                }
            }
            RetrieveSpace::Thumb => {
                page_ids_set.insert(h.id);
                if let Some(did) = h.document_id {
                    doc_ids_set.insert(did);
                }
            }
            RetrieveSpace::Wiki => {
                wiki_ids_set.insert(h.id);
            }
            RetrieveSpace::Memory => {
                mem_ids_set.insert(h.id);
            }
        }
    }
    for (space, h) in bm25_hits {
        match space {
            RetrieveSpace::Chunks => {
                chunk_ids_set.insert(h.id);
                if let Some(did) = h.document_id {
                    doc_ids_set.insert(did);
                }
            }
            RetrieveSpace::Thumb => {
                page_ids_set.insert(h.id);
                if let Some(did) = h.document_id {
                    doc_ids_set.insert(did);
                }
            }
            RetrieveSpace::Wiki => {
                wiki_ids_set.insert(h.id);
            }
            RetrieveSpace::Memory => {
                mem_ids_set.insert(h.id);
            }
        }
    }
    for thit in tag_only_hits {
        match thit.space {
            RetrieveSpace::Chunks => {
                chunk_ids_set.insert(thit.id);
                if let Some(did) = thit.document_id {
                    doc_ids_set.insert(did);
                }
            }
            RetrieveSpace::Thumb => {
                page_ids_set.insert(thit.id);
                if let Some(did) = thit.document_id {
                    doc_ids_set.insert(did);
                }
            }
            RetrieveSpace::Wiki => {
                wiki_ids_set.insert(thit.id);
            }
            RetrieveSpace::Memory => {
                mem_ids_set.insert(thit.id);
            }
        }
    }

    let mut hydrated = HashMap::new();
    let mut docs_map: HashMap<Uuid, DocRowMeta> = HashMap::new();

    if !doc_ids_set.is_empty() {
        let doc_ids: Vec<Uuid> = doc_ids_set.into_iter().collect();
        let rows = sqlx::query(
            "SELECT id, title, min_auth_level, revision, valid_from, valid_to, status, tags
             FROM corpus.documents
             WHERE org_id = $1 AND id = ANY($2::uuid[])",
        )
        .bind(org_id)
        .bind(&doc_ids)
        .fetch_all(&mut **tx.tx_mut())
        .await
        .map_err(|e| OpenRagError::Database(format!("failed to hydrate corpus.documents: {e}")))?;

        for row in rows {
            let id: Uuid = row
                .try_get("id")
                .map_err(|e| OpenRagError::Database(format!("invalid doc id: {e}")))?;
            docs_map.insert(
                id,
                DocRowMeta {
                    title: row
                        .try_get("title")
                        .map_err(|e| OpenRagError::Database(format!("invalid doc title: {e}")))?,
                    min_auth_level: row.try_get("min_auth_level").map_err(|e| {
                        OpenRagError::Database(format!("invalid doc min_auth_level: {e}"))
                    })?,
                    revision: row.try_get("revision").map_err(|e| {
                        OpenRagError::Database(format!("invalid doc revision: {e}"))
                    })?,
                    valid_from: row.try_get("valid_from").map_err(|e| {
                        OpenRagError::Database(format!("invalid doc valid_from: {e}"))
                    })?,
                    valid_to: row.try_get("valid_to").map_err(|e| {
                        OpenRagError::Database(format!("invalid doc valid_to: {e}"))
                    })?,
                    status: row
                        .try_get("status")
                        .map_err(|e| OpenRagError::Database(format!("invalid doc status: {e}")))?,
                    tags: row
                        .try_get("tags")
                        .map_err(|e| OpenRagError::Database(format!("invalid doc tags: {e}")))?,
                },
            );
        }
    }

    if !chunk_ids_set.is_empty() {
        let chunk_ids: Vec<Uuid> = chunk_ids_set.into_iter().collect();
        let sql = if include_embeddings {
            "SELECT id, document_id, page_id, redacted_text, bbox_norm_1000, min_auth_level,
                    acl_scope_key, valid_from, valid_to, tags, embedding
             FROM corpus.chunks
             WHERE org_id = $1 AND id = ANY($2::uuid[])"
        } else {
            "SELECT id, document_id, page_id, redacted_text, bbox_norm_1000, min_auth_level,
                    acl_scope_key, valid_from, valid_to, tags, NULL::vector(256) AS embedding
             FROM corpus.chunks
             WHERE org_id = $1 AND id = ANY($2::uuid[])"
        };
        let rows = sqlx::query(sql)
            .bind(org_id)
            .bind(&chunk_ids)
            .fetch_all(&mut **tx.tx_mut())
            .await
            .map_err(|e| OpenRagError::Database(format!("failed to hydrate corpus.chunks: {e}")))?;

        for row in rows {
            let id: Uuid = row
                .try_get("id")
                .map_err(|e| OpenRagError::Database(format!("invalid chunk id: {e}")))?;
            let doc_id: Uuid = row
                .try_get("document_id")
                .map_err(|e| OpenRagError::Database(format!("invalid chunk document_id: {e}")))?;
            let Some(doc) = docs_map.get(&doc_id) else {
                // Fail-closed if parent document is not visible under RLS.
                continue;
            };
            let page_id: Option<Uuid> = row
                .try_get("page_id")
                .map_err(|e| OpenRagError::Database(format!("invalid chunk page_id: {e}")))?;
            let text: String = row
                .try_get("redacted_text")
                .map_err(|e| OpenRagError::Database(format!("invalid chunk redacted_text: {e}")))?;
            let bbox_arr: Option<Vec<i32>> = row.try_get("bbox_norm_1000").map_err(|e| {
                OpenRagError::Database(format!("invalid chunk bbox_norm_1000: {e}"))
            })?;
            let bbox_norm_1000 = bbox_arr.and_then(|arr| {
                if arr.len() == 4 {
                    BBoxNorm1000::from_array([arr[0], arr[1], arr[2], arr[3]]).ok()
                } else {
                    None
                }
            });
            let chunk_level: i16 = row.try_get("min_auth_level").map_err(|e| {
                OpenRagError::Database(format!("invalid chunk min_auth_level: {e}"))
            })?;
            let acl_scope_key: String = row
                .try_get("acl_scope_key")
                .map_err(|e| OpenRagError::Database(format!("invalid chunk acl_scope_key: {e}")))?;
            let chunk_valid_from: DateTime<Utc> = row
                .try_get("valid_from")
                .map_err(|e| OpenRagError::Database(format!("invalid chunk valid_from: {e}")))?;
            let chunk_valid_to: Option<DateTime<Utc>> = row
                .try_get("valid_to")
                .map_err(|e| OpenRagError::Database(format!("invalid chunk valid_to: {e}")))?;
            let chunk_tags: Vec<String> = row
                .try_get("tags")
                .map_err(|e| OpenRagError::Database(format!("invalid chunk tags: {e}")))?;
            let emb: Option<Vector> = row
                .try_get("embedding")
                .map_err(|e| OpenRagError::Database(format!("invalid chunk embedding: {e}")))?;

            let tags = if chunk_tags.is_empty() {
                doc.tags.clone()
            } else {
                chunk_tags
            };

            hydrated.insert(
                (RetrieveSpace::Chunks, id),
                HydratedCandidate {
                    id,
                    document_id: Some(doc_id),
                    page_id,
                    space: RetrieveSpace::Chunks,
                    title: Some(doc.title.clone()),
                    text,
                    tags,
                    min_auth_level: Some(chunk_level.max(doc.min_auth_level)),
                    revision: Some(doc.revision),
                    valid_from: Some(doc.valid_from.max(chunk_valid_from)),
                    valid_to: doc.valid_to.or(chunk_valid_to),
                    is_archived: doc.status == "archived",
                    is_superseded: false,
                    thumb_uri: None,
                    bbox_norm_1000,
                    acl_scope_key,
                    embedding: emb.map(|v| v.to_vec()),
                },
            );
        }
    }

    if !page_ids_set.is_empty() {
        let page_ids: Vec<Uuid> = page_ids_set.into_iter().collect();
        // Pages carry no vector (migration 012), so there is no embedding to hydrate.
        let sql = "SELECT id, document_id,
                    COALESCE(redacted_markdown, raw_markdown, '') AS page_text,
                    COALESCE(redacted_thumb_uri, thumb_uri) AS effective_thumb_uri,
                    min_auth_level, acl_scope_key
             FROM corpus.pages
             WHERE org_id = $1 AND id = ANY($2::uuid[])";
        let rows = sqlx::query(sql)
            .bind(org_id)
            .bind(&page_ids)
            .fetch_all(&mut **tx.tx_mut())
            .await
            .map_err(|e| OpenRagError::Database(format!("failed to hydrate corpus.pages: {e}")))?;

        for row in rows {
            let id: Uuid = row
                .try_get("id")
                .map_err(|e| OpenRagError::Database(format!("invalid page id: {e}")))?;
            let doc_id: Uuid = row
                .try_get("document_id")
                .map_err(|e| OpenRagError::Database(format!("invalid page document_id: {e}")))?;
            let Some(doc) = docs_map.get(&doc_id) else {
                continue;
            };
            let text: String = row
                .try_get("page_text")
                .map_err(|e| OpenRagError::Database(format!("invalid page_text: {e}")))?;
            let thumb_uri: Option<String> = row
                .try_get("effective_thumb_uri")
                .map_err(|e| OpenRagError::Database(format!("invalid thumb_uri: {e}")))?;
            let page_level: i16 = row
                .try_get("min_auth_level")
                .map_err(|e| OpenRagError::Database(format!("invalid page min_auth_level: {e}")))?;
            let acl_scope_key: String = row
                .try_get("acl_scope_key")
                .map_err(|e| OpenRagError::Database(format!("invalid page acl_scope_key: {e}")))?;
            hydrated.insert(
                (RetrieveSpace::Thumb, id),
                HydratedCandidate {
                    id,
                    document_id: Some(doc_id),
                    page_id: Some(id),
                    space: RetrieveSpace::Thumb,
                    title: Some(doc.title.clone()),
                    text,
                    tags: doc.tags.clone(),
                    min_auth_level: Some(page_level.max(doc.min_auth_level)),
                    revision: Some(doc.revision),
                    valid_from: Some(doc.valid_from),
                    valid_to: doc.valid_to,
                    is_archived: doc.status == "archived",
                    is_superseded: false,
                    thumb_uri,
                    bbox_norm_1000: None,
                    acl_scope_key,
                    embedding: None,
                },
            );
        }
    }

    if !wiki_ids_set.is_empty() {
        let wiki_ids: Vec<Uuid> = wiki_ids_set.into_iter().collect();
        let superseded_rows = sqlx::query_scalar::<_, Uuid>(
            "SELECT DISTINCT dst_page_id
             FROM wiki.edges
             WHERE org_id = $1
               AND relation_type = 'supersedes'
               AND dst_page_id = ANY($2::uuid[])",
        )
        .bind(org_id)
        .bind(&wiki_ids)
        .fetch_all(&mut **tx.tx_mut())
        .await
        .map_err(|e| OpenRagError::Database(format!("failed to hydrate wiki.edges: {e}")))?;
        let superseded_set: HashSet<Uuid> = superseded_rows.into_iter().collect();

        let sql = if include_embeddings {
            "SELECT id, title, body_md, version, min_auth_level, acl_scope_key, tags, created_at, embedding
             FROM wiki.pages
             WHERE org_id = $1 AND id = ANY($2::uuid[]) AND status = 'active'"
        } else {
            "SELECT id, title, body_md, version, min_auth_level, acl_scope_key, tags, created_at, NULL::vector(256) AS embedding
             FROM wiki.pages
             WHERE org_id = $1 AND id = ANY($2::uuid[]) AND status = 'active'"
        };
        let rows = sqlx::query(sql)
            .bind(org_id)
            .bind(&wiki_ids)
            .fetch_all(&mut **tx.tx_mut())
            .await
            .map_err(|e| OpenRagError::Database(format!("failed to hydrate wiki.pages: {e}")))?;

        for row in rows {
            let id: Uuid = row
                .try_get("id")
                .map_err(|e| OpenRagError::Database(format!("invalid wiki id: {e}")))?;
            let title: String = row
                .try_get("title")
                .map_err(|e| OpenRagError::Database(format!("invalid wiki title: {e}")))?;
            let body_md: String = row
                .try_get("body_md")
                .map_err(|e| OpenRagError::Database(format!("invalid wiki body_md: {e}")))?;
            let version: i32 = row
                .try_get("version")
                .map_err(|e| OpenRagError::Database(format!("invalid wiki version: {e}")))?;
            let min_auth_level: i16 = row
                .try_get("min_auth_level")
                .map_err(|e| OpenRagError::Database(format!("invalid wiki min_auth_level: {e}")))?;
            let acl_scope_key: String = row
                .try_get("acl_scope_key")
                .map_err(|e| OpenRagError::Database(format!("invalid wiki acl_scope_key: {e}")))?;
            let tags: Vec<String> = row
                .try_get("tags")
                .map_err(|e| OpenRagError::Database(format!("invalid wiki tags: {e}")))?;
            let created_at: DateTime<Utc> = row
                .try_get("created_at")
                .map_err(|e| OpenRagError::Database(format!("invalid wiki created_at: {e}")))?;
            let emb: Option<Vector> = row
                .try_get("embedding")
                .map_err(|e| OpenRagError::Database(format!("invalid wiki embedding: {e}")))?;

            hydrated.insert(
                (RetrieveSpace::Wiki, id),
                HydratedCandidate {
                    id,
                    document_id: None,
                    page_id: None,
                    space: RetrieveSpace::Wiki,
                    title: Some(title),
                    text: body_md,
                    tags,
                    min_auth_level: Some(min_auth_level),
                    revision: Some(version),
                    valid_from: Some(created_at),
                    valid_to: None,
                    is_archived: false,
                    is_superseded: superseded_set.contains(&id),
                    thumb_uri: None,
                    bbox_norm_1000: None,
                    acl_scope_key,
                    embedding: emb.map(|v| v.to_vec()),
                },
            );
        }
    }

    if !mem_ids_set.is_empty() {
        let mem_ids: Vec<Uuid> = mem_ids_set.into_iter().collect();
        let superseded_rows = sqlx::query_scalar::<_, Uuid>(
            "SELECT DISTINCT dst_memory_id
             FROM memory.edges
             WHERE org_id = $1
               AND relation_type = 'supersedes'
               AND dst_memory_id = ANY($2::uuid[])",
        )
        .bind(org_id)
        .bind(&mem_ids)
        .fetch_all(&mut **tx.tx_mut())
        .await
        .map_err(|e| OpenRagError::Database(format!("failed to hydrate memory.edges: {e}")))?;
        let superseded_set: HashSet<Uuid> = superseded_rows.into_iter().collect();

        let sql = if include_embeddings {
            "SELECT id, title, effective_text, version, valid_from, valid_to, min_auth_level,
                    acl_scope_key, tags, superseded_by, embedding
             FROM memory.entries
             WHERE org_id = $1 AND id = ANY($2::uuid[]) AND status = 'active'"
        } else {
            "SELECT id, title, effective_text, version, valid_from, valid_to, min_auth_level,
                    acl_scope_key, tags, superseded_by, NULL::vector(256) AS embedding
             FROM memory.entries
             WHERE org_id = $1 AND id = ANY($2::uuid[]) AND status = 'active'"
        };
        let rows = sqlx::query(sql)
            .bind(org_id)
            .bind(&mem_ids)
            .fetch_all(&mut **tx.tx_mut())
            .await
            .map_err(|e| {
                OpenRagError::Database(format!("failed to hydrate memory.entries: {e}"))
            })?;

        for row in rows {
            let id: Uuid = row
                .try_get("id")
                .map_err(|e| OpenRagError::Database(format!("invalid memory id: {e}")))?;
            let title: Option<String> = row
                .try_get("title")
                .map_err(|e| OpenRagError::Database(format!("invalid memory title: {e}")))?;
            let effective_text: String = row.try_get("effective_text").map_err(|e| {
                OpenRagError::Database(format!("invalid memory effective_text: {e}"))
            })?;
            let version: i32 = row
                .try_get("version")
                .map_err(|e| OpenRagError::Database(format!("invalid memory version: {e}")))?;
            let valid_from: DateTime<Utc> = row
                .try_get("valid_from")
                .map_err(|e| OpenRagError::Database(format!("invalid memory valid_from: {e}")))?;
            let valid_to: Option<DateTime<Utc>> = row
                .try_get("valid_to")
                .map_err(|e| OpenRagError::Database(format!("invalid memory valid_to: {e}")))?;
            let min_auth_level: i16 = row.try_get("min_auth_level").map_err(|e| {
                OpenRagError::Database(format!("invalid memory min_auth_level: {e}"))
            })?;
            let acl_scope_key: String = row.try_get("acl_scope_key").map_err(|e| {
                OpenRagError::Database(format!("invalid memory acl_scope_key: {e}"))
            })?;
            let tags: Vec<String> = row
                .try_get("tags")
                .map_err(|e| OpenRagError::Database(format!("invalid memory tags: {e}")))?;
            let superseded_by: Option<Uuid> = row.try_get("superseded_by").map_err(|e| {
                OpenRagError::Database(format!("invalid memory superseded_by: {e}"))
            })?;
            let emb: Option<Vector> = row
                .try_get("embedding")
                .map_err(|e| OpenRagError::Database(format!("invalid memory embedding: {e}")))?;

            hydrated.insert(
                (RetrieveSpace::Memory, id),
                HydratedCandidate {
                    id,
                    document_id: None,
                    page_id: None,
                    space: RetrieveSpace::Memory,
                    title,
                    text: effective_text,
                    tags,
                    min_auth_level: Some(min_auth_level),
                    revision: Some(version),
                    valid_from: Some(valid_from),
                    valid_to,
                    is_archived: false,
                    is_superseded: superseded_by.is_some() || superseded_set.contains(&id),
                    thumb_uri: None,
                    bbox_norm_1000: None,
                    acl_scope_key,
                    embedding: emb.map(|v| v.to_vec()),
                },
            );
        }
    }

    Ok(hydrated)
}

// ============================================================================
// Offline In-Memory RetrieveBackend (for deterministic unit/BVA testing)
// ============================================================================

#[derive(Debug, Clone)]
pub struct InMemoryDocRecord {
    pub id: Uuid,
    pub org_id: Uuid,
    pub bucket_id: Option<Uuid>,
    pub dept_id: Uuid,
    pub owner_principal_id: Uuid,
    pub access_scope: AccessScope,
    pub allowed_group_ids: Vec<Uuid>,
    pub denied_group_ids: Vec<Uuid>,
    pub min_auth_level: i16,
    pub space: RetrieveSpace,
    pub document_id: Option<Uuid>,
    pub page_id: Option<Uuid>,
    pub title: Option<String>,
    pub text: String,
    pub tags: Vec<String>,
    /// Optional document-level tags (used to model `CorpusPages` where tags live on `corpus.documents`!).
    pub document_tags: Option<Vec<String>>,
    pub revision: i32,
    pub valid_from: DateTime<Utc>,
    pub valid_to: Option<DateTime<Utc>>,
    pub is_archived: bool,
    pub is_superseded: bool,
    pub thumb_uri: Option<String>,
    pub bbox_norm_1000: Option<BBoxNorm1000>,
    pub acl_scope_key: String,
    pub embedding: Option<Vec<f32>>,
}

impl InMemoryDocRecord {
    pub fn can_read(&self, ctx: &PrincipalContext) -> bool {
        if self.org_id != ctx.org_id {
            return false;
        }
        if self
            .denied_group_ids
            .iter()
            .any(|g| ctx.group_ids.contains(g))
        {
            return false;
        }
        if i16::from(ctx.auth_level.as_u8()) < self.min_auth_level {
            return false;
        }
        match self.access_scope {
            AccessScope::Private => {
                self.owner_principal_id == ctx.principal_id
                    || ctx.dept_admin_ids.contains(&self.dept_id)
            }
            AccessScope::Dept => {
                ctx.dept_ids.contains(&self.dept_id)
                    || self
                        .allowed_group_ids
                        .iter()
                        .any(|g| ctx.group_ids.contains(g))
            }
            AccessScope::Group => self
                .allowed_group_ids
                .iter()
                .any(|g| ctx.group_ids.contains(g)),
            AccessScope::Org => true,
        }
    }
}

pub struct InMemoryRetrieveBackend {
    records: Arc<RwLock<Vec<InMemoryDocRecord>>>,
    concepts: Arc<InMemoryConceptStore>,
    embedder: Arc<dyn TextEmbedder>,
}

impl Default for InMemoryRetrieveBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl InMemoryRetrieveBackend {
    pub fn new() -> Self {
        Self {
            records: Arc::new(RwLock::new(Vec::new())),
            concepts: Arc::new(InMemoryConceptStore::new()),
            embedder: Arc::new(DeterministicTextEmbedder::default()),
        }
    }

    pub fn concepts(&self) -> &Arc<InMemoryConceptStore> {
        &self.concepts
    }

    pub async fn insert(&self, record: InMemoryDocRecord) {
        self.records.write().await.push(record);
    }
}

#[async_trait]
impl RetrieveBackend for InMemoryRetrieveBackend {
    async fn execute_plan(
        &self,
        ctx: &PrincipalContext,
        input: &BackendExecutionInput,
    ) -> Result<BackendExecutionOutput, OpenRagError> {
        let lookup_terms = build_concept_lookup_terms(&input.clean_query);
        let expanded_terms = if lookup_terms.is_empty() {
            Vec::new()
        } else {
            self.concepts.expand_terms(ctx.org_id, &lookup_terms).await
        };

        let mut alias_map: HashMap<String, Vec<String>> = HashMap::new();
        if input.effective_tag_filter.expand_tag_aliases {
            let mut to_resolve = Vec::new();
            to_resolve.extend(input.effective_tag_filter.tags_any.iter().cloned());
            to_resolve.extend(input.effective_tag_filter.tags_all.iter().cloned());
            to_resolve.extend(input.effective_tag_filter.tags_exclude.iter().cloned());
            for tag in to_resolve {
                if let Some(resolved) = self.concepts.resolve(ctx.org_id, &tag).await {
                    let mut list = vec![resolved.concept.canonical_key];
                    for al in resolved.aliases {
                        list.push(al.alias_text);
                        list.push(al.normalized_alias);
                        list.push(al.compact_alias);
                    }
                    alias_map.insert(normalize_tag(&tag), list);
                }
            }
        }

        let expanded_tag_filter = if input.effective_tag_filter.is_empty() {
            None
        } else {
            Some(expand_tag_filter_in_memory(
                &input.effective_tag_filter,
                |t| {
                    alias_map
                        .get(&normalize_tag(t))
                        .cloned()
                        .unwrap_or_default()
                },
            ))
        };

        let query_vector = match &input.explicit_query_vector {
            Some(v) => v.clone(),
            None => {
                let embed_text = build_embed_query_text(
                    &input.clean_query,
                    &expanded_terms,
                    &input.effective_tag_filter,
                );
                self.embedder.embed_query(&embed_text).await?
            }
        };

        let guard = self.records.read().await;
        let visible: Vec<&InMemoryDocRecord> = guard
            .iter()
            .filter(|r| !r.is_archived)
            .filter(|r| r.can_read(ctx))
            .filter(|r| match input.bucket_id {
                Some(bid) => r.bucket_id == Some(bid),
                None => true,
            })
            .collect();

        let mut vector_hits = Vec::new();
        let mut bm25_hits = Vec::new();
        let mut tag_only_hits = Vec::new();
        let mut hydrated = HashMap::new();
        let has_searchable = query_has_searchable_tokens(&input.clean_query);

        for &space in &input.spaces {
            let table = space.vector_table();
            let is_thumb = space == RetrieveSpace::Thumb;
            let space_recs: Vec<&InMemoryDocRecord> = visible
                .iter()
                .copied()
                .filter(|r| r.space == space)
                .filter(|r| match &expanded_tag_filter {
                    Some(etf) if !etf.is_empty() => {
                        let eff_tags = if is_thumb {
                            r.document_tags.as_ref().unwrap_or(&r.tags)
                        } else {
                            &r.tags
                        };
                        etf.matches_tags(eff_tags)
                    }
                    _ => true,
                })
                .collect();

            let space_limit = input.candidate_limit_per_space;

            // Vector search (archived & tag filters applied before truncating to space_limit).
            let mut v_cands = Vec::new();
            for rec in &space_recs {
                let effective_tags = if is_thumb {
                    rec.document_tags
                        .clone()
                        .unwrap_or_else(|| rec.tags.clone())
                } else {
                    rec.tags.clone()
                };
                // Page thumbnails carry no vector: the Thumb space is lexical/tag only.
                if let Some(emb) = rec.embedding.as_ref().filter(|_| table.has_vectors()) {
                    if let Ok(sim) = cosine_similarity(&query_vector, emb) {
                        v_cands.push(VectorHit {
                            id: rec.id,
                            document_id: rec.document_id,
                            table,
                            cosine_sim: sim,
                            distance: 1.0 - sim,
                            acl_scope_key: rec.acl_scope_key.clone(),
                            tags: effective_tags,
                            metadata: serde_json::Map::new(),
                        });
                    }
                }
            }
            v_cands.sort_by(|a, b| {
                a.distance
                    .total_cmp(&b.distance)
                    .then_with(|| a.id.cmp(&b.id))
            });
            v_cands.truncate(space_limit);
            for h in v_cands {
                vector_hits.push((space, h));
            }

            // BM25 search (using query_has_searchable_tokens and document_tags for Thumb).
            if has_searchable {
                if !(input.skip_lexical && table.has_vectors()) {
                    let bm25_docs: Vec<Bm25DocInput> = space_recs
                        .iter()
                        .map(|r| {
                            let effective_tags = if is_thumb {
                                r.document_tags.clone().unwrap_or_else(|| r.tags.clone())
                            } else {
                                r.tags.clone()
                            };
                            Bm25DocInput {
                                id: r.id,
                                document_id: r.document_id,
                                text: format!("{} {}", r.title.as_deref().unwrap_or(""), r.text),
                                tags: effective_tags,
                            }
                        })
                        .collect();
                    let idx = Bm25Index::from_docs(&bm25_docs);
                    let b_hits = idx.search_with_expansion(
                        table,
                        &input.clean_query,
                        &expanded_terms,
                        expanded_tag_filter.as_ref(),
                        space_limit,
                    );
                    for h in b_hits {
                        bm25_hits.push((space, h));
                    }
                }
            } else if input.tag_only_query {
                if let Some(etf) = &expanded_tag_filter {
                    for rec in &space_recs {
                        let effective_tags = rec.document_tags.as_ref().unwrap_or(&rec.tags);
                        if !rec.is_archived && etf.matches_tags(effective_tags) {
                            tag_only_hits.push(TagOnlyHit {
                                space,
                                id: rec.id,
                                document_id: rec.document_id,
                                acl_scope_key: rec.acl_scope_key.clone(),
                                tags: effective_tags.clone(),
                            });
                        }
                    }
                }
            }

            for rec in space_recs {
                let effective_tags = if is_thumb {
                    rec.document_tags
                        .clone()
                        .unwrap_or_else(|| rec.tags.clone())
                } else {
                    rec.tags.clone()
                };
                hydrated.insert(
                    (space, rec.id),
                    HydratedCandidate {
                        id: rec.id,
                        document_id: rec.document_id,
                        page_id: rec.page_id,
                        space,
                        title: rec.title.clone(),
                        text: rec.text.clone(),
                        tags: effective_tags,
                        min_auth_level: Some(rec.min_auth_level),
                        revision: Some(rec.revision),
                        valid_from: Some(rec.valid_from),
                        valid_to: rec.valid_to,
                        is_archived: rec.is_archived,
                        is_superseded: rec.is_superseded,
                        thumb_uri: rec.thumb_uri.clone(),
                        bbox_norm_1000: rec.bbox_norm_1000,
                        acl_scope_key: rec.acl_scope_key.clone(),
                        embedding: if input.include_embeddings {
                            rec.embedding.clone()
                        } else {
                            None
                        },
                    },
                );
            }
        }

        Ok(BackendExecutionOutput {
            vector_hits,
            bm25_hits,
            tag_only_hits,
            expanded_terms,
            expanded_tag_filter,
            query_vector,
            hydrated,
        })
    }

    async fn compute_bucket_facets(
        &self,
        ctx: &PrincipalContext,
        spaces: &[RetrieveSpace],
        bucket_id: Option<Uuid>,
        limit: i64,
    ) -> Result<BTreeMap<String, i64>, OpenRagError> {
        if limit <= 0 {
            return Ok(BTreeMap::new());
        }
        let guard = self.records.read().await;
        let mut seen_entities: HashSet<(u8, Uuid)> = HashSet::new();
        let mut counts: HashMap<String, i64> = HashMap::new();

        for rec in guard.iter() {
            if !spaces.contains(&rec.space) || rec.is_archived || !rec.can_read(ctx) {
                continue;
            }
            if let Some(bid) = bucket_id {
                if rec.bucket_id != Some(bid) {
                    continue;
                }
            }
            let ek = match rec.space {
                RetrieveSpace::Chunks | RetrieveSpace::Thumb => {
                    (0_u8, rec.document_id.unwrap_or(rec.id))
                }
                RetrieveSpace::Wiki => (1_u8, rec.id),
                RetrieveSpace::Memory => (2_u8, rec.id),
            };
            if !seen_entities.insert(ek) {
                continue;
            }
            let tags = rec.document_tags.as_ref().unwrap_or(&rec.tags);
            let unique_tags: BTreeSet<String> = tags
                .iter()
                .map(|t| normalize_tag(t))
                .filter(|t| !t.is_empty())
                .collect();
            for tag in unique_tags {
                *counts.entry(tag).or_insert(0) += 1;
            }
        }

        let mut sorted: Vec<(String, i64)> = counts.into_iter().collect();
        sorted.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        sorted.truncate(limit as usize);
        Ok(sorted.into_iter().collect())
    }
}
