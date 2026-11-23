use async_trait::async_trait;
use chrono::{DateTime, Utc};
use openrag_core::{ExternalMirrorBackend, OpenRagError, PgPineconeEngine};
use openrag_db::{DbPool, RlsTransaction};
use pgvector::Vector;
use serde::{Deserialize, Serialize};
use serde_json::json;
use sqlx::{Postgres, QueryBuilder, Row};
use uuid::Uuid;

use crate::s3_vectors::{s3_bucket_arn, s3_bucket_name, NON_FILTERABLE_KEYS};
use crate::tags::{expand_tag_filter_rls, push_tag_filter_predicates};
use crate::types::{validate_vector, StoreMode, VectorHit, VectorSearchQuery, VectorTable, VECTOR_DIM};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IndexBinding {
    pub id: Uuid,
    pub org_id: Uuid,
    pub table_name: String,
    pub vector_column: String,
    pub pg_pinecone_engine: PgPineconeEngine,
    pub external_backend: ExternalMirrorBackend,
    pub external_config: serde_json::Value,
    pub dimension: i16,
    pub filter_column: Option<String>,
    pub status: String,
    pub updated_at: DateTime<Utc>,
}

#[async_trait]
pub trait RemoteVectorQuerier: Send + Sync {
    async fn query_remote_ids(
        &self,
        binding: &IndexBinding,
        query: &VectorSearchQuery,
    ) -> Result<Vec<(Uuid, f32)>, OpenRagError>;
}

#[derive(Default)]
pub struct PgPineconeManager {
    remote_querier: Option<Box<dyn RemoteVectorQuerier>>,
}

impl PgPineconeManager {
    pub fn new() -> Self {
        Self {
            remote_querier: None,
        }
    }

    pub fn with_remote_querier(mut self, querier: Box<dyn RemoteVectorQuerier>) -> Self {
        self.remote_querier = Some(querier);
        self
    }

    pub async fn set_engine(
        worker_pool: &DbPool,
        org_id: Uuid,
        table: VectorTable,
        engine: PgPineconeEngine,
        mode: StoreMode,
        pinecone_api_key: Option<&str>,
    ) -> Result<IndexBinding, OpenRagError> {
        if engine == PgPineconeEngine::Remote
            && mode == StoreMode::Live
            && pinecone_api_key.map(str::trim).unwrap_or("").is_empty()
        {
            return Err(OpenRagError::ExternalAdapter {
                adapter: "pinecone".into(),
                message: "503 Service Unavailable: PINECONE_API_KEY is required when switching to remote engine in live mode".into(),
            });
        }

        let caps = worker_pool.capabilities().await?;
        let table_str = table.as_str();
        let vec_col = table.require_vector_column()?;
        let engine_str = match engine {
            PgPineconeEngine::Native => "native",
            PgPineconeEngine::Pgvector => "pgvector",
            PgPineconeEngine::Remote => "remote",
        };

        if caps.has_pg_pinecone {
            let attempt = sqlx::query(
                "SELECT pinecone.create_index($1, $2, engine => $3, filter_column => 'acl_scope_key')",
            )
            .bind(table_str)
            .bind(vec_col)
            .bind(engine_str)
            .execute(worker_pool.inner())
            .await;

            if attempt.is_err() {
                Self::ensure_pgvector_hnsw(worker_pool, table).await?;
            }
        } else {
            Self::ensure_pgvector_hnsw(worker_pool, table).await?;
        }

        let row = sqlx::query(
            "INSERT INTO vectors.index_bindings
             (org_id, table_name, vector_column, pg_pinecone_engine, dimension, filter_column, status, updated_at)
             VALUES ($1, $2, $3, $4, $5, 'acl_scope_key', 'ready', now())
             ON CONFLICT (org_id, table_name, vector_column)
             DO UPDATE SET
                 pg_pinecone_engine = EXCLUDED.pg_pinecone_engine,
                 status = 'ready',
                 updated_at = now()
             RETURNING id, org_id, table_name, vector_column, pg_pinecone_engine, external_backend,
                       external_config, dimension, filter_column, status, updated_at",
        )
        .bind(org_id)
        .bind(table_str)
        .bind(vec_col)
        .bind(engine_str)
        .bind(VECTOR_DIM as i16)
        .fetch_one(worker_pool.inner())
        .await
        .map_err(|e| OpenRagError::Database(format!("failed to upsert index_binding: {e}")))?;

        row_to_binding(&row)
    }

    async fn ensure_pgvector_hnsw(
        worker_pool: &DbPool,
        table: VectorTable,
    ) -> Result<(), OpenRagError> {
        let (schema, rel) = table
            .as_str()
            .split_once('.')
            .ok_or_else(|| OpenRagError::Internal("invalid VectorTable format".into()))?;
        let idx_name = table
            .hnsw_index_name()
            .ok_or_else(|| OpenRagError::Validation(format!("{} has no vector index", table.as_str())))?;
        let exists: bool = sqlx::query_scalar(
            "SELECT EXISTS (
                 SELECT 1 FROM pg_indexes
                 WHERE schemaname = $1 AND tablename = $2 AND indexname = $3
             )",
        )
        .bind(schema)
        .bind(rel)
        .bind(idx_name)
        .fetch_one(worker_pool.inner())
        .await
        .map_err(|e| OpenRagError::Database(format!("failed to check pg_indexes: {e}")))?;
        if exists {
            return Ok(());
        }

        let ddl = format!(
            "CREATE INDEX IF NOT EXISTS {} ON {} USING hnsw ({} vector_cosine_ops)",
            idx_name,
            table.as_str(),
            table.require_vector_column()?
        );
        sqlx::query(&ddl)
            .execute(worker_pool.inner())
            .await
            .map_err(|e| OpenRagError::Database(format!("failed to ensure pgvector HNSW index: {e}")))?;
        Ok(())
    }

    pub async fn set_external_mirror(
        worker_pool: &DbPool,
        org_id: Uuid,
        table: VectorTable,
        backend: ExternalMirrorBackend,
        mut external_config: serde_json::Value,
    ) -> Result<IndexBinding, OpenRagError> {
        let backend_str = match backend {
            ExternalMirrorBackend::None => "none",
            ExternalMirrorBackend::S3Vectors => "s3_vectors",
            ExternalMirrorBackend::Qdrant => "qdrant",
            ExternalMirrorBackend::Faiss => "faiss",
            ExternalMirrorBackend::Usearch => "usearch",
        };

        if backend == ExternalMirrorBackend::S3Vectors {
            let prefix = external_config
                .get("bucket_prefix")
                .and_then(|v| v.as_str())
                .unwrap_or("sv-prod")
                .to_string();
            let region = external_config
                .get("region")
                .and_then(|v| v.as_str())
                .unwrap_or("ap-northeast-2")
                .to_string();
            let account_id = external_config
                .get("account_id")
                .and_then(|v| v.as_str())
                .unwrap_or("000000000000")
                .to_string();
            let bucket_name = s3_bucket_name(&prefix, org_id);
            let bucket_arn = s3_bucket_arn(&region, &account_id, &prefix, org_id);

            let obj = external_config
                .as_object_mut()
                .ok_or_else(|| OpenRagError::Validation("external_config must be a JSON object".into()))?;
            obj.insert("bucket_name".into(), json!(bucket_name));
            obj.insert("bucket_arn".into(), json!(bucket_arn));
            obj.insert("region".into(), json!(region));
            obj.insert("non_filterable_keys".into(), json!(NON_FILTERABLE_KEYS));
        }

        let row = sqlx::query(
            "INSERT INTO vectors.index_bindings
             (org_id, table_name, vector_column, external_backend, external_config, dimension, filter_column, status, updated_at)
             VALUES ($1, $2, $3, $4, $5, $6, 'acl_scope_key', 'ready', now())
             ON CONFLICT (org_id, table_name, vector_column)
             DO UPDATE SET
                 external_backend = EXCLUDED.external_backend,
                 external_config = EXCLUDED.external_config,
                 status = 'ready',
                 updated_at = now()
             RETURNING id, org_id, table_name, vector_column, pg_pinecone_engine, external_backend,
                       external_config, dimension, filter_column, status, updated_at",
        )
        .bind(org_id)
        .bind(table.as_str())
        .bind(table.require_vector_column()?)
        .bind(backend_str)
        .bind(external_config)
        .bind(VECTOR_DIM as i16)
        .fetch_one(worker_pool.inner())
        .await
        .map_err(|e| OpenRagError::Database(format!("failed to set external mirror binding: {e}")))?;

        row_to_binding(&row)
    }

    pub async fn get_binding(
        worker_pool: &DbPool,
        org_id: Uuid,
        table: VectorTable,
    ) -> Result<Option<IndexBinding>, OpenRagError> {
        let row = sqlx::query(
            "SELECT id, org_id, table_name, vector_column, pg_pinecone_engine, external_backend,
                    external_config, dimension, filter_column, status, updated_at
             FROM vectors.index_bindings
             WHERE org_id = $1 AND table_name = $2 AND vector_column = $3",
        )
        .bind(org_id)
        .bind(table.as_str())
        .bind(table.require_vector_column()?)
        .fetch_optional(worker_pool.inner())
        .await
        .map_err(|e| OpenRagError::Database(format!("failed to fetch index_binding: {e}")))?;

        row.as_ref().map(row_to_binding).transpose()
    }

    pub async fn list_bindings(
        worker_pool: &DbPool,
        org_id: Uuid,
    ) -> Result<Vec<IndexBinding>, OpenRagError> {
        let rows = sqlx::query(
            "SELECT id, org_id, table_name, vector_column, pg_pinecone_engine, external_backend,
                    external_config, dimension, filter_column, status, updated_at
             FROM vectors.index_bindings
             WHERE org_id = $1
             ORDER BY table_name ASC, vector_column ASC",
        )
        .bind(org_id)
        .fetch_all(worker_pool.inner())
        .await
        .map_err(|e| OpenRagError::Database(format!("failed to list index_bindings: {e}")))?;

        rows.iter().map(row_to_binding).collect()
    }

    /// Single invariant 256d RLS vector search path across `native`, `pgvector`, and `remote`.
    pub async fn search_rls(
        &self,
        tx: &mut RlsTransaction<'_>,
        binding: Option<&IndexBinding>,
        query: &VectorSearchQuery,
    ) -> Result<Vec<VectorHit>, OpenRagError> {
        validate_vector(&query.query_vector)?;
        if query.top_k == 0 {
            return Ok(Vec::new());
        }

        // If binding is Remote with a live remote querier, fetch remote candidate IDs
        // and still intersect through the RLS-protected PostgreSQL query below.
        let remote_candidate_ids: Option<Vec<Uuid>> = if let (Some(b), Some(querier)) = (binding, &self.remote_querier) {
            if b.pg_pinecone_engine == PgPineconeEngine::Remote {
                let pairs = querier.query_remote_ids(b, query).await?;
                if pairs.is_empty() {
                    return Ok(Vec::new());
                }
                Some(pairs.into_iter().map(|(id, _)| id).collect())
            } else {
                None
            }
        } else {
            None
        };

        let expanded_tags = if let Some(tf) = &query.tag_filter {
            Some(expand_tag_filter_rls(tx, query.org_id, tf).await?)
        } else {
            None
        };

        let qvec = Vector::from(query.query_vector.clone());
        let vec_col = query.table.require_vector_column()?;

        let mut qb: QueryBuilder<'_, Postgres> = QueryBuilder::new("SELECT t.id, ");
        if query.table.has_document_id() {
            qb.push("t.document_id");
        } else {
            qb.push("NULL::uuid AS document_id");
        }
        qb.push(", t.acl_scope_key, ");
        if query.table.has_tags_column() {
            qb.push("t.tags");
        } else {
            qb.push("ARRAY[]::text[] AS tags");
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
        qb.push(query.table.as_str());
        qb.push(" AS t");

        if matches!(
            query.table,
            VectorTable::CorpusChunks | VectorTable::CorpusPages | VectorTable::CorpusPageFigures
        ) && query.bucket_id.is_some()
        {
            qb.push(" JOIN corpus.documents AS d ON d.id = t.document_id AND d.org_id = t.org_id");
        }

        qb.push(" WHERE t.org_id = ");
        qb.push_bind(query.org_id);
        qb.push(" AND t.");
        qb.push(vec_col);
        qb.push(" IS NOT NULL");

        if query.table.has_status_active() {
            qb.push(" AND t.status = 'active'");
        }
        if let Some(scope_key) = &query.acl_scope_key {
            qb.push(" AND t.acl_scope_key = ");
            qb.push_bind(scope_key);
        }
        if let Some(bid) = query.bucket_id {
            match query.table {
                VectorTable::CorpusChunks
                | VectorTable::CorpusPages
                | VectorTable::CorpusPageFigures => {
                    qb.push(" AND d.bucket_id = ");
                    qb.push_bind(bid);
                }
                VectorTable::WikiPages | VectorTable::MemoryEntries => {
                    qb.push(" AND t.bucket_id = ");
                    qb.push_bind(bid);
                }
            }
        }
        if let Some(ids) = remote_candidate_ids {
            qb.push(" AND t.id = ANY(");
            qb.push_bind(ids);
            qb.push("::uuid[])");
        }
        if let Some(tf) = &expanded_tags {
            if query.table.has_tags_column() {
                push_tag_filter_predicates(&mut qb, "t.tags", tf);
            }
        }

        qb.push(" ORDER BY t.");
        qb.push(vec_col);
        qb.push(" <=> ");
        qb.push_bind(qvec);
        qb.push(" ASC, t.id ASC LIMIT ");
        qb.push_bind(query.top_k as i64);

        let rows = qb
            .build()
            .fetch_all(&mut **tx.tx_mut())
            .await
            .map_err(|e| OpenRagError::Database(format!("vector search query failed: {e}")))?;

        let mut hits = Vec::with_capacity(rows.len());
        for row in rows {
            let id: Uuid = row
                .try_get("id")
                .map_err(|e| OpenRagError::Database(format!("invalid hit id: {e}")))?;
            let document_id: Option<Uuid> = row
                .try_get("document_id")
                .map_err(|e| OpenRagError::Database(format!("invalid hit document_id: {e}")))?;
            let acl_scope_key: String = row
                .try_get("acl_scope_key")
                .map_err(|e| OpenRagError::Database(format!("invalid hit acl_scope_key: {e}")))?;
            let tags: Vec<String> = row
                .try_get("tags")
                .map_err(|e| OpenRagError::Database(format!("invalid hit tags: {e}")))?;
            let cosine_sim: f32 = row
                .try_get("cosine_sim")
                .map_err(|e| OpenRagError::Database(format!("invalid hit cosine_sim: {e}")))?;
            let distance: f32 = row
                .try_get("distance")
                .map_err(|e| OpenRagError::Database(format!("invalid hit distance: {e}")))?;
            hits.push(VectorHit {
                id,
                document_id,
                table: query.table,
                cosine_sim,
                distance,
                acl_scope_key,
                tags,
                metadata: serde_json::Map::new(),
            });
        }
        Ok(hits)
    }
}

fn row_to_binding(row: &sqlx::postgres::PgRow) -> Result<IndexBinding, OpenRagError> {
    let eng_str: String = row
        .try_get("pg_pinecone_engine")
        .map_err(|e| OpenRagError::Database(format!("invalid pg_pinecone_engine: {e}")))?;
    let ext_str: String = row
        .try_get("external_backend")
        .map_err(|e| OpenRagError::Database(format!("invalid external_backend: {e}")))?;

    let pg_pinecone_engine = match eng_str.as_str() {
        "native" => PgPineconeEngine::Native,
        "pgvector" => PgPineconeEngine::Pgvector,
        "remote" => PgPineconeEngine::Remote,
        other => {
            return Err(OpenRagError::Database(format!(
                "unknown pg_pinecone_engine '{other}'"
            )));
        }
    };
    let external_backend = match ext_str.as_str() {
        "none" => ExternalMirrorBackend::None,
        "s3_vectors" => ExternalMirrorBackend::S3Vectors,
        "qdrant" => ExternalMirrorBackend::Qdrant,
        "faiss" => ExternalMirrorBackend::Faiss,
        "usearch" => ExternalMirrorBackend::Usearch,
        other => {
            return Err(OpenRagError::Database(format!(
                "unknown external_backend '{other}'"
            )));
        }
    };

    Ok(IndexBinding {
        id: row
            .try_get("id")
            .map_err(|e| OpenRagError::Database(format!("invalid id: {e}")))?,
        org_id: row
            .try_get("org_id")
            .map_err(|e| OpenRagError::Database(format!("invalid org_id: {e}")))?,
        table_name: row
            .try_get("table_name")
            .map_err(|e| OpenRagError::Database(format!("invalid table_name: {e}")))?,
        vector_column: row
            .try_get("vector_column")
            .map_err(|e| OpenRagError::Database(format!("invalid vector_column: {e}")))?,
        pg_pinecone_engine,
        external_backend,
        external_config: row
            .try_get("external_config")
            .map_err(|e| OpenRagError::Database(format!("invalid external_config: {e}")))?,
        dimension: row
            .try_get("dimension")
            .map_err(|e| OpenRagError::Database(format!("invalid dimension: {e}")))?,
        filter_column: row
            .try_get("filter_column")
            .map_err(|e| OpenRagError::Database(format!("invalid filter_column: {e}")))?,
        status: row
            .try_get("status")
            .map_err(|e| OpenRagError::Database(format!("invalid status: {e}")))?,
        updated_at: row
            .try_get("updated_at")
            .map_err(|e| OpenRagError::Database(format!("invalid updated_at: {e}")))?,
    })
}
