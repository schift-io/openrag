-- OpenRAG Migration 002: Vectors Bindings (128d) & Corpus Tables + Fail-Closed RLS

CREATE TABLE vectors.index_bindings (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id              UUID NOT NULL REFERENCES iam.orgs(id) ON DELETE CASCADE,
    table_name          TEXT NOT NULL,
    vector_column       TEXT NOT NULL DEFAULT 'embedding',
    pg_pinecone_engine  TEXT NOT NULL DEFAULT 'native'
                        CHECK (pg_pinecone_engine IN ('native', 'pgvector', 'remote')),
    external_backend    TEXT NOT NULL DEFAULT 'none'
                        CHECK (external_backend IN ('none', 's3_vectors', 'qdrant', 'faiss', 'usearch')),
    external_config     JSONB NOT NULL DEFAULT '{}'::jsonb,
    dimension           SMALLINT NOT NULL DEFAULT 128 CHECK (dimension > 0),
    filter_column       TEXT DEFAULT 'acl_scope_key',
    status              TEXT NOT NULL DEFAULT 'ready',
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (org_id, table_name, vector_column)
);

-- Note: pipeline_config_id FK to eval.pipeline_configs is added in 006_eval_bench.sql
CREATE TABLE corpus.buckets (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id              UUID NOT NULL REFERENCES iam.orgs(id) ON DELETE CASCADE,
    dept_id             UUID NOT NULL,
    owner_principal_id  UUID NOT NULL,
    slug                TEXT NOT NULL,
    name                TEXT NOT NULL,
    access_scope        TEXT NOT NULL DEFAULT 'dept'
                        CHECK (access_scope IN ('private', 'dept', 'group', 'org')),
    allowed_group_ids   UUID[] NOT NULL DEFAULT ARRAY[]::UUID[],
    denied_group_ids    UUID[] NOT NULL DEFAULT ARRAY[]::UUID[],
    min_auth_level      SMALLINT NOT NULL DEFAULT 1 CHECK (min_auth_level BETWEEN 1 AND 10),
    arn                 TEXT NOT NULL UNIQUE,
    pipeline_config_id  UUID,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (org_id, slug),
    UNIQUE (org_id, id),
    FOREIGN KEY (org_id, dept_id) REFERENCES iam.departments(org_id, id),
    FOREIGN KEY (org_id, owner_principal_id) REFERENCES iam.principals(org_id, id)
);

CREATE TABLE corpus.documents (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id              UUID NOT NULL REFERENCES iam.orgs(id) ON DELETE CASCADE,
    bucket_id           UUID NOT NULL,
    dept_id             UUID NOT NULL,
    owner_principal_id  UUID NOT NULL,
    access_scope        TEXT NOT NULL DEFAULT 'private'
                        CHECK (access_scope IN ('private', 'dept', 'group', 'org')),
    allowed_group_ids   UUID[] NOT NULL DEFAULT ARRAY[]::UUID[],
    denied_group_ids    UUID[] NOT NULL DEFAULT ARRAY[]::UUID[],
    min_auth_level      SMALLINT NOT NULL DEFAULT 3 CHECK (min_auth_level BETWEEN 1 AND 10),
    arn                 TEXT NOT NULL UNIQUE,
    acl_scope_key       TEXT NOT NULL,
    title               TEXT NOT NULL,
    source_uri          TEXT,
    mime_type           TEXT,
    content_sha256      TEXT NOT NULL,
    sensitivity         TEXT NOT NULL DEFAULT 'normal'
                        CHECK (sensitivity IN ('share', 'private', 'zdr', 'normal', 'confidential', 'secret')),
    status              TEXT NOT NULL DEFAULT 'indexed'
                        CHECK (status IN ('pending', 'indexed', 'redacted', 'archived', 'failed')),
    valid_from          TIMESTAMPTZ NOT NULL DEFAULT now(),
    valid_to            TIMESTAMPTZ,
    revision            INT NOT NULL DEFAULT 1,
    metadata            JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (org_id, id),
    FOREIGN KEY (org_id, bucket_id) REFERENCES corpus.buckets(org_id, id) ON DELETE CASCADE,
    FOREIGN KEY (org_id, dept_id) REFERENCES iam.departments(org_id, id),
    FOREIGN KEY (org_id, owner_principal_id) REFERENCES iam.principals(org_id, id)
);

CREATE TABLE corpus.pages (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id              UUID NOT NULL REFERENCES iam.orgs(id) ON DELETE CASCADE,
    document_id         UUID NOT NULL,
    dept_id             UUID NOT NULL,
    owner_principal_id  UUID NOT NULL,
    access_scope        TEXT NOT NULL DEFAULT 'private'
                        CHECK (access_scope IN ('private', 'dept', 'group', 'org')),
    allowed_group_ids   UUID[] NOT NULL DEFAULT ARRAY[]::UUID[],
    denied_group_ids    UUID[] NOT NULL DEFAULT ARRAY[]::UUID[],
    min_auth_level      SMALLINT NOT NULL DEFAULT 3 CHECK (min_auth_level BETWEEN 1 AND 10),
    arn                 TEXT NOT NULL UNIQUE,
    acl_scope_key       TEXT NOT NULL,
    page_no             INT NOT NULL CHECK (page_no >= 1),
    raw_markdown        TEXT,
    redacted_markdown   TEXT,
    page_state          JSONB NOT NULL DEFAULT '{}'::jsonb,
    thumb_uri           TEXT,
    redacted_thumb_uri  TEXT,
    thumb_embedding     vector(128),
    content_sha256      TEXT,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (document_id, page_no),
    UNIQUE (org_id, id),
    UNIQUE (org_id, document_id, id),
    FOREIGN KEY (org_id, document_id) REFERENCES corpus.documents(org_id, id) ON DELETE CASCADE,
    FOREIGN KEY (org_id, dept_id) REFERENCES iam.departments(org_id, id),
    FOREIGN KEY (org_id, owner_principal_id) REFERENCES iam.principals(org_id, id)
);

CREATE TABLE corpus.page_figures (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id              UUID NOT NULL REFERENCES iam.orgs(id) ON DELETE CASCADE,
    page_id             UUID NOT NULL,
    document_id         UUID NOT NULL,
    dept_id             UUID NOT NULL,
    owner_principal_id  UUID NOT NULL,
    access_scope        TEXT NOT NULL DEFAULT 'private'
                        CHECK (access_scope IN ('private', 'dept', 'group', 'org')),
    allowed_group_ids   UUID[] NOT NULL DEFAULT ARRAY[]::UUID[],
    denied_group_ids    UUID[] NOT NULL DEFAULT ARRAY[]::UUID[],
    min_auth_level      SMALLINT NOT NULL DEFAULT 3 CHECK (min_auth_level BETWEEN 1 AND 10),
    arn                 TEXT NOT NULL UNIQUE,
    acl_scope_key       TEXT NOT NULL,
    figure_idx          INT NOT NULL CHECK (figure_idx >= 0),
    kind                TEXT NOT NULL DEFAULT 'image',
    caption             TEXT,
    bbox_norm_1000      INT[4],
    figure_uri          TEXT,
    redacted_figure_uri TEXT,
    figure_embedding    vector(128),
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (page_id, figure_idx),
    FOREIGN KEY (org_id, document_id) REFERENCES corpus.documents(org_id, id) ON DELETE CASCADE,
    FOREIGN KEY (org_id, document_id, page_id) REFERENCES corpus.pages(org_id, document_id, id) ON DELETE CASCADE,
    FOREIGN KEY (org_id, dept_id) REFERENCES iam.departments(org_id, id),
    FOREIGN KEY (org_id, owner_principal_id) REFERENCES iam.principals(org_id, id)
);

CREATE TABLE corpus.chunks (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id              UUID NOT NULL REFERENCES iam.orgs(id) ON DELETE CASCADE,
    document_id         UUID NOT NULL,
    page_id             UUID,
    dept_id             UUID NOT NULL,
    owner_principal_id  UUID NOT NULL,
    access_scope        TEXT NOT NULL DEFAULT 'private'
                        CHECK (access_scope IN ('private', 'dept', 'group', 'org')),
    allowed_group_ids   UUID[] NOT NULL DEFAULT ARRAY[]::UUID[],
    denied_group_ids    UUID[] NOT NULL DEFAULT ARRAY[]::UUID[],
    min_auth_level      SMALLINT NOT NULL DEFAULT 3 CHECK (min_auth_level BETWEEN 1 AND 10),
    arn                 TEXT NOT NULL UNIQUE,
    acl_scope_key       TEXT NOT NULL,
    chunk_idx           INT NOT NULL CHECK (chunk_idx >= 0),
    redacted_text       TEXT NOT NULL,
    bm25_text           TEXT NOT NULL,
    table_html          TEXT,
    bbox_norm_1000      INT[4],
    text_hash           TEXT NOT NULL,
    embedding           vector(128),
    embedding_model     TEXT,
    valid_from          TIMESTAMPTZ NOT NULL DEFAULT now(),
    valid_to            TIMESTAMPTZ,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (document_id, chunk_idx),
    UNIQUE (org_id, id),
    FOREIGN KEY (org_id, document_id) REFERENCES corpus.documents(org_id, id) ON DELETE CASCADE,
    FOREIGN KEY (org_id, document_id, page_id) REFERENCES corpus.pages(org_id, document_id, id) ON DELETE SET NULL (page_id),
    FOREIGN KEY (org_id, dept_id) REFERENCES iam.departments(org_id, id),
    FOREIGN KEY (org_id, owner_principal_id) REFERENCES iam.principals(org_id, id)
);

-- KoParaRel 6-label edges between chunks
CREATE TABLE corpus.chunk_edges (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id          UUID NOT NULL REFERENCES iam.orgs(id) ON DELETE CASCADE,
    src_chunk_id    UUID NOT NULL,
    dst_chunk_id    UUID NOT NULL,
    relation_type   TEXT NOT NULL CHECK (relation_type IN ('equivalent', 'entails', 'contradicts', 'supersedes', 'refines', 'related')),
    confidence      REAL NOT NULL DEFAULT 1.0,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    FOREIGN KEY (org_id, src_chunk_id) REFERENCES corpus.chunks(org_id, id) ON DELETE CASCADE,
    FOREIGN KEY (org_id, dst_chunk_id) REFERENCES corpus.chunks(org_id, id) ON DELETE CASCADE
);

CREATE INDEX idx_corpus_docs_org_bucket ON corpus.documents (org_id, bucket_id, status);
CREATE INDEX idx_corpus_chunks_org_doc ON corpus.chunks (org_id, document_id, chunk_idx);
CREATE INDEX idx_corpus_chunks_acl_scope ON corpus.chunks (org_id, acl_scope_key);

-- Capability-guarded Vector & BM25 indexes
DO $$
BEGIN
    IF vectors.has_pg_pinecone() THEN
        BEGIN
            PERFORM pinecone.create_index('corpus.chunks', 'embedding', engine => 'native', filter_column => 'acl_scope_key');
            PERFORM pinecone.create_index('corpus.pages', 'thumb_embedding', engine => 'native', filter_column => 'acl_scope_key');
            PERFORM pinecone.create_index('corpus.page_figures', 'figure_embedding', engine => 'native', filter_column => 'acl_scope_key');
        EXCEPTION WHEN OTHERS THEN
            RAISE NOTICE 'pinecone.create_index failed (%); creating pgvector HNSW fallback', SQLERRM;
            CREATE INDEX IF NOT EXISTS idx_corpus_chunks_vec_hnsw ON corpus.chunks USING hnsw (embedding vector_cosine_ops);
            CREATE INDEX IF NOT EXISTS idx_corpus_pages_thumb_hnsw ON corpus.pages USING hnsw (thumb_embedding vector_cosine_ops);
            CREATE INDEX IF NOT EXISTS idx_corpus_figures_vec_hnsw ON corpus.page_figures USING hnsw (figure_embedding vector_cosine_ops);
        END;
    ELSE
        CREATE INDEX IF NOT EXISTS idx_corpus_chunks_vec_hnsw ON corpus.chunks USING hnsw (embedding vector_cosine_ops);
        CREATE INDEX IF NOT EXISTS idx_corpus_pages_thumb_hnsw ON corpus.pages USING hnsw (thumb_embedding vector_cosine_ops);
        CREATE INDEX IF NOT EXISTS idx_corpus_figures_vec_hnsw ON corpus.page_figures USING hnsw (figure_embedding vector_cosine_ops);
    END IF;
END;
$$;

DO $$
BEGIN
    IF vectors.has_pg_search() THEN
        BEGIN
            EXECUTE 'CREATE INDEX IF NOT EXISTS idx_corpus_chunks_bm25 ON corpus.chunks USING bm25 (id, (bm25_text::pdb.lindera(korean)))';
        EXCEPTION WHEN OTHERS THEN
            RAISE NOTICE 'pg_search bm25 index failed (%); creating pg_trgm + tsvector fallback', SQLERRM;
            CREATE INDEX IF NOT EXISTS idx_corpus_chunks_bm25_trgm ON corpus.chunks USING gin (bm25_text gin_trgm_ops);
            CREATE INDEX IF NOT EXISTS idx_corpus_chunks_bm25_tsv ON corpus.chunks USING gin (to_tsvector('simple', bm25_text));
        END;
    ELSE
        CREATE INDEX IF NOT EXISTS idx_corpus_chunks_bm25_trgm ON corpus.chunks USING gin (bm25_text gin_trgm_ops);
        CREATE INDEX IF NOT EXISTS idx_corpus_chunks_bm25_tsv ON corpus.chunks USING gin (to_tsvector('simple', bm25_text));
    END IF;
END;
$$;

-- Fail-Closed RLS (ENABLE + FORCE on all corpus tables)
ALTER TABLE corpus.buckets      ENABLE ROW LEVEL SECURITY;
ALTER TABLE corpus.buckets      FORCE ROW LEVEL SECURITY;
ALTER TABLE corpus.documents    ENABLE ROW LEVEL SECURITY;
ALTER TABLE corpus.documents    FORCE ROW LEVEL SECURITY;
ALTER TABLE corpus.pages        ENABLE ROW LEVEL SECURITY;
ALTER TABLE corpus.pages        FORCE ROW LEVEL SECURITY;
ALTER TABLE corpus.page_figures ENABLE ROW LEVEL SECURITY;
ALTER TABLE corpus.page_figures FORCE ROW LEVEL SECURITY;
ALTER TABLE corpus.chunks       ENABLE ROW LEVEL SECURITY;
ALTER TABLE corpus.chunks       FORCE ROW LEVEL SECURITY;
ALTER TABLE corpus.chunk_edges  ENABLE ROW LEVEL SECURITY;
ALTER TABLE corpus.chunk_edges  FORCE ROW LEVEL SECURITY;

CREATE POLICY rls_buckets_read ON corpus.buckets FOR SELECT
    USING (iam.rls_can_read(org_id, dept_id, owner_principal_id, access_scope, allowed_group_ids, arn, min_auth_level, denied_group_ids, 'bucket', id));
CREATE POLICY rls_buckets_insert ON corpus.buckets FOR INSERT
    WITH CHECK (
        iam.rls_can_insert(org_id, dept_id, owner_principal_id, arn, min_auth_level)
        AND (
            corpus.buckets.pipeline_config_id IS NULL
            OR iam.pipeline_config_belongs_to_org(corpus.buckets.org_id, corpus.buckets.pipeline_config_id)
        )
    );
CREATE POLICY rls_buckets_update ON corpus.buckets FOR UPDATE
    USING (iam.rls_can_access(org_id, dept_id, owner_principal_id, access_scope, allowed_group_ids, arn, min_auth_level, denied_group_ids, 'write', 'bucket', id))
    WITH CHECK (
        iam.rls_can_access(org_id, dept_id, owner_principal_id, access_scope, allowed_group_ids, arn, min_auth_level, denied_group_ids, 'write', 'bucket', id)
        AND (
            corpus.buckets.pipeline_config_id IS NULL
            OR iam.pipeline_config_belongs_to_org(corpus.buckets.org_id, corpus.buckets.pipeline_config_id)
        )
    );
CREATE POLICY rls_buckets_delete ON corpus.buckets FOR DELETE
    USING (iam.rls_can_access(org_id, dept_id, owner_principal_id, access_scope, allowed_group_ids, arn, min_auth_level, denied_group_ids, 'write', 'bucket', id));

CREATE POLICY rls_documents_read ON corpus.documents FOR SELECT
    USING (iam.rls_can_read(org_id, dept_id, owner_principal_id, access_scope, allowed_group_ids, arn, min_auth_level, denied_group_ids, 'document', id));
CREATE POLICY rls_documents_insert ON corpus.documents FOR INSERT
    WITH CHECK (
        iam.rls_can_insert(org_id, dept_id, owner_principal_id, arn, min_auth_level)
        AND EXISTS (
            SELECT 1 FROM corpus.buckets AS b
            WHERE b.id = corpus.documents.bucket_id
              AND b.org_id = corpus.documents.org_id
              AND iam.rls_can_access(b.org_id, b.dept_id, b.owner_principal_id, b.access_scope, b.allowed_group_ids, b.arn, b.min_auth_level, b.denied_group_ids, 'write', 'bucket', b.id)
        )
    );
CREATE POLICY rls_documents_update ON corpus.documents FOR UPDATE
    USING (iam.rls_can_access(org_id, dept_id, owner_principal_id, access_scope, allowed_group_ids, arn, min_auth_level, denied_group_ids, 'write', 'document', id))
    WITH CHECK (
        iam.rls_can_access(org_id, dept_id, owner_principal_id, access_scope, allowed_group_ids, arn, min_auth_level, denied_group_ids, 'write', 'document', id)
        AND EXISTS (
            SELECT 1 FROM corpus.buckets AS b
            WHERE b.id = corpus.documents.bucket_id
              AND b.org_id = corpus.documents.org_id
              AND iam.rls_can_access(b.org_id, b.dept_id, b.owner_principal_id, b.access_scope, b.allowed_group_ids, b.arn, b.min_auth_level, b.denied_group_ids, 'write', 'bucket', b.id)
        )
    );
CREATE POLICY rls_documents_delete ON corpus.documents FOR DELETE
    USING (iam.rls_can_access(org_id, dept_id, owner_principal_id, access_scope, allowed_group_ids, arn, min_auth_level, denied_group_ids, 'write', 'document', id));

CREATE POLICY rls_pages_read ON corpus.pages FOR SELECT
    USING (iam.rls_can_read(org_id, dept_id, owner_principal_id, access_scope, allowed_group_ids, arn, min_auth_level, denied_group_ids, 'page', id));
CREATE POLICY rls_pages_insert ON corpus.pages FOR INSERT
    WITH CHECK (
        iam.rls_can_insert(org_id, dept_id, owner_principal_id, arn, min_auth_level)
        AND EXISTS (
            SELECT 1 FROM corpus.documents AS d
            WHERE d.id = corpus.pages.document_id
              AND d.org_id = corpus.pages.org_id
              AND iam.rls_can_access(d.org_id, d.dept_id, d.owner_principal_id, d.access_scope, d.allowed_group_ids, d.arn, d.min_auth_level, d.denied_group_ids, 'write', 'document', d.id)
        )
    );
CREATE POLICY rls_pages_update ON corpus.pages FOR UPDATE
    USING (iam.rls_can_access(org_id, dept_id, owner_principal_id, access_scope, allowed_group_ids, arn, min_auth_level, denied_group_ids, 'write', 'page', id))
    WITH CHECK (
        iam.rls_can_access(org_id, dept_id, owner_principal_id, access_scope, allowed_group_ids, arn, min_auth_level, denied_group_ids, 'write', 'page', id)
        AND EXISTS (
            SELECT 1 FROM corpus.documents AS d
            WHERE d.id = corpus.pages.document_id
              AND d.org_id = corpus.pages.org_id
              AND iam.rls_can_access(d.org_id, d.dept_id, d.owner_principal_id, d.access_scope, d.allowed_group_ids, d.arn, d.min_auth_level, d.denied_group_ids, 'write', 'document', d.id)
        )
    );
CREATE POLICY rls_pages_delete ON corpus.pages FOR DELETE
    USING (iam.rls_can_access(org_id, dept_id, owner_principal_id, access_scope, allowed_group_ids, arn, min_auth_level, denied_group_ids, 'write', 'page', id));

CREATE POLICY rls_page_figures_read ON corpus.page_figures FOR SELECT
    USING (iam.rls_can_read(org_id, dept_id, owner_principal_id, access_scope, allowed_group_ids, arn, min_auth_level, denied_group_ids, 'figure', id));
CREATE POLICY rls_page_figures_insert ON corpus.page_figures FOR INSERT
    WITH CHECK (
        iam.rls_can_insert(org_id, dept_id, owner_principal_id, arn, min_auth_level)
        AND EXISTS (
            SELECT 1 FROM corpus.documents AS d
            WHERE d.id = corpus.page_figures.document_id
              AND d.org_id = corpus.page_figures.org_id
              AND iam.rls_can_access(d.org_id, d.dept_id, d.owner_principal_id, d.access_scope, d.allowed_group_ids, d.arn, d.min_auth_level, d.denied_group_ids, 'write', 'document', d.id)
        )
        AND EXISTS (
            SELECT 1 FROM corpus.pages AS p
            WHERE p.id = corpus.page_figures.page_id
              AND p.org_id = corpus.page_figures.org_id
              AND p.document_id = corpus.page_figures.document_id
              AND iam.rls_can_access(p.org_id, p.dept_id, p.owner_principal_id, p.access_scope, p.allowed_group_ids, p.arn, p.min_auth_level, p.denied_group_ids, 'write', 'page', p.id)
        )
    );
CREATE POLICY rls_page_figures_update ON corpus.page_figures FOR UPDATE
    USING (iam.rls_can_access(org_id, dept_id, owner_principal_id, access_scope, allowed_group_ids, arn, min_auth_level, denied_group_ids, 'write', 'figure', id))
    WITH CHECK (
        iam.rls_can_access(org_id, dept_id, owner_principal_id, access_scope, allowed_group_ids, arn, min_auth_level, denied_group_ids, 'write', 'figure', id)
        AND EXISTS (
            SELECT 1 FROM corpus.documents AS d
            WHERE d.id = corpus.page_figures.document_id
              AND d.org_id = corpus.page_figures.org_id
              AND iam.rls_can_access(d.org_id, d.dept_id, d.owner_principal_id, d.access_scope, d.allowed_group_ids, d.arn, d.min_auth_level, d.denied_group_ids, 'write', 'document', d.id)
        )
        AND EXISTS (
            SELECT 1 FROM corpus.pages AS p
            WHERE p.id = corpus.page_figures.page_id
              AND p.org_id = corpus.page_figures.org_id
              AND p.document_id = corpus.page_figures.document_id
              AND iam.rls_can_access(p.org_id, p.dept_id, p.owner_principal_id, p.access_scope, p.allowed_group_ids, p.arn, p.min_auth_level, p.denied_group_ids, 'write', 'page', p.id)
        )
    );
CREATE POLICY rls_page_figures_delete ON corpus.page_figures FOR DELETE
    USING (iam.rls_can_access(org_id, dept_id, owner_principal_id, access_scope, allowed_group_ids, arn, min_auth_level, denied_group_ids, 'write', 'figure', id));

CREATE POLICY rls_chunks_read ON corpus.chunks FOR SELECT
    USING (iam.rls_can_read(org_id, dept_id, owner_principal_id, access_scope, allowed_group_ids, arn, min_auth_level, denied_group_ids, 'chunk', id));
CREATE POLICY rls_chunks_insert ON corpus.chunks FOR INSERT
    WITH CHECK (
        iam.rls_can_insert(org_id, dept_id, owner_principal_id, arn, min_auth_level)
        AND EXISTS (
            SELECT 1 FROM corpus.documents AS d
            WHERE d.id = corpus.chunks.document_id
              AND d.org_id = corpus.chunks.org_id
              AND iam.rls_can_access(d.org_id, d.dept_id, d.owner_principal_id, d.access_scope, d.allowed_group_ids, d.arn, d.min_auth_level, d.denied_group_ids, 'write', 'document', d.id)
        )
        AND (
            corpus.chunks.page_id IS NULL
            OR EXISTS (
                SELECT 1 FROM corpus.pages AS p
                WHERE p.id = corpus.chunks.page_id
                  AND p.org_id = corpus.chunks.org_id
                  AND p.document_id = corpus.chunks.document_id
                  AND iam.rls_can_access(p.org_id, p.dept_id, p.owner_principal_id, p.access_scope, p.allowed_group_ids, p.arn, p.min_auth_level, p.denied_group_ids, 'write', 'page', p.id)
            )
        )
    );
CREATE POLICY rls_chunks_update ON corpus.chunks FOR UPDATE
    USING (iam.rls_can_access(org_id, dept_id, owner_principal_id, access_scope, allowed_group_ids, arn, min_auth_level, denied_group_ids, 'write', 'chunk', id))
    WITH CHECK (
        iam.rls_can_access(org_id, dept_id, owner_principal_id, access_scope, allowed_group_ids, arn, min_auth_level, denied_group_ids, 'write', 'chunk', id)
        AND EXISTS (
            SELECT 1 FROM corpus.documents AS d
            WHERE d.id = corpus.chunks.document_id
              AND d.org_id = corpus.chunks.org_id
              AND iam.rls_can_access(d.org_id, d.dept_id, d.owner_principal_id, d.access_scope, d.allowed_group_ids, d.arn, d.min_auth_level, d.denied_group_ids, 'write', 'document', d.id)
        )
        AND (
            corpus.chunks.page_id IS NULL
            OR EXISTS (
                SELECT 1 FROM corpus.pages AS p
                WHERE p.id = corpus.chunks.page_id
                  AND p.org_id = corpus.chunks.org_id
                  AND p.document_id = corpus.chunks.document_id
                  AND iam.rls_can_access(p.org_id, p.dept_id, p.owner_principal_id, p.access_scope, p.allowed_group_ids, p.arn, p.min_auth_level, p.denied_group_ids, 'write', 'page', p.id)
            )
        )
    );
CREATE POLICY rls_chunks_delete ON corpus.chunks FOR DELETE
    USING (iam.rls_can_access(org_id, dept_id, owner_principal_id, access_scope, allowed_group_ids, arn, min_auth_level, denied_group_ids, 'write', 'chunk', id));

CREATE POLICY rls_chunk_edges_read ON corpus.chunk_edges FOR SELECT
    USING (
        corpus.chunk_edges.org_id = iam.safe_uuid(current_setting('app.org_id', true))
        AND EXISTS (
            SELECT 1 FROM corpus.chunks AS sc
            WHERE sc.id = corpus.chunk_edges.src_chunk_id
              AND sc.org_id = corpus.chunk_edges.org_id
              AND iam.rls_can_read(sc.org_id, sc.dept_id, sc.owner_principal_id, sc.access_scope, sc.allowed_group_ids, sc.arn, sc.min_auth_level, sc.denied_group_ids, 'chunk', sc.id)
        )
        AND EXISTS (
            SELECT 1 FROM corpus.chunks AS dc
            WHERE dc.id = corpus.chunk_edges.dst_chunk_id
              AND dc.org_id = corpus.chunk_edges.org_id
              AND iam.rls_can_read(dc.org_id, dc.dept_id, dc.owner_principal_id, dc.access_scope, dc.allowed_group_ids, dc.arn, dc.min_auth_level, dc.denied_group_ids, 'chunk', dc.id)
        )
    );
CREATE POLICY rls_chunk_edges_insert ON corpus.chunk_edges FOR INSERT
    WITH CHECK (
        corpus.chunk_edges.org_id = iam.safe_uuid(current_setting('app.org_id', true))
        AND EXISTS (
            SELECT 1 FROM corpus.chunks AS sc
            WHERE sc.id = corpus.chunk_edges.src_chunk_id
              AND sc.org_id = corpus.chunk_edges.org_id
              AND iam.rls_can_access(sc.org_id, sc.dept_id, sc.owner_principal_id, sc.access_scope, sc.allowed_group_ids, sc.arn, sc.min_auth_level, sc.denied_group_ids, 'write', 'chunk', sc.id)
        )
        AND EXISTS (
            SELECT 1 FROM corpus.chunks AS dc
            WHERE dc.id = corpus.chunk_edges.dst_chunk_id
              AND dc.org_id = corpus.chunk_edges.org_id
              AND iam.rls_can_read(dc.org_id, dc.dept_id, dc.owner_principal_id, dc.access_scope, dc.allowed_group_ids, dc.arn, dc.min_auth_level, dc.denied_group_ids, 'chunk', dc.id)
        )
    );
CREATE POLICY rls_chunk_edges_update ON corpus.chunk_edges FOR UPDATE
    USING (
        corpus.chunk_edges.org_id = iam.safe_uuid(current_setting('app.org_id', true))
        AND EXISTS (
            SELECT 1 FROM corpus.chunks AS sc
            WHERE sc.id = corpus.chunk_edges.src_chunk_id
              AND sc.org_id = corpus.chunk_edges.org_id
              AND iam.rls_can_access(sc.org_id, sc.dept_id, sc.owner_principal_id, sc.access_scope, sc.allowed_group_ids, sc.arn, sc.min_auth_level, sc.denied_group_ids, 'write', 'chunk', sc.id)
        )
    )
    WITH CHECK (
        corpus.chunk_edges.org_id = iam.safe_uuid(current_setting('app.org_id', true))
        AND EXISTS (
            SELECT 1 FROM corpus.chunks AS sc
            WHERE sc.id = corpus.chunk_edges.src_chunk_id
              AND sc.org_id = corpus.chunk_edges.org_id
              AND iam.rls_can_access(sc.org_id, sc.dept_id, sc.owner_principal_id, sc.access_scope, sc.allowed_group_ids, sc.arn, sc.min_auth_level, sc.denied_group_ids, 'write', 'chunk', sc.id)
        )
        AND EXISTS (
            SELECT 1 FROM corpus.chunks AS dc
            WHERE dc.id = corpus.chunk_edges.dst_chunk_id
              AND dc.org_id = corpus.chunk_edges.org_id
              AND iam.rls_can_read(dc.org_id, dc.dept_id, dc.owner_principal_id, dc.access_scope, dc.allowed_group_ids, dc.arn, dc.min_auth_level, dc.denied_group_ids, 'chunk', dc.id)
        )
    );
CREATE POLICY rls_chunk_edges_delete ON corpus.chunk_edges FOR DELETE
    USING (
        corpus.chunk_edges.org_id = iam.safe_uuid(current_setting('app.org_id', true))
        AND EXISTS (
            SELECT 1 FROM corpus.chunks AS sc
            WHERE sc.id = corpus.chunk_edges.src_chunk_id
              AND sc.org_id = corpus.chunk_edges.org_id
              AND iam.rls_can_access(sc.org_id, sc.dept_id, sc.owner_principal_id, sc.access_scope, sc.allowed_group_ids, sc.arn, sc.min_auth_level, sc.denied_group_ids, 'write', 'chunk', sc.id)
        )
    );

-- Prevent non-admin reassignment of org_id, dept_id, or owner_principal_id
CREATE TRIGGER trg_corpus_buckets_ownership
    BEFORE UPDATE ON corpus.buckets
    FOR EACH ROW EXECUTE FUNCTION iam.prevent_non_admin_ownership_reassignment();
CREATE TRIGGER trg_corpus_documents_ownership
    BEFORE UPDATE ON corpus.documents
    FOR EACH ROW EXECUTE FUNCTION iam.prevent_non_admin_ownership_reassignment();
CREATE TRIGGER trg_corpus_pages_ownership
    BEFORE UPDATE ON corpus.pages
    FOR EACH ROW EXECUTE FUNCTION iam.prevent_non_admin_ownership_reassignment();
CREATE TRIGGER trg_corpus_page_figures_ownership
    BEFORE UPDATE ON corpus.page_figures
    FOR EACH ROW EXECUTE FUNCTION iam.prevent_non_admin_ownership_reassignment();
CREATE TRIGGER trg_corpus_chunks_ownership
    BEFORE UPDATE ON corpus.chunks
    FOR EACH ROW EXECUTE FUNCTION iam.prevent_non_admin_ownership_reassignment();
