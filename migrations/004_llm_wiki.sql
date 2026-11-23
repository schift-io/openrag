-- OpenRAG Migration 004: DB-Native LLM Wiki + KoParaRel Edges + Obsidian Sync + RLS

CREATE TABLE wiki.pages (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id              UUID NOT NULL REFERENCES iam.orgs(id) ON DELETE CASCADE,
    bucket_id           UUID,
    dept_id             UUID NOT NULL,
    owner_principal_id  UUID NOT NULL,
    access_scope        TEXT NOT NULL DEFAULT 'dept'
                        CHECK (access_scope IN ('private', 'dept', 'group', 'org')),
    allowed_group_ids   UUID[] NOT NULL DEFAULT ARRAY[]::UUID[],
    denied_group_ids    UUID[] NOT NULL DEFAULT ARRAY[]::UUID[],
    min_auth_level      SMALLINT NOT NULL DEFAULT 1 CHECK (min_auth_level BETWEEN 1 AND 10),
    arn                 TEXT NOT NULL UNIQUE,
    acl_scope_key       TEXT NOT NULL,
    slug                TEXT NOT NULL,
    title               TEXT NOT NULL,
    body_md             TEXT NOT NULL,
    bm25_text           TEXT NOT NULL,
    body_sha256         TEXT NOT NULL,
    embedding           vector(128),
    status              TEXT NOT NULL DEFAULT 'active'
                        CHECK (status IN ('active', 'superseded', 'archived', 'redacted')),
    version             INT NOT NULL DEFAULT 1,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (org_id, bucket_id, slug),
    UNIQUE (org_id, id),
    FOREIGN KEY (org_id, bucket_id) REFERENCES corpus.buckets(org_id, id) ON DELETE SET NULL (bucket_id),
    FOREIGN KEY (org_id, dept_id) REFERENCES iam.departments(org_id, id),
    FOREIGN KEY (org_id, owner_principal_id) REFERENCES iam.principals(org_id, id)
);

CREATE TABLE wiki.revisions (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id      UUID NOT NULL REFERENCES iam.orgs(id) ON DELETE CASCADE,
    page_id     UUID NOT NULL,
    version     INT NOT NULL CHECK (version >= 1),
    body_md     TEXT NOT NULL,
    body_sha256 TEXT NOT NULL,
    editor_type TEXT NOT NULL CHECK (editor_type IN ('human', 'agent', 'compiler', 'redactor', 'sync')),
    editor_id   TEXT NOT NULL,
    change_note TEXT,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (page_id, version),
    FOREIGN KEY (org_id, page_id) REFERENCES wiki.pages(org_id, id) ON DELETE CASCADE
);

CREATE TABLE wiki.edges (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id          UUID NOT NULL REFERENCES iam.orgs(id) ON DELETE CASCADE,
    src_page_id     UUID NOT NULL,
    dst_page_id     UUID NOT NULL,
    relation_type   TEXT NOT NULL CHECK (relation_type IN (
        'equivalent', 'entails', 'contradicts', 'supersedes', 'refines', 'related', 'references', 'parent_of'
    )),
    confidence      REAL NOT NULL DEFAULT 1.0,
    resolved_at     TIMESTAMPTZ,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    FOREIGN KEY (org_id, src_page_id) REFERENCES wiki.pages(org_id, id) ON DELETE CASCADE,
    FOREIGN KEY (org_id, dst_page_id) REFERENCES wiki.pages(org_id, id) ON DELETE CASCADE
);

CREATE TABLE wiki.page_sources (
    org_id      UUID NOT NULL REFERENCES iam.orgs(id) ON DELETE CASCADE,
    page_id     UUID NOT NULL,
    source_type TEXT NOT NULL CHECK (source_type IN ('chunk', 'memory', 'document')),
    source_id   UUID NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (page_id, source_type, source_id),
    FOREIGN KEY (org_id, page_id) REFERENCES wiki.pages(org_id, id) ON DELETE CASCADE
);

CREATE TABLE wiki.sync_changes (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id      UUID NOT NULL REFERENCES iam.orgs(id) ON DELETE CASCADE,
    bucket_id   UUID,
    page_id     UUID,
    file_path   TEXT NOT NULL,
    direction   TEXT NOT NULL CHECK (direction IN ('import', 'export')),
    body_sha256 TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'synced' CHECK (status IN ('pending', 'synced', 'conflict', 'failed')),
    synced_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    FOREIGN KEY (org_id, bucket_id) REFERENCES corpus.buckets(org_id, id) ON DELETE SET NULL (bucket_id),
    FOREIGN KEY (org_id, page_id) REFERENCES wiki.pages(org_id, id) ON DELETE SET NULL (page_id)
);

CREATE INDEX idx_wiki_pages_active ON wiki.pages (org_id, dept_id, status) WHERE status = 'active';
CREATE INDEX idx_wiki_edges_src ON wiki.edges (org_id, src_page_id, relation_type);

-- Capability-guarded Vector & BM25 indexes
DO $$
BEGIN
    IF vectors.has_pg_pinecone() THEN
        BEGIN
            PERFORM pinecone.create_index('wiki.pages', 'embedding', engine => 'native', filter_column => 'acl_scope_key');
        EXCEPTION WHEN OTHERS THEN
            RAISE NOTICE 'pinecone.create_index on wiki.pages failed (%); creating pgvector HNSW fallback', SQLERRM;
            CREATE INDEX IF NOT EXISTS idx_wiki_pages_vec_hnsw ON wiki.pages USING hnsw (embedding vector_cosine_ops);
        END;
    ELSE
        CREATE INDEX IF NOT EXISTS idx_wiki_pages_vec_hnsw ON wiki.pages USING hnsw (embedding vector_cosine_ops);
    END IF;
END;
$$;

DO $$
BEGIN
    IF vectors.has_pg_search() THEN
        BEGIN
            EXECUTE 'CREATE INDEX IF NOT EXISTS idx_wiki_pages_bm25 ON wiki.pages USING bm25 (id, (bm25_text::pdb.lindera(korean)))';
        EXCEPTION WHEN OTHERS THEN
            RAISE NOTICE 'pg_search bm25 index on wiki.pages failed (%); creating pg_trgm + tsvector fallback', SQLERRM;
            CREATE INDEX IF NOT EXISTS idx_wiki_pages_bm25_trgm ON wiki.pages USING gin (bm25_text gin_trgm_ops);
            CREATE INDEX IF NOT EXISTS idx_wiki_pages_bm25_tsv ON wiki.pages USING gin (to_tsvector('simple', bm25_text));
        END;
    ELSE
        CREATE INDEX IF NOT EXISTS idx_wiki_pages_bm25_trgm ON wiki.pages USING gin (bm25_text gin_trgm_ops);
        CREATE INDEX IF NOT EXISTS idx_wiki_pages_bm25_tsv ON wiki.pages USING gin (to_tsvector('simple', bm25_text));
    END IF;
END;
$$;

-- Fail-Closed RLS on wiki tables
ALTER TABLE wiki.pages        ENABLE ROW LEVEL SECURITY;
ALTER TABLE wiki.pages        FORCE ROW LEVEL SECURITY;
ALTER TABLE wiki.revisions    ENABLE ROW LEVEL SECURITY;
ALTER TABLE wiki.revisions    FORCE ROW LEVEL SECURITY;
ALTER TABLE wiki.edges        ENABLE ROW LEVEL SECURITY;
ALTER TABLE wiki.edges        FORCE ROW LEVEL SECURITY;
ALTER TABLE wiki.page_sources ENABLE ROW LEVEL SECURITY;
ALTER TABLE wiki.page_sources FORCE ROW LEVEL SECURITY;
ALTER TABLE wiki.sync_changes ENABLE ROW LEVEL SECURITY;
ALTER TABLE wiki.sync_changes FORCE ROW LEVEL SECURITY;

CREATE POLICY rls_wiki_read ON wiki.pages FOR SELECT
    USING (status = 'active' AND iam.rls_can_read(org_id, dept_id, owner_principal_id, access_scope, allowed_group_ids, arn, min_auth_level, denied_group_ids, 'wiki', id));
CREATE POLICY rls_wiki_insert ON wiki.pages FOR INSERT
    WITH CHECK (
        iam.rls_can_insert(org_id, dept_id, owner_principal_id, arn, min_auth_level)
        AND (
            wiki.pages.bucket_id IS NULL
            OR EXISTS (
                SELECT 1 FROM corpus.buckets AS b
                WHERE b.id = wiki.pages.bucket_id
                  AND b.org_id = wiki.pages.org_id
                  AND iam.rls_can_access(b.org_id, b.dept_id, b.owner_principal_id, b.access_scope, b.allowed_group_ids, b.arn, b.min_auth_level, b.denied_group_ids, 'write', 'bucket', b.id)
            )
        )
    );
CREATE POLICY rls_wiki_update ON wiki.pages FOR UPDATE
    USING (iam.rls_can_access(org_id, dept_id, owner_principal_id, access_scope, allowed_group_ids, arn, min_auth_level, denied_group_ids, 'write', 'wiki', id))
    WITH CHECK (
        iam.rls_can_access(org_id, dept_id, owner_principal_id, access_scope, allowed_group_ids, arn, min_auth_level, denied_group_ids, 'write', 'wiki', id)
        AND (
            wiki.pages.bucket_id IS NULL
            OR EXISTS (
                SELECT 1 FROM corpus.buckets AS b
                WHERE b.id = wiki.pages.bucket_id
                  AND b.org_id = wiki.pages.org_id
                  AND iam.rls_can_access(b.org_id, b.dept_id, b.owner_principal_id, b.access_scope, b.allowed_group_ids, b.arn, b.min_auth_level, b.denied_group_ids, 'write', 'bucket', b.id)
            )
        )
    );
CREATE POLICY rls_wiki_delete ON wiki.pages FOR DELETE
    USING (iam.rls_can_access(org_id, dept_id, owner_principal_id, access_scope, allowed_group_ids, arn, min_auth_level, denied_group_ids, 'write', 'wiki', id));

CREATE TRIGGER trg_wiki_pages_ownership
    BEFORE UPDATE ON wiki.pages
    FOR EACH ROW EXECUTE FUNCTION iam.prevent_non_admin_ownership_reassignment();

CREATE POLICY rls_wiki_revisions_read ON wiki.revisions FOR SELECT
    USING (
        wiki.revisions.org_id = iam.safe_uuid(current_setting('app.org_id', true))
        AND EXISTS (
            SELECT 1 FROM wiki.pages AS p
            WHERE p.id = wiki.revisions.page_id
              AND p.org_id = wiki.revisions.org_id
              AND p.status = 'active'
              AND iam.rls_can_read(p.org_id, p.dept_id, p.owner_principal_id, p.access_scope, p.allowed_group_ids, p.arn, p.min_auth_level, p.denied_group_ids, 'wiki', p.id)
        )
    );
CREATE POLICY rls_wiki_revisions_insert ON wiki.revisions FOR INSERT
    WITH CHECK (
        wiki.revisions.org_id = iam.safe_uuid(current_setting('app.org_id', true))
        AND EXISTS (
            SELECT 1 FROM wiki.pages AS p
            WHERE p.id = wiki.revisions.page_id
              AND p.org_id = wiki.revisions.org_id
              AND iam.rls_can_access(p.org_id, p.dept_id, p.owner_principal_id, p.access_scope, p.allowed_group_ids, p.arn, p.min_auth_level, p.denied_group_ids, 'write', 'wiki', p.id)
        )
    );
CREATE POLICY rls_wiki_revisions_update ON wiki.revisions FOR UPDATE
    USING (
        wiki.revisions.org_id = iam.safe_uuid(current_setting('app.org_id', true))
        AND EXISTS (
            SELECT 1 FROM wiki.pages AS p
            WHERE p.id = wiki.revisions.page_id
              AND p.org_id = wiki.revisions.org_id
              AND iam.rls_can_access(p.org_id, p.dept_id, p.owner_principal_id, p.access_scope, p.allowed_group_ids, p.arn, p.min_auth_level, p.denied_group_ids, 'write', 'wiki', p.id)
        )
    )
    WITH CHECK (
        wiki.revisions.org_id = iam.safe_uuid(current_setting('app.org_id', true))
        AND EXISTS (
            SELECT 1 FROM wiki.pages AS p
            WHERE p.id = wiki.revisions.page_id
              AND p.org_id = wiki.revisions.org_id
              AND iam.rls_can_access(p.org_id, p.dept_id, p.owner_principal_id, p.access_scope, p.allowed_group_ids, p.arn, p.min_auth_level, p.denied_group_ids, 'write', 'wiki', p.id)
        )
    );
CREATE POLICY rls_wiki_revisions_delete ON wiki.revisions FOR DELETE
    USING (
        wiki.revisions.org_id = iam.safe_uuid(current_setting('app.org_id', true))
        AND EXISTS (
            SELECT 1 FROM wiki.pages AS p
            WHERE p.id = wiki.revisions.page_id
              AND p.org_id = wiki.revisions.org_id
              AND iam.rls_can_access(p.org_id, p.dept_id, p.owner_principal_id, p.access_scope, p.allowed_group_ids, p.arn, p.min_auth_level, p.denied_group_ids, 'write', 'wiki', p.id)
        )
    );

CREATE POLICY rls_wiki_edges_read ON wiki.edges FOR SELECT
    USING (
        wiki.edges.org_id = iam.safe_uuid(current_setting('app.org_id', true))
        AND EXISTS (
            SELECT 1 FROM wiki.pages AS sp
            WHERE sp.id = wiki.edges.src_page_id
              AND sp.org_id = wiki.edges.org_id
              AND sp.status = 'active'
              AND iam.rls_can_read(sp.org_id, sp.dept_id, sp.owner_principal_id, sp.access_scope, sp.allowed_group_ids, sp.arn, sp.min_auth_level, sp.denied_group_ids, 'wiki', sp.id)
        )
        AND EXISTS (
            SELECT 1 FROM wiki.pages AS dp
            WHERE dp.id = wiki.edges.dst_page_id
              AND dp.org_id = wiki.edges.org_id
              AND dp.status = 'active'
              AND iam.rls_can_read(dp.org_id, dp.dept_id, dp.owner_principal_id, dp.access_scope, dp.allowed_group_ids, dp.arn, dp.min_auth_level, dp.denied_group_ids, 'wiki', dp.id)
        )
    );
CREATE POLICY rls_wiki_edges_insert ON wiki.edges FOR INSERT
    WITH CHECK (
        wiki.edges.org_id = iam.safe_uuid(current_setting('app.org_id', true))
        AND EXISTS (
            SELECT 1 FROM wiki.pages AS sp
            WHERE sp.id = wiki.edges.src_page_id
              AND sp.org_id = wiki.edges.org_id
              AND iam.rls_can_access(sp.org_id, sp.dept_id, sp.owner_principal_id, sp.access_scope, sp.allowed_group_ids, sp.arn, sp.min_auth_level, sp.denied_group_ids, 'write', 'wiki', sp.id)
        )
        AND EXISTS (
            SELECT 1 FROM wiki.pages AS dp
            WHERE dp.id = wiki.edges.dst_page_id
              AND dp.org_id = wiki.edges.org_id
              AND dp.status = 'active'
              AND iam.rls_can_read(dp.org_id, dp.dept_id, dp.owner_principal_id, dp.access_scope, dp.allowed_group_ids, dp.arn, dp.min_auth_level, dp.denied_group_ids, 'wiki', dp.id)
        )
    );
CREATE POLICY rls_wiki_edges_update ON wiki.edges FOR UPDATE
    USING (
        wiki.edges.org_id = iam.safe_uuid(current_setting('app.org_id', true))
        AND EXISTS (
            SELECT 1 FROM wiki.pages AS sp
            WHERE sp.id = wiki.edges.src_page_id
              AND sp.org_id = wiki.edges.org_id
              AND iam.rls_can_access(sp.org_id, sp.dept_id, sp.owner_principal_id, sp.access_scope, sp.allowed_group_ids, sp.arn, sp.min_auth_level, sp.denied_group_ids, 'write', 'wiki', sp.id)
        )
    )
    WITH CHECK (
        wiki.edges.org_id = iam.safe_uuid(current_setting('app.org_id', true))
        AND EXISTS (
            SELECT 1 FROM wiki.pages AS sp
            WHERE sp.id = wiki.edges.src_page_id
              AND sp.org_id = wiki.edges.org_id
              AND iam.rls_can_access(sp.org_id, sp.dept_id, sp.owner_principal_id, sp.access_scope, sp.allowed_group_ids, sp.arn, sp.min_auth_level, sp.denied_group_ids, 'write', 'wiki', sp.id)
        )
        AND EXISTS (
            SELECT 1 FROM wiki.pages AS dp
            WHERE dp.id = wiki.edges.dst_page_id
              AND dp.org_id = wiki.edges.org_id
              AND dp.status = 'active'
              AND iam.rls_can_read(dp.org_id, dp.dept_id, dp.owner_principal_id, dp.access_scope, dp.allowed_group_ids, dp.arn, dp.min_auth_level, dp.denied_group_ids, 'wiki', dp.id)
        )
    );
CREATE POLICY rls_wiki_edges_delete ON wiki.edges FOR DELETE
    USING (
        wiki.edges.org_id = iam.safe_uuid(current_setting('app.org_id', true))
        AND EXISTS (
            SELECT 1 FROM wiki.pages AS sp
            WHERE sp.id = wiki.edges.src_page_id
              AND sp.org_id = wiki.edges.org_id
              AND iam.rls_can_access(sp.org_id, sp.dept_id, sp.owner_principal_id, sp.access_scope, sp.allowed_group_ids, sp.arn, sp.min_auth_level, sp.denied_group_ids, 'write', 'wiki', sp.id)
        )
    );

-- Replaced in 007_memory_redaction_indexing.sql to also validate 'memory' against memory.entries
CREATE OR REPLACE FUNCTION iam.wiki_source_can_read(
    p_org_id UUID,
    p_source_type TEXT,
    p_source_id UUID
)
RETURNS BOOLEAN
LANGUAGE plpgsql
STABLE
SECURITY DEFINER
SET search_path = iam, corpus, wiki, pg_catalog
AS $$
BEGIN
    IF p_org_id IS NULL OR p_source_type IS NULL OR p_source_id IS NULL THEN
        RETURN false;
    END IF;

    IF p_source_type = 'document' THEN
        RETURN EXISTS (
            SELECT 1
            FROM corpus.documents AS d
            WHERE d.id = p_source_id
              AND d.org_id = p_org_id
              AND iam.rls_can_read(
                  d.org_id, d.dept_id, d.owner_principal_id, d.access_scope,
                  d.allowed_group_ids, d.arn, d.min_auth_level, d.denied_group_ids,
                  'document', d.id
              )
        );
    ELSIF p_source_type = 'chunk' THEN
        RETURN EXISTS (
            SELECT 1
            FROM corpus.chunks AS c
            WHERE c.id = p_source_id
              AND c.org_id = p_org_id
              AND iam.rls_can_read(
                  c.org_id, c.dept_id, c.owner_principal_id, c.access_scope,
                  c.allowed_group_ids, c.arn, c.min_auth_level, c.denied_group_ids,
                  'chunk', c.id
              )
        );
    END IF;

    RETURN false;
END;
$$;

CREATE POLICY rls_wiki_page_sources_read ON wiki.page_sources FOR SELECT
    USING (
        wiki.page_sources.org_id = iam.safe_uuid(current_setting('app.org_id', true))
        AND EXISTS (
            SELECT 1 FROM wiki.pages AS p
            WHERE p.id = wiki.page_sources.page_id
              AND p.org_id = wiki.page_sources.org_id
              AND p.status = 'active'
              AND iam.rls_can_read(p.org_id, p.dept_id, p.owner_principal_id, p.access_scope, p.allowed_group_ids, p.arn, p.min_auth_level, p.denied_group_ids, 'wiki', p.id)
        )
    );
CREATE POLICY rls_wiki_page_sources_insert ON wiki.page_sources FOR INSERT
    WITH CHECK (
        wiki.page_sources.org_id = iam.safe_uuid(current_setting('app.org_id', true))
        AND EXISTS (
            SELECT 1 FROM wiki.pages AS p
            WHERE p.id = wiki.page_sources.page_id
              AND p.org_id = wiki.page_sources.org_id
              AND iam.rls_can_access(p.org_id, p.dept_id, p.owner_principal_id, p.access_scope, p.allowed_group_ids, p.arn, p.min_auth_level, p.denied_group_ids, 'write', 'wiki', p.id)
        )
        AND iam.wiki_source_can_read(wiki.page_sources.org_id, wiki.page_sources.source_type, wiki.page_sources.source_id)
    );
CREATE POLICY rls_wiki_page_sources_update ON wiki.page_sources FOR UPDATE
    USING (
        wiki.page_sources.org_id = iam.safe_uuid(current_setting('app.org_id', true))
        AND EXISTS (
            SELECT 1 FROM wiki.pages AS p
            WHERE p.id = wiki.page_sources.page_id
              AND p.org_id = wiki.page_sources.org_id
              AND iam.rls_can_access(p.org_id, p.dept_id, p.owner_principal_id, p.access_scope, p.allowed_group_ids, p.arn, p.min_auth_level, p.denied_group_ids, 'write', 'wiki', p.id)
        )
    )
    WITH CHECK (
        wiki.page_sources.org_id = iam.safe_uuid(current_setting('app.org_id', true))
        AND EXISTS (
            SELECT 1 FROM wiki.pages AS p
            WHERE p.id = wiki.page_sources.page_id
              AND p.org_id = wiki.page_sources.org_id
              AND iam.rls_can_access(p.org_id, p.dept_id, p.owner_principal_id, p.access_scope, p.allowed_group_ids, p.arn, p.min_auth_level, p.denied_group_ids, 'write', 'wiki', p.id)
        )
        AND iam.wiki_source_can_read(wiki.page_sources.org_id, wiki.page_sources.source_type, wiki.page_sources.source_id)
    );
CREATE POLICY rls_wiki_page_sources_delete ON wiki.page_sources FOR DELETE
    USING (
        wiki.page_sources.org_id = iam.safe_uuid(current_setting('app.org_id', true))
        AND EXISTS (
            SELECT 1 FROM wiki.pages AS p
            WHERE p.id = wiki.page_sources.page_id
              AND p.org_id = wiki.page_sources.org_id
              AND iam.rls_can_access(p.org_id, p.dept_id, p.owner_principal_id, p.access_scope, p.allowed_group_ids, p.arn, p.min_auth_level, p.denied_group_ids, 'write', 'wiki', p.id)
        )
    );

CREATE POLICY rls_wiki_sync_changes_read ON wiki.sync_changes FOR SELECT
    USING (
        wiki.sync_changes.org_id = iam.safe_uuid(current_setting('app.org_id', true))
        AND (
            (wiki.sync_changes.bucket_id IS NOT NULL AND EXISTS (
                SELECT 1 FROM corpus.buckets AS b
                WHERE b.id = wiki.sync_changes.bucket_id
                  AND b.org_id = wiki.sync_changes.org_id
                  AND iam.rls_can_read(b.org_id, b.dept_id, b.owner_principal_id, b.access_scope, b.allowed_group_ids, b.arn, b.min_auth_level, b.denied_group_ids, 'bucket', b.id)
            ))
            OR (wiki.sync_changes.bucket_id IS NULL AND wiki.sync_changes.page_id IS NOT NULL)
        )
        AND (
            wiki.sync_changes.page_id IS NULL
            OR EXISTS (
                SELECT 1 FROM wiki.pages AS p
                WHERE p.id = wiki.sync_changes.page_id
                  AND p.org_id = wiki.sync_changes.org_id
                  AND p.status = 'active'
                  AND iam.rls_can_read(p.org_id, p.dept_id, p.owner_principal_id, p.access_scope, p.allowed_group_ids, p.arn, p.min_auth_level, p.denied_group_ids, 'wiki', p.id)
            )
        )
    );
CREATE POLICY rls_wiki_sync_changes_insert ON wiki.sync_changes FOR INSERT
    WITH CHECK (
        wiki.sync_changes.org_id = iam.safe_uuid(current_setting('app.org_id', true))
        AND (
            (wiki.sync_changes.bucket_id IS NOT NULL AND EXISTS (
                SELECT 1 FROM corpus.buckets AS b
                WHERE b.id = wiki.sync_changes.bucket_id
                  AND b.org_id = wiki.sync_changes.org_id
                  AND iam.rls_can_access(b.org_id, b.dept_id, b.owner_principal_id, b.access_scope, b.allowed_group_ids, b.arn, b.min_auth_level, b.denied_group_ids, 'write', 'bucket', b.id)
            ))
            OR (wiki.sync_changes.bucket_id IS NULL AND wiki.sync_changes.page_id IS NOT NULL)
        )
        AND (
            wiki.sync_changes.page_id IS NULL
            OR EXISTS (
                SELECT 1 FROM wiki.pages AS p
                WHERE p.id = wiki.sync_changes.page_id
                  AND p.org_id = wiki.sync_changes.org_id
                  AND iam.rls_can_access(p.org_id, p.dept_id, p.owner_principal_id, p.access_scope, p.allowed_group_ids, p.arn, p.min_auth_level, p.denied_group_ids, 'write', 'wiki', p.id)
            )
        )
    );
CREATE POLICY rls_wiki_sync_changes_update ON wiki.sync_changes FOR UPDATE
    USING (
        wiki.sync_changes.org_id = iam.safe_uuid(current_setting('app.org_id', true))
        AND (
            (wiki.sync_changes.bucket_id IS NOT NULL AND EXISTS (
                SELECT 1 FROM corpus.buckets AS b
                WHERE b.id = wiki.sync_changes.bucket_id
                  AND b.org_id = wiki.sync_changes.org_id
                  AND iam.rls_can_access(b.org_id, b.dept_id, b.owner_principal_id, b.access_scope, b.allowed_group_ids, b.arn, b.min_auth_level, b.denied_group_ids, 'write', 'bucket', b.id)
            ))
            OR (wiki.sync_changes.bucket_id IS NULL AND wiki.sync_changes.page_id IS NOT NULL)
        )
        AND (
            wiki.sync_changes.page_id IS NULL
            OR EXISTS (
                SELECT 1 FROM wiki.pages AS p
                WHERE p.id = wiki.sync_changes.page_id
                  AND p.org_id = wiki.sync_changes.org_id
                  AND iam.rls_can_access(p.org_id, p.dept_id, p.owner_principal_id, p.access_scope, p.allowed_group_ids, p.arn, p.min_auth_level, p.denied_group_ids, 'write', 'wiki', p.id)
            )
        )
    )
    WITH CHECK (
        wiki.sync_changes.org_id = iam.safe_uuid(current_setting('app.org_id', true))
        AND (
            (wiki.sync_changes.bucket_id IS NOT NULL AND EXISTS (
                SELECT 1 FROM corpus.buckets AS b
                WHERE b.id = wiki.sync_changes.bucket_id
                  AND b.org_id = wiki.sync_changes.org_id
                  AND iam.rls_can_access(b.org_id, b.dept_id, b.owner_principal_id, b.access_scope, b.allowed_group_ids, b.arn, b.min_auth_level, b.denied_group_ids, 'write', 'bucket', b.id)
            ))
            OR (wiki.sync_changes.bucket_id IS NULL AND wiki.sync_changes.page_id IS NOT NULL)
        )
        AND (
            wiki.sync_changes.page_id IS NULL
            OR EXISTS (
                SELECT 1 FROM wiki.pages AS p
                WHERE p.id = wiki.sync_changes.page_id
                  AND p.org_id = wiki.sync_changes.org_id
                  AND iam.rls_can_access(p.org_id, p.dept_id, p.owner_principal_id, p.access_scope, p.allowed_group_ids, p.arn, p.min_auth_level, p.denied_group_ids, 'write', 'wiki', p.id)
            )
        )
    );
CREATE POLICY rls_wiki_sync_changes_delete ON wiki.sync_changes FOR DELETE
    USING (
        wiki.sync_changes.org_id = iam.safe_uuid(current_setting('app.org_id', true))
        AND (
            (wiki.sync_changes.bucket_id IS NOT NULL AND EXISTS (
                SELECT 1 FROM corpus.buckets AS b
                WHERE b.id = wiki.sync_changes.bucket_id
                  AND b.org_id = wiki.sync_changes.org_id
                  AND iam.rls_can_access(b.org_id, b.dept_id, b.owner_principal_id, b.access_scope, b.allowed_group_ids, b.arn, b.min_auth_level, b.denied_group_ids, 'write', 'bucket', b.id)
            ))
            OR (wiki.sync_changes.bucket_id IS NULL AND wiki.sync_changes.page_id IS NOT NULL)
        )
        AND (
            wiki.sync_changes.page_id IS NULL
            OR EXISTS (
                SELECT 1 FROM wiki.pages AS p
                WHERE p.id = wiki.sync_changes.page_id
                  AND p.org_id = wiki.sync_changes.org_id
                  AND iam.rls_can_access(p.org_id, p.dept_id, p.owner_principal_id, p.access_scope, p.allowed_group_ids, p.arn, p.min_auth_level, p.denied_group_ids, 'write', 'wiki', p.id)
            )
        )
    );
