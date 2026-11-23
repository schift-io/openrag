-- OpenRAG Migration 007: CCLG Memory Patch Ledger, Cascade Redaction & Incremental Indexing Outbox

-- ============================================================================
-- A. CCLG Immutable Memory Episodes, Effective Entries & Patch Ledger (memory.*)
-- ============================================================================

CREATE TABLE memory.episodes (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id              UUID NOT NULL REFERENCES iam.orgs(id) ON DELETE CASCADE,
    dept_id             UUID NOT NULL,
    owner_principal_id  UUID NOT NULL,
    access_scope        TEXT NOT NULL DEFAULT 'private'
                        CHECK (access_scope IN ('private', 'dept', 'group', 'org')),
    allowed_group_ids   UUID[] NOT NULL DEFAULT ARRAY[]::UUID[],
    denied_group_ids    UUID[] NOT NULL DEFAULT ARRAY[]::UUID[],
    min_auth_level      SMALLINT NOT NULL DEFAULT 3 CHECK (min_auth_level BETWEEN 1 AND 10),
    arn                 TEXT NOT NULL UNIQUE,
    source_harness      TEXT NOT NULL CHECK (source_harness IN ('chat_ui', 'claude_code', 'codex', 'cursor', 'kordoc', 'docufinder')),
    session_key         TEXT,
    summary_md          TEXT NOT NULL,
    raw_redacted_json   JSONB,
    content_sha256      TEXT NOT NULL,
    redaction_applied   BOOLEAN NOT NULL DEFAULT true,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (org_id, id),
    FOREIGN KEY (org_id, dept_id) REFERENCES iam.departments(org_id, id),
    FOREIGN KEY (org_id, owner_principal_id) REFERENCES iam.principals(org_id, id)
);

CREATE TABLE memory.entries (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id              UUID NOT NULL REFERENCES iam.orgs(id) ON DELETE CASCADE,
    bucket_id           UUID,
    dept_id             UUID NOT NULL,
    owner_principal_id  UUID NOT NULL,
    access_scope        TEXT NOT NULL DEFAULT 'private'
                        CHECK (access_scope IN ('private', 'dept', 'group', 'org')),
    allowed_group_ids   UUID[] NOT NULL DEFAULT ARRAY[]::UUID[],
    denied_group_ids    UUID[] NOT NULL DEFAULT ARRAY[]::UUID[],
    min_auth_level      SMALLINT NOT NULL DEFAULT 3 CHECK (min_auth_level BETWEEN 1 AND 10),
    arn                 TEXT NOT NULL UNIQUE,
    acl_scope_key       TEXT NOT NULL,
    memory_kind         TEXT NOT NULL CHECK (memory_kind IN ('fact', 'decision', 'procedure', 'preference', 'correction', 'daily_log')),
    title               TEXT,
    effective_text      TEXT NOT NULL,
    bm25_text           TEXT NOT NULL,
    content_sha256      TEXT NOT NULL,
    embedding           vector(128),
    embedding_model     TEXT,
    source_episode_id   UUID,
    source_document_id  UUID,
    promoted_wiki_id    UUID,
    confidence          REAL NOT NULL DEFAULT 0.8,
    status              TEXT NOT NULL DEFAULT 'active'
                        CHECK (status IN ('active', 'superseded', 'retracted', 'redacted', 'archived')),
    superseded_by       UUID,
    version             INT NOT NULL DEFAULT 1,
    valid_from          TIMESTAMPTZ NOT NULL DEFAULT now(),
    valid_to            TIMESTAMPTZ,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (org_id, id),
    FOREIGN KEY (org_id, dept_id) REFERENCES iam.departments(org_id, id),
    FOREIGN KEY (org_id, owner_principal_id) REFERENCES iam.principals(org_id, id),
    FOREIGN KEY (org_id, bucket_id) REFERENCES corpus.buckets(org_id, id) ON DELETE SET NULL (bucket_id),
    FOREIGN KEY (org_id, source_episode_id) REFERENCES memory.episodes(org_id, id) ON DELETE SET NULL (source_episode_id),
    FOREIGN KEY (org_id, source_document_id) REFERENCES corpus.documents(org_id, id) ON DELETE SET NULL (source_document_id),
    FOREIGN KEY (org_id, promoted_wiki_id) REFERENCES wiki.pages(org_id, id) ON DELETE SET NULL (promoted_wiki_id),
    FOREIGN KEY (org_id, superseded_by) REFERENCES memory.entries(org_id, id) ON DELETE SET NULL (superseded_by)
);

CREATE INDEX idx_memory_entries_active ON memory.entries (org_id, dept_id, status) WHERE status = 'active';

CREATE TABLE memory.patches (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id          UUID NOT NULL REFERENCES iam.orgs(id) ON DELETE CASCADE,
    memory_entry_id UUID NOT NULL,
    patch_op        TEXT NOT NULL CHECK (patch_op IN ('amend', 'supersede', 'retract', 'redact', 'promote_scope', 'promote_wiki')),
    prev_version    INT NOT NULL,
    new_version     INT NOT NULL,
    prev_text       TEXT,
    new_text        TEXT,
    prev_scope      TEXT,
    new_scope       TEXT,
    reason          TEXT,
    actor_type      TEXT NOT NULL CHECK (actor_type IN ('human', 'agent', 'redactor', 'compiler')),
    actor_id        TEXT NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    FOREIGN KEY (org_id, memory_entry_id) REFERENCES memory.entries(org_id, id) ON DELETE CASCADE
);

-- Capability-guarded Vector & BM25 indexes on memory.entries
DO $$
BEGIN
    IF vectors.has_pg_pinecone() THEN
        BEGIN
            PERFORM pinecone.create_index('memory.entries', 'embedding', engine => 'native', filter_column => 'acl_scope_key');
        EXCEPTION WHEN OTHERS THEN
            RAISE NOTICE 'pinecone.create_index on memory.entries failed (%); creating pgvector HNSW fallback', SQLERRM;
            CREATE INDEX IF NOT EXISTS idx_memory_entries_vec_hnsw ON memory.entries USING hnsw (embedding vector_cosine_ops);
        END;
    ELSE
        CREATE INDEX IF NOT EXISTS idx_memory_entries_vec_hnsw ON memory.entries USING hnsw (embedding vector_cosine_ops);
    END IF;
END;
$$;

DO $$
BEGIN
    IF vectors.has_pg_search() THEN
        BEGIN
            EXECUTE 'CREATE INDEX IF NOT EXISTS idx_memory_entries_bm25 ON memory.entries USING bm25 (id, (bm25_text::pdb.lindera(korean)))';
        EXCEPTION WHEN OTHERS THEN
            RAISE NOTICE 'pg_search bm25 index on memory.entries failed (%); creating pg_trgm + tsvector fallback', SQLERRM;
            CREATE INDEX IF NOT EXISTS idx_memory_entries_bm25_trgm ON memory.entries USING gin (bm25_text gin_trgm_ops);
            CREATE INDEX IF NOT EXISTS idx_memory_entries_bm25_tsv ON memory.entries USING gin (to_tsvector('simple', bm25_text));
        END;
    ELSE
        CREATE INDEX IF NOT EXISTS idx_memory_entries_bm25_trgm ON memory.entries USING gin (bm25_text gin_trgm_ops);
        CREATE INDEX IF NOT EXISTS idx_memory_entries_bm25_tsv ON memory.entries USING gin (to_tsvector('simple', bm25_text));
    END IF;
END;
$$;

-- Fail-Closed RLS (private memories restricted to owner or same-dept dept_admin)
ALTER TABLE memory.episodes ENABLE ROW LEVEL SECURITY;
ALTER TABLE memory.episodes FORCE ROW LEVEL SECURITY;
ALTER TABLE memory.entries  ENABLE ROW LEVEL SECURITY;
ALTER TABLE memory.entries  FORCE ROW LEVEL SECURITY;
ALTER TABLE memory.patches  ENABLE ROW LEVEL SECURITY;
ALTER TABLE memory.patches  FORCE ROW LEVEL SECURITY;

CREATE POLICY rls_episodes_read ON memory.episodes FOR SELECT
    USING (iam.rls_can_read(org_id, dept_id, owner_principal_id, access_scope, allowed_group_ids, arn, min_auth_level, denied_group_ids, 'episode', id));
CREATE POLICY rls_episodes_insert ON memory.episodes FOR INSERT
    WITH CHECK (iam.rls_can_insert(org_id, dept_id, owner_principal_id, arn, min_auth_level));
CREATE POLICY rls_episodes_update ON memory.episodes FOR UPDATE
    USING (iam.rls_can_access(org_id, dept_id, owner_principal_id, access_scope, allowed_group_ids, arn, min_auth_level, denied_group_ids, 'write', 'episode', id))
    WITH CHECK (iam.rls_can_access(org_id, dept_id, owner_principal_id, access_scope, allowed_group_ids, arn, min_auth_level, denied_group_ids, 'write', 'episode', id));
CREATE POLICY rls_episodes_delete ON memory.episodes FOR DELETE
    USING (iam.rls_can_access(org_id, dept_id, owner_principal_id, access_scope, allowed_group_ids, arn, min_auth_level, denied_group_ids, 'write', 'episode', id));

CREATE POLICY rls_memory_read ON memory.entries FOR SELECT
    USING (status = 'active' AND iam.rls_can_read(org_id, dept_id, owner_principal_id, access_scope, allowed_group_ids, arn, min_auth_level, denied_group_ids, 'memory', id));
CREATE POLICY rls_memory_insert ON memory.entries FOR INSERT
    WITH CHECK (
        iam.rls_can_insert(org_id, dept_id, owner_principal_id, arn, min_auth_level)
        AND (
            memory.entries.bucket_id IS NULL
            OR EXISTS (
                SELECT 1 FROM corpus.buckets AS b
                WHERE b.id = memory.entries.bucket_id
                  AND b.org_id = memory.entries.org_id
                  AND iam.rls_can_access(b.org_id, b.dept_id, b.owner_principal_id, b.access_scope, b.allowed_group_ids, b.arn, b.min_auth_level, b.denied_group_ids, 'write', 'bucket', b.id)
            )
        )
        AND (
            memory.entries.source_episode_id IS NULL
            OR EXISTS (
                SELECT 1 FROM memory.episodes AS ep
                WHERE ep.id = memory.entries.source_episode_id
                  AND ep.org_id = memory.entries.org_id
                  AND iam.rls_can_read(ep.org_id, ep.dept_id, ep.owner_principal_id, ep.access_scope, ep.allowed_group_ids, ep.arn, ep.min_auth_level, ep.denied_group_ids, 'episode', ep.id)
            )
        )
        AND (
            memory.entries.source_document_id IS NULL
            OR EXISTS (
                SELECT 1 FROM corpus.documents AS d
                WHERE d.id = memory.entries.source_document_id
                  AND d.org_id = memory.entries.org_id
                  AND iam.rls_can_read(d.org_id, d.dept_id, d.owner_principal_id, d.access_scope, d.allowed_group_ids, d.arn, d.min_auth_level, d.denied_group_ids, 'document', d.id)
            )
        )
        AND (
            memory.entries.promoted_wiki_id IS NULL
            OR EXISTS (
                SELECT 1 FROM wiki.pages AS w
                WHERE w.id = memory.entries.promoted_wiki_id
                  AND w.org_id = memory.entries.org_id
                  AND iam.rls_can_read(w.org_id, w.dept_id, w.owner_principal_id, w.access_scope, w.allowed_group_ids, w.arn, w.min_auth_level, w.denied_group_ids, 'wiki', w.id)
            )
        )
        AND (
            memory.entries.superseded_by IS NULL
            OR EXISTS (
                SELECT 1 FROM memory.entries AS me
                WHERE me.id = memory.entries.superseded_by
                  AND me.org_id = memory.entries.org_id
                  AND iam.rls_can_read(me.org_id, me.dept_id, me.owner_principal_id, me.access_scope, me.allowed_group_ids, me.arn, me.min_auth_level, me.denied_group_ids, 'memory', me.id)
            )
        )
    );
CREATE POLICY rls_memory_update ON memory.entries FOR UPDATE
    USING (iam.rls_can_access(org_id, dept_id, owner_principal_id, access_scope, allowed_group_ids, arn, min_auth_level, denied_group_ids, 'write', 'memory', id))
    WITH CHECK (
        iam.rls_can_access(org_id, dept_id, owner_principal_id, access_scope, allowed_group_ids, arn, min_auth_level, denied_group_ids, 'write', 'memory', id)
        AND (
            memory.entries.bucket_id IS NULL
            OR EXISTS (
                SELECT 1 FROM corpus.buckets AS b
                WHERE b.id = memory.entries.bucket_id
                  AND b.org_id = memory.entries.org_id
                  AND iam.rls_can_access(b.org_id, b.dept_id, b.owner_principal_id, b.access_scope, b.allowed_group_ids, b.arn, b.min_auth_level, b.denied_group_ids, 'write', 'bucket', b.id)
            )
        )
        AND (
            memory.entries.source_episode_id IS NULL
            OR EXISTS (
                SELECT 1 FROM memory.episodes AS ep
                WHERE ep.id = memory.entries.source_episode_id
                  AND ep.org_id = memory.entries.org_id
                  AND iam.rls_can_read(ep.org_id, ep.dept_id, ep.owner_principal_id, ep.access_scope, ep.allowed_group_ids, ep.arn, ep.min_auth_level, ep.denied_group_ids, 'episode', ep.id)
            )
        )
        AND (
            memory.entries.source_document_id IS NULL
            OR EXISTS (
                SELECT 1 FROM corpus.documents AS d
                WHERE d.id = memory.entries.source_document_id
                  AND d.org_id = memory.entries.org_id
                  AND iam.rls_can_read(d.org_id, d.dept_id, d.owner_principal_id, d.access_scope, d.allowed_group_ids, d.arn, d.min_auth_level, d.denied_group_ids, 'document', d.id)
            )
        )
        AND (
            memory.entries.promoted_wiki_id IS NULL
            OR EXISTS (
                SELECT 1 FROM wiki.pages AS w
                WHERE w.id = memory.entries.promoted_wiki_id
                  AND w.org_id = memory.entries.org_id
                  AND iam.rls_can_read(w.org_id, w.dept_id, w.owner_principal_id, w.access_scope, w.allowed_group_ids, w.arn, w.min_auth_level, w.denied_group_ids, 'wiki', w.id)
            )
        )
        AND (
            memory.entries.superseded_by IS NULL
            OR EXISTS (
                SELECT 1 FROM memory.entries AS me
                WHERE me.id = memory.entries.superseded_by
                  AND me.org_id = memory.entries.org_id
                  AND iam.rls_can_read(me.org_id, me.dept_id, me.owner_principal_id, me.access_scope, me.allowed_group_ids, me.arn, me.min_auth_level, me.denied_group_ids, 'memory', me.id)
            )
        )
    );
CREATE POLICY rls_memory_delete ON memory.entries FOR DELETE
    USING (iam.rls_can_access(org_id, dept_id, owner_principal_id, access_scope, allowed_group_ids, arn, min_auth_level, denied_group_ids, 'write', 'memory', id));

CREATE TRIGGER trg_memory_episodes_ownership
    BEFORE UPDATE ON memory.episodes
    FOR EACH ROW EXECUTE FUNCTION iam.prevent_non_admin_ownership_reassignment();
CREATE TRIGGER trg_memory_entries_ownership
    BEFORE UPDATE ON memory.entries
    FOR EACH ROW EXECUTE FUNCTION iam.prevent_non_admin_ownership_reassignment();

CREATE POLICY rls_memory_patches_read ON memory.patches FOR SELECT
    USING (
        memory.patches.org_id = iam.safe_uuid(current_setting('app.org_id', true))
        AND EXISTS (
            SELECT 1 FROM memory.entries AS e
            WHERE e.id = memory.patches.memory_entry_id
              AND e.org_id = memory.patches.org_id
              AND e.status = 'active'
              AND iam.rls_can_read(e.org_id, e.dept_id, e.owner_principal_id, e.access_scope, e.allowed_group_ids, e.arn, e.min_auth_level, e.denied_group_ids, 'memory', e.id)
        )
    );
CREATE POLICY rls_memory_patches_insert ON memory.patches FOR INSERT
    WITH CHECK (
        memory.patches.org_id = iam.safe_uuid(current_setting('app.org_id', true))
        AND EXISTS (
            SELECT 1 FROM memory.entries AS e
            WHERE e.id = memory.patches.memory_entry_id
              AND e.org_id = memory.patches.org_id
              AND iam.rls_can_access(e.org_id, e.dept_id, e.owner_principal_id, e.access_scope, e.allowed_group_ids, e.arn, e.min_auth_level, e.denied_group_ids, 'write', 'memory', e.id)
        )
    );
CREATE POLICY rls_memory_patches_update ON memory.patches FOR UPDATE
    USING (
        memory.patches.org_id = iam.safe_uuid(current_setting('app.org_id', true))
        AND EXISTS (
            SELECT 1 FROM memory.entries AS e
            WHERE e.id = memory.patches.memory_entry_id
              AND e.org_id = memory.patches.org_id
              AND iam.rls_can_access(e.org_id, e.dept_id, e.owner_principal_id, e.access_scope, e.allowed_group_ids, e.arn, e.min_auth_level, e.denied_group_ids, 'write', 'memory', e.id)
        )
    )
    WITH CHECK (
        memory.patches.org_id = iam.safe_uuid(current_setting('app.org_id', true))
        AND EXISTS (
            SELECT 1 FROM memory.entries AS e
            WHERE e.id = memory.patches.memory_entry_id
              AND e.org_id = memory.patches.org_id
              AND iam.rls_can_access(e.org_id, e.dept_id, e.owner_principal_id, e.access_scope, e.allowed_group_ids, e.arn, e.min_auth_level, e.denied_group_ids, 'write', 'memory', e.id)
        )
    );
CREATE POLICY rls_memory_patches_delete ON memory.patches FOR DELETE
    USING (
        memory.patches.org_id = iam.safe_uuid(current_setting('app.org_id', true))
        AND EXISTS (
            SELECT 1 FROM memory.entries AS e
            WHERE e.id = memory.patches.memory_entry_id
              AND e.org_id = memory.patches.org_id
              AND iam.rls_can_access(e.org_id, e.dept_id, e.owner_principal_id, e.access_scope, e.allowed_group_ids, e.arn, e.min_auth_level, e.denied_group_ids, 'write', 'memory', e.id)
        )
    );

-- Replace wiki_source_can_read from 004_llm_wiki.sql to also validate 'memory' against memory.entries
CREATE OR REPLACE FUNCTION iam.wiki_source_can_read(
    p_org_id UUID,
    p_source_type TEXT,
    p_source_id UUID
)
RETURNS BOOLEAN
LANGUAGE plpgsql
STABLE
SECURITY DEFINER
SET search_path = iam, corpus, wiki, memory, pg_catalog
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
    ELSIF p_source_type = 'memory' THEN
        RETURN EXISTS (
            SELECT 1
            FROM memory.entries AS me
            WHERE me.id = p_source_id
              AND me.org_id = p_org_id
              AND me.status = 'active'
              AND iam.rls_can_read(
                  me.org_id, me.dept_id, me.owner_principal_id, me.access_scope,
                  me.allowed_group_ids, me.arn, me.min_auth_level, me.denied_group_ids,
                  'memory', me.id
              )
        );
    END IF;

    RETURN false;
END;
$$;

-- Deferred FK from plugins.source_entries (created in 005)
ALTER TABLE plugins.source_entries
    ADD CONSTRAINT fk_plugins_source_entries_episode
    FOREIGN KEY (org_id, memory_episode_id) REFERENCES memory.episodes(org_id, id) ON DELETE SET NULL (memory_episode_id);

-- ============================================================================
-- B. Full-Layer Cascade Redaction Ledger (pii.redaction_*)
-- ============================================================================

CREATE TABLE pii.redaction_jobs (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id              UUID NOT NULL REFERENCES iam.orgs(id) ON DELETE CASCADE,
    requested_by        UUID NOT NULL,
    target_scope        TEXT NOT NULL CHECK (target_scope IN ('document', 'chunk', 'memory', 'wiki', 'pattern_org', 'pattern_dept', 'vault_entry')),
    target_ref          JSONB NOT NULL,
    redaction_mode      TEXT NOT NULL DEFAULT 'mask_keep_vault'
                        CHECK (redaction_mode IN ('mask_keep_vault', 'hard_purge', 'tombstone_delete')),
    redact_thumbnails   BOOLEAN NOT NULL DEFAULT true,
    reason              TEXT NOT NULL,
    status              TEXT NOT NULL DEFAULT 'queued' CHECK (status IN ('queued', 'running', 'completed', 'failed')),
    affected_counts     JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    completed_at        TIMESTAMPTZ,
    UNIQUE (org_id, id),
    FOREIGN KEY (org_id, requested_by) REFERENCES iam.principals(org_id, id)
);

CREATE TABLE pii.redaction_events (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    redaction_job_id    UUID NOT NULL,
    org_id              UUID NOT NULL REFERENCES iam.orgs(id) ON DELETE CASCADE,
    target_table        TEXT NOT NULL,
    target_id           UUID NOT NULL,
    action_taken        TEXT NOT NULL,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    FOREIGN KEY (org_id, redaction_job_id) REFERENCES pii.redaction_jobs(org_id, id) ON DELETE CASCADE
);

-- ============================================================================
-- C. Incremental Index Write Outbox, Watermarks & Quarantine (indexing.*)
-- ============================================================================

CREATE TABLE indexing.outbox (
    id              BIGSERIAL PRIMARY KEY,
    org_id          UUID NOT NULL REFERENCES iam.orgs(id) ON DELETE CASCADE,
    target_table    TEXT NOT NULL CHECK (target_table IN ('corpus.chunks', 'corpus.pages', 'corpus.page_figures', 'wiki.pages', 'memory.entries')),
    target_id       UUID NOT NULL,
    op              TEXT NOT NULL CHECK (op IN ('upsert_embed', 'delete_vector', 'recompile_wiki', 'sync_external_index', 'redact_reindex')),
    content_sha256  TEXT,
    priority        SMALLINT NOT NULL DEFAULT 5 CHECK (priority BETWEEN 1 AND 10),
    payload         JSONB NOT NULL DEFAULT '{}'::jsonb,
    status          TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'processing', 'done', 'quarantined')),
    attempts        INT NOT NULL DEFAULT 0,
    max_attempts    INT NOT NULL DEFAULT 3,
    last_error      TEXT,
    idempotency_key TEXT,
    available_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    processed_at    TIMESTAMPTZ,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (org_id, id)
);

CREATE INDEX idx_indexing_outbox_claim ON indexing.outbox (status, priority, available_at, id)
    WHERE status = 'pending';
CREATE UNIQUE INDEX idx_indexing_outbox_idem ON indexing.outbox (org_id, idempotency_key)
    WHERE idempotency_key IS NOT NULL AND status IN ('pending', 'processing');

CREATE TABLE indexing.watermarks (
    org_id          UUID NOT NULL REFERENCES iam.orgs(id) ON DELETE CASCADE,
    target_table    TEXT NOT NULL,
    backend_name    TEXT NOT NULL,
    last_outbox_id  BIGINT NOT NULL DEFAULT 0,
    indexed_count   BIGINT NOT NULL DEFAULT 0,
    last_synced_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (org_id, target_table, backend_name)
);

CREATE TABLE indexing.quarantine (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    outbox_id       BIGINT,
    org_id          UUID NOT NULL REFERENCES iam.orgs(id) ON DELETE CASCADE,
    target_table    TEXT NOT NULL,
    target_id       UUID NOT NULL,
    op              TEXT NOT NULL,
    error_class     TEXT NOT NULL,
    error_detail    TEXT NOT NULL,
    payload         JSONB NOT NULL DEFAULT '{}'::jsonb,
    resolved_at     TIMESTAMPTZ,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    FOREIGN KEY (org_id, outbox_id) REFERENCES indexing.outbox(org_id, id) ON DELETE SET NULL (outbox_id)
);

-- SECURITY DEFINER helper verifying indexing.outbox target belongs to org and caller has p_action access
CREATE OR REPLACE FUNCTION iam.outbox_target_accessible(
    p_org_id UUID,
    p_target_table TEXT,
    p_target_id UUID,
    p_action TEXT
)
RETURNS BOOLEAN
LANGUAGE plpgsql
STABLE
SECURITY DEFINER
SET search_path = iam, corpus, wiki, memory, pg_catalog
AS $$
BEGIN
    IF p_org_id IS NULL OR p_target_table IS NULL OR p_target_id IS NULL OR p_action IS NULL THEN
        RETURN false;
    END IF;
    IF p_action NOT IN ('read', 'write') THEN
        RETURN false;
    END IF;

    IF p_target_table = 'corpus.chunks' THEN
        RETURN EXISTS (
            SELECT 1 FROM corpus.chunks AS c
            WHERE c.id = p_target_id
              AND c.org_id = p_org_id
              AND iam.rls_can_access(
                  c.org_id, c.dept_id, c.owner_principal_id, c.access_scope,
                  c.allowed_group_ids, c.arn, c.min_auth_level, c.denied_group_ids,
                  p_action, 'chunk', c.id
              )
        );
    ELSIF p_target_table = 'corpus.pages' THEN
        RETURN EXISTS (
            SELECT 1 FROM corpus.pages AS p
            WHERE p.id = p_target_id
              AND p.org_id = p_org_id
              AND iam.rls_can_access(
                  p.org_id, p.dept_id, p.owner_principal_id, p.access_scope,
                  p.allowed_group_ids, p.arn, p.min_auth_level, p.denied_group_ids,
                  p_action, 'page', p.id
              )
        );
    ELSIF p_target_table = 'corpus.page_figures' THEN
        RETURN EXISTS (
            SELECT 1 FROM corpus.page_figures AS pf
            WHERE pf.id = p_target_id
              AND pf.org_id = p_org_id
              AND iam.rls_can_access(
                  pf.org_id, pf.dept_id, pf.owner_principal_id, pf.access_scope,
                  pf.allowed_group_ids, pf.arn, pf.min_auth_level, pf.denied_group_ids,
                  p_action, 'figure', pf.id
              )
        );
    ELSIF p_target_table = 'wiki.pages' THEN
        RETURN EXISTS (
            SELECT 1 FROM wiki.pages AS w
            WHERE w.id = p_target_id
              AND w.org_id = p_org_id
              AND (p_action <> 'read' OR w.status = 'active')
              AND iam.rls_can_access(
                  w.org_id, w.dept_id, w.owner_principal_id, w.access_scope,
                  w.allowed_group_ids, w.arn, w.min_auth_level, w.denied_group_ids,
                  p_action, 'wiki', w.id
              )
        );
    ELSIF p_target_table = 'memory.entries' THEN
        RETURN EXISTS (
            SELECT 1 FROM memory.entries AS me
            WHERE me.id = p_target_id
              AND me.org_id = p_org_id
              AND (p_action <> 'read' OR me.status = 'active')
              AND iam.rls_can_access(
                  me.org_id, me.dept_id, me.owner_principal_id, me.access_scope,
                  me.allowed_group_ids, me.arn, me.min_auth_level, me.denied_group_ids,
                  p_action, 'memory', me.id
              )
        );
    END IF;

    RETURN false;
END;
$$;

-- Fail-Closed RLS on openrag_app-accessible indexing.outbox
ALTER TABLE indexing.outbox ENABLE ROW LEVEL SECURITY;
ALTER TABLE indexing.outbox FORCE ROW LEVEL SECURITY;

CREATE POLICY rls_indexing_outbox_read ON indexing.outbox FOR SELECT
    USING (EXISTS (
        SELECT 1 FROM iam.session_active_principal() AS s
        WHERE s.v_ok
          AND indexing.outbox.org_id = s.v_org_id
          AND iam.outbox_target_accessible(
              indexing.outbox.org_id,
              indexing.outbox.target_table,
              indexing.outbox.target_id,
              'read'
          )
    ));

CREATE POLICY rls_indexing_outbox_insert ON indexing.outbox FOR INSERT
    WITH CHECK (EXISTS (
        SELECT 1 FROM iam.session_active_principal() AS s
        WHERE s.v_ok
          AND indexing.outbox.org_id = s.v_org_id
          AND iam.outbox_target_accessible(
              indexing.outbox.org_id,
              indexing.outbox.target_table,
              indexing.outbox.target_id,
              'write'
          )
    ));
