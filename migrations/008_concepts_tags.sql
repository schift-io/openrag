-- OpenRAG Migration 008: Canonical Concept/Alias Graph (F36), First-Class Tags (F37) & Memory Belief-Revision Edges (F18)

-- ============================================================================
-- A. Security Helpers (prevent 23503 cross-tenant existence oracles under RLS)
-- ============================================================================
CREATE OR REPLACE FUNCTION iam.wiki_page_readable_in_org(
    p_org_id UUID,
    p_page_id UUID
)
RETURNS BOOLEAN
LANGUAGE sql
STABLE
SECURITY DEFINER
SET search_path = wiki, iam, pg_catalog
AS $$
    SELECT p_org_id IS NOT NULL
       AND p_page_id IS NOT NULL
       AND EXISTS (
           SELECT 1
           FROM wiki.pages AS p
           WHERE p.id = p_page_id
             AND p.org_id = p_org_id
             AND p.status = 'active'
             AND iam.rls_can_read(
                 p.org_id, p.dept_id, p.owner_principal_id, p.access_scope,
                 p.allowed_group_ids, p.arn, p.min_auth_level, p.denied_group_ids,
                 'wiki', p.id
             )
       );
$$;

-- ============================================================================
-- B. Canonical Concepts & Aliases (F36)
-- ============================================================================
CREATE TABLE wiki.concepts (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id          UUID NOT NULL REFERENCES iam.orgs(id) ON DELETE CASCADE,
    canonical_key   TEXT NOT NULL,
    display_name    TEXT NOT NULL,
    category        TEXT NOT NULL DEFAULT 'general'
                    CHECK (category IN ('general', 'domain', 'entity', 'acronym', 'safeword_alias')),
    description     TEXT,
    wiki_page_id    UUID,
    embedding       vector(128),
    created_by      UUID,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (org_id, canonical_key),
    UNIQUE (org_id, id),
    FOREIGN KEY (org_id, wiki_page_id) REFERENCES wiki.pages(org_id, id) ON DELETE SET NULL (wiki_page_id),
    FOREIGN KEY (org_id, created_by) REFERENCES iam.principals(org_id, id) ON DELETE SET NULL (created_by)
);

CREATE OR REPLACE FUNCTION iam.concept_belongs_to_org(
    p_org_id UUID,
    p_concept_id UUID
)
RETURNS BOOLEAN
LANGUAGE sql
STABLE
SECURITY DEFINER
SET search_path = wiki, iam, pg_catalog
AS $$
    SELECT p_org_id IS NOT NULL
       AND p_concept_id IS NOT NULL
       AND EXISTS (
           SELECT 1
           FROM wiki.concepts AS c
           WHERE c.id = p_concept_id
             AND c.org_id = p_org_id
       );
$$;

CREATE TABLE wiki.concept_aliases (
    id               UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id           UUID NOT NULL REFERENCES iam.orgs(id) ON DELETE CASCADE,
    concept_id       UUID NOT NULL,
    alias_text       TEXT NOT NULL,
    normalized_alias TEXT NOT NULL,
    compact_alias    TEXT NOT NULL,
    lang             TEXT NOT NULL DEFAULT 'und',
    source           TEXT NOT NULL DEFAULT 'manual'
                     CHECK (source IN ('manual', 'ingest_parenthetical', 'wiki_frontmatter', 'semantic_auto', 'seed')),
    confidence       REAL NOT NULL DEFAULT 1.0 CHECK (confidence >= 0.0 AND confidence <= 1.0),
    created_by       UUID,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (org_id, concept_id, normalized_alias),
    UNIQUE (org_id, id),
    FOREIGN KEY (org_id, concept_id) REFERENCES wiki.concepts(org_id, id) ON DELETE CASCADE,
    FOREIGN KEY (org_id, created_by) REFERENCES iam.principals(org_id, id) ON DELETE SET NULL (created_by)
);

CREATE INDEX idx_wiki_concepts_org_key ON wiki.concepts (org_id, canonical_key);
CREATE INDEX idx_wiki_concept_aliases_norm ON wiki.concept_aliases (org_id, normalized_alias);
CREATE INDEX idx_wiki_concept_aliases_compact ON wiki.concept_aliases (org_id, compact_alias);
CREATE INDEX idx_wiki_concept_aliases_concept ON wiki.concept_aliases (org_id, concept_id);
CREATE INDEX idx_wiki_concept_aliases_trgm ON wiki.concept_aliases USING gin (normalized_alias gin_trgm_ops);
CREATE INDEX idx_wiki_concepts_vec_hnsw ON wiki.concepts USING hnsw (embedding vector_cosine_ops);

ALTER TABLE wiki.concepts        ENABLE ROW LEVEL SECURITY;
ALTER TABLE wiki.concepts        FORCE ROW LEVEL SECURITY;
ALTER TABLE wiki.concept_aliases ENABLE ROW LEVEL SECURITY;
ALTER TABLE wiki.concept_aliases FORCE ROW LEVEL SECURITY;

CREATE POLICY rls_wiki_concepts_read ON wiki.concepts FOR SELECT
    USING (EXISTS (
        SELECT 1 FROM iam.session_active_principal() AS s
        WHERE s.v_ok AND wiki.concepts.org_id = s.v_org_id
    ));

CREATE POLICY rls_wiki_concepts_insert ON wiki.concepts FOR INSERT
    WITH CHECK (EXISTS (
        SELECT 1 FROM iam.session_active_principal() AS s
        WHERE s.v_ok
          AND wiki.concepts.org_id = s.v_org_id
          AND (wiki.concepts.created_by IS NULL OR wiki.concepts.created_by = s.v_principal_id)
          AND (wiki.concepts.wiki_page_id IS NULL OR iam.wiki_page_readable_in_org(wiki.concepts.org_id, wiki.concepts.wiki_page_id))
    ));

CREATE POLICY rls_wiki_concepts_update ON wiki.concepts FOR UPDATE
    USING (EXISTS (
        SELECT 1 FROM iam.session_active_principal() AS s
        WHERE s.v_ok
          AND wiki.concepts.org_id = s.v_org_id
          AND (
              wiki.concepts.created_by IS NULL
              OR wiki.concepts.created_by = s.v_principal_id
              OR s.v_is_org_admin
              OR cardinality(s.v_dept_admin_ids) > 0
          )
    ))
    WITH CHECK (EXISTS (
        SELECT 1 FROM iam.session_active_principal() AS s
        WHERE s.v_ok
          AND wiki.concepts.org_id = s.v_org_id
          AND (wiki.concepts.created_by IS NULL OR iam.principal_active_in_org(wiki.concepts.org_id, wiki.concepts.created_by))
          AND (wiki.concepts.wiki_page_id IS NULL OR iam.wiki_page_readable_in_org(wiki.concepts.org_id, wiki.concepts.wiki_page_id))
    ));

CREATE POLICY rls_wiki_concepts_delete ON wiki.concepts FOR DELETE
    USING (EXISTS (
        SELECT 1 FROM iam.session_active_principal() AS s
        WHERE s.v_ok
          AND wiki.concepts.org_id = s.v_org_id
          AND (
              wiki.concepts.created_by IS NULL
              OR wiki.concepts.created_by = s.v_principal_id
              OR s.v_is_org_admin
              OR cardinality(s.v_dept_admin_ids) > 0
          )
    ));

CREATE POLICY rls_wiki_concept_aliases_read ON wiki.concept_aliases FOR SELECT
    USING (EXISTS (
        SELECT 1 FROM iam.session_active_principal() AS s
        WHERE s.v_ok AND wiki.concept_aliases.org_id = s.v_org_id
    ));

CREATE POLICY rls_wiki_concept_aliases_insert ON wiki.concept_aliases FOR INSERT
    WITH CHECK (EXISTS (
        SELECT 1 FROM iam.session_active_principal() AS s
        WHERE s.v_ok
          AND wiki.concept_aliases.org_id = s.v_org_id
          AND (wiki.concept_aliases.created_by IS NULL OR wiki.concept_aliases.created_by = s.v_principal_id)
          AND iam.concept_belongs_to_org(wiki.concept_aliases.org_id, wiki.concept_aliases.concept_id)
    ));

CREATE POLICY rls_wiki_concept_aliases_update ON wiki.concept_aliases FOR UPDATE
    USING (EXISTS (
        SELECT 1 FROM iam.session_active_principal() AS s
        WHERE s.v_ok
          AND wiki.concept_aliases.org_id = s.v_org_id
          AND (
              wiki.concept_aliases.created_by IS NULL
              OR wiki.concept_aliases.created_by = s.v_principal_id
              OR s.v_is_org_admin
              OR cardinality(s.v_dept_admin_ids) > 0
          )
    ))
    WITH CHECK (EXISTS (
        SELECT 1 FROM iam.session_active_principal() AS s
        WHERE s.v_ok
          AND wiki.concept_aliases.org_id = s.v_org_id
          AND (wiki.concept_aliases.created_by IS NULL OR iam.principal_active_in_org(wiki.concept_aliases.org_id, wiki.concept_aliases.created_by))
          AND iam.concept_belongs_to_org(wiki.concept_aliases.org_id, wiki.concept_aliases.concept_id)
    ));

CREATE POLICY rls_wiki_concept_aliases_delete ON wiki.concept_aliases FOR DELETE
    USING (EXISTS (
        SELECT 1 FROM iam.session_active_principal() AS s
        WHERE s.v_ok
          AND wiki.concept_aliases.org_id = s.v_org_id
          AND (
              wiki.concept_aliases.created_by IS NULL
              OR wiki.concept_aliases.created_by = s.v_principal_id
              OR s.v_is_org_admin
              OR cardinality(s.v_dept_admin_ids) > 0
          )
    ));

-- ============================================================================
-- C. First-Class Tag-Based Search & Faceted Filtering (F37)
-- ============================================================================
ALTER TABLE corpus.documents ADD COLUMN IF NOT EXISTS tags TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[];
ALTER TABLE corpus.chunks    ADD COLUMN IF NOT EXISTS tags TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[];
ALTER TABLE wiki.pages       ADD COLUMN IF NOT EXISTS tags TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[];
ALTER TABLE memory.entries   ADD COLUMN IF NOT EXISTS tags TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[];

CREATE INDEX IF NOT EXISTS idx_corpus_documents_tags_gin ON corpus.documents USING gin (tags);
CREATE INDEX IF NOT EXISTS idx_corpus_chunks_tags_gin    ON corpus.chunks    USING gin (tags);
CREATE INDEX IF NOT EXISTS idx_wiki_pages_tags_gin       ON wiki.pages       USING gin (tags);
CREATE INDEX IF NOT EXISTS idx_memory_entries_tags_gin   ON memory.entries   USING gin (tags);

-- ============================================================================
-- D. Memory Belief-Revision Typed Edges (F18)
-- ============================================================================
CREATE TABLE memory.edges (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id          UUID NOT NULL REFERENCES iam.orgs(id) ON DELETE CASCADE,
    src_memory_id   UUID NOT NULL,
    dst_memory_id   UUID NOT NULL,
    relation_type   TEXT NOT NULL CHECK (relation_type IN ('derived_from', 'supports', 'contradicts', 'supersedes')),
    confidence      REAL NOT NULL DEFAULT 1.0 CHECK (confidence >= 0.0 AND confidence <= 1.0),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (org_id, src_memory_id, dst_memory_id, relation_type),
    FOREIGN KEY (org_id, src_memory_id) REFERENCES memory.entries(org_id, id) ON DELETE CASCADE,
    FOREIGN KEY (org_id, dst_memory_id) REFERENCES memory.entries(org_id, id) ON DELETE CASCADE
);

CREATE INDEX idx_memory_edges_src ON memory.edges (org_id, src_memory_id, relation_type);
CREATE INDEX idx_memory_edges_dst ON memory.edges (org_id, dst_memory_id, relation_type);

ALTER TABLE memory.edges ENABLE ROW LEVEL SECURITY;
ALTER TABLE memory.edges FORCE ROW LEVEL SECURITY;

CREATE POLICY rls_memory_edges_read ON memory.edges FOR SELECT
    USING (
        memory.edges.org_id = iam.safe_uuid(current_setting('app.org_id', true))
        AND EXISTS (
            SELECT 1 FROM memory.entries AS sm
            WHERE sm.id = memory.edges.src_memory_id
              AND sm.org_id = memory.edges.org_id
              AND sm.status = 'active'
              AND iam.rls_can_read(sm.org_id, sm.dept_id, sm.owner_principal_id, sm.access_scope, sm.allowed_group_ids, sm.arn, sm.min_auth_level, sm.denied_group_ids, 'memory', sm.id)
        )
        AND EXISTS (
            SELECT 1 FROM memory.entries AS dm
            WHERE dm.id = memory.edges.dst_memory_id
              AND dm.org_id = memory.edges.org_id
              AND (
                  dm.status = 'active'
                  OR (memory.edges.relation_type = 'supersedes' AND dm.status = 'superseded')
              )
              AND iam.rls_can_read(dm.org_id, dm.dept_id, dm.owner_principal_id, dm.access_scope, dm.allowed_group_ids, dm.arn, dm.min_auth_level, dm.denied_group_ids, 'memory', dm.id)
        )
    );

CREATE POLICY rls_memory_edges_insert ON memory.edges FOR INSERT
    WITH CHECK (
        memory.edges.org_id = iam.safe_uuid(current_setting('app.org_id', true))
        AND EXISTS (
            SELECT 1 FROM memory.entries AS sm
            WHERE sm.id = memory.edges.src_memory_id
              AND sm.org_id = memory.edges.org_id
              AND sm.status = 'active'
              AND iam.rls_can_access(sm.org_id, sm.dept_id, sm.owner_principal_id, sm.access_scope, sm.allowed_group_ids, sm.arn, sm.min_auth_level, sm.denied_group_ids, 'write', 'memory', sm.id)
        )
        AND EXISTS (
            SELECT 1 FROM memory.entries AS dm
            WHERE dm.id = memory.edges.dst_memory_id
              AND dm.org_id = memory.edges.org_id
              AND (
                  dm.status = 'active'
                  OR (memory.edges.relation_type = 'supersedes' AND dm.status = 'superseded')
              )
              AND iam.rls_can_read(dm.org_id, dm.dept_id, dm.owner_principal_id, dm.access_scope, dm.allowed_group_ids, dm.arn, dm.min_auth_level, dm.denied_group_ids, 'memory', dm.id)
        )
    );

CREATE POLICY rls_memory_edges_update ON memory.edges FOR UPDATE
    USING (
        memory.edges.org_id = iam.safe_uuid(current_setting('app.org_id', true))
        AND EXISTS (
            SELECT 1 FROM memory.entries AS sm
            WHERE sm.id = memory.edges.src_memory_id
              AND sm.org_id = memory.edges.org_id
              AND iam.rls_can_access(sm.org_id, sm.dept_id, sm.owner_principal_id, sm.access_scope, sm.allowed_group_ids, sm.arn, sm.min_auth_level, sm.denied_group_ids, 'write', 'memory', sm.id)
        )
    )
    WITH CHECK (
        memory.edges.org_id = iam.safe_uuid(current_setting('app.org_id', true))
        AND EXISTS (
            SELECT 1 FROM memory.entries AS sm
            WHERE sm.id = memory.edges.src_memory_id
              AND sm.org_id = memory.edges.org_id
              AND sm.status = 'active'
              AND iam.rls_can_access(sm.org_id, sm.dept_id, sm.owner_principal_id, sm.access_scope, sm.allowed_group_ids, sm.arn, sm.min_auth_level, sm.denied_group_ids, 'write', 'memory', sm.id)
        )
        AND EXISTS (
            SELECT 1 FROM memory.entries AS dm
            WHERE dm.id = memory.edges.dst_memory_id
              AND dm.org_id = memory.edges.org_id
              AND (
                  dm.status = 'active'
                  OR (memory.edges.relation_type = 'supersedes' AND dm.status = 'superseded')
              )
              AND iam.rls_can_read(dm.org_id, dm.dept_id, dm.owner_principal_id, dm.access_scope, dm.allowed_group_ids, dm.arn, dm.min_auth_level, dm.denied_group_ids, 'memory', dm.id)
        )
    );

CREATE POLICY rls_memory_edges_delete ON memory.edges FOR DELETE
    USING (
        memory.edges.org_id = iam.safe_uuid(current_setting('app.org_id', true))
        AND EXISTS (
            SELECT 1 FROM memory.entries AS sm
            WHERE sm.id = memory.edges.src_memory_id
              AND sm.org_id = memory.edges.org_id
              AND iam.rls_can_access(sm.org_id, sm.dept_id, sm.owner_principal_id, sm.access_scope, sm.allowed_group_ids, sm.arn, sm.min_auth_level, sm.denied_group_ids, 'write', 'memory', sm.id)
        )
    );

-- Fail-Closed status transition helper for memory.entries (Retract / Supersede / Redact)
-- Necessary because rls_memory_read enforces status = 'active', which PostgreSQL checks against
-- the new tuple on direct UPDATE by openrag_app when WHERE/RETURNING reads columns.
CREATE OR REPLACE FUNCTION iam.memory_transition_status(
    p_org_id UUID,
    p_entry_id UUID,
    p_new_status TEXT,
    p_superseded_by UUID,
    p_new_version INT,
    p_effective_text TEXT DEFAULT NULL,
    p_bm25_text TEXT DEFAULT NULL,
    p_content_sha256 TEXT DEFAULT NULL
)
RETURNS VOID
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = memory, iam, pg_catalog
AS $$
DECLARE
    v_org_id UUID;
    v_principal_id UUID;
    v_is_org_admin BOOLEAN;
    v_dept_admin_ids UUID[];
    v_ok BOOLEAN;
    v_entry memory.entries%ROWTYPE;
BEGIN
    SELECT s.v_org_id, s.v_principal_id, s.v_is_org_admin, s.v_dept_admin_ids, s.v_ok
    INTO v_org_id, v_principal_id, v_is_org_admin, v_dept_admin_ids, v_ok
    FROM iam.session_active_principal() AS s;

    IF NOT COALESCE(v_ok, false) OR p_org_id IS NULL OR p_org_id IS DISTINCT FROM v_org_id THEN
        RAISE EXCEPTION 'session not active or org_id mismatch on memory transition'
            USING ERRCODE = '42501';
    END IF;

    IF p_new_status IS NULL OR p_new_status NOT IN ('superseded', 'retracted', 'redacted', 'archived') THEN
        RAISE EXCEPTION 'invalid memory target status %', p_new_status
            USING ERRCODE = '22023';
    END IF;

    IF p_superseded_by IS NOT NULL THEN
        IF p_new_status <> 'superseded' THEN
            RAISE EXCEPTION 'superseded_by is only valid when transitioning to superseded'
                USING ERRCODE = '22023';
        END IF;
        IF p_superseded_by = p_entry_id THEN
            RAISE EXCEPTION 'memory entry % cannot supersede itself', p_entry_id
                USING ERRCODE = '22023';
        END IF;
    END IF;

    SELECT * INTO v_entry
    FROM memory.entries
    WHERE org_id = p_org_id AND id = p_entry_id
    FOR UPDATE;

    -- Uniform fail-closed 42501 error across missing, inactive, and unauthorized entries
    -- to prevent same-org existence oracles.
    IF NOT FOUND
       OR v_entry.status <> 'active'
       OR NOT iam.rls_can_access(
           v_entry.org_id, v_entry.dept_id, v_entry.owner_principal_id, v_entry.access_scope,
           v_entry.allowed_group_ids, v_entry.arn, v_entry.min_auth_level, v_entry.denied_group_ids,
           'write', 'memory', v_entry.id
       ) THEN
        RAISE EXCEPTION 'RLS denied or unavailable memory.entries %', p_entry_id
            USING ERRCODE = '42501';
    END IF;

    IF p_new_version IS NULL OR p_new_version <= v_entry.version THEN
        RAISE EXCEPTION 'non-monotonic version transition on memory.entries %: % -> %',
            p_entry_id, v_entry.version, p_new_version
            USING ERRCODE = '22023';
    END IF;

    IF p_superseded_by IS NOT NULL THEN
        IF NOT EXISTS (
            SELECT 1 FROM memory.entries AS me
            WHERE me.id = p_superseded_by
              AND me.org_id = p_org_id
              AND me.status = 'active'
              AND iam.rls_can_read(
                  me.org_id, me.dept_id, me.owner_principal_id, me.access_scope,
                  me.allowed_group_ids, me.arn, me.min_auth_level, me.denied_group_ids,
                  'memory', me.id
              )
        ) THEN
            RAISE EXCEPTION 'invalid superseded_by target %', p_superseded_by
                USING ERRCODE = '42501';
        END IF;
    END IF;

    UPDATE memory.entries
    SET status = p_new_status,
        valid_to = now(),
        superseded_by = COALESCE(p_superseded_by, superseded_by),
        effective_text = COALESCE(p_effective_text, effective_text),
        bm25_text = COALESCE(p_bm25_text, bm25_text),
        content_sha256 = COALESCE(p_content_sha256, content_sha256),
        version = p_new_version,
        updated_at = now()
    WHERE org_id = p_org_id AND id = p_entry_id;
END;
$$;
