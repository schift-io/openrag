-- OpenRAG Migration 003: PII Capsule Vault + Configurable PII & Safe-Word Guard

CREATE TABLE pii.detector_profiles (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id              UUID NOT NULL REFERENCES iam.orgs(id) ON DELETE CASCADE,
    name                TEXT NOT NULL,
    use_schift_v7       BOOLEAN NOT NULL DEFAULT true,
    use_marker_ko_pii   BOOLEAN NOT NULL DEFAULT false,
    use_schift_v6_org   BOOLEAN NOT NULL DEFAULT false,
    custom_dict_terms   TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
    allowlist_terms     TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
    is_default          BOOLEAN NOT NULL DEFAULT false,
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (org_id, name)
);

CREATE TABLE pii.vault_keys (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id          UUID NOT NULL REFERENCES iam.orgs(id) ON DELETE CASCADE,
    key_version     INT NOT NULL CHECK (key_version >= 1),
    wrapped_dek     BYTEA NOT NULL,
    algorithm       TEXT NOT NULL DEFAULT 'AES-256-GCM',
    status          TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'rotated', 'revoked')),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (org_id, key_version),
    UNIQUE (org_id, id)
);

CREATE TABLE pii.vault_entries (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id          UUID NOT NULL REFERENCES iam.orgs(id) ON DELETE CASCADE,
    vault_key_id    UUID NOT NULL,
    token           TEXT NOT NULL,
    entity_type     TEXT NOT NULL,
    ciphertext      BYTEA NOT NULL,
    nonce           BYTEA NOT NULL,
    detector_source TEXT NOT NULL,
    detector_trace  JSONB,
    purged_at       TIMESTAMPTZ,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (org_id, token),
    UNIQUE (org_id, id),
    FOREIGN KEY (org_id, vault_key_id) REFERENCES pii.vault_keys(org_id, id) ON DELETE RESTRICT
);

CREATE TABLE pii.chunk_spans (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id          UUID NOT NULL REFERENCES iam.orgs(id) ON DELETE CASCADE,
    chunk_id        UUID NOT NULL,
    page_id         UUID,
    vault_entry_id  UUID,
    entity_type     TEXT NOT NULL,
    start_offset    INT NOT NULL CHECK (start_offset >= 0),
    end_offset      INT NOT NULL CHECK (end_offset >= start_offset),
    bbox_norm_1000  INT[4],
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    FOREIGN KEY (org_id, chunk_id) REFERENCES corpus.chunks(org_id, id) ON DELETE CASCADE,
    FOREIGN KEY (org_id, page_id) REFERENCES corpus.pages(org_id, id) ON DELETE CASCADE,
    FOREIGN KEY (org_id, vault_entry_id) REFERENCES pii.vault_entries(org_id, id) ON DELETE SET NULL (vault_entry_id)
);

CREATE TABLE pii.audit_logs (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id          UUID NOT NULL REFERENCES iam.orgs(id) ON DELETE CASCADE,
    principal_id    UUID,
    action          TEXT NOT NULL,
    resource_arn    TEXT,
    vault_entry_id  UUID,
    detector_source TEXT,
    allowed         BOOLEAN NOT NULL,
    reason          TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    FOREIGN KEY (org_id, principal_id) REFERENCES iam.principals(org_id, id) ON DELETE SET NULL (principal_id),
    FOREIGN KEY (org_id, vault_entry_id) REFERENCES pii.vault_entries(org_id, id) ON DELETE SET NULL (vault_entry_id)
);

-- Configurable PII & Safe-Word Guard (allow | warn | mask | block)
CREATE TABLE pii.guard_policies (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id          UUID NOT NULL REFERENCES iam.orgs(id) ON DELETE CASCADE,
    bucket_id       UUID,
    name            TEXT NOT NULL,
    master_enabled  BOOLEAN NOT NULL DEFAULT true,
    default_action  TEXT NOT NULL DEFAULT 'mask'
                    CHECK (default_action IN ('allow', 'warn', 'mask', 'block')),
    allowlist_terms TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
    surfaces        TEXT[] NOT NULL DEFAULT ARRAY['inbound_query', 'llm_egress', 'mcp_egress', 'ingestion_gate']::TEXT[],
    is_active       BOOLEAN NOT NULL DEFAULT true,
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (org_id, name),
    UNIQUE (org_id, id),
    FOREIGN KEY (org_id, bucket_id) REFERENCES corpus.buckets(org_id, id) ON DELETE CASCADE
);

CREATE TABLE pii.guard_rules (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    policy_id           UUID NOT NULL,
    org_id              UUID NOT NULL REFERENCES iam.orgs(id) ON DELETE CASCADE,
    rule_kind           TEXT NOT NULL CHECK (rule_kind IN ('pii_category', 'safe_word', 'keyword', 'regex')),
    pattern_or_category TEXT NOT NULL,
    action              TEXT NOT NULL CHECK (action IN ('allow', 'warn', 'mask', 'block')),
    surfaces            TEXT[] NOT NULL DEFAULT ARRAY['inbound_query', 'llm_egress', 'mcp_egress', 'ingestion_gate']::TEXT[],
    replacement_text    TEXT,
    priority            INT NOT NULL DEFAULT 100,
    enabled             BOOLEAN NOT NULL DEFAULT true,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    FOREIGN KEY (org_id, policy_id) REFERENCES pii.guard_policies(org_id, id) ON DELETE CASCADE
);

-- Guard audit events (metadata only; never stores raw blocked PII/safe-words)
CREATE TABLE pii.guard_events (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id              UUID NOT NULL REFERENCES iam.orgs(id) ON DELETE CASCADE,
    policy_id           UUID,
    principal_id        UUID,
    surface             TEXT NOT NULL CHECK (surface IN ('inbound_query', 'llm_egress', 'mcp_egress', 'ingestion_gate')),
    decision            TEXT NOT NULL CHECK (decision IN ('allow', 'warn', 'mask', 'block')),
    matched_rule_ids    UUID[] NOT NULL DEFAULT ARRAY[]::UUID[],
    matched_categories  TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
    metadata            JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    FOREIGN KEY (org_id, policy_id) REFERENCES pii.guard_policies(org_id, id) ON DELETE SET NULL (policy_id),
    FOREIGN KEY (org_id, principal_id) REFERENCES iam.principals(org_id, id) ON DELETE SET NULL (principal_id)
);

CREATE INDEX idx_pii_chunk_spans_chunk ON pii.chunk_spans (org_id, chunk_id);
CREATE INDEX idx_pii_guard_rules_policy ON pii.guard_rules (policy_id, enabled, priority);
CREATE INDEX idx_pii_guard_events_org ON pii.guard_events (org_id, surface, created_at DESC);

-- SECURITY DEFINER helpers for pii.audit_logs & pii.guard_events RLS (eliminates 23503 cross-tenant existence oracles)
CREATE OR REPLACE FUNCTION iam.vault_entry_belongs_to_org(
    p_org_id UUID,
    p_vault_entry_id UUID
)
RETURNS BOOLEAN
LANGUAGE sql
STABLE
SECURITY DEFINER
SET search_path = pii, iam, pg_catalog
AS $$
    SELECT p_org_id IS NOT NULL
       AND p_vault_entry_id IS NOT NULL
       AND EXISTS (
           SELECT 1
           FROM pii.vault_entries AS ve
           WHERE ve.id = p_vault_entry_id
             AND ve.org_id = p_org_id
       );
$$;

CREATE OR REPLACE FUNCTION iam.guard_policy_belongs_to_org(
    p_org_id UUID,
    p_policy_id UUID
)
RETURNS BOOLEAN
LANGUAGE sql
STABLE
SECURITY DEFINER
SET search_path = pii, iam, pg_catalog
AS $$
    SELECT p_org_id IS NOT NULL
       AND p_policy_id IS NOT NULL
       AND EXISTS (
           SELECT 1
           FROM pii.guard_policies AS gp
           WHERE gp.id = p_policy_id
             AND gp.org_id = p_org_id
       );
$$;

-- Fail-Closed RLS on openrag_app-accessible PII audit & guard event tables
ALTER TABLE pii.audit_logs   ENABLE ROW LEVEL SECURITY;
ALTER TABLE pii.audit_logs   FORCE ROW LEVEL SECURITY;
ALTER TABLE pii.guard_events ENABLE ROW LEVEL SECURITY;
ALTER TABLE pii.guard_events FORCE ROW LEVEL SECURITY;

CREATE POLICY rls_pii_audit_logs_read ON pii.audit_logs FOR SELECT
    USING (EXISTS (
        SELECT 1 FROM iam.session_active_principal() AS s
        WHERE s.v_ok
          AND pii.audit_logs.org_id = s.v_org_id
          AND (
              pii.audit_logs.principal_id = s.v_principal_id
              OR s.v_is_org_admin
          )
    ));

CREATE POLICY rls_pii_audit_logs_insert ON pii.audit_logs FOR INSERT
    WITH CHECK (EXISTS (
        SELECT 1 FROM iam.session_active_principal() AS s
        WHERE s.v_ok
          AND pii.audit_logs.org_id = s.v_org_id
          AND pii.audit_logs.principal_id = s.v_principal_id
          AND (
              pii.audit_logs.vault_entry_id IS NULL
              OR iam.vault_entry_belongs_to_org(pii.audit_logs.org_id, pii.audit_logs.vault_entry_id)
          )
    ));

CREATE POLICY rls_pii_guard_events_read ON pii.guard_events FOR SELECT
    USING (EXISTS (
        SELECT 1 FROM iam.session_active_principal() AS s
        WHERE s.v_ok
          AND pii.guard_events.org_id = s.v_org_id
          AND (
              pii.guard_events.principal_id = s.v_principal_id
              OR s.v_is_org_admin
          )
    ));

CREATE POLICY rls_pii_guard_events_insert ON pii.guard_events FOR INSERT
    WITH CHECK (EXISTS (
        SELECT 1 FROM iam.session_active_principal() AS s
        WHERE s.v_ok
          AND pii.guard_events.org_id = s.v_org_id
          AND pii.guard_events.principal_id = s.v_principal_id
          AND (
              pii.guard_events.policy_id IS NULL
              OR iam.guard_policy_belongs_to_org(pii.guard_events.org_id, pii.guard_events.policy_id)
          )
    ));
