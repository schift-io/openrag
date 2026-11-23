-- OpenRAG Migration 006: 4-Axis Golden Sets, AutoRAG Optimization Trials, 3-Track Bench & Security Gates

CREATE TABLE eval.pipeline_configs (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id          UUID NOT NULL REFERENCES iam.orgs(id) ON DELETE CASCADE,
    name            TEXT NOT NULL,
    profile         TEXT NOT NULL DEFAULT 'standard' CHECK (profile IN ('standard', 'airgap', 'custom')),
    pipeline_yaml   TEXT NOT NULL,
    config_json     JSONB NOT NULL,
    is_promoted     BOOLEAN NOT NULL DEFAULT false,
    promoted_at     TIMESTAMPTZ,
    created_by      UUID,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (org_id, name),
    UNIQUE (org_id, id),
    FOREIGN KEY (org_id, created_by) REFERENCES iam.principals(org_id, id) ON DELETE SET NULL (created_by)
);

-- Real implementation replacing stub from 001_iam_arn.sql
CREATE OR REPLACE FUNCTION iam.pipeline_config_belongs_to_org(
    p_org_id UUID,
    p_pipeline_config_id UUID
)
RETURNS BOOLEAN
LANGUAGE sql
STABLE
SECURITY DEFINER
SET search_path = eval, iam, pg_catalog
AS $$
    SELECT p_org_id IS NOT NULL
       AND p_pipeline_config_id IS NOT NULL
       AND EXISTS (
           SELECT 1
           FROM eval.pipeline_configs AS pc
           WHERE pc.id = p_pipeline_config_id
             AND pc.org_id = p_org_id
       );
$$;

-- Deferred composite FK from corpus.buckets (created in 002)
ALTER TABLE corpus.buckets
    ADD CONSTRAINT fk_corpus_buckets_pipeline_config
    FOREIGN KEY (org_id, pipeline_config_id) REFERENCES eval.pipeline_configs(org_id, id) ON DELETE SET NULL (pipeline_config_id);

CREATE TABLE eval.golden_sets (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id      UUID NOT NULL REFERENCES iam.orgs(id) ON DELETE CASCADE,
    name        TEXT NOT NULL,
    axis        TEXT NOT NULL CHECK (axis IN ('retrieval', 'ocr', 'pii', 'pararel', 'multi')),
    description TEXT,
    version     INT NOT NULL DEFAULT 1,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (org_id, name),
    UNIQUE (org_id, id)
);

CREATE TABLE eval.golden_items (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id              UUID NOT NULL REFERENCES iam.orgs(id) ON DELETE CASCADE,
    golden_set_id       UUID NOT NULL,
    axis                TEXT NOT NULL CHECK (axis IN ('retrieval', 'ocr', 'pii', 'pararel')),
    query_text          TEXT NOT NULL,
    test_principal_id   UUID,
    expected_doc_ids    UUID[] NOT NULL DEFAULT ARRAY[]::UUID[],
    forbidden_doc_ids   UUID[] NOT NULL DEFAULT ARRAY[]::UUID[],
    expected_answer     TEXT,
    expected_pii_tokens TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
    expected_edges      JSONB,
    review_status       TEXT NOT NULL DEFAULT 'approved' CHECK (review_status IN ('pending', 'approved', 'rejected')),
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (org_id, id),
    FOREIGN KEY (org_id, golden_set_id) REFERENCES eval.golden_sets(org_id, id) ON DELETE CASCADE,
    FOREIGN KEY (org_id, test_principal_id) REFERENCES iam.principals(org_id, id) ON DELETE SET NULL (test_principal_id)
);

CREATE TABLE eval.optimization_trials (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id              UUID NOT NULL REFERENCES iam.orgs(id) ON DELETE CASCADE,
    golden_set_id       UUID NOT NULL,
    search_space_yaml   TEXT NOT NULL,
    strategy            TEXT NOT NULL CHECK (strategy IN ('greedy_node', 'grid', 'ab_pair')),
    best_config_yaml    TEXT,
    best_score          REAL,
    security_gate_pass  BOOLEAN NOT NULL DEFAULT false,
    status              TEXT NOT NULL DEFAULT 'running' CHECK (status IN ('running', 'completed', 'blocked_security_gate', 'failed')),
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    completed_at        TIMESTAMPTZ,
    UNIQUE (org_id, id),
    FOREIGN KEY (org_id, golden_set_id) REFERENCES eval.golden_sets(org_id, id) ON DELETE CASCADE
);

CREATE TABLE eval.bench_runs (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id              UUID NOT NULL REFERENCES iam.orgs(id) ON DELETE CASCADE,
    golden_set_id       UUID NOT NULL,
    pipeline_config_id  UUID,
    trial_id            UUID,
    track               TEXT NOT NULL CHECK (track IN ('track1_retrieval', 'track2_e2e_rag', 'track3_security_ocr')),
    metrics             JSONB NOT NULL DEFAULT '{}'::jsonb,
    pii_leak_count      INT NOT NULL DEFAULT 0,
    arn_violation_count INT NOT NULL DEFAULT 0,
    security_gate_pass  BOOLEAN GENERATED ALWAYS AS (pii_leak_count = 0 AND arn_violation_count = 0) STORED,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (org_id, id),
    FOREIGN KEY (org_id, golden_set_id) REFERENCES eval.golden_sets(org_id, id) ON DELETE CASCADE,
    FOREIGN KEY (org_id, pipeline_config_id) REFERENCES eval.pipeline_configs(org_id, id) ON DELETE SET NULL (pipeline_config_id),
    FOREIGN KEY (org_id, trial_id) REFERENCES eval.optimization_trials(org_id, id) ON DELETE SET NULL (trial_id)
);

CREATE TABLE eval.bench_item_results (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    bench_run_id        UUID NOT NULL REFERENCES eval.bench_runs(id) ON DELETE CASCADE,
    golden_item_id      UUID NOT NULL REFERENCES eval.golden_items(id) ON DELETE CASCADE,
    retrieved_doc_ids   UUID[] NOT NULL DEFAULT ARRAY[]::UUID[],
    latency_ms          INT NOT NULL DEFAULT 0,
    score               REAL NOT NULL DEFAULT 0,
    pii_leaked          BOOLEAN NOT NULL DEFAULT false,
    arn_violated        BOOLEAN NOT NULL DEFAULT false,
    details             JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE TABLE eval.search_feedback (
    id                      UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id                  UUID NOT NULL REFERENCES iam.orgs(id) ON DELETE CASCADE,
    principal_id            UUID NOT NULL,
    query_text              TEXT NOT NULL,
    response_text           TEXT,
    cited_chunk_ids         UUID[] NOT NULL DEFAULT ARRAY[]::UUID[],
    verdict                 TEXT NOT NULL CHECK (verdict IN ('up', 'down', 'wrong_evidence', 'missing_citation')),
    comment                 TEXT,
    promoted_golden_item_id UUID,
    created_at              TIMESTAMPTZ NOT NULL DEFAULT now(),
    FOREIGN KEY (org_id, principal_id) REFERENCES iam.principals(org_id, id) ON DELETE CASCADE,
    FOREIGN KEY (org_id, promoted_golden_item_id) REFERENCES eval.golden_items(org_id, id) ON DELETE SET NULL (promoted_golden_item_id)
);

CREATE INDEX idx_eval_golden_items_set ON eval.golden_items (org_id, golden_set_id, axis);
CREATE INDEX idx_eval_bench_runs_org ON eval.bench_runs (org_id, track, created_at DESC);

-- SECURITY DEFINER helper for eval.search_feedback RLS (eliminates 23503 cross-tenant existence oracle)
CREATE OR REPLACE FUNCTION iam.golden_item_belongs_to_org(
    p_org_id UUID,
    p_item_id UUID
)
RETURNS BOOLEAN
LANGUAGE sql
STABLE
SECURITY DEFINER
SET search_path = eval, iam, pg_catalog
AS $$
    SELECT p_org_id IS NOT NULL
       AND p_item_id IS NOT NULL
       AND EXISTS (
           SELECT 1
           FROM eval.golden_items AS gi
           WHERE gi.id = p_item_id
             AND gi.org_id = p_org_id
       );
$$;

-- Fail-Closed RLS on openrag_app-accessible eval.search_feedback
ALTER TABLE eval.search_feedback ENABLE ROW LEVEL SECURITY;
ALTER TABLE eval.search_feedback FORCE ROW LEVEL SECURITY;

CREATE POLICY rls_eval_search_feedback_read ON eval.search_feedback FOR SELECT
    USING (EXISTS (
        SELECT 1 FROM iam.session_active_principal() AS s
        WHERE s.v_ok
          AND eval.search_feedback.org_id = s.v_org_id
          AND (
              eval.search_feedback.principal_id = s.v_principal_id
              OR s.v_is_org_admin
          )
    ));

CREATE POLICY rls_eval_search_feedback_insert ON eval.search_feedback FOR INSERT
    WITH CHECK (EXISTS (
        SELECT 1 FROM iam.session_active_principal() AS s
        WHERE s.v_ok
          AND eval.search_feedback.org_id = s.v_org_id
          AND eval.search_feedback.principal_id = s.v_principal_id
          AND (
              eval.search_feedback.promoted_golden_item_id IS NULL
              OR (
                  s.v_is_org_admin
                  AND iam.golden_item_belongs_to_org(
                      eval.search_feedback.org_id,
                      eval.search_feedback.promoted_golden_item_id
                  )
              )
          )
    ));
