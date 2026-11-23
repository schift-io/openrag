-- OpenRAG Migration 005: Policy LLM Gateway & Plugin Connectors

CREATE TABLE gateway.providers (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id              UUID NOT NULL REFERENCES iam.orgs(id) ON DELETE CASCADE,
    name                TEXT NOT NULL,
    provider_type       TEXT NOT NULL CHECK (provider_type IN ('openrouter', 'openai', 'vllm_local', 'schift_ocr', 'mock')),
    base_url            TEXT NOT NULL,
    model_id            TEXT NOT NULL,
    is_on_premise       BOOLEAN NOT NULL DEFAULT false,
    max_auth_level      SMALLINT NOT NULL DEFAULT 5 CHECK (max_auth_level BETWEEN 1 AND 10),
    allow_unmasked_pii  BOOLEAN NOT NULL DEFAULT false,
    config              JSONB NOT NULL DEFAULT '{}'::jsonb,
    enabled             BOOLEAN NOT NULL DEFAULT true,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (org_id, name),
    UNIQUE (org_id, id)
);

CREATE TABLE gateway.routing_rules (
    id                   UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id               UUID NOT NULL REFERENCES iam.orgs(id) ON DELETE CASCADE,
    priority             INT NOT NULL DEFAULT 100,
    min_auth_level       SMALLINT NOT NULL DEFAULT 1 CHECK (min_auth_level BETWEEN 1 AND 10),
    max_auth_level       SMALLINT NOT NULL DEFAULT 10 CHECK (max_auth_level BETWEEN 1 AND 10),
    require_pii_masked   BOOLEAN NOT NULL DEFAULT true,
    target_provider_id   UUID NOT NULL,
    fallback_provider_id UUID,
    enabled              BOOLEAN NOT NULL DEFAULT true,
    CHECK (min_auth_level <= max_auth_level),
    FOREIGN KEY (org_id, target_provider_id) REFERENCES gateway.providers(org_id, id) ON DELETE CASCADE,
    FOREIGN KEY (org_id, fallback_provider_id) REFERENCES gateway.providers(org_id, id) ON DELETE SET NULL (fallback_provider_id)
);

CREATE TABLE gateway.request_logs (
    id                      UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id                  UUID NOT NULL REFERENCES iam.orgs(id) ON DELETE CASCADE,
    principal_id            UUID,
    provider_id             UUID,
    model_id                TEXT NOT NULL,
    max_context_auth_level  SMALLINT CHECK (max_context_auth_level BETWEEN 1 AND 10),
    pii_masked              BOOLEAN NOT NULL,
    prompt_tokens           INT NOT NULL DEFAULT 0,
    completion_tokens       INT NOT NULL DEFAULT 0,
    total_tokens            INT NOT NULL DEFAULT 0,
    cost_usd                NUMERIC(12, 6) NOT NULL DEFAULT 0,
    latency_ms              INT NOT NULL DEFAULT 0,
    status                  TEXT NOT NULL,
    created_at              TIMESTAMPTZ NOT NULL DEFAULT now(),
    FOREIGN KEY (org_id, principal_id) REFERENCES iam.principals(org_id, id) ON DELETE SET NULL (principal_id),
    FOREIGN KEY (org_id, provider_id) REFERENCES gateway.providers(org_id, id) ON DELETE SET NULL (provider_id)
);

CREATE TABLE plugins.connectors (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id              UUID NOT NULL REFERENCES iam.orgs(id) ON DELETE CASCADE,
    connector_type      TEXT NOT NULL CHECK (connector_type IN ('kordoc_hook', 'docufinder_sync', 'schift_index', 'ai_memory', 'git_wiki')),
    name                TEXT NOT NULL,
    config              JSONB NOT NULL DEFAULT '{}'::jsonb,
    status              TEXT NOT NULL DEFAULT 'healthy' CHECK (status IN ('healthy', 'degraded', 'offline', 'paused')),
    last_heartbeat_at   TIMESTAMPTZ,
    last_error          TEXT,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (org_id, name),
    UNIQUE (org_id, id)
);

-- Note: memory_episode_id FK to memory.episodes is added in 007_memory_redaction_indexing.sql
CREATE TABLE plugins.source_entries (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id              UUID NOT NULL REFERENCES iam.orgs(id) ON DELETE CASCADE,
    connector_id        UUID NOT NULL,
    external_key        TEXT NOT NULL,
    content_sha256      TEXT NOT NULL,
    document_id         UUID,
    memory_episode_id   UUID,
    synced_at           TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (connector_id, external_key),
    FOREIGN KEY (org_id, connector_id) REFERENCES plugins.connectors(org_id, id) ON DELETE CASCADE,
    FOREIGN KEY (org_id, document_id) REFERENCES corpus.documents(org_id, id) ON DELETE SET NULL (document_id)
);

CREATE INDEX idx_gateway_logs_org_created ON gateway.request_logs (org_id, created_at DESC);
CREATE INDEX idx_plugins_source_entries_org ON plugins.source_entries (org_id, connector_id);
