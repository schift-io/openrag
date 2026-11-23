-- OpenRAG Migration 009: One-time CLI authorization codes
--
-- `POST /v1/auth/cli/code-exchange` mints an API key only for a code that an org admin
-- (`POST /api/studio/iam/cli-codes`) or the local database operator (`openrag auth code`)
-- issued beforehand. Codes are stored as SHA-256 hashes, expire quickly and are single-use;
-- the ceilings recorded here can only be narrowed at exchange time, never widened.
--
-- Like iam.api_keys, this table is read and written through the worker pool only.
-- Idempotent so it can be applied on databases migrated with plain psql.

CREATE TABLE IF NOT EXISTS iam.cli_auth_codes (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id              UUID NOT NULL REFERENCES iam.orgs(id) ON DELETE CASCADE,
    principal_id        UUID NOT NULL,
    issued_by           UUID,
    code_hash           TEXT NOT NULL UNIQUE,
    max_auth_level      SMALLINT CHECK (max_auth_level BETWEEN 1 AND 10),
    scoped_arns         TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
    scoped_group_ids    UUID[] NOT NULL DEFAULT ARRAY[]::UUID[],
    expires_at          TIMESTAMPTZ NOT NULL,
    used_at             TIMESTAMPTZ,
    used_key_id         UUID,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    CHECK (expires_at > created_at),
    FOREIGN KEY (org_id, principal_id) REFERENCES iam.principals(org_id, id) ON DELETE CASCADE,
    FOREIGN KEY (org_id, issued_by) REFERENCES iam.principals(org_id, id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_cli_auth_codes_principal
    ON iam.cli_auth_codes (org_id, principal_id, created_at DESC);
