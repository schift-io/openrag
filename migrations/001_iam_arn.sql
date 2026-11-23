-- OpenRAG Migration 001: 3-Axis IAM (ARN + Permission Groups + Permission Levels) & Fail-Closed RLS

CREATE TABLE iam.orgs (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    slug        TEXT NOT NULL UNIQUE,
    name        TEXT NOT NULL,
    settings    JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE iam.departments (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id      UUID NOT NULL REFERENCES iam.orgs(id) ON DELETE CASCADE,
    parent_id   UUID,
    name        TEXT NOT NULL,
    arn_path    TEXT NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (org_id, arn_path),
    UNIQUE (org_id, id),
    FOREIGN KEY (org_id, parent_id) REFERENCES iam.departments(org_id, id) ON DELETE SET NULL (parent_id)
);

CREATE TABLE iam.principals (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id          UUID NOT NULL REFERENCES iam.orgs(id) ON DELETE CASCADE,
    external_sub    TEXT,
    email           TEXT NOT NULL,
    display_name    TEXT NOT NULL,
    auth_level      SMALLINT NOT NULL DEFAULT 3 CHECK (auth_level BETWEEN 1 AND 10),
    can_unmask_pii  BOOLEAN NOT NULL DEFAULT false,
    is_org_admin    BOOLEAN NOT NULL DEFAULT false,
    status          TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'suspended', 'disabled')),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (org_id, email),
    UNIQUE (org_id, id)
);

CREATE TABLE iam.dept_memberships (
    org_id          UUID NOT NULL REFERENCES iam.orgs(id) ON DELETE CASCADE,
    dept_id         UUID NOT NULL,
    principal_id    UUID NOT NULL,
    dept_role       TEXT NOT NULL CHECK (dept_role IN ('dept_admin', 'dept_reviewer', 'member')),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (dept_id, principal_id),
    FOREIGN KEY (org_id, dept_id) REFERENCES iam.departments(org_id, id) ON DELETE CASCADE,
    FOREIGN KEY (org_id, principal_id) REFERENCES iam.principals(org_id, id) ON DELETE CASCADE
);

-- First-Class Permission Groups (cross-department TF / project / functional / clearance)
CREATE TABLE iam.groups (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id          UUID NOT NULL REFERENCES iam.orgs(id) ON DELETE CASCADE,
    dept_id         UUID,
    name            TEXT NOT NULL,
    group_type      TEXT NOT NULL DEFAULT 'permission_group'
                    CHECK (group_type IN ('permission_group', 'project_tf', 'functional', 'clearance')),
    min_auth_level  SMALLINT NOT NULL DEFAULT 1 CHECK (min_auth_level BETWEEN 1 AND 10),
    max_auth_level  SMALLINT NOT NULL DEFAULT 10 CHECK (max_auth_level BETWEEN 1 AND 10),
    allowed_actions TEXT[] NOT NULL DEFAULT ARRAY['read', 'search']::TEXT[],
    description     TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    CHECK (min_auth_level <= max_auth_level),
    UNIQUE (org_id, name),
    UNIQUE (org_id, id),
    FOREIGN KEY (org_id, dept_id) REFERENCES iam.departments(org_id, id) ON DELETE SET NULL (dept_id)
);

CREATE TABLE iam.group_members (
    org_id              UUID NOT NULL REFERENCES iam.orgs(id) ON DELETE CASCADE,
    group_id            UUID NOT NULL,
    principal_id        UUID NOT NULL,
    member_role         TEXT NOT NULL DEFAULT 'member' CHECK (member_role IN ('admin', 'member')),
    action_overrides    TEXT[],
    level_override      SMALLINT CHECK (level_override BETWEEN 1 AND 10),
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (group_id, principal_id),
    FOREIGN KEY (org_id, group_id) REFERENCES iam.groups(org_id, id) ON DELETE CASCADE,
    FOREIGN KEY (org_id, principal_id) REFERENCES iam.principals(org_id, id) ON DELETE CASCADE
);

CREATE TABLE iam.api_keys (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id              UUID NOT NULL REFERENCES iam.orgs(id) ON DELETE CASCADE,
    principal_id        UUID NOT NULL,
    name                TEXT NOT NULL,
    key_prefix          TEXT NOT NULL,
    key_hash            TEXT NOT NULL UNIQUE,
    scoped_arns         TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
    scoped_group_ids    UUID[] NOT NULL DEFAULT ARRAY[]::UUID[],
    max_auth_level      SMALLINT CHECK (max_auth_level BETWEEN 1 AND 10),
    expires_at          TIMESTAMPTZ,
    revoked_at          TIMESTAMPTZ,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    FOREIGN KEY (org_id, principal_id) REFERENCES iam.principals(org_id, id) ON DELETE CASCADE
);

CREATE TABLE iam.policies (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id      UUID NOT NULL REFERENCES iam.orgs(id) ON DELETE CASCADE,
    name        TEXT NOT NULL,
    description TEXT,
    document    JSONB NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (org_id, name),
    UNIQUE (org_id, id)
);

CREATE TABLE iam.policy_attachments (
    id                      UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id                  UUID NOT NULL REFERENCES iam.orgs(id) ON DELETE CASCADE,
    policy_id               UUID NOT NULL,
    target_type             TEXT NOT NULL CHECK (target_type IN ('principal', 'dept', 'group', 'api_key')),
    target_id               UUID NOT NULL,
    min_auth_level_override SMALLINT CHECK (min_auth_level_override BETWEEN 1 AND 10),
    created_at              TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (policy_id, target_type, target_id),
    FOREIGN KEY (org_id, policy_id) REFERENCES iam.policies(org_id, id) ON DELETE CASCADE
);

CREATE TABLE iam.resource_group_bindings (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id          UUID NOT NULL REFERENCES iam.orgs(id) ON DELETE CASCADE,
    resource_type   TEXT NOT NULL CHECK (resource_type IN ('bucket', 'document', 'page', 'figure', 'chunk', 'memory', 'episode', 'wiki')),
    resource_id     UUID NOT NULL,
    group_id        UUID NOT NULL,
    effect          TEXT NOT NULL CHECK (effect IN ('allow', 'deny')),
    actions         TEXT[] NOT NULL DEFAULT ARRAY['read', 'search']::TEXT[],
    min_auth_level  SMALLINT CHECK (min_auth_level BETWEEN 1 AND 10),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (resource_type, resource_id, group_id, effect),
    FOREIGN KEY (org_id, group_id) REFERENCES iam.groups(org_id, id) ON DELETE CASCADE
);

CREATE INDEX idx_iam_dept_memberships_principal ON iam.dept_memberships (principal_id, org_id);
CREATE INDEX idx_iam_group_members_principal ON iam.group_members (principal_id, org_id);
CREATE INDEX idx_iam_policy_attachments_target ON iam.policy_attachments (org_id, target_type, target_id);
CREATE INDEX idx_iam_rgb_resource ON iam.resource_group_bindings (org_id, resource_type, resource_id, effect);

-- Glob '*' ARN matcher
CREATE OR REPLACE FUNCTION iam.arn_matches(p_pattern TEXT, p_arn TEXT)
RETURNS BOOLEAN
LANGUAGE plpgsql
IMMUTABLE
AS $$
DECLARE
    v_escaped TEXT;
BEGIN
    IF p_pattern IS NULL OR p_arn IS NULL OR p_pattern = '' OR p_arn = '' THEN
        RETURN false;
    END IF;
    IF p_pattern = '*' THEN
        RETURN true;
    END IF;
    -- Escape regex metacharacters except '*', then replace '*' with '.*'
    v_escaped := regexp_replace(p_pattern, '([.+?^${}()|\[\]\\])', '\\\1', 'g');
    v_escaped := replace(v_escaped, '*', '.*');
    RETURN p_arn ~ ('^' || v_escaped || '$');
EXCEPTION WHEN OTHERS THEN
    RETURN false;
END;
$$;

-- Helper: evaluate whether a policy condition JSONB matches
CREATE OR REPLACE FUNCTION iam.policy_condition_matches(
    p_condition JSONB,
    p_effective_level SMALLINT,
    p_group_ids UUID[],
    p_access_scope TEXT
)
RETURNS BOOLEAN
LANGUAGE plpgsql
IMMUTABLE
AS $$
DECLARE
    v_cond_min SMALLINT;
    v_cond_max SMALLINT;
    v_req_groups UUID[];
    v_scopes TEXT[];
BEGIN
    IF p_condition IS NULL OR p_condition = 'null'::jsonb THEN
        RETURN true;
    END IF;

    IF p_condition ? 'min_auth_level' AND p_condition->>'min_auth_level' IS NOT NULL THEN
        v_cond_min := (p_condition->>'min_auth_level')::SMALLINT;
        IF p_effective_level < v_cond_min THEN
            RETURN false;
        END IF;
    END IF;

    IF p_condition ? 'max_auth_level' AND p_condition->>'max_auth_level' IS NOT NULL THEN
        v_cond_max := (p_condition->>'max_auth_level')::SMALLINT;
        IF p_effective_level > v_cond_max THEN
            RETURN false;
        END IF;
    END IF;

    IF p_condition ? 'required_group_ids' AND jsonb_typeof(p_condition->'required_group_ids') = 'array' THEN
        SELECT COALESCE(array_agg(elem::UUID), ARRAY[]::UUID[])
        INTO v_req_groups
        FROM jsonb_array_elements_text(p_condition->'required_group_ids') AS elem;
        IF cardinality(v_req_groups) > 0 AND NOT (p_group_ids && v_req_groups) THEN
            RETURN false;
        END IF;
    END IF;

    IF p_condition ? 'allowed_scopes' AND jsonb_typeof(p_condition->'allowed_scopes') = 'array' THEN
        SELECT COALESCE(array_agg(elem), ARRAY[]::TEXT[])
        INTO v_scopes
        FROM jsonb_array_elements_text(p_condition->'allowed_scopes') AS elem;
        IF cardinality(v_scopes) > 0 AND NOT (p_access_scope = ANY(v_scopes)) THEN
            RETURN false;
        END IF;
    END IF;

    RETURN true;
EXCEPTION WHEN OTHERS THEN
    RETURN false;
END;
$$;

-- Core Fail-Closed 3-Axis Access Control (100% parity with openrag-iam)
CREATE OR REPLACE FUNCTION iam.rls_can_access(
    p_org_id UUID,
    p_dept_id UUID,
    p_owner_principal_id UUID,
    p_access_scope TEXT,
    p_allowed_group_ids UUID[],
    p_arn TEXT,
    p_min_auth_level SMALLINT,
    p_denied_group_ids UUID[] DEFAULT ARRAY[]::UUID[],
    p_action TEXT DEFAULT 'read',
    p_resource_type TEXT DEFAULT NULL,
    p_resource_id UUID DEFAULT NULL
)
RETURNS BOOLEAN
LANGUAGE plpgsql
STABLE
SECURITY DEFINER
SET search_path = iam, pg_catalog
AS $$
DECLARE
    v_org_id UUID;
    v_principal_id UUID;
    v_auth_level SMALLINT;
    v_dept_ids UUID[];
    v_dept_admin_ids UUID[];
    v_group_ids UUID[];
    v_guc_scoped_groups UUID[];
    v_deny_group_ids UUID[];
    v_allow_group_ids UUID[];
    v_api_key_raw TEXT;
    v_api_key_id UUID;
    v_scoped_arns TEXT[];

    v_p_auth_level SMALLINT;
    v_can_unmask_pii BOOLEAN;
    v_is_org_admin BOOLEAN;
    v_status TEXT;
    v_effective_level SMALLINT;

    v_ak_max_level SMALLINT;
    v_ak_scoped_groups UUID[];
    v_ak_scoped_arns TEXT[];
    v_ak_expires TIMESTAMPTZ;
    v_ak_revoked TIMESTAMPTZ;

    v_res_min_level SMALLINT;
    v_arn_matched BOOLEAN;
    v_has_explicit_deny BOOLEAN := false;
    v_group_allow BOOLEAN := false;
    v_arn_allow BOOLEAN := false;
BEGIN
    -- 1. Session & Identity Fail-Closed Validation
    v_org_id := iam.safe_uuid(current_setting('app.org_id', true));
    v_principal_id := iam.safe_uuid(current_setting('app.principal_id', true));
    v_auth_level := iam.safe_smallint(current_setting('app.auth_level', true), 0::SMALLINT);
    v_dept_ids := iam.safe_uuid_array(current_setting('app.dept_ids', true));
    v_dept_admin_ids := iam.safe_uuid_array(current_setting('app.dept_admin_ids', true));
    v_group_ids := iam.safe_uuid_array(current_setting('app.group_ids', true));
    v_guc_scoped_groups := iam.safe_uuid_array(current_setting('app.scoped_group_ids', true));
    v_api_key_raw := NULLIF(btrim(current_setting('app.api_key_id', true)), '');
    IF v_api_key_raw IS NOT NULL THEN
        v_api_key_id := iam.safe_uuid(v_api_key_raw);
        IF v_api_key_id IS NULL THEN
            RETURN false;
        END IF;
    ELSE
        v_api_key_id := NULL;
    END IF;
    v_scoped_arns := iam.safe_text_array(current_setting('app.scoped_arns', true));

    IF v_org_id IS NULL OR v_principal_id IS NULL OR p_org_id IS NULL OR v_org_id <> p_org_id THEN
        RETURN false;
    END IF;

    IF p_access_scope IS NULL OR p_action IS NULL THEN
        RETURN false;
    END IF;

    IF v_auth_level IS NULL OR v_auth_level < 1 OR v_auth_level > 10 THEN
        RETURN false;
    END IF;

    IF v_dept_ids IS NULL OR v_dept_admin_ids IS NULL OR v_group_ids IS NULL OR v_guc_scoped_groups IS NULL OR v_scoped_arns IS NULL THEN
        RETURN false;
    END IF;

    SELECT p.auth_level, p.can_unmask_pii, p.is_org_admin, p.status
    INTO v_p_auth_level, v_can_unmask_pii, v_is_org_admin, v_status
    FROM iam.principals AS p
    WHERE p.id = v_principal_id AND p.org_id = v_org_id;

    IF NOT FOUND OR v_status <> 'active' THEN
        RETURN false;
    END IF;

    v_effective_level := LEAST(v_auth_level, v_p_auth_level);

    -- Full group set for Explicit Deny (session GUCs union ALL iam.group_members for principal; NEVER filtered by scoped_group_ids)
    SELECT COALESCE(array_agg(DISTINCT g_id), ARRAY[]::UUID[])
    INTO v_deny_group_ids
    FROM (
        SELECT unnest(v_group_ids) AS g_id
        UNION
        SELECT gm.group_id AS g_id
        FROM iam.group_members AS gm
        WHERE gm.org_id = v_org_id AND gm.principal_id = v_principal_id
    ) AS all_groups
    WHERE g_id IS NOT NULL;

    -- Narrowed group set for Allow only (may be intersected with app.scoped_group_ids / api_keys.scoped_group_ids)
    v_allow_group_ids := v_group_ids;
    IF cardinality(v_guc_scoped_groups) > 0 THEN
        SELECT COALESCE(array_agg(g_id), ARRAY[]::UUID[])
        INTO v_allow_group_ids
        FROM unnest(v_allow_group_ids) AS g_id
        WHERE g_id = ANY(v_guc_scoped_groups);
    END IF;

    -- Enforce API key ceilings if api_key_id is bound
    IF v_api_key_id IS NOT NULL THEN
        SELECT ak.max_auth_level, ak.scoped_group_ids, ak.scoped_arns, ak.expires_at, ak.revoked_at
        INTO v_ak_max_level, v_ak_scoped_groups, v_ak_scoped_arns, v_ak_expires, v_ak_revoked
        FROM iam.api_keys AS ak
        WHERE ak.id = v_api_key_id AND ak.org_id = v_org_id AND ak.principal_id = v_principal_id;

        IF NOT FOUND OR v_ak_revoked IS NOT NULL OR (v_ak_expires IS NOT NULL AND v_ak_expires <= now()) THEN
            RETURN false;
        END IF;

        IF v_ak_max_level IS NOT NULL THEN
            v_effective_level := LEAST(v_effective_level, v_ak_max_level);
        END IF;

        IF v_ak_scoped_groups IS NOT NULL AND cardinality(v_ak_scoped_groups) > 0 THEN
            SELECT COALESCE(array_agg(g_id), ARRAY[]::UUID[])
            INTO v_allow_group_ids
            FROM unnest(v_allow_group_ids) AS g_id
            WHERE g_id = ANY(v_ak_scoped_groups);
        END IF;

        IF v_ak_scoped_arns IS NOT NULL AND cardinality(v_ak_scoped_arns) > 0 THEN
            SELECT EXISTS (
                SELECT 1 FROM unnest(v_ak_scoped_arns) AS pat
                WHERE iam.arn_matches(pat, p_arn)
            ) INTO v_arn_matched;
            IF NOT COALESCE(v_arn_matched, false) THEN
                RETURN false;
            END IF;
        END IF;
    END IF;

    -- Also enforce session app.scoped_arns ceiling if provided
    IF cardinality(v_scoped_arns) > 0 THEN
        SELECT EXISTS (
            SELECT 1 FROM unnest(v_scoped_arns) AS pat
            WHERE iam.arn_matches(pat, p_arn)
        ) INTO v_arn_matched;
        IF NOT COALESCE(v_arn_matched, false) THEN
            RETURN false;
        END IF;
    END IF;

    IF p_action = 'pii:unmask' AND NOT COALESCE(v_can_unmask_pii, false) THEN
        RETURN false;
    END IF;

    -- 2. Explicit Deny > Allow (evaluated against full v_deny_group_ids)
    IF p_denied_group_ids IS NOT NULL AND cardinality(p_denied_group_ids) > 0 THEN
        IF v_deny_group_ids && p_denied_group_ids THEN
            RETURN false;
        END IF;
    END IF;

    IF p_resource_type IS NOT NULL AND p_resource_id IS NOT NULL THEN
        SELECT EXISTS (
            SELECT 1
            FROM iam.resource_group_bindings AS rgb
            WHERE rgb.org_id = v_org_id
              AND rgb.resource_type = p_resource_type
              AND rgb.resource_id = p_resource_id
              AND rgb.effect = 'deny'
              AND rgb.group_id = ANY(v_deny_group_ids)
              AND (p_action = ANY(rgb.actions) OR '*' = ANY(rgb.actions))
        ) INTO v_has_explicit_deny;

        IF COALESCE(v_has_explicit_deny, false) THEN
            RETURN false;
        END IF;
    END IF;

    SELECT EXISTS (
        SELECT 1
        FROM iam.policy_attachments AS pa
        JOIN iam.policies AS pol ON pol.id = pa.policy_id
        CROSS JOIN LATERAL jsonb_array_elements(
            CASE
                WHEN jsonb_typeof(pol.document->'statements') = 'array' THEN pol.document->'statements'
                ELSE '[]'::jsonb
            END
        ) AS stmt
        WHERE pa.org_id = v_org_id
          AND pol.org_id = v_org_id
          AND (
              (pa.target_type = 'principal' AND pa.target_id = v_principal_id) OR
              (pa.target_type = 'dept' AND pa.target_id = ANY(v_dept_ids)) OR
              (pa.target_type = 'group' AND pa.target_id = ANY(v_deny_group_ids)) OR
              (pa.target_type = 'api_key' AND v_api_key_id IS NOT NULL AND pa.target_id = v_api_key_id)
          )
          AND stmt->>'effect' = 'Deny'
          AND EXISTS (
              SELECT 1 FROM jsonb_array_elements_text(
                  CASE WHEN jsonb_typeof(stmt->'actions') = 'array' THEN stmt->'actions' ELSE '[]'::jsonb END
              ) AS act
              WHERE act = p_action OR act = '*'
          )
          AND EXISTS (
              SELECT 1 FROM jsonb_array_elements_text(
                  CASE WHEN jsonb_typeof(stmt->'resources') = 'array' THEN stmt->'resources' ELSE '[]'::jsonb END
              ) AS pat
              WHERE iam.arn_matches(pat, p_arn)
          )
          AND iam.policy_condition_matches(stmt->'condition', v_effective_level, v_deny_group_ids, p_access_scope)
    ) INTO v_has_explicit_deny;

    IF COALESCE(v_has_explicit_deny, false) THEN
        RETURN false;
    END IF;

    -- 3. Resource Base Permission Level
    v_res_min_level := COALESCE(p_min_auth_level, 1);
    IF v_effective_level < v_res_min_level THEN
        RETURN false;
    END IF;

    -- 4. Private Scope Strict Isolation (owner or same-dept dept_admin ONLY; peers blocked even with Allow *)
    IF p_access_scope = 'private' THEN
        IF p_owner_principal_id = v_principal_id THEN
            RETURN true;
        END IF;
        IF p_dept_id IS NOT NULL AND p_dept_id = ANY(v_dept_admin_ids) THEN
            RETURN true;
        END IF;
        RETURN false;
    END IF;

    -- 5. Compute Group Allow (evaluated against narrowed v_allow_group_ids)
    SELECT EXISTS (
        SELECT 1
        FROM iam.groups AS g
        JOIN iam.group_members AS gm
          ON gm.group_id = g.id AND gm.principal_id = v_principal_id AND gm.org_id = v_org_id
        WHERE g.org_id = v_org_id
          AND g.id = ANY(v_allow_group_ids)
          AND p_allowed_group_ids IS NOT NULL
          AND g.id = ANY(p_allowed_group_ids)
          AND (p_action = ANY(COALESCE(gm.action_overrides, g.allowed_actions))
               OR '*' = ANY(COALESCE(gm.action_overrides, g.allowed_actions)))
          AND LEAST(v_effective_level, COALESCE(gm.level_override, v_effective_level)) >= GREATEST(v_res_min_level, g.min_auth_level)
          AND v_res_min_level <= g.max_auth_level
    ) INTO v_group_allow;

    IF NOT COALESCE(v_group_allow, false) AND p_resource_type IS NOT NULL AND p_resource_id IS NOT NULL THEN
        SELECT EXISTS (
            SELECT 1
            FROM iam.resource_group_bindings AS rgb
            JOIN iam.groups AS g ON g.id = rgb.group_id AND g.org_id = v_org_id
            JOIN iam.group_members AS gm
              ON gm.group_id = g.id AND gm.principal_id = v_principal_id AND gm.org_id = v_org_id
            WHERE rgb.org_id = v_org_id
              AND rgb.resource_type = p_resource_type
              AND rgb.resource_id = p_resource_id
              AND rgb.effect = 'allow'
              AND g.id = ANY(v_allow_group_ids)
              AND (p_action = ANY(rgb.actions) OR '*' = ANY(rgb.actions))
              AND (p_action = ANY(COALESCE(gm.action_overrides, g.allowed_actions))
                   OR '*' = ANY(COALESCE(gm.action_overrides, g.allowed_actions)))
              AND LEAST(v_effective_level, COALESCE(gm.level_override, v_effective_level))
                  >= GREATEST(v_res_min_level, g.min_auth_level, COALESCE(rgb.min_auth_level, 1))
              AND v_res_min_level <= g.max_auth_level
        ) INTO v_group_allow;
    END IF;

    -- 6. Compute ARN Policy Allow (evaluated against narrowed v_allow_group_ids)
    SELECT EXISTS (
        SELECT 1
        FROM iam.policy_attachments AS pa
        JOIN iam.policies AS pol ON pol.id = pa.policy_id
        CROSS JOIN LATERAL jsonb_array_elements(
            CASE
                WHEN jsonb_typeof(pol.document->'statements') = 'array' THEN pol.document->'statements'
                ELSE '[]'::jsonb
            END
        ) AS stmt
        WHERE pa.org_id = v_org_id
          AND pol.org_id = v_org_id
          AND (
              (pa.target_type = 'principal' AND pa.target_id = v_principal_id) OR
              (pa.target_type = 'dept' AND pa.target_id = ANY(v_dept_ids)) OR
              (pa.target_type = 'group' AND pa.target_id = ANY(v_allow_group_ids)) OR
              (pa.target_type = 'api_key' AND v_api_key_id IS NOT NULL AND pa.target_id = v_api_key_id)
          )
          AND stmt->>'effect' = 'Allow'
          AND v_effective_level >= GREATEST(v_res_min_level, COALESCE(pa.min_auth_level_override, 1))
          AND EXISTS (
              SELECT 1 FROM jsonb_array_elements_text(
                  CASE WHEN jsonb_typeof(stmt->'actions') = 'array' THEN stmt->'actions' ELSE '[]'::jsonb END
              ) AS act
              WHERE act = p_action OR act = '*'
          )
          AND EXISTS (
              SELECT 1 FROM jsonb_array_elements_text(
                  CASE WHEN jsonb_typeof(stmt->'resources') = 'array' THEN stmt->'resources' ELSE '[]'::jsonb END
              ) AS pat
              WHERE iam.arn_matches(pat, p_arn)
          )
          AND iam.policy_condition_matches(stmt->'condition', v_effective_level, v_allow_group_ids, p_access_scope)
    ) INTO v_arn_allow;

    v_group_allow := COALESCE(v_group_allow, false);
    v_arn_allow := COALESCE(v_arn_allow, false);

    -- 7. Scope Resolution (dept | group | org)
    IF p_access_scope = 'dept' THEN
        IF p_action IN ('read', 'search') THEN
            RETURN COALESCE(p_owner_principal_id = v_principal_id, false)
                OR COALESCE(p_dept_id IS NOT NULL AND p_dept_id = ANY(v_dept_ids), false)
                OR v_group_allow
                OR v_arn_allow;
        ELSE
            RETURN COALESCE(p_owner_principal_id = v_principal_id, false)
                OR COALESCE(p_dept_id IS NOT NULL AND p_dept_id = ANY(v_dept_admin_ids), false)
                OR v_group_allow
                OR v_arn_allow;
        END IF;
    ELSIF p_access_scope = 'group' THEN
        RETURN COALESCE(p_owner_principal_id = v_principal_id, false)
            OR v_group_allow
            OR v_arn_allow;
    ELSIF p_access_scope = 'org' THEN
        IF p_action IN ('read', 'search') THEN
            RETURN true;
        ELSE
            RETURN COALESCE(p_owner_principal_id = v_principal_id, false)
                OR COALESCE(p_dept_id IS NOT NULL AND p_dept_id = ANY(v_dept_admin_ids), false)
                OR COALESCE(v_is_org_admin, false)
                OR v_group_allow
                OR v_arn_allow;
        END IF;
    END IF;

    RETURN false;
END;
$$;

-- Backward-compatible 7-arg + optional 3-arg wrapper for SELECT RLS policies
CREATE OR REPLACE FUNCTION iam.rls_can_read(
    p_org_id UUID,
    p_dept_id UUID,
    p_owner_principal_id UUID,
    p_access_scope TEXT,
    p_allowed_group_ids UUID[],
    p_arn TEXT,
    p_min_auth_level SMALLINT,
    p_denied_group_ids UUID[] DEFAULT ARRAY[]::UUID[],
    p_resource_type TEXT DEFAULT NULL,
    p_resource_id UUID DEFAULT NULL
)
RETURNS BOOLEAN
LANGUAGE sql
STABLE
SECURITY DEFINER
SET search_path = iam, pg_catalog
AS $$
    SELECT COALESCE(iam.rls_can_access(
        p_org_id,
        p_dept_id,
        p_owner_principal_id,
        p_access_scope,
        p_allowed_group_ids,
        p_arn,
        p_min_auth_level,
        p_denied_group_ids,
        'read',
        p_resource_type,
        p_resource_id
    ), false);
$$;

-- Helper: verify department belongs to org (SECURITY DEFINER for trigger/RLS callers without direct table grants)
CREATE OR REPLACE FUNCTION iam.dept_belongs_to_org(
    p_org_id UUID,
    p_dept_id UUID
)
RETURNS BOOLEAN
LANGUAGE sql
STABLE
SECURITY DEFINER
SET search_path = iam, pg_catalog
AS $$
    SELECT p_org_id IS NOT NULL
       AND p_dept_id IS NOT NULL
       AND EXISTS (
           SELECT 1 FROM iam.departments AS d
           WHERE d.id = p_dept_id AND d.org_id = p_org_id
       );
$$;

-- Helper: verify principal is active in org (SECURITY DEFINER for trigger/RLS callers without direct iam.principals SELECT)
CREATE OR REPLACE FUNCTION iam.principal_active_in_org(
    p_org_id UUID,
    p_principal_id UUID
)
RETURNS BOOLEAN
LANGUAGE sql
STABLE
SECURITY DEFINER
SET search_path = iam, pg_catalog
AS $$
    SELECT p_org_id IS NOT NULL
       AND p_principal_id IS NOT NULL
       AND EXISTS (
           SELECT 1 FROM iam.principals AS pr
           WHERE pr.id = p_principal_id AND pr.org_id = p_org_id AND pr.status = 'active'
       );
$$;

-- Stub replaced in 006_eval_bench.sql once eval.pipeline_configs exists
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
    SELECT false;
$$;

-- Helper: verify group belongs to org and one of caller's administered departments (bypasses iam.groups RLS safely)
CREATE OR REPLACE FUNCTION iam.group_belongs_to_dept_admins(
    p_org_id UUID,
    p_group_id UUID,
    p_dept_admin_ids UUID[]
)
RETURNS BOOLEAN
LANGUAGE sql
STABLE
SECURITY DEFINER
SET search_path = iam, pg_catalog, pg_temp
AS $$
    SELECT p_org_id IS NOT NULL
       AND p_group_id IS NOT NULL
       AND p_dept_admin_ids IS NOT NULL
       AND cardinality(p_dept_admin_ids) > 0
       AND EXISTS (
           SELECT 1
           FROM iam.groups AS g
           WHERE g.id = p_group_id
             AND g.org_id = p_org_id
             AND g.dept_id IS NOT NULL
             AND g.dept_id = ANY(p_dept_admin_ids)
       );
$$;

-- Fail-Closed active session + API key validation helper for operational & IAM RLS policies
CREATE OR REPLACE FUNCTION iam.session_active_principal(
    OUT v_org_id UUID,
    OUT v_principal_id UUID,
    OUT v_is_org_admin BOOLEAN,
    OUT v_dept_admin_ids UUID[],
    OUT v_ok BOOLEAN
)
RETURNS RECORD
LANGUAGE plpgsql
STABLE
SECURITY DEFINER
SET search_path = iam, pg_catalog
AS $$
DECLARE
    v_auth_level SMALLINT;
    v_dept_ids UUID[];
    v_group_ids UUID[];
    v_guc_scoped_groups UUID[];
    v_scoped_arns TEXT[];
    v_status TEXT;
    v_api_key_raw TEXT;
    v_api_key_id UUID;
BEGIN
    v_ok := false;
    v_is_org_admin := false;
    v_org_id := iam.safe_uuid(current_setting('app.org_id', true));
    v_principal_id := iam.safe_uuid(current_setting('app.principal_id', true));
    v_auth_level := iam.safe_smallint(current_setting('app.auth_level', true), 0::SMALLINT);
    v_dept_ids := iam.safe_uuid_array(current_setting('app.dept_ids', true));
    v_dept_admin_ids := iam.safe_uuid_array(current_setting('app.dept_admin_ids', true));
    v_group_ids := iam.safe_uuid_array(current_setting('app.group_ids', true));
    v_guc_scoped_groups := iam.safe_uuid_array(current_setting('app.scoped_group_ids', true));
    v_scoped_arns := iam.safe_text_array(current_setting('app.scoped_arns', true));

    IF v_org_id IS NULL
       OR v_principal_id IS NULL
       OR v_auth_level IS NULL
       OR v_auth_level < 1
       OR v_auth_level > 10
       OR v_dept_ids IS NULL
       OR v_dept_admin_ids IS NULL
       OR v_group_ids IS NULL
       OR v_guc_scoped_groups IS NULL
       OR v_scoped_arns IS NULL THEN
        RETURN;
    END IF;

    SELECT p.is_org_admin, p.status
    INTO v_is_org_admin, v_status
    FROM iam.principals AS p
    WHERE p.id = v_principal_id AND p.org_id = v_org_id;

    IF NOT FOUND OR v_status <> 'active' THEN
        v_is_org_admin := false;
        RETURN;
    END IF;

    v_api_key_raw := NULLIF(btrim(current_setting('app.api_key_id', true)), '');
    IF v_api_key_raw IS NOT NULL THEN
        v_api_key_id := iam.safe_uuid(v_api_key_raw);
        IF v_api_key_id IS NULL THEN
            RETURN;
        END IF;

        IF NOT EXISTS (
            SELECT 1
            FROM iam.api_keys AS ak
            WHERE ak.id = v_api_key_id
              AND ak.org_id = v_org_id
              AND ak.principal_id = v_principal_id
              AND ak.revoked_at IS NULL
              AND (ak.expires_at IS NULL OR ak.expires_at > now())
        ) THEN
            RETURN;
        END IF;
    END IF;

    v_ok := true;
END;
$$;

DROP FUNCTION IF EXISTS iam.rls_can_insert(UUID, UUID, UUID, TEXT);

-- Fail-Closed INSERT check (prevents cross-org / cross-owner / nonexistent UUID spoofing + clearance & API key enforcement)
CREATE OR REPLACE FUNCTION iam.rls_can_insert(
    p_org_id UUID,
    p_dept_id UUID,
    p_owner_principal_id UUID,
    p_arn TEXT DEFAULT NULL,
    p_min_auth_level SMALLINT DEFAULT 1
)
RETURNS BOOLEAN
LANGUAGE plpgsql
STABLE
SECURITY DEFINER
SET search_path = iam, pg_catalog
AS $$
DECLARE
    v_org_id UUID;
    v_principal_id UUID;
    v_auth_level SMALLINT;
    v_dept_ids UUID[];
    v_dept_admin_ids UUID[];
    v_scoped_arns TEXT[];
    v_api_key_raw TEXT;
    v_api_key_id UUID;
    v_p_auth_level SMALLINT;
    v_is_org_admin BOOLEAN;
    v_status TEXT;
    v_effective_level SMALLINT;
    v_ak_max_level SMALLINT;
    v_ak_scoped_arns TEXT[];
    v_ak_expires TIMESTAMPTZ;
    v_ak_revoked TIMESTAMPTZ;
    v_arn_matched BOOLEAN;
    v_res_min_level SMALLINT;
    v_arn_level_str TEXT;
    v_arn_level SMALLINT;
BEGIN
    v_org_id := iam.safe_uuid(current_setting('app.org_id', true));
    v_principal_id := iam.safe_uuid(current_setting('app.principal_id', true));
    v_auth_level := iam.safe_smallint(current_setting('app.auth_level', true), 0::SMALLINT);
    v_dept_ids := iam.safe_uuid_array(current_setting('app.dept_ids', true));
    v_dept_admin_ids := iam.safe_uuid_array(current_setting('app.dept_admin_ids', true));
    v_scoped_arns := iam.safe_text_array(current_setting('app.scoped_arns', true));

    IF v_org_id IS NULL OR v_principal_id IS NULL OR p_org_id IS NULL OR v_org_id <> p_org_id THEN
        RETURN false;
    END IF;
    IF p_dept_id IS NULL OR p_owner_principal_id IS NULL THEN
        RETURN false;
    END IF;
    IF v_auth_level IS NULL OR v_auth_level < 1 OR v_auth_level > 10 THEN
        RETURN false;
    END IF;
    IF v_dept_ids IS NULL OR v_dept_admin_ids IS NULL OR v_scoped_arns IS NULL THEN
        RETURN false;
    END IF;

    SELECT p.auth_level, p.is_org_admin, p.status
    INTO v_p_auth_level, v_is_org_admin, v_status
    FROM iam.principals AS p
    WHERE p.id = v_principal_id AND p.org_id = v_org_id;

    IF NOT FOUND OR v_status <> 'active' THEN
        RETURN false;
    END IF;

    v_effective_level := LEAST(v_auth_level, v_p_auth_level);

    -- Validate app.api_key_id if non-blank (fail closed on malformed, missing, revoked, or expired)
    v_api_key_raw := NULLIF(btrim(current_setting('app.api_key_id', true)), '');
    IF v_api_key_raw IS NOT NULL THEN
        v_api_key_id := iam.safe_uuid(v_api_key_raw);
        IF v_api_key_id IS NULL THEN
            RETURN false;
        END IF;

        SELECT ak.max_auth_level, ak.scoped_arns, ak.expires_at, ak.revoked_at
        INTO v_ak_max_level, v_ak_scoped_arns, v_ak_expires, v_ak_revoked
        FROM iam.api_keys AS ak
        WHERE ak.id = v_api_key_id
          AND ak.org_id = v_org_id
          AND ak.principal_id = v_principal_id
          AND ak.revoked_at IS NULL
          AND (ak.expires_at IS NULL OR ak.expires_at > now());

        IF NOT FOUND THEN
            RETURN false;
        END IF;

        IF v_ak_max_level IS NOT NULL THEN
            v_effective_level := LEAST(v_effective_level, v_ak_max_level);
        END IF;

        IF v_ak_scoped_arns IS NOT NULL AND cardinality(v_ak_scoped_arns) > 0 THEN
            IF p_arn IS NULL THEN
                RETURN false;
            END IF;
            SELECT EXISTS (
                SELECT 1 FROM unnest(v_ak_scoped_arns) AS pat
                WHERE iam.arn_matches(pat, p_arn)
            ) INTO v_arn_matched;
            IF NOT COALESCE(v_arn_matched, false) THEN
                RETURN false;
            END IF;
        END IF;
    END IF;

    -- Enforce session app.scoped_arns ceiling if provided
    IF cardinality(v_scoped_arns) > 0 THEN
        IF p_arn IS NULL THEN
            RETURN false;
        END IF;
        SELECT EXISTS (
            SELECT 1 FROM unnest(v_scoped_arns) AS pat
            WHERE iam.arn_matches(pat, p_arn)
        ) INTO v_arn_matched;
        IF NOT COALESCE(v_arn_matched, false) THEN
            RETURN false;
        END IF;
    END IF;

    -- Enforce min_auth_level and mandatory ARN ':L<n>/' clearance bounds
    v_res_min_level := COALESCE(p_min_auth_level, 1::SMALLINT);
    IF v_res_min_level < 1 OR v_res_min_level > 10 OR v_res_min_level > v_effective_level THEN
        RETURN false;
    END IF;

    IF p_arn IS NULL OR btrim(p_arn) = '' THEN
        RETURN false;
    END IF;
    v_arn_level_str := substring(p_arn FROM ':L([0-9]{1,2})/[^:]*$');
    IF v_arn_level_str IS NULL THEN
        RETURN false;
    END IF;
    v_arn_level := iam.safe_smallint(v_arn_level_str, 0::SMALLINT);
    IF v_arn_level IS NULL OR v_arn_level < 1 OR v_arn_level > 10 THEN
        RETURN false;
    END IF;
    IF v_arn_level < v_res_min_level OR v_arn_level > v_effective_level THEN
        RETURN false;
    END IF;

    -- Verify target dept_id and owner_principal_id belong to v_org_id BEFORE role checks (eliminates 23503 oracle)
    IF NOT EXISTS (
        SELECT 1 FROM iam.departments AS d
        WHERE d.id = p_dept_id AND d.org_id = v_org_id
    ) THEN
        RETURN false;
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM iam.principals AS pr
        WHERE pr.id = p_owner_principal_id AND pr.org_id = v_org_id AND pr.status = 'active'
    ) THEN
        RETURN false;
    END IF;

    IF COALESCE(v_is_org_admin, false) THEN
        RETURN true;
    END IF;

    IF p_dept_id = ANY(v_dept_admin_ids) THEN
        RETURN true;
    END IF;

    RETURN COALESCE(
        p_owner_principal_id = v_principal_id
        AND p_dept_id = ANY(v_dept_ids),
        false
    );
END;
$$;

-- Prevent non-admin or cross-org reassignment of org_id, dept_id, or owner_principal_id on UPDATE
CREATE OR REPLACE FUNCTION iam.prevent_non_admin_ownership_reassignment()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
DECLARE
    v_bypass BOOLEAN := false;
    v_dept_admin_ids UUID[];
BEGIN
    SELECT r.rolsuper OR r.rolbypassrls
    INTO v_bypass
    FROM pg_catalog.pg_roles AS r
    WHERE r.rolname = current_user;

    IF COALESCE(v_bypass, false) THEN
        RETURN NEW;
    END IF;

    IF NEW.org_id IS DISTINCT FROM OLD.org_id THEN
        RAISE EXCEPTION 'cannot reassign org_id on %.%', TG_TABLE_SCHEMA, TG_TABLE_NAME
            USING ERRCODE = '42501';
    END IF;

    IF NEW.dept_id IS DISTINCT FROM OLD.dept_id
       OR NEW.owner_principal_id IS DISTINCT FROM OLD.owner_principal_id THEN
        v_dept_admin_ids := iam.safe_uuid_array(current_setting('app.dept_admin_ids', true));
        IF v_dept_admin_ids IS NULL OR NOT COALESCE(OLD.dept_id = ANY(v_dept_admin_ids), false) THEN
            RAISE EXCEPTION 'non-admin cannot reassign dept_id or owner_principal_id on %.%', TG_TABLE_SCHEMA, TG_TABLE_NAME
                USING ERRCODE = '42501';
        END IF;
        IF NEW.dept_id IS DISTINCT FROM OLD.dept_id
           AND (NOT COALESCE(NEW.dept_id = ANY(v_dept_admin_ids), false)
                OR NOT iam.dept_belongs_to_org(OLD.org_id, NEW.dept_id)) THEN
            RAISE EXCEPTION 'cannot reassign dept_id outside admin departments in org on %.%', TG_TABLE_SCHEMA, TG_TABLE_NAME
                USING ERRCODE = '42501';
        END IF;
        IF NEW.owner_principal_id IS DISTINCT FROM OLD.owner_principal_id
           AND NOT iam.principal_active_in_org(OLD.org_id, NEW.owner_principal_id) THEN
            RAISE EXCEPTION 'cannot reassign owner_principal_id outside org on %.%', TG_TABLE_SCHEMA, TG_TABLE_NAME
                USING ERRCODE = '42501';
        END IF;
    END IF;

    RETURN NEW;
END;
$$;

-- Fail-Closed RLS on openrag_app-accessible IAM directory tables (SELECT ONLY)
ALTER TABLE iam.orgs          ENABLE ROW LEVEL SECURITY;
ALTER TABLE iam.orgs          FORCE ROW LEVEL SECURITY;
ALTER TABLE iam.departments   ENABLE ROW LEVEL SECURITY;
ALTER TABLE iam.departments   FORCE ROW LEVEL SECURITY;
ALTER TABLE iam.groups        ENABLE ROW LEVEL SECURITY;
ALTER TABLE iam.groups        FORCE ROW LEVEL SECURITY;
ALTER TABLE iam.group_members ENABLE ROW LEVEL SECURITY;
ALTER TABLE iam.group_members FORCE ROW LEVEL SECURITY;

CREATE POLICY rls_iam_orgs_read ON iam.orgs FOR SELECT
    USING (EXISTS (
        SELECT 1 FROM iam.session_active_principal() AS s
        WHERE s.v_ok AND iam.orgs.id = s.v_org_id
    ));

CREATE POLICY rls_iam_departments_read ON iam.departments FOR SELECT
    USING (EXISTS (
        SELECT 1 FROM iam.session_active_principal() AS s
        WHERE s.v_ok AND iam.departments.org_id = s.v_org_id
    ));

CREATE POLICY rls_iam_groups_read ON iam.groups FOR SELECT
    USING (EXISTS (
        SELECT 1 FROM iam.session_active_principal() AS s
        WHERE s.v_ok AND iam.groups.org_id = s.v_org_id
    ));

CREATE POLICY rls_iam_group_members_read ON iam.group_members FOR SELECT
    USING (EXISTS (
        SELECT 1 FROM iam.session_active_principal() AS s
        WHERE s.v_ok
          AND iam.group_members.org_id = s.v_org_id
          AND (
              iam.group_members.principal_id = s.v_principal_id
              OR s.v_is_org_admin
              OR iam.group_belongs_to_dept_admins(
                  s.v_org_id,
                  iam.group_members.group_id,
                  s.v_dept_admin_ids
              )
          )
    ));
