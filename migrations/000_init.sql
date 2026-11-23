-- OpenRAG Migration 000: Extensions, 10 Schemas, Capability & Safe GUC Helpers
-- Safe for single-transaction execution on PostgreSQL 17/18 with or without
-- compiled pg_pinecone / pg_search extensions.

CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- Optional extensions (available in full Docker image; graceful fallback locally)
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_available_extensions WHERE name = 'pg_pinecone') THEN
        CREATE EXTENSION IF NOT EXISTS pg_pinecone;
    ELSE
        RAISE NOTICE 'pg_pinecone not installed; using pgvector HNSW fallback';
    END IF;
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'pg_pinecone unavailable (%); using pgvector HNSW fallback', SQLERRM;
END;
$$;

DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_available_extensions WHERE name = 'pg_search') THEN
        CREATE EXTENSION IF NOT EXISTS pg_search;
    ELSE
        RAISE NOTICE 'pg_search not installed; using pg_trgm + tsvector fallback';
    END IF;
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'pg_search unavailable (%); using pg_trgm + tsvector fallback', SQLERRM;
END;
$$;

-- 10 Core Schemas
CREATE SCHEMA IF NOT EXISTS iam;
CREATE SCHEMA IF NOT EXISTS vectors;
CREATE SCHEMA IF NOT EXISTS corpus;
CREATE SCHEMA IF NOT EXISTS pii;
CREATE SCHEMA IF NOT EXISTS memory;
CREATE SCHEMA IF NOT EXISTS wiki;
CREATE SCHEMA IF NOT EXISTS indexing;
CREATE SCHEMA IF NOT EXISTS gateway;
CREATE SCHEMA IF NOT EXISTS plugins;
CREATE SCHEMA IF NOT EXISTS eval;

-- Extension Capability Probes
CREATE OR REPLACE FUNCTION vectors.has_pg_pinecone()
RETURNS BOOLEAN
LANGUAGE sql
STABLE
AS $$
    SELECT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'pg_pinecone');
$$;

CREATE OR REPLACE FUNCTION vectors.has_pg_search()
RETURNS BOOLEAN
LANGUAGE sql
STABLE
AS $$
    SELECT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'pg_search');
$$;

-- Fail-Closed Session GUC Parsers (never throw on empty '' or malformed values)
CREATE OR REPLACE FUNCTION iam.safe_uuid(p_raw TEXT)
RETURNS UUID
LANGUAGE plpgsql
STABLE
AS $$
DECLARE
    v_trimmed TEXT;
BEGIN
    v_trimmed := NULLIF(btrim(p_raw), '');
    IF v_trimmed IS NULL THEN
        RETURN NULL;
    END IF;
    RETURN v_trimmed::UUID;
EXCEPTION WHEN OTHERS THEN
    RETURN NULL;
END;
$$;

CREATE OR REPLACE FUNCTION iam.safe_smallint(p_raw TEXT, p_default SMALLINT DEFAULT 0)
RETURNS SMALLINT
LANGUAGE plpgsql
STABLE
AS $$
DECLARE
    v_trimmed TEXT;
BEGIN
    v_trimmed := NULLIF(btrim(p_raw), '');
    IF v_trimmed IS NULL THEN
        RETURN p_default;
    END IF;
    RETURN v_trimmed::SMALLINT;
EXCEPTION WHEN OTHERS THEN
    RETURN p_default;
END;
$$;

-- Returns ARRAY[]::UUID[] on NULL/blank/'{}', or NULL if any token is malformed or NULL (Fail-Closed signal)
CREATE OR REPLACE FUNCTION iam.safe_uuid_array(p_raw TEXT)
RETURNS UUID[]
LANGUAGE plpgsql
STABLE
AS $$
DECLARE
    v_trimmed TEXT;
    v_parsed  UUID[];
BEGIN
    v_trimmed := NULLIF(btrim(p_raw), '');
    IF v_trimmed IS NULL OR v_trimmed = '{}' THEN
        RETURN ARRAY[]::UUID[];
    END IF;
    IF left(v_trimmed, 1) = '{' THEN
        v_parsed := v_trimmed::UUID[];
    ELSE
        v_parsed := string_to_array(v_trimmed, ',')::UUID[];
    END IF;
    IF v_parsed IS NULL OR array_position(v_parsed, NULL) IS NOT NULL THEN
        RETURN NULL;
    END IF;
    RETURN v_parsed;
EXCEPTION WHEN OTHERS THEN
    RETURN NULL;
END;
$$;

-- Returns ARRAY[]::TEXT[] on NULL/blank/'{}', or NULL if malformed or contains NULL element (Fail-Closed signal)
CREATE OR REPLACE FUNCTION iam.safe_text_array(p_raw TEXT)
RETURNS TEXT[]
LANGUAGE plpgsql
STABLE
AS $$
DECLARE
    v_trimmed TEXT;
    v_parsed  TEXT[];
BEGIN
    v_trimmed := NULLIF(btrim(p_raw), '');
    IF v_trimmed IS NULL OR v_trimmed = '{}' THEN
        RETURN ARRAY[]::TEXT[];
    END IF;
    IF left(v_trimmed, 1) = '{' THEN
        v_parsed := v_trimmed::TEXT[];
    ELSE
        v_parsed := string_to_array(v_trimmed, ',');
    END IF;
    IF v_parsed IS NULL OR array_position(v_parsed, NULL) IS NOT NULL THEN
        RETURN NULL;
    END IF;
    RETURN v_parsed;
EXCEPTION WHEN OTHERS THEN
    RETURN NULL;
END;
$$;
