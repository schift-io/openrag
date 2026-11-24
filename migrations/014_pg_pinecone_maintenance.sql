-- pg_pinecone builds a native index's IVF clusters from the rows present at CREATE INDEX. The
-- migrations create the indexes on empty tables, and the planner costs such an index out
-- (disable_cost), so every search was an exact sequential scan: measured 2026-11-24 on 100k chunks,
-- p50 1.3 s before a rebuild, 0.23 s after. Register every pg_pinecone index for maintenance: the
-- pg_pinecone worker REINDEXes it once rows inserted since the last build reach 20% of the table.
-- The worker acts only with pinecone.auto_maintain = on (docker/docker-compose.yml sets it).
-- Without pg_pinecone (pgvector HNSW fallback) this does nothing.
DO $$
DECLARE
    r record;
BEGIN
    IF to_regprocedure('pinecone.schedule_maintenance(regclass,double precision)') IS NULL THEN
        RAISE NOTICE 'pg_pinecone maintenance is not available; skipping';
        RETURN;
    END IF;
    FOR r IN
        SELECT i.indexrelid::regclass AS idx
        FROM pg_index i
        JOIN pg_class c ON c.oid = i.indexrelid
        JOIN pg_am a ON a.oid = c.relam
        WHERE a.amname = 'pinecone'
    LOOP
        PERFORM pinecone.schedule_maintenance(r.idx, 0.2);
    END LOOP;
END $$;
