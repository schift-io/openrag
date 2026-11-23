-- OpenRAG Migration 011: every stored vector space moves from 128d to 256d
-- (text: corpus.chunks, wiki.pages, memory.entries; visual: corpus.pages thumbnails,
-- corpus.page_figures; concepts: wiki.concepts).
--
-- A 128d vector cannot be widened into the 256d space (it came from a different model or
-- feature layout), so existing vectors are cleared and must be re-embedded. Rows stay
-- searchable by BM25 and tags meanwhile. ALTER COLUMN TYPE rebuilds each column's vector
-- index (pgvector HNSW or pg_pinecone) in place.
--
-- Idempotent: a column is only altered while it is still vector(128).

DO $$
DECLARE
    c record;
BEGIN
    FOR c IN
        SELECT *
        FROM (VALUES
            ('corpus', 'chunks', 'embedding'),
            ('corpus', 'pages', 'thumb_embedding'),
            ('corpus', 'page_figures', 'figure_embedding'),
            ('wiki', 'pages', 'embedding'),
            ('memory', 'entries', 'embedding'),
            ('wiki', 'concepts', 'embedding')
        ) AS t(sch, tbl, col)
    LOOP
        IF EXISTS (
            SELECT 1
            FROM pg_attribute a
            JOIN pg_class k ON k.oid = a.attrelid
            JOIN pg_namespace n ON n.oid = k.relnamespace
            WHERE n.nspname = c.sch
              AND k.relname = c.tbl
              AND a.attname = c.col
              AND NOT a.attisdropped
              AND format_type(a.atttypid, a.atttypmod) = 'vector(128)'
        ) THEN
            EXECUTE format(
                'ALTER TABLE %I.%I ALTER COLUMN %I TYPE vector(256) USING NULL::vector(256)',
                c.sch, c.tbl, c.col
            );
        END IF;
    END LOOP;
END;
$$;

ALTER TABLE vectors.index_bindings ALTER COLUMN dimension SET DEFAULT 256;
UPDATE vectors.index_bindings SET dimension = 256, updated_at = now() WHERE dimension = 128;
