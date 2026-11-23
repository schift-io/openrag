-- OpenRAG Migration 012: page thumbnails and figures are no longer embedded.
--
-- The images stay (display, PII masking); retrieval reaches pages through their text and the
-- parent document's tags. Dropping a column also drops the vector index built on it
-- (pgvector HNSW or pg_pinecone native), and the bindings for those columns go with it.
--
-- Idempotent.

ALTER TABLE corpus.pages DROP COLUMN IF EXISTS thumb_embedding;
ALTER TABLE corpus.page_figures DROP COLUMN IF EXISTS figure_embedding;

DELETE FROM vectors.index_bindings
WHERE (table_name, vector_column) IN (
    ('corpus.pages', 'thumb_embedding'),
    ('corpus.page_figures', 'figure_embedding')
);
