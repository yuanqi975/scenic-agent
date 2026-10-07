-- Migrate an existing pgvector database from vector(1536) to halfvec(2560).
-- Existing embeddings cannot be reused: clear them before changing the typmod,
-- then regenerate all vectors with the configured 2560-dimensional model. halfvec is
-- required because pgvector's HNSW vector type is limited to 2000 dimensions.
BEGIN;

UPDATE document_chunks SET embedding = NULL WHERE embedding IS NOT NULL;

DROP INDEX IF EXISTS document_chunks_embedding_idx;

ALTER TABLE document_chunks
    ALTER COLUMN embedding TYPE halfvec(2560)
    USING NULL::halfvec(2560);

CREATE INDEX document_chunks_embedding_idx
    ON document_chunks USING hnsw (embedding halfvec_cosine_ops);

COMMIT;
