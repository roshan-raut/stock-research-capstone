-- ============================================================================
-- Vector embeddings for semantic retrieval over unstructured news text.
-- Replace {{EMBEDDING_DIM}} with your model's output dimension before running
-- (e.g. 384 for sentence-transformers/all-MiniLM-L6-v2, 768 for
-- all-mpnet-base-v2 / bge-base-en-v1.5).
--
-- WHY THE STAGING TABLES EXIST:
-- Spark's generic JDBC writer (df.write.jdbc(...)) has no idea what a
-- Postgres `vector` column is - it can only write standard SQL types. So the
-- pipeline computes embeddings as a Spark array<float> column, writes that
-- via plain JDBC into a TEXT-typed staging table (the array serialized as a
-- Postgres array literal string), and a short psycopg2 step does the final
-- `INSERT ... SELECT ... embedding::vector` cast + upsert into the real
-- pgvector table. This is the standard pattern for getting Spark output into
-- a pgvector column.
-- ============================================================================

CREATE EXTENSION IF NOT EXISTS vector;

-- ----------------------------------------------------------------------------
-- Document-level embeddings (title + description) - one row per article.
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS ticker_news_embeddings (
    id            TEXT PRIMARY KEY,
    ticker        TEXT NOT NULL,
    title         TEXT NOT NULL,
    published_utc TIMESTAMPTZ,
    embedding     VECTOR({{EMBEDDING_DIM}}) NOT NULL,
    model_name    TEXT NOT NULL,
    embedded_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_ticker_news_embeddings_embedding
    ON ticker_news_embeddings USING hnsw (embedding vector_cosine_ops);

CREATE TABLE IF NOT EXISTS ticker_news_embeddings_staging (
    id            TEXT,
    ticker        TEXT,
    title         TEXT,
    published_utc TIMESTAMPTZ,
    embedding_str TEXT NOT NULL,   -- e.g. "[0.012,-0.44,...]" - cast to vector on merge
    model_name    TEXT NOT NULL
);

-- ----------------------------------------------------------------------------
-- Chunk-level embeddings (full article body, split into overlapping chunks)
-- - fine-grained passages for RAG retrieval beyond title/description.
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS ticker_news_chunk_embeddings (
    id            TEXT PRIMARY KEY,
    article_id    TEXT NOT NULL,
    ticker        TEXT NOT NULL,
    chunk_index   INT NOT NULL,
    chunk_text    TEXT NOT NULL,
    embedding     VECTOR({{EMBEDDING_DIM}}) NOT NULL,
    model_name    TEXT NOT NULL,
    embedded_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_ticker_news_chunk_embeddings_embedding
    ON ticker_news_chunk_embeddings USING hnsw (embedding vector_cosine_ops);

CREATE TABLE IF NOT EXISTS ticker_news_chunk_embeddings_staging (
    id            TEXT,
    article_id    TEXT,
    ticker        TEXT,
    chunk_index   INT,
    chunk_text    TEXT,
    embedding_str TEXT NOT NULL,
    model_name    TEXT NOT NULL
);
