-- ============================================================================
-- Raw unstructured news documents (the source text that gets embedded for
-- semantic retrieval in 03_embeddings_tables.sql). Same shape as the Day 2
-- table so existing data keeps working if you're extending that repo.
-- ============================================================================

CREATE TABLE IF NOT EXISTS ticker_news_documents (
    id                    TEXT PRIMARY KEY,
    ticker                TEXT NOT NULL,
    title                 TEXT NOT NULL,
    description           TEXT,
    author                TEXT,
    article_url           TEXT,
    publisher_name        TEXT,
    keywords              JSONB,
    sentiment             TEXT,
    sentiment_reasoning   TEXT,
    published_utc         TIMESTAMPTZ,
    payload               JSONB NOT NULL,
    synced_at             TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_ticker_news_documents_ticker
    ON ticker_news_documents (ticker);

CREATE INDEX IF NOT EXISTS idx_ticker_news_documents_published
    ON ticker_news_documents (published_utc DESC);

-- Staging table the Spark pipeline writes raw fetched articles into via a
-- plain JDBC append (no ON CONFLICT support over JDBC); a short psycopg2
-- step then MERGEs staging -> ticker_news_documents. Truncated each run.
CREATE TABLE IF NOT EXISTS ticker_news_documents_staging (
    id                    TEXT,
    ticker                TEXT,
    title                 TEXT,
    description           TEXT,
    author                TEXT,
    article_url           TEXT,
    publisher_name        TEXT,
    keywords              TEXT,   -- JSON-encoded string; cast to JSONB on merge
    sentiment             TEXT,
    sentiment_reasoning   TEXT,
    published_utc         TIMESTAMPTZ,
    payload               TEXT    -- JSON-encoded string; cast to JSONB on merge
);
