-- ============================================================================
-- Core relational tables for the AI Stock Market Research Assistant.
-- Run this once against your Lakebase Postgres database (SQL editor, psql,
-- or a notebook %sql cell) before deploying the app / MCP server / pipeline.
-- ============================================================================

-- ----------------------------------------------------------------------------
-- watchlist: one row per (email, symbol) a user is tracking.
-- Kept as a single denormalized table (rather than separate users /
-- watchlists / watchlist_tickers tables) since this app has exactly one
-- watchlist per user - simpler to query, same information.
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS watchlist (
    symbol       TEXT NOT NULL,
    email        TEXT NOT NULL,
    latest_price NUMERIC,
    added_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (symbol, email)
);

CREATE INDEX IF NOT EXISTS idx_watchlist_email ON watchlist (email);

-- ----------------------------------------------------------------------------
-- companies: fundamentals / reference data for a ticker, refreshed
-- periodically by the Spark pipeline from Massive's ticker-details endpoint.
-- One row per ticker (upserted, not appended).
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS companies (
    symbol           TEXT PRIMARY KEY,
    name             TEXT,
    sector           TEXT,
    industry         TEXT,
    description      TEXT,
    market_cap       NUMERIC,
    employees        INTEGER,
    homepage_url     TEXT,
    exchange         TEXT,
    currency         TEXT,
    payload          JSONB,
    updated_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Staging table the Spark pipeline appends fundamentals into via plain JDBC
-- (no ON CONFLICT support over JDBC); a short psycopg2 step then upserts
-- staging -> companies, keyed on symbol. Truncated at the start of each run.
CREATE TABLE IF NOT EXISTS companies_staging (
    symbol           TEXT,
    name             TEXT,
    sector           TEXT,
    industry         TEXT,
    description      TEXT,
    market_cap       NUMERIC,
    employees        INTEGER,
    homepage_url     TEXT,
    exchange         TEXT,
    currency         TEXT,
    payload          TEXT   -- JSON-encoded string; cast to JSONB on merge
);

-- ----------------------------------------------------------------------------
-- price_snapshots: append-only time series of price observations per ticker.
-- The Spark pipeline appends one row per ticker per run, so history
-- accumulates over time (this is what "historical price data" and
-- "notable price move" detection are built on).
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS price_snapshots (
    id              BIGSERIAL PRIMARY KEY,
    symbol          TEXT NOT NULL,
    price           NUMERIC NOT NULL,
    open            NUMERIC,
    high            NUMERIC,
    low             NUMERIC,
    volume          BIGINT,
    change          NUMERIC,
    change_percent  NUMERIC,
    as_of           TIMESTAMPTZ,
    captured_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_price_snapshots_symbol_captured
    ON price_snapshots (symbol, captured_at DESC);

-- ----------------------------------------------------------------------------
-- research_notes: freeform notes a user (or the agent, on the user's behalf)
-- jots down about a ticker.
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS research_notes (
    id          BIGSERIAL PRIMARY KEY,
    email       TEXT NOT NULL,
    symbol      TEXT NOT NULL,
    note_text   TEXT NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_research_notes_email_symbol
    ON research_notes (email, symbol, created_at DESC);

-- ----------------------------------------------------------------------------
-- analysis_reports: structured, agent-generated (or user-authored) writeups
-- tied to a ticker - a thesis, a summary, and the sources it drew on.
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS analysis_reports (
    id          BIGSERIAL PRIMARY KEY,
    email       TEXT NOT NULL,
    symbol      TEXT NOT NULL,
    thesis      TEXT,
    summary     TEXT NOT NULL,
    sources     JSONB,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_analysis_reports_email_symbol
    ON analysis_reports (email, symbol, created_at DESC);

-- ----------------------------------------------------------------------------
-- user_checkpoints: last time each user "checked in", so the agent can flag
-- notable price moves / news published since then.
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS user_checkpoints (
    email          TEXT PRIMARY KEY,
    last_seen_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);
