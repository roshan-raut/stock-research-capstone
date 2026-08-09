"""
Core logic for the AI Stock Market Research Assistant.

Every function here is a plain read or write against Lakebase (+ live Massive
API calls where noted). This module is imported by both the MCP server
(research_mcp_server.py, which exposes these as agent tools) and the Flask
app (app.py, which calls the same functions directly for the human-facing UI)
so the two surfaces never drift out of sync.
"""

import json
import logging
import os
from datetime import datetime, timezone

import lakebase
from massive_client import MassiveClient

logger = logging.getLogger("research-tools")

WATCHLIST_TABLE = os.environ.get("WATCHLIST_TABLE_NAME", "watchlist")
COMPANIES_TABLE = os.environ.get("COMPANIES_TABLE_NAME", "companies")
PRICE_SNAPSHOTS_TABLE = os.environ.get("PRICE_SNAPSHOTS_TABLE_NAME", "price_snapshots")
NEWS_TABLE = os.environ.get("NEWS_TABLE_NAME", "ticker_news_documents")
EMBEDDINGS_TABLE = os.environ.get("EMBEDDINGS_TABLE_NAME", "ticker_news_embeddings")
CHUNK_EMBEDDINGS_TABLE = os.environ.get("CHUNK_EMBEDDINGS_TABLE_NAME", "ticker_news_chunk_embeddings")
RESEARCH_NOTES_TABLE = os.environ.get("RESEARCH_NOTES_TABLE_NAME", "research_notes")
ANALYSIS_REPORTS_TABLE = os.environ.get("ANALYSIS_REPORTS_TABLE_NAME", "analysis_reports")
USER_CHECKPOINTS_TABLE = os.environ.get("USER_CHECKPOINTS_TABLE_NAME", "user_checkpoints")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# Flag a price move as "notable" when the daily change is at least this many
# percentage points, or an article was published, since the user last visited.
NOTABLE_MOVE_THRESHOLD_PCT = float(os.environ.get("NOTABLE_MOVE_THRESHOLD_PCT", "3.0"))

_TICKER_RE_MSG = "Ticker symbols must be 1-10 letters, optionally with a share-class suffix (e.g. BRK.B)."


def _clean_symbol(symbol: str) -> str:
    symbol = (symbol or "").strip().upper()
    if not symbol or len(symbol) > 10:
        raise ValueError(_TICKER_RE_MSG)
    return symbol


_embedding_model = None


def _get_embedding_model():
    """Lazy-load the sentence-transformers model (expensive; load once per process)."""
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer

        logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    return _embedding_model


# ----------------------------------------------------------------------------
# Watchlist (read + write)
# ----------------------------------------------------------------------------


def get_watchlist(email: str) -> list[dict]:
    """Return everything on a user's watchlist, most recently added first."""
    return lakebase.run_query(
        f"""
        SELECT symbol, latest_price, added_at, updated_at
        FROM {WATCHLIST_TABLE}
        WHERE email = %s
        ORDER BY added_at DESC
        """,
        (email,),
    )


def add_to_watchlist(email: str, symbol: str) -> dict:
    """Fetch a live quote for `symbol` and add/update it on the user's watchlist."""
    symbol = _clean_symbol(symbol)
    quote = MassiveClient().get_quote(symbol)

    lakebase.run_write(
        f"""
        INSERT INTO {WATCHLIST_TABLE} (symbol, email, latest_price, added_at, updated_at)
        VALUES (%s, %s, %s, now(), now())
        ON CONFLICT (symbol, email) DO UPDATE
            SET latest_price = EXCLUDED.latest_price, updated_at = now()
        """,
        (symbol, email, quote["price"]),
    )
    return {"status": "success", "symbol": symbol, "quote": quote}


def remove_from_watchlist(email: str, symbol: str) -> dict:
    """Remove a symbol from the user's watchlist."""
    symbol = _clean_symbol(symbol)
    deleted = lakebase.run_write(
        f"DELETE FROM {WATCHLIST_TABLE} WHERE symbol = %s AND email = %s",
        (symbol, email),
    )
    if not deleted:
        return {"status": "not_found", "symbol": symbol}
    return {"status": "success", "symbol": symbol}


# ----------------------------------------------------------------------------
# Prices and fundamentals (read; live quote is a write-through cache)
# ----------------------------------------------------------------------------


def get_quote(symbol: str) -> dict:
    """Live quote straight from Massive (not from Lakebase - always fresh)."""
    return MassiveClient().get_quote(_clean_symbol(symbol))


def get_price_history(symbol: str, days: int = 30) -> list[dict]:
    """Historical price snapshots for a ticker, oldest first.

    Reads from `price_snapshots`, which the Spark pipeline appends to once
    per run - so history accumulates the longer the pipeline has been
    scheduled, rather than being backfilled from a single API call.
    """
    symbol = _clean_symbol(symbol)
    rows = lakebase.run_query(
        f"""
        SELECT price, open, high, low, volume, change, change_percent, as_of, captured_at
        FROM {PRICE_SNAPSHOTS_TABLE}
        WHERE symbol = %s AND captured_at >= now() - (%s || ' days')::interval
        ORDER BY captured_at ASC
        """,
        (symbol, days),
    )
    return rows


def get_company_fundamentals(symbol: str) -> dict:
    """Company fundamentals for a ticker - from Lakebase if the Spark pipeline
    has already synced it, otherwise fetched live from Massive as a fallback
    (and NOT persisted here - persistence is the pipeline's job)."""
    symbol = _clean_symbol(symbol)
    rows = lakebase.run_query(
        f"SELECT * FROM {COMPANIES_TABLE} WHERE symbol = %s",
        (symbol,),
    )
    if rows:
        return rows[0]

    logger.info(f"{symbol} not yet in {COMPANIES_TABLE}; fetching live from Massive")
    return MassiveClient().get_company_details(symbol)


def compare_tickers(symbols: list[str]) -> dict:
    """Side-by-side comparison of multiple tickers: latest price, fundamentals,
    and most recent news sentiment for each."""
    results = []
    for raw_symbol in symbols:
        try:
            symbol = _clean_symbol(raw_symbol)
        except ValueError:
            continue

        entry: dict = {"symbol": symbol}
        try:
            entry["quote"] = get_quote(symbol)
        except Exception as exc:
            entry["quote_error"] = str(exc)

        try:
            entry["fundamentals"] = get_company_fundamentals(symbol)
        except Exception as exc:
            entry["fundamentals_error"] = str(exc)

        recent_news = lakebase.run_query(
            f"""
            SELECT title, sentiment, published_utc, article_url
            FROM {NEWS_TABLE}
            WHERE ticker = %s
            ORDER BY published_utc DESC
            LIMIT 5
            """,
            (symbol,),
        )
        entry["recent_news"] = recent_news
        results.append(entry)

    return {"comparison": results}


# ----------------------------------------------------------------------------
# News + semantic (vector) search (read)
# ----------------------------------------------------------------------------


def get_recent_news(symbol: str, limit: int = 10) -> list[dict]:
    """Recent synced news articles for a ticker, most recent first."""
    symbol = _clean_symbol(symbol)
    return lakebase.run_query(
        f"""
        SELECT id, title, description, sentiment, sentiment_reasoning,
               article_url, publisher_name, published_utc
        FROM {NEWS_TABLE}
        WHERE ticker = %s
        ORDER BY published_utc DESC
        LIMIT %s
        """,
        (symbol, limit),
    )


def vector_search(query: str, limit: int = 10, search_chunks: bool = True) -> dict:
    """Semantic search over ticker news using pgvector cosine similarity.

    Embeds `query` with the same model used to embed the news corpus, then
    retrieves the closest document-level and (optionally) chunk-level
    passages - e.g. "companies exposed to rising interest rates in the
    regional banking sector" rather than a plain keyword/ticker lookup.
    """
    if not query or not query.strip():
        return {"error": "Query text is required"}

    embedding = _get_embedding_model().encode(query).tolist()
    embedding_str = str(embedding)

    documents = lakebase.run_query(
        f"""
        SELECT e.id, e.ticker, e.title, e.published_utc,
               1 - (e.embedding <=> %s::vector) AS similarity,
               d.description, d.article_url, d.sentiment
        FROM {EMBEDDINGS_TABLE} e
        LEFT JOIN {NEWS_TABLE} d ON e.id = d.id
        ORDER BY e.embedding <=> %s::vector
        LIMIT %s
        """,
        (embedding_str, embedding_str, limit),
    )

    chunks = []
    if search_chunks:
        chunks = lakebase.run_query(
            f"""
            SELECT c.id, c.article_id, c.ticker, c.chunk_index, c.chunk_text,
                   1 - (c.embedding <=> %s::vector) AS similarity,
                   d.title, d.article_url, d.published_utc
            FROM {CHUNK_EMBEDDINGS_TABLE} c
            LEFT JOIN {NEWS_TABLE} d ON c.article_id = d.id
            ORDER BY c.embedding <=> %s::vector
            LIMIT %s
            """,
            (embedding_str, embedding_str, limit),
        )

    return {"query": query, "documents": documents, "chunks": chunks, "model": EMBEDDING_MODEL}


# ----------------------------------------------------------------------------
# Research notes and analysis reports (read + write)
# ----------------------------------------------------------------------------


def save_research_note(email: str, symbol: str, note_text: str) -> dict:
    symbol = _clean_symbol(symbol)
    if not note_text or not note_text.strip():
        raise ValueError("note_text is required")

    row = lakebase.run_query(
        f"""
        INSERT INTO {RESEARCH_NOTES_TABLE} (email, symbol, note_text, created_at)
        VALUES (%s, %s, %s, now())
        RETURNING id, symbol, created_at
        """,
        (email, symbol, note_text.strip()),
    )
    return {"status": "success", **row[0]}


def get_research_notes(email: str, symbol: str | None = None, limit: int = 20) -> list[dict]:
    if symbol:
        return lakebase.run_query(
            f"""
            SELECT id, symbol, note_text, created_at FROM {RESEARCH_NOTES_TABLE}
            WHERE email = %s AND symbol = %s
            ORDER BY created_at DESC LIMIT %s
            """,
            (email, _clean_symbol(symbol), limit),
        )
    return lakebase.run_query(
        f"""
        SELECT id, symbol, note_text, created_at FROM {RESEARCH_NOTES_TABLE}
        WHERE email = %s
        ORDER BY created_at DESC LIMIT %s
        """,
        (email, limit),
    )


def save_analysis_report(
    email: str, symbol: str, summary: str, thesis: str | None = None, sources: list | None = None
) -> dict:
    symbol = _clean_symbol(symbol)
    if not summary or not summary.strip():
        raise ValueError("summary is required")

    row = lakebase.run_query(
        f"""
        INSERT INTO {ANALYSIS_REPORTS_TABLE} (email, symbol, thesis, summary, sources, created_at)
        VALUES (%s, %s, %s, %s, %s, now())
        RETURNING id, symbol, created_at
        """,
        (email, symbol, thesis, summary.strip(), json.dumps(sources or [])),
    )
    return {"status": "success", **row[0]}


def get_analysis_reports(email: str, symbol: str | None = None, limit: int = 20) -> list[dict]:
    if symbol:
        return lakebase.run_query(
            f"""
            SELECT id, symbol, thesis, summary, sources, created_at FROM {ANALYSIS_REPORTS_TABLE}
            WHERE email = %s AND symbol = %s
            ORDER BY created_at DESC LIMIT %s
            """,
            (email, _clean_symbol(symbol), limit),
        )
    return lakebase.run_query(
        f"""
        SELECT id, symbol, thesis, summary, sources, created_at FROM {ANALYSIS_REPORTS_TABLE}
        WHERE email = %s
        ORDER BY created_at DESC LIMIT %s
        """,
        (email, limit),
    )


# ----------------------------------------------------------------------------
# "Since your last visit" (read + write - reads price/news history, writes
# the visit checkpoint)
# ----------------------------------------------------------------------------


def get_notable_changes(email: str) -> dict:
    """Flag notable price moves or news for the user's watchlist since their
    last recorded visit. Does NOT advance the checkpoint - call mark_visit()
    once the user has actually seen these results."""
    checkpoint_rows = lakebase.run_query(
        f"SELECT last_seen_at FROM {USER_CHECKPOINTS_TABLE} WHERE email = %s",
        (email,),
    )
    last_seen_at = checkpoint_rows[0]["last_seen_at"] if checkpoint_rows else None

    watchlist_symbols = [row["symbol"] for row in get_watchlist(email)]
    if not watchlist_symbols:
        return {"last_seen_at": last_seen_at, "price_moves": [], "news": []}

    if last_seen_at is None:
        # First-ever visit: nothing to compare against yet.
        return {"last_seen_at": None, "price_moves": [], "news": [], "note": "First visit - no baseline yet."}

    price_moves = lakebase.run_query(
        f"""
        SELECT symbol, price, change_percent, captured_at
        FROM {PRICE_SNAPSHOTS_TABLE}
        WHERE symbol = ANY(%s)
          AND captured_at > %s
          AND ABS(change_percent) >= %s
        ORDER BY ABS(change_percent) DESC
        """,
        (watchlist_symbols, last_seen_at, NOTABLE_MOVE_THRESHOLD_PCT),
    )

    news = lakebase.run_query(
        f"""
        SELECT ticker, title, sentiment, published_utc, article_url
        FROM {NEWS_TABLE}
        WHERE ticker = ANY(%s) AND published_utc > %s
        ORDER BY published_utc DESC
        LIMIT 25
        """,
        (watchlist_symbols, last_seen_at),
    )

    return {"last_seen_at": last_seen_at, "price_moves": price_moves, "news": news}


def mark_visit(email: str) -> dict:
    """Advance the user's checkpoint to now - call after showing them
    get_notable_changes() results, so the next check starts from here."""
    lakebase.run_write(
        f"""
        INSERT INTO {USER_CHECKPOINTS_TABLE} (email, last_seen_at)
        VALUES (%s, now())
        ON CONFLICT (email) DO UPDATE SET last_seen_at = now()
        """,
        (email,),
    )
    return {"status": "success", "last_seen_at": datetime.now(timezone.utc).isoformat()}
