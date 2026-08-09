# Databricks notebook source
# MAGIC %md
# MAGIC # Market Data Pipeline (Spark) - AI Stock Market Research Assistant
# MAGIC
# MAGIC This is the capstone's Spark data pipeline. It is a genuine Spark job -
# MAGIC every read, transform, and write below uses Spark DataFrames, JDBC, and
# MAGIC distributed pandas UDFs (not a plain Python/pandas script running on a
# MAGIC single-node driver).
# MAGIC
# MAGIC Each run:
# MAGIC 1. Reads distinct watchlisted tickers from `watchlist` (Spark JDBC read).
# MAGIC 2. Fetches a price quote, company details, and recent news per ticker
# MAGIC    from the Massive API (rate-limited, so this part runs on the driver -
# MAGIC    see note below).
# MAGIC 3. Writes price quotes to `price_snapshots` (Spark JDBC append - this
# MAGIC    table's history is what "historical price" / "notable move" features
# MAGIC    read from).
# MAGIC 4. Writes company fundamentals through a staging table into `companies`
# MAGIC    (Spark JDBC append + a small psycopg2 upsert, since generic JDBC
# MAGIC    can't do `ON CONFLICT`).
# MAGIC 5. Flattens raw news JSON with genuine Spark DataFrame operations
# MAGIC    (`explode`, `from_json`, window functions) and upserts into
# MAGIC    `ticker_news_documents`.
# MAGIC 6. Computes document-level embeddings for any new articles using a
# MAGIC    **Spark pandas_udf** - the embedding model runs once per executor
# MAGIC    and is applied to a Spark DataFrame column, distributing the
# MAGIC    embedding workload across the cluster.
# MAGIC 7. Computes chunk-level embeddings for full article bodies using
# MAGIC    **`mapInPandas`** - each partition's executor fetches article HTML
# MAGIC    (via `trafilatura`), splits it into overlapping chunks, and embeds
# MAGIC    each chunk, so the network I/O + embedding work for different
# MAGIC    articles happens in parallel across the cluster.
# MAGIC
# MAGIC WHY THE FETCH STEP (2) ISN'T DISTRIBUTED: the Massive free tier caps
# MAGIC requests at ~5/min, so spreading these calls across executors buys
# MAGIC nothing (you'd still be limited to 5/min in aggregate, just harder to
# MAGIC reason about) and each executor would need its own copy of the API
# MAGIC secret. Steps 6-7 are the actual CPU/IO-bound work worth distributing,
# MAGIC and that's exactly where Spark is used.

# COMMAND ----------

# DBTITLE 1,Install dependencies
# MAGIC %pip install -q "databricks-sdk>=0.30.0" psycopg2-binary sentence-transformers trafilatura requests
dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Widgets (config)
dbutils.widgets.text("watchlist_table_name", "watchlist")
dbutils.widgets.text("companies_table_name", "companies")
dbutils.widgets.text("price_snapshots_table_name", "price_snapshots")
dbutils.widgets.text("news_table_name", "ticker_news_documents")
dbutils.widgets.text("embeddings_table_name", "ticker_news_embeddings")
dbutils.widgets.text("chunk_embeddings_table_name", "ticker_news_chunk_embeddings")
dbutils.widgets.text("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
dbutils.widgets.text("lakebase_secret_scope", "database")
dbutils.widgets.text("lakebase_secret_key", "lakebase-url")
dbutils.widgets.text("massive_secret_scope", "massive")
dbutils.widgets.text("massive_secret_key", "api-key")
dbutils.widgets.text("massive_api_base_url", "https://api.massive.com")
dbutils.widgets.text("news_fetch_limit", "20")
dbutils.widgets.text("max_requests_per_minute", "5")
dbutils.widgets.text("company_refresh_days", "7")
dbutils.widgets.text("chunk_size", "800")
dbutils.widgets.text("chunk_overlap", "100")
dbutils.widgets.text(
    "model_cache_volume_path",
    "/Volumes/workspace/default/model_cache",
    "UC Volume path to cache the embedding model (create the volume first; adjust catalog/schema to match your workspace)",
)

WATCHLIST_TABLE = dbutils.widgets.get("watchlist_table_name")
COMPANIES_TABLE = dbutils.widgets.get("companies_table_name")
PRICE_SNAPSHOTS_TABLE = dbutils.widgets.get("price_snapshots_table_name")
NEWS_TABLE = dbutils.widgets.get("news_table_name")
EMBEDDINGS_TABLE = dbutils.widgets.get("embeddings_table_name")
CHUNK_EMBEDDINGS_TABLE = dbutils.widgets.get("chunk_embeddings_table_name")
EMBEDDING_MODEL_NAME = dbutils.widgets.get("embedding_model")
LAKEBASE_SECRET_SCOPE = dbutils.widgets.get("lakebase_secret_scope")
LAKEBASE_SECRET_KEY = dbutils.widgets.get("lakebase_secret_key")
MASSIVE_SECRET_SCOPE = dbutils.widgets.get("massive_secret_scope")
MASSIVE_SECRET_KEY = dbutils.widgets.get("massive_secret_key")
MASSIVE_API_BASE_URL = dbutils.widgets.get("massive_api_base_url")
NEWS_FETCH_LIMIT = int(dbutils.widgets.get("news_fetch_limit"))
MAX_REQUESTS_PER_MINUTE = int(dbutils.widgets.get("max_requests_per_minute"))
COMPANY_REFRESH_DAYS = int(dbutils.widgets.get("company_refresh_days"))
CHUNK_SIZE = int(dbutils.widgets.get("chunk_size"))
CHUNK_OVERLAP = int(dbutils.widgets.get("chunk_overlap"))
MODEL_CACHE_VOLUME_PATH = dbutils.widgets.get("model_cache_volume_path")

EMBEDDING_DIMS = {
    "sentence-transformers/all-MiniLM-L6-v2": 384,
    "sentence-transformers/all-mpnet-base-v2": 768,
    "BAAI/bge-small-en-v1.5": 384,
    "BAAI/bge-base-en-v1.5": 768,
    "BAAI/bge-large-en-v1.5": 1024,
}
EMBEDDING_DIM = EMBEDDING_DIMS.get(EMBEDDING_MODEL_NAME, 384)

# COMMAND ----------

# DBTITLE 1,Resolve Lakebase connection (JDBC + psycopg2)
import base64
from urllib.parse import urlparse

from databricks.sdk import WorkspaceClient

w = WorkspaceClient()


def _secret(scope: str, key: str) -> str:
    return base64.b64decode(w.secrets.get_secret(scope=scope, key=key).value).decode("utf-8")


lakebase_url = _secret(LAKEBASE_SECRET_SCOPE, LAKEBASE_SECRET_KEY)
parsed = urlparse(lakebase_url)

db_host = parsed.hostname
db_port = parsed.port or 5432
db_name = parsed.path.lstrip("/")
db_user = parsed.username
db_password = parsed.password

# Spark's JDBC data source needs a jdbc:// URL + a properties dict (not the
# postgresql:// URL psycopg2 uses) - same credentials, different format.
JDBC_URL = f"jdbc:postgresql://{db_host}:{db_port}/{db_name}?sslmode=require"
JDBC_PROPERTIES = {
    "user": db_user,
    "password": db_password,
    "driver": "org.postgresql.Driver",
}

print(f"Lakebase: {db_host}:{db_port}/{db_name} (JDBC + psycopg2 both configured)")

# COMMAND ----------

# DBTITLE 1,psycopg2 connection helper (closes the socket, not just the txn)
from contextlib import contextmanager

import psycopg2


@contextmanager
def pg_connect():
    """psycopg2's own `with conn:` only commits/rolls back the transaction -
    it does NOT close the connection. Use this instead so every merge step
    below actually releases its socket when done."""
    conn = psycopg2.connect(
        host=db_host, port=db_port, dbname=db_name, user=db_user, password=db_password, sslmode="require"
    )
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cache the embedding model to a Unity Catalog Volume
# MAGIC
# MAGIC **Why this matters on Free Edition (or any tier):** the pandas UDF and
# MAGIC `mapInPandas` steps below run on Spark executors, not the driver.
# MAGIC Executors are not guaranteed the same general internet egress as the
# MAGIC notebook driver - so if `SentenceTransformer(...)` tries to download
# MAGIC model weights from huggingface.co *inside* the UDF, it can silently
# MAGIC hang or fail depending on the workspace's network policy. Downloading
# MAGIC the model once here (on the driver, which we already know has internet
# MAGIC access - it's how `%pip install` above worked) and saving it to a UC
# MAGIC Volume means executors only ever need read access to governed Databricks
# MAGIC storage, not the open internet.
# MAGIC
# MAGIC If `MODEL_CACHE_VOLUME_PATH` doesn't exist yet, create it first, e.g.:
# MAGIC ```sql
# MAGIC CREATE VOLUME IF NOT EXISTS workspace.default.model_cache;
# MAGIC ```
# MAGIC (adjust the catalog/schema to match your workspace, and the
# MAGIC `model_cache_volume_path` widget to match).

# COMMAND ----------

# DBTITLE 1,Warm + cache the model; resolve the path executors will load from
import os as _os

from sentence_transformers import SentenceTransformer

_model_dir_name = EMBEDDING_MODEL_NAME.replace("/", "__")
_model_volume_path = f"{MODEL_CACHE_VOLUME_PATH.rstrip('/')}/{_model_dir_name}"

MODEL_LOAD_PATH = EMBEDDING_MODEL_NAME  # fallback: load by name (needs executor internet)
try:
    if not _os.path.isdir(_model_volume_path) or not _os.listdir(_model_volume_path):
        print(f"Downloading {EMBEDDING_MODEL_NAME} on the driver and saving to {_model_volume_path} ...")
        _driver_model = SentenceTransformer(EMBEDDING_MODEL_NAME, cache_folder="/tmp/.cache/huggingface")
        _os.makedirs(_model_volume_path, exist_ok=True)
        _driver_model.save(_model_volume_path)
    MODEL_LOAD_PATH = _model_volume_path
    print(f"Executors will load the model from the volume: {MODEL_LOAD_PATH}")
except Exception as exc:
    print(
        f"WARNING: could not cache the model to {_model_volume_path} ({exc}). "
        f"Falling back to loading '{EMBEDDING_MODEL_NAME}' by name inside each UDF - "
        f"this requires executors to have direct internet access to huggingface.co. "
        f"If Steps 6/7 hang or fail below, create the volume "
        f"(CREATE VOLUME IF NOT EXISTS <catalog>.<schema>.model_cache) and rerun this cell."
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1 - Read watchlisted tickers (Spark JDBC read)

# COMMAND ----------

# DBTITLE 1,Read distinct tickers via Spark JDBC
watchlist_df = (
    spark.read.format("jdbc")
    .option("url", JDBC_URL)
    .option("dbtable", WATCHLIST_TABLE)
    .options(**JDBC_PROPERTIES)
    .load()
)

tickers = [
    row["symbol"].strip().upper()
    for row in watchlist_df.select("symbol").distinct().collect()
    if row["symbol"]
]
print(f"Found {len(tickers)} watchlisted tickers: {tickers}")

if not tickers:
    dbutils.notebook.exit("No tickers on any watchlist yet - nothing to do.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2 - Fetch quotes, fundamentals, and news from Massive
# MAGIC
# MAGIC Serial + rate-limited (driver-side) - see the note in the intro cell for
# MAGIC why this part is not distributed across Spark workers.

# COMMAND ----------

# DBTITLE 1,Fetch from Massive API
import json
import time
from datetime import datetime, timedelta, timezone

import requests

_session = requests.Session()
_session.headers.update(
    {"Authorization": f"Bearer {_secret(MASSIVE_SECRET_SCOPE, MASSIVE_SECRET_KEY)}"}
)
_seconds_between_requests = 60.0 / MAX_REQUESTS_PER_MINUTE


def _rate_limited_get(path: str, params: dict | None = None) -> dict:
    resp = _session.get(f"{MASSIVE_API_BASE_URL}{path}", params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def _fetch_quote(ticker: str) -> dict | None:
    """GET /v2/aggs/ticker/{ticker}/prev -> previous day's OHLCV bar."""
    try:
        data = _rate_limited_get(f"/v2/aggs/ticker/{ticker}/prev")
    except requests.exceptions.RequestException as exc:
        print(f"  quote fetch failed for {ticker}: {exc}")
        return None
    results = data.get("results") or []
    if not results:
        return None
    bar = results[0]
    open_price = float(bar.get("o", 0) or 0)
    close_price = float(bar.get("c", 0) or 0)
    change = close_price - open_price
    change_pct = (change / open_price * 100) if open_price else None
    ts = bar.get("t")
    as_of = datetime.fromtimestamp(ts / 1000, tz=timezone.utc).isoformat() if ts else None
    return {
        "symbol": ticker,
        "price": close_price,
        "open": open_price,
        "high": float(bar.get("h", 0) or 0),
        "low": float(bar.get("l", 0) or 0),
        "volume": int(bar.get("v", 0) or 0),
        "change": round(change, 4),
        "change_percent": round(change_pct, 4) if change_pct is not None else None,
        "as_of": as_of,
    }


def _fetch_company_details(ticker: str) -> dict | None:
    """GET /v3/reference/tickers/{ticker} -> name, sector, market cap, etc.

    NOTE: adjust the field mapping below if Massive's real response shape
    differs from this (Polygon-compatible) reference-ticker shape.
    """
    try:
        data = _rate_limited_get(f"/v3/reference/tickers/{ticker}")
    except requests.exceptions.RequestException as exc:
        print(f"  company details fetch failed for {ticker}: {exc}")
        return None
    result = data.get("results") or {}
    if not result:
        return None
    return {
        "symbol": ticker,
        "name": result.get("name"),
        "sector": result.get("sic_description"),
        "industry": result.get("type"),
        "description": result.get("description"),
        "market_cap": result.get("market_cap"),
        "employees": result.get("total_employees"),
        "homepage_url": result.get("homepage_url"),
        "exchange": result.get("primary_exchange"),
        "currency": result.get("currency_name"),
        "payload": json.dumps(result),
    }


def _fetch_news(ticker: str, limit: int) -> list[dict]:
    """GET /v2/reference/news -> recent articles for one ticker."""
    try:
        data = _rate_limited_get(
            "/v2/reference/news",
            params={"ticker": ticker, "limit": limit, "order": "desc", "sort": "published_utc"},
        )
    except requests.exceptions.RequestException as exc:
        print(f"  news fetch failed for {ticker}: {exc}")
        return []
    return data.get("results") or []


# Skip re-fetching fundamentals for companies refreshed recently.
_stale_cutoff = datetime.now(timezone.utc) - timedelta(days=COMPANY_REFRESH_DAYS)
try:
    _fresh_companies = {
        row["symbol"]
        for row in spark.read.format("jdbc")
        .option("url", JDBC_URL)
        .option("dbtable", COMPANIES_TABLE)
        .options(**JDBC_PROPERTIES)
        .load()
        .filter(f"updated_at >= '{_stale_cutoff.isoformat()}'")
        .select("symbol")
        .collect()
    }
except Exception:
    _fresh_companies = set()

quote_rows, company_rows, news_rows = [], [], []

for i, ticker in enumerate(tickers):
    if i > 0:
        time.sleep(_seconds_between_requests)

    quote = _fetch_quote(ticker)
    if quote:
        quote_rows.append(quote)

    if ticker not in _fresh_companies:
        time.sleep(_seconds_between_requests)
        details = _fetch_company_details(ticker)
        if details:
            company_rows.append(details)

    time.sleep(_seconds_between_requests)
    for article in _fetch_news(ticker, NEWS_FETCH_LIMIT):
        article["_watchlist_ticker"] = ticker  # tag which ticker this fetch was for
        news_rows.append(article)

print(f"Fetched: {len(quote_rows)} quotes, {len(company_rows)} company profiles, {len(news_rows)} news articles")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3 - Write price snapshots (Spark JDBC append)
# MAGIC
# MAGIC `price_snapshots` is append-only, so a plain JDBC append is all we need
# MAGIC - no upsert/merge step required for this table.

# COMMAND ----------

# DBTITLE 1,Spark DataFrame -> price_snapshots
from pyspark.sql import Row
from pyspark.sql.types import (
    StringType,
    StructField,
    StructType,
    DoubleType,
    LongType,
    TimestampType,
)

price_schema = StructType(
    [
        StructField("symbol", StringType(), False),
        StructField("price", DoubleType(), False),
        StructField("open", DoubleType(), True),
        StructField("high", DoubleType(), True),
        StructField("low", DoubleType(), True),
        StructField("volume", LongType(), True),
        StructField("change", DoubleType(), True),
        StructField("change_percent", DoubleType(), True),
        StructField("as_of", StringType(), True),
    ]
)

from pyspark.sql import functions as F

if quote_rows:
    price_df = spark.createDataFrame(quote_rows, schema=price_schema).withColumn(
        "as_of", F.to_timestamp("as_of")
    )

    (
        price_df.write.format("jdbc")
        .option("url", JDBC_URL)
        .option("dbtable", PRICE_SNAPSHOTS_TABLE)
        .options(**JDBC_PROPERTIES)
        .mode("append")
        .save()
    )
    print(f"Appended {price_df.count()} rows to {PRICE_SNAPSHOTS_TABLE}")
else:
    print("No quotes fetched this run.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4 - Upsert company fundamentals (Spark JDBC append -> psycopg2 merge)

# COMMAND ----------

# DBTITLE 1,Spark DataFrame -> companies_staging -> psycopg2 merge
import psycopg2

STAGING_TABLE = f"{COMPANIES_TABLE}_staging"

if company_rows:
    company_df = spark.createDataFrame(company_rows)

    with pg_connect() as _conn:
        with _conn.cursor() as _cur:
            _cur.execute(f"TRUNCATE TABLE {STAGING_TABLE}")

    (
        company_df.write.format("jdbc")
        .option("url", JDBC_URL)
        .option("dbtable", STAGING_TABLE)
        .options(**JDBC_PROPERTIES)
        .mode("append")
        .save()
    )

    with pg_connect() as _conn:
        with _conn.cursor() as _cur:
            _cur.execute(
                f"""
                INSERT INTO {COMPANIES_TABLE} (
                    symbol, name, sector, industry, description, market_cap,
                    employees, homepage_url, exchange, currency, payload, updated_at
                )
                SELECT
                    symbol, name, sector, industry, description, market_cap,
                    employees, homepage_url, exchange, currency, payload::jsonb, now()
                FROM {STAGING_TABLE}
                ON CONFLICT (symbol) DO UPDATE SET
                    name = EXCLUDED.name,
                    sector = EXCLUDED.sector,
                    industry = EXCLUDED.industry,
                    description = EXCLUDED.description,
                    market_cap = EXCLUDED.market_cap,
                    employees = EXCLUDED.employees,
                    homepage_url = EXCLUDED.homepage_url,
                    exchange = EXCLUDED.exchange,
                    currency = EXCLUDED.currency,
                    payload = EXCLUDED.payload,
                    updated_at = now()
                """
            )
    print(f"Upserted {len(company_rows)} company profiles into {COMPANIES_TABLE}")
else:
    print("No fresh company profiles to upsert this run.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5 - Flatten + upsert raw news (Spark DataFrame transforms)
# MAGIC
# MAGIC This replaces a hand-written Python loop with real Spark DataFrame
# MAGIC operations: `explode` over the nested `insights` array to find the
# MAGIC sentiment entry that matches the ticker we fetched this article for.

# COMMAND ----------

# DBTITLE 1,Spark transform: flatten sentiment insights
from pyspark.sql import functions as F

if news_rows:
    # Keep the raw payload as JSON text (schema-on-read via from_json below)
    # so odd/missing fields across articles don't break schema inference.
    raw_news_df = spark.createDataFrame(
        [
            Row(
                id=str(a.get("id")),
                ticker=a.get("_watchlist_ticker"),
                title=a.get("title", "") or "",
                description=a.get("description"),
                author=a.get("author"),
                article_url=a.get("article_url"),
                publisher_name=(a.get("publisher") or {}).get("name"),
                keywords=json.dumps(a.get("keywords", [])),
                published_utc=a.get("published_utc"),
                insights_json=json.dumps(a.get("insights", [])),
                payload=json.dumps(a),
            )
            for a in news_rows
        ]
    )

    insight_schema = "array<struct<ticker:string,sentiment:string,sentiment_reasoning:string>>"

    flattened_df = (
        raw_news_df.withColumn("insights", F.from_json("insights_json", insight_schema))
        .withColumn(
            "matched_insight",
            F.expr("filter(insights, x -> x.ticker = ticker)")[0],
        )
        .withColumn("sentiment", F.col("matched_insight.sentiment"))
        .withColumn("sentiment_reasoning", F.col("matched_insight.sentiment_reasoning"))
        .withColumn("published_utc", F.to_timestamp("published_utc"))
        .select(
            "id",
            "ticker",
            "title",
            "description",
            "author",
            "article_url",
            "publisher_name",
            "keywords",
            "sentiment",
            "sentiment_reasoning",
            "published_utc",
            "payload",
        )
        .dropDuplicates(["id"])
    )

    NEWS_STAGING_TABLE = f"{NEWS_TABLE}_staging"

    with pg_connect() as _conn:
        with _conn.cursor() as _cur:
            _cur.execute(f"TRUNCATE TABLE {NEWS_STAGING_TABLE}")

    (
        flattened_df.write.format("jdbc")
        .option("url", JDBC_URL)
        .option("dbtable", NEWS_STAGING_TABLE)
        .options(**JDBC_PROPERTIES)
        .mode("append")
        .save()
    )

    with pg_connect() as _conn:
        with _conn.cursor() as _cur:
            _cur.execute(
                f"""
                INSERT INTO {NEWS_TABLE} (
                    id, ticker, title, description, author, article_url,
                    publisher_name, keywords, sentiment, sentiment_reasoning,
                    published_utc, payload, synced_at
                )
                SELECT
                    id, ticker, title, description, author, article_url,
                    publisher_name, keywords::jsonb, sentiment, sentiment_reasoning,
                    published_utc, payload::jsonb, now()
                FROM {NEWS_STAGING_TABLE}
                ON CONFLICT (id) DO UPDATE SET
                    sentiment = EXCLUDED.sentiment,
                    sentiment_reasoning = EXCLUDED.sentiment_reasoning,
                    synced_at = now()
                """
            )
    print(f"Upserted {flattened_df.count()} news articles into {NEWS_TABLE}")
else:
    print("No news articles fetched this run.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6 - Document-level embeddings (distributed `pandas_udf`)
# MAGIC
# MAGIC Reads any `ticker_news_documents` rows that don't yet have an embedding
# MAGIC (a Spark `left_anti` join), then computes embeddings for `title +
# MAGIC description` using a **pandas UDF** - the embedding model loads once per
# MAGIC executor process and every partition's batch is encoded in parallel.

# COMMAND ----------

# DBTITLE 1,pandas_udf embedding computation
from pyspark.sql.types import ArrayType, FloatType
from pyspark.sql.functions import pandas_udf
import pandas as pd

docs_df = (
    spark.read.format("jdbc")
    .option("url", JDBC_URL)
    .option("dbtable", NEWS_TABLE)
    .options(**JDBC_PROPERTIES)
    .load()
)
existing_embeddings_df = (
    spark.read.format("jdbc")
    .option("url", JDBC_URL)
    .option("dbtable", EMBEDDINGS_TABLE)
    .options(**JDBC_PROPERTIES)
    .load()
    .select("id")
)

new_docs_df = (
    docs_df.join(existing_embeddings_df, on="id", how="left_anti")
    .withColumn(
        "embedding_text",
        F.concat_ws(". ", F.col("title"), F.coalesce(F.col("description"), F.lit(""))),
    )
    .withColumn("model_name", F.lit(EMBEDDING_MODEL_NAME))
    .select("id", "ticker", "title", "published_utc", "embedding_text", "model_name")
)

new_docs_count = new_docs_df.count()
print(f"{new_docs_count} articles need document-level embeddings")

if new_docs_count > 0:

    @pandas_udf(ArrayType(FloatType()))
    def embed_text(texts: pd.Series) -> pd.Series:
        # Runs once per batch, per executor process. sentence-transformers
        # caches the loaded model on the executor's Python worker across
        # batches within the same task, avoiding a reload per row. Loads
        # from MODEL_LOAD_PATH (a UC Volume, ideally - see the caching cell
        # above) so this doesn't need executor-level internet access.
        from sentence_transformers import SentenceTransformer

        global _model
        try:
            model = _model
        except NameError:
            model = SentenceTransformer(MODEL_LOAD_PATH, cache_folder="/tmp/.cache/huggingface")
            _model = model
        vectors = model.encode(texts.tolist(), show_progress_bar=False)
        return pd.Series([v.tolist() for v in vectors])

    embedded_docs_df = new_docs_df.withColumn("embedding", embed_text(F.col("embedding_text")))

    # pgvector's column type isn't a JDBC-writable type, so serialize the
    # float array as a Postgres array-literal string ("[0.1,0.2,...]") and
    # cast it with ::vector during the psycopg2 merge step below.
    embedded_docs_df = embedded_docs_df.withColumn(
        "embedding_str",
        F.concat(F.lit("["), F.concat_ws(",", F.col("embedding")), F.lit("]")),
    ).select("id", "ticker", "title", "published_utc", "embedding_str", "model_name")

    EMBEDDINGS_STAGING_TABLE = f"{EMBEDDINGS_TABLE}_staging"

    with pg_connect() as _conn:
        with _conn.cursor() as _cur:
            _cur.execute(f"TRUNCATE TABLE {EMBEDDINGS_STAGING_TABLE}")

    (
        embedded_docs_df.write.format("jdbc")
        .option("url", JDBC_URL)
        .option("dbtable", EMBEDDINGS_STAGING_TABLE)
        .options(**JDBC_PROPERTIES)
        .mode("append")
        .save()
    )

    with pg_connect() as _conn:
        with _conn.cursor() as _cur:
            _cur.execute(
                f"""
                INSERT INTO {EMBEDDINGS_TABLE} (id, ticker, title, published_utc, embedding, model_name, embedded_at)
                SELECT id, ticker, title, published_utc, embedding_str::vector, model_name, now()
                FROM {EMBEDDINGS_STAGING_TABLE}
                ON CONFLICT (id) DO NOTHING
                """
            )
    print(f"Wrote {new_docs_count} document embeddings into {EMBEDDINGS_TABLE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 7 - Chunk-level embeddings (driver fetch -> distributed `mapInPandas` compute)
# MAGIC
# MAGIC Fetching each article's full body (via `trafilatura`) means an HTTP call
# MAGIC to an arbitrary news-publisher domain. Executors aren't guaranteed the
# MAGIC same general internet egress as the driver (see the model-caching note
# MAGIC above), so - unlike the original single-pass design - this fetch runs on
# MAGIC the **driver** (thread-pooled, since these are independent I/O-bound
# MAGIC calls with no shared rate limit to respect, unlike the Massive API in
# MAGIC Step 2). Only the CPU-bound work - chunking + embedding the fetched
# MAGIC text, no network involved - is distributed via **`mapInPandas`**.

# COMMAND ----------

# DBTITLE 1,Fetch article bodies (driver, thread-pooled)
existing_chunk_article_ids_df = (
    spark.read.format("jdbc")
    .option("url", JDBC_URL)
    .option("dbtable", CHUNK_EMBEDDINGS_TABLE)
    .options(**JDBC_PROPERTIES)
    .load()
    .select(F.col("article_id").alias("id"))
    .distinct()
)

articles_needing_chunks_df = (
    docs_df.filter(F.col("article_url").isNotNull())
    .join(existing_chunk_article_ids_df, on="id", how="left_anti")
    .select("id", "ticker", "article_url")
)

articles_to_fetch = [row.asDict() for row in articles_needing_chunks_df.collect()]
print(f"{len(articles_to_fetch)} articles need chunk-level embeddings")

import trafilatura
from concurrent.futures import ThreadPoolExecutor, as_completed


def _fetch_body(article: dict) -> dict | None:
    try:
        downloaded = trafilatura.fetch_url(article["article_url"])
        body = trafilatura.extract(downloaded) if downloaded else None
    except Exception:
        body = None
    if not body:
        return None
    return {**article, "article_body": body}

fetched_bodies = []
if articles_to_fetch:
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(_fetch_body, a) for a in articles_to_fetch]
        for fut in as_completed(futures):
            result = fut.result()
            if result:
                fetched_bodies.append(result)

print(f"Fetched {len(fetched_bodies)}/{len(articles_to_fetch)} article bodies")

# COMMAND ----------

# DBTITLE 1,mapInPandas: chunk + embed fetched bodies (no network - safe to distribute)
if fetched_bodies:
    bodies_df = spark.createDataFrame(fetched_bodies)

    # Cap partitions to something sane for a small watchlist-sized workload.
    n_partitions = max(1, min(8, len(fetched_bodies)))
    bodies_df = bodies_df.repartition(n_partitions)

    chunk_output_schema = StructType(
        [
            StructField("id", StringType(), False),
            StructField("article_id", StringType(), False),
            StructField("ticker", StringType(), False),
            StructField("chunk_index", LongType(), False),
            StructField("chunk_text", StringType(), False),
            StructField("embedding_str", StringType(), False),
            StructField("model_name", StringType(), False),
        ]
    )

    def _chunk_and_embed(pdf_iterator):
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(MODEL_LOAD_PATH, cache_folder="/tmp/.cache/huggingface")

        def _chunk(text: str, size: int, overlap: int) -> list[str]:
            chunks, start = [], 0
            while start < len(text):
                end = start + size
                chunks.append(text[start:end])
                start = end - overlap
                if start <= 0:
                    break
            return [c for c in chunks if c.strip()]

        for pdf in pdf_iterator:
            out_rows = []
            for _, row in pdf.iterrows():
                text_chunks = _chunk(row["article_body"], CHUNK_SIZE, CHUNK_OVERLAP)
                if not text_chunks:
                    continue

                vectors = model.encode(text_chunks, show_progress_bar=False)
                for idx, (chunk_text, vec) in enumerate(zip(text_chunks, vectors)):
                    out_rows.append(
                        {
                            "id": f"{row['id']}-{idx}",
                            "article_id": row["id"],
                            "ticker": row["ticker"],
                            "chunk_index": idx,
                            "chunk_text": chunk_text,
                            "embedding_str": "[" + ",".join(str(float(v)) for v in vec) + "]",
                            "model_name": EMBEDDING_MODEL_NAME,
                        }
                    )
            yield pd.DataFrame(out_rows, columns=[f.name for f in chunk_output_schema.fields])

    chunks_df = bodies_df.mapInPandas(_chunk_and_embed, schema=chunk_output_schema)

    CHUNK_STAGING_TABLE = f"{CHUNK_EMBEDDINGS_TABLE}_staging"

    with pg_connect() as _conn:
        with _conn.cursor() as _cur:
            _cur.execute(f"TRUNCATE TABLE {CHUNK_STAGING_TABLE}")

    (
        chunks_df.write.format("jdbc")
        .option("url", JDBC_URL)
        .option("dbtable", CHUNK_STAGING_TABLE)
        .options(**JDBC_PROPERTIES)
        .mode("append")
        .save()
    )

    with pg_connect() as _conn:
        with _conn.cursor() as _cur:
            _cur.execute(
                f"""
                INSERT INTO {CHUNK_EMBEDDINGS_TABLE} (
                    id, article_id, ticker, chunk_index, chunk_text, embedding, model_name, embedded_at
                )
                SELECT id, article_id, ticker, chunk_index, chunk_text, embedding_str::vector, model_name, now()
                FROM {CHUNK_STAGING_TABLE}
                ON CONFLICT (id) DO NOTHING
                """
            )
    print(f"Wrote chunk embeddings for {len(fetched_bodies)} articles into {CHUNK_EMBEDDINGS_TABLE}")

# COMMAND ----------

print("Pipeline run complete.")
