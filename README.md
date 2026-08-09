# AI Stock Market Research Assistant (Databricks Capstone)

A research-only stock assistant built on Databricks Free Edition: Lakebase for
relational + vector storage, a genuine Spark pipeline for ingestion and
embeddings, a Databricks App frontend, and an Agent Bricks agent with tools
that read *and* write against your data.

This extends the `databricks-lakebase-app-day-2` (Massive + Lakebase +
watchlist + embeddings) pattern, and intentionally leaves out Day 3's
Alpaca paper-trading piece - this capstone is a research assistant, not a
trading bot.

## Capstone requirement -> where it lives

| Requirement | Implementation |
|---|---|
| Spark data pipeline | `notebooks/market_data_pipeline.py` - Spark JDBC reads/writes, `explode`/`from_json` DataFrame transforms, a **`pandas_udf`** for document embeddings, and **`mapInPandas`** for distributed article fetch+chunk+embed |
| Third-party API | Massive Stocks API (`mcp_server/massive_client.py`, `app/massive_client.py`) - quotes, fundamentals, news |
| Unstructured data processing | News article title/description + full body text, chunked and embedded into `pgvector` columns for semantic retrieval |
| Databricks App + frontend | `app/` - Flask app + `templates/index.html` ("Research Desk" UI: watchlist, price history, fundamentals, news, notes, reports, semantic search, optional agent chat) |
| AI agent with read + write tools | `mcp_server/research_mcp_server.py` - 15 tools registered with Agent Bricks (see table below) |

## Architecture

```
                         ┌─────────────────────────┐
                         │   Massive Stocks API     │
                         └────────────┬─────────────┘
                                      │
        ┌─────────────────────────────┼─────────────────────────────┐
        │                              │                              │
┌───────▼────────┐          ┌──────────▼──────────┐         ┌─────────▼────────┐
│  Spark pipeline │          │   MCP server (App)   │         │  Flask app (App)  │
│  (scheduled     │  writes  │  research_mcp_server │  calls  │  app.py + UI      │
│  Workflow)      │─────────▶│  .py  (read+write     │◀───────▶│  research_tools.py│
└────────┬────────┘  Lakebase│   tools)              │  shared └────────┬─────────┘
         │                   └──────────┬───────────┘  module          │
         │                              │ MCP                          │
         │                     ┌────────▼─────────┐                    │
         │                     │  Agent Bricks     │                   │
         │                     │  agent            │◀──chat────────────┘
         │                     └───────────────────┘   (optional /api/ask)
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│                Lakebase (Postgres + pgvector)             │
│  watchlist · companies · price_snapshots · research_notes  │
│  analysis_reports · user_checkpoints · ticker_news_documents│
│  ticker_news_embeddings · ticker_news_chunk_embeddings      │
└─────────────────────────────────────────────────────────┘
```

`mcp_server/` and `app/` each carry their own copies of `lakebase.py`,
`massive_client.py`, and `research_tools.py` - Databricks Apps deploy
independently from their own folder, so there's no shared-package install
step across apps. Both import the *same* `research_tools.py` logic, so the
UI and the agent are never out of sync - if you change how a note gets
saved, change it once and copy it to both folders.

## Agent capabilities -> tools

| Spec capability | Tool(s) |
|---|---|
| Pull current + historical price, summarize performance | `get_quote`, `get_price_history` |
| Surface/summarize relevant news & filings | `get_recent_news`, `vector_search` |
| Compare multiple tickers | `compare_tickers` |
| Add/remove watchlist tickers | `add_to_watchlist` (write), `remove_from_watchlist` (write) |
| Save a research note / analysis report | `save_research_note` (write), `save_analysis_report` (write) |
| Flag notable moves/news since last visit | `get_notable_changes`, `mark_visit` (write) |

Full list also includes `get_company_fundamentals`, `get_watchlist`,
`get_research_notes`, `get_analysis_reports`.

## Step-by-step setup

### 1. Create a Lakebase instance (skip if reusing one from Day 2)

Follow Day 2's instructions: Catalog > Lakebase > Create instance, enable
native password auth, create a role, copy the connection URL.

### 2. Create the schema

Run these against your Lakebase Postgres database (SQL editor, `psql`, or a
notebook `%sql` cell), in order:

```sql
-- sql/01_core_tables.sql
-- sql/02_news_tables.sql
-- sql/03_embeddings_tables.sql  (replace {{EMBEDDING_DIM}} with 384 for the
--                                 default all-MiniLM-L6-v2 model first)
```

### 3. Store secrets

From a Databricks notebook cell:

```python
%sh python setup_secrets.py
```

Prompts for your Lakebase connection URL and Massive API key (get the key at
[massive.com](https://massive.com)).

### 4. Get a few tickers onto a watchlist

The pipeline only fetches data for tickers that are already on someone's
watchlist, so seed at least one before the first pipeline run - easiest way
is to deploy the app (step 6) and use "Add" in the UI, or insert directly:

```sql
INSERT INTO watchlist (symbol, email, added_at, updated_at)
VALUES ('AAPL', 'you@example.com', now(), now());
```

### 5. Run the Spark pipeline once manually

Import `notebooks/market_data_pipeline.py` into your workspace (via a Git
folder - see step 6 below) and run it interactively first to confirm it
completes and populates `price_snapshots`, `companies`,
`ticker_news_documents`, and both embedding tables. Then schedule it:

- **Asset Bundle** (version-controlled): set your workspace URL in
  `databricks.yml`, then `databricks bundle deploy -t dev` and
  `databricks bundle run market_data_pipeline_job -t dev`. Flip
  `pause_status` to `UNPAUSED` in `resources/market_data_pipeline_job.yml`
  once you've confirmed a clean run.
- **Workflows UI** (no CLI): Workflows > Create Job > Notebook task pointed
  at `notebooks/market_data_pipeline.py`, with the same widget values shown
  in `resources/market_data_pipeline_job.yml`'s `base_parameters`, on a daily
  trigger.

### 6. Deploy the two Databricks Apps

1. **Create a Git folder** for this repo (Workspace > Create > Git folder).
2. **Deploy the MCP server**: Compute > Apps > Create app > Custom, point it
   at the Git folder's `mcp_server/` subfolder, deploy, and copy its app URL.
3. **Deploy the frontend**: repeat, pointing at `app/`, deploy, and open its
   URL - you should see the Research Desk UI with an empty watchlist.

### 7. Register the MCP server + build the Agent Bricks agent

1. **AI Gateway > MCPs > Add MCP**: paste the MCP server app's URL from step
   6.2 (streamable HTTP). Databricks will introspect it and list all 15
   tools.
2. **Agents > Agent Bricks > Create agent** (Custom LLM or Multi-agent
   supervisor). Add the registered MCP server as a tool source.
3. System prompt, e.g.:

   > You are a stock market research assistant. Use `get_watchlist` and
   > `get_notable_changes` to orient yourself on what the user is tracking
   > and what's changed. Use `get_quote`/`get_price_history` for price
   > questions, `vector_search`/`get_recent_news` for news and thematic
   > questions, and `compare_tickers` when asked to compare. When you
   > produce a real conclusion or recommendation, offer to save it via
   > `save_research_note` or `save_analysis_report` rather than only saying
   > it in chat. Never fabricate a price or fact you didn't get from a tool.

4. Deploy the agent, copy its Model Serving endpoint name, and (optionally)
   set `AGENT_ENDPOINT_NAME` in `app/app.yaml` to that name, then redeploy
   the frontend app to enable its in-app "Ask" chat tab.

## Files

```
sql/                                  Lakebase schema (core tables, news, embeddings)
notebooks/market_data_pipeline.py     Spark data pipeline
databricks.yml + resources/           Asset Bundle to schedule the pipeline
app/                                  Databricks App: Flask backend + Research Desk UI
mcp_server/                           Databricks App: MCP server exposing agent tools
setup_secrets.py                      One-time secret setup
.env.example                          Local dev env var template
```

## Will this run on Databricks Free Edition?

**Proven already, by your own Day 1-3 repos** - nothing here changes it:
Lakebase via psycopg2/SQLAlchemy, Databricks secrets, Databricks Apps making
outbound calls to a third-party API (Massive, and Alpaca in Day 3), and a
scheduled notebook that does `%pip install` + outbound `requests` calls from
the driver. If Day 3 worked for you, these pieces will too.

**New in this pipeline, and where the real risk is:** `pandas_udf` and
`mapInPandas` run on Spark **executors**, not the notebook driver - and
executors aren't guaranteed the same general internet egress as the driver,
even on paid tiers, let alone Free Edition. Two things in the original
single-pass design would have depended on that egress:

1. `SentenceTransformer(...)` downloading model weights from huggingface.co
   *inside* a UDF.
2. `trafilatura` fetching arbitrary news-publisher URLs *inside* a UDF.

I removed both dependencies before finalizing this: the pipeline now
downloads the embedding model once on the driver and caches it to a Unity
Catalog Volume (executors just read local governed storage), and article
bodies are fetched on the driver via a thread pool (the same proven network
path as the Massive API calls) - `mapInPandas` is only used for the
CPU-only chunk+embed step, which needs no network at all. Spark JDBC reads/
writes to Lakebase are unaffected either way, since Lakebase-Spark
integration is a first-party supported pattern.

**What I can't verify from here:** whether Free Edition's serverless Spark
fully supports `pandas_udf`/`mapInPandas` at all (vs. some older/limited
serverless tiers that didn't), and exact Volume-path behavior on Free
Edition specifically. My knowledge here may be stale - Databricks ships
platform changes faster than model training cutoffs track. **Test cheaply
before trusting the full run:**

```python
# Paste into a fresh cell on your Free Edition workspace and run it alone.
from pyspark.sql.functions import pandas_udf
import pandas as pd

@pandas_udf("int")
def add_one(s: pd.Series) -> pd.Series:
    return s + 1

spark.range(3).withColumn("plus_one", add_one("id")).show()
```

If that works, `pandas_udf`/`mapInPandas` are fine on your workspace and the
pipeline should run as designed. If it errors out on serverless, the
honest fallback is to drop Steps 6-7's UDFs and compute embeddings in a
plain Python loop instead (exactly what the original Day 2 notebook did) -
you'd lose the "distributed via Spark" flourish, but the pipeline still
reads/writes/transforms through genuine Spark DataFrames via JDBC (Steps
1, 3-5), which is what actually satisfies "a data pipeline in Spark."

One more Free-Edition-specific judgment call: `databricks bundle deploy`
(the Asset Bundle path in `databricks.yml`) needs CLI auth and workspace
permissions that I can't confirm are fully available on every Free Edition
account. The Workflows UI path in the setup steps below has no such
dependency - use that first if the CLI gives you trouble.

## Notes and other known limitations

- **Fundamentals endpoint is speculative.** `massive_client.py`'s
  `get_company_details` assumes a Polygon-compatible
  `/v3/reference/tickers/{ticker}` shape, matching the existing
  `/v2/aggs/.../prev` and `/v2/reference/news` calls already validated in
  Day 1/2. Adjust the field mapping if Massive's real response differs.
- **Historical price depth builds over time.** The free Massive tier only
  exposes a previous-day aggregate per call, so `price_snapshots` grows one
  row per ticker per pipeline run rather than being backfilled - run the
  pipeline daily for a week or two before "recent performance" answers get
  interesting.
- **pgvector isn't JDBC-writable.** Spark's generic JDBC writer can't target
  a `vector` column directly, so every embedding write goes: Spark computes
  the vector -> serializes it to a Postgres array-literal string -> JDBC
  append into a `_staging` table -> a short `psycopg2` step casts
  `::vector` and upserts into the real table. See the comments at the top of
  `sql/03_embeddings_tables.sql`.
- **Rate limits.** Massive's free tier is strict (default: 5 req/min in the
  pipeline). A larger watchlist means a slower pipeline run, not failures -
  the pipeline paces itself.
