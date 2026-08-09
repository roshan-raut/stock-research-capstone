"""
AI Stock Market Research Assistant - Databricks App frontend.

A Flask app that:
- Serves the research assistant UI (templates/index.html)
- Reads/writes Lakebase directly for watchlist, notes, reports, price
  history, and fundamentals (via research_tools.py - the SAME module the MCP
  server uses, so the UI and the agent never see different logic)
- Optionally proxies chat messages to the deployed Agent Bricks agent
  (Databricks Model Serving endpoint) so users can talk to the same agent
  that has the read/write tools, from inside this app

Run locally:
    python app.py
Deploy as a Databricks App using app.yaml.
"""

import logging
import os

from databricks.sdk import WorkspaceClient
from flask import Flask, jsonify, render_template, request

import research_tools

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("research-app")

app = Flask(__name__)
_w = WorkspaceClient()

# Name of the Model Serving endpoint behind your deployed Agent Bricks agent
# (Compute > Serving, or the Agent Bricks agent's "Endpoint" tab). Leave unset
# to disable the in-app chat panel and use the MCP tools via Playground/API
# directly instead.
AGENT_ENDPOINT_NAME = os.environ.get("AGENT_ENDPOINT_NAME", "")


def _current_user_email() -> str:
    """Databricks Apps inject the logged-in user's identity via the
    X-Forwarded-Email header. Fall back to the SDK's current-user API for
    local development where that header isn't set."""
    header_email = request.headers.get("X-Forwarded-Email")
    if header_email:
        return header_email
    return _w.current_user.me().user_name


@app.errorhandler(Exception)
def handle_exception(err):
    """Return JSON (not an HTML error page) on any unhandled error, so the
    frontend's resp.json() calls never choke."""
    logger.exception("Unhandled exception")
    status_code = getattr(err, "code", 500)
    if not isinstance(status_code, int):
        status_code = 500
    return jsonify({"error": str(err)}), status_code


@app.route("/healthz")
def healthz():
    return jsonify({"status": "ok"})


@app.route("/")
def index():
    return render_template("index.html", agent_enabled=bool(AGENT_ENDPOINT_NAME))


# ----------------------------------------------------------------------------
# Watchlist
# ----------------------------------------------------------------------------


@app.route("/api/watchlist", methods=["GET"])
def api_get_watchlist():
    return jsonify(research_tools.get_watchlist(_current_user_email()))


@app.route("/api/watchlist", methods=["POST"])
def api_add_to_watchlist():
    symbol = (request.json or {}).get("symbol", "")
    result = research_tools.add_to_watchlist(_current_user_email(), symbol)
    return jsonify(result)


@app.route("/api/watchlist/<symbol>", methods=["DELETE"])
def api_remove_from_watchlist(symbol: str):
    result = research_tools.remove_from_watchlist(_current_user_email(), symbol)
    status = 404 if result.get("status") == "not_found" else 200
    return jsonify(result), status


# ----------------------------------------------------------------------------
# Prices and fundamentals
# ----------------------------------------------------------------------------


@app.route("/api/quote/<symbol>")
def api_get_quote(symbol: str):
    return jsonify(research_tools.get_quote(symbol))


@app.route("/api/price-history/<symbol>")
def api_get_price_history(symbol: str):
    days = int(request.args.get("days", 30))
    return jsonify({"symbol": symbol.upper(), "history": research_tools.get_price_history(symbol, days)})


@app.route("/api/fundamentals/<symbol>")
def api_get_fundamentals(symbol: str):
    return jsonify(research_tools.get_company_fundamentals(symbol))


@app.route("/api/compare", methods=["POST"])
def api_compare_tickers():
    symbols = (request.json or {}).get("symbols", [])
    return jsonify(research_tools.compare_tickers(symbols))


# ----------------------------------------------------------------------------
# News + semantic search
# ----------------------------------------------------------------------------


@app.route("/api/news/<symbol>")
def api_get_news(symbol: str):
    limit = int(request.args.get("limit", 10))
    return jsonify({"symbol": symbol.upper(), "news": research_tools.get_recent_news(symbol, limit)})


@app.route("/api/search", methods=["POST"])
def api_vector_search():
    body = request.json or {}
    return jsonify(
        research_tools.vector_search(
            body.get("query", ""), int(body.get("limit", 10)), bool(body.get("search_chunks", True))
        )
    )


# ----------------------------------------------------------------------------
# Research notes + analysis reports
# ----------------------------------------------------------------------------


@app.route("/api/notes", methods=["GET"])
def api_get_notes():
    symbol = request.args.get("symbol")
    return jsonify(research_tools.get_research_notes(_current_user_email(), symbol))


@app.route("/api/notes", methods=["POST"])
def api_save_note():
    body = request.json or {}
    result = research_tools.save_research_note(_current_user_email(), body.get("symbol", ""), body.get("note_text", ""))
    return jsonify(result)


@app.route("/api/reports", methods=["GET"])
def api_get_reports():
    symbol = request.args.get("symbol")
    return jsonify(research_tools.get_analysis_reports(_current_user_email(), symbol))


@app.route("/api/reports", methods=["POST"])
def api_save_report():
    body = request.json or {}
    result = research_tools.save_analysis_report(
        _current_user_email(),
        body.get("symbol", ""),
        body.get("summary", ""),
        body.get("thesis"),
        body.get("sources"),
    )
    return jsonify(result)


# ----------------------------------------------------------------------------
# "Since your last visit"
# ----------------------------------------------------------------------------


@app.route("/api/notable-changes")
def api_notable_changes():
    return jsonify(research_tools.get_notable_changes(_current_user_email()))


@app.route("/api/mark-visit", methods=["POST"])
def api_mark_visit():
    return jsonify(research_tools.mark_visit(_current_user_email()))


# ----------------------------------------------------------------------------
# Chat passthrough to the deployed Agent Bricks agent (optional)
# ----------------------------------------------------------------------------


@app.route("/api/ask", methods=["POST"])
def api_ask_agent():
    """Forward a chat message to the deployed Agent Bricks agent's Model
    Serving endpoint, so users can talk to the same agent that holds the
    read/write tools directly from this app."""
    if not AGENT_ENDPOINT_NAME:
        return jsonify({"error": "AGENT_ENDPOINT_NAME is not configured for this app."}), 501

    message = (request.json or {}).get("message", "")
    if not message.strip():
        return jsonify({"error": "message is required"}), 400

    try:
        response = _w.serving_endpoints.query(
            name=AGENT_ENDPOINT_NAME,
            messages=[{"role": "user", "content": message}],
        )
        choice = response.choices[0] if getattr(response, "choices", None) else None
        reply = choice.message.content if choice else str(response)
        return jsonify({"reply": reply})
    except Exception as exc:
        logger.exception("Agent query failed")
        return jsonify({"error": f"Agent query failed: {exc}"}), 502


if __name__ == "__main__":
    host = os.getenv("FLASK_RUN_HOST", "0.0.0.0")
    port = int(os.getenv("FLASK_RUN_PORT", 8000))
    app.run(debug=True, host=host, port=port)
