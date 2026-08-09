"""
AI Stock Market Research Assistant - MCP server.

Exposes research_tools.py's read/write functions over MCP (Model Context
Protocol) so a Databricks Agent Bricks agent can call them like any other
tool. Deploy this as its own Databricks App (see app.yaml), then register
its URL as an external MCP server for the agent (see the top-level README).

Tools:
  Read:  get_quote, get_price_history, get_company_fundamentals,
         compare_tickers, get_recent_news, vector_search, get_watchlist,
         get_research_notes, get_analysis_reports, get_notable_changes
  Write: add_to_watchlist, remove_from_watchlist, save_research_note,
         save_analysis_report, mark_visit

Run locally:
    python research_mcp_server.py
"""

import logging
import os
from contextvars import ContextVar

from fastmcp import FastMCP
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

import research_tools

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("research-mcp-server")

mcp = FastMCP("stock-research-assistant")

_request_context: ContextVar[dict] = ContextVar("request_context", default={})

_FALLBACK_EMAIL = os.environ.get("FALLBACK_USER_EMAIL", "researcher@example.com")


def _current_user_email() -> str:
    """The end user's email, from the X-Forwarded-Email header Databricks Apps
    inject on every request. Falls back to the service principal (local dev)."""
    headers = _request_context.get()
    forwarded_email = headers.get("x-forwarded-email") or headers.get("x-forwarded-user")
    if forwarded_email:
        return forwarded_email
    try:
        from databricks.sdk import WorkspaceClient

        return WorkspaceClient().current_user.me().user_name or _FALLBACK_EMAIL
    except Exception:
        return _FALLBACK_EMAIL


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Captures the end-user identity headers Databricks injects."""

    async def dispatch(self, request: Request, call_next):
        _request_context.set(
            {
                "x-forwarded-user": request.headers.get("x-forwarded-user"),
                "x-forwarded-email": request.headers.get("x-forwarded-email"),
            }
        )
        return await call_next(request)


# ----------------------------------------------------------------------------
# Read tools
# ----------------------------------------------------------------------------


@mcp.tool
def get_quote(symbol: str) -> dict:
    """Get a live quote (price, volume, day change) for a stock ticker.

    Args:
        symbol: Stock ticker symbol, e.g. "AAPL".
    """
    return research_tools.get_quote(symbol)


@mcp.tool
def get_price_history(symbol: str, days: int = 30) -> dict:
    """Get historical daily price snapshots for a ticker to summarize recent
    performance (e.g. "how has this stock done over the last month?").

    Args:
        symbol: Stock ticker symbol, e.g. "AAPL".
        days: How many days back to look (default 30).
    """
    return {"symbol": symbol.upper(), "history": research_tools.get_price_history(symbol, days)}


@mcp.tool
def get_company_fundamentals(symbol: str) -> dict:
    """Get company fundamentals (sector, market cap, description, etc.) for a ticker.

    Args:
        symbol: Stock ticker symbol, e.g. "AAPL".
    """
    return research_tools.get_company_fundamentals(symbol)


@mcp.tool
def compare_tickers(symbols: list[str]) -> dict:
    """Compare multiple tickers side by side on price, fundamentals, and recent news sentiment.

    Args:
        symbols: List of stock ticker symbols to compare, e.g. ["AAPL", "MSFT"].
    """
    return research_tools.compare_tickers(symbols)


@mcp.tool
def get_recent_news(symbol: str, limit: int = 10) -> dict:
    """Get recent synced news headlines and sentiment for a ticker.

    Args:
        symbol: Stock ticker symbol, e.g. "AAPL".
        limit: Max number of articles to return (default 10).
    """
    return {"symbol": symbol.upper(), "news": research_tools.get_recent_news(symbol, limit)}


@mcp.tool
def vector_search(query: str, limit: int = 10, search_chunks: bool = True) -> dict:
    """Semantic search over ticker news using vector embeddings - for natural
    language questions like "companies exposed to rising interest rates in
    the regional banking sector" rather than a plain ticker/keyword lookup.

    Args:
        query: Natural language search query.
        limit: Maximum number of results to return (default 10).
        search_chunks: Whether to also search full-article-body chunk embeddings.
    """
    return research_tools.vector_search(query, limit, search_chunks)


@mcp.tool
def get_watchlist() -> dict:
    """Get the authenticated user's current watchlist."""
    email = _current_user_email()
    return {"email": email, "watchlist": research_tools.get_watchlist(email)}


@mcp.tool
def get_research_notes(symbol: str | None = None, limit: int = 20) -> dict:
    """Get the authenticated user's saved research notes, optionally filtered to one ticker.

    Args:
        symbol: Optional stock ticker symbol to filter by.
        limit: Max number of notes to return (default 20).
    """
    email = _current_user_email()
    return {"email": email, "notes": research_tools.get_research_notes(email, symbol, limit)}


@mcp.tool
def get_analysis_reports(symbol: str | None = None, limit: int = 20) -> dict:
    """Get the authenticated user's saved analysis reports, optionally filtered to one ticker.

    Args:
        symbol: Optional stock ticker symbol to filter by.
        limit: Max number of reports to return (default 20).
    """
    email = _current_user_email()
    return {"email": email, "reports": research_tools.get_analysis_reports(email, symbol, limit)}


@mcp.tool
def get_notable_changes() -> dict:
    """Flag notable price moves and news for the user's watchlist since their
    last visit. Call mark_visit after presenting these to the user."""
    return research_tools.get_notable_changes(_current_user_email())


# ----------------------------------------------------------------------------
# Write tools
# ----------------------------------------------------------------------------


@mcp.tool
def add_to_watchlist(symbol: str) -> dict:
    """Add a ticker to the authenticated user's watchlist (fetches a live quote first).

    Args:
        symbol: Stock ticker symbol to add, e.g. "AAPL".
    """
    try:
        return research_tools.add_to_watchlist(_current_user_email(), symbol)
    except Exception as exc:
        logger.exception("add_to_watchlist failed")
        return {"status": "error", "message": str(exc)}


@mcp.tool
def remove_from_watchlist(symbol: str) -> dict:
    """Remove a ticker from the authenticated user's watchlist.

    Args:
        symbol: Stock ticker symbol to remove, e.g. "AAPL".
    """
    try:
        return research_tools.remove_from_watchlist(_current_user_email(), symbol)
    except Exception as exc:
        logger.exception("remove_from_watchlist failed")
        return {"status": "error", "message": str(exc)}


@mcp.tool
def save_research_note(symbol: str, note_text: str) -> dict:
    """Save a freeform research note tied to a ticker for the authenticated user.

    Args:
        symbol: Stock ticker symbol the note is about, e.g. "AAPL".
        note_text: The note content.
    """
    try:
        return research_tools.save_research_note(_current_user_email(), symbol, note_text)
    except Exception as exc:
        logger.exception("save_research_note failed")
        return {"status": "error", "message": str(exc)}


@mcp.tool
def save_analysis_report(symbol: str, summary: str, thesis: str = "", sources: list[str] | None = None) -> dict:
    """Save a structured analysis report (thesis + summary + sources) tied to a ticker.

    Args:
        symbol: Stock ticker symbol the report is about, e.g. "AAPL".
        summary: The report's summary/body text.
        thesis: Optional one-line investing thesis.
        sources: Optional list of source URLs or article IDs the report drew on.
    """
    try:
        return research_tools.save_analysis_report(
            _current_user_email(), symbol, summary, thesis or None, sources
        )
    except Exception as exc:
        logger.exception("save_analysis_report failed")
        return {"status": "error", "message": str(exc)}


@mcp.tool
def mark_visit() -> dict:
    """Record that the authenticated user has just visited, advancing the
    checkpoint get_notable_changes compares against next time."""
    return research_tools.mark_visit(_current_user_email())


if __name__ == "__main__":
    if hasattr(mcp, "app") and mcp.app is not None:
        mcp.app.add_middleware(RequestContextMiddleware)

    port = int(os.getenv("DATABRICKS_APP_PORT", os.getenv("PORT", 8000)))
    mcp.run(transport="http", host="0.0.0.0", port=port)
