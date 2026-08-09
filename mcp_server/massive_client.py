"""
Client for the Massive Stocks API.

The API key is stored in a Databricks secret scope (see setup_secrets.py) and
resolved at runtime via the Databricks SDK - never stored in code or env files.

Endpoints used (Polygon-compatible shape - adjust field mappings below if
Massive's real response shape differs):
  - GET /v2/aggs/ticker/{ticker}/prev      -> latest quote
  - GET /v3/reference/tickers/{ticker}     -> company fundamentals
  - GET /v2/reference/news                 -> recent news for a ticker
"""

import base64
import os
from datetime import datetime, timezone
from typing import Any

import requests
from databricks.sdk import WorkspaceClient

_w = WorkspaceClient()

_SCOPE = os.environ.get("MASSIVE_SECRET_SCOPE", "massive")
_KEY = os.environ.get("MASSIVE_SECRET_KEY", "api-key")
_BASE_URL = os.environ.get("MASSIVE_API_BASE_URL", "https://api.massive.com")

_api_key: str | None = None


def _get_api_key() -> str:
    global _api_key
    if _api_key is None:
        secret = _w.secrets.get_secret(scope=_SCOPE, key=_KEY)
        _api_key = base64.b64decode(secret.value).decode("utf-8")
    return _api_key


class MassiveClient:
    """Thin wrapper around the Massive Stocks API."""

    def __init__(self, base_url: str | None = None, timeout: int = 30):
        self.base_url = (base_url or _BASE_URL).rstrip("/")
        self.timeout = timeout
        self._session = requests.Session()
        self._session.headers.update({"Authorization": f"Bearer {_get_api_key()}"})

    def _get(self, path: str, params: dict[str, Any] | None = None) -> dict:
        resp = self._session.get(f"{self.base_url}{path}", params=params, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def get_quote(self, symbol: str) -> dict:
        """Latest previous-day OHLCV bar for a ticker (one API call)."""
        symbol = symbol.strip().upper()
        data = self._get(f"/v2/aggs/ticker/{symbol}/prev")
        results = data.get("results") or []
        if not results:
            raise RuntimeError(f"No quote data available for {symbol}")
        bar = results[0]
        open_price = float(bar.get("o", 0) or 0)
        close_price = float(bar.get("c", 0) or 0)
        change = close_price - open_price
        change_pct = (change / open_price * 100) if open_price else None
        ts = bar.get("t")
        as_of = datetime.fromtimestamp(ts / 1000, tz=timezone.utc).isoformat() if ts else None
        return {
            "symbol": symbol,
            "price": close_price,
            "open": open_price,
            "high": float(bar.get("h", 0) or 0),
            "low": float(bar.get("l", 0) or 0),
            "volume": int(bar.get("v", 0) or 0),
            "change": round(change, 4),
            "change_percent": round(change_pct, 4) if change_pct is not None else None,
            "as_of": as_of,
        }

    def get_company_details(self, symbol: str) -> dict:
        """Fundamentals/reference data for a ticker (one API call)."""
        symbol = symbol.strip().upper()
        data = self._get(f"/v3/reference/tickers/{symbol}")
        result = data.get("results") or {}
        if not result:
            raise RuntimeError(f"No company details available for {symbol}")
        return {
            "symbol": symbol,
            "name": result.get("name"),
            "sector": result.get("sic_description"),
            "industry": result.get("type"),
            "description": result.get("description"),
            "market_cap": result.get("market_cap"),
            "employees": result.get("total_employees"),
            "homepage_url": result.get("homepage_url"),
            "exchange": result.get("primary_exchange"),
            "currency": result.get("currency_name"),
        }

    def get_news(self, ticker: str, limit: int = 20) -> list[dict]:
        """Recent news articles for a ticker (one API call)."""
        ticker = ticker.strip().upper()
        data = self._get(
            "/v2/reference/news",
            params={"ticker": ticker, "limit": limit, "order": "desc", "sort": "published_utc"},
        )
        return data.get("results", [])
