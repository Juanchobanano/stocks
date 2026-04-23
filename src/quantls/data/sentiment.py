"""
Sentiment data from StockTwits.

Fetches the most recent message stream for each ticker and computes a
bull_minus_bear score normalized to [-1, +1].

Limitations of the free StockTwits API:
  - Returns only the most recent ~30 messages per symbol.
  - No historical data — suitable for live/forward trading only, not backtesting.
  - Rate limit: ~200 requests/hour. Use `delay_seconds` to stay within it.

In backtesting, pass `sentiment=None` to `compute_factors`; it will fall
back to the short-term price reversal proxy automatically.
"""

import logging
import time
from typing import List, Optional

import pandas as pd
import requests

log = logging.getLogger(__name__)

STOCKTWITS_BASE = "https://api.stocktwits.com/api/2"


# ── StockTwits fetch ──────────────────────────────────────────────────────────

def _fetch_symbol_stream(ticker: str) -> dict:
    """Fetch the latest StockTwits message stream for a symbol."""
    url = f"{STOCKTWITS_BASE}/streams/symbol/{ticker}.json"
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            return resp.json()
        if resp.status_code == 429:
            log.warning(f"StockTwits rate limit hit for {ticker} — waiting 60 s …")
            time.sleep(60)
        else:
            log.warning(f"StockTwits {resp.status_code} for {ticker}")
    except requests.RequestException as e:
        log.warning(f"StockTwits request failed for {ticker}: {e}")
    return {}


def _bull_minus_bear(messages: list) -> float:
    """
    Count Bullish vs Bearish tagged messages; return net score in [-1, +1].
    Messages without a sentiment tag are ignored.
    """
    bull  = sum(1 for m in messages if m.get("entities", {}).get("sentiment", {}).get("basic") == "Bullish")
    bear  = sum(1 for m in messages if m.get("entities", {}).get("sentiment", {}).get("basic") == "Bearish")
    total = bull + bear
    return (bull - bear) / total if total > 0 else 0.0


# ── Public API ────────────────────────────────────────────────────────────────

def fetch_sentiment(
    tickers: List[str],
    delay_seconds: float = 0.5,
) -> pd.Series:
    """
    Fetch current StockTwits bull_minus_bear sentiment for each ticker.

    Returns a Series indexed by ticker with values in [-1, +1]:
      +1 = all messages bullish
      -1 = all messages bearish
       0 = neutral or no tagged messages

    Note: This is a point-in-time snapshot. It cannot be used to reconstruct
    historical sentiment for backtesting — use it for live signal generation only.
    """
    scores = {}
    for i, ticker in enumerate(tickers):
        log.info(f"  Fetching sentiment {i + 1}/{len(tickers)}: {ticker}")
        data     = _fetch_symbol_stream(ticker)
        messages = data.get("messages", [])
        scores[ticker] = _bull_minus_bear(messages)
        time.sleep(delay_seconds)

    return pd.Series(scores, name="sentiment")
