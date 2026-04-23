import logging
import time
from typing import List

import pandas as pd
import requests
from transformers import pipeline as hf_pipeline
from src.quantls.pipeline.store import FeatureStore

log = logging.getLogger(__name__)

POLYGON_BASE = "https://api.polygon.io"
_finbert = None   # lazy-loaded on first use


def _load_finbert():
    global _finbert
    if _finbert is None:
        log.info("Loading FinBERT model (first run may take a moment) …")
        _finbert = hf_pipeline(
            "sentiment-analysis",
            model="ProsusAI/finbert",
            tokenizer="ProsusAI/finbert",
            truncation=True,
            max_length=512,
        )
    return _finbert


def _score_texts(texts: List[str], batch_size: int = 32) -> List[float]:
    """
    Score a list of texts with FinBERT.
    Returns a list of floats in [-1, +1]:  P(positive) - P(negative).
    """
    pipe = _load_finbert()
    scores = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        results = pipe(batch)
        for r in results:
            label = r["label"].lower()
            score = r["score"]
            if label == "positive":
                scores.append(score)
            elif label == "negative":
                scores.append(-score)
            else:
                scores.append(0.0)
    return scores


def _fetch_news(ticker: str, date: str, api_key: str, limit: int = 50) -> List[str]:
    """
    Fetch news headlines + descriptions published on `date` for `ticker`.
    Returns a list of text strings (title + description).
    """
    url = f"{POLYGON_BASE}/v2/reference/news"
    params = {
        "ticker":                ticker,
        "published_utc.gte":     f"{date}T00:00:00Z",
        "published_utc.lte":     f"{date}T23:59:59Z",
        "limit":                 limit,
        "order":                 "asc",
        "sort":                  "published_utc",
        "apiKey":                api_key,
    }
    try:
        resp = requests.get(url, params=params, timeout=15)
        if resp.status_code == 200:
            articles = resp.json().get("results", [])
            texts = []
            for a in articles:
                title = a.get("title", "")
                desc = a.get("description", "") or ""
                texts.append(f"{title}. {desc[:200]}".strip())
            return texts
        log.warning(f"Polygon news {resp.status_code} for {ticker} on {date}")
    except requests.RequestException as e:
        log.warning(f"Polygon news request failed for {ticker}: {e}")
    return []


def _compute_daily_sentiment(
    tickers: List[str],
    dates: List[str],
    api_key: str,
    requests_per_minute: int,
) -> pd.DataFrame:
    """
    For each (ticker, date), fetch articles and compute a FinBERT sentiment score.
    Returns a DataFrame with columns [date, ticker, bull_minus_bear].
    """
    delay = 60.0 / requests_per_minute
    rows = []
    total = len(tickers) * len(dates)
    done = 0

    for ticker in tickers:
        for date in dates:
            done += 1
            texts = _fetch_news(ticker, date, api_key)
            if texts:
                sent_scores = _score_texts(texts)
                daily_score = sum(sent_scores) / len(sent_scores)
            else:
                daily_score = 0.0
            rows.append({"date": date, "ticker": ticker, "bull_minus_bear": daily_score})
            if done % 20 == 0:
                log.info(f"  Sentiment: {done}/{total} (ticker/date pairs) processed …")
            time.sleep(delay)

    return pd.DataFrame(rows)


def _add_rolling_sma(df: pd.DataFrame, window: int = 3) -> pd.DataFrame:
    """Add a `bull_minus_bear_3d` column: 3-day rolling mean per ticker."""
    df = df.sort_values(["ticker", "date"])
    df["bull_minus_bear_3d"] = (
        df.groupby("ticker")["bull_minus_bear"]
        .transform(lambda s: s.rolling(window, min_periods=1).mean())
    )
    return df


def run(
    tickers: List[str],
    start: str,
    end: str,
    api_key: str,
    store: FeatureStore,
    requests_per_minute: int = 5,
) -> None:
    """
    Compute FinBERT sentiment scores for all tickers across the date range
    and persist to the sentiment table.
    """
    # Only process dates where at least one ticker is missing sentiment data
    dates_all = pd.bdate_range(start=start, end=end).strftime("%Y-%m-%d").tolist()
    dates_todo = [
        d for d in dates_all
        if any(not store.has_data("sentiment", t, d, d) for t in tickers)
    ]

    if not dates_todo:
        log.info("Sentiment data already up to date — skipping.")
        return

    log.info(f"Running FinBERT sentiment for {len(tickers)} tickers × {len(dates_todo)} days …")
    raw = _compute_daily_sentiment(tickers, dates_todo, api_key, requests_per_minute)
    df = _add_rolling_sma(raw)
    store.save("sentiment", df)
    log.info("Sentiment pipeline complete.")
