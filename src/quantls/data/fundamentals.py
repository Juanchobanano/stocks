import logging
import os
import time
from typing import List, Optional

import pandas as pd
import requests

log = logging.getLogger(__name__)

POLYGON_BASE = "https://api.polygon.io"


def _get_ticker_financials(ticker: str, api_key: str, limit: int = 40) -> list:
    """Fetch quarterly financials from Polygon for a single ticker."""
    url = f"{POLYGON_BASE}/vX/reference/financials"
    params = {
        "ticker":    ticker,
        "timeframe": "quarterly",
        "limit":     limit,
        "apiKey":    api_key,
    }
    try:
        resp = requests.get(url, params=params, timeout=15)
        if resp.status_code == 200:
            return resp.json().get("results", [])
        log.warning(f"Polygon {resp.status_code} for {ticker}: {resp.text[:120]}")
    except requests.RequestException as e:
        log.warning(f"Polygon request failed for {ticker}: {e}")
    return []


def _val(section: dict, field: str) -> Optional[float]:
    """Safely extract a numeric value from a Polygon financials section."""
    return section.get(field, {}).get("value")


def _parse_results(results: list, ticker: str) -> pd.DataFrame:
    """
    Convert raw Polygon financials results into a flat DataFrame.
    Rows are sorted by period_of_report_date so revenue_growth can be computed
    sequentially.
    """
    rows = []
    prev_revenue = None

    for r in sorted(results, key=lambda x: x.get("period_of_report_date", "")):
        try:
            inc = r.get("financials", {}).get("income_statement", {})
            bal = r.get("financials", {}).get("balance_sheet", {})

            filing_date = r.get("filing_date") or r.get("period_of_report_date")
            if not filing_date:
                continue

            revenues = _val(inc, "revenues")
            operating_income = _val(inc, "operating_income_loss")
            net_income = _val(inc, "net_income_loss")
            equity = _val(bal, "equity") or _val(bal, "equity_attributable_to_parent")
            assets = _val(bal, "assets")
            long_term_debt = _val(bal, "long_term_debt") or 0.0
            cash = _val(bal, "cash_and_cash_equivalents_and_short_term_investments") or 0.0

            revenue_growth = None
            if revenues and prev_revenue and prev_revenue != 0:
                revenue_growth = (revenues - prev_revenue) / abs(prev_revenue)
            prev_revenue = revenues

            rows.append({
                "ticker":          ticker,
                "filing_date":     pd.Timestamp(filing_date),
                "operating_income": operating_income,
                "net_income":       net_income,
                "revenues":         revenues,
                "equity":           equity,
                "assets":           assets,
                "long_term_debt":   long_term_debt,
                "cash":             cash,
                "revenue_growth":   revenue_growth,
            })
        except Exception as e:
            log.debug(f"Skipping filing for {ticker}: {e}")

    return pd.DataFrame(rows)


def fetch_fundamentals(
    tickers: List[str],
    api_key: str,
    cache_path: str = ".cache/fundamentals.csv",
    requests_per_minute: int = 5,
) -> pd.DataFrame:
    """
    Fetch historical quarterly fundamentals for all tickers from Polygon.io.
    Results are cached to `cache_path`; delete the file to force a re-fetch.

    Returns a DataFrame indexed by (filing_date, ticker).
    """
    if os.path.exists(cache_path):
        log.info(f"Loading fundamentals from cache: {cache_path}")
        df = pd.read_csv(cache_path, parse_dates=["filing_date"])
        return df.set_index(["filing_date", "ticker"])

    os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
    delay = 60.0 / requests_per_minute
    frames = []

    for i, ticker in enumerate(tickers):
        log.info(f"Fetching fundamentals {i + 1}/{len(tickers)}: {ticker} …")
        results = _get_ticker_financials(ticker, api_key)
        if results:
            df = _parse_results(results, ticker)
            if not df.empty:
                frames.append(df)
        time.sleep(delay)

    if not frames:
        log.warning("No fundamental data retrieved from Polygon.")
        return pd.DataFrame()

    combined = (
        pd.concat(frames, ignore_index=True)
        .dropna(subset=["filing_date"])
        .sort_values("filing_date")
    )
    combined.to_csv(cache_path, index=False)
    log.info(f"Fundamentals cached → {cache_path}")
    return combined.set_index(["filing_date", "ticker"])


def get_fundamentals_asof(
    fundamentals: pd.DataFrame,
    date: pd.Timestamp,
    prices: pd.Series,
) -> pd.DataFrame:
    """
    Return the most recently filed fundamental metrics for each ticker as of
    `date`, preventing look-ahead bias.

    `prices` (indexed by ticker) is used to compute ebit_to_price — a proxy
    for EBIT/EV that avoids needing historical shares-outstanding data.
    """
    if fundamentals.empty:
        return pd.DataFrame()

    filed = fundamentals[
        fundamentals.index.get_level_values("filing_date") <= date
    ]
    if filed.empty:
        return pd.DataFrame()

    # Most recent filing per ticker as of this date
    latest = (
        filed.reset_index()
        .sort_values("filing_date")
        .groupby("ticker")
        .last()
        .drop(columns=["filing_date"], errors="ignore")
    )

    eps = 1e-9
    latest["roe"] = latest["net_income"] / (latest["equity"].abs() + eps)
    latest["roa"] = latest["net_income"] / (latest["assets"].abs() + eps)

    shared = latest.index.intersection(prices.index)
    latest.loc[shared, "ebit_to_price"] = (
        latest.loc[shared, "operating_income"]
        / prices[shared].replace(0, float("nan"))
    )

    return latest[["ebit_to_price", "roe", "roa", "revenue_growth"]].dropna(how="all")
