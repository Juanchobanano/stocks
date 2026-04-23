import logging
import time
from typing import List, Optional

import pandas as pd
import requests

from src.quantls.pipeline.store import FeatureStore

log = logging.getLogger(__name__)

POLYGON_BASE = "https://api.polygon.io"


def _fetch_filings(ticker: str, api_key: str, limit: int = 40) -> list:
    url = f"{POLYGON_BASE}/vX/reference/financials"
    params = {"ticker": ticker, "timeframe": "quarterly", "limit": limit, "apiKey": api_key}
    try:
        resp = requests.get(url, params=params, timeout=15)
        if resp.status_code == 200:
            return resp.json().get("results", [])
        log.warning(f"Polygon {resp.status_code} for {ticker}")
    except requests.RequestException as e:
        log.warning(f"Polygon request failed for {ticker}: {e}")
    return []


def _val(section: dict, *keys: str) -> Optional[float]:
    """Try multiple field names, return the first non-null value found."""
    for key in keys:
        v = section.get(key, {}).get("value")
        if v is not None:
            return float(v)
    return None


def _parse_filings(results: list, ticker: str) -> pd.DataFrame:
    """
    Convert raw Polygon filing results into a DataFrame indexed by filing_date.
    Computes derived metrics: sustainable_growth_rate and equity_per_share_growth.
    """
    rows = []
    prev_equity_ps = None
    prev_revenue = None

    for r in sorted(results, key=lambda x: x.get("period_of_report_date", "")):
        try:
            inc = r.get("financials", {}).get("income_statement", {})
            bal = r.get("financials", {}).get("balance_sheet", {})
            cf = r.get("financials", {}).get("cash_flow_statement", {})

            filing_date = r.get("filing_date") or r.get("period_of_report_date")
            if not filing_date:
                continue

            net_income = _val(inc, "net_income_loss")
            revenue = _val(inc, "revenues")
            ebit = _val(inc, "operating_income_loss")
            eps = _val(inc, "basic_earnings_per_share")
            equity = _val(bal, "equity", "equity_attributable_to_parent")
            assets = _val(bal, "assets")
            debt = _val(bal, "long_term_debt") or 0.0
            cash = _val(bal, "cash_and_cash_equivalents_and_short_term_investments") or 0.0
            dividends = abs(_val(cf, "payment_of_dividends") or 0.0)

            # Shares outstanding estimated from net_income / EPS
            shares = None
            if net_income and eps and eps != 0:
                shares = net_income / eps

            # equity per share
            equity_ps = None
            if equity and shares and shares > 0:
                equity_ps = equity / shares

            # QoQ equity-per-share growth
            eps_growth = None
            if equity_ps and prev_equity_ps and prev_equity_ps != 0:
                eps_growth = (equity_ps - prev_equity_ps) / abs(prev_equity_ps)
            prev_equity_ps = equity_ps

            # QoQ revenue growth
            rev_growth = None
            if revenue and prev_revenue and prev_revenue != 0:
                rev_growth = (revenue - prev_revenue) / abs(prev_revenue)
            prev_revenue = revenue

            # Sustainable growth rate = ROE × retention ratio
            sgr = None
            if net_income and equity and equity != 0 and net_income != 0:
                roe = net_income / equity
                payout = min(dividends / abs(net_income), 1.0) if net_income else 0.4
                retention = 1.0 - payout
                sgr = roe * retention

            rows.append({
                "filing_date":            pd.Timestamp(filing_date),
                "ticker":                 ticker,
                "ebit":                   ebit,
                "net_income":             net_income,
                "equity":                 equity,
                "assets":                 assets,
                "eps":                    eps,
                "equity_per_share":       equity_ps,
                "shares":                 shares,
                "long_term_debt":         debt,
                "cash":                   cash,
                "revenue":                revenue,
                "revenue_growth":         rev_growth,
                "equity_per_share_growth": eps_growth,
                "sustainable_growth_rate": sgr,
            })
        except Exception as e:
            log.debug(f"Skipping filing for {ticker}: {e}")

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).set_index("filing_date").drop(columns=["ticker"])
    return df.sort_index()


def _to_daily(filings_df: pd.DataFrame, ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Forward-fill quarterly filing data to a daily business-day calendar.
    Data becomes available only on its filing_date (point-in-time correct).
    """
    bdays = pd.bdate_range(start=start, end=end)
    daily = filings_df.reindex(bdays, method="ffill")
    daily = daily.dropna(how="all")
    daily.index.name = "date"
    daily.insert(0, "ticker", ticker)
    return daily.reset_index()


def run(
    tickers: List[str],
    start: str,
    end: str,
    api_key: str,
    store: FeatureStore,
    requests_per_minute: int = 5,
) -> None:
    """
    Fetch quarterly filings from Polygon for all tickers, forward-fill to
    daily, and persist to the fundamentals table.
    """
    delay = 60.0 / requests_per_minute

    for i, ticker in enumerate(tickers):
        if store.has_data("fundamentals", ticker, start, end):
            log.info(f"  [{i+1}/{len(tickers)}] {ticker} — fundamentals cached, skipping.")
            continue

        log.info(f"  [{i+1}/{len(tickers)}] {ticker} — fetching from Polygon …")
        results = _fetch_filings(ticker, api_key)

        if not results:
            log.warning(f"  No filings returned for {ticker}.")
            time.sleep(delay)
            continue

        filings_df = _parse_filings(results, ticker)
        if filings_df.empty:
            time.sleep(delay)
            continue

        daily_df = _to_daily(filings_df, ticker, start, end)
        store.save("fundamentals", daily_df)
        time.sleep(delay)

    log.info("Fundamentals pipeline complete.")
