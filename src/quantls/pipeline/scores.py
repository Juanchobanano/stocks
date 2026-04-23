import logging
from typing import List

import numpy as np
import pandas as pd

from src.quantls.pipeline.store import FeatureStore

log = logging.getLogger(__name__)


def _zscore(s: pd.Series) -> pd.Series:
    return (s - s.mean()) / (s.std() + 1e-9)


def _cross_sectional_composite(*series: pd.Series) -> pd.Series:
    """Z-score each series then average, ignoring NaN columns per row."""
    zscored = [_zscore(s) for s in series if s.notna().sum() > 1]
    if not zscored:
        return pd.Series(np.nan, index=series[0].index)
    return pd.concat(zscored, axis=1).mean(axis=1)


def _compute_scores_for_date(
    fund: pd.DataFrame,
    price: pd.DataFrame,
    date: pd.Timestamp,
    tickers: List[str],
) -> pd.DataFrame:
    """Compute all score columns for a single date cross-section."""
    eps = 1e-9

    f = fund.reindex(tickers)
    p = price.reindex(tickers)

    close = p.get("close", pd.Series(index=tickers, dtype=float))
    close = close.replace(0, np.nan)

    # ── Filing-derived ratios (no price needed) ───────────────────────────────
    roe = f["net_income"] / (f["equity"].abs() + eps)
    roa = f["net_income"] / (f["assets"].abs() + eps)

    # ── Price-dependent ratios ────────────────────────────────────────────────
    pe_ratio = close / (f["eps"].replace(0, np.nan))
    pb_ratio = close / (f["equity_per_share"].replace(0, np.nan))
    earning_yield = f["eps"] / close

    # Enterprise value = market_cap + debt − cash
    market_cap = close * f["shares"].fillna(0)
    ev = market_cap + f["long_term_debt"].fillna(0) - f["cash"].fillna(0)
    ebit_to_ev = f["ebit"] / ev.replace(0, np.nan)

    # ── Composite scores ──────────────────────────────────────────────────────
    growth_score = _cross_sectional_composite(
        f["revenue_growth"],
        f["equity_per_share_growth"],
        f["sustainable_growth_rate"],
    )

    value_score = _cross_sectional_composite(
        earning_yield,
        pb_ratio.mul(-1),   # lower P/B → more value
        pe_ratio.mul(-1),   # lower P/E → more value
    )

    # Style = equal blend of value and growth (weight 2.0 applied in factors.py)
    style_score = (
        _zscore(value_score.fillna(0)) * 0.5
        + _zscore(growth_score.fillna(0)) * 0.5
    )

    result = pd.DataFrame({
        "date":          date.strftime("%Y-%m-%d"),
        "ticker":        tickers,
        "roe":           roe.values,
        "roa":           roa.values,
        "pe_ratio":      pe_ratio.values,
        "pb_ratio":      pb_ratio.values,
        "earning_yield": earning_yield.values,
        "ebit_to_ev":    ebit_to_ev.values,
        "growth_score":  growth_score.values,
        "value_score":   value_score.values,
        "style_score":   style_score.values,
    })
    return result


def run(
    tickers: List[str],
    start: str,
    end: str,
    store: FeatureStore,
) -> None:
    """
    Compute daily scores for all tickers across the full date range and
    persist to the scores table.
    """
    fund_range = store.get_range(start, end, tickers, "fundamentals")
    price_range = store.get_range(start, end, tickers, "prices")

    if fund_range.empty or price_range.empty:
        log.warning("Scores pipeline: missing fundamentals or prices — skipping.")
        return

    dates = fund_range.index.get_level_values("date").unique().sort_values()
    all_rows = []

    for date in dates:
        try:
            fund_day = fund_range.xs(date, level="date")
            price_day = price_range.xs(date, level="date") if date in price_range.index.get_level_values("date") else pd.DataFrame()
            if price_day.empty:
                continue
            day_scores = _compute_scores_for_date(fund_day, price_day, date, tickers)
            all_rows.append(day_scores)
        except Exception as e:
            log.debug(f"Scores failed for {date}: {e}")

    if all_rows:
        store.save("scores", pd.concat(all_rows, ignore_index=True))

    log.info(f"Scores pipeline complete: {len(all_rows)} dates processed.")
