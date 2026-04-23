import logging
from typing import List
import pandas as pd
import yfinance as yf
from src.quantls.pipeline.store import FeatureStore

log = logging.getLogger(__name__)


def run(
    tickers: List[str],
    start: str,
    end: str,
    store: FeatureStore,
) -> None:
    """Download OHLCV for all tickers and persist to the prices table."""
    to_fetch = [
        t for t in tickers
        if not store.has_data("prices", t, start, end)
    ]
    if not to_fetch:
        log.info("Price data already up to date — skipping download.")
        return

    log.info(f"Downloading prices for {len(to_fetch)} tickers …")
    raw = yf.download(
        to_fetch, start=start, end=end,
        auto_adjust=True, progress=False, threads=True,
    )

    close = raw["Close"]
    open_ = raw["Open"]
    volume = raw["Volume"]

    # Drop tickers with < 80% coverage
    min_rows = int(0.8 * len(close))
    close = close.dropna(axis=1, thresh=min_rows)
    valid = close.columns.tolist()
    open_ = open_[valid]
    volume = volume[valid]

    rows = []
    for ticker in valid:
        for date, o, c, v in zip(
            close.index,
            open_[ticker],
            close[ticker],
            volume[ticker],
        ):
            rows.append({
                "date": date,
                "ticker": ticker,
                "open": float(o) if pd.notna(o) else None,
                "close": float(c) if pd.notna(c) else None,
                "volume": float(v) if pd.notna(v) else None,
            })

    store.save("prices", pd.DataFrame(rows))
    log.info(f"Price pipeline complete: {len(valid)} tickers.")
