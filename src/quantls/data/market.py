import logging
from typing import List, Tuple

import pandas as pd
import yfinance as yf

log = logging.getLogger(__name__)


def fetch_market_data(
    tickers: List[str], start: str, end: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Bulk download adjusted close prices and volume.
    Drops tickers with < 80% data coverage to filter delisted / stale stocks.
    """
    log.info(f"Downloading data for {len(tickers)} tickers ({start} → {end}) …")
    raw = yf.download(
        tickers, start=start, end=end,
        auto_adjust=True, progress=False, threads=True,
    )
    close  = raw["Close"]
    volume = raw["Volume"]
    min_rows = int(0.8 * len(close))
    close  = close.dropna(axis=1, thresh=min_rows)
    volume = volume[close.columns]
    log.info(f"Clean data: {len(close.columns)} tickers × {len(close)} days")
    return close, volume
