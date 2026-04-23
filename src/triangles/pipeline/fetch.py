import logging
from typing import Dict, List

import pandas as pd

from src.broker.ib import IBBroker
from src.triangles.pipeline.config import TrianglePipelineConfig

log = logging.getLogger(__name__)

# IB historical data duration string for the configured lookback
_DURATION_MAP = {
    "1 day": "4 M",
    "1 hour": "1 M",
    "30 mins": "1 M",
}
_DEFAULT_DURATION = "4 M"


def get_symbols(broker: IBBroker, cfg: TrianglePipelineConfig) -> List[str]:
    symbols = broker.get_scanner_symbols(cfg.scanner_code, cfg.scanner_n_symbols)
    log.info(f"Scanner returned {len(symbols)} symbols.")
    return symbols


def get_ohlcv(
    broker: IBBroker,
    symbols: List[str],
    cfg: TrianglePipelineConfig,
) -> Dict[str, pd.DataFrame]:
    """
    Download OHLCV bars for every symbol. Returns only symbols that have
    at least 80% bar coverage over the requested lookback window.
    """
    duration = _DURATION_MAP.get(cfg.bar_size, _DEFAULT_DURATION)
    bars: Dict[str, pd.DataFrame] = {}
    min_rows = int(0.8 * cfg.lookback_bars)

    for ticker in symbols:
        try:
            df = broker.get_historical_bars(ticker, duration, cfg.bar_size)
            if df.empty or len(df) < min_rows:
                log.debug(f"{ticker}: insufficient bars ({len(df)}) — skipped.")
                continue
            bars[ticker] = df
        except Exception as exc:
            log.warning(f"{ticker}: failed to fetch bars — {exc}")

    log.info(f"Fetched OHLCV for {len(bars)}/{len(symbols)} symbols.")
    return bars
