import logging
import math
from dataclasses import dataclass
from typing import Dict, List

import pandas as pd

from src.triangles.pipeline.config import TrianglePipelineConfig
from src.triangles.pipeline.store import TriangleStore

log = logging.getLogger(__name__)


@dataclass
class TradeSignal:
    ticker: str
    pattern: str
    direction: str          # 'LONG' | 'SHORT'
    entry_price: float
    take_profit: float
    stop_loss: float
    height: float
    quantity: float


def run(
    patterns: Dict[str, dict],
    bars: Dict[str, pd.DataFrame],
    equity: float,
    cfg: TrianglePipelineConfig,
    store: TriangleStore,
) -> List[TradeSignal]:
    signals: List[TradeSignal] = []

    for ticker, geo in patterns.items():
        reason = _evaluate(ticker, geo, bars.get(ticker), equity, cfg, store)
        if isinstance(reason, TradeSignal):
            signals.append(reason)
        else:
            log.debug(f"{ticker}: filtered — {reason}")

    log.info(f"Signal stage complete: {len(signals)} actionable signal(s).")
    return signals


def _evaluate(ticker, geo, df, equity, cfg, store):
    if not geo["is_consolidating"]:
        return "not consolidating"

    if geo["apex_distance_bars"] < cfg.min_bars_to_apex:
        return f"apex too close ({geo['apex_distance_bars']:.1f} bars)"

    if df is None or len(df) < cfg.volume_lookback:
        return "insufficient bars for volume check"

    vol_recent = float(df["Volume"].iloc[-5:].mean())
    vol_baseline = float(df["Volume"].iloc[-cfg.volume_lookback:].mean())
    if vol_baseline <= 0:
        return "zero baseline volume"
    if (vol_recent / vol_baseline) >= cfg.volume_ratio_threshold:
        return f"volume not declining ({vol_recent/vol_baseline:.2f} >= {cfg.volume_ratio_threshold})"

    if store.has_open_order(ticker):
        return "open order already exists"

    pattern_type = geo["pattern_type"]
    height = geo["height"]

    if pattern_type in ("Ascending Triangle", "Symmetrical Triangle"):
        direction = "LONG"
        entry = geo["resistance_at_now"]
        take_profit = entry + height / 2
        stop_loss = entry - height * (2 / 3)
    else:  # Descending Triangle
        direction = "SHORT"
        entry = geo["support_at_now"]
        take_profit = entry - height / 2
        stop_loss = entry + height * (2 / 3)

    quantity = math.floor((equity * cfg.position_pct) / entry)
    if quantity < 1:
        return f"position too small (qty={quantity})"

    return TradeSignal(
        ticker=ticker,
        pattern=pattern_type,
        direction=direction,
        entry_price=round(entry, 2),
        take_profit=round(take_profit, 2),
        stop_loss=round(stop_loss, 2),
        height=round(height, 2),
        quantity=quantity,
    )
