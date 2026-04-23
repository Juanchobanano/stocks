import logging
from typing import List
from src.broker.ib import IBBroker
from src.triangles.pipeline.config import TrianglePipelineConfig
from src.triangles.pipeline.signal import TradeSignal
from src.triangles.pipeline.store import TriangleStore

log = logging.getLogger(__name__)


def run(
    signals: List[TradeSignal],
    broker: IBBroker,
    cfg: TrianglePipelineConfig,
    store: TriangleStore,
) -> None:
    if not signals:
        log.info("No signals to trade.")
        return

    prefix = "[DRY RUN] " if cfg.dry_run else ""

    for sig in signals:
        log.info(
            f"{prefix}{sig.direction} {sig.ticker} | {sig.pattern} | "
            f"entry={sig.entry_price:.2f}  tp={sig.take_profit:.2f}  "
            f"sl={sig.stop_loss:.2f}  qty={sig.quantity}  height={sig.height:.2f}"
        )

        if not cfg.dry_run:
            action = "BUY" if sig.direction == "LONG" else "SELL"
            broker.place_bracket_order(
                ticker=sig.ticker,
                action=action,
                entry_price=sig.entry_price,
                take_profit=sig.take_profit,
                stop_loss=sig.stop_loss,
                quantity=sig.quantity,
            )

        store.save(
            ticker=sig.ticker,
            pattern=sig.pattern,
            direction=sig.direction,
            entry_price=sig.entry_price,
            take_profit=sig.take_profit,
            stop_loss=sig.stop_loss,
            height=sig.height,
            quantity=sig.quantity,
            status="placed",
        )

    n = len(signals)
    log.info(
        f"Trade stage complete: {n} bracket order(s) "
        f"{'simulated' if cfg.dry_run else 'submitted'}."
    )
