import logging
from src.broker.ib import IBBroker
from src.triangles.pipeline import fetch, detect, signal, trade
from src.triangles.pipeline.config import TrianglePipelineConfig
from src.triangles.pipeline.store import TriangleStore

log = logging.getLogger(__name__)


class TrianglePipelineRunner:

    def __init__(self, cfg: TrianglePipelineConfig | None = None):
        self.cfg = cfg or TrianglePipelineConfig()
        self.store = TriangleStore(self.cfg.db_path)

    def run(self) -> None:
        cfg = self.cfg

        with IBBroker(
            host=cfg.ib_host,
            port=cfg.ib_port,
            client_id=cfg.ib_client_id,
            order_type="LMT",
            dry_run=cfg.dry_run,
        ) as broker:
            log.info("=== Stage 1/4: fetch symbols + OHLCV ===")
            symbols = fetch.get_symbols(broker, cfg)
            bars = fetch.get_ohlcv(broker, symbols, cfg)

            log.info("=== Stage 2/4: detect triangle geometry ===")
            patterns = detect.run(bars, cfg)

            log.info("=== Stage 3/4: evaluate signals ===")
            equity = broker._get_equity()
            signals = signal.run(patterns, bars, equity, cfg, self.store)

            log.info("=== Stage 4/4: place bracket orders ===")
            trade.run(signals, broker, cfg, self.store)

        log.info("=== Pipeline complete ===")
