import logging
from typing import List

from src.quantls.config import Config
from src.quantls.pipeline import fundamentals, price, scores, sentiment
from src.quantls.pipeline.store import FeatureStore

log = logging.getLogger(__name__)


class PipelineRunner:

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.store = FeatureStore(cfg.db_path)

    def run(self, tickers: List[str], start: str, end: str) -> FeatureStore:
        """
        Run all pipeline stages for the given tickers and date range.
        Returns the populated FeatureStore for use by the backtest engine.
        """
        if not self.cfg.polygon_api_key:
            log.warning(
                "polygon_api_key not set in Config — "
                "running price pipeline only. "
                "Fundamental and sentiment features will fall back to price proxies."
            )
            price.run(tickers, start, end, self.store)
            return self.store

        log.info("=== Pipeline stage 1/4: prices ===")
        price.run(tickers, start, end, self.store)

        log.info("=== Pipeline stage 2/4: fundamentals ===")
        fundamentals.run(
            tickers, start, end,
            api_key=self.cfg.polygon_api_key,
            store=self.store,
            requests_per_minute=self.cfg.polygon_requests_per_minute,
        )

        log.info("=== Pipeline stage 3/4: scores ===")
        scores.run(tickers, start, end, self.store)

        log.info("=== Pipeline stage 4/4: sentiment ===")
        sentiment.run(
            tickers, start, end,
            api_key=self.cfg.polygon_api_key,
            store=self.store,
            requests_per_minute=self.cfg.polygon_requests_per_minute,
        )

        log.info("=== Pipeline complete ===")
        return self.store
