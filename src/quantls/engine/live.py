import logging
import os
from datetime import datetime

import pandas as pd

from src.quantls.config import Config
from src.quantls.data import get_sp500_tickers
from src.quantls.engine.backtest import _load_prices
from src.quantls.pipeline import PipelineRunner
from src.quantls.portfolio import optimize_portfolio
from src.quantls.signals import Predictor, compute_factors

log = logging.getLogger(__name__)


class LiveEngine:
    """
    Production signal generator.

    retrain()          — fits and saves the Lasso model to disk
    generate_signals() — loads the saved model and returns target weights
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.predictor = Predictor(cfg)
        self.model_path = os.path.join(
            os.path.dirname(cfg.db_path), "predictor.joblib"
        )

    def _prepare(self) -> tuple:
        """Fetch universe, run pipeline, load prices from store."""
        tickers = get_sp500_tickers()[: self.cfg.universe_size]

        end = datetime.today().strftime("%Y-%m-%d")
        start = (
            pd.Timestamp(end) - pd.tseries.offsets.BDay(self.cfg.lookback_days + 20)
        ).strftime("%Y-%m-%d")

        runner = PipelineRunner(self.cfg)
        store = runner.run(tickers, start, end)

        close, volume = _load_prices(store, tickers, start, end)
        return close, volume, store

    def retrain(self) -> None:
        """
        Fit the model on the most recent lookback window and save it to disk.
        Run this EOD on your rebalance day (e.g. Friday after close).
        """
        log.info("LiveEngine: starting model retraining …")
        close, volume, store = self._prepare()

        self.predictor.train(close, volume, store=store)
        self.predictor.save(self.model_path)
        log.info(f"Model saved → {self.model_path}")

    def generate_signals(self) -> pd.Series:
        """
        Load the saved model and return target portfolio weights.
        Run this at market open on your rebalance day (e.g. Friday morning).
        """
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"No saved model found at '{self.model_path}'. "
                "Run retrain() first."
            )

        log.info("LiveEngine: generating signals …")
        close, volume, store = self._prepare()

        self.predictor.load(self.model_path)
        ml_scores = self.predictor.infer(close, volume, store=store)

        # Use today's feature snapshot for the combined factor
        tickers = list(close.columns)
        features = store.get(close.index[-1], tickers)
        factors = compute_factors(close, volume, self.cfg, features=features)

        ml_zscored = (ml_scores - ml_scores.mean()) / (ml_scores.std() + 1e-9)
        combined = factors["combined"].add(
            ml_zscored * self.cfg.ml_weight, fill_value=0)

        universe = (
            combined.nlargest(self.cfg.long_n).index
            .union(combined.nsmallest(self.cfg.short_n).index)
        )
        weights = optimize_portfolio(combined[universe], self.cfg)

        log.info(
            f"Signals ready — {(weights > 0).sum()} longs, "
            f"{(weights < 0).sum()} shorts."
        )
        return weights
