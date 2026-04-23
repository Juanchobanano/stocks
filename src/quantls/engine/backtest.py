import logging
import pandas as pd
from src.quantls.config import Config
from src.quantls.data import get_sp500_tickers
from src.quantls.pipeline import PipelineRunner
from src.quantls.pipeline.store import FeatureStore
from src.quantls.portfolio import optimize_portfolio
from src.quantls.signals import Predictor, compute_factors

log = logging.getLogger(__name__)


def _load_prices(store: FeatureStore, tickers: list, start: str, end: str):
    """
    Return (close, volume) DataFrames from the feature store — avoids a
    second yfinance download after the pipeline has already cached prices.
    """
    df = store.get_range(start, end, tickers, "prices")
    if df.empty:
        raise RuntimeError("Price data missing from store — run PipelineRunner first.")
    close = df["close"].unstack("ticker")
    volume = df["volume"].unstack("ticker")
    return close, volume


class Backtest:
    """
    Event-driven backtest loop.

    On startup, the pipeline runner fetches and caches all features
    (prices, fundamentals, scores, sentiment) for the full date range.
    Each rebalance reads a point-in-time feature slice from the store.
    """

    def __init__(self, cfg: Config):
        self.cfg       = cfg
        self.predictor = Predictor(cfg)

    def _get_weights(
        self,
        close: pd.DataFrame,
        volume: pd.DataFrame,
        as_of: pd.Timestamp,
        store,
    ) -> pd.Series:
        """Compute target portfolio weights as of a given rebalance date."""
        loc = close.index.get_loc(as_of)
        start = max(0, loc - self.cfg.lookback_days + 1)
        c_win = close.iloc[start:loc + 1]
        v_win = volume.iloc[start:loc + 1]

        if len(c_win) < 21:
            return pd.Series(dtype=float)

        # Point-in-time feature snapshot for this rebalance date
        features = store.get(as_of, list(c_win.columns))
        factors = compute_factors(c_win, v_win, self.cfg, features=features)
        ml_scores = self.predictor.fit_predict(c_win, v_win, store=store, as_of=as_of)

        # Z-score ML scores to match combined_factor scale, then blend per-stock
        ml_zscored = (ml_scores - ml_scores.mean()) / (ml_scores.std() + 1e-9)
        combined = factors["combined"].add(ml_zscored * self.cfg.ml_weight, fill_value=0)

        universe = (
            combined.nlargest(self.cfg.long_n).index
            .union(combined.nsmallest(self.cfg.short_n).index)
        )
        return optimize_portfolio(combined[universe], self.cfg)

    def run(self) -> pd.DataFrame:
        """Run the full backtest; return a DataFrame of daily portfolio values."""
        log.info("Fetching universe …")
        tickers = get_sp500_tickers()[: self.cfg.universe_size]

        data_start = (
            pd.Timestamp(self.cfg.start_date)
            - pd.tseries.offsets.BDay(self.cfg.lookback_days + 20)
        ).strftime("%Y-%m-%d")

        # ── Run all pipeline stages (skips stages already cached) ─────────────
        runner = PipelineRunner(self.cfg)
        store = runner.run(tickers, data_start, self.cfg.end_date)

        close, volume = _load_prices(store, tickers, data_start, self.cfg.end_date)

        bt_close = close.loc[self.cfg.start_date :]
        rebalance_dates = set(
            pd.date_range(bt_close.index[0], bt_close.index[-1], freq=self.cfg.rebalance_freq)
            .intersection(bt_close.index)
        )

        capital = self.cfg.initial_capital
        weights = pd.Series(dtype=float)
        records = []
        prev_date = None

        for date in bt_close.index:
            # Daily mark-to-market
            if prev_date is not None and not weights.empty:
                shared = weights.index.intersection(close.columns)
                daily_ret = (close.loc[date, shared] / close.loc[prev_date, shared] - 1).fillna(0)
                capital *= 1.0 + (weights[shared] * daily_ret).sum()

            # Rebalance if scheduled
            if date in rebalance_dates:
                log.info(f"Rebalancing on {date.date()} …")
                weights = self._get_weights(close, volume, date, store)

            records.append({"date": date, "portfolio_value": capital})
            prev_date = date

        return pd.DataFrame(records).set_index("date")
