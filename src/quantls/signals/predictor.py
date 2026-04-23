import logging
import os
from typing import TYPE_CHECKING, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler

from src.quantls.config import Config

if TYPE_CHECKING:
    from src.quantls.pipeline.store import FeatureStore

log = logging.getLogger(__name__)

# Fundamental feature columns sourced from the scores / fundamentals tables
_FUND_COLS = [
    "earning_yield",
    "pb_ratio",
    "pe_ratio",
    "roa",
    "sustainable_growth_rate",
    "equity_per_share_growth",
    "growth_score",
    "value_score",
]


class Predictor:
    """
    Lasso regression trained on lagged price/volume features to predict
    next-day returns. Expanded to include fundamental features when a
    FeatureStore is available, matching the original Quantopian Predictor.

    Training: all (date × ticker) pairs where the forward return is known.
    Prediction: the most recent row of features for each ticker.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.model = Lasso(alpha=cfg.lasso_alpha, max_iter=5_000)
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy="median")
        self._is_trained = False
        self._last_trained_date: Optional[pd.Timestamp] = None

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Persist the fitted model, scaler, and imputer to disk."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        joblib.dump(
            {"model": self.model, "scaler": self.scaler, "imputer": self.imputer},
            path,
        )
        log.info(f"Predictor saved → {path}")

    def load(self, path: str) -> None:
        """Load a previously saved model, scaler, and imputer from disk."""
        state = joblib.load(path)
        self.model = state["model"]
        self.scaler = state["scaler"]
        self.imputer = state["imputer"]
        self._is_trained = True
        log.info(f"Predictor loaded ← {path}")

    @property
    def is_trained(self) -> bool:
        return self._is_trained

    # ── Feature construction ──────────────────────────────────────────────────

    def _price_features(
        self, close: pd.DataFrame, volume: pd.DataFrame
    ) -> np.ndarray:
        """Build (T, N, 4) price feature array."""
        ret_1d = close.pct_change(1).values
        ret_5d = close.pct_change(5).values
        ret_21d = close.pct_change(21).values
        vol_ratio = (
            volume.rolling(21).mean() / (volume.rolling(63).mean() + 1e-9)
        ).values
        return np.stack([ret_1d, ret_5d, ret_21d, vol_ratio], axis=-1)

    def _fund_features(
        self,
        close: pd.DataFrame,
        store: "FeatureStore",
    ) -> np.ndarray:
        """
        Build (T, N, 8) fundamental feature array.
        Unstacks the store data once to avoid repeated MultiIndex lookups.
        """
        dates = close.index
        tickers = list(close.columns)
        T, N = len(dates), len(_FUND_COLS)
        result = np.full((T, len(tickers), N), np.nan)

        start = dates[0].strftime("%Y-%m-%d")
        end   = dates[-1].strftime("%Y-%m-%d")

        scores_range = store.get_range(start, end, tickers, "scores")
        fund_range   = store.get_range(start, end, tickers, "fundamentals")
        all_data = pd.concat(
            [x for x in [scores_range, fund_range] if not x.empty], axis=1
        )

        if all_data.empty:
            return result

        available = [c for c in _FUND_COLS if c in all_data.columns]
        if not available:
            return result

        # Unstack once: index=date, columns=MultiIndex(feature, ticker)
        unstacked = all_data[available].unstack(level="ticker")

        for i, date in enumerate(dates):
            if date not in unstacked.index:
                continue
            for j, col in enumerate(_FUND_COLS):
                if col not in available:
                    continue
                result[i, :, j] = unstacked.loc[date, col].reindex(tickers).values

        return result

    # ── Main interface ────────────────────────────────────────────────────────

    def train(
        self,
        close: pd.DataFrame,
        volume: pd.DataFrame,
        store: Optional["FeatureStore"] = None,
        as_of: Optional[pd.Timestamp] = None,
    ) -> None:
        """
        Fit the Lasso model on all labeled history in the supplied window.

        When `as_of` is provided, retraining is skipped if fewer than
        `cfg.ml_retrain_days` trading days have passed since the last fit —
        avoiding expensive weekly refits during backtesting.
        """
        if as_of is not None and self._last_trained_date is not None:
            days_since = (as_of - self._last_trained_date).days
            if days_since < self.cfg.ml_retrain_days:
                log.debug(
                    f"Skipping retrain: only {days_since}d since last train "
                    f"(threshold={self.cfg.ml_retrain_days}d)."
                )
                return

        n_hold = self.cfg.holding_days
        n_tickers = len(close.columns)

        fwd_ret = close.pct_change(n_hold).shift(-n_hold).values  # (T, N)

        price_feats = self._price_features(close, volume)
        if store is not None:
            fund_feats = self._fund_features(close, store)
            features = np.concatenate([price_feats, fund_feats], axis=-1)
        else:
            features = price_feats

        n_feats = features.shape[-1]
        X_flat = features.reshape(-1, n_feats)
        y_flat = fwd_ret.reshape(-1)

        valid = ~np.isnan(y_flat)
        X_train = X_flat[valid]
        y_train = y_flat[valid]

        if len(X_train) < n_tickers:
            log.warning("Insufficient training data — model not updated.")
            return

        X_train = self.imputer.fit_transform(X_train)
        X_train = self.scaler.fit_transform(X_train)
        self.model.fit(X_train, y_train)
        self._is_trained = True
        self._last_trained_date = as_of
        log.info(f"Predictor trained on {len(X_train)} samples × {n_feats} features.")

    def infer(
        self,
        close: pd.DataFrame,
        volume: pd.DataFrame,
        store: Optional["FeatureStore"] = None,
    ) -> pd.Series:
        """
        Generate prediction scores using the already-fitted model.
        Call this at signal generation time (e.g. Friday open) after load().
        Raises RuntimeError if the model has not been trained or loaded yet.
        """
        if not self._is_trained:
            raise RuntimeError(
                "Predictor has not been trained. Call train() or load() first."
            )

        price_feats = self._price_features(close, volume)
        if store is not None:
            fund_feats = self._fund_features(close, store)
            features = np.concatenate([price_feats, fund_feats], axis=-1)
        else:
            features = price_feats

        X_pred = self.imputer.transform(features[-1])
        X_pred = self.scaler.transform(X_pred)
        return pd.Series(self.model.predict(X_pred), index=close.columns)

    def fit_predict(
        self,
        close: pd.DataFrame,
        volume: pd.DataFrame,
        store: Optional["FeatureStore"] = None,
        as_of: Optional[pd.Timestamp] = None,
    ) -> pd.Series:
        """
        Convenience method used by the backtester: train then immediately infer.
        Pass `as_of` to enable the retrain-frequency gate (recommended).
        Not intended for production — use train()+save() / load()+infer() instead.
        """
        self.train(close, volume, store, as_of=as_of)
        return self.infer(close, volume, store)
