from typing import Optional
import numpy as np
import pandas as pd
from src.quantls.config import Config


def winsorize(s: pd.Series, low: float, high: float) -> pd.Series:
    return s.clip(s.quantile(low), s.quantile(high))


def zscore(s: pd.Series) -> pd.Series:
    return (s - s.mean()) / (s.std() + 1e-9)


def process_factor(s: pd.Series, low: float = 0.05, high: float = 0.95) -> pd.Series:
    """Winsorize then z-score a cross-sectional factor series."""
    return zscore(winsorize(s.dropna(), low, high))


def compute_factors(
    close: pd.DataFrame,
    volume: pd.DataFrame,
    cfg: Config,
    features: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Build a cross-sectional factor table (one row per ticker) as of the last
    date in the window.

    `features` is a DataFrame indexed by ticker returned by FeatureStore.get().
    When a column is present it replaces the corresponding price/volume proxy.

    Combined factor formula uses weights from cfg.factor_weight_* fields.
    """
    w_lo, w_hi = cfg.winsorize_low, cfg.winsorize_high
    tickers = close.columns

    def _feat(col: str) -> Optional[pd.Series]:
        if features is not None and col in features.columns:
            return features[col].reindex(tickers)
        return None

    def _proc(s: pd.Series) -> pd.Series:
        return process_factor(s, w_lo, w_hi)

    def _or(feat_val, fallback):
        return feat_val if feat_val is not None else fallback

    value = _or(_feat("ebit_to_ev"), close.pct_change(21).iloc[-1])
    quality = _or(_feat("roe"), close.pct_change().rolling(21).std().iloc[-1].mul(-1))
    sentiment_score = _or(_feat("bull_minus_bear_3d"), close.pct_change(5).iloc[-1].mul(-1))
    growth_score = _or(
        _feat("growth_score"),
        close.pct_change(252).iloc[-1] if len(close) >= 252
        else pd.Series(np.nan, index=tickers),
    )
    value_score = _or(_feat("value_score"), close.pct_change(63).iloc[-1])
    style_score = _or(
        _feat("style_score"),
        (volume.rolling(21).mean() / (volume.rolling(63).mean() + 1e-9)).iloc[-1],
    )

    v_ = _proc(value)
    q_ = _proc(quality)
    s_ = _proc(sentiment_score)
    g_ = _proc(growth_score)
    vs_ = _proc(value_score)
    ss_ = _proc(style_score)

    combined = (
        v_ * cfg.factor_weight_value
        + q_ * cfg.factor_weight_quality
        + s_ * cfg.factor_weight_sentiment
        + g_ * cfg.factor_weight_growth
        + vs_ * cfg.factor_weight_value_score
        + ss_ * cfg.factor_weight_style_score
    )

    return pd.DataFrame({
        "value":        v_,
        "quality":      q_,
        "sentiment":    s_,
        "growth_score": g_,
        "value_score":  vs_,
        "style_score":  ss_,
        "combined":     combined,
    }).reindex(tickers)
