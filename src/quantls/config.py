from dataclasses import dataclass


@dataclass
class Config:
    # Universe
    universe_size: int = 100

    # Portfolio
    total_positions: int = 40
    max_gross_leverage: float = 1.0

    # ML model
    holding_days: int = 1
    lookback_days: int = 126
    lasso_alpha: float = 0.01
    ml_weight: float = 1.5       # scale applied to z-scored ML scores before blending
    ml_retrain_days: int = 21    # minimum calendar days between Lasso retrains

    # Factor processing
    winsorize_low: float = 0.05
    winsorize_high: float = 0.95

    # Backtest
    start_date: str = "2022-01-01"
    end_date: str = "2023-12-31"
    rebalance_freq: str = "W-FRI"
    initial_capital: float = 1_000_000.0

    # Polygon.io — leave empty to run on price/volume proxies only
    polygon_api_key: str = ""
    polygon_requests_per_minute: int = 5   # free tier: 5; paid tiers: higher

    # Factor weights for combined signal (must sum to 1.0 for interpretability)
    factor_weight_value: float = 0.20
    factor_weight_quality: float = 0.20
    factor_weight_sentiment: float = 0.10
    factor_weight_growth: float = 0.15
    factor_weight_value_score: float = 0.15
    factor_weight_style_score: float = 0.20

    # Feature store
    db_path: str = ".cache/features.db"

    @property
    def long_n(self) -> int:
        return self.total_positions // 2

    @property
    def short_n(self) -> int:
        return self.total_positions // 2

    @property
    def max_position_size(self) -> float:
        return 2.0 / self.total_positions
