import logging

from src.quantls.config import Config
from src.quantls.data import fetch_market_data, get_sp500_tickers
from src.quantls.engine import Backtest
from src.quantls.portfolio import optimize_portfolio
from src.quantls.reporting import print_summary
from src.quantls.signals import Predictor, compute_factors
import warnings
warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Example 1 — Standard backtest with default settings
# ─────────────────────────────────────────────────────────────────────────────
def example_standard_backtest():
    print("\n" + "─" * 54)
    print("  Example 1: Standard Backtest")
    print("─" * 54)

    cfg = Config(
        universe_size=100,
        total_positions=40,
        start_date="2022-01-01",
        end_date="2023-12-31",
    )

    results = Backtest(cfg).run()
    print_summary(results, cfg)
    results.to_csv("results_standard.csv")
    log.info("Saved → results_standard.csv")


# ─────────────────────────────────────────────────────────────────────────────
# Example 2 — Compare aggressive vs. conservative configurations
# ─────────────────────────────────────────────────────────────────────────────
def example_compare_configs():
    print("\n" + "─" * 54)
    print("  Example 2: Aggressive vs. Conservative")
    print("─" * 54)

    aggressive = Config(
        universe_size=100,
        total_positions=60,
        max_gross_leverage=1.5,
        lasso_alpha=0.001,
        start_date="2022-01-01",
        end_date="2023-12-31",
    )
    conservative = Config(
        universe_size=100,
        total_positions=20,
        max_gross_leverage=0.8,
        lasso_alpha=0.1,
        start_date="2022-01-01",
        end_date="2023-12-31",
    )

    for label, cfg in [("Aggressive", aggressive), ("Conservative", conservative)]:
        log.info(f"Running {label} backtest …")
        results = Backtest(cfg).run()
        print(f"\n  [{label}]")
        print_summary(results, cfg)
        results.to_csv(f"results_{label.lower()}.csv")
        log.info(f"Saved → results_{label.lower()}.csv")


# ─────────────────────────────────────────────────────────────────────────────
# Example 3 — Inspect factor scores for a snapshot date
# ─────────────────────────────────────────────────────────────────────────────
def example_factor_inspection():
    print("\n" + "─" * 54)
    print("  Example 3: Factor Score Inspection")
    print("─" * 54)

    cfg = Config(universe_size=30, lookback_days=126)
    tickers = get_sp500_tickers()[:30]
    close, volume = fetch_market_data(tickers, start="2023-10-01", end="2024-01-01")

    factors = compute_factors(close, volume, cfg)
    ranked = factors.sort_values("combined", ascending=False)

    print("\n  Top 5 stocks by combined factor score:")
    print(ranked[["mom_1m", "mom_3m", "reversal", "combined"]].head(5).to_string())
    print("\n  Bottom 5 stocks by combined factor score:")
    print(ranked[["mom_1m", "mom_3m", "reversal", "combined"]].tail(5).to_string())


# ─────────────────────────────────────────────────────────────────────────────
# Example 4 — Run ML predictor and optimizer standalone
# ─────────────────────────────────────────────────────────────────────────────
def example_predictor_and_optimizer():
    print("\n" + "─" * 54)
    print("  Example 4: Standalone Predictor + Optimizer")
    print("─" * 54)

    cfg = Config(universe_size=20, total_positions=10, lookback_days=126)
    tickers = get_sp500_tickers()[:20]
    close, volume = fetch_market_data(tickers, start="2023-07-01", end="2024-01-01")

    ml_scores = Predictor(cfg).fit_predict(close, volume)
    print("\n  ML prediction scores (top 5):")
    print(ml_scores.nlargest(5).to_string())

    weights = optimize_portfolio(ml_scores, cfg)
    longs = weights[weights > 0].sort_values(ascending=False)
    shorts = weights[weights < 0].sort_values()

    print(f"\n  Long positions  ({len(longs)}):")
    print(longs.to_string())
    print(f"\n  Short positions ({len(shorts)}):")
    print(shorts.to_string())
    print(f"\n  Gross leverage: {weights.abs().sum():.4f}")
    print(f"  Net exposure:   {weights.sum():.6f}")


# ─────────────────────────────────────────────────────────────────────────────
# Example 5 — Live production trading via Interactive Brokers
# ─────────────────────────────────────────────────────────────────────────────
def example_live_ib():
    """
    Full production workflow.

    Prerequisites:
      - TWS or IB Gateway is running and API connections are enabled
      - Fractional share trading is enabled in account settings (optional)
      - Run engine.retrain() at least once before calling generate_signals()

    Ports:
      TWS paper  → 7497   TWS live  → 7496
      GW paper   → 4002   GW live   → 4001
    """
    print("\n" + "─" * 54)
    print("  Example 5: Live IB Trading (dry_run=True)")
    print("─" * 54)

    from broker import IBBroker
    from src.quantls.engine import LiveEngine

    cfg = Config(
        polygon_api_key="YOUR_POLYGON_KEY",
        universe_size=100,
        total_positions=40,
    )

    engine = LiveEngine(cfg)

    # ── Friday EOD: retrain the model ─────────────────────────────────────────
    # engine.retrain()

    # ── Friday open: generate target weights ──────────────────────────────────
    weights = engine.generate_signals()
    log.info(f"Target weights:\n{weights.sort_values().to_string()}")

    # ── Submit to IB ──────────────────────────────────────────────────────────
    with IBBroker(
        port=7497,              # paper trading port; change to 7496 for live
        order_type="MKT",       # "MKT" or "LMT"
        stagger_seconds=0.5,    # 0.5 s between orders
        dry_run=True,           # set False to send real orders
        fractional=True,
    ) as broker:
        broker.rebalance(weights)


# ─────────────────────────────────────────────────────────────────────────────
# Run all examples
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Comment out any examples you don't want to run.
    example_standard_backtest()
    example_compare_configs()
    example_factor_inspection()
    example_predictor_and_optimizer()
    # example_live_ib()   # uncomment when TWS / IB Gateway is running
