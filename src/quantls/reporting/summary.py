import numpy as np
import pandas as pd

from src.quantls.config import Config


def compute_metrics(results: pd.DataFrame, cfg: Config) -> dict:
    """Return a dict of performance metrics for programmatic use."""
    pv = results["portfolio_value"]
    daily_ret = pv.pct_change().dropna()

    total_ret   = (pv.iloc[-1] / cfg.initial_capital - 1) * 100
    ann_vol     = daily_ret.std() * np.sqrt(252) * 100
    sharpe      = daily_ret.mean() / (daily_ret.std() + 1e-9) * np.sqrt(252)
    max_dd      = (pv / pv.cummax() - 1).min() * 100
    calmar      = (total_ret / 100) / (abs(max_dd / 100) + 1e-9)

    downside    = daily_ret[daily_ret < 0]
    sortino     = (
        daily_ret.mean() / (downside.std() + 1e-9) * np.sqrt(252)
        if len(downside) > 1 else np.nan
    )

    win_days    = (daily_ret > 0).sum()
    win_rate    = win_days / len(daily_ret) * 100

    return {
        "total_return_pct":  total_ret,
        "ann_volatility_pct": ann_vol,
        "sharpe":            sharpe,
        "sortino":           sortino,
        "max_drawdown_pct":  max_dd,
        "calmar":            calmar,
        "win_rate_pct":      win_rate,
    }


def print_summary(results: pd.DataFrame, cfg: Config) -> None:
    m = compute_metrics(results, cfg)

    print("\n" + "=" * 54)
    print("  BACKTEST RESULTS")
    print("=" * 54)
    print(f"  Period:          {cfg.start_date}  →  {cfg.end_date}")
    print(f"  Universe:        {cfg.universe_size} S&P 500 stocks")
    print(f"  Positions:       {cfg.long_n} long / {cfg.short_n} short")
    print(f"  Total Return:    {m['total_return_pct']:+.2f}%")
    print(f"  Ann. Volatility: {m['ann_volatility_pct']:.2f}%")
    print(f"  Sharpe Ratio:    {m['sharpe']:.3f}")
    print(f"  Sortino Ratio:   {m['sortino']:.3f}")
    print(f"  Max Drawdown:    {m['max_drawdown_pct']:.2f}%")
    print(f"  Calmar Ratio:    {m['calmar']:.3f}")
    print(f"  Win Rate:        {m['win_rate_pct']:.1f}%")
    print("=" * 54 + "\n")
