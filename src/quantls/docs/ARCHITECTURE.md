# Architecture

## Overview

This is a **long-short equity backtesting system** built around three distinct stages: signal generation, portfolio construction, and execution simulation. The system ranks stocks by a combination of rule-based factors and a machine learning model, goes long on the highest-ranked and short on the lowest-ranked, and optimizes position weights subject to risk constraints.

---

## Project Structure

```
stocks/
├── config.py           # All tunable parameters
├── main.py             # Entry point
├── example.py          # Usage examples
│
├── data/               # Data acquisition
│   ├── universe.py     # S&P 500 ticker list
│   └── market.py       # Price + volume download
│
├── signals/            # Alpha generation
│   ├── factors.py      # Rule-based cross-sectional factors
│   └── predictor.py    # Lasso ML model
│
├── portfolio/          # Portfolio construction
│   └── optimizer.py    # Convex optimization (cvxpy)
│
├── engine/             # Orchestration
│   └── backtest.py     # Backtest loop + scheduling
│
└── reporting/          # Output
    └── summary.py      # Performance metrics
```

---

## Data Flow

```
         ┌─────────────┐
         │  data/      │
         │  universe   │──► S&P 500 tickers (up to universe_size)
         │  market     │──► close prices, volume  (adjusted, yfinance)
         └──────┬──────┘
                │  rolling lookback window (lookback_days)
                ▼
    ┌───────────────────────┐
    │       signals/        │
    │                       │
    │  factors.py           │──► combined factor score  (rule-based)
    │  ┌─────────────────┐  │
    │  │ winsorize        │  │
    │  │ z-score          │  │
    │  │ equal-weight avg │  │
    │  └─────────────────┘  │
    │                       │
    │  predictor.py         │──► ML score  (Lasso regression)
    │  ┌─────────────────┐  │
    │  │ impute + scale   │  │
    │  │ fit on history   │  │
    │  │ predict today    │  │
    │  └─────────────────┘  │
    └───────────┬───────────┘
                │  combined = factor_score + ml_score
                │  universe = top-N ∪ bottom-N
                ▼
    ┌───────────────────────┐
    │     portfolio/        │
    │     optimizer         │──► target weights (one per stock)
    │  ┌─────────────────┐  │
    │  │ maximize alpha   │  │
    │  │ dollar neutral   │  │
    │  │ leverage cap     │  │
    │  │ position limits  │  │
    │  └─────────────────┘  │
    └───────────┬───────────┘
                │
                ▼
    ┌───────────────────────┐
    │       engine/         │
    │       backtest        │──► daily portfolio value
    │  ┌─────────────────┐  │
    │  │ mark-to-market   │  │
    │  │ weekly rebalance │  │
    │  └─────────────────┘  │
    └───────────┬───────────┘
                │
                ▼
    ┌───────────────────────┐
    │      reporting/       │──► total return, Sharpe, drawdown
    └───────────────────────┘
```

---

## Stage 1 — Signal Generation (`signals/`)

### Rule-Based Factors (`factors.py`)

Six cross-sectional factors are computed from price and volume data. Each is independently winsorized (clipped at the 5th and 95th percentile) and z-scored before being averaged into a single `combined` score.

| Factor | Computation | Economic intuition |
|---|---|---|
| `mom_1m` | 21-day return | Short-term momentum |
| `mom_3m` | 63-day return | Medium-term momentum |
| `mom_12m` | 252-day return | Long-term trend |
| `reversal` | −(5-day return) | Short-term mean reversion |
| `neg_volatility` | −(21-day realized vol) | Quality proxy — low vol stocks tend to outperform |
| `vol_trend` | 21-day avg vol / 63-day avg vol | Rising interest / liquidity signal |

**Processing pipeline per factor:**
```
raw cross-section → winsorize(5%, 95%) → z-score → combined = mean of all factors
```

### ML Predictor (`predictor.py`)

A **Lasso regression** is retrained at every rebalance using all available labeled history. The model learns which lagged features predict next-day returns.

**Feature matrix** — built by stacking all `(date × ticker)` pairs into a flat `(T×N, 4)` matrix:

| Feature | Computation |
|---|---|
| `ret_1d` | 1-day return |
| `ret_5d` | 5-day return |
| `ret_21d` | 21-day return |
| `vol_ratio` | 21-day avg volume / 63-day avg volume |

**Target**: `n_hold`-day forward return (default: 1 day), aligned back to the feature date.

**Training / prediction split:**
```
All (date × ticker) pairs where forward return is known → train
Last N rows (today's tickers) → predict
```

Lasso is used because it performs feature selection via L1 regularization — irrelevant features are driven to zero, which matters when the feature set grows.

---

## Stage 2 — Portfolio Construction (`portfolio/`)

### Optimizer (`optimizer.py`)

The combined signal (factor score + ML score) is treated as proportional to expected return. The optimizer finds the weight vector that maximizes expected return subject to three constraints, formulated as a **linear program** solved by cvxpy (OSQP solver):

```
maximize    αᵀw

subject to  Σ wᵢ = 0                    (dollar neutral)
            Σ |wᵢ| ≤ max_gross_leverage  (gross leverage cap)
            −max_pos ≤ wᵢ ≤ max_pos      (position concentration)
```

**Constraint rationale:**

| Constraint | Purpose |
|---|---|
| Dollar neutral | Isolates stock-selection alpha from market beta |
| Gross leverage | Controls total capital at risk |
| Position limits | Prevents over-concentration in any single name |

The tradeable universe passed to the optimizer is restricted to the top-N and bottom-N stocks by combined score, so the LP is small and fast even with large universes.

---

## Stage 3 — Backtest Engine (`engine/`)

### Backtest Loop (`backtest.py`)

The engine iterates over every trading day in the backtest window and performs two operations:

1. **Daily mark-to-market** — portfolio value is updated using the realized return of each held position.
2. **Weekly rebalance** (default: every Friday) — new target weights are computed by calling `_get_weights()`, which runs the full signal + optimization pipeline on the rolling lookback window ending on that date.

```
for each trading day:
    capital *= 1 + Σ (wᵢ × daily_returnᵢ)   ← mark-to-market
    if rebalance day:
        weights = _get_weights(close, volume, date)
```

**Lookback window**: the most recent `lookback_days` trading days of data are passed to the signal and ML stages on each rebalance. This simulates the information actually available at that point in time and prevents look-ahead bias.

---

## Configuration (`config.py`)

All parameters are in a single `Config` dataclass. The most impactful ones:

| Parameter | Default | Effect |
|---|---|---|
| `universe_size` | 100 | Number of S&P 500 stocks to consider |
| `total_positions` | 40 | 20 long + 20 short |
| `lookback_days` | 126 | ML training window length (~6 months) |
| `lasso_alpha` | 0.01 | Regularization strength; higher = sparser model |
| `max_gross_leverage` | 1.0 | Total capital deployed (1.0 = fully invested) |
| `rebalance_freq` | `W-FRI` | How often weights are updated |

---

## Comparison with the Original Quantopian Algorithm

| Dimension | Original (Quantopian) | This Rewrite |
|---|---|---|
| Universe | `QTradableStocksUS` (~1,500 stocks) | S&P 500 top-N |
| Fundamentals | Morningstar (EBIT, ROE, growth scores) | Not available — replaced with price/volume factors |
| Sentiment | StockTwits `bull_minus_bear` | Replaced with short-term price reversal |
| Factor weights | Style score weighted 20× others | Equal-weight average |
| Positions | 600 (300L / 300S) | 40 default (configurable) |
| Risk model | `RiskModelExposure` (sector/style neutralization) | Not implemented — no equivalent free data source |
| Rebalance | Daily, 30 min after open | Weekly (API rate limit constraint) |
| Optimizer | `quantopian.optimize` | `cvxpy` with equivalent LP constraints |
| ML features | Open price, volume, 8 Morningstar fundamentals | 4 price/volume-derived features |
