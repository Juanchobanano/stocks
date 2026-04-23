# CLAUDE.md

Approach
Think before acting. Read existing files before writing code.
Be concise in output but thorough in reasoning.
Prefer editing over rewriting whole files.
Do not re-read files you have already read unless the file may have changed.
Skip files over 100KB unless explicitly required.
Suggest running /cost when a session is running long to monitor cache ratio.
Recommend starting a new session when switching to an unrelated task.
Test your code before declaring done.
No sycophantic openers or closing fluff.
Keep solutions simple and direct.
User instructions always override this file.


This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Coding style
Respect linter and PEP-8

## Running the algorithm

```bash
pip install -r requirements.txt
python main.py       # full backtest
python example.py    # four standalone usage examples
```

Always run from the project root — all package imports are relative to it.

## Architecture

Long-short equity backtesting system rewritten from a Quantopian algorithm. Code is organized into five packages plus a root-level `config.py`.

**Data flow:**

```
data/ ──────────────────────────────────────┐
  universe.py   → S&P 500 tickers           │
  market.py     → price + volume download   │
                                            ▼
signals/ ───────────────────────────► engine/backtest.py ──► reporting/
  factors.py    → cross-sectional scores        ▲
  predictor.py  → Lasso ML model                │
                                            portfolio/
portfolio/                                  optimizer.py
  optimizer.py  → cvxpy LP weights
```

**Package responsibilities:**

- `config.py` — Single `Config` dataclass; all tunable parameters live here (universe size, leverage, dates, rebalance frequency, Lasso alpha).
- `data/` — Fetching only. `universe.py` pulls S&P 500 tickers from Wikipedia; `market.py` bulk-downloads adjusted close + volume via yfinance and drops tickers with < 80% coverage.
- `signals/` — Signal generation. `factors.py` builds cross-sectional factor scores (winsorized + z-scored momentum, reversal, volatility, volume trend). `predictor.py` wraps a Lasso model trained on stacked `(date × ticker, feature)` matrices.
- `portfolio/` — Portfolio construction. `optimizer.py` solves a cvxpy LP: maximize alpha subject to dollar-neutrality, gross leverage cap, and per-position concentration limits.
- `engine/` — Backtest loop. `backtest.py` iterates over trading days, calls `_get_weights()` on rebalance days, and marks the portfolio to market daily.
- `reporting/` — Output only. `summary.py` computes total return, Sharpe, annualized volatility, and max drawdown.

## Key design decisions

- **`combined` signal blend** in `engine/backtest.py:_get_weights` adds the rule-based factor score (`factors["combined"]`) to the ML predictions before passing to the optimizer. Adjust the blend here.
- **Weekly rebalancing** (`W-FRI`) is the default; daily is impractical with yfinance free-tier rate limits.
- **No historical fundamentals**: all factors are price/volume-derived. Adding a paid data source (Polygon, Tiingo) would touch `data/market.py` and `signals/factors.py` only.
- **`lookback_days`** controls both the factor computation window and the ML training window. Longer = more training data, slower rebalances.
- Each `__init__.py` re-exports its package's public API so callers import from the package name, not the submodule (e.g. `from signals import Predictor`, not `from signals.predictor import Predictor`).
