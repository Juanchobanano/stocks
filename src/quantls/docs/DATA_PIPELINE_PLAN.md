# Data Engineering Pipeline Plan

## Goal

Build a pipeline that makes every feature from the original `old_algorithm.txt` available
as a clean, point-in-time daily time series, ready to be consumed by `signals/factors.py`
and `signals/predictor.py` without look-ahead bias.

---

## Feature Inventory

### Combined Factor Features (rule-based signal)

| Feature | Original source | Availability | How to obtain |
|---|---|---|---|
| `value` = EBIT / EV | Morningstar via Quantopian | Computable | EBIT from Polygon income statement; EV = market_cap + debt − cash |
| `quality` = ROE | Morningstar via Quantopian | Computable | net_income / equity from Polygon balance sheet |
| `sentiment_score` = 3-day SMA of bull_minus_bear | StockTwits / psychsignal | Paid only | StockTwits Data API (historical) or Polygon news + NLP |
| `growth_score` | Morningstar proprietary | Approximable | Composite of revenue growth + EPS growth + equity growth |
| `value_score` | Morningstar proprietary | Approximable | Composite of earning_yield + P/B + P/S |
| `style_score` | Morningstar proprietary | Approximable | Weighted blend of value_score + growth_score composites |

### ML Model Features (Predictor inputs)

| Feature | Original source | Availability | How to obtain |
|---|---|---|---|
| `Open Price` | USEquityPricing | ✅ Free | yfinance |
| `Volume` | USEquityPricing | ✅ Free | yfinance |
| `equity_per_share_growth` | Morningstar | Computable | QoQ change in (equity / shares_outstanding) from Polygon |
| `growth_score` | Morningstar | Approximable | Same composite as above |
| `value_score` | Morningstar | Approximable | Same composite as above |
| `sustainable_growth_rate` | Morningstar | Computable | ROE × (1 − dividend_payout_ratio) |
| `earning_yield` | Morningstar | Computable | EPS / price (= 1 / P/E) |
| `pb_ratio` | Morningstar | Computable | price / (equity / shares_outstanding) |
| `pe_ratio` | Morningstar | Computable | price / EPS |
| `roa` | Morsonstar | Computable | net_income / total_assets |

---

## Data Sources

### 1. yfinance — Price & Volume
- **What**: Daily adjusted OHLCV, current shares outstanding, current P/E, P/B
- **Limitations**: Fundamental data is point-in-time snapshot only (not historical series)
- **Use for**: Open price, volume, current market cap

### 2. Polygon.io — Historical Fundamentals
- **What**: Quarterly income statement, balance sheet, cash flow statement per ticker
- **Key fields**:
  - Income: `revenues`, `operating_income_loss` (EBIT), `net_income_loss`, `basic_earnings_per_share`, `diluted_shares_outstanding`
  - Balance: `equity`, `assets`, `long_term_debt`, `cash_and_cash_equivalents_and_short_term_investments`
- **Filing lag**: Reports filed 1–3 months after period end — must use `filing_date`, not `period_of_report_date`
- **Rate limits**: Free = 5 req/min; Starter ($29/mo) = unlimited

### 3. StockTwits — Historical Sentiment
- **Option A** – StockTwits Data API (paid): provides historical daily bull/bear message counts per symbol
- **Option B** – Polygon News API + NLP: fetch news headlines per ticker per day, score with a sentiment model (VADER or FinBERT)
- **Option B is free** and produces a daily sentiment time series usable in backtesting

---

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        RAW INGESTION                                │
│                                                                     │
│  yfinance          Polygon.io           StockTwits / Polygon News   │
│  (daily OHLCV)     (quarterly filings)  (daily sentiment)           │
└────────┬───────────────────┬─────────────────────┬──────────────────┘
         │                   │                     │
         ▼                   ▼                     ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     TRANSFORMATION LAYER                            │
│                                                                     │
│  price_pipeline.py      fundamental_pipeline.py   sentiment_pipeline.py │
│  ─────────────────       ────────────────────────  ──────────────────── │
│  • Adjust splits        • Parse Polygon JSON        • Count bull/bear   │
│  • Compute market cap   • Compute derived ratios    • 3-day rolling SMA │
│  • Align to trade days  • Forward-fill to daily     • Normalize [-1,+1] │
│                         • Point-in-time alignment                   │
└────────┬───────────────────┬─────────────────────┬──────────────────┘
         │                   │                     │
         ▼                   ▼                     ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       SCORE LAYER                                   │
│                                                                     │
│  score_pipeline.py                                                  │
│  ──────────────────────────────────────────────────────────────     │
│  • Compute growth_score  = composite(rev_growth, eps_growth,        │
│                                      equity_growth)                 │
│  • Compute value_score   = composite(earning_yield, pb, ps)         │
│  • Compute style_score   = blend(value_score, growth_score)         │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       FEATURE STORE                                 │
│                                                                     │
│  .cache/                                                            │
│  ├── prices.parquet          daily OHLCV + market_cap               │
│  ├── fundamentals.parquet    daily forward-filled fundamental ratios│
│  ├── scores.parquet          daily growth/value/style scores        │
│  └── sentiment.parquet       daily bull_minus_bear SMA              │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    EXISTING SIGNAL CODE                             │
│                                                                     │
│  signals/factors.py     → reads fundamentals + scores + sentiment  │
│  signals/predictor.py   → reads prices + fundamentals + scores     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Stage Details

### Stage 1 — Price Pipeline (`pipeline/price.py`)

**Input**: yfinance daily OHLCV  
**Output**: `prices.parquet` — columns: `open`, `close`, `volume`, `market_cap`

Steps:
1. Download adjusted OHLCV via `yf.download()` (already in `data/market.py`)
2. Compute `market_cap = close × shares_outstanding` using shares from the most recent
   Polygon filing (forward-filled)
3. Store as wide-format parquet: rows = dates, columns = (ticker, field)

---

### Stage 2 — Fundamental Pipeline (`pipeline/fundamentals.py`)

**Input**: Polygon.io quarterly filings  
**Output**: `fundamentals.parquet` — one row per (date, ticker), forward-filled daily

Key computations:

```
ebit              = operating_income_loss
enterprise_value  = market_cap + long_term_debt − cash
value (EBIT/EV)   = ebit / enterprise_value

roe               = net_income / equity
roa               = net_income / assets
equity_per_share  = equity / shares_outstanding
eps               = basic_earnings_per_share

pe_ratio          = price / eps
pb_ratio          = price / equity_per_share
earning_yield     = eps / price   (= 1 / pe_ratio)

sustainable_growth_rate = roe × (1 − dividend_payout_ratio)
    where dividend_payout_ratio = dividends_paid / net_income  (from cash flow)

equity_per_share_growth = QoQ change in equity_per_share
```

**Point-in-time rule**: A filing is only made available on its `filing_date`, not its
`period_of_report_date`. This prevents using Q3 data before it was actually published.

**Forward-fill rule**: After a filing is published, its values are held constant until the
next filing arrives. This mirrors how an investor would use the data.

---

### Stage 3 — Score Pipeline (`pipeline/scores.py`)

**Input**: `fundamentals.parquet`  
**Output**: `scores.parquet` — daily composite growth, value, and style scores

Since Morningstar's proprietary scores are unavailable, we approximate them with
equally-weighted composites of their constituent signals:

```
growth_score = mean(
    z_score(revenue_growth_ttm),      # trailing twelve month revenue growth
    z_score(eps_growth_ttm),           # trailing twelve month EPS growth
    z_score(equity_per_share_growth),  # quarterly equity growth
)

value_score = mean(
    z_score(earning_yield),            # EPS / price
    z_score(−pb_ratio),                # lower P/B = higher value
    z_score(−pe_ratio),                # lower P/E = higher value
)

style_score = 0.5 × value_score + 0.5 × growth_score
```

Each component is z-scored cross-sectionally before averaging — the same processing
used in the original `combined_factor` pipeline.

---

### Stage 4 — Sentiment Pipeline (`pipeline/sentiment.py`)

**Input**: StockTwits Data API (paid) or Polygon News API + NLP (free)  
**Output**: `sentiment.parquet` — daily `bull_minus_bear` per ticker, 3-day SMA

**Option A — StockTwits historical (paid, most faithful to original)**:
- Endpoint: `GET /api/2/streams/symbol/{ticker}.json?since={date}`
- Aggregate daily: bull_count, bear_count per ticker
- `bull_minus_bear = (bull − bear) / (bull + bear + ε)`
- Apply 3-day rolling mean to match original `SimpleMovingAverage(window_length=3)`

**Option B — Polygon News + FinBERT (free)**:
- Endpoint: `GET /v2/reference/news?ticker={ticker}&published_utc.gte={date}`
- Score each headline with FinBERT (financial domain sentiment model)
- Aggregate to daily score per ticker
- Apply 3-day rolling mean

Option B is recommended for backtesting since it is free and produces a true historical
time series. Option A is more faithful to the original signal definition.

---

### Stage 5 — Feature Store (`pipeline/store.py`)

All four parquet files share the same index structure: `(date, ticker)`.

```
.cache/
├── prices.parquet
│     index: (date, ticker)
│     cols:  open, close, volume, market_cap
│
├── fundamentals.parquet
│     index: (date, ticker)
│     cols:  ebit, enterprise_value, roe, roa, pe_ratio, pb_ratio,
│            earning_yield, sustainable_growth_rate, equity_per_share_growth
│
├── scores.parquet
│     index: (date, ticker)
│     cols:  growth_score, value_score, style_score
│
└── sentiment.parquet
      index: (date, ticker)
      cols:  bull_minus_bear, bull_minus_bear_sma3
```

A single `FeatureStore` class will:
- Load all parquet files into memory at startup
- Expose `get(date, tickers)` → flat DataFrame of all features for a given date
- Handle forward-filling automatically on read

---

## New Files to Create

```
pipeline/
├── __init__.py
├── price.py           Stage 1 — OHLCV + market cap
├── fundamentals.py    Stage 2 — Polygon filings → daily ratios
├── scores.py          Stage 3 — composite growth/value/style scores
├── sentiment.py       Stage 4 — StockTwits or Polygon news NLP
└── store.py           Stage 5 — FeatureStore read/write interface
```

Existing files to modify:

```
signals/factors.py     reads from FeatureStore instead of computing from raw prices
signals/predictor.py   feature set expanded to all original ML inputs
engine/backtest.py     initialises FeatureStore once, passes slices to signals
config.py              add stocktwits_api_key, sentiment_source ("stocktwits"|"polygon_news")
```

---

## Update Frequency

| Dataset | Update frequency | Notes |
|---|---|---|
| Prices | Daily (after market close) | Re-download last 5 days to catch corrections |
| Fundamentals | On new filing detected | Polygon webhook or daily polling |
| Scores | Daily (recomputed from fundamentals) | Cheap: just arithmetic on cached data |
| Sentiment | Daily | Fetch previous day's messages/news |

---

## Open Questions Before Implementation

1. **Sentiment source**: StockTwits paid API or Polygon News + FinBERT?
   - StockTwits: closer to original, requires subscription
   - FinBERT: free, requires running an NLP model locally (adds dependency)

2. **style_score weight**: The original gave `style_score` a weight of **2.0** (20× others).
   Should the composite `style_score` keep that weight, or should all factors be equal-weighted?

3. **Parquet vs database**: For larger universes (>500 tickers, multi-year history),
   parquet files may become slow to load. Should the store use DuckDB or SQLite instead?

4. **Pipeline trigger**: Should the pipeline run on a schedule (cron), on-demand before
   each backtest, or continuously as a background service?
