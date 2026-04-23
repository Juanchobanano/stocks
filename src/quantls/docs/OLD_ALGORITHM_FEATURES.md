# Feature Definition, Processing & Usage — Original Quantopian Algorithm

This document describes every feature used in the original `old_algorithm.txt`, how each one is constructed, how it is processed before use, and how it feeds into the final trading signal.

---

## 1. Pipeline Overview

The algorithm uses two parallel feature pipelines that are merged before optimization:

```
Fundamentals + Sentiment
        │
        ▼
Rule-Based Factors ──────────────────────────────┐
  (value, quality, sentiment, growth,              │
   value_score, style_score)                       ├──► combined_factor + ML_score
        │                                          │         │
        ▼                                          │         ▼
  combined_factor ──────────────────────────────►  │    MaximizeAlpha(final_data)
                                                   │
ML Model (Predictor CustomFactor) ────────────────┘
  (open price, volume, 8 Morningstar fundamentals)
```

---

## 2. Rule-Based Pipeline Features

These six features are defined in `make_pipeline()` and form the `combined_factor` baseline signal.

### 2.1 Value
```python
value = Fundamentals.ebit.latest / Fundamentals.enterprise_value.latest
```
**Definition**: EBIT-to-Enterprise-Value ratio — a measure of how cheaply a company's operating earnings can be acquired.  
**Type**: Point-in-time fundamental (Morningstar).  
**Direction**: Higher = cheaper = more attractive for longs.

---

### 2.2 Quality
```python
quality = Fundamentals.roe.latest
```
**Definition**: Return on Equity — net income divided by shareholders' equity.  
**Type**: Point-in-time fundamental (Morningstar).  
**Direction**: Higher = more profitable = more attractive for longs.

---

### 2.3 Sentiment Score
```python
sentiment_score = SimpleMovingAverage(
    inputs=[stocktwits.bull_minus_bear],
    window_length=3,
)
```
**Definition**: 3-day moving average of the StockTwits `bull_minus_bear` signal — the net count of bullish minus bearish messages about a stock on StockTwits.  
**Type**: Alternative data (psychsignal / StockTwits feed via Quantopian).  
**Direction**: Higher = more bullish social sentiment = more attractive for longs.  
**Note**: The 3-day window smooths out single-day spikes in social media activity.

---

### 2.4 Growth Score
```python
growth_score = Fundamentals.growth_score.latest
```
**Definition**: Morningstar's proprietary composite growth score, incorporating revenue growth, earnings growth, and book value growth.  
**Type**: Point-in-time fundamental (Morningstar).  
**Direction**: Higher = stronger growth profile = more attractive for longs.

---

### 2.5 Value Score
```python
value_score = Fundamentals.value_score.latest
```
**Definition**: Morningstar's proprietary composite value score, based on P/E, P/B, P/S, P/CF, and dividend yield.  
**Type**: Point-in-time fundamental (Morningstar).  
**Direction**: Higher = cheaper on multiple valuation metrics = more attractive for longs.

---

### 2.6 Style Score
```python
style_score = Fundamentals.style_score.latest
```
**Definition**: Morningstar's proprietary style score that classifies a stock along the value–growth spectrum.  
**Type**: Point-in-time fundamental (Morningstar).  
**Direction**: Higher = stronger style alignment = more attractive for longs.  
**Note**: This factor receives a weight of **2.0** vs. 0.1 for all others — it has **20× more influence** on the combined signal than any individual factor.

---

## 3. Rule-Based Feature Processing

Every factor above goes through the same two-step normalization before combining:

### Step 1 — Winsorization
```python
value_winsorized = value.winsorize(min_percentile=0.05, max_percentile=0.95)
```
Clips values below the 5th percentile to the 5th percentile and above the 95th to the 95th. This limits the influence of extreme outliers (e.g. a company with a distorted EBIT/EV ratio due to a one-time event).

### Step 2 — Z-Scoring
```python
combined_factor = (
    value_winsorized.zscore()      * 0.1 +
    quality_winsorized.zscore()    * 0.1 +
    sentiment_score_winsorized.zscore() * 0.1 +
    growth_score_winsorized.zscore()    * 0.1 +
    value_score_winsorized.zscore()     * 0.1 +
    style_score_winsorized.zscore()     * 2.0
)
```
Z-scoring (subtract mean, divide by std) puts all factors on a common scale regardless of their original units. The weights are then applied after z-scoring so they represent true relative influence.

**Combined factor weight summary:**

| Factor | Weight | Share of total (2.5) |
|---|---|---|
| Value (EBIT/EV) | 0.1 | 4% |
| Quality (ROE) | 0.1 | 4% |
| Sentiment (StockTwits) | 0.1 | 4% |
| Growth Score | 0.1 | 4% |
| Value Score | 0.1 | 4% |
| **Style Score** | **2.0** | **80%** |

---

## 4. ML Model Features (`Predictor` CustomFactor)

The `Predictor` class is a Quantopian `CustomFactor` that trains a Lasso regression inside the pipeline. It uses a different, larger feature set drawn from both price data and Morningstar fundamentals.

### 4.1 Feature Definitions

| Name | Source | Description |
|---|---|---|
| `Open Price` | `USEquityPricing.open` | Daily open price — used **only** to compute the return target (y), then deleted from X |
| `Volume` | `USEquityPricing.volume` | Daily traded volume |
| `Equity Growth` | `Fundamentals.equity_per_share_growth` | Year-over-year growth in book value per share |
| `Growth Score` | `Fundamentals.growth_score` | Morningstar composite growth score (same as rule-based pipeline) |
| `Value Score` | `Fundamentals.value_score` | Morningstar composite value score (same as rule-based pipeline) |
| `Sustainable Growth Rate` | `Fundamentals.sustainable_growth_rate` | ROE × retention ratio — the growth rate a company can sustain without external financing |
| `earning_yield` | `Fundamentals.earning_yield` | Earnings per share / price (inverse of P/E) |
| `pb_ratio` | `Fundamentals.pb_ratio` | Price-to-Book ratio |
| `pe_ratio` | `Fundamentals.pe_ratio` | Price-to-Earnings ratio |
| `roa` | `Fundamentals.roa` | Return on Assets |

The window length is `days_for_fundamentals_analysis = 30` trading days.

### 4.2 Target Variable (y)

```python
y = (np.log(inputs['Open Price']) - np.log(inputs['Open Price'].shift(num_holding_days))) \
    .shift(-num_holding_days - 1) \
    .dropna(axis=0, how='all') \
    .stack(dropna=False)
```

**Definition**: Log return of the open price over `num_holding_days` (1 day), shifted back by `num_holding_days + 1` so that the label aligns with the feature date.  
**Shape after stacking**: A flat 1-D array of `(days × securities)` return values.

Open Price is then **deleted** from the feature dict so it does not leak into X:
```python
del inputs['Open Price']
```

### 4.3 Feature Matrix (X)

```python
x = pd.concat([df.stack(dropna=False) for df in inputs.values()], axis=1)
```

Each of the 9 remaining feature DataFrames (shape: `days × securities`) is stacked into a flat column. The result is a matrix of shape `(days × securities, 9 features)`.

---

## 5. ML Feature Processing

### Step 1 — Null handling (input data)
```python
inputs = OrderedDict([
    (name, pd.DataFrame(arr).fillna(method='ffill', axis=1).fillna(method='bfill', axis=1))
    for name, arr in ...
])
```
Forward-fill then backward-fill along the time axis (axis=1) to handle missing fundamental data. Fundamentals are often reported quarterly and held constant between releases.

### Step 2 — Imputation
```python
x = Imputer(strategy='median', axis=1).fit_transform(x)
y = np.ravel(Imputer(strategy='median', axis=1).fit_transform(y))
```
Any remaining NaNs (e.g. at the start of the series where ffill/bfill can't reach) are replaced with the column median. Applied to both X and y.

### Step 3 — Standardization (X only)
```python
scaler = StandardScaler()
x = scaler.fit_transform(x)
```
Z-scores each feature column across the full stacked matrix. Required for Lasso since the L1 penalty is not scale-invariant — without this, features on larger scales would be penalized more.

### Step 4 — Train / Predict Split
```python
model_x = x[:-num_secs * (num_holding_days + 1), :]
model.fit(model_x, y)
out[:] = model.predict(x[-num_secs:, :])
```

| Slice | Content |
|---|---|
| `x[:-num_secs*(num_holding_days+1)]` | All rows except the most recent `(holding_days+1)` periods per security — these have no known forward return yet |
| `x[-num_secs:]` | The last `num_secs` rows — today's feature values for each security |

The model is **retrained from scratch on every pipeline execution** (daily). There is no persistence of model weights between days.

### Step 5 — Lasso Regression
```python
model = Lasso()
```
Default sklearn parameters (`alpha=1.0`). Lasso is chosen for its L1 penalty, which zeroes out weak features — effectively performing automatic feature selection across the 9 fundamentals + volume inputs.

---

## 6. Signal Combination and Usage in `rebalance()`

The ML predictions and rule-based combined factor are merged in the `rebalance()` function:

### Step 1 — Normalize ML predictions
```python
todays_predictions = pipeline_data.Model
target_weight_series = todays_predictions.sub(todays_predictions.mean())
target_weight_series = target_weight_series / target_weight_series.abs().sum()
```
Centers the predictions around zero and normalizes so they sum to 1 in absolute terms — making them comparable in scale to the combined factor.

### Step 2 — Z-scale and extract scalar
```python
df['zscore'] = (df.Model - df.Model.mean()) / df.Model.std()
df['zscore'] = df['zscore'] * 1.5
target_weight_series = df["zscore"].iloc[0]
```
Re-z-scores the predictions and scales by 1.5 to amplify the ML signal's magnitude, then extracts a **single scalar** (the score of the first stock in the DataFrame). This scalar is added uniformly to all stocks' rule-based scores.

### Step 3 — Add to combined factor
```python
final_data = pipeline_data.combined_factor.add(target_weight_series, fill_value=0)
```
The scalar ML adjustment shifts the entire combined factor distribution up or down, effectively acting as a market-timing overlay on top of the cross-sectional ranking.

### Step 4 — Optimize
```python
objective = opt.MaximizeAlpha(final_data)
```
The adjusted scores are passed to Quantopian's `MaximizeAlpha` optimizer, which finds weights proportional to expected returns subject to the portfolio constraints.

---

## 7. Portfolio Constraints Applied to Features

The final weights derived from the features above are subject to:

| Constraint | Value | Purpose |
|---|---|---|
| `MaxGrossExposure` | 1.0 | Total long + short ≤ 100% of capital |
| `DollarNeutral` | — | Long market value = Short market value |
| `RiskModelExposure` | version=0 | Neutralizes Quantopian's sector and style risk factor loadings |
| `PositionConcentration` | ±2/600 | No single position > 0.33% of portfolio |

---

## 8. Feature Availability Summary

| Feature | Still Available? | Free Alternative |
|---|---|---|
| EBIT / Enterprise Value | No (Morningstar via Quantopian) | `yfinance` info dict (snapshot only) |
| ROE | No (Morningstar via Quantopian) | `yfinance` info dict (snapshot only) |
| StockTwits bull/bear | No (paid psychsignal feed) | None free; paid: StockTwits API |
| Growth Score | No (Morningstar proprietary) | None direct |
| Value Score | No (Morningstar proprietary) | None direct |
| Style Score | No (Morningstar proprietary) | None direct |
| Open Price / Volume | **Yes** | `yfinance` |
| Equity per share growth | No (Morningstar via Quantopian) | `yfinance` info dict (snapshot only) |
| Sustainable growth rate | No (Morningstar via Quantopian) | Computable from ROE + payout ratio |
| Earning yield | No (Morningstar via Quantopian) | `yfinance` info dict (snapshot only) |
| P/B ratio | No (Morningstar via Quantopian) | `yfinance` info dict (snapshot only) |
| P/E ratio | No (Morningstar via Quantopian) | `yfinance` info dict (snapshot only) |
| ROA | No (Morningstar via Quantopian) | `yfinance` info dict (snapshot only) |
