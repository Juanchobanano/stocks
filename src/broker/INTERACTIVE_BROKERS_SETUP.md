# Interactive Brokers Setup

Step-by-step guide to connect this algorithm to Interactive Brokers.

---

## 1. Create an IB Account

1. Go to [interactivebrokers.com](https://www.interactivebrokers.com) and open an account.
2. During signup, enable **Margin** and **Short Selling** — the algorithm holds both long and short positions.
3. Optional but recommended: open a **Paper Trading** account first to validate the integration without risking capital. IB provides one automatically alongside live accounts.

---

## 2. Enable Fractional Shares (optional)

Required if running with `fractional=True` (the default).

1. Log in to [Client Portal](https://www.interactivebrokers.com/portal).
2. Go to **Settings → Account Settings → Trading Permissions**.
3. Enable **Fractional Share Trading** for US stocks.

---

## 3. Install TWS or IB Gateway

You need one of these running locally for the API connection.

| Application | Best for | Download |
|---|---|---|
| **Trader Workstation (TWS)** | Manual + automated trading | [Download TWS](https://www.interactivebrokers.com/en/trading/tws.php) |
| **IB Gateway** | Automated only, lighter weight | [Download Gateway](https://www.interactivebrokers.com/en/trading/ib-gateway.php) |

IB Gateway is recommended for production — it uses less memory and has no GUI overhead.

---

## 4. Enable the API in TWS

1. Open TWS and log in.
2. Go to **Edit → Global Configuration → API → Settings**.
3. Check **Enable ActiveX and Socket Clients**.
4. Set **Socket port** (note the port you choose — you will pass it to `IBBroker`).
5. Check **Allow connections from localhost only** (keep this on for security).
6. Uncheck **Read-Only API** — the algorithm needs to place orders.
7. Click **OK** and restart TWS.

### Ports reference

| Application | Account type | Port |
|---|---|---|
| TWS | Paper trading | **7497** |
| TWS | Live trading | **7496** |
| IB Gateway | Paper trading | **4002** |
| IB Gateway | Live trading | **4001** |

---

## 5. Enable the API in IB Gateway

1. Open IB Gateway and log in.
2. Click **Configure → Settings → API → Settings**.
3. Check **Enable ActiveX and Socket Clients**.
4. Set the port (4002 for paper, 4001 for live).
5. Uncheck **Read-Only API**.
6. Click **OK**.

---

## 6. Install Python dependencies

```bash
pip install -r requirements.txt
```

This installs `ib_insync`, which wraps the official TWS API with a cleaner Python interface.

---

## 7. Validate the connection

Run this snippet to confirm the API is reachable before running any trading logic:

```python
from ib_insync import IB

ib = IB()
ib.connect("127.0.0.1", 7497, clientId=1)  # change port if needed
print(ib.accountValues())
ib.disconnect()
```

If you see a list of account values, the connection works.

---

## 8. Run the algorithm

### Step 1 — Retrain the model (Friday EOD)

```python
from config import Config
from engine import LiveEngine

cfg = Config(polygon_api_key="YOUR_POLYGON_KEY")
engine = LiveEngine(cfg)
engine.retrain()   # saves model to .cache/predictor.joblib
```

### Step 2 — Generate signals and rebalance (Friday open)

```python
from broker import IBBroker

weights = engine.generate_signals()

with IBBroker(
    port=7497,             # paper: 7497 / live: 7496
    order_type="MKT",      # "MKT" or "LMT"
    stagger_seconds=0.5,
    dry_run=True,          # set False to submit real orders
    fractional=True,
) as broker:
    broker.rebalance(weights)
```

Always run with `dry_run=True` first and inspect the logged orders before switching to `dry_run=False`.

---

## 9. Order type guidance

| Scenario | Recommended `order_type` |
|---|---|
| Rebalancing at market open | `"MKT"` — fills guaranteed, price risk is low at open |
| Rebalancing mid-session | `"LMT"` — avoids chasing price, but monitor for non-fills |
| Large positions (>$10k per leg) | `"LMT"` — reduces market impact |

With `"LMT"`, orders are priced at the current bid/ask mid. If the market moves away, the order may not fill. You are responsible for monitoring open orders in TWS.

---

## 10. Scheduling (optional)

To automate the weekly workflow on macOS/Linux, add two cron jobs:

```cron
# Friday 4:30 PM ET — retrain after market close
30 16 * * 5 cd /path/to/stocks && python -c "
from config import Config; from engine import LiveEngine
LiveEngine(Config(polygon_api_key='KEY')).retrain()
"

# Friday 9:35 AM ET — rebalance 5 minutes after open
35 9 * * 5 cd /path/to/stocks && python -c "
from config import Config; from engine import LiveEngine; from broker import IBBroker
cfg = Config(polygon_api_key='KEY')
weights = LiveEngine(cfg).generate_signals()
with IBBroker(port=7496, order_type='MKT', dry_run=False, fractional=True) as b:
    b.rebalance(weights)
"
```

Edit crontab with `crontab -e`. Make sure TWS or IB Gateway is already running before the cron jobs fire — IB does not start automatically.

---

## Troubleshooting

| Error | Cause | Fix |
|---|---|---|
| `Connection refused` | TWS/Gateway not running or API not enabled | Start TWS, check API settings |
| `clientId already in use` | Another session is connected with the same `client_id` | Pass a different `client_id` to `IBBroker` |
| `No price data for [TICKER]` | Ticker not found on SMART routing | Check the symbol is listed on a US exchange |
| Orders not filling (LMT) | Market moved away from mid-price | Switch to `"MKT"` or manually cancel/resubmit in TWS |
| `Read-Only` error | API is in read-only mode | Uncheck Read-Only API in TWS settings |
