import logging
import time
from typing import List, Optional

import pandas as pd
from ib_insync import IB, LimitOrder, MarketOrder, ScannerSubscription, Stock, StopOrder

log = logging.getLogger(__name__)


class IBBroker:
    """
    Rebalances an IB account to target portfolio weights.

    Parameters
    ----------
    host : str
        TWS / IB Gateway host. Usually 127.0.0.1.
    port : int
        TWS paper=7497, TWS live=7496, Gateway paper=4002, Gateway live=4001.
    client_id : int
        Unique client ID for this connection. Use different IDs for concurrent sessions.
    order_type : str
        "MKT"  — market orders; fills are guaranteed but price is not.
        "LMT"  — limit orders at the current mid-price; better price, risk of non-fill.
    stagger_seconds : float
        Seconds to wait between consecutive order submissions.
        0 = submit all at once. Use 0.5–2.0 to reduce burst impact.
    dry_run : bool
        When True, logs what would be traded without submitting any orders.
    fractional : bool
        When True, order quantities are sent as decimals (requires fractional
        share trading to be enabled in the IB account settings).
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 7497,
        client_id: int = 1,
        order_type: str = "MKT",
        stagger_seconds: float = 0.0,
        dry_run: bool = False,
        fractional: bool = True,
    ):
        if order_type.upper() not in ("MKT", "LMT"):
            raise ValueError(f"order_type must be 'MKT' or 'LMT', got '{order_type}'")

        self.host = host
        self.port = port
        self.client_id = client_id
        self.order_type = order_type.upper()
        self.stagger_seconds = stagger_seconds
        self.dry_run = dry_run
        self.fractional = fractional
        self._ib: Optional[IB] = None

    # ── Connection management ─────────────────────────────────────────────────

    def connect(self) -> None:
        self._ib = IB()
        self._ib.connect(self.host, self.port, clientId=self.client_id)
        log.info(f"Connected to IB at {self.host}:{self.port} (client_id={self.client_id})")

    def disconnect(self) -> None:
        if self._ib and self._ib.isConnected():
            self._ib.disconnect()
            log.info("Disconnected from IB.")

    def __enter__(self) -> "IBBroker":
        self.connect()
        return self

    def __exit__(self, *_) -> None:
        self.disconnect()

    def _get_equity(self) -> float:
        """Return net liquidation value in USD."""
        vals = {
            v.tag: v.value
            for v in self._ib.accountValues()
            if v.currency == "USD"
        }
        return float(vals["NetLiquidation"])

    def _get_positions(self) -> pd.Series:
        """Return current share counts indexed by ticker symbol."""
        return pd.Series(
            {pos.contract.symbol: pos.position for pos in self._ib.positions()},
            dtype=float,
        )

    def _get_prices(self, tickers: list) -> dict[str, float]:
        """
        Batch-request snapshot prices for all tickers at once.
        Uses mid-price for LMT orders, last/close for MKT.
        """
        contracts = [Stock(t, "SMART", "USD") for t in tickers]
        self._ib.qualifyContracts(*contracts)
        snapshots = self._ib.reqTickers(*contracts)

        prices: dict[str, float] = {}
        for snap in snapshots:
            symbol = snap.contract.symbol
            if self.order_type == "LMT" and snap.bid > 0 and snap.ask > 0:
                prices[symbol] = (snap.bid + snap.ask) / 2.0
            else:
                price = snap.last if snap.last and snap.last > 0 else snap.close
                if price and price > 0:
                    prices[symbol] = price

        missing = set(tickers) - set(prices)
        if missing:
            log.warning(f"No price data for: {sorted(missing)} — these will be skipped.")
        return prices

    # ── Order construction ────────────────────────────────────────────────────

    def _make_order(self, action: str, qty: float, price: float):
        if self.order_type == "LMT":
            return LimitOrder(action, qty, round(price, 2))
        return MarketOrder(action, qty)

    def get_scanner_symbols(
        self,
        scan_code: str = "MOST_ACTIVE",
        n: int = 100,
    ) -> List[str]:
        """Return up to n ticker symbols from an IB market scanner."""
        sub = ScannerSubscription(
            instrument="STK",
            locationCode="STK.US.MAJOR",
            scanCode=scan_code,
            numberOfRows=n,
        )
        results = self._ib.reqScannerData(sub)
        symbols = [item.contractDetails.contract.symbol for item in results]
        log.info(f"Scanner '{scan_code}' returned {len(symbols)} symbols.")
        return symbols

    def get_historical_bars(
        self,
        ticker: str,
        duration: str = "4 M",
        bar_size: str = "1 day",
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV bars for a single ticker.

        Returns a DataFrame with columns Open, High, Low, Close, Volume
        indexed by date. Returns an empty DataFrame if data is unavailable.
        """
        contract = Stock(ticker, "SMART", "USD")
        self._ib.qualifyContracts(contract)
        bars = self._ib.reqHistoricalData(
            contract,
            endDateTime="",
            durationStr=duration,
            barSizeSetting=bar_size,
            whatToShow="TRADES",
            useRTH=True,
            formatDate=1,
        )
        if not bars:
            log.warning(f"{ticker}: no historical data returned.")
            return pd.DataFrame()

        df = pd.DataFrame([
            {
                "date":   b.date,
                "Open":   b.open,
                "High":   b.high,
                "Low":    b.low,
                "Close":  b.close,
                "Volume": b.volume,
            }
            for b in bars
        ])
        df["date"] = pd.to_datetime(df["date"])
        return df.set_index("date")

    def place_bracket_order(
        self,
        ticker: str,
        action: str,
        entry_price: float,
        take_profit: float,
        stop_loss: float,
        quantity: float,
    ) -> None:
        """
        Place a bracket order: limit entry with attached TP limit and SL stop.

        The three child orders are linked via IB's OCA mechanism so that
        when one child fills the other is automatically cancelled.

        action : 'BUY' for long, 'SELL' for short.
        """
        action = action.upper()
        if action not in ("BUY", "SELL"):
            raise ValueError(f"action must be 'BUY' or 'SELL', got '{action}'")

        close_action = "SELL" if action == "BUY" else "BUY"

        prefix = "[DRY RUN] " if self.dry_run else ""
        log.info(
            f"{prefix}Bracket {action} {quantity} {ticker} "
            f"entry={entry_price:.2f} tp={take_profit:.2f} sl={stop_loss:.2f}"
        )

        if self.dry_run:
            return

        contract = Stock(ticker, "SMART", "USD")
        self._ib.qualifyContracts(contract)

        parent = LimitOrder(action, quantity, round(entry_price, 2))
        parent.transmit = False

        tp_order = LimitOrder(close_action, quantity, round(take_profit, 2))
        tp_order.parentId = parent.orderId
        tp_order.transmit = False

        sl_order = StopOrder(close_action, quantity, round(stop_loss, 2))
        sl_order.parentId = parent.orderId
        sl_order.transmit = True  # transmit all three together

        for order in (parent, tp_order, sl_order):
            self._ib.placeOrder(contract, order)

        log.debug(f"Bracket order submitted for {ticker}.")

    def rebalance(self, weights: pd.Series) -> None:
        """
        Rebalance the IB account to match target portfolio weights.

        Parameters
        ----------
        weights : pd.Series
            Signed target weights indexed by ticker (e.g. from LiveEngine).
            Positive = long, negative = short.
            The sum of absolute weights should be ≤ max_gross_leverage.
        """
        equity = self._get_equity()
        current = self._get_positions()
        log.info(f"Account equity: ${equity:,.2f}")

        # Include any open positions not in the new target (they need to be closed)
        all_tickers = weights.index.union(current.index).tolist()
        target_weights = weights.reindex(all_tickers).fillna(0.0)

        prices = self._get_prices(all_tickers)

        # Compute delta shares for each ticker
        orders: list[tuple[str, float, float]] = []  # (ticker, delta_shares, price)
        for ticker in all_tickers:
            price = prices.get(ticker)
            if price is None:
                continue

            target_shares = (target_weights[ticker] * equity) / price
            current_shares = current.get(ticker, 0.0)
            delta = target_shares - current_shares

            if not self.fractional:
                delta = round(delta)

            min_delta = 0.01 if self.fractional else 1.0
            if abs(delta) < min_delta:
                continue

            orders.append((ticker, delta, price))

        if not orders:
            log.info("Rebalance: no changes required.")
            return

        # Submit sells before buys to free up margin
        orders.sort(key=lambda x: x[1])  # ascending: sells (negative) first

        prefix = "[DRY RUN] " if self.dry_run else ""
        for ticker, delta, price in orders:
            action = "BUY" if delta > 0 else "SELL"
            qty = abs(delta)
            price_str = f"~{price:.2f}" if self.order_type == "LMT" else f"{price:.2f}"
            log.info(f"{prefix}{action} {qty:.4f} {ticker} @ {price_str}")

            if not self.dry_run:
                contract = Stock(ticker, "SMART", "USD")
                self._ib.qualifyContracts(contract)
                order = self._make_order(action, qty, price)
                trade = self._ib.placeOrder(contract, order)
                log.debug(f"Trade object: {trade}")

            if self.stagger_seconds > 0:
                time.sleep(self.stagger_seconds)

        n_buys = sum(1 for _, d, _ in orders if d > 0)
        n_sells = sum(1 for _, d, _ in orders if d < 0)
        log.info(
            f"Rebalance {'simulated' if self.dry_run else 'submitted'}: "
            f"{n_buys} buys, {n_sells} sells."
        )
