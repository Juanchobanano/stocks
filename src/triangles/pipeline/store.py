import logging
import sqlite3
from contextlib import contextmanager
from typing import List
import pandas as pd

log = logging.getLogger(__name__)

_SCHEMA = """
    CREATE TABLE IF NOT EXISTS pattern_orders (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp   TEXT NOT NULL,
        ticker      TEXT NOT NULL,
        pattern     TEXT NOT NULL,
        direction   TEXT NOT NULL,
        entry_price REAL NOT NULL,
        take_profit REAL NOT NULL,
        stop_loss   REAL NOT NULL,
        height      REAL NOT NULL,
        quantity    REAL NOT NULL,
        status      TEXT NOT NULL
    )
"""


class TriangleStore:

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(_SCHEMA)

    def has_open_order(self, ticker: str) -> bool:
        """Return True if a 'placed' order already exists for this ticker."""
        query = "SELECT COUNT(*) FROM pattern_orders WHERE ticker = ? AND status = 'placed'"
        with self._connect() as conn:
            count = conn.execute(query, [ticker]).fetchone()[0]
        return count > 0

    def save(
        self,
        ticker: str,
        pattern: str,
        direction: str,
        entry_price: float,
        take_profit: float,
        stop_loss: float,
        height: float,
        quantity: float,
        status: str = "placed",
    ) -> None:
        sql = """
            INSERT INTO pattern_orders
                (timestamp, ticker, pattern, direction, entry_price,
                 take_profit, stop_loss, height, quantity, status)
            VALUES (datetime('now'), ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        with self._connect() as conn:
            conn.execute(
                sql,
                [ticker, pattern, direction, entry_price,
                 take_profit, stop_loss, height, quantity, status],
            )
        log.info(f"Stored order: {direction} {ticker} entry={entry_price:.2f} "
                 f"tp={take_profit:.2f} sl={stop_loss:.2f}")

    def get_all(self) -> pd.DataFrame:
        """Return all rows as a DataFrame for inspection."""
        with self._connect() as conn:
            return pd.read_sql_query("SELECT * FROM pattern_orders", conn)

    def cancel_ticker(self, ticker: str) -> None:
        """Mark all placed orders for a ticker as cancelled."""
        sql = "UPDATE pattern_orders SET status = 'cancelled' WHERE ticker = ? AND status = 'placed'"
        with self._connect() as conn:
            conn.execute(sql, [ticker])

    def get_open_orders(self) -> List[dict]:
        """Return all rows with status='placed'."""
        with self._connect() as conn:
            cur = conn.execute("SELECT * FROM pattern_orders WHERE status = 'placed'")
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]
