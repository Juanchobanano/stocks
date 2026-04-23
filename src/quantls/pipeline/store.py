import logging
import sqlite3
from contextlib import contextmanager
from typing import List

import pandas as pd

log = logging.getLogger(__name__)

_SCHEMA = {
    "prices": """
        CREATE TABLE IF NOT EXISTS prices (
            date   TEXT NOT NULL,
            ticker TEXT NOT NULL,
            open   REAL,
            close  REAL,
            volume REAL,
            PRIMARY KEY (date, ticker)
        )""",
    "fundamentals": """
        CREATE TABLE IF NOT EXISTS fundamentals (
            date                    TEXT NOT NULL,
            ticker                  TEXT NOT NULL,
            ebit                    REAL,
            net_income              REAL,
            equity                  REAL,
            assets                  REAL,
            eps                     REAL,
            equity_per_share        REAL,
            shares                  REAL,
            long_term_debt          REAL,
            cash                    REAL,
            revenue                 REAL,
            revenue_growth          REAL,
            equity_per_share_growth REAL,
            sustainable_growth_rate REAL,
            PRIMARY KEY (date, ticker)
        )""",
    "scores": """
        CREATE TABLE IF NOT EXISTS scores (
            date          TEXT NOT NULL,
            ticker        TEXT NOT NULL,
            roe           REAL,
            roa           REAL,
            pe_ratio      REAL,
            pb_ratio      REAL,
            earning_yield REAL,
            ebit_to_ev    REAL,
            growth_score  REAL,
            value_score   REAL,
            style_score   REAL,
            PRIMARY KEY (date, ticker)
        )""",
    "sentiment": """
        CREATE TABLE IF NOT EXISTS sentiment (
            date               TEXT NOT NULL,
            ticker             TEXT NOT NULL,
            bull_minus_bear    REAL,
            bull_minus_bear_3d REAL,
            PRIMARY KEY (date, ticker)
        )""",
}


class FeatureStore:
    """
    Unified read/write interface to the SQLite feature database.

    All DataFrames saved via `save()` must have `date` and `ticker` as either
    columns or a MultiIndex (date, ticker). Date values must be strings in
    YYYY-MM-DD format or pandas Timestamps.
    """

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
            for ddl in _SCHEMA.values():
                conn.execute(ddl)

    def save(self, table: str, df: pd.DataFrame) -> None:
        """Upsert all rows in `df` into `table`."""
        if df.empty:
            return
        flat = df.reset_index() if df.index.names[0] is not None else df
        # Ensure date column is a string
        if "date" in flat.columns:
            flat["date"] = pd.to_datetime(flat["date"]).dt.strftime("%Y-%m-%d")

        cols = list(flat.columns)
        col_str = ", ".join(f'"{c}"' for c in cols)
        placeholders = ", ".join("?" * len(cols))
        sql = f'INSERT OR REPLACE INTO "{table}" ({col_str}) VALUES ({placeholders})'
        rows = [tuple(r) for r in flat.itertuples(index=False, name=None)]

        with self._connect() as conn:
            conn.executemany(sql, rows)
        log.info(f"Saved {len(rows)} rows → '{table}'")

    def get(self, date: pd.Timestamp, tickers: List[str]) -> pd.DataFrame:
        """
        Return all features for a single date as a DataFrame indexed by ticker.
        Columns cover all four tables; missing data appears as NaN.
        """
        date_str = date.strftime("%Y-%m-%d")
        placeholders = ",".join("?" * len(tickers))
        frames = []

        with self._connect() as conn:
            for table in _SCHEMA:
                query = f"""
                    SELECT * FROM "{table}"
                    WHERE date = ? AND ticker IN ({placeholders})
                """
                df = pd.read_sql_query(query, conn, params=[date_str] + list(tickers))
                if not df.empty:
                    frames.append(df.drop(columns=["date"]).set_index("ticker"))

        if not frames:
            return pd.DataFrame(index=tickers)
        return pd.concat(frames, axis=1).reindex(tickers)

    def get_range(
        self,
        start: str,
        end: str,
        tickers: List[str],
        table: str,
    ) -> pd.DataFrame:
        """
        Return all rows from `table` for a date range.
        Returns a DataFrame indexed by (date, ticker).
        """
        placeholders = ",".join("?" * len(tickers))
        query = f"""
            SELECT * FROM "{table}"
            WHERE date >= ? AND date <= ? AND ticker IN ({placeholders})
        """
        with self._connect() as conn:
            df = pd.read_sql_query(
                query, conn,
                params=[start, end] + list(tickers),
                parse_dates=["date"],
            )
        if df.empty:
            return pd.DataFrame()
        return df.set_index(["date", "ticker"])

    def has_data(self, table: str, ticker: str, start: str, end: str) -> bool:
        """Return True if the table already contains data for ticker in [start, end]."""
        query = f"""
            SELECT COUNT(*) FROM "{table}"
            WHERE ticker = ? AND date >= ? AND date <= ?
        """
        with self._connect() as conn:
            count = conn.execute(query, [ticker, start, end]).fetchone()[0]
        return count > 0
