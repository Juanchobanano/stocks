from typing import List

import pandas as pd


def get_sp500_tickers() -> List[str]:
    """Fetch current S&P 500 tickers from Wikipedia."""
    tables = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
    tickers = tables[0]["Symbol"].tolist()
    return [t.replace(".", "-") for t in tickers]  # BRK.B → BRK-B for yfinance
