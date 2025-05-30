"""
Data ingestion: load price & fundamental data.
"""

import pandas as pd
from typing import List
import yaml

def load_price_data(tickers: List[str], start_date: str, end_date: str, path: str = "data/processed/") -> pd.DataFrame:
    """
    Load adjusted close price data for given tickers and date range.
    Expects CSVs named `<TICKER>.csv` in `path`.
    Returns a DataFrame indexed by date, columns=tickers.
    """
    prices = {}
    for t in tickers:
        df = pd.read_csv(f"{path}{t}.csv", parse_dates=["Date"], index_col="Date")
        df = df.loc[start_date:end_date]
        prices[t] = df["Adj Close"]
    return pd.DataFrame(prices)

def load_fundamental_data(path: str = "data/processed/fundamentals.csv") -> pd.DataFrame:
    """
    Load a single CSV of fundamental metrics (e.g., earnings, book value).
    Returns a DataFrame indexed by date and ticker.
    """
    df = pd.read_csv(path, parse_dates=["Date"])
    df.set_index(["Date", "Ticker"], inplace=True)
    return df