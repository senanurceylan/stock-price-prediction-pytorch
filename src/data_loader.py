"""Utilities for downloading historical stock market data."""

from pathlib import Path

import yfinance as yf


DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def download_stock_data(symbol: str, start_date: str, end_date: str):
    """Download historical stock data for the given symbol and date range."""
    data = yf.download(symbol, start=start_date, end=end_date, auto_adjust=False)

    if data.empty:
        raise ValueError(f"No data returned for symbol: {symbol}")

    if data.columns.nlevels > 1:
        data.columns = data.columns.get_level_values(0)

    return data


def save_stock_data(data, symbol: str) -> Path:
    """Save stock data as a CSV file inside the data directory."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = DATA_DIR / f"{symbol.upper()}.csv"
    data.to_csv(output_path)
    return output_path


if __name__ == "__main__":
    stock_symbol = "MSFT"
    stock_data = download_stock_data(stock_symbol, "2015-01-01", "2026-01-01")
    output_file = save_stock_data(stock_data, stock_symbol)

    print(stock_data.head())
    print(f"\nTotal rows: {len(stock_data)}")
    print(f"Saved to: {output_file}")
