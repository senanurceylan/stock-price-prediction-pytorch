"""Visualization helpers for stock market data."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def plot_close_price(symbol: str) -> None:
    """Plot the closing-price history from a locally saved CSV file."""
    csv_path = DATA_DIR / f"{symbol.upper()}.csv"

    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} was not found. Run src/data_loader.py first."
        )

    data = pd.read_csv(csv_path, index_col=0, parse_dates=True)

    if "Close" not in data.columns:
        raise ValueError("The CSV file does not contain a Close column.")

    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data["Close"])
    plt.title(f"{symbol.upper()} Closing Price")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_close_price("MSFT")
