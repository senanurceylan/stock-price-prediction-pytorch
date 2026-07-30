"""Preprocessing utilities for turning raw stock data into model-ready sequences."""

from pathlib import Path

import numpy as np
import pandas as pd


DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DEFAULT_SEQUENCE_LENGTH = 60


def load_close_prices(symbol: str) -> pd.Series:
    """Load the Close price column from a locally saved CSV file."""
    csv_path = DATA_DIR / f"{symbol.upper()}.csv"

    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} was not found. Run src/data_loader.py first."
        )

    data = pd.read_csv(csv_path, index_col=0, parse_dates=True)

    if "Close" not in data.columns:
        raise ValueError("The CSV file does not contain a Close column.")

    return data["Close"].astype(float)


def normalize_series(series: pd.Series) -> tuple[np.ndarray, float, float]:
    """Scale values to the [0, 1] range using min-max normalization."""
    values = series.to_numpy(dtype=float)
    min_val = values.min()
    max_val = values.max()

    normalized = (values - min_val) / (max_val - min_val)
    return normalized, min_val, max_val


def create_sequences(
    values: np.ndarray, sequence_length: int = DEFAULT_SEQUENCE_LENGTH
) -> tuple[np.ndarray, np.ndarray]:
    """Build sliding-window sequences (X) and their next-day targets (y)."""
    X, y = [], []

    for i in range(len(values) - sequence_length):
        X.append(values[i : i + sequence_length])
        y.append(values[i + sequence_length])

    return np.array(X), np.array(y)


def train_test_split_sequences(
    X: np.ndarray, y: np.ndarray, train_ratio: float = 0.8
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split sequences into train/test sets by index order (no shuffling)."""
    split_index = int(len(X) * train_ratio)

    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    return X_train, y_train, X_test, y_test


def prepare_data(
    symbol: str,
    sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
    train_ratio: float = 0.8,
):
    """Run the full preprocessing pipeline for a symbol's Close price."""
    close_prices = load_close_prices(symbol)
    normalized, min_val, max_val = normalize_series(close_prices)
    X, y = create_sequences(normalized, sequence_length)
    X_train, y_train, X_test, y_test = train_test_split_sequences(X, y, train_ratio)

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "min_val": min_val,
        "max_val": max_val,
    }


if __name__ == "__main__":
    result = prepare_data("MSFT")

    print(f"X_train shape: {result['X_train'].shape}")
    print(f"y_train shape: {result['y_train'].shape}")
    print(f"X_test shape: {result['X_test'].shape}")
    print(f"y_test shape: {result['y_test'].shape}")
