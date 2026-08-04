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


def train_val_test_split_sequences(
    X: np.ndarray,
    y: np.ndarray,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    """Split sequences into chronological train/val/test sets (no shuffling).

    The remaining fraction after train_ratio and val_ratio becomes the test
    set. Order is preserved throughout so the test set stays untouched,
    future data relative to training/validation.
    """
    train_end = int(len(X) * train_ratio)
    val_end = int(len(X) * (train_ratio + val_ratio))

    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]

    return X_train, y_train, X_val, y_val, X_test, y_test


def prepare_data(
    symbol: str,
    sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
):
    """Run the full preprocessing pipeline for a symbol's Close price."""
    close_prices = load_close_prices(symbol)
    normalized, min_val, max_val = normalize_series(close_prices)
    X, y = create_sequences(normalized, sequence_length)
    X_train, y_train, X_val, y_val, X_test, y_test = train_val_test_split_sequences(
        X, y, train_ratio, val_ratio
    )

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
        "min_val": min_val,
        "max_val": max_val,
    }


if __name__ == "__main__":
    result = prepare_data("MSFT")

    print(f"X_train shape: {result['X_train'].shape}")
    print(f"y_train shape: {result['y_train'].shape}")
    print(f"X_val shape: {result['X_val'].shape}")
    print(f"y_val shape: {result['y_val'].shape}")
    print(f"X_test shape: {result['X_test'].shape}")
    print(f"y_test shape: {result['y_test'].shape}")
