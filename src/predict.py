"""Predict the next trading day's closing price using a trained checkpoint."""

import numpy as np
import torch

from evaluate import inverse_transform, load_checkpoint
from preprocessing import load_close_prices


def get_latest_window(symbol: str, sequence_length: int) -> np.ndarray:
    """Return the most recent `sequence_length` closing prices as a NumPy array."""
    close_prices = load_close_prices(symbol)

    if len(close_prices) < sequence_length:
        raise ValueError(
            f"Not enough data for {symbol}: need at least {sequence_length} rows, "
            f"found {len(close_prices)}."
        )

    return close_prices.tail(sequence_length).to_numpy(dtype=float)


def normalize_window(window: np.ndarray, min_val: float, max_val: float) -> np.ndarray:
    """Scale a raw price window using the checkpoint's saved min/max (not recomputed)."""
    return (window - min_val) / (max_val - min_val)


def predict_next_close(symbol: str = "MSFT") -> dict:
    """Predict the next trading day's closing price for `symbol`.

    Loads the trained checkpoint, feeds the most recent `sequence_length`
    closing prices through the model, and returns the result as a dict.
    Reusable directly by other code (e.g. a future Streamlit dashboard),
    not just as a script.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_checkpoint(symbol, device)

    sequence_length = checkpoint["sequence_length"]
    min_val, max_val = checkpoint["min_val"], checkpoint["max_val"]

    window = get_latest_window(symbol, sequence_length)
    latest_close = float(window[-1])

    normalized_window = normalize_window(window, min_val, max_val)
    input_tensor = (
        torch.tensor(normalized_window, dtype=torch.float32)
        .view(1, sequence_length, 1)
        .to(device)
    )

    with torch.no_grad():
        predicted_norm = model(input_tensor).item()

    predicted_close = float(
        inverse_transform(np.array([predicted_norm]), min_val, max_val)[0]
    )
    percent_change = (predicted_close - latest_close) / latest_close * 100

    return {
        "symbol": symbol.upper(),
        "latest_close": latest_close,
        "predicted_close": predicted_close,
        "percent_change": percent_change,
    }


def print_prediction(result: dict) -> None:
    """Pretty-print a prediction result dict returned by predict_next_close."""
    print(f"Symbol: {result['symbol']}")
    print(f"Latest closing price: {result['latest_close']:.2f}")
    print(f"Predicted next closing price: {result['predicted_close']:.2f}")
    print(f"Expected change: {result['percent_change']:+.2f}%")


if __name__ == "__main__":
    print_prediction(predict_next_close("MSFT"))
