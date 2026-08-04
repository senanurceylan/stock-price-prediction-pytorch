"""Evaluate a trained LSTM checkpoint on the held-out test set."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from dataset import create_dataloaders
from model import LSTMRegressor
from preprocessing import prepare_data


MODELS_DIR = Path(__file__).resolve().parents[1] / "models"


def load_checkpoint(symbol: str, device: torch.device):
    """Load a saved checkpoint and rebuild the matching model architecture."""
    checkpoint_path = MODELS_DIR / f"{symbol.upper()}_lstm.pt"

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"{checkpoint_path} was not found. Run src/train.py first."
        )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model = LSTMRegressor(
        hidden_size=checkpoint["hidden_size"], num_layers=checkpoint["num_layers"]
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, checkpoint


def inverse_transform(values: np.ndarray, min_val: float, max_val: float) -> np.ndarray:
    """Undo min-max normalization to recover real price values."""
    return values * (max_val - min_val) + min_val


def predict(model, loader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """Run inference over a DataLoader, preserving its (unshuffled) order."""
    predictions, actuals = [], []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            output = model(X_batch)
            predictions.append(output.cpu().numpy())
            actuals.append(y_batch.numpy())

    return np.concatenate(predictions).flatten(), np.concatenate(actuals).flatten()


def compute_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    """Compute RMSE, MAE, and MAPE on real (denormalized) price values."""
    errors = predicted - actual
    rmse = np.sqrt(np.mean(errors**2))
    mae = np.mean(np.abs(errors))
    mape = np.mean(np.abs(errors / actual)) * 100

    return {"rmse": rmse, "mae": mae, "mape": mape}


def plot_predictions(
    actual: np.ndarray, predicted: np.ndarray, symbol: str, output_path: Path
) -> None:
    """Save an actual-vs-predicted line chart for the test set."""
    plt.figure(figsize=(12, 6))
    plt.plot(actual, label="Actual")
    plt.plot(predicted, label="Predicted")
    plt.title(f"{symbol.upper()} - Actual vs Predicted (Test Set)")
    plt.xlabel("Time step")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Plot saved to: {output_path}")


def evaluate_model(
    symbol: str = "MSFT", sequence_length: int = 60, batch_size: int = 32
) -> dict:
    """Load the checkpoint, predict on the test set, and report metrics."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_checkpoint(symbol, device)

    data = prepare_data(symbol, sequence_length=sequence_length)
    _, _, test_loader = create_dataloaders(
        data["X_train"],
        data["y_train"],
        data["X_val"],
        data["y_val"],
        data["X_test"],
        data["y_test"],
        batch_size=batch_size,
    )

    predicted_norm, actual_norm = predict(model, test_loader, device)

    min_val, max_val = checkpoint["min_val"], checkpoint["max_val"]
    predicted = inverse_transform(predicted_norm, min_val, max_val)
    actual = inverse_transform(actual_norm, min_val, max_val)

    metrics = compute_metrics(actual, predicted)
    print(f"RMSE: {metrics['rmse']:.4f}")
    print(f"MAE: {metrics['mae']:.4f}")
    print(f"MAPE: {metrics['mape']:.2f}%")

    plot_path = MODELS_DIR / f"{symbol.upper()}_test_predictions.png"
    plot_predictions(actual, predicted, symbol, plot_path)

    return metrics


if __name__ == "__main__":
    evaluate_model()
