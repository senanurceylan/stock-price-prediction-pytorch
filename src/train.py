"""Training loop for the LSTM stock price model."""

import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from config import Config
from dataset import create_dataloaders
from model import LSTMRegressor
from preprocessing import prepare_data


MODELS_DIR = Path(__file__).resolve().parents[1] / "models"


def set_seed(seed: int) -> None:
    """Seed all RNGs involved so a training run is reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_epoch(model, loader, criterion, device, optimizer=None) -> float:
    """Run one pass over `loader`. Trains if `optimizer` is given, else only evaluates."""
    is_training = optimizer is not None
    model.train() if is_training else model.eval()

    total_loss = 0.0
    total_samples = 0

    with torch.set_grad_enabled(is_training):
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)

            if is_training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * X_batch.size(0)
            total_samples += X_batch.size(0)

    return total_loss / total_samples


def train_model(config: Config) -> Path:
    """Train the LSTM model with early stopping and save the best checkpoint."""
    set_seed(config.random_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = prepare_data(config.symbol, sequence_length=config.sequence_length)
    train_loader, val_loader, _ = create_dataloaders(
        data["X_train"],
        data["y_train"],
        data["X_val"],
        data["y_val"],
        data["X_test"],
        data["y_test"],
        batch_size=config.batch_size,
    )

    model = LSTMRegressor(
        hidden_size=config.hidden_size, num_layers=config.num_layers
    ).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = MODELS_DIR / f"{config.symbol.upper()}_lstm.pt"

    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, config.epochs + 1):
        train_loss = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss = run_epoch(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch}/{config.epochs} "
            f"- train_loss: {train_loss:.6f} - val_loss: {val_loss:.6f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "min_val": data["min_val"],
                    "max_val": data["max_val"],
                    "sequence_length": config.sequence_length,
                    "hidden_size": config.hidden_size,
                    "num_layers": config.num_layers,
                    "symbol": config.symbol,
                    "best_validation_loss": best_val_loss,
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.patience:
                print(
                    f"Early stopping at epoch {epoch} "
                    f"(no improvement for {config.patience} epochs)."
                )
                break

    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Checkpoint saved to: {checkpoint_path}")
    return checkpoint_path


if __name__ == "__main__":
    train_model(Config())
