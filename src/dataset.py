"""PyTorch Dataset and DataLoader utilities for stock price sequences."""

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class StockSequenceDataset(Dataset):
    """Wraps sliding-window X/y NumPy arrays as tensors shaped for an LSTM."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        # LSTM input shape: (samples, sequence_length, input_size=1)
        self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, index: int):
        return self.X[index], self.y[index]


def create_dataloaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    batch_size: int = 32,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build train/val/test DataLoaders. Only the train loader is shuffled."""
    train_dataset = StockSequenceDataset(X_train, y_train)
    val_dataset = StockSequenceDataset(X_val, y_val)
    test_dataset = StockSequenceDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
