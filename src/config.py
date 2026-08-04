"""Central configuration for training the LSTM stock price model."""

from dataclasses import dataclass


@dataclass
class Config:
    symbol: str = "MSFT"
    sequence_length: int = 60
    batch_size: int = 32
    hidden_size: int = 64
    num_layers: int = 2
    learning_rate: float = 1e-3
    epochs: int = 50
    patience: int = 7
    random_seed: int = 42
