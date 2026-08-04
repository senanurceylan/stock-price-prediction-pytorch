"""A simple LSTM regression model for next-day price prediction."""

import torch
import torch.nn as nn


class LSTMRegressor(nn.Module):
    """LSTM followed by a linear layer that outputs a single predicted value."""

    def __init__(self, input_size: int = 1, hidden_size: int = 64, num_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, sequence_length, input_size)
        lstm_out, _ = self.lstm(x)
        last_step = lstm_out[:, -1, :]  # hidden state from the final time step
        return self.fc(last_step)
