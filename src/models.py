"""
The neural network: a Long Short-Term Memory (LSTM) network followed by a
small fully connected "head".

    input window (batch, 24 hours, 8 features)
        -> LSTM layer 1 -> LSTM layer 2      reads the 24 hours one by one
        -> hidden state after the LAST hour  (batch, hidden_size)
        -> Linear -> ReLU -> Dropout -> Linear
        -> one number per forecast horizon   (batch, 6)
"""
from __future__ import annotations

import torch
from torch import nn


class DstLSTM(nn.Module):
    def __init__(self, n_features: int, n_outputs: int, hidden_size: int = 64,
                 num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,                       # tensors are (batch, time, features)
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, n_outputs),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # out: hidden state at EVERY time step, shape (batch, time, hidden_size)
        out, (h_n, c_n) = self.lstm(x)
        last = out[:, -1, :]          # we only need the summary after the last hour
        return self.head(last)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
