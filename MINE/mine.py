import numpy as np

import torch
from torch import nn


class MINE(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.input_dim = input_dim * 2

        self.MINE = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        combined_input = torch.cat((x, y), dim=1)
        return self.MINE(combined_input)


class EMA:
    def __init__(self, decay: float = 0.01, init_value: float = 1.0):
        self.decay = decay
        self.value = torch.tensor(init_value, dtype=torch.float32)
        self.initialized = False

    def update(self, new_value: torch.Tensor) -> torch.Tensor:
        if not self.initialized:
            self.value = new_value.mean().item()
            self.initialized = True
        else:
            self.value = (1 - self.decay) * self.value + self.decay * new_value.mean().item()
        return torch.tensor(self.value)


def true_mutual_information(rho: float, dim: int) -> float:
    return -(dim / 2) * np.log(1 - rho**2)
