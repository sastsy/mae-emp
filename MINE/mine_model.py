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

    def update(self, new_value: torch.Tensor) -> torch.Tensor:
        self.value = self.decay * new_value.detach() + (1 - self.decay) * self.value
        return self.value


def true_mutual_information(rho: float, dim: int) -> float:
    return -(dim / 2) * np.log(1 - rho**2)
