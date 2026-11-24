"""PyTorch architectures for Gas Sensor Array Drift classification."""

from __future__ import annotations

import torch
import torch.nn as nn


class GasMLP(nn.Module):
    """Multi-Layer Perceptron for tabular or UMAP-reduced sensor features."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int = 6,
        hidden_dims: tuple[int, ...] = (256, 128, 64),
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, h_dim),
                    nn.BatchNorm1d(h_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Gas1DCNN(nn.Module):
    """1D Convolutional Neural Network treating the 16 sensors as sequence steps.

    Reshapes flat (B, 128) input into (B, 16, 8) and permutes to (B, 8, 16)
    so each of the 8 extracted features per sensor acts as a channel across
    the 16 chemical sensors.
    """

    def __init__(
        self,
        num_sensors: int = 16,
        features_per_sensor: int = 8,
        num_classes: int = 6,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.num_sensors = num_sensors
        self.features_per_sensor = features_per_sensor

        self.conv_blocks = nn.Sequential(
            nn.Conv1d(features_per_sensor, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), self.num_sensors, self.features_per_sensor)
        x = x.permute(0, 2, 1)  # (B, 8, 16)
        feats = self.conv_blocks(x)
        pooled = torch.cat(
            [self.avg_pool(feats).squeeze(-1), self.max_pool(feats).squeeze(-1)],
            dim=1,
        )
        return self.head(pooled)
