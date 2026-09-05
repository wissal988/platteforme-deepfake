import torch
import torch.nn as nn
import torch.nn.functional as F


class RawCNNBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = F.leaky_relu(x, negative_slope=0.2)
        x = self.pool(x)
        return torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


class RawCNN(nn.Module):
    """
    Lightweight raw-waveform CNN for complementary anti-spoofing.

    Input:
        (B, 1, T)
    Output:
        logits (B,)
    """

    def __init__(
        self,
        input_channels: int = 1,
        hidden_dim: int = 128,
        dropout: float = 0.5,
    ):
        super().__init__()

        self.blocks = nn.Sequential(
            RawCNNBlock(input_channels, 32, kernel_size=31, stride=2),
            RawCNNBlock(32, 64, kernel_size=15, stride=2),
            RawCNNBlock(64, 96, kernel_size=9, stride=2),
            RawCNNBlock(96, 128, kernel_size=7, stride=2),
            RawCNNBlock(128, 160, kernel_size=5, stride=2),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(160, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(1)
        if x.ndim != 3:
            raise ValueError(f"RawCNN attend (B, 1, T), recu {tuple(x.shape)}")

        x = self.blocks(x)
        x = self.pool(x).squeeze(-1)
        x = self.dropout(x)
        x = F.relu(self.fc(x))
        x = self.dropout(x)
        logits = self.classifier(x).squeeze(-1)
        return torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)
