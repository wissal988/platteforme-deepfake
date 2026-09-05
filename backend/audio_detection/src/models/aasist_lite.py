import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualConvBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 5, dilation: int = 1):
        super().__init__()
        padding = (kernel_size // 2) * dilation
        self.conv1 = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.bn2 = nn.BatchNorm1d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.leaky_relu(self.bn1(self.conv1(x)), negative_slope=0.2)
        x = self.bn2(self.conv2(x))
        x = F.leaky_relu(x + residual, negative_slope=0.2)
        return torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


class AASISTLite(nn.Module):
    """
    AASIST-inspired raw-waveform baseline.

    This is intentionally a local, lightweight implementation rather than a
    copied full AASIST release: convolutional raw encoder, temporal attention,
    and attentive statistics pooling.
    """

    def __init__(
        self,
        input_channels: int = 1,
        channels: int = 128,
        hidden_dim: int = 160,
        attention_heads: int = 4,
        max_attn_frames: int = 256,
        dropout: float = 0.4,
    ):
        super().__init__()
        if channels % attention_heads != 0:
            raise ValueError("channels doit etre divisible par attention_heads")

        self.max_attn_frames = max_attn_frames
        self.stem = nn.Sequential(
            nn.Conv1d(input_channels, channels, kernel_size=31, stride=2, padding=15, bias=False),
            nn.BatchNorm1d(channels),
            nn.LeakyReLU(0.2),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )
        self.encoder = nn.Sequential(
            ResidualConvBlock(channels, kernel_size=7, dilation=1),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            ResidualConvBlock(channels, kernel_size=5, dilation=2),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            ResidualConvBlock(channels, kernel_size=3, dilation=4),
        )
        self.self_attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=attention_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(channels)
        self.pool_attn = nn.Sequential(
            nn.Linear(channels, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(channels * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(1)
        if x.ndim != 3:
            raise ValueError(f"AASISTLite attend (B, 1, T), recu {tuple(x.shape)}")

        x = self.stem(x)
        x = self.encoder(x)

        if x.shape[-1] > self.max_attn_frames:
            x = F.adaptive_avg_pool1d(x, self.max_attn_frames)

        x = x.transpose(1, 2)

        attn_out, _ = self.self_attn(x, x, x, need_weights=False)
        x = self.attn_norm(x + attn_out)

        weights = torch.softmax(self.pool_attn(x).squeeze(-1), dim=1).unsqueeze(-1)
        mean = torch.sum(weights * x, dim=1)
        var = torch.sum(weights * (x - mean.unsqueeze(1)).pow(2), dim=1)
        pooled = torch.cat([mean, torch.sqrt(var.clamp_min(1e-8))], dim=1)

        logits = self.classifier(pooled).squeeze(-1)
        return torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)
