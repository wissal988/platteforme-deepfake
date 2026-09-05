# ============================================================
# src/models/lcnn.py
#
# LCNN pour détection de deepfake audio
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F


class MaxFeatureMap(nn.Module):
    """
    Max-Feature-Map activation.

    Input  : (B, 2C, H, W)
    Output : (B, C, H, W)
    """

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        if x.shape[1] % 2 != 0:

            raise ValueError(
                f"MFM nécessite un nombre pair de canaux. "
                f"Reçu : {x.shape}"
            )

        x1, x2 = torch.chunk(
            x,
            2,
            dim=1,
        )

        return torch.max(x1, x2)


class ConvBlock(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        use_batchnorm: bool = True,
    ):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels,
            out_channels * 2,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

        self.bn = (
            nn.BatchNorm2d(out_channels * 2)
            if use_batchnorm
            else nn.Identity()
        )

        self.mfm = MaxFeatureMap()

        self._init_weights()

    def _init_weights(self):

        nn.init.kaiming_normal_(
            self.conv.weight,
            nonlinearity="linear",
        )

        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        x = self.conv(x)

        x = self.bn(x)

        x = self.mfm(x)

        x = torch.nan_to_num(
            x,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        return x


class LCNN(nn.Module):

    def __init__(
        self,
        input_channels: int = 1,
        hidden_dim: int = 64,
        dropout: float = 0.5,
        use_batchnorm: bool = True,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim

        self.features = nn.Sequential(

            ConvBlock(
                input_channels,
                32,
                kernel_size=5,
                padding=2,
                use_batchnorm=use_batchnorm,
            ),

            nn.MaxPool2d(2),

            ConvBlock(
                32,
                48,
                use_batchnorm=use_batchnorm,
            ),

            nn.MaxPool2d(2),

            ConvBlock(
                48,
                64,
                use_batchnorm=use_batchnorm,
            ),

            nn.MaxPool2d(2),

            ConvBlock(
                64,
                128,
                use_batchnorm=use_batchnorm,
            ),

            nn.MaxPool2d(2),

            ConvBlock(
                128,
                64,
                use_batchnorm=use_batchnorm,
            ),
        )

        self.dropout = nn.Dropout(dropout)

        # lazy layers PyTorch
        self.fc = nn.LazyLinear(
            hidden_dim
        )

        self.classifier = nn.Linear(
            hidden_dim,
            1,
        )

        self._init_classifier()

    def _init_classifier(self):

        nn.init.kaiming_normal_(
            self.classifier.weight,
            nonlinearity="linear",
        )

        if self.classifier.bias is not None:
            nn.init.zeros_(
                self.classifier.bias
            )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Input :
            (B, 1, F, T)

        Output :
            logits : (B,)
        """

        x = self.features(x)

        x = torch.nan_to_num(
            x,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        x = x.reshape(
            x.shape[0],
            -1,
        )

        x = self.dropout(x)

        x = F.relu(
            self.fc(x)
        )

        x = self.dropout(x)

        logits = self.classifier(x)

        logits = logits.squeeze(-1)

        logits = torch.nan_to_num(
            logits,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        return logits

    def get_embedding(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        x = self.features(x)

        x = x.reshape(
            x.shape[0],
            -1,
        )

        x = self.dropout(x)

        embedding = F.relu(
            self.fc(x)
        )

        embedding = torch.nan_to_num(
            embedding,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        return embedding