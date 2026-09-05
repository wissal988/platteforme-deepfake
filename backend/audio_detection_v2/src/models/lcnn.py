# ============================================================
# src/models/lcnn.py
#
# LCNN pour détection de deepfake audio
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation channel attention.

    Recalibre les canaux en fonction de leur importance globale.
    Ref: Hu et al., "Squeeze-and-Excitation Networks", CVPR 2018.

    Input / Output : (B, C, H, W) — inchangé spatiallement.
    """

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        reduced = max(channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, reduced),
            nn.ReLU(),
            nn.Linear(reduced, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.pool(x).view(x.shape[0], -1)
        scale = self.fc(scale).view(x.shape[0], x.shape[1], 1, 1)
        return x * scale


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
        norm_type: str = "instance",
    ):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels,
            out_channels * 2,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

        # InstanceNorm2d : normalise par (batch_item, channel) indépendamment.
        # Pas de running_mean/var → statistiques recalculées à chaque sample.
        # Résultat : domain-agnostic, généralise mieux cross-dataset que BatchNorm.
        # Ref : Ulyanov et al., "Instance Normalization", arXiv 2016.
        #
        # BatchNorm2d : stocke running_mean/var sur le dataset d'entraînement.
        # Sur un dataset non vu → les stats ne matchent plus → dégradation silencieuse.
        if norm_type == "instance":
            self.norm = nn.InstanceNorm2d(out_channels * 2, affine=True)
        elif norm_type == "batch":
            self.norm = nn.BatchNorm2d(out_channels * 2)
        else:
            self.norm = nn.Identity()

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

        x = self.norm(x)

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
        norm_type: str = "instance",
    ):
        """
        Args:
            norm_type : "instance" (défaut, recommandé pour généralisation cross-dataset)
                        "batch"    (plus stable mais overfit la distribution d'entraînement)
                        "none"     (aucune normalisation)
        """
        super().__init__()

        self.hidden_dim = hidden_dim

        self.features = nn.Sequential(

            ConvBlock(
                input_channels,
                32,
                kernel_size=5,
                padding=2,
                norm_type=norm_type,
            ),

            nn.MaxPool2d(2),

            ConvBlock(
                32,
                48,
                norm_type=norm_type,
            ),

            nn.MaxPool2d(2),

            ConvBlock(
                48,
                64,
                norm_type=norm_type,
            ),

            nn.MaxPool2d(2),

            ConvBlock(
                64,
                96,
                norm_type=norm_type,
            ),

            nn.MaxPool2d(2),

            ConvBlock(
                96,
                128,
                norm_type=norm_type,
            ),

            ConvBlock(
                128,
                64,
                norm_type=norm_type,
            ),

            # recalibrage des canaux par attention globale
            SEBlock(64),
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