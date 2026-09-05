import torch
import torch.nn as nn


class Wav2Vec2SSLClassifier(nn.Module):
    """
    SSL classifier using torchaudio wav2vec2 features.

    By default the pretrained SSL encoder is frozen and only the small pooling
    classifier is trained. This is much more realistic on CPU.
    """

    def __init__(
        self,
        bundle_name: str = "WAV2VEC2_BASE",
        hidden_dim: int = 256,
        dropout: float = 0.4,
        freeze_ssl: bool = True,
    ):
        super().__init__()
        try:
            import torchaudio
        except ImportError as exc:
            raise ImportError("torchaudio est requis pour wav2vec2_ssl") from exc

        if not hasattr(torchaudio, "pipelines") or not hasattr(torchaudio.pipelines, bundle_name):
            raise ValueError(f"Bundle torchaudio inconnu: {bundle_name}")

        self.bundle_name = bundle_name
        self.freeze_ssl = freeze_ssl
        bundle = getattr(torchaudio.pipelines, bundle_name)
        self.ssl = bundle.get_model()

        if freeze_ssl:
            self.ssl.eval()
            for param in self.ssl.parameters():
                param.requires_grad = False

        # Wav2Vec2 base/large expose encoder_embed_dim on recent torchaudio.
        feature_dim = int(getattr(bundle._params, "encoder_embed_dim", 768))
        self.classifier = nn.Sequential(
            nn.LayerNorm(feature_dim * 2),
            nn.Linear(feature_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_ssl:
            self.ssl.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            x = x.squeeze(1)
        if x.ndim != 2:
            raise ValueError(f"Wav2Vec2SSLClassifier attend (B, T), recu {tuple(x.shape)}")

        if self.freeze_ssl:
            with torch.no_grad():
                features, _ = self.ssl.extract_features(x)
        else:
            features, _ = self.ssl.extract_features(x)

        if isinstance(features, (list, tuple)):
            features = features[-1]

        mean = features.mean(dim=1)
        std = features.std(dim=1).clamp_min(1e-8)
        pooled = torch.cat([mean, std], dim=1)

        logits = self.classifier(pooled).squeeze(-1)
        return torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)
