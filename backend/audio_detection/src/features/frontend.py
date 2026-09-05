# ============================================================
# src/features/frontend.py
#
# Frontends audio :
# - LFCC
# - Log-Mel
# - Combined (LFCC + Log-Mel)
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T


def _normalize_features(x: torch.Tensor) -> torch.Tensor:
    """
    Mean-variance normalization par instance.

    Réduit :
    - différences de gain
    - biais inter-datasets
    - instabilité numérique
    """
    mean = x.mean(dim=(1, 2), keepdim=True)

    std = x.std(dim=(1, 2), keepdim=True)

    std = std.clamp(min=1e-8)

    x = (x - mean) / std

    # sécurité numérique
    x = torch.nan_to_num(
        x,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    return x


def _prepare_waveform(waveform: torch.Tensor) -> torch.Tensor:
    """
    Standardise les shapes.

    Accepte :
    - (samples,)
    - (1, samples)
    - (batch, samples)
    - (batch, 1, samples)

    Retour :
    - (batch, samples)
    """

    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)

    elif waveform.dim() == 3:
        waveform = waveform.squeeze(1)

    return waveform


class LFCCFrontend(nn.Module):

    def __init__(
        self,
        sample_rate: int = 16000,
        n_lfcc: int = 60,
        n_fft: int = 512,
        hop_length: int = 160,
        win_length: int = 400,
        n_filter: int = 20,
        with_delta: bool = True,
        normalize: bool = True,
    ):
        super().__init__()

        self.with_delta = with_delta
        self.normalize = normalize

        self.lfcc_transform = T.LFCC(
            sample_rate=sample_rate,
            n_lfcc=n_lfcc,
            speckwargs={
                "n_fft": n_fft,
                "hop_length": hop_length,
                "win_length": win_length,
            },
            f_min=0.0,
            f_max=sample_rate / 2,
            n_filter=n_filter,
            log_lf=True,
        )

        if with_delta:
            self.delta = T.ComputeDeltas(win_length=5)

        self.n_features = n_lfcc * (3 if with_delta else 1)

    def forward(self, waveform: torch.Tensor):

        x = _prepare_waveform(waveform)

        # LFCC
        lfcc = self.lfcc_transform(x)

        if self.with_delta:

            delta1 = self.delta(lfcc)

            delta2 = self.delta(delta1)

            features = torch.cat(
                [lfcc, delta1, delta2],
                dim=1,
            )

        else:
            features = lfcc

        if self.normalize:
            features = _normalize_features(features)

        return features.unsqueeze(1)

class LogMelFrontend(nn.Module):

    def __init__(
        self,
        sample_rate: int = 16000,
        n_mels: int = 128,
        n_fft: int = 512,
        hop_length: int = 160,
        win_length: int = 400,
        f_min: float = 20.0,
        f_max: float = 8000.0,
        normalize: bool = True,
    ):
        super().__init__()

        self.normalize = normalize

        self.mel_spectrogram = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max,
            power=2.0,
        )

        self.amplitude_to_db = T.AmplitudeToDB(
            stype="power",
            top_db=80.0,
        )

        self.n_features = n_mels

    def forward(self, waveform: torch.Tensor):

        x = _prepare_waveform(waveform)

        mel = self.mel_spectrogram(x)

        log_mel = self.amplitude_to_db(mel)

        if self.normalize:
            log_mel = _normalize_features(log_mel)

        return log_mel.unsqueeze(1)


class CombinedFrontend(nn.Module):

    def __init__(
        self,
        sample_rate: int = 16000,
        n_lfcc: int = 60,
        n_mels: int = 80,
        n_fft: int = 512,
        hop_length: int = 160,
        win_length: int = 400,
        n_filter: int = 20,
        with_delta: bool = True,
        f_min: float = 20.0,
        f_max: float = 8000.0,
        normalize: bool = True,
    ):
        super().__init__()

        self.lfcc = LFCCFrontend(
            sample_rate=sample_rate,
            n_lfcc=n_lfcc,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_filter=n_filter,
            with_delta=with_delta,
            normalize=normalize,
        )

        self.logmel = LogMelFrontend(
            sample_rate=sample_rate,
            n_mels=n_mels,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            f_min=f_min,
            f_max=f_max,
            normalize=normalize,
        )

    def forward(self, waveform: torch.Tensor):

        lfcc_feat = self.lfcc(waveform)

        mel_feat = self.logmel(waveform)

        # alignement temporel
        T_lfcc = lfcc_feat.shape[-1]

        T_mel = mel_feat.shape[-1]

        if T_lfcc != T_mel:

            mel_feat = F.interpolate(
                mel_feat,
                size=(mel_feat.shape[2], T_lfcc),
                mode="nearest",
            )

        return torch.cat(
            [lfcc_feat, mel_feat],
            dim=2,
        )


def get_frontend(
    name: str,
    sample_rate: int = 16000,
    config: dict | None = None,
):
    config = config or {}

    lfcc_kwargs = {
        "sample_rate": sample_rate,
        "n_lfcc": config.get("n_lfcc", 60),
        "n_fft": config.get("n_fft", 512),
        "hop_length": config.get("hop_length", 160),
        "win_length": config.get("win_length", 400),
        "n_filter": config.get("n_filter", 20),
        "with_delta": config.get("with_delta", True),
        "normalize": config.get("normalize", True),
    }

    logmel_kwargs = {
        "sample_rate": sample_rate,
        "n_mels": config.get("n_mels", 128),
        "n_fft": config.get("n_fft", 512),
        "hop_length": config.get("hop_length", 160),
        "win_length": config.get("win_length", 400),
        "f_min": config.get("f_min", 20.0),
        "f_max": config.get("f_max", sample_rate / 2),
        "normalize": config.get("normalize", True),
    }

    combined_kwargs = {
        "sample_rate": sample_rate,
        "n_lfcc": config.get("n_lfcc", 60),
        "n_mels": config.get("n_mels", 80),
        "n_fft": config.get("n_fft", 512),
        "hop_length": config.get("hop_length", 160),
        "win_length": config.get("win_length", 400),
        "n_filter": config.get("n_filter", 20),
        "with_delta": config.get("with_delta", True),
        "f_min": config.get("f_min", 20.0),
        "f_max": config.get("f_max", sample_rate / 2),
        "normalize": config.get("normalize", True),
    }

    frontends = {
        "lfcc": lambda: LFCCFrontend(**lfcc_kwargs),

        "logmel": lambda: LogMelFrontend(**logmel_kwargs),

        "combined": lambda: CombinedFrontend(**combined_kwargs),
    }

    if name not in frontends:

        raise ValueError(
            f"Frontend inconnu : {name}\n"
            f"Choix disponibles : {list(frontends.keys())}"
        )

    return frontends[name]()
