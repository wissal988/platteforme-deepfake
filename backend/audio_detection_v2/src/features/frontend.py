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
        n_filter: int = 70,
        with_delta: bool = True,
        normalize: bool = True,
        augment: bool = False,
    ):
        super().__init__()

        self.with_delta = with_delta
        self.normalize = normalize
        self.augment = augment

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

        if augment:
            # SpecAugment : double masquage fréquence + double masquage temps.
            # Ancien : 1 masque fréquence (n//12) + 1 masque temps (40).
            # Nouveau : 2 masques fréquence (n//6 + n//10) + 2 masques temps.
            # Masques plus larges → régularisation plus forte → meilleure généralisation.
            # Ref: Park et al., "SpecAugment", Interspeech 2019 (Table 3 montre
            # que LB policy avec 2 masques > 1 masque sur LibriSpeech).
            n_freq_bins = n_lfcc * (3 if with_delta else 1)
            self.freq_mask1 = T.FrequencyMasking(freq_mask_param=max(1, n_freq_bins // 6))
            self.freq_mask2 = T.FrequencyMasking(freq_mask_param=max(1, n_freq_bins // 10))
            self.time_mask1 = T.TimeMasking(time_mask_param=40)
            self.time_mask2 = T.TimeMasking(time_mask_param=20)

    def forward(self, waveform: torch.Tensor):

        x = _prepare_waveform(waveform)

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

        if self.augment and self.training:
            features = self.freq_mask1(features)
            features = self.freq_mask2(features)
            features = self.time_mask1(features)
            features = self.time_mask2(features)

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
        augment: bool = False,
    ):
        super().__init__()

        self.normalize = normalize
        self.augment = augment

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

        if augment:
            self.freq_mask1 = T.FrequencyMasking(freq_mask_param=max(1, n_mels // 6))
            self.freq_mask2 = T.FrequencyMasking(freq_mask_param=max(1, n_mels // 10))
            self.time_mask1 = T.TimeMasking(time_mask_param=40)
            self.time_mask2 = T.TimeMasking(time_mask_param=20)

    def forward(self, waveform: torch.Tensor):

        x = _prepare_waveform(waveform)

        mel = self.mel_spectrogram(x)

        log_mel = self.amplitude_to_db(mel)

        if self.normalize:
            log_mel = _normalize_features(log_mel)

        if self.augment and self.training:
            log_mel = self.freq_mask1(log_mel)
            log_mel = self.freq_mask2(log_mel)
            log_mel = self.time_mask1(log_mel)
            log_mel = self.time_mask2(log_mel)

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
        n_filter: int = 70,
        with_delta: bool = True,
        f_min: float = 20.0,
        f_max: float = 8000.0,
        normalize: bool = True,
        augment: bool = False,
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
            augment=augment,
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
            augment=augment,
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
        "n_filter": config.get("n_filter", 70),
        "with_delta": config.get("with_delta", True),
        "normalize": config.get("normalize", True),
        "augment": config.get("augment", False),
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
        "augment": config.get("augment", False),
    }

    combined_kwargs = {
        "sample_rate": sample_rate,
        "n_lfcc": config.get("n_lfcc", 60),
        "n_mels": config.get("n_mels", 80),
        "n_fft": config.get("n_fft", 512),
        "hop_length": config.get("hop_length", 160),
        "win_length": config.get("win_length", 400),
        "n_filter": config.get("n_filter", 70),
        "with_delta": config.get("with_delta", True),
        "f_min": config.get("f_min", 20.0),
        "f_max": config.get("f_max", sample_rate / 2),
        "normalize": config.get("normalize", True),
        "augment": config.get("augment", False),
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
