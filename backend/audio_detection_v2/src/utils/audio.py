# ============================================================
# src/utils/audio.py
#
# Preprocessing audio pour deepfake detection
# ============================================================

from pathlib import Path
import subprocess
import tempfile

import numpy as np
import torch
import torchaudio


# ============================================================
# Constantes globales
# ============================================================

TARGET_SR = 16000

TARGET_CHANNELS = 1


# cache des resamplers
_RESAMPLERS = {}


# ============================================================
# Chargement
# ============================================================

def load_audio(filepath: str | Path):
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"Fichier audio introuvable : {filepath}")

    try:
        waveform, sample_rate = torchaudio.load(str(filepath))
        waveform = waveform.float()
    except Exception:
        import soundfile as sf

        try:
            waveform_np, sample_rate = sf.read(str(filepath), always_2d=True)
            waveform = torch.from_numpy(waveform_np.T).float()
        except Exception:
            tmp_path = None
            try:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    tmp_path = Path(tmp.name)

                subprocess.run(
                    [
                        "ffmpeg",
                        "-y",
                        "-hide_banner",
                        "-loglevel",
                        "error",
                        "-i",
                        str(filepath),
                        "-acodec",
                        "pcm_s16le",
                        str(tmp_path),
                    ],
                    check=True,
                )
                waveform_np, sample_rate = sf.read(str(tmp_path), always_2d=True)
                waveform = torch.from_numpy(waveform_np.T.astype(np.float32)).float()
            finally:
                if tmp_path and tmp_path.exists():
                    tmp_path.unlink()

    if waveform.numel() == 0:
        raise ValueError(f"Audio vide : {filepath}")

    return waveform, sample_rate

# ============================================================
# Mono
# ============================================================

def to_mono(waveform: torch.Tensor):

    if waveform.dim() != 2:

        raise ValueError(
            f"Shape audio invalide : {waveform.shape}\n"
            f"Attendu : (channels, samples)"
        )

    if waveform.shape[0] > 1:

        waveform = waveform.mean(
            dim=0,
            keepdim=True,
        )

    return waveform


# ============================================================
# Resample
# ============================================================

def _get_resampler(
    orig_sr: int,
    target_sr: int,
):

    key = (orig_sr, target_sr)

    if key not in _RESAMPLERS:

        _RESAMPLERS[key] = torchaudio.transforms.Resample(
            orig_freq=orig_sr,
            new_freq=target_sr,
            lowpass_filter_width=64,
            rolloff=0.9985,
        )

    return _RESAMPLERS[key]


def resample(
    waveform: torch.Tensor,
    orig_sr: int,
    target_sr: int = TARGET_SR,
):

    if orig_sr == target_sr:
        return waveform

    resampler = _get_resampler(
        orig_sr,
        target_sr,
    )

    waveform = resampler(waveform)

    waveform = torch.nan_to_num(
        waveform,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    return waveform


# ============================================================
# Normalisation
# ============================================================

def normalize_amplitude(
    waveform: torch.Tensor,
    eps: float = 1e-8,
):

    peak = waveform.abs().max()

    if peak < eps:
        return waveform

    waveform = waveform / peak

    waveform = torch.clamp(
        waveform,
        min=-1.0,
        max=1.0,
    )

    waveform = torch.nan_to_num(
        waveform,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    return waveform


# ============================================================
# Padding / truncature
# ============================================================

def pad_or_truncate(
    waveform: torch.Tensor,
    target_length: int,
):

    n_samples = waveform.shape[-1]

    if n_samples <= 0:

        raise ValueError(
            "Signal audio vide après preprocessing"
        )

    # truncature
    if n_samples >= target_length:

        return waveform[:, :target_length]

    # padding cyclique
    repeats = (target_length // n_samples) + 1

    waveform = waveform.repeat(
        1,
        repeats,
    )

    return waveform[:, :target_length]


# ============================================================
# Pipeline complet
# ============================================================

def preprocess_audio(
    filepath: str | Path,
    target_sr: int = TARGET_SR,
    max_duration_sec: float = 4.0,
):

    target_length = int(
        target_sr * max_duration_sec
    )

    waveform, orig_sr = load_audio(filepath)

    waveform = to_mono(waveform)

    waveform = resample(
        waveform,
        orig_sr,
        target_sr,
    )

    waveform = normalize_amplitude(
        waveform
    )

    waveform = pad_or_truncate(
        waveform,
        target_length,
    )

    # sécurité finale
    waveform = torch.nan_to_num(
        waveform,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    return waveform


# ============================================================
# Infos audio
# ============================================================

def get_audio_info(
    filepath: str | Path,
):

    info = torchaudio.info(
        str(filepath)
    )

    return {
        "sample_rate": info.sample_rate,
        "num_channels": info.num_channels,
        "num_frames": info.num_frames,
        "duration_sec": (
            info.num_frames / info.sample_rate
        ),
        "encoding": info.encoding,
    }
