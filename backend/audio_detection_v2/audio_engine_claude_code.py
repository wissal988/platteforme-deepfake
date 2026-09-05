"""
Adaptateur backend portable pour le modele audio de `claude code`.

Structure attendue:
  audio_engine_claude_code.py
  configs/
  src/
  trained_models/attack_agnostic/

Usage FastAPI:
  from audio_engine_claude_code import load_audio_engine, run_audio_inference

  load_audio_engine()
  result = run_audio_inference(audio_bytes, filename)
"""

from __future__ import annotations

import json
import os
import random
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch import nn


PACKAGE_DIR = Path(__file__).resolve().parent
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))

from src.features.frontend import get_frontend  # noqa: E402
from src.models.lcnn import LCNN  # noqa: E402
from src.models.raw_cnn import RawCNN  # noqa: E402
from src.utils.audio import load_audio, normalize_amplitude, resample, to_mono  # noqa: E402


DEFAULT_CONFIG = PACKAGE_DIR / "configs" / "attack_agnostic_lcnn_logmel.yaml"
DEFAULT_CHECKPOINT_DIR = PACKAGE_DIR / "trained_models" / "attack_agnostic"

_ENGINE: dict | None = None


class LegacyMaxFeatureMap(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = torch.chunk(x, 2, dim=1)
        return torch.max(x1, x2)


class LegacyConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, padding: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels * 2, kernel_size=kernel_size, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels * 2)
        self.mfm = LegacyMaxFeatureMap()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mfm(self.bn(self.conv(x)))


class LegacyLCNN(nn.Module):
    def __init__(self, input_channels: int = 1, hidden_dim: int = 160, dropout: float = 0.0):
        super().__init__()
        self.features = nn.Sequential(
            LegacyConvBlock(input_channels, 32, kernel_size=5, padding=2),
            nn.MaxPool2d(2),
            LegacyConvBlock(32, 48),
            nn.MaxPool2d(2),
            LegacyConvBlock(48, 64),
            nn.MaxPool2d(2),
            LegacyConvBlock(64, 128),
            nn.MaxPool2d(2),
            LegacyConvBlock(128, 64),
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.LazyLinear(hidden_dim)
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.reshape(x.shape[0], -1)
        x = self.dropout(x)
        x = F.relu(self.fc(x))
        x = self.dropout(x)
        return self.classifier(x).squeeze(-1)


def get_audio_engine() -> dict | None:
    return _ENGINE


def load_audio_engine(
    config_path: str | Path = DEFAULT_CONFIG,
    checkpoint_dir: str | Path = DEFAULT_CHECKPOINT_DIR,
    folds: tuple[int, ...] = (0, 1, 2),
) -> dict:
    global _ENGINE

    config_path = Path(config_path)
    checkpoint_dir = Path(checkpoint_dir)
    if not config_path.exists():
        raise FileNotFoundError(f"Config introuvable: {config_path}")
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Dossier checkpoints introuvable: {checkpoint_dir}")

    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fold_models = _load_fold_models(cfg, device, checkpoint_dir, folds=folds)
    _ENGINE = {"cfg": cfg, "fold_models": fold_models, "device": device}
    return _ENGINE


def run_audio_inference(audio_bytes: bytes, filename: str = "audio.wav", max_segments: int = 5) -> dict:
    engine = get_audio_engine()
    if engine is None:
        engine = load_audio_engine()

    suffix = Path(filename).suffix or ".wav"
    fd, tmp = tempfile.mkstemp(suffix=suffix)
    try:
        os.write(fd, audio_bytes)
        os.close(fd)
        result = _predict(
            Path(tmp),
            engine["cfg"],
            engine["fold_models"],
            engine["device"],
            max_segments=max_segments,
        )
    except Exception as exc:
        return _stub(str(exc))
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass

    verdict = result["predicted_label"]
    return {
        "verdict": verdict,
        "confidence": result["reliability_percent"],
        "predicted_label": verdict,
        "decision": result["decision"],
        "compliance_status": result["compliance_status"],
        "reliability_percent": result["reliability_percent"],
        "mean_score": result["mean_score"],
        "mean_margin": result["mean_margin"],
        "calibrated_threshold": result["calibrated_threshold"],
        "fake_votes": result["fake_votes"],
        "real_votes": result["real_votes"],
        "folds": result["folds"],
        "stub": False,
    }


def _stub(error: str | None = None) -> dict:
    score = random.uniform(0.3, 0.9)
    verdict = "fake" if score > 0.5 else "real"
    return {
        "verdict": verdict,
        "confidence": round(score * 100, 2),
        "predicted_label": verdict,
        "decision": "mode stub",
        "compliance_status": "A VERIFIER",
        "reliability_percent": round(score * 100, 2),
        "mean_score": round(score, 4),
        "mean_margin": None,
        "calibrated_threshold": None,
        "fake_votes": 2 if verdict == "fake" else 1,
        "real_votes": 1 if verdict == "fake" else 2,
        "folds": [],
        "stub": True,
        "error": error,
    }


def _detect_norm_type(state: dict) -> str:
    has_running = any("running_mean" in key for key in state)
    has_norm = any(".norm." in key for key in state)
    has_bn = any(".bn." in key for key in state)
    if has_running and (has_norm or has_bn):
        return "batch"
    if has_norm:
        return "instance"
    return "none"


def _is_true_legacy(state: dict) -> bool:
    has_se = "features.10.fc.0.weight" in state or "features.10.pool.weight" in state
    has_f9 = any(key.startswith("features.9.") for key in state)
    return not (has_se or has_f9)


def _load_threshold(model_name: str, result_dir: Path) -> float:
    result_path = result_dir / f"{model_name}_results.json"
    if not result_path.exists():
        return 0.5
    data = json.loads(result_path.read_text(encoding="utf-8"))
    return float(data["metrics"].get("threshold_used", 0.5))


def _load_fold_models(
    cfg: dict,
    device: torch.device,
    checkpoint_dir: Path,
    folds: tuple[int, ...],
) -> list[dict]:
    frontend_name = cfg["features"]["type"]
    architecture_name = cfg["model"].get("name", "lcnn").lower()
    sample_rate = int(cfg["data"]["sample_rate"])
    models = []

    for fold in folds:
        saved_frontend_name = "raw" if architecture_name == "raw_cnn" else frontend_name
        model_name = f"attack_agnostic_fold{fold}_{architecture_name}_{saved_frontend_name}"
        checkpoint_path = checkpoint_dir / f"{model_name}_best.pth"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint fold {fold} introuvable: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state = checkpoint["model_state_dict"]
        hidden_dim = int(
            state.get(
                "classifier.weight",
                torch.empty(1, cfg["model"].get("hidden_dim", 160)),
            ).shape[1]
        )

        if architecture_name == "raw_cnn":
            frontend = nn.Identity().to(device)
            model = RawCNN(
                input_channels=int(cfg["model"].get("input_channels", 1)),
                hidden_dim=hidden_dim,
                dropout=float(cfg["model"].get("dropout", 0.5)),
            ).to(device)
        elif architecture_name == "lcnn":
            frontend = get_frontend(frontend_name, sample_rate=sample_rate, config=cfg["features"]).to(device)
            if _is_true_legacy(state):
                model = LegacyLCNN(
                    input_channels=int(cfg["model"].get("input_channels", 1)),
                    hidden_dim=hidden_dim,
                    dropout=0.0,
                ).to(device)
            else:
                model = LCNN(
                    input_channels=int(cfg["model"].get("input_channels", 1)),
                    hidden_dim=hidden_dim,
                    dropout=float(cfg["model"].get("dropout", 0.5)),
                    norm_type=_detect_norm_type(state),
                ).to(device)
                if any(".bn." in key for key in state):
                    state = {key.replace(".bn.", ".norm."): value for key, value in state.items()}
        else:
            raise ValueError(f"Modele inconnu: {architecture_name}")

        model.load_state_dict(state)
        frontend_state = checkpoint.get("frontend_state_dict", {})
        if frontend_state:
            frontend.load_state_dict(frontend_state)

        model.eval()
        frontend.eval()
        models.append(
            {
                "fold": fold,
                "name": model_name,
                "threshold": _load_threshold(model_name, checkpoint_dir),
                "model": model,
                "frontend": frontend,
            }
        )

    return models


def _load_segments(audio_path: Path, target_sr: int, duration_sec: float, max_segments: int) -> list[torch.Tensor]:
    waveform, sr = load_audio(audio_path)
    waveform = normalize_amplitude(to_mono(resample(waveform, sr, target_sr)))
    segment_len = int(target_sr * duration_sec)
    total_len = waveform.shape[-1]
    if total_len <= 0:
        raise ValueError(f"Audio vide: {audio_path}")

    if total_len <= segment_len:
        repeats = (segment_len // total_len) + 1
        return [waveform.repeat(1, repeats)[:, :segment_len].unsqueeze(0)]

    starts = np.linspace(0, total_len - segment_len, num=max(max_segments, 1), dtype=int).tolist()
    seen: set[int] = set()
    segments = []
    for start in starts:
        if start in seen:
            continue
        seen.add(start)
        segment = waveform[:, start : start + segment_len]
        segments.append(normalize_amplitude(segment).unsqueeze(0))
    return segments


@torch.no_grad()
def _predict(
    audio_path: Path,
    cfg: dict,
    fold_models: list[dict],
    device: torch.device,
    max_segments: int = 5,
    suspect_margin: float = 0.05,
) -> dict:
    sample_rate = int(cfg["data"]["sample_rate"])
    duration = float(cfg["data"]["max_duration_seconds"])
    segments = _load_segments(audio_path, sample_rate, duration, max_segments)

    fold_results = []
    raw_scores = []
    margins = []
    for fold_model in fold_models:
        segment_scores = []
        for segment in segments:
            features = fold_model["frontend"](segment.to(device))
            logits = fold_model["model"](features)
            if isinstance(logits, tuple):
                logits = logits[0]
            segment_scores.append(torch.sigmoid(logits.view(-1))[0].item())

        score = float(sum(segment_scores) / len(segment_scores))
        threshold = float(fold_model["threshold"])
        margin = score - threshold
        raw_scores.append(score)
        margins.append(margin)
        fold_results.append(
            {
                "fold": fold_model["fold"],
                "score": round(score, 4),
                "threshold": round(threshold, 4),
                "margin": round(margin, 4),
                "vote": "FAKE" if score >= threshold else "REAL",
                "segment_scores": [round(value, 4) for value in segment_scores],
            }
        )

    fake_votes = sum(1 for row in fold_results if row["vote"] == "FAKE")
    real_votes = len(fold_results) - fake_votes
    mean_score = float(sum(raw_scores) / len(raw_scores))
    mean_margin = float(sum(margins) / len(margins))
    calibrated_threshold = float(fold_models[0]["threshold"]) if fold_models else 0.5
    delta = mean_score - calibrated_threshold

    if abs(delta) <= suspect_margin:
        decision = "SUSPECT / a verifier"
        predicted_label = "suspect"
    elif fake_votes > 0 and real_votes > 0 and abs(delta) <= suspect_margin * 2:
        decision = "SUSPECT / conflit folds"
        predicted_label = "suspect"
    elif mean_score >= calibrated_threshold:
        decision = "FAKE probable"
        predicted_label = "fake"
    else:
        decision = "REAL probable"
        predicted_label = "real"

    reliability_percent = _estimate_reliability_percent(
        mean_score=mean_score,
        threshold=calibrated_threshold,
        fake_votes=fake_votes,
        real_votes=real_votes,
        suspect_margin=suspect_margin,
        predicted_label=predicted_label,
    )

    return {
        "predicted_label": predicted_label,
        "decision": decision,
        "compliance_status": _compliance_status(predicted_label),
        "reliability_percent": reliability_percent,
        "mean_score": round(mean_score, 4),
        "mean_margin": round(mean_margin, 4),
        "calibrated_threshold": round(calibrated_threshold, 4),
        "fake_votes": fake_votes,
        "real_votes": real_votes,
        "folds": fold_results,
    }


def _estimate_reliability_percent(
    mean_score: float,
    threshold: float,
    fake_votes: int,
    real_votes: int,
    suspect_margin: float,
    predicted_label: str,
) -> float:
    total_votes = max(fake_votes + real_votes, 1)
    vote_agreement = abs(fake_votes - real_votes) / total_votes
    scale = max(suspect_margin * 2.0, 0.15)
    margin_strength = min(abs(mean_score - threshold) / scale, 1.0)
    reliability = 50.0 + 50.0 * (0.65 * margin_strength + 0.35 * vote_agreement)
    if predicted_label == "suspect":
        reliability = min(reliability, 65.0)
    return round(float(reliability), 2)


def _compliance_status(predicted_label: str) -> str:
    if predicted_label == "real":
        return "CONFORME"
    if predicted_label == "fake":
        return "NON CONFORME"
    return "A VERIFIER"
