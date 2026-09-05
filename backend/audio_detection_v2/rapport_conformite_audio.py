import argparse
import csv
import json
import tkinter as tk
from pathlib import Path
from tkinter import filedialog

import numpy as np
import torch
import torchaudio
import yaml
from torch import nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from tqdm import tqdm

from src.features.frontend import get_frontend
from src.models.lcnn import LCNN
from src.models.raw_cnn import RawCNN
from src.utils.audio import load_audio, normalize_amplitude, resample, to_mono


AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}
DEFAULT_CHECKPOINT_DIR = Path("trained_models/attack_agnostic")
DEFAULT_RESULT_DIR = Path("trained_models/attack_agnostic")


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
    """Architecture des anciens checkpoints attack_agnostic / attack_agnostic_600."""

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


def select_audio_file() -> Path:
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    selected = filedialog.askopenfilename(
        title="Selectionner un fichier audio",
        filetypes=[
            ("Audio files", "*.wav *.mp3 *.flac *.ogg *.m4a *.aac"),
            ("All files", "*.*"),
        ],
    )
    root.destroy()
    if not selected:
        raise ValueError("Aucun fichier audio selectionne.")
    return Path(selected)


def load_segments(
    audio_path: Path,
    target_sr: int,
    duration_sec: float,
    max_segments: int,
) -> list[torch.Tensor]:
    waveform, sr = load_audio(audio_path)
    waveform = to_mono(waveform)
    waveform = resample(waveform, sr, target_sr)
    waveform = normalize_amplitude(waveform)

    segment_len = int(target_sr * duration_sec)
    total_len = waveform.shape[-1]
    if total_len <= 0:
        raise ValueError(f"Audio vide: {audio_path}")

    if total_len <= segment_len:
        repeats = (segment_len // total_len) + 1
        segment = waveform.repeat(1, repeats)[:, :segment_len]
        return [segment.unsqueeze(0)]

    segments = []
    if max_segments <= 1:
        starts = [0]
    else:
        max_start = total_len - segment_len
        starts = np.linspace(0, max_start, num=max_segments, dtype=int).tolist()

    seen = set()
    for start in starts:
        if start in seen:
            continue
        seen.add(start)
        end = start + segment_len
        segment = waveform[:, start:end]
        segment = normalize_amplitude(segment)
        segments.append(segment.unsqueeze(0))

    return segments


def compute_eer(labels: list[int], scores: list[float]) -> float | None:
    if len(set(labels)) < 2:
        return None
    fpr, tpr, _ = roc_curve(labels, scores)
    fnr = 1.0 - tpr
    idx = min(range(len(fpr)), key=lambda i: abs(fpr[i] - fnr[i]))
    return float((fpr[idx] + fnr[idx]) / 2.0 * 100.0)


def load_threshold(model_name: str, result_dir: Path) -> float:
    result_path = result_dir / f"{model_name}_results.json"
    if not result_path.exists():
        return 0.5
    data = json.loads(result_path.read_text(encoding="utf-8"))
    return float(data["metrics"].get("threshold_used", 0.5))


def load_fold_models(cfg: dict, device: torch.device, checkpoint_dir: Path):
    models = []
    frontend_name = cfg["features"]["type"]
    architecture_name = cfg["model"].get("name", "lcnn").lower()
    sr = int(cfg["data"]["sample_rate"])

    for fold in (0, 1, 2):
        saved_frontend_name = "raw" if architecture_name == "raw_cnn" else frontend_name
        model_name = f"attack_agnostic_fold{fold}_{architecture_name}_{saved_frontend_name}"
        checkpoint_path = checkpoint_dir / f"{model_name}_best.pth"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint fold {fold} introuvable: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state = checkpoint["model_state_dict"]
        is_legacy_lcnn = any(".bn." in key for key in state)
        hidden_dim = int(state.get("classifier.weight", torch.empty(1, cfg["model"].get("hidden_dim", 160))).shape[1])

        if architecture_name == "raw_cnn":
            frontend = nn.Identity().to(device)
            model = RawCNN(
                input_channels=int(cfg["model"].get("input_channels", 1)),
                hidden_dim=hidden_dim,
                dropout=float(cfg["model"].get("dropout", 0.5)),
            ).to(device)
        elif architecture_name == "lcnn":
            frontend = get_frontend(frontend_name, sample_rate=sr, config=cfg["features"]).to(device)
            if is_legacy_lcnn:
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
                    norm_type=str(cfg["model"].get("norm_type", "instance")),
                ).to(device)
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
                "checkpoint": str(checkpoint_path),
                "threshold": load_threshold(model_name, checkpoint_dir),
                "model": model,
                "frontend": frontend,
            }
        )

    return models


def load_calibration(path: str | None) -> dict | None:
    if not path:
        return None
    calibration_path = Path(path)
    if not calibration_path.exists():
        raise FileNotFoundError(f"Calibration introuvable: {calibration_path}")
    return json.loads(calibration_path.read_text(encoding="utf-8"))


def compliance_status(predicted_label: str) -> str:
    if predicted_label == "real":
        return "CONFORME"
    if predicted_label == "fake":
        return "NON CONFORME"
    return "A VERIFIER"


def estimate_reliability_percent(
    mean_score: float,
    threshold: float | None,
    mean_margin: float,
    fake_votes: int,
    real_votes: int,
    suspect_margin: float | None,
    predicted_label: str,
) -> float:
    """Heuristic confidence for the final report, not a statistical guarantee."""
    total_votes = max(fake_votes + real_votes, 1)
    vote_agreement = abs(fake_votes - real_votes) / total_votes

    if threshold is not None:
        scale = max((suspect_margin or 0.0) * 2.0, 0.15)
        margin_strength = min(abs(mean_score - threshold) / scale, 1.0)
    else:
        margin_strength = min(abs(mean_margin) / 0.25, 1.0)

    reliability = 50.0 + 50.0 * (0.65 * margin_strength + 0.35 * vote_agreement)
    if predicted_label == "suspect":
        reliability = min(reliability, 65.0)
    return round(float(reliability), 2)


@torch.no_grad()
def predict_audio(
    audio_path: Path,
    cfg: dict,
    fold_models: list[dict],
    device: torch.device,
    max_segments: int,
    calibration: dict | None = None,
    suspect_margin: float = 0.05,
    decision_threshold: float | None = None,
):
    sr = int(cfg["data"]["sample_rate"])
    duration = float(cfg["data"]["max_duration_seconds"])
    segments = load_segments(audio_path, sr, duration, max_segments=max_segments)

    fold_results = []
    normalized_margins = []
    raw_scores = []

    for fold_model in fold_models:
        segment_scores = []
        for segment in segments:
            features = fold_model["frontend"](segment.to(device))
            logits = fold_model["model"](features)
            if isinstance(logits, tuple):
                logits = logits[0]
            score = torch.sigmoid(logits.view(-1))[0].item()
            segment_scores.append(score)

        score = float(sum(segment_scores) / len(segment_scores))
        threshold = float(fold_model["threshold"])
        margin = score - threshold
        vote = "FAKE" if score >= threshold else "REAL"

        raw_scores.append(score)
        normalized_margins.append(margin)
        fold_results.append(
            {
                "fold": fold_model["fold"],
                "score": score,
                "threshold": threshold,
                "margin": margin,
                "vote": vote,
                "segment_scores": segment_scores,
            }
        )

    fake_votes = sum(1 for row in fold_results if row["vote"] == "FAKE")
    real_votes = len(fold_results) - fake_votes
    mean_margin = float(sum(normalized_margins) / len(normalized_margins))
    mean_score = float(sum(raw_scores) / len(raw_scores))

    calibrated_threshold = None
    if decision_threshold is not None:
        calibrated_threshold = float(decision_threshold)
    elif calibration is not None:
        calibrated_threshold = float(calibration["thresholds"]["mean_score"])

    if calibrated_threshold is not None:
        delta = mean_score - calibrated_threshold
        has_fold_disagreement = fake_votes > 0 and real_votes > 0
        if abs(delta) <= suspect_margin:
            decision = "SUSPECT / a verifier"
            predicted_label = "suspect"
        elif has_fold_disagreement and abs(delta) <= suspect_margin * 2.0:
            decision = "SUSPECT / conflit entre folds"
            predicted_label = "suspect"
        elif mean_score >= calibrated_threshold:
            decision = "FAKE probable"
            predicted_label = "fake"
        else:
            decision = "REAL probable"
            predicted_label = "real"
    else:
        if fake_votes >= 2:
            decision = "FAKE probable"
            predicted_label = "fake"
        elif real_votes >= 2:
            decision = "REAL probable"
            predicted_label = "real"
        else:
            decision = "INCERTAIN"
            predicted_label = "uncertain"

    confidence_gap = abs(fake_votes - real_votes)
    confidence = "elevee" if confidence_gap == 3 and abs(mean_margin) >= 0.15 else "moyenne"
    if confidence_gap == 1 or abs(mean_margin) < 0.08:
        confidence = "faible"

    reliability_percent = estimate_reliability_percent(
        mean_score=mean_score,
        threshold=calibrated_threshold,
        mean_margin=mean_margin,
        fake_votes=fake_votes,
        real_votes=real_votes,
        suspect_margin=suspect_margin if calibrated_threshold is not None else None,
        predicted_label=predicted_label,
    )

    return {
        "audio": str(audio_path),
        "n_segments": len(segments),
        "decision": decision,
        "predicted_label": predicted_label,
        "compliance_status": compliance_status(predicted_label),
        "reliability_percent": reliability_percent,
        "fake_votes": fake_votes,
        "real_votes": real_votes,
        "mean_score": mean_score,
        "mean_margin": mean_margin,
        "calibrated_threshold": calibrated_threshold,
        "suspect_margin": suspect_margin if calibrated_threshold is not None else None,
        "confidence": confidence,
        "folds": fold_results,
    }


def find_labeled_audio(root: Path) -> list[tuple[Path, int, str]]:
    rows = []
    seen = set()
    for dirname, label, label_name in (
        ("REAL", 0, "real"),
        ("real", 0, "real"),
        ("FAKE", 1, "fake"),
        ("fake", 1, "fake"),
    ):
        folder = root / dirname
        if not folder.exists():
            continue
        for path in sorted(folder.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in AUDIO_EXTENSIONS:
                continue
            key = str(path.resolve()).lower()
            if key in seen:
                continue
            seen.add(key)
            rows.append((path, label, label_name))
    if not rows:
        raise FileNotFoundError(f"Aucun audio trouve dans {root}. Structure attendue: REAL/ et FAKE/.")
    return rows


def print_single_report(result: dict, cfg: dict) -> None:
    print("\n" + "=" * 80)
    print("RAPPORT DE CONFORMITE AUDIO")
    print("=" * 80)
    print(f"Audio analyse : {result['audio']}")
    print(f"Frontend      : {cfg['features']['type']}")
    print(f"Segments      : {result['n_segments']}")
    print("-" * 80)
    for row in result["folds"]:
        print(
            f"Fold {row['fold']} | vote={row['vote']:<4} | "
            f"score={row['score']:.4f} | threshold={row['threshold']:.4f} | "
            f"margin={row['margin']:+.4f}"
        )
    print("-" * 80)
    print(f"Votes FAKE    : {result['fake_votes']}")
    print(f"Votes REAL    : {result['real_votes']}")
    print(f"Score moyen   : {result['mean_score']:.4f}")
    if result["calibrated_threshold"] is not None:
        print(f"Seuil calibre : {result['calibrated_threshold']:.4f}")
        print(f"Marge suspect : +/-{result['suspect_margin']:.4f}")
    print(f"Marge moyenne : {result['mean_margin']:+.4f}")
    print(f"Statut        : {result['compliance_status']}")
    print(f"Decision      : {result['decision']}")
    print(f"Fiabilite     : {result['reliability_percent']:.2f}%")
    print(f"Niveau        : {result['confidence']} (estimation interne)")
    print("=" * 80)


def evaluate_folder(
    root: Path,
    cfg: dict,
    fold_models: list[dict],
    device: torch.device,
    max_segments: int,
    output_csv: Path,
    calibration: dict | None = None,
    suspect_margin: float = 0.05,
    decision_threshold: float | None = None,
):
    audio_rows = find_labeled_audio(root)
    rows = []
    y_true = []
    y_pred = []
    y_score = []
    decided_true = []
    decided_pred = []

    for audio_path, label, label_name in tqdm(audio_rows, desc="external ensemble"):
        result = predict_audio(
            audio_path,
            cfg,
            fold_models,
            device,
            max_segments=max_segments,
            calibration=calibration,
            suspect_margin=suspect_margin,
            decision_threshold=decision_threshold,
        )
        pred_label = result["predicted_label"]
        pred_binary = 1 if pred_label == "fake" else 0

        y_true.append(label)
        y_pred.append(pred_binary)
        y_score.append(result["mean_score"])
        if pred_label in {"real", "fake"}:
            decided_true.append(label)
            decided_pred.append(pred_binary)

        rows.append(
            {
                "audio": str(audio_path),
                "expected_label": label_name,
                "predicted_label": pred_label,
                "compliance_status": result["compliance_status"],
                "decision": result["decision"],
                "is_correct": pred_label == label_name,
                "reliability_percent": result["reliability_percent"],
                "fake_votes": result["fake_votes"],
                "real_votes": result["real_votes"],
                "mean_score": round(result["mean_score"], 6),
                "calibrated_threshold": result["calibrated_threshold"],
                "suspect_margin": result["suspect_margin"],
                "mean_margin": round(result["mean_margin"], 6),
                "confidence": result["confidence"],
            }
        )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    auc = roc_auc_score(y_true, y_score) if len(set(y_true)) == 2 else None
    eer = compute_eer(y_true, y_score)
    n_suspect = sum(1 for row in rows if row["predicted_label"] == "suspect")
    n_decided = len(rows) - n_suspect
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    decided_metrics = {}
    if decided_true:
        d_tn, d_fp, d_fn, d_tp = confusion_matrix(
            decided_true,
            decided_pred,
            labels=[0, 1],
        ).ravel()
        decided_metrics = {
            "accuracy_decided": accuracy_score(decided_true, decided_pred),
            "balanced_accuracy_decided": balanced_accuracy_score(decided_true, decided_pred),
            "precision_spoof_decided": precision_score(
                decided_true,
                decided_pred,
                pos_label=1,
                zero_division=0,
            ),
            "recall_spoof_decided": recall_score(
                decided_true,
                decided_pred,
                pos_label=1,
                zero_division=0,
            ),
            "f1_spoof_decided": f1_score(
                decided_true,
                decided_pred,
                pos_label=1,
                zero_division=0,
            ),
            "f1_macro_decided": f1_score(
                decided_true,
                decided_pred,
                average="macro",
                zero_division=0,
            ),
            "true_real_decided": int(d_tn),
            "false_fake_on_real_decided": int(d_fp),
            "false_real_on_fake_decided": int(d_fn),
            "true_fake_decided": int(d_tp),
        }
    else:
        decided_metrics = {
            "accuracy_decided": None,
            "balanced_accuracy_decided": None,
            "precision_spoof_decided": None,
            "recall_spoof_decided": None,
            "f1_spoof_decided": None,
            "f1_macro_decided": None,
            "true_real_decided": None,
            "false_fake_on_real_decided": None,
            "false_real_on_fake_decided": None,
            "true_fake_decided": None,
        }

    summary = {
        "n_total": len(rows),
        "n_real": sum(1 for y in y_true if y == 0),
        "n_fake": sum(1 for y in y_true if y == 1),
        "coverage": n_decided / max(len(rows), 1),
        "accuracy_all_suspect_as_real": accuracy_score(y_true, y_pred),
        "balanced_accuracy_all_suspect_as_real": balanced_accuracy_score(y_true, y_pred),
        "precision_spoof_all_suspect_as_real": precision_score(
            y_true,
            y_pred,
            pos_label=1,
            zero_division=0,
        ),
        "recall_spoof_all_suspect_as_real": recall_score(
            y_true,
            y_pred,
            pos_label=1,
            zero_division=0,
        ),
        "f1_spoof_all_suspect_as_real": f1_score(
            y_true,
            y_pred,
            pos_label=1,
            zero_division=0,
        ),
        "f1_macro_all_suspect_as_real": f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),
        **decided_metrics,
        "auc": auc,
        "eer_%": eer,
        "true_real": int(tn),
        "false_fake_on_real": int(fp),
        "false_real_on_fake": int(fn),
        "true_fake": int(tp),
        "fake_to_real": int(fn),
        "real_to_fake": int(fp),
        "suspect": n_suspect,
        "fake_to_suspect": sum(1 for row in rows if row["expected_label"] == "fake" and row["predicted_label"] == "suspect"),
        "real_to_suspect": sum(1 for row in rows if row["expected_label"] == "real" and row["predicted_label"] == "suspect"),
        "output_csv": str(output_csv),
    }

    print("\nExternal ensemble evaluation")
    print(f"Audios       : {summary['n_total']} | real={summary['n_real']} | fake={summary['n_fake']}")
    print(f"Coverage     : {summary['coverage'] * 100:.2f}%")
    if summary["accuracy_decided"] is not None:
        print(f"Accuracy dec.: {summary['accuracy_decided'] * 100:.2f}%")
        print(f"Bal Acc dec. : {summary['balanced_accuracy_decided'] * 100:.2f}%")
        print(f"Precision FK : {summary['precision_spoof_decided']:.4f}")
        print(f"Recall FK    : {summary['recall_spoof_decided']:.4f}")
        print(f"F1 spoof dec.: {summary['f1_spoof_decided']:.4f}")
        print(f"F1 dec.      : {summary['f1_macro_decided']:.4f}")
    print(f"Accuracy raw : {summary['accuracy_all_suspect_as_real'] * 100:.2f}%")
    print(f"Bal Acc raw  : {summary['balanced_accuracy_all_suspect_as_real'] * 100:.2f}%")
    print(f"F1 spoof raw : {summary['f1_spoof_all_suspect_as_real']:.4f}")
    print(f"F1 macro raw : {summary['f1_macro_all_suspect_as_real']:.4f}")
    if auc is not None:
        print(f"AUC          : {auc:.4f}")
    if eer is not None:
        print(f"EER          : {eer:.2f}%")
    print(
        f"Confusion    : TN={summary['true_real']} | FP={summary['false_fake_on_real']} | "
        f"FN={summary['false_real_on_fake']} | TP={summary['true_fake']}"
    )
    print(f"Fake -> real : {summary['fake_to_real']}")
    print(f"Real -> fake : {summary['real_to_fake']}")
    print(f"Suspect      : {summary['suspect']} | fake={summary['fake_to_suspect']} | real={summary['real_to_suspect']}")
    print(f"CSV          : {output_csv}")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Detecteur ensemble attack-agnostic.")
    parser.add_argument("--audio", type=str, default=None, help="Audio a analyser. Si absent, ouvre une fenetre.")
    parser.add_argument("--root", type=str, default=None, help="Dossier externe avec REAL/ et FAKE/.")
    parser.add_argument("--config", type=str, default="configs/debug_cpu_lcnn_logmel.yaml")
    parser.add_argument("--checkpoint_dir", type=str, default=str(DEFAULT_CHECKPOINT_DIR))
    parser.add_argument("--max_segments", type=int, default=5)
    parser.add_argument("--output_csv", type=str, default="report_assets/results/external_attack_agnostic_ensemble.csv")
    parser.add_argument("--output_json", type=str, default=None)
    parser.add_argument(
        "--calibration",
        type=str,
        default=None,
        help="JSON de calibration produit par scripts/calibrer_seuil_systeme.py.",
    )
    parser.add_argument(
        "--suspect_margin",
        type=float,
        default=0.05,
        help="Zone autour du seuil calibre ou la decision devient SUSPECT.",
    )
    parser.add_argument(
        "--decision_threshold",
        type=float,
        default=None,
        help="Remplace le seuil du fichier de calibration pour tester un point de fonctionnement.",
    )
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fold_models = load_fold_models(cfg, device, Path(args.checkpoint_dir))
    calibration = load_calibration(args.calibration)

    if args.root:
        summary = evaluate_folder(
            Path(args.root),
            cfg,
            fold_models,
            device,
            max_segments=args.max_segments,
            output_csv=Path(args.output_csv),
            calibration=calibration,
            suspect_margin=args.suspect_margin,
            decision_threshold=args.decision_threshold,
        )
        output_json = Path(args.output_json) if args.output_json else Path(args.output_csv).with_suffix(".json")
        output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return

    audio_path = Path(args.audio) if args.audio else select_audio_file()
    result = predict_audio(
        audio_path,
        cfg,
        fold_models,
        device,
        max_segments=args.max_segments,
        calibration=calibration,
        suspect_margin=args.suspect_margin,
        decision_threshold=args.decision_threshold,
    )
    print_single_report(result, cfg)

    output_path = DEFAULT_RESULT_DIR / f"detection_ensemble_{audio_path.stem}.json"
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Rapport sauvegarde : {output_path}")


if __name__ == "__main__":
    main()
