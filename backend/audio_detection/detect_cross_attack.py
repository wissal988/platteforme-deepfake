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
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve
from tqdm import tqdm

from src.features.frontend import get_frontend
from src.models.aasist_lite import AASISTLite
from src.models.lcnn import LCNN
from src.models.raw_cnn import RawCNN
from src.models.wav2vec2_ssl import Wav2Vec2SSLClassifier
from src.utils.audio import load_audio, normalize_amplitude, resample, to_mono


AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}
DEFAULT_CHECKPOINT_DIR = Path("trained_models/cross_attack")
DEFAULT_RESULT_DIR = Path("trained_models/cross_attack")


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


def load_fold_models(
    cfg: dict,
    device: torch.device,
    checkpoint_dir: Path,
    folds: tuple[int, ...] | list[int] = (0, 1, 2),
):
    models = []
    if not checkpoint_dir.exists() and "cross_attack" in str(checkpoint_dir):
        legacy_dir = Path(str(checkpoint_dir).replace("cross_attack", "attack_agnostic"))
        if legacy_dir.exists():
            checkpoint_dir = legacy_dir

    frontend_name = cfg["features"]["type"]
    architecture_name = cfg["model"].get("name", "lcnn").lower()
    sr = int(cfg["data"]["sample_rate"])

    for fold in folds:
        raw_models = {"raw_cnn", "aasist_lite", "wav2vec2_ssl"}
        saved_frontend_name = "raw" if architecture_name in raw_models else frontend_name
        model_name = f"cross_attack_fold{fold}_{architecture_name}_{saved_frontend_name}"
        checkpoint_path = checkpoint_dir / f"{model_name}_best.pth"
        legacy_name = f"attack_agnostic_fold{fold}_{architecture_name}_{saved_frontend_name}"
        if not checkpoint_path.exists():
            legacy_path = checkpoint_dir / f"{legacy_name}_best.pth"
            if legacy_path.exists():
                model_name = legacy_name
                checkpoint_path = legacy_path
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint fold {fold} introuvable: {checkpoint_path}")

        if architecture_name == "raw_cnn":
            frontend = nn.Identity().to(device)
            model = RawCNN(
                input_channels=int(cfg["model"].get("input_channels", 1)),
                hidden_dim=int(cfg["model"].get("hidden_dim", 128)),
                dropout=float(cfg["model"].get("dropout", 0.5)),
            ).to(device)
        elif architecture_name == "aasist_lite":
            frontend = nn.Identity().to(device)
            model = AASISTLite(
                input_channels=int(cfg["model"].get("input_channels", 1)),
                channels=int(cfg["model"].get("channels", 128)),
                hidden_dim=int(cfg["model"].get("hidden_dim", 160)),
                attention_heads=int(cfg["model"].get("attention_heads", 4)),
                max_attn_frames=int(cfg["model"].get("max_attn_frames", 256)),
                dropout=float(cfg["model"].get("dropout", 0.4)),
            ).to(device)
        elif architecture_name == "wav2vec2_ssl":
            frontend = nn.Identity().to(device)
            model = Wav2Vec2SSLClassifier(
                bundle_name=str(cfg["model"].get("bundle_name", "WAV2VEC2_BASE")),
                hidden_dim=int(cfg["model"].get("hidden_dim", 256)),
                dropout=float(cfg["model"].get("dropout", 0.4)),
                freeze_ssl=bool(cfg["model"].get("freeze_ssl", True)),
            ).to(device)
        elif architecture_name == "lcnn":
            frontend = get_frontend(frontend_name, sample_rate=sr, config=cfg["features"]).to(device)
            model = LCNN(
                input_channels=int(cfg["model"].get("input_channels", 1)),
                hidden_dim=int(cfg["model"].get("hidden_dim", 160)),
                dropout=float(cfg["model"].get("dropout", 0.5)),
            ).to(device)
        else:
            raise ValueError(f"Modele inconnu: {architecture_name}")

        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
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


def aggregate_scores(scores: list[float], strategy: str = "mean_score") -> float:
    if not scores:
        raise ValueError("Aucun score a agreger.")
    sorted_scores = sorted(scores, reverse=True)
    if strategy == "max_score":
        return float(sorted_scores[0])
    if strategy == "top2_mean":
        top = sorted_scores[: min(2, len(sorted_scores))]
        return float(sum(top) / len(top))
    if strategy == "median_score":
        mid = len(sorted_scores) // 2
        if len(sorted_scores) % 2 == 1:
            return float(sorted_scores[mid])
        return float((sorted_scores[mid - 1] + sorted_scores[mid]) / 2.0)
    return float(sum(scores) / len(scores))


def confidence_percentages(
    decision_score: float,
    threshold: float | None,
    temperature: float = 0.10,
) -> tuple[float, float]:
    """
    Convertit un score de decision en pourcentages lisibles.

    Ce n'est pas une probabilite bayesienne stricte: c'est une confiance
    operationnelle centree sur le seuil calibre.
    """
    if threshold is None:
        fake_percent = max(0.0, min(100.0, decision_score * 100.0))
    else:
        temperature = max(float(temperature), 1e-6)
        x = (decision_score - threshold) / temperature
        fake_percent = 100.0 / (1.0 + np.exp(-x))
    real_percent = 100.0 - fake_percent
    return float(fake_percent), float(real_percent)


def build_conformity_report(
    predicted_label: str,
    confidence: str,
    fake_percent: float,
    real_percent: float,
    decision_score: float,
    threshold: float | None,
    score_strategy: str,
    fold_results: list[dict],
    margin: float | None,
    low_confidence_margin: float,
) -> dict:
    fake_votes = sum(1 for row in fold_results if row["vote"] == "FAKE")
    real_votes = len(fold_results) - fake_votes
    high_fold_scores = [row for row in fold_results if row["score"] >= 0.80]
    contradictory_high_fake = predicted_label == "real" and bool(high_fold_scores)
    reasons = []

    if threshold is not None:
        comparator = ">=" if decision_score >= threshold else "<"
        reasons.append(
            f"Score de decision {decision_score:.4f} {comparator} seuil calibre {threshold:.4f}."
        )
        if margin is not None and abs(margin) <= low_confidence_margin:
            reasons.append(
                "Le score est proche du seuil: decision binaire affichee, mais prudence recommandee."
            )
    else:
        reasons.append("Aucun seuil calibre fourni: decision basee sur les votes des folds.")

    reasons.append(f"Strategie d'agregation utilisee: {score_strategy}.")
    reasons.append(f"Votes internes: {fake_votes} FAKE / {real_votes} REAL.")

    if high_fold_scores:
        folds = ", ".join(f"fold {row['fold']}={row['score']:.3f}" for row in high_fold_scores)
        reasons.append(f"Scores fake eleves observes sur: {folds}.")
        if contradictory_high_fake:
            reasons.append(
                "Conclusion REAL affaiblie: au moins un fold detecte fortement un signal fake."
            )
    else:
        reasons.append("Aucun fold ne depasse 0.80 en score fake brut.")

    if fake_votes > 0 and real_votes > 0:
        reasons.append("Desaccord entre folds: le fichier peut etre hors distribution ou ambigu.")
    if confidence == "faible":
        reasons.append("Confiance faible: resultat a confirmer par ecoute humaine ou autre modele.")

    status = "conforme"
    if confidence == "faible" or contradictory_high_fake:
        status = "conforme_avec_reserve"

    return {
        "status": status,
        "predicted_label": predicted_label,
        "fake_percent": round(fake_percent, 2),
        "real_percent": round(real_percent, 2),
        "confidence": confidence,
        "reasons": reasons,
        "disclaimer": (
            "Ces pourcentages sont des scores de confiance operationnels calibres, "
            "pas une preuve mathematique que l'audio est reel ou fake."
        ),
    }


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
    max_score = aggregate_scores(raw_scores, "max_score")
    top2_mean = aggregate_scores(raw_scores, "top2_mean")
    median_score = aggregate_scores(raw_scores, "median_score")

    calibrated_threshold = None
    score_strategy = "mean_score"
    if decision_threshold is not None:
        calibrated_threshold = float(decision_threshold)
    elif calibration is not None:
        score_strategy = str(calibration.get("score_strategy", "mean_score"))
        calibrated_threshold = float(
            calibration["thresholds"].get(score_strategy, calibration["thresholds"]["mean_score"])
        )
    decision_score = {
        "mean_score": mean_score,
        "max_score": max_score,
        "top2_mean": top2_mean,
        "median_score": median_score,
    }.get(score_strategy, mean_score)

    margin_to_threshold = None
    if calibrated_threshold is not None:
        delta = decision_score - calibrated_threshold
        margin_to_threshold = delta
        if decision_score >= calibrated_threshold:
            decision = "FAKE probable"
            predicted_label = "fake"
        else:
            decision = "REAL probable"
            predicted_label = "real"
    else:
        if fake_votes > real_votes:
            decision = "FAKE probable"
            predicted_label = "fake"
        elif real_votes > fake_votes:
            decision = "REAL probable"
            predicted_label = "real"
        else:
            decision = "REAL probable"
            predicted_label = "real"

    confidence_gap = abs(fake_votes - real_votes)
    confidence = "elevee" if confidence_gap == 3 and abs(mean_margin) >= 0.15 else "moyenne"
    if confidence_gap == 1 or abs(mean_margin) < 0.08:
        confidence = "faible"
    if margin_to_threshold is not None and abs(margin_to_threshold) <= suspect_margin:
        confidence = "faible"
    if predicted_label == "real" and max_score >= 0.80:
        confidence = "faible"

    fake_percent, real_percent = confidence_percentages(
        decision_score,
        calibrated_threshold,
        temperature=max(suspect_margin, 0.05),
    )
    conformity_report = build_conformity_report(
        predicted_label=predicted_label,
        confidence=confidence,
        fake_percent=fake_percent,
        real_percent=real_percent,
        decision_score=decision_score,
        threshold=calibrated_threshold,
        score_strategy=score_strategy,
        fold_results=fold_results,
        margin=margin_to_threshold,
        low_confidence_margin=suspect_margin,
    )

    return {
        "audio": str(audio_path),
        "n_segments": len(segments),
        "decision": decision,
        "predicted_label": predicted_label,
        "fake_percent": fake_percent,
        "real_percent": real_percent,
        "fake_votes": fake_votes,
        "real_votes": real_votes,
        "mean_score": mean_score,
        "max_score": max_score,
        "top2_mean": top2_mean,
        "median_score": median_score,
        "decision_score": decision_score,
        "score_strategy": score_strategy,
        "mean_margin": mean_margin,
        "calibrated_threshold": calibrated_threshold,
        "low_confidence_margin": suspect_margin if calibrated_threshold is not None else None,
        "confidence": confidence,
        "conformity_report": conformity_report,
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
    print("RAPPORT DETECTION DEEPFAKE AUDIO - ENSEMBLE CROSS-ATTACK")
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
    print(f"Score max     : {result['max_score']:.4f}")
    print(f"Top-2 moyen   : {result['top2_mean']:.4f}")
    print(f"Score decision: {result['decision_score']:.4f} ({result['score_strategy']})")
    if result["calibrated_threshold"] is not None:
        print(f"Seuil calibre : {result['calibrated_threshold']:.4f}")
        print(f"Zone prudence : +/-{result['low_confidence_margin']:.4f}")
    print(f"Marge moyenne : {result['mean_margin']:+.4f}")
    print(f"Decision      : {result['decision']}")
    print(f"Pourcentage   : FAKE {result['fake_percent']:.2f}% | REAL {result['real_percent']:.2f}%")
    print(f"Confiance     : {result['confidence']} (interne, non garantie externe)")
    print("-" * 80)
    print("RAPPORT DE CONFORMITE")
    print(f"Statut        : {result['conformity_report']['status']}")
    for reason in result["conformity_report"]["reasons"]:
        print(f"- {reason}")
    print(f"Note          : {result['conformity_report']['disclaimer']}")
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
    y_pred_decided = []
    y_true_decided = []
    y_score = []

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

        y_true.append(label)
        y_score.append(result["decision_score"])
        if pred_label in {"fake", "real"}:
            y_true_decided.append(label)
            y_pred_decided.append(1 if pred_label == "fake" else 0)

        rows.append(
            {
                "audio": str(audio_path),
                "expected_label": label_name,
                "predicted_label": pred_label,
                "decision": result["decision"],
                "is_correct": pred_label == label_name,
                "fake_percent": round(result["fake_percent"], 2),
                "real_percent": round(result["real_percent"], 2),
                "fake_votes": result["fake_votes"],
                "real_votes": result["real_votes"],
                "mean_score": round(result["mean_score"], 6),
                "max_score": round(result["max_score"], 6),
                "top2_mean": round(result["top2_mean"], 6),
                "decision_score": round(result["decision_score"], 6),
                "score_strategy": result["score_strategy"],
                "calibrated_threshold": result["calibrated_threshold"],
                "low_confidence_margin": result["low_confidence_margin"],
                "mean_margin": round(result["mean_margin"], 6),
                "confidence": result["confidence"],
                "conformity_status": result["conformity_report"]["status"],
            }
        )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    auc = roc_auc_score(y_true, y_score) if len(set(y_true)) == 2 else None
    eer = compute_eer(y_true, y_score)
    decided_accuracy = (
        accuracy_score(y_true_decided, y_pred_decided)
        if y_pred_decided
        else None
    )
    decided_f1 = (
        f1_score(y_true_decided, y_pred_decided, average="macro")
        if y_pred_decided and len(set(y_true_decided)) == 2
        else None
    )
    summary = {
        "n_total": len(rows),
        "n_real": sum(1 for y in y_true if y == 0),
        "n_fake": sum(1 for y in y_true if y == 1),
        "n_decided": len(y_pred_decided),
        "coverage": len(y_pred_decided) / len(rows),
        "accuracy_decided": decided_accuracy,
        "f1_macro_decided": decided_f1,
        "auc": auc,
        "eer_%": eer,
        "fake_to_real": sum(1 for row in rows if row["expected_label"] == "fake" and row["predicted_label"] == "real"),
        "real_to_fake": sum(1 for row in rows if row["expected_label"] == "real" and row["predicted_label"] == "fake"),
        "low_confidence": sum(1 for row in rows if row["confidence"] == "faible"),
        "output_csv": str(output_csv),
    }

    print("\nExternal ensemble evaluation")
    print(f"Audios       : {summary['n_total']} | real={summary['n_real']} | fake={summary['n_fake']}")
    print(f"Decides      : {summary['n_decided']} | coverage={summary['coverage'] * 100:.2f}%")
    if summary["accuracy_decided"] is not None:
        print(f"Accuracy     : {summary['accuracy_decided'] * 100:.2f}%")
    if summary["f1_macro_decided"] is not None:
        print(f"F1 macro     : {summary['f1_macro_decided']:.4f}")
    if auc is not None:
        print(f"AUC          : {auc:.4f}")
    if eer is not None:
        print(f"EER          : {eer:.2f}%")
    print(f"Fake -> real : {summary['fake_to_real']}")
    print(f"Real -> fake : {summary['real_to_fake']}")
    print(f"Conf. faible : {summary['low_confidence']}")
    print(f"CSV          : {output_csv}")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Detecteur ensemble cross-attack.")
    parser.add_argument("--audio", type=str, default=None, help="Audio a analyser. Si absent, ouvre une fenetre.")
    parser.add_argument("--root", type=str, default=None, help="Dossier externe avec REAL/ et FAKE/.")
    parser.add_argument("--config", type=str, default="configs/debug_cpu_lcnn_logmel.yaml")
    parser.add_argument("--checkpoint_dir", type=str, default=str(DEFAULT_CHECKPOINT_DIR))
    parser.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2], choices=[0, 1, 2])
    parser.add_argument("--max_segments", type=int, default=5)
    parser.add_argument("--output_csv", type=str, default="report_assets/results/external_cross_attack_ensemble.csv")
    parser.add_argument("--output_json", type=str, default=None)
    parser.add_argument(
        "--calibration",
        type=str,
        default=None,
        help="JSON de calibration produit par scripts/calibrate_cross_attack.py.",
    )
    parser.add_argument(
        "--suspect_margin",
        type=float,
        default=0.05,
        help="Zone autour du seuil calibre marquee comme confiance faible, sans classe SUSPECT.",
    )
    parser.add_argument(
        "--low_confidence_margin",
        dest="suspect_margin",
        type=float,
        help="Alias plus clair de --suspect_margin.",
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
    fold_models = load_fold_models(cfg, device, Path(args.checkpoint_dir), folds=args.folds)
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

    output_path = Path(args.checkpoint_dir) / f"detection_ensemble_{audio_path.stem}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Rapport sauvegarde : {output_path}")


if __name__ == "__main__":
    main()
