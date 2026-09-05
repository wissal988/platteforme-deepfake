import sys
import os
from pathlib import Path

AUDIO_DIR = Path(__file__).parent.parent.parent / "audio_detection_v2"
sys.path.insert(0, str(AUDIO_DIR))

import torch
from PIL import Image
import numpy as np
from tensorflow import keras

from app.ml.image_model import predict_image
from app.ml.video_model import predict_video

print("⏳ Chargement modèle image...")
from app.ml.image_model import load_image_model
load_image_model()
print("✅ Modèle image chargé")

print("⏳ Chargement modèle vidéo...")
from app.ml.video_model import load_video_model, load_feature_extractor
load_video_model()
load_feature_extractor()
print("✅ Modèle vidéo chargé")

print("⏳ Chargement modèle audio...")
from audio_engine_claude_code import load_audio_engine, run_audio_inference
load_audio_engine()
print("✅ Modèle audio chargé")


def score_to_confidence(prob_fake: float, label: str) -> float:
    """
    Retourne le pourcentage de confiance sur la décision.
    Toujours entre 50% et 100%.
    """
    if label == "FAKE":
        confidence = prob_fake * 100
    else:
        confidence = (1 - prob_fake) * 100
    return round(max(50.0, min(100.0, confidence)), 2)


# ── Image ─────────────────────────────────────────────────────────────────────
def run_image_model_with_confidence(file_path: str) -> dict:
    try:
        result = predict_image(file_path)
        label = result.get("label", "REAL")
        prob_fake = float(result.get("prob_fake", 0.5))
        confidence = score_to_confidence(prob_fake, label)
        return {
            "success": True,
            "result": {
                **result,
                "score": confidence,          # ✅ score = % confiance
                "confidence_pct": confidence,
                "confidence_label": f"{confidence:.1f}% certain que c'est {label}",
            }
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


# ── Audio seul ────────────────────────────────────────────────────────────────
def run_audio_model(file_path: str) -> dict:
    try:
        with open(file_path, "rb") as f:
            audio_bytes = f.read()
        filename = Path(file_path).name
        result = run_audio_inference(audio_bytes, filename, max_segments=5)

        predicted = result.get("predicted_label", "real").lower()
        label = "FAKE" if predicted in ("fake", "suspect") else "REAL"

        mean_score = float(result.get("mean_score", 0.5))
        reliability = float(result.get("reliability_percent", 50.0))
        confidence = score_to_confidence(mean_score, label)
        final_confidence = round(max(confidence, min(reliability, 100.0)), 2)

        return {
            "success": True,
            "result": {
                "label": label,
                "prob_fake": mean_score,
                "prob_real": round(1 - mean_score, 4),
                "score": final_confidence,    # ✅ score = % confiance
                "confidence_pct": final_confidence,
                "confidence_label": f"{final_confidence:.1f}% certain que c'est {label}",
                "decision": result.get("decision", "—"),
                "compliance_status": result.get("compliance_status", "—"),
                "fake_votes": result.get("fake_votes", 0),
                "real_votes": result.get("real_votes", 0),
                "model_name": "attack_agnostic_lcnn_logmel",
                "threshold": result.get("calibrated_threshold", 0.5),
            },
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


# ── Vidéo + Audio combinés ────────────────────────────────────────────────────
def run_video_with_audio(file_path: str) -> dict:
    try:
        import tempfile
        import subprocess

        video_result = predict_video(file_path)
        video_label = video_result.get("label", "REAL")
        video_prob_fake = float(video_result.get("prob_fake", 0.5))
        video_confidence = score_to_confidence(video_prob_fake, video_label)

        audio_label = None
        audio_result = None
        audio_confidence = None

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_audio_path = tmp.name

        try:
            ret = subprocess.run(
                ["ffmpeg", "-y", "-i", file_path,
                 "-vn", "-acodec", "pcm_s16le",
                 "-ar", "16000", "-ac", "1",
                 tmp_audio_path],
                capture_output=True, timeout=60
            )
            if ret.returncode == 0 and Path(tmp_audio_path).stat().st_size > 0:
                audio_result = run_audio_model(tmp_audio_path)
                if audio_result.get("success"):
                    audio_label = audio_result["result"]["label"]
                    audio_confidence = audio_result["result"]["confidence_pct"]
        except Exception:
            audio_result = None
        finally:
            if Path(tmp_audio_path).exists():
                os.remove(tmp_audio_path)

        final_label = "FAKE" if video_label == "FAKE" or audio_label == "FAKE" else "REAL"

        if audio_confidence is not None:
            if final_label == "FAKE":
                if video_label == "FAKE" and audio_label == "FAKE":
                    final_confidence = round(video_confidence * 0.6 + audio_confidence * 0.4, 2)
                elif video_label == "FAKE":
                    final_confidence = video_confidence
                else:
                    final_confidence = audio_confidence
            else:
                final_confidence = round((video_confidence + audio_confidence) / 2, 2)
        else:
            final_confidence = video_confidence

        return {
            "success": True,
            "result": {
                "label": final_label,
                "score": final_confidence,    # ✅ score = % confiance
                "confidence_pct": final_confidence,
                "confidence_label": f"{final_confidence:.1f}% certain que c'est {final_label}",
                "model_name": "deepfake_video_model.h5 + attack_agnostic_lcnn_logmel",
                "decision_rule": "FAKE si vidéo OU audio détecte FAKE",
                "video_analysis": {
                    **video_result,
                    "confidence_pct": video_confidence,
                },
                "audio_analysis": audio_result.get("result") if audio_result and audio_result.get("success") else None,
            },
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


# ── Point d'entrée ────────────────────────────────────────────────────────────
def run_model(media_type: str, file_path: str) -> dict:
    if media_type == "IMAGE":
        return run_image_model_with_confidence(file_path)
    elif media_type == "VIDEO":
        return run_video_with_audio(file_path)
    elif media_type == "AUDIO":
        return run_audio_model(file_path)
    else:
        return {"success": False, "error": f"Type non supporté: {media_type}"}