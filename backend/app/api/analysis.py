import os
import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, UploadFile, File, HTTPException
from sqlalchemy.orm import Session

from app.db.deps import get_db
from app.api.deps import get_current_user
from app.models.media_file import MediaFile
from app.models.analysis_run import AnalysisRun
from app.models.user import User
from app.services.ml_service import run_model

router = APIRouter(prefix="/analysis", tags=["Analysis"])

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


def detect_media_type(filename: str) -> str:
    ext = Path(filename).suffix.lower()
    if ext in [".jpg", ".jpeg", ".png", ".webp", ".jfif"]:
        return "IMAGE"
    if ext in [".mp4", ".avi", ".mov", ".mkv"]:
        return "VIDEO"
    if ext in [".mp3", ".wav", ".flac"]:
        return "AUDIO"
    raise HTTPException(status_code=400, detail="Unsupported file type")


@router.post("/upload")
def upload_media(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    try:
        media_type = detect_media_type(file.filename)
        file_ext = Path(file.filename).suffix
        stored_name = f"{uuid.uuid4()}{file_ext}"
        saved_path = UPLOAD_DIR / stored_name

        with open(saved_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        size = os.path.getsize(saved_path)

        media = MediaFile(
            original_name=file.filename,
            stored_name=stored_name,
            media_type=media_type,
            storage_path=str(saved_path.resolve()),
            mime_type=file.content_type,
            file_size=size,
            status="PENDING",
            uploaded_by=current_user.id  # ✅
        )

        db.add(media)
        db.commit()
        db.refresh(media)

        return {
            "id": str(media.id),
            "original_name": media.original_name,
            "stored_name": media.stored_name,
            "media_type": media.media_type,
            "storage_path": media.storage_path,
            "status": media.status,
        }

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        print("UPLOAD ERROR:", repr(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{media_id}/run")
def run_analysis(
    media_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    media = db.query(MediaFile).filter(MediaFile.id == media_id).first()

    if not media:
        raise HTTPException(status_code=404, detail="Media not found")

    result = run_model(media.media_type, media.storage_path)

    if not result.get("success"):
        media.status = "FAILED"
        db.commit()
        return result

    result_data = result["result"]

    # ✅ Extraire le score selon le type de résultat
    raw_score = (
        result_data.get("score")
        or result_data.get("prob_fake")
        or (result_data.get("video_analysis") or {}).get("prob_fake")
        or 0.0
    )

    analysis = AnalysisRun(
        media_file_id=media.id,
        model_name=result_data.get("model_name"),
        label=result_data.get("label"),
        score=str(raw_score),  # ✅ jamais None
        result_json=result_data,
        status="COMPLETED",
        created_by=current_user.id
    )

    media.status = "ANALYZED"
    db.add(analysis)
    db.commit()
    db.refresh(analysis)

    return result

@router.get("/history")
def get_history(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    UploaderAlias = User.__table__.alias("uploader")

    rows = (
        db.query(AnalysisRun, MediaFile, User)
        .join(MediaFile, AnalysisRun.media_file_id == MediaFile.id)
        .join(User, AnalysisRun.created_by == User.id, isouter=True)
        .order_by(AnalysisRun.created_at.desc())
        .all()
    )

    # Pour récupérer le nom de l'uploader séparément
    result = []
    for r, m, analyzer in rows:
        uploader = db.query(User).filter(User.id == m.uploaded_by).first()
        result.append({
            "id": str(r.id),
            "media_file_id": str(r.media_file_id),
            "original_name": m.original_name,
            "media_type": m.media_type,
            "model_name": r.model_name,
            "label": r.label,
            "score": f"{float(r.score):.1f}%" if r.score else "—",            "status": r.status,
            "created_at": r.created_at,
            "uploaded_by": uploader.full_name if uploader else "—",   # ✅ nom uploader
            "analyzed_by": analyzer.full_name if analyzer else "—",   # ✅ nom analyzer
        })

    return result