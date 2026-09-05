import os
import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from sqlalchemy.orm import Session

from app.db.deps import get_db
from app.api.deps import get_current_user
from app.models.media_file import MediaFile
from app.models.user import User
from app.schemas.media import MediaUploadResponse

router = APIRouter(prefix="/media", tags=["Media"])

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


@router.get("/")
def list_media(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    rows = (
        db.query(MediaFile, User)
        .join(User, MediaFile.uploaded_by == User.id, isouter=True)
        .order_by(MediaFile.uploaded_at.desc())
        .all()
    )
    return [
        {
            "id": str(m.id),
            "original_name": m.original_name,
            "media_type": m.media_type,
            "file_size": m.file_size,
            "status": m.status,
            "uploaded_at": m.uploaded_at,
            "uploaded_by": u.full_name if u else "—",
        }
        for m, u in rows
    ]


@router.post("/upload", response_model=MediaUploadResponse)
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
            uploaded_by=current_user.id
        )

        db.add(media)
        db.commit()
        db.refresh(media)

        return MediaUploadResponse(
            id=str(media.id),
            original_name=media.original_name,
            stored_name=media.stored_name,
            media_type=media.media_type,
            storage_path=media.storage_path,
            status=media.status
        )

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        print("UPLOAD ERROR:", repr(e))
        raise HTTPException(status_code=500, detail=str(e))