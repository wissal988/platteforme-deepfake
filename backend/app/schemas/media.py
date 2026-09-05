from pydantic import BaseModel
from typing import Optional


class MediaUploadResponse(BaseModel):
    id: str
    original_name: str
    stored_name: str
    media_type: str
    storage_path: str
    status: str

    class Config:
        from_attributes = True


class AnalysisResponse(BaseModel):
    success: bool
    result: Optional[dict] = None
    error: Optional[str] = None