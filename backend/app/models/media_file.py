import uuid
from sqlalchemy import Column, String, DateTime, BigInteger, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func
from app.db.session import Base


class MediaFile(Base):
    __tablename__ = "media_files"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    original_name = Column(String(255), nullable=False)
    stored_name = Column(String(255), nullable=False)
    media_type = Column(String(20), nullable=False)
    storage_path = Column(String, nullable=False)
    mime_type = Column(String(100), nullable=True)
    file_size = Column(BigInteger, nullable=True)
    uploaded_by = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)
    status = Column(String(20), nullable=False, default="PENDING")
    uploaded_at = Column(DateTime(timezone=True), server_default=func.now())