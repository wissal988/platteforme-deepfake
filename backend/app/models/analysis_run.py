import uuid
from sqlalchemy import Column, String, DateTime, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func
from sqlalchemy.dialects.postgresql import JSONB
from app.db.session import Base


class AnalysisRun(Base):
    __tablename__ = "analysis_runs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    media_file_id = Column(UUID(as_uuid=True), ForeignKey("media_files.id"), nullable=False)
    model_name = Column(String(255), nullable=True)
    label = Column(String(20), nullable=True)
    score = Column(String(50), nullable=True)
    result_json = Column(JSONB, nullable=True)
    status = Column(String(20), nullable=False, default="COMPLETED")
    created_by = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())