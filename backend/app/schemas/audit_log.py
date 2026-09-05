from uuid import UUID
from datetime import datetime
from typing import Optional, Any
from pydantic import BaseModel, ConfigDict


class AuditLogResponse(BaseModel):
    id: UUID
    user_id: Optional[UUID] = None
    action: str
    entity_type: Optional[str] = None
    entity_id: Optional[UUID] = None
    details: Optional[Any] = None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)