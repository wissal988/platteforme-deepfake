from typing import Optional
from sqlalchemy.orm import Session
from app.models.audit_log import AuditLog


def create_audit_log(
    db: Session,
    user_id=None,
    action: str = "",
    entity_type: Optional[str] = None,
    entity_id=None,
    metadata: Optional[dict] = None
):
    log = AuditLog(
        user_id=user_id,
        action=action,
        entity_type=entity_type,
        entity_id=entity_id,
        details=metadata
    )
    db.add(log)
    db.commit()
    db.refresh(log)
    return log