from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.db.deps import get_db
from app.api.deps import require_super_admin
from app.models.user import User
from app.models.audit_log import AuditLog
from app.schemas.audit_log import AuditLogResponse

router = APIRouter(prefix="/audit-logs", tags=["Audit Logs"])


@router.get("/", response_model=list[AuditLogResponse])
def list_audit_logs(
    db: Session = Depends(get_db),
    current_user: User = Depends(require_super_admin)
):
    logs = db.query(AuditLog).order_by(AuditLog.created_at.desc()).all()
    return logs