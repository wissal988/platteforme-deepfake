from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from app.db.deps import get_db
from app.models.user import User
from app.schemas.user import AdminCreate, AdminUpdate, AdminResponse
from app.core.security import hash_password
from app.api.deps import require_super_admin
from app.services.audit_service import create_audit_log

router = APIRouter(prefix="/admins", tags=["Admins"])


@router.get("/", response_model=list[AdminResponse])
def list_admins(
    db: Session = Depends(get_db),
    current_user: User = Depends(require_super_admin)
):
    return db.query(User).filter(User.role == "ADMIN").order_by(User.full_name).all()


@router.post("/", response_model=AdminResponse)
def create_admin(
    payload: AdminCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_super_admin)
):
    existing_user = db.query(User).filter(User.email == payload.email).first()
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already exists"
        )
    admin = User(
        full_name=payload.full_name,
        email=payload.email,
        password_hash=hash_password(payload.password),
        role="ADMIN",
        is_active=True
    )
    db.add(admin)
    db.commit()
    db.refresh(admin)
    create_audit_log(
        db=db,
        user_id=current_user.id,
        action="CREATE_ADMIN",
        entity_type="USER",
        entity_id=admin.id,
        metadata={"full_name": admin.full_name, "email": admin.email, "role": admin.role}
    )
    return admin


@router.patch("/{admin_id}", response_model=AdminResponse)
def update_admin(
    admin_id: str,
    payload: AdminUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_super_admin)
):
    admin = db.query(User).filter(User.id == admin_id, User.role == "ADMIN").first()
    if not admin:
        raise HTTPException(status_code=404, detail="Admin not found")
    if payload.full_name is not None:
        admin.full_name = payload.full_name
    if payload.email is not None:
        existing = db.query(User).filter(User.email == payload.email, User.id != admin_id).first()
        if existing:
            raise HTTPException(status_code=400, detail="Email already exists")
        admin.email = payload.email
    if payload.password is not None:
        admin.password_hash = hash_password(payload.password)
    db.commit()
    db.refresh(admin)
    return admin


@router.patch("/{admin_id}/activate", response_model=AdminResponse)
def activate_admin(
    admin_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_super_admin)
):
    admin = db.query(User).filter(User.id == admin_id, User.role == "ADMIN").first()
    if not admin:
        raise HTTPException(status_code=404, detail="Admin not found")
    admin.is_active = True
    db.commit()
    db.refresh(admin)
    return admin


@router.patch("/{admin_id}/deactivate", response_model=AdminResponse)
def deactivate_admin(
    admin_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_super_admin)
):
    admin = db.query(User).filter(User.id == admin_id, User.role == "ADMIN").first()
    if not admin:
        raise HTTPException(status_code=404, detail="Admin not found")
    admin.is_active = False
    db.commit()
    db.refresh(admin)
    return admin


@router.delete("/{admin_id}", status_code=204)
def delete_admin(
    admin_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_super_admin)
):
    admin = db.query(User).filter(User.id == admin_id, User.role == "ADMIN").first()
    if not admin:
        raise HTTPException(status_code=404, detail="Admin not found")
    db.delete(admin)
    db.commit()