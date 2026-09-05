from sqlalchemy.orm import Session
from app.db.session import SessionLocal
from app.core.security import hash_password
from app.models.user import User


def create_super_admin():
    db: Session = SessionLocal()

    email = "maalemwissal@gmail.com"

    existing_user = db.query(User).filter(User.email == email).first()
    if existing_user:
        print("Super admin already exists.")
        db.close()
        return

    super_admin = User(
        full_name="Super Admin",
        email=email,
        password_hash=hash_password("wissal123"),
        role="SUPER_ADMIN",
        is_active=True
    )

    db.add(super_admin)
    db.commit()
    db.refresh(super_admin)
    db.close()

    print("Super admin created successfully.")
    print("Email: maalemwissal@gmail.com")
    print("Password: wissal123")


if __name__ == "__main__":
    create_super_admin()