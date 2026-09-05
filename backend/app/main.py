from fastapi import FastAPI
from sqlalchemy import text
from app.db.session import engine, Base
from app.models.user import User
from app.models.audit_log import AuditLog
from app.models.media_file import MediaFile
from app.models.analysis_run import AnalysisRun
from app.api.auth import router as auth_router
from app.api.users import router as users_router
from app.api.audit_logs import router as audit_logs_router
from app.api.media import router as media_router
from app.api.analysis import router as analysis_router
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Deepfake Detection API")

# ✅ CORS doit être AVANT les routes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

Base.metadata.create_all(bind=engine)

app.include_router(auth_router)
app.include_router(users_router)
app.include_router(audit_logs_router)
app.include_router(media_router)
app.include_router(analysis_router)

@app.get("/")
def root():
    return {"message": "API is running 🚀"}

@app.get("/health/db")
def check_db():
    with engine.connect() as connection:
        connection.execute(text("SELECT 1"))
    return {"database": "connected"}