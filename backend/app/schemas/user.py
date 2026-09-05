from uuid import UUID
from typing import Optional
from pydantic import BaseModel, EmailStr, ConfigDict


class AdminCreate(BaseModel):
    full_name: str
    email: EmailStr
    password: str


class AdminUpdate(BaseModel):
    full_name: Optional[str] = None
    email: Optional[EmailStr] = None
    password: Optional[str] = None


class AdminResponse(BaseModel):
    id: UUID
    full_name: str
    email: EmailStr
    role: str
    is_active: bool

    model_config = ConfigDict(from_attributes=True)