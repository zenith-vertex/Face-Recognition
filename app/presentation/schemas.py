from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, List
from enum import Enum


class UserRole(str, Enum):
    ADMIN = "admin"
    INSTRUCTOR = "instructor"
    USER = "user"


class UserCreate(BaseModel):
    full_name: str
    email: Optional[str] = None
    department: Optional[str] = None
    role: UserRole = UserRole.USER
    password: str


class UserRead(BaseModel):
    id: str
    full_name: str
    email: Optional[str]
    department: Optional[str]
    role: UserRole
    created_at: datetime

    class Config:
        from_attributes = True


class ConsentCreate(BaseModel):
    consent_text_version: str


class ConsentRead(BaseModel):
    id: str
    user_id: str
    consent_text_version: str
    granted_at: datetime
    revoked_at: Optional[datetime]

    class Config:
        from_attributes = True


class FaceEnrollResponse(BaseModel):
    embedding_id: str
    model_version: str


class MatchResponse(BaseModel):
    user_id: Optional[str]
    full_name: Optional[str]
    similarity: Optional[float]


class RecognizeResponse(BaseModel):
    match: Optional[MatchResponse]
    message: str
    duplicate: bool = False


class RecognitionLogRead(BaseModel):
    id: str
    user_id: Optional[str]
    camera_id: Optional[str]
    similarity: Optional[float]
    matched_at: datetime

    class Config:
        from_attributes = True


class AuditLogRead(BaseModel):
    id: str
    actor_user_id: Optional[str]
    subject_user_id: Optional[str]
    action: str
    occurred_at: datetime

    class Config:
        from_attributes = True


class Token(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class TokenRefresh(BaseModel):
    refresh_token: str
