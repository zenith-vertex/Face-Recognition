from sqlalchemy import (
    Column,
    String,
    Text,
    TIMESTAMP,
    Boolean,
    ForeignKey,
    func,
    Float,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from app.database.session import Base
import uuid


class User(Base):
    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    full_name = Column(Text, nullable=False)
    email = Column(Text, unique=True, nullable=True)
    department = Column(Text, nullable=True)
    role = Column(String(50), nullable=False, default="user")
    hashed_password = Column(Text, nullable=False)
    created_at = Column(TIMESTAMP(timezone=True), nullable=False, server_default=func.now())

    consents = relationship("BiometricConsent", back_populates="user", cascade="all, delete-orphan")
    face_embeddings = relationship("FaceEmbedding", back_populates="user", cascade="all, delete-orphan")
    recognition_logs = relationship("RecognitionLog", back_populates="user")


class BiometricConsent(Base):
    __tablename__ = "biometric_consents"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    consent_text_version = Column(Text, nullable=False)
    granted_at = Column(TIMESTAMP(timezone=True), nullable=False, server_default=func.now())
    revoked_at = Column(TIMESTAMP(timezone=True), nullable=True)

    user = relationship("User", back_populates="consents")


class FaceEmbedding(Base):
    __tablename__ = "face_embeddings"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    embedding = Column(Text, nullable=False)
    model_version = Column(Text, nullable=False)
    created_at = Column(TIMESTAMP(timezone=True), nullable=False, server_default=func.now())

    user = relationship("User", back_populates="face_embeddings")


class RecognitionLog(Base):
    __tablename__ = "recognition_logs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    camera_id = Column(Text, nullable=True)
    similarity = Column(Float, nullable=True)
    matched_at = Column(TIMESTAMP(timezone=True), nullable=False, server_default=func.now())

    user = relationship("User", back_populates="recognition_logs")


class BiometricAccessAudit(Base):
    __tablename__ = "biometric_access_audit"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    actor_user_id = Column(UUID(as_uuid=True), nullable=True)
    subject_user_id = Column(UUID(as_uuid=True), nullable=True)
    action = Column(Text, nullable=False)
    occurred_at = Column(TIMESTAMP(timezone=True), nullable=False, server_default=func.now())
