import os
from datetime import datetime
from sqlalchemy import Column, Integer, String, Boolean, DateTime, ForeignKey, Float, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship
from pgvector.sqlalchemy import Vector
from database import Base


class Person(Base):
    __tablename__ = "persons"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(150), nullable=False)
    role = Column(String(50), nullable=False, default="unspecified")
    registration_date = Column(DateTime, nullable=False, default=datetime.utcnow)
    is_active = Column(Boolean, nullable=False, default=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)

    embeddings = relationship("FaceEmbedding", back_populates="person", cascade="all, delete-orphan")
    logs = relationship("RecognitionLog", back_populates="person", cascade="all, delete-orphan")


class FaceEmbedding(Base):
    __tablename__ = "face_embeddings"

    id = Column(Integer, primary_key=True, index=True)
    person_id = Column(Integer, ForeignKey("persons.id", ondelete="CASCADE"), nullable=False)
    embedder_model = Column(String(30), nullable=False)
    embedding = Column("embedding", Vector(128), nullable=False)
    source_image_path = Column(Text, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    person = relationship("Person", back_populates="embeddings")


class RecognitionLog(Base):
    __tablename__ = "recognition_logs"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow)
    matched_person_id = Column(Integer, ForeignKey("persons.id", ondelete="SET NULL"), nullable=True)
    confidence_score = Column(Float, nullable=False)
    metric_used = Column(String(20), nullable=False)
    detector_used = Column(String(20), nullable=True)
    embedder_used = Column(String(20), nullable=True)
    decision = Column(String(10), nullable=False)
    source_camera = Column(String(100), nullable=True)
    frame_reference = Column(Text, nullable=True)
    raw_bbox = Column(JSONB, nullable=True)

    person = relationship("Person", back_populates="logs")