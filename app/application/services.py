from app.infrastructure.recognition.detector import FaceDetector
from app.infrastructure.repositories.repositories import (
    ConsentRepository,
    FaceEmbeddingRepository,
    RecognitionLogRepository,
    AuditRepository,
)
from app.core.config import settings
from app.domain.entities import AuditAction
import cv2
import numpy as np
import uuid


class RegisterFace:
    def __init__(self, db, detector: FaceDetector, consent_repo: ConsentRepository, embedding_repo: FaceEmbeddingRepository, audit_repo: AuditRepository):
        self.db = db
        self.detector = detector
        self.consent_repo = consent_repo
        self.embedding_repo = embedding_repo
        self.audit_repo = audit_repo

    def execute(self, user_id: str, image_bytes: bytes) -> dict:
        consent = self.consent_repo.get_active_for_user(user_id)
        if not consent:
            raise PermissionError("Valid biometric consent required before enrollment")

        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Invalid image data")

        embedding = self.detector.extract_embedding(image)
        if embedding is None:
            raise ValueError("No face detected in image")

        model_version = settings.INSIGHTFACE_MODEL
        record = self.embedding_repo.add_embedding(user_id, embedding, model_version)
        self.audit_repo.record(action=AuditAction.ENROLL, subject_user_id=user_id)

        return {"embedding_id": str(record.id), "model_version": model_version}


class RecognizeFace:
    def __init__(self, db, detector: FaceDetector, matcher, embedding_repo: FaceEmbeddingRepository, log_repo: RecognitionLogRepository, audit_repo: AuditRepository):
        self.db = db
        self.detector = detector
        self.matcher = matcher
        self.embedding_repo = embedding_repo
        self.log_repo = log_repo
        self.audit_repo = audit_repo

    def execute(self, image_bytes: bytes, camera_id: str = "default") -> dict:
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Invalid image data")

        embedding = self.detector.extract_embedding(image)
        if embedding is None:
            self.audit_repo.record(action=AuditAction.MATCH_ATTEMPT)
            return {"match": None, "message": "No face detected"}

        match = self.matcher.match(embedding)
        self.audit_repo.record(action=AuditAction.MATCH_ATTEMPT, subject_user_id=match["user_id"] if match else None)

        if match:
            existing = self.log_repo.get_today_for_user(match["user_id"])
            if existing:
                return {
                    "match": match,
                    "message": "Attendance already recorded today",
                    "duplicate": True,
                }
            self.log_repo.add(user_id=match["user_id"], camera_id=camera_id, similarity=match["similarity"])
            return {"match": match, "message": "Attendance recorded", "duplicate": False}
        return {"match": None, "message": "No match above threshold", "duplicate": False}
