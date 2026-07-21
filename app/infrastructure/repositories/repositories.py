from typing import List, Tuple, Optional
from sqlalchemy.orm import Session
from app.database.models import User, BiometricConsent, FaceEmbedding, RecognitionLog, BiometricAccessAudit
from app.domain.entities import AuditAction
from datetime import datetime, timezone
import uuid
import numpy as np
from app.infrastructure.security.encryption import encrypt_embedding, decrypt_embedding


class UserRepository:
    def __init__(self, db: Session):
        self.db = db

    def get_by_id(self, user_id: str) -> Optional[User]:
        return self.db.query(User).filter(User.id == uuid.UUID(user_id)).first()

    def get_by_email(self, email: str) -> Optional[User]:
        return self.db.query(User).filter(User.email == email).first()

    def create(self, full_name: str, email: str = None, department: str = None, role: str = "user", hashed_password: str = "") -> User:
        user = User(full_name=full_name, email=email, department=department, role=role, hashed_password=hashed_password)
        self.db.add(user)
        self.db.commit()
        self.db.refresh(user)
        return user

    def list_all(self) -> List[User]:
        return self.db.query(User).all()


class ConsentRepository:
    def __init__(self, db: Session):
        self.db = db

    def get_active_for_user(self, user_id: str) -> Optional[BiometricConsent]:
        return (
            self.db.query(BiometricConsent)
            .filter(
                BiometricConsent.user_id == uuid.UUID(user_id),
                BiometricConsent.revoked_at.is_(None),
            )
            .first()
        )

    def grant(self, user_id: str, consent_text_version: str) -> BiometricConsent:
        consent = BiometricConsent(
            user_id=uuid.UUID(user_id),
            consent_text_version=consent_text_version,
        )
        self.db.add(consent)
        self.db.commit()
        self.db.refresh(consent)
        return consent

    def revoke(self, user_id: str) -> bool:
        consent = self.get_active_for_user(user_id)
        if not consent:
            return False
        consent.revoked_at = datetime.now(timezone.utc)
        self.db.commit()
        return True


class FaceEmbeddingRepository:
    def __init__(self, db: Session):
        self.db = db

    def add_embedding(self, user_id: str, embedding, model_version: str) -> FaceEmbedding:
        if isinstance(embedding, np.ndarray):
            embedding = embedding.astype(float).tolist()
        encrypted = encrypt_embedding(embedding)
        emb = FaceEmbedding(
            user_id=uuid.UUID(user_id),
            embedding=encrypted,
            model_version=model_version,
        )
        self.db.add(emb)
        self.db.commit()
        self.db.refresh(emb)
        return emb

    def get_embeddings_for_user(self, user_id: str) -> List[FaceEmbedding]:
        return self.db.query(FaceEmbedding).filter(FaceEmbedding.user_id == uuid.UUID(user_id)).all()

    def get_all_enrolled(self) -> List[Tuple[str, str, list]]:
        rows = (
            self.db.query(FaceEmbedding, User)
            .join(User, FaceEmbedding.user_id == User.id)
            .all()
        )
        results = []
        for fe, u in rows:
            try:
                emb = decrypt_embedding(fe.embedding)
                results.append((str(fe.user_id), u.full_name, emb))
            except Exception:
                continue
        return results

    def delete_for_user(self, user_id: str) -> int:
        count = self.db.query(FaceEmbedding).filter(FaceEmbedding.user_id == uuid.UUID(user_id)).delete()
        self.db.commit()
        return count


class RecognitionLogRepository:
    def __init__(self, db: Session):
        self.db = db

    def add(self, user_id: str = None, camera_id: str = None, similarity: float = None) -> RecognitionLog:
        log = RecognitionLog(
            user_id=uuid.UUID(user_id) if user_id else None,
            camera_id=camera_id,
            similarity=similarity,
        )
        self.db.add(log)
        self.db.commit()
        self.db.refresh(log)
        return log

    def get_today_for_user(self, user_id: str) -> List[RecognitionLog]:
        from sqlalchemy import func as sa_func
        today_start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        return (
            self.db.query(RecognitionLog)
            .filter(
                RecognitionLog.user_id == uuid.UUID(user_id),
                RecognitionLog.matched_at >= today_start,
            )
            .all()
        )

    def list_logs(self, user_id: str = None, start: datetime = None, end: datetime = None, limit: int = 100) -> List[RecognitionLog]:
        q = self.db.query(RecognitionLog)
        if user_id:
            q = q.filter(RecognitionLog.user_id == uuid.UUID(user_id))
        if start:
            q = q.filter(RecognitionLog.matched_at >= start)
        if end:
            q = q.filter(RecognitionLog.matched_at <= end)
        return q.order_by(RecognitionLog.matched_at.desc()).limit(limit).all()


class AuditRepository:
    def __init__(self, db: Session):
        self.db = db

    def record(self, actor_user_id: str = None, subject_user_id: str = None, action: AuditAction = AuditAction.MATCH_ATTEMPT):
        entry = BiometricAccessAudit(
            actor_user_id=uuid.UUID(actor_user_id) if actor_user_id else None,
            subject_user_id=uuid.UUID(subject_user_id) if subject_user_id else None,
            action=action.value,
        )
        self.db.add(entry)
        self.db.commit()
        self.db.refresh(entry)
        return entry
