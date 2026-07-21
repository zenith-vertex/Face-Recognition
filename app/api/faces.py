from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File
from sqlalchemy.orm import Session
from app.database.session import get_db
from app.infrastructure.recognition.pipeline import get_detector, invalidate_cache
from app.infrastructure.repositories.repositories import (
    ConsentRepository,
    FaceEmbeddingRepository,
    AuditRepository,
)
from app.application.services import RegisterFace
from app.presentation.schemas import FaceEnrollResponse
from app.domain.entities import AuditAction
from app.core.auth import get_current_user, require_role
from app.core.config import settings
import asyncio

router = APIRouter(prefix="/users/{user_id}/faces", tags=["faces"])


@router.post("", response_model=FaceEnrollResponse, status_code=status.HTTP_201_CREATED)
async def enroll_face(user_id: str, file: UploadFile = File(...), db: Session = Depends(get_db), _: User = Depends(require_role("admin", "instructor"))):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type; expected image")

    content = await file.read()
    detector = get_detector()
    use_case = RegisterFace(
        db=db,
        detector=detector,
        consent_repo=ConsentRepository(db),
        embedding_repo=FaceEmbeddingRepository(db),
        audit_repo=AuditRepository(db),
    )
    try:
        result = await asyncio.to_thread(use_case.execute, user_id, content)
        invalidate_cache()
        return FaceEnrollResponse(**result)
    except PermissionError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))


@router.delete("", status_code=status.HTTP_204_NO_CONTENT)
def delete_faces(user_id: str, db: Session = Depends(get_db), _: User = Depends(require_role("admin"))):
    repo = FaceEmbeddingRepository(db)
    repo.delete_for_user(user_id)
    AuditRepository(db).record(action=AuditAction.DELETE, subject_user_id=user_id)
    invalidate_cache()
    return None
