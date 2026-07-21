from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Request
from sqlalchemy.orm import Session
from app.database.session import get_db
from app.infrastructure.recognition.pipeline import get_detector, get_matcher, get_cached_enrolled, warm_cache, invalidate_cache
from app.infrastructure.repositories.repositories import (
    FaceEmbeddingRepository,
    RecognitionLogRepository,
    AuditRepository,
)
from app.application.services import RecognizeFace
from app.presentation.schemas import RecognizeResponse, MatchResponse
from app.domain.entities import AuditAction
from app.core.config import settings
from slowapi import Limiter
from slowapi.util import get_remote_address
import asyncio

limiter = Limiter(key_func=get_remote_address)

router = APIRouter(prefix="/recognize", tags=["recognize"])


@router.post("", response_model=RecognizeResponse)
@limiter.limit(settings.RATE_LIMIT_RECOGNIZE)
async def recognize(
    request: Request,
    file: UploadFile = File(...),
    camera_id: str = "default",
    db: Session = Depends(get_db),
):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type; expected image")

    content = await file.read()

    detector = get_detector()
    matcher = get_matcher()

    enrolled = get_cached_enrolled()
    if not enrolled:
        emb_repo = FaceEmbeddingRepository(db)
        enrolled = emb_repo.get_all_enrolled()
        warm_cache(enrolled)
    matcher.load_enrolled(enrolled)

    use_case = RecognizeFace(
        db=db,
        detector=detector,
        matcher=matcher,
        embedding_repo=FaceEmbeddingRepository(db),
        log_repo=RecognitionLogRepository(db),
        audit_repo=AuditRepository(db),
    )
    try:
        result = await asyncio.to_thread(use_case.execute, content, camera_id)
        match = result.get("match")
        resp = RecognizeResponse(
            match=MatchResponse(**match) if match else None,
            message=result.get("message", ""),
            duplicate=result.get("duplicate", False),
        )
        return resp
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
