from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from datetime import datetime, timezone
from app.database.session import get_db
from app.infrastructure.repositories.repositories import RecognitionLogRepository
from app.database.models import User
from app.presentation.schemas import RecognitionLogRead
from app.core.auth import get_current_user, require_role

router = APIRouter(prefix="/logs", tags=["logs"])


@router.get("", response_model=list[RecognitionLogRead])
def list_logs(
    user_id: str = Query(None),
    from_: str = Query(None, alias="from"),
    to: str = Query(None),
    db: Session = Depends(get_db),
    _: User = Depends(require_role("admin", "instructor")),
):
    start = datetime.fromisoformat(from_) if from_ else None
    end = datetime.fromisoformat(to) if to else None
    logs = RecognitionLogRepository(db).list_logs(user_id=user_id, start=start, end=end)
    return [
        RecognitionLogRead(
            id=str(l.id),
            user_id=str(l.user_id) if l.user_id else None,
            camera_id=l.camera_id,
            similarity=float(l.similarity) if l.similarity else None,
            matched_at=l.matched_at,
        )
        for l in logs
    ]
