from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from app.database.session import get_db
from app.infrastructure.repositories.repositories import ConsentRepository
from app.presentation.schemas import ConsentCreate, ConsentRead
from app.core.auth import get_current_user, require_role

router = APIRouter(prefix="/users/{user_id}/consent", tags=["consent"])


@router.post("", response_model=ConsentRead, status_code=status.HTTP_201_CREATED)
def grant_consent(user_id: str, payload: ConsentCreate, db: Session = Depends(get_db), _: User = Depends(require_role("admin", "instructor"))):
    repo = ConsentRepository(db)
    existing = repo.get_active_for_user(user_id)
    if existing:
        raise HTTPException(status_code=400, detail="Active consent already exists")
    consent = repo.grant(user_id, payload.consent_text_version)
    return ConsentRead(
        id=str(consent.id),
        user_id=str(consent.user_id),
        consent_text_version=consent.consent_text_version,
        granted_at=consent.granted_at,
        revoked_at=consent.revoked_at,
    )


@router.delete("", status_code=status.HTTP_204_NO_CONTENT)
def revoke_consent(user_id: str, db: Session = Depends(get_db), _: User = Depends(require_role("admin", "instructor"))):
    repo = ConsentRepository(db)
    if not repo.revoke(user_id):
        raise HTTPException(status_code=404, detail="No active consent found")
    return None
