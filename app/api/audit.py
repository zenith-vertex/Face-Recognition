from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from app.database.session import get_db
from app.database.models import BiometricAccessAudit, User
from app.presentation.schemas import AuditLogRead
from app.core.auth import get_current_user, require_role

router = APIRouter(prefix="/audit", tags=["audit"])


@router.get("", response_model=list[AuditLogRead])
def list_audit(
    subject_user_id: str = Query(None),
    db: Session = Depends(get_db),
    _: User = Depends(require_role("admin")),
):
    q = db.query(BiometricAccessAudit)
    if subject_user_id:
        q = q.filter(BiometricAccessAudit.subject_user_id == subject_user_id)
    rows = q.order_by(BiometricAccessAudit.occurred_at.desc()).limit(200).all()
    return [
        AuditLogRead(
            id=str(r.id),
            actor_user_id=str(r.actor_user_id) if r.actor_user_id else None,
            subject_user_id=str(r.subject_user_id) if r.subject_user_id else None,
            action=r.action,
            occurred_at=r.occurred_at,
        )
        for r in rows
    ]
