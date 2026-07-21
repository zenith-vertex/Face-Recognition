from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from app.database.session import get_db
from app.database.models import User
from app.presentation.schemas import UserCreate, UserRead
from app.core.auth import get_current_user, require_role
from app.core.security import hash_password

router = APIRouter(prefix="/users", tags=["users"])


@router.post("", response_model=UserRead, status_code=status.HTTP_201_CREATED)
def create_user(payload: UserCreate, db: Session = Depends(get_db), _: User = Depends(require_role("admin", "instructor"))):
    existing = db.query(User).filter(User.email == payload.email).first()
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")
    user = User(
        full_name=payload.full_name,
        email=payload.email,
        department=payload.department,
        role=payload.role.value,
        hashed_password=hash_password(payload.password),
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return UserRead(
        id=str(user.id),
        full_name=user.full_name,
        email=user.email,
        department=user.department,
        role=user.role,
        created_at=user.created_at,
    )


@router.get("", response_model=list[UserRead])
def list_users(db: Session = Depends(get_db), _: User = Depends(require_role("admin", "instructor"))):
    users = db.query(User).all()
    return [
        UserRead(
            id=str(u.id),
            full_name=u.full_name,
            email=u.email,
            department=u.department,
            role=u.role,
            created_at=u.created_at,
        )
        for u in users
    ]


@router.get("/{user_id}", response_model=UserRead)
def get_user(user_id: str, db: Session = Depends(get_db), _: User = Depends(get_current_user)):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return UserRead(
        id=str(user.id),
        full_name=user.full_name,
        email=user.email,
        department=user.department,
        role=user.role,
        created_at=user.created_at,
    )
