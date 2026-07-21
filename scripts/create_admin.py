#!/usr/bin/env python3
"""Create the first admin user."""
import sys
import uuid
from sqlalchemy.orm import Session
from app.database.session import SessionLocal
from app.database.models import User
from app.core.security import hash_password


def create_admin(email: str, password: str, full_name: str = "Admin"):
    db: Session = SessionLocal()
    try:
        existing = db.query(User).filter(User.email == email).first()
        if existing:
            print(f"User {email} already exists")
            return
        user = User(
            id=uuid.uuid4(),
            full_name=full_name,
            email=email,
            role="admin",
            hashed_password=hash_password(password),
        )
        db.add(user)
        db.commit()
        print(f"Admin user created: {email}")
    finally:
        db.close()


if __name__ == "__main__":
    email = sys.argv[1] if len(sys.argv) > 1 else "admin@example.com"
    password = sys.argv[2] if len(sys.argv) > 2 else "adminpass"
    create_admin(email, password)
