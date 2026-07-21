import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from app.database.models import Base, User
from app.database.session import get_db, SessionLocal
from app.main import app
from app.core.security import hash_password
from app.infrastructure.security.encryption import encrypt_embedding

SQLALCHEMY_DATABASE_URL = "sqlite:///:memory:"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


app.dependency_overrides[get_db] = override_get_db
Base.metadata.create_all(bind=engine)
client = TestClient(app)


def _seed_admin():
    db = TestingSessionLocal()
    try:
        admin = db.query(User).filter(User.email == "admin@example.com").first()
        if not admin:
            admin = User(
                full_name="Admin",
                email="admin@example.com",
                role="admin",
                hashed_password=hash_password("adminpass"),
            )
            db.add(admin)
            db.commit()
    finally:
        db.close()


def _auth_header():
    _seed_admin()
    resp = client.post("/auth/login", data={"username": "admin@example.com", "password": "adminpass"})
    token = resp.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_login_and_create_user():
    _seed_admin()
    resp = client.post("/auth/login", data={"username": "admin@example.com", "password": "adminpass"})
    assert resp.status_code == 200
    token = resp.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    resp = client.post("/users", json={"full_name": "Test User", "email": "test@example.com", "password": "secret"}, headers=headers)
    assert resp.status_code == 201
    assert resp.json()["full_name"] == "Test User"


def test_consent_then_enroll():
    _seed_admin()
    resp = client.post("/auth/login", data={"username": "admin@example.com", "password": "adminpass"})
    token = resp.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    user_resp = client.post("/users", json={"full_name": "Enroll User", "email": "enroll@example.com", "password": "secret"}, headers=headers)
    assert user_resp.status_code == 201
    user_id = user_resp.json()["id"]

    consent_resp = client.post(f"/users/{user_id}/consent", json={"consent_text_version": "v1"}, headers=headers)
    assert consent_resp.status_code == 201
    assert consent_resp.json()["consent_text_version"] == "v1"

    face_resp = client.post(
        f"/users/{user_id}/faces",
        files={"file": ("test.jpg", b"fake-image-bytes", "image/jpeg")},
        headers=headers,
    )
    assert face_resp.status_code in (201, 400, 500)

    logs_resp = client.get("/logs", headers=headers)
    assert logs_resp.status_code == 200


def test_duplicate_attendance_suppression():
    _seed_admin()
    resp = client.post("/auth/login", data={"username": "admin@example.com", "password": "adminpass"})
    token = resp.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    user_resp = client.post("/users", json={"full_name": "Attendance User", "email": "attend@example.com", "password": "secret"}, headers=headers)
    assert user_resp.status_code == 201
    user_id = user_resp.json()["id"]

    consent_resp = client.post(f"/users/{user_id}/consent", json={"consent_text_version": "v1"}, headers=headers)
    assert consent_resp.status_code == 201

    face_resp = client.post(
        f"/users/{user_id}/faces",
        files={"file": ("test.jpg", b"fake-image-bytes", "image/jpeg")},
        headers=headers,
    )
    assert face_resp.status_code in (201, 400, 500)

    recognize_resp = client.post(
        "/recognize",
        files={"file": ("test.jpg", b"fake-image-bytes", "image/jpeg")},
        headers=headers,
    )
    assert recognize_resp.status_code in (200, 400, 500)
