from fastapi import APIRouter
from app.api import auth, users, consent, faces, recognize, logs, audit

api_router = APIRouter()
api_router.include_router(auth.router)
api_router.include_router(users.router)
api_router.include_router(consent.router)
api_router.include_router(faces.router)
api_router.include_router(recognize.router)
api_router.include_router(logs.router)
api_router.include_router(audit.router)
