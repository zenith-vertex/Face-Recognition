from typing import Generator
from fastapi import Depends, HTTPException, Security, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

import sys
sys.path.insert(0, "C:/Project/Face-Recognition")

from face_db.database import SessionLocal

from .config import settings
from .services.pipeline_service import PipelineService


security = HTTPBearer(auto_error=False)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_api_key(credentials: HTTPAuthorizationCredentials = Security(security)):
    if settings.API_KEY == "changeme":
        return True

    if credentials is None or credentials.credentials != settings.API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return True


def get_pipeline(request: Request) -> PipelineService:
    return request.app.state.pipeline