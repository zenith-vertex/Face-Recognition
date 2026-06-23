import os
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    DATABASE_URL: str = "postgresql://postgres:password@localhost:5432/face_recognition_db"
    ACTIVE_DETECTOR: str = "yolov8"
    ACTIVE_EMBEDDER: str = "dlib"
    API_KEY: str = "changeme"
    MAX_IMAGE_DIMENSION: int = 1280
    MATCH_THRESHOLD: float = 0.6

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()