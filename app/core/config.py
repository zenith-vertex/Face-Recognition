from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    DATABASE_URL: str = "postgresql://faceuser:facepass@localhost:5432/facerecognition"
    SECRET_KEY: str = "change-me-in-production-use-a-real-secret"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    INSIGHTFACE_MODEL: str = "buffalo_l"
    FACE_DETECTION_SIZE: int = 640
    SIMILARITY_THRESHOLD: float = 0.65
    UPLOAD_DIR: str = "uploads"
    STORAGE_TYPE: str = "local"
    S3_BUCKET: str = ""
    S3_ENDPOINT: str = ""
    S3_ACCESS_KEY: str = ""
    S3_SECRET_KEY: str = ""
    RATE_LIMIT_DEFAULT: str = "60/minute"
    RATE_LIMIT_RECOGNIZE: str = "20/minute"
    EMBEDDING_CACHE_TTL_SECONDS: int = 300
    EMBEDDING_ENCRYPTION_KEY: str = ""


settings = Settings()
