from datetime import datetime
from typing import Optional
from pydantic import BaseModel


class Bbox(BaseModel):
    x: int
    y: int
    w: int
    h: int


class RecognitionResult(BaseModel):
    bbox: Bbox
    matched_person_id: Optional[int] = None
    matched_name: Optional[str] = None
    confidence: float
    decision: str


class RegisterFaceResponse(BaseModel):
    person_id: int
    name: str
    embeddings_stored: int


class PersonResponse(BaseModel):
    id: int
    name: str
    role: str
    is_active: bool
    registration_date: datetime
    embedding_count: int = 0


class RecognitionLogResponse(BaseModel):
    id: int
    timestamp: datetime
    matched_person_id: Optional[int] = None
    matched_person_name: str
    confidence_score: float
    metric_used: str
    detector_used: Optional[str] = None
    embedder_used: Optional[str] = None
    decision: str
    source_camera: Optional[str] = None
    frame_reference: Optional[str] = None


class StreamStartRequest(BaseModel):
    camera_id: str
    source: str
    sample_rate_fps: float = 1.0


class StreamStartResponse(BaseModel):
    camera_id: str
    status: str


class StreamStatusResponse(BaseModel):
    camera_id: str
    source: str
    status: str
    uptime_seconds: float
    frames_processed: int
    started_at: Optional[datetime] = None


class StopStreamRequest(BaseModel):
    camera_id: str