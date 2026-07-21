from app.infrastructure.recognition.detector import FaceDetector
from app.infrastructure.recognition.matcher import FaceMatcher
from app.infrastructure.recognition.cache import embedding_cache
from app.core.config import settings

_detector: FaceDetector | None = None
_matcher: FaceMatcher | None = None


def get_detector() -> FaceDetector:
    global _detector
    if _detector is None:
        _detector = FaceDetector()
    return _detector


def get_matcher() -> FaceMatcher:
    global _matcher
    if _matcher is None:
        _matcher = FaceMatcher()
    return _matcher


def get_cached_enrolled() -> list:
    return embedding_cache.get()


def warm_cache(records: list):
    embedding_cache.set(records)


def invalidate_cache():
    embedding_cache.invalidate()


def reset_matcher():
    global _matcher
    _matcher = None
