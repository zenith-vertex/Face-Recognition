from dataclasses import dataclass
from enum import Enum


class UserRole(str, Enum):
    ADMIN = "admin"
    INSTRUCTOR = "instructor"
    USER = "user"


class AuditAction(str, Enum):
    ENROLL = "enroll"
    MATCH_ATTEMPT = "match_attempt"
    DELETE = "delete"
    EXPORT = "export"


@dataclass
class MatchResult:
    user_id: str
    full_name: str
    similarity: float
    confidence: str


@dataclass
class ConsentRecord:
    user_id: str
    consent_text_version: str
    granted_at: str
    revoked_at: str | None = None
