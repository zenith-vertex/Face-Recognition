"""Custom exceptions for the face recognition system."""

from __future__ import annotations


class FaceRecognitionError(Exception):
    """Base exception for face recognition system errors."""

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)


class UserAlreadyExistsError(FaceRecognitionError):
    """Raised when attempting to register a user that already exists."""

    def __init__(self, message: str) -> None:
        super().__init__(message)


class NoFacesDetectedError(FaceRecognitionError):
    """Raised when no faces are detected in an image."""

    def __init__(self, message: str) -> None:
        super().__init__(message)


class InsufficientFaceDataError(FaceRecognitionError):
    """Raised when there is not enough face data to proceed."""

    def __init__(self, message: str) -> None:
        super().__init__(message)


class DatabaseError(FaceRecognitionError):
    """Raised when a database operation fails."""

    def __init__(self, message: str) -> None:
        super().__init__(message)


class CameraUnavailableError(FaceRecognitionError):
    """Raised when the camera is not available."""

    def __init__(self, message: str) -> None:
        super().__init__(message)


class RecognitionEngineError(FaceRecognitionError):
    """Raised when a face recognition engine operation fails."""

    def __init__(self, message: str) -> None:
        super().__init__(message)


class ImageProcessingError(FaceRecognitionError):
    """Raised when an image cannot be processed."""

    def __init__(self, message: str) -> None:
        super().__init__(message)


class UserNotFoundError(FaceRecognitionError):
    """Raised when a user is not found in the system."""

    def __init__(self, message: str) -> None:
        super().__init__(message)


class StorageError(FaceRecognitionError):
    """Raised when a file storage operation fails."""

    def __init__(self, message: str) -> None:
        super().__init__(message)


class ConfigurationError(FaceRecognitionError):
    """Raised when configuration is invalid or missing."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
