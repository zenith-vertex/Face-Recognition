"""Presentation module containing CLI interface."""

from __future__ import annotations

from ..application.capture_images import CaptureImagesUseCase
from ..application.export_logs import ExportLogsUseCase
from ..application.manage_users import ManageUsersUseCase
from ..application.recognize import RecognizeFaceUseCase
from ..application.register_user import RegisterUserUseCase
from ..application.train_model import TrainModelUseCase
from ..core.ports.camera_port import CameraPort
from ..core.ports.face_recognizer_port import FaceRecognizerPort
from ..core.ports.repositories import (
    FaceEncodingRepository,
    LogRepository,
    UserRepository,
)
from ..core.ports.storage_port import EmbeddingStoragePort
from ..infrastructure.camera_adapter import OpenCVCameraAdapter
from ..infrastructure.config import Config
from ..infrastructure.embedding_storage import FileSystemEmbeddingStorage
from ..infrastructure.face_recognition_engine import FaceRecognitionEngine
from ..infrastructure.sqlite_repository import SQLiteRepository


class ServiceContainer:
    """Container for all application services."""

    def __init__(
        self,
        config: Config,
        user_repo: UserRepository,
        face_encoding_repo: FaceEncodingRepository,
        log_repo: LogRepository,
        face_recognizer: FaceRecognizerPort,
        camera: CameraPort,
        embedding_storage: EmbeddingStoragePort,
    ) -> None:
        self.config = config
        self.user_repo = user_repo
        self.face_encoding_repo = face_encoding_repo
        self.log_repo = log_repo
        self.face_recognizer = face_recognizer
        self.camera = camera
        self.embedding_storage = embedding_storage

        self.register_user_use_case = RegisterUserUseCase(
            user_repo=user_repo,
            face_encoding_repo=face_encoding_repo,
            face_recognizer=face_recognizer,
        )
        self.capture_images_use_case = CaptureImagesUseCase(
            user_repo=user_repo,
            face_encoding_repo=face_encoding_repo,
            camera=camera,
            face_recognizer=face_recognizer,
            embedding_storage=embedding_storage,
        )
        self.train_model_use_case = TrainModelUseCase(
            user_repo=user_repo,
            face_encoding_repo=face_encoding_repo,
            embedding_storage=embedding_storage,
        )
        self.recognize_use_case = RecognizeFaceUseCase(
            user_repo=user_repo,
            face_encoding_repo=face_encoding_repo,
            log_repo=log_repo,
            face_recognizer=face_recognizer,
            camera=camera,
            embedding_storage=embedding_storage,
        )
        self.manage_users_use_case = ManageUsersUseCase(
            user_repo=user_repo,
            face_encoding_repo=face_encoding_repo,
            log_repo=log_repo,
            embedding_storage=embedding_storage,
        )
        self.export_logs_use_case = ExportLogsUseCase(log_repo=log_repo)


def create_services(config: Config) -> ServiceContainer:
    """Create all required services with their implementations.

    Args:
        config: Application configuration.

    Returns:
        A ServiceContainer with all services instantiated.
    """
    db_repo = SQLiteRepository(database_path=config.database_path)
    face_recognizer = FaceRecognitionEngine()
    camera = OpenCVCameraAdapter(
        source=config.camera_source,
        width=config.camera_width,
        height=config.camera_height,
    )
    embedding_storage = FileSystemEmbeddingStorage(base_dir=config.encodings_dir)

    return ServiceContainer(
        config=config,
        user_repo=db_repo,
        face_encoding_repo=db_repo,
        log_repo=db_repo,
        face_recognizer=face_recognizer,
        camera=camera,
        embedding_storage=embedding_storage,
    )


def create_bootstrap(config_path=None):
    """Create a service container from an optional config path.

    This is used for testing and direct invocation.
    """
    from ..infrastructure.config import Config
    config = Config(env_file=config_path)
    return create_services(config)
