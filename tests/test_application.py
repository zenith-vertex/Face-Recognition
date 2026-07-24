"""Tests for application use cases."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.application.export_logs import (
    ExportLogsInput,
    ExportLogsUseCase,
)
from src.application.manage_users import (
    ManageUsersUseCase,
)
from src.application.recognize import (
    RecognizeFaceUseCase,
    RecognizeInput,
)
from src.application.train_model import (
    TrainModelInput,
    TrainModelUseCase,
)
from src.core.domain.face_data import RecognitionEvent
from src.core.domain.user import User


class TestManageUsersUseCase:
    def test_list_users(self, tmp_path: Path) -> None:
        mock_user_repo = MagicMock()
        mock_user_repo.list_all.return_value = [
            User(id=1, name="Alice"),
            User(id=2, name="Bob"),
        ]

        use_case = ManageUsersUseCase(
            user_repo=mock_user_repo,
            face_encoding_repo=MagicMock(),
            log_repo=MagicMock(),
            embedding_storage=MagicMock(),
        )

        result = use_case.list_users()
        assert result.total == 2
        assert len(result.users) == 2
        assert result.users[0].name == "Alice"
        assert result.users[1].name == "Bob"

    def test_get_user_by_name(self, tmp_path: Path) -> None:
        mock_user_repo = MagicMock()
        mock_user_repo.get_by_name.return_value = User(id=1, name="Alice")

        use_case = ManageUsersUseCase(
            user_repo=mock_user_repo,
            face_encoding_repo=MagicMock(),
            log_repo=MagicMock(),
            embedding_storage=MagicMock(),
        )

        user = use_case.get_user("Alice")
        assert user is not None
        assert user.name == "Alice"

    def test_get_user_not_found(self, tmp_path: Path) -> None:
        mock_user_repo = MagicMock()
        mock_user_repo.get_by_name.return_value = None
        mock_user_repo.get_by_id.return_value = None

        use_case = ManageUsersUseCase(
            user_repo=mock_user_repo,
            face_encoding_repo=MagicMock(),
            log_repo=MagicMock(),
            embedding_storage=MagicMock(),
        )

        user = use_case.get_user("DoesNotExist")
        assert user is None

    def test_delete_user(self, tmp_path: Path) -> None:
        mock_user_repo = MagicMock()
        mock_user_repo.get_by_name.return_value = User(id=1, name="Alice")
        mock_user_repo.delete.return_value = True

        mock_face_encoding_repo = MagicMock()
        mock_face_encoding_repo.get_by_user_id.return_value = []

        mock_log_repo = MagicMock()
        mock_embedding_storage = MagicMock()

        use_case = ManageUsersUseCase(
            user_repo=mock_user_repo,
            face_encoding_repo=mock_face_encoding_repo,
            log_repo=mock_log_repo,
            embedding_storage=mock_embedding_storage,
        )

        result = use_case.delete_user("Alice")
        assert result.user_name == "Alice"
        assert result.user_id == 1
        assert result.deleted is True
        mock_embedding_storage.delete_user_encodings.assert_called_once_with(1)


class TestExportLogsUseCase:
    def test_export_csv(self, tmp_path: Path) -> None:
        mock_log_repo = MagicMock()
        mock_log_repo.get_recent.return_value = [
            RecognitionEvent(
                id=1,
                event_type="recognized",
                user_id=1,
                confidence=0.95,
                image_path=tmp_path / "img.jpg",
                timestamp=datetime.now(),
            ),
        ]

        use_case = ExportLogsUseCase(log_repo=mock_log_repo)

        output_path = tmp_path / "export.csv"
        input_data = ExportLogsInput(output_path=output_path, format="csv")
        result = use_case.execute(input_data)

        assert result.event_count == 1
        assert output_path.exists()
        mock_log_repo.get_recent.assert_called_once_with(limit=1000)

    def test_export_unsupported_format_raises(self, tmp_path: Path) -> None:
        use_case = ExportLogsUseCase(log_repo=MagicMock())
        output_path = tmp_path / "export.xml"
        input_data = ExportLogsInput(output_path=output_path, format="xml")

        with pytest.raises(ValueError, match="Unsupported format"):
            use_case.execute(input_data)


class TestTrainModelUseCase:
    def test_train_no_users(self, tmp_path: Path) -> None:
        mock_user_repo = MagicMock()
        mock_user_repo.list_all.return_value = []

        mock_face_encoding_repo = MagicMock()
        mock_embedding_storage = MagicMock()

        use_case = TrainModelUseCase(
            user_repo=mock_user_repo,
            face_encoding_repo=mock_face_encoding_repo,
            embedding_storage=mock_embedding_storage,
        )

        result = use_case.execute(TrainModelInput())
        assert result.users_trained == 0
        assert result.total_encodings == 0
        assert "No users" in result.message


class TestRecognizeUseCase:
    def test_no_known_encodings_returns_message(self, tmp_path: Path) -> None:
        mock_user_repo = MagicMock()
        mock_face_encoding_repo = MagicMock()
        mock_face_encoding_repo.get_all_encodings.return_value = []
        mock_log_repo = MagicMock()

        use_case = RecognizeFaceUseCase(
            user_repo=mock_user_repo,
            face_encoding_repo=mock_face_encoding_repo,
            log_repo=mock_log_repo,
            face_recognizer=MagicMock(),
            camera=MagicMock(),
            embedding_storage=MagicMock(),
        )

        input_data = RecognizeInput(source="camera")
        result = use_case.execute(input_data)
        assert "No trained encodings" in result.message
