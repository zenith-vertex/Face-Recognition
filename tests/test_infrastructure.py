"""Tests for infrastructure components."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from core.domain.exceptions import DatabaseError
from infrastructure.embedding_storage import (
    FileSystemEmbeddingStorage,
)
from infrastructure.sqlite_repository import SQLiteRepository


class TestFileSystemEmbeddingStorage:
    def test_save_and_load_encoding(self, tmp_path: Path) -> None:
        storage = FileSystemEmbeddingStorage(base_dir=tmp_path)
        encoding = np.random.rand(128).astype(np.float64)

        saved_path = storage.save_encoding(user_id=1, encoding=encoding)
        assert saved_path.exists()

        loaded = storage.load_encoding(saved_path)
        np.testing.assert_array_almost_equal(encoding, loaded)

    def test_delete_encoding(self, tmp_path: Path) -> None:
        storage = FileSystemEmbeddingStorage(base_dir=tmp_path)
        encoding = np.random.rand(128).astype(np.float64)
        saved_path = storage.save_encoding(user_id=1, encoding=encoding)

        storage.delete_encoding(saved_path)
        assert not saved_path.exists()

    def test_load_all_empty(self, tmp_path: Path) -> None:
        storage = FileSystemEmbeddingStorage(base_dir=tmp_path)
        results = storage.load_all_encodings()
        assert results == []

    def test_delete_user_encodings(self, tmp_path: Path) -> None:
        storage = FileSystemEmbeddingStorage(base_dir=tmp_path)
        for i in range(3):
            storage.save_encoding(user_id=1, encoding=np.random.rand(128))

        deleted = storage.delete_user_encodings(user_id=1)
        assert len(deleted) > 0
        for p in tmp_path.glob("user_1_face*.npy"):
            assert not p.exists()


class TestSQLiteRepository:
    def test_create_and_get_user(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test.db"
        repo = SQLiteRepository(database_path=db_path)

        user = repo.create(name="Test User")
        assert user.id > 0
        assert user.name == "Test User"

        fetched = repo.get_by_id(user.id)
        assert fetched is not None
        assert fetched.name == "Test User"

    def test_get_by_name_case_insensitive(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test.db"
        repo = SQLiteRepository(database_path=db_path)
        repo.create(name="Test User")

        found = repo.get_by_name("TEST USER")
        assert found is not None
        assert found.name == "Test User"

    def test_list_and_delete_user(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test.db"
        repo = SQLiteRepository(database_path=db_path)
        user = repo.create(name="ToDelete")

        users = repo.list_all()
        assert len(users) == 1

        deleted = repo.delete(user.id)
        assert deleted is True

        users_after = repo.list_all()
        assert len(users_after) == 0

    def test_duplicate_user_raises(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test.db"
        repo = SQLiteRepository(database_path=db_path)
        repo.create(name="Duplicate")

        with pytest.raises(DatabaseError, match="already exists"):
            repo.create(name="Duplicate")

    def test_face_encoding_crud(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test.db"
        repo = SQLiteRepository(database_path=db_path)
        user = repo.create(name="WithFaces")

        from pathlib import Path as _Path
        encoding_path = _Path("data/encodings/test.npy")
        image_path = _Path("data/captures/test.jpg")
        _Path("data/encodings").mkdir(parents=True, exist_ok=True)
        _Path("data/captures").mkdir(parents=True, exist_ok=True)
        encoding_path.touch()
        image_path.touch()

        fe = repo.add_encoding(
            user_id=user.id,
            encoding_path=encoding_path,
            image_path=image_path,
        )
        assert fe.user_id == user.id

        face_encodings = repo.get_by_user_id(user.id)
        assert len(face_encodings) == 1
        assert face_encodings[0].encoding_path == encoding_path

        count = repo.count_by_user_id(user.id)
        assert count == 1

        deleted_count = repo.delete_by_user_id(user.id)
        assert deleted_count == 1

        remaining = repo.get_by_user_id(user.id)
        assert len(remaining) == 0

        encoding_path.unlink()
        image_path.unlink()
