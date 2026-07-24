"""Use case for managing registered users."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from ..core.domain.exceptions import UserNotFoundError
from ..core.domain.user import User
from ..core.ports.repositories import (
    FaceEncodingRepository,
    LogRepository,
    UserRepository,
)
from ..core.ports.storage_port import EmbeddingStoragePort

logger = logging.getLogger(__name__)


@dataclass
class ListUsersOutput:
    """Output from listing users."""

    users: list[User]
    total: int


@dataclass
class DeleteUserOutput:
    """Output from deleting a user."""

    deleted: bool
    user_id: int
    user_name: str
    message: str


class ManageUsersUseCase:
    """Use case for managing registered users."""

    def __init__(
        self,
        user_repo: UserRepository,
        face_encoding_repo: FaceEncodingRepository,
        log_repo: LogRepository,
        embedding_storage: EmbeddingStoragePort,
    ) -> None:
        self.user_repo = user_repo
        self.face_encoding_repo = face_encoding_repo
        self.log_repo = log_repo
        self.embedding_storage = embedding_storage

    def list_users(self) -> ListUsersOutput:
        """List all registered users.

        Returns:
            Output with list of users.
        """
        users = self.user_repo.list_all()
        return ListUsersOutput(users=users, total=len(users))

    def get_user(self, identifier: str) -> User | None:
        """Retrieve a user by name or ID.

        Args:
            identifier: User name or numeric ID.

        Returns:
            User entity if found, None otherwise.
        """
        try:
            user_id = int(identifier)
            return self.user_repo.get_by_id(user_id)
        except ValueError:
            return self.user_repo.get_by_name(identifier)

    def delete_user(self, identifier: str) -> DeleteUserOutput:
        """Delete a user and all associated data.

        Args:
            identifier: User name or numeric ID.

        Returns:
            Delete result output.

        Raises:
            UserNotFoundError: If user is not found.
        """
        user = self.get_user(identifier)
        if not user:
            raise UserNotFoundError(
                f"User '{identifier}' not found"
            )

        face_encodings = self.face_encoding_repo.get_by_user_id(user.id)
        encoding_paths = [fe.encoding_path for fe in face_encodings]
        self.embedding_storage.delete_user_encodings(user.id)
        self.face_encoding_repo.delete_by_user_id(user.id)
        deleted = self.user_repo.delete(user.id)

        message = (
            f"Deleted user '{user.name}' (ID: {user.id}) and "
            f"{len(encoding_paths)} face encoding(s)"
        )
        logger.info(message)
        return DeleteUserOutput(
            deleted=deleted,
            user_id=user.id,
            user_name=user.name,
            message=message,
        )

    def get_user_face_count(self, identifier: str) -> int | None:
        """Get the number of face encodings for a user.

        Args:
            identifier: User name or numeric ID.

        Returns:
            Number of face encodings, or None if user not found.
        """
        user = self.get_user(identifier)
        if not user:
            return None
        return self.face_encoding_repo.count_by_user_id(user.id)
