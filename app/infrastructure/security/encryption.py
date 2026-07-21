import json
import os
from typing import List

from cryptography.fernet import Fernet


_fernet: Fernet | None = None


def _get_fernet() -> Fernet:
    global _fernet
    if _fernet is None:
        key = os.getenv("EMBEDDING_ENCRYPTION_KEY")
        if not key:
            raise RuntimeError("EMBEDDING_ENCRYPTION_KEY environment variable is not set")
        _fernet = Fernet(key.encode() if isinstance(key, str) else key)
    return _fernet


def encrypt_embedding(embedding: List[float]) -> str:
    data = json.dumps(embedding).encode()
    return _get_fernet().encrypt(data).decode()


def decrypt_embedding(token: str) -> List[float]:
    data = _get_fernet().decrypt(token.encode() if isinstance(token, str) else token)
    return json.loads(data.decode())
