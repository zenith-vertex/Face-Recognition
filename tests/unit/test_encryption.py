import os
import pytest
import numpy as np
from app.infrastructure.recognition.matcher import cosine_similarity, FaceMatcher
from app.infrastructure.security.encryption import encrypt_embedding, decrypt_embedding


@pytest.fixture(autouse=True)
def _encryption_key(monkeypatch):
    from cryptography.fernet import Fernet
    monkeypatch.setenv("EMBEDDING_ENCRYPTION_KEY", Fernet.generate_key().decode())


def test_encryption_roundtrip():
    embedding = [0.1, 0.2, 0.3, 0.4, 0.5] * 100
    token = encrypt_embedding(embedding)
    recovered = decrypt_embedding(token)
    assert len(recovered) == len(embedding)
    for a, b in zip(embedding, recovered):
        assert abs(a - b) < 1e-6


def test_encryption_produces_different_tokens():
    embedding = [0.1, 0.2, 0.3]
    token1 = encrypt_embedding(embedding)
    token2 = encrypt_embedding(embedding)
    assert token1 != token2


def test_match_top_k_ordering():
    matcher = FaceMatcher(threshold=0.0)
    matcher.load_enrolled([
        ("user-1", "Alice", [1.0, 0.0, 0.0]),
        ("user-2", "Bob", [0.0, 1.0, 0.0]),
        ("user-3", "Carol", [0.5, 0.5, 0.0]),
    ])
    query = np.array([0.9, 0.1, 0.0], dtype=np.float32)
    top = matcher.match_top_k(query, k=2)
    assert len(top) == 2
    assert top[0]["user_id"] == "user-1"
    assert top[1]["user_id"] == "user-3"
