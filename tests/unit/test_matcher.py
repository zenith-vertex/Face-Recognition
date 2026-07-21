import numpy as np
from app.infrastructure.recognition.matcher import cosine_similarity, FaceMatcher


def test_cosine_similarity_identical():
    a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    b = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    assert abs(cosine_similarity(a, b) - 1.0) < 1e-6


def test_cosine_similarity_orthogonal():
    a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    b = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    assert abs(cosine_similarity(a, b)) < 1e-6


def test_face_matcher_threshold_gate():
    matcher = FaceMatcher(threshold=0.8)
    matcher.load_enrolled([
        ("user-1", "Alice", [1.0, 0.0, 0.0]),
    ])
    query = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    result = matcher.match(query)
    assert result is None


def test_face_matcher_match_above_threshold():
    matcher = FaceMatcher(threshold=0.5)
    matcher.load_enrolled([
        ("user-1", "Alice", [1.0, 0.0, 0.0]),
    ])
    query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    result = matcher.match(query)
    assert result is not None
    assert result["user_id"] == "user-1"
    assert result["full_name"] == "Alice"
    assert "similarity" in result
