import pytest
import numpy as np
from app.infrastructure.recognition.matcher import FaceMatcher


def test_low_light_variant_similarity():
    matcher = FaceMatcher(threshold=0.6)
    matcher.load_enrolled([
        ("user-1", "Alice", [1.0, 0.0, 0.0]),
    ])
    query = np.array([0.95, 0.05, 0.0], dtype=np.float32)
    result = matcher.match(query)
    assert result is not None
    assert result["similarity"] > 0.6


def test_side_angle_variant_similarity():
    matcher = FaceMatcher(threshold=0.5)
    matcher.load_enrolled([
        ("user-1", "Alice", [0.8, 0.2, 0.0]),
    ])
    query = np.array([0.7, 0.3, 0.0], dtype=np.float32)
    result = matcher.match(query)
    assert result is not None


def test_mask_occlusion_lowers_similarity():
    matcher = FaceMatcher(threshold=0.8)
    matcher.load_enrolled([
        ("user-1", "Alice", [1.0, 0.0, 0.0]),
    ])
    query = np.array([0.3, 0.4, 0.3], dtype=np.float32)
    result = matcher.match(query)
    assert result is None


def test_multiple_candidates_returns_best():
    matcher = FaceMatcher(threshold=0.0)
    matcher.load_enrolled([
        ("user-1", "Alice", [0.9, 0.1, 0.0]),
        ("user-2", "Bob", [0.1, 0.9, 0.0]),
        ("user-3", "Carol", [0.5, 0.5, 0.0]),
    ])
    query = np.array([0.95, 0.05, 0.0], dtype=np.float32)
    top = matcher.match_top_k(query, k=3)
    assert top[0]["user_id"] == "user-1"
    assert top[1]["user_id"] == "user-3"
    assert top[2]["user_id"] == "user-2"
