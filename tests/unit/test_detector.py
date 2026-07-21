import pytest
import numpy as np
from app.infrastructure.recognition.detector import FaceDetector


@pytest.fixture(scope="module")
def detector():
    return FaceDetector()


def test_detector_initialization(detector):
    assert detector.app is not None


def test_detector_no_face(detector):
    blank = np.zeros((100, 100, 3), dtype=np.uint8)
    faces = detector.detect(blank)
    assert len(faces) == 0
