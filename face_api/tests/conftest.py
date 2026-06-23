import pytest
import sys
import os

sys.path.insert(0, "C:/Project/Face-Recognition")

from unittest.mock import patch, MagicMock

from face_api.app.main import create_app
from face_api.app.services.pipeline_service import PipelineService


@pytest.fixture
def client():
    with patch('face_api.app.services.pipeline_service.PipelineService.initialize'):
        app = create_app()
        from fastapi.testclient import TestClient
        yield TestClient(app)


@pytest.fixture
def sample_face_image():
    import numpy as np
    import cv2
    img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    _, buffer = cv2.imencode('.jpg', img)
    return buffer.tobytes()


@pytest.fixture
def mock_pipeline():
    pipeline = MagicMock(spec=PipelineService)
    import numpy as np
    pipeline.detect_faces.return_value = [{"bbox": (10, 10, 100, 100), "confidence": 0.9, "latency_ms": 10}]
    pipeline.crop_face.return_value = np.random.randint(0, 255, (120, 120, 3), dtype=np.uint8)
    pipeline.embed_face.return_value = np.random.randn(128).astype(np.float32)
    pipeline.validate_embedding_dimension.return_value = True
    pipeline.resize_image.return_value = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    pipeline.detector_name = "yolov8"
    pipeline.embedder_name = "dlib"
    return pipeline