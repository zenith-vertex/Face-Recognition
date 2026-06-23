import pytest
import numpy as np
import cv2
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

import sys
sys.path.insert(0, "C:/Project/Face-Recognition")

from face_api.app.main import create_app
from face_api.app.config import settings
from face_api.app.services.pipeline_service import PipelineService


@pytest.fixture
def client():
    app = create_app()
    return TestClient(app)


@pytest.fixture
def sample_face_image():
    img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    _, buffer = cv2.imencode('.jpg', img)
    return buffer.tobytes()


@pytest.fixture
def sample_multi_face_image():
    img = np.random.randint(0, 255, (448, 448, 3), dtype=np.uint8)
    _, buffer = cv2.imencode('.jpg', img)
    return buffer.tobytes()


@pytest.fixture
def mock_pipeline():
    pipeline = MagicMock(spec=PipelineService)
    pipeline.detect_faces.return_value = [{"bbox": (10, 10, 100, 100), "confidence": 0.9, "latency_ms": 10}]
    pipeline.crop_face.return_value = np.random.randint(0, 255, (120, 120, 3), dtype=np.uint8)
    pipeline.embed_face.return_value = np.random.randn(128).astype(np.float32)
    pipeline.validate_embedding_dimension.return_value = True
    pipeline.detector_name = "yolov8"
    pipeline.embedder_name = "dlib"
    return pipeline


class TestRegisterFace:
    def test_register_face_no_images(self, client):
        response = client.post("/persons/register-face", data={"name": "John Doe", "role": "employee"})
        assert response.status_code == 422

    def test_register_face_single_face(self, client, sample_face_image, mock_pipeline):
        with patch('face_api.app.routers.persons.cv2.imdecode') as mock_decode, \
             patch('face_api.app.routers.persons.create_person') as mock_create_person, \
             patch('face_api.app.routers.persons.add_face_embedding') as mock_add_emb, \
             patch('face_api.app.dependencies.get_pipeline', return_value=mock_pipeline):

            mock_decode.return_value = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            mock_create_person.return_value = 1
            mock_add_emb.return_value = 1

            response = client.post(
                "/persons/register-face",
                data={"name": "John Doe", "role": "employee"},
                files={"files": ("test.jpg", sample_face_image, "image/jpeg")}
            )
            assert response.status_code == 200
            assert response.json()["name"] == "John Doe"
            assert response.json()["person_id"] == 1


class TestRecognize:
    def test_recognize_no_faces(self, client, sample_face_image, mock_pipeline):
        with patch('face_api.app.routers.recognize.cv2.imdecode') as mock_decode, \
             patch('face_api.app.dependencies.get_pipeline', return_value=mock_pipeline):

            mock_decode.return_value = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

            response = client.post(
                "/recognize",
                files={"file": ("test.jpg", sample_face_image, "image/jpeg")}
            )
            assert response.status_code == 200
            assert response.json() == []

    def test_recognize_with_match(self, client, sample_face_image, mock_pipeline):
        with patch('face_api.app.routers.recognize.cv2.imdecode') as mock_decode, \
             patch('face_api.app.dependencies.get_pipeline', return_value=mock_pipeline), \
             patch('face_api.app.routers.recognize.find_nearest_match') as mock_find:

            mock_frame = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            mock_decode.return_value = mock_frame
            mock_find.return_value = [{"person_id": 1, "person_name": "John Doe", "distance": 0.3}]

            response = client.post(
                "/recognize",
                files={"file": ("test.jpg", sample_face_image, "image/jpeg")}
            )
            assert response.status_code == 200
            data = response.json()
            assert len(data) == 1
            assert data[0]["decision"] == "match"
            assert data[0]["matched_person_id"] == 1
            assert data[0]["matched_name"] == "John Doe"

    def test_recognize_no_match(self, client, sample_face_image, mock_pipeline):
        with patch('face_api.app.routers.recognize.cv2.imdecode') as mock_decode, \
             patch('face_api.app.dependencies.get_pipeline', return_value=mock_pipeline), \
             patch('face_api.app.routers.recognize.find_nearest_match') as mock_find:

            mock_frame = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            mock_decode.return_value = mock_frame
            mock_find.return_value = []

            response = client.post(
                "/recognize",
                files={"file": ("test.jpg", sample_face_image, "image/jpeg")}
            )
            assert response.status_code == 200
            data = response.json()
            assert len(data) == 1
            assert data[0]["decision"] == "no_match"


class TestLogs:
    def test_get_logs_filtering(self, client):
        with patch('face_api.app.routers.logs.get_recognition_logs') as mock_logs:
            mock_logs.return_value = [
                {
                    "id": 1,
                    "timestamp": "2026-01-01T00:00:00",
                    "matched_person_id": 1,
                    "matched_person_name": "John Doe",
                    "confidence_score": 0.85,
                    "metric_used": "cosine",
                    "detector_used": "yolov8",
                    "embedder_used": "dlib",
                    "decision": "match",
                    "source_camera": "cam1"
                }
            ]

            response = client.get("/logs?person_id=1&decision=match&limit=50")
            assert response.status_code == 200
            assert len(response.json()) == 1


class TestStreams:
    def test_start_stream_already_running(self, client, mock_pipeline):
        with patch('face_api.app.dependencies.get_pipeline', return_value=mock_pipeline), \
             patch('face_api.app.routers.streams.active_streams', {'cam1': MagicMock(status="running")}):
            response = client.post(
                "/streams/start",
                json={"camera_id": "cam1", "source": "webcam", "sample_rate_fps": 1.0}
            )
            assert response.status_code == 409

    def test_stop_stream_not_found(self, client):
        response = client.post(
            "/streams/stop",
            json={"camera_id": "nonexistent"}
        )
        assert response.status_code == 404

    def test_get_stream_status(self, client):
        response = client.get("/streams/status")
        assert response.status_code == 200
        assert response.json() == []

    def test_start_stop_stream_lifecycle(self, client, mock_pipeline):
        with patch('face_api.app.dependencies.get_pipeline', return_value=mock_pipeline), \
             patch('face_api.app.routers.streams.StreamWorker') as MockWorker:

            mock_worker_instance = MagicMock()
            mock_worker_instance.status = "running"
            mock_worker_instance.get_status.return_value = {
                "camera_id": "cam1",
                "source": "webcam",
                "status": "running",
                "uptime_seconds": 5.0,
                "frames_processed": 10,
                "started_at": "2026-01-01T00:00:00"
            }
            MockWorker.return_value = mock_worker_instance

            response = client.post(
                "/streams/start",
                json={"camera_id": "cam1", "source": "webcam", "sample_rate_fps": 1.0}
            )
            assert response.status_code == 200
            assert response.json()["status"] == "started"

            response = client.post(
                "/streams/stop",
                json={"camera_id": "cam1"}
            )
            assert response.status_code == 200


class TestWebSocket:
    def test_websocket_connection(self, client):
        with client.websocket_connect("/streams/ws/live/test_cam") as websocket:
            websocket.send_text("ping")
            websocket.receive_text()


    def test_stream_reconnect_on_failure(self, client, mock_pipeline):
        with patch('face_api.app.dependencies.get_pipeline', return_value=mock_pipeline), \
             patch('face_api.app.services.stream_worker.cv2.VideoCapture') as MockCapture, \
             patch('face_api.app.services.stream_worker.StreamWorker.start') as mock_start:

            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = True
            mock_cap.read.side_effect = [
                (True, np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)),
                (False, None),
            ]
            MockCapture.return_value = mock_cap

            mock_start.return_value = None

            import time
            import threading
            with patch('time.sleep') as mock_sleep:
                mock_sleep.return_value = None
                client.post(
                    "/streams/start",
                    json={"camera_id": "cam1", "source": "webcam", "sample_rate_fps": 1.0}
                )