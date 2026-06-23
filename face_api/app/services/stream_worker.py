import time
import cv2
import threading
import asyncio
from datetime import datetime
from typing import Optional, Callable, Dict, Any

import sys
sys.path.insert(0, "C:/Project/Face-Recognition")

from face_db.repository import find_nearest_match, log_recognition


class StreamWorker:
    def __init__(self, camera_id: str, source: str, sample_rate_fps: float,
                 pipeline_service, websocket_manager, threshold: float = 0.6):
        self.camera_id = camera_id
        self.source = source
        self.sample_rate_fps = sample_rate_fps
        self.pipeline_service = pipeline_service
        self.websocket_manager = websocket_manager
        self.threshold = threshold

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.frames_processed = 0
        self.started_at: Optional[datetime] = None
        self.status = "stopped"
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def _open_capture(self) -> cv2.VideoCapture:
        if self.source == "webcam":
            return cv2.VideoCapture(0)
        elif self.source.isdigit():
            return cv2.VideoCapture(int(self.source))
        else:
            return cv2.VideoCapture(self.source)

    def _run(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self.status = "starting"
        self.frames_processed = 0

        backoff_delay = 1.0
        max_backoff = 30.0
        cap = None

        try:
            while not self._stop_event.is_set():
                try:
                    cap = self._open_capture()
                    if not cap.isOpened():
                        raise RuntimeError(f"Cannot open video source: {self.source}")

                    self.status = "running"
                    self.started_at = datetime.now()
                    backoff_delay = 1.0

                    frame_interval = 1.0 / self.sample_rate_fps
                    last_frame_time = time.time()

                    while not self._stop_event.is_set() and cap.isOpened():
                        current_time = time.time()
                        if current_time - last_frame_time < frame_interval:
                            time.sleep(0.01)
                            continue

                        ret, frame = cap.read()
                        if not ret:
                            raise RuntimeError("Failed to read frame from stream")

                        last_frame_time = current_time
                        self.frames_processed += 1

                        frame = self.pipeline_service.resize_image(frame)
                        detections = self.pipeline_service.detect_faces(frame)

                        for det in detections:
                            bbox = det["bbox"]
                            x, y, w, h = bbox
                            latency_ms = det.get("latency_ms", 0)

                            face_crop = self.pipeline_service.crop_face(frame, bbox)
                            query_emb = self.pipeline_service.embed_face(face_crop)
                            self.pipeline_service.validate_embedding_dimension(query_emb)

                            nearest = find_nearest_match(query_emb, self.pipeline_service.embedder_name, "cosine", 1)

                            if nearest:
                                match_info = nearest[0]
                                matched_person_id = match_info["person_id"]
                                confidence = 1.0 - match_info["distance"]
                                decision = "match"
                                matched_name = match_info["person_name"]
                            else:
                                matched_person_id = None
                                confidence = 0.0
                                decision = "no_match"
                                matched_name = None

                            log_recognition(
                                matched_person_id=matched_person_id,
                                confidence_score=confidence,
                                metric_used="cosine",
                                decision=decision,
                                detector_used=self.pipeline_service.detector_name,
                                embedder_used=self.pipeline_service.embedder_name,
                                source_camera=self.camera_id,
                                raw_bbox={"x": x, "y": y, "w": w, "h": h}
                            )

                            ws_message = {
                                "camera_id": self.camera_id,
                                "frame_number": self.frames_processed,
                                "bbox": {"x": x, "y": y, "w": w, "h": h},
                                "matched_person_id": matched_person_id,
                                "matched_name": matched_name,
                                "confidence": confidence,
                                "decision": decision,
                            }

                            asyncio.run_coroutine_threadsafe(
                                self.websocket_manager.broadcast(self.camera_id, ws_message),
                                self._loop
                            )

                except Exception as e:
                    self.status = "reconnecting"
                    print(f"Stream {self.camera_id} error: {e}. Retrying in {backoff_delay}s...")

                    time.sleep(backoff_delay)
                    backoff_delay = min(backoff_delay * 2, max_backoff)

                    if self._stop_event.is_set():
                        break

                    if cap:
                        cap.release()
                        cap = None

        finally:
            if cap:
                cap.release()
            self.status = "stopped"
            self._loop.close()

    def start(self):
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError(f"Stream already running for camera_id: {self.camera_id}")

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        if self._thread is None or not self._thread.is_alive():
            raise RuntimeError(f"No active stream for camera_id: {self.camera_id}")

        self._stop_event.set()
        self._thread.join(timeout=5.0)
        self._thread = None

    def get_status(self) -> Dict[str, Any]:
        uptime = 0.0
        if self.started_at and self.status == "running":
            uptime = (datetime.now() - self.started_at).total_seconds()

        return {
            "camera_id": self.camera_id,
            "source": self.source,
            "status": self.status,
            "uptime_seconds": uptime,
            "frames_processed": self.frames_processed,
            "started_at": self.started_at.isoformat() if self.started_at else None,
        }