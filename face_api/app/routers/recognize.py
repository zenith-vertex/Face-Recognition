from typing import List, Optional
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
import numpy as np
import cv2

import sys
sys.path.insert(0, "C:/Project/Face-Recognition")

from face_db.repository import find_nearest_match, log_recognition

from ..config import settings
from ..dependencies import get_pipeline, get_api_key
from ..schemas import RecognitionResult
from ..services.pipeline_service import PipelineService

router = APIRouter(
    tags=["recognition"]
)


def _process_recognition(frame: np.ndarray, pipeline: PipelineService, source_camera: Optional[str] = None) -> list:
    frame = pipeline.resize_image(frame)
    detections = pipeline.detect_faces(frame)

    if len(detections) == 0:
        return []

    results = []

    for det in detections:
        bbox = det["bbox"]
        face_crop = pipeline.crop_face(frame, bbox)
        query_emb = pipeline.embed_face(face_crop)
        pipeline.validate_embedding_dimension(query_emb)

        nearest = find_nearest_match(query_emb, settings.ACTIVE_EMBEDDER, "cosine", 1)

        if nearest:
            match_info = nearest[0]
            matched_person_id = match_info["person_id"]
            confidence = 1.0 - match_info["distance"]
            matched_name = match_info["person_name"]
            decision = "match"
        else:
            matched_person_id = None
            confidence = 0.0
            matched_name = None
            decision = "no_match"

        log_recognition(
            matched_person_id=matched_person_id,
            confidence_score=confidence,
            metric_used="cosine",
            decision=decision,
            detector_used=settings.ACTIVE_DETECTOR,
            embedder_used=settings.ACTIVE_EMBEDDER,
            source_camera=source_camera,
            raw_bbox={"x": bbox[0], "y": bbox[1], "w": bbox[2], "h": bbox[3]}
        )

        results.append(RecognitionResult(
            bbox={"x": bbox[0], "y": bbox[1], "w": bbox[2], "h": bbox[3]},
            matched_person_id=matched_person_id,
            matched_name=matched_name,
            confidence=confidence,
            decision=decision
        ))

    return results


@router.post("/recognize", response_model=List[RecognitionResult])
async def recognize(
    file: UploadFile = File(...),
    source_camera: Optional[str] = Form(None),
    pipeline: PipelineService = Depends(get_pipeline),
    _: bool = Depends(get_api_key)
):
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None or frame.size == 0:
            raise HTTPException(status_code=400, detail="Invalid or corrupt image file")

        results = await run_in_threadpool(_process_recognition, frame, pipeline, source_camera)
        return results

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")