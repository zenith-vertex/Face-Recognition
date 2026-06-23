from typing import Dict, Optional, List
from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from datetime import datetime

import sys
sys.path.insert(0, "C:/Project/Face-Recognition")

from ..config import settings
from ..dependencies import get_pipeline, get_api_key
from ..schemas import StreamStartRequest, StreamStartResponse, StreamStatusResponse, StopStreamRequest
from ..services.stream_worker import StreamWorker
from ..websocket_manager import websocket_manager

router = APIRouter(
    prefix="/streams",
    tags=["streams"]
)

active_streams: Dict[str, StreamWorker] = {}


@router.post("/start", response_model=StreamStartResponse)
async def start_stream(
    request: StreamStartRequest,
    pipeline = Depends(get_pipeline),
    _: bool = Depends(get_api_key)
):
    if request.camera_id in active_streams:
        worker = active_streams[request.camera_id]
        if worker.status in ("running", "starting"):
            raise HTTPException(status_code=409, detail=f"Stream already running for camera_id: {request.camera_id}")

    worker = StreamWorker(
        camera_id=request.camera_id,
        source=request.source,
        sample_rate_fps=request.sample_rate_fps,
        pipeline_service=pipeline,
        websocket_manager=websocket_manager,
        threshold=settings.MATCH_THRESHOLD
    )

    worker.start()
    active_streams[request.camera_id] = worker

    return StreamStartResponse(camera_id=request.camera_id, status="started")


@router.post("/stop")
async def stop_stream(
    request: StopStreamRequest,
    _: bool = Depends(get_api_key)
):
    camera_id = request.camera_id
    if camera_id not in active_streams:
        raise HTTPException(status_code=404, detail=f"No active stream found for camera_id: {camera_id}")

    worker = active_streams[camera_id]
    worker.stop()
    del active_streams[camera_id]

    return {"detail": "Stream stopped successfully"}


@router.get("/status", response_model=List[StreamStatusResponse])
async def get_stream_status(
    _: bool = Depends(get_api_key)
):
    results = []
    for camera_id, worker in active_streams.items():
        status = worker.get_status()
        results.append(StreamStatusResponse(**status))
    return results


@router.websocket("/ws/live/{camera_id}")
async def websocket_endpoint(websocket: WebSocket, camera_id: str):
    await websocket.accept()
    await websocket_manager.connect(camera_id, websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        await websocket_manager.disconnect(camera_id, websocket)