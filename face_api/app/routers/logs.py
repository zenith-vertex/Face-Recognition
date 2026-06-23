from datetime import datetime
from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException, Query

import sys
sys.path.insert(0, "C:/Project/Face-Recognition")

from face_db.repository import get_recognition_logs

from ..dependencies import get_api_key
from ..schemas import RecognitionLogResponse

router = APIRouter(
    prefix="/logs",
    tags=["logs"]
)


@router.get("/", response_model=List[RecognitionLogResponse])
async def get_logs(
    person_id: Optional[int] = Query(None),
    start_date: Optional[datetime] = Query(None),
    end_date: Optional[datetime] = Query(None),
    decision: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    _: bool = Depends(get_api_key)
):
    logs = get_recognition_logs(
        person_id=person_id,
        start=start_date,
        end=end_date,
        decision=decision
    )

    return logs[offset:offset + limit]