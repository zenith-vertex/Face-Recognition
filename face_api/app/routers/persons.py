import os
from typing import List, Optional
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
import numpy as np
import cv2

import sys
sys.path.insert(0, "C:/Project/Face-Recognition")

from face_db.repository import create_person, deactivate_person, add_face_embedding
from face_db.models import Person, FaceEmbedding

from ..config import settings
from ..dependencies import get_pipeline, get_api_key
from ..schemas import RegisterFaceResponse, PersonResponse
from ..services.pipeline_service import PipelineService

router = APIRouter(
    prefix="/persons",
    tags=["persons"]
)


def _process_registration(file_bytes: bytes, filename: str, pipeline: PipelineService) -> tuple:
    nparr = np.frombuffer(file_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None or frame.size == 0:
        raise HTTPException(status_code=400, detail=f"Invalid or corrupt image: {filename}")

    detections = pipeline.detect_faces(frame)

    if len(detections) == 0:
        raise HTTPException(status_code=400, detail=f"No face detected in image: {filename}")
    if len(detections) > 1:
        raise HTTPException(status_code=400, detail=f"Multiple faces detected in image: {filename}. Enrollment images must contain exactly one face.")

    bbox = detections[0]["bbox"]
    face_crop = pipeline.crop_face(frame, bbox)
    embedding = pipeline.embed_face(face_crop)
    pipeline.validate_embedding_dimension(embedding)

    embedding = embedding / np.linalg.norm(embedding)

    return embedding


@router.post("/register-face", response_model=RegisterFaceResponse)
async def register_face(
    name: str = Form(...),
    role: str = Form("unspecified"),
    files: List[UploadFile] = File(...),
    pipeline: PipelineService = Depends(get_pipeline),
    _: bool = Depends(get_api_key)
):
    if not files:
        raise HTTPException(status_code=400, detail="At least one image file is required")

    embeddings_stored = 0
    person_id = None

    for upload_file in files:
        try:
            contents = await upload_file.read()

            embedding = await run_in_threadpool(
                _process_registration, contents, upload_file.filename, pipeline
            )

            if person_id is None:
                person_id = create_person(name=name, role=role)

            add_face_embedding(
                person_id=person_id,
                embedding=embedding,
                embedder_model=settings.ACTIVE_EMBEDDER
            )
            embeddings_stored += 1

        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error processing image {upload_file.filename}: {str(e)}")

    if person_id is None:
        raise HTTPException(status_code=500, detail="Failed to create person record")

    return RegisterFaceResponse(person_id=person_id, name=name, embeddings_stored=embeddings_stored)


@router.get("/", response_model=List[PersonResponse])
async def list_persons(
    limit: int = 50,
    offset: int = 0,
    _: bool = Depends(get_api_key)
):
    from face_db.database import SessionLocal
    db = SessionLocal()
    try:
        persons = db.query(Person).filter(Person.is_active == True).offset(offset).limit(limit).all()
        results = []
        for person in persons:
            embedding_count = db.query(FaceEmbedding).filter(FaceEmbedding.person_id == person.id).count()
            results.append(PersonResponse(
                id=person.id,
                name=person.name,
                role=person.role,
                is_active=person.is_active,
                registration_date=person.registration_date,
                embedding_count=embedding_count
            ))
        return results
    finally:
        db.close()


@router.get("/{person_id}", response_model=PersonResponse)
async def get_person_info(
    person_id: int,
    _: bool = Depends(get_api_key)
):
    from face_db.database import SessionLocal
    db = SessionLocal()
    try:
        person = db.query(Person).filter(Person.id == person_id).first()
        if person is None:
            raise HTTPException(status_code=404, detail="Person not found")

        embedding_count = db.query(FaceEmbedding).filter(FaceEmbedding.person_id == person_id).count()
    finally:
        db.close()

    return PersonResponse(
        id=person.id,
        name=person.name,
        role=person.role,
        is_active=person.is_active,
        registration_date=person.registration_date,
        embedding_count=embedding_count
    )


@router.delete("/{person_id}")
async def delete_person(
    person_id: int,
    _: bool = Depends(get_api_key)
):
    person = None
    from face_db.database import SessionLocal
    db = SessionLocal()
    try:
        person = db.query(Person).filter(Person.id == person_id).first()
        if person is None:
            raise HTTPException(status_code=404, detail="Person not found")
    finally:
        db.close()

    deactivate_person(person_id)
    return {"detail": "Person deactivated successfully"}