import os
from datetime import datetime
from typing import Optional
import numpy as np
from sqlalchemy.orm import Session
from sqlalchemy import text

from models import Person, FaceEmbedding, RecognitionLog
from database import SessionLocal

VECTOR_DIMENSIONS = {
    "dlib": 128,
    "facenet": 512,
    "arcface": 512
}


def get_vector_backend():
    return os.getenv("VECTOR_BACKEND", "pgvector")


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(1 - np.dot(a, b))


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def create_person(name: str, role: str = "unspecified") -> int:
    db = SessionLocal()
    try:
        person = Person(name=name, role=role)
        db.add(person)
        db.commit()
        db.refresh(person)
        return person.id
    finally:
        db.close()


def get_person(person_id: int) -> Optional[dict]:
    db = SessionLocal()
    try:
        person = db.query(Person).filter(Person.id == person_id).first()
        if person is None:
            return None
        return {
            "id": person.id,
            "name": person.name,
            "role": person.role,
            "is_active": person.is_active,
            "registration_date": person.registration_date,
            "created_at": person.created_at,
            "updated_at": person.updated_at
        }
    finally:
        db.close()


def deactivate_person(person_id: int) -> None:
    db = SessionLocal()
    try:
        person = db.query(Person).filter(Person.id == person_id).first()
        if person:
            person.is_active = False
            db.commit()
    finally:
        db.close()


def add_face_embedding(person_id: int, embedding: np.ndarray, embedder_model: str, source_image_path: str = None) -> int:
    if embedder_model not in VECTOR_DIMENSIONS:
        raise ValueError(f"Unknown embedder model: {embedder_model}. Available: {list(VECTOR_DIMENSIONS.keys())}")

    expected_dim = VECTOR_DIMENSIONS[embedder_model]
    embedding = np.asarray(embedding, dtype=np.float32)
    if len(embedding) != expected_dim:
        raise ValueError(f"Embedding dimension mismatch: expected {expected_dim}, got {len(embedding)}")

    embedding = embedding / np.linalg.norm(embedding)

    db = SessionLocal()
    try:
        face_embedding = FaceEmbedding(
            person_id=person_id,
            embedder_model=embedder_model,
            embedding=embedding.tolist(),
            source_image_path=source_image_path
        )
        db.add(face_embedding)
        db.commit()
        db.refresh(face_embedding)
        return face_embedding.id
    finally:
        db.close()


def get_embeddings_for_person(person_id: int, embedder_model: str) -> list:
    db = SessionLocal()
    try:
        embeddings = db.query(FaceEmbedding).filter(
            FaceEmbedding.person_id == person_id,
            FaceEmbedding.embedder_model == embedder_model
        ).all()
        return [np.array(e.embedding, dtype=np.float32) for e in embeddings]
    finally:
        db.close()


def find_nearest_match(query_embedding: np.ndarray, embedder_model: str, metric: str = "cosine", top_k: int = 1) -> list:
    if len(query_embedding) == 0:
        return []

    backend = get_vector_backend()
    db = SessionLocal()

    try:
        if backend == "pgvector":
            op = "<=>" if metric == "cosine" else "<->"
            stmt = text(f"""
                SELECT fe.id, fe.person_id, p.name as person_name,
                       (embedding {op} :query_vec) as distance
                FROM face_embeddings fe
                JOIN persons p ON fe.person_id = p.id
                WHERE fe.embedder_model = :embedder_model
                ORDER BY distance
                LIMIT :top_k
            """)
            result = db.execute(stmt, {
                "query_vec": query_embedding.tolist(),
                "embedder_model": embedder_model,
                "top_k": top_k
            }).fetchall()

            return [{
                "id": row.id,
                "person_id": row.person_id,
                "person_name": row.person_name,
                "distance": float(row.distance)
            } for row in result]
        else:
            embeddings = db.query(FaceEmbedding).join(Person).filter(
                FaceEmbedding.embedder_model == embedder_model
            ).all()

            results = []
            for emb in embeddings:
                stored_emb = np.array(emb.embedding, dtype=np.float32)
                
                if metric == "cosine":
                    distance = cosine_distance(query_embedding, stored_emb)
                else:
                    distance = euclidean_distance(query_embedding, stored_emb)

                results.append({
                    "id": emb.id,
                    "person_id": emb.person_id,
                    "person_name": emb.person.name,
                    "distance": distance
                })

            results.sort(key=lambda x: x["distance"])
            return results[:top_k]
    finally:
        db.close()


def log_recognition(matched_person_id: Optional[int], confidence_score: float, metric_used: str,
                     decision: str, detector_used: str = None, embedder_used: str = None,
                     source_camera: str = None, frame_reference: str = None,
                     raw_bbox: dict = None) -> int:
    if decision == "match" and matched_person_id is None:
        raise ValueError("Match decision requires matched_person_id")
    if decision == "no_match" and matched_person_id is not None:
        raise ValueError("No_match decision requires matched_person_id to be None")

    db = SessionLocal()
    try:
        log = RecognitionLog(
            matched_person_id=matched_person_id,
            confidence_score=confidence_score,
            metric_used=metric_used,
            detector_used=detector_used,
            embedder_used=embedder_used,
            decision=decision,
            source_camera=source_camera,
            frame_reference=frame_reference,
            raw_bbox=raw_bbox
        )
        db.add(log)
        db.commit()
        db.refresh(log)
        return log.id
    finally:
        db.close()


def get_recognition_logs(person_id: Optional[int] = None, start: datetime = None,
                         end: datetime = None, decision: str = None) -> list:
    db = SessionLocal()
    try:
        query = db.query(RecognitionLog)

        if person_id is not None:
            query = query.filter(RecognitionLog.matched_person_id == person_id)
        if start is not None:
            query = query.filter(RecognitionLog.timestamp >= start)
        if end is not None:
            query = query.filter(RecognitionLog.timestamp <= end)
        if decision is not None:
            query = query.filter(RecognitionLog.decision == decision)

        logs = query.order_by(RecognitionLog.timestamp.desc()).all()
        return [{
            "id": log.id,
            "timestamp": log.timestamp,
            "matched_person_id": log.matched_person_id,
            "matched_person_name": log.person.name if log.person else "Unknown",
            "confidence_score": log.confidence_score,
            "metric_used": log.metric_used,
            "detector_used": log.detector_used,
            "embedder_used": log.embedder_used,
            "decision": log.decision,
            "source_camera": log.source_camera,
            "frame_reference": log.frame_reference
        } for log in logs]
    finally:
        db.close()