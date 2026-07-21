"""Threshold calibration utility.

Builds genuine/impostor score distributions from enrolled embeddings,
sweeps candidate thresholds, and reports FAR/FRR at each step.
Run after collecting a representative validation set through your
actual camera/pipeline.

Usage:
    python scripts/calibrate_threshold.py
"""

from __future__ import annotations

import argparse
import os
from collections import defaultdict

import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.core.config import settings
from app.database.models import Base, FaceEmbedding, User
from app.infrastructure.recognition.matcher import cosine_similarity


def load_enrolled(database_url: str) -> list[tuple[str, str, np.ndarray]]:
    engine = create_engine(database_url)
    Base.metadata.bind = engine
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        rows = (
            session.query(FaceEmbedding, User)
            .join(User, FaceEmbedding.user_id == User.id)
            .all()
        )
        enrolled = []
        for fe, u in rows:
            emb = np.array(fe.embedding, dtype=np.float32)
            enrolled.append((str(fe.user_id), u.full_name, emb))
        return enrolled
    finally:
        session.close()


def compute_scores(enrolled: list[tuple[str, str, np.ndarray]]):
    by_user = defaultdict(list)
    for user_id, full_name, emb in enrolled:
        by_user[user_id].append(emb)

    genuine = []
    impostor = []
    users = list(by_user.keys())
    for user_id, embeddings in by_user.items():
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                genuine.append(cosine_similarity(embeddings[i], embeddings[j]))
        for other_id in users:
            if other_id == user_id:
                continue
            for emb in embeddings:
                for other_emb in by_user[other_id][:3]:
                    impostor.append(cosine_similarity(emb, other_emb))
    return np.array(genuine), np.array(impostor)


def evaluate_threshold(genuine_scores: np.ndarray, impostor_scores: np.ndarray, threshold: float):
    far = float(np.mean(impostor_scores >= threshold))
    frr = float(np.mean(genuine_scores < threshold))
    return far, frr


def main():
    parser = argparse.ArgumentParser(description="Calibrate face recognition similarity threshold")
    parser.add_argument("--database-url", default=os.getenv("DATABASE_URL", settings.DATABASE_URL))
    parser.add_argument("--min", type=float, default=0.30)
    parser.add_argument("--max", type=float, default=0.95)
    parser.add_argument("--step", type=float, default=0.05)
    args = parser.parse_args()

    enrolled = load_enrolled(args.database_url)
    if len(enrolled) < 2:
        print("Need at least 2 enrolled users with embeddings to calibrate.")
        return

    genuine, impostor = compute_scores(enrolled)
    print(f"Genuine pairs: {len(genuine)}, Impostor pairs: {len(impostor)}")
    print(f"Genuine mean={genuine.mean():.4f} std={genuine.std():.4f}")
    print(f"Impostor mean={impostor.mean():.4f} std={impostor.std():.4f}")
    print()
    print(f"{'threshold':>10} {'FAR':>10} {'FRR':>10}")
    print("-" * 32)
    best = None
    for t in np.arange(args.min, args.max + args.step / 2, args.step):
        far, frr = evaluate_threshold(genuine, impostor, float(t))
        print(f"{t:10.2f} {far:10.4f} {frr:10.4f}")
        if best is None or (far + frr) < best[0]:
            best = (far + frr, t, far, frr)
    if best:
        _, t, far, frr = best
        print()
        print(f"Suggested operating point (min FAR+FRR): threshold={t:.2f} FAR={far:.4f} FRR={frr:.4f}")


if __name__ == "__main__":
    main()
