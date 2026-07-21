import numpy as np
from typing import List, Optional
from app.core.config import settings


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    if a_norm == 0 or b_norm == 0:
        return 0.0
    return float(np.dot(a, b) / (a_norm * b_norm))


class FaceMatcher:
    def __init__(self, threshold: float = None):
        self.threshold = threshold if threshold is not None else settings.SIMILARITY_THRESHOLD
        self._enrolled: List[tuple[str, str, np.ndarray]] = []

    def load_enrolled(self, records: List[tuple[str, str, list]]):
        self._enrolled = []
        for user_id, full_name, embedding_list in records:
            emb = np.array(embedding_list, dtype=np.float32)
            self._enrolled.append((user_id, full_name, emb))

    def match(self, embedding: np.ndarray) -> Optional[dict]:
        best = None
        best_score = -1.0
        for user_id, full_name, enrolled_emb in self._enrolled:
            score = cosine_similarity(embedding, enrolled_emb)
            if score > best_score:
                best_score = score
                best = (user_id, full_name, score)
        if best and best[2] >= self.threshold:
            return {
                "user_id": best[0],
                "full_name": best[1],
                "similarity": best[2],
            }
        return None

    def match_top_k(self, embedding: np.ndarray, k: int = 5) -> List[dict]:
        scores = []
        for user_id, full_name, enrolled_emb in self._enrolled:
            score = cosine_similarity(embedding, enrolled_emb)
            scores.append((score, user_id, full_name))
        scores.sort(reverse=True)
        return [
            {"user_id": uid, "full_name": name, "similarity": score}
            for score, uid, name in scores[:k]
        ]
