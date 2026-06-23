import numpy as np


def match(query_embedding: np.ndarray, known_embeddings: dict, metric: str = "cosine", threshold: float = 0.6) -> dict:
    if query_embedding is None or len(query_embedding) == 0:
        return {"name": "Unknown", "score": 0.0, "metric": metric}
    
    if not known_embeddings:
        return {"name": "Unknown", "score": 0.0, "metric": metric}
    
    best_name = "Unknown"
    best_score = 0.0
    best_distance = float('inf')
    
    for name, known_emb in known_embeddings.items():
        if known_emb is None or len(known_emb) == 0:
            continue
        
        distance = float(np.linalg.norm(query_embedding - known_emb))
        similarity = 1.0 - distance
        
        if metric == "cosine":
            if similarity > best_score:
                best_score = similarity
                best_name = name
                best_distance = distance
        else:
            if distance < best_distance:
                best_distance = distance
                best_name = name
                best_score = similarity
    
    if metric == "cosine":
        if best_score >= threshold:
            return {"name": best_name, "score": best_score, "metric": metric}
    else:
        if best_distance <= threshold:
            return {"name": best_name, "score": best_score, "metric": metric}
    
    return {"name": "Unknown", "score": best_score, "metric": metric}