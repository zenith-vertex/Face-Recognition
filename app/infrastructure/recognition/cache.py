import time
import threading
from typing import List, Tuple


class EmbeddingCache:
    def __init__(self, ttl_seconds: int = 300):
        self._enrolled: List[Tuple[str, str, list]] = []
        self._timestamp = 0.0
        self._ttl = ttl_seconds
        self._lock = threading.Lock()

    def get(self) -> List[Tuple[str, str, list]]:
        with self._lock:
            return list(self._enrolled)

    def set(self, records: List[Tuple[str, str, list]]):
        with self._lock:
            self._enrolled = list(records)
            self._timestamp = time.time()

    def invalidate(self):
        with self._lock:
            self._enrolled = []
            self._timestamp = 0.0

    def is_stale(self) -> bool:
        with self._lock:
            return (time.time() - self._timestamp) > self._ttl


embedding_cache = EmbeddingCache()
