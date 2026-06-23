from abc import ABC, abstractmethod
import numpy as np


class BaseEmbedder(ABC):
    @abstractmethod
    def embed(self, face_crop: np.ndarray) -> np.ndarray:
        pass