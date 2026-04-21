# src/xai_clinical/models/base_model.py

"""
Base model interface for XAI clinical prediction.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import numpy as np


class BaseModel(ABC):
    """
    Abstract base class for all clinical prediction models.
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.model = None
        self.is_fitted = False

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the model."""
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        pass

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities."""
        pass

    def get_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        return self.config
