# src/xai_clinical/models/__init__.py

from .base_model import BaseModel
from .classifiers import ClassifierFactory
from .trainer import ModelTrainer

# SUPPRIME ces lignes:
# from .data.preprocessor import DataPreprocessor  # ← data/ n'est pas dans models/
# from .data.loader import DataLoader  # ← data/ n'est pas dans models/

__all__ = ["BaseModel", "ClassifierFactory", "ModelTrainer"]
