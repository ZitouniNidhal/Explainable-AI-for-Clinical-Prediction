# src/xai_clinical/data/__init__.py

from .loader import DataLoader
from .preprocessor import DataPreprocessor
from .synthetic_generator import SyntheticDataGenerator

# SUPPRIME cette ligne:
# from .shap_explainer import SHAPExplainer  # ← ERREUR! C'est dans explainability/

__all__ = ["DataLoader", "DataPreprocessor", "SyntheticDataGenerator"]
