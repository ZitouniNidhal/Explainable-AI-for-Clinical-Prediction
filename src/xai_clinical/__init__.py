
"""
XAI Clinical Prediction - Explainable model for clinical outcome prediction
"""

__version__ = "1.0.0"
__author__ = "Nidhal Zitouni"

from .config import Config
from .pipeline import ClinicalPipeline

__all__ = ["Config", "ClinicalPipeline"]