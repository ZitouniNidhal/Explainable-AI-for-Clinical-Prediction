# src/xai_clinical/__init__.py
"""
XAI Clinical Prediction - Explainable model for clinical outcome prediction
"""

__version__ = "1.0.0"
__author__ = "Your Name"

from .config import Config
from .pipeline import ClinicalPipeline

__all__ = ["Config", "ClinicalPipeline"]