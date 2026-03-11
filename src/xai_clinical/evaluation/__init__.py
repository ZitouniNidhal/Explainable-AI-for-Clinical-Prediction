"""
Evaluation module.

Provides tools for building and training classifiers.

Classes:
    ClassifierFactory: Factory for creating classifier instances.
    ModelTrainer: Handles model training workflows.

Usage:
    from evaluation import ClassifierFactory, ModelTrainer
"""

from .classifiers import ClassifierFactory
from .trainer import ModelTrainer

__all__ = [
    "ClassifierFactory",
    "ModelTrainer",
]