"""
Project configuration management
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field


class ProjectConfig(BaseModel):
    name: str
    version: str
    random_state: int = 42


class DataConfig(BaseModel):
    dataset_name: str = "brca_tcga"
    data_dir: str = "data/raw/brca_tcga"
    pancan_dir: str = "data/raw/pancan"
    clinical_file: str = "data_clinical_patient.txt"
    genomic_files: list = []
    pancan_files: Dict[str, str] = {}
    target_column: str = "OS"
    filter_cancer_type: Optional[str] = "BRCA"
    use_synthetic: bool = False
    synthetic_samples: int = 2000
    external_data_path: Optional[str] = None
    test_size: float = 0.2
    validation_size: float = 0.1
    missing_threshold: float = 0.3
    imputation_strategy: str = "iterative"


class PreprocessingConfig(BaseModel):
    scaling: str = "robust"
    balance_classes: bool = True
    balance_method: str = "smote"
    feature_selection: str = "mutual_info"
    max_features: int = 50


class ModelConfig(BaseModel):
    classifiers: list
    hyperparameter_tuning: Dict[str, Any]


class ExplainabilityConfig(BaseModel):
    global_methods: list
    local_methods: list
    shap: Dict[str, Any]
    lime: Dict[str, Any]


class RobustnessConfig(BaseModel):
    noise_tests: list
    stability_metrics: list


class ConfigLoader:
    """Main configuration manager"""

    def __init__(self, config_path: str = "config/config.yaml"):
        self.config_path = Path(config_path)
        self._config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        with open(self.config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    @property
    def project(self) -> ProjectConfig:
        return ProjectConfig(**self._config["project"])

    @property
    def data(self) -> DataConfig:
        return DataConfig(**self._config["data"])

    @property
    def preprocessing(self) -> PreprocessingConfig:
        return PreprocessingConfig(**self._config["preprocessing"])

    @property
    def models(self) -> ModelConfig:
        return ModelConfig(**self._config["models"])

    @property
    def explainability(self) -> ExplainabilityConfig:
        return ExplainabilityConfig(**self._config["explainability"])

    @property
    def robustness(self) -> RobustnessConfig:
        return RobustnessConfig(**self._config["robustness"])

    def get_path(self, key: str) -> Path:
        """Return absolute path for a given key"""
        base_path = Path(__file__).parent.parent.parent
        return base_path / self._config["paths"][key]

    def to_dict(self) -> Dict[str, Any]:
        """Convert entire configuration to dictionary"""
        return self._config
