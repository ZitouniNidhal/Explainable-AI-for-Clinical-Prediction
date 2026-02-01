# src/xai_clinical/data/loader.py
"""
Data loading utilities for various clinical data sources
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import logging
import requests
import zipfile
import io
from sklearn.datasets import fetch_openml

logger = logging.getLogger(__name__)


class DataLoader:
    """Handles data loading from various sources including online datasets"""
    
    def __init__(self, config):
        self.config = config
        self.supported_formats = ['.csv', '.parquet', '.xlsx', '.json', '.feather']
    
    def load_data(self, data_path: Optional[str] = None, **kwargs) -> pd.DataFrame:
        """
        Load data from file with automatic format detection
        
        Args:
            data_path: Path to data file
            **kwargs: Additional arguments for specific formats
            
        Returns:
            Loaded DataFrame
        """
        if data_path is None:
            data_path = self.config.data.external_data_path
            
        if data_path is None:
            raise ValueError("No data path provided")
            
        path = Path(data_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {path}")
        
        logger.info(f"Loading data from: {path}")
        
        if path.suffix == '.csv':
            return pd.read_csv(path, **kwargs)
        elif path.suffix == '.parquet':
            return pd.read_parquet(path, **kwargs)
        elif path.suffix == '.xlsx':
            return pd.read_excel(path, **kwargs)
        elif path.suffix == '.json':
            return pd.read_json(path, **kwargs)
        elif path.suffix == '.feather':
            return pd.read_feather(path, **kwargs)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
    
    def load_synthetic_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Load or generate synthetic data"""
        synthetic_path = self.config.get_path('data') / 'synthetic_data.csv'
        target_path = self.config.get_path('data') / 'synthetic_target.csv'
        
        if synthetic_path.exists() and target_path.exists():
            logger.info("Loading existing synthetic data")
            X = pd.read_csv(synthetic_path)
            y = pd.read_csv(target_path).iloc[:, 0]  # First column
            return X, y
        else:
            logger.info("Generating new synthetic data")
            from .synthetic_generator import SyntheticDataGenerator
            
            generator = SyntheticDataGenerator(
                n_samples=self.config.data.synthetic_samples,
                random_state=self.config.project.random_state
            )
            X, y = generator.generate()
            
            # Save for future use
            X.to_csv(synthetic_path, index=False)
            y.to_csv(target_path, index=False)
            
            return X, y
    
    def load_mimic_sample(self, n_samples: int = 1000) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load a sample of MIMIC-III data (simplified version for demonstration)
        
        Note: In real implementation, this would connect to actual MIMIC database
        """
        logger.info(f"Loading MIMIC-III sample data ({n_samples} patients)")
        
        # For demonstration, we'll create synthetic data that mimics MIMIC structure
        np.random.seed(42)
        
        # Generate synthetic MIMIC-like data
        data = {
            'subject_id': range(n_samples),
            'age': np.random.normal(65, 15, n_samples).clip(18, 95).astype(int),
            'gender': np.random.choice(['M', 'F'], n_samples),
            'admission_type': np.random.choice(['ELECTIVE', 'EMERGENCY', 'URGENT'], n_samples, p=[0.5, 0.3, 0.2]),
            'length_of_stay': np.random.lognormal(2, 0.5, n_samples).round().astype(int),
            'icu_los': np.random.exponential(2, n_samples).round().astype(int),
            'sofa_score': np.random.poisson(4, n_samples).clip(0, 20),
            'sapsii_score': np.random.poisson(30, n_samples).clip(0, 100),
            'heart_rate_mean': np.random.normal(85, 15, n_samples).clip(40, 150).round(1),
            'sys_bp_mean': np.random.normal(120, 20, n_samples).clip(80, 200).round(1),
            'temp_mean': np.random.normal(37, 0.5, n_samples).clip(35, 40).round(1),
            'spo2_mean': np.random.normal(97, 2, n_samples).clip(85, 100).round(1),
            'glucose_max': np.random.normal(140, 30, n_samples).clip(70, 400).round(1),
            'creatinine_max': np.random.normal(1.2, 0.5, n_samples).clip(0.5, 5).round(2),
            'bilirubin_max': np.random.exponential(0.8, n_samples).clip(0.1, 10).round(2),
            'platelet_min': np.random.normal(250, 80, n_samples).clip(50, 500).astype(int),
            'gcs_min': np.random.choice([3, 6, 9, 12, 15], n_samples, p=[0.05, 0.1, 0.15, 0.3, 0.4]),
            'ventilation_hours': np.random.exponential(12, n_samples).clip(0, 168).round(1),
            'vasopressor': np.random.binomial(1, 0.3, n_samples),
            'dialysis': np.random.binomial(1, 0.1, n_samples),
            'mortality': np.random.binomial(1, 0.15, n_samples)
        }
        
        X = pd.DataFrame(data)
        y = X.pop('mortality')
        
        return X, y
    
    def load_chexpert_reports(self, reports_path: str) -> pd.DataFrame:
        """
        Load CheXpert radiology reports
        
        Args:
            reports_path: Path to CheXpert reports CSV
            
        Returns:
            DataFrame with reports and labels
        """
        logger.info(f"Loading CheXpert reports from: {reports_path}")
        
        df = self.load_data(reports_path)
        
        # Standard CheXpert columns
        expected_cols = [
            'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly',
            'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation',
            'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion',
            'Pleural Other', 'Fracture', 'Support Devices'
        ]
        
        # Keep only existing columns
        available_cols = [col for col in expected_cols if col in df.columns]
        
        if available_cols:
            logger.info(f"Found CheXpert labels: {available_cols}")
        
        return df
    
    def load_uptodate_format(self, file_path: str) -> pd.DataFrame:
        """
        Load data in UptoDate format (custom clinical format)
        
        Args:
            file_path: Path to UptoDate format file
            
        Returns:
            Parsed DataFrame
        """
        logger.info(f"Loading UptoDate format from: {file_path}")
        
        # Custom parsing logic for UptoDate format
        # This is a placeholder - implement based on actual format
        try:
            # Try standard formats first
            return self.load_data(file_path)
        except:
            # Custom parsing logic here
            logger.warning("Standard parsing failed, attempting custom UptoDate parsing")
            
            # Example custom parsing
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            # Parse based on UptoDate format specifications
            # This would need to be customized based on actual format
            data = []
            for line in lines:
                if line.strip() and not line.startswith('#'):
                    # Custom parsing logic
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        data.append(parts)
            
            return pd.DataFrame(data[1:], columns=data[0]) if data else pd.DataFrame()
    
    def validate_clinical_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate clinical data quality
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Validation report
        """
        logger.info("Validating clinical data quality")
        
        report = {
            'shape': df.shape,
            'missing_values': df.isnull().sum().to_dict(),
            'missing_percentage': (df.isnull().sum() / len(df) * 100).to_dict(),
            'duplicates': df.duplicated().sum(),
            'data_types': df.dtypes.to_dict(),
            'memory_usage': df.memory_usage(deep=True).sum() / 1024**2,  # MB
        }
        
        # Numeric columns analysis
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            report['numeric_summary'] = df[numeric_cols].describe().to_dict()
        
        # Categorical columns analysis
        categorical_cols = df.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            report['categorical_summary'] = {
                col: df[col].value_counts().head(10).to_dict()
                for col in categorical_cols
            }
        
        # Potential issues
        issues = []
        
        if report['duplicates'] > 0:
            issues.append(f"Found {report['duplicates']} duplicate rows")
        
        high_missing = [col for col, pct in report['missing_percentage'].items() if pct > 50]
        if high_missing:
            issues.append(f"High missing values (>50%) in columns: {high_missing}")
        
        report['issues'] = issues
        
        return report
    
    def download_and_extract(self, url: str, extract_dir: str) -> str:
        """
        Download and extract a zip file
        
        Args:
            url: URL to download
            extract_dir: Directory to extract to
            
        Returns:
            Path to extracted directory
        """
        logger.info(f"Downloading from: {url}")
        
        response = requests.get(url)
        response.raise_for_status()
        
        extract_path = Path(extract_dir)
        extract_path.mkdir(parents=True, exist_ok=True)
        
        with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
            zip_ref.extractall(extract_path)
        
        logger.info(f"Extracted to: {extract_path}")
        return str(extract_path)
    
    def load_from_openml(self, dataset_id: int) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load dataset from OpenML
        
        Args:
            dataset_id: OpenML dataset ID
            
        Returns:
            Features and target
        """
        logger.info(f"Loading dataset {dataset_id} from OpenML")
        
        dataset = fetch_openml(data_id=dataset_id, as_frame=True)
        X = dataset.data
        y = dataset.target
        
        return X, y