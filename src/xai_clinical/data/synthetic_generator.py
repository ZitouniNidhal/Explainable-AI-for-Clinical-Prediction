# src/xai_clinical/data/synthetic_generator.py
"""
Synthetic clinical data generator for demonstration
"""

import numpy as np
import pandas as pd
from typing import Tuple, List
from sklearn.datasets import make_classification


class SyntheticDataGenerator:
    """
    Generates realistic synthetic clinical data for demonstration.
    Simulates a post-operative complication risk prediction scenario.
    """

    def __init__(self, n_samples: int = 2000, random_state: int = 42):
        self.n_samples = n_samples
        self.random_state = random_state
        self.feature_names = self._get_feature_names()

    def _get_feature_names(self) -> List[str]:
        """Define clinical variable names"""
        return [
            # Demographics
            "age",
            "sex",
            "bmi",
            # Medical history
            "diabetes",
            "hypertension",
            "heart_disease",
            "renal_insufficiency",
            # Pre-operative parameters
            "hemoglobin",
            "creatinine",
            "leukocytes",
            "platelets",
            "sodium",
            "potassium",
            "albumin",
            # Respiratory function
            "respiratory_rate",
            "oxygen_saturation",
            "pao2_fio2_ratio",
            # Cardiac function
            "heart_rate",
            "systolic_bp",
            "diastolic_bp",
            # Risk scores
            "asa_score",
            "nyha_class",
            # Surgery characteristics
            "surgery_duration",
            "emergency",
            "surgery_type",
            # Peri-operative parameters
            "blood_loss",
            "transfusion",
            "hypothermia",
        ]

    def generate(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Generates synthetic clinical dataset with realistic relationships
        between variables and clinical outcome.
        """
        np.random.seed(self.random_state)
        n = self.n_samples

        # Generate base features
        data = {}

        # 1. Demographics
        data["age"] = np.random.normal(65, 15, n).clip(18, 95).astype(int)
        data["sex"] = np.random.binomial(1, 0.55, n)  # 1=Male, 0=Female
        data["bmi"] = np.random.normal(27, 5, n).clip(15, 45)

        # 2. Medical history (correlated with age)
        data["diabetes"] = np.random.binomial(
            1, 0.3 + 0.01 * (data["age"] - 50).clip(0) / 10, n
        )
        data["hypertension"] = np.random.binomial(
            1, 0.4 + 0.008 * (data["age"] - 55).clip(0) / 10, n
        )
        data["heart_disease"] = np.random.binomial(
            1, 0.2 + 0.015 * (data["age"] - 60).clip(0) / 10, n
        )
        data["renal_insufficiency"] = np.random.binomial(
            1, 0.1 + 0.01 * data["diabetes"], n
        )

        # 3. Biological parameters
        data["hemoglobin"] = (
            np.random.normal(13, 2, n) - 0.5 * data["renal_insufficiency"]
        )
        data["creatinine"] = (
            np.random.normal(80, 20, n)
            + 30 * data["renal_insufficiency"]
            + 10 * data["diabetes"]
        )
        data["leukocytes"] = np.random.normal(8, 2, n)
        data["platelets"] = np.random.normal(250, 80, n)
        data["sodium"] = np.random.normal(140, 3, n)
        data["potassium"] = np.random.normal(4, 0.5, n)
        data["albumin"] = np.random.normal(38, 5, n) - 3 * data["heart_disease"]

        # 4. Respiratory function
        data["respiratory_rate"] = (
            np.random.normal(16, 3, n) + 2 * data["heart_disease"]
        )
        data["oxygen_saturation"] = (
            np.random.normal(97, 2, n) - 3 * data["heart_disease"]
        )
        data["pao2_fio2_ratio"] = (
            np.random.normal(300, 50, n) - 40 * data["heart_disease"]
        )

        # 5. Cardiac function
        data["heart_rate"] = np.random.normal(75, 12, n) + 5 * data["hypertension"]
        data["systolic_bp"] = np.random.normal(130, 20, n) + 15 * data["hypertension"]
        data["diastolic_bp"] = np.random.normal(80, 10, n) + 10 * data["hypertension"]

        # 6. Risk scores
        data["asa_score"] = np.random.choice(
            [1, 2, 3, 4, 5], n, p=[0.1, 0.3, 0.35, 0.2, 0.05]
        )
        data["nyha_class"] = np.random.choice([1, 2, 3, 4], n, p=[0.4, 0.35, 0.2, 0.05])

        # 7. Surgical characteristics
        data["surgery_duration"] = np.random.lognormal(4, 0.5, n)  # minutes
        data["emergency"] = np.random.binomial(1, 0.2, n)
        data["surgery_type"] = np.random.choice(
            [0, 1, 2], n, p=[0.5, 0.3, 0.2]
        )  # 0=minor, 1=major, 2=complex

        # 8. Peri-operative parameters
        data["blood_loss"] = np.random.exponential(200, n) * (
            1 + 0.5 * data["surgery_type"]
        )
        data["transfusion"] = (data["blood_loss"] > 500).astype(
            int
        ) * np.random.binomial(1, 0.7, n)
        data["hypothermia"] = np.random.binomial(
            1, 0.15 + 0.1 * data["surgery_duration"] / 100, n
        )

        # Create DataFrame
        df = pd.DataFrame(data)

        # Calculate theoretical risk (ground truth for generation)
        risk_score = (
            0.02 * (df["age"] - 50)
            + 0.5 * df["diabetes"]
            + 0.4 * df["hypertension"]
            + 0.8 * df["heart_disease"]
            + 0.6 * df["renal_insufficiency"]
            + 0.1 * (df["creatinine"] - 80) / 10
            + 0.3 * (df["asa_score"] - 2)
            + 0.2 * df["nyha_class"]
            + 0.001 * (df["surgery_duration"] - 100)
            + 0.5 * df["emergency"]
            + 0.3 * df["surgery_type"]
            + 0.0005 * df["blood_loss"]
            + 0.4 * df["hypothermia"]
            + np.random.normal(0, 0.5, n)  # noise
        )

        # Convert to probability using logistic function
        probability = 1 / (1 + np.exp(-risk_score))

        # Generate binary target (post-operative complication)
        target = (probability > 0.5).astype(int)

        # Add some realistic missing values (MNAR)
        missing_cols = ["albumin", "pao2_fio2_ratio", "potassium"]
        for col in missing_cols:
            mask = np.random.binomial(1, 0.05, n).astype(bool)
            df.loc[mask, col] = np.nan

        return df, pd.Series(target, name="clinical_outcome")

    def get_feature_descriptions(self) -> dict:
        """Return feature descriptions for clinical interpretation"""
        return {
            "age": "Patient age (years)",
            "sex": "Sex (1=Male, 0=Female)",
            "bmi": "Body Mass Index (kg/m²)",
            "diabetes": "Diabetes history (0/1)",
            "hypertension": "Hypertension history (0/1)",
            "heart_disease": "Known heart disease (0/1)",
            "renal_insufficiency": "Chronic renal insufficiency (0/1)",
            "hemoglobin": "Hemoglobin level (g/dL)",
            "creatinine": "Serum creatinine (µmol/L)",
            "leukocytes": "Leukocyte count (G/L)",
            "platelets": "Platelets (thousands/µL)",
            "sodium": "Serum sodium (mmol/L)",
            "potassium": "Serum potassium (mmol/L)",
            "albumin": "Serum albumin (g/L)",
            "respiratory_rate": "Respiratory rate (/min)",
            "oxygen_saturation": "Oxygen saturation (%)",
            "pao2_fio2_ratio": "PaO2/FiO2 ratio",
            "heart_rate": "Heart rate (bpm)",
            "systolic_bp": "Systolic blood pressure (mmHg)",
            "diastolic_bp": "Diastolic blood pressure (mmHg)",
            "asa_score": "ASA Score (1-5)",
            "nyha_class": "NYHA Class (1-4)",
            "surgery_duration": "Surgery duration (minutes)",
            "emergency": "Emergency surgery (0/1)",
            "surgery_type": "Surgery type (0=minor, 1=major, 2=complex)",
            "blood_loss": "Estimated blood loss (mL)",
            "transfusion": "Transfusion performed (0/1)",
            "hypothermia": "Peri-operative hypothermia (0/1)",
        }
