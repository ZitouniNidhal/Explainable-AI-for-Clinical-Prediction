# XAI Clinical Prediction 🔬

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Testing: pytest](https://img.shields.io/badge/testing-pytest-blue.svg)](https://docs.pytest.org/)

**Explainable Machine Learning Model for Clinical Outcome Prediction** with comprehensible interpretations for clinicians.

## 🎯 Objectives

- **Predict** post-operative complication risk with high accuracy
- **Explain** predictions using SHAP and LIME for clinical transparency
- **Validate** model robustness against data perturbations
- **Facilitate** clinical adoption through interpretable AI

## 🌟 Key Features

- ✅ **Multiple ML Models**: XGBoost, LightGBM, Random Forest, Logistic Regression
- ✅ **Advanced XAI**: SHAP global/local explanations + LIME comparisons
- ✅ **Robustness Testing**: Stability analysis with noise and missing data
- ✅ **Clinical Validation**: Medical sense checking of predictions
- ✅ **Interactive Interface**: Streamlit web application
- ✅ **Production Ready**: Docker support, comprehensive testing, logging

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/NidhalZitouni/xai-clinical-prediction.git
cd xai-clinical-prediction

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
