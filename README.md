# 🌟 StarHarbor - AI-Powered Exoplanet Vetting System

[![NASA Space Apps Challenge 2025](https://img.shields.io/badge/NASA_Space_Apps-2025-blue)](https://spaceappschallenge.org)
[![Python](https://img.shields.io/badge/Python-3.11+-green)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-Latest-teal)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue)](https://docker.com)

## 🚀 Project Overview

**StarHarbor** is an advanced AI-powered system for automated exoplanet detection and vetting, designed for the NASA Space Apps Challenge 2025. Our solution processes data from multiple NASA missions (Kepler, K2, TESS) and uses machine learning to classify astronomical objects as false positives, planetary candidates, or confirmed exoplanets.

### 🎯 Challenge Addressed
**"Exoplanet Discovery Using Machine Learning"** - Creating an AI/ML model trained on NASA's open-source exoplanet datasets to automatically analyze survey data and accurately identify exoplanets.

### ✨ Key Features
- 🤖 **Advanced AI Classification**: Random Forest model with 85-86% accuracy
- 🌍 **Multi-Mission Support**: Unified processing for Kepler, K2, and TESS data
- 🇺🇦 **Ukrainian Interface**: Accessible space science for Ukrainian researchers
- ⚡ **Real-time Analysis**: Process thousands of candidates in seconds
- 📊 **Statistical Validation**: Conformal prediction for uncertainty quantification
- 🔍 **Explainable AI**: SHAP-based feature importance analysis
- 🐳 **Production Ready**: Full Docker containerization

## 🛠 Technical Architecture

### Machine Learning Pipeline
```
Raw NASA Data → Schema Normalization → Feature Engineering → ML Classification → Statistical Validation → Results
```

### Technology Stack
- **Backend**: Python 3.11, FastAPI, scikit-learn, pandas
- **ML Models**: Random Forest, ONNX runtime, feature scaling
- **Frontend**: HTML5, CSS3, JavaScript (Ukrainian localization)
- **Deployment**: Docker, Docker Compose
- **Data Sources**: NASA Exoplanet Archive (KOI, K2, TOI catalogs)

### System Components
```
📦 StarHarbor/
├── 🔧 api/               # FastAPI backend services
│   ├── models/           # Pydantic data models
│   ├── routers/          # REST API endpoints
│   ├── services/         # ML pipeline & business logic
│   └── utils/            # Core utilities
├── 📊 data/              # Data processing pipeline
│   ├── schema/           # Mission-specific normalization
│   ├── samples/          # Test datasets
│   └── processed/        # Processed NASA catalogs
├── 🌐 frontend/          # Ukrainian web interface
├── 🤖 models/            # Trained ML models & artifacts
├── 📚 docs/              # Documentation & demo
└── 🐳 docker-compose.yml # Container orchestration
```

## Quick Start Guide

### 📋 Prerequisites
- Python 3.11+
- All dependencies from `requirements.txt`
- Trained models in the `models/` directory

### 🚀 Running the System

#### Option 1: Direct Python
```bash
# Install dependencies
pip install -r requirements.txt

# Start the API server
python -m uvicorn api.utils.main:app --host 0.0.0.0 --port 8000 --reload

# Open frontend
# Navigate to frontend/index.html in your browser
# Or serve with a simple HTTP server:
cd frontend
python -m http.server 8080
```

#### Option 2: Docker (Recommended)
```bash
# Build and start all services
docker-compose up -d

# Check logs
docker-compose logs -f

# Stop services
docker-compose down
```

### 🔧 Testing the API

Run the test script:
```bash
python test_api.py
```

### 📊 Using the Web Interface

1. Open `http://localhost:80` (Docker) or `frontend/index.html` (direct)
2. Upload a CSV file with exoplanet data
3. Specify the mission (kepler/k2/tess) if using raw NASA data
4. Click "Аналізувати" to run the AI analysis
5. View results with predictions and statistics

### 📁 Sample Data

Test files are available in `data/samples/`:
- `sample_exoplanets.csv` - Kepler mission data
- `sample_k2.csv` - K2 mission data

### 🔍 API Endpoints

- `GET /inference/health` - Health check
- `POST /inference/upload` - Upload and parse dataset
- `POST /inference/predict` - Make predictions
- `POST /inference/predict-file` - Predict from uploaded file
- `POST /inference/explain` - SHAP explanations
- `POST /inference/conformal` - Conformal prediction confidence
- `POST /inference/vet` - Quality control vetting

### 📈 Features

- **Multi-mission support**: Kepler, K2, TESS
- **Robust CSV parsing**: Handles NASA Exoplanet Archive format
- **AI-powered classification**: False Positive / Candidate / Confirmed
- **Quality control**: Automated vetting flags
- **Explainable AI**: SHAP feature importance
- **Conformal prediction**: Uncertainty quantification
- **Web interface**: User-friendly upload and analysis

### 🛠 Troubleshooting

1. **API not responding**: Check if server is running on port 8000
2. **Empty predictions**: Ensure CSV contains actual data, not just headers
3. **File parsing errors**: Try specifying the mission parameter
4. **Missing models**: Ensure all model files are in the `models/` directory

### 📚 Architecture

```
StarHarbor/
├── api/               # FastAPI backend
│   ├── models/        # Pydantic models
│   ├── routers/       # API endpoints
│   ├── services/      # Business logic
│   └── utils/         # Utilities
├── data/              # Data processing
│   ├── schema/        # Mission-specific schemas
│   └── samples/       # Test data
├── frontend/          # Web interface
├── models/            # Trained ML models
└── docker-compose.yml # Container orchestration
```

### 🏆 NASA Space Apps Challenge 2025

This system was developed for the NASA Space Apps Challenge 2025, focusing on AI-powered exoplanet detection and vetting. It combines machine learning, conformal prediction, and quality control to help astronomers identify genuine exoplanet candidates from transit survey data.