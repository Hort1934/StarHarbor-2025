# 🌟 StarHarbor - Exoplanet Detection with AI

## NASA Space Apps Challenge 2025

StarHarbor is an AI-powered exoplanet detection system that analyzes space mission data to identify confirmed exoplanets, planetary candidates, and false positives.

## 🚀 Quick Start

### 1. Run with Docker (Recommended)

```bash
# Clone the repository
git clone <your-repo-url>
cd StarHarbor-2025

# Start the full stack
docker-compose up -d

# Access the application
# Frontend: http://localhost:3000
# API: http://localhost:8000/docs
```

### 2. Run Locally

#### Prerequisites
- Python 3.11+
- Node.js (for frontend development)

#### Setup API
```bash
# Install dependencies
pip install -r requirements.txt

# Start the API server
python scripts/run_server.py
```

#### Setup Frontend
```bash
# Start the frontend server
cd frontend
python -m http.server 3000
```

## 📊 Usage

1. **Open the web interface** at http://localhost:3000
2. **Upload your dataset** (CSV or Parquet format)
   - Supported missions: Kepler, K2, TESS
   - Sample data available in `test_data.csv`
3. **Click "Аналізувати"** to start the AI analysis
4. **View results** showing:
   - Confirmed exoplanets
   - Planetary candidates
   - False positives
   - Detection statistics

## 🔬 Supported Data Formats

### Input Columns
The system accepts data from multiple space missions with these key columns:

- `period_days` - Orbital period
- `duration_hours` - Transit duration  
- `depth_ppm` - Transit depth in parts per million
- `snr` - Signal-to-noise ratio
- `impact` - Impact parameter
- `stellar_teff_k` - Stellar effective temperature
- `stellar_radius_rsun` - Stellar radius
- And many more...

### Missions Supported
- **Kepler**: KOI (Kepler Objects of Interest)
- **K2**: EPIC targets 
- **TESS**: TOI (TESS Objects of Interest)

## 🤖 AI Models

Our system uses multiple machine learning approaches:

1. **Tabular Model**: XGBoost classifier for structured data
2. **Neural Network**: ONNX-based CNN for light curve analysis
3. **Ensemble Model**: Fusion of multiple predictions
4. **Quality Control**: Automated vetting and validation

## 📈 API Endpoints

- `GET /inference/health` - Health check
- `POST /inference/upload` - Upload and parse dataset
- `POST /inference/predict` - Make predictions
- `POST /inference/predict-file` - Predict from uploaded file
- `POST /inference/explain` - SHAP explanations
- `POST /inference/vet` - Quality control vetting

## 🛠️ Development

### Project Structure
```
StarHarbor-2025/
├── api/                 # FastAPI backend
│   ├── models/         # Pydantic models
│   ├── routers/        # API endpoints
│   ├── services/       # Business logic
│   └── utils/          # Utilities
├── data/               # Datasets and processing
├── models/             # Trained ML models
├── frontend/           # Web interface
├── notebooks/          # Jupyter notebooks
└── scripts/            # Utility scripts
```

### Adding New Features

1. **New API endpoint**: Add to `api/routers/`
2. **New model**: Update `api/services/pipeline.py`
3. **Frontend changes**: Modify `frontend/index.html`

## 🧪 Testing

### Test the API
```bash
# Health check
curl http://localhost:8000/inference/health

# Upload test data
curl -X POST -F "file=@test_data.csv" http://localhost:8000/inference/upload
```

### Run with Sample Data
A sample CSV file `test_data.csv` is included with known exoplanet data for testing.

## 🌌 Team

**StarHarbor Team - NASA Space Apps Challenge 2025**

- Ihor Marchenko (@hort19345) - Team Lead, Ukraine
- Alina Koyun (@alinakoyun) - Developer, Ukraine  
- Veronika (@horobets) - Developer, Ukraine

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🏆 NASA Space Apps Challenge

This project was created for the 2025 NASA Space Apps Challenge as part of the Astrophysics Division challenge to develop AI/ML models for exoplanet detection using open-source datasets from NASA space missions.

**Challenge**: Create an AI/ML model trained on exoplanet datasets that can analyze new data to accurately identify exoplanets.

## 🔗 Links

- [NASA Space Apps Challenge](https://www.spaceappschallenge.org/)
- [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Docker Documentation](https://docs.docker.com/)
