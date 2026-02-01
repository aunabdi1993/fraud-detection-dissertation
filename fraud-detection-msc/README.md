# Financial Fraud Detection: Class Imbalance Techniques Comparison

**MSc Computer Science Dissertation Project**  
**University of London, City **

---

## 📋 Project Overview

This project provides a comprehensive comparison of class imbalance handling techniques for financial fraud detection, with emphasis on production deployability and practical constraints.

### Research Question
> "How do different class imbalance handling techniques compare in performance and practical deployability for financial fraud detection?"

### Key Objectives
1. **Rigorous Comparison**: Evaluate 10-12 existing class imbalance techniques
2. **Production Implementation**: Build a deployment-ready fraud detection API
3. **Practical Insights**: Analyze trade-offs between accuracy, latency, and explainability
4. **Statistical Rigor**: Apply proper experimental methodology with significance testing

---

## 🎯 What Makes This Project Valuable

- ✅ Addresses real-world problem (credit card fraud detection)
- ✅ Comprehensive experimental methodology with statistical validation
- ✅ Production-ready system (not just notebooks)
- ✅ Focus on deployment constraints (latency, explainability, resource requirements)
- ✅ Clear documentation and reproducible results
- ✅ Practical insights for ML practitioners

---

## 🗂️ Project Structure

```
fraud-detection-msc/
│
├── README.md                          # This file
├── DISSERTATION_PLAN.md               # Detailed 11-month timeline
├── requirements.txt                   # Python dependencies
├── .gitignore                         # Git ignore rules
├── .env.example                       # Environment variables template
│
├── notebooks/                         # Jupyter notebooks for exploration
│   ├── 01_eda.ipynb                  # Exploratory Data Analysis
│   ├── 02_feature_engineering.ipynb   # Feature creation experiments
│   ├── 03_baseline_models.ipynb       # Initial model testing
│   ├── 04_sampling_comparison.ipynb   # Class imbalance techniques
│   └── 05_results_analysis.ipynb      # Final results visualization
│
├── data/                              # Data directory (gitignored)
│   ├── raw/                           # Original Kaggle dataset
│   ├── processed/                     # Cleaned, split data
│   └── features/                      # Feature-engineered datasets
│
├── src/                               # Source code (production quality)
│   ├── data/                          # Data handling modules
│   ├── features/                      # Feature engineering
│   ├── models/                        # Model implementations
│   ├── evaluation/                    # Metrics and evaluation
│   └── api/                           # FastAPI application
│
├── experiments/                       # Experiment tracking
│   ├── configs/                       # Experiment configurations (YAML)
│   └── runs/                          # MLflow experiment runs
│
├── models/                            # Saved models (gitignored)
│   ├── baseline/                      # Baseline model artifacts
│   ├── sampling/                      # Sampling technique models
│   └── production/                    # Best model for deployment
│
├── tests/                             # Unit and integration tests
│   ├── test_data.py
│   ├── test_features.py
│   ├── test_models.py
│   └── test_api.py
│
├── scripts/                           # Utility scripts
│   ├── download_data.py               # Download Kaggle dataset
│   ├── run_experiments.py             # Execute all experiments
│   ├── train_best_model.py            # Train final production model
│   └── evaluate_all.py                # Comprehensive evaluation
│
├── results/                           # Results and visualizations
│   ├── figures/                       # All plots (PR curves, ROC curves)
│   ├── tables/                        # Results tables (CSV/LaTeX)
│   └── metrics/                       # JSON metrics files
│
├── dissertation/                      # Dissertation writing
│   ├── latex/                         # LaTeX source files
│   └── notes/                         # Literature notes, drafts
│
├── docker/                            # Docker configuration
│   ├── Dockerfile
│   └── docker-compose.yml
│
└── docs/                              # Project documentation
    ├── api_documentation.md
    ├── experiment_guide.md
    └── deployment_guide.md
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- pip or conda
- Git
- (Optional) Kaggle API credentials

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/fraud-detection-msc.git
cd fraud-detection-msc
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**
```bash
cp .env.example .env
# Edit .env with your Kaggle credentials (optional)
```

5. **Download dataset**
```bash
python scripts/download_data.py
```

### Running the Project

**Exploratory Data Analysis**
```bash
jupyter notebook notebooks/01_eda.ipynb
```

**Train Baseline Models**
```bash
python scripts/run_experiments.py --config experiments/configs/baseline.yaml
```

**Run All Experiments**
```bash
python scripts/run_experiments.py --config experiments/configs/sampling.yaml
python scripts/run_experiments.py --config experiments/configs/ensemble.yaml
```

**Start API Server (Development)**
```bash
uvicorn src.api.main:app --reload
```

**Run Tests**
```bash
pytest tests/ -v --cov=src
```

---

## 🔬 Methodology

### Dataset
- **Source**: [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- **Size**: 284,807 transactions
- **Class Distribution**: 0.172% fraud (492 fraudulent out of 284,807)
- **Features**: 28 PCA-transformed features + Time + Amount

### Feature Engineering
- **RFM Features**: Recency, Frequency, Monetary value
- **Velocity Features**: Transaction rate, amount velocity
- **Aggregation Features**: Rolling statistics, historical patterns

### Models Evaluated

**Baseline Models**
- Logistic Regression
- Decision Tree
- Random Forest

**Sampling Techniques**
- Random Undersampling
- Random Oversampling
- SMOTE (Synthetic Minority Over-sampling Technique)
- SMOTE + Tomek Links
- ADASYN (Adaptive Synthetic Sampling)

**Ensemble Methods**
- Balanced Random Forest
- RUSBoost
- EasyEnsemble
- XGBoost with `scale_pos_weight`
- LightGBM with `is_unbalance`

**Cost-Sensitive Learning**
- Class weights optimization
- Custom threshold tuning

### Evaluation Metrics
- **Primary**: PR-AUC (Precision-Recall Area Under Curve)
- **Secondary**: ROC-AUC, F1-Score, Precision, Recall
- **Practical**: Inference latency (ms), Memory footprint
- **Statistical**: Paired t-tests for significance

### Experimental Design
- **Data Split**: 60% train, 20% validation, 20% test (stratified)
- **Cross-Validation**: 5-fold stratified CV on training set
- **Random Seeds**: Fixed for reproducibility
- **Hyperparameter Tuning**: Optuna with validation set

---

## 📊 Expected Results

The project will deliver:

1. **Comprehensive Comparison Table**
   - Performance metrics for all 10-12 approaches
   - Statistical significance tests
   - Latency analysis

2. **Visualization Suite**
   - PR curves and ROC curves
   - Confusion matrices
   - Feature importance plots
   - Performance vs. latency trade-off chart

3. **Production API**
   - RESTful API with FastAPI
   - `/predict` endpoint (single transaction)
   - `/batch-predict` endpoint (multiple transactions)
   - Deployed on Railway/Render with health checks

4. **Dissertation Document**
   - 10,000-word thesis
   - Literature review of 25 key papers
   - Complete methodology and results chapters
   - Discussion of practical implications

---

## 🛠️ Technology Stack

**Core ML Stack**
- `scikit-learn` - Baseline models, preprocessing
- `imbalanced-learn` - Sampling techniques
- `xgboost` / `lightgbm` - Gradient boosting
- `pandas` / `numpy` - Data manipulation

**Experiment Tracking**
- `mlflow` - Experiment logging and model registry
- `optuna` - Hyperparameter optimization

**API & Deployment**
- `fastapi` - REST API framework
- `pydantic` - Data validation
- `uvicorn` - ASGI server
- `docker` - Containerization

**Testing & Quality**
- `pytest` - Unit and integration tests
- `pytest-cov` - Code coverage
- `black` / `flake8` - Code formatting and linting

**Visualization**
- `matplotlib` / `seaborn` - Statistical plots
- `plotly` - Interactive visualizations

---

## 📅 Project Timeline

| Phase | Months | Key Deliverables |
|-------|--------|------------------|
| Literature Review | Feb-Mar | Research proposal, 25 papers reviewed |
| Data & Baseline | Apr-May | EDA report, 3 baseline models |
| Model Comparison | Jun-Jul | 10-12 techniques evaluated |
| Production API | Aug-Sep | Deployed fraud detection API |
| Results & Analysis | Oct-Nov | Complete experiments, statistical tests |
| Writing Sprint | Dec | 10,000-word dissertation submitted |

*See [DISSERTATION_PLAN.md](DISSERTATION_PLAN.md) for detailed weekly breakdown.*

---

## 🧪 Running Experiments

### Single Experiment
```bash
python scripts/run_experiments.py \
  --config experiments/configs/baseline.yaml \
  --output results/baseline/
```

### Full Comparison
```bash
# Run all experiments sequentially
./scripts/run_all_experiments.sh

# View results in MLflow UI
mlflow ui --backend-store-uri experiments/runs/
```

### Reproduce Paper Results
```bash
# Train best model from paper
python scripts/train_best_model.py \
  --model xgboost \
  --sampling smote \
  --output models/production/

# Evaluate on test set
python scripts/evaluate_all.py \
  --model models/production/best_model.pkl \
  --test-data data/processed/test.csv
```

---

## 🌐 API Usage

### Start Server
```bash
# Development
uvicorn src.api.main:app --reload --port 8000

# Production (Docker)
docker-compose up
```

### Example Requests

**Single Prediction**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "time": 406,
    "amount": 123.45,
    "v1": -1.35, "v2": -0.07, ..., "v28": 0.13
  }'
```

**Batch Prediction**
```bash
curl -X POST "http://localhost:8000/batch-predict" \
  -H "Content-Type: application/json" \
  -d '{
    "transactions": [
      {"time": 406, "amount": 123.45, ...},
      {"time": 407, "amount": 67.89, ...}
    ]
  }'
```

**Health Check**
```bash
curl http://localhost:8000/health
```

*Full API documentation available at: http://localhost:8000/docs (Swagger UI)*

---

## 🧰 Development

### Code Quality
```bash
# Format code
black src/ tests/

# Lint
flake8 src/ tests/

# Type checking
mypy src/
```

### Testing
```bash
# Run all tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=src --cov-report=html

# Specific test file
pytest tests/test_api.py -v
```

### Pre-commit Hooks
```bash
# Install pre-commit hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

---

## 📝 Documentation

- **API Documentation**: Auto-generated Swagger UI at `/docs`
- **Experiment Guide**: See `docs/experiment_guide.md`
- **Deployment Guide**: See `docs/deployment_guide.md`
- **Code Documentation**: Docstrings follow Google style

---

## 📚 Citation

If you use this work, please cite:

```bibtex
@mastersthesis{yourname2026fraud,
  title={Class Imbalance Techniques for Financial Fraud Detection: A Comprehensive Comparison},
  author={Your Name},
  year={2026},
  school={University of London, City},
  type={MSc Dissertation}
}
```

---

*This README is a living document and will be updated throughout the project lifecycle.*