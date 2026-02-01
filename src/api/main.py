"""
FastAPI application for fraud detection.
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict
import numpy as np
import joblib
from pathlib import Path

app = FastAPI(
    title="Fraud Detection API",
    description="Credit card fraud detection using machine learning",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model storage
model = None
scaler = None


class Transaction(BaseModel):
    """Single transaction schema."""
    time: float
    amount: float
    v1: float
    v2: float
    v3: float
    v4: float
    v5: float
    v6: float
    v7: float
    v8: float
    v9: float
    v10: float
    v11: float
    v12: float
    v13: float
    v14: float
    v15: float
    v16: float
    v17: float
    v18: float
    v19: float
    v20: float
    v21: float
    v22: float
    v23: float
    v24: float
    v25: float
    v26: float
    v27: float
    v28: float


class PredictionResponse(BaseModel):
    """Prediction response schema."""
    is_fraud: bool
    fraud_probability: float
    transaction_id: str = None


class BatchPredictionRequest(BaseModel):
    """Batch prediction request schema."""
    transactions: List[Transaction]


@app.on_event("startup")
async def load_model():
    """Load model and scaler on startup."""
    global model, scaler

    # Load model (placeholder path)
    model_path = Path("models/production/best_model.pkl")
    if model_path.exists():
        model = joblib.load(model_path)

    # Load scaler (if exists)
    scaler_path = Path("models/production/scaler.pkl")
    if scaler_path.exists():
        scaler = joblib.load(scaler_path)


@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Fraud Detection API", "version": "1.0.0"}


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }


@app.get("/model-info")
async def model_info():
    """Get model information."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return {
        "model_type": type(model).__name__,
        "status": "ready"
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(transaction: Transaction):
    """
    Predict fraud for a single transaction.

    Args:
        transaction: Transaction data

    Returns:
        Prediction response with fraud probability
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Convert to feature array
        features = np.array([[
            transaction.time, transaction.amount,
            transaction.v1, transaction.v2, transaction.v3, transaction.v4,
            transaction.v5, transaction.v6, transaction.v7, transaction.v8,
            transaction.v9, transaction.v10, transaction.v11, transaction.v12,
            transaction.v13, transaction.v14, transaction.v15, transaction.v16,
            transaction.v17, transaction.v18, transaction.v19, transaction.v20,
            transaction.v21, transaction.v22, transaction.v23, transaction.v24,
            transaction.v25, transaction.v26, transaction.v27, transaction.v28
        ]])

        # Scale features if scaler available
        if scaler is not None:
            features = scaler.transform(features)

        # Make prediction
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0][1]

        return PredictionResponse(
            is_fraud=bool(prediction),
            fraud_probability=float(probability)
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/batch-predict")
async def batch_predict(request: BatchPredictionRequest):
    """
    Predict fraud for multiple transactions.

    Args:
        request: Batch prediction request

    Returns:
        List of predictions
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        predictions = []
        for transaction in request.transactions:
            result = await predict(transaction)
            predictions.append(result)

        return {"predictions": predictions}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")
