"""
Pydantic models for API request/response validation.
"""
from pydantic import BaseModel, Field
from typing import Optional, List


class TransactionInput(BaseModel):
    """Input schema for a single transaction."""
    time: float = Field(..., description="Seconds elapsed between this transaction and first transaction")
    amount: float = Field(..., ge=0, description="Transaction amount")
    v1: float = Field(..., description="PCA feature V1")
    v2: float = Field(..., description="PCA feature V2")
    v3: float = Field(..., description="PCA feature V3")
    v4: float = Field(..., description="PCA feature V4")
    v5: float = Field(..., description="PCA feature V5")
    v6: float = Field(..., description="PCA feature V6")
    v7: float = Field(..., description="PCA feature V7")
    v8: float = Field(..., description="PCA feature V8")
    v9: float = Field(..., description="PCA feature V9")
    v10: float = Field(..., description="PCA feature V10")
    v11: float = Field(..., description="PCA feature V11")
    v12: float = Field(..., description="PCA feature V12")
    v13: float = Field(..., description="PCA feature V13")
    v14: float = Field(..., description="PCA feature V14")
    v15: float = Field(..., description="PCA feature V15")
    v16: float = Field(..., description="PCA feature V16")
    v17: float = Field(..., description="PCA feature V17")
    v18: float = Field(..., description="PCA feature V18")
    v19: float = Field(..., description="PCA feature V19")
    v20: float = Field(..., description="PCA feature V20")
    v21: float = Field(..., description="PCA feature V21")
    v22: float = Field(..., description="PCA feature V22")
    v23: float = Field(..., description="PCA feature V23")
    v24: float = Field(..., description="PCA feature V24")
    v25: float = Field(..., description="PCA feature V25")
    v26: float = Field(..., description="PCA feature V26")
    v27: float = Field(..., description="PCA feature V27")
    v28: float = Field(..., description="PCA feature V28")

    class Config:
        schema_extra = {
            "example": {
                "time": 406.0,
                "amount": 123.45,
                "v1": -1.35,
                "v2": -0.07,
                "v3": 2.54,
                "v4": 1.38,
                "v5": -0.34,
                "v6": 0.46,
                "v7": 0.24,
                "v8": 0.10,
                "v9": 0.36,
                "v10": 0.09,
                "v11": -0.55,
                "v12": -0.62,
                "v13": -0.99,
                "v14": -0.31,
                "v15": 1.47,
                "v16": -0.47,
                "v17": 0.21,
                "v18": 0.03,
                "v19": 0.40,
                "v20": 0.25,
                "v21": -0.02,
                "v22": 0.28,
                "v23": -0.11,
                "v24": 0.07,
                "v25": 0.13,
                "v26": 0.15,
                "v27": 0.02,
                "v28": 0.01
            }
        }


class PredictionOutput(BaseModel):
    """Output schema for fraud prediction."""
    is_fraud: bool = Field(..., description="Whether transaction is predicted as fraud")
    fraud_probability: float = Field(..., ge=0.0, le=1.0, description="Probability of fraud (0-1)")
    confidence: Optional[str] = Field(None, description="Confidence level (low/medium/high)")


class ModelInfoResponse(BaseModel):
    """Model information response."""
    model_name: str
    model_version: str
    trained_date: Optional[str]
    performance_metrics: Optional[dict]
