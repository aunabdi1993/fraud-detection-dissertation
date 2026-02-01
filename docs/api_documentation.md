# API Documentation

## Fraud Detection API

This API provides endpoints for predicting credit card fraud using machine learning models.

### Base URL
```
http://localhost:8000
```

## Endpoints

### 1. Health Check

**GET** `/health`

Check if the API is running and model is loaded.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

### 2. Model Information

**GET** `/model-info`

Get information about the loaded model.

**Response:**
```json
{
  "model_type": "XGBClassifier",
  "status": "ready"
}
```

### 3. Single Prediction

**POST** `/predict`

Predict fraud for a single transaction.

**Request Body:**
```json
{
  "time": 406.0,
  "amount": 123.45,
  "v1": -1.35,
  "v2": -0.07,
  "v3": 2.54,
  ...
  "v28": 0.01
}
```

**Response:**
```json
{
  "is_fraud": false,
  "fraud_probability": 0.023,
  "transaction_id": null
}
```

### 4. Batch Prediction

**POST** `/batch-predict`

Predict fraud for multiple transactions.

**Request Body:**
```json
{
  "transactions": [
    {
      "time": 406.0,
      "amount": 123.45,
      "v1": -1.35,
      ...
    },
    {
      "time": 407.0,
      "amount": 67.89,
      "v1": -0.98,
      ...
    }
  ]
}
```

**Response:**
```json
{
  "predictions": [
    {
      "is_fraud": false,
      "fraud_probability": 0.023
    },
    {
      "is_fraud": true,
      "fraud_probability": 0.87
    }
  ]
}
```

## Error Codes

- `400` - Bad Request (invalid input)
- `500` - Internal Server Error
- `503` - Service Unavailable (model not loaded)

## Interactive Documentation

When the server is running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Example Usage

### Python
```python
import requests

url = "http://localhost:8000/predict"
data = {
    "time": 406.0,
    "amount": 123.45,
    "v1": -1.35,
    # ... other features
}

response = requests.post(url, json=data)
result = response.json()
print(f"Fraud: {result['is_fraud']}, Probability: {result['fraud_probability']}")
```

### cURL
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "time": 406,
    "amount": 123.45,
    "v1": -1.35,
    ...
  }'
```
