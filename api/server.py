import os
import sys

# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Optional
import pickle
import pandas as pd
import numpy as np
import logging
from model.repayment_capability_model import predict_repayment_capability

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Financial Prediction API",
    description="API for predicting customer repayment capability and income",
    version="1.0.0"
)

# Load both models
try:
    logger.info("Attempting to load repayment model...")
    with open('model/repayment_capability_model.pkl', 'rb') as f:
        repayment_model, repayment_scaler = pickle.load(f)
    logger.info("Successfully loaded repayment model")
    
    logger.info("Attempting to load income model...")
    with open('model/income_prediction_model.pkl', 'rb') as f:
        income_model, income_scaler = pickle.load(f)
    logger.info("Successfully loaded income model")
except FileNotFoundError as e:
    logger.error(f"Model file not found: {e}")
    repayment_model = None
    repayment_scaler = None
    income_model = None
    income_scaler = None
except Exception as e:
    logger.error(f"Error loading models: {e}")
    repayment_model = None
    repayment_scaler = None
    income_model = None
    income_scaler = None

class PredictionRequest(BaseModel):
    features: Dict[str, float | str]
    return_actual_values: Optional[bool] = False

class PredictionResponse(BaseModel):
    repayment_capability_score: float
    predicted_income: float
    confidence_score: float
    actual_repayment_score: Optional[float] = None
    actual_income: Optional[float] = None

@app.get("/")
async def root():
    return {
        "message": "Financial Prediction API is running",
        "endpoints": {
            "repayment": "/predict/repayment",
            "income": "/predict/income",
            "combined": "/predict/combined"
        }
    }

@app.get("/health")
async def health_check():
    models_status = {
        "repayment_model": repayment_model is not None,
        "income_model": income_model is not None
    }
    if not all(models_status.values()):
        raise HTTPException(
            status_code=503,
            detail=f"Models not loaded: {models_status}"
        )
    return {"status": "healthy", "models_loaded": models_status}

@app.post("/predict/repayment", response_model=PredictionResponse)
async def predict_repayment(request: PredictionRequest):
    if repayment_model is None or repayment_scaler is None:
        raise HTTPException(status_code=503, detail="Repayment model not loaded")
    
    try:
        # Get repayment prediction
        score, confidence = predict_repayment_capability(
            request.features,
            return_actual_income=request.return_actual_values
        )
        
        # If return_actual_values is True, score is already the actual value
        if request.return_actual_values:
            actual_score = score
            score = repayment_scaler.transform([[actual_score]])[0][0]
        else:
            actual_score = repayment_scaler.inverse_transform([[score]])[0][0]
        
        return PredictionResponse(
            repayment_capability_score=float(score),
            predicted_income=0.0,  # Not applicable for repayment endpoint
            confidence_score=float(confidence),
            actual_repayment_score=float(actual_score),
            actual_income=None
        )
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict/income", response_model=PredictionResponse)
async def predict_income(request: PredictionRequest):
    if income_model is None or income_scaler is None:
        raise HTTPException(status_code=503, detail="Income model not loaded")
    
    try:
        # Get income prediction
        input_data = pd.DataFrame([request.features])
        predicted_scaled = income_model.predict(input_data)[0]
        confidence = np.mean(income_model.feature_importances_)
        
        # Convert to actual income if requested
        if request.return_actual_values:
            actual_income = income_scaler.inverse_transform([[predicted_scaled]])[0][0]
            predicted_scaled = income_scaler.transform([[actual_income]])[0][0]
        else:
            actual_income = income_scaler.inverse_transform([[predicted_scaled]])[0][0]
        
        return PredictionResponse(
            repayment_capability_score=0.0,  # Not applicable for income endpoint
            predicted_income=float(predicted_scaled),
            confidence_score=float(confidence),
            actual_repayment_score=None,
            actual_income=float(actual_income)
        )
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict/combined", response_model=PredictionResponse)
async def predict_combined(request: PredictionRequest):
    if (repayment_model is None or repayment_scaler is None or 
        income_model is None or income_scaler is None):
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    try:
        # Get repayment prediction
        repayment_score, repayment_confidence = predict_repayment_capability(
            request.features,
            return_actual_income=request.return_actual_values
        )
        
        # Get income prediction
        input_data = pd.DataFrame([request.features])
        income_scaled = income_model.predict(input_data)[0]
        income_confidence = np.mean(income_model.feature_importances_)
        
        # Convert values if requested
        if request.return_actual_values:
            actual_repayment = repayment_score
            actual_income = income_scaler.inverse_transform([[income_scaled]])[0][0]
            repayment_score = repayment_scaler.transform([[actual_repayment]])[0][0]
            income_scaled = income_scaler.transform([[actual_income]])[0][0]
        else:
            actual_repayment = repayment_scaler.inverse_transform([[repayment_score]])[0][0]
            actual_income = income_scaler.inverse_transform([[income_scaled]])[0][0]
        
        # Average confidence scores
        combined_confidence = (repayment_confidence + income_confidence) / 2
        
        return PredictionResponse(
            repayment_capability_score=float(repayment_score),
            predicted_income=float(income_scaled),
            confidence_score=float(combined_confidence),
            actual_repayment_score=float(actual_repayment),
            actual_income=float(actual_income)
        )
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/feature-list")
async def get_feature_list():
    """Return the list of required features for prediction"""
    from model.repayment_capability_model import features as repayment_features
    # Assuming income model uses the same features
    return {
        "features": repayment_features,
        "note": "Both models use the same feature set"
    }

if __name__ == "__main__":
    import uvicorn
    try:
        logger.info("Starting FastAPI server...")
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8000,
            log_level="info"
        )
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        sys.exit(1) 