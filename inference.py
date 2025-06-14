import os
import json
import logging
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from model.Rigorious.income_prediction_model import IncomePredictionModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Income Prediction API",
    description="API for predicting income based on credit bureau data",
    version="1.0.0"
)

# Initialize model
model = None

class ApplicantData(BaseModel):
    """Input data model for income prediction"""
    age: int = Field(..., ge=18, le=100, description="Age of the applicant")
    gender: str = Field(..., description="Gender of the applicant (MALE/FEMALE/OTHER)")
    marital_status: str = Field(..., description="Marital status of the applicant")
    city: str = Field(..., description="City of residence")
    state: str = Field(..., description="State of residence")
    residence_ownership: str = Field(..., description="Type of residence ownership")
    credit_score: float = Field(..., ge=300, le=900, description="Credit score")
    total_loan_amount: float = Field(..., ge=0, description="Total loan amount")
    total_credit_limit: float = Field(..., ge=0, description="Total credit limit")
    total_emi: float = Field(..., ge=0, description="Total EMI amount")
    
    class Config:
        schema_extra = {
            "example": {
                "age": 35,
                "gender": "MALE",
                "marital_status": "MARRIED",
                "city": "Mumbai",
                "state": "Maharashtra",
                "residence_ownership": "SELF-OWNED",
                "credit_score": 750,
                "total_loan_amount": 500000,
                "total_credit_limit": 1000000,
                "total_emi": 15000
            }
        }

class PredictionResponse(BaseModel):
    """Output data model for income prediction"""
    predicted_income: float = Field(..., description="Predicted annual income in INR")
    confidence_score: float = Field(..., description="Model confidence score (0-1)")

def load_model():
    """Load the trained model"""
    global model
    try:
        model = IncomePredictionModel()
        model.load_model()
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise HTTPException(status_code=500, detail="Error loading model")

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    load_model()

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"status": "healthy", "message": "Income Prediction API is running"}

@app.post("/predict", response_model=PredictionResponse)
async def predict(applicant_data: ApplicantData):
    """Make income prediction for applicant data"""
    try:
        # Convert input data to dictionary
        input_data = applicant_data.dict()
        
        # Make prediction
        predicted_income = model.predict(input_data)
        
        # Calculate confidence score (placeholder - replace with actual confidence calculation)
        confidence_score = 0.85  # This should be replaced with actual confidence calculation
        
        return PredictionResponse(
            predicted_income=float(predicted_income),
            confidence_score=confidence_score
        )
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# SageMaker inference functions
def model_fn(model_dir: str) -> IncomePredictionModel:
    """Load model for SageMaker inference"""
    model = IncomePredictionModel()
    model.load_model(os.path.join(model_dir, "income_prediction_model.pkl"))
    return model

def input_fn(request_body: str, request_content_type: str) -> Dict[str, Any]:
    """Parse input data for SageMaker inference"""
    if request_content_type == "application/json":
        return json.loads(request_body)
    raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data: Dict[str, Any], model: IncomePredictionModel) -> Dict[str, float]:
    """Make prediction for SageMaker inference"""
    try:
        predicted_income = model.predict(input_data)
        return {
            "predicted_income": float(predicted_income),
            "confidence_score": 0.85  # Replace with actual confidence calculation
        }
    except Exception as e:
        logger.error(f"SageMaker prediction error: {str(e)}")
        raise

def output_fn(prediction: Dict[str, float], response_content_type: str) -> str:
    """Format output for SageMaker inference"""
    if response_content_type == "application/json":
        return json.dumps(prediction)
    raise ValueError(f"Unsupported content type: {response_content_type}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 