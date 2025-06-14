import os
import sys
import logging
from pathlib import Path
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from typing import Dict, Optional, Union, List
import pickle
import pandas as pd
import numpy as np
from model.Rigorious.bureau_loan_repayment_model import BureauLoanRepaymentModel

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(project_root / 'api.log')
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Financial Prediction API",
    description="API for predicting bureau loan repayment and income",
    version="1.0.0"
)

# Model instances
bureau_model = None
    income_model = None
    income_scaler = None

# Request/Response Models
class Features(BaseModel):
    var_0: float = Field(..., description="Balance 1")
    var_1: float = Field(..., description="Balance 2")
    var_2: float = Field(..., description="Credit limit 1")
    age: int = Field(..., ge=18, le=100, description="Age of the applicant")
    gender: str = Field(..., description="Gender of the applicant")
    marital_status: str = Field(..., description="Marital status of the applicant")
    city: str = Field(..., description="City of residence")
    state: str = Field(..., description="State of residence")
    residence_ownership: str = Field(..., description="Residence ownership status")

    @validator('gender')
    def validate_gender(cls, v):
        valid_genders = ['MALE', 'FEMALE', 'OTHER']
        if v.upper() not in valid_genders:
            raise ValueError(f'Gender must be one of {valid_genders}')
        return v.upper()

    @validator('marital_status')
    def validate_marital_status(cls, v):
        valid_statuses = ['SINGLE', 'MARRIED', 'DIVORCED', 'WIDOWED']
        if v.upper() not in valid_statuses:
            raise ValueError(f'Marital status must be one of {valid_statuses}')
        return v.upper()

    @validator('residence_ownership')
    def validate_residence_ownership(cls, v):
        valid_types = ['SELF-OWNED', 'RENTED', 'PARENTAL', 'COMPANY-PROVIDED']
        if v.upper() not in valid_types:
            raise ValueError(f'Residence ownership must be one of {valid_types}')
        return v.upper()

class PredictionRequest(BaseModel):
    features: Features
    threshold: Optional[float] = Field(0.5, ge=0.0, le=1.0, description="Threshold for bureau loan repayment prediction")

class BureauResponse(BaseModel):
    prediction: int = Field(..., description="Predicted repayment status (1 for Good, 0 for Bad)")
    probability: float = Field(..., ge=0.0, le=1.0, description="Probability of good repayment")
    status: str = Field(..., description="Predicted repayment status (Good/Bad)")
    threshold: float = Field(..., description="Threshold used for prediction")

class IncomeResponse(BaseModel):
    predicted_income: float = Field(..., ge=0.0, description="Predicted income")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Confidence score of the prediction")

class CombinedResponse(BaseModel):
    bureau_prediction: BureauResponse
    income_prediction: IncomeResponse

class ErrorResponse(BaseModel):
    detail: str
    error_type: str

# Exception handlers
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global error handler caught: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            detail=str(exc),
            error_type=exc.__class__.__name__
        ).dict()
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    logger.error(f"HTTP error handler caught: {str(exc)}")
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            detail=exc.detail,
            error_type="HTTPException"
        ).dict()
    )

# Load models on startup
@app.on_event("startup")
async def startup_event():
    global bureau_model, income_model, income_scaler
    
    try:
        # Initialize bureau model
        logger.info("Loading bureau loan repayment model...")
        bureau_model = BureauLoanRepaymentModel()
        bureau_model.load_column_mapping()
        
        bureau_model_path = project_root / 'model' / 'Rigorious' / 'bureau_loan_repayment_model.pkl'
        try:
            bureau_model.load_model(str(bureau_model_path))
            logger.info("Successfully loaded bureau model")
        except FileNotFoundError:
            logger.info("Training new bureau model...")
            bureau_model.train()
            bureau_model.save_model(str(bureau_model_path))
            logger.info("Successfully trained and saved bureau model")

        # Load income model
        logger.info("Loading income prediction model...")
        income_model_path = project_root / 'model' / 'Rigorious' / 'income_prediction_model.pkl'
        try:
            with open(income_model_path, 'rb') as f:
                income_model, income_scaler = pickle.load(f)
            logger.info("Successfully loaded income model")
        except FileNotFoundError:
            logger.error(f"Income prediction model not found at {income_model_path}")
            income_model = None
            income_scaler = None

    except Exception as e:
        logger.error(f"Error during startup: {e}", exc_info=True)
        raise

@app.get("/")
async def root():
    return {
        "message": "Financial Prediction API is running",
        "version": "1.0.0",
        "endpoints": {
            "bureau": "/predict/bureau",
            "income": "/predict/income",
            "combined": "/predict/combined",
            "health": "/health"
        }
    }

@app.get("/health")
async def health_check():
    models_status = {
        "bureau_model": bureau_model is not None,
        "income_model": income_model is not None and income_scaler is not None
    }
    if not all(models_status.values()):
        raise HTTPException(
            status_code=503,
            detail=f"Models not loaded: {models_status}"
        )
    return {
        "status": "healthy",
        "models_loaded": models_status,
        "version": "1.0.0"
    }

@app.post("/predict/bureau", response_model=BureauResponse)
async def predict_bureau(request: PredictionRequest):
    if bureau_model is None:
        raise HTTPException(status_code=503, detail="Bureau model not loaded")
    
    try:
        # Convert features to dict
        features_dict = request.features.dict()
        
        # Set custom threshold if provided
        if request.threshold is not None:
            bureau_model.set_threshold(request.threshold)
        
        # Get prediction
        prediction = bureau_model.predict(features_dict)
        probability = bureau_model.predict(features_dict, return_proba=True)
        
        return BureauResponse(
            prediction=int(prediction),
            probability=float(probability),
            status="Good" if prediction == 1 else "Bad",
            threshold=float(bureau_model.threshold)
        )
    
    except Exception as e:
        logger.error(f"Error in bureau prediction: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict/income", response_model=IncomeResponse)
async def predict_income(request: PredictionRequest):
    if income_model is None or income_scaler is None:
        raise HTTPException(status_code=503, detail="Income model not loaded")
    
    try:
        # Convert features to DataFrame
        features_dict = request.features.dict()
        input_data = pd.DataFrame([features_dict])
        
        # Get prediction
        predicted_scaled = income_model.predict(input_data)[0]
        confidence = np.mean(income_model.feature_importances_)
        
        # Convert to actual income
            actual_income = income_scaler.inverse_transform([[predicted_scaled]])[0][0]
        
        return IncomeResponse(
            predicted_income=float(actual_income),
            confidence_score=float(confidence)
        )
    
    except Exception as e:
        logger.error(f"Error in income prediction: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict/combined", response_model=CombinedResponse)
async def predict_combined(request: PredictionRequest):
    if bureau_model is None or income_model is None or income_scaler is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    try:
        # Get bureau prediction
        bureau_response = await predict_bureau(request)
        
        # Get income prediction
        income_response = await predict_income(request)
        
        return CombinedResponse(
            bureau_prediction=bureau_response,
            income_prediction=income_response
        )
    
    except Exception as e:
        logger.error(f"Error in combined prediction: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))

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
        logger.error(f"Failed to start server: {e}", exc_info=True)
        sys.exit(1) 