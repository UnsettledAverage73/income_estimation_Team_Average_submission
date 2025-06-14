import os
import sys
import logging
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
from typing import Dict, Optional, Union, List
import pickle
import pandas as pd
import numpy as np

# Add project root to Python path
project_root = Path(__file__).parent
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
    title="Bureau Loan Repayment Prediction API",
    description="API for predicting bureau loan repayment status",
    version="1.0.0"
)

# Model instance
model_data = None

# Request/Response Models
class Features(BaseModel):
    var_0: float = Field(..., description="Balance 1")
    var_1: float = Field(..., description="Balance 2")
    var_2: float = Field(..., description="Credit limit 1")
    var_3: float = Field(..., description="Credit limit 2")
    var_4: float = Field(..., description="Balance 3")
    var_5: float = Field(..., description="Credit limit 3")
    var_6: float = Field(..., description="Loan amount 1")
    var_7: float = Field(..., description="Loan amount 2")
    var_8: float = Field(..., description="Business balance")
    var_9: float = Field(..., description="Total EMI 1")
    var_10: float = Field(..., description="Active credit limit 1")
    var_11: float = Field(..., description="Credit limit recent 1")
    var_12: float = Field(..., description="Credit limit 4")
    var_13: float = Field(..., description="Loan amount large tenure")
    var_14: float = Field(..., description="Primary loan amount")
    var_15: float = Field(..., description="Total inquiries 1")
    var_16: float = Field(..., description="Total inquiries 2")
    var_17: float = Field(..., description="Total EMI 2")
    var_18: float = Field(..., description="Balance 4")
    var_19: float = Field(..., description="Balance 5")
    var_20: float = Field(..., description="Loan amount 3")
    var_21: float = Field(..., description="Balance 6")
    var_22: float = Field(..., description="Credit limit 5")
    var_23: float = Field(..., description="Credit limit 6")
    var_24: float = Field(..., description="Loan amount recent")
    var_25: float = Field(..., description="Total inquiries recent")
    var_26: float = Field(..., description="Credit limit 7")
    var_27: float = Field(..., description="Credit limit 8")
    age: int = Field(..., ge=18, le=100, description="Age of the applicant")
    var_28: float = Field(..., description="Credit limit 9")
    var_29: float = Field(..., description="Credit limit 10")
    var_30: float = Field(..., description="Balance 7")
    var_31: float = Field(..., description="Loan amount 4")
    var_32: float = Field(..., description="Credit score")
    var_33: float = Field(..., description="Credit limit 11")
    var_34: float = Field(..., description="Balance 8")
    var_35: float = Field(..., description="Balance 9")
    var_36: float = Field(..., description="Loan amount 5")
    var_37: float = Field(..., description="Repayment 1")
    var_38: float = Field(..., description="Balance 10")
    var_39: float = Field(..., description="Loan amount 6")
    var_40: float = Field(..., description="Closed loan")
    var_41: float = Field(..., description="Total EMI 3")
    var_42: float = Field(..., description="Loan amount 7")
    var_43: float = Field(..., description="Total EMI 4")
    var_44: float = Field(..., description="Credit limit 12")
    var_45: float = Field(..., description="Total inquiries 3")
    var_46: float = Field(..., description="Total EMI 5")
    var_47: float = Field(..., description="Credit limit 13")
    var_48: float = Field(..., description="Repayment 2")
    var_49: float = Field(..., description="Repayment 3")
    var_50: float = Field(..., description="Repayment 4")
    var_51: float = Field(..., description="Total EMI 6")
    var_52: float = Field(..., description="Repayment 5")
    var_53: float = Field(..., description="Total loans 1")
    var_54: float = Field(..., description="Closed total loans")
    var_55: float = Field(..., description="Repayment 6")
    var_56: float = Field(..., description="Total EMI 7")
    var_57: float = Field(..., description="Total loans 2")
    var_58: float = Field(..., description="Total inquiries 4")
    var_59: float = Field(..., description="Balance 11")
    var_60: float = Field(..., description="Total loans 2")
    var_61: float = Field(..., description="Total inquiries 5")
    var_62: float = Field(..., description="Total loan recent")
    var_63: float = Field(..., description="Total loans 3")
    var_64: float = Field(..., description="Total loans 4")
    var_65: float = Field(..., description="Loan amount 8")
    var_66: float = Field(..., description="Total loans 5")
    var_67: float = Field(..., description="Repayment 7")
    var_68: float = Field(..., description="Balance 12")
    var_69: float = Field(..., description="Repayment 8")
    var_70: float = Field(..., description="Repayment 9")
    var_71: float = Field(..., description="Total inquiries 6")
    var_72: float = Field(..., description="Loan amount 9")
    var_73: float = Field(..., description="Repayment 10")
    var_74: str = Field(..., description="Score comments")
    var_75: str = Field(..., description="Score type")
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

# Load model on startup
@app.on_event("startup")
async def startup_event():
    global model_data
    
    try:
        logger.info("Loading bureau loan repayment model...")
        model_path = project_root / 'model' / 'Rigorious' / 'bureau_loan_repayment_model.pkl'
        
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        logger.info("Successfully loaded bureau model")
        
    except Exception as e:
        logger.error(f"Error during startup: {e}", exc_info=True)
        raise

@app.get("/")
async def root():
    return {
        "message": "Bureau Loan Repayment Prediction API is running",
        "version": "1.0.0",
        "endpoints": {
            "bureau": "/predict/bureau",
            "health": "/health"
        }
    }

@app.get("/health")
async def health_check():
    if model_data is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded"
        )
    return {
        "status": "healthy",
        "model_loaded": True,
        "version": "1.0.0"
    }

@app.post("/predict/bureau", response_model=BureauResponse)
async def predict_bureau(request: PredictionRequest):
    try:
        # Convert features to DataFrame
        features_dict = request.features.model_dump()  # Updated from dict() to model_dump()
        input_df = pd.DataFrame([features_dict])
        
        # Get prediction using the model pipeline (regressor)
        model = model_data['model']
        prediction = float(model.predict(input_df)[0])
        
        # Convert regression output to probability using sigmoid
        probability = 1 / (1 + np.exp(-prediction))
        
        # Apply threshold if provided
        threshold = request.threshold if request.threshold is not None else 0.5
        prediction_class = "GOOD" if probability >= threshold else "BAD"
        
        return BureauResponse(
            prediction=prediction_class,
            probability=probability,
            threshold=threshold
        )
    except Exception as e:
        logger.error(f"Error in bureau prediction: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=400,
            detail=f"Error making prediction: {str(e)}"
        )

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