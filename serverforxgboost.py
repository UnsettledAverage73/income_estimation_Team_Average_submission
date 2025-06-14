from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import numpy as np
import pandas as pd
import pickle
from typing import List, Dict, Optional
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

# Initialize FastAPI app
app = FastAPI(
    title="Income Prediction API",
    description="API for predicting income using XGBoost model",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Load the saved model and preprocessor
try:
    with open('ensemble_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('preprocessor.pkl', 'rb') as f:
        preprocessor = pickle.load(f)
    print("✅ Model and preprocessor loaded successfully")
except Exception as e:
    print(f"Error loading model or preprocessor: {e}")
    raise

# Define input data model using Pydantic
class PredictionInput(BaseModel):
    # Add all the features that your model expects
    # These should match the features used during training
    age: Optional[float] = Field(None, description="Age of the person")
    credit_score: Optional[float] = Field(None, description="Credit score")
    total_credit_limit_overall: Optional[float] = Field(None, description="Total credit limit")
    total_balance_overall: Optional[float] = Field(None, description="Total balance")
    total_emi_overall: Optional[float] = Field(None, description="Total EMI")
    total_inquiries_overall: Optional[float] = Field(None, description="Total inquiries")
    total_loan_amount_overall: Optional[float] = Field(None, description="Total loan amount")
    city: Optional[str] = Field(None, description="City of residence")
    state: Optional[str] = Field(None, description="State of residence")
    marital_status: Optional[str] = Field(None, description="Marital status")
    gender: Optional[str] = Field(None, description="Gender")
    residence_ownership: Optional[str] = Field(None, description="Residence ownership status")
    
    # Add any other features your model expects
    # Make sure to include all features that were used during training

class PredictionResponse(BaseModel):
    predicted_income: float = Field(..., description="Predicted income in original scale")
    prediction_confidence: float = Field(..., description="Confidence score of the prediction")

@app.get("/")
async def root():
    """Root endpoint returning API information"""
    return {
        "message": "Welcome to the Income Prediction API",
        "endpoints": {
            "/predict": "POST - Make a prediction",
            "/health": "GET - Check API health",
            "/model-info": "GET - Get model information"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model_loaded": model is not None}

@app.get("/model-info")
async def model_info():
    """Get information about the model"""
    return {
        "model_type": type(model).__name__,
        "preprocessor_type": type(preprocessor).__name__,
        "feature_names": preprocessor.get_feature_names_out().tolist()
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(input_data: PredictionInput):
    """
    Make a prediction using the loaded model
    """
    try:
        # Convert input data to DataFrame
        input_dict = input_data.dict()
        input_df = pd.DataFrame([input_dict])
        
        # Preprocess the input data
        processed_input = preprocessor.transform(input_df)
        
        # Make prediction
        prediction_log = model.predict(processed_input)[0]
        
        # Convert prediction back to original scale
        prediction_original = np.expm1(prediction_log)
        
        # Calculate a simple confidence score (you might want to implement a more sophisticated method)
        # This is just a placeholder - you should implement proper confidence scoring
        confidence_score = 0.8  # Placeholder confidence score
        
        return PredictionResponse(
            predicted_income=float(prediction_original),
            prediction_confidence=float(confidence_score)
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )

if __name__ == "__main__":
    # Run the server
    uvicorn.run(app, host="0.0.0.0", port=8000) 