import os
import pickle
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import xgboost as xgb

# --- Load (or Train) Repayment Capability Model ---
repayment_model_path = 'repayment_capability_model.pkl'
if os.path.exists(repayment_model_path):
    with open(repayment_model_path, 'rb') as f:
        repayment_model = pickle.load(f)
    print("Repayment Capability Model loaded from 'repayment_capability_model.pkl'.")
else:
    # (Optional) If you want to train the model on startup, uncomment and adjust the training code.
    # For example, you can import and call a training function from model/repayment_capability_model.py.
    # (Here, we assume the model is already trained and saved.)
    raise FileNotFoundError("Repayment Capability Model pickle file not found. Please train and save the model first.")

# --- Load (or Train) Income Estimation Model ---
income_model_path = 'income_estimation_model.pkl'
if os.path.exists(income_model_path):
    with open(income_model_path, 'rb') as f:
        income_model = pickle.load(f)
    print("Income Estimation Model loaded from 'income_estimation_model.pkl'.")
else:
    # (Temporary block for testing: Train a minimal income model if the pickle file is not found.)
    print("Income Estimation Model pickle file not found. Training a minimal income model for testing...")
    data_path = 'data/Hackathon_bureau_data_400.csv'
    data = pd.read_csv(data_path)
    feature_cols = [col for col in data.columns if col.startswith('var_')]
    numeric_feature_cols = [col for col in feature_cols if np.issubdtype(data[col].dtype, np.number) or pd.to_numeric(data[col], errors='coerce').notnull().any()]
    demographic_features = ['age', 'gender', 'marital_status', 'residence_ownership']
    categorical_features = ['gender', 'marital_status', 'residence_ownership']
    features = numeric_feature_cols + demographic_features
    X = data[features]
    y = data['target_income']
    mask = ~(X.isna().any(axis=1) | y.isna())
    X = X[mask]
    y = y[mask]
    numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
    categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])
    preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, numeric_feature_cols), ('cat', categorical_transformer, categorical_features)])
    income_model = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', xgb.XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=6, min_child_weight=2, subsample=0.8, colsample_bytree=0.8, gamma=0.1, reg_alpha=0.1, reg_lambda=1, random_state=42))])
    income_model.fit(X, y)
    with open(income_model_path, 'wb') as f:
        pickle.dump(income_model, f)
    print("Income Estimation Model trained and saved as 'income_estimation_model.pkl'.")

# --- FastAPI Integration ---
# Define a Pydantic model for the input (prediction request) that includes keys for repayment and income.
class PredictionRequest(BaseModel):
    repayment: Dict[str, Any] = Field(..., description="Dictionary of features for repayment (repayment capability) prediction (e.g. var_1, age, gender, marital_status, residence_ownership)")
    income: Dict[str, Any] = Field(..., description="Dictionary of features for income estimation prediction (e.g. var_1, age, gender, marital_status, residence_ownership)")

# Define a Pydantic model for the output (prediction response) that includes predicted repayment (income) and predicted income (income) along with confidence scores.
class PredictionResponse(BaseModel):
    repayment_prediction: float = Field(..., description="Predicted repayment (income) (in ₹)")
    repayment_confidence: float = Field(..., description="Confidence score (mean of repayment model feature importances)")
    income_prediction: float = Field(..., description="Predicted income (in ₹)")
    income_confidence: float = Field(..., description="Confidence score (mean of income model feature importances)")

# Instantiate a FastAPI app.
app = FastAPI(title="Repayment & Income Prediction API", description="API to predict repayment (income) and income using trained models.")

# Define a POST endpoint /predict that accepts a PredictionRequest and returns a PredictionResponse.
@app.post("/predict", response_model=PredictionResponse, summary="Predict repayment (income) and income for a new customer.")
async def predict(request: PredictionRequest) -> PredictionResponse:
    try:
        # --- Repayment (Income) Prediction ---
        repayment_input = pd.DataFrame([request.repayment])
        repayment_prediction = repayment_model.predict(repayment_input)[0]
        repayment_feature_importance = repayment_model.named_steps['regressor'].feature_importances_
        repayment_confidence = float(np.mean(repayment_feature_importance))

        # --- Income Estimation Prediction ---
        income_input = pd.DataFrame([request.income])
        income_prediction = income_model.predict(income_input)[0]
        income_feature_importance = income_model.named_steps['regressor'].feature_importances_
        income_confidence = float(np.mean(income_feature_importance))

        return PredictionResponse(
            repayment_prediction=float(repayment_prediction),
            repayment_confidence=repayment_confidence,
            income_prediction=float(income_prediction),
            income_confidence=income_confidence
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# --- Example usage (if run as a script) ---
if __name__ == "__main__":
    import uvicorn
    # (Optional) run a simple test prediction if run as a script.
    repayment_example = {col: 0 for col in repayment_model.feature_names_in_ if col.startswith('var_')}
    repayment_example.update({'var_1': 100, 'var_2': 0.8, 'var_3': 1.0, 'var_4': 500, 'var_5': 0.5, 'age': 35, 'gender': 'M', 'marital_status': 'Married', 'residence_ownership': 'Own'})
    income_example = {col: 0 for col in income_model.feature_names_in_ if col.startswith('var_')}
    income_example.update({'var_1': 50, 'var_2': 0.3, 'var_3': 0.5, 'var_4': 200, 'var_5': 0.2, 'age': 25, 'gender': 'F', 'marital_status': 'Single', 'residence_ownership': 'Rent'})
    test_request = PredictionRequest(repayment=repayment_example, income=income_example)
    try:
        resp = predict(test_request)
        print("\nTest prediction (via API):")
        print(f"Repayment (Income) Prediction: ₹{resp.repayment_prediction:,.2f} (Confidence: {resp.repayment_confidence:.3f})")
        print(f"Income Prediction: ₹{resp.income_prediction:,.2f} (Confidence: {resp.income_confidence:.3f})")
    except Exception as e:
        print(f"Test prediction error: {e}")

    # (Optional) start the FastAPI server (using uvicorn) if run as a script.
    uvicorn.run(app, host="0.0.0.0", port=8000) 