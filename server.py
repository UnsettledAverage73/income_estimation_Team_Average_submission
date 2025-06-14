from fastapi import FastAPI, HTTPException
from typing import Dict, Any
import pandas as pd
import numpy as np
import pickle
import joblib
import uvicorn

# === Load artifacts ===
try:
    model = pickle.load(open("ensemble_model.pkl", "rb"))
    preprocessor = joblib.load("preprocessor.joblib")
    target_encoder = pickle.load(open("target_encoder.pkl", "rb"))  # << MUST be saved during training
    expected_columns = preprocessor.feature_names_in_
except Exception as e:
    raise RuntimeError(f"Error loading model/preprocessor/encoder: {e}")

# === FastAPI Setup ===
app = FastAPI(
    title="Income Prediction API",
    description="Predict income using an ensemble model (XGBoost + CatBoost + RF)",
    version="1.0"
)

@app.get("/")
def read_root():
    return {"message": "✅ Income Prediction API is up and running!"}

@app.post("/predict")
def predict(input_data: Dict[str, Any]):
    try:
        # Convert to DataFrame
        df_input = pd.DataFrame([input_data])

        # --- Apply Target Encoding (city, state, gender, etc.) ---
        df_encoded = target_encoder.transform(df_input)

        # --- Ensure all expected columns exist ---
        for col in expected_columns:
            if col not in df_encoded.columns:
                df_encoded[col] = np.nan  # or a sensible default

        df_encoded = df_encoded[expected_columns]  # reorder

        # --- Preprocess and Predict ---
        processed_input = preprocessor.transform(df_encoded)
        prediction_log = model.predict(processed_input)
        predicted_income = np.expm1(prediction_log[0])  # back-transform

        return {
            "predicted_income": round(predicted_income, 2)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

# === Run API Server ===
if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)

