# Income Prediction Model

## Problem Statement
This project aims to predict an individual's income based on their credit bureau data and demographic information. The model uses various features such as credit utilization, loan behavior, credit limits, EMI payments, and demographic data to make accurate income predictions.

## Dataset
The model is trained on credit bureau data containing various financial and demographic features. The dataset includes:
- Credit utilization metrics
- Loan behavior patterns
- Credit limits and EMI information
- Demographic information (age, gender, location, etc.)
- Credit scores and risk indicators

The data is preprocessed to handle missing values, outliers, and categorical variables before training.

## Model Architecture
The income prediction model uses a pipeline approach with the following components:

1. **Preprocessing**:
   - Standard scaling for numerical features
   - One-hot encoding for categorical features
   - Feature engineering for derived metrics

2. **Model**: XGBoost Regressor with the following key features:
   - Log-transformed target variable for better prediction
   - Hyperparameter tuning using GridSearchCV
   - Feature importance analysis
   - Cross-validation for robust performance

## Performance Metrics
The model is evaluated using:
- RMSE (Root Mean Square Error)
- R² Score (Coefficient of Determination)

## How to Run

### Local Development
1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Train the model:
```bash
python income_prediction_model.py
```

### FastAPI Server
1. Start the FastAPI server:
```bash
uvicorn inference:app --reload
```

2. Access the API documentation at `http://localhost:8000/docs`

### API Endpoints
- `POST /predict`: Make income predictions
  - Input: JSON with applicant details
  - Output: Predicted income

Example request:
```json
{
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
```

## Model Files
- `income_prediction_model.py`: Main model implementation
- `inference.py`: FastAPI server for model deployment
- `requirements.txt`: Project dependencies
- `data/`: Directory containing training data and column mappings

## Future Improvements
- Integration with AWS SageMaker for production deployment
- Additional feature engineering
- Model versioning and monitoring
- API authentication and rate limiting