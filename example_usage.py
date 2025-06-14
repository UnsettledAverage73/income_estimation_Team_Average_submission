import pandas as pd
import numpy as np
import pickle
import json
import joblib
from typing import Dict, Any, Optional

def load_model_and_preprocessor():
    """Load the trained ensemble model and preprocessor from disk."""
    try:
        # Load the ensemble model (saved as pickle) and the preprocessor (saved as joblib)
        with open("ensemble_model.pkl", "rb") as f:
            model = pickle.load(f)
        preprocessor = joblib.load("preprocessor.joblib")
        print("Model and preprocessor loaded successfully.")
        return model, preprocessor
    except Exception as e:
        print(f"Error loading model or preprocessor: {e}")
        raise

def get_all_expected_features() -> Dict[str, Any]:
    """Get all features expected by the preprocessor."""
    # Basic features
    base_features = {
        # Demographic
        'age': None,
        'gender': None,
        'marital_status': None,
        'city': None,
        'state': None,
        'residence_ownership': None,
        'device_category': None,
        'device_manufacturer': None,
        'device_model': None,
        'platform': None,
        'score_type': None,
        'score_comments': None,
        
        # Financial - Credit Limits
        'credit_limit_1': None,
        'credit_limit_2': None,
        'credit_limit_3': None,
        'credit_limit_4': None,
        'credit_limit_5': None,
        'credit_limit_6': None,
        'credit_limit_7': None,
        'credit_limit_8': None,
        'credit_limit_9': None,
        'credit_limit_10': None,
        'credit_limit_11': None,
        'credit_limit_12': None,
        'credit_limit_13': None,
        'credit_limit_recent_1': None,
        'active_credit_limit_1': None,
        
        # Financial - Balances
        'balance_1': None,
        'balance_2': None,
        'balance_3': None,
        'balance_4': None,
        'balance_5': None,
        'balance_6': None,
        'balance_7': None,
        'balance_8': None,
        'balance_9': None,
        'balance_10': None,
        'balance_11': None,
        'balance_12': None,
        'business_balance': None,
        
        # Financial - EMIs
        'total_emi_1': None,
        'total_emi_2': None,
        'total_emi_3': None,
        'total_emi_4': None,
        'total_emi_5': None,
        'total_emi_6': None,
        'total_emi_7': None,
        
        # Financial - Inquiries
        'total_inquiries_1': None,
        'total_inquiries_2': None,
        'total_inquiries_recent': None,
        'total_inquires_3': None,
        'total_inquires_4': None,
        'total_inquires_5': None,
        'total_inquires_6': None,
        
        # Financial - Loans
        'loan_amt_1': None,
        'loan_amt_2': None,
        'loan_amt_3': None,
        'loan_amt_4': None,
        'loan_amt_5': None,
        'loan_amt_6': None,
        'loan_amt_7': None,
        'loan_amt_8': None,
        'loan_amt_9': None,
        'loan_amt_recent': None,
        'loan_amt_large_tenure': None,
        'primary_loan_amt': None,
        
        # Financial - Repayments
        'repayment_1': None,
        'repayment_2': None,
        'repayment_3': None,
        'repayment_4': None,
        'repayment_5': None,
        'repayment_6': None,
        'repayment_7': None,
        'repayment_8': None,
        'repayment_9': None,
        'repayment_10': None,
        
        # Financial - Loans Count
        'total_loans_1': None,
        'total_loans_2': None,
        'total_loans_2_dup1': None,
        'total_loans_3': None,
        'total_loans_4': None,
        'total_loans_5': None,
        'closed_loan': None,
        'closed_total_loans': None,
        
        # Other
        'credit_score': None,
        'pin_code_frequency': None
    }
    
    # Add missing value indicators for all base features
    missing_indicators = {f"{col}_is_missing": 0 for col in base_features.keys()}
    
    # Add derived features
    derived_features = {
        'total_credit_limit_overall': None,
        'avg_credit_limit_overall': None,
        'total_balance_overall': None,
        'avg_balance_overall': None,
        'total_emi_overall': None,
        'avg_emi_overall': None,
        'total_inquiries_overall': None,
        'avg_inquiries_overall': None,
        'total_loan_amount_overall': None,
        'avg_loan_amount_overall': None,
        'overall_credit_util_ratio': None,
        'credit_util_ratio_1': None,
        'credit_util_ratio_2': None,
        'credit_util_ratio_3': None,
        'debt_to_income_ratio': None,
        'income_to_emi_ratio': None,
        'income_to_loan_ratio': None,
        'payment_to_income_ratio': None,
        'emi_per_inquiry_ratio': None,
        'loan_amount_per_inquiry_ratio': None,
        'repayment_consistency_score': None,
        'avg_repayment': None,
        'age_x_total_inquiries_overall': None,
        'total_emi_overall_x_credit_score': None,
        'age_bin': None
    }
    
    # Combine all features
    all_features = {**base_features, **missing_indicators, **derived_features}
    return all_features

def calculate_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate derived features based on base features."""
    # Credit limits
    credit_limit_cols = [col for col in df.columns if col.startswith('credit_limit_')]
    if credit_limit_cols:
        df['total_credit_limit_overall'] = df[credit_limit_cols].sum(axis=1)
        df['avg_credit_limit_overall'] = df[credit_limit_cols].mean(axis=1)
    
    # Balances
    balance_cols = [col for col in df.columns if col.startswith('balance_')]
    if balance_cols:
        df['total_balance_overall'] = df[balance_cols].sum(axis=1)
        df['avg_balance_overall'] = df[balance_cols].mean(axis=1)
    
    # EMIs
    emi_cols = [col for col in df.columns if col.startswith('total_emi_')]
    if emi_cols:
        df['total_emi_overall'] = df[emi_cols].sum(axis=1)
        df['avg_emi_overall'] = df[emi_cols].mean(axis=1)
    
    # Inquiries
    inquiry_cols = [col for col in df.columns if col.startswith('total_inquiries_') or col.startswith('total_inquires_')]
    if inquiry_cols:
        df['total_inquiries_overall'] = df[inquiry_cols].sum(axis=1)
        df['avg_inquiries_overall'] = df[inquiry_cols].mean(axis=1)
    
    # Loans
    loan_cols = [col for col in df.columns if col.startswith('loan_amt_')]
    if loan_cols:
        df['total_loan_amount_overall'] = df[loan_cols].sum(axis=1)
        df['avg_loan_amount_overall'] = df[loan_cols].mean(axis=1)
    
    # Credit utilization ratios
    if 'total_balance_overall' in df.columns and 'total_credit_limit_overall' in df.columns:
        df['overall_credit_util_ratio'] = df['total_balance_overall'] / (df['total_credit_limit_overall'] + 1e-6)
    
    # Individual credit utilization ratios
    for i in range(1, 4):
        bal_col = f'balance_{i}'
        lim_col = f'credit_limit_{i}'
        if bal_col in df.columns and lim_col in df.columns:
            df[f'credit_util_ratio_{i}'] = df[bal_col] / (df[lim_col] + 1e-6)
    
    # Age binning
    if 'age' in df.columns:
        df['age_bin'] = pd.cut(df['age'],
                              bins=[0, 25, 35, 45, 55, np.inf],
                              labels=['<25', '25-34', '35-44', '45-54', '55+'],
                              right=False)
    
    # Interaction features
    if 'age' in df.columns and 'total_inquiries_overall' in df.columns:
        df['age_x_total_inquiries_overall'] = df['age'] * df['total_inquiries_overall']
    
    if 'total_emi_overall' in df.columns and 'credit_score' in df.columns:
        df['total_emi_overall_x_credit_score'] = df['total_emi_overall'] * df['credit_score']
    
    return df

def validate_and_prepare_input(input_data: Dict[str, Any], preprocessor=None) -> pd.DataFrame:
    """Validate and prepare input data for prediction, ensuring all expected columns are present and types match the preprocessor."""
    # Get all expected features
    all_features = get_all_expected_features()
    # Update with provided values
    for key, value in input_data.items():
        if key in all_features:
            all_features[key] = value
    # Validate required fields
    required_fields = ['age', 'gender', 'marital_status', 'city', 'state', 'residence_ownership']
    missing_required = [k for k in required_fields if all_features[k] is None]
    if missing_required:
        raise ValueError(f"Missing required fields: {', '.join(missing_required)}")
    # Convert to DataFrame
    df = pd.DataFrame([all_features])
    # Add missing value indicators
    for col in df.columns:
        if not col.endswith('_is_missing'):
            missing_col = f"{col}_is_missing"
            if missing_col in df.columns:
                df[missing_col] = df[col].isna().astype(int)
    # Calculate derived features
    df = calculate_derived_features(df)
    # If preprocessor is provided, ensure all columns match its expected input
    if preprocessor is not None:
        try:
            expected_cols = list(preprocessor.feature_names_in_)
        except AttributeError:
            expected_cols = None
        if expected_cols:
            for col in expected_cols:
                if col not in df.columns:
                    df[col] = np.nan
            df = df[expected_cols]
        # Detect numeric and categorical columns from the preprocessor's transformers_
        numeric_cols = []
        categorical_cols = []
        if hasattr(preprocessor, 'transformers_'):
            for name, trans, cols in preprocessor.transformers_:
                if name == 'num':
                    numeric_cols.extend(cols)
                elif name == 'cat':
                    categorical_cols.extend(cols)
        # Cast columns to correct types
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        for col in categorical_cols:
            if col in df.columns:
                df[col] = df[col].astype(str)
    else:
        # Fallback: guess types
        numeric_cols = [col for col in df.columns if not col.endswith('_is_missing') and col not in ['gender', 'marital_status', 'city', 'state', 'residence_ownership', 'device_category', 'device_manufacturer', 'device_model', 'platform', 'score_type', 'score_comments', 'age_bin']]
        categorical_cols = ['gender', 'marital_status', 'city', 'state', 'residence_ownership', 'device_category', 'device_manufacturer', 'device_model', 'platform', 'score_type', 'score_comments', 'age_bin']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        for col in categorical_cols:
            if col in df.columns:
                df[col] = df[col].astype(str)
    return df

def predict_income(input_data: Dict[str, Any], model, preprocessor) -> Dict[str, Any]:
    """Make predictions using the model and preprocessor."""
    try:
        # Validate and prepare input
        df = validate_and_prepare_input(input_data, preprocessor=preprocessor)
        # Count available features
        available_features = sum(1 for v in df.iloc[0] if pd.notna(v))
        total_features = len(df.columns)
        feature_completeness = (available_features / total_features) * 100
        # Preprocess the input data
        X_processed = preprocessor.transform(df)
        # Make prediction (this will be in log scale)
        y_pred_log = model.predict(X_processed)
        # Convert back to original scale
        y_pred_original = np.expm1(y_pred_log)
        # Get individual model predictions if available (for confidence)
        if hasattr(model, 'estimators_'):
            individual_predictions = []
            for estimator in model.estimators_:
                if hasattr(estimator, 'predict'):
                    pred = np.expm1(estimator.predict(X_processed))
                    individual_predictions.append(pred[0])
            if individual_predictions:
                pred_std = np.std(individual_predictions)
                pred_mean = np.mean(individual_predictions)
                confidence_score = 1 - (pred_std / pred_mean) if pred_mean != 0 else 0
                confidence_score = max(0, min(1, confidence_score))
            else:
                confidence_score = None
        else:
            confidence_score = None
        return {
            'predicted_annual_income': float(y_pred_original[0]),
            'predicted_monthly_income': float(y_pred_original[0] / 12),
            'feature_completeness': float(feature_completeness),
            'confidence_score': float(confidence_score) if confidence_score is not None else None,
            'available_features': int(available_features),
            'total_features': int(total_features)
        }
    except Exception as e:
        raise Exception(f"Error making prediction: {str(e)}")

def main():
    # Example 1: Complete data
    print("Example 1: Complete data")
    complete_data = {
        'age': 35,
        'gender': 'M',
        'marital_status': 'Married',
        'city': 'Mumbai',
        'state': 'Maharashtra',
        'residence_ownership': 'Owned',
        'credit_score': 750,
        'credit_limit_1': 50000,
        'balance_1': 25000,
        'total_emi_1': 5000,
        'total_inquiries_1': 2,
        'loan_amt_1': 200000,
        'repayment_1': 4500,
        'device_category': 'Smartphone',
        'device_manufacturer': 'Samsung',
        'device_model': 'Galaxy S21',
        'platform': 'Android',
        'score_type': 'FICO',
        'score_comments': 'Good',
        # Simulated/external features
        'digital_literacy_score': 0.85,
        'ecommerce_purchase_frequency': 10,
        'digital_service_subscriptions': 3,
        'upi_transaction_frequency': 25,
        'total_loan_recent_is_missing': 0,
        'market_density_score': 0.8,
        'app_diversity_count': 25,
        'mobile_recharge_frequency': 5,
        'bill_payment_consistency': 0.9,
        'night_light_intensity': 75.5,
        'local_literacy_rate': 85.0,
        'total_loan_recent': 0
    }
    
    # Example 2: Minimal required data
    print("\nExample 2: Minimal required data")
    minimal_data = {
        'age': 28,
        'gender': 'F',
        'marital_status': 'Single',
        'city': 'Delhi',
        'state': 'Delhi',
        'residence_ownership': 'Rented'
    }
    
    try:
        # Load model and preprocessor
        print("Loading model and preprocessor...")
        model, preprocessor = load_model_and_preprocessor()
        
        # Test complete data
        print("\nTesting with complete data:")
        print("Input data:", json.dumps(complete_data, indent=2))
        result_complete = predict_income(complete_data, model, preprocessor)
        print("\nPrediction results:")
        print(json.dumps(result_complete, indent=2))
        
        # Test minimal data
        print("\nTesting with minimal required data:")
        print("Input data:", json.dumps(minimal_data, indent=2))
        result_minimal = predict_income(minimal_data, model, preprocessor)
        print("\nPrediction results:")
        print(json.dumps(result_minimal, indent=2))
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main() 