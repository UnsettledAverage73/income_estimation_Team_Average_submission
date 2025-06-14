import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import logging
import hashlib
from typing import Dict, Any
from datetime import datetime
import pytz
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.multioutput import MultiOutputEstimator

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='imputation_training_log.log'
)
logger = logging.getLogger(__name__)

# Define imputation timestamp (09:11 PM IST, June 14, 2025)
IMPUTATION_TIMESTAMP = datetime(2025, 6, 14, 21, 11, tzinfo=pytz.timezone('Asia/Kolkata')).astimezone(pytz.UTC)

# Mock PIN-to-city/state mapping
PIN_MAPPING: Dict[str, Dict[str, str]] = {
    '400001': {'city': 'Mumbai', 'state': 'Maharashtra'},
    '560100': {'city': 'Bangalore', 'state': 'Karnataka'},
    '110091': {'city': 'Delhi', 'state': 'Delhi'},
    '700001': {'city': 'Kolkata', 'state': 'West Bengal'},
    '600001': {'city': 'Chennai', 'state': 'Tamil Nadu'},
}

def anonymize_pin(pin: str) -> str:
    """Anonymize PIN code for DPDP compliance."""
    return hashlib.sha256(str(pin).encode()).hexdigest()

def impute_rule_based(df: pd.DataFrame, numeric_features: list, categorical_features: list, date_features: list) -> pd.DataFrame:
    """Impute missing values using rule-based logic from Feature-by-Feature Strategy, preserving non-missing values."""
    try:
        logger.info("Starting rule-based imputation...")
        
        # Numeric features
        for col in numeric_features:
            if col in df.columns and df[col].isnull().any():
                logger.info(f"Imputing {col}...")
                # Coerce to numeric, replace non-numeric with NaN
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
                if col == 'age':
                    if 'target_income' in df.columns:
                        mean_age = df['age'].mean()
                        df[col] = df[col].where(df[col].notna(), np.where(
                            df['target_income'] > 50000, mean_age, 30
                        ))
                    else:
                        df[col] = df[col].fillna(30)
                
                elif col == 'var_0':  # balance_1
                    if 'target_income' in df.columns:
                        df[col] = df[col].where(df[col].notna(), df['target_income'] * 0.125)
                    else:
                        df[col] = df[col].fillna(df[col].median())
                
                elif col == 'var_1':  # balance_2
                    if 'var_0' in df.columns:
                        df[col] = df[col].where(df[col].notna(), df['var_0'] * 0.4)
                    else:
                        df[col] = df[col].fillna(0)
                
                elif col == 'var_4':  # balance_3
                    if 'target_income' in df.columns:
                        df[col] = df[col].where(df[col].notna(), np.where(
                            df['target_income'] > df['target_income'].quantile(0.75), df[col].median(), 0
                        ))
                    else:
                        df[col] = df[col].fillna(0)
                
                elif col == 'var_8':  # business_balance
                    if 'target_income' in df.columns:
                        df[col] = df[col].where(df[col].notna(), np.where(
                            df['target_income'] > 60000, df['target_income'] * 2.5, 0
                        ))
                    else:
                        df[col] = df[col].fillna(0)
                
                elif col in ['var_18', 'var_22', 'var_27', 'var_36', 'var_49', 'var_54', 'var_60', 'var_68']:  # balance_4 to balance_12
                    df[col] = df[col].where(df[col].notna(), 0)
                
                elif col == 'var_2':  # credit_limit_1
                    if 'target_income' in df.columns:
                        df[col] = df[col].where(df[col].notna(), df['target_income'] * 4)
                    else:
                        df[col] = df[col].fillna(df[col].median())
                
                elif col == 'var_3':  # credit_limit_2
                    if 'var_2' in df.columns:
                        df[col] = df[col].where(df[col].notna(), df['var_2'] * 0.5)
                    else:
                        df[col] = df[col].fillna(0)
                
                elif col in ['var_5', 'var_10', 'var_14', 'var_19', 'var_25', 'var_30', 'var_38', 'var_43', 'var_47', 'var_55', 'var_62']:  # credit_limit_3 to credit_limit_13
                    df[col] = df[col].where(df[col].notna(), 0)
                
                elif col == 'var_6':  # loan_amt_1
                    if 'target_income' in df.columns:
                        df[col] = df[col].where(df[col].notna(), df['target_income'] * 2.5 * 12)
                    else:
                        df[col] = df[col].fillna(0)
                
                elif col == 'var_7':  # loan_amt_2
                    if 'var_6' in df.columns:
                        df[col] = df[col].where(df[col].notna(), df['var_6'] * 0.4)
                    else:
                        df[col] = df[col].fillna(0)
                
                elif col in ['var_20', 'var_24', 'var_31', 'var_41', 'var_50', 'var_61', 'var_72']:  # loan_amt_3 to loan_amt_9
                    df[col] = df[col].where(df[col].notna(), 0)
                
                elif col == 'var_9':  # total_emi_1
                    if 'target_income' in df.columns and 'var_6' in df.columns:
                        df[col] = df[col].where(df[col].notna(), df['var_6'] * 0.008)
                    else:
                        df[col] = df[col].fillna(0)
                
                elif col in ['var_17', 'var_26', 'var_34', 'var_42', 'var_51', 'var_56']:  # total_emi_2 to total_emi_7
                    df[col] = df[col].where(df[col].notna(), 0)
                
                elif col == 'var_15':  # total_inquiries_1
                    if 'age' in df.columns:
                        df[col] = df[col].where(df[col].notna(), np.where(
                            df['age'] < 35, 3, 1
                        ))
                    else:
                        df[col] = df[col].fillna(1)
                
                elif col in ['var_16', 'var_21', 'var_28', 'var_35', 'var_71']:  # total_inquiries_2 to total_inquiries_6
                    df[col] = df[col].where(df[col].notna(), 0)
                
                elif col == 'var_37':  # repayment_1
                    if 'var_32' in df.columns:
                        df[col] = df[col].where(df[col].notna(), np.where(
                            df['var_32'] > 700, 1, 0
                        ))
                    else:
                        df[col] = df[col].fillna(0)
                
                elif col in ['var_44', 'var_48', 'var_52', 'var_58', 'var_59', 'var_63', 'var_64', 'var_69', 'var_73']:  # repayment_2 to repayment_10
                    df[col] = df[col].where(df[col].notna(), df.get('var_37', pd.Series(0)))
                
                elif col == 'var_53':  # total_loans_1
                    loan_cols = ['var_6', 'var_7', 'var_20', 'var_24', 'var_31', 'var_41', 'var_50', 'var_61', 'var_72']
                    if any(c in df.columns for c in loan_cols):
                        df[col] = df[col].where(df[col].notna(), sum((df.get(c, pd.Series(0)) > 0).astype(int) for c in loan_cols if c in df.columns))
                    else:
                        df[col] = df[col].fillna(0)
                
                elif col in ['var_57', 'var_65', 'var_66', 'var_70']:  # total_loans_2 to total_loans_5
                    df[col] = df[col].where(df[col].notna(), df.get('var_53', pd.Series(0)))
                
                elif col == 'var_32':  # credit_score
                    if 'target_income' in df.columns and 'age' in df.columns:
                        df[col] = df[col].where(df[col].notna(), np.where(
                            (df['target_income'] > 50000) & (df['age'] > 30), 750, 675
                        ))
                    else:
                        df[col] = df[col].fillna(675)
                
                elif col == 'var_40':  # closed_loan
                    if 'var_53' in df.columns and 'age' in df.columns:
                        df[col] = df[col].where(df[col].notna(), np.where(
                            df['age'] > 40, df['var_53'] * 0.5, 0
                        ))
                    else:
                        df[col] = df[col].fillna(0)
                
                else:
                    df[col] = df[col].fillna(df[col].median())
                
                logger.info(f"Imputed {col}. Missing values: {df[col].isna().sum()}")

        # Categorical features
        for col in categorical_features:
            if col in df.columns and df[col].isnull().any():
                logger.info(f"Imputing {col}...")
                if col == 'marital_status':
                    if 'age' in df.columns:
                        df[col] = df[col].where(df[col].notna(), np.where(
                            df['age'] > 28, 'Married', 'Single'
                        ))
                    else:
                        df[col] = df[col].fillna('Single')
                
                elif col == 'gender':
                    if 'target_income' in df.columns:
                        mode_gender = df['gender'].mode()[0] if not df['gender'].mode().empty else 'Unknown'
                        df[col] = df[col].where(df[col].notna(), np.where(
                            df['target_income'] > 40000, 'Male', mode_gender
                        ))
                    else:
                        df[col] = df[col].fillna(df['gender'].mode()[0] if not df['gender'].mode().empty else 'Unknown')
                
                elif col == 'residence_ownership':
                    if 'target_income' in df.columns and 'age' in df.columns:
                        df[col] = df[col].where(df[col].notna(), np.where(
                            (df['target_income'] > 60000) & (df['age'] > 35), 'SELF-OWNED', 'RENTED'
                        ))
                    else:
                        df[col] = df[col].fillna('RENTED')
                
                elif col == 'city':
                    if 'pin' in df.columns:
                        def impute_city(row):
                            if pd.isna(row['city']):
                                pin = str(row['pin'])
                                return PIN_MAPPING.get(pin, {'city': df['city'].mode()[0] if not df['city'].mode().empty else 'Unknown'})['city']
                            return row['city']
                        df['city'] = df.apply(impute_city, axis=1)
                    else:
                        df[col] = df[col].fillna(df['city'].mode()[0] if not df['city'].mode().empty else 'Unknown')
                
                elif col == 'state':
                    if 'pin' in df.columns:
                        def impute_state(row):
                            if pd.isna(row['state']):
                                pin = str(row['pin'])
                                return PIN_MAPPING.get(pin, {'state': df['state'].mode()[0] if not df['state'].mode().empty else 'Unknown'})['state']
                            return row['state']
                        df['state'] = df.apply(impute_state, axis=1)
                    else:
                        df[col] = df[col].fillna(df['state'].mode()[0] if not df['state'].mode().empty else 'Unknown')
                
                elif col == 'device_model':
                    if 'target_income' in df.columns:
                        df[col] = df[col].where(df[col].notna(), np.where(
                            df['target_income'] > 50000, 'iPhone', 'Generic Android'
                        ))
                    else:
                        df[col] = df[col].fillna('Generic Android')
                
                elif col == 'device_category':
                    if 'device_model' in df.columns:
                        df[col] = df[col].where(df[col].notna(), np.where(
                            df['device_model'].str.contains('iPhone|Samsung', case=False, na=False), 'SMART PHONE', 'FEATURE PHONE'
                        ))
                    else:
                        df[col] = df[col].fillna('SMART PHONE')
                
                elif col == 'platform':
                    if 'device_model' in df.columns:
                        df[col] = df[col].where(df[col].notna(), np.where(
                            df['device_model'].str.contains('iPhone', case=False, na=False), 'IOS', 'Android'
                        ))
                    else:
                        df[col] = df[col].fillna('Android')
                
                elif col == 'device_manufacturer':
                    if 'device_model' in df.columns:
                        df[col] = df[col].where(df[col].notna(), np.where(
                            df['device_model'].str.contains('iPhone', case=False, na=False), 'Apple', 'Samsung'
                        ))
                    else:
                        df[col] = df[col].fillna('Samsung')
                
                elif col == 'var_74':  # score_comments
                    if 'var_32' in df.columns:
                        df[col] = df[col].where(df[col].notna(), np.where(
                            df['var_32'] > 700, 'Low Risk', 'Medium Risk'
                        ))
                    else:
                        df[col] = df[col].fillna('Medium Risk')
                
                elif col == 'var_75':  # score_type
                    df[col] = df[col].where(df[col].notna(), 'Primary Score')
                
                else:
                    df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown')
                
                logger.info(f"Imputed {col}. Missing values: {df[col].isna().sum()}")

        # Date features
        for col in date_features:
            if col in df.columns and df[col].isnull().any():
                logger.info(f"Imputing {col} with timestamp {IMPUTATION_TIMESTAMP}...")
                df[col] = df[col].where(df[col].notna(), IMPUTATION_TIMESTAMP)
                logger.info(f"Imputed {col}. Missing values: {df[col].isna().sum()}")

        return df
    except Exception as e:
        logger.error(f"Error in rule-based imputation: {str(e)}")
        return df

def plot_missing_data_reduction(initial_missing: pd.Series, final_missing: pd.Series, output_path: str):
    """Plot missing data reduction."""
    plt.figure(figsize=(12, 6))
    missing_df = pd.DataFrame({
        'Feature': initial_missing.index,
        'Before (%)': initial_missing * 100,
        'After (%)': final_missing * 100
    }).melt(id_vars='Feature', var_name='Stage', value_name='Missing Percentage')
    
    before_missing = missing_df[missing_df['Stage'] == 'Before (%)']
    features_to_keep = before_missing[before_missing['Missing Percentage'] > 0]['Feature']
    filtered_df = missing_df[missing_df['Feature'].isin(features_to_keep)]
    
    sns.barplot(x='Feature', y='Missing Percentage', hue='Stage', data=filtered_df)
    plt.title('Missing Data Reduction After Imputation')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def predict_repayment_capability(features_dict: Dict[str, Any], model: Pipeline, income_scaler: MinMaxScaler, features: list, numeric_features: list, categorical_features: list, return_actual_income: bool = False) -> tuple:
    """Predict repayment capability with partial input, using defaults for missing features."""
    input_dict = {}
    for col in features:
        if col in features_dict:
            input_dict[col] = features_dict[col]
        else:
            input_dict[col] = 0 if col in numeric_features else 'Unknown'
    input_data = pd.DataFrame([input_dict])
    predicted_scaled = model.predict(input_data)[0]
    feature_importance = model.named_steps['regressor'].feature_importances_
    confidence_score = np.mean(feature_importance)
    
    if return_actual_income:
        predicted_income = income_scaler.inverse_transform([[predicted_scaled]])[0][0]
        return float(predicted_income), float(confidence_score)
    else:
        return float(predicted_scaled), float(confidence_score)

def main():
    # File path
    data_path = 'data/Hackathon_bureau_data_400.csv'
    output_path = data_path.replace('.csv', '_imputed.csv')
    
    # Load data
    try:
        df = pd.read_csv(data_path)
        logger.info(f"Loaded {data_path} with {len(df)} rows")
    except Exception as e:
        logger.error(f"Failed to load data: {str(e)}")
        print(f"Error: Failed to load {data_path}. Check file path and format.")
        return
    
    # Validate required columns
    required_cols = ['target_income']
    missing_required = [col for col in required_cols if col not in df.columns]
    if missing_required:
        logger.error(f"Missing required columns: {missing_required}")
        print(f"Error: Missing required columns {missing_required}")
        return
    
    # Log available columns
    logger.info(f"Available columns: {list(df.columns)}")
    
    # Identify all features with missing data
    initial_missing = df.isnull().mean()
    missing_features = initial_missing[initial_missing > 0].index.tolist()
    logger.info(f"Features with missing data (%):\n{initial_missing[missing_features] * 100}")
    
    # Define all features for imputation
    all_numeric_features = [
        'age', 'var_0', 'var_1', 'var_2', 'var_3', 'var_4', 'var_5', 'var_6', 'var_7', 'var_8', 'var_9',
        'var_10', 'var_14', 'var_15', 'var_16', 'var_17', 'var_18', 'var_19', 'var_20', 'var_21', 'var_22',
        'var_24', 'var_25', 'var_26', 'var_27', 'var_30', 'var_31', 'var_32', 'var_35', 'var_36', 'var_37',
        'var_38', 'var_40', 'var_41', 'var_42', 'var_43', 'var_44', 'var_47', 'var_48', 'var_49', 'var_50',
        'var_51', 'var_52', 'var_53', 'var_54', 'var_55', 'var_56', 'var_57', 'var_58', 'var_59', 'var_60',
        'var_61', 'var_62', 'var_63', 'var_64', 'var_65', 'var_66', 'var_68', 'var_69', 'var_70', 'var_71',
        'var_72', 'var_73'
    ]
    all_categorical_features = [
        'gender', 'city', 'state', 'residence_ownership', 'device_model', 'device_category',
        'platform', 'device_manufacturer', 'marital_status', 'var_74', 'var_75'
    ]
    date_features = ['last_updated', 'transaction_date']
    
    # Impute missing values
    df = impute_rule_based(df, all_numeric_features, all_categorical_features, date_features)
    
    # Add imputation timestamp and anonymize PIN
    df['imputation_timestamp'] = IMPUTATION_TIMESTAMP
    if 'pin' in df.columns:
        df['pin_anonymized'] = df['pin'].apply(anonymize_pin)
    
    # Save to new file to avoid PermissionError
    try:
        df.to_csv(output_path, index=False)
        logger.info(f"Saved imputed data to {output_path}")
    except PermissionError as e:
        logger.error(f"PermissionError saving {output_path}: {str(e)}")
        print(f"Error: Cannot save {output_path}. Close the file if open, or run with admin privileges.")
        return
    
    # Missing data reduction plot
    final_missing = df[missing_features].isnull().mean() if missing_features else pd.Series()
    if not final_missing.empty:
        plot_missing_data_reduction(initial_missing[missing_features], final_missing, 'missing_data_reduction.png')
    
    # Define features for model: Use ALL var_* columns
    feature_cols = [col for col in df.columns if col.startswith('var_')]
    demographic_features = ['age', 'gender', 'marital_status', 'residence_ownership']
    
    # Split var_* into numeric and categorical
    numeric_feature_cols = []
    categorical_feature_cols = []
    for col in feature_cols:
        if col in df.columns:
            try:
                pd.to_numeric(df[col], errors='raise')
                numeric_feature_cols.append(col)
            except (ValueError, TypeError):
                logger.info(f"Column {col} identified as categorical (e.g., '{df[col].dropna().iloc[0]}').")
                categorical_feature_cols.append(col)
    
    # Combine features
    categorical_features = list(set(categorical_feature_cols + ['gender', 'marital_status', 'residence_ownership']))
    numeric_features = numeric_feature_cols + ['age']
    features = numeric_features + categorical_features
    
    logger.info(f"Total features for modeling: {len(features)} ({len(numeric_features)} numeric, {len(categorical_features)} categorical)")
    logger.info(f"Features: {features}")
    
    if not features:
        logger.error("No valid features found for modeling")
        print("Error: No valid features found for modeling")
        return
    
    # Preprocessing pipeline
    numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='mean')), ('scaler', StandardScaler())])
    categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')), ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
    preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, numeric_features), ('cat', categorical_transformer, categorical_features)])

    # Model pipeline
    repayment_estimator = XGBClassifier(objective='binary:logistic', random_state=42, n_estimators=200, learning_rate=0.05, max_depth=6, min_child_weight=2, subsample=0.8, colsample_bytree=0.8, gamma=0.1, reg_alpha=0.1, reg_lambda=1)
    income_estimator = XGBRegressor(objective='reg:squarederror', random_state=42, n_estimators=200, learning_rate=0.05, max_depth=6, min_child_weight=2, subsample=0.8, colsample_bytree=0.8, gamma=0.1, reg_alpha=0.1, reg_lambda=1)
    multi_estimator = MultiOutputEstimator(estimator=[repayment_estimator, income_estimator])

    # Combined pipeline
    combined_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('multi_estimator', multi_estimator)])

    # Prepare X (features) and y (repayment_ok and target_income) for training
    df['repayment_ok'] = (df['target_income'] > 50000).astype(int)
    X = df[features]
    y_repayment = df['repayment_ok']
    y_income = df['target_income']
    y_combined = np.column_stack((y_repayment, y_income))

    # Scale target_income (using MinMaxScaler) so that it is in [0,1] (for regression)
    income_scaler = MinMaxScaler()
    y_income_scaled = income_scaler.fit_transform(y_income.values.reshape(-1, 1)).ravel()
    y_combined_scaled = np.column_stack((y_repayment, y_income_scaled))

    # Split X, y_combined_scaled into train and test (using train_test_split)
    X_train, X_test, y_train_scaled, y_test_scaled = train_test_split(X, y_combined_scaled, test_size=0.2, random_state=42)

    # Train the combined pipeline (using fit) on X_train, y_train_scaled
    print("\nTraining combined (XGBoostClassifier for repayment_ok and XGBoostRegressor for income) model...")
    combined_pipeline.fit(X_train, y_train_scaled)

    # Evaluate (using predict) on X_test and compute metrics (accuracy for repayment_ok, MSE and R² for income)
    y_pred_scaled = combined_pipeline.predict(X_test)
    y_pred_repayment, y_pred_income_scaled = y_pred_scaled[:, 0], y_pred_scaled[:, 1]
    y_test_repayment, y_test_income_scaled = y_test_scaled[:, 0], y_test_scaled[:, 1]

    # Compute repayment (classification) metrics (accuracy)
    repayment_accuracy = (y_pred_repayment == y_test_repayment).mean()
    print(f"\nRepayment (Classifier) Test Accuracy: {repayment_accuracy:.3f}")

    # Compute income (regression) metrics (MSE, R²) – convert scaled predictions back to original scale
    y_test_income = income_scaler.inverse_transform(y_test_income_scaled.reshape(-1, 1)).ravel()
    y_pred_income = income_scaler.inverse_transform(y_pred_income_scaled.reshape(-1, 1)).ravel()
    mse = mean_squared_error(y_test_income, y_pred_income)
    r2 = r2_score(y_test_income, y_pred_income)
    print(f"Income (Regressor) Test MSE: {mse:.4f}, R²: {r2:.3f}")

    # Save the combined pipeline (and scaler) so that later you can predict repayment and income simultaneously
    with open('repayment_capability_model.pkl', 'wb') as f:
        pickle.dump((combined_pipeline, income_scaler), f)
    print("\nCombined (XGBoostClassifier and XGBoostRegressor) model and scaler saved as 'repayment_capability_model.pkl'")

    # Use combined_pipeline for feature importance (if available) and predictions below
    model = combined_pipeline

if __name__ == "__main__":
    main()