import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import pickle

data_path = 'data/Hackathon_bureau_data_400.csv'
data = pd.read_csv(data_path)

# Define features
feature_cols = [col for col in data.columns if col.startswith('var_')]
numeric_feature_cols = [
    col for col in feature_cols
    if np.issubdtype(data[col].dtype, np.number) or pd.to_numeric(data[col], errors='coerce').notnull().any()
]
demographic_features = ['age', 'gender', 'marital_status', 'residence_ownership']
categorical_features = ['gender', 'marital_status', 'residence_ownership']
features = numeric_feature_cols + demographic_features

# Clean data
X = data[features]
y = data['target_income']
mask = ~(X.isna().any(axis=1) | y.isna())
X = X[mask]
y = y[mask]

# Scale target variable to 0-1 range
income_scaler = MinMaxScaler()
y_scaled = income_scaler.fit_transform(y.values.reshape(-1, 1)).ravel()

# Preprocessing
numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_feature_cols),
        ('cat', categorical_transformer, categorical_features)
    ])

# Model pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', xgb.XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        min_child_weight=2,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1,
        random_state=42
    ))
])

print("\nTraining model on the entire dataset...")
model.fit(X, y_scaled)
y_pred_scaled = model.predict(X)
mse = mean_squared_error(y_scaled, y_pred_scaled)
r2 = r2_score(y_scaled, y_pred_scaled)
print(f"Training MSE: {mse:.4f}")
print(f"Training R2: {r2:.3f}")

# Save the model and scaler
with open('repayment_capability_model.pkl', 'wb') as f:
    pickle.dump((model, income_scaler), f)
print("\nModel and scaler saved as 'repayment_capability_model.pkl'")

def predict_repayment_capability(features_dict, return_actual_income=False):
    """
    Predict repayment capability for a new customer on a 0-1 scale
    Parameters:
    features_dict (dict): Dictionary containing feature values for the customer
    return_actual_income (bool): If True, returns actual income instead of scaled score
    Returns:
    tuple: (repayment_capability_score, confidence_score)
        repayment_capability_score: Score between 0 and 1 (or actual income if return_actual_income=True)
        confidence_score: Model confidence in the prediction (0-1)
    """
    input_data = pd.DataFrame([{col: features_dict.get(col, 0) for col in features}])
    predicted_scaled = model.predict(input_data)[0]
    feature_importance = model.named_steps['regressor'].feature_importances_
    confidence_score = np.mean(feature_importance)
    
    if return_actual_income:
        predicted_income = income_scaler.inverse_transform([[predicted_scaled]])[0][0]
        return float(predicted_income), float(confidence_score)
    else:
        return float(predicted_scaled), float(confidence_score)

print("\nExample predictions (0-1 scale):")
example1 = {col: 0 for col in features}
example1.update({
    'var_1': 100, 'var_2': 0.8, 'var_3': 1.0, 'var_4': 500, 'var_5': 0.5,
    'age': 35, 'gender': 'M', 'marital_status': 'Married', 'residence_ownership': 'Own'
})
example2 = {col: 0 for col in features}
example2.update({
    'var_1': 50, 'var_2': 0.3, 'var_3': 0.5, 'var_4': 200, 'var_5': 0.2,
    'age': 25, 'gender': 'F', 'marital_status': 'Single', 'residence_ownership': 'Rent'
})

for i, example in enumerate([example1, example2], 1):
    score, confidence = predict_repayment_capability(example)
    actual_income, _ = predict_repayment_capability(example, return_actual_income=True)
    print(f"\nExample {i}:")
    print(f"Repayment Capability Score (0-1): {score:.3f}")
    print(f"Actual Predicted Income: ₹{actual_income:,.2f}")
    print(f"Confidence Score: {confidence:.3f}")
    print(f"Demographic Info: Age={example['age']}, Gender={example['gender']}, "
          f"Marital Status={example['marital_status']}, Residence={example['residence_ownership']}") 