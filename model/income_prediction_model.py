import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import re

def load_data_in_batches(file_path, batch_size=100):
    """
    Load data in batches of specified size
    """
    # Read the first batch to get column names
    first_batch = pd.read_csv(file_path, nrows=batch_size)
    total_rows = sum(1 for _ in open(file_path)) - 1  # subtract header
    
    for start_row in range(0, total_rows, batch_size):
        print(f"Processing batch starting at row {start_row}")
        batch = pd.read_csv(file_path, skiprows=range(1, start_row + 1), nrows=batch_size)
        if batch.empty:
            break
        yield batch

# Load the first batch to get feature columns
data_path = 'data/Hackathon_bureau_data_400.csv'  # Updated path to the correct data file
first_batch_df = next(load_data_in_batches(data_path, batch_size=100))
feature_cols = [col for col in first_batch_df.columns if col.startswith('var_')]
# Keep only numeric columns (dtype or at least one value convertible)
numeric_feature_cols = [
    col for col in feature_cols
    if np.issubdtype(first_batch_df[col].dtype, np.number) or pd.to_numeric(first_batch_df[col], errors='coerce').notnull().any()
]
target = 'target_income'
features = numeric_feature_cols

# Define preprocessing steps
numeric_features = features  # All selected var_* columns are numeric
categorical_features = []    # No categorical features in var_* columns

# Create preprocessing pipelines
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Create the XGBoost model pipeline
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

# Initialize lists to store metrics
all_mse = []
all_rmse = []
all_mae = []
all_r2 = []
all_predictions = []
all_actuals = []

# Process data in batches
print("Starting batch processing...")
first_batch = True

for batch in load_data_in_batches(data_path, batch_size=100):
    # Separate features and target
    X_batch = batch[features]
    y_batch = batch[target]
    
    # Split the batch
    X_train, X_test, y_train, y_test = train_test_split(X_batch, y_batch, test_size=0.2, random_state=42)
    
    if first_batch:
        # Train the model on first batch
        print("Training model on first batch...")
        model.fit(X_train, y_train)
        first_batch = False
    else:
        # Update the model with subsequent batches
        print("Updating model with new batch...")
        model.fit(X_train, y_train, regressor__xgb_model=model.named_steps['regressor'].get_booster())
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Store predictions and actuals for later analysis
    all_predictions.extend(y_pred)
    all_actuals.extend(y_test)
    
    # Calculate metrics for this batch
    batch_mse = mean_squared_error(y_test, y_pred)
    batch_rmse = np.sqrt(batch_mse)
    batch_mae = mean_absolute_error(y_test, y_pred)
    batch_r2 = r2_score(y_test, y_pred)
    
    all_mse.append(batch_mse)
    all_rmse.append(batch_rmse)
    all_mae.append(batch_mae)
    all_r2.append(batch_r2)
    
    print(f"Batch metrics - MSE: {batch_mse:,.2f}, RMSE: {batch_rmse:,.2f}, MAE: {batch_mae:,.2f}, R2: {batch_r2:.3f}")

# Calculate and print overall metrics
print("\nOverall Model Performance Metrics:")
print(f"Mean Squared Error: {np.mean(all_mse):,.2f}")
print(f"Root Mean Squared Error: {np.mean(all_rmse):,.2f}")
print(f"Mean Absolute Error: {np.mean(all_mae):,.2f}")
print(f"R-squared Score: {np.mean(all_r2):.3f}")

# Get feature names after preprocessing
if categorical_features:
    feature_names = (
        numeric_features +
        [f"{col}_{val}" for col, vals in zip(
            categorical_features,
            model.named_steps['preprocessor']
            .named_transformers_['cat']
            .named_steps['onehot']
            .categories_
        ) for val in vals]
    )
else:
    feature_names = numeric_features

# Get feature importance
feature_importance = model.named_steps['regressor'].feature_importances_

# Create feature importance DataFrame
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importance
})
importance_df = importance_df.sort_values('Importance', ascending=False)

# Plot feature importance
plt.figure(figsize=(12, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df.head(15))
plt.title('Top 15 Most Important Features for Income Prediction')
plt.tight_layout()
plt.savefig('income_prediction_feature_importance.png')
plt.close()

# Save the model
with open('income_prediction_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("\nModel saved as 'income_prediction_model.pkl'")
print("Feature importance plot saved as 'income_prediction_feature_importance.png'")

# Plot actual vs predicted values using all data
plt.figure(figsize=(10, 6))
plt.scatter(all_actuals, all_predictions, alpha=0.5)
plt.plot([min(all_actuals), max(all_actuals)], [min(all_actuals), max(all_actuals)], 'r--', lw=2)
plt.xlabel('Actual Income')
plt.ylabel('Predicted Income')
plt.title('Actual vs Predicted Income')
plt.tight_layout()
plt.savefig('income_prediction_actual_vs_predicted.png')
plt.close()

# Example prediction function
def predict_income(features_dict):
    """
    Make income prediction for a new customer
    """
    input_data = pd.DataFrame([{col: features_dict.get(col, 0) for col in features}])
    prediction = model.predict(input_data)[0]
    return round(prediction, 2)

# Example usage
print("\nExample predictions:")
example1 = {col: 0 for col in features}
example1.update({'var_1': 100, 'var_2': 0.8, 'var_3': 1.0, 'var_4': 500, 'var_5': 0.5})
example2 = {col: 0 for col in features}
example2.update({'var_1': 50, 'var_2': 0.3, 'var_3': 0.5, 'var_4': 200, 'var_5': 0.2})
examples = [example1, example2]
for i, example in enumerate(examples, 1):
    prediction = predict_income(example)
    print(f"Example {i} prediction: {prediction:.2f}")

# Cross-validation score
cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
print(f"\nCross-validation R2 scores: {cv_scores}")
print(f"Average CV R2 score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})") 