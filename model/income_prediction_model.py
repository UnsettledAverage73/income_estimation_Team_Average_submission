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

# Load the dataset
data = pd.read_csv('../Synthetic-Data/data.csv')

# Define features and target
target = 'income'
features = ['occupation', 'upi_txns_month', 'mobile_recharge_freq', 'night_light',
           'education', 'is_urban', 'bill_payment_consistency', 'ecommerce_freq',
           'skill_level', 'household_size', 'market_density', 'public_transport_access',
           'housing_type']

# Separate features and target
X = data[features]
y = data[target]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define preprocessing steps
numeric_features = ['upi_txns_month', 'night_light', 'bill_payment_consistency',
                   'ecommerce_freq', 'household_size', 'market_density',
                   'public_transport_access']
categorical_features = ['occupation', 'mobile_recharge_freq', 'education',
                       'is_urban', 'skill_level', 'housing_type']

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
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        min_child_weight=1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    ))
])

# Train the model
print("Training the model...")
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Performance Metrics:")
print(f"Mean Squared Error: {mse:,.2f}")
print(f"Root Mean Squared Error: {rmse:,.2f}")
print(f"Mean Absolute Error: {mae:,.2f}")
print(f"R-squared Score: {r2:.3f}")

# Get feature names after preprocessing
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

# Cross-validation score
cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
print(f"\nCross-validation R2 scores: {cv_scores}")
print(f"Average CV R2 score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")

# Plot actual vs predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Income')
plt.ylabel('Predicted Income')
plt.title('Actual vs Predicted Income')
plt.tight_layout()
plt.savefig('income_prediction_actual_vs_predicted.png')
plt.close()

# Example prediction function
def predict_income(occupation, upi_txns_month, mobile_recharge_freq, night_light,
                  education, is_urban, bill_payment_consistency, ecommerce_freq,
                  skill_level, household_size, market_density, public_transport_access,
                  housing_type):
    """
    Make income prediction for a new customer
    """
    input_data = pd.DataFrame({
        'occupation': [occupation],
        'upi_txns_month': [upi_txns_month],
        'mobile_recharge_freq': [mobile_recharge_freq],
        'night_light': [night_light],
        'education': [education],
        'is_urban': [is_urban],
        'bill_payment_consistency': [bill_payment_consistency],
        'ecommerce_freq': [ecommerce_freq],
        'skill_level': [skill_level],
        'household_size': [household_size],
        'market_density': [market_density],
        'public_transport_access': [public_transport_access],
        'housing_type': [housing_type]
    })
    
    prediction = model.predict(input_data)[0]
    return round(prediction, 2)

# Example usage
print("\nExample predictions:")
examples = [
    {
        'occupation': 'Small Shop Owner',
        'upi_txns_month': 30,
        'mobile_recharge_freq': 'Weekly',
        'night_light': 70,
        'education': 'Graduate',
        'is_urban': 1,
        'bill_payment_consistency': 1,
        'ecommerce_freq': 8,
        'skill_level': 'Skilled',
        'household_size': 3,
        'market_density': 0.6,
        'public_transport_access': 1,
        'housing_type': 'Pucca'
    },
    {
        'occupation': 'Farmer',
        'upi_txns_month': 5,
        'mobile_recharge_freq': 'Monthly',
        'night_light': 40,
        'education': 'Primary',
        'is_urban': 0,
        'bill_payment_consistency': 0,
        'ecommerce_freq': 2,
        'skill_level': 'Semi-Skilled',
        'household_size': 5,
        'market_density': 0.3,
        'public_transport_access': 1,
        'housing_type': 'Semi-Pucca'
    }
]

for i, example in enumerate(examples, 1):
    prediction = predict_income(**example)
    print(f"\nExample {i}:")
    print(f"Occupation: {example['occupation']}")
    print(f"Education: {example['education']}")
    print(f"Skill Level: {example['skill_level']}")
    print(f"Predicted Income: ₹{prediction:,.2f}") 