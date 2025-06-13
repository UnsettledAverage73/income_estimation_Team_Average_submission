import json
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
from dotenv import load_dotenv

def load_config():
    """Load configuration from config.json"""
    with open('config/config.json', 'r') as f:
        return json.load(f)

def load_data(config):
    """Load and preprocess the data"""
    df = pd.read_csv(config['data']['input_file'])
    return df

def prepare_data(df, config):
    """Prepare data for training"""
    # TODO: Add your data preprocessing steps here
    # This is a placeholder - you'll need to implement proper feature engineering
    X = df.drop(columns=[config['data']['target_column']])
    y = df[config['data']['target_column']]
    
    return train_test_split(
        X, y,
        test_size=config['data']['test_size'],
        random_state=config['data']['random_state']
    )

def train_model(X_train, y_train, config):
    """Train the model"""
    model = RandomForestRegressor(**config['model']['model_params'])
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate the model performance"""
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return {
        'mse': mse,
        'rmse': np.sqrt(mse),
        'r2': r2
    }

def save_model(model, config):
    """Save the trained model"""
    os.makedirs(os.path.dirname(config['training']['save_model_path']), exist_ok=True)
    joblib.dump(model, config['training']['save_model_path'])

def main():
    # Load environment variables
    load_dotenv()
    
    # Load configuration
    config = load_config()
    
    # Load and prepare data
    df = load_data(config)
    X_train, X_test, y_train, y_test = prepare_data(df, config)
    
    # Train model
    model = train_model(X_train, y_train, config)
    
    # Evaluate model
    metrics = evaluate_model(model, X_test, y_test)
    print("Model Performance Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Save model
    save_model(model, config)
    print(f"Model saved to {config['training']['save_model_path']}")

if __name__ == "__main__":
    main() 