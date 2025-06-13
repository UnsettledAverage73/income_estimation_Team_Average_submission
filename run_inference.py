import json
import os
import pandas as pd
import joblib
from dotenv import load_dotenv

def load_config():
    """Load configuration from config.json"""
    with open('config/config.json', 'r') as f:
        return json.load(f)

def load_model(config):
    """Load the trained model"""
    return joblib.load(config['training']['save_model_path'])

def preprocess_input(df):
    """Preprocess input data for inference"""
    # TODO: Add your preprocessing steps here
    # This should match the preprocessing done in main.py
    return df

def run_inference(input_file, output_file):
    """Run inference on input data and save predictions"""
    # Load configuration and model
    config = load_config()
    model = load_model(config)
    
    # Load and preprocess input data
    df = pd.read_csv(input_file)
    X = preprocess_input(df)
    
    # Make predictions
    predictions = model.predict(X)
    
    # Save predictions
    output_df = pd.DataFrame({
        'id': df['id'] if 'id' in df.columns else range(len(predictions)),
        'predicted_income': predictions
    })
    output_df.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")

def main():
    # Load environment variables
    load_dotenv()
    
    # Load configuration
    config = load_config()
    
    # Get input and output file paths from command line arguments or config
    input_file = config['data']['input_file']
    output_file = config['data']['output_file']
    
    # Run inference
    run_inference(input_file, output_file)

if __name__ == "__main__":
    main() 