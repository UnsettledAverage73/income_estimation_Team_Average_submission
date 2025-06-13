# Income Estimation Model - Team Average

This repository contains the implementation of an income estimation model developed by Team Average for the hackathon.

## Project Structure

```
income_estimation_Team_Average_submission/
├── main.py              # Training script
├── run_inference.py     # Inference script for judging
├── requirements.txt     # Project dependencies
├── config/
│   └── config.json     # Configuration parameters
├── data/
│   └── Hackathon_bureau_data_400.csv
├── output/
│   └── output_Hackathon_bureau_data_400.csv
└── .env                # Environment variables (not committed)
```

## Setup Instructions

1. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Linux/Mac
   # or
   .\venv\Scripts\activate  # On Windows
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the root directory for any required API keys (if applicable)

## Usage

### Training the Model

To train the model:
```bash
python main.py
```

This will:
- Load and preprocess the data
- Train the model
- Save the trained model
- Output performance metrics

### Running Inference

To run inference on new data:
```bash
python run_inference.py
```

This will:
- Load the trained model
- Process the input data
- Generate predictions
- Save the predictions to the output file

## Model Details

- Model Type: Random Forest Regressor
- Features: [To be added after feature engineering]
- Target Variable: Income
- Performance Metrics: MSE, RMSE, R²

## Team Members

[Add team members here]

## License

[Add license information here]