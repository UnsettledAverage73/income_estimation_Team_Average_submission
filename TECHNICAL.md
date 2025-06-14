# Technical Documentation: Income Prediction Model

## System Architecture

### 1. Model Architecture
```
IncomePredictionModel
├── Data Preprocessing Pipeline
│   ├── Column Mapping
│   ├── Feature Engineering
│   └── Data Cleaning
├── Model Pipeline
│   ├── Preprocessor (ColumnTransformer)
│   │   ├── Numerical Features (StandardScaler)
│   │   └── Categorical Features (OneHotEncoder)
│   └── XGBoost Regressor
└── Model Persistence
    └── Pickle Serialization
```

### 2. API Architecture
```
FastAPI Application
├── Input Validation (Pydantic Models)
├── Model Loading & Caching
├── Prediction Endpoint
└── Error Handling & Logging
```

## Technical Implementation Details

### 1. Data Processing Pipeline

#### Column Mapping
- Maps raw column names to human-readable descriptions
- Maintains bidirectional mapping (raw → description and description → raw)
- Implemented in `load_column_mapping()` method
- Uses CSV file for mapping configuration

#### Feature Engineering
Key engineered features:
1. Credit Utilization Metrics
   - Average credit utilization
   - Maximum credit utilization
   - Credit utilization ratio

2. Loan Behavior Features
   - Total loan amount
   - Average loan amount
   - Loan frequency
   - Average loan size

3. Credit Limit Features
   - Total credit limit
   - Average credit limit

4. EMI Features
   - Total EMI
   - Average EMI

5. Repayment Behavior
   - Average repayment
   - Repayment consistency (std)

6. Risk Indicators
   - Credit risk score
   - Debt-to-income ratio
   - Credit utilization ratio

#### Data Cleaning
1. Missing Value Handling:
   - Drop columns with >50% missing values
   - Fill numerical columns with median values
   - Fill categorical columns with mode values
   - Drop rows with missing target values

2. Outlier Handling:
   - Replace infinite values with NaN
   - Log-transform target and key numeric features
   - Handle missing values after transformation

### 2. Model Implementation

#### Preprocessing Pipeline
```python
ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)
```

#### XGBoost Configuration
```python
XGBRegressor(
    objective='reg:squarederror',
    random_state=42,
    n_estimators=200,
    learning_rate=0.1,
    max_depth=5
)
```

#### Hyperparameter Tuning
GridSearchCV parameters:
```python
param_grid = {
    'regressor__max_depth': [3, 5, 7],
    'regressor__learning_rate': [0.01, 0.05, 0.1],
    'regressor__n_estimators': [100, 200],
    'regressor__min_child_weight': [1, 3, 5],
    'regressor__subsample': [0.8, 0.9, 1.0]
}
```

### 3. API Implementation

#### Input Validation
Pydantic models for request/response validation:
```python
class ApplicantData(BaseModel):
    age: int = Field(..., ge=18, le=100)
    gender: str
    marital_status: str
    # ... other fields

class PredictionResponse(BaseModel):
    predicted_income: float
    confidence_score: float
```

#### Model Loading
- Lazy loading on first request
- Global model instance caching
- Error handling for model loading failures

#### Error Handling
- HTTP 500 for model loading errors
- HTTP 422 for validation errors
- Detailed error messages in logs

### 4. SageMaker Integration

#### Model Serving
Required functions for SageMaker deployment:
1. `model_fn`: Loads model from S3
2. `input_fn`: Parses JSON input
3. `predict_fn`: Makes predictions
4. `output_fn`: Formats JSON output

#### Container Requirements
- Python 3.8+
- Dependencies from requirements.txt
- Model artifacts in /opt/ml/model
- Input/output handling for SageMaker endpoints

## Performance Considerations

### 1. Model Performance
- Log-transformed target for better prediction
- Cross-validation for robust evaluation
- Feature importance analysis for model interpretability

### 2. API Performance
- Model caching for faster inference
- Input validation before model inference
- Efficient error handling and logging

### 3. Memory Usage
- Efficient data structures (pandas DataFrame)
- Proper cleanup of temporary objects
- Memory-efficient preprocessing pipeline

## Security Considerations

### 1. Input Validation
- Strict type checking
- Range validation for numeric fields
- Enum validation for categorical fields

### 2. Error Handling
- No sensitive data in error messages
- Proper logging with appropriate log levels
- Graceful failure handling

### 3. API Security
- Input sanitization
- Rate limiting (to be implemented)
- Authentication (to be implemented)

## Future Technical Improvements

### 1. Model Improvements
- Implement confidence score calculation
- Add model versioning
- Implement A/B testing capability
- Add model monitoring

### 2. API Improvements
- Add authentication
- Implement rate limiting
- Add request/response compression
- Implement caching layer

### 3. Deployment Improvements
- Container optimization
- Auto-scaling configuration
- Monitoring and alerting
- CI/CD pipeline

## Development Guidelines

### 1. Code Style
- Follow PEP 8 guidelines
- Use type hints
- Document all functions and classes
- Write unit tests

### 2. Git Workflow
- Feature branches
- Pull request reviews
- Semantic versioning
- Conventional commits

### 3. Testing
- Unit tests for model components
- Integration tests for API
- Load testing for performance
- Security testing

## Dependencies

### Core Dependencies
- numpy>=1.24.0
- pandas>=2.0.0
- scikit-learn>=1.3.0
- xgboost>=2.0.0
- fastapi>=0.100.0
- uvicorn>=0.23.0
- pydantic>=2.0.0

### Development Dependencies
- pytest>=7.4.0
- black>=23.7.0
- flake8>=6.1.0

## Environment Setup

### Local Development
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Run tests
pytest

# Start development server
uvicorn inference:app --reload
```

### Production Deployment
```bash
# Build Docker image
docker build -t income-prediction-api .

# Run container
docker run -p 8000:8000 income-prediction-api
``` 