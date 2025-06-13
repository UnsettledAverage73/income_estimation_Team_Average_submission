import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
import pickle
import json

class BureauLoanRepaymentModel:
    def __init__(self):
        self.model = None
        self.column_mapping = None
        self.reverse_mapping = None
        self.feature_importance = None
        self.preprocessor = None
        
    def load_column_mapping(self, mapping_file='data/participant_col_mapping.csv'):
        """Load and create column mappings"""
        mapping_df = pd.read_csv(mapping_file)
        self.column_mapping = dict(zip(mapping_df['column_name'], mapping_df['description']))
        self.reverse_mapping = {v: k for k, v in self.column_mapping.items()}
        
    def map_columns(self, df, reverse=False):
        """Map column names using the mapping dictionary"""
        mapping = self.reverse_mapping if reverse else self.column_mapping
        return df.rename(columns=mapping)
    
    def load_data(self, data_file='data/Hackathon_bureau_data_400.csv'):
        """Load and preprocess the bureau data"""
        data = pd.read_csv(data_file)
        
        # Map column names to their descriptions
        if self.column_mapping is None:
            self.load_column_mapping()
        data = self.map_columns(data)
        
        # Drop columns that are not useful for prediction, but keep 'target'
        columns_to_drop = [
            'unique_id', 'pin code', 'score_comments', 'score_type', 
            'device_model', 'device_category', 'platform', 'device_manufacturer'
        ]
        columns_to_drop = [col for col in columns_to_drop if col in data.columns and col != 'target']
        data = data.drop(columns=columns_to_drop, errors='ignore')
        
        # Ensure target column exists
        if 'target' not in data.columns:
            raise ValueError("Target column 'target' not found in the data after mapping and dropping columns.")
        
        return data
    
    def engineer_features(self, data):
        """Create new features from the bureau data"""
        # Credit utilization features
        balance_cols = [col for col in data.columns if 'balance' in col]
        data['avg_credit_utilization'] = data[balance_cols].mean(axis=1) if balance_cols else 0
        data['max_credit_utilization'] = data[balance_cols].max(axis=1) if balance_cols else 0
        
        # Loan behavior features
        loan_amt_cols = [col for col in data.columns if 'loan_amt' in col]
        data['total_loan_amount'] = data[loan_amt_cols].sum(axis=1) if loan_amt_cols else 0
        data['avg_loan_amount'] = data[loan_amt_cols].mean(axis=1) if loan_amt_cols else 0
        data['loan_frequency'] = data[loan_amt_cols].count(axis=1) if loan_amt_cols else 0
        
        # Credit limit features
        credit_limit_cols = [col for col in data.columns if 'credit_limit' in col]
        data['total_credit_limit'] = data[credit_limit_cols].sum(axis=1) if credit_limit_cols else 0
        data['avg_credit_limit'] = data[credit_limit_cols].mean(axis=1) if credit_limit_cols else 0
        
        # EMI features
        emi_cols = [col for col in data.columns if 'total_emi' in col]
        data['total_emi'] = data[emi_cols].sum(axis=1) if emi_cols else 0
        data['avg_emi'] = data[emi_cols].mean(axis=1) if emi_cols else 0
        
        # Repayment behavior features
        repayment_cols = [col for col in data.columns if 'repayment' in col]
        data['avg_repayment'] = data[repayment_cols].mean(axis=1) if repayment_cols else 0
        data['repayment_consistency'] = data[repayment_cols].std(axis=1) if repayment_cols else 0
        
        # Credit inquiry features
        inquiries_cols = [col for col in data.columns if 'inquiries' in col]
        data['total_inquiries'] = data[inquiries_cols].sum(axis=1) if inquiries_cols else 0
        # Use correct mapped name for recent inquiries
        recent_inquiries_col = 'total inquiries recent'
        data['recent_inquiries'] = data[recent_inquiries_col] if recent_inquiries_col in data.columns else 0
        
        # Risk indicators
        data['credit_risk_score'] = data['credit_score'] if 'credit_score' in data.columns else 0
        data['debt_to_income'] = data['total_emi'] / data['target'] if 'total_emi' in data.columns and 'target' in data.columns else 0
        data['credit_utilization_ratio'] = data['avg_credit_utilization'] / data['total_credit_limit'] if 'avg_credit_utilization' in data.columns and 'total_credit_limit' in data.columns else 0
        
        return data
    
    def prepare_features(self, data):
        """Prepare features for model training"""
        # Separate numeric and categorical features, but exclude 'target'
        numeric_features = [col for col in data.columns if pd.api.types.is_numeric_dtype(data[col]) and col != 'target']
        categorical_features = [col for col in data.columns if pd.api.types.is_object_dtype(data[col]) and col != 'target']
        
        # Create preprocessing pipeline
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
            ])
        
        # Create model pipeline
        self.model = Pipeline(steps=[
            ('preprocessor', self.preprocessor),
            ('regressor', XGBRegressor(
                objective='reg:squarederror',
                random_state=42,
                n_estimators=200,
                learning_rate=0.1,
                max_depth=5
            ))
        ])
        
        return numeric_features, categorical_features
    
    def train(self, data_file='data/Hackathon_bureau_data_400.csv'):
        """Train the model with improved pipeline"""
        # Load and preprocess data
        data = self.load_data(data_file)
        print(f"Data shape after loading: {data.shape}")
        print("\nTarget variable statistics before cleaning:")
        print(data['target'].describe())
        
        data = self.engineer_features(data)
        print(f"\nData shape after feature engineering: {data.shape}")
        
        # Drop features with >50% missing values
        missing_frac = data.isna().mean()
        cols_to_drop = missing_frac[missing_frac > 0.5].index.tolist()
        print(f"Dropping columns with >50% missing values: {cols_to_drop}")
        data = data.drop(columns=cols_to_drop)
        
        # Check for problematic values before cleaning
        print("\nChecking for problematic values:")
        nan_cols = data.columns[data.isna().any()].tolist()
        numeric_data = data.select_dtypes(include=np.number)
        inf_mask = np.isinf(numeric_data).any()
        inf_cols = numeric_data.columns[inf_mask].tolist()
        print(f"Columns with NaN values: {nan_cols}")
        print(f"Columns with inf values: {inf_cols}")
        
        # More targeted cleaning approach
        data = data.replace([np.inf, -np.inf], np.nan)
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col != 'target':
                median_val = data[col].median()
                data[col] = data[col].fillna(median_val)
        categorical_cols = data.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            mode_val = data[col].mode()[0]
            data[col] = data[col].fillna(mode_val)
        data = data.dropna(subset=['target'])
        print(f"\nData shape after cleaning: {data.shape}")
        print("\nTarget variable statistics after cleaning:")
        print(data['target'].describe())
        
        # Log-transform target and key numeric features
        data['target_log'] = np.log1p(data['target'])
        for col in ['total_loan_amount', 'total_credit_limit', 'total_emi', 'income', 'loan_amt_1', 'loan_amt_2']:
            if col in data.columns:
                data[col + '_log'] = np.log1p(data[col])
        
        # Prepare features (exclude original target)
        features = [col for col in data.columns if col not in ['target', 'target_log']]
        X = data[features]
        y = data['target_log']
        
        # Identify numeric and categorical features
        numeric_features = [col for col in X.columns if pd.api.types.is_numeric_dtype(X[col])]
        categorical_features = [col for col in X.columns if pd.api.types.is_object_dtype(X[col])]
        
        # Preprocessing pipeline
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
            ])
        
        # Model pipeline
        self.model = Pipeline(steps=[
            ('preprocessor', self.preprocessor),
            ('regressor', XGBRegressor(objective='reg:squarederror', random_state=42))
        ])
        
        # Hyperparameter tuning with GridSearchCV
        param_grid = {
            'regressor__max_depth': [3, 5, 7],
            'regressor__learning_rate': [0.01, 0.05, 0.1],
            'regressor__n_estimators': [100, 200]
        }
        grid_search = GridSearchCV(self.model, param_grid, cv=5, scoring='r2', n_jobs=-1, verbose=1)
        grid_search.fit(X, y)
        self.model = grid_search.best_estimator_
        print(f"\nBest parameters: {grid_search.best_params_}")
        
        # Cross-validation score
        from sklearn.model_selection import cross_val_score
        cv_scores = cross_val_score(self.model, X, y, cv=5, scoring='r2')
        print(f"\nCross-validation R2 scores: {cv_scores}")
        print(f"Average CV R2 score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        
        # Evaluate on a hold-out set
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
        y_pred_log = self.model.predict(X_test)
        y_pred = np.expm1(y_pred_log)
        y_test_actual = np.expm1(y_test)
        rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred))
        r2 = r2_score(y_test_actual, y_pred)
        print(f"\nModel Performance on Hold-out Set:")
        print(f"RMSE: {rmse:.2f}")
        print(f"R² Score: {r2:.4f}")
        
        # Store feature importances
        feature_importances = self.model.named_steps['regressor'].feature_importances_
        feature_names = numeric_features + self.model.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(categorical_features).tolist()
        self.feature_importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': feature_importances
        }).sort_values('Importance', ascending=False)
        
        return rmse, r2
    
    def predict(self, input_data):
        """Make predictions for new data"""
        if isinstance(input_data, dict):
            input_data = pd.DataFrame([input_data])
        
        # Map column names if needed
        if self.column_mapping is not None:
            input_data = self.map_columns(input_data)
        
        # Engineer features
        input_data = self.engineer_features(input_data)
        
        # Ensure all columns used in training are present
        model_features = self.model.named_steps['preprocessor'].feature_names_in_
        for col in model_features:
            if col not in input_data.columns:
                input_data[col] = 0
        input_data = input_data[model_features]
        
        # Make prediction
        prediction = self.model.predict(input_data)
        return prediction[0]
    
    def save_model(self, model_path='bureau_loan_repayment_model.pkl'):
        """Save the model and related information"""
        model_info = {
            'model': self.model,
            'column_mapping': self.column_mapping,
            'reverse_mapping': self.reverse_mapping,
            'feature_importance': self.feature_importance,
            'preprocessor': self.preprocessor
        }
        with open(model_path, 'wb') as f:
            pickle.dump(model_info, f)
    
    def load_model(self, model_path='bureau_loan_repayment_model.pkl'):
        """Load a saved model"""
        with open(model_path, 'rb') as f:
            model_info = pickle.load(f)
        
        self.model = model_info['model']
        self.column_mapping = model_info['column_mapping']
        self.reverse_mapping = model_info['reverse_mapping']
        self.feature_importance = model_info['feature_importance']
        self.preprocessor = model_info['preprocessor']
    
    def plot_feature_importance(self, top_n=20):
        """Plot feature importance"""
        if self.feature_importance is None:
            print("Model needs to be trained first")
            return
        
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Importance', y='Feature', 
                   data=self.feature_importance.head(top_n))
        plt.title(f'Top {top_n} Feature Importances')
        plt.tight_layout()
        plt.show()

# Example usage
if __name__ == "__main__":
    # Initialize and train model
    model = BureauLoanRepaymentModel()
    model.load_column_mapping()
    rmse, r2 = model.train()
    
    # Plot feature importance
    model.plot_feature_importance()
    
    # Save model
    model.save_model()
    
    # Example prediction
    example_applicant = {
        'var_0': -0.13,  # balance_1
        'var_1': -0.45,  # balance_2
        'var_2': -0.26,  # credit_limit_1
        'age': 35,
        'gender': 'MALE',
        'marital_status': 'MARRIED',
        'city': 'Mumbai',
        'state': 'Maharashtra',
        'residence_ownership': 'SELF-OWNED'
    }
    
    prediction = model.predict(example_applicant)
    print(f"\nPredicted income for example applicant: {prediction:.2f}") 