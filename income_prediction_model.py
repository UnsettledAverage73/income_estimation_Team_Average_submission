import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectFromModel
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pickle
import json
import warnings
warnings.filterwarnings('ignore')

class IncomePredictionModel:
    def __init__(self):
        self.model = None
        self.column_mapping = None
        self.reverse_mapping = None
        self.feature_importance = None
        self.preprocessor = None
        self.feature_selector = None
        self.best_threshold = None
        
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
        """Load and preprocess the data with improved handling"""
        data = pd.read_csv(data_file)
        
        # Map column names to their descriptions
        if self.column_mapping is None:
            self.load_column_mapping()
        data = self.map_columns(data)
        
        # Drop columns that are not useful for prediction
        columns_to_drop = [
            'unique_id', 'pin code', 'score_comments', 'score_type', 
            'device_model', 'device_category', 'platform', 'device_manufacturer'
        ]
        columns_to_drop = [col for col in columns_to_drop if col in data.columns and col != 'target']
        data = data.drop(columns=columns_to_drop, errors='ignore')
        
        # Ensure target column exists
        if 'target' not in data.columns:
            raise ValueError("Target column 'target' not found in the data")
        
        return data
    
    def engineer_features(self, data):
        """Enhanced feature engineering for income prediction"""
        # Credit utilization features with improved handling
        balance_cols = [col for col in data.columns if 'balance' in col]
        if balance_cols:
            data['avg_credit_utilization'] = data[balance_cols].mean(axis=1)
            data['max_credit_utilization'] = data[balance_cols].max(axis=1)
            data['credit_utilization_std'] = data[balance_cols].std(axis=1)
            data['credit_utilization_range'] = data[balance_cols].max(axis=1) - data[balance_cols].min(axis=1)
        
        # Loan behavior features with ratios
        loan_amt_cols = [col for col in data.columns if 'loan_amt' in col]
        if loan_amt_cols:
            data['total_loan_amount'] = data[loan_amt_cols].sum(axis=1)
            data['avg_loan_amount'] = data[loan_amt_cols].mean(axis=1)
            data['loan_frequency'] = data[loan_amt_cols].count(axis=1)
            data['loan_amount_std'] = data[loan_amt_cols].std(axis=1)
            data['loan_amount_range'] = data[loan_amt_cols].max(axis=1) - data[loan_amt_cols].min(axis=1)
        
        # Credit limit features with ratios
        credit_limit_cols = [col for col in data.columns if 'credit_limit' in col]
        if credit_limit_cols:
            data['total_credit_limit'] = data[credit_limit_cols].sum(axis=1)
            data['avg_credit_limit'] = data[credit_limit_cols].mean(axis=1)
            data['credit_limit_std'] = data[credit_limit_cols].std(axis=1)
            data['credit_limit_range'] = data[credit_limit_cols].max(axis=1) - data[credit_limit_cols].min(axis=1)
        
        # EMI features with ratios
        emi_cols = [col for col in data.columns if 'total_emi' in col]
        if emi_cols:
            data['total_emi'] = data[emi_cols].sum(axis=1)
            data['avg_emi'] = data[emi_cols].mean(axis=1)
            data['emi_std'] = data[emi_cols].std(axis=1)
            data['emi_range'] = data[emi_cols].max(axis=1) - data[emi_cols].min(axis=1)
        
        # Repayment behavior features
        repayment_cols = [col for col in data.columns if 'repayment' in col]
        if repayment_cols:
            data['avg_repayment'] = data[repayment_cols].mean(axis=1)
            data['repayment_consistency'] = data[repayment_cols].std(axis=1)
            data['repayment_range'] = data[repayment_cols].max(axis=1) - data[repayment_cols].min(axis=1)
        
        # Credit inquiry features
        inquiries_cols = [col for col in data.columns if 'inquiries' in col]
        if inquiries_cols:
            data['total_inquiries'] = data[inquiries_cols].sum(axis=1)
            recent_inquiries_col = 'total inquiries recent'
            if recent_inquiries_col in data.columns:
                data['recent_inquiries'] = data[recent_inquiries_col]
                data['inquiry_intensity'] = data['recent_inquiries'] / (data['total_inquiries'] + 1)
        
        # Risk and financial health indicators
        if 'credit_score' in data.columns:
            data['credit_risk_score'] = data['credit_score']
            data['credit_score_normalized'] = (data['credit_score'] - data['credit_score'].min()) / (data['credit_score'].max() - data['credit_score'].min())
        
        # Additional income-specific features
        if 'age' in data.columns:
            data['age_squared'] = data['age'] ** 2
            data['age_cubed'] = data['age'] ** 3
        
        # Financial ratios
        if 'total_loan_amount' in data.columns and 'total_credit_limit' in data.columns:
            data['loan_to_credit_ratio'] = data['total_loan_amount'] / (data['total_credit_limit'] + 1)
        
        if 'total_emi' in data.columns and 'total_loan_amount' in data.columns:
            data['emi_to_loan_ratio'] = data['total_emi'] / (data['total_loan_amount'] + 1)
        
        if 'total_credit_limit' in data.columns and 'avg_credit_utilization' in data.columns:
            data['credit_utilization_ratio'] = data['avg_credit_utilization'] / (data['total_credit_limit'] + 1)
        
        # Interaction features
        if 'credit_score' in data.columns and 'age' in data.columns:
            data['credit_score_age_interaction'] = data['credit_score'] * data['age']
        
        if 'total_loan_amount' in data.columns and 'credit_score' in data.columns:
            data['loan_amount_credit_score_interaction'] = data['total_loan_amount'] * data['credit_score']
        
        return data
    
    def train(self, data_file='data/Hackathon_bureau_data_400.csv'):
        """Enhanced training process with improved preprocessing and feature selection"""
        # Load and preprocess data
        data = self.load_data(data_file)
        print(f"Data shape after loading: {data.shape}")
        print("\nTarget variable (income) statistics before cleaning:")
        print(data['target'].describe())
        
        # Engineer features
        data = self.engineer_features(data)
        print(f"\nData shape after feature engineering: {data.shape}")
        
        # Handle missing values more intelligently
        missing_frac = data.isna().mean()
        cols_to_drop = missing_frac[missing_frac > 0.7].index.tolist()  # Increased threshold
        print(f"Dropping columns with >70% missing values: {cols_to_drop}")
        data = data.drop(columns=cols_to_drop)
        
        # Clean data with improved handling
        data = data.replace([np.inf, -np.inf], np.nan)
        
        # Separate numeric and categorical columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        categorical_cols = data.select_dtypes(include=['object']).columns
        
        # Handle missing values with more sophisticated approach
        for col in numeric_cols:
            if col != 'target':
                col_non_na = data[col].dropna()
                if len(col_non_na) == 0:
                    # If all values are NaN, fill with 0
                    data[col] = data[col].fillna(0)
                else:
                    skewness = col_non_na.skew()
                    if pd.isna(skewness) or skewness > 1:
                        data[col] = data[col].fillna(col_non_na.median())
                    else:
                        data[col] = data[col].fillna(col_non_na.mean())
        
        for col in categorical_cols:
            data[col] = data[col].fillna(data[col].mode()[0])
        
        data = data.dropna(subset=['target'])
        print(f"\nData shape after cleaning: {data.shape}")
        
        # Log-transform target and key numeric features with improved handling
        data['target_log'] = np.log1p(data['target'].clip(lower=0))
        for col in ['total_loan_amount', 'total_credit_limit', 'total_emi', 'income']:
            if col in data.columns:
                data[col + '_log'] = np.log1p(data[col].clip(lower=0))
        
        # Prepare features
        features = [col for col in data.columns if col not in ['target', 'target_log']]
        X = data[features]
        y = data['target_log']
        
        # Split data with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Identify numeric and categorical features
        numeric_features = [col for col in X.columns if pd.api.types.is_numeric_dtype(X[col])]
        categorical_features = [col for col in X.columns if pd.api.types.is_object_dtype(X[col])]
        
        # Create enhanced preprocessing pipeline
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', RobustScaler())  # More robust to outliers
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])
        
        # Create base model for feature selection
        base_model = XGBRegressor(
            objective='reg:squarederror',
            random_state=42,
            n_estimators=100
        )
        
        # Feature selection
        selector = SelectFromModel(
            base_model,
            prefit=False,
            threshold='median'
        )
        
        # Create enhanced model pipeline
        self.model = Pipeline(steps=[
            ('preprocessor', self.preprocessor),
            ('feature_selector', selector),
            ('regressor', XGBRegressor(
                objective='reg:squarederror',
                random_state=42
            ))
        ])
        
        # Enhanced hyperparameter tuning
        param_grid = {
            'regressor__max_depth': [3, 5, 7],
            'regressor__learning_rate': [0.01, 0.05, 0.1],
            'regressor__n_estimators': [200, 300],
            'regressor__min_child_weight': [1, 3, 5],
            'regressor__subsample': [0.8, 0.9, 1.0],
            'regressor__colsample_bytree': [0.8, 0.9, 1.0],
            'regressor__gamma': [0, 0.1, 0.2],
            'regressor__reg_alpha': [0, 0.1, 0.5],
            'regressor__reg_lambda': [0.1, 1.0, 5.0]
        }
        
        # Use KFold for more robust cross-validation
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        
        grid_search = GridSearchCV(
            self.model,
            param_grid,
            cv=kf,
            scoring='r2',
            n_jobs=-1,
            verbose=1
        )
        
        print("\nStarting enhanced model training with hyperparameter tuning...")
        grid_search.fit(X_train, y_train)
        self.model = grid_search.best_estimator_
        print(f"\nBest parameters: {grid_search.best_params_}")
        
        # Evaluate model with multiple metrics
        y_pred_log = self.model.predict(X_test)
        y_pred = np.expm1(y_pred_log)
        y_test_actual = np.expm1(y_test)
        
        rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred))
        mae = mean_absolute_error(y_test_actual, y_pred)
        r2 = r2_score(y_test_actual, y_pred)
        
        print("\nEnhanced Model Performance:")
        print(f"RMSE: {rmse:.2f}")
        print(f"MAE: {mae:.2f}")
        print(f"R² Score: {r2:.4f}")
        
        # Cross-validation with multiple metrics
        cv_scores = cross_val_score(
            self.model,
            X,
            y,
            cv=kf,
            scoring='r2'
        )
        print(f"\nCross-validation R2 scores: {cv_scores}")
        print(f"Average CV R2 score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        
        # Store feature importances
        feature_importances = self.model.named_steps['regressor'].feature_importances_
        selected_features = self.model.named_steps['feature_selector'].get_support()
        feature_names = (
            numeric_features +
            self.model.named_steps['preprocessor']
            .named_transformers_['cat']
            .named_steps['onehot']
            .get_feature_names_out(categorical_features)
            .tolist()
        )
        selected_feature_names = [f for f, s in zip(feature_names, selected_features) if s]
        
        self.feature_importance = pd.DataFrame({
            'Feature': selected_feature_names,
            'Importance': feature_importances
        }).sort_values('Importance', ascending=False)
        
        # Plot actual vs predicted values
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test_actual, y_pred, alpha=0.5)
        plt.plot([y_test_actual.min(), y_test_actual.max()],
                [y_test_actual.min(), y_test_actual.max()],
                'r--', lw=2)
        plt.xlabel('Actual Income')
        plt.ylabel('Predicted Income')
        plt.title('Actual vs Predicted Income')
        plt.tight_layout()
        plt.savefig('income_prediction_actual_vs_predicted.png')
        plt.close()
        
        return rmse, mae, r2
    
    def predict(self, input_data):
        """Enhanced prediction with better error handling"""
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
        
        # Make prediction with error handling
        try:
            prediction_log = self.model.predict(input_data)
            prediction = np.expm1(prediction_log)
            return prediction[0]
        except Exception as e:
            print(f"Error making prediction: {str(e)}")
            return None
    
    def save_model(self, model_path='income_prediction_model.pkl'):
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
    
    def load_model(self, model_path='income_prediction_model.pkl'):
        """Load a saved model"""
        with open(model_path, 'rb') as f:
            model_info = pickle.load(f)
        
        self.model = model_info['model']
        self.column_mapping = model_info['column_mapping']
        self.reverse_mapping = model_info['reverse_mapping']
        self.feature_importance = model_info['feature_importance']
        self.preprocessor = model_info['preprocessor']
    
    def plot_feature_importance(self, top_n=20):
        """Enhanced feature importance plotting"""
        if self.feature_importance is None:
            print("Model needs to be trained first")
            return
        
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Importance', y='Feature',
                   data=self.feature_importance.head(top_n))
        plt.title(f'Top {top_n} Features for Income Prediction')
        plt.tight_layout()
        plt.savefig('income_prediction_feature_importance.png')
        plt.close()

# Example usage
if __name__ == "__main__":
    # Initialize and train model
    model = IncomePredictionModel()
    model.load_column_mapping()
    rmse, mae, r2 = model.train()
    
    # Plot feature importance
    model.plot_feature_importance()
    
    # Save model
    model.save_model()
    
    # Example prediction with more realistic data
    example_applicant = {
        'age': 35,
        'gender': 'MALE',
        'marital_status': 'MARRIED',
        'city': 'Mumbai',
        'state': 'Maharashtra',
        'residence_ownership': 'SELF-OWNED',
        'credit_score': 750,
        'total_loan_amount': 500000,
        'total_credit_limit': 1000000,
        'total_emi': 15000,
        'var_1': 100,  # balance_1
        'var_2': 0.8,  # balance_2
        'var_3': 1.0,  # credit_limit_1
        'var_4': 500,  # loan_amt_1
        'var_5': 0.5   # repayment_1
    }
    
    predicted_income = model.predict(example_applicant)
    if predicted_income is not None:
        print(f"\nPredicted income for example applicant: ₹{predicted_income:,.2f}") 