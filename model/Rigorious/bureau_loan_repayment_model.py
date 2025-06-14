import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
from sklearn.model_selection import GridSearchCV
import pickle
import json
import os

class BureauLoanRepaymentModel:
    def __init__(self):
        self.model = None
        self.column_mapping = None
        self.reverse_mapping = None
        self.feature_importance = None
        self.preprocessor = None
        self.threshold = 0.5  # Default threshold for classification
        
        # Set up base paths
        self.base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.data_dir = os.path.join(self.base_dir, 'data')
        
    def load_column_mapping(self, mapping_file=None):
        """Load and create column mappings"""
        if mapping_file is None:
            mapping_file = os.path.join(self.data_dir, 'participant_col_mapping.csv')
        
        try:
            mapping_df = pd.read_csv(mapping_file)
            self.column_mapping = dict(zip(mapping_df['column_name'], mapping_df['description']))
            self.reverse_mapping = {v: k for k, v in self.column_mapping.items()}
            print(f"Successfully loaded column mapping from {mapping_file}")
        except FileNotFoundError:
            print(f"Warning: Column mapping file not found at {mapping_file}")
            print("Creating default column mapping...")
            # Create a default mapping if file not found
            self.column_mapping = {
                'var_0': 'balance_1',
                'var_1': 'balance_2',
                'var_2': 'credit_limit_1',
                'age': 'age',
                'gender': 'gender',
                'marital_status': 'marital_status',
                'city': 'city',
                'state': 'state',
                'residence_ownership': 'residence_ownership'
            }
            self.reverse_mapping = {v: k for k, v in self.column_mapping.items()}
    
    def map_columns(self, df, reverse=False):
        """Map column names using the mapping dictionary"""
        mapping = self.reverse_mapping if reverse else self.column_mapping
        return df.rename(columns=mapping)
    
    def load_data(self, data_file=None):
        """Load and preprocess the bureau data"""
        if data_file is None:
            data_file = os.path.join(self.data_dir, 'Hackathon_bureau_data_400.csv')
        
        try:
            data = pd.read_csv(data_file)
            print(f"Successfully loaded data from {data_file}")
            
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
        except FileNotFoundError:
            raise FileNotFoundError(f"Data file not found at {data_file}. Please ensure the file exists and the path is correct.")
    
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
        
        # Add repayment status features
        if 'target' in data.columns:
            # Create binary target based on income threshold (e.g., median income)
            median_income = data['target'].median()
            data['repayment_status'] = (data['target'] >= median_income).astype(int)
            
            # Add risk-based features
            data['income_to_loan_ratio'] = data['target'] / (data['total_loan_amount'] + 1)
            data['income_to_emi_ratio'] = data['target'] / (data['total_emi'] + 1)
            data['credit_utilization_ratio'] = data['avg_credit_utilization'] / (data['total_credit_limit'] + 1)
            
            # Add payment behavior features
            data['payment_to_income_ratio'] = data['avg_repayment'] / (data['target'] + 1)
            data['payment_consistency_score'] = 1 / (data['repayment_consistency'] + 1)
        
        return data
    
    def prepare_features(self, data):
        """Prepare features for model training"""
        # Separate numeric and categorical features, but exclude target columns
        numeric_features = [col for col in data.columns 
                          if pd.api.types.is_numeric_dtype(data[col]) 
                          and col not in ['target', 'repayment_status']]
        categorical_features = [col for col in data.columns 
                              if pd.api.types.is_object_dtype(data[col])]
        
        # Create preprocessing pipeline
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
            ])
        
        # Create model pipeline with XGBoost Classifier
        self.model = Pipeline(steps=[
            ('preprocessor', self.preprocessor),
            ('classifier', XGBClassifier(
                objective='binary:logistic',
                random_state=42,
                n_estimators=200,
                learning_rate=0.1,
                max_depth=5,
                eval_metric='auc'
            ))
        ])
        
        return numeric_features, categorical_features
    
    def train(self, data_file=None):
        """Train the model with classification metrics"""
        # Load and preprocess data
        data = self.load_data(data_file)
        print(f"Data shape after loading: {data.shape}")
        
        data = self.engineer_features(data)
        print(f"\nData shape after feature engineering: {data.shape}")
        
        # Drop features with >50% missing values
        missing_frac = data.isna().mean()
        cols_to_drop = missing_frac[missing_frac > 0.5].index.tolist()
        print(f"Dropping columns with >50% missing values: {cols_to_drop}")
        data = data.drop(columns=cols_to_drop)
        
        # Clean data
        data = data.replace([np.inf, -np.inf], np.nan)
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col not in ['target', 'repayment_status']:
                median_val = data[col].median()
                data[col] = data[col].fillna(median_val)
        categorical_cols = data.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            mode_val = data[col].mode()[0]
            data[col] = data[col].fillna(mode_val)
        data = data.dropna(subset=['target', 'repayment_status'])
        
        # Prepare features (initialize model pipeline)
        numeric_features, categorical_features = self.prepare_features(data)
        
        # Prepare features
        features = [col for col in data.columns 
                   if col not in ['target', 'repayment_status']]
        X = data[features]
        y = data['repayment_status']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Hyperparameter tuning
        param_grid = {
            'classifier__max_depth': [3, 5, 7],
            'classifier__learning_rate': [0.01, 0.05, 0.1],
            'classifier__n_estimators': [100, 200],
            'classifier__min_child_weight': [1, 3, 5],
            'classifier__subsample': [0.8, 0.9, 1.0]
        }
        
        grid_search = GridSearchCV(
            self.model, param_grid, cv=5, 
            scoring='f1', n_jobs=-1, verbose=1
        )
        grid_search.fit(X_train, y_train)
        self.model = grid_search.best_estimator_
        print(f"\nBest parameters: {grid_search.best_params_}")
        
        # Evaluate model
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        print("\nModel Performance Metrics:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"AUC-ROC: {auc:.4f}")
        
        # Print classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Plot confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()
        
        # Store feature importances
        feature_importances = self.model.named_steps['classifier'].feature_importances_
        feature_names = (numeric_features + 
                        self.model.named_steps['preprocessor']
                        .named_transformers_['cat']
                        .get_feature_names_out(categorical_features).tolist())
        
        self.feature_importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': feature_importances
        }).sort_values('Importance', ascending=False)
        
        return accuracy, precision, recall, f1, auc
    
    def predict(self, input_data, return_proba=False):
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
        if return_proba:
            prediction = self.model.predict_proba(input_data)[:, 1]
        else:
            prediction = self.model.predict(input_data)
        
        return prediction[0]
    
    def set_threshold(self, threshold):
        """Set custom threshold for classification"""
        if not 0 <= threshold <= 1:
            raise ValueError("Threshold must be between 0 and 1")
        self.threshold = threshold
    
    def predict_with_threshold(self, input_data):
        """Make predictions using custom threshold"""
        proba = self.predict(input_data, return_proba=True)
        return int(proba >= self.threshold)
    
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
    try:
        # Initialize and train model
        model = BureauLoanRepaymentModel()
        print("\nInitializing model...")
        
        # Load column mapping
        print("\nLoading column mapping...")
        model.load_column_mapping()
        
        # Train model
        print("\nTraining model...")
        accuracy, precision, recall, f1, auc = model.train()
        
        # Plot feature importance
        print("\nPlotting feature importance...")
        model.plot_feature_importance()
        
        # Save model
        print("\nSaving model...")
        model.save_model()
        
        # Example prediction
        print("\nMaking example prediction...")
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
        
        # Get prediction and probability
        prediction = model.predict(example_applicant)
        probability = model.predict(example_applicant, return_proba=True)
        
        print(f"\nPrediction for example applicant:")
        print(f"Repayment Status: {'Good' if prediction == 1 else 'Bad'}")
        print(f"Probability of Good Repayment: {probability:.4f}")
        
        # Try different threshold
        model.set_threshold(0.7)  # More conservative threshold
        prediction_threshold = model.predict_with_threshold(example_applicant)
        print(f"\nPrediction with 0.7 threshold:")
        print(f"Repayment Status: {'Good' if prediction_threshold == 1 else 'Bad'}")
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        print("\nStack trace:")
        import traceback
        traceback.print_exc() 