import numpy as np
import pandas as pd
from scipy import stats
import random
from datetime import datetime

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# Number of records to generate
n_records = 10000

# Generate synthetic data
def generate_synthetic_data(n_records):
    # Basic demographic features
    education_levels = ['High School', 'Bachelor', 'Master', 'PhD', 'Diploma']
    housing_types = ['Rented', 'Owned', 'Joint', 'Government']
    occupations = ['Professional', 'Service', 'Business', 'Skilled Worker', 'Unskilled Worker', 
                  'Student', 'Retired', 'Unemployed']
    
    # Generate base features
    data = {
        'id': range(1, n_records + 1),
        'occupation': np.random.choice(occupations, n_records, p=[0.3, 0.2, 0.15, 0.15, 0.1, 0.05, 0.03, 0.02]),
        'education': np.random.choice(education_levels, n_records, p=[0.3, 0.4, 0.2, 0.05, 0.05]),
        'is_urban': np.random.choice([True, False], n_records, p=[0.7, 0.3]),
        'housing_type': np.random.choice(housing_types, n_records, p=[0.4, 0.4, 0.1, 0.1]),
        'household_size': np.random.randint(1, 8, n_records),
    }
    
    # Generate income (base distribution)
    base_income = np.random.lognormal(mean=10.5, sigma=0.5, size=n_records)
    
    # Adjust income based on education and occupation
    education_multiplier = {
        'High School': 0.7,
        'Diploma': 0.85,
        'Bachelor': 1.0,
        'Master': 1.3,
        'PhD': 1.5
    }
    
    occupation_multiplier = {
        'Professional': 1.4,
        'Service': 0.9,
        'Business': 1.3,
        'Skilled Worker': 1.0,
        'Unskilled Worker': 0.7,
        'Student': 0.5,
        'Retired': 0.8,
        'Unemployed': 0.3
    }
    
    # Apply multipliers to income
    income = base_income.copy()
    for i in range(n_records):
        income[i] *= education_multiplier[data['education'][i]] * occupation_multiplier[data['occupation'][i]]
    
    data['income'] = np.round(income, 2)
    
    # Generate correlated financial features
    data['upi_txns_month'] = np.random.poisson(lam=15, size=n_records) + np.random.randint(0, 10, n_records)
    data['mobile_recharge_freq'] = np.random.poisson(lam=2, size=n_records)
    data['night_light'] = np.random.choice([True, False], n_records, p=[0.3, 0.7])
    data['bill_payment_consistency'] = np.random.uniform(0.5, 1.0, n_records)
    data['ecommerce_freq'] = np.random.poisson(lam=3, size=n_records)
    
    # Generate skill level (correlated with education and income)
    base_skill = (np.array([education_multiplier[edu] for edu in data['education']]) * 
                 np.array([occupation_multiplier[occ] for occ in data['occupation']]))
    data['skill_level'] = np.round(base_skill * np.random.uniform(0.8, 1.2, n_records), 2)
    
    # Generate location-based features
    data['market_density'] = np.random.uniform(0.1, 1.0, n_records)
    data['public_transport_access'] = np.random.uniform(0.1, 1.0, n_records)
    
    # Generate loan-related features
    max_loan_amount = data['income'] * 5  # Maximum loan amount based on income
    data['loan_amount'] = np.round(np.random.uniform(0, max_loan_amount), 2)
    data['income_to_loan_ratio'] = np.round(data['income'] / (data['loan_amount'] + 1), 2)  # Add 1 to avoid division by zero
    
    # Generate repayment capability (correlated with income, skill level, and bill payment consistency)
    base_repayment = (data['income'] * data['skill_level'] * data['bill_payment_consistency']) / 1000
    data['repayment_capability'] = np.round(base_repayment * np.random.uniform(0.8, 1.2, n_records), 2)
    
    return pd.DataFrame(data)

# Generate the dataset
df = generate_synthetic_data(n_records)

# Save to CSV
output_file = 'synthetic_financial_data.csv'
df.to_csv(output_file, index=False)

# Print some basic statistics
print(f"\nGenerated {n_records} records of synthetic data")
print("\nBasic statistics:")
print(df.describe())
print("\nData saved to:", output_file)

# Print correlation matrix for numerical columns
numerical_cols = df.select_dtypes(include=[np.number]).columns
correlation_matrix = df[numerical_cols].corr()
print("\nCorrelation matrix for numerical features:")
print(correlation_matrix)