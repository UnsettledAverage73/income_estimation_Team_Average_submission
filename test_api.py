import requests
import json
import sys
import time
from typing import Dict, Any

BASE_URL = "http://localhost:8000"
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds

def wait_for_server(max_retries: int = MAX_RETRIES, delay: int = RETRY_DELAY) -> bool:
    """Wait for server to be ready"""
    for i in range(max_retries):
        try:
            response = requests.get(f"{BASE_URL}/health")
            if response.status_code == 200:
                print("Server is ready!")
                return True
        except requests.exceptions.ConnectionError:
            if i < max_retries - 1:
                print(f"Server not ready, retrying in {delay} seconds...")
                time.sleep(delay)
            continue
    return False

def test_endpoint(endpoint: str, method: str, data: Dict[str, Any] = None) -> bool:
    """Test an API endpoint"""
    print(f"\nTesting {method} {endpoint}")
    try:
        if method == "GET":
            response = requests.get(f"{BASE_URL}{endpoint}")
        else:
            response = requests.post(f"{BASE_URL}{endpoint}", json=data)
        
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            print("Response:")
            print(json.dumps(response.json(), indent=2))
            return True
        else:
            print(f"Error: {response.text}")
            return False
    except Exception as e:
        print(f"Error: {str(e)}")
        return False

def main():
    # Test data with only the real feature names required by the model
    test_features = {
        "total_loan_amount": 100000,
        "loan_amt_4": 5000,
        "credit_utilization_ratio": 0.3,
        "total_emi": 2000,
        "total_loans_1": 2,
        "loan_amt_7": 3000,
        "total_emi_4": 1000,
        "balance_6": 0,
        "total_loans_4": 1,
        "loan_frequency": 0.5,
        "balance_7": 0,
        "total_loans_5": 1,
        "total_loan_recent": 10000,
        "repayment_1": 0.9,
        "total_emi_2": 1500,
        "avg_emi": 1200,
        "credit_limit_8": 20000,
        "recent_inquiries": 1,
        "balance_5": 0,
        "credit_limit_9": 25000,
        "loan_amt_recent": 4000,
        "loan_amt_8": 3500,
        "avg_credit_utilization": 0.25,
        "credit_score": 700,
        "max_credit_utilization": 0.5,
        "balance_1": 0,
        "balance_8": 0,
        "total_inquiries_1": 1,
        "total_inquires_5": 0,
        "repayment_consistency": 0.8,
        "total_emi_log": 7.6,
        "loan_amt_1_log": 8.5,
        "primary_loan_amt": 5000,
        "repayment_2": 0.8,
        "total_inquiries_2": 1,
        "total_credit_limit_log": 10.5,
        "balance_9": 0,
        "balance_2": 0,
        "balance_4": 0,
        "closed_loan": 0,
        "total_emi_1": 1000,
        "loan_amt_3": 2000,
        "total_inquires_6": 0,
        "avg_repayment": 0.75,
        "debt_to_income": 0.3,
        "repayment_5": 0.7,
        "total_emi_7": 900,
        "repayment_4": 0.7,
        "loan_amt_large_tenure": 6000,
        "business_balance": 0,
        "total_inquires_4": 0,
        "repayment_7": 0.6,
        "credit_risk_score": 650,
        "total_loan_amount_log": 11.5,
        "total_emi_6": 800,
        "avg_loan_amount": 3000,
        "loan_amt_1": 1000,
        "repayment_8": 0.5,
        "avg_credit_limit": 15000,
        "credit_limit_recent_1": 18000,
        "closed_total_loans": 0,
        "balance_12": 0,
        "repayment_10": 0.4,
        "repayment_9": 0.45,
        "total_credit_limit": 120000,
        "loan_amt_5": 2500,
        "total_inquiries": 2,
        "total_inquires_3": 0,
        "loan_amt_6": 2200,
        # Add any other features required by your model here
        "age": 35,
        "gender": "MALE",
        "marital_status": "MARRIED",
        "residence_ownership": "SELF-OWNED",
        "city": "MUMBAI",
        "state": "MAHARASHTRA"
    }
    
    # Wait for server to be ready
    if not wait_for_server():
        print("Failed to connect to server. Make sure it's running.")
        return 1
    
    # Test all endpoints
    endpoints = [
        ("/", "GET"),
        ("/health", "GET"),
        ("/predict/bureau", "POST", {"features": test_features, "threshold": 0.5})
    ]
    
    success_count = 0
    for endpoint, method, *args in endpoints:
        data = args[0] if args else None
        if test_endpoint(endpoint, method, data):
            success_count += 1
    
    print(f"\nTest Summary: {success_count}/{len(endpoints)} endpoints successful")
    return 0 if success_count == len(endpoints) else 1

if __name__ == "__main__":
    sys.exit(main()) 