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

def test_health():
    """Test the health endpoint"""
    response = requests.get(f"{BASE_URL}/health")
    print("\nTesting health endpoint:")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")

def test_model_info():
    """Test the model info endpoint"""
    response = requests.get(f"{BASE_URL}/model-info")
    print("\nTesting model info endpoint:")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

def test_prediction():
    """Test the prediction endpoint with sample data"""
    # Sample input data
    sample_input = {
        "age": 35,
        "credit_score": 750,
        "total_credit_limit_overall": 500000,
        "total_balance_overall": 200000,
        "total_emi_overall": 15000,
        "total_inquiries_overall": 5,
        "total_loan_amount_overall": 300000,
        "city": "Mumbai",
        "state": "Maharashtra",
        "marital_status": "Married",
        "gender": "Male",
        "residence_ownership": "Owned"
    }
    
    print("\nTesting prediction endpoint:")
    print(f"Input data: {json.dumps(sample_input, indent=2)}")
    
    response = requests.post(f"{BASE_URL}/predict", json=sample_input)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

def main():
    """Run all tests"""
    print("Starting API tests...")
    
    try:
        # Test health endpoint
        test_health()
        
        # Test model info endpoint
        test_model_info()
        
        # Test prediction endpoint
        test_prediction()
        
    except requests.exceptions.ConnectionError:
        print("\nError: Could not connect to the API. Make sure the server is running on http://localhost:8000")
    except Exception as e:
        print(f"\nError during testing: {str(e)}")

if __name__ == "__main__":
    sys.exit(main()) 