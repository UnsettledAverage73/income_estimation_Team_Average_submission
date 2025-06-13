import requests
import json

BASE_URL = "http://localhost:8000"

def test_health():
    response = requests.get(f"{BASE_URL}/health")
    print("\nHealth Check:")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

def test_feature_list():
    response = requests.get(f"{BASE_URL}/feature-list")
    print("\nFeature List:")
    print(f"Status Code: {response.status_code}")
    print(f"Features: {json.dumps(response.json(), indent=2)}")

def test_predictions():
    # Example features
    features = {
        'var_1': 100,
        'var_2': 0.8,
        'var_3': 1.0,
        'var_4': 500,
        'var_5': 0.5,
        'age': 35,
        'gender': 'M',
        'marital_status': 'Married',
        'residence_ownership': 'Own'
    }
    
    # Test repayment prediction
    print("\nTesting Repayment Prediction:")
    response = requests.post(
        f"{BASE_URL}/predict/repayment",
        json={"features": features, "return_actual_values": False}
    )
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    # Test income prediction
    print("\nTesting Income Prediction:")
    response = requests.post(
        f"{BASE_URL}/predict/income",
        json={"features": features, "return_actual_values": False}
    )
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    # Test combined prediction
    print("\nTesting Combined Prediction:")
    response = requests.post(
        f"{BASE_URL}/predict/combined",
        json={"features": features, "return_actual_values": True}
    )
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

if __name__ == "__main__":
    print("Testing Financial Prediction API...")
    test_health()
    test_feature_list()
    test_predictions() 