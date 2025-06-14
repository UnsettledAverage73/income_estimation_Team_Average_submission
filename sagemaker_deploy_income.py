import boto3
import time

# AWS Configuration
role = "arn:aws:iam::926435054259:role/SageMakerExecutionRole"
region = "us-west-2"

# Create a SageMaker boto3 client
sagemaker_client = boto3.client("sagemaker", region_name=region)

# Model configuration
model_name = "income-estimation-model"
endpoint_config_name = "income-endpoint-config"
endpoint_name = "income-endpoint"
instance_type = "ml.m5.large"

# Container configuration for scikit-learn
container_image = "763104351884.dkr.ecr.us-west-2.amazonaws.com/sagemaker-scikit-learn:0.23-1-cpu-py3"
model_data = "s3://averagessgmce/income_estimation_model.pkl"

# Define the primary container
primary_container = {
    "Image": container_image,
    "ModelDataUrl": model_data,
    "Environment": {
        "SAGEMAKER_PROGRAM": "inference.py",
        "SAGEMAKER_SUBMIT_DIRECTORY": "/opt/ml/model"
    }
}

# Define the production variant
production_variant = {
    "VariantName": "AllTraffic",
    "ModelName": model_name,
    "InitialInstanceCount": 1,
    "InstanceType": instance_type
}

# Define the endpoint configuration
endpoint_config = {
    "EndpointConfigName": endpoint_config_name,
    "ProductionVariants": [production_variant]
}

def deploy_model():
    """Deploy the model to SageMaker"""
    try:
        # Create the model
        print("Creating model...")
        sagemaker_client.create_model(
            ModelName=model_name,
            PrimaryContainer=primary_container,
            ExecutionRoleArn=role
        )
        print(f"Model created: {model_name}")

        # Create the endpoint configuration
        print("Creating endpoint configuration...")
        sagemaker_client.create_endpoint_config(**endpoint_config)
        print(f"Endpoint configuration created: {endpoint_config_name}")

        # Create the endpoint
        print("Creating endpoint...")
        sagemaker_client.create_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=endpoint_config_name
        )
        print(f"Endpoint creation initiated: {endpoint_name}")

        # Wait for the endpoint to be in service
        print("Waiting for endpoint to be in service...")
        waiter = sagemaker_client.get_waiter('endpoint_in_service')
        waiter.wait(EndpointName=endpoint_name)
        print(f"Endpoint is now in service: {endpoint_name}")

        return True

    except Exception as e:
        print(f"Error during deployment: {str(e)}")
        return False

def test_endpoint():
    """Test the deployed endpoint with sample data"""
    runtime = boto3.client('sagemaker-runtime', region_name=region)
    
    # Sample input data
    sample_input = {
        "age": 35,
        "gender": "MALE",
        "marital_status": "MARRIED",
        "city": "Mumbai",
        "state": "Maharashtra",
        "residence_ownership": "SELF-OWNED",
        "credit_score": 750,
        "total_loan_amount": 500000,
        "total_credit_limit": 1000000,
        "total_emi": 15000
    }

    try:
        response = runtime.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType='application/json',
            Body=json.dumps(sample_input)
        )
        result = json.loads(response['Body'].read().decode())
        print("Test prediction result:", result)
        return True
    except Exception as e:
        print(f"Error testing endpoint: {str(e)}")
        return False

if __name__ == "__main__":
    import json
    
    # Deploy the model
    if deploy_model():
        print("\nDeployment successful!")
        
        # Test the endpoint
        print("\nTesting endpoint...")
        test_endpoint()
    else:
        print("\nDeployment failed!") 