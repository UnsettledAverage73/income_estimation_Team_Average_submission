import boto3
import json
import re

# Replace these with your AWS account ID and SageMaker execution role name (e.g. "AmazonSageMaker-ExecutionRole-xxxx" or "user/test")
role = "arn:aws:iam::926435054259:role/SageMakerExecutionRole"

# (Optional) You can also set your AWS region (e.g. "us-east-1") if not set in your environment.
# boto3.setup_default_session(region_name="us-west-2")

# Create a SageMaker boto3 client.
sagemaker_client = boto3.client("sagemaker", region_name="us-west-2")

# (Optional) Define a model name (or use a default one).
model_name = "repayment-model-boto3"

# (Optional) Define an endpoint config name (or use a default one).
endpoint_config_name = "repayment-endpoint-config-boto3"

# (Optional) Define an endpoint name (or use a default one).
endpoint_name = "repayment-endpoint-boto3"

# (Optional) Define the instance type (e.g. "ml.m5.large").
instance_type = "ml.m5.large"

# (Optional) Define the container image (for scikit-learn built-in container).
# (Adjust the framework version if your scikit-learn version differs.)
container_image = "763104351884.dkr.ecr.us-west-2.amazonaws.com/sagemaker-scikit-learn:0.23-1-cpu-py3"

# (Optional) Define the model artifact (S3 URI) (e.g. "s3://averagessgmce/repayment_capability_model.pkl").
model_data = "s3://averagessgmce/repayment_capability_model.pkl"

# (Optional) Define the primary container (for CreateModel).
primary_container = {
    "Image": container_image,
    "ModelDataUrl": model_data,
    "Environment": { "SAGEMAKER_PROGRAM": "inference.py", "SAGEMAKER_SUBMIT_DIRECTORY": "/opt/ml/model" }
}

# (Optional) Define the production variant (for CreateEndpointConfig).
production_variant = {
    "VariantName": "AllTraffic",
    "ModelName": model_name,
    "InitialInstanceCount": 1,
    "InstanceType": instance_type
}

# (Optional) Define the endpoint config (for CreateEndpointConfig).
endpoint_config = {
    "EndpointConfigName": endpoint_config_name,
    "ProductionVariants": [production_variant]
}

# (Optional) Define the endpoint (for CreateEndpoint).
endpoint = {
    "EndpointName": endpoint_name,
    "EndpointConfigName": endpoint_config_name
}

# (Optional) Create the SageMaker model (using CreateModel).
try:
    sagemaker_client.create_model(
        ModelName=model_name,
        PrimaryContainer=primary_container,
        ExecutionRoleArn=role
    )
    print("Model created:", model_name)
except Exception as e:
    print("Error creating model:", e)

# (Optional) Create the endpoint config (using CreateEndpointConfig).
try:
    sagemaker_client.create_endpoint_config(**endpoint_config)
    print("Endpoint config created:", endpoint_config_name)
except Exception as e:
    print("Error creating endpoint config:", e)

# (Optional) Create the endpoint (using CreateEndpoint).
try:
    sagemaker_client.create_endpoint(**endpoint)
    print("Endpoint created (or in progress):", endpoint_name)
except Exception as e:
    print("Error creating endpoint:", e)

# (Optional) Wait (or poll) for the endpoint to be "InService" (using DescribeEndpoint).
# (You can use a loop or a waiter if you want to wait for the endpoint to be "InService".)
# (For example, you can use "sagemaker_client.get_waiter("endpoint_in_service").wait(EndpointName=endpoint_name)")
# (Or you can poll "sagemaker_client.describe_endpoint(EndpointName=endpoint_name)" until "EndpointStatus" is "InService".)

# (Optional) Print the endpoint name (or endpoint_name) for reference.
print("Endpoint (or endpoint config) deployed:", endpoint_name)

# Add Bedrock client setup
bedrock_client = boto3.client('bedrock-runtime', region_name='us-west-2')

def process_natural_language_query(query, prediction):
    """
    Process natural language queries about income predictions using AWS Bedrock.
    """
    prompt = f"""You are a financial advisor assistant. Given the following income prediction score and query, provide a natural response.
    
    Income Prediction Score: {prediction}
    User Query: {query}
    
    Please provide a clear, professional response about the loan eligibility based on the prediction score.
    If the score is above 0.8, mention eligibility for loans above $12,000.
    If the score is between 0.6 and 0.8, mention eligibility for loans up to $12,000.
    If the score is below 0.6, mention that the person may need to improve their financial profile.
    """
    
    try:
        response = bedrock_client.invoke_model(
            modelId='anthropic.claude-v2',
            body=json.dumps({
                "prompt": prompt,
                "max_tokens_to_sample": 200,
                "temperature": 0.5,
                "top_p": 1,
            })
        )
        
        response_body = json.loads(response['body'].read())
        return response_body['completion']
    except Exception as e:
        return f"Error processing query: {str(e)}"

def get_prediction_from_endpoint(input_data):
    """
    Get prediction from SageMaker endpoint
    """
    try:
        runtime = boto3.client('sagemaker-runtime')
        response = runtime.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType='application/json',
            Body=json.dumps(input_data)
        )
        prediction = json.loads(response['Body'].read().decode())
        return prediction
    except Exception as e:
        return f"Error getting prediction: {str(e)}"

# Example usage
if __name__ == "__main__":
    # Example input data (adjust according to your model's requirements)
    sample_input = {
        "features": [0.5, 0.3, 0.2]  # Replace with actual feature values
    }
    
    # Get prediction
    prediction = get_prediction_from_endpoint(sample_input)
    
    # Example natural language query
    query = "What does my income predictability score mean for loan eligibility?"
    response = process_natural_language_query(query, prediction)
    print("\nNatural Language Response:")
    print(response) 