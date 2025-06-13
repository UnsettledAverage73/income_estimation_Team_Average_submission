import sagemaker
from sagemaker.sklearn.model import SKLearnModel

# Replace these with your AWS account ID and SageMaker execution role name (e.g. "AmazonSageMaker-ExecutionRole-xxxx")
role = "arn:aws:iam::926435054259:user/test"

# Create a SageMaker model using the built-in scikit-learn container.
# (The model artifact is uploaded at s3://averagessgmce/repayment_capability_model.pkl.)
model = SKLearnModel(
    model_data="s3://averagessgmce/repayment_capability_model.pkl",
    role=role,
    entry_point=None,  # (Using built-in inference, no custom inference script.)
    framework_version="0.23-1"  # (Adjust if your scikit-learn version differs.)
)

# Deploy the model as an endpoint (using an ml.m5.large instance) named "repayment-endpoint".
predictor = model.deploy(instance_type="ml.m5.large", endpoint_name="repayment-endpoint")

# (Optional) Print the endpoint name (or predictor.endpoint_name) for reference.
print("Endpoint deployed:", predictor.endpoint_name) 