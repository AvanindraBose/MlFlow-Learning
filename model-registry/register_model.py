# client demo
from mlflow.tracking import MlflowClient
import mlflow
import dagshub
# Initialize the MLflow Client

mlflow.set_tracking_uri("https://dagshub.com/AvanindraBose/MlFlow-Learning.mlflow")
dagshub.init(repo_owner='AvanindraBose', repo_name='MlFlow-Learning', mlflow=True)

client = MlflowClient()

# Replace with the run_id of the run where the model was logged
# run_id = "6c824ecc9cb44edc8e88a9fc7dcb69ec"

# Replace with the path to the logged model within the run
model_id = "m-cbc37b33eee8466eb7722cf23ca4ea5a"

# Construct the model URI
model_uri = f"models:/{model_id}"

# Register the model in the model registry
model_name = "diabities-rf"
result = mlflow.register_model(model_uri, model_name)

import time
time.sleep(5)

# Add a description to the registered model version
client.update_model_version(
    name=model_name,
    version=result.version,
    description="This is a RandomForest model trained to predict diabetes outcomes based on Pima Indians Diabetes Dataset."
)

client.set_model_version_tag(
    name=model_name,
    version=result.version,
    key="experiment",
    value="diabetes prediction"
)

client.set_model_version_tag(
    name=model_name,
    version=result.version,
    key="day",
    value="sat"
)
print(f"Model registered with name: {model_name} and version: {result.version}")
print(f"Added tags to model {model_name} version {result.version}")

# Get and print the registered model information
registered_model = client.get_registered_model(model_name)
print("Registered Model Information:")
print(f"Name: {registered_model.name}")
print(f"Creation Timestamp: {registered_model.creation_timestamp}")
print(f"Last Updated Timestamp: {registered_model.last_updated_timestamp}")
print(f"Description: {registered_model.description}")
