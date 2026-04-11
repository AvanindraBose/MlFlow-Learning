from mlflow.tracking import MlflowClient
import dagshub
import mlflow
# Initialize the MLflow Client
mlflow.set_tracking_uri("https://dagshub.com/AvanindraBose/MlFlow-Learning.mlflow")
dagshub.init(repo_owner='AvanindraBose', repo_name='MlFlow-Learning', mlflow=True)
client = MlflowClient()

# Define the model name and version
model_name = "diabities-rf"
model_version = 3  # Replace with the specific version number you want to transition

# Transition the model version to a new stage
new_stage = "Production"  # Possible values: "None", "Staging", "Production", "Archived"
client.transition_model_version_stage(
    name=model_name,
    version=model_version,
    stage=new_stage,
    archive_existing_versions=True
)

print(f"Model version {model_version} transitioned to {new_stage}")