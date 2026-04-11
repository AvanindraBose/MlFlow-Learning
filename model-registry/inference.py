import mlflow.pyfunc
import mlflow
import numpy as np
import dagshub

mlflow.set_tracking_uri("https://dagshub.com/AvanindraBose/MlFlow-Learning.mlflow")
dagshub.init(repo_owner='AvanindraBose', repo_name='MlFlow-Learning', mlflow=True)

data = np.array([1,85,66,29,0,26.6,0.351,31]).reshape(1,-1)

model_name = "diabities-rf"
model_version = 3

model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{model_version}")

print(model.predict(data))
