import mlflow.pyfunc
import mlflow
import pandas as pd
import dagshub

mlflow.set_tracking_uri("https://dagshub.com/AvanindraBose/MlFlow-Learning.mlflow")
dagshub.init(repo_owner='AvanindraBose', repo_name='MlFlow-Learning', mlflow=True)

data = pd.DataFrame({
    'Pregnancies': [1],
    'Glucose': [85],
    'BloodPressure': [66],
    'SkinThickness': [29],
    'Insulin': [0],
    'BMI': [26.6],
    'DiabetesPedigreeFunction': [0.351],
    'Age': [31]
})

model_name = "diabities-rf"
model_version = 3

model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{model_version}")

result = model.predict(data)
print(f"Prediction: {result}")
print(f"0 = No Diabetes, 1 = Diabetes")
