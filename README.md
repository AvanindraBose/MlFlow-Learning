This Repo will help Avanindra to learn about mlflow basics that knwledge will be used for making production grade projects.

# MLflow Interview + Practical Learning Summary

This document summarizes my end-to-end understanding of MLflow from both a **practical implementation** and **interview preparation** perspective.

It covers:

1. What MLflow is
2. Experiment vs Run
3. Local storage architecture (`mlruns` vs `mlflow.db + mlartifacts`)
4. Tracking server architecture
5. Production-grade tracking setups (DagsHub vs AWS)
6. Common MLflow APIs
7. Model Registry
8. `MlflowClient` APIs
9. FastAPI serving integration
10. Model lineage
11. Why `mlflow.autolog()` is avoided in production
12. MLflow vs DVC

---

# 1) What is MLflow?

MLflow is an open-source platform used to manage the machine learning lifecycle.

It helps with:

* Experiment tracking
* Logging parameters
* Logging metrics
* Logging artifacts
* Model versioning
* Model registry
* Deployment

### Problem it solves

Suppose I train multiple models:

* Random Forest
* XGBoost
* Logistic Regression

Each model has:

* different hyperparameters
* different metrics
* different artifacts

Without MLflow:

* difficult to track experiments
* hard to compare models
* easy to lose trained artifacts

MLflow solves this by centralizing experiment management.

---

# 2) Experiment vs Run

This is a very common interview question.

## Experiment

An experiment is simply a **folder/group** of related training attempts.

Example:

`Fraud Detection Project`

---

## Run

A run is one individual training attempt.

Example:

Run 1:

* learning rate = 0.01
* accuracy = 91%

Run 2:

* learning rate = 0.1
* accuracy = 93%

---

### Easy analogy

Experiment = Folder

Run = File inside folder

---

# 3) Local MLflow Storage Architecture

This is commonly asked in interviews.

---

## Older architecture (`mlruns/`)

Older MLflow versions stored everything in one folder.

```
mlruns/
   experiment_id/
      run_id/
         params/
         metrics/
         tags/
         artifacts/
```

It stored:

* parameters
* metrics
* tags
* artifacts
* model files

### Limitation

Everything was stored together.

Not scalable for large teams.

---

# 4) Newer Architecture (`mlflow.db + mlartifacts`)

Newer MLflow setups often separate metadata and artifacts.

---

## `mlflow.db`

Stores metadata:

* experiment IDs
* run IDs
* parameters
* metrics
* tags
* timestamps

This is typically:

* SQLite (local)
* MySQL
* PostgreSQL

---

## `mlartifacts/`

Stores actual files.

Generalized structure:

```
mlartifacts/
   experiment_id/
      run_id/
         artifacts/

      models/
         model_id/
            artifacts/
               model.pkl
               MLmodel
               conda.yaml
               python_env.yaml
               requirements.txt
```

---

## Important file: `MLmodel`

This tells MLflow:

* model flavor
* loading instructions
* metadata

---

# 5) Tracking Server Architecture

A production-grade MLflow setup usually contains 3 layers:

```
Training Code
      ↓
Tracking Server
      ↓
Backend Store + Artifact Store
```

---

## Backend Store

Stores metadata.

Examples:

* PostgreSQL
* MySQL
* SQLite

---

## Artifact Store

Stores actual files.

Examples:

* S3
* GCS
* Azure Blob
* local mlartifacts

---

# 6) DagsHub Setup

Useful for:

* personal projects
* portfolio projects
* small teams

Architecture:

```
Training Code
    ↓
DagsHub Tracking Server
    ↓
Managed Metadata + Managed Artifact Storage
```

### Benefits

* quick setup
* no infrastructure management

### Limitations

* less customization
* vendor dependency

---

# 7) AWS Production Setup

Used in enterprise environments.

Architecture:

```
Training Code
    ↓
MLflow Tracking Server (EC2/ECS/Kubernetes)
    ↓
RDS (metadata)
    ↓
S3 (artifacts)
```

---

## Common AWS components

Tracking Server:

* EC2
* ECS
* Kubernetes

Backend Store:

* PostgreSQL
* MySQL

Artifact Store:

* S3

---

# 8) Common MLflow Functions

These are frequently asked in interviews.

---

## Set tracking URI

```python
mlflow.set_tracking_uri(...)
```

Used to connect code to tracking server.

---

## Set experiment

```python
mlflow.set_experiment("fraud_detection")
```

Creates/selects experiment.

---

## Start run

```python
mlflow.start_run()
```

Creates one run.

---

## Log parameters

```python
mlflow.log_param()
mlflow.log_params()
```

---

## Log metrics

```python
mlflow.log_metric()
mlflow.log_metrics()
```

---

## Log artifacts

```python
mlflow.log_artifact()
mlflow.log_artifacts()
```

---

## Log model

```python
mlflow.sklearn.log_model()
```

Also available for:

* XGBoost
* PyTorch
* TensorFlow

---

## Search runs

```python
mlflow.search_runs()
```

---

## Load model

```python
mlflow.pyfunc.load_model()
```

Used heavily during serving.

---

# 9) Nested Runs

Very important during hyperparameter tuning.

Example:

Parent run:

`Random Forest Tuning`

Child runs:

* depth=5
* depth=10
* depth=15

Implementation:

```python
mlflow.start_run(nested=True)
```

Helps organize tuning experiments.

---

# 10) `mlflow.autolog()`

```python
mlflow.autolog()
```

Automatically logs:

* parameters
* metrics
* models

---

## Why it is avoided in production

* less control
* logs unnecessary artifacts
* harder debugging
* doesn't capture custom business metrics
* inconsistent behavior across frameworks

Production teams prefer manual logging.

---

# 11) Model Registry

One of MLflow's biggest strengths.

Used for model lifecycle management.

Example:

```
Version 1 → Staging
Version 2 → Production
Version 3 → Archived
```

Registry helps with:

* versioning
* promotion
* rollback
* deployment management

---

# 12) Important `MlflowClient` APIs

Used for automation.

---

## Create client

```python
from mlflow.tracking import MlflowClient
client = MlflowClient()
```

---

## Get latest production model

```python
client.get_latest_versions()
```

Returns latest model version.

Example return:

* version
* run_id
* stage

---

## Create model URI

```python
model_uri = f"models:/fraud_model/{version}"
```

---

## Transition model stage

```python
client.transition_model_version_stage()
```

Moves model:

* Staging
  n- Production
* Archived

---

## Create model version

```python
client.create_model_version()
```

---

## Search model versions

```python
client.search_model_versions()
```

---

# 13) FastAPI Integration

Common production workflow:

### Step 1

Fetch latest production model

### Step 2

Create model URI

### Step 3

Load model

```python
mlflow.pyfunc.load_model()
```

### Step 4

Serve predictions through FastAPI

---

# 14) Model Lineage

Model lineage means understanding:

```
Dataset → Run → Parameters → Metrics → Model → Deployment
```

In MLflow, lineage can be seen through:

* run details page
* model registry page
* source run references
* tags

---

# 15) MLflow vs DVC

This is a very common interview question.

---

## DVC is stronger for

* dataset versioning
* pipeline reproducibility
* dataset lineage

---

## MLflow is stronger for

* experiment tracking
* model registry
* model deployment
* visualization

---

## Why not use DVC experiment tracking?

DVC experiment tracking limitations:

* CLI heavy
* weak visualization
* limited model registry
* harder collaboration
* weaker deployment integration

---

## Real-world architecture

Many teams use both:

```
DVC → dataset versioning
MLflow → experiment tracking + model registry
FastAPI → serving
AWS/DagsHub → infrastructure
```

---

# Final Interview Summary

MLflow helps manage the full model lifecycle:

* experiment tracking
* artifact logging
* model versioning
* model registry
* deployment support

For complete production workflows:

DVC handles data reproducibility.

MLflow handles experiment and model lifecycle management.

Together they form a strong MLOps stack.

# Point to Note for Production Grade Project

# Nuances of MLFlow Tracking Server
![alt text](image.png)

# ML artifacts folder structure
This structure has significantly changed

![alt text](mlartifacts-folder-structure.png)

# Importnat Point to note about MLFlow 3.xx

![alt text](Point-to-remember-for-mlflow-3.0.png)
