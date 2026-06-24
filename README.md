# End_to_End_ML_project_for_Ad_Click_Through_Rate_Prediction

# 🚀 End-to-End Ad Click Through Rate (CTR) Prediction MLOps Platform

<p align="center">

<img src="https://img.shields.io/badge/Python-3.11-blue">
<img src="https://img.shields.io/badge/XGBoost-ML_Model-green">
<img src="https://img.shields.io/badge/FastAPI-Production_API-success">
<img src="https://img.shields.io/badge/DVC-Data_Versioning-purple">
<img src="https://img.shields.io/badge/MLflow-Experiment_Tracking-orange">
<img src="https://img.shields.io/badge/Docker-Containerized-blue">
<img src="https://img.shields.io/badge/AWS-Cloud_Deployment-yellow">
<img src="https://img.shields.io/badge/GitHub_Actions-CI/CD-red">

</p>

---

# 📌 Project Overview

This project implements a complete **production-grade MLOps platform** for predicting whether a user will click on an advertisement.

The solution combines:

* Machine Learning Engineering
* Data Engineering
* MLOps
* Software Engineering
* Cloud Deployment

into a reproducible, scalable, and production-ready system.

The platform predicts user click behavior using:

* User Features
* Advertisement Features
* Advertiser Features
* Page Features
* Context Features
* Historical Engagement Features

---

# 🎯 Business Problem

Digital advertising platforms generate billions of ad impressions every day.

Displaying the right advertisement to the right user at the right time is critical for:

* Revenue Growth
* User Engagement
* Campaign Optimization
* Advertising ROI

The objective is:

> Predict the probability that a user clicks on a given advertisement.

Prediction Output:

```text
0 → No Click

1 → Click
```

---

# 🏢 Real-World Applications

This type of system is widely used in:

* Google Ads
* Meta Ads
* Amazon Advertising
* LinkedIn Ads
* TikTok Ads
* Programmatic Advertising Platforms
* DSP Platforms
* RTB Systems

---

# 🏗️ Complete System Architecture

```text
Raw Dataset
    │
    ▼
Data Ingestion
    │
    ▼
Data Validation
    │
    ▼
Data Preprocessing
    │
    ▼
Feature Engineering
    │
    ▼
Data Transformation
    │
    ▼
Model Training
    │
    ▼
MLflow Tracking
    │
    ▼
Model Evaluation
    │
    ▼
Model Promotion
    │
    ▼
Production Model
    │
    ▼
FastAPI Inference Layer
    │
    ▼
Docker Container
    │
    ▼
AWS Deployment
```

---

# ⚙️ Tech Stack

## Machine Learning

* XGBoost
* Scikit-Learn
* Pandas
* NumPy

## MLOps

* DVC
* MLflow

## Backend

* FastAPI
* Uvicorn

## Deployment

* Docker
* AWS S3
* AWS ECR
* AWS EC2

## CI/CD

* GitHub Actions

## Version Control

* Git
* GitHub

---

# 📂 Project Structure

```text
Ad_CTR_Prediction/

├── .github/
│   └── workflows/
│       └── aws.yaml
│
├── artifacts/
│
├── config/
│   ├── config.yaml
│   └── schema.yaml
│
├── params.yaml
├── dvc.yaml
│
├── logs/
│
├── research/
│
├── templates/
│   └── index.html
│
├
│
├── src/
│
│   └── ad_ctr_prediction/
│
│       ├── components/
│       ├── pipeline/
│       ├── entity/
│       ├── config/
│       ├── constants/
│       ├── transformers/
│       ├── utils/
│      
│     
│
├── Dockerfile
├── app.py
├── requirements.txt
├── setup.py
└── README.md
```

---

# 📊 Dataset Features

## User Features

```text
user_age_group
user_gender
user_device_type
user_os
user_browser
user_interest_category

user_total_impressions
user_total_clicks
user_historical_ctr
user_avg_session_duration
user_days_active
```

## Advertisement Features

```text
ad_type
ad_format
ad_position

ad_quality_score
ad_relevance_score
```

## Advertiser Features

```text
advertiser_industry
advertiser_historical_ctr
advertiser_budget_utilization
```

## Page Features

```text
page_category
page_quality_score
page_dwell_time
```

## Context Features

```text
hour_of_day
day_of_week
month
is_weekend
```

## Interaction Features

```text
interest_match

sequence_position

previous_ad_clicks_today

time_since_last_click

time_since_last_impression
```

---

# 🧠 Advanced Feature Engineering

The project creates domain-specific CTR features.

### User Features

```python
user_click_ratio
activity_score
```

### Advertiser Features

```python
budget_efficiency
ad_strength
```

### Engagement Features

```python
engagement_score

user_ad_affinity

interest_quality

user_page_affinity
```

### Behavioral Features

```python
recent_clicker

recent_impression

ad_fatigue

click_momentum
```

### Time Features

```python
peak_hour

night_user

weekend_evening

quarter

month_sin

month_cos
```

---

# 🔄 DVC Pipeline

The entire machine learning workflow is automated using DVC.

## Stage 01

### Data Ingestion

* Download data
* Load source files
* Store raw artifacts

---

## Stage 02

### Data Validation

Checks:

* Schema Validation
* Missing Values
* Duplicate Records
* Data Types
* Column Consistency

---

## Stage 03

### Data Preprocessing

Operations:

* Missing Value Handling
* Data Cleaning
* Type Conversion
* Data Standardization

---

## Stage 04

### Feature Engineering

Creates:

* Behavioral Features
* Temporal Features
* CTR Features
* Engagement Features

---

## Stage 05

### Data Transformation

Includes:

* Frequency Encoding
* Feature Selection
* Scaling
* Train Validation Test Split

---

## Stage 06

### Model Training

Algorithm:

```text
XGBoost Classifier
```

Supports:

* Early Stopping
* Hyperparameter Logging
* Reproducible Training

---

## Stage 07

### Model Evaluation

Metrics:

```text
Accuracy
Precision
Recall
F1 Score
ROC AUC
PR AUC
Log Loss
```

---

## Stage 08

### Model Promotion

Promotes model automatically when evaluation thresholds are satisfied.

---

## Stage 09

### Model Inference

Generates production-ready predictions.

---

# 📈 MLflow Experiment Tracking

Every experiment logs:

## Parameters

```yaml
learning_rate
max_depth
n_estimators
subsample
colsample_bytree
```

## Metrics

```yaml
accuracy
precision
recall
f1_score
roc_auc
pr_auc
log_loss
```

## Artifacts

```yaml
model.pkl
metrics.json
feature_importance.png
roc_curve.png
confusion_matrix.png
```

Run MLflow:

```bash
mlflow ui
```

Open:

```text
http://localhost:5000
```

---

# 🚀 FastAPI Service

## Features

### Single Prediction

```http
POST /predict
```

### Batch Prediction

```http
POST /predict/batch
```

### Health Check

```http
GET /health
```

### Readiness Probe

```http
GET /health/ready
```

### Liveness Probe

```http
GET /health/live
```

---

# 🐳 Docker Deployment

Build Docker Image

```bash
docker build -t ad-ctr-prediction .
```

Run Container

```bash
docker run -d \
-p 8000:8000 \
--name adctr \
ad-ctr-prediction
```

Verify

```bash
docker ps
```

---

# ☁️ AWS Deployment Architecture

```text
GitHub
   │
   ▼
GitHub Actions
   │
   ▼
Docker Build
   │
   ▼
Amazon ECR
   │
   ▼
AWS EC2
   │
   ▼
FastAPI Container
   │
   ▼
Production API
```

---

# 🪣 AWS S3 Integration

S3 stores:

```text

Models

Preprocessors

Feature Selectors

Feature Names
```

Configure Remote:

```bash
dvc remote add -d s3remote s3://your-bucket-name
```

Push Artifacts:

```bash
dvc push
```

Pull Artifacts:

```bash
dvc pull
```

---

# 🔄 GitHub Actions CI/CD

Workflow executes automatically on push.

## Continuous Integration

* Code Validation
* Unit Testing
* DVC Validation
* Dependency Checks

## Continuous Deployment

* Docker Build
* Push Image to ECR
* Deploy to EC2
* Health Verification
* Deployment Validation

---

# 📊 Production Monitoring

Future Enhancements:

### Infrastructure Monitoring

* AWS CloudWatch
* Prometheus
* Grafana

### ML Monitoring

* Data Drift Detection
* Feature Drift Detection
* Prediction Drift Monitoring
* Model Performance Monitoring

---

# 🔐 Production Features

### Engineering

✅ Modular Architecture

✅ OOP Design

✅ Structured Logging

✅ Exception Handling

✅ Configuration Management

---

### MLOps

✅ DVC Pipelines

✅ MLflow Tracking

✅ Automated Evaluation

✅ Model Promotion Workflow

---

### Deployment

✅ FastAPI

✅ Docker

✅ AWS S3

✅ AWS ECR

✅ AWS EC2

✅ GitHub Actions

---

# 🏆 Key Achievements

* End-to-End MLOps Pipeline
* Production-Grade FastAPI Service
* Automated Experiment Tracking
* Reproducible Data Pipelines
* Cloud Artifact Management
* Automated CI/CD Deployment
* Enterprise-Level Project Structure
* Scalable Inference Architecture

---

# 🚀 Getting Started

### Clone Repository

```bash
git clone https://github.com/ajaychaudhary8104/End_to_End_ML_project_for_Ad_Click_Through_Rate_Prediction.git

cd Ad_CTR_Prediction
```

### Create Environment

```bash
conda create -n adctr python=3.11

conda activate adctr
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Pipeline

```bash
dvc repro
```

### Start FastAPI

```bash
python app.py
```

Open:

```text
http://localhost:8000
```

Swagger:

```text
http://localhost:8000/api/docs
```

---

# 👨‍💻 Author

**Ajay Chaudhary**

Machine Learning Engineer | MLOps Engineer | Data Scientist

### Technologies Used

```text
Python
Pandas
NumPy
Scikit-Learn
XGBoost
DVC
MLflow
FastAPI
Docker
AWS S3
AWS ECR
AWS EC2
GitHub Actions
```

⭐ If you found this project useful, please consider giving it a star.


# Configuration Workflow

To modify the pipeline:

1. Update `config/config.yaml` - Set paths and parameters
2. Update `params.yaml` - Modify hyperparameters
3. Update `entity/config_entity.py` - Define configuration entities
4. Update `config/configuration.py` - Implement configuration manager
5. Update components in `components/` - Modify pipeline stages
6. Update pipeline stages in `pipeline/` - Update pipeline logic
7. Update `main.py` - Execute the pipeline
8. Update `dvc.yaml` - Define DVC pipeline stages


set MLFLOW_TRACKING_URI=https://dagshub.com/ajaychaudhary8104/End_to_End_ML_project_for_Ad_Click_Through_Rate_Prediction.mlflow
set MLFLOW_TRACKING_USERNAME=ajaychaudhary8104
set MLFLOW_TRACKING_PASSWORD=gangapur8955


## Build Docker Image

```bash
docker build -t ad-ctr-prediction:latest .
```

## Run Docker Container

```bash
docker run -p 8000:8000 ad-ctr-prediction:latest
```

# AWS CI/CD Deployment with GitHub Actions

## Step 1: Login to AWS Console

Go to:

[AWS Console](https://aws.amazon.com/console/?utm_source=chatgpt.com)

---

# Step 2: Create IAM User for Deployment

Go to:

* IAM → Users → Create User

## Required Permissions

Attach these policies:

* `AmazonEC2FullAccess`
* `AmazonEC2ContainerRegistryFullAccess`
* `AmazonS3FullAccess`

## Purpose of These Permissions

### EC2 Access

Used to manage virtual machines.

### ECR Access

Used to store Docker images in AWS Elastic Container Registry.

---

# CI/CD Deployment Flow

1. Build Docker image from source code
2. Push Docker image to Amazon ECR
3. Launch EC2 instance
4. Pull Docker image from ECR inside EC2
5. Run Docker container on EC2

---

# Step 3: Create ECR Repository

Go to:

* Elastic Container Registry (ECR)
* Create Repository

Example Repository URI:

```bash
577124149610.dkr.ecr.us-east-1.amazonaws.com/at/ctr/repo
```

Save this URI for GitHub Secrets.

---

# Step 4: Launch EC2 Instance

Recommended Configuration:

* Ubuntu Server 22.04
* t2.medium or higher
* Minimum 20GB storage

Allow These Inbound Rules:

| Type       | Port |
| ---------- | ---- |
| SSH        | 22   |
| HTTP       | 80   |
| HTTPS      | 443  |
| Custom TCP | 8000 |

---

# Step 5: Install Docker on EC2

Connect to EC2:

```bash
ssh -i key.pem ubuntu@<EC2_PUBLIC_IP>
```

Run:

```bash
sudo apt-get update -y

sudo apt-get upgrade -y
```

Install Docker:

```bash
curl -fsSL https://get.docker.com -o get-docker.sh

sudo sh get-docker.sh
```

Add Ubuntu user to Docker group:

```bash
sudo usermod -aG docker ubuntu
```

Activate group changes:

```bash
newgrp docker
```

Verify Docker:

```bash
docker --version
```

---

# Step 6: Configure EC2 as GitHub Self-Hosted Runner

Go to your GitHub repository:

```text
Settings → Actions → Runners → New Self-hosted Runner
```

Choose:

* Linux
* x64

Run all commands provided by GitHub one-by-one on EC2.

Example:

```bash
mkdir actions-runner && cd actions-runner

curl -o actions-runner-linux-x64.tar.gz -L https://github.com/actions/runner/releases/download/v2.317.0/actions-runner-linux-x64-2.317.0.tar.gz

tar xzf ./actions-runner-linux-x64.tar.gz
```

Configure runner:

```bash
./config.sh --url https://github.com/<username>/<repo> --token <TOKEN>
```

Start runner:

```bash
./run.sh
```

For background service:

```bash
sudo ./svc.sh install

sudo ./svc.sh start
```

---

# Step 7: Configure GitHub Secrets

Go to:

```text
Repository → Settings → Secrets and variables → Actions
```

Add:

```bash
AWS_ACCESS_KEY_ID=

AWS_SECRET_ACCESS_KEY=

AWS_DEFAULT_REGION=us-east-1

AWS_ECR_LOGIN_URI=

ECR_REPOSITORY_NAME=
```

---