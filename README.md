# 🍷 End-to-End Machine Learning: Wine Quality Prediction

This is a full-fledged end-to-end machine learning pipeline to predict wine quality based on physicochemical features. The project integrates everything from data processing and model training to containerized deployment using AWS services like EC2, ECS, ECR, and CI/CD with GitHub Actions.

---

## 📌 Features

- **Data Handling**: Automatic data validation, preprocessing, and feature engineering.
- **Model Training**: Uses Random Forest with tunable hyperparameters.
- **Web Interface**: Flask app for real-time wine quality prediction.
- **Cloud Integration**:
  - **Amazon S3** – Stores datasets and trained models.
  - **Amazon ECR** – Hosts Docker images.
  - **Amazon ECS (Fargate)** – Runs the containerized application.
  - **EC2** – Can be used for development or backend compute.
- **Deployment**:
  - **Docker** – Containerizes the app for portability.
  - **GitHub Actions** – CI/CD for automatic testing, Docker build, push to ECR, and deploy to ECS.

---

## 🏗️ Project Structure
End_to_end_ML_Wine_Quality_Prediction/
├── .github/workflows/ # CI/CD pipeline (GitHub Actions)
├── artifacts/ # Trained models, saved assets
├── config/ # Configuration files for training and schema
├── logs/ # Logging outputs
├── research/ # EDA notebooks
├── src/ # Source code (pipeline, model, etc.)
├── static/ # Static files for Flask
├── templates/ # HTML templates for the web UI
├── app.py # Flask application
├── main.py # Main training pipeline
├── Dockerfile # Docker build instructions
├── requirements.txt # Python dependencies
├── params.yaml # Model hyperparameters
├── schema.yaml # Input schema for validation

---

## 🔧 Tech Stack

- **Python 3.8+**
- **scikit-learn, pandas, numpy, seaborn, matplotlib**
- **Flask** – Web API and frontend
- **Docker** – Containerization
- **AWS S3** – Model/data storage
- **AWS ECR** – Docker image repository
- **AWS ECS (Fargate)** – App hosting
- **AWS EC2** – Compute instances for dev/test
- **GitHub Actions** – Automated CI/CD

---
