# Machine Learning Classification Project
https://ml-classification-project-idornugzkmkvdlbsby9xqk.streamlit.app/
## Project Overview

This project implements and compares multiple machine learning classification algorithms using real-world datasets.

The project includes:

1. Random Forest Classification
2. Logistic Regression
3. XGBoost Classification
4. Decision Tree Classification

The project also includes an interactive Streamlit web application for viewing the model results.

## Machine Learning Models

### 1. Random Forest

Dataset: Breast Cancer Wisconsin dataset

The Random Forest model is used to classify tumors as malignant or benign.

### 2. Logistic Regression

Dataset: Pima Indians Diabetes dataset

The Logistic Regression model predicts whether a patient has diabetes.

### 3. XGBoost

Dataset: Titanic dataset

The XGBoost model predicts whether a passenger survived the Titanic disaster.

### 4. Decision Tree

Dataset: Pima Indians Diabetes dataset

Two Decision Tree models are compared:

- Full-depth Decision Tree
- Restricted Decision Tree with maximum depth of 3

## Evaluation Metrics

The models are evaluated using:

- Accuracy
- Confusion Matrix
- Precision
- Recall
- F1 Score
- ROC-AUC
- Feature Importance where applicable

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Matplotlib
- Streamlit

## How to Run the Project

Install the required packages:

```bash
pip install -r requirements.txt
