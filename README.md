# 🏥 Hospital Readmission Prediction

An end-to-end Machine Learning project that predicts whether a diabetic patient is likely to be readmitted to the hospital within 30 days.

## 🚀 Live Demo

https://qyco6zjfz6rqcuy7xgfokd.streamlit.app/

## 📌 Project Overview

Hospital readmission is an important healthcare problem. This project uses patient information and hospital encounter data to build a machine learning model for predicting 30-day readmission.

The project includes:

- Data preprocessing
- Exploratory Data Analysis
- Feature engineering
- Machine Learning model training
- XGBoost model
- Random Forest comparison
- Model evaluation
- Interactive Streamlit dashboard
- 30-day readmission prediction

## 📊 Dataset

Dataset contains:

- 101,766 patient records
- 47 processed features
- Patient demographic information
- Hospital stay information
- Previous hospital visits
- Medication information
- Diagnosis categories

## 🤖 Machine Learning

The main prediction model used in the project is:

**XGBoost Classifier**

The trained model is saved as:

`models/hospital_readmission_model.pkl`

### Model Performance

| Metric | XGBoost |
|---|---:|
| Accuracy | 65.62% |
| Precision | 18.52% |
| Recall | 61.21% |
| F1 Score | 28.43% |
| ROC-AUC | 68.38% |

## 🔍 Model Comparison

The project also compares XGBoost with Random Forest.

| Metric | XGBoost | Random Forest |
|---|---:|---:|
| Accuracy | 0.6562 | 0.6958 |
| Precision | 0.1852 | 0.1859 |
| Recall | 0.6121 | 0.5108 |
| F1 Score | 0.2843 | 0.2726 |
| ROC-AUC | 0.6838 | 0.6642 |

## 📈 Visualizations

The project contains visualizations for:

- Readmission rate by age group
- Readmission rate by hospital stay
- Readmission rate by gender
- Confusion matrix
- ROC curve
- Model comparison

## 💻 Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Joblib
- Plotly
- Streamlit

## 📂 Project Structure

```text
Hospital_Readmission_Prediction/
│
├── data/
│   └── hospital_readmission_dashboard.csv
│
├── models/
│   └── hospital_readmission_model.pkl
│
├── app.py
├── hospital_readmission.py
├── evaluate_model.py
├── model_comparison.py
├── model_visualization.py
├── check_model.py
├── model_comparison.csv
├── requirements.txt
│
├── Figure_1.png
├── Figure_2.png
├── Figure_3.png
├── Figure_4.png
├── Figure_5.png
├── Figure_6.png
├── Figure_7_Confusion_Matrix.png
├── Figure_8_ROC_Curve.png
└── Figure_9_Model_Comparison.png
