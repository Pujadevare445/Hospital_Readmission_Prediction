# ==========================================
# HOSPITAL READMISSION PREDICTION
# Random Forest + XGBoost
# ==========================================

import kagglehub
import os
import pandas as pd
import numpy as np


# ==========================================
# STEP 1: DOWNLOAD DATASET
# ==========================================

path = kagglehub.dataset_download("brandao/diabetes")

print("Dataset path:")
print(path)


# ==========================================
# STEP 2: CHECK DATASET FILES
# ==========================================

print("\nFiles inside dataset:")
print(os.listdir(path))


# ==========================================
# STEP 3: LOAD DATASET
# ==========================================

file_path = os.path.join(path, "diabetic_data.csv")

df = pd.read_csv(file_path)

print("\nDataset loaded successfully!")
print("Shape:", df.shape)


# ==========================================
# STEP 4: FIRST 5 ROWS
# ==========================================

print("\nFirst 5 rows:")
print(df.head())


# ==========================================
# STEP 5: CHECK ORIGINAL TARGET
# ==========================================

print("\nOriginal Readmission Distribution:")
print(df["readmitted"].value_counts())


# ==========================================
# STEP 6: CREATE BINARY TARGET
# ==========================================

df["readmission_30"] = df["readmitted"].apply(
    lambda x: 1 if x == "<30" else 0
)

print("\nNew Target Distribution:")
print(df["readmission_30"].value_counts())


print("\nTarget Percentage:")
print(
    df["readmission_30"]
    .value_counts(normalize=True) * 100
)


# ==========================================
# STEP 7: REMOVE ID + ORIGINAL TARGET
# ==========================================

df = df.drop(
    columns=[
        "encounter_id",
        "patient_nbr",
        "readmitted"
    ]
)

print("\nShape after removing ID columns:")
print(df.shape)


# ==========================================
# STEP 8: REPLACE ? WITH NaN
# ==========================================

df = df.replace("?", np.nan)

print("\nMissing Values:")
print(
    df.isnull()
    .sum()
    .sort_values(ascending=False)
    .head(20)
)


# ==========================================
# STEP 9: MISSING VALUE PERCENTAGE
# ==========================================

missing_percentage = (
    df.isnull()
    .mean() * 100
).sort_values(ascending=False)

print("\nMissing Value Percentage:")
print(
    missing_percentage[
        missing_percentage > 0
    ]
)


# ==========================================
# STEP 10: DROP HIGH MISSING COLUMNS
# ==========================================

drop_columns = [
    "weight",
    "payer_code",
    "medical_specialty"
]

df = df.drop(columns=drop_columns)

print("\nShape after dropping high-missing columns:")
print(df.shape)


# ==========================================
# STEP 11: CHECK REMAINING MISSING VALUES
# ==========================================

print("\nRemaining Missing Values:")

remaining_missing = (
    df.isnull()
    .sum()
    .sort_values(ascending=False)
)

print(
    remaining_missing[
        remaining_missing > 0
    ]
)


# ==========================================
# STEP 12: CHECK TARGET AGAIN
# ==========================================

print("\nFinal Target Distribution:")

print(
    df["readmission_30"]
    .value_counts()
)


print("\nFinal Target Percentage:")

print(
    df["readmission_30"]
    .value_counts(normalize=True) * 100
)


# ==========================================
# STEP 13: FINAL DATASET SHAPE
# ==========================================

print("\nFinal Dataset Shape:")
print(df.shape)


# ==============================
# STEP 2: EXPLORATORY DATA ANALYSIS
# ==============================

import matplotlib.pyplot as plt
import seaborn as sns

# Readmission Distribution
plt.figure(figsize=(6, 4))

sns.countplot(
    data=df,
    x="readmission_30"
)

plt.title("30-Day Readmission Distribution")
plt.xlabel("Readmission within 30 Days")
plt.ylabel("Number of Patients")

plt.show()



# ==========================================
# STEP 2.2: AGE-WISE READMISSION ANALYSIS
# ==========================================

age_readmission = pd.crosstab(
    df["age"],
    df["readmission_30"]
)

print("\nAge-wise Readmission Count:")
print(age_readmission)

# Plot
age_readmission.plot(
    kind="bar",
    figsize=(10, 5)
)

plt.title("Age-wise 30-Day Readmission")
plt.xlabel("Age Group")
plt.ylabel("Number of Patients")
plt.xticks(rotation=45)
plt.legend(["No Readmission", "Readmitted <30 Days"])

plt.tight_layout()
plt.show()



# ==========================================
# STEP 2.3: HOSPITAL STAY VS READMISSION
# ==========================================

stay_readmission = pd.crosstab(
    df["time_in_hospital"],
    df["readmission_30"]
)

print("\nHospital Stay vs Readmission:")
print(stay_readmission)

# Plot
stay_readmission.plot(
    kind="bar",
    figsize=(10, 5)
)

plt.title("Hospital Stay vs 30-Day Readmission")
plt.xlabel("Days in Hospital")
plt.ylabel("Number of Patients")
plt.legend(["No Readmission", "Readmitted <30 Days"])

plt.tight_layout()
plt.show()


# ==========================================
# STEP 2.4: AGE-WISE READMISSION PERCENTAGE
# ==========================================

age_readmission_percentage = pd.crosstab(
    df["age"],
    df["readmission_30"],
    normalize="index"
) * 100

print("\nAge-wise Readmission Percentage:")
print(age_readmission_percentage)

# Plot readmission percentage
plt.figure(figsize=(10, 5))

age_readmission_percentage[1].plot(
    kind="bar"
)

plt.title("30-Day Readmission Rate by Age Group")
plt.xlabel("Age Group")
plt.ylabel("Readmission Rate (%)")
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()


# ==========================================
# STEP 2.5: HOSPITAL STAY-WISE READMISSION RATE
# ==========================================

stay_readmission_percentage = pd.crosstab(
    df["time_in_hospital"],
    df["readmission_30"],
    normalize="index"
) * 100

print("\nHospital Stay-wise Readmission Percentage:")
print(stay_readmission_percentage)

# Plot readmission rate
plt.figure(figsize=(10, 5))

stay_readmission_percentage[1].plot(
    kind="bar"
)

plt.title("30-Day Readmission Rate by Hospital Stay")
plt.xlabel("Days in Hospital")
plt.ylabel("Readmission Rate (%)")
plt.xticks(rotation=0)

plt.tight_layout()
plt.show()


# ==========================================
# STEP 2.6: GENDER-WISE READMISSION ANALYSIS
# ==========================================

gender_readmission = pd.crosstab(
    df["gender"],
    df["readmission_30"]
)

print("\nGender-wise Readmission Count:")
print(gender_readmission)

# ==========================================
# GENDER-WISE READMISSION RATE
# ==========================================

gender_readmission_percentage = pd.crosstab(
    df["gender"],
    df["readmission_30"],
    normalize="index"
) * 100

print("\nGender-wise Readmission Percentage:")
print(gender_readmission_percentage)

# ==========================================
# PLOT
# ==========================================

plt.figure(figsize=(7, 5))

gender_readmission_percentage[1].plot(
    kind="bar"
)

plt.title("30-Day Readmission Rate by Gender")
plt.xlabel("Gender")
plt.ylabel("Readmission Rate (%)")
plt.xticks(rotation=0)

plt.tight_layout()
plt.show()

# ==========================================
# STEP 3.1: DIAGNOSIS FEATURE ENGINEERING
# ==========================================

def categorize_diagnosis(code):

    if pd.isna(code):
        return "Unknown"

    code = str(code)

    try:
        code_num = float(code)
    except ValueError:
        return "Other"

    if 390 <= code_num <= 459 or code_num == 785:
        return "Circulatory"

    elif 460 <= code_num <= 519 or code_num == 786:
        return "Respiratory"

    elif 520 <= code_num <= 579 or code_num == 787:
        return "Digestive"

    elif 250 <= code_num < 251:
        return "Diabetes"

    elif 800 <= code_num <= 999:
        return "Injury"

    elif 140 <= code_num <= 239:
        return "Neoplasms"

    elif 580 <= code_num <= 629:
        return "Genitourinary"

    elif 1 <= code_num <= 139:
        return "Infectious"

    else:
        return "Other"


# Apply function to diagnosis columns

df["diag_1_category"] = df["diag_1"].apply(categorize_diagnosis)
df["diag_2_category"] = df["diag_2"].apply(categorize_diagnosis)
df["diag_3_category"] = df["diag_3"].apply(categorize_diagnosis)


print("\nDiagnosis Categories Created Successfully!")

print("\nPrimary Diagnosis Categories:")
print(df["diag_1_category"].value_counts())

print("\nSecondary Diagnosis Categories:")
print(df["diag_2_category"].value_counts())

print("\nTertiary Diagnosis Categories:")
print(df["diag_3_category"].value_counts())



# ==========================================
# STEP 3.2: PREVIOUS VISIT FEATURE ENGINEERING
# ==========================================

df["total_previous_visits"] = (
    df["number_outpatient"]
    + df["number_emergency"]
    + df["number_inpatient"]
)

print("\nTotal Previous Visits Feature Created Successfully!")

print("\nTotal Previous Visits Statistics:")
print(
    df["total_previous_visits"].describe()
)

print("\nSample of Previous Visit Features:")
print(
    df[
        [
            "number_outpatient",
            "number_emergency",
            "number_inpatient",
            "total_previous_visits"
        ]
    ].head(10)
)


# ==========================================
# STEP 3.3: MEDICATION FEATURE ENGINEERING
# ==========================================

medication_columns = [
    "metformin",
    "repaglinide",
    "nateglinide",
    "chlorpropamide",
    "glimepiride",
    "acetohexamide",
    "glipizide",
    "glyburide",
    "tolbutamide",
    "pioglitazone",
    "rosiglitazone",
    "acarbose",
    "miglitol",
    "troglitazone",
    "tolazamide",
    "examide",
    "citoglipton",
    "insulin",
    "glyburide-metformin",
    "glipizide-metformin",
    "glimepiride-pioglitazone",
    "metformin-rosiglitazone",
    "metformin-pioglitazone"
]

df["num_medications_used"] = (
    df[medication_columns] != "No"
).sum(axis=1)

print("\nNumber of Medications Feature Created Successfully!")

print("\nMedication Usage Statistics:")
print(df["num_medications_used"].describe())

print("\nSample Medication Features:")
print(
    df[
        [
            "num_medications_used",
            "diabetesMed",
            "change"
        ]
    ].head(10)
)


# ==========================================
# STEP 3.4: MEDICATION CHANGE FEATURE
# ==========================================

df["medication_changed"] = df["change"].apply(
    lambda x: 1 if x == "Ch" else 0
)

print("\nMedication Change Feature Created Successfully!")

print("\nMedication Change Distribution:")
print(df["medication_changed"].value_counts())

print("\nMedication Change Percentage:")
print(
    df["medication_changed"]
    .value_counts(normalize=True) * 100
)

print("\nSample Medication Change Features:")
print(
    df[
        [
            "change",
            "medication_changed",
            "diabetesMed"
        ]
    ].head(10)
)

# ==========================================
# STEP 3.5: FINAL FEATURE SELECTION & CLEANUP
# ==========================================

# Remove raw diagnosis codes
# We already created diagnosis category features
df = df.drop(
    columns=[
        "diag_1",
        "diag_2",
        "diag_3"
    ]
)

# Remove original medication change column
# Numeric feature "medication_changed" is already created
df = df.drop(columns=["change"])

print("\nFinal Feature Cleanup Completed Successfully!")

print("\nFinal Dataset Shape:")
print(df.shape)

print("\nFinal Columns:")
print(df.columns.tolist())

print("\nRemaining Missing Values:")
remaining_missing = df.isnull().sum().sort_values(ascending=False)
print(remaining_missing[remaining_missing > 0])

# ==========================================
# SAVE DATASET FOR STREAMLIT DASHBOARD
# ==========================================

os.makedirs("data", exist_ok=True)

df.to_csv(
    "data/hospital_readmission_dashboard.csv",
    index=False
)

print("\nDashboard Dataset Saved Successfully!")
print("File: data/hospital_readmission_dashboard.csv")

# ==========================================
# STEP 4: TRAIN-TEST SPLIT
# ==========================================

from sklearn.model_selection import train_test_split

# Separate features and target
X = df.drop(columns=["readmission_30"])
y = df["readmission_30"]

# Split dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTrain-Test Split Completed Successfully!")

print("\nTraining Data Shape:")
print(X_train.shape)

print("\nTesting Data Shape:")
print(X_test.shape)

print("\nTraining Target Distribution:")
print(y_train.value_counts())

print("\nTesting Target Distribution:")
print(y_test.value_counts())

print("\nTraining Target Percentage:")
print(y_train.value_counts(normalize=True) * 100)

print("\nTesting Target Percentage:")
print(y_test.value_counts(normalize=True) * 100)

# ==========================================
# STEP 5: PREPROCESSING PIPELINE
# ==========================================

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

# Numeric columns
numeric_features = X_train.select_dtypes(
    include=["int64", "float64"]
).columns

# Categorical columns
categorical_features = X_train.select_dtypes(
    include=["object"]
).columns

# Numeric preprocessing
numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

# Categorical preprocessing
categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

# Combine preprocessing
preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

print("\nPreprocessing Pipeline Created Successfully!")
print("Numeric Features:", len(numeric_features))
print("Categorical Features:", len(categorical_features))

# ==========================================
# STEP 6: RANDOM FOREST MODEL
# ==========================================

from sklearn.ensemble import RandomForestClassifier

rf_model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    ))
])

rf_model.fit(X_train, y_train)

print("\nRandom Forest Model Trained Successfully!")

# ==========================================
# STEP 7: RANDOM FOREST EVALUATION
# ==========================================

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

y_pred_rf = rf_model.predict(X_test)

print("\nRandom Forest Evaluation:")

print("Accuracy :", accuracy_score(y_test, y_pred_rf))
print("Precision:", precision_score(y_test, y_pred_rf))
print("Recall   :", recall_score(y_test, y_pred_rf))
print("F1 Score :", f1_score(y_test, y_pred_rf))

print("\nClassification Report:")
print(classification_report(y_test, y_pred_rf))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_rf))


# ==========================================
# STEP 8: XGBOOST MODEL
# ==========================================

from xgboost import XGBClassifier

xgb_model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
        eval_metric="logloss",
        n_jobs=-1
    ))
])

xgb_model.fit(X_train, y_train)

print("\nXGBoost Model Trained Successfully!")

# ==========================================
# STEP 9: XGBOOST EVALUATION
# ==========================================

y_pred_xgb = xgb_model.predict(X_test)

print("\nXGBoost Evaluation:")

print("Accuracy :", accuracy_score(y_test, y_pred_xgb))
print("Precision:", precision_score(y_test, y_pred_xgb))
print("Recall   :", recall_score(y_test, y_pred_xgb))
print("F1 Score :", f1_score(y_test, y_pred_xgb))

print("\nClassification Report:")
print(classification_report(y_test, y_pred_xgb))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_xgb))

# ==========================================
# STEP 10: XGBOOST WITH CLASS IMBALANCE
# ==========================================

negative = y_train.value_counts()[0]
positive = y_train.value_counts()[1]

scale_pos_weight = negative / positive

print("\nScale Pos Weight:", scale_pos_weight)

xgb_balanced = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric="logloss",
        n_jobs=-1
    ))
])

xgb_balanced.fit(X_train, y_train)

print("\nBalanced XGBoost Model Trained Successfully!")

# ==========================================
# STEP 11: BALANCED XGBOOST EVALUATION
# ==========================================

y_pred_balanced = xgb_balanced.predict(X_test)

print("\nBalanced XGBoost Evaluation:")

print("Accuracy :", accuracy_score(y_test, y_pred_balanced))
print("Precision:", precision_score(y_test, y_pred_balanced))
print("Recall   :", recall_score(y_test, y_pred_balanced))
print("F1 Score :", f1_score(y_test, y_pred_balanced))

print("\nClassification Report:")
print(classification_report(y_test, y_pred_balanced))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_balanced))

# ==========================================
# STEP 12: ROC-AUC AND PR-AUC
# ==========================================

from sklearn.metrics import roc_auc_score, average_precision_score

# Probability predictions
y_prob_rf = rf_model.predict_proba(X_test)[:, 1]
y_prob_xgb = xgb_model.predict_proba(X_test)[:, 1]
y_prob_balanced = xgb_balanced.predict_proba(X_test)[:, 1]

print("\nROC-AUC Scores:")
print("Random Forest      :", roc_auc_score(y_test, y_prob_rf))
print("XGBoost            :", roc_auc_score(y_test, y_prob_xgb))
print("Balanced XGBoost   :", roc_auc_score(y_test, y_prob_balanced))

print("\nPR-AUC Scores:")
print("Random Forest      :", average_precision_score(y_test, y_prob_rf))
print("XGBoost            :", average_precision_score(y_test, y_prob_xgb))
print("Balanced XGBoost   :", average_precision_score(y_test, y_prob_balanced))

# ==========================================
# STEP 13: THRESHOLD TUNING
# ==========================================

from sklearn.metrics import precision_recall_curve

precision, recall, thresholds = precision_recall_curve(
    y_test,
    y_prob_balanced
)

f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)

best_index = np.argmax(f1_scores)

best_threshold = thresholds[best_index]
best_f1 = f1_scores[best_index]

print("\nBest Threshold:", best_threshold)
print("Best F1 Score:", best_f1)

# ==========================================
# STEP 14: SAVE FINAL MODEL
# ==========================================

import joblib
import os

os.makedirs("models", exist_ok=True)

joblib.dump(
    xgb_balanced,
    "models/hospital_readmission_model.pkl"
)

print("\nFinal Model Saved Successfully!")
print("File: models/hospital_readmission_model.pkl")

# ==========================================
# STEP 15: CHECK SAVED MODEL
# ==========================================

loaded_model = joblib.load(
    "models/hospital_readmission_model.pkl"
)

print("\nSaved Model Loaded Successfully!")
print(type(loaded_model))