import pandas as pd
import joblib
import time

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer


# ==========================================
# 1. LOAD DATA
# ==========================================

df = pd.read_csv(
    "data/hospital_readmission_dashboard.csv"
)

print("Dataset Shape:", df.shape)


# ==========================================
# 2. FEATURES AND TARGET
# ==========================================

X = df.drop(columns=["readmission_30"])
y = df["readmission_30"]


# ==========================================
# 3. TRAIN / TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# ==========================================
# 4. IDENTIFY COLUMN TYPES
# ==========================================
categorical_columns = X.select_dtypes(
    include=["object", "string"]
).columns

numeric_columns = X.select_dtypes(
    exclude=["object", "string"]
).columns

print("\nCategorical Features:", len(categorical_columns))
print("Numerical Features:", len(numeric_columns))


# ==========================================
# 5. PREPROCESSING
# ==========================================

numeric_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        )
    ]
)


categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ]
)


preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            numeric_pipeline,
            numeric_columns
        ),
        (
            "cat",
            categorical_pipeline,
            categorical_columns
        )
    ]
)


# ==========================================
# 6. RANDOM FOREST MODEL
# ==========================================

rf_classifier = RandomForestClassifier(
    n_estimators=100,
    max_depth=12,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)


rf_model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "classifier",
            rf_classifier
        )
    ]
)


# ==========================================
# 7. TRAIN RANDOM FOREST
# ==========================================

print("\nTraining Random Forest...")

start_time = time.time()

rf_model.fit(
    X_train,
    y_train
)

rf_training_time = time.time() - start_time

print(
    f"Random Forest Training Time: "
    f"{rf_training_time:.2f} seconds"
)


# ==========================================
# 8. RANDOM FOREST PREDICTIONS
# ==========================================

rf_pred = rf_model.predict(X_test)

rf_probability = rf_model.predict_proba(
    X_test
)[:, 1]


# ==========================================
# 9. RANDOM FOREST METRICS
# ==========================================

rf_accuracy = accuracy_score(
    y_test,
    rf_pred
)

rf_precision = precision_score(
    y_test,
    rf_pred,
    zero_division=0
)

rf_recall = recall_score(
    y_test,
    rf_pred,
    zero_division=0
)

rf_f1 = f1_score(
    y_test,
    rf_pred,
    zero_division=0
)

rf_auc = roc_auc_score(
    y_test,
    rf_probability
)


# ==========================================
# 10. LOAD XGBOOST MODEL
# ==========================================

print("\nLoading XGBoost model...")

xgb_model = joblib.load(
    "models/hospital_readmission_model.pkl"
)


# ==========================================
# 11. XGBOOST PREDICTIONS
# ==========================================

start_time = time.time()

xgb_pred = xgb_model.predict(X_test)

xgb_probability = xgb_model.predict_proba(
    X_test
)[:, 1]

xgb_prediction_time = time.time() - start_time


# ==========================================
# 12. XGBOOST METRICS
# ==========================================

xgb_accuracy = accuracy_score(
    y_test,
    xgb_pred
)

xgb_precision = precision_score(
    y_test,
    xgb_pred,
    zero_division=0
)

xgb_recall = recall_score(
    y_test,
    xgb_pred,
    zero_division=0
)

xgb_f1 = f1_score(
    y_test,
    xgb_pred,
    zero_division=0
)

xgb_auc = roc_auc_score(
    y_test,
    xgb_probability
)


# ==========================================
# 13. COMPARISON TABLE
# ==========================================

comparison = pd.DataFrame({

    "Metric": [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "ROC-AUC"
    ],

    "XGBoost": [
        xgb_accuracy,
        xgb_precision,
        xgb_recall,
        xgb_f1,
        xgb_auc
    ],

    "Random Forest": [
        rf_accuracy,
        rf_precision,
        rf_recall,
        rf_f1,
        rf_auc
    ]
})


# ==========================================
# 14. DISPLAY RESULTS
# ==========================================

print("\n==========================================")
print("XGBOOST vs RANDOM FOREST")
print("==========================================")

print(
    comparison.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ==========================================
# 15. SAVE RESULTS
# ==========================================

comparison.to_csv(
    "model_comparison.csv",
    index=False
)

print("\n==========================================")
print("COMPARISON COMPLETED")
print("==========================================")

print(
    "Saved: model_comparison.csv"
)

# ==========================================
# 16. MODEL COMPARISON GRAPH
# ==========================================

import matplotlib.pyplot as plt
import numpy as np

metrics = comparison["Metric"]

xgboost_values = comparison["XGBoost"]
random_forest_values = comparison["Random Forest"]

x = np.arange(len(metrics))
width = 0.35

plt.figure(figsize=(10, 6))

plt.bar(
    x - width / 2,
    xgboost_values,
    width,
    label="XGBoost"
)

plt.bar(
    x + width / 2,
    random_forest_values,
    width,
    label="Random Forest"
)

plt.xlabel("Evaluation Metrics")
plt.ylabel("Score")
plt.title("XGBoost vs Random Forest Performance")

plt.xticks(
    x,
    metrics
)

plt.ylim(0, 1)

plt.legend()

plt.tight_layout()

plt.savefig(
    "Figure_9_Model_Comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print(
    "\nSaved: Figure_9_Model_Comparison.png"
)