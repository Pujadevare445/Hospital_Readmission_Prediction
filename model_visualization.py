import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc
)

# ==========================================
# 1. LOAD DATA
# ==========================================

df = pd.read_csv(
    "data/hospital_readmission_dashboard.csv"
)

X = df.drop(columns=["readmission_30"])
y = df["readmission_30"]


# ==========================================
# 2. TRAIN / TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ==========================================
# 3. LOAD SAVED MODEL
# ==========================================

model = joblib.load(
    "models/hospital_readmission_model.pkl"
)

print("Model loaded successfully!")


# ==========================================
# 4. PREDICTIONS
# ==========================================

y_pred = model.predict(X_test)

y_probability = model.predict_proba(X_test)[:, 1]


# ==========================================
# 5. CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\nConfusion Matrix:")
print(cm)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[
        "No Readmission",
        "Readmitted <30 Days"
    ]
)

fig, ax = plt.subplots(figsize=(7, 6))

disp.plot(
    ax=ax,
    cmap="Blues",
    values_format="d"
)

plt.title(
    "Confusion Matrix - Hospital Readmission Prediction"
)

plt.tight_layout()

plt.savefig(
    "Figure_7_Confusion_Matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ==========================================
# 6. ROC CURVE
# ==========================================

fpr, tpr, thresholds = roc_curve(
    y_test,
    y_probability
)

roc_auc = auc(
    fpr,
    tpr
)

print("\nROC-AUC:", round(roc_auc, 4))


plt.figure(figsize=(7, 6))

plt.plot(
    fpr,
    tpr,
    label=f"XGBoost (AUC = {roc_auc:.3f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title(
    "ROC Curve - Hospital Readmission Prediction"
)

plt.legend(
    loc="lower right"
)

plt.tight_layout()

plt.savefig(
    "Figure_8_ROC_Curve.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ==========================================
# 7. FINAL MESSAGE
# ==========================================

print("\n==========================================")
print("VISUALIZATION COMPLETED")
print("==========================================")

print(
    "Saved: Figure_7_Confusion_Matrix.png"
)

print(
    "Saved: Figure_8_ROC_Curve.png"
)