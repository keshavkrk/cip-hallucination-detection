# cip/src/classifier/train_model.py

import numpy as np
import joblib

from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix


# -------------------------------------------------
# Load Features
# -------------------------------------------------

X = np.load("../data/processed/X.npy")
y = np.load("../data/processed/y.npy")

print("Dataset shape:", X.shape)
print("Label distribution:", np.bincount(y))


# -------------------------------------------------
# Train / Test Split
# -------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)


# -------------------------------------------------
# Build Calibrated Linear SVM
# -------------------------------------------------

base_svm = LinearSVC(
    C=1.0,
    class_weight="balanced",
    random_state=42,
    max_iter=5000
)

calibrated_svm = CalibratedClassifierCV(
    estimator=base_svm,
    method="sigmoid",   # Platt scaling
    cv=5
)

model = Pipeline([
    ("scaler", StandardScaler()),
    ("calibrated_svm", calibrated_svm)
])


# -------------------------------------------------
# Train
# -------------------------------------------------

print("Training Calibrated Linear SVM (5-fold calibration)...")

model.fit(X_train, y_train)


# -------------------------------------------------
# Evaluate
# -------------------------------------------------

probs = model.predict_proba(X_test)[:, 1]
preds = (probs > 0.5).astype(int)

print("\nEvaluation Results")
print("-------------------------------------------------")
print("Accuracy:", (preds == y_test).mean())
print("ROC-AUC :", roc_auc_score(y_test, probs))

print("\nClassification Report:")
print(classification_report(y_test, preds))

print("Confusion Matrix:")
print(confusion_matrix(y_test, preds))


# -------------------------------------------------
# Save Model
# -------------------------------------------------

joblib.dump(model, "../data/processed/hallucination_model.pkl")

print("\nCalibrated model saved successfully.")