# cip/src/classifier/train_model.py

import numpy as np
import joblib
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix


X = np.load("../data/processed/X.npy")
y = np.load("../data/processed/y.npy")

print("Splitting dataset...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

print("Training Linear SVM (770-d)...")

model = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(
        kernel="linear",
        C=1.0,
        probability=True,
        class_weight="balanced",
        random_state=42
    ))
])

model.fit(X_train, y_train)

preds = model.predict(X_test)
probs = model.predict_proba(X_test)[:, 1]

print("\nClassification Report:")
print(classification_report(y_test, preds))

print("ROC-AUC:", roc_auc_score(y_test, probs))
print("Confusion Matrix:")
print(confusion_matrix(y_test, preds))

joblib.dump(model, "../data/processed/hallucination_model.pkl")

print("\nModel saved successfully.")
