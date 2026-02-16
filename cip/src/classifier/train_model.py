import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score


X = np.load("../data/processed/X.npy")
y = np.load("../data/processed/y.npy")

print("Training Logistic Regression...")

model = LogisticRegression(
    max_iter=1000,
    n_jobs=-1
)

model.fit(X, y)

preds = model.predict(X)
probs = model.predict_proba(X)[:, 1]

print("\nClassification Report:")
print(classification_report(y, preds))

print("ROC-AUC:", roc_auc_score(y, probs))

joblib.dump(model, "../data/processed/hallucination_model.pkl")

print("\nModel saved successfully.")
