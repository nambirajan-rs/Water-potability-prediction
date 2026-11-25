"""Train the water potability model used by the Flask app.

The whole preprocessing chain (missing-value imputation + scaling + model) is
saved as ONE scikit-learn Pipeline, so the web app applies exactly the same
transformations as training.

Usage:
    python train.py
"""
import pickle

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from features import FEATURES

df = pd.read_csv("data/water_potability.csv")
X, y = df[FEATURES], df["Potability"]

# Split first, then impute inside the pipeline: missing values are filled
# from training data only, never using the target or the test set.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scale", StandardScaler()),
    # Depth/leaf limits keep model.pkl small (~2-3 MB) with no loss in accuracy
    ("rf", RandomForestClassifier(n_estimators=150, max_depth=10, min_samples_leaf=5,
                                  class_weight="balanced", random_state=42, n_jobs=-1)),
])
model.fit(X_train, y_train)

pred = model.predict(X_test)
print(f"Test accuracy: {accuracy_score(y_test, pred):.3f}")
print(classification_report(y_test, pred, target_names=["not potable", "potable"]))

# Refit on all data for the saved model
model.fit(X, y)
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)
print("Saved model.pkl")
