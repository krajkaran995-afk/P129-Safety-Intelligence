
"""Train and evaluate an NLP root-cause classifier for P_129."""

from pathlib import Path
import json
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "safety_incidents_cleaned.csv"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

TEXT_COL = "incident_description"
TARGET_COL = "root_cause_category"

df = pd.read_csv(DATA_PATH, low_memory=False)
df.columns = [str(c).strip().lower() for c in df.columns]

required = {TEXT_COL, TARGET_COL}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns: {sorted(missing)}")

df = df[[TEXT_COL, TARGET_COL]].dropna()
df[TEXT_COL] = df[TEXT_COL].astype(str).str.strip()
df[TARGET_COL] = df[TARGET_COL].astype(str).str.strip()
df = df[(df[TEXT_COL] != "") & (df[TARGET_COL] != "")]

# Avoid putting duplicate narratives in both training and test sets.
df = df.drop_duplicates(subset=[TEXT_COL]).reset_index(drop=True)

class_counts = df[TARGET_COL].value_counts()
keep_classes = class_counts[class_counts >= 2].index
df = df[df[TARGET_COL].isin(keep_classes)].reset_index(drop=True)

if df[TARGET_COL].nunique() < 2:
    raise ValueError("At least two root-cause categories are required.")

if df[TARGET_COL].value_counts().min() < 2:
    raise ValueError("Each category needs at least two examples.")

X_train, X_test, y_train, y_test = train_test_split(
    df[TEXT_COL],
    df[TARGET_COL],
    test_size=0.2,
    random_state=42,
    stratify=df[TARGET_COL],
)

model = Pipeline([
    ("tfidf", TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        max_features=30000,
        sublinear_tf=True,
    )),
    ("classifier", LinearSVC(class_weight="balanced")),
])

model.fit(X_train, y_train)
predictions = model.predict(X_test)

report = classification_report(
    y_test,
    predictions,
    output_dict=True,
    zero_division=0,
)

results = {
    "task": "Predict root_cause_category from incident_description",
    "model": "TF-IDF + LinearSVC",
    "records_used": int(len(df)),
    "training_records": int(len(X_train)),
    "test_records": int(len(X_test)),
    "number_of_categories": int(df[TARGET_COL].nunique()),
    "accuracy": float(accuracy_score(y_test, predictions)),
    "classification_report": report,
    "confusion_matrix": confusion_matrix(
        y_test,
        predictions,
        labels=model.classes_,
    ).tolist(),
    "class_labels": model.classes_.tolist(),
    "limitations": [
        "Performance reflects this dataset and is not proof of real-world accuracy.",
        "If descriptions are synthetic or templated, results may not generalize to actual incidents.",
        "Predictions should be reviewed by a qualified safety professional.",
    ],
}

model_path = MODEL_DIR / "root_cause_pipeline.joblib"
report_path = REPORT_DIR / "root_cause_model_evaluation.json"

joblib.dump(model, model_path)
report_path.write_text(
    json.dumps(results, indent=2, ensure_ascii=False),
    encoding="utf-8",
)

print("Root-cause model trained successfully.")
print(f"Records used: {len(df)}")
print(f"Training records: {len(X_train)}")
print(f"Test records: {len(X_test)}")
print(f"Accuracy: {results['accuracy']:.3f}")
print(f"Model saved to: {model_path}")
print(f"Evaluation report saved to: {report_path}")
print("\nPer-class evaluation:")
print(classification_report(y_test, predictions, zero_division=0))
