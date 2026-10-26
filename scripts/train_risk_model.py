
"""Train and evaluate a risk-level classifier for P_129."""

from pathlib import Path
import json
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.ensemble import RandomForestClassifier

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "safety_incidents_cleaned_final.csv"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "risk_level"

# Only use information that could reasonably be known at assessment time.
# Exclude risk_score, severity, root-cause labels, injuries and post-incident actions.
CATEGORICAL_FEATURES = [
    "department",
    "location",
    "shift",
    "employee_role",
    "experience_level",
    "equipment_type",
    "equipment_status",
    "incident_type",
    "hazard_type",
]

TEXT_FEATURE = "incident_description"

df = pd.read_csv(DATA_PATH, low_memory=False)
df.columns = [str(c).strip().lower() for c in df.columns]

required = set(CATEGORICAL_FEATURES + [TEXT_FEATURE, TARGET])
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns: {sorted(missing)}")

df = df[CATEGORICAL_FEATURES + [TEXT_FEATURE, TARGET]].copy()
df[TARGET] = df[TARGET].astype("string").str.strip()
df[TEXT_FEATURE] = df[TEXT_FEATURE].fillna("").astype(str).str.strip()

df = df.dropna(subset=[TARGET])
df = df[df[TARGET] != ""]

# Remove duplicate records before splitting.
df = df.drop_duplicates().reset_index(drop=True)

counts = df[TARGET].value_counts()
eligible = counts[counts >= 2].index
df = df[df[TARGET].isin(eligible)].reset_index(drop=True)

if df[TARGET].nunique() < 2:
    raise ValueError("At least two risk-level classes are required.")

if df[TARGET].value_counts().min() < 2:
    raise ValueError("Each risk-level class needs at least two records.")

X = df[CATEGORICAL_FEATURES + [TEXT_FEATURE]]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore")),
])

text_pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        max_features=20000,
        sublinear_tf=True,
    )),
])

preprocessor = ColumnTransformer([
    ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
    ("text", text_pipeline, TEXT_FEATURE),
])

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=250,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )),
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
    "task": "Predict risk_level from eligible incident features",
    "model": "OneHotEncoder + TF-IDF + RandomForestClassifier",
    "target": TARGET,
    "excluded_fields": [
        "risk_score",
        "severity_level",
        "root_cause_category",
        "root_cause_description",
        "injury_type",
        "body_part_affected",
        "medical_treatment_required",
        "immediate_action",
        "corrective_action",
        "preventive_action",
        "lost_time_days",
        "investigation_status",
        "incident_status",
    ],
    "records_used": int(len(df)),
    "training_records": int(len(X_train)),
    "test_records": int(len(X_test)),
    "accuracy": float(accuracy_score(y_test, predictions)),
    "classification_report": report,
    "class_labels": model.classes_.tolist(),
    "confusion_matrix": confusion_matrix(
        y_test,
        predictions,
        labels=model.classes_,
    ).tolist(),
    "limitations": [
        "Risk labels may have been generated from rules or fields unavailable at prediction time.",
        "This random split does not prove generalization to future incidents or other facilities.",
        "Predictions require review by a qualified safety professional.",
    ],
}

model_path = MODEL_DIR / "risk_level_pipeline.joblib"
report_path = REPORT_DIR / "risk_model_evaluation.json"

joblib.dump(model, model_path)
report_path.write_text(
    json.dumps(results, indent=2, ensure_ascii=False),
    encoding="utf-8",
)

print("Risk-level model training complete.")
print(f"Records used: {len(df)}")
print(f"Training records: {len(X_train)}")
print(f"Test records: {len(X_test)}")
print(f"Accuracy: {results['accuracy']:.3f}")
print(f"Macro F1: {report['macro avg']['f1-score']:.3f}")
print(f"Model saved: {model_path}")
print(f"Evaluation report saved: {report_path}")
print("\nPer-class evaluation:")
print(classification_report(y_test, predictions, zero_division=0))
