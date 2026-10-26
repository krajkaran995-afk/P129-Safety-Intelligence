
"""Audit the supplied P_129 datasets and save a readable report.

Run from the project root:
python scripts/audit_data.py
"""

from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "reports"
OUT.mkdir(parents=True, exist_ok=True)

main_path = RAW / "safety_incidents_cleaned_final.csv"
nlp_path = RAW / "safety_incidents_cleaned.csv"

main = pd.read_csv(main_path, low_memory=False)
nlp = pd.read_csv(nlp_path, low_memory=False)

main.columns = [str(c).strip().lower() for c in main.columns]
nlp.columns = [str(c).strip().lower() for c in nlp.columns]

shared = [
    "incident_description",
    "incident_type",
    "risk_level",
    "root_cause_category",
]

checks = {}

for col in shared:
    if col not in main.columns or col not in nlp.columns:
        checks[f"same_row_order_{col}"] = False
    else:
        checks[f"same_row_order_{col}"] = bool(
            main[col].fillna("").astype(str).equals(
                nlp[col].fillna("").astype(str)
            )
        )

checks["same_number_of_rows"] = len(main) == len(nlp)
checks["all_shared_fields_match_row_by_row"] = all(
    checks[f"same_row_order_{col}"] for col in shared
)
checks["description_sets_match"] = (
    "incident_description" in main.columns
    and "incident_description" in nlp.columns
    and set(main["incident_description"].fillna("").astype(str))
    == set(nlp["incident_description"].fillna("").astype(str))
)

def distribution(df, column, dropna=False):
    if column not in df.columns:
        return {}
    counts = df[column].value_counts(dropna=dropna)
    return {str(k): int(v) for k, v in counts.items()}

def missing_values(df):
    return {
        str(k): int(v)
        for k, v in df.isna().sum().items()
        if v
    }

main_dates = (
    pd.to_datetime(main["incident_date"], errors="coerce")
    if "incident_date" in main.columns
    else pd.Series(dtype="datetime64[ns]")
)
risk_scores = (
    pd.to_numeric(main["risk_score"], errors="coerce")
    if "risk_score" in main.columns
    else pd.Series(dtype="float64")
)

quality = {
    "main_dataset": {
        "file": main_path.name,
        "rows": int(len(main)),
        "columns": int(len(main.columns)),
        "duplicate_full_rows": int(main.duplicated().sum()),
        "missing_values_by_column": missing_values(main),
        "severity_distribution": distribution(
            main, "severity_level", dropna=False
        ),
        "risk_level_distribution": distribution(
            main, "risk_level", dropna=False
        ),
        "incident_type_top_10": distribution(
            main, "incident_type"
        ),
        "root_cause_top_10": distribution(
            main, "root_cause_category"
        ),
        "incident_date_min": str(main_dates.min()),
        "incident_date_max": str(main_dates.max()),
        "risk_score_min": (
            float(risk_scores.min()) if risk_scores.notna().any() else None
        ),
        "risk_score_max": (
            float(risk_scores.max()) if risk_scores.notna().any() else None
        ),
    },
    "nlp_dataset": {
        "file": nlp_path.name,
        "rows": int(len(nlp)),
        "columns": int(len(nlp.columns)),
        "duplicate_full_rows": int(nlp.duplicated().sum()),
        "missing_values_by_column": missing_values(nlp),
        "clean_description_unique_values": (
            int(nlp["clean_description"].nunique(dropna=True))
            if "clean_description" in nlp.columns
            else None
        ),
    },
    "cross_dataset_checks": checks,
    "interpretation_and_cautions": [
        "Matching descriptions and labels suggest the NLP file is a derived view, not an independent dataset.",
        "Matching content does not establish real-world data provenance.",
        "Preserve missing injury and body-part fields until their meaning is confirmed.",
        "Do not use risk_level or risk_score as input features when predicting risk_level.",
        "Do not use root_cause_category as an input when predicting root_cause_category.",
        "Evaluate classifiers on held-out data using per-class precision, recall, F1 and a confusion matrix.",
    ],
}

json_path = OUT / "initial_data_audit.json"
json_path.write_text(
    json.dumps(quality, indent=2, ensure_ascii=False),
    encoding="utf-8",
)

lines = [
    "# P_129 Initial Data Audit",
    "",
    "## Dataset overview",
    f"- Main dataset: **{len(main):,} rows × {len(main.columns)} columns** (`{main_path.name}`)",
    f"- NLP dataset: **{len(nlp):,} rows × {len(nlp.columns)} columns** (`{nlp_path.name}`)",
    f"- Main dataset date range: **{main_dates.min()} to {main_dates.max()}**",
    f"- Main dataset duplicate full rows: **{main.duplicated().sum():,}**",
    f"- NLP dataset duplicate full rows: **{nlp.duplicated().sum():,}**",
    "",
    "## Relationship between files",
    f"- Same number of rows: **{checks['same_number_of_rows']}**",
    f"- Shared fields match row-by-row: **{checks['all_shared_fields_match_row_by_row']}**",
    f"- Incident description sets match: **{checks['description_sets_match']}**",
    "- Treat the NLP file as a derived view if these checks confirm matching records.",
    "",
    "## Missing values in main dataset",
]

if quality["main_dataset"]["missing_values_by_column"]:
    lines.extend(
        f"- `{col}`: {count:,}"
        for col, count in quality["main_dataset"]["missing_values_by_column"].items()
    )
else:
    lines.append("- None detected")

lines += ["", "## Severity distribution"]
lines.extend(
    f"- {key}: {value:,}"
    for key, value in quality["main_dataset"]["severity_distribution"].items()
)

lines += ["", "## Risk-level distribution"]
lines.extend(
    f"- {key}: {value:,}"
    for key, value in quality["main_dataset"]["risk_level_distribution"].items()
)

lines += ["", "## Important cautions"]
lines.extend(
    f"- {item}" for item in quality["interpretation_and_cautions"]
)

md_path = OUT / "initial_data_audit.md"
md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

print(f"Audit complete: {md_path}")
print(f"JSON report: {json_path}")
print(
    "Cross-file shared-label row match:",
    checks["all_shared_fields_match_row_by_row"],
)
