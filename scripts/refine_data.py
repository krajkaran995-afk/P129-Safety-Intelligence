"""Create a cleaned copy and a basic quality report. Original CSVs remain unchanged."""
from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
source = ROOT / "data" / "raw" / "safety_incidents_cleaned_final.csv"
out_dir = ROOT / "data" / "processed"
out_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(source, low_memory=False)
df.columns = [str(c).strip().lower() for c in df.columns]
for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].map(lambda x: x.strip() if isinstance(x, str) else x)

if "incident_date" in df:
    df["incident_date"] = pd.to_datetime(df["incident_date"], errors="coerce")
if "incident_time" in df:
    # Convert date-time-like values to HH:MM; preserve unparseable values for review.
    parsed = pd.to_datetime(df["incident_time"], errors="coerce")
    df["incident_time"] = parsed.dt.strftime("%H:%M").where(parsed.notna(), df["incident_time"].astype(str))

# Standardize common category whitespace while preserving source labels.
for col in ["severity_level", "risk_level", "incident_type", "root_cause_category", "investigation_status", "incident_status"]:
    if col in df:
        df[col] = df[col].map(lambda x: x.strip().title() if isinstance(x, str) else x)

# Do not delete rows simply because optional injury/body-part fields are missing.
duplicates = int(df.duplicated().sum())
missing = {str(k): int(v) for k, v in df.isna().sum().items()}
quality = {
    "source_file": str(source.relative_to(ROOT)),
    "rows": int(len(df)),
    "columns": int(len(df.columns)),
    "duplicate_rows_before_deduplication": duplicates,
    "missing_values_by_column": missing,
    "date_parse_failures": int(df["incident_date"].isna().sum()) if "incident_date" in df else None,
    "notes": [
        "Original raw data was not overwritten.",
        "Missing values were preserved because some fields may be not applicable.",
        "Review label consistency and data provenance before model training.",
        "Do not use risk_level/risk_score or root-cause label as model input when that field is the prediction target."
    ]
}
df.to_csv(out_dir / "safety_incidents_refined.csv", index=False)
(out_dir / "data_quality_report.json").write_text(json.dumps(quality, indent=2), encoding="utf-8")
print(f"Saved: {out_dir / 'safety_incidents_refined.csv'}")
print(f"Saved: {out_dir / 'data_quality_report.json'}")
print(f"Rows: {len(df):,}; columns: {len(df.columns)}; duplicate rows: {duplicates}")
