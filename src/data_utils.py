from pathlib import Path
from io import BytesIO
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MAIN_PATH = ROOT / "data" / "raw" / "safety_incidents_cleaned_final.csv"
NLP_PATH = ROOT / "data" / "raw" / "safety_incidents_cleaned.csv"

def normalize_text(value):
    if pd.isna(value):
        return ""
    return " ".join(str(value).strip().split())

def load_main_data():
    if not MAIN_PATH.exists():
        return pd.DataFrame(), f"Main dataset not found at {MAIN_PATH}"
    df = pd.read_csv(MAIN_PATH, low_memory=False)
    df.columns = [str(c).strip().lower() for c in df.columns]
    if "incident_date" in df.columns:
        df["incident_date"] = pd.to_datetime(df["incident_date"], errors="coerce")
    if "incident_time" in df.columns:
        # Source uses a date-time-like string with a placeholder date; extract clock time.
        parsed = pd.to_datetime(df["incident_time"], errors="coerce")
        df["incident_time"] = parsed.dt.strftime("%H:%M").fillna(df["incident_time"].astype(str))
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].map(lambda x: x.strip() if isinstance(x, str) else x)
    if "incident_date" in df.columns:
        df["_year"] = df["incident_date"].dt.year
        df["_month"] = df["incident_date"].dt.to_period("M").dt.to_timestamp()
    note = f"Loaded {len(df):,} records from the supplied main CSV. Missing values are preserved where they may mean 'not recorded' or 'not applicable'."
    return df, note

def load_nlp_data():
    if not NLP_PATH.exists():
        return pd.DataFrame(), "NLP dataset not found."
    df = pd.read_csv(NLP_PATH, low_memory=False)
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df, f"NLP dataset available: {len(df):,} rows and {len(df.columns)} columns. Row-level matching to the main dataset has not yet been verified."

def read_uploaded_report(filename, content):
    ext = Path(filename).suffix.lower()
    if ext != ".pdf":
        raise ValueError("Only PDF files can be passed to the text extraction function.")
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("Install pypdf with: pip install pypdf") from exc
    reader = PdfReader(BytesIO(content))
    chunks = []
    for page in reader.pages:
        chunks.append(page.extract_text() or "")
    return "\n\n".join(chunks)
