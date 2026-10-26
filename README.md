# HCL P_129 — Safety Incident Prediction and Root Cause Analysis using NLP

A Streamlit-first safety intelligence prototype using the two supplied CSV datasets, reproducible data preparation, and experimental trained models for root-cause classification and risk-level prediction.

## Features in this starter

- Dashboard with dataset-based KPIs and Plotly charts
- Incident Explorer with search, filters and detailed records
- Report Upload supporting PDF, CSV and Excel files
- Select previous incidents in Report Upload, Root Cause Analysis and Risk Assessment
- Model-generated root-cause and risk-level predictions for selected incidents
- Preliminary safety-professional review prompts based on model and incident labels
- Evaluation reports for the trained root-cause and risk-level models
- Filtered PDF safety reports with severity and risk charts
- PDF text extraction for text-based PDFs
- CSV/Excel preview
- Analytics and filtered CSV export
- Data Quality page with missing-value review and cross-file consistency check
- Data audit script and reproducible data-refinement report
- Streamlit-native UI; no custom HTML/CSS frontend

## Environment setup (Windows PowerShell)

Activate the Conda environment you already created:

```powershell
conda activate p129
```

If PowerShell says Conda is not initialized in a new terminal, run:

```powershell
(& "$env:USERPROFILE\miniconda3\shell\condabin\conda-hook.ps1")
conda activate p129
```

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Run the app from this folder:

```powershell
streamlit run app.py
```

## Data audit and refinement

Run the audit first:

```powershell
python scripts/audit_data.py
```

This writes `reports/initial_data_audit.md` and `reports/initial_data_audit.json`. The supplied files have been checked: both contain 8,000 rows, and the shared description/type/risk/root-cause fields match row-by-row. Treat the smaller NLP CSV as a derived view of the main CSV, not as extra independent incidents.

Then create the refined copy:

```powershell
python scripts/refine_data.py
```

## Data layout

- `data/raw/safety_incidents_cleaned_final.csv`: primary structured incident dataset
- `data/raw/safety_incidents_cleaned.csv`: NLP-focused dataset
- `data/processed/`: generated refined CSV and quality report

Raw files are retained unchanged. Run:

```powershell
python scripts/refine_data.py
```

This creates `data/processed/safety_incidents_refined.csv` and `data/processed/data_quality_report.json`.

## Important data/model notes

1. Confirm the source/provenance of the datasets before describing them as verified industrial incident records.
2. The two CSVs both contain 8,000 rows, but that alone does not prove the rows match. Verify record-level correspondence before joining them.
3. Missing injury/body-part fields may be not applicable; don't drop all incomplete rows automatically.
4. Avoid target leakage. If predicting `risk_level`, do not use `risk_level` or a score derived from it as an input feature. If predicting `root_cause_category`, don't include the target or text that directly repeats it without documenting the experimental design.
5. Root Cause Analysis and Risk Assessment can generate model predictions for selected incidents. These experimental predictions are not verified findings and require qualified human review.
6. PDF text extraction works for text-based PDFs. Scanned PDFs require OCR, which is not configured.
7. MySQL, authentication, further model validation and production deployment are future tasks. PDF reports and review prompts summarize selected data; they do not establish verified causes or approved safety procedures.

## Suggested next implementation milestones

1. Data audit and source verification
2. Establish a defensible train/validation/test split
3. Review and validate the existing root-cause and risk classifiers using class imbalance, per-class precision/recall/F1 and confusion matrices
4. Investigate whether the model labels and evaluation splits support reliable use on future incidents
6. Add MySQL after data contracts and analytical modules stabilize
7. Add tests, logging, error handling and final documentation
