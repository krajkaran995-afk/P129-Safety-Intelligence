# P_129 Initial Data Audit

## Dataset overview
- Main dataset: **8,000 rows × 29 columns** (`safety_incidents_cleaned_final.csv`)
- NLP dataset: **8,000 rows × 5 columns** (`safety_incidents_cleaned.csv`)
- Main dataset date range: **2021-01-01 00:00:00 to 2026-08-31 00:00:00**
- Main dataset duplicate full rows: **0**
- NLP dataset duplicate full rows: **0**

## Relationship between files
- Same number of rows: **True**
- Shared fields match row-by-row: **True**
- Incident description sets match: **True**
- Treat the NLP file as a derived view if these checks confirm matching records.

## Missing values in main dataset
- `injury_type`: 1,135
- `body_part_affected`: 1,135
- `root_cause_description`: 41
- `contributing_factor_2`: 2,578

## Severity distribution
- Minor: 2,721
- Moderate: 2,700
- Serious: 1,639
- Severe: 777
- Critical: 163

## Risk-level distribution
- Medium: 3,323
- High: 2,109
- Low: 2,009
- Critical: 559

## Important cautions
- Matching descriptions and labels suggest the NLP file is a derived view, not an independent dataset.
- Matching content does not establish real-world data provenance.
- Preserve missing injury and body-part fields until their meaning is confirmed.
- Do not use risk_level or risk_score as input features when predicting risk_level.
- Do not use root_cause_category as an input when predicting root_cause_category.
- Evaluate classifiers on held-out data using per-class precision, recall, F1 and a confusion matrix.
