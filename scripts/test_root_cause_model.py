
from pathlib import Path
import joblib

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "root_cause_pipeline.joblib"

model = joblib.load(MODEL_PATH)

description = input(
    "Enter an incident description: "
).strip()

if not description:
    raise ValueError("Please enter a non-empty incident description.")

predicted_cause = model.predict([description])[0]

print("\n--- P_129 NLP Analysis ---")
print(f"Incident description: {description}")
print(f"Predicted root cause: {predicted_cause}")
print("\nNote: This is a model prediction, not a verified finding.")
print("A safety professional should review the result.")
