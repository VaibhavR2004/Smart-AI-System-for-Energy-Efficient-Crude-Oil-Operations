import pandas as pd
import joblib
import os
# ─────────────────────────────────────────────
# 1. Load the saved model
# ─────────────────────────────────────────────
MODEL_PATH = "models/extraction/extraction_model.pkl"

if not os.path.exists(MODEL_PATH):
    print(f"❌ Model not found at: {MODEL_PATH}")
    print("   Make sure you run train_extraction.py first.")
    exit()

model = joblib.load(MODEL_PATH)
print("✅ Model loaded successfully\n")

# ─────────────────────────────────────────────
# 2. Import feature columns from your contract
# ─────────────────────────────────────────────
from pipelines.feature_contract_extraction import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    SENSOR_FEATURES,
    WELL_FEATURES,
    HEALTH_FEATURES
)

# ─────────────────────────────────────────────
# 3. Define sample test inputs
#    Using EXACT column names from feature contract
# ─────────────────────────────────────────────
test_cases = pd.DataFrame([
    {
        # ── Well A: High-performing, new well, healthy equipment
        # SENSOR FEATURES
        "pressure_gas_lift":          92.0,
        "pressure_production_casing": 88.4,
        "pressure_toptubing":         1.95e7,
        "temp_toptubing":             145.0,
        "temp_downholegauge":         185.0,
        # WELL FEATURES
        "well_age_years":             1.5,
        "attribute_score":            9.2,
        "aggregate_quality_score":    8.8,
        # HEALTH FEATURES
        "mtbf_days":                  300,
        "mttr_days":                  4.0,
        "availability_pct":           97.5,
    },
    {
        # ── Well B: Average well, mid-age, moderate health
        # SENSOR FEATURES
        "pressure_gas_lift":          70.0,
        "pressure_production_casing": 65.0,
        "pressure_toptubing":         1.20e7,
        "temp_toptubing":             122.0,
        "temp_downholegauge":         155.0,
        # WELL FEATURES
        "well_age_years":             5.0,
        "attribute_score":            6.5,
        "aggregate_quality_score":    6.0,
        # HEALTH FEATURES
        "mtbf_days":                  180,
        "mttr_days":                  8.4,
        "availability_pct":           85.2,
    },
    {
        # ── Well C: Degraded, aging well, poor equipment health
        # SENSOR FEATURES
        "pressure_gas_lift":          40.0,
        "pressure_production_casing": 35.0,
        "pressure_toptubing":         0.60e7,
        "temp_toptubing":             95.0,
        "temp_downholegauge":         115.0,
        # WELL FEATURES
        "well_age_years":             12.0,
        "attribute_score":            3.8,
        "aggregate_quality_score":    3.2,
        # HEALTH FEATURES
        "mtbf_days":                  60,
        "mttr_days":                  18.0,
        "availability_pct":           55.0,
    },
])

# ─────────────────────────────────────────────
# 4. Align columns to match training order
# ─────────────────────────────────────────────
test_cases = test_cases[FEATURE_COLUMNS]

# ─────────────────────────────────────────────
# 5. Predict
# ─────────────────────────────────────────────
predictions = model.predict(test_cases)

# ─────────────────────────────────────────────
# 6. Display results
# ─────────────────────────────────────────────
print("=" * 55)
print("   EXTRACTION RATE PREDICTION — DEMO TEST")
print("=" * 55)

wells = [
    ("Well A", "High-Performing  | New Well      | Healthy Equipment"),
    ("Well B", "Average          | Mid-Age Well  | Moderate Health"),
    ("Well C", "Degraded         | Aging Well    | Poor Equipment"),
]

for i, ((name, desc), pred) in enumerate(zip(wells, predictions)):

    print(f"\n🛢️  {name} — {desc}")
    print(f"   {'─' * 50}")

    print("   📡 SENSOR READINGS:")
    for col in SENSOR_FEATURES:
        val = test_cases.iloc[i][col]
        print(f"      {col:<35} = {val:>12.2f}")

    print("   🏗️  WELL ATTRIBUTES:")
    for col in WELL_FEATURES:
        val = test_cases.iloc[i][col]
        print(f"      {col:<35} = {val:>12.2f}")

    print("   🔧 EQUIPMENT HEALTH:")
    for col in HEALTH_FEATURES:
        val = test_cases.iloc[i][col]
        print(f"      {col:<35} = {val:>12.2f}")

    print(f"   {'─' * 50}")
    print(f"   ➤  Predicted {TARGET_COLUMN:<22} = {pred:>10,.2f} BPD")

    # Production status flag
    if pred >= 5000:
        flag = "🟢 HIGH PRODUCTION"
    elif pred >= 2000:
        flag = "🟡 MODERATE PRODUCTION"
    else:
        flag = "🔴 LOW PRODUCTION — Maintenance Review Required"
    print(f"   ➤  Production Status               : {flag}")

print("\n" + "=" * 55)
print("✅ Demo test complete.")
print("=" * 55)