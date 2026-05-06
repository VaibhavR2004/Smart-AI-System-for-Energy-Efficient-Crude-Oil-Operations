import json
import joblib
import pandas as pd
import numpy as np

# Load artifacts
MODEL_PATH = "models/failure/xgb_failure_model.pkl"
THRESHOLD_PATH = "models/failure/threshold.json"
FEATURE_CONTRACT_PATH = "models/failure/feature_contract.json"

model = joblib.load(MODEL_PATH)

with open(THRESHOLD_PATH, "r") as f:
    THRESHOLD = json.load(f)["failure_threshold"]

with open(FEATURE_CONTRACT_PATH, "r") as f:
    FEATURE_COLUMNS = json.load(f)["features"]

print("✅ Model & artifacts loaded")

# Prediction function
def predict_failure(input_df: pd.DataFrame):
    """
    input_df: DataFrame with required feature columns
    """

    # Ensure correct features
    missing = [c for c in FEATURE_COLUMNS if c not in input_df.columns]
    if missing:
        raise ValueError(f"Missing required features: {missing}")

    X = input_df[FEATURE_COLUMNS].copy()

    # Fill missing numeric values
    X = X.fillna(X.median(numeric_only=True))

    # Predict probability
    failure_prob = model.predict_proba(X)[:, 1]

    # Apply threshold
    if failure_prob*100 >= THRESHOLD:
        failure_pred = 1
    else:
        failure_pred = 0
        print("Failure probability:", failure_prob)
        print("Threshold:", THRESHOLD)

    # Return results
    result = input_df.copy()
    result["failure_probability"] = failure_prob
    result["failure_prediction"] = failure_pred

    return result

