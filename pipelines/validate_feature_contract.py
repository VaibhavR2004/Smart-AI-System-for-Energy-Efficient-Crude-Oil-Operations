import pandas as pd
from pipelines.feature_contract import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    FORBIDDEN_COLUMNS
)

# Load processed failure dataset
df = pd.read_csv("data/processed/failure_dataset.csv")

# 1️⃣ Forbidden column check
for col in FORBIDDEN_COLUMNS:
    if col in FEATURE_COLUMNS:
        raise AssertionError(f" Forbidden feature used in model: {col}")

# 2️⃣ Target leakage check
if TARGET_COLUMN in FEATURE_COLUMNS:
    raise AssertionError(" Target leakage detected: target column in features")

# 3️⃣ Missing feature check
missing_features = [f for f in FEATURE_COLUMNS if f not in df.columns]
if missing_features:
    raise AssertionError(f" Missing required features: {missing_features}")

# 4️⃣ Extra columns warning (optional)
extra_columns = [
    c for c in df.columns
    if c not in FEATURE_COLUMNS + [TARGET_COLUMN]
]

if extra_columns:
    print(f" Extra columns found (ignored by model): {extra_columns}")

# Validation success
print(" Feature contract validated successfully!")
print(f" Features used: {len(FEATURE_COLUMNS)}")
print(f" Target column: {TARGET_COLUMN}")
print(f" Dataset shape: {df.shape}")
