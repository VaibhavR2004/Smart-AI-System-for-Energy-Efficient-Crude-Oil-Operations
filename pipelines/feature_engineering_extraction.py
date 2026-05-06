import pandas as pd
from sklearn.preprocessing import StandardScaler
from pipelines.feature_contract_extraction import (
    FEATURE_COLUMNS,
    TARGET_COLUMN
)

# Load base dataset
df = pd.read_csv(r"E:\Research_Paper\oil_gas_ai\data\processed\failure_dataset.csv")

df["extraction_rate_bpd"] = (
    df["flow_gaslift"] * 86400 * 0.159
)

# Keep only allowed columns
df = df[FEATURE_COLUMNS + [TARGET_COLUMN]]

# Handle missing values
df = df.fillna(df.median(numeric_only=True))

# Scale numeric features
scaler = StandardScaler()
df[FEATURE_COLUMNS] = scaler.fit_transform(df[FEATURE_COLUMNS])

# Save artifacts
df.to_csv("data/features/extraction_training_dataset.csv", index=False)



print("✅ Extraction training dataset ready")
print("📐 Shape:", df.shape)
