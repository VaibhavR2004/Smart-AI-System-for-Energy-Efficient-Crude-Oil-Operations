import pandas as pd
import numpy as np

# Load base dataset
df = pd.read_csv("data/processed/failure_dataset.csv")

# Ensure correct sorting (CRITICAL)
df = df.sort_values(["equipment_id", "timestamp"])

# --------------------------------------------------
# Create future failure labels (24h prediction window)
# --------------------------------------------------
WINDOW_HOURS = 24

df["failure_next_24h"] = (
    df.groupby("equipment_id")["failure"]
      .shift(-WINDOW_HOURS)
      .fillna(0)
      .astype(int)
)

# --------------------------------------------------
# Temporal rolling features
# --------------------------------------------------
ROLLING_WINDOWS = [6, 12, 24]

SENSOR_COLS = [
    "pressure_gas_lift",
    "pressure_production_casing",
    "flow_gaslift",
    "temp_toptubing",
    "temp_downholegauge",
]

for col in SENSOR_COLS:
    for w in ROLLING_WINDOWS:
        df[f"{col}_mean_{w}h"] = (
            df.groupby("equipment_id")[col]
              .rolling(w)
              .mean()
              .reset_index(level=0, drop=True)
        )

        df[f"{col}_std_{w}h"] = (
            df.groupby("equipment_id")[col]
              .rolling(w)
              .std()
              .reset_index(level=0, drop=True)
        )

# --------------------------------------------------
# Clean NaNs (rolling creates NaNs)
# --------------------------------------------------
df = df.fillna(method="bfill").fillna(method="ffill")

# --------------------------------------------------
# Save engineered dataset
# --------------------------------------------------
output_path = "data/features/failure_training_dataset.csv"
df.to_csv(output_path, index=False)

print("✅ Feature engineering completed")
print("📐 Shape:", df.shape)
