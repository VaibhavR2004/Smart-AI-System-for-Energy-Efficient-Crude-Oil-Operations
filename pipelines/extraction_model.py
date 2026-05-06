import pandas as pd
import joblib
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

from pipelines.feature_contract_extraction import (
    FEATURE_COLUMNS,
    TARGET_COLUMN
)

# ─────────────────────────────────────────────
# 1. Load raw dataset (NO scaling)
# ─────────────────────────────────────────────
print("📂 Loading dataset...")
df = pd.read_csv("data/features/extraction_training_dataset.csv")
print(f"   Shape: {df.shape}")

# ─────────────────────────────────────────────
# 2. Check if data is pre-scaled and undo it
#    (values between -5 and 5 = already scaled)
# ─────────────────────────────────────────────
sample_val = df["pressure_gas_lift"].abs().max()

if sample_val < 100:
    print("\n⚠️  WARNING: Data appears to be pre-scaled (Z-score normalized).")
    print("   Re-generating raw features from build script is recommended.")
    print("   Proceeding with scaled data — test inputs must also be scaled.\n")
else:
    print("   ✅ Data is in raw units — no scaling needed.\n")

# ─────────────────────────────────────────────
# 3. Features and target — raw, no scaling
# ─────────────────────────────────────────────
X = df[FEATURE_COLUMNS]
y = df[TARGET_COLUMN]

print(f"   Feature columns : {FEATURE_COLUMNS}")
print(f"   Target range    : {y.min():.1f} to {y.max():.1f} BPD")
print(f"   Target mean     : {y.mean():.1f} BPD\n")

# ─────────────────────────────────────────────
# 4. Train-test split
# ─────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"   Train size: {len(X_train):,} | Test size: {len(X_test):,}")

# ─────────────────────────────────────────────
# 5. Model — XGBoost (NO scaling needed for trees)
# ─────────────────────────────────────────────
model = XGBRegressor(
    n_estimators=400,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,          # use all CPU cores
    tree_method="hist"  # faster training
)

print("\n🚀 Training model...")
model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=50
)

# ─────────────────────────────────────────────
# 6. Evaluate
# ─────────────────────────────────────────────
y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))  # true RMSE
mae  = mean_absolute_error(y_test, y_pred)
r2   = r2_score(y_test, y_pred)

print("\n" + "=" * 45)
print("   EXTRACTION MODEL PERFORMANCE")
print("=" * 45)
print(f"   RMSE : {rmse:>12,.3f} BPD")
print(f"   MAE  : {mae:>12,.3f} BPD")
print(f"   R²   : {r2:>12.4f}")
print("=" * 45)

# ─────────────────────────────────────────────
# 7. Save model ONLY (no scaler needed)
# ─────────────────────────────────────────────
os.makedirs("models/extraction", exist_ok=True)
joblib.dump(model, "models/extraction/extraction_model.pkl")
print("\n💾 Model saved → models/extraction/extraction_model.pkl")
print("   ✅ No scaler saved — raw inputs work directly.\n")

# ─────────────────────────────────────────────
# 8. Quick sanity check — prediction range
# ─────────────────────────────────────────────
print("🔍 Sanity check — prediction spread on test set:")
print(f"   Min predicted : {y_pred.min():>10,.2f} BPD")
print(f"   Max predicted : {y_pred.max():>10,.2f} BPD")
print(f"   Std predicted : {y_pred.std():>10,.2f} BPD")
print(f"   (If all values are identical → data is still scaled)")