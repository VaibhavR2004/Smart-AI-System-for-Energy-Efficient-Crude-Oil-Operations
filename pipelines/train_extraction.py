import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

from pipelines.feature_contract_extraction import (
    FEATURE_COLUMNS,
    TARGET_COLUMN
)

# Load dataset
df = pd.read_csv("data/features/extraction_training_dataset.csv")

X = df[FEATURE_COLUMNS]
y = df[TARGET_COLUMN]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = XGBRegressor(
    n_estimators=400,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
print(y)
# Train
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)

rmse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Extraction Model Performance")
print(f"RMSE: {rmse:.3f}")
print(f"MAE : {mae:.3f}")
print(f"R²  : {r2:.3f}")

# Save model
os.makedirs("models/extraction", exist_ok=True)
joblib.dump(model, "models/extraction/extraction_model.pkl")
print("💾 Model saved → models/extraction/extraction_model.pkl")