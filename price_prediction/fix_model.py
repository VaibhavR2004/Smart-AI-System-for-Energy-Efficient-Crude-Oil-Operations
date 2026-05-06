"""
Diagnostic + Fix script for corrupted .pkl model files
Run this first to identify which files are broken, then re-save them.

Run: python fix_models.py
"""
import numpy as np
import pandas as pd
import pickle
import json
from pathlib import Path

MODEL_DIR = Path("models/economics")

# ─────────────────────────────────────────────
# 1. Diagnose all pkl files
# ─────────────────────────────────────────────

files = {
    "price_lgbm.pkl":      "lgbm",
    "price_xgb.pkl":       "xgb",
    "price_ridge.pkl":     "ridge",
    "scaler.pkl":          "scaler",
    "feature_columns.pkl": "feature_columns",
}

print("── Diagnosing model files ───────────────────────────")
loaded = {}
for fname, key in files.items():
    fpath = MODEL_DIR / fname
    try:
        with open(fpath, "rb") as f:
            obj = pickle.load(f)
        loaded[key] = obj
        print(f"  ✅ {fname:<25} → loaded OK  (type: {type(obj).__name__})")
    except Exception as e:
        loaded[key] = None
        print(f"  ❌ {fname:<25} → FAILED: {e}")

try:
    with open(MODEL_DIR / "meta.json", "r") as f:
        meta = json.load(f)
    print(f"  ✅ {'meta.json':<25} → loaded OK")
except Exception as e:
    meta = {}
    print(f"  ❌ {'meta.json':<25} → FAILED: {e}")

print()

# ─────────────────────────────────────────────
# 2. Try alternative loaders for broken files
# ─────────────────────────────────────────────

def try_load_lgbm(path):
    """Try loading with lightgbm's native loader."""
    try:
        import lightgbm as lgb
        model = lgb.Booster(model_file=str(path).replace(".pkl", ".txt"))
        print(f"  ✅ Loaded via lgbm native loader")
        return model
    except Exception as e:
        print(f"  ❌ lgbm native loader failed: {e}")

    try:
        import joblib
        model = joblib.load(path)
        print(f"  ✅ Loaded via joblib")
        return model
    except Exception as e:
        print(f"  ❌ joblib failed: {e}")

    return None


def try_load_xgb(path):
    """Try loading with xgboost's native loader."""
    try:
        import xgboost as xgb
        model = xgb.XGBRegressor()
        model.load_model(str(path).replace(".pkl", ".json"))
        print(f"  ✅ Loaded via xgb native loader (.json)")
        return model
    except Exception as e:
        print(f"  ❌ xgb native json loader failed: {e}")

    try:
        import joblib
        model = joblib.load(path)
        print(f"  ✅ Loaded via joblib")
        return model
    except Exception as e:
        print(f"  ❌ joblib failed: {e}")

    return None


def try_load_generic(path):
    """Try joblib as fallback for any pkl."""
    try:
        import joblib
        obj = joblib.load(path)
        print(f"  ✅ Loaded via joblib")
        return obj
    except Exception as e:
        print(f"  ❌ joblib failed: {e}")
    return None


print("── Attempting alternative loaders for failed files ──")
for fname, key in files.items():
    if loaded[key] is None:
        fpath = MODEL_DIR / fname
        print(f"\n  Trying alternatives for {fname}:")
        if key == "lgbm":
            loaded[key] = try_load_lgbm(fpath)
        elif key == "xgb":
            loaded[key] = try_load_xgb(fpath)
        else:
            loaded[key] = try_load_generic(fpath)

print()

# ─────────────────────────────────────────────
# 3. Re-save recovered files with pickle
# ─────────────────────────────────────────────

print("── Re-saving recovered files ────────────────────────")
for fname, key in files.items():
    if loaded[key] is not None:
        backup_path = MODEL_DIR / fname
        fixed_path  = MODEL_DIR / fname.replace(".pkl", "_fixed.pkl")
        try:
            with open(fixed_path, "wb") as f:
                pickle.dump(loaded[key], f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"  ✅ Re-saved → {fixed_path.name}")
        except Exception as e:
            print(f"  ❌ Could not re-save {fname}: {e}")
    else:
        print(f"  ⚠️  Skipping {fname} — still not loaded")

print()

# ─────────────────────────────────────────────
# 4. If ridge is unrecoverable — retrain it
# ─────────────────────────────────────────────

if loaded.get("ridge") is None:
    print("── Ridge model unrecoverable — retraining ───────────")
    print("   (Uses the same scaler + feature_columns already loaded)\n")

    import numpy as np
    import pandas as pd
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import train_test_split

    scaler_obj         = loaded.get("scaler")
    feature_columns    = loaded.get("feature_columns")

    if scaler_obj is None or feature_columns is None:
        print("  ❌ Cannot retrain — scaler or feature_columns also failed to load.")
        print("     Please re-run the original training script to regenerate all files.")
    else:
        print(f"  Features: {feature_columns}")
        print("  Generating placeholder training data — replace with your actual data!\n")

        # ── REPLACE THIS BLOCK with your actual training data ──
        np.random.seed(42)
        n = 300
        X_dummy = np.random.randn(n, len(feature_columns))
        y_dummy = X_dummy[:, 0] * 10 + np.random.randn(n) * 2  # fake target

        X_scaled = scaler_obj.transform(X_dummy)
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_dummy, test_size=0.2, random_state=42
        )

        ridge_new = Ridge(alpha=1.0)
        ridge_new.fit(X_train, y_train)

        score = ridge_new.score(X_test, y_test)
        print(f"  Retrained Ridge R² on test set: {score:.4f}")

        ridge_fixed_path = MODEL_DIR / "price_ridge_fixed.pkl"
        with open(ridge_fixed_path, "wb") as f:
            pickle.dump(ridge_new, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"  ✅ Ridge retrained and saved → {ridge_fixed_path.name}")
        loaded["ridge"] = ridge_new

print()

# ─────────────────────────────────────────────
# 5. Quick smoke test with all recovered models
# ─────────────────────────────────────────────

print("── Smoke Test ───────────────────────────────────────")
scaler_obj      = loaded.get("scaler")
feature_columns = loaded.get("feature_columns")

if scaler_obj and feature_columns:
    dummy_input = pd.DataFrame(
        [np.ones(len(feature_columns))],
        columns=feature_columns
    )
    X = scaler_obj.transform(dummy_input)

    for key in ["lgbm", "xgb", "ridge"]:
        model = loaded.get(key)
        if model:
            try:
                pred = model.predict(X)
                print(f"  ✅ {key:<10} predict OK → {pred[0]:.4f}")
            except Exception as e:
                print(f"  ❌ {key:<10} predict FAILED: {e}")
        else:
            print(f"  ⚠️  {key:<10} not loaded — skipping")
else:
    print("  ⚠️  Scaler or feature_columns missing — cannot run smoke test")

print("\nDone. Use the '_fixed.pkl' files in your main prediction script.")
print("Rename them (remove '_fixed') or update the paths in crude_oil_prediction_demo.py")