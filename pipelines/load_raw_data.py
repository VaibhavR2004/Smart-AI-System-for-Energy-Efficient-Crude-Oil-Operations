import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA = PROJECT_ROOT / "data" / "processed"

PROCESSED_DATA.mkdir(exist_ok=True)


# Global Rules
DROP_IDENTIFIER_COLS = [
    "well_id",
    "equipment_id",
    "facility_name",
    "operator"
]

# Utility Functions
def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convert columns to snake_case lowercase"""
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("-", "_")
    )
    return df


def drop_identifier_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop forbidden identifier columns"""
    drop_cols = [c for c in DROP_IDENTIFIER_COLS if c in df.columns]
    return df.drop(columns=drop_cols)


def convert_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Convert date-like columns to datetime"""
    for col in df.columns:
        if "date" in col or "time" in col:
            try:
                df[col] = pd.to_datetime(df[col], errors="coerce")
            except Exception:
                pass
    return df


def convert_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Convert sensor-like columns to float"""
    for col in df.columns:
        if df[col].dtype == "object":
            try:
                df[col] = pd.to_numeric(df[col])
            except Exception:
                pass
    return df


def clean_dataset(file_name: str) -> pd.DataFrame:
    """Complete cleaning pipeline for a dataset"""
    print(f"\n Loading: {file_name}")
    df = pd.read_csv(RAW_DATA / file_name)

    df = standardize_columns(df)
    df = convert_dates(df)
    df = convert_numeric(df)
    df = drop_identifier_columns(df)

    print(f" Cleaned: {file_name} | Shape: {df.shape}")
    return df



def main():

    datasets = {
        "3w_clean.csv": "cleaned_3w_dataset.csv",
        "pm_clean.csv": "cleaned_predictive_maintenance.csv",
        "oreda_clean.csv": "cleaned_oreda_reliability.csv",
        "refinery_clean.csv": "cleaned_refinery_capacity.csv",
        "ai4i_clean.csv": "cleaned_ai4i_pm.csv"
    }

    for out_file, raw_file in datasets.items():
        df = clean_dataset(raw_file)
        df.to_csv(PROCESSED_DATA / out_file, index=False)
        print(f" Saved → data/processed/{out_file}")

    print("\n PHASE 1 STEP 1 COMPLETE — RAW DATA SANITIZED")


if __name__ == "__main__":
    main()
