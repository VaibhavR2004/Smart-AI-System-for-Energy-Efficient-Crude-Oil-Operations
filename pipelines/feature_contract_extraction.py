"""
Feature Contract — Extraction Rate Prediction
"""

# SENSOR FEATURES (3W)
SENSOR_FEATURES = [
    "pressure_gas_lift",
    "pressure_production_casing",
    "pressure_toptubing",
    "temp_toptubing",
    "temp_downholegauge"
]

# WELL FEATURES
WELL_FEATURES = [
    "well_age_years",
    "attribute_score",
    "aggregate_quality_score"
]

# EQUIPMENT HEALTH (FROM FAILURE MODEL OUTPUT / OREDA)
HEALTH_FEATURES = [
    "mtbf_days",
    "mttr_days",
    "availability_pct"
]

FEATURE_COLUMNS = (
    SENSOR_FEATURES +
    WELL_FEATURES +
    HEALTH_FEATURES
)

# TARGET
TARGET_COLUMN = "extraction_rate_bpd"


# FORBIDDEN (LEAKAGE)
FORBIDDEN_COLUMNS = [
    "flow_gaslift",
    "ExtractionRate_bpd",
    "Well_ID",
    "Equipment_ID"
]
