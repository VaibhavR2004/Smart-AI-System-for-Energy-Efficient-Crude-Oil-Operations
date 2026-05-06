"""
Feature Contract for Failure Prediction Model

Defines:
- Allowed input features
- Target variable
- Forbidden columns (leakage prevention)
"""

# ================================
# SENSOR FEATURES (3W Dataset)
# ================================
SENSOR_FEATURES = [
    "pressure_gas_lift",
    "pressure_production_casing",
    "pressure_toptubing",
    "flow_gaslift",
    "temp_toptubing",
    "temp_downholegauge",
    "temp_checkproduction"
]

# ================================
# VALVE / STATE FEATURES
# ================================
STATE_FEATURES = [
    "valvestatus_dhsv",
    "valvestatus_master1",
    "valvestatus_master2",
    "valvestatus_xmastree",
    "valvestatus_shutdowngaslift"
]

# ================================
# RELIABILITY FEATURES (OREDA)
# ================================
RELIABILITY_FEATURES = [
    "mtbf_days",
    "mttr_days",
    "failure_rate_per_year_expected",
    "availability_pct"
]

# ================================
# FINAL FEATURE LIST (ORDER MATTERS)
# ================================
FEATURE_COLUMNS = (
    SENSOR_FEATURES
    + STATE_FEATURES
    + RELIABILITY_FEATURES
)

# ================================
# TARGET VARIABLE
# ================================
TARGET_COLUMN = "failure"

# ================================
# FORBIDDEN COLUMNS (LEAKAGE PREVENTION)
# ================================
FORBIDDEN_COLUMNS = [
    "well_id",        # ID leakage
    "equipment_id",   # ID leakage
    "facility_name",
    "operator",
    "latitude",
    "longitude",
    "date"            # temporal leakage
]
