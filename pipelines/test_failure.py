import pandas as pd
from pipelines.predict_failure import predict_failure

sample_input = pd.DataFrame([{
    "pressure_gas_lift": 92.0,
  "pressure_production_casing": 88.4,
  "pressure_toptubing": 1.95e7,
  "flow_gaslift": 0.3,
  "temp_toptubing": 122,
  "temp_downholegauge": 118,
  "temp_checkproduction": 110,
  "valvestatus_dhsv": 0,
  "valvestatus_master1": 1,
  "valvestatus_master2": 1,
  "valvestatus_xmastree": 1,
  "valvestatus_shutdowngaslift": 1,
  "mtbf_days": 180,
  "mttr_days": 8.4,
  "failure_rate_per_year_expected": 2.1,
  "availability_pct": 85.2
}])

result = predict_failure(sample_input)
print("Result: ",result)


