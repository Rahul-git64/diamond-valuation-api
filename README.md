# Diamond Valuation API

End-to-end gemological diamond valuation pipeline featuring strict GIA ordinal encodings (`cut`, `color`, `clarity`), physical geometry validation (`volume_mm3`, `table_to_depth_ratio`), single (`POST /predict`) and high-throughput batch (`POST /predict/batch`, up to 500 stones) FastAPI inference, and KS/PSI drift supervision.
