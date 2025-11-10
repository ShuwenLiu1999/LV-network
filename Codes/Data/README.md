# Dwelling input summary

`dwelling_summary.csv` is intentionally left without sample rows so you can drop in the exact dwelling list from your summary export (for example, the file that lives under `Codes/Data` in your working copy). Only keep the columns you plan to edit—at a minimum the optimiser needs `dwelling_id`, `technology`, `R1`, `C1`, and `g`. Every other operational quantity that the optimiser needs is injected automatically by `run_dwelling_optimization.py`, so you only have to edit the thermal coefficients and leave the dwelling identifiers untouched.

When you run the batch script it now evaluates up to three preset demand patterns so you can compare how the same dwelling behaves under different occupancy assumptions without editing the CSV:

| Pattern key | Description | Comfort windows (set to `setpoint_comfort_c`) |
| --- | --- | --- |
| `two_peaks` | Morning and evening peaks matching the original schedule. | Defaults to 06:00–10:00 and 17:00–21:00 but honours any per-dwelling window overrides supplied in the summary CSV. |
| `evening_peak` | Single evening occupancy block for commuters. | 17:00–22:00 |
| `daytime_continuous` | Daytime presence with a long comfort window. | 08:00–20:00 |

Use the `--patterns` CLI flag to pick one or more profiles (defaults to all three). Results for each pattern are written to a dedicated subfolder beneath the chosen output directory.

The optimiser continues to use the repository's built-in cosy tariff profile, matching the previous workflow.

## Automatically supplied defaults

When the summary file does not provide an explicit value, the batch runner supplies the following defaults:

| Parameter | Default | Notes |
| --- | --- | --- |
| `initial_temp_c` | 21.0 °C | Matches the start temperature used in the worked example in `RC_Optimization.py`. |
| `cop` | 3.5 | Same COP that the original optimisation example assumes for the heat pump. |
| `boiler_efficiency` | 0.9 for HHP dwellings, 1.0 for HP-only | Aligns with the efficiency value used in `RC_Optimization.py`; pure HP homes never call the boiler so a neutral efficiency is applied. |
| `hp_capacity_kw` | 7 kW for HP dwellings, 4 kW for HHP dwellings | Mirrors the 7 kW HP and 4 kW HHP test cases packaged with the repository datasets. |
| `boiler_capacity_kw` | 24 kW for HHP dwellings, 0 for HP dwellings | Replicates the boiler ceiling from the optimisation example; the cap is removed for HP-only cases. |
| `tolerance_c` | ±1 °C | Matches the comfort band enforced in the example optimisation script. |
| `setpoint_comfort_c` / `setpoint_setback_c` | 21 °C during comfort windows, 15 °C otherwise | Same comfort vs. setback temperatures used in the example script. |
| `morning_start_hour` – `morning_end_hour` | 06:00–10:00 | Defines the first comfort window. |
| `evening_start_hour` – `evening_end_hour` | 17:00–21:00 | Defines the second comfort window. |
| `day_ahead` | `False` | Runs the optimiser across the whole horizon in one pass (matching the default example). |

These defaults are applied per dwelling, and the resolved values (including an audit of which fields fell back to defaults) are exported to `resolved_dwelling_parameters.csv` alongside the optimisation outputs so you can double-check exactly what was used.

If you do want to override any of the defaults later, simply add the relevant column(s) back into `dwelling_summary.csv`; any non-empty entry takes precedence over the fallback shown above. Extra columns that are not used by the optimiser are ignored, so you can keep the rest of your original summary export alongside the thermal fields if that is more convenient.
