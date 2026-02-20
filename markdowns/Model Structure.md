# LV Network - Model Structure

## Scope
This document is the working map of the project structure, model responsibilities, and assistant operating rules for prompt handling.

## 1) Repository map

- `Codes/`
  - Main notebooks, scripts, and core Python modules.
- `Codes/sourcecode/`
  - Reusable modeling, optimization, aggregation, and analytics modules.
- `Input data/`
  - Raw/processed inputs (weather, demand, hot water, occupancy-linked profiles, metadata).
- `Output Data/`
  - Simulation outputs (batch run CSVs, per-dwelling breakdown files, metrics, plots).
- `markdowns/`
  - Project documentation, including this file.
- Supporting data folders:
  - `Data_for_CIGRE_Network/`
  - `Modified_116_LV_CSV/`
  - `Other Demand Profiles/`
  - `UoNewCastle/`

## 2) Module catalog (what each module does)

### Core optimization and workflow

| Module | What it does | Main inputs | Main outputs | Used by |
|---|---|---|---|---|
| `Codes/sourcecode/RC_Optimization.py` | Solves thermal-energy dispatch optimization with Gurobi. Supports hybrid/monovalent heating logic, hot-water modes, EV charging constraints, and day-ahead/full-horizon solve. Builds tariffs too. | Building RC params (`R1,C1,g`), tariff, weather (`Tout,S`), comfort setpoints/tolerance, device capacities, EV and HW settings | Per-step optimal schedules (`Q_hp_space`, `Q_bo_space`, `Q_hp_hw`, `Q_bo_hw`, `P_ev_charge`, temperatures, storage states) and objective costs | `FullEnergyOptimizationDemo11.ipynb`, `stochastic_baseload_multiple_building_simulation_and_aggregation.py`, demand generation scripts |
| `Codes/sourcecode/stochastic_baseload_multiple_building_simulation_and_aggregation.py` | Orchestrates end-to-end stochastic simulation workflow for many dwellings. Handles profile sampling by occupancy, EV travel synthesis, Monte Carlo runs, run aggregation, summary plots, EV-power sweep experiments, per-dwelling breakdown export, and cache-based EV-penetration x HHP-share experiments. Supports optional on-the-fly homogeneous EV profile generation to replace cached EV components in penetration studies, and explicit per-pixel tqdm progress updates. | Metadata CSV, weather CSV, stochastic demand profiles, configuration dictionaries (`optim_params_cfg`, `ev_params_cfg`, `hw_params_cfg`), cached single-dwelling breakdown folders (`hybrid`/`monovalent`), optional EV-generation parameter dictionary | Monte Carlo result dicts, run CSVs (optional), aggregated curves, summary tables, per-dwelling breakdown CSV, penetration-grid maximum-demand summary table, contour-plot-ready surface table | `Codes/FullEnergyOptimizationDemo11.ipynb` |

### Network simulation and aggregation

| Module | What it does | Main inputs | Main outputs | Used by |
|---|---|---|---|---|
| `Codes/sourcecode/Residential_CIGRE_LV_network.py` | Builds and trims CIGRE LV pandapower network to the intended bus subset (`Bus R*` + `Bus 0`). | None (internally creates CIGRE LV network) | Pandapower `net` object | `MC_simulation.py`, network studies |
| `Codes/sourcecode/Load_aggregation.py` | Assigns household technologies (HP/HHP) by penetration rate and aggregates load components per network node. | Pandapower network loads, HHP/HP profile data, baseload profile, HHP share | `df_load_info` (tech allocation), `df_load_by_nodes` (time-indexed nodal demand) | `MC_simulation.py` |
| `Codes/sourcecode/MC_simulation.py` | Performs Monte Carlo network power flow simulation for random HHP allocation and records extreme line/trafo/bus metrics and their timestamps. | Number of samples, HHP penetration, profile files, network | CSV with per-sample extremes, percentile summaries of loading/voltage | Network impact analysis pipeline |
| `Codes/sourcecode/Network_Plotting.py` | Visualizes power-flow results on network graph with bus voltage colors, line flow colors/widths, and transformer sizing annotation. | Solved pandapower network (`pp.runpp` already executed) | Saved network plot image | Manual analysis and reporting |

### Demand profile generation and analytics

| Module | What it does | Main inputs | Main outputs | Used by |
|---|---|---|---|---|
| `Codes/sourcecode/generate_demand_profiles.py` | Generates dwelling-level heating demand profiles from summary RC parameters and weather window by calling optimization for each dwelling. | Summary file (`dataset,R1,C1,g`), weather profile, tariff assumptions, device limits | Per-dwelling demand CSVs in `Codes/Output/DemandProfiles` | Downstream metrics scripts |
| `Codes/sourcecode/generate_demand_metrics.py` | Aggregates generated demand profiles into reporting metrics (peak window, total electricity, total gas) and joins thermal params. | Demand profile folders + summary file | `demand_metrics_summary.csv` | `analyze_peak_reduction.py`, reporting |
| `Codes/sourcecode/analyze_peak_reduction.py` | Computes peak-demand reduction (Flat -> ToU, extreme weather), derives thermal/HTC and HP heat-share indicators, and produces hist/scatter visualizations. | Metrics CSV or recomputed metrics | Processed reduction table and plots | Post-analysis notebooks/scripts |

## 3) Notebook role map

| Notebook | Role |
|---|---|
| `Codes/FullEnergyOptimizationDemo11.ipynb` | Primary experiment notebook: workflow setup, MC runs, MHP/HHP sweeps, single/all-dwelling breakdown runs, convergence analytics, and cache-based EV-penetration x HHP-share maximum-demand sweep. |
| `Codes/Generate_Occupancy_based_demand_with_CREST_model.ipynb` | Demand profile generation and occupancy-linked preprocessing. |
| `Codes/Data Preprocessing.ipynb` | Data cleaning/transformation utilities. |
| `Codes/Main.ipynb`, `Codes/Test.ipynb`, `Codes/IEA_Con_Result_Analysis.ipynb` | Scenario assembly, experimentation, and result analysis utilities. |

## 4) Output contracts

- Batch run file (when enabled): `Output Data/<subdir>/mc_run_XX.csv`
- Per-dwelling run breakdown: `Output Data/<subdir>/dwelling_<id>_runs_breakdown.csv`
- Optional per-dwelling metrics: `..._run_metrics.csv`
- Optional convergence plots: `..._convergence.png`
- Optional per-dwelling stacked consumption plots: `Output Data/<subdir>/plots/exp5_cache_stackplots/<case>/dwelling_<id>_stacked_consumption.png`
- Optional aggregate stacked consumption plot: `Output Data/<subdir>/plots/exp5_cache_stackplots/<case>/aggregate_stacked_consumption.png`
- Penetration sweep summary table: `Output Data/<subdir>/ev_hhp_penetration_max_demand.csv`
- Penetration contour plot: `Output Data/<subdir>/ev_hhp_penetration_contour_max_demand.png`

## 5) Prompt handling protocol (must follow)

### A) Required steps on every prompt

1. Re-open and read this file first: `markdowns/Model Structure.md`.
2. Sync with latest code/notebook state before editing.
3. Prioritize reading code structure over reading raw data contents.
4. Avoid deep inspection of large data files unless explicitly requested.
5. Keep notebook-safe outputs (clear cell-friendly formatting).
6. Check/report time around code edits for traceability.
7. Ask concise clarifying questions when requirements are ambiguous.

### B) Structural change documentation rule

- Every code-structure change must be reflected in this file in the same working session.
- Structural changes include:
  - new modules
  - removed modules
  - renamed modules
  - major responsibility shifts between modules
  - new standard workflow outputs.

## 6) Structure change log

- `2026-02-20`:
  - Added an Experiment 3 tariff override in `FullEnergyOptimizationDemo11.ipynb` so optimization can be run with flat or cosy tariffs without rebuilding the global workflow context.
  - Added a safety fallback in the Experiment 3 cell to rebuild the workflow context if it is missing, to prevent `NameError` when running the cell in isolation.
  - Added Experiment 5 to plot per-dwelling stacked electricity consumption curves from cached breakdown files, using the same cache folders as Experiment 4.
  - Added plotting helpers in `stochastic_baseload_multiple_building_simulation_and_aggregation.py` to render stacked curves from breakdown CSVs.
  - Updated Experiment 5 to include full-case/EV/tariff metadata in plot titles and to support aggregated plots when no dwelling IDs are specified.
  - Updated stacked consumption plotting style to remove band edge lines and emphasize the total mean curve for readability.
  - Reduced the total mean line thickness in stacked plots for a cleaner look.
  - Updated Experiment 5 to use explicit case-folder configuration in the cell (no dependence on Experiment 4 globals) with auto-parsing of EV power and tariff labels from folder names.
  - Updated Experiment 5 to derive cache folder paths from case variables (tariff, case type, EV charger power) instead of hard-coded folder lists.
  - Updated Experiment 5 to name plot files using the same caption text used in plot titles.
  - Reordered stacked plots to show appliance at bottom, heat in the middle, EV on top, and thinned the total mean line.
  - Restored tariff subplot support for stacked plots after a stale-kernel mismatch and added alignment markers to aggregate plots.
  - Reverted stacked plot color palette to the original defaults while keeping alignment markers visible.
  - Set stacked plot colors to EV blue, heat pump yellow, and appliance purple.
  - Reduced saturation for EV/HP/appliance stack colors and thinned the total mean line for readability.
- `2026-02-18`:
  - Replaced short project note with full repository + module catalog.
  - Added explicit per-module responsibilities, I/O expectations, and usage mapping.
  - Added mandatory prompt-handling workflow and structural-change documentation rules.
  - Added cache-based EV penetration x HHP share Monte Carlo experiment support:
    - new function in `stochastic_baseload_multiple_building_simulation_and_aggregation.py` for pixel-wise maximum-demand evaluation from cached single-dwelling runs.
    - new notebook experiment cell in `FullEnergyOptimizationDemo11.ipynb` to run and export penetration-grid results.
- `2026-02-19`:
  - Extended Experiment 4 in `FullEnergyOptimizationDemo11.ipynb` with a 3D terrain surface plot over EV penetration x HHP share for maximum-demand reporting.
  - Updated Experiment 4 visualization from 3D terrain to 2D contour for EV penetration x HHP share maximum-demand reporting.
  - Hardened notebook import behavior in `FullEnergyOptimizationDemo11.ipynb`:
    - source module path is forced to the front of `sys.path` and experiment module is explicitly reloaded.
    - prevents stale rolled-back function signatures from persisting in-kernel across edits.
  - Added optional EV profile replacement for Experiment 4:
    - homogeneous EV profiles can be generated on the fly with user parameters and used instead of cached EV profiles.
    - added a notebook snippet before Experiment 4 for EV generation parameter setup.
  - Improved Experiment 4 progress behavior:
    - switched cache-penetration progress tracking to explicit tqdm per-pixel updates with refresh so progress appears during execution.
  - Added explicit cache folder overrides for Experiment 4:
    - `exp4_hybrid_cache_dir` and `exp4_monovalent_cache_dir` can now point to any two folders after a directory restructure.
