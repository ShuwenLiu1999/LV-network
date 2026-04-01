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
  - Project documentation, including this file and `key assumptions.md`.
- Supporting data folders:
  - `Data_for_CIGRE_Network/`
  - `Modified_116_LV_CSV/`
  - `Other Demand Profiles/`
  - `UoNewCastle/`

## 2) Module catalog (what each module does)

### Core optimization and workflow

| Module | What it does | Main inputs | Main outputs | Used by |
|---|---|---|---|---|
| `Codes/sourcecode/RC_Optimization.py` | Solves thermal-energy dispatch optimization with Gurobi. Supports hybrid/monovalent heating logic, hot-water modes, EV charging constraints, and day-ahead/full-horizon solve. Builds tariffs for `flat`, `cozy`, and `agile` (Agile electricity from CSV + constant gas price). | Building RC params (`R1,C1,g`), tariff, weather (`Tout,S`), comfort setpoints/tolerance, device capacities, EV and HW settings | Per-step optimal schedules (`Q_hp_space`, `Q_bo_space`, `Q_hp_hw`, `Q_bo_hw`, `P_ev_charge`, temperatures, storage states) and objective costs | `FullEnergyOptimizationDemo11.ipynb`, `stochastic_baseload_multiple_building_simulation_and_aggregation.py`, demand generation scripts |
| `Codes/sourcecode/stochastic_baseload_multiple_building_simulation_and_aggregation.py` | Orchestrates end-to-end stochastic simulation workflow for many dwellings. Handles profile sampling by occupancy, EV travel synthesis, Monte Carlo runs, run aggregation, summary plots, EV-power sweep experiments, per-dwelling breakdown export, and cache-based EV-penetration x HHP-share experiments. Supports optional on-the-fly homogeneous EV profile generation to replace cached EV components in penetration studies, explicit per-pixel tqdm progress updates, and randomized-tariff-offset MC workflows (including offset-range scans across `hybrid`/`monovalent` cases). | Metadata CSV, weather CSV, stochastic demand profiles, configuration dictionaries (`optim_params_cfg`, `ev_params_cfg`, `hw_params_cfg`), cached single-dwelling breakdown folders (`hybrid`/`monovalent`), optional EV-generation parameter dictionary | Monte Carlo result dicts, run CSVs (optional), aggregated curves, summary tables, per-dwelling breakdown CSV, penetration-grid maximum-demand summary table, contour-plot-ready surface table | `Codes/FullEnergyOptimizationDemo11.ipynb` |

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
| `Codes/FullEnergyOptimizationDemo11.ipynb` | Primary experiment notebook: workflow setup, MC runs, MHP/HHP sweeps, single/all-dwelling breakdown runs, convergence analytics, cache-based EV-penetration x HHP-share maximum-demand sweep, randomized cozy-tariff offset scans across both heating cases, and post-scan energy-cost component summarization from randomized-offset breakdown outputs (HP/EV/baseload electricity + gas) with case-wise component-cost vs peak-demand plots. |
| `Codes/Diagnose_HHP_Infeasibility.ipynb` | Replays infeasible cached Experiment 6 `(dwelling, run)` cases and applies A/B relaxation tests (EV targets vs thermal constraints), plus capacity-limit relaxations (EV charge-cap lift and monovalent HP-cap lift), to classify likely infeasibility drivers and export diagnosis summaries including selected feasible HP capacities. |
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
- Experiment 6a electricity/gas/energy-cost summary: `Output Data/Single Dwelling Runs/randomized offset/exp6a_energy_cost_summary.csv` (includes infeasible-handling counters when infeasible run curves are replaced by feasible-run mean dwelling curves; includes component breakdown columns for HP electricity, EV electricity, baseload electricity, and gas costs/energy)
- Experiment 6 per-folder monovalent HP-capacity summary: `Output Data/Single Dwelling Runs/randomized offset/<tariff>_monovalent_EV_<kW>kW_offset<X>h/dwelling_monovalent_hp_capacity_summary.csv` (one row per dwelling with max selected HP capacity across MC runs)
- Diagnosis A/B per-pair summary: `Output Data/Single Dwelling Runs/randomized offset/<case>/diagnosis_ab_test_summary.csv` (includes `*_hp_capacity_kw` columns for replayed scenarios)
- Diagnosis capacity-relaxation status summary: `Output Data/Single Dwelling Runs/randomized offset/<case>/diagnosis_capacity_relaxation_summary.csv`

## 5) Prompt handling protocol (must follow)

### A) Required steps on every prompt

1. Re-open and read this file first: `markdowns/main.md`.
2. Sync with latest code/notebook state before editing.
3. Prioritize reading code structure over reading raw data contents.
4. Avoid deep inspection of large data files unless explicitly requested.
5. Keep notebook-safe outputs (clear cell-friendly formatting).
6. Check/report time around code edits for traceability.
7. Ask concise clarifying questions when requirements are ambiguous or can be interpreted in multiple ways.
8. When possible, provide multiple implementation options before making changes.
9. When multiple options are presented, wait for the user to choose before implementing.
10. When the user asks to do work in a notebook, implement it as notebook code cells (not prose-only instructions).
11. Before changing the local runtime environment (install/upgrade/remove packages, interpreter/kernel/toolchain changes), ask the user for explicit permission first.
12. When any model assumption changes, update `markdowns/key assumptions.md` in place in the same working session.
13. Reproducibility rule: never run experiment/data-update tasks directly; always provide runnable code for the user to execute.
14. On every new user task, re-open `markdowns/main.md` first before analysis, edits, or command execution.

### B) Structural change documentation rule

- Every code-structure change must be reflected in this file in the same working session.
- Always track changes in two places when applicable:
  - update the change log in Section 6.
  - update impacted structure sections in place (module catalog, notebook role map, output contracts, or protocol sections).
- Structural changes include:
  - new modules
  - removed modules
  - renamed modules
  - major responsibility shifts between modules
  - new standard workflow outputs.

### C) Code commenting standard

- Add straightforward and concise comments for key variables/functions and the purpose of each code block.
- Keep comments practical and minimal: explain intent and usage, not obvious syntax.

## 6) Structure change log

- `2026-04-01`:
  - Updated `Experiment 6a` in `Codes/FullEnergyOptimizationDemo11.ipynb` to compute component-level energy and cost breakdowns from cached randomized-offset breakdowns:
    - electricity split into `hp_elec_kw`, `ev_charge_kw`, and `appliance_kw` components,
    - gas kept as `boiler_gas_kw`,
    - all components priced on original un-offset tariff and exported in `exp6a_energy_cost_summary.csv`.
  - Updated `Experiment 6a` infeasible-run replacement workflow to apply replacement at component level (not only total load), then recompute total electricity/gas curves from replaced components.
  - Replaced Experiment 6a comparison plots with a two-subplot case layout:
    - top subplot: `monovalent`,
    - bottom subplot: `hybrid`,
    - each subplot shows grouped component energy-cost bars (with explicit component legend) plus a line for mean peak demand.
  - Updated `Experiment 6a` component-cost plot styling:
    - keeps grouped component bars,
    - renders legend from the top axis layer to avoid hidden legend in dual-axis plots,
    - sets the peak-demand line to white with outlined markers for visibility.
  - Fixed `Experiment 6a` grouped-bar legend labels to avoid Matplotlib `_nolegend_` entries by explicitly supplying component legend text (`baseload`, `heat pump`, `EV`, `gas`) plus mean-peak line label.
  - Restored legacy `Experiment 6a` total-energy-cost vs peak-demand comparison line plots (both `peak_extreme_demand_kw` and `peak_mean_demand_kw`) alongside the component grouped-bar figure.
- `2026-03-30`:
  - Added protocol rule: never execute experiment/data-update tasks directly; always provide runnable code for user-side execution.
  - Added protocol rule reinforcement: always re-open `markdowns/main.md` at the start of every new task before any actions.
  - Updated `Experiment 6a` in `Codes/FullEnergyOptimizationDemo11.ipynb` infeasible handling:
    - excludes infeasible run curves from direct aggregation,
    - replaces each dwelling infeasible run with the dwelling's feasible-run mean load curve,
    - falls back to case-level feasible mean curve (then dwelling all-runs mean if no feasible data exists),
    - adds infeasible replacement counters to `exp6a_energy_cost_summary.csv`.
  - Updated `Experiment 6` in `Codes/FullEnergyOptimizationDemo11.ipynb` to track monovalent selected HP capacities per dwelling across all MC runs in each monovalent case folder and export `dwelling_monovalent_hp_capacity_summary.csv`.
- `2026-03-27`:
  - Renamed `markdowns/Model Structure.md` to `markdowns/main.md`.
  - Added `markdowns/key assumptions.md` containing reviewed model assumptions and experiment summaries.
  - Added protocol rule: when a model assumption changes, update `markdowns/key assumptions.md` in place.
  - Updated `Codes/Diagnose_HHP_Infeasibility.ipynb` to record selected `hp_capacity_kw` from each replayed solve and include it in output columns for both original A/B and capacity-relaxation paths.
  - Added monovalent HP-capacity tracking fields for capacity-relaxation outputs (`thermal_capacity_relaxed_hp_capacity_kw`, `both_capacity_relaxed_hp_capacity_kw`).
- `2026-03-26`:
  - Added explicit protocol rule: when user requests notebook implementation, provide runnable notebook code cells directly.
  - Added explicit protocol rule: ask for user permission before any local environment change (package/interpreter/kernel/tooling).
  - Enhanced `Codes/Diagnose_HHP_Infeasibility.ipynb` while keeping original A/B workflow:
    - added EV charging-capacity relaxation path (`ev_charge_max` lift, default 10 kW).
    - added monovalent HP-capacity upper-limit relaxation path (capacity sweep up to 15 kW).
    - added combined capacity-relaxation status reporting.
    - added export `diagnosis_capacity_relaxation_summary.csv` alongside `diagnosis_ab_test_summary.csv`.
- `2026-02-20`:
  - Added an Experiment 3 tariff override in `FullEnergyOptimizationDemo11.ipynb` so optimization can be run with flat or cozy tariffs without rebuilding the global workflow context.
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
- `2026-02-23`:
  - Added a pre-Experiment 4 pixel-convergence check in `FullEnergyOptimizationDemo11.ipynb` to estimate required MC runs.
  - Added `run_penetration_pixel_convergence_from_cache` in `stochastic_baseload_multiple_building_simulation_and_aggregation.py` to return per-run peaks and running mean.
- `2026-03-23`:
  - Added `Experiment 3a` in `FullEnergyOptimizationDemo11.ipynb` to aggregate Experiment 3 per-dwelling breakdown outputs by case and plot stacked demand curves with component breakdown.
  - Updated `Experiment 3a` case-path selection to auto-load all folders under `Output Data/Single Dwelling Runs/randomized offset`, while keeping the old manual case-path list commented in the notebook cell for reference.
  - Added randomized per-dwelling, per-day tariff-offset support to `run_monte_carlo_batch` in `stochastic_baseload_multiple_building_simulation_and_aggregation.py` via `tariff_random_offset_cfg`.
  - Added `Experiment 6` in `FullEnergyOptimizationDemo11.ipynb` for cozy-tariff Monte Carlo with randomized daily switching-point offsets and outputs saved under `Output Data/Single Dwelling Runs/randomized offset`.
  - Updated `Experiment 6` in `FullEnergyOptimizationDemo11.ipynb` to sweep offset maxima (`exp6_offset_max_hours_list`) with the same `mc_runs` across both `hybrid` and `monovalent` cases, using a single visible tqdm progress bar over all dwelling jobs.
  - Applied full-notebook comment standardization in `FullEnergyOptimizationDemo11.ipynb` with concise intent-focused comments for key variables, helper functions, and code blocks.
  - Updated `Experiment 3a` to add a shared original cozy-tariff subplot and synchronized dotted switch-point alignment lines across all subplots.
  - Integrated updated workflow governance rules:
    - always re-check this file on each prompt.
    - provide multiple options before implementation when possible.
    - wait for user selection when multiple options are presented.
    - ask clarification questions on ambiguous instructions.
    - apply concise, intent-focused code comments.
    - enforce dual change tracking (change log + in-place structure updates).
- `2026-03-24`:
  - Added `Experiment 6a` in `FullEnergyOptimizationDemo11.ipynb` to compute total electricity cost (using cozy electricity tariff as wholesale price) for each randomized-offset case folder (`case + offset`) from Experiment 6 outputs, using all runs.
  - Added CSV export for Experiment 6a summary at `Output Data/Single Dwelling Runs/randomized offset/exp6a_electricity_cost_summary.csv`.
  - Simplified `Experiment 6a` method: for each `case + offset`, first compute the mean aggregated electricity-demand curve (aggregate across dwellings per run, then average across runs), then compute total electricity cost using the original un-offset cozy tariff.
  - Extended `Experiment 6a` summary columns to include `total_gas_cost_gbp_unoffset_tariff` and `total_energy_cost_gbp_unoffset_tariff` (electricity + gas), using the same mean aggregated curves and original un-offset cozy tariff prices.
  - Added `peak_extreme_demand_kw` and `peak_mean_demand_kw` to the Experiment 6a summary table per `case + offset`.
  - Added two Experiment 6a line plots comparing both cases across offset values:
    - peak extreme demand vs total energy cost
    - peak mean demand vs total energy cost
  - Updated Experiment 6a plot axes so `x = peak demand` and `y = total energy cost` for both peak-extreme and peak-mean comparison charts.
  - Updated `build_tariff` in `Codes/sourcecode/RC_Optimization.py` to support `type='agile'`:
    - electricity tariff is loaded from `Input data/csv_agile_F_North_Eastern_England.csv` for the requested time window.
    - gas price is fixed at `6.0 p/kWh`.
  - Fixed Experiment 6 notebook variable naming collision (`exp6_tariff`) by separating tariff type string and tariff DataFrame (`exp6_tariff_type`, `exp6_tariff_df`) so output folder names no longer embed DataFrame text.
  - Adapted Experiment 6a to read tariff type from case-folder names (e.g., `agile_*`, `cozy_*`) and build the corresponding original un-offset tariff per folder before cost calculation.
  - Standardized tariff naming to `cozy` across active Experiment 3a/6/6a notebook cells and randomized-offset case folders.
  - Added `exp6a_plot_tariff_filter` in `Experiment 6a` to choose plotting scope: `all`, `agile`, or `cozy`.
  - Kept cost valuation logic in `Experiment 6a` on original un-offset tariff profiles, including explicit un-offset `agile` valuation for agile folders.
  - Updated Experiment 6a output filename to `exp6a_energy_cost_summary.csv` to reflect electricity, gas, and total energy cost columns.
  - Updated `Experiment 3a` to include both original `cozy` and original `agile` tariff subplots for the same period, with shared dotted switch markers aligned across all demand/tariff subplots.
- `2026-03-25`:
  - Added `Codes/Diagnose_HHP_Infeasibility.ipynb` as a standalone diagnostic notebook to analyze infeasible cached Experiment 6 runs by replaying exact `(dwelling, run)` seeds and comparing baseline vs EV-relaxed vs thermal-relaxed vs combined-relaxed solves.
  - Added diagnosis export contract in the notebook workflow: per-case `diagnosis_ab_test_summary.csv` written to the analyzed case folder.
  - Extended per-dwelling multi-run breakdown export in `run_monte_carlo_batch` (`stochastic_baseload_multiple_building_simulation_and_aggregation.py`) to include optimizer system variables:
    - thermal states: `Tin_C`, `T_tank_C`
    - thermal dispatch: `Q_hp_space_w`, `Q_bo_space_w`, `Q_hp_hw_w`, `Q_bo_hw_w`
    - EV state: `ev_soc_kwh`
  - Added per-folder setpoint+metadata export file `dwelling_setpoints_metadata.csv`:
    - stores `T_set_C`, `T_low_C`, `T_high_C` time series per dwelling
    - stores dwelling/model metadata and key optimization/EV/HW parameter fields.
  - Optimized setpoint/metadata CSV assembly to avoid DataFrame fragmentation warnings by batching metadata column creation via single-step DataFrame concatenation.
  - Updated `Experiment 6` notebook progress display to use notebook-friendly tqdm with forced refresh (`set_postfix_str(..., refresh=True)` and `bar.refresh()`), so progress is visible during long runs.
  - Updated `Experiment 6` to use plain-text tqdm (`from tqdm import tqdm`, stdout-backed) for more reliable progress rendering in PyCharm/Jupyter frontends.
  - Reduced warning noise during `Experiment 6`:
    - changed `dwelling_setpoints_metadata.csv` reads to `pd.read_csv(..., low_memory=False)` to avoid repeated mixed-dtype chunk warnings.
    - hardened `summary_runs` status aggregation in the notebook to skip empty/all-NA frames before `pd.concat(...)`, avoiding concat deprecation warnings.
  - Further hardened `Experiment 6` status aggregation to avoid DataFrame concatenation entirely (direct per-frame status counting), eliminating residual concat FutureWarnings.
  - Switched `Experiment 6` progress display back to Jupyter-style tqdm (`tqdm.notebook` with fallback) per user preference while keeping warning-suppression changes in place.
  - Replaced `Experiment 6` progress handling with a custom notebook widget helper (`ipywidgets.IntProgress` + label) and text-tqdm fallback, improving visibility in PyCharm/Jupyter environments where standard tqdm rendering is inconsistent.
  - Removed per-folder `dwelling_setpoints_metadata.csv` export from `run_monte_carlo_batch` to reduce heavy per-dwelling file I/O and restore faster Experiment 6 progression.
  - Reverted `Experiment 6` progress display to the simpler original single-bar `tqdm.auto` pattern in the notebook cell per user preference.
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
