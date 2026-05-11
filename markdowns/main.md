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
- `README.md`
  - Human-facing project entry point and quick orientation.
- `AGENTS.md`
  - Root-level operating instructions for future coding-agent sessions.
- Supporting data folders:
  - `Data_for_CIGRE_Network/`
  - `Modified_116_LV_CSV/`
  - `Other Demand Profiles/`
  - `UoNewCastle/`

## 2) Module catalog (what each module does)

### Core optimization and workflow

| Module | What it does | Main inputs | Main outputs | Used by |
|---|---|---|---|---|
| `Codes/sourcecode/RC_Optimization.py` | Solves thermal-energy dispatch optimization with Gurobi. Supports hybrid/monovalent heating logic, hot-water modes, EV charging constraints, and day-ahead/full-horizon solve. Builds tariffs for `flat`, `cozy`, and `agile` (Agile electricity from CSV + constant gas price), and allows optional removal of the upper indoor-temperature comfort bound for heating-only studies. | Building RC params (`R1,C1,g`), tariff, weather (`Tout,S`), comfort setpoints/tolerance, device capacities, EV and HW settings | Per-step optimal schedules (`Q_hp_space`, `Q_bo_space`, `Q_hp_hw`, `Q_bo_hw`, `P_ev_charge`, temperatures, storage states) and objective costs | `Codes/simulation.ipynb`, `Codes/analysis.ipynb`, `stochastic_baseload_multiple_building_simulation_and_aggregation.py`, demand generation scripts |
| `Codes/sourcecode/stochastic_baseload_multiple_building_simulation_and_aggregation.py` | Orchestrates end-to-end stochastic simulation workflow for many dwellings. Handles profile sampling by occupancy, EV travel synthesis, Monte Carlo runs, run aggregation, summary plots, EV-power sweep experiments, per-dwelling breakdown export, and cache-based penetration studies. Supports optional on-the-fly homogeneous EV profile generation to replace cached EV components, explicit per-pixel tqdm progress updates, randomized-tariff-offset MC workflows (`hybrid`/`monovalent`/`boiler_only`), fixed-EV penetration sweeps over mixed `HHP`/`MHP`/`boiler` shares with electricity-only peak-demand aggregation in Experiment 4a, and fixed-EV annual gas/CO2 sweeps in Experiment 4b. | Metadata CSV, weather CSV, stochastic demand profiles, configuration dictionaries (`optim_params_cfg`, `ev_params_cfg`, `hw_params_cfg`), cached single-dwelling breakdown folders (`hybrid`/`monovalent`/`boiler_only`), optional EV-generation parameter dictionary | Monte Carlo result dicts, run CSVs (optional), aggregated curves, summary tables, per-dwelling breakdown CSV, penetration-grid maximum-demand summary table, annual gas/CO2 sweep table, contour-plot-ready surface tables | `Codes/simulation.ipynb`, `Codes/analysis.ipynb` |

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

### Notebook workflow utilities

| Module | What it does | Main inputs | Main outputs | Used by |
|---|---|---|---|---|
| `Codes/sourcecode/notebook_workflow.py` | Centralizes notebook repository-root detection, canonical path bundles, output-directory creation, and compact path-status printing. | Notebook working directory or repo root | `NotebookPaths` bundle and normalized output directories | `Codes/simulation.ipynb`, `Codes/analysis.ipynb`, `Codes/Plotting illustration.ipynb` |
| `Codes/sourcecode/artifact_naming.py` | Provides shared filename-token helpers for percentages, grids, contour levels, EV-case labels, tariff-offset labels, and natural sorting. | Numeric scenario values, labels, paths | Stable filename/path tokens | `Codes/analysis.ipynb`, `Codes/Plotting illustration.ipynb` |
| `Codes/sourcecode/plotting_style.py` | Provides shared white-background Matplotlib styling, stack-plot colors, axis cleanup, and high-contrast legend helpers. | Matplotlib axes/legend objects | Styled plots and reusable color dictionaries | `Codes/Plotting illustration.ipynb`; available to analysis sections |
| `Codes/sourcecode/simulation_instance_runner.py` | Runs simulation notebook instances by expanding list-valued case/EV/offset fields, slicing context date windows, applying tariff settings, handling per-dwelling cache continuation, and writing manifest rows. | Base workflow context, `simulation_defaults`, `simulation_instances` | Per-instance output folders, per-dwelling breakdown files, optional aggregate artifacts, and simulation instance manifest CSVs | `Codes/simulation.ipynb` |

## 3) Notebook role map

`Codes/FullEnergyOptimizationDemo11.ipynb` was the former combined experiment notebook. It is no longer present in the current working tree; the active workflow is split between `Codes/simulation.ipynb` and `Codes/analysis.ipynb`.

| Notebook | Role |
|---|---|
| `Codes/simulation.ipynb` | Active run-generation notebook: setup/context build and instance-based simulation execution. Each run is specified through `simulation_instances` with case type, tariff, tariff offset, EV charger power, date window, dwelling selection, run count, continuation behavior, and output location. Uses `simulation_instance_runner.py` for per-dwelling cache and aggregate-batch execution. Technology-penetration variables are intentionally excluded from this notebook and handled later in `analysis.ipynb`. |
| `Codes/analysis.ipynb` | Active post-processing and cache-analysis notebook: file-driven follow-up calculations/visualization with per-section dataset-selection cells placed immediately before each analysis block. Runs in ordered workflow (baseline, Exp1, Exp2, Exp3a, Exp4-family generation, Exp4-pre, Exp4, Exp4a, Exp4b, Exp4c, Exp5, Exp6a, Exp6b, Exp6c), including cache-based generation from existing run outputs, annual gas/CO2 sweeps, carbon-saving line plots along peak-demand contours with multi-EV-case overlays, highest-saving segment selection, and inset percentage-change tables, plus gas-consumption post-analysis for HHP vs pure boiler. Shared path and artifact-tag helpers now live in `notebook_workflow.py` and `artifact_naming.py`. |
| `Codes/Diagnose_HHP_Infeasibility.ipynb` | Replays infeasible cached Experiment 6 `(dwelling, run)` cases and applies A/B relaxation tests (EV targets vs thermal constraints), plus capacity-limit relaxations (EV charge-cap lift and monovalent HP-cap lift), to classify likely infeasibility drivers and export diagnosis summaries including selected feasible HP capacities. |
| `Codes/Generate_Occupancy_based_demand_with_CREST_model.ipynb` | Demand profile generation and occupancy-linked preprocessing. |
| `Codes/Data Preprocessing.ipynb` | Data cleaning/transformation utilities. |
| `Codes/Calculate_Demand_batch.ipynb` | Batch demand-profile generation wrapper around source modules. |
| `Codes/Plotting illustration.ipynb` | Figure-generation and illustration notebook for cached experiment outputs, including Experiment 4 single-pixel aggregate electricity stack plots by tariff-offset case and EV-charger-capacity peak-demand sweeps from parameterized `Output Data/Simulation Cache` folders. Uses shared notebook path setup, artifact naming, and plotting-style helpers so presentation cells stay focused on figure choices. |
| `Codes/playground.ipynb` | Scratchpad notebook for small plotting/data checks. |
| `Codes/Main.ipynb`, `Codes/Test.ipynb`, `Codes/IEA_Con_Result_Analysis.ipynb` | Older scenario assembly, experimentation, and result analysis utilities. |
| `Codes/sourcecode/resul_analysis_Network.ipynb` | Legacy network-result analysis notebook. |

## 4) Output contracts

- Batch run file (when enabled): `Output Data/<subdir>/mc_run_XX.csv`
- Per-dwelling run breakdown: `Output Data/<subdir>/dwelling_<id>_runs_breakdown.csv`
- Simulation bottom-level output folders generated by `simulation_instance_runner.py` use parameterized names:
  `cache_or_batch_tariff-<tariff>_case-<case>_EV<kW>kW_offset<h>h_start<YYYYMMDD>_days<N>_runs<N>_seed<N>_dw<selection>`.
- Simulation instance manifest: `Output Data/<subdir-or-group>/simulation_instance_manifest.csv` (one row per aggregate instance or per dwelling-cache instance, including case, tariff, EV charger power, tariff offset, date window/run counts, continuation status, `output_folder`, `output_subdir`, full output path, and status counts)
- Optional per-dwelling metrics: `..._run_metrics.csv`
- Optional convergence plots: `..._convergence.png`
- Optional per-dwelling stacked consumption plots: `Output Data/<subdir>/plots/exp5_cache_stackplots/<case>/dwelling_<id>_stacked_consumption.png`
- Optional aggregate stacked consumption plot: `Output Data/<subdir>/plots/exp5_cache_stackplots/<case>/aggregate_stacked_consumption.png`
- Experiment 3a aggregated demand + tariff plot: `Output Data/plots/exp3a_scenario_<scenario>_cases_<casegroup>_<N>cases_plot_stacked_demand_with_tariffs.png`
- Experiment 4-family CSV outputs are written under `Output Data/<subdir>/csv/`; Experiment 4-family plots are written under `Output Data/<subdir>/plots/`.
- Experiment 4 pre-check convergence CSV/plot filenames include fixed EV penetration, fixed HHP penetration, run count, random seed, and EV-source tag.
- Experiment 4 EV/HHP sweep CSV filename includes EV range, grid resolution, run count, random seed, and EV-source tag; the contour plot filename additionally includes the plotted metric.
- Experiment 4a fixed-EV HHP/MHP/boiler sweep CSV filename includes fixed EV penetration, residual-boiler tag, grid resolution, run count, random seed, and EV-source tag; the contour plot filename additionally includes the plotted metric and highlighted peak levels.
- Experiment 4b annual gas/CO2 sweep CSV filename includes fixed EV penetration, residual-boiler tag, grid resolution, run count, random seed, and CO2 factor; gas and CO2 contour plot filenames additionally include the plotted metric.
- Experiment 4 single-pixel aggregate electricity stack plot from `Codes/Plotting illustration.ipynb`: `Output Data/plots/exp4_single_pixel_electricity_stack_simcache_by_tariff_offset_ci95.png`; the plotting cell discovers parameterized `Output Data/Simulation Cache/cache_tariff-..._case-..._EV..._offset...` folders from tariff, case, EV charger power, and tariff-offset variables, then renders one subplot per tariff-offset case.
- Experiment 4 EV-capacity peak-demand sweep plot from `Codes/Plotting illustration.ipynb`: `Output Data/plots/exp4_ev_capacity_sweep_peak_p97p5_tariff-<tariff>_evkw<min>-<max>_ev100_hhp50_mhp50_offsets<offset-tags>.png`; the plotting cell keeps EV penetration at 100%, mixes HHP/MHP 50/50, compares 0 h and 2 h offsets, and plots P97.5 of MC aggregate peak demand over EV charger capacity.
- Experiment 4c sampled peak-contour CO2/carbon-saving CSV filename includes EV case labels, requested peak levels, peak metric, CO2 metric, x-axis, best-carbon-segment tag, and tCO2 unit tag; rows include case labels/source CSVs and retain only the contour segment with the highest mean carbon saving when a requested peak contour splits into multiple segments.
- Experiment 4c carbon-saving line plot filename uses the same key tags as the Experiment 4c sampled CSV; the plot can overlay multiple EV-case peak-demand CSVs, displays carbon saving in tCO2, uses color for EV case and linestyle for peak-contour level, and includes a numbered inset legend table with columns `Case`, `EV %`, `Peak kW`, `HHP%`, and `Additional Saving`.
- Experiment 6a electricity/gas/energy-cost summary: `Output Data/Single Dwelling Runs/randomized offset/exp6a_energy_cost_summary.csv` (includes infeasible-handling counters when infeasible run curves are replaced by feasible-run mean dwelling curves; includes component breakdown columns for HP electricity, EV electricity, baseload electricity, and gas costs/energy; peak/cost statistics are estimated from dwelling-level resampling replicates; includes 95% CI bound columns for mean peak demand, total energy cost, and component costs used in the grouped bar chart; `peak_extreme_demand_*` bound columns are point-equal boundaries because extreme peak is reported as an extreme overlay)
- Experiment 6a plot outputs:
  - `Output Data/plots/exp6a_scenario_<scenario>_cases_<casegroup>_tariff_<tarifffilter>_plot_component_cost_with_peak_mean.png`
  - `Output Data/plots/exp6a_scenario_<scenario>_cases_<casegroup>_tariff_<tarifffilter>_plot_total_cost_vs_peak_mean.png` (mean-peak CI crosshair with overlaid dotted extreme-peak boundary line)
- Experiment 6b single-dwelling gas-convergence CSV: `Output Data/Single Dwelling Runs/all-year/<case>/exp6b_single_dwelling_gas_convergence.csv`
- Experiment 6b single-dwelling gas-convergence plot: `Output Data/plots/exp6b_single_dwelling_gas_convergence.png`
- Experiment 6c period gas per-run detail (total + space + hot-water): `Output Data/Single Dwelling Runs/<dataset_dir>/exp6c_period_gas_per_run.csv`
- Experiment 6c period gas dwelling-average table (total + space + hot-water): `Output Data/Single Dwelling Runs/<dataset_dir>/exp6c_period_gas_dwelling_average.csv`
- Experiment 6c period gas summary (HHP vs pure boiler, total + space + hot-water): `Output Data/Single Dwelling Runs/<dataset_dir>/exp6c_period_gas_summary.csv`
- Experiment 6c period gas histogram plot by component (HHP vs pure boiler): `Output Data/Single Dwelling Runs/<dataset_dir>/exp6c_period_gas_hhp_vs_boiler_hist.png`
- Experiment 6c run-feasibility pre-check matrix (`rows=runs`, `cols=dwelling+case`): `Output Data/Single Dwelling Runs/all-year/exp6c_run_feasibility_matrix.csv`
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

### D) Cross-chat maintenance standard

- Keep `README.md` as the human-facing project entry point.
- Keep `AGENTS.md` as the root-level agent entry point; it should point future coding-agent sessions back to this file.
- Keep `markdowns/chat_handoff.md` current when active workflow state, staged/unstaged work, or handoff instructions materially change.
- In a fresh chat, the first working step must still be to re-open `markdowns/main.md`; `AGENTS.md` and `chat_handoff.md` are reminders, not replacements for this file.

## 6) Structure change log

- `2026-05-11`:
  - Refreshed documentation entry points with a human-facing `README.md`, root-level `AGENTS.md`, and `markdowns/chat_handoff.md` for cross-chat continuity.
  - Updated the active notebook map to treat `Codes/simulation.ipynb` and `Codes/analysis.ipynb` as the current split workflow, with `Codes/FullEnergyOptimizationDemo11.ipynb` no longer present in the working tree.
  - Documented Experiment 4b annual gas/CO2 cache sweeps and Experiment 4c carbon-saving analysis outputs in the module catalog, notebook role map, and output contracts.
  - Added cross-chat maintenance rules to keep future agent sessions aligned with `markdowns/main.md`.
  - Updated `Codes/analysis.ipynb` Experiment 4c so split peak-demand contours keep only the segment with the highest mean carbon saving, and so multiple EV-case peak-demand CSVs (for example 10% and 40% EV) can be overlaid in the same carbon-saving plot.
  - Updated `Codes/analysis.ipynb` Experiment 4c carbon-saving plot to include an inset table summarizing each plotted line's percentage increase from near-0% HHP to max HHP.
  - Refined `Codes/analysis.ipynb` Experiment 4c plot styling: table text is forced to high-contrast black, carbon saving is plotted in tCO2, and legends explicitly map color to EV case and linestyle to peak-contour level.
  - Updated `Codes/analysis.ipynb` Experiment 4c inset legend table to number plotted cases and show separate `EV %`, `Peak kW`, `HHP%`, and `Additional Saving` columns.
  - Updated `Codes/analysis.ipynb` Experiment 4-family output paths so CSV files write under `Output Data/Penetration Sweep/csv/`, plots write under `Output Data/Penetration Sweep/plots/`, and filenames include key scenario variables while omitting HHP/MHP range tags.
  - Updated `Codes/Plotting illustration.ipynb` Experiment 4 single-pixel aggregate electricity stack plot so it can render multiple tariff-offset cases as subplots from `exp4_pixel_tariff_offset_cases`.
  - Updated `Codes/Plotting illustration.ipynb` Experiment 4 single-pixel aggregate electricity stack plot to discover the new parameterized `Output Data/Simulation Cache` folders by tariff, case, EV charger power, and tariff offset, with optional run/seed/dwelling-selection filters and an updated `simcache` output filename.
  - Added a `Codes/Plotting illustration.ipynb` Experiment 4 EV-capacity sweep plot that keeps EV penetration at 100%, HHP/MHP at 50/50, compares 0 h and 2 h tariff offsets, and plots P97.5 of aggregate peak demand against EV charger capacity from Simulation Cache folders.
  - Streamlined the three active notebooks by adding shared setup/artifact/style modules (`notebook_workflow.py`, `artifact_naming.py`, `plotting_style.py`), updating notebook opening contracts, and moving repeated path, filename-token, and plotting-style glue out of notebook cells.
  - Tidied `Codes/simulation.ipynb` into an instance-driven runner: separate model defaults from run instances, preview expanded plans before execution, gate expensive execution with `run_simulation_now`, and move continuation/output/manifest plumbing into `Codes/sourcecode/simulation_instance_runner.py`.
  - Updated simulation instance output naming so generated bottom-level output folders encode the key run parameters (`mode`, tariff, case, EV charger power or EV-off state, tariff offset, start date, duration, target runs, seed, and dwelling selection), with `output_folder` and `output_subdir` recorded in the preview table and manifest.

- `2026-04-21`:
  - Added optimizer flag `enforce_upper_comfort_bound` in `Codes/sourcecode/RC_Optimization.py` so the upper indoor-temperature comfort bound can be disabled while preserving the lower bound.
  - Exposed `enforce_upper_comfort_bound` in `Codes/simulation.ipynb` through `optim_params_cfg` for notebook-level control during simulation runs.

- `2026-04-20`:
  - Updated Experiment 6c in `analysis.ipynb` to compute annual gas statistics at dwelling level first (mean annual gas per dwelling over selected runs), then compare `hybrid` vs `boiler_only` using case histograms.
  - Added Experiment 6c dwelling-average CSV output (`exp6c_annual_gas_dwelling_average.csv`) and renamed the Exp 6c figure output to `exp6c_annual_gas_hhp_vs_boiler_hist.png`.
  - Added an Experiment 6c pre-check cell in `analysis.ipynb` to scan all selected case/dwelling breakdown files and export a run-feasibility matrix CSV (`exp6c_run_feasibility_matrix.csv`) with rows as run IDs and columns as dwelling+case identifiers.
  - Standardized Experiment 6c feasibility-matrix column names to `<dwelling_id_3digits>_<case>` (for example `001_hybrid`, `001_boiler_only`).
  - Updated Experiment 6c gas metric from annualized values to dataset-period aggregate values and split gas into space-heating and hot-water components (using `Q_bo_space_w`/`Q_bo_hw_w` shares of `boiler_gas_kw` per step), with updated period-output filenames.

- `2026-04-17`:
  - Added `Codes/simulation.ipynb` and `Codes/analysis.ipynb` as an explicit workflow split of `Codes/FullEnergyOptimizationDemo11.ipynb`.
  - Refined split to remove plotting/post-processing overlap from `simulation.ipynb`; simulation now exports run outputs plus manifest CSV artifacts only.
  - Added/updated analysis sections in `analysis.ipynb` for file-driven follow-up metrics/plots from simulation artifacts (baseline checks, Experiment 1, Experiment 2, and Experiment 4 pre-check, plus existing Experiment 3a/4/4a/5/6a analysis blocks).
  - Moved cache-based Experiment 4-family execution (pre-check, Experiment 4, Experiment 4a CSV generation from existing breakdown runs) from `simulation.ipynb` into `analysis.ipynb`; simulation now only keeps a handoff note for these sections.
  - Merged standalone Experiment 3 run-generation flow into the unified Experiment 6 runner in `simulation.ipynb` (Exp3-equivalent achieved via single-case + zero-offset settings).
  - Extended unified Experiment 6 tariff selector in `simulation.ipynb` to explicitly support `flat` alongside `cozy` and `agile`.
  - Reordered `analysis.ipynb` sections into experiment order and added explicit dataset-selection configuration cells immediately before each analysis block.
  - Updated analysis blocks to consume the preceding configuration cells (instead of hard-coded dataset paths), including configurable Exp3a/Exp4-family/Exp5/Exp6a dataset inputs.
  - Added Experiment 6b and Experiment 6c analysis blocks in `analysis.ipynb` for all-year gas post-processing (single-dwelling run-mean convergence and HHP-vs-boiler annual gas comparison with CSV and plot outputs in `Output Data/Single Dwelling Runs/all-year`).
  - Updated Experiment 6c in `analysis.ipynb` with `exp6c_max_runs_per_dwelling` (default `5`) so annual-gas comparison can read only the first N run IDs from each dwelling breakdown file.
  - Kept `Codes/FullEnergyOptimizationDemo11.ipynb` in place as the original combined notebook alongside the split notebooks.

- `2026-04-10`:
  - Added `run_hhp_mhp_boiler_penetration_experiment_from_cache` in `Codes/sourcecode/stochastic_baseload_multiple_building_simulation_and_aggregation.py` for fixed EV penetration with mixed `HHP` + `MHP` + residual `boiler` share pixels (`HHP+MHP<=100%`), including compatibility with generated homogeneous EV profiles.
  - Extended `_load_case_breakdown_cache` to retain parsed component arrays (`hp_by_dwelling`, `appliance_by_dwelling`) in addition to existing combined base and EV arrays, enabling boiler-share sampling from appliance-only load.
  - Added `Experiment 4a` in `Codes/FullEnergyOptimizationDemo11.ipynb` to run and export fixed-EV penetration sweeps over `HHP` (y-axis) and `MHP` (x-axis) shares with boiler remainder, plus contour visualization aligned to Experiment 4 style.
  - Updated Experiment 4a cache mixing to use an explicit `boiler_only` cache folder while keeping peak-demand aggregation electricity-only (boiler dwellings contribute appliance + EV electric demand only).
  - Extended `_load_case_breakdown_cache` parsing to include per-dwelling `boiler_gas_kw` arrays (`boiler_by_dwelling`) with backward-compatible zero fallback when the column is absent.
  - Enforced Experiment 4a penetration-grid constraint at sampling stage so only valid pixels with `HHP + MHP <= 100%` are simulated (invalid combinations are no longer emitted as NaN result rows).

- `2026-04-09`:
  - Added explicit `boiler_only` case support in `run_monte_carlo_batch` (`Codes/sourcecode/stochastic_baseload_multiple_building_simulation_and_aggregation.py`) with default HP-disabled settings (`Qhp_max=0`, `Q_hp_hw_max=0`) and boiler-supplied space/DHW heating.
  - Updated Experiment 6 in `Codes/FullEnergyOptimizationDemo11.ipynb` to include `boiler_only` in the same randomized offset sweep as `hybrid` and `monovalent`, using case-specific configuration mapping (`exp6_case_cfg`) for optimization and hot-water settings.
  - Updated the notebook-level Monte Carlo configuration block to include `boiler_only` case configuration (`mc_case_cfg`) for baseline/single-case runs.
  - Updated Experiment 6a folder parsing and component-cost plotting to include `boiler_only` outputs as a third case in the case panel plot.

- `2026-04-06`:
  - Updated `Experiment 6a` in `Codes/FullEnergyOptimizationDemo11.ipynb` to compute peak-demand and total-energy-cost statistics via dwelling-level resampling (each resample draws one run per dwelling, aggregates, and repeats for CI/mean/extreme estimates).
  - Updated Experiment 6a peak-vs-total-cost plotting to keep 95% CI crosshair error bars for mean peak demand and total cost, while overlaying extreme peak demand as a dotted boundary line in the same mean-peak plot.
  - Updated Experiment 6a grouped component-cost vs peak-demand plot to render 95% CI error bars on component bars, plus mean-peak 95% CI and extreme-peak dotted overlays on the peak line axis.
  - Removed the separate `total_cost_vs_peak_extreme` plot output from the standard Experiment 6a output contract.
  - Extended `exp6a_energy_cost_summary.csv` with CI bound columns for component costs (HP, EV, baseload, gas) in addition to mean peak demand and total energy cost, plus extreme-peak point-equal boundary columns and bootstrap sample count metadata.
- `2026-04-03`:
  - Updated `Experiment 6` in `Codes/FullEnergyOptimizationDemo11.ipynb` to support a user-defined total run target (`exp6_target_total_runs`) per dwelling-case-offset combination.
  - Added existing-run detection for continuation using breakdown CSV run statistics so reruns only execute missing runs needed to reach the total target (rather than always adding a fixed batch size).
  - Extended Experiment 6 summary output columns to include existing runs, runs added this execution, target runs, and final total runs per dwelling.
- `2026-04-02`:
  - Inspected and aligned plotting outputs to repository-level plot folder structure under `Output Data/plots`.
  - Updated `Experiment 3a` in `Codes/FullEnergyOptimizationDemo11.ipynb` to save the aggregated multi-case demand+tariff figure to `Output Data/plots` with scenario/case/plot-type naming.
  - Updated `Experiment 6a` in `Codes/FullEnergyOptimizationDemo11.ipynb` to save all generated figures (component grouped bars + two legacy peak-vs-total-cost lines) to `Output Data/plots` with scenario/case/tariff-filter/plot-type naming.
  - Updated `Experiment 6` in `Codes/FullEnergyOptimizationDemo11.ipynb` continuation workflow so reruns add new MC runs to existing per-dwelling breakdown files (merge + dedupe by `run,time`) instead of overwriting prior data, and so monovalent HP-capacity summary CSV values are accumulated across existing + newly added runs.
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
