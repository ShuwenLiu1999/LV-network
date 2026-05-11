# Chat Handoff

Last updated: 2026-05-11 22:25:24 +01:00

## Start Procedure

For any fresh chat or resumed agent session:

1. Read `AGENTS.md`.
2. Re-open and read `markdowns/main.md`.
3. Read this file.
4. Run a lightweight status check such as `git status --short`.
5. Inspect code/notebook structure before editing; avoid raw data/output inspection unless needed.

## Current Workflow

- Active run-generation notebook: `Codes/simulation.ipynb`.
- Active analysis notebook: `Codes/analysis.ipynb`.
- Diagnostic notebook: `Codes/Diagnose_HHP_Infeasibility.ipynb`.
- Former combined notebook: `Codes/FullEnergyOptimizationDemo11.ipynb`; it is currently absent from the working tree and staged as deleted.
- The three active workflow notebooks have been streamlined with shared helper modules:
  - `Codes/sourcecode/notebook_workflow.py` for repo-root/path setup, output-directory creation, and path-status printing.
  - `Codes/sourcecode/artifact_naming.py` for stable artifact filename tokens and natural sorting.
  - `Codes/sourcecode/plotting_style.py` for shared white-background Matplotlib style, stack colors, axis cleanup, and legend styling.
  - `Codes/sourcecode/simulation_instance_runner.py` for simulation-instance expansion, context slicing, per-dwelling cache continuation, and manifest writing.
- `Codes/simulation.ipynb` has been tidied into an instance-driven runner. It now separates shared model defaults from `simulation_instances`, previews the expanded run plan, and gates execution behind `run_simulation_now`. Technology penetration is intentionally excluded from the simulation notebook and remains an analysis/cache-mixing concern.
- Simulation instance output folders are now generated systematically from run parameters, using tokens for mode, tariff, case, EV charger power or EV-off state, tariff offset, start date, duration, target runs, seed, and dwelling selection. The preview table and manifest include `output_folder` and `output_subdir` for later analysis lookup.
- `Codes/analysis.ipynb` and `Codes/Plotting illustration.ipynb` now include clearer opening notebook contracts and use the shared setup helpers where appropriate. The plotting notebook now starts with a setup cell before figure cells.
- `Codes/analysis.ipynb` Experiment 4c now supports overlaying multiple EV-case peak-demand CSVs in one carbon-saving plot; default cases are 10% EV and 40% EV. When a requested peak-demand contour has multiple segments, the analysis keeps only the segment with the highest mean carbon saving. The plot shows carbon saving in tCO2, maps color to EV case and linestyle to peak-contour level, and includes a high-contrast numbered inset legend table with columns `Case`, `EV %`, `Peak kW`, `HHP%`, and `Additional Saving`.
- Experiment 4-family outputs in `Codes/analysis.ipynb` now write CSV artifacts under `Output Data/Penetration Sweep/csv/` and plot artifacts under `Output Data/Penetration Sweep/plots/`. Filenames include key variables such as EV penetration/range, grid size, run count, seed, EV source, metric, highlighted peak levels, CO2 factor, Exp 4c EV cases, and Exp 4c axis/segment/unit tags; HHP/MHP range tags are intentionally omitted.
- `Codes/Plotting illustration.ipynb` Experiment 4 single-pixel aggregate electricity stack plot now discovers parameterized `Output Data/Simulation Cache/cache_tariff-..._case-..._EV..._offset...` folders from tariff, case, EV charger power, and tariff-offset variables, then renders one subplot per tariff-offset case. The current defaults use agile, 5 kW EV, `0` and `2` hour offsets, with optional run/seed/dwelling-selection filters, and save to `Output Data/plots/exp4_single_pixel_electricity_stack_simcache_by_tariff_offset_ci95.png` when run.
- `Codes/Plotting illustration.ipynb` now also has an Experiment 4 EV-capacity sweep plot immediately after the single-pixel stack plot. It keeps EV penetration at 100%, HHP/MHP at 50/50, compares `0` and `2` hour offsets, sweeps EV charger capacity `[3, 5, 7, 9, 11]` kW from `Output Data/Simulation Cache`, and plots P97.5 of MC aggregate peak demand. It saves to `Output Data/plots/exp4_ev_capacity_sweep_peak_p97p5_tariff-agile_evkw3-11_ev100_hhp50_mhp50_offsets0p0h_2p0h.png` when run.

## Observed Working Tree State

Observed at 2026-05-11 16:42:00 +01:00:

- Staged deletion: `Codes/FullEnergyOptimizationDemo11.ipynb`.
- Unstaged notebook changes: `Codes/simulation.ipynb`, `Codes/analysis.ipynb`, and `Codes/Plotting illustration.ipynb`.
- Untracked helper modules: `Codes/sourcecode/notebook_workflow.py`, `Codes/sourcecode/artifact_naming.py`, `Codes/sourcecode/plotting_style.py`, and `Codes/sourcecode/simulation_instance_runner.py`.
- Unstaged generated-output changes:
  - `Codes/Output/DemandProfiles/figures/1/hp_heat_proportion_hist_hhp.png`
  - `Output Data/Penetration Sweep/exp4c_peak_contour_carbon_saving_line.png`
  - `Output Data/Penetration Sweep/exp4c_peak_contour_co2_samples.csv`

Re-check this state in future sessions because it may change outside this handoff file.

## Maintenance Checklist

- If code/notebook structure changes, update `markdowns/main.md` structure sections and Section 6 change log.
- If assumptions change, update `markdowns/key assumptions.md`.
- If active workflow state or pending work changes materially, update this file.
- Keep `README.md` human-facing and concise.
- Keep `AGENTS.md` agent-facing and concise.

## New Chat Prompt

Use this prompt when moving to a different chat:

```text
You are working in the LV-network repository. Before doing anything else, read AGENTS.md, then re-open markdowns/main.md and follow its prompt-handling protocol. After that, read markdowns/chat_handoff.md for current workflow state. Start by running a lightweight git status check. Do not inspect raw data/output files unless explicitly needed; map structure via filenames, notebook headings, and source function summaries. If code/notebook structure changes, update markdowns/main.md Section 6 and the affected structure/output sections in the same session. If model assumptions change, update markdowns/key assumptions.md too.
```
