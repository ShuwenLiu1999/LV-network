# Key Assumptions

## Scope
This file captures the reviewed assumptions and experiment definitions for the building-optimization workflow (primarily `Codes/FullEnergyOptimizationDemo11.ipynb` and `Codes/sourcecode/*`).

## Building Optimization Assumptions
- Building thermal model is single-zone `1R1C` linear dynamics (`Tin` state, parameters `R1`, `C1`, solar gain factor `g`).
- Heat pump COP is constant in time; boiler efficiency is constant in time.
- Tariffs are exogenous and price-taking; optimization minimizes direct energy cost only.
- Comfort bounds are hard constraints:
  - if setpoint >= 19 degC: both lower and upper bounds are enforced (`T_set +/- tol`).
  - if setpoint < 19 degC: lower bound is fixed at 15 degC; no upper bound is enforced.
- Space-heating and DHW outputs share device capacity each timestep:
  - `Q_hp_space + Q_hp_hw <= Qhp_max`
  - `Q_bo_space + Q_bo_hw <= Qbo_max`
- Hot-water demand is exogenous (volume/time-step) and must be met by system constraints.
- Hybrid defaults use boiler-only DHW (`Q_hp_hw_max = 0`) unless overridden.
- Monovalent defaults use HP storage DHW with boiler contribution disabled (`Q_bo_hw_max = 0`) unless overridden.
- Storage formulation (when enabled) uses fixed tank temperature (`T_stor = T_stor_max`) and volume-energy balance; no explicit standby-loss dynamics.
- EV availability profile is exogenous binary mask.
- EV travel energy is exogenous and subtracted from SOC each step.
- EV charging efficiency and retention are constant parameters.
- EV pre-departure target SOC is hard-constrained at the first daily unplug event; final SOC minimum fraction is hard-constrained.
- Appliance/base electric demand is non-controllable and always added to aggregate electric load.
- Thermal internal gains are exogenous and added to thermal dynamics.
- Stochastic daily profiles are sampled by seeded RNG using run index and dwelling identifier for replayability.
- Tariff random offset experiment shifts each day profile by an integer number of time steps (uniform in configured range) using deterministic seeds.
- Day-ahead mode solves per day while carrying terminal states (Tin, EV SOC, storage state) forward.
- Infeasible dwelling-run cases are retained in aggregation as appliance-only electric demand (zero HP/EV/gas contribution in run outputs).

## Data/Preprocessing Assumptions
- Weather temperature column is converted from Kelvin to degC if median suggests Kelvin scale.
- Solar irradiance/energy column is converted to power if values suggest accumulated energy.
- Appliance profile values are interpreted as kW and scaled to W when magnitude indicates that unit.
- Occupancy-specific demand CSV selection uses filename patterns (`occ1..occ5`) with fallback mapping when exactly five files exist.

## Experiment Summary (FullEnergyOptimizationDemo11)
- Baseline MC block (pre-numbered experiments):
  - Runs configured case (`hybrid` or `monovalent`) and selected dwellings.
  - Exports run CSVs and summary plots.
- Experiment 1 (EV power sweep):
  - Sweeps EV charging limit across configured values for both MHP and HHP.
  - Compares aggregated demand envelopes and feasibility status.
- Experiment 2 (HHP convergence at fixed EV power):
  - Uses fixed EV charger limit (default 5 kW) for hybrid case.
  - Supports continuation from existing run files.
  - Produces peak-demand histogram and running-mean convergence plots.
- Experiment 3 (all-dwelling breakdown runs):
  - Runs multi-run MC per dwelling and writes one per-dwelling breakdown CSV.
  - Optional per-dwelling convergence diagnostics/plots.
- Experiment 3a (aggregate from cached breakdowns):
  - Aggregates per-dwelling breakdowns by case folder and common run IDs.
  - Plots stacked demand curves with aligned tariff change markers.
- Experiment 4 pre-check (single-pixel convergence):
  - Repeats one penetration pixel to estimate MC run count sufficiency.
- Experiment 4 (EV penetration x HHP share from cache):
  - Performs grid MC mixing of hybrid/monovalent cached breakdowns.
  - Reports max-demand statistics per pixel and contour visualization.
- Experiment 5 (cache-based stacked profiles):
  - Generates per-dwelling or all-dwelling stacked demand plots from cache folders.
- Experiment 6 (randomized daily tariff offsets):
  - Sweeps offset ranges across hybrid and monovalent cases.
  - Saves per-dwelling multi-run breakdown outputs per case/offset folder.
- Experiment 6a (cost summary on original tariff):
  - Uses dwelling-level resampling over cached Experiment 6 runs: each replicate samples one run per dwelling, aggregates demand, and computes peak demand plus total energy cost under the un-offset original tariff.
  - Repeats the replicate process to report mean and 95% CI for energy cost, and mean/95% CI plus extreme value for peak demand, alongside comparison plots.

## Diagnostic Notebook Summary
- `Codes/Diagnose_HHP_Infeasibility.ipynb`:
  - Replays infeasible `(dwelling_id, run)` pairs with deterministic seeds.
  - Supports mode selection: `ab_only`, `capacity_only`, `both`.
  - A/B path: EV-target and thermal-constraint relaxations.
  - Capacity path: EV charging-cap lift and monovalent HP-cap sweep lift.
  - Outputs include replayed `hp_capacity_kw` fields for each scenario.
