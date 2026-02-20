"""Workflow helpers for FullEnergyOptimizationDemo11 notebook.

This module keeps heavy logic out of the notebook so notebook cells only keep
configuration and execution calls.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence
from collections import Counter
import re

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from RC_Optimization import build_tariff, optimize_full_energy_system


DEFAULT_META_COLUMNS = {"occupants": "No_occupants", "R1": "R1", "C1": "C1", "g": "g"}
DEFAULT_WEATHER_COLUMNS = {"time": "timestamp", "t2m": "t2m", "ssrd": "ssrd"}
DEFAULT_PROFILE_COLUMN_MAP = {
    "hotwater": ["hotwater_L"],
    "appliance": ["appliance_demand_W"],
    "ev_availability": ["ev_available"],
    "thermal_gains": ["Thermal Gains_W"],
}


def find_repo_root(start: Path | None = None) -> Path:
    """Walk upward from start and return the first folder containing .git."""
    path = Path(start or Path.cwd()).resolve()
    for candidate in [path] + list(path.parents):
        if (candidate / ".git").exists():
            return candidate
    return path


def _load_daily_samples_from_array(data: np.ndarray, steps_per_day: int) -> tuple[np.ndarray, int]:
    """Convert profile vector/matrix to daily samples and return the sample axis."""
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    if data.shape[0] == steps_per_day:
        return data, 1
    if data.shape[1] == steps_per_day:
        return data, 0
    flat = data.reshape(-1)
    if flat.size % steps_per_day == 0:
        return flat.reshape(-1, steps_per_day), 0
    raise ValueError(f"Cannot infer daily samples from shape {data.shape}")


def _select_profile_by_index(samples: np.ndarray, axis: int, idx: int, steps_per_day: int) -> np.ndarray:
    """Pick one daily profile and enforce consistent step count."""
    profile = samples[:, idx] if axis == 1 else samples[idx, :]
    if len(profile) < steps_per_day:
        return np.pad(profile, (0, steps_per_day - len(profile)), mode="edge")
    if len(profile) > steps_per_day:
        return profile[:steps_per_day]
    return profile


def find_occ_profile_file(root: Path, occ: int) -> Path:
    """Find the stochastic profile file for a given occupant count."""
    root = Path(root)
    csv_paths = sorted(root.glob("*.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found in {root}")

    occ = int(occ)
    for path in csv_paths:
        name = path.stem.lower()
        if f"occ{occ}" in name or f"occ_{occ}" in name or f"occ-{occ}" in name:
            return path

    # Common fallback: five files map to occupant 1..5 by sorted order.
    if len(csv_paths) == 5 and 1 <= occ <= 5:
        return csv_paths[occ - 1]

    raise FileNotFoundError(f"No occupant-specific CSV found for occ={occ} in {root}")


def sample_stochastic_profiles(
    path: Path,
    n_days: int,
    steps_per_day: int,
    rng: np.random.Generator,
    profile_column_map: Mapping[str, Sequence[str]],
) -> tuple[dict[str, np.ndarray], dict[str, list[int]]]:
    """Sample daily stochastic profiles for each required demand type."""
    df = pd.read_csv(path)
    profile_data = {k: [] for k in profile_column_map}
    chosen_indices: list[int] = []

    for _ in range(n_days):
        run_idx = int(rng.integers(0, 1000))
        chosen_indices.append(run_idx)
        resolved: list[tuple[str, np.ndarray, int]] = []
        min_days: int | None = None

        for key, cols in profile_column_map.items():
            col = f"run_{run_idx}_{cols[0]}"
            if col not in df.columns:
                raise KeyError(f"Expected column {col} in {path}")

            series = pd.to_numeric(df[col], errors="coerce").fillna(0).to_numpy()
            samples, axis = _load_daily_samples_from_array(series, steps_per_day)
            resolved.append((key, samples, axis))
            n_candidates = samples.shape[1] if axis == 1 else samples.shape[0]
            min_days = n_candidates if min_days is None else min(min_days, n_candidates)

        if not min_days:
            raise ValueError(f"No daily samples in {path} for run {run_idx}")

        day_idx = int(rng.integers(0, min_days))
        for key, samples, axis in resolved:
            profile_data[key].append(_select_profile_by_index(samples, axis, day_idx, steps_per_day))

    for key in profile_column_map:
        profile_data[key] = np.concatenate(profile_data[key])

    return profile_data, {key: chosen_indices for key in profile_column_map}


def _load_weather(weather_path: Path, weather_columns: Mapping[str, str], step: str) -> pd.DataFrame:
    """Load weather CSV and normalize it to a datetime index."""
    weather = pd.read_csv(weather_path, comment="#")
    time_col = weather_columns["time"]
    if time_col in weather.columns:
        weather[time_col] = pd.to_datetime(weather[time_col], errors="coerce", dayfirst=True)
        weather = weather.dropna(subset=[time_col]).set_index(time_col)
    else:
        weather.index = pd.date_range("2020-01-01", periods=len(weather), freq=step)
    return weather.sort_index()


def build_setpoint_sequences(
    tariff_index: pd.DatetimeIndex,
    comfort_start_hour: int = 8,
    comfort_end_hour: int = 21,
    include_flex: bool = False,
) -> list[np.ndarray]:
    """Build one or two set-point schedules used by optimization."""
    hours = tariff_index.hour + tariff_index.minute / 60.0
    comfort = np.where((hours >= comfort_start_hour) & (hours < comfort_end_hour), 21.0, 17.0)
    if not include_flex:
        return [comfort]
    flex = np.where(((hours >= 6) & (hours < 8)) | ((hours >= 18) & (hours < 21)), 20.0, 16.0)
    return [comfort, flex]

def prepare_workflow_inputs(
    repo_root: Path | None = None,
    data_root: Path | None = None,
    meta_path: Path | None = None,
    weather_path: Path | None = None,
    profile_root: Path | None = None,
    *,
    step: str = "30min",
    start_date: pd.Timestamp | str | None = None,
    n_days: int = 7,
    tariff_type: str = "cosy",
    max_dwellings: int | None = None,
    random_seed: int = 42,
    meta_columns: Mapping[str, str] | None = None,
    weather_columns: Mapping[str, str] | None = None,
    profile_column_map: Mapping[str, Sequence[str]] | None = None,
    include_flex_setpoint: bool = False,
) -> dict[str, Any]:
    """Prepare metadata, weather, tariff, and dwelling-level stochastic inputs."""
    repo_root = Path(repo_root or find_repo_root())
    data_root = Path(data_root or (repo_root / "Input data"))
    meta_path = Path(meta_path or (data_root / "Newcastle_Urban_Case_meta_updated.csv"))
    weather_path = Path(weather_path or (data_root / "NEcase_20_21_t2m_ssrd_30min.csv"))
    profile_root = Path(profile_root or (data_root / "Stochastic_Demands"))

    meta_columns = dict(DEFAULT_META_COLUMNS if meta_columns is None else meta_columns)
    weather_columns = dict(DEFAULT_WEATHER_COLUMNS if weather_columns is None else weather_columns)
    profile_column_map = dict(DEFAULT_PROFILE_COLUMN_MAP if profile_column_map is None else profile_column_map)

    steps_per_day = int(pd.Timedelta("1D") / pd.Timedelta(step))
    meta = pd.read_csv(meta_path)
    weather = _load_weather(weather_path, weather_columns, step)

    temp_col, solar_col = weather_columns["t2m"], weather_columns["ssrd"]
    if temp_col not in weather.columns:
        raise KeyError(f"Column {temp_col!r} not found in weather CSV")
    if solar_col not in weather.columns:
        raise KeyError(f"Column {solar_col!r} not found in weather CSV")

    # Convert to expected physical units for the optimization model.
    tout = weather[temp_col].astype(float)
    if tout.median() > 100:
        tout = tout - 273.15
    solar = weather[solar_col].astype(float)

    if len(weather.index) < 2:
        raise ValueError("Weather data must have at least two rows")
    dt_seconds = (weather.index[1] - weather.index[0]).total_seconds()
    if solar.max() > 2000:
        solar = solar / dt_seconds

    start_ts = weather.index[0].normalize() if start_date is None else pd.Timestamp(start_date)
    end_ts = start_ts + pd.Timedelta(days=int(n_days))
    window = weather.loc[start_ts : end_ts - pd.Timedelta(step)]
    if window.empty:
        raise ValueError(f"Weather data does not cover {start_ts:%Y-%m-%d} to {end_ts:%Y-%m-%d}")

    tout_arr = tout.loc[window.index].to_numpy()
    solar_arr = solar.loc[window.index].to_numpy()

    tariff = build_tariff(window.index[0], n_days=int(n_days), step=step, type=tariff_type)
    if not tariff.index.equals(window.index):
        tariff = tariff.reindex(window.index, method="ffill")

    setpoint_sequences = build_setpoint_sequences(tariff.index, include_flex=include_flex_setpoint)

    occ_col = meta_columns["occupants"]
    r_col = meta_columns["R1"]
    c_col = meta_columns["C1"]
    g_col = meta_columns["g"]

    meta_iter = meta.head(max_dwellings).copy() if max_dwellings else meta
    rng = np.random.default_rng(random_seed)

    dwelling_inputs: dict[Any, dict[str, Any]] = {}
    for idx, row in meta_iter.iterrows():
        occ = int(row[occ_col])
        profile_path = find_occ_profile_file(profile_root, occ)
        profiles, chosen = sample_stochastic_profiles(
            profile_path, int(n_days), steps_per_day, rng, profile_column_map
        )

        appliance = profiles["appliance"]
        # If appliance values are very small, treat them as kW and convert to W.
        base_electric = appliance * 1000 if np.nanmax(appliance) < 50 else appliance

        dwelling_inputs[idx] = {
            "occ": occ,
            "meta": row,
            "hw_demand": profiles["hotwater"],
            "base_electric": base_electric,
            "thermal_gains": profiles["thermal_gains"],
            "ev_availability": profiles["ev_availability"],
            "samples": chosen,
            "profile_path": str(profile_path),
        }

    return {
        "repo_root": repo_root,
        "data_root": data_root,
        "meta_path": meta_path,
        "weather_path": weather_path,
        "profile_root": profile_root,
        "meta": meta,
        "weather": weather,
        "window": window,
        "step": step,
        "steps_per_day": steps_per_day,
        "n_days": int(n_days),
        "dt_seconds": float(dt_seconds),
        "tariff": tariff,
        "n_steps": len(tariff),
        "Tout": tout_arr,
        "S": solar_arr,
        "setpoint_sequences": setpoint_sequences,
        "meta_columns": meta_columns,
        "weather_columns": weather_columns,
        "profile_column_map": profile_column_map,
        "occ_col": occ_col,
        "r_col": r_col,
        "c_col": c_col,
        "g_col": g_col,
        "dwelling_inputs": dwelling_inputs,
    }


def summarize_dwelling_inputs(dwelling_inputs: Mapping[Any, Mapping[str, Any]], n_preview: int = 3) -> pd.DataFrame:
    """Build a compact preview table for notebook display."""
    rows: list[dict[str, Any]] = []
    for i, (d_id, d_input) in enumerate(dwelling_inputs.items()):
        rows.append(
            {
                "dwelling_id": d_id,
                "occupants": d_input["occ"],
                "profile_path": d_input["profile_path"],
                "sampled_runs_preview": d_input["samples"]["hotwater"][:3],
            }
        )
        if i + 1 >= int(n_preview):
            break
    return pd.DataFrame(rows)


def _stable_dwelling_seed(dwelling_id: Any) -> int:
    """Convert dwelling identifiers to deterministic integer seeds."""
    try:
        return int(dwelling_id)
    except Exception:
        return abs(hash(str(dwelling_id))) % 100_000


def _compute_ev_travel_energy(
    ev_availability: np.ndarray,
    tariff_index: pd.DatetimeIndex,
    n_days: int,
    *,
    rng_seed: int,
    energy_per_mile_kwh: float = 0.25,
) -> np.ndarray:
    """Create per-step EV travel energy from stochastic daily mileage."""
    mean_miles = 2.61 * 8.28
    sd_miles = 14.92
    mean_kwh = mean_miles * energy_per_mile_kwh
    sd_kwh = sd_miles * energy_per_mile_kwh
    shape = (mean_kwh / sd_kwh) ** 2
    scale = (sd_kwh**2) / mean_kwh

    rng = np.random.default_rng(rng_seed)
    daily_ev_kwh = rng.gamma(shape, scale, size=int(n_days))
    ev_travel_energy = np.zeros(len(tariff_index))

    # Spread each day's travel energy over away timesteps.
    for day, day_start in enumerate(tariff_index.normalize().unique()):
        day_mask = tariff_index.normalize() == day_start
        away_idx = np.where(day_mask & (ev_availability < 0.5))[0]
        if len(away_idx) == 0:
            continue
        ev_travel_energy[away_idx] = daily_ev_kwh[day] / len(away_idx)

    return ev_travel_energy


def run_single_dwelling_case(
    context: Mapping[str, Any],
    selected_dwelling_id: Any,
    *,
    params_config: Mapping[str, Any] | None = None,
    ev_config: Mapping[str, Any] | None = None,
    hw_config: Mapping[str, Any] | None = None,
    setpoint_sequences: Sequence[np.ndarray] | None = None,
    day_ahead: bool = True,
) -> dict[str, Any]:
    """Run one dwelling optimization with user-defined parameter dictionaries."""
    dwelling_inputs = context["dwelling_inputs"]
    if selected_dwelling_id not in dwelling_inputs:
        selected_dwelling_id = int(selected_dwelling_id)
    selected = dwelling_inputs[selected_dwelling_id]

    row = selected["meta"]
    params = {
        "R1": float(row[context["r_col"]]),
        "C1": float(row[context["c_col"]]),
        "g": float(row[context["g_col"]]),
        "dt": context["dt_seconds"],
        "T0": 21.0,
        "tol": 1.0,
        "COP": 3.5,
        "etaB": 0.9,
        "Qhp_max": 7000.0,
        "Qbo_max": 0.0,
    }
    if params_config:
        params.update(dict(params_config))

    hw_demand_m3 = np.asarray(selected["hw_demand"], dtype=float) / 1000.0
    base_electric = np.asarray(selected["base_electric"], dtype=float)
    thermal_gains = np.asarray(selected["thermal_gains"], dtype=float)
    ev_availability = np.asarray(selected["ev_availability"], dtype=float)

    ev_params = {
        "ev_capacity": 60.0,
        "ev_target": 30.0,
        "ev_charge_max": 2.0,
        "eta_ev_charge": 0.95,
        "ev_soc_init": 30.0,
        "ev_min_final_fraction": 0.5,
    }
    if ev_config:
        ev_params.update(dict(ev_config))

    ev_seed = int(ev_params.pop("ev_seed", 123))
    ev_travel_energy = _compute_ev_travel_energy(
        ev_availability, context["tariff"].index, context["n_days"], rng_seed=ev_seed
    )
    ev_params["ev_availability"] = ev_availability
    ev_params["ev_travel_energy"] = ev_travel_energy

    hw_params = {
        "hw_mode": "hp_storage",
        "V_stor": 0.2,
        "V_stor_init": 0.12,
        "T_mains": 10.0,
        "T_hw_supply": 40.0,
    }
    if hw_config:
        hw_params.update(dict(hw_config))
    if hw_params.get("hw_mode") == "boiler_only":
        hw_params["V_stor"] = 0.0
        hw_params["V_stor_init"] = 0.0

    schedules = list(setpoint_sequences) if setpoint_sequences is not None else context["setpoint_sequences"]
    results = optimize_full_energy_system(
        tariff=context["tariff"],
        Tout=context["Tout"],
        S=context["S"],
        setpoint_sequences=schedules,
        hw_demand=hw_demand_m3,
        base_electric=base_electric,
        thermal_gains=thermal_gains,
        day_ahead=bool(day_ahead),
        **params,
        **ev_params,
        **hw_params,
    )

    best_key = results.get("best_key", next(k for k in results if k.startswith("schedule_")))
    if "best_result" in results:
        best, best_cost = results["best_result"]["results"], results["best_result"]["cost"]
    else:
        best, best_cost = results[best_key]["results"], results[best_key]["cost"]

    return {
        "selected_dwelling_id": selected_dwelling_id,
        "selected": selected,
        "params": params,
        "ev_params": ev_params,
        "hw_params": hw_params,
        "hw_demand_m3": hw_demand_m3,
        "base_electric": base_electric,
        "thermal_gains": thermal_gains,
        "results": results,
        "best_key": best_key,
        "best": best,
        "best_cost": best_cost,
    }

def compute_storage_balance_residual(
    best: pd.DataFrame,
    hw_demand_m3: np.ndarray,
    dt_seconds: float,
    hw_params: Mapping[str, Any],
) -> pd.Series:
    """Compute storage-volume energy-balance residual for diagnostics."""
    if not np.isfinite(best["V_stor"]).any():
        return pd.Series(dtype=float)

    T_mains = float(hw_params.get("T_mains", 10.0))
    T_hw_supply = float(hw_params.get("T_hw_supply", 40.0))
    T_stor_max = float(hw_params.get("T_stor_max", 55.0))
    eta_hw_store = float(hw_params.get("eta_hw_store", 1.0))
    C_hw = float(hw_params.get("C_hw", 4180.0))
    rho_hw = float(hw_params.get("rho_hw", 1000.0))
    vstor_initial = float(hw_params.get("V_stor_init", hw_params.get("V_stor", 0.0)))

    if T_stor_max == T_mains:
        hw_draw_equiv_m3 = hw_demand_m3
    else:
        hw_draw_equiv_m3 = (
            hw_demand_m3
            * (T_hw_supply - T_mains)
            / (T_stor_max - T_mains)
            / max(eta_hw_store, 1e-6)
        )
    hw_draw_energy = hw_draw_equiv_m3 * (T_stor_max - T_mains) * C_hw * rho_hw

    q_hw = (best["Q_hp_hw"] + best["Q_bo_hw"]).to_numpy()
    vstor = best["V_stor"].to_numpy()
    v_prev = np.r_[vstor_initial, vstor[:-1]]
    lhs = (T_stor_max - T_mains) * C_hw * rho_hw * (vstor - v_prev)
    rhs = q_hw * dt_seconds - hw_draw_energy

    return pd.Series(lhs - rhs, index=best.index, name="storage_balance_residual_J")


def plot_single_dwelling_detail(
    *,
    best: pd.DataFrame,
    tariff: pd.DataFrame,
    params: Mapping[str, Any],
    ev_params: Mapping[str, Any],
    hw_params: Mapping[str, Any],
    base_electric: np.ndarray,
    thermal_gains: np.ndarray,
    hw_demand_m3: np.ndarray,
    case_label: str = "Demo",
    mc_runs: int = 1,
    dwellings_count: int = 1,
) -> None:
    """Detailed multi-panel plot for one dwelling solution."""
    use_storage = np.isfinite(best["T_stor"]).any()

    T_mains = float(hw_params.get("T_mains", 10.0))
    T_hw_supply = float(hw_params.get("T_hw_supply", 40.0))
    T_stor_max = float(hw_params.get("T_stor_max", 55.0))
    eta_hw_store = float(hw_params.get("eta_hw_store", 1.0))
    if T_stor_max == T_mains:
        hw_draw_equiv_m3 = hw_demand_m3
    else:
        hw_draw_equiv_m3 = (
            hw_demand_m3
            * (T_hw_supply - T_mains)
            / (T_stor_max - T_mains)
            / max(eta_hw_store, 1e-6)
        )

    hw_draw_series = pd.Series(hw_draw_equiv_m3, index=best.index)
    hw_draw_plot = hw_draw_series.shift(-1).fillna(0.0)

    n_rows = 8 if use_storage else 7
    fig, axs = plt.subplots(n_rows, 1, figsize=(12, 22 if use_storage else 20), sharex=True)
    fig.suptitle(
        f"Detailed Results - case: {case_label}, runs: {mc_runs}, dwellings: {dwellings_count}", y=0.995
    )

    axs[0].plot(tariff.index, tariff["elec_price"], label="Electricity price (p/kWh)")
    axs[0].plot(tariff.index, tariff["gas_price"], label="Gas price (p/kWh)")
    axs[0].legend(loc="upper right")
    axs[0].set_ylabel("Tariff")
    axs[0].grid(True, alpha=0.3)

    axs[1].plot(best.index, best["Tin"], label="Indoor temp")
    axs[1].plot(best.index, best["T_set"], "--", label="Set-point")
    axs[1].fill_between(best.index, best["T_low"], best["T_high"], color="lightblue", alpha=0.3, label="Comfort band")
    axs[1].legend(loc="upper right")
    axs[1].set_ylabel("Temperature (C)")
    axs[1].grid(True, alpha=0.3)

    axs[2].plot(best.index, best["Q_hp_space"] / 1000, label="HP space (kW)")
    axs[2].plot(best.index, best["Q_bo_space"] / 1000, label="Boiler space (kW)")
    axs[2].plot(best.index, best["Q_hp_hw"] / 1000, label="HP hot water (kW)")
    axs[2].plot(best.index, best["Q_bo_hw"] / 1000, label="Boiler hot water (kW)")
    axs[2].legend(loc="upper right")
    axs[2].set_ylabel("Thermal output (kW)")
    axs[2].grid(True, alpha=0.3)

    hw_ax = axs[3]
    hw_ax.plot(best.index, (best["Q_hp_hw"] + best["Q_bo_hw"]) / 1000, label="HW heat (kW)")
    hw_ax.set_ylabel("HW heat (kW)")
    hw_ax.grid(True, alpha=0.3)
    hw_ax.legend(loc="upper left")

    demand_ax = hw_ax.twinx()
    demand_ax.bar(best.index, hw_draw_plot, width=0.02, alpha=0.3, color="tab:green", label="HW demand equiv (m3)")
    demand_ax.set_ylabel("HW demand equiv (m3)")
    demand_ax.legend(loc="upper right")

    gain_ax = axs[4]
    gain_ax.plot(best.index, thermal_gains / 1000, label="Thermal gains (kW)")
    gain_ax.legend(loc="upper right")
    gain_ax.set_ylabel("Heat gains (kW)")
    gain_ax.grid(True, alpha=0.3)

    if use_storage:
        tank_ax = axs[5]
        vol_ax = tank_ax.twinx()
        tank_ax.plot(best.index, best["T_stor"], label="Storage water temp (C)")
        tank_ax.axhline(hw_params.get("T_mains", np.nan), color="k", linestyle=":", label="Mains temp")
        vol_ax.plot(best.index, best["V_stor"], color="tab:orange", label="Stored volume (m3)")
        vol_ax.bar(best.index, hw_draw_plot, width=0.02, alpha=0.3, color="tab:green", label="HW demand equiv (m3)")
        tank_ax.legend(loc="upper left")
        vol_ax.legend(loc="upper right")
        tank_ax.set_ylabel("Tank temp (C)")
        vol_ax.set_ylabel("Stored volume (m3)")
        tank_ax.grid(True, alpha=0.3)

    ev_ax = axs[6] if use_storage else axs[5]
    plugin_series = pd.Series(ev_params["ev_availability"], index=best.index)
    plugged = plugin_series > 0.5
    if plugged.any():
        spans = []
        start = None
        for ts, avail in plugged.items():
            if avail and start is None:
                start = ts
            elif (not avail) and start is not None:
                spans.append((start, ts))
                start = None
        if start is not None:
            spans.append((start, plugin_series.index[-1]))
        for i, (s, e) in enumerate(spans):
            ev_ax.axvspan(s, e, color="gray", alpha=0.2, label="Plug-in window" if i == 0 else None)

    ev_ax.plot(best.index, best["ev_soc"], label="EV SOC (kWh)")
    ev_ax.bar(best.index, best["P_ev_charge"], width=0.02, alpha=0.4, label="EV charge (kW)")
    ev_ax.legend(loc="upper right")
    ev_ax.set_ylabel("EV metrics")
    ev_ax.grid(True, alpha=0.3)

    heat_pump_elec = (best["Q_hp_space"] + best["Q_hp_hw"]) / float(params["COP"]) / 1000
    other_elec = base_electric / 1000
    ev_elec = best["P_ev_charge"]
    gas_input = (best["Q_bo_space"] + best["Q_bo_hw"]) / float(params["etaB"]) / 1000

    agg_ax = axs[7] if use_storage else axs[6]
    agg_ax.plot(best.index, heat_pump_elec + other_elec + ev_elec, label="Electric load (kW)")
    agg_ax.plot(best.index, gas_input, label="Gas input (kW)")
    agg_ax.legend(loc="upper right")
    agg_ax.set_ylabel("Power (kW)")
    agg_ax.grid(True, alpha=0.3)

    plt.xlabel("Time")
    plt.tight_layout()
    plt.show()


def _resolve_dwellings_to_run(
    dwelling_inputs: Mapping[Any, Mapping[str, Any]],
    selected_dwellings: Sequence[Any] | Any | None,
) -> list[tuple[Any, Mapping[str, Any]]]:
    """Resolve selection input to a concrete list of dwellings."""
    if selected_dwellings is None:
        return list(dwelling_inputs.items())
    selected_list = [selected_dwellings] if isinstance(selected_dwellings, (int, str)) else list(selected_dwellings)
    out = []
    for dwelling_id in selected_list:
        resolved_id = dwelling_id if dwelling_id in dwelling_inputs else int(dwelling_id)
        out.append((resolved_id, dwelling_inputs[resolved_id]))
    return out


def run_monte_carlo_batch(
    context: Mapping[str, Any],
    *,
    mc_runs: int = 10,
    run_index_start: int = 0,
    case: str = "monovalent",
    output_subdir: str = "Test",
    selected_dwellings: Sequence[Any] | Any | None = None,
    capacity_candidates_kw: Sequence[float] | None = None,
    optim_params_cfg: Mapping[str, Any] | None = None,
    ev_params_cfg: Mapping[str, Any] | None = None,
    hw_params_cfg: Mapping[str, Any] | None = None,
    single_dwelling_id: Any | None = None,
    single_dwelling_output_path: str | Path | None = None,
    day_ahead: bool = True,
    save_outputs: bool = True,
    show_progress: bool = True,
) -> dict[str, Any]:
    """Run batch Monte Carlo and save per-run CSV outputs.

    Override dictionaries let the notebook control model parameters directly.
    Precedence is: internal defaults -> case defaults -> user overrides.
    `run_index_start` controls deterministic continuation:
    run indices `[run_index_start, ..., run_index_start + mc_runs - 1]`
    are used for RNG seeds and output filenames (`mc_run_XX.csv`).
    """
    tariff = context["tariff"]
    n_steps = context["n_steps"]
    n_days = context["n_days"]
    steps_per_day = context["steps_per_day"]
    repo_root = context["repo_root"]
    dwelling_inputs = context["dwelling_inputs"]
    setpoint_sequences = context["setpoint_sequences"]
    profile_root = context["profile_root"]
    r_col, c_col, g_col = context["r_col"], context["c_col"], context["g_col"]
    dt_seconds = context["dt_seconds"]
    Tout, S = context["Tout"], context["S"]
    optim_params_cfg = dict(optim_params_cfg or {})
    ev_params_cfg = dict(ev_params_cfg or {})
    hw_params_cfg = dict(hw_params_cfg or {})

    case = str(case).lower().strip()
    capacity_candidates = np.asarray(capacity_candidates_kw if capacity_candidates_kw is not None else np.arange(7.0, 10.5, 0.5), dtype=float)

    dwellings_to_run = _resolve_dwellings_to_run(dwelling_inputs, selected_dwellings)
    mc_runs = int(mc_runs)
    run_index_start = int(run_index_start)
    if mc_runs < 0:
        raise ValueError("mc_runs must be >= 0")
    if run_index_start < 0:
        raise ValueError("run_index_start must be >= 0")
    output_dir = repo_root / "Output Data" / output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    case_label = "HHP" if case == "hybrid" else ("MHP" if case == "monovalent" else str(case))

    is_single = (mc_runs == 1) and (len(dwellings_to_run) == 1)
    single_detail = None

    summary_runs, profile_usage = [], []
    agg_curves, agg_gas_curves, agg_hp_curves, agg_ev_curves, agg_app_curves = [], [], [], [], []
    failure_counter: Counter[str] = Counter()
    single_dwelling_rows: list[pd.DataFrame] = []
    if single_dwelling_output_path is not None:
        single_dwelling_output_path = Path(single_dwelling_output_path)

    run_iter = range(run_index_start, run_index_start + mc_runs)
    if show_progress:
        run_iter = tqdm(
            run_iter,
            desc=f"MC runs [{run_index_start + 1}..{run_index_start + mc_runs}]",
            position=0,
            leave=True,
            dynamic_ncols=True,
        )

    for run in run_iter:
        agg_elec_kw = np.zeros(n_steps)
        agg_gas_kw = np.zeros(n_steps)
        agg_hp_kw = np.zeros(n_steps)
        agg_ev_kw = np.zeros(n_steps)
        agg_app_kw = np.zeros(n_steps)
        summary_rows = []
        run_elec_cols, run_gas_cols = {}, {}

        dwell_iter = dwellings_to_run
        # Avoid a redundant nested bar when only one dwelling is being simulated.
        if show_progress and len(dwellings_to_run) > 1:
            dwell_iter = tqdm(dwellings_to_run, desc=f"Dwellings (run {run+1})", position=1, leave=False, dynamic_ncols=True)

        for dwelling_id, d_input in dwell_iter:
            if single_dwelling_id is not None and dwelling_id != single_dwelling_id:
                continue
            occ = int(d_input["occ"])
            profile_path = find_occ_profile_file(profile_root, occ)

            rng_profiles = np.random.default_rng(10_000 * run + _stable_dwelling_seed(dwelling_id))
            profiles, chosen = sample_stochastic_profiles(profile_path, n_days, steps_per_day, rng_profiles, context["profile_column_map"])

            hw_demand_m3 = np.asarray(profiles["hotwater"], dtype=float) / 1000.0
            appliance_profile = np.asarray(profiles["appliance"], dtype=float)
            base_electric = appliance_profile * 1000 if np.nanmax(appliance_profile) < 50 else appliance_profile
            thermal_gains = np.asarray(profiles["thermal_gains"], dtype=float)
            ev_availability = np.asarray(profiles["ev_availability"], dtype=float)
            # Appliance demand does not depend on optimization feasibility.
            other_elec = base_electric / 1000.0
            agg_app_kw += np.asarray(other_elec)
            agg_elec_kw += np.asarray(other_elec)

            for day_idx, run_idx in enumerate(chosen["hotwater"]):
                profile_usage.append({"run": run, "dwelling_id": dwelling_id, "day": day_idx, "stochastic_run_idx": run_idx})

            ev_params = {
                "ev_capacity": 60.0,
                "ev_soc_init": 0.8 * 60.0,
                "ev_target": 0.8 * 60.0,
                "ev_charge_max": 3.0,
                "eta_ev_charge": 0.95,
                "ev_min_final_fraction": 0.8,
                "ev_retention": 0.999,
            }
            ev_params.update(ev_params_cfg)

            # Always align EV availability to the sampled profile unless explicitly overridden.
            if "ev_availability" in ev_params_cfg:
                ev_availability_arr = np.asarray(ev_params_cfg["ev_availability"], dtype=float)
            else:
                ev_availability_arr = ev_availability
            if len(ev_availability_arr) != n_steps:
                raise ValueError("ev_availability override must match simulation horizon length.")
            ev_params["ev_availability"] = ev_availability_arr

            # Travel profile can be fully overridden, or generated from seeded stochastic mileage.
            if "ev_travel_energy" in ev_params_cfg:
                ev_travel_energy = np.asarray(ev_params_cfg["ev_travel_energy"], dtype=float)
                if len(ev_travel_energy) != n_steps:
                    raise ValueError("ev_travel_energy override must match simulation horizon length.")
            else:
                ev_seed = int(ev_params_cfg.get("ev_seed", 1_000 * run + _stable_dwelling_seed(dwelling_id)))
                ev_travel_energy = _compute_ev_travel_energy(
                    ev_availability_arr,
                    tariff.index,
                    n_days,
                    rng_seed=ev_seed,
                )
            ev_params["ev_travel_energy"] = ev_travel_energy
            # ev_seed is a convenience config key and should not be passed to the optimizer.
            ev_params.pop("ev_seed", None)

            if case == "hybrid":
                cap_candidates, qbo_max_kw = np.array([4.0]), 30.0
                hw_params = {"hw_mode": "boiler_only", "V_stor": 0.0, "V_stor_init": 0.0, "T_mains": 10.0, "T_hw_supply": 40.0, "Q_hp_hw_max": 0.0}
            elif case == "monovalent":
                cap_candidates, qbo_max_kw = capacity_candidates, 0.0
                hw_params = {"hw_mode": "hp_storage", "V_stor": 0.2, "V_stor_init": 0.12, "T_mains": 10.0, "T_hw_supply": 40.0, "Q_bo_hw_max": 0.0}
            else:
                raise ValueError("case must be either 'hybrid' or 'monovalent'")
            hw_params.update(hw_params_cfg)
            if str(hw_params.get("hw_mode", "")).lower() == "boiler_only":
                hw_params["V_stor"] = 0.0
                hw_params["V_stor_init"] = 0.0

            missing_cols = [col for col in (r_col, c_col, g_col) if col not in d_input["meta"].index]
            if missing_cols:
                raise KeyError(f"Missing meta columns for dwelling {dwelling_id}: {missing_cols}")

            # Allow fixed capacity overrides from configuration by collapsing sweep values.
            qhp_override = optim_params_cfg.get("Qhp_max")
            if qhp_override is not None:
                cap_candidates = np.array([float(qhp_override) / 1000.0], dtype=float)

            qbo_override = optim_params_cfg.get("Qbo_max")
            qbo_max_w = float(qbo_override) if qbo_override is not None else float(qbo_max_kw) * 1000.0

            solved = False
            last_failure_reason = ""
            for cap_kw in cap_candidates:
                params = {
                    "R1": float(d_input["meta"][r_col]),
                    "C1": float(d_input["meta"][c_col]),
                    "g": float(d_input["meta"][g_col]),
                    "dt": dt_seconds,
                    "T0": 21.0,
                    "tol": 1.0,
                    "COP": 3.5,
                    "etaB": 0.9,
                    "Qhp_max": float(cap_kw) * 1000.0,
                    "Qbo_max": qbo_max_w,
                }
                # Apply optimization parameter overrides after defaults.
                for key, value in optim_params_cfg.items():
                    if key not in {"Qhp_max", "Qbo_max"}:
                        params[key] = value

                try:
                    results = optimize_full_energy_system(
                        tariff=tariff,
                        Tout=Tout,
                        S=S,
                        setpoint_sequences=setpoint_sequences,
                        hw_demand=hw_demand_m3,
                        base_electric=base_electric,
                        thermal_gains=thermal_gains,
                        day_ahead=bool(day_ahead),
                        **params,
                        **ev_params,
                        **hw_params,
                    )
                except RuntimeError as exc:
                    # Capture solver failure reason (e.g. infeasible status code) for diagnostics.
                    last_failure_reason = str(exc)
                    failure_counter[last_failure_reason] += 1
                    continue

                best_key = results.get("best_key", next(k for k in results if k.startswith("schedule_")))
                if "best_result" in results:
                    best, best_cost = results["best_result"]["results"], results["best_result"]["cost"]
                else:
                    best, best_cost = results[best_key]["results"], results[best_key]["cost"]

                if is_single:
                    single_detail = {
                        "best": best,
                        "tariff": tariff,
                        "params": params,
                        "ev_params": ev_params,
                        "hw_params": hw_params,
                        "base_electric": base_electric,
                        "thermal_gains": thermal_gains,
                        "hw_demand_m3": hw_demand_m3,
                    }

                heat_pump_elec = (best["Q_hp_space"] + best["Q_hp_hw"]) / params["COP"] / 1000
                ev_elec = best["P_ev_charge"]
                gas_input = (best["Q_bo_space"] + best["Q_bo_hw"]) / params["etaB"] / 1000

                agg_hp_kw += np.asarray(heat_pump_elec)
                agg_ev_kw += np.asarray(ev_elec)
                # Appliance load was already added above for every dwelling.
                agg_elec_kw += np.asarray(heat_pump_elec + ev_elec)
                agg_gas_kw += np.asarray(gas_input)

                summary_rows.append({
                    "run": run,
                    "dwelling_id": dwelling_id,
                    "hp_capacity_kw": float(cap_kw),
                    "best_cost": float(best_cost),
                    "solve_status": "optimal",
                    "failure_reason": "",
                })
                run_elec_cols[f"dwelling_{dwelling_id}"] = (heat_pump_elec + other_elec + ev_elec).to_numpy()
                run_gas_cols[f"dwelling_{dwelling_id}"] = gas_input.to_numpy()
                if single_dwelling_output_path is not None:
                    single_dwelling_rows.append(
                        pd.DataFrame(
                            {
                                "time": tariff.index,
                                "run": int(run + 1),
                                "dwelling_id": dwelling_id,
                                "hp_elec_kw": np.asarray(heat_pump_elec, dtype=float),
                                "boiler_gas_kw": np.asarray(gas_input, dtype=float),
                                "ev_charge_kw": np.asarray(ev_elec, dtype=float),
                                "appliance_kw": np.asarray(other_elec, dtype=float),
                                "solve_status": "optimal",
                            }
                        )
                    )
                solved = True
                break

            if not solved:
                # Keep unsolved dwellings in outputs with appliance-only electric profile.
                summary_rows.append({
                    "run": run,
                    "dwelling_id": dwelling_id,
                    "hp_capacity_kw": None,
                    "best_cost": None,
                    "solve_status": "infeasible",
                    "failure_reason": last_failure_reason,
                })
                run_elec_cols[f"dwelling_{dwelling_id}"] = np.asarray(other_elec, dtype=float)
                run_gas_cols[f"dwelling_{dwelling_id}"] = np.zeros(n_steps, dtype=float)
                if single_dwelling_output_path is not None:
                    single_dwelling_rows.append(
                        pd.DataFrame(
                            {
                                "time": tariff.index,
                                "run": int(run + 1),
                                "dwelling_id": dwelling_id,
                                "hp_elec_kw": np.zeros(n_steps, dtype=float),
                                "boiler_gas_kw": np.zeros(n_steps, dtype=float),
                                "ev_charge_kw": np.zeros(n_steps, dtype=float),
                                "appliance_kw": np.asarray(other_elec, dtype=float),
                                "solve_status": "infeasible",
                            }
                        )
                    )

        if save_outputs:
            run_df = pd.DataFrame(run_elec_cols, index=tariff.index)
            run_df.columns = [f"elec_kw_{c}" for c in run_df.columns]
            run_gas_df = pd.DataFrame(run_gas_cols, index=tariff.index)
            run_gas_df.columns = [f"gas_kw_{c}" for c in run_gas_df.columns]
            out_df = pd.concat([run_df, run_gas_df], axis=1)
            out_df.index.name = "time"
            out_df.to_csv(output_dir / f"mc_run_{run + 1:02d}.csv")

        agg_curves.append(agg_elec_kw.copy())
        agg_gas_curves.append(agg_gas_kw.copy())
        agg_hp_curves.append(agg_hp_kw.copy())
        agg_ev_curves.append(agg_ev_kw.copy())
        agg_app_curves.append(agg_app_kw.copy())
        summary_runs.append(pd.DataFrame(summary_rows))

    if single_dwelling_output_path is not None and single_dwelling_rows:
        single_dwelling_output_path.parent.mkdir(parents=True, exist_ok=True)
        single_df = pd.concat(single_dwelling_rows, ignore_index=True)
        single_df.to_csv(single_dwelling_output_path, index=False)

    return {
        "summary_runs": summary_runs,
        "profile_usage": profile_usage,
        "agg_curves": agg_curves,
        "agg_gas_curves": agg_gas_curves,
        "agg_hp_curves": agg_hp_curves,
        "agg_ev_curves": agg_ev_curves,
        "agg_app_curves": agg_app_curves,
        "dwellings_to_run": dwellings_to_run,
        "is_single": is_single,
        "single_detail": single_detail,
        "case_label": case_label,
        "mc_runs": mc_runs,
        "run_index_start": run_index_start,
        "output_dir": output_dir,
        "setpoint_sequences": setpoint_sequences,
        "failure_reasons": dict(failure_counter),
    }

def plot_profile_usage_table(profile_usage: Sequence[Mapping[str, Any]], dwelling_id: Any | None = None) -> pd.DataFrame:
    """Return profile-usage table, optionally filtered to one dwelling."""
    df = pd.DataFrame(list(profile_usage))
    if dwelling_id is not None and not df.empty:
        return df.loc[df["dwelling_id"] == dwelling_id]
    return df


def _compute_temperature_band(
    setpoint_sequences: Sequence[np.ndarray] | np.ndarray,
    n_steps: int,
    tol_for_plot: float,
) -> tuple[bool, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """Build set-point mean and comfort envelope for summary plot."""
    if isinstance(setpoint_sequences, np.ndarray) and setpoint_sequences.ndim == 1:
        sequence_candidates = [setpoint_sequences]
    else:
        sequence_candidates = [np.asarray(seq) for seq in setpoint_sequences]

    valid_sequences = []
    for seq in sequence_candidates:
        seq_arr = np.asarray(seq).reshape(-1)
        if len(seq_arr) == n_steps:
            valid_sequences.append(seq_arr.astype(float))

    if not valid_sequences:
        return False, None, None, None

    stack = np.vstack(valid_sequences)
    low_stack = np.where(stack >= 19.0, stack - tol_for_plot, 15.0)
    high_stack = np.where(stack >= 19.0, stack + tol_for_plot, np.nan)

    setpoint_mean = np.nanmean(stack, axis=0)
    comfort_lower = np.nanmin(low_stack, axis=0)
    comfort_upper = np.full(n_steps, np.nan)
    for i in range(n_steps):
        vals = high_stack[:, i][np.isfinite(high_stack[:, i])]
        if vals.size:
            comfort_upper[i] = vals.max()
    comfort_upper = np.where(np.isfinite(comfort_upper), comfort_upper, setpoint_mean)

    return True, setpoint_mean, comfort_lower, comfort_upper


def plot_monte_carlo_summary(
    mc_results: Mapping[str, Any],
    tariff: pd.DataFrame,
    *,
    setpoint_sequences: Sequence[np.ndarray] | np.ndarray | None = None,
    tol_for_plot: float = 1.0,
    envelope_mode: str = "percentile",
    envelope_low_pct: float = 10.0,
    envelope_high_pct: float = 90.0,
    save_path: str | Path | None = None,
    save_dpi: int = 300,
) -> None:
    """Plot MC summary or single-dwelling detail depending on run settings.

    By default this uses percentile envelopes instead of raw min/max to avoid
    noisy spikes dominating the chart when stochastic runs diverge sharply.
    """
    if mc_results["is_single"] and mc_results["single_detail"] is not None:
        detail = mc_results["single_detail"]
        plot_single_dwelling_detail(
            best=detail["best"],
            tariff=detail["tariff"],
            params=detail["params"],
            ev_params=detail["ev_params"],
            hw_params=detail["hw_params"],
            base_electric=detail["base_electric"],
            thermal_gains=detail["thermal_gains"],
            hw_demand_m3=detail["hw_demand_m3"],
            case_label=mc_results["case_label"],
            mc_runs=mc_results["mc_runs"],
            dwellings_count=len(mc_results["dwellings_to_run"]),
        )
        return

    hp_array = np.vstack(mc_results["agg_hp_curves"])
    ev_array = np.vstack(mc_results["agg_ev_curves"])
    app_array = np.vstack(mc_results["agg_app_curves"])
    hp_mean = hp_array.mean(axis=0)
    ev_mean = ev_array.mean(axis=0)
    app_mean = app_array.mean(axis=0)
    total_mean = hp_mean + ev_mean + app_mean

    agg_array = np.vstack(mc_results["agg_curves"])
    mode = str(envelope_mode).strip().lower()
    if mode == "extreme":
        low_curve = agg_array.min(axis=0)
        high_curve = agg_array.max(axis=0)
        low_label = "Lower extreme"
        high_label = "Upper extreme"
    else:
        # Percentile envelope is robust to outlier runs and easier to read.
        low_curve = np.percentile(agg_array, float(envelope_low_pct), axis=0)
        high_curve = np.percentile(agg_array, float(envelope_high_pct), axis=0)
        low_label = f"P{envelope_low_pct:g}"
        high_label = f"P{envelope_high_pct:g}"

    elec_prices = tariff["elec_price"].to_numpy()
    gas_prices = tariff["gas_price"].to_numpy()
    change_mask = np.zeros_like(elec_prices, dtype=bool)
    change_mask[1:] = (elec_prices[1:] != elec_prices[:-1]) | (gas_prices[1:] != gas_prices[:-1])
    change_times = tariff.index[change_mask]

    n_steps = len(tariff.index)
    source_sequences = setpoint_sequences if setpoint_sequences is not None else mc_results.get("setpoint_sequences", [])
    temp_ok, setpoint_mean, comfort_lower, comfort_upper = _compute_temperature_band(
        source_sequences, n_steps, float(tol_for_plot)
    )

    case_label_norm = str(mc_results.get("case_label", "")).strip().lower()
    include_hhp_gas = case_label_norm in {"hhp", "hybrid"}
    gas_mean: np.ndarray | None = None
    if include_hhp_gas and mc_results.get("agg_gas_curves"):
        gas_array = np.vstack(mc_results["agg_gas_curves"])
        gas_mean = gas_array.mean(axis=0)

    if include_hhp_gas:
        fig, (ax_main, ax_temp, ax_gas, ax_tariff) = plt.subplots(
            4, 1, sharex=True, figsize=(11, 9), gridspec_kw={"height_ratios": [3, 1.2, 1.2, 1]}
        )
    else:
        fig, (ax_main, ax_temp, ax_tariff) = plt.subplots(
            3, 1, sharex=True, figsize=(11, 8), gridspec_kw={"height_ratios": [3, 1.2, 1]}
        )
        ax_gas = None

    ax_main.stackplot(
        tariff.index, hp_mean, ev_mean, app_mean, labels=["Heat pump", "EV", "Appliance"], alpha=0.6
    )
    ax_main.plot(tariff.index, total_mean, color="k", linewidth=1.5, label="Total (mean)")
    ax_main.plot(tariff.index, low_curve, color="tab:red", linewidth=1.2, linestyle="--", label=low_label)
    ax_main.plot(tariff.index, high_curve, color="tab:green", linewidth=1.2, linestyle="--", label=high_label)
    for t in change_times:
        ax_main.axvline(t, color="k", linestyle=":", alpha=0.35)
    ax_main.set_title(
        f"Aggregated Electricity Consumption - case: {mc_results['case_label']}, "
        f"runs: {mc_results['mc_runs']}, dwellings: {len(mc_results['dwellings_to_run'])}"
    )
    ax_main.set_ylabel("Power (kW)")
    ax_main.grid(True, alpha=0.3)
    ax_main.legend(ncol=2)

    if temp_ok and setpoint_mean is not None and comfort_lower is not None and comfort_upper is not None:
        ax_temp.plot(
            tariff.index,
            setpoint_mean,
            linestyle="--",
            color="tab:blue",
            linewidth=1.2,
            label="Set-point (mean)",
        )
        ax_temp.fill_between(
            tariff.index,
            comfort_lower,
            comfort_upper,
            color="lightblue",
            alpha=0.4,
            label="Temperature constraint band",
        )
    else:
        ax_temp.text(
            0.5,
            0.5,
            "Temperature constraints unavailable",
            transform=ax_temp.transAxes,
            ha="center",
            va="center",
        )

    for t in change_times:
        ax_temp.axvline(t, color="k", linestyle=":", alpha=0.35)
    ax_temp.set_title(
        f"Temperature Constraints - case: {mc_results['case_label']}, "
        f"runs: {mc_results['mc_runs']}, dwellings: {len(mc_results['dwellings_to_run'])}"
    )
    ax_temp.set_ylabel("Temperature (C)")
    ax_temp.grid(True, alpha=0.3)
    if ax_temp.get_legend_handles_labels()[0]:
        ax_temp.legend(loc="upper right")

    if ax_gas is not None:
        if gas_mean is None:
            gas_mean = np.zeros(n_steps, dtype=float)
        ax_gas.plot(tariff.index, gas_mean, color="tab:orange", linewidth=1.5, label="Boiler gas input (mean)")
        for t in change_times:
            ax_gas.axvline(t, color="k", linestyle=":", alpha=0.35)
        ax_gas.set_title(
            f"Average Boiler Gas Consumption - case: {mc_results['case_label']}, "
            f"runs: {mc_results['mc_runs']}, dwellings: {len(mc_results['dwellings_to_run'])}"
        )
        ax_gas.set_ylabel("Gas input (kW)")
        ax_gas.grid(True, alpha=0.3)
        ax_gas.legend(loc="upper right")

    ax_tariff.step(tariff.index, tariff["elec_price"], where="post", label="Electricity price (p/kWh)")
    ax_tariff.step(tariff.index, tariff["gas_price"], where="post", label="Gas price (p/kWh)")
    for t in change_times:
        ax_tariff.axvline(t, color="k", linestyle=":", alpha=0.35)
    ax_tariff.set_title(
        f"Tariff - case: {mc_results['case_label']}, "
        f"runs: {mc_results['mc_runs']}, dwellings: {len(mc_results['dwellings_to_run'])}"
    )
    ax_tariff.set_ylabel("Tariff (p/kWh)")
    ax_tariff.grid(True, alpha=0.3)
    ax_tariff.legend(loc="upper right")
    ax_tariff.set_xlabel("Time")

    plt.tight_layout()
    if save_path is not None:
        out_path = Path(save_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=save_dpi, bbox_inches="tight")
    plt.show()


def _load_breakdown_arrays(
    breakdown_path: str | Path,
    *,
    only_optimal: bool = False,
) -> tuple[
    pd.DatetimeIndex,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[str, int],
    Any | None,
    np.ndarray,
]:
    """Load a per-dwelling breakdown CSV and return stacked component arrays."""
    df = pd.read_csv(breakdown_path)
    if "time" not in df.columns:
        raise ValueError(f"Breakdown file missing 'time' column: {breakdown_path}")
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    if df["time"].isna().all():
        raise ValueError(f"Could not parse time column in {breakdown_path}")

    if only_optimal and "solve_status" in df.columns:
        df = df[df["solve_status"].astype(str).str.lower() == "optimal"]
        if df.empty:
            raise ValueError(f"No optimal rows remaining in {breakdown_path}")

    required = {"run", "hp_elec_kw", "ev_charge_kw", "appliance_kw"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Breakdown file missing required columns {sorted(missing)}: {breakdown_path}")

    status_counts: dict[str, int] = {}
    if "solve_status" in df.columns:
        status_counts = (
            df["solve_status"]
            .astype(str)
            .str.lower()
            .value_counts(dropna=False)
            .to_dict()
        )

    dwelling_id = None
    if "dwelling_id" in df.columns and not df.empty:
        dwelling_id = df["dwelling_id"].iloc[0]

    runs = sorted(df["run"].unique())
    if not runs:
        raise ValueError(f"No runs found in {breakdown_path}")

    hp_list, ev_list, app_list = [], [], []
    time_index = None
    n_steps = None
    for run in runs:
        run_df = df[df["run"] == run].sort_values("time")
        if time_index is None:
            time_index = pd.DatetimeIndex(run_df["time"])
            n_steps = len(time_index)
        elif len(run_df) != n_steps:
            raise ValueError(f"Inconsistent step count in {breakdown_path} for run {run}")
        hp_list.append(run_df["hp_elec_kw"].to_numpy(dtype=float))
        ev_list.append(run_df["ev_charge_kw"].to_numpy(dtype=float))
        app_list.append(run_df["appliance_kw"].to_numpy(dtype=float))

    return (
        time_index,
        np.vstack(hp_list),
        np.vstack(ev_list),
        np.vstack(app_list),
        status_counts,
        dwelling_id,
        np.asarray(runs, dtype=int),
    )


def _infer_step_str(time_index: pd.DatetimeIndex) -> str:
    """Infer a pandas frequency string from a datetime index."""
    if time_index.freq is not None:
        return time_index.freqstr
    if len(time_index) < 2:
        return "30min"
    delta = time_index[1] - time_index[0]
    total_seconds = int(delta.total_seconds())
    if total_seconds % 60 == 0:
        minutes = total_seconds // 60
        if minutes % 60 == 0:
            hours = minutes // 60
            return f"{hours}H"
        return f"{minutes}min"
    return f"{total_seconds}s"


def _build_tariff_for_time_index(time_index: pd.DatetimeIndex, tariff_type: str) -> pd.DataFrame:
    """Build a tariff DataFrame aligned to the provided time index."""
    tariff_norm = str(tariff_type).strip().lower()
    if tariff_norm == "cozy":
        tariff_norm = "cosy"
    if len(time_index) < 2:
        raise ValueError("Time index must have at least two points to build a tariff.")
    delta = time_index[1] - time_index[0]
    steps_per_day = int(round(pd.Timedelta("1D") / delta))
    if steps_per_day <= 0:
        steps_per_day = len(time_index)
    n_days = int(round(len(time_index) / steps_per_day))
    step = _infer_step_str(time_index)
    tariff = build_tariff(time_index[0], n_days=n_days, step=step, type=tariff_norm)
    if not tariff.index.equals(time_index):
        tariff = tariff.reindex(time_index, method="ffill")
    return tariff


def _tariff_change_times(tariff: pd.DataFrame) -> pd.DatetimeIndex:
    """Return timestamps where tariff prices change (for alignment markers)."""
    elec = tariff["elec_price"].to_numpy()
    gas = tariff["gas_price"].to_numpy()
    change_mask = np.zeros_like(elec, dtype=bool)
    change_mask[1:] = (elec[1:] != elec[:-1]) | (gas[1:] != gas[:-1])
    return tariff.index[change_mask]


def plot_dwelling_stackplot_from_breakdown(
    breakdown_path: str | Path,
    *,
    output_path: str | Path | None = None,
    envelope_mode: str = "percentile",
    envelope_low_pct: float = 10.0,
    envelope_high_pct: float = 90.0,
    only_optimal: bool = False,
    title_prefix: str | None = None,
    title: str | None = None,
    tariff_type: str | None = None,
    save_dpi: int = 300,
) -> dict[str, Any]:
    """Plot stacked consumption for one dwelling from its breakdown CSV."""
    time_index, hp_array, ev_array, app_array, status_counts, dwelling_id, _runs = _load_breakdown_arrays(
        breakdown_path, only_optimal=only_optimal
    )

    hp_mean = hp_array.mean(axis=0)
    ev_mean = ev_array.mean(axis=0)
    app_mean = app_array.mean(axis=0)
    total_mean = hp_mean + ev_mean + app_mean
    total_array = hp_array + ev_array + app_array

    mode = str(envelope_mode).strip().lower()
    if mode == "extreme":
        low_curve = total_array.min(axis=0)
        high_curve = total_array.max(axis=0)
        low_label = "Lower extreme"
        high_label = "Upper extreme"
    else:
        low_curve = np.percentile(total_array, float(envelope_low_pct), axis=0)
        high_curve = np.percentile(total_array, float(envelope_high_pct), axis=0)
        low_label = f"P{envelope_low_pct:g}"
        high_label = f"P{envelope_high_pct:g}"

    if tariff_type:
        fig, (ax, ax_tariff) = plt.subplots(
            2, 1, sharex=True, figsize=(10, 6.2), gridspec_kw={"height_ratios": [3, 1]}
        )
        tariff = _build_tariff_for_time_index(time_index, tariff_type)
        change_times = _tariff_change_times(tariff)
    else:
        fig, ax = plt.subplots(figsize=(10, 4.5))
        ax_tariff = None
        change_times = []

    ax.stackplot(
        time_index,
        app_mean,
        hp_mean,
        ev_mean,
        labels=["Appliance", "Heat pump", "EV"],
        colors=["#B39DDB", "#F6D365", "#90CAF9"],
        alpha=0.6,
        linewidth=0,
        edgecolor="none",
    )
    total_line = ax.plot(time_index, total_mean, color="white", linewidth=1.2, label="Total (mean)", zorder=5)[0]
    total_line.set_path_effects([pe.Stroke(linewidth=1.8, foreground="black"), pe.Normal()])
    ax.plot(time_index, low_curve, color="tab:red", linewidth=1.1, linestyle="--", label=low_label)
    ax.plot(time_index, high_curve, color="tab:green", linewidth=1.1, linestyle="--", label=high_label)

    if title:
        ax.set_title(str(title))
    else:
        title_bits = []
        if title_prefix:
            title_bits.append(str(title_prefix))
        if dwelling_id is not None:
            title_bits.append(f"Dwelling {dwelling_id}")
        if title_bits:
            ax.set_title(" - ".join(title_bits))
    ax.set_ylabel("Power (kW)")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2)
    ax.set_xlabel("Time")
    for t in change_times:
        ax.axvline(t, color="#444444", linestyle=":", linewidth=1.0, alpha=0.7)

    if ax_tariff is not None:
        ax_tariff.step(tariff.index, tariff["elec_price"], where="post", label="Electricity price (p/kWh)")
        ax_tariff.step(tariff.index, tariff["gas_price"], where="post", label="Gas price (p/kWh)")
        for t in change_times:
            ax_tariff.axvline(t, color="#444444", linestyle=":", linewidth=1.0, alpha=0.7)
        ax_tariff.set_ylabel("Tariff (p/kWh)")
        ax_tariff.grid(True, alpha=0.3)
        ax_tariff.legend(loc="upper right")
    fig.tight_layout()

    if output_path is not None:
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=save_dpi, bbox_inches="tight")
    plt.close(fig)

    return {
        "dwelling_id": dwelling_id,
        "runs": int(hp_array.shape[0]),
        "n_steps": int(hp_array.shape[1]),
        "status_counts": status_counts,
    }


def plot_aggregate_stackplot_from_cache(
    cache_dir: str | Path,
    *,
    output_path: str | Path | None = None,
    envelope_mode: str = "percentile",
    envelope_low_pct: float = 10.0,
    envelope_high_pct: float = 90.0,
    only_optimal: bool = False,
    title: str | None = None,
    tariff_type: str | None = None,
    save_dpi: int = 300,
) -> dict[str, Any]:
    """Plot stacked consumption curves aggregated across all dwellings in a cache folder."""
    cache_dir = Path(cache_dir)
    breakdown_files = sorted(cache_dir.glob("dwelling_*_runs_breakdown.csv"))
    if not breakdown_files:
        raise FileNotFoundError(f"No breakdown files found in {cache_dir}")

    common_runs: set[int] | None = None
    time_index: pd.DatetimeIndex | None = None

    # First pass: validate time index and determine common run IDs.
    for breakdown_path in breakdown_files:
        d_time, _hp, _ev, _app, _status_counts, _dwelling_id, run_ids = _load_breakdown_arrays(
            breakdown_path, only_optimal=only_optimal
        )
        if time_index is None:
            time_index = d_time
        elif not d_time.equals(time_index):
            raise ValueError(f"Time index mismatch in {breakdown_path}")
        run_set = set(run_ids.tolist())
        common_runs = run_set if common_runs is None else (common_runs & run_set)
        if not common_runs:
            raise ValueError("No common run IDs across breakdown files.")

    common_runs_sorted = sorted(common_runs or [])
    if not common_runs_sorted:
        raise ValueError("No common run IDs across breakdown files.")

    # Second pass: sum arrays aligned to the common run IDs.
    sum_hp = np.zeros((len(common_runs_sorted), len(time_index)), dtype=float)
    sum_ev = np.zeros_like(sum_hp)
    sum_app = np.zeros_like(sum_hp)

    for breakdown_path in breakdown_files:
        (
            _d_time,
            hp_array,
            ev_array,
            app_array,
            _status_counts,
            _dwelling_id,
            run_ids,
        ) = _load_breakdown_arrays(breakdown_path, only_optimal=only_optimal)

        run_index_map = {int(r): idx for idx, r in enumerate(run_ids)}
        sel_idx = [run_index_map[r] for r in common_runs_sorted]
        sum_hp += hp_array[sel_idx, :]
        sum_ev += ev_array[sel_idx, :]
        sum_app += app_array[sel_idx, :]

    if time_index is None:
        raise RuntimeError("Failed to assemble aggregated curves from cache.")

    hp_mean = sum_hp.mean(axis=0)
    ev_mean = sum_ev.mean(axis=0)
    app_mean = sum_app.mean(axis=0)
    total_mean = hp_mean + ev_mean + app_mean
    total_array = sum_hp + sum_ev + sum_app

    mode = str(envelope_mode).strip().lower()
    if mode == "extreme":
        low_curve = total_array.min(axis=0)
        high_curve = total_array.max(axis=0)
        low_label = "Lower extreme"
        high_label = "Upper extreme"
    else:
        low_curve = np.percentile(total_array, float(envelope_low_pct), axis=0)
        high_curve = np.percentile(total_array, float(envelope_high_pct), axis=0)
        low_label = f"P{envelope_low_pct:g}"
        high_label = f"P{envelope_high_pct:g}"

    if tariff_type:
        fig, (ax, ax_tariff) = plt.subplots(
            2, 1, sharex=True, figsize=(10, 6.2), gridspec_kw={"height_ratios": [3, 1]}
        )
        tariff = _build_tariff_for_time_index(time_index, tariff_type)
        change_times = _tariff_change_times(tariff)
    else:
        fig, ax = plt.subplots(figsize=(10, 4.5))
        ax_tariff = None
        change_times = []

    ax.stackplot(
        time_index,
        app_mean,
        hp_mean,
        ev_mean,
        labels=["Appliance", "Heat pump", "EV"],
        colors=["#B39DDB", "#F6D365", "#90CAF9"],
        alpha=0.6,
        linewidth=0,
        edgecolor="none",
    )
    total_line = ax.plot(time_index, total_mean, color="white", linewidth=1.2, label="Total (mean)", zorder=5)[0]
    total_line.set_path_effects([pe.Stroke(linewidth=1.8, foreground="black"), pe.Normal()])
    ax.plot(time_index, low_curve, color="tab:red", linewidth=1.1, linestyle="--", label=low_label)
    ax.plot(time_index, high_curve, color="tab:green", linewidth=1.1, linestyle="--", label=high_label)
    if title:
        ax.set_title(title)
    ax.set_ylabel("Power (kW)")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2)
    ax.set_xlabel("Time")
    for t in change_times:
        ax.axvline(t, color="#444444", linestyle=":", linewidth=1.0, alpha=0.7)

    if ax_tariff is not None:
        ax_tariff.step(tariff.index, tariff["elec_price"], where="post", label="Electricity price (p/kWh)")
        ax_tariff.step(tariff.index, tariff["gas_price"], where="post", label="Gas price (p/kWh)")
        for t in change_times:
            ax_tariff.axvline(t, color="#444444", linestyle=":", linewidth=1.0, alpha=0.7)
        ax_tariff.set_ylabel("Tariff (p/kWh)")
        ax_tariff.grid(True, alpha=0.3)
        ax_tariff.legend(loc="upper right")
    fig.tight_layout()

    if output_path is not None:
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=save_dpi, bbox_inches="tight")
    plt.close(fig)

    return {
        "dwelling_count": len(breakdown_files),
        "runs": int(total_array.shape[0]),
        "n_steps": int(total_array.shape[1]),
    }


def plot_dwelling_stackplots_from_cache(
    *,
    hybrid_cache_dir: str | Path | None = None,
    monovalent_cache_dir: str | Path | None = None,
    cases: Sequence[str] | None = None,
    output_dir: str | Path,
    envelope_mode: str = "percentile",
    envelope_low_pct: float = 10.0,
    envelope_high_pct: float = 90.0,
    only_optimal: bool = False,
    max_dwellings: int | None = None,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Plot stacked consumption curves for each dwelling using cached breakdowns."""
    case_dirs: dict[str, Path] = {}
    if hybrid_cache_dir is not None:
        case_dirs["hybrid"] = Path(hybrid_cache_dir)
    if monovalent_cache_dir is not None:
        case_dirs["monovalent"] = Path(monovalent_cache_dir)
    if not case_dirs:
        raise ValueError("Provide hybrid_cache_dir and/or monovalent_cache_dir.")

    case_list = [c.lower() for c in (cases or case_dirs.keys())]
    output_dir = Path(output_dir)
    rows: list[dict[str, Any]] = []

    for case in case_list:
        if case not in case_dirs:
            raise ValueError(f"Unknown case '{case}'. Available: {sorted(case_dirs)}")
        case_dir = case_dirs[case]
        if not case_dir.exists():
            raise FileNotFoundError(f"Cache folder not found: {case_dir}")
        breakdown_files = sorted(case_dir.glob("dwelling_*_runs_breakdown.csv"))
        if max_dwellings is not None:
            breakdown_files = breakdown_files[: int(max_dwellings)]

        iter_files = (
            tqdm(breakdown_files, desc=f"Exp5 {case} dwellings", unit="dwelling", dynamic_ncols=True)
            if show_progress else breakdown_files
        )

        for breakdown_path in iter_files:
            dwelling_token = breakdown_path.stem.replace("_runs_breakdown", "")
            plot_path = output_dir / case / f"{dwelling_token}_stacked_consumption.png"
            summary = plot_dwelling_stackplot_from_breakdown(
                breakdown_path,
                output_path=plot_path,
                envelope_mode=envelope_mode,
                envelope_low_pct=envelope_low_pct,
                envelope_high_pct=envelope_high_pct,
                only_optimal=only_optimal,
                title_prefix=f"{case.upper()}",
            )
            rows.append(
                {
                    "case": case,
                    "dwelling_id": summary["dwelling_id"],
                    "breakdown_path": str(breakdown_path),
                    "plot_path": str(plot_path),
                    "runs": summary["runs"],
                    "n_steps": summary["n_steps"],
                    "status_counts": summary["status_counts"],
                }
            )

    return pd.DataFrame(rows)


def run_ev_power_sweep_experiment(
    context: Mapping[str, Any],
    *,
    ev_power_limits_kw: Sequence[float] = (3, 5, 7, 9, 11),
    mc_runs: int = 10,
    selected_dwellings: Sequence[Any] | Any | None = None,
    mhp_capacity_candidates_kw: Sequence[float] | None = None,
    output_subdir: str = "Test",
    plot_dir: str | Path = "test/plots",
    optim_params_cfg: Mapping[str, Any] | None = None,
    hw_params_cfg: Mapping[str, Any] | None = None,
    mhp_optim_params_cfg: Mapping[str, Any] | None = None,
    mhp_hw_params_cfg: Mapping[str, Any] | None = None,
    hhp_optim_params_cfg: Mapping[str, Any] | None = None,
    hhp_hw_params_cfg: Mapping[str, Any] | None = None,
    day_ahead: bool = True,
    envelope_mode: str = "percentile",
    envelope_low_pct: float = 10.0,
    envelope_high_pct: float = 90.0,
    save_outputs: bool = True,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Run EV charging-power sweep for MHP and HHP cases and save summary plots.

    Cases:
    - MHP: monovalent HP case
    - HHP: hybrid case with default 4 kW HP and 30 kW boiler in this project

    Override behavior:
    - `optim_params_cfg` / `hw_params_cfg` are treated as shared defaults for MHP.
    - HHP uses its own defaults unless you pass `hhp_optim_params_cfg` / `hhp_hw_params_cfg`.
    """

    plot_dir = Path(plot_dir)
    if not plot_dir.is_absolute():
        plot_dir = Path(context["repo_root"]) / plot_dir
    plot_dir.mkdir(parents=True, exist_ok=True)

    shared_optim_cfg = dict(optim_params_cfg or {})
    shared_hw_cfg = dict(hw_params_cfg or {})
    mhp_optim_cfg = dict(shared_optim_cfg)
    if mhp_optim_params_cfg:
        mhp_optim_cfg.update(dict(mhp_optim_params_cfg))
    mhp_hw_cfg = dict(shared_hw_cfg)
    if mhp_hw_params_cfg:
        mhp_hw_cfg.update(dict(mhp_hw_params_cfg))
    hhp_optim_cfg = dict(hhp_optim_params_cfg or {})
    hhp_hw_cfg = dict(hhp_hw_params_cfg or {})
    mhp_caps = np.asarray(
        mhp_capacity_candidates_kw if mhp_capacity_candidates_kw is not None else np.arange(7.0, 10.5, 0.5),
        dtype=float,
    )

    results_rows: list[dict[str, Any]] = []
    cases = [("MHP", "monovalent"), ("HHP", "hybrid")]

    for case_label, case_mode in cases:
        for ev_power in ev_power_limits_kw:
            ev_cfg = {"ev_charge_max": float(ev_power)}
            if case_mode == "monovalent":
                case_optim_cfg = mhp_optim_cfg
                case_hw_cfg = mhp_hw_cfg
            else:
                # Keep HHP safe defaults unless explicit HHP override is provided.
                case_optim_cfg = hhp_optim_cfg
                case_hw_cfg = hhp_hw_cfg

            mc_results = run_monte_carlo_batch(
                context,
                mc_runs=mc_runs,
                case=case_mode,
                output_subdir=output_subdir,
                selected_dwellings=selected_dwellings,
                capacity_candidates_kw=mhp_caps if case_mode == "monovalent" else None,
                optim_params_cfg=case_optim_cfg,
                ev_params_cfg=ev_cfg,
                hw_params_cfg=case_hw_cfg,
                day_ahead=day_ahead,
                save_outputs=save_outputs,
                show_progress=show_progress,
            )

            # Plot filename format: MHP_ev_3kW_summary.png, HHP_ev_11kW_summary.png, etc.
            plot_path = plot_dir / f"{case_label}_ev_{int(ev_power)}kW_summary.png"
            plot_monte_carlo_summary(
                mc_results,
                context["tariff"],
                setpoint_sequences=context.get("setpoint_sequences"),
                tol_for_plot=float(case_optim_cfg.get("tol", 1.0)),
                envelope_mode=envelope_mode,
                envelope_low_pct=envelope_low_pct,
                envelope_high_pct=envelope_high_pct,
                save_path=plot_path,
            )

            if mc_results["summary_runs"]:
                summary_df = pd.concat(mc_results["summary_runs"], ignore_index=True)
                status_counts = summary_df["solve_status"].value_counts(dropna=False).to_dict()
            else:
                status_counts = {}

            results_rows.append(
                {
                    "case_label": case_label,
                    "case_mode": case_mode,
                    "ev_charge_max_kw": float(ev_power),
                    "mc_runs": int(mc_runs),
                    "n_dwellings": len(mc_results["dwellings_to_run"]),
                    "status_counts": status_counts,
                    "failure_reasons": mc_results.get("failure_reasons", {}),
                    "plot_path": str(plot_path),
                }
            )

    return pd.DataFrame(results_rows)


def _normalize_penetration_values(values: Sequence[float]) -> np.ndarray:
    """Normalize percentage inputs to fractions in [0, 1]."""
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == 0:
        raise ValueError("Penetration list must contain at least one value.")
    if np.any(arr < 0):
        raise ValueError("Penetration values must be non-negative.")
    if np.any(arr > 1.0):
        if np.any(arr > 100.0):
            raise ValueError("Penetration values above 100 are not valid percentages.")
        arr = arr / 100.0
    if np.any(arr > 1.0):
        raise ValueError("Penetration values must be in [0,1] or [0,100].")
    return arr


def _dwelling_sort_key(dwelling_id: str) -> tuple[int, Any]:
    """Sort numeric dwelling identifiers before non-numeric identifiers."""
    if dwelling_id.isdigit():
        return (0, int(dwelling_id))
    return (1, dwelling_id)


def _load_case_breakdown_cache(case_dir: Path) -> dict[str, Any]:
    """Load one case cache directory into dense arrays for fast MC mixing."""
    files = sorted(case_dir.glob("dwelling_*_runs_breakdown.csv"))
    if not files:
        raise FileNotFoundError(f"No dwelling breakdown cache files found in {case_dir}")

    base_by_dwelling: dict[str, np.ndarray] = {}
    ev_by_dwelling: dict[str, np.ndarray] = {}
    n_runs_ref: int | None = None
    n_steps_ref: int | None = None

    for path in files:
        match = re.match(r"^dwelling_(.+)_runs_breakdown\.csv$", path.name)
        if not match:
            continue
        dwelling_id = match.group(1)
        df = pd.read_csv(path, usecols=["run", "time", "hp_elec_kw", "ev_charge_kw", "appliance_kw"])
        if df.empty:
            continue

        run_counts = df.groupby("run").size()
        if run_counts.empty:
            continue
        if run_counts.nunique() != 1:
            raise ValueError(f"Inconsistent step count per run in {path}")

        n_runs = int(run_counts.shape[0])
        n_steps = int(run_counts.iloc[0])
        if n_runs_ref is None:
            n_runs_ref = n_runs
        elif n_runs_ref != n_runs:
            raise ValueError(
                f"Run count mismatch in {path}: expected {n_runs_ref}, got {n_runs}"
            )
        if n_steps_ref is None:
            n_steps_ref = n_steps
        elif n_steps_ref != n_steps:
            raise ValueError(
                f"Step count mismatch in {path}: expected {n_steps_ref}, got {n_steps}"
            )

        df = df.sort_values(["run", "time"], kind="mergesort")
        hp = pd.to_numeric(df["hp_elec_kw"], errors="coerce").fillna(0.0).to_numpy()
        app = pd.to_numeric(df["appliance_kw"], errors="coerce").fillna(0.0).to_numpy()
        ev = pd.to_numeric(df["ev_charge_kw"], errors="coerce").fillna(0.0).to_numpy()

        base_by_dwelling[dwelling_id] = (hp + app).reshape(n_runs, n_steps).astype(np.float32, copy=False)
        ev_by_dwelling[dwelling_id] = ev.reshape(n_runs, n_steps).astype(np.float32, copy=False)

    if not base_by_dwelling:
        raise FileNotFoundError(f"No valid dwelling cache files parsed in {case_dir}")
    if n_runs_ref is None or n_steps_ref is None:
        raise ValueError(f"Unable to infer run/step dimensions from {case_dir}")

    return {
        "base_by_dwelling": base_by_dwelling,
        "ev_by_dwelling": ev_by_dwelling,
        "n_runs": int(n_runs_ref),
        "n_steps": int(n_steps_ref),
    }


def _build_charging_window_mask(hours: np.ndarray, start_hour: float, end_hour: float) -> np.ndarray:
    """Return a boolean mask for a charging window on a 24-hour clock."""
    if start_hour <= end_hour:
        return (hours >= start_hour) & (hours < end_hour)
    return (hours >= start_hour) | (hours < end_hour)


def _generate_homogeneous_ev_profile_pool(
    n_profiles: int,
    n_steps: int,
    rng: np.random.Generator,
    params: Mapping[str, Any] | None = None,
) -> np.ndarray:
    """Generate a reusable pool of homogeneous EV charging profiles in kW."""
    cfg = dict(params or {})
    step_hours = float(cfg.get("step_hours", 0.5))
    if step_hours <= 0:
        raise ValueError("step_hours must be > 0")

    day_steps = int(round(24.0 / step_hours))
    if day_steps <= 0:
        raise ValueError("Computed day_steps must be > 0")
    n_days = int(np.ceil(n_steps / day_steps))

    ev_charge_max_kw = float(cfg.get("ev_charge_max_kw", 7.0))
    daily_energy_mean_kwh = float(cfg.get("daily_energy_mean_kwh", 8.0))
    daily_energy_std_kwh = float(cfg.get("daily_energy_std_kwh", 2.0))
    daily_energy_min_kwh = float(cfg.get("daily_energy_min_kwh", 0.0))
    daily_energy_max_kwh = float(cfg.get("daily_energy_max_kwh", np.inf))
    charge_start_hour = float(cfg.get("charge_start_hour", 18.0))
    charge_end_hour = float(cfg.get("charge_end_hour", 7.0))
    start_hour_jitter = float(cfg.get("start_hour_jitter", 1.5))

    if ev_charge_max_kw < 0:
        raise ValueError("ev_charge_max_kw must be >= 0")
    if daily_energy_std_kwh < 0:
        raise ValueError("daily_energy_std_kwh must be >= 0")

    profiles = np.zeros((int(n_profiles), int(n_steps)), dtype=np.float32)
    for p_idx in range(int(n_profiles)):
        for day in range(n_days):
            s = day * day_steps
            e = min((day + 1) * day_steps, n_steps)
            if s >= e:
                continue

            day_len = e - s
            hours = np.arange(day_len, dtype=float) * step_hours
            shift = float(rng.normal(0.0, start_hour_jitter))
            day_start = (charge_start_hour + shift) % 24.0
            day_end = (charge_end_hour + shift) % 24.0
            avail = np.flatnonzero(_build_charging_window_mask(hours, day_start, day_end))
            if avail.size == 0 or ev_charge_max_kw == 0:
                continue

            daily_energy = float(rng.normal(daily_energy_mean_kwh, daily_energy_std_kwh))
            daily_energy = float(np.clip(daily_energy, daily_energy_min_kwh, daily_energy_max_kwh))
            if daily_energy <= 0:
                continue

            start_pos = int(rng.integers(0, avail.size))
            order = np.concatenate((avail[start_pos:], avail[:start_pos]))
            remain = daily_energy
            for slot in order:
                p_kw = min(ev_charge_max_kw, remain / step_hours)
                profiles[p_idx, s + int(slot)] = np.float32(p_kw)
                remain -= p_kw * step_hours
                if remain <= 1e-9:
                    break
    return profiles


def run_hhp_mhp_ev_penetration_experiment_from_cache(
    *,
    cache_root: str | Path | None = None,
    hybrid_cache_dir: str | Path | None = None,
    monovalent_cache_dir: str | Path | None = None,
    ev_penetrations: Sequence[float],
    hhp_percentages: Sequence[float],
    mc_runs_per_pixel: int = 100,
    random_seed: int = 42,
    save_path: str | Path | None = None,
    use_generated_ev_profiles: bool = False,
    ev_gen_params: Mapping[str, Any] | None = None,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Run cache-based penetration MC and report maximum aggregated demand.

    For each (EV penetration, HHP share) pixel:
    - each dwelling is assigned either HHP (hybrid cache) or MHP (monovalent cache)
    - EV homes are sampled independently by EV penetration
    - non-EV homes have EV charging component forced to zero
    - one stochastic run profile is sampled per dwelling from its selected case cache
    - optional EV replacement can inject generated homogeneous EV profiles
    - maximum aggregated electric demand (kW) is recorded per Monte Carlo replicate
    """
    if hybrid_cache_dir is None or monovalent_cache_dir is None:
        if cache_root is None:
            raise ValueError("Provide cache_root or both hybrid_cache_dir and monovalent_cache_dir.")
        cache_root = Path(cache_root)
        hybrid_cache_dir = cache_root / "hybrid"
        monovalent_cache_dir = cache_root / "monovalent"

    hybrid_cache = _load_case_breakdown_cache(Path(hybrid_cache_dir))
    monovalent_cache = _load_case_breakdown_cache(Path(monovalent_cache_dir))

    hybrid_ids = set(hybrid_cache["base_by_dwelling"].keys())
    monovalent_ids = set(monovalent_cache["base_by_dwelling"].keys())
    common_ids = sorted(hybrid_ids & monovalent_ids, key=_dwelling_sort_key)
    if not common_ids:
        raise ValueError("No common dwelling IDs between hybrid and monovalent cache folders.")

    n_runs_h = int(hybrid_cache["n_runs"])
    n_runs_m = int(monovalent_cache["n_runs"])
    n_steps_h = int(hybrid_cache["n_steps"])
    n_steps_m = int(monovalent_cache["n_steps"])
    if n_steps_h != n_steps_m:
        raise ValueError(f"Step count mismatch: hybrid={n_steps_h}, monovalent={n_steps_m}")

    hybrid_base = np.stack([hybrid_cache["base_by_dwelling"][d] for d in common_ids], axis=0)
    hybrid_ev = np.stack([hybrid_cache["ev_by_dwelling"][d] for d in common_ids], axis=0)
    monovalent_base = np.stack([monovalent_cache["base_by_dwelling"][d] for d in common_ids], axis=0)
    monovalent_ev = np.stack([monovalent_cache["ev_by_dwelling"][d] for d in common_ids], axis=0)

    n_dwellings = len(common_ids)
    n_steps = n_steps_h
    ev_fracs = _normalize_penetration_values(ev_penetrations)
    hhp_fracs = _normalize_penetration_values(hhp_percentages)
    mc_runs_per_pixel = int(mc_runs_per_pixel)
    if mc_runs_per_pixel <= 0:
        raise ValueError("mc_runs_per_pixel must be > 0")

    rng = np.random.default_rng(int(random_seed))
    ev_pool = None
    if use_generated_ev_profiles:
        cfg = dict(ev_gen_params or {})
        pool_size = int(cfg.pop("ev_profile_pool_size", 2000))
        if pool_size <= 0:
            raise ValueError("ev_profile_pool_size must be > 0")
        ev_pool = _generate_homogeneous_ev_profile_pool(
            n_profiles=pool_size,
            n_steps=n_steps,
            rng=rng,
            params=cfg,
        )

    pixels: list[tuple[float, float]] = [
        (float(ev_frac), float(hhp_frac)) for hhp_frac in hhp_fracs for ev_frac in ev_fracs
    ]
    results: list[dict[str, Any]] = []
    pbar = None
    if show_progress:
        pbar = tqdm(total=len(pixels), desc="Penetration pixels", dynamic_ncols=True, leave=True, mininterval=0.1)
        pbar.refresh()

    for ev_frac, hhp_frac in pixels:
        n_hhp = int(round(hhp_frac * n_dwellings))
        n_ev = int(round(ev_frac * n_dwellings))
        n_hhp = min(max(n_hhp, 0), n_dwellings)
        n_ev = min(max(n_ev, 0), n_dwellings)

        peaks = np.zeros(mc_runs_per_pixel, dtype=np.float32)
        for i in range(mc_runs_per_pixel):
            hhp_idx = rng.choice(n_dwellings, size=n_hhp, replace=False) if n_hhp > 0 else np.array([], dtype=int)
            mhp_idx = np.setdiff1d(np.arange(n_dwellings, dtype=int), hhp_idx, assume_unique=True)

            ev_idx = rng.choice(n_dwellings, size=n_ev, replace=False) if n_ev > 0 else np.array([], dtype=int)
            ev_mask = np.zeros(n_dwellings, dtype=bool)
            ev_mask[ev_idx] = True

            agg = np.zeros(n_steps, dtype=np.float32)
            if hhp_idx.size:
                run_idx_h = rng.integers(0, n_runs_h, size=hhp_idx.size)
                agg += hybrid_base[hhp_idx, run_idx_h, :].sum(axis=0)
                if not use_generated_ev_profiles:
                    hhp_ev_mask = ev_mask[hhp_idx]
                    if np.any(hhp_ev_mask):
                        agg += hybrid_ev[hhp_idx[hhp_ev_mask], run_idx_h[hhp_ev_mask], :].sum(axis=0)

            if mhp_idx.size:
                run_idx_m = rng.integers(0, n_runs_m, size=mhp_idx.size)
                agg += monovalent_base[mhp_idx, run_idx_m, :].sum(axis=0)
                if not use_generated_ev_profiles:
                    mhp_ev_mask = ev_mask[mhp_idx]
                    if np.any(mhp_ev_mask):
                        agg += monovalent_ev[mhp_idx[mhp_ev_mask], run_idx_m[mhp_ev_mask], :].sum(axis=0)

            if use_generated_ev_profiles and n_ev > 0:
                if ev_pool is None:
                    raise RuntimeError("EV profile pool was not initialized.")
                ev_pick = rng.integers(0, ev_pool.shape[0], size=n_ev)
                agg += ev_pool[ev_pick, :].sum(axis=0)

            peaks[i] = float(agg.max())

        results.append(
            {
                "ev_penetration": ev_frac,
                "hhp_percentage": hhp_frac,
                "n_dwellings": n_dwellings,
                "n_ev_homes": n_ev,
                "n_hhp_homes": n_hhp,
                "mc_runs_per_pixel": mc_runs_per_pixel,
                "max_demand_mean_kw": float(np.mean(peaks)),
                "max_demand_std_kw": float(np.std(peaks, ddof=0)),
                "max_demand_p50_kw": float(np.percentile(peaks, 50)),
                "max_demand_p90_kw": float(np.percentile(peaks, 90)),
                "max_demand_p95_kw": float(np.percentile(peaks, 95)),
                "max_demand_max_kw": float(np.max(peaks)),
                "max_demand_min_kw": float(np.min(peaks)),
            }
        )
        if pbar is not None:
            pbar.update(1)
            pbar.set_postfix({"EV%": round(ev_frac * 100, 1), "HHP%": round(hhp_frac * 100, 1)}, refresh=False)
            pbar.refresh()

    if pbar is not None:
        pbar.close()

    result_df = pd.DataFrame(results).sort_values(["hhp_percentage", "ev_penetration"]).reset_index(drop=True)
    if save_path is not None:
        out_path = Path(save_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        result_df.to_csv(out_path, index=False)
    return result_df
