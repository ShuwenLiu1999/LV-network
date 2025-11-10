"""Generate heating demand profiles for dwellings using RC optimization.

This script loads dwelling-specific thermal parameters from the
``Codes/Data/1R1C1P1S_filtered.csv`` summary file, prepares the weather
profile for EOH2303 between 9–13 February 2022, and runs the existing
hybrid heat pump optimization routine to compute the heating demand for
each dwelling.

The output is saved to ``Codes/Output/demand_profiles_EOH2303_Feb09-13.csv``
with one column group per dwelling containing the optimized dispatch
variables and aggregated heat demand.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from RC_Optimization import build_tariff, optimize_hhp_operation


DATA_DIR = Path(__file__).resolve().parents[1] / "Data"
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "Output"
SUMMARY_PATH = DATA_DIR / "1R1C1P1S_filtered.csv"
WEATHER_PATH = DATA_DIR / "EOH2303_merged_30min.csv"
OUTPUT_PATH = OUTPUT_DIR / "demand_profiles_EOH2303_Feb09-13.csv"

START_DATE = pd.Timestamp("2022-02-09 00:00:00")
END_DATE = pd.Timestamp("2022-02-13 23:30:00")


@dataclass(frozen=True)
class DwellingParameters:
    dwelling_id: str
    R1: float
    C1: float
    g: float


def load_dwelling_parameters(summary_path: Path) -> Dict[str, DwellingParameters]:
    """Return a mapping from dwelling ID to its RC model parameters."""
    df = pd.read_csv(summary_path)
    df["dwelling_id"] = df["dataset"].str.extract(r"(EOH\d{4})")
    df = df.dropna(subset=["dwelling_id", "R1", "C1", "g"])

    params: Dict[str, DwellingParameters] = {}
    for _, row in df.iterrows():
        dwelling_id = str(row["dwelling_id"])
        if dwelling_id in params:
            # Keep the first occurrence to avoid overwriting with duplicates.
            continue
        params[dwelling_id] = DwellingParameters(
            dwelling_id=dwelling_id,
            R1=float(row["R1"]),
            C1=float(row["C1"]),
            g=float(row["g"]),
        )
    return params


def prepare_weather_profile(weather_path: Path, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Load and clean the weather profile for the requested period."""
    df = pd.read_csv(weather_path, parse_dates=["time"])
    df = df.set_index("time").sort_index()

    target_index = pd.date_range(start=start, end=end, freq="30min")
    df = df.reindex(target_index)

    numeric_cols = [
        "External_Air_Temperature",
        "t2m_°C",
        "ssrd_J_m2_30min",
        "Internal_Air_Temperature",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "External_Air_Temperature" in df.columns and "t2m_°C" in df.columns:
        df["External_Air_Temperature"] = df["External_Air_Temperature"].fillna(df["t2m_°C"])

    if "External_Air_Temperature" in df.columns:
        df["External_Air_Temperature"] = df["External_Air_Temperature"].interpolate(limit_direction="both")

    if "ssrd_J_m2_30min" in df.columns:
        df["ssrd_J_m2_30min"] = df["ssrd_J_m2_30min"].fillna(0.0)

    if "Internal_Air_Temperature" in df.columns:
        df["Internal_Air_Temperature"] = df["Internal_Air_Temperature"].interpolate(limit_direction="both")

    return df


def build_setpoint_schedule(index: pd.DatetimeIndex) -> np.ndarray:
    """Create a simple heating setpoint profile with comfort bands."""
    hours = index.hour
    setpoint = np.full(len(index), 18.0)

    morning = (hours >= 6) & (hours < 9)
    daytime = (hours >= 9) & (hours < 17)
    evening = (hours >= 17) & (hours < 22)

    setpoint[morning | evening] = 21.0
    setpoint[daytime & ~morning & ~evening] = 19.0

    # Overnight setback remains at 18°C
    return setpoint


def run_dispatch_for_dwelling(
    params: DwellingParameters,
    weather: pd.DataFrame,
    tariff: pd.DataFrame,
    tol: float = 1.0,
    cop: float = 3.5,
    eta_boiler: float = 0.9,
    q_hp_max: float = 4_000.0,
    q_bo_max: float = 24_000.0,
) -> pd.DataFrame:
    """Execute the optimization for a single dwelling and return results."""
    tout = weather["External_Air_Temperature"].to_numpy()
    if "ssrd_J_m2_30min" in weather.columns:
        solar = np.clip(weather["ssrd_J_m2_30min"].to_numpy() / 1800.0, a_min=0.0, a_max=None)
    else:
        solar = np.zeros_like(tout)

    timestep_seconds = int(pd.Timedelta(tariff.index.freq or "30min").total_seconds())

    setpoint = build_setpoint_schedule(tariff.index)
    indoor_measured = weather.get("Internal_Air_Temperature")
    t0 = float(indoor_measured.dropna().iloc[0]) if indoor_measured is not None and indoor_measured.notna().any() else float(setpoint[0])

    results = optimize_hhp_operation(
        params.R1,
        params.C1,
        params.g,
        tariff,
        tout,
        solar,
        timestep_seconds,
        t0,
        T_setpoint=setpoint,
        tol=tol,
        COP=cop,
        etaB=eta_boiler,
        Qhp_max=q_hp_max,
        Qbo_max=q_bo_max,
        day_ahead=True,
    )

    results = results.copy()
    results["heat_demand"] = results["Q_hp"] + results["Q_bo"]
    results["heat_demand_kw"] = results["heat_demand"] / 1000.0
    return results


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    params_map = load_dwelling_parameters(SUMMARY_PATH)
    if not params_map:
        raise RuntimeError("No dwelling parameters found in the summary file.")

    weather = prepare_weather_profile(WEATHER_PATH, START_DATE, END_DATE)

    n_days = (END_DATE.normalize() - START_DATE.normalize()).days + 1
    tariff = build_tariff(START_DATE.normalize(), n_days=n_days, step="30min")
    tariff = tariff.reindex(weather.index)
    if tariff.isna().any().any():
        raise ValueError("Tariff contains NaN values after alignment; check the requested time range.")

    all_results = {}
    for dwelling_id, params in params_map.items():
        result = run_dispatch_for_dwelling(params, weather, tariff)
        all_results[dwelling_id] = result

    combined = pd.concat(all_results, axis=1)
    combined.to_csv(OUTPUT_PATH, index_label="timestamp")
    print(f"Saved demand profiles for {len(all_results)} dwellings to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
