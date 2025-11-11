"""Utility for generating heat demand profiles for dwellings in the summary file.

The script loads the building parameters (R, C and g) from
``Codes/Data/1R1C1P1S_filtered.csv`` and runs the hybrid heat pump
optimisation for each dwelling using the weather profile of
EOH2303 between 9 and 13 February 2022.  The resulting demand profile is
written to ``Codes/Output/DemandProfiles``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from RC_Optimization import build_tariff, optimize_hhp_operation


@dataclass(frozen=True)
class BuildingParameters:
    """Simple container for the thermal parameters of one dwelling."""

    dataset: str
    R1: float
    C1: float
    g: float

    @property
    def dwelling_id(self) -> str:
        """Return a friendly identifier derived from the dataset filename."""

        if self.dataset.endswith("_merged_30min.parquet"):
            return self.dataset[: -len("_merged_30min.parquet")]
        return Path(self.dataset).stem


def load_summary(summary_path: Path) -> list[BuildingParameters]:
    """Read the summary CSV into :class:`BuildingParameters` objects."""

    df = pd.read_csv(summary_path)
    required_cols = frozenset({"dataset", "R1", "C1", "g"})
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(f"Summary file missing columns: {sorted(missing)}")

    params: list[BuildingParameters] = []
    for _, row in df.iterrows():
        params.append(
            BuildingParameters(
                dataset=str(row["dataset"]),
                R1=float(row["R1"]),
                C1=float(row["C1"]),
                g=float(row["g"]),
            )
        )
    return params


def load_weather(
    weather_path: Path, start: pd.Timestamp, end: pd.Timestamp
) -> pd.DataFrame:
    """Load and tidy the weather/profile data for the required window."""

    df = (
        pd.read_csv(weather_path, parse_dates=["time"])
        .set_index("time")
        .sort_index()
    )
    window = df.loc[start:end].copy()
    if window.empty:
        raise ValueError(
            f"Weather file {weather_path} does not contain data between {start} and {end}."
        )

    for column in (
        "External_Air_Temperature",
        "Internal_Air_Temperature",
        "ssrd_J_m2_30min",
        "GHI",
        "t2m_°C",
    ):
        if column in window.columns:
            window[column] = pd.to_numeric(window[column], errors="coerce")

    return window


def extract_outdoor_temperature(weather: pd.DataFrame) -> np.ndarray:
    """Prefer measured external air temperature and fall back to re-analysis."""

    if "External_Air_Temperature" in weather:
        series = weather["External_Air_Temperature"].ffill().bfill()
    elif "t2m_°C" in weather:
        series = weather["t2m_°C"].ffill().bfill()
    else:
        raise ValueError(
            "Weather dataset does not contain an outdoor temperature column."
        )
    return series.to_numpy()


def extract_solar_gain(weather: pd.DataFrame, step_seconds: float) -> np.ndarray:
    """Return the solar gain proxy expected by the optimisation model."""

    if "GHI" in weather:
        series = weather["GHI"]
    elif "ssrd_J_m2_30min" in weather:
        series = weather["ssrd_J_m2_30min"] / step_seconds
    else:
        series = pd.Series(0.0, index=weather.index)
    return series.fillna(0.0).to_numpy()


def derive_setpoint(index: pd.DatetimeIndex) -> np.ndarray:
    """Construct the comfort setpoint schedule for the optimisation horizon."""

    hours = index.hour
    return np.where(
        ((hours >= 6) & (hours < 9)) | ((hours >= 17) & (hours < 22)),
        21.0,
        18.0,
    )


def derive_initial_temperature(
    weather: pd.DataFrame, fallback: float = 19.0
) -> float:
    """Use the first valid internal temperature if available."""

    if "Internal_Air_Temperature" in weather:
        first_valid = weather["Internal_Air_Temperature"].dropna().head(1)
        if not first_valid.empty:
            return float(first_valid.iat[0])
    return float(fallback)


def ensure_aligned_tariff(
    tariff: pd.DataFrame, index: pd.DatetimeIndex
) -> pd.DataFrame:
    """Align the tariff profile to match the simulation index."""

    tariff = tariff.copy()
    tariff.index = tariff.index.tz_localize(None)
    return tariff.reindex(index)


def run_optimisation(
    params: Iterable[BuildingParameters],
    weather: pd.DataFrame,
    tariff: pd.DataFrame,
    dt_seconds: float,
    setpoint: np.ndarray,
    tolerance: float,
    cop: float,
    eta_boiler: float,
    qhp_max: float,
    qbo_max: float,
    output_dir: Path,
) -> None:
    """Execute the optimisation for each dwelling and persist the results."""

    Tout = extract_outdoor_temperature(weather)
    solar_gain = extract_solar_gain(weather, dt_seconds)
    initial_temp = derive_initial_temperature(weather)

    output_dir.mkdir(parents=True, exist_ok=True)

    for param in params:
        results = optimize_hhp_operation(
            param.R1,
            param.C1,
            param.g,
            tariff,
            Tout,
            solar_gain,
            dt_seconds,
            initial_temp,
            T_setpoint=setpoint,
            tol=tolerance,
            COP=cop,
            etaB=eta_boiler,
            Qhp_max=qhp_max,
            Qbo_max=qbo_max,
            day_ahead=False,
        )

        demand = results.copy()
        demand["total_heat_W"] = demand["Q_hp"] + demand["Q_bo"]
        demand["total_heat_kW"] = demand["total_heat_W"] / 1000.0

        out_path = output_dir / f"{param.dwelling_id}_Feb09-13_2022.csv"
        demand.to_csv(out_path, index_label="time")
        print(f"Saved demand profile for {param.dwelling_id} to {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    repo_root = Path(__file__).resolve().parents[1]
    parser.add_argument(
        "--summary",
        type=Path,
        default=repo_root / "Data" / "1R1C1P1S_filtered.csv",
        help="Path to the summary CSV containing R, C and g parameters.",
    )
    parser.add_argument(
        "--weather",
        type=Path,
        default=repo_root / "Data" / "EOH2303_merged_30min.csv",
        help="Weather/profile CSV for EOH2303.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=repo_root / "Output" / "DemandProfiles",
        help="Directory to store the generated demand profiles.",
    )
    parser.add_argument(
        "--start",
        type=pd.Timestamp,
        default=pd.Timestamp("2022-02-09 00:00"),
        help="Start of the optimisation horizon (inclusive).",
    )
    parser.add_argument(
        "--end",
        type=pd.Timestamp,
        default=pd.Timestamp("2022-02-13 23:30"),
        help="End of the optimisation horizon (inclusive).",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1.0,
        help="Comfort band half-width in °C.",
    )
    parser.add_argument(
        "--cop",
        type=float,
        default=3.5,
        help="Coefficient of performance for the heat pump.",
    )
    parser.add_argument(
        "--eta-boiler",
        dest="eta_boiler",
        type=float,
        default=0.9,
        help="Boiler efficiency.",
    )
    parser.add_argument(
        "--hp-max",
        dest="hp_max",
        type=float,
        default=4000.0,
        help="Maximum thermal power of the heat pump in watts.",
    )
    parser.add_argument(
        "--boiler-max",
        dest="boiler_max",
        type=float,
        default=24000.0,
        help="Maximum thermal power of the boiler in watts.",
    )
    parser.add_argument(
        "--tariff-type",
        choices={"cosy", "flat"},
        default="cosy",
        help="Shape of the electricity tariff profile.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    summary_path = args.summary
    weather_path = args.weather
    output_dir = args.output
    start = args.start
    end = args.end

    params = load_summary(summary_path)
    weather = load_weather(weather_path, start, end)

    dt_seconds = weather.index.to_series().diff().dropna().median().total_seconds()
    if not (np.isfinite(dt_seconds) and dt_seconds > 0):
        raise ValueError("Could not infer timestep from weather data.")

    tariff_days = int((end.normalize() - start.normalize()).days) + 1
    tariff = build_tariff(
        start.normalize(), n_days=tariff_days, step="30min", type=args.tariff_type
    )
    tariff = ensure_aligned_tariff(tariff, weather.index)

    setpoint = derive_setpoint(weather.index)

    run_optimisation(
        params=params,
        weather=weather,
        tariff=tariff,
        dt_seconds=dt_seconds,
        setpoint=setpoint,
        tolerance=args.tolerance,
        cop=args.cop,
        eta_boiler=args.eta_boiler,
        qhp_max=args.hp_max,
        qbo_max=args.boiler_max,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()
