"""Batch runner for the RC optimization model using dwelling metadata."""
from __future__ import annotations

import argparse
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

import RC_Optimization as rc


@dataclass
class DwellingConfig:
    dwelling_id: str
    technology: str
    R1: float
    C1: float
    g: float
    initial_temp_c: float
    cop: float
    boiler_efficiency: float
    hp_capacity_kw: float
    boiler_capacity_kw: float
    tolerance_c: float
    setpoint_comfort_c: float
    setpoint_setback_c: float
    morning_start_hour: float
    morning_end_hour: float
    evening_start_hour: float
    evening_end_hour: float
    day_ahead: bool

    @property
    def hp_capacity_w(self) -> float:
        return self.hp_capacity_kw * 1000.0

    @property
    def boiler_capacity_w(self) -> float:
        return self.boiler_capacity_kw * 1000.0


REQUIRED_COLUMNS = (
    "dwelling_id",
    "technology",
    "R1",
    "C1",
    "g",
)


def parse_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    raise ValueError(f"Cannot interpret '{value}' as a boolean flag")


OPTIONAL_FIELD_SPECS: Tuple[Tuple[str, object, object], ...] = (
    ("initial_temp_c", lambda tech: 21.0, float),
    ("cop", lambda tech: 3.5, float),
    ("boiler_efficiency", lambda tech: 0.9 if tech == "HHP" else 1.0, float),
    ("hp_capacity_kw", lambda tech: 4.0 if tech == "HHP" else 7.0, float),
    ("boiler_capacity_kw", lambda tech: 24.0 if tech == "HHP" else 0.0, float),
    ("tolerance_c", lambda tech: 1.0, float),
    ("setpoint_comfort_c", lambda tech: 21.0, float),
    ("setpoint_setback_c", lambda tech: 15.0, float),
    ("morning_start_hour", lambda tech: 6.0, float),
    ("morning_end_hour", lambda tech: 10.0, float),
    ("evening_start_hour", lambda tech: 17.0, float),
    ("evening_end_hour", lambda tech: 21.0, float),
    ("day_ahead", lambda tech: False, parse_bool),
)


def _resolve_optional(
    row: pd.Series,
    column_name: str,
    technology: str,
    default_factory,
    caster,
):
    if column_name in row.index:
        value = row[column_name]
        if isinstance(value, str):
            value = value.strip()
            if value == "":
                value = np.nan
        if not pd.isna(value):
            if caster is parse_bool:
                return parse_bool(value), False
            return caster(value), False
    default_value = (
        default_factory(technology)
        if callable(default_factory)
        else default_factory
    )
    if caster is parse_bool:
        return bool(default_value), True
    return caster(default_value), True


def load_summary(summary_path: Path) -> Tuple[List[DwellingConfig], pd.DataFrame]:
    df = pd.read_csv(summary_path)
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            "Summary file is missing required columns: " + ", ".join(missing)
        )
    configs: List[DwellingConfig] = []
    resolved_rows: List[dict] = []
    for _, row in df.iterrows():
        dwelling_id = str(row["dwelling_id"]).strip()
        technology = str(row["technology"]).strip().upper()
        defaults_used = []
        values = {
            "dwelling_id": dwelling_id,
            "technology": technology,
            "R1": float(row["R1"]),
            "C1": float(row["C1"]),
            "g": float(row["g"]),
        }
        for name, default_factory, caster in OPTIONAL_FIELD_SPECS:
            value, used_default = _resolve_optional(row, name, technology, default_factory, caster)
            values[name] = value
            if used_default:
                defaults_used.append(name)
        config_kwargs = {field: values[field] for field in DwellingConfig.__dataclass_fields__}
        configs.append(DwellingConfig(**config_kwargs))
        resolved_row = config_kwargs.copy()
        resolved_row["defaults_used"] = ", ".join(defaults_used)
        resolved_rows.append(resolved_row)
    resolved_df = pd.DataFrame(resolved_rows)
    return configs, resolved_df


def load_weather(weather_path: Path, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    df = pd.read_csv(weather_path, parse_dates=["time"])
    # Some datasets include an unnamed index column. Drop if present.
    if df.columns[0].startswith("Unnamed") or df.columns[0] == "":
        df = df.drop(columns=df.columns[0])
    df = df.set_index("time").sort_index()
    sliced = df.loc[start:end]
    if sliced.empty:
        raise ValueError("Weather file does not cover the requested time range")
    required_cols = ["External_Air_Temperature", "ssrd_J_m2_30min"]
    for col in required_cols:
        if col not in sliced.columns:
            raise ValueError(f"Weather file missing required column '{col}'")
    return sliced


PATTERN_WINDOWS = {
    "two_peaks": ((6.0, 10.0), (17.0, 21.0)),
    "evening_peak": ((17.0, 22.0),),
    "daytime_continuous": ((8.0, 20.0),),
}


def build_setpoint_series(
    config: DwellingConfig, index: pd.DatetimeIndex, pattern: str
) -> np.ndarray:
    if pattern not in PATTERN_WINDOWS:
        raise ValueError(
            f"Unknown demand pattern '{pattern}'. Expected one of: {', '.join(PATTERN_WINDOWS)}"
        )
    hours = index.hour + index.minute / 60.0
    in_comfort = np.zeros(len(index), dtype=bool)
    if pattern == "two_peaks":
        windows = (
            (config.morning_start_hour, config.morning_end_hour),
            (config.evening_start_hour, config.evening_end_hour),
        )
    else:
        windows = PATTERN_WINDOWS[pattern]
    for start_hour, end_hour in windows:
        mask = (hours >= start_hour) & (hours < end_hour)
        in_comfort |= mask
    return np.where(in_comfort, config.setpoint_comfort_c, config.setpoint_setback_c)


def ensure_tariff(
    start: pd.Timestamp, end: pd.Timestamp, step_minutes: int, tariff_type: str
) -> pd.DataFrame:
    n_days = (end.normalize() - start.normalize()).days + 1
    step_str = f"{int(step_minutes)}min"
    tariff = rc.build_tariff(start.normalize(), n_days=n_days, step=step_str, type=tariff_type)
    return tariff.loc[start:end]


def compute_consumption(results: pd.DataFrame, config: DwellingConfig, dt_seconds: float) -> pd.DataFrame:
    df = results.copy()
    df["hp_electric_W"] = df["Q_hp"] / config.cop
    if config.technology == "HHP" and config.boiler_capacity_kw > 0:
        df["boiler_gas_W"] = np.where(df["Q_bo"] > 0, df["Q_bo"] / config.boiler_efficiency, 0.0)
    else:
        df["boiler_gas_W"] = 0.0
    df["total_heat_W"] = df["Q_hp"] + df["Q_bo"]
    step_hours = dt_seconds / 3600.0
    df["hp_electric_kWh"] = df["hp_electric_W"] * step_hours / 1000.0
    df["boiler_gas_kWh"] = df["boiler_gas_W"] * step_hours / 1000.0
    df["delivered_heat_kWh"] = df["total_heat_W"] * step_hours / 1000.0
    return df


def summarize_consumption(consumption: pd.DataFrame, dwelling_id: str) -> dict:
    return {
        "dwelling_id": dwelling_id,
        "hp_electric_kWh": consumption["hp_electric_kWh"].sum(),
        "boiler_gas_kWh": consumption["boiler_gas_kWh"].sum(),
        "delivered_heat_kWh": consumption["delivered_heat_kWh"].sum(),
        "min_indoor_temp_C": consumption["Tin"].min(),
        "max_indoor_temp_C": consumption["Tin"].max(),
        "peak_hp_heat_kW": consumption["Q_hp"].max() / 1000.0,
        "peak_boiler_heat_kW": consumption["Q_bo"].max() / 1000.0,
    }


def run_batch(
    summary_path: Path,
    weather_path: Path,
    output_dir: Path,
    start: pd.Timestamp,
    end: pd.Timestamp,
    step_minutes: int,
    tariff_type: str,
    demand_patterns: Sequence[str],
) -> None:
    configs, resolved_params = load_summary(summary_path)
    if not configs:
        raise RuntimeError("No dwelling configurations found in the summary file")

    weather = load_weather(weather_path, start, end)
    step = pd.Timedelta(minutes=step_minutes)
    index = pd.date_range(start=start, end=end, freq=step)
    weather = weather.reindex(index, method="nearest")
    dt_seconds = step.total_seconds()

    tariff = ensure_tariff(start, end, step_minutes, tariff_type)
    tariff = tariff.reindex(index, method="nearest")

    output_dir.mkdir(parents=True, exist_ok=True)

    demand_patterns = list(dict.fromkeys(demand_patterns))  # deduplicate, preserve order
    if not demand_patterns:
        raise ValueError("At least one demand pattern must be specified")

    resolved_params.to_csv(
        output_dir / "resolved_dwelling_parameters.csv", index=False
    )

    solar_power = weather["ssrd_J_m2_30min"].to_numpy(dtype=float) / dt_seconds
    tout = weather["External_Air_Temperature"].to_numpy(dtype=float)

    all_summary_frames: List[pd.DataFrame] = []

    single_pattern = len(demand_patterns) == 1

    for pattern in demand_patterns:
        pattern_dir = output_dir / pattern
        pattern_dir.mkdir(parents=True, exist_ok=True)

        summaries = []
        aggregated_electric = []
        aggregated_ids = []

        for config in configs:
            setpoints = build_setpoint_series(config, index, pattern)
            results = rc.optimize_hhp_operation(
                config.R1,
                config.C1,
                config.g,
                tariff,
                tout,
                solar_power,
                dt_seconds,
                config.initial_temp_c,
                setpoints,
                config.tolerance_c,
                config.cop,
                config.boiler_efficiency,
                config.hp_capacity_w,
                config.boiler_capacity_w,
                day_ahead=config.day_ahead,
            )
            consumption = compute_consumption(results, config, dt_seconds)
            summary_row = summarize_consumption(consumption, config.dwelling_id)
            summary_row["pattern"] = pattern
            summaries.append(summary_row)

            dispatch_path = pattern_dir / f"{config.dwelling_id}_dispatch.csv"
            consumption.reset_index().rename(columns={"index": "timestamp"}).to_csv(
                dispatch_path,
                index=False,
            )
            if single_pattern:
                shutil.copyfile(dispatch_path, output_dir / dispatch_path.name)
            aggregated_electric.append(
                consumption["hp_electric_W"].to_numpy(dtype=float)
            )
            aggregated_ids.append(config.dwelling_id)

        summary_df = pd.DataFrame(summaries)
        summary_df.to_csv(pattern_dir / "consumption_summary.csv", index=False)
        all_summary_frames.append(summary_df)

        electric_df = pd.DataFrame(
            np.column_stack(aggregated_electric),
            index=index,
            columns=aggregated_ids,
        )
        electric_df.index.name = "timestamp"
        electric_df.to_csv(pattern_dir / "electric_consumption_by_dwelling.csv")
        if single_pattern:
            electric_df.to_csv(output_dir / "electric_consumption_by_dwelling.csv")

    if len(all_summary_frames) > 1:
        combined = pd.concat(all_summary_frames, ignore_index=True)
        combined.to_csv(output_dir / "consumption_summary_all_patterns.csv", index=False)
    else:
        # For a single pattern, mirror the summary at the root for convenience.
        summary_df = all_summary_frames[0]
        summary_df.to_csv(output_dir / "consumption_summary.csv", index=False)


def parse_args(args: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the RC optimization for each dwelling defined in the summary file."
    )
    parser.add_argument("--summary", type=Path, required=True, help="Path to the dwelling summary CSV")
    parser.add_argument("--weather", type=Path, required=True, help="Weather CSV with time, External_Air_Temperature, ssrd_J_m2_30min")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory where results will be written")
    parser.add_argument("--start", type=pd.Timestamp, required=True, help="Start timestamp (inclusive)")
    parser.add_argument("--end", type=pd.Timestamp, required=True, help="End timestamp (inclusive)")
    parser.add_argument("--step-minutes", type=int, default=30, help="Timestep in minutes (default: 30)")
    parser.add_argument("--tariff", type=str, default="cosy", choices=["cosy"], help="Tariff profile to use (cosy tariff only)")
    parser.add_argument(
        "--patterns",
        nargs="+",
        default=list(PATTERN_WINDOWS.keys()),
        choices=list(PATTERN_WINDOWS.keys()),
        help=(
            "Demand pattern(s) to evaluate. Choose one or more of: two_peaks, "
            "evening_peak, daytime_continuous"
        ),
    )
    return parser.parse_args(list(args))


def main(argv: Iterable[str] | None = None) -> None:
    if argv is None:
        argv = sys.argv[1:]
    ns = parse_args(list(argv))
    run_batch(
        summary_path=ns.summary,
        weather_path=ns.weather,
        output_dir=ns.output_dir,
        start=ns.start,
        end=ns.end,
        step_minutes=ns.step_minutes,
        tariff_type=ns.tariff,
        demand_patterns=ns.patterns,
    )


if __name__ == "__main__":
    main()
