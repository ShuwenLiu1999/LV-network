"""Utility to aggregate demand profile metrics across scenario folders.

This script reads the synthetic heating demand profiles located in
``Codes/Output/DemandProfiles`` and the building parameter summary stored in
``Codes/Data/1R1C1P1S_filtered_filtered.csv``.  It calculates the metrics needed
for the reporting template provided by the user and writes them to
``Codes/Output/demand_metrics_summary.csv``.  All energy results are reported in
kWh and the peak power in kW.

Example usage::

    python Codes/sourcecode/generate_demand_metrics.py
"""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DEMAND_PROFILES_ROOT = REPO_ROOT / "Codes" / "Output" / "DemandProfiles"
SUMMARY_FILE = REPO_ROOT / "Codes" / "Data" / "1R1C1P1S_filtered_filtered.csv"
OUTPUT_FILE = REPO_ROOT / "Codes" / "Output" / "demand_metrics_summary.csv"

# Physical parameters shared across all scenarios.
HEAT_PUMP_COP = 3.5
BOILER_EFFICIENCY = 0.9
TIME_STEP_HOURS = 0.5  # 30 minute time-step in the demand profiles.

# Date ranges required by the template.
PEAK_START = pd.Timestamp("2022-02-11 16:00:00")
PEAK_END = pd.Timestamp("2022-02-11 19:30:00")  # inclusive of the 19:00-19:30 slot.
AGGREGATION_START = pd.Timestamp("2022-02-10 00:00:00")
AGGREGATION_END = pd.Timestamp("2022-02-13 00:00:00")  # stop at 00:00 on Feb 13th.


@dataclass(frozen=True)
class ScenarioMetadata:
    """Metadata describing the demand scenario used to build the metrics."""

    technology: str  # "HHP" or "mHP"
    capacity_kw: float
    tariff: str  # e.g. "Flat", "Time-of-use"
    weather: str  # "mild" or "extreme"

    def device_label(self) -> str:
        if math.isnan(self.capacity_kw):
            return self.technology
        return f"{self.technology} {self.capacity_kw:.2f} kW"


def parse_capacity(folder: Path, csv_path: Path, frame: pd.DataFrame) -> float:
    """Infer the installed capacity from the profile data or file naming."""

    if "hp_capacity_kW" in frame.columns:
        value = pd.to_numeric(frame["hp_capacity_kW"], errors="coerce").dropna()
        if not value.empty:
            return float(value.iloc[0])

    match = re.search(r"_(\d+(?:\.\d+)?)kW", csv_path.stem)
    if match:
        return float(match.group(1))

    numeric_suffix = re.findall(r"(\d+)", folder.name)
    if len(numeric_suffix) >= 2:
        # Treat "HHP_4_24" as 4.24 kW for example.
        return float(f"{numeric_suffix[0]}.{numeric_suffix[1]}")
    if numeric_suffix:
        return float(numeric_suffix[0])

    return float("nan")


def decode_scenario(folder: Path, csv_path: Path, frame: pd.DataFrame) -> ScenarioMetadata:
    technology = "HHP" if folder.name.lower().startswith("hhp") else "mHP"
    capacity_kw = parse_capacity(folder, csv_path, frame)
    tariff = "Flat" if "flat" in folder.name.lower() else "Time-of-use"
    weather = "extreme" if "extreme" in folder.name.lower() else "mild"
    return ScenarioMetadata(technology=technology, capacity_kw=capacity_kw, tariff=tariff, weather=weather)


def load_summary(summary_path: Path) -> pd.DataFrame:
    summary = pd.read_csv(summary_path)
    summary["Dwelling ID"] = summary["dataset"].str.split("_", n=1).str[0]
    return summary[["Dwelling ID", "R1", "C1", "g"]].drop_duplicates()


def iter_profile_files(root: Path) -> Iterable[tuple[Path, Path]]:
    for folder in sorted(root.iterdir()):
        if not folder.is_dir():
            continue
        for csv_file in sorted(folder.glob("*.csv")):
            yield folder, csv_file


def compute_metrics(
    demand_profiles_root: Path = DEMAND_PROFILES_ROOT,
    summary_path: Path = SUMMARY_FILE,
) -> pd.DataFrame:
    summary = load_summary(summary_path)
    records = []

    for folder, csv_file in iter_profile_files(demand_profiles_root):
        frame = pd.read_csv(csv_file)
        frame["time"] = pd.to_datetime(frame["time"])
        frame["Q_hp"] = pd.to_numeric(frame["Q_hp"], errors="coerce").fillna(0.0)
        frame["Q_bo"] = pd.to_numeric(frame["Q_bo"], errors="coerce").fillna(0.0)

        scenario = decode_scenario(folder, csv_file, frame)
        dwelling_id = csv_file.stem.split("_")[0]

        frame["hp_elec_kw"] = frame["Q_hp"] / HEAT_PUMP_COP / 1000.0
        frame["boiler_gas_kw"] = frame["Q_bo"] / BOILER_EFFICIENCY / 1000.0

        peak_mask = (frame["time"] >= PEAK_START) & (frame["time"] < PEAK_END)
        aggregation_mask = (frame["time"] >= AGGREGATION_START) & (frame["time"] < AGGREGATION_END)

        peak_kw = frame.loc[peak_mask, "hp_elec_kw"].max()
        if pd.isna(peak_kw):
            peak_kw = 0.0

        total_elec_kwh = (frame.loc[aggregation_mask, "hp_elec_kw"] * TIME_STEP_HOURS).sum()
        total_gas_kwh = (frame.loc[aggregation_mask, "boiler_gas_kw"] * TIME_STEP_HOURS).sum()

        records.append(
            {
                "Dwelling ID": dwelling_id,
                "Device (HHP/mHP+capacity)": scenario.device_label(),
                "Tariff Type": scenario.tariff,
                "Weather (mild/extreme)": scenario.weather,
                "Peak Electricity Consumption (Feb 11 1600-1900) [kW]": round(peak_kw, 3),
                "Total Electricity Consumption (Feb10-12) [kWh]": round(total_elec_kwh, 3),
                "Total Gas Consumption (Feb10-12) [kWh]": round(total_gas_kwh, 3),
            }
        )

    metrics = pd.DataFrame.from_records(records)
    metrics = metrics.merge(summary, on="Dwelling ID", how="left")
    metrics.rename(columns={"R1": "R (K/W)", "C1": "C (J/K)", "g": "g (m^2)"}, inplace=True)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate heating demand profile metrics.")
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_FILE,
        help="Path to the CSV file where the metrics will be saved.",
    )
    args = parser.parse_args()

    metrics = compute_metrics()
    metrics.to_csv(args.output, index=False)
    print(f"Wrote {len(metrics)} rows to {args.output}")


if __name__ == "__main__":
    main()
