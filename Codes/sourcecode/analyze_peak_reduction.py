"""Compute and visualise peak demand reductions under extreme weather scenarios."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from Codes.sourcecode.generate_demand_metrics import (
    BOILER_EFFICIENCY,
    DEMAND_PROFILES_ROOT,
    HEAT_PUMP_COP,
    OUTPUT_FILE,
    compute_metrics,
)


DWELLING_COL = "Dwelling ID"
PEAK_COL = "Peak Electricity Consumption (Feb 11) 1600-1900"
TOTAL_ELECTRICITY_COL = "Total Electricity Consumption (Feb10-12)"
TOTAL_GAS_COL = "Total gas Consumption (Feb10-12)"
TARIFF_COL = "Tariff Type"
WEATHER_COL = "Weather(mild/extreme)"
DEVICE_COL = "Device(HHP/mHP+capacity)"
R_COL = "R(K/W)"
C_COL = "C(J/K)"
G_COL = "g(m^2)"
THERMAL_CONSTANT_COL = "Thermal Constant (R*C)"
REDUCTION_COL = "Peak Demand Reduction (Flat -> Time-of-use) [kWh]"
HTC_COL = "HTC (1/R)"
HP_PROPORTION_PREFIX = "HP Heat Proportion"
HP_PROPORTION_FLAT_COL = f"{HP_PROPORTION_PREFIX} (Flat)"
HP_PROPORTION_TOU_COL = f"{HP_PROPORTION_PREFIX} (Time-of-use)"

REQUIRED_COLUMNS = [
    DWELLING_COL,
    PEAK_COL,
    TOTAL_ELECTRICITY_COL,
    TOTAL_GAS_COL,
    TARIFF_COL,
    WEATHER_COL,
    DEVICE_COL,
    R_COL,
    C_COL,
    G_COL,
]

COLUMN_ALIASES = {
    "dwellingid": DWELLING_COL,
    "devicehhpmhpcapacity": DEVICE_COL,
    "tarifftype": TARIFF_COL,
    "weathermildextreme": WEATHER_COL,
    "peakelectricityconsumptionfeb1116001900": PEAK_COL,
    "peakelectricityconsumptionfeb1116001900kwh": PEAK_COL,
    "totalelectricityconsumptionfeb1012": "Total Electricity Consumption (Feb10-12)",
    "totalelectricityconsumptionfeb1012kwh": "Total Electricity Consumption (Feb10-12)",
    "totalgasconsumptionfeb1012": "Total gas Consumption (Feb10-12)",
    "totalgasconsumptionfeb1012kwh": "Total gas Consumption (Feb10-12)",
    "rkw": R_COL,
    "cjk": C_COL,
    "gm2": G_COL,
}


def _candidate_summaries(explicit: Path | None) -> Iterable[Path]:
    """Yield candidate summary CSV locations to load the metrics from."""

    seen: set[Path] = set()
    for candidate in (
        explicit,
        DEMAND_PROFILES_ROOT / "demand_metrics_summary.csv",
        OUTPUT_FILE,
    ):
        if candidate is None:
            continue
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        yield candidate


def _normalise_column_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).lower())


def _standardise_columns(frame: pd.DataFrame) -> pd.DataFrame:
    rename: dict[str, str] = {}
    for column in frame.columns:
        key = _normalise_column_name(column)
        target = COLUMN_ALIASES.get(key)
        if target:
            rename[column] = target

    if rename:
        frame = frame.rename(columns=rename)

    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise KeyError(
            "Missing required columns in metrics CSV: "
            + ", ".join(missing)
        )

    return frame


def load_metrics(summary_path: Path | None = None) -> pd.DataFrame:
    """Load the pre-computed metrics or build them if no CSV is present."""

    for candidate in _candidate_summaries(summary_path):
        if candidate.exists():
            return _standardise_columns(pd.read_csv(candidate))

    # Fall back to recomputing everything from the demand profiles.
    return _standardise_columns(compute_metrics())


def _extract_device_fields(frame: pd.DataFrame) -> pd.DataFrame:
    """Split the device description into technology and numeric capacity."""

    frame = frame.copy()

    technology = frame[DEVICE_COL].str.extract(r"(HHP|mHP)", expand=False)
    fallback_tech = frame[DEVICE_COL].str.split().str[0]
    frame["Technology"] = technology.fillna(fallback_tech)

    capacity_kw = frame[DEVICE_COL].str.extract(
        r"([0-9]+(?:\.[0-9]+)?)\s*kW",
        expand=False,
    )
    if capacity_kw.isna().any():
        loose_match = frame[DEVICE_COL].str.extract(
            r"([0-9]+(?:\.[0-9]+)?)",
            expand=False,
        )
        capacity_kw = capacity_kw.fillna(loose_match)

    frame["Capacity kW"] = pd.to_numeric(capacity_kw, errors="coerce")
    return frame


def _normalise_tariff(values: pd.Series) -> pd.Series:
    """Coerce tariff labels into the Flat/Time-of-use buckets used downstream."""

    def _bucket(value: str) -> str:
        text = str(value).strip().lower()
        if "flat" in text:
            return "Flat"
        if "time" in text or "tou" in text:
            return "Time-of-use"
        # Fall back to the original value so unexpected labels are still visible.
        return str(value)

    return values.apply(_bucket)


def compute_peak_reduction(metrics: pd.DataFrame) -> pd.DataFrame:
    """Calculate the peak demand reduction from Flat to Time-of-use tariffs."""

    if PEAK_COL not in metrics.columns:
        raise KeyError(f"'{PEAK_COL}' column is required in the metrics table")

    weather_series = metrics[WEATHER_COL].astype(str).str.strip().str.lower()
    extreme = metrics.loc[weather_series == "extreme"].copy()
    if extreme.empty:
        return pd.DataFrame(
            columns=[
                DWELLING_COL,
                "Technology",
                "Capacity kW",
                "Flat",
                "Time-of-use",
                R_COL,
                C_COL,
                G_COL,
                REDUCTION_COL,
                THERMAL_CONSTANT_COL,
                HTC_COL,
                HP_PROPORTION_FLAT_COL,
                HP_PROPORTION_TOU_COL,
            ]
        )

    extreme = _extract_device_fields(extreme)
    extreme[TARIFF_COL] = _normalise_tariff(extreme[TARIFF_COL])

    index_cols = [DWELLING_COL, "Technology", "Capacity kW"]

    grouped = extreme.groupby(
        index_cols + [TARIFF_COL],
        dropna=False,
    )[PEAK_COL]
    aggregated = grouped.mean().reset_index()

    total_electricity = pd.to_numeric(
        extreme[TOTAL_ELECTRICITY_COL], errors="coerce"
    ).fillna(0.0)
    total_gas = pd.to_numeric(extreme[TOTAL_GAS_COL], errors="coerce").fillna(0.0)
    extreme = extreme.assign(
        _hp_heat_kwh=total_electricity * HEAT_PUMP_COP,
        _boiler_heat_kwh=total_gas * BOILER_EFFICIENCY,
    )

    heat_totals = (
        extreme.groupby(index_cols + [TARIFF_COL], dropna=False)[
            ["_hp_heat_kwh", "_boiler_heat_kwh"]
        ]
        .sum()
        .reset_index()
    )
    heat_totals["_total_heat_kwh"] = (
        heat_totals["_hp_heat_kwh"] + heat_totals["_boiler_heat_kwh"]
    )
    with pd.option_context("mode.use_inf_as_na", True):
        heat_totals["hp_heat_proportion"] = (
            heat_totals["_hp_heat_kwh"]
            .div(heat_totals["_total_heat_kwh"])
            .where(heat_totals["_total_heat_kwh"] > 0)
        )

    proportion_pivot = (
        heat_totals.pivot(
            index=index_cols, columns=TARIFF_COL, values="hp_heat_proportion"
        )
        .rename(
            columns={
                "Flat": HP_PROPORTION_FLAT_COL,
                "Time-of-use": HP_PROPORTION_TOU_COL,
            }
        )
        .reset_index()
    )
    for column in (HP_PROPORTION_FLAT_COL, HP_PROPORTION_TOU_COL):
        if column not in proportion_pivot.columns:
            proportion_pivot[column] = float("nan")

    pivot = (
        aggregated.pivot(index=index_cols, columns=TARIFF_COL, values=PEAK_COL)
        .reindex(columns=["Flat", "Time-of-use"])
        .rename_axis(columns=None)
        .reset_index()
    )

    parameters = (
        extreme.groupby(index_cols, dropna=False)[[R_COL, C_COL, G_COL]].first().reset_index()
    )
    pivot = pivot.merge(parameters, on=index_cols, how="left")
    pivot = pivot.merge(proportion_pivot, on=index_cols, how="left")

    pivot = pivot.dropna(subset=["Flat", "Time-of-use"])

    pivot[REDUCTION_COL] = pivot["Flat"] - pivot["Time-of-use"]
    pivot[THERMAL_CONSTANT_COL] = pivot[R_COL] * pivot[C_COL]
    pivot[HTC_COL] = (
        (1.0 / pivot[R_COL]).where(pivot[R_COL].notna() & (pivot[R_COL] != 0))
    )

    column_order = [
        DWELLING_COL,
        "Technology",
        "Capacity kW",
        "Flat",
        "Time-of-use",
        HP_PROPORTION_FLAT_COL,
        HP_PROPORTION_TOU_COL,
        R_COL,
        C_COL,
        G_COL,
        HTC_COL,
        THERMAL_CONSTANT_COL,
        REDUCTION_COL,
    ]
    existing = [column for column in column_order if column in pivot.columns]
    remaining = [column for column in pivot.columns if column not in existing]
    pivot = pivot[existing + remaining]

    return pivot.sort_values(["Technology", DWELLING_COL]).reset_index(drop=True)


def plot_histograms(
    reduction: pd.DataFrame,
    output_dir: Path,
    metric_col: str = REDUCTION_COL,
    metric_label: str = "Peak demand reduction (kWh)",
    file_prefix: str = "peak_demand_reduction_hist",
    units: str | None = "kWh",
) -> dict[str, Path]:
    """Save histogram figures for each technology and return their file paths."""

    output_dir.mkdir(parents=True, exist_ok=True)
    figure_paths: dict[str, Path] = {}

    for technology, subset in reduction.groupby("Technology"):
        values = subset[metric_col].dropna()
        if values.empty:
            continue

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(
            values,
            bins=30,
            color="tab:blue" if technology == "HHP" else "tab:orange",
            edgecolor="black",
        )
        ax.set_title(f"{metric_label} distribution ({technology})")
        ax.set_xlabel(metric_label)
        ax.set_ylabel("Number of dwellings")
        mean_value = values.mean()
        mean_label = f"Mean: {mean_value:.2f}"
        if units:
            mean_label += f" {units}"
        ax.axvline(mean_value, color="red", linestyle="--", label=mean_label)
        ax.legend()
        fig.tight_layout()

        filename = f"{file_prefix}_{technology.lower()}.png"
        path = output_dir / filename
        fig.savefig(path, dpi=300)
        plt.close(fig)
        figure_paths[technology] = path

    return figure_paths


def plot_scatter_matrix(
    reduction: pd.DataFrame,
    output_path: Path,
    metric_col: str = REDUCTION_COL,
    metric_label: str = "Peak demand reduction (kWh)",
    parameters: list[tuple[str, str]] | None = None,
) -> Path:
    """Create a scatter matrix of the metric against building parameters."""

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if parameters is None:
        parameters = [
            (R_COL, "R (K/W)"),
            (C_COL, "C (J/K)"),
            (G_COL, "g (m$^2$)"),
            (THERMAL_CONSTANT_COL, "Thermal constant (R*C)"),
            (HTC_COL, "Heat transfer coefficient (1/R)"),
        ]
    colors = {"HHP": "tab:blue", "mHP": "tab:orange"}

    num_plots = len(parameters)
    if num_plots == 0:
        raise ValueError("No parameters provided for scatter plotting")

    ncols = 3 if num_plots > 1 else 1
    nrows = (num_plots + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    axes_array = np.array(axes, ndmin=1)
    axes = [ax for ax in axes_array.ravel()]

    for ax, (column, label) in zip(axes, parameters):
        for technology, subset in reduction.groupby("Technology"):
            valid = subset[column].notna() & subset[metric_col].notna()
            if not valid.any():
                continue
            ax.scatter(
                subset.loc[valid, column],
                subset.loc[valid, metric_col],
                label=technology,
                alpha=0.7,
                s=30,
                color=colors.get(technology, "tab:gray"),
                edgecolors="none",
            )
        ax.set_xlabel(label)
        ax.set_ylabel(metric_label)
        ax.grid(True, alpha=0.3)

    for ax in axes[num_plots:]:
        ax.set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=len(labels))

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary",
        type=Path,
        default=None,
        help="Path to the pre-computed demand metrics summary CSV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEMAND_PROFILES_ROOT / "peak_demand_reduction_extreme.csv",
        help="Where to write the peak reduction table.",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=DEMAND_PROFILES_ROOT / "figures",
        help="Directory to store generated figures.",
    )
    parser.add_argument(
        "--scatter-name",
        type=str,
        default="peak_demand_reduction_scatter.png",
        help="Filename for the scatter matrix plot.",
    )
    args = parser.parse_args()

    metrics = load_metrics(args.summary)
    reduction = compute_peak_reduction(metrics)
    reduction.to_csv(args.output, index=False)
    print(f"Saved peak reduction table with {len(reduction)} rows to {args.output}")

    hist_paths = plot_histograms(reduction, args.figures_dir)
    for tech, path in hist_paths.items():
        print(f"Saved {tech} histogram to {path}")

    scatter_path = plot_scatter_matrix(reduction, args.figures_dir / args.scatter_name)
    print(f"Saved scatter matrix to {scatter_path}")

    hhp_mask = reduction["Technology"].astype(str).str.upper() == "HHP"
    hhp_reduction = reduction.loc[hhp_mask].copy()

    if not hhp_reduction.empty:
        proportion_hist_paths = plot_histograms(
            hhp_reduction,
            args.figures_dir,
            metric_col=HP_PROPORTION_TOU_COL,
            metric_label="Heat pump heat proportion (Time-of-use)",
            file_prefix="hp_heat_proportion_hist",
            units=None,
        )
        for tech, path in proportion_hist_paths.items():
            print(f"Saved {tech} heat proportion histogram to {path}")

        proportion_scatter_path = plot_scatter_matrix(
            hhp_reduction,
            args.figures_dir / "hp_heat_proportion_scatter.png",
            metric_col=HP_PROPORTION_TOU_COL,
            metric_label="Heat pump heat proportion (Time-of-use)",
        )
        print(f"Saved heat proportion scatter matrix to {proportion_scatter_path}")
    else:
        print("No HHP rows available for heat proportion plots; skipping figure generation.")


if __name__ == "__main__":
    main()
