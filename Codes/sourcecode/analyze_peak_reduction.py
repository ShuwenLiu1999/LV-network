"""Compute and visualise peak demand reductions under extreme weather scenarios."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd

from Codes.sourcecode.generate_demand_metrics import (
    DEMAND_PROFILES_ROOT,
    OUTPUT_FILE,
    compute_metrics,
)


PEAK_COL = "Peak Electricity Consumption (Feb 11 1600-1900) [kWh]"
TARIFF_COL = "Tariff Type"
WEATHER_COL = "Weather (mild/extreme)"
DEVICE_COL = "Device (HHP/mHP+capacity)"
R_COL = "R (K/W)"
C_COL = "C (J/K)"
G_COL = "g (m^2)"
THERMAL_CONSTANT_COL = "Thermal Constant (R*C)"
REDUCTION_COL = "Peak Demand Reduction (Flat -> Time-of-use) [kWh]"


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


def load_metrics(summary_path: Path | None = None) -> pd.DataFrame:
    """Load the pre-computed metrics or build them if no CSV is present."""

    for candidate in _candidate_summaries(summary_path):
        if candidate.exists():
            return pd.read_csv(candidate)

    # Fall back to recomputing everything from the demand profiles.
    return compute_metrics()


def _extract_device_fields(frame: pd.DataFrame) -> pd.DataFrame:
    extracted = frame[DEVICE_COL].str.extract(
        r"^(?P<Technology>\\w+)(?:\\s+(?P<Capacity>[0-9.]+))?", expand=True
    )
    frame = frame.copy()
    frame["Technology"] = extracted["Technology"]
    frame["Capacity kW"] = pd.to_numeric(extracted["Capacity"], errors="coerce")
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

    extreme = metrics.loc[metrics[WEATHER_COL].str.lower() == "extreme"].copy()
    if extreme.empty:
        return pd.DataFrame(
            columns=[
                "Dwelling ID",
                "Technology",
                "Capacity kW",
                "Flat",
                "Time-of-use",
                R_COL,
                C_COL,
                G_COL,
                REDUCTION_COL,
                THERMAL_CONSTANT_COL,
            ]
        )

    extreme = _extract_device_fields(extreme)
    extreme[TARIFF_COL] = _normalise_tariff(extreme[TARIFF_COL])

    index_cols = ["Dwelling ID", "Technology", "Capacity kW"]

    pivot = (
        extreme.pivot_table(
            index=index_cols,
            columns=TARIFF_COL,
            values=PEAK_COL,
            aggfunc="first",
        )
        .reset_index()
        .rename_axis(columns=None)
    )

    for tariff in ("Flat", "Time-of-use"):
        if tariff not in pivot.columns:
            pivot[tariff] = float("nan")

    parameters = (
        extreme.groupby(index_cols, as_index=False)[[R_COL, C_COL, G_COL]].first()
    )
    pivot = pivot.merge(parameters, on=index_cols, how="left")

    pivot = pivot.dropna(subset=["Flat", "Time-of-use"])

    pivot[REDUCTION_COL] = pivot["Flat"] - pivot["Time-of-use"]
    pivot[THERMAL_CONSTANT_COL] = pivot[R_COL] * pivot[C_COL]

    return pivot.sort_values(["Technology", "Dwelling ID"]).reset_index(drop=True)


def plot_histograms(reduction: pd.DataFrame, output_dir: Path) -> dict[str, Path]:
    """Save histogram figures for each technology and return their file paths."""

    output_dir.mkdir(parents=True, exist_ok=True)
    figure_paths: dict[str, Path] = {}

    for technology, subset in reduction.groupby("Technology"):
        values = subset[REDUCTION_COL].dropna()
        if values.empty:
            continue

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(values, bins=30, color="tab:blue" if technology == "HHP" else "tab:orange", edgecolor="black")
        ax.set_title(f"Peak demand reduction distribution ({technology})")
        ax.set_xlabel("Peak demand reduction (kWh)")
        ax.set_ylabel("Number of dwellings")
        ax.axvline(values.mean(), color="red", linestyle="--", label=f"Mean: {values.mean():.2f} kWh")
        ax.legend()
        fig.tight_layout()

        filename = f"peak_demand_reduction_hist_{technology.lower()}.png"
        path = output_dir / filename
        fig.savefig(path, dpi=300)
        plt.close(fig)
        figure_paths[technology] = path

    return figure_paths


def plot_scatter_matrix(reduction: pd.DataFrame, output_path: Path) -> Path:
    """Create a scatter matrix of the reduction metric against building parameters."""

    output_path.parent.mkdir(parents=True, exist_ok=True)

    parameters = [
        (R_COL, "R (K/W)"),
        (C_COL, "C (J/K)"),
        (G_COL, "g (m$^2$)"),
        (THERMAL_CONSTANT_COL, "Thermal constant (R*C)")
    ]
    colors = {"HHP": "tab:blue", "mHP": "tab:orange"}

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for ax, (column, label) in zip(axes, parameters):
        for technology, subset in reduction.groupby("Technology"):
            ax.scatter(
                subset[column],
                subset[REDUCTION_COL],
                label=technology,
                alpha=0.7,
                s=30,
                color=colors.get(technology, "tab:gray"),
                edgecolors="none",
            )
        ax.set_xlabel(label)
        ax.set_ylabel("Peak demand reduction (kWh)")
        ax.grid(True, alpha=0.3)

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


if __name__ == "__main__":
    main()
