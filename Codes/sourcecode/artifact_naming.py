"""Artifact-name token helpers shared across notebooks."""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


def safe_tag(value: object) -> str:
    """Return a filesystem-friendly token while keeping readable words."""

    text = str(value).strip().replace(" ", "_").replace("/", "_per_")
    return "".join(ch if ch.isalnum() or ch in ["_", "-"] else "_" for ch in text).strip("_")


def num_tag(value: float | int) -> str:
    """Format a numeric value as a compact filename token."""

    text = f"{float(value):.6g}".replace("-", "m").replace(".", "p")
    return text or "0"


def pct_tag(value: float | int) -> str:
    """Format a fraction as a percentage token."""

    return num_tag(float(value) * 100.0) + "pct"


def range_tag(values: Iterable[float | int]) -> str:
    """Format the min/max of a numeric iterable as a percentage range token."""

    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return "none"
    return pct_tag(np.min(arr)) + "-" + pct_tag(np.max(arr))


def grid_tag(*value_sets: Iterable[float | int]) -> str:
    """Format the smallest non-zero grid spacing across one or more axes."""

    diffs: list[float] = []
    for values in value_sets:
        arr = np.unique(np.asarray(list(values), dtype=float))
        if arr.size > 1:
            diffs.extend(float(abs(diff)) for diff in np.diff(arr) if abs(diff) > 1e-12)
    return "gridSingle" if not diffs else "grid" + pct_tag(min(diffs))


def ev_source_tag(use_generated_ev_profiles: bool) -> str:
    """Tokenize whether EV profiles came from generated or cached curves."""

    return "evsrcGenerated" if bool(use_generated_ev_profiles) else "evsrcCached"


def levels_tag(raw_levels: object, *, prefix: str = "peaks", suffix: str = "kw") -> str:
    """Tokenize a scalar/list/string of contour levels."""

    if raw_levels is None:
        return f"{prefix}None"
    if isinstance(raw_levels, str):
        raw_levels = [part.strip() for part in raw_levels.split(",") if part.strip()]
    arr = np.atleast_1d(raw_levels).astype(float)
    arr = np.unique(arr[np.isfinite(arr)])
    if arr.size == 0:
        return f"{prefix}None"
    return prefix + "-".join(num_tag(value) for value in arr) + suffix


def co2_factor_tag(value: float | int) -> str:
    """Tokenize a CO2 factor measured in kg/kWh."""

    return "co2_" + num_tag(value) + "kgkwh"


def case_label_tag(label: object) -> str:
    """Convert an EV case label, such as `10% EV`, into a short token."""

    text = str(label).strip().lower().replace("% ev", "").replace("ev", "").replace("%", "").strip()
    try:
        return num_tag(float(text)) + "pct"
    except ValueError:
        return safe_tag(label).lower() or "case"


def ev_cases_tag(case_configs: Sequence[object]) -> str:
    """Tokenize the EV cases included in a multi-case plot/table."""

    labels: list[str] = []
    for case_cfg in case_configs:
        if isinstance(case_cfg, dict):
            fallback = Path(case_cfg.get("peak_result_csv", "case")).stem
            labels.append(case_label_tag(case_cfg.get("label", fallback)))
        else:
            labels.append(case_label_tag(Path(case_cfg).stem))
    return "evcases" + "-".join(labels) if labels else "evcasesNone"


def offset_hour_token(value: float | int) -> str:
    """Format an offset hour value using the existing case-folder convention."""

    return f"{float(value):.1f}".replace("-", "m").replace(".", "p") + "h"


def fraction_0_1(value: float | int, name: str) -> float:
    """Accept either a fraction or 0-100 percentage and return a fraction."""

    value = float(value)
    if value < 0:
        raise ValueError(f"{name} must be non-negative.")
    if value > 1.0:
        if value > 100.0:
            raise ValueError(f"{name} must be in [0, 1] or [0, 100].")
        value /= 100.0
    if value > 1.0 or math.isnan(value):
        raise ValueError(f"{name} must be in [0, 1] or [0, 100].")
    return value


def natural_key(path: object) -> list[object]:
    """Sort strings containing numbers in human/natural order."""

    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", str(path))]
