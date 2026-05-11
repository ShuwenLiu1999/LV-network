"""Instance-based simulation runner helpers for `simulation.ipynb`.

The simulation notebook should describe what to run. This module owns the
repeatable plumbing: context slicing, case parameter assembly, continuation
checks, per-dwelling cache writes, and manifest rows.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import stochastic_baseload_multiple_building_simulation_and_aggregation as sbm
from artifact_naming import offset_hour_token, safe_tag


SUPPORTED_CASES = {"hybrid", "monovalent", "boiler_only"}
SUPPORTED_TARIFFS = {"flat", "cozy", "agile"}
SUPPORTED_MODES = {"aggregate_batch", "per_dwelling_cache"}


def _as_list(value: Any, *, default: Any = None) -> list[Any]:
    if value is None:
        value = default
    if isinstance(value, (list, tuple, set, np.ndarray, pd.Index)):
        return list(value)
    return [value]


def _status_counts(summary_runs: Sequence[pd.DataFrame]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for frame in summary_runs:
        if not isinstance(frame, pd.DataFrame) or frame.empty or "solve_status" not in frame.columns:
            continue
        value_counts = frame["solve_status"].dropna().astype(str).value_counts(dropna=False)
        for key, value in value_counts.items():
            counts[str(key)] = int(counts.get(str(key), 0) + int(value))
    return counts


def _run_number_from_path(path: Path) -> int:
    try:
        return int(path.stem.split("_")[-1])
    except Exception:
        return -1


def _ev_kw_token(value: float) -> str:
    value = float(value)
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return str(value).replace(".", "p")


def _date_token(value: Any) -> str:
    if value is None:
        return "startBase"
    try:
        return "start" + pd.Timestamp(value).strftime("%Y%m%d")
    except Exception:
        return "start" + safe_tag(value)


def _days_token(value: Any) -> str:
    if value is None:
        return "daysBase"
    try:
        return "days" + str(int(value))
    except Exception:
        return "days" + safe_tag(value)


def _runs_token(instance: Mapping[str, Any]) -> str:
    value = instance.get("target_runs", instance.get("mc_runs"))
    if value is None:
        return "runsBase"
    try:
        return "runs" + str(int(value))
    except Exception:
        return "runs" + safe_tag(value)


def _seed_token(instance: Mapping[str, Any]) -> str:
    value = instance.get("offset_seed_base")
    if value is None:
        return "seedBase"
    try:
        return "seed" + str(int(value))
    except Exception:
        return "seed" + safe_tag(value)


def _dwelling_selection_token(value: Any) -> str:
    if value is None:
        return "dwAll"
    if isinstance(value, int):
        return f"dwFirst{int(value)}"
    if isinstance(value, str):
        return "dwID" + safe_tag(value)
    values = list(value)
    if len(values) <= 3:
        return "dwIDs" + "-".join(safe_tag(item) for item in values)
    return f"dw{len(values)}IDs"


def _dwelling_token(value: Any) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))


def build_output_folder_name(instance: Mapping[str, Any], *, effective_ev_kw: float | None = None) -> str:
    """Build the systematic bottom-level output folder name for an instance."""

    tariff = str(instance.get("tariff_type", "agile")).strip().lower()
    case = str(instance.get("case", "monovalent")).strip().lower()
    mode = str(instance.get("mode", "per_dwelling_cache")).strip().lower()
    mode_token = "batch" if mode == "aggregate_batch" else "cache"
    ev_kw = float(instance.get("ev_charge_max_kw", 5.0) if effective_ev_kw is None else effective_ev_kw)
    ev_token = "EVoff" if bool(instance.get("bypass_ev", False)) else f"EV{_ev_kw_token(ev_kw)}kW"
    offset_token = "offset" + offset_hour_token(float(instance.get("offset_max_hours", 0.0)))
    parts = [
        mode_token,
        "tariff-" + tariff,
        "case-" + case,
        ev_token,
        offset_token,
        _date_token(instance.get("start_date")),
        _days_token(instance.get("n_days")),
        _runs_token(instance),
        _seed_token(instance),
        _dwelling_selection_token(instance.get("selected_dwellings")),
    ]
    return safe_tag("_".join(parts))


def expand_simulation_instances(instances: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Expand list-valued case/EV/offset fields into one runnable instance each."""

    expanded: list[dict[str, Any]] = []
    for base_index, raw_instance in enumerate(instances):
        base = dict(raw_instance)
        base_name = base.pop("name", None)
        cases = _as_list(base.pop("cases", base.pop("case", None)), default="monovalent")
        ev_values = _as_list(base.pop("ev_charge_max_kw_list", base.pop("ev_charge_max_kw", None)), default=5.0)
        offsets = _as_list(base.pop("offset_max_hours_list", base.pop("offset_max_hours", None)), default=0.0)
        n_combinations = len(cases) * len(ev_values) * len(offsets)
        for case in cases:
            for ev_kw in ev_values:
                for offset_hours in offsets:
                    instance = dict(base)
                    instance["case"] = str(case).strip().lower()
                    instance["ev_charge_max_kw"] = float(ev_kw)
                    instance["offset_max_hours"] = float(offset_hours)
                    if base_name and n_combinations == 1:
                        instance["name"] = base_name
                    elif base_name:
                        instance["name"] = _default_instance_name(
                            {**instance, "name_prefix": base_name},
                            fallback_index=base_index,
                        )
                    else:
                        instance["name"] = _default_instance_name(instance, fallback_index=base_index)
                    expanded.append(instance)
    return expanded


def _default_instance_name(instance: Mapping[str, Any], *, fallback_index: int = 0) -> str:
    prefix = str(instance.get("name_prefix", f"{fallback_index:02d}"))
    return safe_tag(f"{prefix}_{build_output_folder_name(instance)}")


def _effective_ev_params(base_ev_params: Mapping[str, Any], *, ev_charge_max_kw: float, bypass_ev: bool) -> tuple[dict[str, Any], float]:
    ev_params = dict(base_ev_params)
    if bypass_ev:
        ev_params.update(
            {
                "ev_capacity": 0.0,
                "ev_soc_init": 0.0,
                "ev_target": 0.0,
                "ev_charge_max": 0.0,
                "ev_min_final_fraction": 0.0,
                "ev_retention": 1.0,
                "ev_precheck_enabled": False,
            }
        )
        return ev_params, 0.0
    ev_params["ev_charge_max"] = float(ev_charge_max_kw)
    return ev_params, float(ev_charge_max_kw)


def _case_parameters(
    case: str,
    *,
    base_optim_params: Mapping[str, Any],
    case_parameter_cfg: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    case = str(case).strip().lower()
    if case not in SUPPORTED_CASES:
        raise ValueError(f"Unsupported case {case!r}; expected one of {sorted(SUPPORTED_CASES)}")
    if case not in case_parameter_cfg:
        raise ValueError(f"Missing case_parameter_cfg entry for {case!r}")
    cfg = dict(case_parameter_cfg[case])
    optim = dict(base_optim_params)
    optim.update(dict(cfg.get("optim", {})))
    hw = dict(cfg.get("hw", {}))
    return optim, hw


def build_instance_context(
    base_context: Mapping[str, Any],
    *,
    start_date: Any = None,
    n_days: int | None = None,
    tariff_type: str | None = None,
    include_flex_setpoint: bool = False,
) -> dict[str, Any]:
    """Return a context copy for a specific date window and tariff."""

    run_context = dict(base_context)
    base_window = base_context["window"]
    step = base_context["step"]
    steps_per_day = int(base_context["steps_per_day"])
    base_start = pd.Timestamp(base_window.index[0]).normalize()
    base_end = pd.Timestamp(base_window.index[-1]).normalize()
    effective_start = base_start if start_date is None else pd.Timestamp(start_date).normalize()
    effective_n_days = int(base_context["n_days"]) if n_days is None else int(n_days)

    if effective_n_days <= 0:
        raise ValueError("n_days must be positive.")
    if effective_start < base_start or effective_start > base_end:
        raise ValueError(
            f"start_date={effective_start.date()} is outside prepared context range "
            f"[{base_start.date()} .. {base_end.date()}]."
        )

    end_exclusive = effective_start + pd.Timedelta(days=effective_n_days)
    window = base_window.loc[effective_start : end_exclusive - pd.Timedelta(step)]
    if window.empty:
        raise ValueError(f"Selected period is empty: start={effective_start.date()}, n_days={effective_n_days}.")

    trailing_steps = int(len(window) % steps_per_day)
    if trailing_steps:
        window = window.iloc[:-trailing_steps]
    if len(window) < steps_per_day:
        raise ValueError(f"Selected period is shorter than one day after alignment: {len(window)} steps.")

    run_context["window"] = window
    run_context["n_days"] = int(len(window) // steps_per_day)
    run_context["requested_n_days"] = int(effective_n_days)
    run_context["n_steps"] = int(len(window))

    tout_series = pd.Series(np.asarray(base_context["Tout"], dtype=float), index=base_window.index)
    solar_series = pd.Series(np.asarray(base_context["S"], dtype=float), index=base_window.index)
    run_context["Tout"] = tout_series.loc[window.index].to_numpy(dtype=float)
    run_context["S"] = solar_series.loc[window.index].to_numpy(dtype=float)

    tariff_name = str(tariff_type or run_context.get("tariff_type", "agile")).strip().lower()
    if tariff_name not in SUPPORTED_TARIFFS:
        raise ValueError(f"tariff_type must be one of {sorted(SUPPORTED_TARIFFS)}")
    tariff = sbm.build_tariff(window.index[0], n_days=run_context["n_days"], step=step, type=tariff_name)
    if not tariff.index.equals(window.index):
        tariff = tariff.reindex(window.index, method="ffill")
    run_context["tariff"] = tariff
    run_context["tariff_type"] = tariff_name
    run_context["setpoint_sequences"] = sbm.build_setpoint_sequences(
        tariff.index,
        include_flex=bool(include_flex_setpoint),
    )
    return run_context


def resolve_dwellings(context: Mapping[str, Any], selected_dwellings: Any = None) -> list[Any]:
    """Resolve a notebook dwelling selection to ordered dwelling IDs."""

    available = list(context["dwelling_inputs"].keys())
    if selected_dwellings is None:
        return available
    if isinstance(selected_dwellings, int):
        return available[: int(selected_dwellings)]
    raw_values = [selected_dwellings] if isinstance(selected_dwellings, str) else list(selected_dwellings)
    resolved: list[Any] = []
    seen: set[Any] = set()
    for value in raw_values:
        candidate = value if value in context["dwelling_inputs"] else int(value)
        if candidate not in context["dwelling_inputs"]:
            raise KeyError(f"Unknown dwelling id: {value}")
        if candidate in seen:
            continue
        seen.add(candidate)
        resolved.append(candidate)
    return resolved


def _default_output_subdir(instance: Mapping[str, Any], *, effective_ev_kw: float) -> str:
    if instance.get("output_subdir"):
        return str(instance["output_subdir"]).replace("\\", "/")
    output_group = str(instance.get("output_group", "Single Dwelling Runs/instance runs")).strip("/\\")
    folder = build_output_folder_name(instance, effective_ev_kw=effective_ev_kw)
    return f"{output_group}/{folder}" if output_group else folder


def preview_simulation_instances(
    base_context: Mapping[str, Any],
    instances: Sequence[Mapping[str, Any]],
    *,
    defaults: Mapping[str, Any],
) -> pd.DataFrame:
    """Return a dry-run manifest of expanded simulation instances."""

    rows: list[dict[str, Any]] = []
    for instance in expand_simulation_instances(instances):
        merged = {**defaults, **instance}
        mode = str(merged.get("mode", "per_dwelling_cache")).strip().lower()
        if mode not in SUPPORTED_MODES:
            raise ValueError(f"mode must be one of {sorted(SUPPORTED_MODES)}")
        ev_params, effective_ev_kw = _effective_ev_params(
            defaults.get("ev_params_cfg", {}),
            ev_charge_max_kw=float(merged.get("ev_charge_max_kw", defaults.get("ev_charge_max_kw", 5.0))),
            bypass_ev=bool(merged.get("bypass_ev", False)),
        )
        run_context = build_instance_context(
            base_context,
            start_date=merged.get("start_date"),
            n_days=merged.get("n_days"),
            tariff_type=merged.get("tariff_type", defaults.get("tariff_type", "agile")),
            include_flex_setpoint=bool(defaults.get("include_flex_setpoint", False)),
        )
        output_subdir = _default_output_subdir(merged, effective_ev_kw=effective_ev_kw)
        output_folder = Path(output_subdir).name
        dwellings = resolve_dwellings(run_context, merged.get("selected_dwellings", defaults.get("selected_dwellings")))
        rows.append(
            {
                "name": merged.get("name"),
                "mode": mode,
                "case": str(merged.get("case", defaults.get("case", "monovalent"))).strip().lower(),
                "tariff_type": str(merged.get("tariff_type", defaults.get("tariff_type", "agile"))).strip().lower(),
                "start": run_context["window"].index.min(),
                "end": run_context["window"].index.max(),
                "n_days": int(run_context["n_days"]),
                "n_dwellings": int(len(dwellings)),
                "target_runs": int(merged.get("target_runs", merged.get("mc_runs", defaults.get("target_runs", 1)))),
                "ev_charge_max_kw": float(effective_ev_kw),
                "offset_max_hours": float(merged.get("offset_max_hours", 0.0)),
                "continue_from_existing": bool(merged.get("continue_from_existing", defaults.get("continue_from_existing", False))),
                "output_folder": output_folder,
                "output_subdir": output_subdir,
                "ev_precheck_enabled": bool(ev_params.get("ev_precheck_enabled", True)),
            }
        )
    return pd.DataFrame(rows)


def _existing_run_stats(breakdown_path: Path) -> dict[str, int]:
    if not breakdown_path.exists():
        return {"completed_runs": 0, "max_run": 0}
    try:
        run_series = pd.read_csv(breakdown_path, usecols=["run"])["run"]
    except Exception:
        return {"completed_runs": 0, "max_run": 0}
    run_values = pd.to_numeric(run_series, errors="coerce").dropna().astype(int)
    if run_values.empty:
        return {"completed_runs": 0, "max_run": 0}
    return {"completed_runs": int(run_values.nunique()), "max_run": int(run_values.max())}


def _merge_new_runs(existing_path: Path, new_runs_path: Path) -> None:
    if not new_runs_path.exists():
        return
    try:
        new_df = pd.read_csv(new_runs_path)
    except Exception:
        new_runs_path.unlink(missing_ok=True)
        return
    if new_df.empty:
        new_runs_path.unlink(missing_ok=True)
        return

    existing_df = pd.DataFrame()
    if existing_path.exists() and existing_path.stat().st_size > 0:
        try:
            existing_df = pd.read_csv(existing_path)
        except Exception:
            existing_df = pd.DataFrame()

    merged = pd.concat([existing_df, new_df], ignore_index=True)
    dedupe_cols = [col for col in ["run", "time"] if col in merged.columns]
    merged = merged.drop_duplicates(subset=dedupe_cols, keep="last") if dedupe_cols else merged.drop_duplicates(keep="last")
    sort_cols = [col for col in ["run", "time"] if col in merged.columns]
    if sort_cols:
        merged = merged.sort_values(sort_cols, kind="mergesort")
    merged.to_csv(existing_path, index=False)
    new_runs_path.unlink(missing_ok=True)


def _export_aggregate_artifacts(mc_results: Mapping[str, Any], output_dir: Path, *, prefix: str) -> dict[str, str]:
    artifact_dir = output_dir / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    profile_usage_csv = artifact_dir / f"{prefix}_profile_usage.csv"
    pd.DataFrame(mc_results.get("profile_usage", [])).to_csv(profile_usage_csv, index=False)

    run_summary_frames = [frame for frame in mc_results.get("summary_runs", []) if isinstance(frame, pd.DataFrame)]
    run_summary_df = pd.concat(run_summary_frames, ignore_index=True) if run_summary_frames else pd.DataFrame()
    run_summary_csv = artifact_dir / f"{prefix}_run_summary.csv"
    run_summary_df.to_csv(run_summary_csv, index=False)

    run_files = sorted(output_dir.glob("mc_run_*.csv"), key=_run_number_from_path)
    run_files_csv = artifact_dir / f"{prefix}_run_files.csv"
    pd.DataFrame({"run": [_run_number_from_path(path) for path in run_files], "path": [str(path) for path in run_files]}).to_csv(
        run_files_csv,
        index=False,
    )
    return {
        "profile_usage_csv": str(profile_usage_csv),
        "run_summary_csv": str(run_summary_csv),
        "run_files_csv": str(run_files_csv),
    }


def run_simulation_instances(
    base_context: Mapping[str, Any],
    instances: Sequence[Mapping[str, Any]],
    *,
    defaults: Mapping[str, Any],
) -> pd.DataFrame:
    """Execute expanded simulation instances and return a manifest table."""

    expanded = expand_simulation_instances(instances)
    manifest_rows: list[dict[str, Any]] = []

    for instance in expanded:
        cfg = {**defaults, **instance}
        mode = str(cfg.get("mode", "per_dwelling_cache")).strip().lower()
        case = str(cfg.get("case", defaults.get("case", "monovalent"))).strip().lower()
        tariff_type = str(cfg.get("tariff_type", defaults.get("tariff_type", "agile"))).strip().lower()
        target_runs = int(cfg.get("target_runs", cfg.get("mc_runs", defaults.get("target_runs", 1))))
        continue_from_existing = bool(cfg.get("continue_from_existing", defaults.get("continue_from_existing", False)))
        show_progress = bool(cfg.get("show_progress", defaults.get("show_progress", True)))
        day_ahead = bool(cfg.get("day_ahead", defaults.get("day_ahead", True)))

        ev_params_cfg, effective_ev_kw = _effective_ev_params(
            defaults.get("ev_params_cfg", {}),
            ev_charge_max_kw=float(cfg.get("ev_charge_max_kw", defaults.get("ev_charge_max_kw", 5.0))),
            bypass_ev=bool(cfg.get("bypass_ev", False)),
        )
        optim_params_cfg, hw_params_cfg = _case_parameters(
            case,
            base_optim_params=defaults.get("optim_params_cfg", {}),
            case_parameter_cfg=defaults.get("case_parameter_cfg", {}),
        )
        run_context = build_instance_context(
            base_context,
            start_date=cfg.get("start_date"),
            n_days=cfg.get("n_days"),
            tariff_type=tariff_type,
            include_flex_setpoint=bool(defaults.get("include_flex_setpoint", False)),
        )
        selected_dwellings = cfg.get("selected_dwellings", defaults.get("selected_dwellings"))
        dwellings = resolve_dwellings(run_context, selected_dwellings)
        output_subdir = _default_output_subdir(cfg, effective_ev_kw=effective_ev_kw)
        output_folder = Path(output_subdir).name
        output_dir = Path(run_context["repo_root"]) / "Output Data" / output_subdir
        output_dir.mkdir(parents=True, exist_ok=True)

        offset_hours = float(cfg.get("offset_max_hours", 0.0))
        tariff_random_offset_cfg = {
            "enabled": bool(abs(offset_hours) > 1e-12),
            "max_offset_hours": offset_hours,
            "seed_base": int(cfg.get("offset_seed_base", defaults.get("offset_seed_base", 42))),
        }

        if mode == "aggregate_batch":
            mc_results = sbm.run_monte_carlo_batch(
                run_context,
                mc_runs=target_runs,
                run_index_start=int(cfg.get("run_index_start", 0)),
                case=case,
                output_subdir=output_subdir,
                selected_dwellings=selected_dwellings,
                capacity_candidates_kw=cfg.get("capacity_candidates_kw", defaults.get("capacity_candidates_kw")),
                optim_params_cfg=optim_params_cfg,
                ev_params_cfg=ev_params_cfg,
                hw_params_cfg=hw_params_cfg,
                tariff_random_offset_cfg=tariff_random_offset_cfg,
                day_ahead=day_ahead,
                save_outputs=bool(cfg.get("save_outputs", defaults.get("save_outputs", True))),
                show_progress=show_progress,
            )
            artifact_paths = {}
            if bool(cfg.get("write_artifacts", False)):
                artifact_paths = _export_aggregate_artifacts(
                    mc_results,
                    output_dir,
                    prefix=safe_tag(str(cfg.get("artifact_prefix", cfg.get("name", "simulation")))),
                )
            manifest_rows.append(
                {
                    "name": cfg.get("name"),
                    "mode": mode,
                    "tariff_type": tariff_type,
                    "case": case,
                    "ev_bypassed": bool(cfg.get("bypass_ev", False)),
                    "ev_charge_max_kw": float(effective_ev_kw),
                    "offset_max_hours": offset_hours,
                    "n_days": int(run_context["n_days"]),
                    "n_dwellings": int(len(dwellings)),
                    "runs_added": int(target_runs),
                    "target_total_runs": int(target_runs),
                    "final_total_runs": int(target_runs),
                    "status_counts": _status_counts(mc_results.get("summary_runs", [])),
                    "output_folder": output_folder,
                    "output_subdir": output_subdir,
                    "output_dir": str(output_dir),
                    **artifact_paths,
                }
            )
            continue

        if mode != "per_dwelling_cache":
            raise ValueError(f"Unsupported simulation mode {mode!r}; expected one of {sorted(SUPPORTED_MODES)}")

        capacity_filename = "dwelling_monovalent_hp_capacity_summary.csv"
        capacity_csv = output_dir / capacity_filename
        capacity_rows: list[dict[str, Any]] = []
        existing_capacity_df = pd.DataFrame()
        if case == "monovalent" and continue_from_existing and capacity_csv.exists():
            try:
                existing_capacity_df = pd.read_csv(capacity_csv)
            except Exception:
                existing_capacity_df = pd.DataFrame()

        progress = tqdm(dwellings, desc=str(cfg.get("name", case)), unit="dwelling", dynamic_ncols=True) if show_progress else dwellings
        for dwelling_id in progress:
            token = _dwelling_token(dwelling_id)
            single_output_path = output_dir / f"dwelling_{token}_runs_breakdown.csv"
            stats = _existing_run_stats(single_output_path) if continue_from_existing else {"completed_runs": 0, "max_run": 0}
            existing_completed = int(stats["completed_runs"])
            existing_max_run = int(stats["max_run"])
            runs_to_add = max(0, target_runs - existing_completed) if continue_from_existing else target_runs
            run_index_start = existing_max_run if continue_from_existing else 0

            batch_output_path = single_output_path
            if continue_from_existing and single_output_path.exists() and runs_to_add > 0:
                batch_output_path = output_dir / f"dwelling_{token}_runs_breakdown_new_runs.csv"

            if runs_to_add > 0:
                mc_results = sbm.run_monte_carlo_batch(
                    run_context,
                    mc_runs=runs_to_add,
                    run_index_start=run_index_start,
                    case=case,
                    output_subdir=output_subdir,
                    selected_dwellings=[dwelling_id],
                    capacity_candidates_kw=cfg.get("capacity_candidates_kw", defaults.get("capacity_candidates_kw")),
                    optim_params_cfg=optim_params_cfg,
                    ev_params_cfg=ev_params_cfg,
                    hw_params_cfg=hw_params_cfg,
                    tariff_random_offset_cfg=tariff_random_offset_cfg,
                    single_dwelling_id=dwelling_id,
                    single_dwelling_output_path=batch_output_path,
                    day_ahead=day_ahead,
                    save_outputs=False,
                    show_progress=False,
                )
                if batch_output_path != single_output_path:
                    _merge_new_runs(single_output_path, batch_output_path)
            else:
                mc_results = {"summary_runs": []}

            status_counts = _status_counts(mc_results.get("summary_runs", []))
            max_hp_capacity_kw = np.nan
            n_optimal_runs = 0
            n_infeasible_runs = 0
            if case == "monovalent":
                hp_capacity_values: list[float] = []
                for frame in mc_results.get("summary_runs", []):
                    if not isinstance(frame, pd.DataFrame) or frame.empty:
                        continue
                    if "solve_status" in frame.columns:
                        status_norm = frame["solve_status"].astype(str).str.lower()
                        n_optimal_runs += int(status_norm.eq("optimal").sum())
                        n_infeasible_runs += int(status_norm.eq("infeasible").sum())
                    if "hp_capacity_kw" in frame.columns:
                        caps = pd.to_numeric(frame["hp_capacity_kw"], errors="coerce").dropna()
                        hp_capacity_values.extend(caps.tolist())
                if hp_capacity_values:
                    max_hp_capacity_kw = float(np.nanmax(np.asarray(hp_capacity_values, dtype=float)))

                prev_opt = 0
                prev_infeasible = 0
                prev_max_hp_capacity_kw = np.nan
                if not existing_capacity_df.empty and "dwelling_id" in existing_capacity_df.columns:
                    prev_row = existing_capacity_df.loc[existing_capacity_df["dwelling_id"].astype(str) == str(dwelling_id)]
                    if not prev_row.empty:
                        if "n_optimal_runs" in prev_row.columns:
                            prev_opt = int(pd.to_numeric(prev_row["n_optimal_runs"], errors="coerce").fillna(0).iloc[-1])
                        if "n_infeasible_runs" in prev_row.columns:
                            prev_infeasible = int(pd.to_numeric(prev_row["n_infeasible_runs"], errors="coerce").fillna(0).iloc[-1])
                        if "max_hp_capacity_kw" in prev_row.columns:
                            prev_caps = pd.to_numeric(prev_row["max_hp_capacity_kw"], errors="coerce").dropna()
                            if not prev_caps.empty:
                                prev_max_hp_capacity_kw = float(prev_caps.iloc[-1])

                combined_max = max_hp_capacity_kw
                if np.isnan(combined_max):
                    combined_max = prev_max_hp_capacity_kw
                elif np.isfinite(prev_max_hp_capacity_kw):
                    combined_max = float(max(combined_max, prev_max_hp_capacity_kw))
                max_hp_capacity_kw = combined_max
                capacity_rows.append(
                    {
                        "dwelling_id": dwelling_id,
                        "max_hp_capacity_kw": combined_max,
                        "n_optimal_runs": int(prev_opt + n_optimal_runs),
                        "n_infeasible_runs": int(prev_infeasible + n_infeasible_runs),
                        "mc_runs": int(prev_opt + prev_infeasible + n_optimal_runs + n_infeasible_runs),
                    }
                )

            manifest_rows.append(
                {
                    "name": cfg.get("name"),
                    "mode": mode,
                    "tariff_type": tariff_type,
                    "case": case,
                    "ev_bypassed": bool(cfg.get("bypass_ev", False)),
                    "ev_charge_max_kw": float(effective_ev_kw),
                    "offset_max_hours": offset_hours,
                    "n_days": int(run_context["n_days"]),
                    "dwelling_id": dwelling_id,
                    "breakdown_path": str(single_output_path),
                    "existing_completed_runs": int(existing_completed),
                    "runs_added": int(runs_to_add),
                    "target_total_runs": int(target_runs),
                    "final_total_runs": int(existing_completed + runs_to_add),
                    "status_counts": status_counts,
                    "max_monovalent_hp_capacity_kw": max_hp_capacity_kw,
                    "output_folder": output_folder,
                    "output_subdir": output_subdir,
                    "output_dir": str(output_dir),
                }
            )

        if case == "monovalent":
            capacity_df = pd.DataFrame(capacity_rows)
            if not existing_capacity_df.empty:
                capacity_df = pd.concat([existing_capacity_df, capacity_df], ignore_index=True)
                if "dwelling_id" in capacity_df.columns:
                    capacity_df = capacity_df.drop_duplicates(subset=["dwelling_id"], keep="last")
            if not capacity_df.empty:
                capacity_df = capacity_df.sort_values("dwelling_id").reset_index(drop=True)
                capacity_df.to_csv(capacity_csv, index=False)

    manifest_df = pd.DataFrame(manifest_rows)
    if not manifest_df.empty:
        sort_cols = [col for col in ["name", "case", "offset_max_hours", "dwelling_id"] if col in manifest_df.columns]
        manifest_df = manifest_df.sort_values(sort_cols).reset_index(drop=True)
        first = {**defaults, **expanded[0]}
        manifest_output = first.get("manifest_output_csv")
        if manifest_output is None:
            first_ev_params, first_ev_kw = _effective_ev_params(
                defaults.get("ev_params_cfg", {}),
                ev_charge_max_kw=float(first.get("ev_charge_max_kw", defaults.get("ev_charge_max_kw", 5.0))),
                bypass_ev=bool(first.get("bypass_ev", False)),
            )
            first_output_subdir = _default_output_subdir(first, effective_ev_kw=first_ev_kw)
            first_output_dir = Path(base_context["repo_root"]) / "Output Data" / first_output_subdir
            manifest_output = first_output_dir.parent / "simulation_instance_manifest.csv"
        manifest_output = Path(manifest_output)
        if not manifest_output.is_absolute():
            manifest_output = Path(base_context["repo_root"]) / "Output Data" / manifest_output
        manifest_output.parent.mkdir(parents=True, exist_ok=True)
        manifest_df.to_csv(manifest_output, index=False)
        manifest_df.attrs["manifest_output_csv"] = str(manifest_output)
    return manifest_df
