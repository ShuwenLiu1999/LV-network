"""Shared setup helpers for repository notebooks.

The active notebooks should stay focused on workflow choices. This module keeps
common path resolution and artifact-folder setup in one small place.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


@dataclass(frozen=True)
class NotebookPaths:
    """Canonical repository paths used by the main notebooks."""

    repo_root: Path
    source_dir: Path
    input_data_dir: Path
    output_data_dir: Path
    plots_dir: Path
    penetration_dir: Path
    penetration_csv_dir: Path
    penetration_plot_dir: Path


def find_repo_root(start: Path | str | None = None) -> Path:
    """Find the repository root from a notebook working directory."""

    current = Path.cwd().resolve() if start is None else Path(start).resolve()
    candidates = [current, *current.parents]
    for candidate in candidates:
        if (candidate / ".git").exists() or (candidate / "Codes").exists():
            return candidate
    raise FileNotFoundError(f"Could not find repository root from {current}")


def ensure_source_path(repo_root: Path | str, *, front: bool = True) -> Path:
    """Put `Codes/sourcecode` on `sys.path` and return that directory."""

    source_dir = Path(repo_root).resolve() / "Codes" / "sourcecode"
    source_dir_str = str(source_dir)
    if source_dir_str in sys.path:
        sys.path.remove(source_dir_str)
    if front:
        sys.path.insert(0, source_dir_str)
    else:
        sys.path.append(source_dir_str)
    return source_dir


def build_notebook_paths(repo_root: Path | str | None = None) -> NotebookPaths:
    """Return the standard path bundle used by simulation, analysis, and plots."""

    root = find_repo_root(repo_root)
    output_data_dir = root / "Output Data"
    penetration_dir = output_data_dir / "Penetration Sweep"
    return NotebookPaths(
        repo_root=root,
        source_dir=root / "Codes" / "sourcecode",
        input_data_dir=root / "Input data",
        output_data_dir=output_data_dir,
        plots_dir=output_data_dir / "plots",
        penetration_dir=penetration_dir,
        penetration_csv_dir=penetration_dir / "csv",
        penetration_plot_dir=penetration_dir / "plots",
    )


def ensure_directories(*paths: Path | str) -> tuple[Path, ...]:
    """Create artifact directories and return normalized `Path` objects."""

    normalized = tuple(Path(path) for path in paths)
    for path in normalized:
        path.mkdir(parents=True, exist_ok=True)
    return normalized


def print_path_summary(title: str, paths: Mapping[str, Path | str]) -> None:
    """Print compact notebook-friendly path status lines."""

    print(title)
    for label, path in paths.items():
        path_obj = Path(path)
        print(f"- {label}: {path_obj} (exists={path_obj.exists()})")
