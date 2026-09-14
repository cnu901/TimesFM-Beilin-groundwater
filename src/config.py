"""Configuration loading and shared project paths.

The complete analysis is parameterised through ``config/config.yaml`` so that
``scripts/run_all.py`` can reproduce every published number without editing
source files.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "config.yaml"


class ConfigError(RuntimeError):
    """Raised when the configuration file is missing or malformed."""


@dataclass(frozen=True)
class ProjectPaths:
    """Resolved absolute paths for every project directory."""

    root: Path
    raw: Path
    external: Path
    processed: Path
    results: Path
    forecasts: Path
    anomaly_scores: Path
    anomaly_segments: Path
    statistics: Path
    sensitivity: Path
    preprocessing: Path
    figures: Path
    tables_main: Path
    tables_supplementary: Path
    logs: Path

    def ensure(self) -> "ProjectPaths":
        """Create every output directory if it does not yet exist."""
        for value in self.__dict__.values():
            if isinstance(value, Path):
                value.mkdir(parents=True, exist_ok=True)
        return self


class Config(dict):
    """Nested dictionary with dotted access and a resolved path bundle."""

    def __init__(self, data: dict[str, Any], paths: ProjectPaths):
        super().__init__(data)
        self.paths = paths

    def get_path(self, dotted: str, default: Any = None) -> Any:
        """Return a nested value using ``a.b.c`` notation."""
        node: Any = self
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node


def build_paths(raw: dict[str, Any]) -> ProjectPaths:
    data = raw.get("data", {})
    output = raw.get("output", {})
    results = PROJECT_ROOT / output.get("results_dir", "results")
    figures = PROJECT_ROOT / output.get("figures_dir", "figures")
    tables = PROJECT_ROOT / output.get("tables_dir", "tables")
    logs = PROJECT_ROOT / output.get("logs_dir", "logs")
    return ProjectPaths(
        root=PROJECT_ROOT,
        raw=PROJECT_ROOT / data.get("raw_dir", "data/raw"),
        external=PROJECT_ROOT / data.get("external_dir", "data/external"),
        processed=PROJECT_ROOT / data.get("processed_dir", "data/processed"),
        results=results,
        forecasts=results / "forecasts",
        anomaly_scores=results / "anomaly_scores",
        anomaly_segments=results / "anomaly_segments",
        statistics=results / "statistics",
        sensitivity=results / "sensitivity",
        preprocessing=results / "preprocessing",
        figures=figures,
        tables_main=tables / "main",
        tables_supplementary=tables / "supplementary",
        logs=logs,
    )


def load_config(path: Path | str | None = None) -> Config:
    """Load the YAML configuration and attach resolved project paths."""
    config_path = Path(path) if path is not None else DEFAULT_CONFIG
    if not config_path.is_file():
        raise ConfigError(f"Configuration file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ConfigError(f"Configuration file is not a mapping: {config_path}")
    return Config(raw, build_paths(raw))


def figure_dir(name: str, *, create: bool = True) -> Path:
    """Return the dedicated output directory for one manuscript figure."""
    path = PROJECT_ROOT / "figures" / name
    if create:
        path.mkdir(parents=True, exist_ok=True)
        (path / "source_data").mkdir(exist_ok=True)
    return path
