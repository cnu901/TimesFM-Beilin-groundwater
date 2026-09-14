"""Shared data-access helpers for the analysis pipeline.

The daily table produced here is the single input to every downstream step:
forecasting, anomaly detection, statistics, figures and tables.  Centralising
it is what keeps the manuscript numbers consistent across outputs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config
from .features import FEATURE_NAMES, build_daily_covariates
from .preprocessing import long_gap_hour_mask


DAILY_COLUMNS = [
    "water_level",
    "water_level_original",
    "air_pressure_hpa",
    "rainfall_mm",
    "imputation_flag",
]


def canonical_hourly_path(config: Config) -> Path:
    return config.paths.processed / "beilin_hourly_corrected.csv"


def canonical_daily_path(config: Config) -> Path:
    return config.paths.processed / "beilin_daily.csv"


def canonical_hourly_reference_path(config: Config) -> Path:
    return config.paths.processed / "beilin_hourly_corrected_reference.csv"


def read_hourly_table(path: Path) -> pd.DataFrame:
    """Read a processed hourly CSV with a normalised time index."""
    frame = pd.read_csv(path)
    frame.columns = ["time"] + list(frame.columns[1:])
    frame["time"] = pd.to_datetime(frame["time"])
    frame = frame.set_index("time").sort_index()
    frame.index.name = "time"
    return frame


def build_daily_from_hourly(hourly: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the corrected hourly record onto the daily analysis grid."""
    daily = pd.DataFrame(
        {
            "water_level": hourly["water_level_final"].resample("D").mean(),
            "water_level_original": hourly["water_level_original"].resample("D").mean(),
            "air_pressure_hpa": hourly["air_pressure_hpa"].resample("D").mean(),
            "rainfall_mm": hourly["rainfall_mm"].resample("D").sum(min_count=1),
        }
    )
    # The imputation flag marks every day that contains at least one hour in
    # a missing run longer than the linear-interpolation limit.  Short gaps are
    # linearly interpolated between observed values and are negligible for a
    # daily-mean analysis; only the model-filled long gaps are excluded.
    if "water_level_original" in hourly.columns:
        daily["imputation_flag"] = (
            long_gap_hour_mask(hourly["water_level_original"].isna())
            .resample("D")
            .max()
            .astype(bool)
        )
    else:
        daily["imputation_flag"] = False
    daily.index.name = "date"
    return daily


def load_daily(config: Config, *, rebuild_covariates: bool = True) -> pd.DataFrame:
    """Load the analysis-ready daily table, optionally rebuilding covariates.

    Covariates are rebuilt from the corrected hourly pressure and rainfall
    record so that the daily table is reproducible from a single source.
    """
    daily_path = canonical_daily_path(config)
    if not daily_path.is_file():
        raise FileNotFoundError(
            f"Daily dataset not found: {daily_path}. Run scripts/01_prepare_data.py first."
        )
    daily = pd.read_csv(daily_path, parse_dates=["date"]).set_index("date").sort_index()
    daily.index.name = "date"
    daily["imputation_flag"] = daily["imputation_flag"].astype(bool)

    if rebuild_covariates:
        hourly = read_hourly_table(canonical_hourly_path(config))
        covariates = build_daily_covariates(
            hourly["air_pressure_hpa"], hourly["rainfall_mm"]
        )
        covariates = covariates.reindex(daily.index)
        for name in FEATURE_NAMES:
            daily[name] = covariates[name].to_numpy()

    config["feature_names"] = list(FEATURE_NAMES)
    return daily


def load_reference_daily(config: Config) -> pd.DataFrame | None:
    """Load the archived hourly reference series for validation, if present."""
    reference = canonical_hourly_reference_path(config)
    if not reference.is_file():
        return None
    hourly = read_hourly_table(reference)
    return build_daily_from_hourly(hourly)


def residual_segment_starts(segments: pd.DataFrame) -> list[pd.Timestamp]:
    """Return the start dates of the residual (screening) segments."""
    if segments.empty:
        return []
    return [pd.Timestamp(value) for value in segments.loc[segments["cause"] == "residual", "start"]]


def ensure_present(path: Path, description: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{description} not found: {path}")
    return path
