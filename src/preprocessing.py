"""Quality control and correction of the Beilin monitoring records.

The raw files are CEA ``EQT`` exports.  Three documented defects are handled:

1. A confirmed 0.354 m instrument step on 2015-08-26.  All earlier values are
   shifted by the step so that the series is continuous.
2. A polarity inversion between 2021-04-13 and 2021-06-21.  The inverted
   values are mirrored about the mean of the two anchor hours on 13 April
   2021, which makes the corrected record continuous across the boundary.
3. Instrument instability from 2018-06-30 to 2018-07-20.  This is *not* a level
   step.  The interval is flagged as missing; imputed values are retained only
   so that model context windows stay continuous, and affected target dates
   are excluded from scoring and earthquake matching.

Gap filling uses linear interpolation for gaps of at most
``gap_filling.linear_limit_hours`` and a bidirectional forecast for longer
gaps.  The forecast filler is injected by the caller so that this module stays
testable without loading a deep-learning model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from .config import Config
from .eqt import read_eqt_series


LongGapFiller = Callable[[pd.Series], pd.Series]


@dataclass
class PreprocessingReport:
    """Diagnostics written to ``results/preprocessing``."""

    hourly_start: str
    hourly_end: str
    n_hourly: int
    n_water_missing_raw: int
    n_water_missing_2018: int
    n_water_filled_short: int
    n_water_filled_long: int
    n_pressure_rejected: int
    n_rainfall_missing: int
    longest_rainfall_gap_h: int
    step_date: str
    step_offset_m: float
    polarity_start: str
    polarity_end: str
    polarity_centre_m: float
    n_daily: int

    def as_rows(self) -> list[tuple[str, object]]:
        return [
            ("hourly_start", self.hourly_start),
            ("hourly_end", self.hourly_end),
            ("n_hourly_records", self.n_hourly),
            ("water_level_missing_raw_hours", self.n_water_missing_raw),
            ("water_level_missing_2018_hours", self.n_water_missing_2018),
            ("water_level_filled_short_gap_hours", self.n_water_filled_short),
            ("water_level_filled_long_gap_hours", self.n_water_filled_long),
            ("pressure_rejected_hours", self.n_pressure_rejected),
            ("rainfall_missing_hours", self.n_rainfall_missing),
            ("longest_rainfall_gap_hours", self.longest_rainfall_gap_h),
            ("step_correction_date", self.step_date),
            ("step_correction_offset_m", self.step_offset_m),
            ("polarity_start", self.polarity_start),
            ("polarity_end", self.polarity_end),
            ("polarity_centre_m", self.polarity_centre_m),
            ("n_daily_records", self.n_daily),
        ]


def _interpolate_short_gaps(series: pd.Series, limit_hours: int) -> tuple[pd.Series, pd.Series]:
    """Linearly interpolate gaps up to ``limit_hours`` and report filled hours."""
    filled = series.interpolate(method="linear", limit=limit_hours, limit_area="inside")
    was_missing = series.isna()
    return filled, was_missing & filled.notna()


def _fill_long_gaps(
    series: pd.Series,
    short_filled: pd.Series,
    long_gap_filler: LongGapFiller,
) -> tuple[pd.Series, pd.Series]:
    """Fill the remaining gaps with a bidirectional forecast."""
    remaining = short_filled.isna()
    if not remaining.any():
        return short_filled, remaining
    completed = long_gap_filler(short_filled)
    if len(completed) != len(short_filled):
        raise ValueError("Long-gap filler returned a series of the wrong length")
    completed = completed.copy()
    completed.index = short_filled.index
    previously_observed = ~short_filled.isna()
    completed[previously_observed] = short_filled[previously_observed]
    return completed, remaining & completed.notna()


def linear_long_gap_filler(series: pd.Series) -> pd.Series:
    """Fallback filler: two-sided linear interpolation across every gap."""
    return series.interpolate(method="linear", limit_direction="both")


def apply_step_correction(series: pd.Series, date: str, offset_m: float) -> pd.Series:
    """Shift all values recorded before ``date`` by ``offset_m``."""
    corrected = series.copy()
    mask = corrected.index < pd.Timestamp(date)
    corrected.loc[mask] = corrected.loc[mask] + offset_m
    return corrected


def replace_transition_hours(
    series: pd.Series,
    window: list[str] | None,
    reference: str | None,
) -> pd.Series:
    """Replace the recorder settling hours with the first stable value."""
    if not window or not reference:
        return series
    corrected = series.copy()
    value = corrected.loc[pd.Timestamp(reference)]
    mask = (corrected.index >= pd.Timestamp(window[0])) & (corrected.index <= pd.Timestamp(window[1]))
    corrected.loc[mask] = value
    return corrected


def apply_polarity_correction(
    series: pd.Series,
    start: str,
    end: str,
    anchor_before: str,
    anchor_after: str,
    centre_m: float | None = None,
) -> tuple[pd.Series, float]:
    """Mirror the inverted interval about the two-anchor mean.

    ``anchor_before`` is the last hour before the inversion and ``anchor_after``
    is the first inverted hour.  The mirror centre is the mean of the two
    values, so the corrected record is continuous across the boundary.  Passing
    ``centre_m`` overrides the derived value; the published analysis used
    ``31.26405 m``, which is the two-anchor mean obtained after the gap fill.
    """
    if centre_m is not None:
        centre = float(centre_m)
    else:
        before = series.loc[pd.Timestamp(anchor_before)]
        after = series.loc[pd.Timestamp(anchor_after)]
        if not np.isfinite(before) or not np.isfinite(after):
            raise ValueError(
                "Polarity anchors must be finite: "
                f"{anchor_before}={before}, {anchor_after}={after}"
            )
        centre = float((before + after) / 2.0)
    corrected = series.copy()
    mask = (corrected.index >= pd.Timestamp(start)) & (corrected.index <= pd.Timestamp(end))
    corrected.loc[mask] = 2.0 * centre - corrected.loc[mask]
    return corrected, centre


def longest_false_run(mask: pd.Series) -> int:
    """Length of the longest consecutive run of ``True`` values."""
    if mask.empty:
        return 0
    groups = (mask != mask.shift()).cumsum()
    runs = mask.groupby(groups).sum()
    return int(runs.max()) if len(runs) else 0


def long_gap_hour_mask(missing: pd.Series, *, limit_hours: int = 24) -> pd.Series:
    """Mark hours inside missing runs longer than ``limit_hours``.

    ``limit_hours`` must match ``preprocessing.gap_filling.linear_limit_hours``:
    gaps at or below this length are linearly interpolated and considered
    negligible for daily-mean anomaly screening, while longer gaps are filled
    by the bidirectional model and are therefore flagged for exclusion.
    """
    groups = (missing != missing.shift()).cumsum()
    run_length = missing.groupby(groups).transform("size")
    return missing & (run_length > limit_hours)


def build_hourly_table(
    config: Config,
    *,
    long_gap_filler: LongGapFiller | None = None,
) -> tuple[pd.DataFrame, PreprocessingReport]:
    """Build the corrected hourly table and a preprocessing report."""
    paths = config.paths
    data_cfg = config["data"]
    prep = config["preprocessing"]
    start, end = data_cfg["study_period"]
    sentinel = float(prep["missing_sentinel"])

    water = read_eqt_series(paths.raw / data_cfg["water_level_hourly"], missing_sentinel=sentinel)
    pressure = read_eqt_series(paths.raw / data_cfg["pressure_hourly"], missing_sentinel=sentinel)
    rainfall = read_eqt_series(paths.raw / data_cfg["rainfall_hourly"], missing_sentinel=sentinel)

    grid = pd.date_range(pd.Timestamp(start), pd.Timestamp(end), freq="h")
    water = water.reindex(grid)
    pressure = pressure.reindex(grid)
    rainfall = rainfall.reindex(grid)

    water_raw = water.copy()
    n_missing_raw = int(water.isna().sum())

    gap_cfg = prep["flagged_gap"]
    gap_mask = (grid >= pd.Timestamp(gap_cfg["start"])) & (grid <= pd.Timestamp(gap_cfg["end"]))
    n_missing_2018 = int(water.isna().where(gap_mask, False).sum())

    step_cfg = prep["step_correction"]
    if step_cfg["enabled"]:
        water = apply_step_correction(water, step_cfg["date"], float(step_cfg["offset_m"]))
        water = replace_transition_hours(
            water,
            step_cfg.get("transition_window"),
            step_cfg.get("transition_reference"),
        )

    fill_cfg = prep["gap_filling"]
    water, short_mask = _interpolate_short_gaps(water, int(fill_cfg["linear_limit_hours"]))
    filler = long_gap_filler or linear_long_gap_filler
    water, long_mask = _fill_long_gaps(water, water, filler)

    pol_cfg = prep["polarity_correction"]
    polarity_centre = float("nan")
    if pol_cfg["enabled"]:
        anchor_date = pd.Timestamp(pol_cfg["anchor_date"])
        water, polarity_centre = apply_polarity_correction(
            water,
            pol_cfg["start"],
            pol_cfg["end"],
            (anchor_date + pd.Timedelta(hours=9)).isoformat(),
            (anchor_date + pd.Timedelta(hours=10)).isoformat(),
            pol_cfg.get("centre_m"),
        )

    pressure_cfg = prep["pressure_quality"]
    rejected = pressure.isna()
    if pressure_cfg["zero_is_missing"]:
        rejected |= pressure == 0
    rejected |= pressure < float(pressure_cfg["valid_min_hpa"])
    rejected |= pressure > float(pressure_cfg["valid_max_hpa"])
    pressure = pressure.mask(rejected)
    pressure = pressure.interpolate(method="linear", limit=72, limit_area="inside")
    pressure = pressure.bfill().ffill()
    n_pressure_rejected = int(rejected.sum())

    rainfall_cfg = prep["rainfall_quality"]
    missing_rain = rainfall.isna()
    n_rainfall_missing = int(missing_rain.sum())
    longest_rain = longest_false_run(missing_rain)
    if rainfall_cfg["negative_to_zero"]:
        rainfall = rainfall.where(rainfall >= 0, 0.0)
    rainfall = rainfall.fillna(float(rainfall_cfg["missing_fill_value"]))

    table = pd.DataFrame(
        {
            "water_level_original": water_raw,
            "water_level_final": water,
            "air_pressure_hpa": pressure,
            "rainfall_mm": rainfall,
            "imputation_flag": long_mask,
        },
        index=grid,
    )
    table.index.name = "time"

    report = PreprocessingReport(
        hourly_start=grid.min().isoformat(sep=" "),
        hourly_end=grid.max().isoformat(sep=" "),
        n_hourly=len(grid),
        n_water_missing_raw=n_missing_raw,
        n_water_missing_2018=n_missing_2018,
        n_water_filled_short=int(short_mask.sum()),
        n_water_filled_long=int(long_mask.sum()),
        n_pressure_rejected=n_pressure_rejected,
        n_rainfall_missing=n_rainfall_missing,
        longest_rainfall_gap_h=longest_rain,
        step_date=str(step_cfg["date"]),
        step_offset_m=float(step_cfg["offset_m"]),
        polarity_start=str(pol_cfg["start"]),
        polarity_end=str(pol_cfg["end"]),
        polarity_centre_m=polarity_centre,
        n_daily=len(table.resample("D").mean()),
    )
    return table, report


def to_daily(hourly: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the corrected hourly table to the analysis-ready daily grid."""
    daily = pd.DataFrame(
        {
            "water_level": hourly["water_level_final"].resample("D").mean(),
            "water_level_original": hourly["water_level_original"].resample("D").mean(),
            "air_pressure_hpa": hourly["air_pressure_hpa"].resample("D").mean(),
            "rainfall_mm": hourly["rainfall_mm"].resample("D").sum(min_count=1),
            "imputation_flag": long_gap_hour_mask(
                hourly["water_level_original"].isna()
            ).resample("D").max(),
        }
    )
    daily.index.name = "date"
    return daily
