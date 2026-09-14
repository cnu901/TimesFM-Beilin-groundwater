"""Rebuild the corrected hourly record and the analysis-ready daily table.

Two things are produced and they are deliberately separated:

``data/processed/beilin_daily.csv``
    The daily analysis grid used by every downstream step.

``results/preprocessing``
    A validation report that compares a full rebuild from the raw CEA files
    against the canonical corrected series, plus the preprocessing counts used
    in the manuscript.

The canonical hourly series retains the bidirectional gap fill used for the
published analysis.  The rebuild is provided so that every correction can be
inspected and reproduced from the raw files.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bootstrap import banner  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.config import load_config  # noqa: E402
from src.pipeline import (  # noqa: E402
    build_daily_from_hourly,
    canonical_daily_path,
    canonical_hourly_path,
    read_hourly_table,
)
from src.preprocessing import build_hourly_table, linear_long_gap_filler  # noqa: E402


def main() -> None:
    banner("Step 1 - prepare data")
    config = load_config()
    config.paths.ensure()

    hourly_path = canonical_hourly_path(config)
    if not hourly_path.is_file():
        raise FileNotFoundError(
            f"Canonical corrected series missing: {hourly_path}. "
            "Run with --rebuild-only to generate it from the raw CEA files."
        )

    hourly = read_hourly_table(hourly_path)
    daily = build_daily_from_hourly(hourly)
    daily_path = canonical_daily_path(config)
    daily.to_csv(daily_path, encoding="utf-8-sig")
    print(f"daily table      : {daily_path}  ({len(daily)} days)")

    rebuilt, report = build_hourly_table(config, long_gap_filler=linear_long_gap_filler)
    rebuilt_daily = build_daily_from_hourly(rebuilt)

    missing_hours = hourly["water_level_original"].isna().resample("D").sum()
    (config.paths.preprocessing / "preprocessing_counts.csv").write_text(
        "metric,value\n"
        + "\n".join(f"{key},{value}" for key, value in report.as_rows())
        + "\n",
        encoding="utf-8-sig",
    )

    comparison = pd.DataFrame(
        {
            "date": daily.index,
            "canonical_water_level_m": daily["water_level"].to_numpy(),
            "rebuilt_water_level_m": rebuilt_daily["water_level"].reindex(daily.index).to_numpy(),
            "canonical_pressure_hpa": daily["air_pressure_hpa"].to_numpy(),
            "rebuilt_pressure_hpa": rebuilt_daily["air_pressure_hpa"].reindex(daily.index).to_numpy(),
            "canonical_rainfall_mm": daily["rainfall_mm"].to_numpy(),
            "rebuilt_rainfall_mm": rebuilt_daily["rainfall_mm"].reindex(daily.index).to_numpy(),
            "imputation_flag": daily["imputation_flag"].to_numpy(),
            "missing_hours": missing_hours.reindex(daily.index).to_numpy(),
        }
    )
    comparison["water_level_difference_m"] = (
        comparison["rebuilt_water_level_m"] - comparison["canonical_water_level_m"]
    )
    comparison.to_csv(
        config.paths.preprocessing / "preprocessing_validation.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pressure_difference = (
        comparison["rebuilt_pressure_hpa"] - comparison["canonical_pressure_hpa"]
    ).abs()
    rainfall_difference = (
        comparison["rebuilt_rainfall_mm"] - comparison["canonical_rainfall_mm"]
    ).abs()
    water_difference = comparison["water_level_difference_m"].abs()
    gap_filled_days = comparison["missing_hours"] > 0
    summary = pd.DataFrame(
        [
            {
                "quantity": "water_level_all_days_vs_canonical",
                "n_days": int(water_difference.notna().sum()),
                "max_abs_difference": float(water_difference.max()),
                "n_days_gt_1mm": int((water_difference > 0.001).sum()),
            },
            {
                "quantity": "water_level_days_without_missing_hours",
                "n_days": int((~gap_filled_days).sum()),
                "max_abs_difference": float(water_difference[~gap_filled_days].max()),
                "n_days_gt_1mm": int((water_difference[~gap_filled_days] > 0.001).sum()),
            },
            {
                "quantity": "water_level_days_with_gap_filling",
                "n_days": int(gap_filled_days.sum()),
                "max_abs_difference": float(water_difference[gap_filled_days].max()),
                "n_days_gt_1mm": int((water_difference[gap_filled_days] > 0.001).sum()),
            },
            {
                "quantity": "air_pressure",
                "n_days": int(pressure_difference.notna().sum()),
                "max_abs_difference": float(pressure_difference.max()),
                "n_days_gt_1mm": int((pressure_difference > 0.001).sum()),
            },
            {
                "quantity": "rainfall",
                "n_days": int(rainfall_difference.notna().sum()),
                "max_abs_difference": float(rainfall_difference.max()),
                "n_days_gt_1mm": int((rainfall_difference > 0.001).sum()),
            },
        ]
    )
    summary.to_csv(
        config.paths.preprocessing / "preprocessing_validation_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print(summary.to_string(index=False))
    print(f"days with any missing hour            : {int(gap_filled_days.sum())}")
    print(f"days containing any filled water-level hour : {int(daily['imputation_flag'].sum())}")
    print(f"2018 flagged interval missing hours   : {report.n_water_missing_2018}")
    print(f"missing rainfall hours                : {report.n_rainfall_missing}")


if __name__ == "__main__":
    main()
