"""Descriptive statistics quoted in the manuscript.

Produces ``results/statistics/descriptive_statistics.csv`` with the record
length, value ranges, secular trend, annual amplitude, covariate-contribution
diagnostics and the hourly barometric response.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bootstrap import banner  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.linear_model import Ridge  # noqa: E402

from src.config import load_config  # noqa: E402
from src.features import FEATURE_NAMES, standardise_frame  # noqa: E402
from src.pipeline import canonical_hourly_path, load_daily, read_hourly_table  # noqa: E402


def main() -> None:
    banner("Step 11 - descriptive statistics")
    config = load_config()
    config.paths.ensure()
    daily = load_daily(config)
    hourly = read_hourly_table(canonical_hourly_path(config))

    water = daily["water_level"]
    days = np.arange(len(daily), dtype="float64")
    linear_trend = float(np.polyfit(days, water.to_numpy(), 1)[0] * len(daily))
    # The manuscript quotes the change between the first and last complete
    # annual means, which is the change a reader sees in the yearly series.
    yearly_means = water.groupby(daily.index.year).mean()
    secular_trend = float(yearly_means.iloc[-1] - yearly_means.iloc[0])
    # The annual amplitude is the range of the long-term monthly means.
    monthly_means = water.groupby(daily.index.month).mean()
    annual_amplitude = float(monthly_means.max() - monthly_means.min())

    target = water.to_numpy(dtype="float64")
    valid = np.isfinite(target) & ~daily["imputation_flag"].to_numpy()
    scaled, _ = standardise_frame(daily[FEATURE_NAMES], mask=pd.Series(valid, index=daily.index))
    features = scaled.to_numpy(dtype="float64")
    model = Ridge(alpha=1.0).fit(features[valid], target[valid])
    r_squared = float(model.score(features[valid], target[valid]))
    coefficient_table = dict(zip(FEATURE_NAMES, model.coef_))

    months = daily.index.month
    summer = valid & np.isin(months, [6, 7, 8])
    winter = valid & np.isin(months, [12, 1, 2])
    pressure_index = FEATURE_NAMES.index("pressure")
    summer_pressure = float(
        Ridge(alpha=1.0).fit(features[summer], target[summer]).coef_[pressure_index]
    )
    winter_pressure = float(
        Ridge(alpha=1.0).fit(features[winter], target[winter]).coef_[pressure_index]
    )

    paired = pd.concat([hourly["water_level_final"], hourly["air_pressure_hpa"]], axis=1).dropna()
    water_valid, pressure_valid = paired.iloc[:, 0], paired.iloc[:, 1]
    water_highpass = water_valid - water_valid.rolling(25, center=True, min_periods=1).mean()
    pressure_highpass = pressure_valid - pressure_valid.rolling(25, center=True, min_periods=1).mean()
    highpass_correlation = float(np.corrcoef(water_highpass, pressure_highpass)[0, 1])
    highpass_slope = float(np.polyfit(pressure_highpass, water_highpass, 1)[0] * 1000.0)

    window = 24 * 7
    rolling = []
    pressure_array = pressure_highpass.to_numpy()
    water_array = water_highpass.to_numpy()
    for position in range(window, len(water_array), 24):
        span = slice(position - window, position)
        rolling.append(np.polyfit(pressure_array[span], water_array[span], 1)[0] * 1000.0)
    rolling = np.asarray(rolling)

    rows = [
        ("daily_records", len(daily), "days"),
        ("hourly_records", len(hourly), "hours"),
        ("water_level_min", float(water.min()), "m"),
        ("water_level_max", float(water.max()), "m"),
        ("water_level_mean", float(water.mean()), "m"),
        ("pressure_min", float(daily["air_pressure_hpa"].min()), "hPa"),
        ("pressure_max", float(daily["air_pressure_hpa"].max()), "hPa"),
        ("rainfall_max_hourly", float(hourly["rainfall_mm"].max()), "mm/h"),
        ("rainfall_positive_fraction", float((hourly["rainfall_mm"] > 0).mean()), "-"),
        ("secular_trend_first_to_last_year", secular_trend, "m"),
        ("linear_trend_over_record", linear_trend, "m"),
        ("annual_amplitude_monthly_mean_range", annual_amplitude, "m"),
        ("full_period_ridge_r_squared", r_squared, "-"),
        ("coefficient_rainfall_30d", float(coefficient_table["rainfall_30d"]), "-"),
        ("coefficient_pressure", float(coefficient_table["pressure"]), "-"),
        ("summer_pressure_coefficient", summer_pressure, "-"),
        ("winter_pressure_coefficient", winter_pressure, "-"),
        ("hourly_highpass_correlation", highpass_correlation, "-"),
        ("hourly_highpass_slope", highpass_slope, "mm/hPa"),
        ("hourly_highpass_barometric_efficiency", abs(highpass_slope) / 10.0 / 1.02, "-"),
        ("rolling_7day_barometric_mean", float(rolling.mean()), "mm/hPa"),
        ("rolling_7day_barometric_median", float(np.median(rolling)), "mm/hPa"),
    ]
    frame = pd.DataFrame(rows, columns=["metric", "value", "unit"])
    frame.to_csv(
        config.paths.statistics / "descriptive_statistics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
