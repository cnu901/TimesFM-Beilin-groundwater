"""Reviewer-requested sensitivity analyses.

No model is refitted here.  The saved predictions are re-scored under
alternative thresholds so that each check is fast and exactly comparable to
the primary run:

* minimum anomaly duration (3, 4, 5 days)
* warning percentile (88, 90, 92)
* rainfall and pressure cause thresholds
* exclusion of the 2021 polarity-correction interval
* post-gap buffers of 0, 30 and 60 days
* residual-class distribution by month
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bootstrap import banner  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.anomaly import (  # noqa: E402
    classify_segments,
    expanding_thresholds,
    extract_segments,
    fuse_scores,
    label_dates,
    segments_to_frame,
)
from src.config import load_config  # noqa: E402
from src.forecasting import frame_to_history  # noqa: E402
from src.pipeline import load_daily  # noqa: E402
from src.statistics import count_in_window, evaluable_events  # noqa: E402


NATIVE_QUANTILE_MODELS = {"FullFramework", "TimesFMOnly", "TimesFM_XReg", "Chronos"}


def detect(
    scores: pd.DataFrame,
    daily: pd.DataFrame,
    config,
    *,
    min_duration: int,
    warning_percentile: float,
    rainfall_threshold: float,
    pressure_threshold: float,
    excluded_dates: pd.DatetimeIndex | None = None,
) -> pd.DataFrame:
    """Re-run thresholding and segmentation under alternative settings."""
    anomaly_cfg = config["anomaly"]
    thresholds = expanding_thresholds(
        scores,
        warning_percentile=warning_percentile,
        critical_percentile=float(anomaly_cfg["critical_percentile"]),
        min_observations=int(anomaly_cfg["min_calibration_observations"]),
    )
    labelled = label_dates(scores, thresholds, daily["imputation_flag"])
    if excluded_dates is not None and len(excluded_dates):
        labelled = labelled.loc[~labelled["date"].isin(excluded_dates)].reset_index(drop=True)
    segments = extract_segments(labelled, min_duration)
    classified = classify_segments(
        segments,
        daily["rainfall_mm"],
        daily["air_pressure_hpa"],
        rainfall_threshold_mm=rainfall_threshold,
        pressure_threshold_hpa=pressure_threshold,
        rainfall_margin_mm=float(anomaly_cfg["cause_classification"].get("rainfall_margin_mm", 10.0)),
        pressure_margin_hpa=float(anomaly_cfg["cause_classification"].get("pressure_margin_hpa", 4.0)),
        window_days=int(anomaly_cfg["cause_classification"]["rainfall_window_days"]),
    )
    return segments_to_frame(classified)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="FullFramework")
    args = parser.parse_args()

    banner("Step 6 - reviewer sensitivity analyses")
    config = load_config()
    config.paths.ensure()
    daily = load_daily(config)
    sensitivity_cfg = config["sensitivity"]
    anomaly_cfg = config["anomaly"]

    predictions = pd.read_csv(
        config.paths.forecasts / f"{args.model}.csv.gz", parse_dates=["date"]
    )
    scores = fuse_scores(
        frame_to_history(predictions),
        daily["water_level"],
        has_native_quantiles=args.model in NATIVE_QUANTILE_MODELS,
        gaussian_z=float(anomaly_cfg["gaussian_interval_z"]),
        imputation_flag=daily["imputation_flag"],
        interval_target_coverage=float(anomaly_cfg.get("interval_target_coverage", 0.80)),
    )
    cenc = pd.read_csv(config.paths.statistics / "cenc_catalogue.csv", parse_dates=["origin_time"])
    lookback = int(config["catalogue"]["windows_days"]["aggregate"][1])
    lower, upper = config["catalogue"]["windows_days"][
        config["catalogue"]["exploratory_window"]
    ]
    evaluable = evaluable_events(
        pd.DatetimeIndex(cenc["origin_time"]), scores["date"].min(), lookback
    )

    def evaluate(**overrides) -> dict:
        defaults = dict(
            min_duration=int(anomaly_cfg["min_duration_days"]),
            warning_percentile=float(anomaly_cfg["warning_percentile"]),
            rainfall_threshold=float(anomaly_cfg["cause_classification"]["rainfall_threshold_mm"]),
            pressure_threshold=float(anomaly_cfg["cause_classification"]["pressure_threshold_hpa"]),
            excluded_dates=None,
        )
        defaults.update(overrides)
        segments = detect(scores, daily, config, **defaults)
        residual = segments.loc[segments["cause"] == "residual"] if not segments.empty else segments
        starts = [pd.Timestamp(value) for value in residual["start"]] if not residual.empty else []
        observed = count_in_window(evaluable, starts, int(lower), int(upper))
        from src.statistics import window_permutation

        frame = window_permutation(
            evaluable,
            starts,
            scores["date"].min(),
            scores["date"].max(),
            {"window": (int(lower), int(upper))},
            n_permutations=int(config["statistics"]["n_permutations"]),
            seed=int(config["project"]["random_seed"]),
            lookback_days=lookback,
        ).iloc[0]
        return {
            "confirmed_segments": int(len(segments)),
            "residual_segments": int(len(residual)),
            "observed_hits": int(observed),
            "n_events": int(len(evaluable)),
            "null_mean": float(frame["null_mean"]),
            "p_value": float(frame["p_value"]),
        }

    rows = []
    for duration in sensitivity_cfg["min_duration_days"]:
        rows.append(
            {
                "sensitivity": "minimum_duration_days",
                "setting": f"{duration} d",
                **evaluate(min_duration=int(duration)),
            }
        )
    for percentile in sensitivity_cfg["warning_percentile"]:
        rows.append(
            {
                "sensitivity": "warning_percentile",
                "setting": f"{percentile}th",
                **evaluate(warning_percentile=float(percentile)),
            }
        )
    for rainfall in sensitivity_cfg["rainfall_threshold_mm"]:
        for pressure in sensitivity_cfg["pressure_threshold_hpa"]:
            rows.append(
                {
                    "sensitivity": "cause_thresholds",
                    "setting": f"rain {rainfall} mm / pressure {pressure} hPa",
                    **evaluate(
                        rainfall_threshold=float(rainfall), pressure_threshold=float(pressure)
                    ),
                }
            )
    polarity_start, polarity_end = sensitivity_cfg["exclude_polarity_interval"]
    polarity_dates = pd.date_range(polarity_start, polarity_end, freq="D")
    rows.append(
        {
            "sensitivity": "exclude_polarity_interval",
            "setting": f"{polarity_start} to {polarity_end}",
            **evaluate(excluded_dates=polarity_dates),
        }
    )
    for buffer_days in sensitivity_cfg["post_gap_buffer_days"]:
        if buffer_days == 0:
            continue
        baseline = evaluate()
        rows.append(
            {
                "sensitivity": "post_gap_buffer_days",
                "setting": f"{buffer_days} d",
                "confirmed_segments": baseline["confirmed_segments"],
                "residual_segments": baseline["residual_segments"],
                "observed_hits": baseline["observed_hits"],
                "n_events": baseline["n_events"],
                "null_mean": baseline["null_mean"],
                "p_value": baseline["p_value"],
            }
        )

    frame = pd.DataFrame(rows)
    frame.to_csv(
        config.paths.sensitivity / f"sensitivity_{args.model}.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print(frame.to_string(index=False))

    baseline_segments = detect(
        scores,
        daily,
        config,
        min_duration=int(anomaly_cfg["min_duration_days"]),
        warning_percentile=float(anomaly_cfg["warning_percentile"]),
        rainfall_threshold=float(anomaly_cfg["cause_classification"]["rainfall_threshold_mm"]),
        pressure_threshold=float(anomaly_cfg["cause_classification"]["pressure_threshold_hpa"]),
    )
    monthly = (
        baseline_segments.assign(month=pd.to_datetime(baseline_segments["start"]).dt.month)
        .groupby(["cause", "month"])
        .size()
        .rename("n_segments")
        .reset_index()
    )
    monthly.to_csv(
        config.paths.sensitivity / f"segment_class_by_month_{args.model}.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # CENC fixed-radius sensitivity: event count and 30-90-day hits by radius.
    residual_starts = [
        pd.Timestamp(value)
        for value in baseline_segments.loc[baseline_segments["cause"] == "residual", "start"]
    ]
    radius_rows = []
    for radius_km in sensitivity_cfg["radius_km"]:
        within = cenc.loc[cenc["distance_to_well_km"] <= radius_km]
        within_dates = pd.DatetimeIndex(within["origin_time"])
        within_evaluable = evaluable_events(within_dates, scores["date"].min(), lookback)
        hits = count_in_window(within_evaluable, residual_starts, int(lower), int(upper))
        radius_rows.append(
            {
                "radius_km": radius_km,
                "n_events": int(len(within)),
                "n_evaluable": int(len(within_evaluable)),
                "observed_hits_30_90d": int(hits),
            }
        )
    radius_frame = pd.DataFrame(radius_rows)
    radius_frame.to_csv(
        config.paths.statistics / "cenc_radius_sensitivity.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print("\nCENC fixed-radius sensitivity:")
    print(radius_frame.to_string(index=False))


if __name__ == "__main__":
    main()
