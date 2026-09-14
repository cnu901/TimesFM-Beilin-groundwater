"""Fuse forecasts into anomaly scores, thresholds and confirmed segments.

Outputs, per model:

``results/anomaly_scores/<model>.csv``
``results/anomaly_segments/<model>.csv``
``results/statistics/ablation_summary.csv``
    Forecasting accuracy (MAE / RMSE / PICP80) and anomaly counts for Table 3
    and Table 4.
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
from src.forecasting import build_windows, frame_to_history  # noqa: E402
from src.pipeline import load_daily  # noqa: E402


NATIVE_QUANTILE_MODELS = {"FullFramework", "TimesFMOnly", "TimesFM_XReg", "Chronos"}


def metrics_for_model(
    predictions: pd.DataFrame,
    daily: pd.DataFrame,
    *,
    native_quantiles: bool,
    horizons: tuple[int, ...] = (1, 7, 14, 30),
    lead_tolerance: int = 1,
) -> pd.DataFrame:
    """MAE, RMSE and PICP80 for each nominal lead time.

    A forecast counts towards a nominal horizon when it lies within
    ``lead_tolerance`` days of it.  With daily aggregates of hourly data a
    nominal 1-day forecast spans 24-48 hours, so the tolerance keeps the metric
    aligned with the lead definition used in the manuscript.

    Interval coverage is only reported for models that produce predictive
    quantiles natively; substituting a Gaussian band for the other models
    would make the coverage column describe the substitute rather than the
    model.
    """
    merged = predictions.merge(
        daily[["water_level", "imputation_flag"]], left_on="date", right_index=True, how="left"
    )
    merged = merged.loc[~merged["imputation_flag"].astype(bool)]
    rows = []
    for horizon in horizons:
        subset = merged.loc[
            (merged["horizon"] - horizon).abs() <= lead_tolerance
        ].dropna(subset=["water_level", "point"])
        if subset.empty:
            continue
        error = subset["point"] - subset["water_level"]
        record = {
            "horizon_days": horizon,
            "n": len(subset),
            "mae": float(error.abs().mean()),
            "rmse": float(np.sqrt((error ** 2).mean())),
        }
        if native_quantiles and subset["q10"].notna().any() and subset["q90"].notna().any():
            inside = (subset["water_level"] >= subset["q10"]) & (
                subset["water_level"] <= subset["q90"]
            )
            record["picp80"] = float(inside.mean() * 100.0)
        else:
            record["picp80"] = float("nan")
        rows.append(record)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default="", help="Comma-separated subset of models")
    args = parser.parse_args()

    banner("Step 4 - anomaly detection")
    config = load_config()
    config.paths.ensure()
    daily = load_daily(config)
    anomaly_cfg = config["anomaly"]
    windows = build_windows(daily, config)

    available = sorted(
        path.name[: -len(".csv.gz")] for path in config.paths.forecasts.glob("*.csv.gz")
    )
    requested = [name.strip() for name in args.models.split(",") if name.strip()]
    models = requested or list(config["models"]["enabled"])
    models = [name for name in models if name in available]
    print(f"models with forecasts: {models}")

    counts_rows = []
    metric_rows = []
    calibration_rows = []
    for name in models:
        predictions = pd.read_csv(config.paths.forecasts / f"{name}.csv.gz", parse_dates=["date"])
        history = frame_to_history(predictions)
        width_factors: dict = {}
        scores = fuse_scores(
            history,
            daily["water_level"],
            has_native_quantiles=name in NATIVE_QUANTILE_MODELS,
            gaussian_z=float(anomaly_cfg["gaussian_interval_z"]),
            imputation_flag=daily["imputation_flag"],
            interval_target_coverage=float(anomaly_cfg.get("interval_target_coverage", 0.80)),
            width_factors_out=width_factors,
        )
        for horizon, factor in width_factors.items():
            calibration_rows.append(
                {"model": name, "horizon_days": horizon, "width_scale_factor": factor}
            )
        if scores.empty:
            print(f"[{name}] no scores produced")
            continue
        thresholds = expanding_thresholds(
            scores,
            warning_percentile=float(anomaly_cfg["warning_percentile"]),
            critical_percentile=float(anomaly_cfg["critical_percentile"]),
            min_observations=int(anomaly_cfg["min_calibration_observations"]),
        )
        labelled = label_dates(scores, thresholds, daily["imputation_flag"])
        segments = extract_segments(labelled, int(anomaly_cfg["min_duration_days"]))
        cause_cfg = anomaly_cfg["cause_classification"]
        classified = classify_segments(
            segments,
            daily["rainfall_mm"],
            daily["air_pressure_hpa"],
            rainfall_threshold_mm=float(cause_cfg["rainfall_threshold_mm"]),
            pressure_threshold_hpa=float(cause_cfg["pressure_threshold_hpa"]),
            rainfall_margin_mm=float(cause_cfg.get("rainfall_margin_mm", 10.0)),
            pressure_margin_hpa=float(cause_cfg.get("pressure_margin_hpa", 4.0)),
            window_days=int(cause_cfg["rainfall_window_days"]),
        )
        labelled.to_csv(
            config.paths.anomaly_scores / f"{name}.csv", index=False, encoding="utf-8-sig"
        )
        segments_frame = segments_to_frame(classified)
        segments_frame.to_csv(
            config.paths.anomaly_segments / f"{name}.csv", index=False, encoding="utf-8-sig"
        )
        cause_counts = segments_frame["cause"].value_counts().to_dict() if not segments_frame.empty else {}
        counts_rows.append(
            {
                "model": name,
                "scored_days": len(labelled),
                "confirmed_segments": len(segments_frame),
                "residual_segments": int(cause_counts.get("residual", 0)),
                "rainfall_segments": int(cause_counts.get("rainfall", 0)),
                "pressure_segments": int(cause_counts.get("pressure", 0)),
                "mixed_segments": int(cause_counts.get("mixed", 0)),
                "borderline_segments": int(cause_counts.get("borderline", 0)),
            }
        )
        metrics = metrics_for_model(
            predictions, daily, native_quantiles=name in NATIVE_QUANTILE_MODELS
        )
        metrics.insert(0, "model", name)
        metric_rows.append(metrics)
        print(
            f"[{name}] scored {len(labelled)} days, {len(segments_frame)} segments "
            f"(residual {cause_counts.get('residual', 0)})"
        )

    if counts_rows:
        pd.DataFrame(counts_rows).to_csv(
            config.paths.statistics / "anomaly_counts_by_model.csv",
            index=False,
            encoding="utf-8-sig",
        )
    if metric_rows:
        pd.concat(metric_rows, ignore_index=True).to_csv(
            config.paths.statistics / "forecast_metrics_by_model.csv",
            index=False,
            encoding="utf-8-sig",
        )
    if calibration_rows:
        pd.DataFrame(calibration_rows).to_csv(
            config.paths.statistics / "interval_recalibration.csv",
            index=False,
            encoding="utf-8-sig",
        )


if __name__ == "__main__":
    main()
