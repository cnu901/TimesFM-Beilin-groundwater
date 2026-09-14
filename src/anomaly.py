"""Anomaly scoring, calibration-prefix thresholds and cause classification.

Prediction windows that overlap a calendar date are fused into a single score
that weights each forecast by the inverse square of its lead time.  Deviations
are standardised by the predictive interval, so models with different variance
scales remain comparable.

Thresholds are estimated only from a calibration prefix, which removes the
self-reference of the original per-month quantile rule: a target date can
never contribute to its own alarm threshold.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Segment:
    """A confirmed anomaly segment."""

    start: pd.Timestamp
    end: pd.Timestamp
    days: int
    max_score: float
    cause: str = "unclassified"
    rain_total_mm: float = float("nan")
    pressure_change_hpa: float = float("nan")


def gaussian_band_function(history: dict, actual: pd.Series, z: float = 1.2816):
    """Return a lead-dependent Gaussian-equivalent 80% band width.

    Used only for models that do not expose native predictive quantiles.
    """
    residuals: dict[int, list[float]] = defaultdict(list)
    for date_str, predictions in history.items():
        stamp = pd.Timestamp(date_str)
        if stamp not in actual.index:
            continue
        observed = float(actual.loc[stamp])
        if not np.isfinite(observed):
            continue
        for prediction in predictions:
            residuals[prediction["horizon"]].append(observed - prediction["point"])
    sigma = {h: float(np.std(v)) for h, v in residuals.items() if len(v) > 3}
    return lambda horizon: 2.0 * z * sigma.get(horizon, np.nan)


def lead_width_factors(
    history: dict,
    actual: pd.Series,
    *,
    imputation_flag: pd.Series | None = None,
    target_coverage: float = 0.80,
) -> dict[int, float]:
    """Return per-lead multiplicative factors that calibrate q10-q90 coverage.

    The native quantile interval is rescaled symmetrically around the point
    forecast until its empirical coverage over the retained prediction set
    equals ``target_coverage``.  The factor is estimated from the full record,
    which matches the retrospective, Gaussian-equivalent band used for the
    non-quantile baselines.
    """
    per_lead: dict[int, list[tuple[float, float, float, float]]] = defaultdict(list)
    for date_str, predictions in history.items():
        stamp = pd.Timestamp(date_str)
        if stamp not in actual.index:
            continue
        if imputation_flag is not None and stamp in imputation_flag.index:
            if bool(imputation_flag.loc[stamp]):
                continue
        observed = float(actual.loc[stamp])
        if not np.isfinite(observed):
            continue
        for prediction in predictions:
            per_lead[prediction["horizon"]].append(
                (
                    observed,
                    float(prediction["point"]),
                    float(prediction["q10"]),
                    float(prediction["q90"]),
                )
            )

    factors: dict[int, float] = {}
    for horizon, rows in per_lead.items():
        if len(rows) < 5:
            continue
        observed = np.array([row[0] for row in rows])
        point = np.array([row[1] for row in rows])
        q10 = np.array([row[2] for row in rows])
        q90 = np.array([row[3] for row in rows])
        lower, upper = 0.05, 50.0
        factor = 1.0
        for _ in range(60):
            mid = (lower + upper) / 2.0
            calibrated_low = point + mid * (q10 - point)
            calibrated_high = point + mid * (q90 - point)
            coverage = float(
                ((observed >= calibrated_low) & (observed <= calibrated_high)).mean()
            )
            if coverage > target_coverage:
                upper = mid
            else:
                lower = mid
            factor = mid
        factors[horizon] = float(factor)
    return factors


def fuse_scores(
    history: dict,
    actual: pd.Series,
    *,
    has_native_quantiles: bool,
    gaussian_z: float = 1.2816,
    imputation_flag: pd.Series | None = None,
    interval_target_coverage: float = 0.80,
    width_factors_out: dict | None = None,
) -> pd.DataFrame:
    """Fuse rolling-window predictions into one anomaly score per date."""
    band = None if has_native_quantiles else gaussian_band_function(history, actual, gaussian_z)
    width_factors: dict[int, float] = {}
    if has_native_quantiles:
        width_factors = lead_width_factors(
            history,
            actual,
            imputation_flag=imputation_flag,
            target_coverage=interval_target_coverage,
        )
    if width_factors_out is not None:
        width_factors_out.update(width_factors)
    records = []
    for date_str, predictions in history.items():
        if len(predictions) < 2:
            continue
        stamp = pd.Timestamp(date_str)
        if stamp not in actual.index:
            continue
        if imputation_flag is not None and stamp in imputation_flag.index:
            if bool(imputation_flag.loc[stamp]):
                continue
        observed = float(actual.loc[stamp])
        if not np.isfinite(observed):
            continue
        total_weight = 0.0
        total_score = 0.0
        for prediction in predictions:
            weight = 1.0 / (prediction["horizon"] ** 2)
            if band is None:
                factor = width_factors.get(prediction["horizon"], 1.0)
                width = factor * (prediction["q90"] - prediction["q10"])
            else:
                width = band(prediction["horizon"])
            if width is None or not np.isfinite(width) or width < 1e-8:
                width = 1e-8
            deviation = abs(observed - prediction["point"]) / width
            total_score += weight * deviation
            total_weight += weight
        records.append(
            {
                "date": stamp,
                "score": total_score / total_weight if total_weight > 0 else 0.0,
                "actual": observed,
                "n_predictions": len(predictions),
                "month": stamp.month,
            }
        )
    if not records:
        return pd.DataFrame(columns=["date", "score", "actual", "n_predictions", "month"])
    return pd.DataFrame(records).sort_values("date").reset_index(drop=True)


def expanding_thresholds(
    scores: pd.DataFrame,
    *,
    warning_percentile: float,
    critical_percentile: float,
    min_observations: int,
) -> pd.DataFrame:
    """Estimate month-specific thresholds using only past scores.

    For every date, the warning and critical thresholds are computed from
    strictly earlier scores in the same calendar month.  If fewer than
    ``min_observations`` same-month scores are available, the pooled set of
    earlier scores is used.  A date therefore never contributes to its own
    threshold, and no future score leaks into the decision.
    """
    scores = scores.sort_values("date").reset_index(drop=True)
    dates = scores["date"].to_numpy()
    months = scores["month"].to_numpy()
    values = scores["score"].to_numpy()
    n = len(scores)
    warning = np.full(n, np.nan)
    critical = np.full(n, np.nan)
    for position in range(n):
        if position == 0:
            continue
        prior = values[:position]
        same_month = months[:position] == months[position]
        if int(same_month.sum()) >= min_observations:
            base = prior[same_month]
        elif len(prior) >= min_observations:
            base = prior
        else:
            # Not enough history yet for a stable percentile.  Leave the
            # thresholds as NaN so the date is labelled NORMAL rather than
            # flagging it on a near-empty sample.
            continue
        warning[position] = float(np.percentile(base, warning_percentile))
        critical[position] = float(np.percentile(base, critical_percentile))
    return pd.DataFrame(
        {
            "date": pd.DatetimeIndex(dates),
            "warning_threshold": warning,
            "critical_threshold": critical,
        }
    )


def label_dates(
    scores: pd.DataFrame,
    thresholds: pd.DataFrame,
    imputation_flag: pd.Series | None = None,
) -> pd.DataFrame:
    """Assign NORMAL / WARNING / CRITICAL labels to each scored date."""
    merged = scores.merge(thresholds, on="date", how="left")
    labels = []
    for row in merged.itertuples(index=False):
        warning = row.warning_threshold
        critical = row.critical_threshold
        if pd.isna(critical) or pd.isna(warning):
            level = "NORMAL"
        elif row.score >= critical:
            level = "CRITICAL"
        elif row.score >= warning:
            level = "WARNING"
        else:
            level = "NORMAL"
        flagged = False
        if imputation_flag is not None and row.date in imputation_flag.index:
            flagged = bool(imputation_flag.loc[row.date])
        labels.append(
            {
                "date": row.date,
                "score": row.score,
                "level": level,
                "actual": row.actual,
                "n_predictions": row.n_predictions,
                "month": row.month,
                "imputation_flag": flagged,
            }
        )
    frame = pd.DataFrame(labels).sort_values("date").reset_index(drop=True)
    frame["is_anomaly"] = frame["level"].isin(["WARNING", "CRITICAL"])
    return frame


def extract_segments(labelled: pd.DataFrame, min_duration_days: int) -> list[Segment]:
    """Group consecutive anomalous, non-imputed days into confirmed segments."""
    segments: list[Segment] = []
    run: list[pd.Series] = []

    def flush() -> None:
        if len(run) >= min_duration_days:
            segments.append(
                Segment(
                    start=run[0]["date"],
                    end=run[-1]["date"],
                    days=len(run),
                    max_score=float(max(item["score"] for item in run)),
                )
            )

    for _, row in labelled.iterrows():
        if row["imputation_flag"]:
            continue
        contiguous = bool(run) and (row["date"] - run[-1]["date"]).days == 1
        if row["is_anomaly"] and (not run or contiguous):
            run.append(row)
        else:
            flush()
            run = [row] if row["is_anomaly"] else []
    flush()
    return segments


def classify_segments(
    segments: list[Segment],
    rainfall_daily: pd.Series,
    pressure_daily: pd.Series,
    *,
    rainfall_threshold_mm: float,
    pressure_threshold_hpa: float,
    rainfall_margin_mm: float = 10.0,
    pressure_margin_hpa: float = 4.0,
    window_days: int = 7,
) -> list[Segment]:
    """Attach rule-based meteorological cause labels to each segment.

    ``rainfall``, ``pressure`` and ``mixed`` mean the corresponding strict
    rule was exceeded.  ``borderline`` means the segment is below both strict
    rules but within a fixed margin of one of them, and ``residual`` means it
    is clearly below both.  Neither ``borderline`` nor ``residual`` is a
    tectonic diagnosis.
    """
    labelled: list[Segment] = []
    for segment in segments:
        rain_start = segment.start - pd.Timedelta(days=window_days)
        rain_total = float(rainfall_daily.loc[rain_start: segment.end].sum())
        pre_start = segment.start - pd.Timedelta(days=window_days)
        segment_pressure = pressure_daily.loc[segment.start: segment.end].mean()
        pre_pressure = pressure_daily.loc[pre_start: segment.start - pd.Timedelta(days=1)].mean()
        if np.isfinite(segment_pressure) and np.isfinite(pre_pressure):
            pressure_change = float(abs(segment_pressure - pre_pressure))
        else:
            pressure_change = 0.0

        rain_low = rainfall_threshold_mm - rainfall_margin_mm
        pressure_low = pressure_threshold_hpa - pressure_margin_hpa
        if rain_total > rainfall_threshold_mm and pressure_change <= pressure_threshold_hpa:
            cause = "rainfall"
        elif pressure_change > pressure_threshold_hpa and rain_total <= rainfall_threshold_mm:
            cause = "pressure"
        elif rain_total > rainfall_threshold_mm and pressure_change > pressure_threshold_hpa:
            cause = "mixed"
        elif rain_total > rain_low or pressure_change > pressure_low:
            # Near-threshold but not clearly meteorological: keep separate so
            # the strict cut-offs do not dump borderline cases into "residual".
            cause = "borderline"
        else:
            cause = "residual"

        labelled.append(
            Segment(
                start=segment.start,
                end=segment.end,
                days=segment.days,
                max_score=segment.max_score,
                cause=cause,
                rain_total_mm=round(rain_total, 2),
                pressure_change_hpa=round(pressure_change, 2),
            )
        )
    return labelled


def segments_to_frame(segments: list[Segment]) -> pd.DataFrame:
    """Convert segments into a tidy table for CSV export."""
    if not segments:
        return pd.DataFrame(
            columns=[
                "start",
                "end",
                "days",
                "max_score",
                "cause",
                "rain_total_mm",
                "pressure_change_hpa",
            ]
        )
    return pd.DataFrame(
        [
            {
                "start": segment.start,
                "end": segment.end,
                "days": segment.days,
                "max_score": segment.max_score,
                "cause": segment.cause,
                "rain_total_mm": segment.rain_total_mm,
                "pressure_change_hpa": segment.pressure_change_hpa,
            }
            for segment in segments
        ]
    )


def residual_segments(segments: list[Segment]) -> list[Segment]:
    """Return only the residual (meteorologically unexplained) segments."""
    return [segment for segment in segments if segment.cause == "residual"]
