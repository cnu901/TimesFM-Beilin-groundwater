"""Permutation tests, multiplicity control and Molchan error diagrams.

Two distinct questions are evaluated and they must not be conflated:

*Permutation test*
    Holding the detected anomaly segments fixed, earthquake dates are shuffled
    within the period that has a complete 365-day look-back.  This tests
    whether the timing of residual segments is unusual relative to a random
    regional catalogue.

*Molchan error diagram*
    Each residual segment opens a 365-day prospective alarm interval.  Sweeping
    the segment score threshold traces alarm-time fraction against earthquake
    miss rate.  The curve is descriptive; a score near zero means the alarm
    behaviour is close to random.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def count_hits(earthquake_dates, segment_starts, window_days: int = 365) -> int:
    """Number of earthquakes with a segment start in ``(q - window, q)``."""
    hits = 0
    for quake in earthquake_dates:
        if any(0 < (quake - start).days <= window_days for start in segment_starts):
            hits += 1
    return hits


def count_in_window(
    earthquake_dates,
    segment_starts,
    lower_days: int,
    upper_days: int,
) -> int:
    """Number of earthquakes with a segment start in ``(lower, upper]`` days."""
    hits = 0
    for quake in earthquake_dates:
        if any(lower_days < (quake - start).days <= upper_days for start in segment_starts):
            hits += 1
    return hits


def evaluable_events(
    earthquake_dates,
    score_start: pd.Timestamp,
    lookback_days: int = 365,
) -> pd.DatetimeIndex:
    """Events old enough to have a complete look-back window in the score record."""
    earliest = pd.Timestamp(score_start) + pd.Timedelta(days=lookback_days)
    return pd.DatetimeIndex([q for q in earthquake_dates if q >= earliest])


def _shuffle_uniform(
    earthquake_dates,
    score_start: pd.Timestamp,
    score_end: pd.Timestamp,
    n_permutations: int,
    seed: int,
    lookback_days: int = 365,
) -> np.ndarray:
    """Return ``n_permutations`` shuffled date sets on the valid calendar."""
    day_zero = pd.Timestamp(score_start) + pd.Timedelta(days=lookback_days)
    n_days = max(1, (pd.Timestamp(score_end) - day_zero).days)
    rng = np.random.default_rng(seed)
    n_events = len(earthquake_dates)
    # ``Generator.choice`` cannot draw without replacement for a
    # multi-dimensional size, so sample by ranking uniform variates instead.
    scores = rng.random((n_permutations, n_days))
    return np.argsort(scores, axis=1)[:, :n_events]


def window_permutation(
    earthquake_dates,
    segment_starts,
    score_start: pd.Timestamp,
    score_end: pd.Timestamp,
    windows: dict[str, tuple[int, int]],
    *,
    n_permutations: int = 1000,
    seed: int = 42,
    lookback_days: int = 365,
) -> pd.DataFrame:
    """Permutation test for every pre-earthquake window in ``windows``."""
    day_zero = pd.Timestamp(score_start) + pd.Timedelta(days=lookback_days)
    draws = _shuffle_uniform(
        earthquake_dates, score_start, score_end, n_permutations, seed, lookback_days
    )
    rows = []
    for name, (lower, upper) in windows.items():
        observed = count_in_window(earthquake_dates, segment_starts, lower, upper)
        null = np.array(
            [
                count_in_window(
                    pd.DatetimeIndex([day_zero + pd.Timedelta(days=int(offset)) for offset in draw]),
                    segment_starts,
                    lower,
                    upper,
                )
                for draw in draws
            ],
            dtype=int,
        )
        p_value = (1 + int((null >= observed).sum())) / (1 + n_permutations)
        rows.append(
            {
                "window": name,
                "lower_days": lower,
                "upper_days": upper,
                "observed_hits": int(observed),
                "n_events": int(len(earthquake_dates)),
                "null_mean": float(null.mean()),
                "null_std": float(null.std()),
                "p_value": float(p_value),
            }
        )
    frame = pd.DataFrame(rows)
    return frame


def apply_fdr(frame: pd.DataFrame, family: list[str], alpha: float = 0.05) -> pd.DataFrame:
    """Benjamini-Hochberg adjustment restricted to the declared test family."""
    frame = frame.copy()
    frame["q_value"] = np.nan
    frame["significant_fdr"] = False
    subset = frame.index[frame["window"].isin(family)]
    if len(subset) == 0:
        return frame
    p_values = frame.loc[subset, "p_value"].to_numpy(dtype="float64")
    order = np.argsort(p_values)
    n_tests = len(p_values)
    ranked = p_values[order] * n_tests / (np.arange(n_tests) + 1)
    monotone = np.minimum.accumulate(ranked[::-1])[::-1]
    monotone = np.clip(monotone, 0.0, 1.0)
    adjusted = np.empty(n_tests)
    adjusted[order] = monotone
    frame.loc[subset, "q_value"] = adjusted
    frame.loc[subset, "significant_fdr"] = adjusted < alpha
    return frame


def sequence_preserving_permutation(
    earthquake_dates,
    segment_starts,
    sequence_ids: list[str],
    score_start: pd.Timestamp,
    score_end: pd.Timestamp,
    *,
    n_permutations: int = 1000,
    seed: int = 42,
    lookback_days: int = 365,
    window_days: int = 365,
) -> dict:
    """Shift every member of a sequence by the same random offset."""
    day_zero = pd.Timestamp(score_start) + pd.Timedelta(days=lookback_days)
    n_days = max(1, (pd.Timestamp(score_end) - day_zero).days)
    unique_sequences = list(dict.fromkeys(sequence_ids))
    dates_by_sequence: dict[str, list[pd.Timestamp]] = {sequence: [] for sequence in unique_sequences}
    for i, sequence in enumerate(sequence_ids):
        dates_by_sequence[sequence].append(pd.Timestamp(earthquake_dates[i]))
    for sequence in unique_sequences:
        dates_by_sequence[sequence].sort()
    end = pd.Timestamp(score_end)
    rng = np.random.default_rng(seed)
    observed = count_hits(earthquake_dates, segment_starts, window_days)
    null = np.zeros(n_permutations, dtype=int)
    for k in range(n_permutations):
        shifted: list[pd.Timestamp] = []
        for sequence in unique_sequences:
            members = dates_by_sequence[sequence]
            origin = members[0]
            draw = int(rng.integers(0, n_days))
            for member in members:
                candidate = day_zero + pd.Timedelta(days=draw) + (member - origin)
                if day_zero <= candidate <= end:
                    shifted.append(candidate)
        null[k] = count_hits(pd.DatetimeIndex(shifted), segment_starts, window_days)
    p_value = (1 + int((null >= observed).sum())) / (1 + n_permutations)
    return {
        "observed_hits": int(observed),
        "n_events": int(len(earthquake_dates)),
        "null_mean": float(null.mean()),
        "null_std": float(null.std()),
        "p_value": float(p_value),
        "n_sequences": len(unique_sequences),
    }


def leave_one_event_out(
    earthquake_dates,
    segment_starts,
    lower_days: int,
    upper_days: int,
    score_start: pd.Timestamp,
    score_end: pd.Timestamp,
    *,
    n_permutations: int = 1000,
    seed: int = 42,
    lookback_days: int = 365,
) -> pd.DataFrame:
    """Repeat the permutation test with each event removed in turn."""
    dates = pd.DatetimeIndex(earthquake_dates)
    rows = []
    for index in range(len(dates)):
        kept = dates.delete(index)
        frame = window_permutation(
            kept,
            segment_starts,
            score_start,
            score_end,
            {"window": (lower_days, upper_days)},
            n_permutations=n_permutations,
            # Reuse the same null draws for every exclusion so that the
            # comparison is paired and only the observed hit count changes.
            seed=seed,
            lookback_days=lookback_days,
        ).iloc[0]
        rows.append(
            {
                "excluded_event": dates[index].date().isoformat(),
                "excluded_magnitude": float("nan"),
                "n_events": int(len(kept)),
                "observed_hits": int(frame["observed_hits"]),
                "null_mean": float(frame["null_mean"]),
                "p_value": float(frame["p_value"]),
            }
        )
    return pd.DataFrame(rows)


def molchan_curve(
    residual_segments: pd.DataFrame,
    earthquake_dates,
    score_start: pd.Timestamp,
    score_end: pd.Timestamp,
    *,
    alarm_window_days: int = 365,
) -> tuple[pd.DataFrame, dict]:
    """Segment-based Molchan curve with a consistent 365-day alarm rule."""
    evaluation_days = pd.date_range(
        pd.Timestamp(score_start) + pd.Timedelta(days=alarm_window_days),
        pd.Timestamp(score_end),
        freq="D",
    )
    events = pd.DatetimeIndex(earthquake_dates)

    def operating_point(subset: pd.DataFrame) -> tuple[float, float]:
        alarm = np.zeros(len(evaluation_days), dtype=bool)
        for start in subset["start"]:
            alarm |= (evaluation_days > start) & (
                evaluation_days <= start + pd.Timedelta(days=alarm_window_days)
            )
        miss_rate_value = float(
            sum(
                not any(
                    0 < (quake - start).days <= alarm_window_days for start in subset["start"]
                )
                for quake in events
            )
            / len(events)
        )
        return float(alarm.mean()), miss_rate_value

    thresholds = np.r_[np.inf, np.unique(residual_segments["max_score"])[::-1], -np.inf]
    tau_values = []
    nu_values = []
    for threshold in thresholds:
        subset = residual_segments.loc[residual_segments["max_score"] >= threshold]
        tau, nu = operating_point(subset)
        tau_values.append(tau)
        nu_values.append(nu)

    tau_values = np.asarray(tau_values)
    nu_values = np.asarray(nu_values)
    order = np.argsort(tau_values)
    tau_values, nu_values = tau_values[order], nu_values[order]
    unique_tau = np.unique(tau_values)
    unique_nu = np.array([nu_values[tau_values == value].min() for value in unique_tau])
    if unique_tau[0] > 0:
        unique_tau = np.r_[0.0, unique_tau]
        unique_nu = np.r_[1.0, unique_nu]
    if unique_tau[-1] < 1:
        unique_tau = np.r_[unique_tau, 1.0]
        unique_nu = np.r_[unique_nu, 0.0]

    auc = float(np.trapezoid(unique_nu, unique_tau))
    skill = 1.0 - 2.0 * auc
    operating_tau, operating_nu = operating_point(residual_segments)
    curve = pd.DataFrame({"alarm_time_fraction": unique_tau, "miss_rate": unique_nu})
    metrics = {
        "alarm_window_days": alarm_window_days,
        "n_residual_segments": int(len(residual_segments)),
        "n_evaluable_events": int(len(events)),
        "operating_tau": operating_tau,
        "operating_nu": operating_nu,
        "molchan_auc": auc,
        "molchan_skill": skill,
        "random_line_auc": 0.5,
    }
    return curve, metrics


def hit_table(
    earthquake_dates,
    earthquake_magnitudes,
    segment_starts,
    *,
    window_days: int = 365,
) -> pd.DataFrame:
    """Per-event hit table used by Table 2 and Figure 7."""
    rows = []
    for quake, magnitude in zip(earthquake_dates, earthquake_magnitudes):
        in_window = [
            start for start in segment_starts if 0 < (quake - start).days <= window_days
        ]
        rows.append(
            {
                "earthquake_date": quake.date().isoformat(),
                "magnitude": float(magnitude),
                "hit": bool(in_window),
                "lead_days": int(min((quake - start).days for start in in_window)) if in_window else None,
                "first_segment_start": min(in_window).date().isoformat() if in_window else "",
                "n_segments_in_window": len(in_window),
            }
        )
    return pd.DataFrame(rows)
