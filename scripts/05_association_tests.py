"""Earthquake-association statistics for the residual anomaly segments.

Produces the numbers reported in Table 5 and Figure 8:

* event-level permutation test for every pre-earthquake window
* Benjamini-Hochberg false-discovery-rate adjustment over the declared family
* sequence-level permutation on sequence representatives
* leave-one-event-out sensitivity for the exploratory window
* corrected Molchan error diagram with a consistent 365-day alarm definition
* Dobrovolsky preparation-radius sensitivity
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bootstrap import banner  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.config import load_config  # noqa: E402
from src.declustering import sequence_representatives  # noqa: E402
from src.pipeline import load_daily, residual_segment_starts  # noqa: E402
from src.statistics import (  # noqa: E402
    apply_fdr,
    evaluable_events,
    hit_table,
    leave_one_event_out,
    molchan_curve,
    window_permutation,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="FullFramework")
    args = parser.parse_args()

    banner("Step 5 - earthquake association statistics")
    config = load_config()
    config.paths.ensure()
    load_daily(config)
    catalogue_cfg = config["catalogue"]
    statistics_cfg = config["statistics"]

    segments = pd.read_csv(
        config.paths.anomaly_segments / f"{args.model}.csv", parse_dates=["start", "end"]
    )
    scores = pd.read_csv(
        config.paths.anomaly_scores / f"{args.model}.csv", parse_dates=["date"]
    )
    cenc = pd.read_csv(config.paths.statistics / "cenc_catalogue.csv", parse_dates=["origin_time"])
    mapping = pd.read_csv(
        config.paths.statistics / "cenc_sequence_map.csv", parse_dates=["origin_time"]
    )

    starts = residual_segment_starts(segments)
    earthquake_dates = pd.DatetimeIndex(cenc["origin_time"])
    lookback = int(catalogue_cfg["windows_days"]["aggregate"][1])
    score_start = scores["date"].min()
    score_end = scores["date"].max()
    evaluable = evaluable_events(earthquake_dates, score_start, lookback)
    magnitude_lookup = {row.origin_time: float(row.magnitude) for row in cenc.itertuples()}
    print(f"residual segments: {len(starts)}   evaluable events: {len(evaluable)}")

    windows = {
        "imminent": tuple(catalogue_cfg["windows_days"]["imminent"]),
        "short_term": tuple(catalogue_cfg["windows_days"]["short_term"]),
        "medium_term": tuple(catalogue_cfg["windows_days"]["medium_term"]),
        "imminent_short_term": tuple(catalogue_cfg["windows_days"]["imminent_short_term"]),
    }
    permutation = window_permutation(
        evaluable,
        starts,
        score_start,
        score_end,
        windows,
        n_permutations=int(statistics_cfg["n_permutations"]),
        seed=int(config["project"]["random_seed"]),
        lookback_days=lookback,
    )
    permutation = apply_fdr(
        permutation,
        list(statistics_cfg["fdr_family"]),
        alpha=float(statistics_cfg["fdr_alpha"]),
    )
    aggregate = window_permutation(
        evaluable,
        starts,
        score_start,
        score_end,
        {"aggregate": (0, lookback)},
        n_permutations=int(statistics_cfg["n_permutations"]),
        seed=int(config["project"]["random_seed"]) + 1,
        lookback_days=lookback,
    )
    permutation = pd.concat([aggregate, permutation], ignore_index=True)
    permutation.to_csv(
        config.paths.statistics / f"permutation_{args.model}.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print(permutation.to_string(index=False))

    representatives = sequence_representatives(mapping)
    representative_dates = pd.DatetimeIndex(representatives["origin_time"])
    evaluable_representatives = evaluable_events(
        representative_dates, score_start, lookback
    )
    sequence_windows = {
        "sequence_aggregate": (0, lookback),
        "sequence_short_term": windows["short_term"],
    }
    sequence_permutation = window_permutation(
        evaluable_representatives,
        starts,
        score_start,
        score_end,
        sequence_windows,
        n_permutations=int(statistics_cfg["n_permutations"]),
        seed=int(config["project"]["random_seed"]),
        lookback_days=lookback,
    )
    sequence_permutation.to_csv(
        config.paths.statistics / f"sequence_permutation_{args.model}.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print(sequence_permutation.to_string(index=False))

    exploratory = catalogue_cfg["exploratory_window"]
    lower, upper = windows[exploratory]
    loo = leave_one_event_out(
        evaluable,
        starts,
        lower,
        upper,
        score_start,
        score_end,
        n_permutations=int(statistics_cfg["n_permutations"]),
        seed=int(config["project"]["random_seed"]),
        lookback_days=lookback,
    )
    loo["excluded_magnitude"] = [
        magnitude_lookup.get(pd.Timestamp(date), float("nan")) for date in loo["excluded_event"]
    ]
    loo.to_csv(
        config.paths.statistics / f"leave_one_event_out_{args.model}.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print(f"leave-one-event-out p range: {loo['p_value'].min():.4f} - {loo['p_value'].max():.4f}")

    residual = segments.loc[segments["cause"] == "residual"]
    curve, molchan_metrics = molchan_curve(
        residual,
        evaluable,
        score_start,
        score_end,
        alarm_window_days=int(config["molchan"]["alarm_window_days"]),
    )
    curve.to_csv(
        config.paths.statistics / f"molchan_curve_{args.model}.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame([molchan_metrics]).to_csv(
        config.paths.statistics / f"molchan_metrics_{args.model}.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print(
        f"Molchan AUC={molchan_metrics['molchan_auc']:.4f}  "
        f"skill={molchan_metrics['molchan_skill']:.4f}  "
        f"tau={molchan_metrics['operating_tau']:.3f}  nu={molchan_metrics['operating_nu']:.3f}"
    )

    hits = hit_table(
        evaluable,
        [magnitude_lookup[quake] for quake in evaluable],
        starts,
        window_days=lookback,
    )
    hits.to_csv(
        config.paths.statistics / f"event_hits_{args.model}.csv",
        index=False,
        encoding="utf-8-sig",
    )

    dobrovolsky = cenc.loc[cenc["within_dobrovolsky_radius"]].copy()
    rows = dobrovolsky.assign(
        hit_365d=[
            any(0 < (pd.Timestamp(quake) - start).days <= lookback for start in starts)
            for quake in dobrovolsky["origin_time"]
        ],
        hit_short_term=[
            any(lower < (pd.Timestamp(quake) - start).days <= upper for start in starts)
            for quake in dobrovolsky["origin_time"]
        ],
    )
    rows.to_csv(
        config.paths.statistics / f"dobrovolsky_sensitivity_{args.model}.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print(
        "Dobrovolsky radius retains "
        f"{int(cenc['within_dobrovolsky_radius'].sum())} of {len(cenc)} events"
    )


if __name__ == "__main__":
    main()
