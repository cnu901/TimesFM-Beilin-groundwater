"""Assemble the manuscript tables and their source data.

Main tables
    Table 1  covariate definitions
    Table 2  regional catalogue with association outcome
    Table 3  forecast accuracy by model and horizon
    Table 4  anomaly detection and earthquake association by model
    Table 5  permutation tests of the earthquake association

Supplementary tables
    Table S1 catalogue completeness and spatial-radius sensitivity
    Table S2 threshold, duration, gap and polarity sensitivity

Each table is written twice: a presentational CSV/Markdown file and a source
CSV holding the underlying numbers.
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
from src.features import FEATURE_DESCRIPTIONS, FEATURE_NAMES, standardise_frame  # noqa: E402
from src.pipeline import load_daily, residual_segment_starts  # noqa: E402
from src.statistics import count_in_window, evaluable_events  # noqa: E402


def write(frame: pd.DataFrame, directory: Path, name: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    frame.to_csv(directory / f"{name}.csv", index=False, encoding="utf-8-sig")


def main() -> None:
    banner("Step 8 - manuscript tables")
    config = load_config()
    config.paths.ensure()
    daily = load_daily(config)
    main_dir = config.paths.tables_main
    supp_dir = config.paths.tables_supplementary
    source_dir = main_dir / "source_data"
    source_dir.mkdir(parents=True, exist_ok=True)

    # Table 1 - covariate definitions
    table1 = pd.DataFrame(
        {
            "id": [f"f{index + 1}" for index in range(len(FEATURE_NAMES))],
            "feature": FEATURE_NAMES,
            "physical_meaning": [FEATURE_DESCRIPTIONS[name] for name in FEATURE_NAMES],
        }
    )
    write(table1, main_dir, "Table1_covariates")
    write(table1, source_dir, "Table1_source")

    # Table 2 - regional catalogue with association outcome
    cenc = pd.read_csv(config.paths.statistics / "cenc_catalogue.csv", parse_dates=["origin_time"])
    mapping = pd.read_csv(config.paths.statistics / "cenc_sequence_map.csv")
    scores = pd.read_csv(
        config.paths.anomaly_scores / "FullFramework.csv", parse_dates=["date"]
    )
    segments = pd.read_csv(
        config.paths.anomaly_segments / "FullFramework.csv", parse_dates=["start", "end"]
    )
    starts = residual_segment_starts(segments)
    lookback = int(config["catalogue"]["windows_days"]["aggregate"][1])
    windows = config["catalogue"]["windows_days"]
    evaluable = set(evaluable_events(pd.DatetimeIndex(cenc["origin_time"]), scores["date"].min(), lookback))
    table2 = cenc.copy()
    table2["sequence"] = mapping["sequence_id"].to_numpy()
    table2["evaluable"] = table2["origin_time"].isin(evaluable)
    table2["hit_365d"] = [
        bool(any(0 < (quake - start).days <= lookback for start in starts))
        for quake in table2["origin_time"]
    ]
    table2["hit_imminent_0_30d"] = [
        bool(any(windows["imminent"][0] < (quake - start).days <= windows["imminent"][1]
                 for start in starts))
        for quake in table2["origin_time"]
    ]
    table2["hit_short_term_30_90d"] = [
        bool(any(windows["short_term"][0] < (quake - start).days <= windows["short_term"][1]
                 for start in starts))
        for quake in table2["origin_time"]
    ]
    write(table2, main_dir, "Table2_catalogue")
    write(table2, source_dir, "Table2_source")

    # Table 3 - forecast accuracy
    metrics = pd.read_csv(config.paths.statistics / "forecast_metrics_by_model.csv")
    pivot = metrics.pivot_table(index="model", columns="horizon_days",
                                values=["mae", "rmse", "picp80"]).round(4)
    pivot.columns = [f"{kind}_{horizon}d" for kind, horizon in pivot.columns]
    pivot = pivot.reset_index()
    ordered_columns = ["model"] + [
        f"mae_{h}d" for h in (1, 7, 14, 30)
    ] + [f"rmse_{h}d" for h in (1, 30)] + [f"picp80_{h}d" for h in (1, 30)]
    pivot = pivot[[column for column in ordered_columns if column in pivot.columns]]
    write(pivot, main_dir, "Table3_forecast_accuracy")
    write(pivot, source_dir, "Table3_source")

    # Table 4 - anomaly detection and association by model
    counts = pd.read_csv(config.paths.statistics / "anomaly_counts_by_model.csv")
    rows = []
    for record in counts.itertuples():
        model_segments = pd.read_csv(
            config.paths.anomaly_segments / f"{record.model}.csv", parse_dates=["start", "end"]
        )
        model_starts = residual_segment_starts(model_segments)
        hits_all = sum(
            bool(any(0 < (quake - start).days <= lookback for start in model_starts))
            for quake in cenc["origin_time"]
        )
        hits_evaluable = sum(
            bool(any(0 < (quake - start).days <= lookback for start in model_starts))
            for quake in cenc.loc[cenc["origin_time"].isin(evaluable), "origin_time"]
        )
        rows.append(
            {
                "model": record.model,
                "confirmed_segments": record.confirmed_segments,
                "residual_segments": record.residual_segments,
                "borderline_segments": int(getattr(record, "borderline_segments", 0)),
                "unmatched_residual_segments": int(
                    sum(
                        not any(
                            0 < (quake - start).days <= lookback
                            for quake in cenc["origin_time"]
                        )
                        for start in model_starts
                    )
                ),
                "earthquake_hits_12": hits_all,
                "earthquake_hits_11": hits_evaluable,
            }
        )
    table4 = pd.DataFrame(rows)
    write(table4, main_dir, "Table4_anomaly_detection")
    write(table4, source_dir, "Table4_source")

    # Table 5 - permutation tests
    permutation = pd.read_csv(config.paths.statistics / "permutation_FullFramework.csv")
    sequence_path = config.paths.statistics / "sequence_permutation_FullFramework.csv"
    if sequence_path.is_file():
        sequence = pd.read_csv(sequence_path)
        sequence = sequence.assign(q_value=np.nan, significant_fdr=False)
        permutation = pd.concat(
            [
                permutation,
                sequence,
            ],
            ignore_index=True,
        )
    write(permutation, main_dir, "Table5_permutation")
    write(permutation, source_dir, "Table5_source")

    # Supplementary S1 - regional CENC radius sensitivity.
    radius = pd.read_csv(config.paths.statistics / "cenc_radius_sensitivity.csv")
    write(radius, supp_dir, "TableS1_radius_sensitivity")
    write(radius, supp_dir / "source_data", "TableS1_source")
    dob = pd.read_csv(config.paths.statistics / "dobrovolsky_sensitivity_FullFramework.csv")
    write(dob, supp_dir / "source_data", "TableS1_dobrovolsky_events")
    radius = pd.read_csv(config.paths.statistics / "cenc_radius_sensitivity.csv")
    write(radius, supp_dir / "source_data", "TableS1_cenc_radius")

    # Supplementary S2 - threshold and gap sensitivity
    sensitivity = pd.read_csv(config.paths.sensitivity / "sensitivity_FullFramework.csv")
    write(sensitivity, supp_dir, "TableS2_sensitivity")
    write(sensitivity, supp_dir / "source_data", "TableS2_source")

    loo = pd.read_csv(config.paths.statistics / "leave_one_event_out_FullFramework.csv")
    write(loo, supp_dir / "source_data", "TableS2_leave_one_event_out")

    monthly = pd.read_csv(config.paths.sensitivity / "segment_class_by_month_FullFramework.csv")
    write(monthly, supp_dir / "source_data", "TableS2_class_by_month")

    # Supplementary S3 - complete four-season pressure-coefficient audit.
    # Figure 5b intentionally presents only the summer-winter contrast; this
    # table preserves the full seasonal calculation for transparency.
    water_array = daily["water_level"].to_numpy(dtype="float64")
    valid = np.isfinite(water_array) & ~daily["imputation_flag"].to_numpy()
    scaled, _ = standardise_frame(
        daily[config["feature_names"]], mask=pd.Series(valid, index=daily.index)
    )
    feature_array = scaled.to_numpy(dtype="float64")
    pressure_index = config["feature_names"].index("pressure")
    seasons = [
        ("Spring (Mar-May)", [3, 4, 5]),
        ("Summer (Jun-Aug)", [6, 7, 8]),
        ("Autumn (Sep-Nov)", [9, 10, 11]),
        ("Winter (Dec-Feb)", [12, 1, 2]),
    ]
    seasonal_rows = []
    for season, months in seasons:
        mask = valid & np.isin(daily.index.month, months)
        fit = Ridge(alpha=1.0).fit(feature_array[mask], water_array[mask])
        seasonal_rows.append(
            {
                "season": season,
                "months": ",".join(str(month) for month in months),
                "valid_daily_records": int(mask.sum()),
                "pressure_coefficient": float(fit.coef_[pressure_index]),
            }
        )
    table_s3 = pd.DataFrame(seasonal_rows)
    write(table_s3, supp_dir, "TableS3_seasonal_pressure")
    write(table_s3, supp_dir / "source_data", "TableS3_source")

    print("tables written:")
    for path in sorted(main_dir.glob("Table*.csv")):
        print("  ", path.name)
    for path in sorted(supp_dir.glob("Table*.csv")):
        print("  ", path.name)


if __name__ == "__main__":
    main()
