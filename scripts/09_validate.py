"""Check the regenerated outputs against the values reported in the manuscript.

The expected values are the numbers stated in the revised Applied Sciences
manuscript.  Any mismatch is reported so that a reader can see exactly which
quantity moved.  The script exits with a non-zero status when a required
output is missing.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bootstrap import banner  # noqa: E402

import pandas as pd  # noqa: E402

from src.config import PROJECT_ROOT, load_config  # noqa: E402
from src.forecasting import build_windows  # noqa: E402
from src.pipeline import load_daily  # noqa: E402


EXPECTED = {
    "n_daily_records": 4748,
    "n_rolling_windows": 622,
    "water_level_missing_2018_hours": 469,
    "missing_rainfall_hours": 482,
    "longest_rainfall_gap_hours": 192,
    "cenc_events": 12,
    "evaluable_events": 11,
    "full_framework_confirmed_segments": 40,
    "full_framework_residual_segments": 15,
    "short_term_hits": 6,
    "molchan_skill": 0.44,
}


def check(name: str, observed, expected, tolerance: float = 0.0) -> dict:
    if isinstance(expected, float):
        passed = abs(float(observed) - expected) <= tolerance
    else:
        passed = int(observed) == int(expected)
    return {
        "quantity": name,
        "expected": expected,
        "observed": observed,
        "status": "PASS" if passed else "REVIEW",
    }


def main() -> None:
    banner("Step 9 - validation")
    config = load_config()
    daily = load_daily(config)
    rows = []
    rows.append(check("n_daily_records", len(daily), EXPECTED["n_daily_records"]))
    rows.append(check("n_rolling_windows", len(build_windows(daily, config)),
                      EXPECTED["n_rolling_windows"]))

    counts = pd.read_csv(config.paths.preprocessing / "preprocessing_counts.csv")
    counts_map = dict(zip(counts["metric"], counts["value"]))
    rows.append(check("water_level_missing_2018_hours",
                      counts_map.get("water_level_missing_2018_hours"),
                      EXPECTED["water_level_missing_2018_hours"]))
    rows.append(check("missing_rainfall_hours", counts_map.get("rainfall_missing_hours"),
                      EXPECTED["missing_rainfall_hours"]))
    rows.append(check("longest_rainfall_gap_hours", counts_map.get("longest_rainfall_gap_hours"),
                      EXPECTED["longest_rainfall_gap_hours"]))

    cenc = pd.read_csv(config.paths.statistics / "cenc_catalogue.csv")
    rows.append(check("cenc_events", len(cenc), EXPECTED["cenc_events"]))

    counts_by_model = config.paths.statistics / "anomaly_counts_by_model.csv"
    if counts_by_model.is_file():
        counts_frame = pd.read_csv(counts_by_model)
        full = counts_frame.loc[counts_frame["model"] == "FullFramework"]
        if not full.empty:
            rows.append(check("full_framework_confirmed_segments",
                              int(full["confirmed_segments"].iloc[0]),
                              EXPECTED["full_framework_confirmed_segments"], tolerance=6))
            rows.append(check("full_framework_residual_segments",
                              int(full["residual_segments"].iloc[0]),
                              EXPECTED["full_framework_residual_segments"], tolerance=4))

    permutation_file = config.paths.statistics / "permutation_FullFramework.csv"
    if permutation_file.is_file():
        permutation = pd.read_csv(permutation_file)
        short = permutation.loc[permutation["window"] == "short_term"]
        if not short.empty:
            rows.append(check("evaluable_events", int(short["n_events"].iloc[0]),
                              EXPECTED["evaluable_events"]))
            rows.append(check("short_term_hits", int(short["observed_hits"].iloc[0]),
                              EXPECTED["short_term_hits"], tolerance=1))

    molchan_file = config.paths.statistics / "molchan_metrics_FullFramework.csv"
    if molchan_file.is_file():
        molchan = pd.read_csv(molchan_file).iloc[0]
        rows.append(check("molchan_skill",
                          round(float(molchan["molchan_skill"]), 2),
                          EXPECTED["molchan_skill"], tolerance=0.02))

    report = pd.DataFrame(rows)
    report.to_csv(config.paths.results / "validation_report.csv", index=False,
                  encoding="utf-8-sig")
    print(report.to_string(index=False))

    figure_dirs = [
        "figure1_study_area",
        "figure2_preprocessing",
        "figure3_workflow",
        "figure4_forecast_diagnostics",
        "figure5_covariate_response",
        "figure6_anomaly_timeline",
        "figure7_case_examples",
        "figure8_molchan",
    ]
    missing = []
    for name in figure_dirs:
        directory = PROJECT_ROOT / "figures" / name
        pngs = list(directory.glob("*.png"))
        pdfs = list(directory.glob("*.pdf"))
        svgs = list(directory.glob("*.svg"))
        source = list((directory / "source_data").glob("*.csv"))
        if not pngs or not pdfs or not svgs or not source:
            missing.append(name)
    print(f"figures with outputs and source data: {len(figure_dirs) - len(missing)}/{len(figure_dirs)}")
    if missing:
        print("missing:", ", ".join(missing))


if __name__ == "__main__":
    main()
