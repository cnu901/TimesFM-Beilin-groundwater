"""Run the rolling-window forecasters and persist per-lead predictions.

Usage
-----
    python scripts/03_run_models.py                       # every enabled model
    python scripts/03_run_models.py --models FullFramework

Saving the raw per-lead predictions is what allows the residual diagnostics
and the case-study panels to be redrawn from the analysis itself rather than
from previously rendered images.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bootstrap import banner  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.config import load_config  # noqa: E402
from src.forecasting import (  # noqa: E402
    build_windows,
    forecast_chronos,
    forecast_full_framework,
    forecast_lstm,
    forecast_ridge_only,
    forecast_timesfm_only,
    forecast_timesfm_xreg,
    history_to_frame,
    load_timesfm,
    make_naive_forecaster,
)
from src.pipeline import load_daily  # noqa: E402


NEEDS_TIMESFM = {"FullFramework", "TimesFMOnly", "TimesFM_XReg"}


def build_registry():
    return {
        "RidgeOnly": forecast_ridge_only,
        "TimesFMOnly": forecast_timesfm_only,
        "FullFramework": forecast_full_framework,
        "TimesFM_XReg": forecast_timesfm_xreg,
        "Chronos": forecast_chronos,
        "LSTM": forecast_lstm,
        "SeasonalNaive": make_naive_forecaster(seasonal=True),
        "LastValueNaive": make_naive_forecaster(seasonal=False),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default="", help="Comma-separated subset of models")
    parser.add_argument("--force", action="store_true", help="Re-run models with existing output")
    args = parser.parse_args()

    banner("Step 3 - rolling-window forecasts")
    config = load_config()
    config.paths.ensure()
    daily = load_daily(config)
    windows = build_windows(daily, config)
    print(f"daily records: {len(daily)}   rolling windows: {len(windows)}")

    registry = build_registry()
    requested = [name.strip() for name in args.models.split(",") if name.strip()]
    models = requested or list(config["models"]["enabled"])
    unknown = [name for name in models if name not in registry]
    if unknown:
        raise SystemExit(f"Unknown model(s): {unknown}. Available: {sorted(registry)}")

    timesfm_model = None
    for name in models:
        output = config.paths.forecasts / f"{name}.csv.gz"
        if output.is_file() and not args.force:
            print(f"[{name}] cached -> {output.name}")
            continue
        if name in NEEDS_TIMESFM and timesfm_model is None:
            print("loading TimesFM 2.5 checkpoint ...")
            timesfm_model = load_timesfm(config)
        started = time.time()
        print(f"[{name}] running ...")
        try:
            history = registry[name](daily, windows, config, model=timesfm_model)
        except Exception as exc:
            print(f"[{name}] FAILED: {type(exc).__name__}: {exc}")
            continue
        frame = history_to_frame(history)
        frame.to_csv(output, index=False, compression="gzip")
        print(
            f"[{name}] {len(frame)} predictions -> {output.name} "
            f"({time.time() - started:.1f} s)"
        )

    summary = []
    for name in models:
        output = config.paths.forecasts / f"{name}.csv.gz"
        if not output.is_file():
            summary.append({"model": name, "predictions": 0, "status": "missing"})
            continue
        frame = pd.read_csv(output, parse_dates=["date"])
        summary.append(
            {
                "model": name,
                "predictions": len(frame),
                "first_date": frame["date"].min().date().isoformat(),
                "last_date": frame["date"].max().date().isoformat(),
                "status": "ok",
            }
        )
    summary_frame = pd.DataFrame(summary)
    summary_frame.to_csv(
        config.paths.forecasts / "forecast_inventory.csv", index=False, encoding="utf-8-sig"
    )
    print(summary_frame.to_string(index=False))


if __name__ == "__main__":
    main()
