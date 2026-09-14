"""Run the complete reproducible pipeline in order.

    python scripts/run_all.py                # full run
    python scripts/run_all.py --fast         # retained for backward compatibility

Every step writes to ``logs/`` and the script stops at the first failure.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bootstrap import PROJECT_ROOT, banner  # noqa: E402


CORE_MODELS = "RidgeOnly,TimesFMOnly,FullFramework,TimesFM_XReg,SeasonalNaive,LastValueNaive"
FULL_MODELS = CORE_MODELS


def run(script: str, arguments: list[str], log_dir: Path, label: str) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{label}.log"
    command = [sys.executable, "-X", "utf8", str(PROJECT_ROOT / "scripts" / script), *arguments]
    print(f"\n>>> {label}: {' '.join(command[2:])}")
    started = time.time()
    with log_path.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT)
    elapsed = time.time() - started
    if completed.returncode != 0:
        print(f"    FAILED after {elapsed:.1f} s - see {log_path}")
        raise SystemExit(completed.returncode)
    print(f"    done in {elapsed:.1f} s - {log_path.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fast", action="store_true",
                        help="Use the retained six-model reproducible set")
    parser.add_argument("--skip-models", action="store_true",
                        help="Reuse existing forecast files")
    args = parser.parse_args()

    banner("Reproducible pipeline: Beilin groundwater anomaly study")
    log_dir = PROJECT_ROOT / "logs"
    models = CORE_MODELS if args.fast else FULL_MODELS

    run("01_prepare_data.py", [], log_dir, "step01_prepare_data")
    run("02_build_catalog.py", [], log_dir, "step02_build_catalog")
    if not args.skip_models:
        run("03_run_models.py", ["--models", models], log_dir, "step03_run_models")
    run("04_detect_anomalies.py", ["--models", models], log_dir, "step04_detect_anomalies")
    run("05_association_tests.py", [], log_dir, "step05_association_tests")
    run("06_sensitivity.py", [], log_dir, "step06_sensitivity")
    run("07_make_figures.py", [], log_dir, "step07_make_figures")
    run("08_make_tables.py", [], log_dir, "step08_make_tables")
    run("09_validate.py", [], log_dir, "step09_validate")
    run("11_descriptive_statistics.py", [], log_dir, "step11_descriptive_statistics")
    run("10_package_release.py", [], log_dir, "step10_package_release")
    print("\nPipeline finished. See results/validation_report.csv for the checks.")


if __name__ == "__main__":
    main()
