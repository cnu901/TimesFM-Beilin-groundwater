# Public reproducibility package: Covariate-Augmented TimesFM

This archive contains the code, public catalogue inputs, derived results, figure/table files, and metadata supporting the manuscript:

*Covariate-Augmented TimesFM Foundation Model for Groundwater-Level Anomaly Detection and Earthquake-Association Analysis: A 13-Year Case Study at Beilin Well, Northeast China.*

## Scope of the release

The study performs retrospective groundwater anomaly screening followed by event-level earthquake-association testing. It does not demonstrate an earthquake precursor or operational earthquake-prediction ability. The exploratory 30–90-day result is 6/11 evaluable events (p = 0.002; FDR q = 0.008); the sequence-level check is 2/3 (p = 0.060), and the retrospective Molchan skill score is 0.44.

The raw monitoring exports and observation-bearing source arrays are withheld because they are subject to the data owner’s sharing rules. Aggregate statistics, model predictions, anomaly segments, catalogue data, figures, tables, and safe source arrays are included. See `DATA_NOTICE.md` and `docs/DATA_DICTIONARY_PUBLIC.md`.

## Contents

- `src/`, `scripts/`, `config/`, `environment/`: analysis code and configuration.
- `data/external/`: CENC catalogue and the auxiliary Songyuan M_S catalogue.
- `results/`: predictions, anomaly segments, public anomaly scores, statistics, and sensitivity analyses.
- `figures/`: PNG, PDF, SVG, and permitted source arrays for Figures 1–8.
- `tables/`: main and supplementary tables with source data.
- `docs/`: protocol, public data dictionary, reproducibility guide, and change log.
- `metadata/`: Zenodo metadata and SHA-256 manifest.

## Reproduction

The complete pipeline requires the restricted monitoring files and the TimesFM 2.5 checkpoint. Place those files in the paths specified by `config/config.yaml`, install `environment/requirements.txt`, and run:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r environment\requirements.txt
.\.venv\Scripts\python.exe scripts\run_all.py --skip-models
.\.venv\Scripts\python.exe scripts\09_validate.py
```

To regenerate TimesFM forecasts, install the upstream TimesFM source and download the checkpoint described in `docs/REPRODUCIBILITY_GUIDE_FINAL.md`. The released forecast files allow the downstream anomaly, statistics, figure, table, and validation steps to run without recomputing the model.

## Citation

See `CITATION.cff`. After Zenodo reserves a DOI, add the DOI to the article Data Availability Statement and to the repository metadata.
