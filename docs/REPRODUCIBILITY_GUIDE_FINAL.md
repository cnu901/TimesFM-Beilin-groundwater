# Reproducibility guide

1. Install Python 3.11 and the pinned packages in `environment/requirements.txt`.
2. Obtain permission for the raw monitoring files and place them under `data/raw/` using the names in `config/config.yaml`.
3. Install the upstream TimesFM 2.5 source and download the 200M checkpoint as described in the project’s private run notes.
4. Run `scripts/run_all.py` for a full rebuild, or `scripts/run_all.py --skip-models` to reuse the released forecasts.
5. Run `scripts/09_validate.py`; the expected final checks are 4748 daily records, 622 rolling windows, 40 full-framework segments, 15 residual segments, 11 evaluable events, 6 short-term matches, and Molchan skill 0.44.
6. Run `scripts/07_make_figures.py` and `scripts/08_make_tables.py` to regenerate the publication outputs.

The public package is intentionally sufficient for inspection of the published derived results but cannot rebuild the observation series without the restricted inputs.
