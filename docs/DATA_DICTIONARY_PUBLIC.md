# Public data dictionary

## Public inputs

- `data/external/cenc_catalogue_2013_2025.csv`: 12 CENC events with magnitude, location, depth, distance to the well, and catalogue metadata.
- `data/external/吉林松原地震目录.txt`: 659-record local catalogue with a uniform M_S scale; used only for Songyuan sequence grouping.

## Public derived outputs

- `results/forecasts/`: point and q10–q90 forecasts for the six retained models.
- `results/anomaly_segments/`: segment start/end, duration, maximum score, screening class, rainfall total, and pressure change.
- `results/anomaly_scores_public/`: date, score, warning level, number of predictions, month, and anomaly flag; the observed water-level column is removed.
- `results/statistics/`: forecast metrics, event matches, permutation/FDR, sequence checks, Molchan metrics, and radius sensitivity.
- `figures/` and `tables/`: publication outputs and permitted source arrays.

## Withheld files

Raw EQT monitoring exports, corrected hourly/daily water-level files, preprocessing rows containing water-level values, and source arrays that contain or can reconstruct observation values are withheld. They can be added only after written permission from the data owner.
