# Final analysis protocol

The pipeline screens daily Beilin groundwater-level anomalies with a covariate-augmented TimesFM workflow and tests residual-segment timing against a regional CENC catalogue. The primary unit is the earthquake event; sequence-level grouping is a robustness check.

Key fixed settings: 365-day context, 30-day horizon, 7-day stride, 13 Ridge covariates, 90th/95th percentile expanding thresholds, minimum four-day segment duration, rainfall threshold 30 mm with a 10 mm borderline margin, pressure threshold 11 hPa with a 4 hPa borderline margin, 1000 permutations, and Benjamini–Hochberg correction over the four declared timing windows.

Water-level targets intersecting long gaps (>24 h) are excluded from scoring and matching. The 30 June–20 July 2018 instrument-instability interval contains 469 missing hourly values among 504 timestamps.
