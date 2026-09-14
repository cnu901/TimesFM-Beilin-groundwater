"""Construction of the 13 covariates used by the Ridge stage.

The covariate set combines barometric loading terms, rainfall accumulation
windows, a centred linear trend, and annual / semi-annual harmonics.  Features
are built on the hourly grid, standardised once over rows with a valid water
level, and then averaged to the daily grid that the forecast models use.

Building the rainfall accumulations on the hourly grid (rather than on daily
totals) keeps the short accumulation windows aligned with the raw record.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


FEATURE_NAMES = [
    "pressure",
    "pressure_change_72h",
    "pressure_lag_1d",
    "pressure_lag_3d",
    "rainfall",
    "rainfall_3d",
    "rainfall_7d",
    "rainfall_30d",
    "trend",
    "sin_annual",
    "cos_annual",
    "sin_semiannual",
    "cos_semiannual",
]

FEATURE_DESCRIPTIONS = {
    "pressure": "Instantaneous barometric loading (hPa)",
    "pressure_change_72h": "3-day pressure change (hPa)",
    "pressure_lag_1d": "Pressure lagged by 1 day (hPa)",
    "pressure_lag_3d": "Pressure lagged by 3 days (hPa)",
    "rainfall": "Contemporaneous rainfall (mm h-1)",
    "rainfall_3d": "3-day cumulative rainfall (mm)",
    "rainfall_7d": "7-day cumulative rainfall (mm)",
    "rainfall_30d": "30-day cumulative rainfall (mm)",
    "trend": "Centred linear trend (dimensionless)",
    "sin_annual": "Annual sine harmonic",
    "cos_annual": "Annual cosine harmonic",
    "sin_semiannual": "Semi-annual sine harmonic",
    "cos_semiannual": "Semi-annual cosine harmonic",
}


def build_hourly_covariates(
    pressure_hpa: pd.Series,
    rainfall_mm: pd.Series,
) -> pd.DataFrame:
    """Build the 13 raw (unstandardised) covariates on the hourly grid."""
    pressure = pressure_hpa.to_numpy(dtype="float64")
    rainfall = rainfall_mm.to_numpy(dtype="float64")
    n_steps = len(pressure)

    features = np.zeros((n_steps, len(FEATURE_NAMES)), dtype="float64")
    features[:, 0] = pressure

    change = np.zeros(n_steps)
    change[72:] = pressure[72:] - pressure[:-72]
    features[:, 1] = change
    features[24:, 2] = pressure[:-24]
    features[72:, 3] = pressure[:-72]
    features[:, 4] = rainfall

    rainfall_series = pd.Series(rainfall)
    features[:, 5] = rainfall_series.rolling(24 * 3, min_periods=1).sum().to_numpy()
    features[:, 6] = rainfall_series.rolling(24 * 7, min_periods=1).sum().to_numpy()
    features[:, 7] = rainfall_series.rolling(24 * 30, min_periods=1).sum().to_numpy()

    steps = np.arange(n_steps, dtype="float64")
    # ``steps`` is an hourly index.  Dividing by 24 converts it to days, so
    # the annual and semi-annual harmonics now have the intended periods of
    # 365.2425 and 182.62125 days rather than the previous 15.2/7.6 days.
    days = steps / 24.0
    phase = 2.0 * np.pi * days / 365.2425
    features[:, 8] = (steps - steps.mean()) / max(1.0, n_steps)
    features[:, 9] = np.sin(phase)
    features[:, 10] = np.cos(phase)
    features[:, 11] = np.sin(2.0 * phase)
    features[:, 12] = np.cos(2.0 * phase)

    return pd.DataFrame(features, index=pressure_hpa.index, columns=FEATURE_NAMES)


def to_daily(features_hourly: pd.DataFrame) -> pd.DataFrame:
    """Average the hourly covariates onto the daily analysis grid."""
    return features_hourly.resample("D").mean()


def build_daily_covariates(
    pressure_hpa: pd.Series,
    rainfall_mm: pd.Series,
) -> pd.DataFrame:
    """Build raw (unstandardised) daily covariates.

    Standardisation is deliberately *not* applied here.  The rolling forecast
    models standardise each window from its own 365-day training context so
    that no future statistics leak into a forecast.  Descriptive coefficient
    tables standardise their own fitting subset separately.
    """
    hourly = build_hourly_covariates(pressure_hpa, rainfall_mm)
    return to_daily(hourly)


def standardise_frame(
    frame: pd.DataFrame,
    *,
    mask: pd.Series | None = None,
) -> tuple[pd.DataFrame, StandardScaler]:
    """Standardise a descriptive feature frame on an optional fitting subset."""
    scaler = StandardScaler()
    fit = frame if mask is None else frame.loc[mask.to_numpy()]
    scaler.fit(fit)
    scaled = pd.DataFrame(
        scaler.transform(frame),
        index=frame.index,
        columns=frame.columns,
    )
    return scaled, scaler


def standardise_blocks(
    train: np.ndarray,
    apply: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Standardise blocks of features using training-block statistics.

    The mean and standard deviation are estimated only from ``train`` and then
    applied to both ``train`` and ``apply``.  A zero-variance feature is left
    unshifted and unscaled so that a constant covariate cannot divide by zero.
    """
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True)
    std = np.where(std < 1e-12, 1.0, std)
    scaled_train = (train - mean) / std
    scaled_apply = None if apply is None else (apply - mean) / std
    return scaled_train, scaled_apply
