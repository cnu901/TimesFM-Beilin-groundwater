"""Rolling-window forecasting models.

Every model uses the same protocol: a 365-day context, a 30-day horizon, a
7-day stride, and identical exclusion of target dates that intersect the
flagged 2018 instrument-instability interval.  Differences between models are
therefore confined to how each one produces point and interval forecasts.

Each forecaster returns a ``history`` dictionary mapping an ISO date string to
the list of predictions covering that date, so that overlapping windows can be
fused into a single daily anomaly score downstream.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from .config import Config
from .features import standardise_blocks


History = dict[str, list[dict]]
Forecaster = Callable[..., History]


def build_windows(
    daily: pd.DataFrame,
    config: Config,
    *,
    imputation_flag: pd.Series | None = None,
) -> list[int]:
    """Indices of rolling-window origins with a complete context.

    Target-window exclusions are no longer applied here.  Imputed target dates
    are instead dropped at scoring time so that a window can still provide
    context continuity and valid forecasts for the surrounding observed dates.
    """
    window = config["window"]
    context = int(window["context_days"])
    horizon = int(window["horizon_days"])
    stride = int(window["stride_days"])
    n_daily = len(daily)
    origins = []
    for origin in range(context, n_daily - horizon, stride):
        if np.isnan(daily["water_level"].to_numpy()[origin - context: origin]).any():
            continue
        origins.append(origin)
    return origins


def _store(
    history: History,
    origin: int,
    point: np.ndarray,
    quantiles: np.ndarray | None,
    dates: pd.DatetimeIndex,
    horizon: int,
    *,
    q10_index: int = 1,
    q90_index: int = 9,
) -> None:
    """Append one window's forecasts to the shared history dictionary."""
    for lead in range(horizon):
        target = origin + lead
        if target >= len(dates):
            break
        key = str(dates[target].date())
        history[key].append(
            {
                "horizon": lead + 1,
                "point": float(point[lead]),
                "q10": float(quantiles[lead, q10_index]) if quantiles is not None else None,
                "q90": float(quantiles[lead, q90_index]) if quantiles is not None else None,
            }
        )


def _ridge_fit(features: np.ndarray, target: np.ndarray, alpha: float = 1.0) -> Ridge:
    model = Ridge(alpha=alpha, fit_intercept=True)
    model.fit(features, target)
    return model


# ---------------------------------------------------------------------------
# Classical baselines
# ---------------------------------------------------------------------------
def forecast_ridge_only(
    daily: pd.DataFrame,
    windows: list[int],
    config: Config,
    **_: object,
) -> History:
    """Covariate Ridge regression without the TimesFM residual stage."""
    context = int(config["window"]["context_days"])
    horizon = int(config["window"]["horizon_days"])
    water = daily["water_level"].to_numpy(dtype="float64")
    features = daily[config["feature_names"]].to_numpy(dtype="float64")
    dates = daily.index
    history: History = defaultdict(list)
    for origin in windows:
        context_features = features[origin - context: origin]
        future_features = features[origin: origin + horizon]
        context_features, future_features = standardise_blocks(context_features, future_features)
        model = _ridge_fit(context_features, water[origin - context: origin])
        point = model.predict(future_features)
        _store(history, origin, point, None, dates, horizon)
    return history


def make_naive_forecaster(seasonal: bool) -> Forecaster:
    """Build a naive baseline forecaster (last value or same day last year)."""

    def forecaster(
        daily: pd.DataFrame,
        windows: list[int],
        config: Config,
        **_: object,
    ) -> History:
        context = int(config["window"]["context_days"])
        horizon = int(config["window"]["horizon_days"])
        water = daily["water_level"].to_numpy(dtype="float64")
        dates = daily.index
        history: History = defaultdict(list)
        for origin in windows:
            if seasonal:
                point = water[origin - context: origin][:horizon].copy()
                repeated = np.resize(water[origin - 365: origin], horizon)
                point = repeated
            else:
                point = np.repeat(water[origin - 1], horizon)
            _store(history, origin, point, None, dates, horizon)
        return history

    return forecaster


# ---------------------------------------------------------------------------
# TimesFM family
# ---------------------------------------------------------------------------
def load_timesfm(config: Config):
    """Load and compile the local TimesFM 2.5 checkpoint."""
    source_dir = config.paths.root / config["timesfm"]["source_dir"]
    if str(source_dir) not in sys.path:
        sys.path.insert(0, str(source_dir))
    import torch
    import timesfm

    torch.set_float32_matmul_precision("high")
    model_dir = config.paths.root / config["timesfm"]["model_dir"]
    if not model_dir.is_dir():
        raise FileNotFoundError(
            "TimesFM checkpoint not found at "
            f"{model_dir}. See docs/reproducibility_guide.md for download instructions."
        )
    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(str(model_dir))
    model.compile(
        timesfm.ForecastConfig(
            max_context=int(config["window"]["context_days"]),
            max_horizon=int(config["window"]["horizon_days"]),
            normalize_inputs=True,
            use_continuous_quantile_head=True,
            force_flip_invariance=True,
            infer_is_positive=False,
            fix_quantile_crossing=True,
            return_backcast=True,
        )
    )
    return model


def _timesfm_family(
    mode: str,
    daily: pd.DataFrame,
    windows: list[int],
    config: Config,
    model,
) -> History:
    """Shared driver for the Ridge, TimesFM-only and full-framework models."""
    context = int(config["window"]["context_days"])
    horizon = int(config["window"]["horizon_days"])
    batch = int(config["window"]["batch_size"])
    q10_index = int(config["timesfm"]["quantile_indices"]["q10"])
    q90_index = int(config["timesfm"]["quantile_indices"]["q90"])
    water = daily["water_level"].to_numpy(dtype="float64")
    features = daily[config["feature_names"]].to_numpy(dtype="float64")
    dates = daily.index
    history: History = defaultdict(list)

    for start in range(0, len(windows), batch):
        batch_windows = windows[start: start + batch]
        contexts: list[np.ndarray] = []
        covariate_forecasts: list[np.ndarray] = []
        for origin in batch_windows:
            context_water = water[origin - context: origin]
            context_features = features[origin - context: origin]
            future_features = features[origin: origin + horizon]
            context_features, future_features = standardise_blocks(context_features, future_features)
            ridge = _ridge_fit(context_features, context_water)
            covariate_forecasts.append(ridge.predict(future_features))
            if mode == "raw":
                # TimesFM-only sees the raw water level, with no covariate layer.
                contexts.append(context_water.astype(np.float32))
            elif mode == "full":
                residual_context = context_water - ridge.predict(context_features)
                contexts.append(residual_context.astype(np.float32))
            elif mode == "ridge":
                contexts.append(context_water.astype(np.float32))
            else:
                raise ValueError(f"Unknown TimesFM family mode: {mode}")

        if mode == "ridge":
            points = covariate_forecasts
            quantiles = [None] * len(batch_windows)
        else:
            point_raw, quantile_raw = model.forecast(horizon=horizon, inputs=contexts)
            point_raw = [np.asarray(item)[-horizon:] for item in point_raw]
            quantile_raw = [np.asarray(item)[-horizon:] for item in quantile_raw]
            if mode == "full":
                points = [
                    point_raw[i] + covariate_forecasts[i] for i in range(len(batch_windows))
                ]
                quantiles = [
                    quantile_raw[i] + covariate_forecasts[i][:, np.newaxis]
                    for i in range(len(batch_windows))
                ]
            else:
                points = point_raw
                quantiles = quantile_raw

        for index, origin in enumerate(batch_windows):
            _store(
                history,
                origin,
                points[index],
                quantiles[index],
                dates,
                horizon,
                q10_index=q10_index,
                q90_index=q90_index,
            )
    return history


def forecast_timesfm_only(daily, windows, config, *, model=None, **_) -> History:
    return _timesfm_family("raw", daily, windows, config, model)


def forecast_full_framework(daily, windows, config, *, model=None, **_) -> History:
    return _timesfm_family("full", daily, windows, config, model)


def forecast_timesfm_xreg(daily, windows, config, *, model=None, **_) -> History:
    """Native TimesFM 2.5 in-context XReg baseline (``xreg + timesfm``)."""
    context = int(config["window"]["context_days"])
    horizon = int(config["window"]["horizon_days"])
    batch = int(config["window"]["batch_size"])
    q10_index = int(config["timesfm"]["quantile_indices"]["q10"])
    q90_index = int(config["timesfm"]["quantile_indices"]["q90"])
    water = daily["water_level"].to_numpy(dtype="float64")
    features = daily[config["feature_names"]].to_numpy(dtype="float64")
    dates = daily.index
    history: History = defaultdict(list)

    try:
        import jax  # noqa: F401
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(f"Native XReg requires the optional JAX extras: {exc}") from exc

    for start in range(0, len(windows), batch):
        batch_windows = windows[start: start + batch]
        inputs = [water[origin - context: origin].astype(np.float32) for origin in batch_windows]
        dynamic = {f"f{column}": [] for column in range(features.shape[1])}
        for origin in batch_windows:
            context_raw = features[origin - context: origin]
            full_raw = features[origin - context: origin + horizon]
            _, full_standardised = standardise_blocks(context_raw, full_raw)
            for column in range(features.shape[1]):
                dynamic[f"f{column}"].append(full_standardised[:, column].astype(np.float32))
        point_raw, quantile_raw = model.forecast_with_covariates(
            inputs=inputs,
            dynamic_numerical_covariates=dynamic,
            xreg_mode="xreg + timesfm",
            ridge=1.0,
        )
        for index, origin in enumerate(batch_windows):
            point = np.asarray(point_raw[index])[-horizon:]
            quantile = np.asarray(quantile_raw[index])[-horizon:]
            _store(
                history,
                origin,
                point,
                quantile,
                dates,
                min(horizon, len(point)),
                q10_index=q10_index,
                q90_index=q90_index,
            )
    return history


# ---------------------------------------------------------------------------
# Chronos baseline
# ---------------------------------------------------------------------------
def forecast_chronos(daily, windows, config, **_) -> History:
    """Amazon Chronos zero-shot baseline with native quantiles."""
    import torch
    from chronos import ChronosPipeline

    context = int(config["window"]["context_days"])
    horizon = int(config["window"]["horizon_days"])
    model_name = config["models"]["chronos"]["model_name"]
    n_samples = int(config["models"]["chronos"]["num_samples"])
    cache_dir = config.paths.root / "third_party" / "models" / "chronos_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    pipeline = ChronosPipeline.from_pretrained(model_name, cache_dir=str(cache_dir))
    water = daily["water_level"].to_numpy(dtype="float64")
    dates = daily.index
    history: History = defaultdict(list)
    for origin in windows:
        tensor = torch.tensor(water[origin - context: origin], dtype=torch.float32)
        samples = pipeline.predict(tensor, num_samples=n_samples, prediction_length=horizon)
        samples = samples.squeeze(0).numpy()
        point = np.median(samples, axis=0)
        quantiles = np.stack(
            [np.percentile(samples, 10, axis=0), np.percentile(samples, 90, axis=0)], axis=1
        )
        _store(history, origin, point, quantiles, dates, horizon, q10_index=0, q90_index=1)
    return history


# ---------------------------------------------------------------------------
# LSTM baseline
# ---------------------------------------------------------------------------
def forecast_lstm(daily, windows, config, **_) -> History:
    """Two-layer LSTM with causal expanding-window refits."""
    import torch
    import torch.nn as nn

    settings = config["models"]["lstm"]
    context = int(config["window"]["context_days"])
    horizon = int(config["window"]["horizon_days"])
    refit_days = int(settings["refit_days"])
    hidden = int(settings["hidden_size"])
    layers = int(settings["num_layers"])
    epochs = int(settings["epochs"])
    learning_rate = float(settings["learning_rate"])
    batch_size = int(settings["batch_size"])
    seed = int(config["project"]["random_seed"])

    class LSTMForecaster(nn.Module):
        def __init__(self, in_dim: int):
            super().__init__()
            self.lstm = nn.LSTM(in_dim, hidden, layers, batch_first=True)
            self.head = nn.Linear(hidden, horizon)

        def forward(self, x):
            output, _ = self.lstm(x)
            return self.head(output[:, -1, :])

    water = daily["water_level"].to_numpy(dtype="float64")
    features = daily[config["feature_names"]].to_numpy(dtype="float64")
    inputs = np.column_stack([water, features]).astype(np.float32)
    dates = daily.index
    history: History = defaultdict(list)

    groups: dict[int, list[int]] = defaultdict(list)
    for origin in windows:
        groups[max(0, (origin - context) // refit_days)].append(origin)

    for _, group in sorted(groups.items()):
        train_end = group[0]
        if train_end <= context + horizon:
            continue
        torch.manual_seed(seed)
        np.random.seed(seed)
        feature_mean = inputs[:train_end].mean(axis=0)
        feature_std = inputs[:train_end].std(axis=0)
        feature_std[feature_std == 0] = 1.0
        scaled = (inputs - feature_mean) / feature_std
        target_mean = water[:train_end].mean()
        target_std = water[:train_end].std() or 1.0

        sequences, targets = [], []
        for index in range(context, train_end - horizon):
            if np.isnan(scaled[index - context:index]).any():
                continue
            if np.isnan(water[index:index + horizon]).any():
                continue
            sequences.append(scaled[index - context:index])
            targets.append((water[index:index + horizon] - target_mean) / target_std)
        if not sequences:
            continue

        x_tensor = torch.tensor(np.stack(sequences), dtype=torch.float32)
        y_tensor = torch.tensor(np.stack(targets), dtype=torch.float32)
        model = LSTMForecaster(in_dim=inputs.shape[1])
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        loss_fn = nn.MSELoss()
        model.train()
        for _ in range(epochs):
            permutation = torch.randperm(len(x_tensor))
            for start in range(0, len(x_tensor), batch_size):
                batch_index = permutation[start:start + batch_size]
                optimizer.zero_grad()
                loss = loss_fn(model(x_tensor[batch_index]), y_tensor[batch_index])
                loss.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            for origin in group:
                window = torch.tensor(
                    scaled[origin - context:origin][np.newaxis, :, :], dtype=torch.float32
                )
                prediction = model(window).numpy().ravel() * target_std + target_mean
                _store(history, origin, prediction, None, dates, horizon)
    return history


def history_to_frame(history: History) -> pd.DataFrame:
    """Flatten a history dictionary into a tidy prediction table."""
    rows = []
    for date_str, predictions in history.items():
        for prediction in predictions:
            rows.append({"date": date_str, **prediction})
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame["date"] = pd.to_datetime(frame["date"])
        frame = frame.sort_values(["date", "horizon"]).reset_index(drop=True)
    return frame


def frame_to_history(frame: pd.DataFrame) -> History:
    """Rebuild a history dictionary from a tidy prediction table."""
    history: History = defaultdict(list)
    for row in frame.itertuples(index=False):
        history[str(pd.Timestamp(row.date).date())].append(
            {
                "horizon": int(row.horizon),
                "point": float(row.point),
                "q10": None if pd.isna(row.q10) else float(row.q10),
                "q90": None if pd.isna(row.q90) else float(row.q90),
            }
        )
    return history
