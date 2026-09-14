"""Draw manuscript Figures 1-8 from the analysis outputs.

Every figure is written to its own folder under ``figures/`` with three
formats (PNG, PDF, SVG) and a ``source_data`` subfolder holding the exact
numbers plotted.  Figures 1 and 2 are generated from the released CENC,
processed-hourly and configuration files; no untracked GIS or borehole
material is required.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bootstrap import banner  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402
import matplotlib.dates as mdates  # noqa: E402

from src import plotting  # noqa: E402
from src.config import figure_dir, load_config  # noqa: E402
from src.features import FEATURE_DESCRIPTIONS, FEATURE_NAMES, standardise_frame  # noqa: E402
from src.pipeline import (  # noqa: E402
    canonical_hourly_path,
    load_daily,
    read_hourly_table,
    residual_segment_starts,
)


COLORS = plotting.COLORS


# ---------------------------------------------------------------------------
# Figure 1 - study area and monitoring configuration
# ---------------------------------------------------------------------------
def _geodesic_circle(latitude: float, longitude: float, radius_km: float,
                     n_points: int = 361) -> tuple[np.ndarray, np.ndarray]:
    """Return a small-circle approximation on a spherical Earth."""
    earth_radius_km = 6371.0
    bearings = np.linspace(0.0, 2.0 * np.pi, n_points)
    angular = radius_km / earth_radius_km
    lat0 = np.deg2rad(latitude)
    lon0 = np.deg2rad(longitude)
    lat = np.arcsin(
        np.sin(lat0) * np.cos(angular)
        + np.cos(lat0) * np.sin(angular) * np.cos(bearings)
    )
    lon = lon0 + np.arctan2(
        np.sin(bearings) * np.sin(angular) * np.cos(lat0),
        np.cos(angular) - np.sin(lat0) * np.sin(lat),
    )
    return np.rad2deg(lon), np.rad2deg(lat)


def build_figure1(config) -> None:
    """Draw a traceable regional catalogue map and monitoring schematic."""
    directory = figure_dir("figure1_study_area")
    cenc = pd.read_csv(config.paths.statistics / "cenc_catalogue.csv",
                       parse_dates=["origin_time"])
    lat0 = float(config["catalogue"]["well_latitude"])
    lon0 = float(config["catalogue"]["well_longitude"])
    radius = float(config["catalogue"]["radius_km"])
    circle_lon, circle_lat = _geodesic_circle(lat0, lon0, radius)

    fig = plt.figure(figsize=(6.7, 3.9))
    grid = GridSpec(1, 2, figure=fig, width_ratios=[1.55, 0.75],
                    wspace=0.28, left=0.07, right=0.97, top=0.86, bottom=0.14)

    ax = fig.add_subplot(grid[0, 0])
    ax.plot(circle_lon, circle_lat, color=COLORS["orange"], linestyle="--",
            linewidth=1.2, label="300-km catalogue domain", zorder=1)
    magnitude_bins = [(4.0, 5.0, COLORS["blue"]),
                      (5.0, 6.0, COLORS["orange"]),
                      (6.0, 10.0, COLORS["red"])]
    for lower, upper, color in magnitude_bins:
        subset = cenc.loc[(cenc["magnitude"] >= lower) & (cenc["magnitude"] < upper)]
        ax.scatter(subset["longitude"], subset["latitude"],
                   s=28 + 24 * (subset["magnitude"] - 4.0),
                   facecolor=color, edgecolor="white", linewidth=0.55,
                   alpha=0.92, zorder=3,
                   label=(f"{lower:.0f} <= M < {upper:.0f}" if upper < 10
                          else f"M >= {lower:.0f}"))
    ax.scatter([lon0], [lat0], marker="*", s=120, color=COLORS["green"],
               edgecolor="white", linewidth=0.7, zorder=4, label="Beilin well")
    # Label only the three largest events; the complete event list and IDs are
    # supplied as Figure 1 source data and in Table 2, avoiding overlap in the
    # dense Songyuan cluster at manuscript width.
    for event in cenc.nlargest(1, "magnitude").itertuples(index=False):
        ax.annotate(f"M{event.magnitude:.1f}", (event.longitude, event.latitude),
                    xytext=(5, 5), textcoords="offset points", fontsize=6.2,
                    color=COLORS["ink"], zorder=5,
                    arrowprops=dict(arrowstyle="-", color=COLORS["muted"], lw=0.5))
    ax.set_xlim(min(circle_lon.min(), cenc["longitude"].min()) - 0.35,
                max(circle_lon.max(), cenc["longitude"].max()) + 0.35)
    ax.set_ylim(min(circle_lat.min(), cenc["latitude"].min()) - 0.35,
                max(circle_lat.max(), cenc["latitude"].max()) + 0.35)
    ax.set_xlabel("Longitude (E)")
    ax.set_ylabel("Latitude (N)")
    ax.set_title("CENC events and catalogue domain", loc="left")
    ax.grid(True)
    ax.legend(loc="upper left", fontsize=6.2, ncol=2, frameon=True)
    plotting.add_panel_label(ax, "(a)")

    ax = fig.add_subplot(grid[0, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.5, 0.97, "Monitoring configuration", ha="center", va="top",
            fontsize=9.5, fontweight="bold", color=COLORS["ink"])
    ax.add_patch(FancyBboxPatch((0.32, 0.18), 0.36, 0.60,
                                boxstyle="round,pad=0.015,rounding_size=0.01",
                                facecolor="#F7F9FA", edgecolor=COLORS["muted"],
                                linewidth=1.0))
    ax.plot([0.50, 0.50], [0.25, 0.72], color=COLORS["muted"], linewidth=2.0)
    ax.scatter([0.50], [0.73], marker="*", s=100, color=COLORS["green"],
               edgecolor="white", linewidth=0.5, zorder=3)
    ax.text(0.50, 0.16, "Beilin well\n46.62 N, 126.97 E", ha="center", va="top",
            fontsize=7.2, color=COLORS["ink"])
    _arrow(ax, 0.30, 0.68, 0.08, 0.82, COLORS["blue"], lw=1.0, mutation=10)
    ax.text(0.02, 0.88, "Daily water-level\ndepth", ha="left", va="top",
            fontsize=7.0, color=COLORS["blue"])
    _arrow(ax, 0.70, 0.52, 0.93, 0.67, COLORS["red"], lw=1.0, mutation=10)
    ax.text(0.63, 0.80, "Pressure + rainfall\n(Suihua station, ~5 km)",
            ha="left", va="top", fontsize=7.0, color=COLORS["red"])
    ax.text(0.50, 0.05, "Retrospective daily analysis", ha="center", va="bottom",
            fontsize=6.8, color=COLORS["muted"])
    plotting.add_panel_label(ax, "(b)", x=0.0, y=0.99)

    fig.suptitle("Study area and data sources", fontsize=10.5, fontweight="bold", y=0.96)
    plotting.save_figure(fig, directory, "Figure1_study_area")
    events_source = cenc[["origin_time", "latitude", "longitude", "magnitude",
                          "location", "distance_to_well_km"]].copy()
    events_source.insert(0, "event_no", np.arange(1, len(events_source) + 1))
    plotting.write_source_data(directory, "figure1_panel_a_cenc_events", events_source)
    plotting.write_source_data(directory, "figure1_panel_a_domain_boundary",
                               pd.DataFrame({"longitude": circle_lon, "latitude": circle_lat,
                                             "radius_km": radius}))
    plotting.write_source_data(
        directory, "figure1_panel_b_monitoring_configuration",
        pd.DataFrame([
            {"element": "Beilin well", "latitude": lat0, "longitude": lon0,
             "value": "46.62 N, 126.97 E"},
            {"element": "Suihua station", "latitude": np.nan, "longitude": np.nan,
             "value": "approximately 5 km from well"},
        ]),
    )
    print("figure 1 written")


# ---------------------------------------------------------------------------
# Figure 2 - data quality control and preprocessing
# ---------------------------------------------------------------------------
def _format_date_axis(ax) -> None:
    locator = mdates.AutoDateLocator(minticks=3, maxticks=6)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))


def build_figure2(config) -> None:
    """Draw the full record as the main panel with defect-specific insets."""
    directory = figure_dir("figure2_preprocessing")
    hourly = read_hourly_table(canonical_hourly_path(config))
    gap_cfg = config["preprocessing"]["flagged_gap"]
    gap_start, gap_end = pd.Timestamp(gap_cfg["start"]), pd.Timestamp(gap_cfg["end"])
    step_time = pd.Timestamp(config["preprocessing"]["step_correction"]["date"])
    pol_cfg = config["preprocessing"]["polarity_correction"]
    pol_start, pol_end = pd.Timestamp(pol_cfg["start"]), pd.Timestamp(pol_cfg["end"])
    plot_full = hourly[["water_level_original", "water_level_final"]].resample("6h").mean()

    fig, ax = plt.subplots(figsize=(6.7, 5.0))
    fig.subplots_adjust(left=0.09, right=0.98, top=0.92, bottom=0.12)

    ax.plot(plot_full.index, plot_full["water_level_original"], color=COLORS["grey"],
            linewidth=0.45, label="Raw")
    ax.plot(plot_full.index, plot_full["water_level_final"], color=COLORS["water"],
            linewidth=0.65, label="Corrected / filled")
    ax.axvspan(gap_start, gap_end, color=COLORS["orange"], alpha=0.18,
               label="2018 flagged gap")
    ax.axvline(step_time, color=COLORS["red"], linestyle="--", linewidth=0.9,
               label="2015 step")
    ax.axvspan(pol_start, pol_end, color=COLORS["purple"], alpha=0.10,
               label="2021 polarity interval")
    ax.set_ylabel("Water-level depth (m)")
    ax.set_title("Full hourly record (6-h means)", loc="left")
    ax.grid(True); _format_date_axis(ax)
    ax.set_ylim(26, 33.5)
    ax.invert_yaxis()
    ax.set_yticks([26, 28, 30, 32, 33.5])
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    ax.legend(loc="lower left", fontsize=6.2, ncol=2)
    plotting.add_panel_label(ax, "(a)")

    zooms = [
        ("2015-08-24", "2015-08-28", "Confirmed 2015 step",
         "shift before 16:00 by -0.3536 m", step_time, "step"),
        ("2018-06-27", "2018-07-23", "2018 instrument-instability gap",
         "469 of 504 raw hourly values missing", gap_start, "gap"),
        ("2021-04-10", "2021-06-24", "2021 polarity inversion",
         "mirrored interval about the two-anchor centre", pol_start, "pol"),
    ]
    inset_specs = [
        (0.02, 0.56, 0.30, 0.36),
        (0.35, 0.56, 0.30, 0.36),
        (0.68, 0.56, 0.30, 0.36),
    ]
    source_names = ["figure2_panel_b_2015_step", "figure2_panel_c_2018_gap",
                    "figure2_panel_d_2021_polarity"]
    panel_labels = ["(b)", "(c)", "(d)"]
    for (start, end, title, note, marker, kind), (x0, y0, w, h), name, label in zip(
            zooms, inset_specs, source_names, panel_labels):
        start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
        sub = hourly.loc[start_ts:end_ts, ["water_level_original", "water_level_final"]].copy()
        iax = ax.inset_axes([x0, y0, w, h])
        iax.plot(sub.index, sub["water_level_original"], color=COLORS["grey"],
                 linewidth=0.7, label="Raw")
        iax.plot(sub.index, sub["water_level_final"], color=COLORS["water"],
                 linewidth=0.8, label="Corrected / filled")
        if kind == "step":
            iax.axvline(marker, color=COLORS["red"], linestyle="--", linewidth=0.8)
        elif kind == "gap":
            iax.axvspan(gap_start, gap_end, color=COLORS["orange"], alpha=0.20)
            imputed = sub.loc[(sub.index >= gap_start) & (sub.index <= gap_end)]
            iax.plot(imputed.index, imputed["water_level_final"], color=COLORS["orange"],
                     linewidth=1.0, linestyle="--", label="Context-only imputation")
        else:
            iax.axvspan(pol_start, pol_end, color=COLORS["purple"], alpha=0.13)
        iax.text(0.02, 0.04, note, transform=iax.transAxes, ha="left", va="bottom",
                 fontsize=5.6, color=COLORS["ink"],
                 bbox=dict(boxstyle="round,pad=0.18", facecolor="white",
                           edgecolor=COLORS["grid"], alpha=0.90))
        iax.set_title(title, loc="left", fontsize=7.2)
        iax.invert_yaxis()
        iax.grid(True); _format_date_axis(iax)
        iax.legend(loc="best", fontsize=5.4)
        iax.tick_params(axis="both", labelsize=5.8)
        plotting.add_panel_label(iax, label, x=0.02, y=0.98, fontsize=7.5)

    ax.set_xlabel("Date")
    fig.suptitle("Data quality control and preprocessing", fontsize=10.5,
                 fontweight="bold", y=0.98)
    plotting.save_figure(fig, directory, "Figure2_preprocessing")
    plotting.write_source_data(directory, "figure2_panel_a_full_record",
                               plot_full.reset_index().rename(columns={"time": "date"}))
    for name, (start, end, *_rest) in zip(source_names, zooms):
        sub = hourly.loc[pd.Timestamp(start):pd.Timestamp(end),
                         ["water_level_original", "water_level_final"]].reset_index()
        sub["flagged_gap"] = (sub["time"] >= gap_start) & (sub["time"] <= gap_end)
        sub["polarity_interval"] = (sub["time"] >= pol_start) & (sub["time"] <= pol_end)
        plotting.write_source_data(directory, name, sub)
    print("figure 2 written")


# ---------------------------------------------------------------------------
# Figure 3 - workflow
# ---------------------------------------------------------------------------
def _box(ax, x, y, w, h, title, body, edge, fill, title_size=8.0, body_size=7.0):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.006,rounding_size=0.016",
            linewidth=1.1,
            edgecolor=edge,
            facecolor=fill,
            zorder=2,
        )
    )
    ax.text(x + w / 2, y + h - 0.014, title, ha="center", va="top",
            fontsize=title_size, fontweight="bold", color=edge, zorder=3)
    ax.text(x + w / 2, y + 0.012, body, ha="center", va="bottom", fontsize=body_size,
            color=COLORS["ink"], linespacing=1.15, zorder=3)


def _arrow(ax, x1, y1, x2, y2, color=None, lw=0.9, mutation=9):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=mutation,
            linewidth=lw, color=color or COLORS["ink"], shrinkA=2, shrinkB=2, zorder=4,
        )
    )


def _stage(ax, y, number, text, color):
    ax.text(0.04, y, f"{number}  {text}", ha="left", va="center",
            fontsize=8.6, fontweight="bold", color=color, zorder=3)
    ax.plot([0.04, 0.96], [y - 0.022, y - 0.022], color=color, linewidth=0.8, alpha=0.7, zorder=1)


def build_figure3(config) -> None:
    directory = figure_dir("figure3_workflow")
    fig, ax = plt.subplots(figsize=(6.7, 7.3))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.5, 0.982, "Covariate-Augmented TimesFM Hindcast Workflow",
            ha="center", va="top", fontsize=11.5, fontweight="bold", color=COLORS["ink"])
    ax.text(0.5, 0.952,
            "Hourly monitoring records are corrected, decomposed into covariates and residual structure,\n"
            "and tested against a regional earthquake catalogue",
            ha="center", va="top", fontsize=7.0, color=COLORS["muted"], linespacing=1.3)

    x0, width, gap = 0.055, 0.205, 0.028
    columns = [x0 + index * (width + gap) for index in range(4)]

    _stage(ax, 0.905, "1", "DATA AND COVARIATE PREPARATION", COLORS["blue"])
    for column, (title, body) in zip(
        columns,
        [
            ("Hourly observations", "Water level, pressure\nand rainfall, 2013-2025"),
            ("Quality control", "Step and polarity \ncorrection; gap flagging"),
            ("Daily aggregation", "Daily means for water\nlevel and pressure"),
            ("Thirteen covariates", "Pressure lags, rainfall\naccumulations, trend,\nharmonics"),
        ],
    ):
        _box(ax, column, 0.752, width, 0.108, title, body, COLORS["blue"], COLORS["blue_fill"])
    for index in range(3):
        _arrow(ax, columns[index] + width, 0.806, columns[index + 1], 0.806, COLORS["blue"])

    ax.add_patch(
        FancyBboxPatch(
            (0.055, 0.668), 0.890, 0.050,
            boxstyle="round,pad=0.006,rounding_size=0.016",
            linewidth=1.0, edgecolor=COLORS["blue"], facecolor="#F4F9FC", zorder=2,
        )
    )
    ax.text(0.072, 0.693, "Rolling protocol", ha="left", va="center",
            fontsize=7.6, fontweight="bold", color=COLORS["blue"], zorder=3)
    ax.text(0.946, 0.693,
            "365-day context   |   30-day horizon   |   7-day stride   |   622 windows",
            ha="right", va="center", fontsize=7.2, color=COLORS["ink"], zorder=3)
    _arrow(ax, 0.946, 0.668, 0.946, 0.592, COLORS["blue"], lw=1.0, mutation=10)

    _stage(ax, 0.572, "2", "TWO-STAGE FORECASTING", COLORS["green"])
    for column, (title, body) in zip(
        columns,
        [
            ("Ridge regression", "Linear covariate fit\n" + r"$\alpha = 1.0$"),
            ("Residual extraction", r"$r(t) = y(t) - \hat{y}_{cov}(t)$"),
            ("TimesFM 2.5", "Residual forecast,\nquantiles q10-q90"),
            ("Recombination", r"$\hat{y}_{final} = \hat{y}_{cov} + \hat{r}$"),
        ],
    ):
        _box(ax, column, 0.428, width, 0.108, title, body, COLORS["green"], COLORS["green_fill"])
    for index in range(3):
        _arrow(ax, columns[index] + width, 0.482, columns[index + 1], 0.482, COLORS["green"])

    ax.add_patch(
        FancyBboxPatch(
            (0.055, 0.320), 0.890, 0.072,
            boxstyle="round,pad=0.006,rounding_size=0.016",
            linewidth=1.0, edgecolor=COLORS["orange"], facecolor=COLORS["orange_fill"],
            linestyle=(0, (3, 2)), zorder=2,
        )
    )
    ax.text(0.5, 0.386, "Retrospective hindcast", ha="center", va="top",
            fontsize=7.0, fontweight="bold", color="#9A6B00", zorder=3)
    ax.text(0.5, 0.339,
            "Observed pressure and rainfall are used over each forecast horizon.\n"
            "Operational use would require separate covariate forecasts.",
            ha="center", va="center", fontsize=6.4, color=COLORS["ink"],
            linespacing=1.25, zorder=3)
    _arrow(ax, 0.946, 0.320, 0.946, 0.262, COLORS["green"], lw=1.0, mutation=10)

    _stage(ax, 0.242, "3", "ANOMALY INTERPRETATION AND EARTHQUAKE ASSOCIATION", COLORS["orange"])
    for column, (title, body, edge, fill) in zip(
        columns,
        [
            ("Weighted score", r"$S(t)=\sum w_i d_i / \sum w_i$" + "\n" + r"$w_i = 1/h_i^2$",
             COLORS["orange"], COLORS["orange_fill"]),
            ("Calibration\nthresholds", "Expanding monthly T90 / T95\nfrom strictly earlier scores",
             COLORS["orange"], COLORS["orange_fill"]),
            ("Cause\nclassification", "Rainfall, pressure,\nmixed or residual screening",
             COLORS["orange"], COLORS["orange_fill"]),
            ("Earthquake\nassociation", "Sequence grouping,\npermutation, FDR, Molchan",
             COLORS["purple"], COLORS["purple_fill"]),
        ],
    ):
        _box(ax, column, 0.104, width, 0.106, title, body, edge, fill,
             title_size=7.0, body_size=6.4)
    for index in range(3):
        color = COLORS["orange"] if index < 2 else COLORS["purple"]
        _arrow(ax, columns[index] + width, 0.157, columns[index + 1], 0.157, color)

    ax.add_patch(
        FancyBboxPatch(
            (0.055, 0.024), 0.890, 0.048,
            boxstyle="round,pad=0.006,rounding_size=0.016",
            linewidth=1.0, edgecolor="#455A64", facecolor=COLORS["grey_fill"], zorder=2,
        )
    )
    ax.text(0.5, 0.048,
            "Outputs:   anomaly catalogue   |   cause classes   |   timing association statistics",
            ha="center", va="center", fontsize=7.2, color=COLORS["ink"], zorder=3)
    _arrow(ax, 0.946, 0.104, 0.946, 0.072, COLORS["purple"], lw=1.0, mutation=9)

    plotting.save_figure(fig, directory, "Figure3_workflow")
    plotting.write_source_data(
        directory,
        "figure3_layout",
        pd.DataFrame(
            [
                {"stage": 1, "elements": "hourly observations; quality control; daily aggregation; 13 covariates"},
                {"stage": 2, "elements": "Ridge regression; residual extraction; TimesFM forecast; recombination"},
                {"stage": 3, "elements": "weighted score; calibration thresholds; cause classification; association tests"},
            ]
        ),
    )
    print("figure 3 written")


# ---------------------------------------------------------------------------
# Figure 4 - forecast diagnostics
# ---------------------------------------------------------------------------
def build_figure4(config, model: str = "FullFramework") -> None:
    directory = figure_dir("figure4_forecast_diagnostics")
    predictions = pd.read_csv(config.paths.forecasts / f"{model}.csv.gz", parse_dates=["date"])
    daily = load_daily(config)
    metrics_path = config.paths.statistics / "forecast_metrics_by_model.csv"
    metrics = pd.read_csv(metrics_path) if metrics_path.is_file() else pd.DataFrame()

    merged = predictions.merge(
        daily[["water_level", "imputation_flag"]], left_on="date", right_index=True, how="left"
    )
    merged = merged.loc[~merged["imputation_flag"].astype(bool)].copy()
    merged["residual"] = merged["water_level"] - merged["point"]

    lead_one = merged.loc[merged["horizon"] == 1].sort_values("date")
    residual_series = lead_one.set_index("date")["residual"]

    fig = plt.figure(figsize=(6.7, 6.4))
    grid = GridSpec(2, 2, figure=fig, hspace=0.28, wspace=0.22)

    # (a) error by horizon
    ax = fig.add_subplot(grid[0, 0])
    own = metrics.loc[metrics["model"] == model] if not metrics.empty else pd.DataFrame()
    horizons = np.array([1, 7, 14, 30])
    if not own.empty:
        mae = [float(own.loc[own["horizon_days"] == h, "mae"].iloc[0]) for h in horizons]
        rmse = [float(own.loc[own["horizon_days"] == h, "rmse"].iloc[0]) for h in horizons]
    else:
        mae = [merged.loc[merged["horizon"] == h, "residual"].abs().mean() for h in horizons]
        rmse = [
            float(np.sqrt((merged.loc[merged["horizon"] == h, "residual"] ** 2).mean()))
            for h in horizons
        ]
    ax.plot(horizons, mae, marker="o", color=COLORS["blue"], linewidth=1.4, markersize=4,
            label="MAE")
    ax.plot(horizons, rmse, marker="s", color=COLORS["red"], linewidth=1.4, markersize=4,
            label="RMSE")
    ax.set_xticks(horizons)
    ax.set_xlabel("Forecast horizon (days)")
    ax.set_ylabel("Error (m)")
    ax.set_title("Forecast error by horizon", loc="left")
    ax.grid(True)
    ax.legend(loc="upper left")
    plotting.add_panel_label(ax, "(a)")

    # (b) ACF of h=1 residuals
    ax = fig.add_subplot(grid[0, 1])
    values = residual_series.dropna().to_numpy()
    values = values - values.mean()
    n = len(values)
    max_lag = 40
    denominator = float(np.dot(values, values))
    acf = [1.0] + [
        float(np.dot(values[:-lag], values[lag:]) / denominator) for lag in range(1, max_lag + 1)
    ]
    lags = np.arange(0, max_lag + 1)
    ax.bar(lags, acf, width=0.75, color=COLORS["green"])
    ax.axhline(0, color=COLORS["ink"], linewidth=0.7)
    confidence = 1.96 / np.sqrt(n)
    ax.axhline(confidence, color=COLORS["grey"], linestyle="--", linewidth=0.8)
    ax.axhline(-confidence, color=COLORS["grey"], linestyle="--", linewidth=0.8)
    ax.set_xlabel("Lag (days)")
    ax.set_ylabel("Autocorrelation")
    ax.set_title("ACF of 1-day forecast residuals", loc="left")
    ax.set_ylim(-0.4, 1.0)
    ax.grid(True)
    plotting.add_panel_label(ax, "(b)")

    # (c) residual distribution
    ax = fig.add_subplot(grid[1, 0])
    ax.hist(residual_series.dropna(), bins=60, color=COLORS["blue_fill"], edgecolor=COLORS["blue"],
            linewidth=0.5)
    ax.axvline(0, color=COLORS["ink"], linewidth=0.8)
    ax.set_xlabel("1-day forecast residual (m)")
    ax.set_ylabel("Number of days")
    ax.set_title("Residual distribution", loc="left")
    ax.grid(True)
    plotting.add_panel_label(ax, "(c)")

    # (d) representative multi-window hindcast
    ax = fig.add_subplot(grid[1, 1])
    window_start = pd.Timestamp("2018-04-01")
    window_end = pd.Timestamp("2018-04-30")
    subset = merged.loc[
        (merged["date"] >= window_start) & (merged["date"] <= window_end)
    ].copy()
    subset["origin_date"] = subset["date"] - pd.to_timedelta(
        subset["horizon"] - 1, unit="D"
    )
    observed = daily[["water_level"]].loc[window_start:window_end].copy()
    ax.plot(observed.index, observed["water_level"].to_numpy(), color=COLORS["ink"], linewidth=1.4,
            label="Observed", zorder=4)
    ax.scatter(subset["date"], subset["point"], s=5, alpha=0.45, color=COLORS["blue"],
               label="FullFramework point forecasts (h = 1-30)", zorder=3)
    band = subset.loc[subset["horizon"] == 1].sort_values("date")
    if not band.empty:
        lower = band["point"].to_numpy() - band["q10"].to_numpy()
        upper = band["q90"].to_numpy() - band["point"].to_numpy()
        ax.errorbar(
            band["date"],
            band["point"],
            yerr=np.vstack([lower, upper]),
            fmt="none",
            ecolor=COLORS["orange"],
            elinewidth=0.9,
            capsize=2.2,
            label="h = 1 q10-q90 interval (available dates)",
            zorder=5,
        )
        ax.scatter(
            band["date"],
            band["point"],
            s=16,
            facecolor=COLORS["orange"],
            edgecolor="white",
            linewidth=0.45,
            zorder=6,
        )
    ax.set_xlabel("Date")
    ax.set_ylabel("Water-level depth (m)")
    ax.set_title("Representative rolling hindcast (April 2018)", loc="left")
    ax.set_xlim(window_start, window_end)
    ax.grid(True)
    ax.legend(loc="best", fontsize=6.5)
    ax.tick_params(axis="x", rotation=30)
    plotting.add_panel_label(ax, "(d)")

    fig.suptitle("Forecast accuracy and residual diagnostics", fontsize=10.5, fontweight="bold", y=0.98)
    plotting.save_figure(fig, directory, "Figure4_forecast_diagnostics")

    plotting.write_source_data(directory, "figure4_panel_a_error_by_horizon",
                               pd.DataFrame({"horizon_days": horizons, "mae_m": mae, "rmse_m": rmse}))
    plotting.write_source_data(directory, "figure4_panel_b_residual_acf",
                               pd.DataFrame({"lag_days": lags, "autocorrelation": acf,
                                             "confidence_95": confidence}))
    plotting.write_source_data(directory, "figure4_panel_c_residuals",
                               residual_series.rename("residual_m").reset_index())
    forecast_source = subset[
        ["date", "origin_date", "horizon", "point", "q10", "q90"]
    ].rename(columns={"date": "target_date"})
    forecast_source.insert(0, "record_type", "forecast")
    forecast_source["observed_date"] = forecast_source["target_date"]
    forecast_source["observed_water_level"] = forecast_source["target_date"].map(
        observed["water_level"]
    )
    observed_source = observed.reset_index().rename(
        columns={"date": "observed_date", "water_level": "observed_water_level"}
    )
    observed_source.insert(0, "record_type", "observed")
    observed_source["origin_date"] = pd.NaT
    observed_source["target_date"] = pd.NaT
    observed_source["horizon"] = pd.NA
    observed_source["point"] = np.nan
    observed_source["q10"] = np.nan
    observed_source["q90"] = np.nan
    panel_d_source = pd.concat([observed_source, forecast_source], ignore_index=True)
    panel_d_source = panel_d_source[
        [
            "record_type",
            "observed_date",
            "observed_water_level",
            "origin_date",
            "target_date",
            "horizon",
            "point",
            "q10",
            "q90",
        ]
    ].sort_values(["record_type", "observed_date", "target_date"], na_position="last")
    plotting.write_source_data(
        directory, "figure4_panel_d_hindcast_window", panel_d_source
    )
    print("figure 4 written")


# ---------------------------------------------------------------------------
# Figure 5 - covariate response
# ---------------------------------------------------------------------------
def build_figure5(config) -> None:
    from sklearn.linear_model import Ridge

    from src.pipeline import canonical_hourly_path, read_hourly_table

    directory = figure_dir("figure5_covariate_response")
    daily = load_daily(config)
    water = daily["water_level"].to_numpy(dtype="float64")
    valid = np.isfinite(water)
    valid &= ~daily["imputation_flag"].to_numpy()
    scaled, _ = standardise_frame(daily[FEATURE_NAMES], mask=pd.Series(valid, index=daily.index))
    features = scaled.to_numpy(dtype="float64")

    model = Ridge(alpha=1.0).fit(features[valid], water[valid])
    r_squared = float(model.score(features[valid], water[valid]))
    coefficients = pd.DataFrame(
        {
            "feature": FEATURE_NAMES,
            "coefficient": model.coef_,
            "description": [FEATURE_DESCRIPTIONS[name] for name in FEATURE_NAMES],
        }
    )

    months = daily.index.month
    summer = valid & np.isin(months, [6, 7, 8])
    winter = valid & np.isin(months, [12, 1, 2])
    summer_model = Ridge(alpha=1.0).fit(features[summer], water[summer])
    winter_model = Ridge(alpha=1.0).fit(features[winter], water[winter])
    pressure_index = FEATURE_NAMES.index("pressure")
    seasonal = pd.DataFrame(
        {
            "season": ["Summer (Jun-Aug)", "Winter (Dec-Feb)"],
            "pressure_coefficient": [
                float(summer_model.coef_[pressure_index]),
                float(winter_model.coef_[pressure_index]),
            ],
        }
    )

    # Barometric response is a high-frequency signal that is destroyed by daily
    # averaging, so panels (c) and (d) work on the hourly record: the water and
    # pressure series are high-pass filtered with a 25-hour centred rolling mean
    # and the residual cross-response is measured from the filtered series.
    hourly = read_hourly_table(canonical_hourly_path(config))
    hourly_water = hourly["water_level_final"]
    hourly_pressure = hourly["air_pressure_hpa"]
    paired = pd.concat([hourly_water, hourly_pressure], axis=1).dropna()
    water_valid = paired.iloc[:, 0]
    pressure_valid = paired.iloc[:, 1]
    window_hours = 25
    water_highpass = (water_valid - water_valid.rolling(window_hours, center=True, min_periods=1).mean())
    pressure_highpass = (
        pressure_valid - pressure_valid.rolling(window_hours, center=True, min_periods=1).mean()
    )
    correlation = float(np.corrcoef(water_highpass, pressure_highpass)[0, 1])
    highpass_slope = float(np.polyfit(pressure_highpass, water_highpass, 1)[0] * 1000.0)
    detrended = pd.DataFrame(
        {
            "time": water_valid.index,
            "detrended_water_level_m": water_highpass.to_numpy(),
            "detrended_pressure_hpa": pressure_highpass.to_numpy(),
        }
    )

    rolling_window_hours = 24 * 7
    rolling_values = np.full(len(water_highpass), np.nan)
    pressure_array = pressure_highpass.to_numpy()
    water_array = water_highpass.to_numpy()
    stride = 24
    for position in range(rolling_window_hours, len(water_array), stride):
        span = slice(position - rolling_window_hours, position)
        rolling_values[position] = np.polyfit(pressure_array[span], water_array[span], 1)[0] * 1000.0
    rolling = pd.DataFrame(
        {
            "time": water_valid.index,
            "barometric_coefficient_mm_per_hpa": rolling_values,
            "window_hours": rolling_window_hours,
        }
    ).dropna()
    mean_barometric = float(rolling["barometric_coefficient_mm_per_hpa"].mean())
    barometric_efficiency = abs(highpass_slope) / 10.0 / 1.02

    fig = plt.figure(figsize=(6.7, 6.4))
    grid = GridSpec(2, 2, figure=fig, hspace=0.62, wspace=0.34)

    ax = fig.add_subplot(grid[0, 0])
    ordered = coefficients.iloc[np.argsort(np.abs(coefficients["coefficient"]))]
    labels = [name.replace("_", " ") for name in ordered["feature"]]
    colors = [COLORS["orange"] if value >= 0 else COLORS["blue"] for value in ordered["coefficient"]]
    ax.barh(labels, ordered["coefficient"], color=colors, height=0.72)
    ax.axvline(0, color=COLORS["ink"], linewidth=0.7)
    ax.set_xlabel("Standardised Ridge coefficient")
    ax.set_title(f"Covariate contribution ($R^2$ = {r_squared:.3f})", loc="left")
    ax.tick_params(axis="y", labelsize=6.5)
    ax.grid(True, axis="x")
    plotting.add_panel_label(ax, "(a)")

    ax = fig.add_subplot(grid[0, 1])
    ax.bar(seasonal["season"], seasonal["pressure_coefficient"],
           color=[COLORS["orange"], COLORS["blue"]], width=0.55)
    ax.axhline(0, color=COLORS["ink"], linewidth=0.7)
    ax.set_ylabel("Pressure coefficient")
    ax.set_title("Summer-winter pressure coefficient contrast", loc="left")
    ax.grid(True, axis="y")
    for index, value in enumerate(seasonal["pressure_coefficient"]):
        ax.text(index, value, f"{value:.3f}", ha="center",
                va="bottom" if value >= 0 else "top", fontsize=7)
    plotting.add_panel_label(ax, "(b)")

    ax = fig.add_subplot(grid[1, 0])
    sample = detrended.sample(n=min(6000, len(detrended)), random_state=int(config["project"]["random_seed"]))
    ax.scatter(sample["detrended_pressure_hpa"], sample["detrended_water_level_m"],
               s=2, alpha=0.25, color=COLORS["blue"], edgecolors="none")
    slope, intercept = np.polyfit(pressure_highpass, water_highpass, 1)
    x_values = np.linspace(pressure_highpass.min(), pressure_highpass.max(), 50)
    ax.plot(x_values, slope * x_values + intercept, color=COLORS["red"], linewidth=1.3)
    ax.set_xlabel("High-pass pressure (hPa)")
    ax.set_ylabel("High-pass water level (m)")
    ax.set_title(f"Hourly high-pass response ($r$ = {correlation:.3f})", loc="left")
    ax.grid(True)
    plotting.add_panel_label(ax, "(c)")

    ax = fig.add_subplot(grid[1, 1])
    ax.plot(rolling["time"], rolling["barometric_coefficient_mm_per_hpa"],
            color=COLORS["green"], linewidth=0.7)
    ax.axhline(mean_barometric, color=COLORS["red"], linestyle="--", linewidth=1.0,
               label=f"Mean = {mean_barometric:.2f} mm/hPa")
    ax.set_xlabel("Date")
    ax.set_ylabel("Barometric coefficient (mm/hPa)")
    ax.set_title("Seven-day rolling barometric response", loc="left")
    ax.grid(True)
    ax.legend(loc="best", fontsize=6.5)
    ax.tick_params(axis="x", rotation=30)
    plotting.add_panel_label(ax, "(d)")

    fig.suptitle("Covariate contribution and barometric response", fontsize=10.5,
                 fontweight="bold", y=0.98)
    plotting.save_figure(fig, directory, "Figure5_covariate_response")
    plotting.write_source_data(directory, "figure5_panel_a_coefficients", coefficients)
    plotting.write_source_data(directory, "figure5_panel_b_seasonal", seasonal)
    plotting.write_source_data(directory, "figure5_panel_c_detrended", detrended)
    plotting.write_source_data(directory, "figure5_panel_d_rolling_barometric", rolling)
    plotting.write_source_data(
        directory,
        "figure5_diagnostics",
        pd.DataFrame(
            [
                {"metric": "full_period_r_squared", "value": r_squared},
                {"metric": "detrended_water_pressure_correlation", "value": correlation},
                {"metric": "hourly_highpass_slope_mm_per_hpa", "value": highpass_slope},
                {"metric": "mean_barometric_coefficient_mm_per_hpa", "value": mean_barometric},
                {"metric": "barometric_efficiency_dimensionless", "value": barometric_efficiency},
            ]
        ),
    )
    print(
        f"figure 5 written (R2={r_squared:.3f}, r={correlation:.3f}, "
        f"slope={highpass_slope:.2f} mm/hPa, barometric={mean_barometric:.2f} mm/hPa)"
    )


# ---------------------------------------------------------------------------
# Figure 6 - anomaly timeline
# ---------------------------------------------------------------------------
def build_figure6(config, model: str = "FullFramework") -> None:
    """Draw Figure 6 with merged cause-class and earthquake swimlanes."""
    directory = figure_dir("figure6_anomaly_timeline")
    daily = load_daily(config)
    scores = pd.read_csv(
        config.paths.anomaly_scores / f"{model}.csv", parse_dates=["date"]
    )
    segments = pd.read_csv(
        config.paths.anomaly_segments / f"{model}.csv", parse_dates=["start", "end"]
    )
    cenc = pd.read_csv(
        config.paths.statistics / "cenc_catalogue.csv", parse_dates=["origin_time"]
    )

    window_lower, window_upper = map(
        int, config["catalogue"]["windows_days"]["short_term"]
    )
    lookback_days = int(config["catalogue"]["windows_days"]["aggregate"][1])
    earliest_evaluable = scores["date"].min() + pd.Timedelta(days=lookback_days)
    residual_segments = segments.loc[segments["cause"] == "residual"].copy()

    links = []
    for event in cenc.loc[cenc["origin_time"] >= earliest_evaluable].itertuples():
        quake_date = pd.Timestamp(event.origin_time).normalize()
        candidates = residual_segments.copy()
        candidates["lead_days"] = (
            quake_date - candidates["start"].dt.normalize()
        ).dt.days
        candidates = candidates.loc[
            (candidates["lead_days"] > window_lower)
            & (candidates["lead_days"] <= window_upper)
        ].sort_values(["lead_days", "start"])
        if candidates.empty:
            continue
        selected = candidates.iloc[0]
        links.append(
            {
                "earthquake_time": event.origin_time,
                "earthquake_magnitude": float(event.magnitude),
                "earthquake_location": event.location,
                "segment_start": selected["start"],
                "segment_end": selected["end"],
                "lead_days": int(selected["lead_days"]),
                "eligible_segments_in_window": int(len(candidates)),
            }
        )
    link_frame = pd.DataFrame(links)

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(6.7, 5.2),
        sharex=True,
        gridspec_kw={"height_ratios": [0.85, 0.80, 1.40], "hspace": 0.10},
    )
    fig.subplots_adjust(left=0.19, right=0.985, top=0.955, bottom=0.09)

    ax = axes[0]
    ax.plot(daily.index, daily["water_level"], color=COLORS["ink"], linewidth=0.65)
    flagged = daily.loc[daily["imputation_flag"]]
    if not flagged.empty:
        ax.scatter(
            flagged.index,
            flagged["water_level"],
            s=3,
            color=COLORS["grey"],
            label="Context-only imputation",
            zorder=3,
        )
    ax.set_ylabel("Water-level\ndepth (m)")
    ax.invert_yaxis()
    ax.grid(True)
    ax.set_title("Groundwater-level dynamics", loc="left", pad=3)
    if not flagged.empty:
        ax.legend(loc="upper left", fontsize=6.2, frameon=False)
    plotting.add_panel_label(ax, "(a)")

    ax = axes[1]
    ax.plot(scores["date"], scores["score"], color=COLORS["grey"], linewidth=0.5)
    warning = scores.loc[scores["level"] == "WARNING"]
    critical = scores.loc[scores["level"] == "CRITICAL"]
    ax.scatter(
        warning["date"],
        warning["score"],
        s=7,
        color=COLORS["orange"],
        edgecolors="none",
        zorder=3,
        label=f"WARNING ({len(warning)} days)",
    )
    ax.scatter(
        critical["date"],
        critical["score"],
        s=9,
        color=COLORS["red"],
        edgecolors="none",
        zorder=3,
        label=f"CRITICAL ({len(critical)} days)",
    )
    ax.set_ylim(0, max(2.65, float(scores["score"].max()) * 1.05))
    ax.set_ylabel("Weighted\nscore")
    ax.grid(True, axis="y")
    ax.legend(
        loc="upper right",
        ncol=2,
        fontsize=6.3,
        frameon=False,
        handletextpad=0.3,
        columnspacing=0.8,
    )
    ax.set_title("Daily weighted anomaly score", loc="left", pad=3)
    plotting.add_panel_label(ax, "(b)")

    ax = axes[2]
    lanes = [
        ("rainfall", 3.0, COLORS["blue"]),
        ("pressure", 2.0, COLORS["green"]),
        ("residual", 1.0, COLORS["red"]),
        ("borderline", 0.0, COLORS["orange"]),
    ]
    for cause, y_position, color in lanes:
        ax.axhspan(
            y_position - 0.38,
            y_position + 0.38,
            color=color,
            alpha=0.045,
            zorder=0,
        )
        subset = segments.loc[segments["cause"] == cause]
        for segment in subset.itertuples():
            ax.hlines(
                y_position,
                segment.start,
                segment.end,
                color=color,
                linewidth=2.2,
                alpha=0.88,
                zorder=3,
            )
            ax.scatter(
                segment.start,
                y_position + 0.14,
                s=7 + 7 * float(segment.max_score),
                color=color,
                edgecolor="white",
                linewidth=0.4,
                zorder=4,
            )

    earthquake_base = -0.72
    ax.axhspan(-0.88, -0.20, color=COLORS["grey"], alpha=0.055, zorder=0)
    event_y = {}
    magnitude_min = float(cenc["magnitude"].min())
    magnitude_range = max(float(cenc["magnitude"].max()) - magnitude_min, 0.1)
    for event in cenc.itertuples():
        marker_y = -0.80 + 0.48 * (
            float(event.magnitude) - magnitude_min
        ) / magnitude_range
        event_y[pd.Timestamp(event.origin_time)] = marker_y
        event_color = COLORS["red"] if event.magnitude >= 5.0 else COLORS["ink"]
        ax.vlines(
            event.origin_time,
            earthquake_base,
            marker_y,
            color=event_color,
            linewidth=1.15 if event.magnitude >= 5.0 else 0.8,
            zorder=4,
        )
        ax.scatter(
            event.origin_time,
            marker_y,
            s=12 + 12 * (float(event.magnitude) - 4.0),
            color=event_color,
            edgecolor="white",
            linewidth=0.4,
            zorder=5,
        )

    for index, link in link_frame.iterrows():
        target_time = pd.Timestamp(link["earthquake_time"])
        ax.annotate(
            "",
            xy=(target_time, event_y[target_time]),
            xytext=(pd.Timestamp(link["segment_start"]), 1.0),
            arrowprops=dict(
                arrowstyle="-|>",
                mutation_scale=6,
                color=COLORS["muted"],
                linewidth=0.65,
                linestyle=(0, (4, 2)),
                alpha=0.72,
                connectionstyle=(
                    f"arc3,rad={0.08 if index % 2 == 0 else -0.08}"
                ),
            ),
            zorder=2,
        )

    largest_events = cenc.nlargest(3, "magnitude").sort_values("origin_time")
    for event in largest_events.itertuples():
        marker_y = event_y[pd.Timestamp(event.origin_time)]
        ax.annotate(
            f"M{event.magnitude:.1f}",
            (event.origin_time, marker_y),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=6.1,
            color=COLORS["red"],
        )

    for separator in (0.5, 1.5, 2.5):
        ax.axhline(
            separator,
            color=COLORS["grid"],
            linewidth=0.55,
            linestyle=(0, (3, 2)),
            zorder=1,
        )
    ax.set_yticks([3.0, 2.0, 1.0, 0.0, -0.56])
    ax.set_yticklabels(
        [
            "Rainfall-dominated",
            "Pressure-dominated",
            "Residual screening",
            "Borderline",
            "Earthquakes",
        ]
    )
    for tick_label, color in zip(
        ax.get_yticklabels(),
        [COLORS["blue"], COLORS["green"], COLORS["red"], COLORS["orange"], COLORS["ink"]],
    ):
        tick_label.set_color(color)
    ax.set_ylim(-0.92, 3.30)
    ax.set_xlabel("Year")
    ax.set_title(
        r"Cause-classification swimlanes and regional earthquakes "
        r"($M \geq 4.0$, $d \leq 300$ km)",
        loc="left",
        pad=3,
    )
    ax.text(
        0.995,
        0.98,
        f"Line = segment duration; circle = onset (size = maximum score)\n"
        f"Dashed arrows = {window_lower}-{window_upper} d residual-event matches "
        f"({len(link_frame)}/11 evaluable events)",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=6.0,
        color=COLORS["muted"],
        bbox=dict(
            boxstyle="round,pad=0.25",
            facecolor="white",
            edgecolor=COLORS["grid"],
            alpha=0.94,
        ),
        zorder=8,
    )
    plotting.add_panel_label(ax, "(c)")

    axes[-1].set_xlim(pd.Timestamp("2013-01-01"), pd.Timestamp("2026-01-15"))
    axes[-1].xaxis.set_major_locator(mdates.YearLocator(2))
    axes[-1].xaxis.set_minor_locator(mdates.YearLocator(1))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.align_ylabels(axes)

    plotting.save_figure(fig, directory, "Figure6_anomaly_timeline")
    plotting.write_source_data(
        directory,
        "figure6_daily_series",
        daily.reset_index()[["date", "water_level", "imputation_flag"]],
    )
    plotting.write_source_data(
        directory,
        "figure6_anomaly_scores",
        scores[["date", "score", "level"]],
    )
    plotting.write_source_data(directory, "figure6_segments", segments)
    plotting.write_source_data(directory, "figure6_catalogue", cenc)
    plotting.write_source_data(directory, "figure6_short_term_links", link_frame)
    print(f"figure 6 swimlane written ({len(link_frame)} short-term links)")


# ---------------------------------------------------------------------------
# Figure 7 - case examples
# ---------------------------------------------------------------------------
def build_figure7(config, model: str = "FullFramework") -> None:
    directory = figure_dir("figure7_case_examples")
    daily = load_daily(config)
    segments = pd.read_csv(config.paths.anomaly_segments / f"{model}.csv", parse_dates=["start", "end"])
    cenc = pd.read_csv(config.paths.statistics / "cenc_catalogue.csv", parse_dates=["origin_time"])

    cases = [
        ("2017-07-23", 210, 20),
        ("2018-05-28", 210, 20),
        ("2019-05-18", 210, 20),
    ]
    fig, axes = plt.subplots(len(cases), 1, figsize=(6.7, 7.2), sharex=False,
                             gridspec_kw={"hspace": 0.45})
    source_rows = []
    for ax, (date_string, lookback_days, margin_days) in zip(axes, cases):
        event = pd.Timestamp(date_string)
        start = event - pd.Timedelta(days=lookback_days + margin_days)
        end = event + pd.Timedelta(days=margin_days)
        window = daily.loc[start:end]
        event_row = cenc.loc[cenc["origin_time"].dt.date == event.date()]
        magnitude = float(event_row["magnitude"].iloc[0]) if not event_row.empty else float("nan")
        distance = float(event_row["distance_to_well_km"].iloc[0]) if not event_row.empty else float("nan")

        ax.plot(window.index, window["water_level"], color=COLORS["water"], linewidth=0.9,
                label="Water level")
        for segment in segments.itertuples():
            if segment.end < start or segment.start > end:
                continue
            color = plotting.CAUSE_COLORS.get(segment.cause, COLORS["grey"])
            ax.axvspan(segment.start, segment.end, color=color, alpha=0.28)
        ax.axvline(event, color=COLORS["red"], linestyle="--", linewidth=1.1)
        ax.annotate(
            f"{event.date()}  M{magnitude:.1f}  ({distance:.0f} km)",
            xy=(event, window["water_level"].min()),
            xytext=(12, 14),
            textcoords="offset points",
            fontsize=6.8,
            color=COLORS["red"],
        )
        ax.set_ylabel("Depth (m)")
        ax.invert_yaxis()
        ax.grid(True)
        ax.legend(loc="upper left", fontsize=6.2)
        ax.set_title(f"Case {date_string}", loc="left")
        source_rows.append(window.assign(event=date_string).reset_index())
    axes[-1].set_xlabel("Date")
    fig.suptitle("Representative event-window examples", fontsize=10.5,
                 fontweight="bold", y=0.995)
    plotting.save_figure(fig, directory, "Figure7_case_examples")
    plotting.write_source_data(directory, "figure7_case_series", pd.concat(source_rows, ignore_index=True))
    plotting.write_source_data(directory, "figure7_segments", segments)
    print("figure 7 written")


# ---------------------------------------------------------------------------
# Figure 8 - Molchan diagram
# ---------------------------------------------------------------------------
def build_figure8(config, model: str = "FullFramework") -> None:
    directory = figure_dir("figure8_molchan")
    curve = pd.read_csv(config.paths.statistics / f"molchan_curve_{model}.csv")
    metrics = pd.read_csv(config.paths.statistics / f"molchan_metrics_{model}.csv").iloc[0]

    fig, ax = plt.subplots(figsize=(5.4, 5.0))
    ax.plot([0, 1], [1, 0], linestyle="--", color=COLORS["grey"], linewidth=1.2,
            label=r"Random line ($\nu = 1 - \tau$)")
    ax.plot(curve["alarm_time_fraction"], curve["miss_rate"], color=COLORS["blue"],
            linewidth=1.6, marker="o", markersize=2.5,
            label="Residual-segment threshold curve")
    ax.scatter([metrics["operating_tau"]], [metrics["operating_nu"]], marker="*", s=170,
               color=COLORS["red"], zorder=5,
               label=fr"Operating point ($\tau$ = {metrics['operating_tau']:.3f}, "
                     fr"$\nu$ = {metrics['operating_nu']:.3f})")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel(r"Alarm-time fraction, $\tau$")
    ax.set_ylabel(r"Miss rate, $\nu$")
    ax.set_title("Molchan error diagram (365-day alarm windows)", loc="left")
    ax.grid(True)
    ax.legend(loc="upper right", fontsize=6.8)
    ax.text(
        0.97, 0.42,
        f"AUC = {metrics['molchan_auc']:.4f}\n"
        f"Molchan skill = {metrics['molchan_skill']:.2f}\n"
        f"Residual segments = {int(metrics['n_residual_segments'])}\n"
        f"Evaluable events = {int(metrics['n_evaluable_events'])}",
        transform=ax.transAxes, ha="right", va="center", fontsize=7.2,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor=COLORS["grid"]),
    )
    plotting.save_figure(fig, directory, "Figure8_molchan")
    plotting.write_source_data(directory, "figure8_molchan_curve", curve)
    plotting.write_source_data(directory, "figure8_molchan_metrics",
                               pd.DataFrame([metrics.to_dict()]))
    print(f"figure 8 written (AUC={metrics['molchan_auc']:.4f}, skill={metrics['molchan_skill']:.3f})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figures", default="1,2,3,4,5,6,7,8")
    parser.add_argument("--model", default="FullFramework")
    args = parser.parse_args()

    banner("Step 7 - manuscript figures")
    plotting.apply_style()
    config = load_config()
    config.paths.ensure()
    requested = {item.strip() for item in args.figures.split(",") if item.strip()}
    if "1" in requested:
        build_figure1(config)
    if "2" in requested:
        build_figure2(config)
    if "3" in requested:
        build_figure3(config)
    if "4" in requested:
        build_figure4(config, args.model)
    if "5" in requested:
        build_figure5(config)
    if "6" in requested:
        build_figure6(config, args.model)
    if "7" in requested:
        build_figure7(config, args.model)
    if "8" in requested:
        build_figure8(config, args.model)


if __name__ == "__main__":
    main()
