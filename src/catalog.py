"""Earthquake catalogue handling for the Beilin well.

Two catalogues are used, and their roles are deliberately different:

``CENC``
    The regional ``M >= 4.0`` catalogue that defines the twelve events of the
    event-level analysis.  Huanan and Nenjiang events are used as-is.

``Songyuan M_S``
    A dense local catalogue used *only* to decide which CENC events belong to
    the same Songyuan sequence.  It never replaces the CENC event list.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd


EARTH_RADIUS_KM = 6371.0


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two coordinates in kilometres."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2.0 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))


def hypocentral_km(event_a: dict, event_b: dict) -> float:
    """Hypocentral distance: horizontal distance plus depth separation."""
    horizontal = haversine_km(
        event_a["latitude"], event_a["longitude"], event_b["latitude"], event_b["longitude"]
    )
    return math.hypot(horizontal, event_b["depth_km"] - event_a["depth_km"])


def dobrovolsky_radius_km(magnitude: float, exponent: float = 0.43) -> float:
    """Dobrovolsky preparation radius ``10 ** (0.43 M)`` in kilometres."""
    return 10.0 ** (exponent * magnitude)


def load_cenc_catalogue(path: Path) -> pd.DataFrame:
    """Load the regional catalogue and add distance and radius columns."""
    frame = pd.read_csv(path, parse_dates=["origin_time"])
    return frame.sort_values("origin_time").reset_index(drop=True)


def add_well_geometry(
    frame: pd.DataFrame,
    well_latitude: float,
    well_longitude: float,
    magnitude_column: str = "magnitude",
) -> pd.DataFrame:
    """Attach distance to the well and the event-specific preparation radius."""
    frame = frame.copy()
    frame["distance_to_well_km"] = [
        haversine_km(well_latitude, well_longitude, row.latitude, row.longitude)
        for row in frame.itertuples()
    ]
    frame["dobrovolsky_radius_km"] = [
        dobrovolsky_radius_km(getattr(row, magnitude_column)) for row in frame.itertuples()
    ]
    frame["within_dobrovolsky_radius"] = (
        frame["distance_to_well_km"] <= frame["dobrovolsky_radius_km"]
    )
    return frame


def parse_songyuan_catalogue(path: Path) -> pd.DataFrame:
    """Parse the fixed-width Songyuan ``M_S`` export.

    The file has no header: ``origin_time latitude longitude magnitude depth
    location`` with timestamps in ``YYYYMMDDHHMMSS`` form.  It is encoded in
    GBK, so it is read with ``cp936``.
    """
    records = []
    with path.open("r", encoding="cp936", errors="replace") as handle:
        for line_number, line in enumerate(handle, start=1):
            fields = line.split()
            if len(fields) < 5:
                continue
            try:
                stamp = pd.to_datetime(fields[0], format="%Y%m%d%H%M%S")
                latitude, longitude, magnitude, depth = (float(value) for value in fields[1:5])
            except (ValueError, IndexError):
                continue
            records.append(
                {
                    "origin_time": stamp,
                    "latitude": latitude,
                    "longitude": longitude,
                    "magnitude_ms": magnitude,
                    "depth_km": depth,
                    "location": " ".join(fields[5:]),
                    "source_line": line_number,
                }
            )
    frame = pd.DataFrame(records).sort_values("origin_time").reset_index(drop=True)
    frame["magnitude_scale"] = "M_S"
    return frame


def catalogue_completeness(frame: pd.DataFrame, column: str = "magnitude_ms", bin_width: float = 0.1) -> dict:
    """Maximum-curvature magnitude of completeness and the Aki b-value."""
    magnitudes = frame[column].to_numpy(dtype="float64")
    binned = np.floor(magnitudes / bin_width) * bin_width
    values, counts = np.unique(np.round(binned, 3), return_counts=True)
    mc = float(values[int(np.argmax(counts))])
    above = magnitudes[magnitudes >= mc]
    if above.size > 1:
        mean_magnitude = float(above.mean())
        denominator = mean_magnitude - (mc - bin_width / 2.0)
        b_value = float(np.log10(np.e) / denominator) if denominator > 0 else float("nan")
    else:
        b_value = float("nan")
    return {
        "max_curvature_mc": mc,
        "max_curvature_bin_count": int(counts.max()),
        "aki_b_value_above_mc": b_value,
        "n_records": int(len(frame)),
        "n_above_mc": int(above.size),
    }
