"""Reader for the Chinese Earthquake Administration (EQT) text format.

EQT files start with a short header (station code, station name, value range)
followed by one ``timestamp value`` record per line.  Timestamps are fixed
width integers whose length encodes the sampling interval:

===========  ==============
width        interval
===========  ==============
8            daily
10           hourly
12           minute
===========  ==============

The sentinel value ``999999`` marks a missing record.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


TIMESTAMP_FORMATS = {
    8: "%Y%m%d",
    10: "%Y%m%d%H",
    12: "%Y%m%d%H%M",
}


def read_eqt_header(path: Path) -> dict[str, str]:
    """Return the station metadata stored in the first two header lines."""
    lines: list[str] = []
    with path.open("r", encoding="gb18030", errors="replace") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            lines.append(stripped)
            if len(lines) == 2:
                break
    header: dict[str, str] = {}
    if lines:
        fields = lines[0].split()
        header["record_count"] = fields[0] if fields else ""
        header["station_item_code"] = " ".join(fields[1:]) if len(fields) > 1 else ""
    if len(lines) > 1:
        header["station_item_name"] = lines[1]
    return header


def read_eqt_series(
    path: Path,
    *,
    missing_sentinel: float = 999999.0,
    encoding: str = "gb18030",
) -> pd.Series:
    """Parse one EQT file into a time-indexed float series.

    Values equal to ``missing_sentinel`` become ``NaN``.  Duplicate timestamps
    keep their first occurrence so that a file can be re-read safely.
    """
    if not path.is_file():
        raise FileNotFoundError(f"EQT file not found: {path}")

    stamps: list[str] = []
    values: list[float] = []
    detected_width: int | None = None
    with path.open("r", encoding=encoding, errors="replace") as handle:
        for line in handle:
            fields = line.split()
            if len(fields) != 2:
                continue
            stamp, raw_value = fields
            fmt = TIMESTAMP_FORMATS.get(len(stamp))
            if fmt is None or not stamp.isdigit():
                continue
            try:
                value = float(raw_value)
            except ValueError:
                continue
            stamps.append(stamp)
            values.append(value)
            detected_width = len(stamp)

    if not stamps:
        raise ValueError(f"No data records parsed from {path}")

    index = pd.to_datetime(pd.Index(stamps), format=TIMESTAMP_FORMATS[detected_width])
    numeric = np.asarray(values, dtype="float64")
    numeric[numeric == missing_sentinel] = np.nan
    series = pd.Series(numeric, index=index, dtype="float64")
    series = series[~series.index.duplicated(keep="first")].sort_index()
    series.name = path.stem
    return series


def resample_hourly(series: pd.Series, start: str, end: str, how: str = "mean") -> pd.Series:
    """Reindex a series onto the complete hourly grid and aggregate to daily."""
    hourly = series.loc[pd.Timestamp(start): pd.Timestamp(end)]
    full = pd.date_range(hourly.index.min().floor("h"), hourly.index.max().ceil("h"), freq="h")
    hourly = hourly.reindex(full)
    if how == "mean":
        return hourly.resample("D").mean()
    if how == "sum":
        return hourly.resample("D").sum(min_count=1)
    raise ValueError(f"Unsupported aggregation: {how}")
