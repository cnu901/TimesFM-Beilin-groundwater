"""Sequence grouping of the regional CENC event list.

For the sequence-level robustness check, CENC events are grouped with
forward-chained Gardner--Knopoff time and distance windows.  This is a simple
transparent grouping of the twelve regional events, not a declustering of a
dense auxiliary catalogue.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .catalog import hypocentral_km


def gardner_knopoff_window_days(magnitude: float) -> float:
    """Gardner--Knopoff aftershock time window for a magnitude."""
    return 10.0 ** (0.5409 * magnitude - 0.547)


def gardner_knopoff_radius_km_value(magnitude: float) -> float:
    """Gardner--Knopoff aftershock distance window for a magnitude."""
    return 10.0 ** (0.1238 * magnitude + 0.983)


def group_cenc_sequences(cenc: pd.DataFrame) -> pd.DataFrame:
    """Assign CENC events to sequences using forward-chained GK windows.

    Events are sorted chronologically.  An event joins an existing sequence
    when it is within the Gardner--Knopoff time and distance windows of that
    sequence's most recent member, with the window evaluated at the largest
    magnitude currently in the sequence.  This grows the window as larger
    events are linked, which keeps the 2017-2019 Songyuan activity in one
    sequence while leaving the 2022 event separate.
    """
    events = cenc.sort_values("origin_time").reset_index(drop=True)
    sequence_of = np.full(len(events), -1, dtype=int)
    sequences: list[list[int]] = []
    for index in range(len(events)):
        placed = False
        for sequence_index, members in enumerate(sequences):
            largest = max(events.iloc[member]["magnitude"] for member in members)
            last = members[-1]
            delta_days = (
                events.iloc[index]["origin_time"] - events.iloc[last]["origin_time"]
            ).days
            distance = hypocentral_km(
                events.iloc[last].to_dict(), events.iloc[index].to_dict()
            )
            if (
                delta_days <= gardner_knopoff_window_days(largest)
                and distance <= gardner_knopoff_radius_km_value(largest)
            ):
                members.append(index)
                sequence_of[index] = sequence_index
                placed = True
                break
        if not placed:
            sequence_of[index] = len(sequences)
            sequences.append([index])
    sequence_ids = [f"SEQ{value + 1}" for value in sequence_of]
    return pd.DataFrame({"origin_time": events["origin_time"], "sequence_id": sequence_ids})


def sequence_representatives(mapping: pd.DataFrame) -> pd.DataFrame:
    """Collapse each sequence to its largest event for the sequence-level test."""
    rows = []
    for sequence_id, group in mapping.groupby("sequence_id", sort=False):
        representative = group.iloc[[int(np.argmax(group["magnitude"].to_numpy()))]]
        rows.append(representative.iloc[0])
    return pd.DataFrame(rows).sort_values("origin_time").reset_index(drop=True)
