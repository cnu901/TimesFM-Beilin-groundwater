"""Build the regional CENC catalogue and its sequence grouping.

Outputs
-------
``results/statistics/cenc_catalogue.csv``
    Regional catalogue with distance to the well and Dobrovolsky radii.
``results/statistics/cenc_sequence_map.csv``
    Sequence membership of each CENC event (Gardner-Knopoff windows).
``results/statistics/cenc_sequence_representatives.csv``
    One representative event per sequence.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bootstrap import banner  # noqa: E402

from src.catalog import add_well_geometry, load_cenc_catalogue  # noqa: E402
from src.config import load_config  # noqa: E402
from src.declustering import group_cenc_sequences, sequence_representatives  # noqa: E402


def main() -> None:
    banner("Step 2 - regional earthquake catalogue and sequence grouping")
    config = load_config()
    config.paths.ensure()
    catalogue_cfg = config["catalogue"]

    cenc = load_cenc_catalogue(config.paths.external / "cenc_catalogue_2013_2025.csv")
    cenc = add_well_geometry(
        cenc, catalogue_cfg["well_latitude"], catalogue_cfg["well_longitude"]
    )
    cenc.to_csv(
        config.paths.statistics / "cenc_catalogue.csv", index=False, encoding="utf-8-sig"
    )
    print(f"CENC events: {len(cenc)}")

    sequence_map = group_cenc_sequences(cenc)
    mapping = cenc[["origin_time", "magnitude", "region", "distance_to_well_km"]].merge(
        sequence_map, on="origin_time", how="left"
    )
    mapping.to_csv(
        config.paths.statistics / "cenc_sequence_map.csv",
        index=False,
        encoding="utf-8-sig",
    )

    representatives = sequence_representatives(mapping)
    representatives.to_csv(
        config.paths.statistics / "cenc_sequence_representatives.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print(f"CENC sequences: {len(representatives)}")


if __name__ == "__main__":
    main()
