"""Shared I/O helpers for the Paper Z (CMERF) pipeline.

All scripts read parameters from config.yaml (single source of truth) and
write machine-readable outputs to results/. The raw FMUCD table is read-only.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
RAW_CSV = ROOT / "data" / "raw" / "FMUCD.csv"

# Columns of FMUCD.csv relevant to Paper Z (weather columns and building
# condition metrics are deliberately not loaded).
USECOLS = [
    "UniversityID",
    "Country",
    "State/Province",
    "BuildingID",
    "SystemCode",
    "SystemDescription",
    "SubsystemCode",
    "SubsystemDescription",
    "WOID",
    "WOPriority",
    "WOStartDate",
    "WOEndDate",
    "PPM/UPM",
    "LaborCost",
    "MaterialCost",
    "OtherCost",
    "TotalCost",
    "LaborHours",
]

# Categorical columns are read as plain strings first (mixed int/str content in
# some columns breaks per-chunk categorical unioning), then converted post-load.
CATEGORICAL = [
    "Country",
    "State/Province",
    "BuildingID",
    "SystemCode",
    "SystemDescription",
    "SubsystemCode",
    "SubsystemDescription",
    "WOPriority",
    "PPM/UPM",
]

DTYPES = {
    "UniversityID": "int16",
    "WOID": "string",
    "LaborCost": "float64",
    "MaterialCost": "float64",
    "OtherCost": "float64",
    "TotalCost": "float64",
    "LaborHours": "float64",
    **{c: "string" for c in CATEGORICAL},
}


def load_config() -> dict:
    with open(ROOT / "config.yaml") as fh:
        return yaml.safe_load(fh)


def load_raw(usecols: list[str] | None = None, parse_dates: bool = True) -> pd.DataFrame:
    """Load the raw FMUCD table with memory-efficient dtypes."""
    cols = usecols if usecols is not None else USECOLS
    dtypes = {c: DTYPES[c] for c in cols if c in DTYPES}
    # pyarrow.csv directly: multi-threaded, handles quoted cells that span
    # multiple lines (WODescription contains embedded newlines), and avoids a
    # chunk-concatenation bug the pandas C engine hits on mixed-type columns.
    import pyarrow as pa
    import pyarrow.csv as pacsv

    string_cols = [c for c in cols if c not in ("UniversityID",) and c not in
                   ("LaborCost", "MaterialCost", "OtherCost", "TotalCost", "LaborHours")]
    column_types = {c: pa.string() for c in string_cols}
    for c in ("LaborCost", "MaterialCost", "OtherCost", "TotalCost", "LaborHours"):
        if c in cols:
            column_types[c] = pa.float64()
    if "UniversityID" in cols:
        column_types["UniversityID"] = pa.int16()
    tbl = pacsv.read_csv(
        RAW_CSV,
        parse_options=pacsv.ParseOptions(newlines_in_values=True),
        convert_options=pacsv.ConvertOptions(include_columns=cols, column_types=column_types),
    )
    df = tbl.to_pandas()
    for c in CATEGORICAL:
        if c in df.columns:
            df[c] = df[c].astype("category")
    if parse_dates:
        for c in ("WOStartDate", "WOEndDate"):
            if c in df.columns:
                df[c] = pd.to_datetime(df[c], format="%Y-%m-%d %H:%M:%S", errors="coerce")
    return df


def write_json(obj, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def _default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (pd.Timestamp,)):
            return str(o)
        raise TypeError(f"not JSON serializable: {type(o)}")

    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2, default=_default)


def results_path(name: str) -> Path:
    p = ROOT / "results" / name
    p.parent.mkdir(parents=True, exist_ok=True)
    return p
