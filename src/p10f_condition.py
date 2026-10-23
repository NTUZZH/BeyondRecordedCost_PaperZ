"""Phase 10f: condition data in FMUCD and how record-derived burden relates
to them at the one retained campus that records them.

(a) Availability: per university, the number of buildings carrying a
    facility condition index (FCI), current replacement value (CRV),
    deferred maintenance cost (DMC), and gross size.
(b) U01 building-level check: for buildings with condition data, the
    Spearman association of realized annual maintenance burden (recorded
    cost, labor hours, work orders, all in the U01 analysis window) with
    DMC, CRV, FCI, and size, and of cost per unit size with FCI.

Output: results/p10f_condition.json
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pc
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from p2_ledgers import load_clean
from utils.io import RAW_CSV, load_config, results_path, write_json

COLS = {"UniversityID": "campus", "BuildingID": "building", "Size": "size",
        "FCI (facility condition index)": "fci",
        "CRV (current replacement value)": "crv",
        "DMC (deferred maintenance cost)": "dmc"}


def main() -> None:
    cfg = load_config()
    t = pc.read_csv(
        RAW_CSV, read_options=pc.ReadOptions(block_size=1 << 26),
        parse_options=pc.ParseOptions(newlines_in_values=True),
        convert_options=pc.ConvertOptions(
            include_columns=list(COLS),
            column_types={c: pa.string() for c in COLS if c != "UniversityID"}),
    ).to_pandas().rename(columns=COLS)
    # building attributes carry free-text missing markers; coerce to numbers
    for c in ("size", "fci", "crv", "dmc"):
        t[c] = pd.to_numeric(t[c], errors="coerce")
    t.loc[t["building"].str.strip().isin(["", "NA", "nan"]), "building"] = None
    b = (t.dropna(subset=["building"]).groupby(["campus", "building"])
         [["size", "fci", "crv", "dmc"]].first().reset_index())
    avail = b.groupby("campus").agg(buildings=("building", "size"),
                                    fci=("fci", "count"), crv=("crv", "count"),
                                    dmc=("dmc", "count"), size=("size", "count"))
    avail = avail.reindex(sorted(t["campus"].unique()), fill_value=0)
    out = {"availability": {int(c): {k: int(v) for k, v in r.items()}
                            for c, r in avail.iterrows()}}
    out["n_universities_with_fci"] = int((avail["fci"] > 0).sum())
    out["n_universities"] = int(len(avail))
    out["retained_with_fci"] = [int(c) for c in cfg["campuses_retained"]
                                if avail.loc[c, "fci"] > 0]

    # U01 building-level association
    wo = load_clean(cfg)
    w = wo[wo["campus"] == 1]
    ny = cfg["campus_valid_window"][1][1] - cfg["campus_valid_window"][1][0] + 1
    g = w.groupby("building", observed=True).agg(
        cost=("cost2021", "sum"), labor=("labor", "sum"), wos=("year", "size"))
    g = g / ny
    bc = b[b["campus"] == 1].set_index("building")
    m = g.join(bc, how="inner").dropna(subset=["fci", "dmc", "crv", "size"])
    m = m[(m["crv"] > 0) & (m["size"] > 0)]
    res = {"n_buildings": int(len(m))}
    for x in ("cost", "labor", "wos"):
        for y in ("dmc", "crv", "fci", "size"):
            res[f"rho_{x}_{y}"] = round(float(spearmanr(m[x], m[y]).statistic), 3)
    res["rho_costpersize_fci"] = round(float(spearmanr(
        m["cost"] / m["size"], m["fci"]).statistic), 3)
    # deferred maintenance relative to annual realized spending
    res["median_dmc_over_annual_cost"] = round(float((m["dmc"] / m["cost"].clip(lower=1)).median()), 1)
    out["U01_buildings"] = res
    write_json(out, results_path("p10f_condition.json"))
    import json
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
