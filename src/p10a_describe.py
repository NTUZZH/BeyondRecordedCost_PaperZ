"""Phase 10a: descriptive statistics per campus and criterion scale.

Answers the request for the size of each campus portfolio and for basic
statistics of the criterion values per campus. Everything is computed from
the cleaned, windowed work-order table and the revised criterion table, so
the numbers match the analysis set exactly.

Per campus: buildings with recorded work (where building identifiers
exist), work orders per year, recorded maintenance expenditure per year
(constant 2021 local currency), labor hours per year, share of planned
preventive work orders, and, over the comparable entities, the median and
interquartile range of the five criteria expressed per window year.

Also: the size of the budget-risk criterion relative to each entity's
mean annual cost, and the cost-labor association by campus, including
non-labor cost against labor hours.

Outputs: results/p10a_describe.json, results/p10a_campus_stats.csv,
         results/p10a_criteria_stats.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

sys.path.insert(0, str(Path(__file__).resolve().parent))
from p2_ledgers import comparable_wo, load_clean
from utils.io import load_config, results_path, write_json

CAMPUS_LABEL = {1: "U01", 2: "U02", 4: "U04", 5: "U05", 8: "U08", 9: "U09"}


def q(x, p):
    return float(np.nanpercentile(x, p))


def example(ent: pd.DataFrame, cfg) -> dict:
    """The worked example of the Introduction: at U08, the system that tops
    the recorded-cost ranking and its place in the work-order-volume
    ranking, and the system that replaces it in the volume top set of the
    same size as the campus top-20% shortlist."""
    c = 8
    g = ent[ent["campus"] == c].copy()
    ny = cfg["campus_valid_window"][c][1] - cfg["campus_valid_window"][c][0] + 1
    k = int(np.ceil(0.2 * len(g)))
    g["pos_cost"] = g["L1"].rank(ascending=False, method="min").astype(int)
    g["pos_vol"] = g["L3"].rank(ascending=False, method="min").astype(int)
    top = g.sort_values("L1", ascending=False).iloc[0]
    cost_set = set(g.nlargest(k, "L1")["entity"])
    vol_set = set(g.nlargest(k, "L3")["entity"])
    entering = sorted(vol_set - cost_set)
    leaving = sorted(cost_set - vol_set)
    e_in = g[g["entity"] == entering[0]].iloc[0]
    return {"campus": "U08", "n": int(len(g)), "k": k,
            "top_system": str(top["entity"]),
            "top_cost_per_year_m": float(top["L1"] / ny / 1e6),
            "top_wo_per_year": float(top["n_wos"] / ny),
            "top_pos_vol": int(top["pos_vol"]),
            "leaving": leaving, "entering": entering,
            "entering_cost_per_year_m": float(e_in["L1"] / ny / 1e6),
            "entering_wo_per_year": float(e_in["n_wos"] / ny),
            "entering_pos_cost": int(e_in["pos_cost"])}


def main() -> None:
    cfg = load_config()
    wo = load_clean(cfg)
    ent = pd.read_csv(results_path("p2b_ledgers_entity.csv"))
    out = {"campus": {}}
    rows, crit_rows = [], []
    for c, g in wo.groupby("campus", observed=True):
        c = int(c)
        y0, y1 = cfg["campus_valid_window"][c]
        ny = y1 - y0 + 1
        bld = g["building"].dropna().astype(str)
        bld = bld[bld.str.len() > 0]
        n_bld = int(bld.nunique())
        rec = {
            "campus": CAMPUS_LABEL[c],
            "currency": "CAD" if c in cfg["campuses_cad"] else "USD",
            "window": f"{y0}-{y1}",
            "n_years": ny,
            "n_buildings": n_bld if n_bld > 1 else None,
            "wo_per_year": float(len(g) / ny),
            "cost_per_year_m": float(np.nansum(g["cost2021"]) / ny / 1e6),
            "labor_hours_per_year_k": float(np.nansum(g["labor"]) / ny / 1e3),
            "cost_recorded_share": float(g["cost2021"].notna().mean()),
            "labor_recorded_share": float(g["labor"].notna().mean()),
            "ppm_share": float((g["wo_type"] == "PPM").mean()),
            "upm_share": float((g["wo_type"] == "UPM").mean()),
        }
        e = ent[ent["campus"] == c]
        rec["n_comparable"] = int(len(e))
        rows.append(rec)
        # criteria per window year (budget risk is already per year)
        crit = {
            "cost_k": e["L1"] / ny / 1e3,
            "labor_h": e["L2"] / ny,
            "volume": e["L3"] / ny,
            "tail_k": e["L4x"] / ny / 1e3,
            "risk_k": e["L5r"] / 1e3,
            "risk_rel": e["L5r"] / e["mean_annual_cost"],
        }
        crow = {"campus": CAMPUS_LABEL[c], "n": int(len(e))}
        for k, v in crit.items():
            crow[f"{k}_med"] = q(v, 50)
            crow[f"{k}_q1"] = q(v, 25)
            crow[f"{k}_q3"] = q(v, 75)
        crit_rows.append(crow)
        # cost-labor association within campus, and non-labor vs labor
        out["campus"][CAMPUS_LABEL[c]] = {
            "taub_cost_labor": float(kendalltau(e["L1"], e["L2"]).statistic),
            "taub_nonlabor_labor": float(
                kendalltau(e["nonlabor_cost2021"], e["L2"]).statistic),
            "taub_cost_volume": float(kendalltau(e["L1"], e["L3"]).statistic),
            "taub_labor_volume": float(kendalltau(e["L2"], e["L3"]).statistic),
        }
    cs = pd.DataFrame(rows)
    cr = pd.DataFrame(crit_rows)
    cs.to_csv(results_path("p10a_campus_stats.csv"), index=False)
    cr.to_csv(results_path("p10a_criteria_stats.csv"), index=False)

    # pooled summaries used in the text
    rel = ent["L5r"] / ent["mean_annual_cost"]
    out["risk_rel_median"] = float(rel.median())
    out["risk_rel_q1"] = q(rel, 25)
    out["risk_rel_q3"] = q(rel, 75)
    t = pd.DataFrame(out["campus"]).T
    out["taub_cost_labor_min"] = float(t["taub_cost_labor"].min())
    out["taub_cost_labor_max"] = float(t["taub_cost_labor"].max())
    out["taub_nonlabor_labor_min"] = float(t["taub_nonlabor_labor"].min())
    out["taub_nonlabor_labor_max"] = float(t["taub_nonlabor_labor"].max())
    # pooled (within-campus percentile ranks) association, non-labor vs labor
    nl = ent.groupby("campus")["nonlabor_cost2021"].rank(pct=True)
    out["taub_nonlabor_labor_pooled"] = float(
        kendalltau(nl, ent["r_L2"]).statistic)
    out["taub_cost_labor_pooled"] = float(
        kendalltau(ent["r_L1"], ent["r_L2"]).statistic)
    out["buildings_total"] = int(cs["n_buildings"].fillna(0).sum())
    out["campuses_with_building_ids"] = int(cs["n_buildings"].notna().sum())
    out["wo_per_year_min"] = float(cs["wo_per_year"].min())
    out["wo_per_year_max"] = float(cs["wo_per_year"].max())
    out["ppm_share_min"] = float(cs["ppm_share"].min())
    out["ppm_share_max"] = float(cs["ppm_share"].max())
    out["example"] = example(ent, cfg)
    write_json(out, results_path("p10a_describe.json"))
    pd.set_option("display.width", 200)
    print(cs.to_string())
    print(cr.round(2).to_string())
    print({k: v for k, v in out.items() if k != "campus"})
    print(t.round(3).to_string())


if __name__ == "__main__":
    main()
