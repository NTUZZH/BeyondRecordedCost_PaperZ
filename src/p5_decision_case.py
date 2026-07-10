"""Phase 5: budget-allocation decision case.

Scenario: an FM director allocates priority attention to the top-k entities
per campus. Selection rules: recorded cost, each other single ledger, WO
count, mean cost per WO, total burden (L6), consensus.

Outputs: results/p5_overlap.csv           cost set vs alternative sets
         results/p5_captured_burden.csv   selection rule x evaluated ledger
         results/p5_decision_case.json    headline numbers ([[TBD]] keys)
         results/p5_micro_example.json    one-campus worked example
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils.io import load_config, results_path, write_json
from p3_stability import k_of, topk_sets

LEDGERS = ["L1", "L2", "L3", "L4", "L5"]

RULES = {
    "recorded_cost": lambda g: g["L1"].to_numpy(dtype=float),
    "labor_hours": lambda g: g["L2"].to_numpy(dtype=float),
    "chronicity": lambda g: g["L3"].to_numpy(dtype=float),
    "shock": lambda g: g["L4"].to_numpy(dtype=float),
    "volatility": lambda g: g["L5"].to_numpy(dtype=float),
    "wo_count": lambda g: g["n_wos"].to_numpy(dtype=float),
    "mean_cost_per_wo": lambda g: (g["L1"] / g["n_wos"]).to_numpy(dtype=float),
    "total_burden_L6": lambda g: g["L6"].to_numpy(dtype=float),
    "consensus": lambda g: g["mean_rank"].to_numpy(dtype=float),
}


def main() -> None:
    cfg = load_config()
    led = pd.read_csv(results_path("p2_ledgers_entity.csv"))
    led = led.sort_values(["campus", "entity"]).reset_index(drop=True)

    specs = (("abs5", cfg["topk_abs"]), ("pct10", 0.10))
    overlap_rows, cb_rows = [], []
    for spec_name, spec in specs:
        for rname, rfun in RULES.items():
            ov, cbs = [], {l: [] for l in LEDGERS}
            for c, g in led.groupby("campus"):
                g = g.reset_index(drop=True)
                k = k_of(len(g), spec)
                sel = topk_sets(rfun(g), k)
                cost_sel = topk_sets(RULES["recorded_cost"](g), k)
                ov.append(len(sel & cost_sel) / k)
                for l in LEDGERS:
                    burden = g[l].to_numpy(dtype=float)
                    tot = np.nansum(burden)
                    cbs[l].append(np.nansum(burden[list(sel)]) / tot if tot > 0 else np.nan)
            overlap_rows.append({
                "k": spec_name, "rule": rname,
                "overlap_with_cost_set": round(float(np.mean(ov)), 4),
            })
            row = {"k": spec_name, "rule": rname}
            for l in LEDGERS:
                row[f"captured_{l}"] = round(float(np.nanmean(cbs[l])), 4)
            cb_rows.append(row)

    overlap = pd.DataFrame(overlap_rows)
    cb = pd.DataFrame(cb_rows)
    overlap.to_csv(results_path("p5_overlap.csv"), index=False)
    cb.to_csv(results_path("p5_captured_burden.csv"), index=False)

    cost5 = cb[(cb["k"] == "abs5") & (cb["rule"] == "recorded_cost")].iloc[0]
    cons5 = cb[(cb["k"] == "abs5") & (cb["rule"] == "consensus")].iloc[0]
    headline = {
        "cb_cost_set_labor": float(cost5["captured_L2"]),
        "cb_cost_set_chron": float(cost5["captured_L3"]),
        "cb_cost_set_shock": float(cost5["captured_L4"]),
        "cb_cost_set_vol": float(cost5["captured_L5"]),
        "cb_cost_set_cost": float(cost5["captured_L1"]),
        "cb_consensus_labor": float(cons5["captured_L2"]),
        "cb_consensus_chron": float(cons5["captured_L3"]),
        "cb_consensus_cost": float(cons5["captured_L1"]),
        "cb_consensus_vol": float(cons5["captured_L5"]),
        "note": "abs5 = top 5 entities per campus, mean across campuses; "
                "captured shares are of campus-total (comparable set) ledger burden",
    }

    # worked micro-example: lowest Kendall's W among campuses with >= 10
    # comparable entities (a top-5 list needs a meaningful candidate pool)
    import json
    with open(results_path("p3_stability.json")) as fh:
        stab = json.load(fh)
    sizes = led.groupby("campus").size()
    eligible = {k: v for k, v in stab["W_per_campus"].items() if sizes[int(k)] >= 10}
    micro_c = int(min(eligible, key=lambda k: eligible[k]))
    g = led[led["campus"] == micro_c].reset_index(drop=True)
    k = k_of(len(g), cfg["topk_abs"])
    micro = {"campus": f"U{micro_c:02d}", "W": stab["W_per_campus"][str(micro_c)],
             "n_entities": int(len(g)), "top5_by_rule": {}}
    wo = pd.read_parquet(Path(__file__).resolve().parents[1] / "data/interim/wo_clean.parquet")
    desc_map = (
        wo[wo["system"].str.strip() != ""]
        .groupby("system")["system_desc"].agg(lambda x: x.mode().iloc[0]).to_dict()
    )
    for rname in ("recorded_cost", "labor_hours", "chronicity", "shock", "volatility", "consensus"):
        sel = topk_sets(RULES[rname](g), k)
        order = sorted(sel, key=lambda i: -RULES[rname](g)[i])
        micro["top5_by_rule"][rname] = [
            {"system": g.loc[i, "entity"], "desc": desc_map.get(g.loc[i, "entity"], "")}
            for i in order
        ]
    write_json(headline, results_path("p5_decision_case.json"))
    write_json(micro, results_path("p5_micro_example.json"))
    print("Phase 5 complete.")
    print(f"  cost top-5 captures: labor {headline['cb_cost_set_labor']:.3f}, "
          f"chron {headline['cb_cost_set_chron']:.3f}, vol {headline['cb_cost_set_vol']:.3f}")
    print(f"  consensus top-5 captures: labor {headline['cb_consensus_labor']:.3f}, "
          f"chron {headline['cb_consensus_chron']:.3f}, cost {headline['cb_consensus_cost']:.3f}")
    print(f"  micro-example campus: U{micro_c:02d}")


if __name__ == "__main__":
    main()
