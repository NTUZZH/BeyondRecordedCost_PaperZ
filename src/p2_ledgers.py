"""Phase 2: ledger computation.

Computes L1-L5 raw scores and within-campus percentile ranks for the
comparable entities at both entity levels, plus the derived L6 ledger.

Outputs: results/p2_ledgers_entity.csv        campus x system (primary)
         results/p2_ledgers_bldg_entity.csv   campus x building x system
         results/p2_ledger_summary.json
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils.io import ROOT, load_config, results_path, write_json
from utils.ledgers import PRIMITIVE_LEDGERS, add_l6, add_ranks, ledger_scores


def load_clean(cfg) -> pd.DataFrame:
    wo = pd.read_parquet(ROOT / "data/interim/wo_clean.parquet")
    wo = wo.rename(columns={"labor_hours": "labor"})
    return wo


def comparable_wo(wo: pd.DataFrame, level: str) -> pd.DataFrame:
    if level == "system":
        ent = pd.read_parquet(ROOT / "data/panel/entity_system.parquet")
        keys = ["campus", "system"]
    else:
        ent = pd.read_parquet(ROOT / "data/panel/entity_bldg_system.parquet")
        keys = ["campus", "building", "system"]
    comp = ent.loc[ent["comparable"], keys]
    out = wo.merge(comp, on=keys, how="inner")
    out["entity"] = out[keys[1:]].astype(str).agg("|".join, axis=1)
    return out


def wage_map(cfg) -> dict:
    return {
        int(c): (cfg["wage_rate_cad_per_hour"] if c in cfg["campuses_cad"] else cfg["wage_rate_usd_per_hour"])
        for c in cfg["campuses_retained"]
    }


def main() -> None:
    cfg = load_config()
    wq = {int(k): (v[1] - v[0] + 1) * 4 for k, v in cfg["campus_valid_window"].items()}
    wy = {int(k): v[1] - v[0] + 1 for k, v in cfg["campus_valid_window"].items()}
    wo = load_clean(cfg)

    summary = {}
    for level, out_name in (("system", "p2_ledgers_entity.csv"), ("bldg", "p2_ledgers_bldg_entity.csv")):
        w = comparable_wo(wo, level)
        scores = ledger_scores(
            w, wq, wy,
            tau_pct=cfg["shock_tau_pct"], vol_min_years=cfg["vol_min_years"],
        )
        scores = add_ranks(scores)
        scores = add_l6(scores, wage_map(cfg))
        scores.to_csv(results_path(out_name), index=False)

        rank_cols = [f"r_{l}" for l in PRIMITIVE_LEDGERS]
        tau_by_campus = scores.groupby("campus")["tau_campus"].first()
        summary[level] = {
            "n_entities": int(len(scores)),
            "n_entities_full_five": int((scores["n_ledgers"] == 5).sum()),
            "n_entities_four": int((scores["n_ledgers"] == 4).sum()),
            "n_entities_lt_four": int((scores["n_ledgers"] < 4).sum()),
            "tau_by_campus": {int(k): round(float(v), 2) for k, v in tau_by_campus.items()},
            "raw_score_medians": {l: float(scores[l].median()) for l in PRIMITIVE_LEDGERS},
            "meii_median": float(scores["meii"].median()),
        }
        # smoke test: pooled pairwise Kendall tau-b on ranks
        from scipy.stats import kendalltau
        taub = {}
        for i, a in enumerate(PRIMITIVE_LEDGERS):
            for b in PRIMITIVE_LEDGERS[i + 1:]:
                m = scores[[f"r_{a}", f"r_{b}"]].dropna()
                taub[f"{a}-{b}"] = round(float(kendalltau(m[f"r_{a}"], m[f"r_{b}"]).statistic), 3)
        summary[level]["pooled_taub_smoke"] = taub

    write_json(summary, results_path("p2_ledger_summary.json"))
    print("Phase 2 complete.")
    for level in summary:
        s = summary[level]
        print(f"[{level}] n={s['n_entities']} five-ledger={s['n_entities_full_five']} "
              f"four={s['n_entities_four']} lt4={s['n_entities_lt_four']}")
        print(f"  tau-b smoke: {s['pooled_taub_smoke']}")


if __name__ == "__main__":
    main()
