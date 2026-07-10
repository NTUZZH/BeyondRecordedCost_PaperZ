"""Phase 6b: missing-data robustness for the revised ledger layer.

R13 completeness-threshold sweep: recompute the comparable set and the
  revised headline statistics under stricter cost/labor completeness
  thresholds (baseline 70/60 up to 95/90 and complete-case).
R14 missing-cost treatments for L1/L4x/L5r:
  observed   missing cost contributes nothing (baseline; sums observed)
  adjusted   entity total scaled by 1/coverage (coverage-adjusted sum)
  bound_lo   missing = 0 (identical to observed for sums)
  bound_hi   missing = campus-system stratum median cost
R15 coverage-vs-outcome associations: does recording completeness predict
  rank, MEII_b, or the risk gap?

Output: results/p6b_missing.json
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from p2_ledgers import load_clean
from utils.boot import kendalls_w
from utils.io import ROOT, load_config, results_path, write_json
from utils.ledgers import REVISED_BURDEN, pct_rank, revised_ledger_scores

BURDEN_R = [f"r_{l}" for l in REVISED_BURDEN]


def entity_wo(wo: pd.DataFrame) -> pd.DataFrame:
    w = wo[wo["system"].astype(str).str.strip() != ""].copy()
    w["entity"] = w["system"].astype(str)
    return w


def comparable_under(ent_stats: pd.DataFrame, cost_thr: float,
                     labor_thr: float, cfg) -> pd.DataFrame:
    m = ent_stats
    return m[(m["n_wos"] >= cfg["min_wo_entity_system"])
             & (m["n_years_active"] >= cfg["min_active_years"])
             & (m["cost_cov"] >= cost_thr) & (m["labor_cov"] >= labor_thr)]


def add_burden_ranks(scores: pd.DataFrame) -> pd.DataFrame:
    df = scores.copy()
    for led in REVISED_BURDEN + ["L5r"]:
        df[f"r_{led}"] = np.nan
        for c, idx in df.groupby("campus", observed=True).groups.items():
            df.loc[idx, f"r_{led}"] = pct_rank(df.loc[idx, led].to_numpy())
    df["meii_burden"] = df[BURDEN_R].std(axis=1, ddof=1)
    return df


def headline(df: pd.DataFrame, base: pd.DataFrame | None = None) -> dict:
    out = {
        "n_entities": int(len(df)),
        "W_burden": round(float(kendalls_w(df[BURDEN_R].to_numpy())), 4),
        "meii_b_mean": round(float(df["meii_burden"].mean()), 4),
    }
    if base is not None:
        m = base.merge(df, on=["campus", "entity"], suffixes=("_b", "_v"))
        if len(m) >= 10:
            taus = []
            for col in ("r_L1", "r_L2", "r_L3", "r_L4x"):
                taus.append(kendalltau(m[f"{col}_b"], m[f"{col}_v"]).statistic)
            out["mean_taub_vs_baseline"] = round(float(np.mean(taus)), 3)
            out["n_common"] = int(len(m))
    return out


def main() -> None:
    cfg = load_config()
    wq = {int(k): (v[1] - v[0] + 1) * 4 for k, v in cfg["campus_valid_window"].items()}
    wy = {int(k): tuple(v) for k, v in cfg["campus_valid_window"].items()}
    wo = entity_wo(load_clean(cfg))
    wo = wo[wo["campus"].isin(cfg["campuses_retained"])]

    # per-entity coverage stats (pre-filter universe)
    ent_stats = wo.groupby(["campus", "entity"], observed=True).agg(
        n_wos=("cost", "size"),
        n_years_active=("year", "nunique"),
        cost_cov=("cost", lambda s: s.notna().mean()),
        labor_cov=("labor", lambda s: s.notna().mean()),
    ).reset_index()

    out = {}
    # ---------------- R13 threshold sweep ----------------
    sweeps = [(0.70, 0.60), (0.80, 0.70), (0.90, 0.80), (0.95, 0.90),
              (0.999, 0.999)]
    base_df = None
    r13 = {}
    for ct, lt in sweeps:
        comp = comparable_under(ent_stats, ct, lt, cfg)
        sub = wo.merge(comp[["campus", "entity"]], on=["campus", "entity"])
        sc = revised_ledger_scores(sub, wq, wy, tau_pct=cfg["shock_tau_pct"])
        sc = add_burden_ranks(sc)
        tag = f"cost{int(ct*100)}_labor{int(lt*100)}"
        if base_df is None:
            base_df = sc[["campus", "entity"] + BURDEN_R].copy()
            r13[tag] = headline(sc)
        else:
            r13[tag] = headline(sc, base_df)
    out["R13_threshold_sweep"] = r13

    # ---------------- R14 missing-cost treatments ----------------
    comp = comparable_under(ent_stats, 0.70, 0.60, cfg)
    sub = wo.merge(comp[["campus", "entity"]], on=["campus", "entity"])
    r14 = {}
    variants = {}
    variants["observed"] = sub
    adj = sub.copy()          # coverage-adjusted: scale costs by 1/coverage
    covmap = comp.set_index(["campus", "entity"])["cost_cov"]
    key = list(zip(adj["campus"], adj["entity"]))
    adj["cost2021"] = adj["cost2021"] / covmap.loc[key].to_numpy()
    variants["adjusted"] = adj
    hi = sub.copy()           # bound: missing cost = stratum median
    med = hi.groupby(["campus", "entity"], observed=True)["cost2021"].transform("median")
    hi["cost2021"] = hi["cost2021"].fillna(med)
    variants["bound_hi"] = hi
    base_ranks = None
    for name, v in variants.items():
        sc = add_burden_ranks(
            revised_ledger_scores(v, wq, wy, tau_pct=cfg["shock_tau_pct"]))
        if base_ranks is None:
            base_ranks = sc[["campus", "entity"] + BURDEN_R].copy()
            r14[name] = headline(sc)
        else:
            r14[name] = headline(sc, base_ranks)
    out["R14_missing_treatments"] = r14

    # ---------------- R15 coverage vs outcomes ----------------
    p3d = pd.read_csv(results_path("p3d_entity.csv"))
    m = p3d.merge(ent_stats, on=["campus", "entity"])
    r15 = {}
    for yvar in ("meii_burden", "gap_risk", "mean_burden_rank"):
        r15[f"spearman_costcov_vs_{yvar}"] = round(
            float(spearmanr(m["cost_cov"], m[yvar], nan_policy="omit").statistic), 3)
    r15["spearman_laborcov_vs_meii_b"] = round(
        float(spearmanr(m["labor_cov"], m["meii_burden"],
                        nan_policy="omit").statistic), 3)
    out["R15_coverage_associations"] = r15

    write_json(out, results_path("p6b_missing.json"))
    print("p6b written")
    import json
    print(json.dumps(out, indent=1)[:1500])


if __name__ == "__main__":
    main()
