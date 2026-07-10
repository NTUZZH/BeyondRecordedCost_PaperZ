"""Phase 2b: REVISED ledger layer (two-layer framework).

Burden family (related resource-burden representations, constant currency
where monetary): L1 recorded expenditure, L2 labor hours, L3 persistent
work-order volume, L4x excess tail cost.
Distinct objective: L5r budget risk (SD of two-way-demeaned annual cost).

Outputs
  results/p2b_ledgers_entity.csv   revised scores + within-campus ranks +
                                   burden/risk diagnostics per entity
  results/p2b_diagnostics.json     per-ledger reliability: split-half
                                   consistency, leave-one-year-out rank
                                   correlation, size dependence, and the
                                   correlation of L5 variants
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, rankdata, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from p2_ledgers import comparable_wo, load_clean, wage_map
from utils.io import load_config, results_path, write_json
from utils.ledgers import (REVISED_ALL, REVISED_BURDEN, pct_rank,
                           revised_ledger_scores)

RANK_COLS = [f"r_{l}" for l in REVISED_ALL]


def add_revised_ranks(scores: pd.DataFrame) -> pd.DataFrame:
    df = scores.copy()
    for led in REVISED_ALL + ["L5r_shrunk", "L5mae", "L5cv"]:
        df[f"r_{led}"] = np.nan
        for c, idx in df.groupby("campus", observed=True).groups.items():
            df.loc[idx, f"r_{led}"] = pct_rank(df.loc[idx, led].to_numpy())
    burden = [f"r_{l}" for l in REVISED_BURDEN]
    df["n_burden_ledgers"] = df[burden].notna().sum(axis=1)
    df["mean_burden_rank"] = df[burden].mean(axis=1)
    df["median_burden_rank"] = df[burden].median(axis=1)
    df["meii_burden"] = df[burden].std(axis=1, ddof=1)
    df["gap_risk"] = (df["r_L5r"] - df["median_burden_rank"]).abs()
    return df


def add_l6star(df: pd.DataFrame, cfg) -> pd.DataFrame:
    """L6* = non-labor recorded cost (constant currency) + rho x labor hours,
    with rho (a 2025 nominal wage) deflated to the 2021 price year per
    country, so no labor cost is double counted and all terms share one
    price level."""
    out = df.copy()
    wages = wage_map(cfg)
    defl = {int(c): (cfg["wage_deflator_cad"] if c in cfg["campuses_cad"]
                     else cfg["wage_deflator_usd"])
            for c in cfg["campuses_retained"]}
    rho = out["campus"].map(wages) * out["campus"].map(defl)
    out["L6star"] = out["nonlabor_cost2021"] + rho * out["L2"]
    for c, idx in out.groupby("campus", observed=True).groups.items():
        out.loc[idx, "r_L6star"] = pct_rank(out.loc[idx, "L6star"].to_numpy())
    return out


def split_half_consistency(wo: pd.DataFrame, cfg, wq, wy,
                           n_splits: int = 50, seed: int = 20260710) -> dict:
    """Spearman between entity rankings computed on random halves of each
    entity's work orders (per revised ledger, pooled within campus)."""
    rng = np.random.default_rng(seed)
    acc = {l: [] for l in REVISED_ALL}
    for s in range(n_splits):
        wo2 = wo.copy()
        wo2["half"] = rng.integers(0, 2, size=len(wo2))
        halves = []
        for h in (0, 1):
            sc = revised_ledger_scores(wo2[wo2["half"] == h], wq, wy,
                                       tau_pct=cfg["shock_tau_pct"])
            halves.append(sc.set_index(["campus", "entity"]))
        a, b = halves[0].align(halves[1], join="inner", axis=0)
        for l in REVISED_ALL:
            rhos = []
            for c in a.index.get_level_values(0).unique():
                x = a.loc[c, l].to_numpy()
                y = b.loc[c, l].to_numpy()
                ok = ~(np.isnan(x) | np.isnan(y))
                if ok.sum() >= 5:
                    rhos.append(spearmanr(x[ok], y[ok]).statistic)
            if rhos:
                acc[l].append(float(np.mean(rhos)))
    return {l: {"mean": round(float(np.mean(v)), 3),
                "p5": round(float(np.percentile(v, 5)), 3)}
            for l, v in acc.items() if v}


def loyo_rank_consistency(wo: pd.DataFrame, cfg, wq, wy) -> dict:
    """tau-b between each ledger's full-window ranking and its ranking with
    one window year removed (min over years = worst-case)."""
    base = revised_ledger_scores(wo, wq, wy, tau_pct=cfg["shock_tau_pct"])
    base = base.set_index(["campus", "entity"])
    years = sorted(wo["year"].unique())
    out = {l: [] for l in REVISED_ALL}
    for y in years:
        sub = revised_ledger_scores(wo[wo["year"] != y], wq, wy,
                                    tau_pct=cfg["shock_tau_pct"])
        sub = sub.set_index(["campus", "entity"])
        a, b = base.align(sub, join="inner", axis=0)
        for l in REVISED_ALL:
            taus = []
            for c in a.index.get_level_values(0).unique():
                x, z = a.loc[c, l].to_numpy(), b.loc[c, l].to_numpy()
                ok = ~(np.isnan(x) | np.isnan(z))
                if ok.sum() >= 5:
                    taus.append(kendalltau(x[ok], z[ok]).statistic)
            if taus:
                out[l].append(float(np.mean(taus)))
    return {l: {"min": round(min(v), 3), "mean": round(float(np.mean(v)), 3)}
            for l, v in out.items() if v}


def main() -> None:
    cfg = load_config()
    wq = {int(k): (v[1] - v[0] + 1) * 4 for k, v in cfg["campus_valid_window"].items()}
    wy = {int(k): tuple(v) for k, v in cfg["campus_valid_window"].items()}
    wo = comparable_wo(load_clean(cfg), "system")

    scores = revised_ledger_scores(wo, wq, wy, tau_pct=cfg["shock_tau_pct"])
    scores = add_revised_ranks(scores)
    scores = add_l6star(scores, cfg)
    scores.to_csv(results_path("p2b_ledgers_entity.csv"), index=False)
    print(f"p2b entity table: {len(scores)} entities, "
          f"{int(scores['r_L5r'].notna().sum())} with L5r")

    diag = {
        "l5_variant_rank_corr": {},
        "size_dependence": {},
    }
    # L5 variant agreement (pooled within-campus tau-b)
    for va, vb in (("L5r", "L5cv"), ("L5r", "L5mae"), ("L5r", "L5r_shrunk"),
                   ("L5cv", "L5mae")):
        taus = []
        for c, g in scores.groupby("campus", observed=True):
            x, y = g[va].to_numpy(), g[vb].to_numpy()
            ok = ~(np.isnan(x) | np.isnan(y))
            if ok.sum() >= 5:
                taus.append(kendalltau(x[ok], y[ok]).statistic)
        diag["l5_variant_rank_corr"][f"{va}_vs_{vb}"] = round(float(np.mean(taus)), 3)
    # size dependence: pooled within-campus Spearman of each rank vs log n_wos
    logn = np.log(scores["n_wos"].to_numpy())
    for l in REVISED_ALL + ["L5cv"]:
        rhos = []
        for c, g in scores.groupby("campus", observed=True):
            x = g[f"r_{l}"].to_numpy()
            ln = np.log(g["n_wos"].to_numpy())
            ok = ~np.isnan(x)
            if ok.sum() >= 5:
                rhos.append(spearmanr(x[ok], ln[ok]).statistic)
        diag["size_dependence"][l] = round(float(np.mean(rhos)), 3)
    diag["split_half"] = split_half_consistency(wo, cfg, wq, wy)
    diag["loyo_rank_consistency"] = loyo_rank_consistency(wo, cfg, wq, wy)
    write_json(diag, results_path("p2b_diagnostics.json"))
    print("p2b diagnostics written")


if __name__ == "__main__":
    main()
