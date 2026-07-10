"""Phase 5b: priority-shortlisting analysis (revised decision layer).

Framing: selecting a portfolio shortlist of systems for further attention,
audit, or condition assessment (NOT a monetary budget allocation). Because
every criterion is already a within-campus percentile rank, entities are
comparable across campuses and the shortlist is chosen at the portfolio
level: the top q of all comparable entities, K = ceil(q*N). The main
selection fraction is q=20% (K=17 of 82); q=10% (K=9) is a sensitivity.

Decision rules compared:
  five single-criterion rankings (L1, L2, L3, L4*, L5*),
  burden mean-rank and burden median-rank consensus,
  all-criteria mean rank,
  an exactly optimal minimax-regret shortlist (mixed-integer program).

Evaluation: ordinal top-K regret per criterion l for a selection S,
  regret_l(S) = 1 - |S cap Top_K(l)| / K   (recall shortfall, in [0,1]),
and worst_regret(S) = max_l regret_l(S). Monetary captured shares are
reported only for the additive ledgers (L1, L2, L4*, L6*), averaged
within campus to respect currency.

Also: a stakeholder-weight simplex (robust core / contested frontier),
the Pareto-efficient set, and per-campus top-5 as a granularity check.

Output: results/p5b_shortlist.json, results/p5b_entity.csv
"""

from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils.io import load_config, results_path, write_json

CRITERIA = ["r_L1", "r_L2", "r_L3", "r_L4x", "r_L5r"]
BURDEN = ["r_L1", "r_L2", "r_L3", "r_L4x"]
ADDITIVE = {"L1": "L1", "L2": "L2", "L4x": "L4x", "L6star": "L6star"}
RULE_ORDER = ["single_cost", "single_labor", "single_volume", "single_tail",
              "single_risk", "burden_mean", "burden_median", "all_mean",
              "minimax"]


def pooled_topsets(df: pd.DataFrame, K: int) -> dict:
    """Top_K index set per criterion over the pooled comparable entities."""
    return {col: set(df.nlargest(K, col).index) for col in CRITERIA}


def regret_vec(S: set, tops: dict, K: int) -> dict:
    return {col: 1.0 - len(S & tops[col]) / K for col in CRITERIA}


def minimax_shortlist(df: pd.DataFrame, tops: dict, K: int) -> set:
    """Exactly minimize the worst-case criterion regret by MILP:
    maximize t s.t. |S cap Top_K(l)| >= t for every criterion l, |S| = K,
    x binary. Minimizing max_l (1 - cov_l/K) == maximizing min_l cov_l."""
    idx = list(df.index)
    n = len(idx)
    pos = {e: i for i, e in enumerate(idx)}
    # variables: [x_0..x_{n-1}, t]; maximize t  ->  minimize -t
    c = np.zeros(n + 1)
    c[-1] = -1.0
    rows, lb, ub = [], [], []
    # |S| = K
    r = np.zeros(n + 1)
    r[:n] = 1.0
    rows.append(r)
    lb.append(K)
    ub.append(K)
    # for each criterion: sum_{i in T_l} x_i - t >= 0
    for col in CRITERIA:
        r = np.zeros(n + 1)
        for e in tops[col]:
            r[pos[e]] = 1.0
        r[-1] = -1.0
        rows.append(r)
        lb.append(0.0)
        ub.append(np.inf)
    A = np.vstack(rows)
    cons = LinearConstraint(A, lb, ub)
    integ = np.ones(n + 1)
    integ[-1] = 0                      # t continuous
    bounds = Bounds(np.r_[np.zeros(n), 0], np.r_[np.ones(n), K])
    res = milp(c, constraints=cons, integrality=integ, bounds=bounds)
    x = np.round(res.x[:n]).astype(bool)
    return {idx[i] for i in range(n) if x[i]}


def rule_selections(df: pd.DataFrame, q: float) -> dict:
    K = int(np.ceil(q * len(df)))
    tops = pooled_topsets(df, K)
    sel = {}
    for col, lab in zip(CRITERIA, ["cost", "labor", "volume", "tail", "risk"]):
        sel[f"single_{lab}"] = set(df.nlargest(K, col).index)
    sel["burden_mean"] = set(df.nlargest(K, "mean_burden_rank").index)
    sel["burden_median"] = set(df.nlargest(K, "median_burden_rank").index)
    sel["all_mean"] = set(df.assign(_m=df[CRITERIA].mean(axis=1))
                          .nlargest(K, "_m").index)
    sel["minimax"] = minimax_shortlist(df, tops, K)
    out = {"K": K}
    for name, S in sel.items():
        rv = regret_vec(S, tops, K)
        burden = [rv[c] for c in BURDEN]
        out[name] = {
            "worst_regret": round(max(rv.values()), 4),
            "burden_worst": round(max(burden), 4),
            "risk_regret": round(rv["r_L5r"], 4),
            "mean_regret": {c.replace("r_", ""): round(rv[c], 4)
                            for c in CRITERIA},
            "_sel": S,
        }
    return out


def captured_shares(df: pd.DataFrame, S: set) -> dict:
    """Within-campus averaged captured share for additive ledgers (respects
    currency: each campus normalized by its own total)."""
    out = {}
    for name, led in ADDITIVE.items():
        shares = []
        for c, g in df.groupby("campus", observed=True):
            tot = g[led].sum()
            if tot > 0:
                shares.append(float(g.loc[g.index.isin(S), led].sum() / tot))
        out[name] = round(float(np.mean(shares)), 4)
    return out


def weight_simplex(df: pd.DataFrame, q: float, n_w: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    W = rng.dirichlet(np.ones(len(CRITERIA)), size=n_w)
    K = int(np.ceil(q * len(df)))
    tops = pooled_topsets(df, K)
    freq = pd.Series(0.0, index=df.index)
    worst = []
    X = df[CRITERIA].to_numpy()
    for w in W:
        score = pd.Series(X @ w, index=df.index)
        S = set(score.nlargest(K).index)
        freq.loc[list(S)] += 1
        worst.append(max(regret_vec(S, tops, K).values()))
    freq /= n_w
    scen = {}
    for name, w in (("finance", [.6, .1, .1, .1, .1]),
                    ("workforce", [.1, .6, .1, .1, .1]),
                    ("operations", [.1, .1, .4, .3, .1]),
                    ("risk", [.1, .1, .1, .1, .6])):
        S = set(pd.Series(X @ np.array(w), index=df.index).nlargest(K).index)
        rv = regret_vec(S, tops, K)
        scen[name] = {c.replace("r_", ""): round(rv[c], 3) for c in CRITERIA}
    return {"freq": freq, "worst_mean": float(np.mean(worst)), "scenarios": scen}


def pareto_front(df: pd.DataFrame) -> pd.Series:
    """Non-dominated entities within campus on the five criterion ranks."""
    flag = pd.Series(False, index=df.index)
    for c, g in df.groupby("campus", observed=True):
        R = g[CRITERIA].fillna(-np.inf).to_numpy()
        dom = np.zeros(len(g), bool)
        for i in range(len(g)):
            ge = (R >= R[i]).all(axis=1)
            gt = (R > R[i]).any(axis=1)
            dom[i] = bool((ge & gt).any())
        flag.loc[g.index] = ~dom
    return flag


def per_campus_top5_check(df: pd.DataFrame) -> dict:
    """Granularity sensitivity: per-campus top-5 selection, worst-case regret
    averaged over campuses (coarse, reported only as a robustness note)."""
    out = {}
    for name in RULE_ORDER:
        worst = []
        for c, g in df.groupby("campus", observed=True):
            k = min(5, len(g))
            tl = {col: set(g.nlargest(k, col).index) for col in CRITERIA}
            if name.startswith("single_"):
                col = CRITERIA[["cost", "labor", "volume", "tail",
                                "risk"].index(name.split("_")[1])]
                S = set(g.nlargest(k, col).index)
            elif name == "burden_mean":
                S = set(g.nlargest(k, "mean_burden_rank").index)
            elif name == "burden_median":
                S = set(g.nlargest(k, "median_burden_rank").index)
            elif name == "all_mean":
                S = set(g.assign(_m=g[CRITERIA].mean(axis=1)).nlargest(k, "_m").index)
            else:  # minimax by enumeration (small k)
                pool = sorted(set().union(*tl.values()))
                best, bv = None, np.inf
                for combo in combinations(pool, min(k, len(pool))):
                    Sx = set(combo)
                    v = max(1 - len(Sx & tl[col]) / k for col in CRITERIA)
                    if v < bv - 1e-12:
                        best, bv = Sx, v
                S = best
            worst.append(max(1 - len(S & tl[col]) / k for col in CRITERIA))
        out[name] = round(float(np.mean(worst)), 4)
    return out


def main() -> None:
    cfg = load_config()
    df = pd.read_csv(results_path("p2b_ledgers_entity.csv"))
    out = {}
    ent = df[["campus", "entity"]].copy()

    for q, tag in ((0.20, "q20"), (0.10, "q10")):
        rules = rule_selections(df, q)
        out[tag] = {"K": rules["K"]}
        for name in RULE_ORDER:
            r = {k: v for k, v in rules[name].items() if k != "_sel"}
            out[tag][name] = r
        out[tag]["captured_cost_rule"] = captured_shares(df, rules["single_cost"]["_sel"])
        out[tag]["captured_minimax"] = captured_shares(df, rules["minimax"]["_sel"])
        if tag == "q20":
            sw = weight_simplex(df, q, n_w=1000, seed=cfg["seed"])
            ent["weight_sel_freq"] = sw["freq"]
            out["weights_q20"] = {
                "scenarios": sw["scenarios"],
                "worst_regret_mean_over_weights": round(sw["worst_mean"], 4),
                "share_robust_core": round(float((sw["freq"] >= 0.90).mean()), 4),
                "share_contested": round(
                    float(((sw["freq"] >= 0.20) & (sw["freq"] < 0.80)).mean()), 4),
            }
            ent["in_minimax_q20"] = ent.index.isin(rules["minimax"]["_sel"])
            ent["in_cost_q20"] = ent.index.isin(rules["single_cost"]["_sel"])

    out["per_campus_top5"] = per_campus_top5_check(df)

    # L6* wage sweep: top-5 overlap under the (2021-deflated) wage 20-60
    defl = {int(c): (cfg["wage_deflator_cad"] if c in cfg["campuses_cad"]
                     else cfg["wage_deflator_usd"])
            for c in cfg["campuses_retained"]}
    dfl = df["campus"].map(defl)
    base_l6 = df["L6star"]
    overlaps = []
    for wage in range(cfg["wage_sweep"][0], cfg["wage_sweep"][1] + 1, 5):
        alt = df["nonlabor_cost2021"] + wage * dfl * df["L2"]
        ov = []
        for c, g in df.groupby("campus", observed=True):
            k = min(5, len(g))
            a = set(base_l6.loc[g.index].nlargest(k).index)
            b = set(alt.loc[g.index].nlargest(k).index)
            ov.append(len(a & b) / k)
        overlaps.append(float(np.mean(ov)))
    out["wage_sweep_min_overlap"] = round(min(overlaps), 4)

    ent["pareto_efficient"] = pareto_front(df)
    out["pareto_share"] = round(float(ent["pareto_efficient"].mean()), 4)
    ent["robust_core"] = ent["weight_sel_freq"] >= 0.90
    ent["contested"] = (ent["weight_sel_freq"] >= 0.20) & (ent["weight_sel_freq"] < 0.80)
    ent.to_csv(results_path("p5b_entity.csv"), index=False)
    write_json(out, results_path("p5b_shortlist.json"))
    print("p5b written")
    for tag in ("q20", "q10"):
        print(f"--- {tag} (K={out[tag]['K']}) worst-case regret ---")
        for name in RULE_ORDER:
            print(f"  {name:16s} worst={out[tag][name]['worst_regret']:.3f}"
                  f"  burden={out[tag][name]['burden_worst']:.3f}"
                  f"  risk={out[tag][name]['risk_regret']:.3f}")


if __name__ == "__main__":
    main()
