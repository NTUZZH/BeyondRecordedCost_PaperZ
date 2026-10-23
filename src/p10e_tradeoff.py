"""Phase 10e: the shortlist as a multi-objective selection, and the
contested systems.

(a) Epsilon-constraint trade-off. For every required number r of the
    budget-risk top-K entities (r = 0..K), a mixed-integer program chooses
    the K-entity shortlist that maximizes the smallest coverage of the four
    burden top-K sets. The non-dominated (burden coverage, risk coverage)
    pairs trace the trade-off curve between the two layers; every decision
    rule of the main analysis is placed on the same axes.
(b) Contested systems. For the entities selected under 20%-80% of the
    sampled stakeholder weightings, the criterion whose weight most raises
    the chance of selection (point-biserial correlation between that
    criterion's weight and selection over the weight sample).

Output: results/p10e_tradeoff.json, results/p10e_contested.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p5b_shortlist as SL
from utils.io import load_config, results_path, write_json

CRIT = SL.CRITERIA
BURDEN = SL.BURDEN
NAMES = {"r_L1": "cost", "r_L2": "labor", "r_L3": "volume", "r_L4x": "tail",
         "r_L5r": "risk"}


def eps_constraint(df: pd.DataFrame, K: int, tops: dict, r_min: int):
    idx = list(df.index)
    n = len(idx)
    pos = {e: i for i, e in enumerate(idx)}
    c = np.zeros(n + 1)
    c[-1] = -1.0
    rows, lb, ub = [], [], []
    r = np.zeros(n + 1); r[:n] = 1; rows.append(r); lb.append(K); ub.append(K)
    for col in BURDEN:
        r = np.zeros(n + 1)
        for e in tops[col]:
            r[pos[e]] = 1
        r[-1] = -1
        rows.append(r); lb.append(0); ub.append(np.inf)
    r = np.zeros(n + 1)
    for e in tops["r_L5r"]:
        r[pos[e]] = 1
    rows.append(r); lb.append(r_min); ub.append(np.inf)
    res = milp(c, constraints=LinearConstraint(np.vstack(rows), lb, ub),
               integrality=np.r_[np.ones(n), 0],
               bounds=Bounds(np.r_[np.zeros(n), 0], np.r_[np.ones(n), K]))
    if not res.success:
        return None
    S = {idx[i] for i in range(n) if res.x[i] > 0.5}
    return S


def coverage(S: set, tops: dict, K: int) -> tuple[float, float]:
    b = min(len(S & tops[c]) for c in BURDEN) / K
    rk = len(S & tops["r_L5r"]) / K
    return round(b, 4), round(rk, 4)


def main() -> None:
    cfg = load_config()
    df = pd.read_csv(results_path("p2b_ledgers_entity.csv"))
    K = int(np.ceil(0.20 * len(df)))
    tops = SL.pooled_topsets(df, K)
    pts = []
    for r_min in range(K + 1):
        S = eps_constraint(df, K, tops, r_min)
        if S is None:
            continue
        pts.append(coverage(S, tops, K))
    pts = sorted(set(pts))
    front = [p for p in pts
             if not any((q[0] >= p[0] and q[1] >= p[1] and q != p) for q in pts)]
    sel = SL.rule_selections(df, 0.20)
    rules = {r: coverage(sel[r]["_sel"], tops, K) for r in SL.RULE_ORDER}
    out = {"K": K, "frontier": front, "rules": rules}

    # contested systems: which criterion weight drives selection
    rng = np.random.default_rng(cfg["seed"])
    W = rng.dirichlet(np.ones(len(CRIT)), size=1000)
    X = df[CRIT].to_numpy()
    sel_mat = np.zeros((len(W), len(df)), bool)
    for t, w in enumerate(W):
        sc = X @ w
        sel_mat[t, np.argsort(-sc, kind="stable")[:K]] = True
    freq = sel_mat.mean(axis=0)
    ent = pd.read_csv(results_path("p5b_entity.csv"))
    assert np.allclose(freq, ent["weight_sel_freq"].to_numpy(), atol=1e-9), \
        "weight sample differs from p5b"
    rows = []
    for i in np.flatnonzero((freq >= 0.05) & (freq < 0.95)):
        cors = {NAMES[c]: float(np.corrcoef(W[:, j], sel_mat[:, i])[0, 1])
                for j, c in enumerate(CRIT)}
        rows.append({"campus": int(df.at[i, "campus"]), "entity": df.at[i, "entity"],
                     "freq": round(float(freq[i]), 3),
                     "contested": bool(0.2 <= freq[i] < 0.8),
                     **{f"rank_{NAMES[c]}": round(float(df.at[i, c]), 3) for c in CRIT},
                     "favoring": max(cors, key=cors.get),
                     "disfavoring": min(cors, key=cors.get),
                     "in_minimax": bool(i in sel["minimax"]["_sel"]),
                     "in_cost": bool(i in sel["single_cost"]["_sel"])})
    con = pd.DataFrame(rows).sort_values("freq", ascending=False)
    con.to_csv(results_path("p10e_contested.csv"), index=False)
    out["n_contested"] = int(con["contested"].sum())
    out["n_boundary_5_95"] = int(len(con))
    out["contested_in_minimax"] = int(con.loc[con["contested"], "in_minimax"].sum())
    write_json(out, results_path("p10e_tradeoff.json"))
    import json
    print(json.dumps(out, indent=1))
    print(con.to_string())


if __name__ == "__main__":
    main()
