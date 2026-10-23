"""Phase 10d: year-by-year priorities and a forward check of shortlists.

(a) Annual burden ranks. For every campus and window year the four burden
    measures are computed on that year's work orders alone (tail
    expenditure uses the full-window campus threshold, so the threshold is
    fixed across years), ranked within campus, and compared across
    consecutive years: rank persistence (Kendall tau-b), the share of each
    year's campus top-20% that is again in the top 20% the next year, and
    the number of entities whose top-20% membership depends on the burden
    measure in each year.
(b) Forward check. Using only the first h window years of every campus
    (h = 4, and h = 5 where the window allows), the criteria and the
    shortlists are computed as in the main analysis; each shortlist is then
    compared with the entities that rank in the top 20% on each burden
    measure in the following year, which played no part in the selection.

Output: results/p10d_dynamics.json
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p5b_shortlist as SL
from p2_ledgers import comparable_wo, load_clean
from p2b_revised import add_revised_ranks
from utils.io import load_config, results_path, write_json
from utils.ledgers import pct_rank, quantile_type1, revised_ledger_scores

MEAS = ["cost", "labor", "volume", "tail"]


def annual_table(wo: pd.DataFrame, cfg) -> pd.DataFrame:
    rows = []
    for c, gc in wo.groupby("campus", observed=True):
        c = int(c)
        tau = quantile_type1(gc["cost2021"].to_numpy(), cfg["shock_tau_pct"] / 100)
        y0, y1 = cfg["campus_valid_window"][c]
        ents = sorted(gc["entity"].unique())
        for y in range(y0, y1 + 1):
            gy = gc[gc["year"] == y]
            for e in ents:
                g = gy[gy["entity"] == e]
                c21 = g["cost2021"]
                rows.append({
                    "campus": c, "year": y, "entity": e,
                    "cost": float(np.nansum(c21)),
                    "labor": float(np.nansum(g["labor"])),
                    "volume": g["quarter"].nunique() / 4 * len(g),
                    "tail": float(np.nansum((c21 - tau).clip(lower=0))),
                })
    d = pd.DataFrame(rows)
    for m in MEAS:
        d[f"r_{m}"] = d.groupby(["campus", "year"])[m].transform(
            lambda s: pd.Series(pct_rank(s.to_numpy()), index=s.index))
    d["r_mean"] = d[[f"r_{m}" for m in MEAS]].mean(axis=1)
    return d


def top_set(g: pd.DataFrame, col: str, k: int) -> set:
    thr = g[col].nlargest(k).min()
    return set(g.loc[g[col] >= thr, "entity"])


def persistence(d: pd.DataFrame) -> dict:
    taus = {m: [] for m in MEAS + ["mean"]}
    keep = {m: [] for m in MEAS + ["mean"]}
    dep = []
    for c, gc in d.groupby("campus"):
        years = sorted(gc["year"].unique())
        n = gc["entity"].nunique()
        k = int(np.ceil(0.2 * n))
        for y in years:
            g = gc[gc["year"] == y]
            cnt = pd.Series(0, index=g["entity"])
            for m in MEAS:
                cnt[list(top_set(g, f"r_{m}", k))] += 1
            dep.append(int(cnt.isin([1, 2, 3]).sum()))
        for y, y2 in zip(years[:-1], years[1:]):
            a = gc[gc["year"] == y].set_index("entity")
            b = gc[gc["year"] == y2].set_index("entity").loc[a.index]
            for m in MEAS + ["mean"]:
                col = f"r_{m}"
                taus[m].append(kendalltau(a[col], b[col]).statistic)
                ta = top_set(a.reset_index(), col, k)
                tb = top_set(b.reset_index(), col, k)
                keep[m].append(len(ta & tb) / len(ta))
    return {"taub_consecutive_mean": {m: round(float(np.mean(v)), 3) for m, v in taus.items()},
            "taub_consecutive_min": {m: round(float(np.min(v)), 3) for m, v in taus.items()},
            "top20_retained_mean": {m: round(float(np.mean(v)), 3) for m, v in keep.items()},
            "n_campus_year_pairs": len(taus["cost"]),
            "topk_dependent_per_campus_year_mean": round(float(np.mean(dep)), 2),
            "topk_dependent_per_campus_year_total_mean": round(
                float(np.mean(dep)) * d["campus"].nunique(), 1)}


def forward_check(wo: pd.DataFrame, d: pd.DataFrame, cfg, h: int) -> dict:
    camps = [c for c, w in cfg["campus_valid_window"].items()
             if w[1] - w[0] + 1 >= h + 1]
    wq = {int(c): h * 4 for c in camps}
    wy = {int(c): (cfg["campus_valid_window"][c][0],
                   cfg["campus_valid_window"][c][0] + h - 1) for c in camps}
    hist = wo[wo["campus"].isin(camps)
              & (wo["year"] <= wo["campus"].map({c: v[1] for c, v in wy.items()}))]
    sc = add_revised_ranks(revised_ledger_scores(hist, wq, wy,
                                                 tau_pct=cfg["shock_tau_pct"]))
    sel = SL.rule_selections(sc, 0.20)
    K = sel["K"]
    # next-year evaluation sets (pooled, K entities per measure)
    nxt = []
    for c in camps:
        y = wy[c][1] + 1
        nxt.append(d[(d["campus"] == c) & (d["year"] == y)])
    nxt = pd.concat(nxt)
    key = lambda df_, i: (int(df_.at[i, "campus"]), str(df_.at[i, "entity"]))
    eval_sets = {m: {key(nxt, i) for i in nxt.nlargest(K, f"r_{m}").index}
                 for m in MEAS + ["mean"]}
    out = {"h": h, "campuses": len(camps), "N": int(len(sc)), "K": K}
    for r in SL.RULE_ORDER:
        S = {key(sc, i) for i in sel[r]["_sel"]}
        cap = {m: round(len(S & eval_sets[m]) / K, 3) for m in eval_sets}
        cap["worst_burden"] = round(min(cap[m] for m in MEAS), 3)
        out[r] = cap
    # chance level for a random shortlist of size K
    out["chance"] = round(K / len(sc), 3)
    return out


def main() -> None:
    cfg = load_config()
    wo = comparable_wo(load_clean(cfg), "system")
    d = annual_table(wo, cfg)
    out = {"persistence": persistence(d)}
    out["forward_h4"] = forward_check(wo, d, cfg, 4)
    out["forward_h5"] = forward_check(wo, d, cfg, 5)
    write_json(out, results_path("p10d_dynamics.json"))
    import json
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
