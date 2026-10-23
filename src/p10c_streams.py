"""Phase 10c: the framework rerun separately on planned preventive (PPM) and
unplanned corrective (UPM) work orders.

Work-order records are not purely reactive: FMUCD labels each record as
planned preventive maintenance or unplanned work. This stage repeats the
burden-agreement, noise-floor, risk, and shortlist analyses on each stream
alone, with the same comparability minima applied within the stream, and
relates each entity's preventive share to its diagnostics.

Output: results/p10c_streams.json, results/p10c_entity.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import kendalltau, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p5b_shortlist as SL
from p2_ledgers import comparable_wo, load_clean
from p2b_revised import add_revised_ranks
from p3d_revised import topk_change
from utils.boot import RevisedCampusBootstrap, kendalls_w, rank_matrix
from utils.io import load_config, results_path, write_json
from utils.ledgers import REVISED_BURDEN, revised_ledger_scores

BURDEN_R = [f"r_{l}" for l in REVISED_BURDEN]


def floors(wo: pd.DataFrame, cfg, wq, wy, mode: str, B: int, jobs: int,
           seed: int) -> pd.DataFrame:
    """Per-entity 95th-percentile rank-spread floor (recentred null, as in
    p3d_revised.floors_from_boot) for one resampling design."""
    rows = []
    for c, g in wo.groupby("campus", observed=True):
        c = int(c)
        ents = sorted(g["entity"].unique())
        bt = RevisedCampusBootstrap(c, g, ents, wq[c], wy[c],
                                    cfg["shock_tau_pct"], mode=mode)
        nb = 125  # fixed batch: replicates independent of the job count
        keys = [((seed, c, b0, 7 + (mode == "year_block")), min(nb, B - b0))
                for b0 in range(0, B, nb)]
        parts = Parallel(n_jobs=jobs)(delayed(bt.run_batch)(k, n) for k, n in keys)
        ranks = np.concatenate(parts, axis=0)
        obs = rank_matrix(bt.observed_scores())
        dev = ranks - obs[None, :, :]
        base = np.nanmean(obs[:, :4], axis=1)
        rec = np.clip(base[None, :, None] + dev[:, :, :4], 1e-6, 1.0)
        floor = np.nanpercentile(np.nanstd(rec, axis=2, ddof=1), 95, axis=0)
        sd = np.nanstd(obs[:, :4], axis=1, ddof=1)
        for i, e in enumerate(ents):
            rows.append({"campus": c, "entity": e, f"sig_{mode}": bool(sd[i] > floor[i])})
    return pd.DataFrame(rows)


def stream_analysis(wo_s: pd.DataFrame, cfg, wq, wy, B, jobs) -> tuple[dict, pd.DataFrame]:
    # comparability minima applied within the stream
    stats = wo_s.groupby(["campus", "entity"], observed=True).agg(
        n=("year", "size"), yrs=("year", "nunique")).reset_index()
    keep = stats[(stats["n"] >= cfg["min_wo_entity_system"])
                 & (stats["yrs"] >= cfg["min_active_years"])]
    w = wo_s.merge(keep[["campus", "entity"]], on=["campus", "entity"])
    # campuses need at least 5 entities for within-campus ranks to mean much
    cc = keep.groupby("campus").size()
    w = w[w["campus"].isin(cc[cc >= 5].index)]
    sc = add_revised_ranks(revised_ledger_scores(w, wq, wy,
                                                 tau_pct=cfg["shock_tau_pct"]))
    out = {"n_entities": int(len(sc)),
           "n_campuses": int(sc["campus"].nunique()),
           "n_work_orders": int(len(w))}
    out["W_pooled"] = round(float(kendalls_w(sc[BURDEN_R].to_numpy())), 3)
    out["W_by_campus"] = {int(c): round(float(kendalls_w(g[BURDEN_R].to_numpy())), 3)
                          for c, g in sc.groupby("campus")}
    taus = [kendalltau(sc[BURDEN_R[i]], sc[BURDEN_R[j]]).statistic
            for i in range(4) for j in range(i + 1, 4)]
    out["taub_burden_min"] = round(float(min(taus)), 3)
    out["taub_burden_max"] = round(float(max(taus)), 3)
    out["taub_risk_medburden"] = round(float(kendalltau(
        sc["r_L5r"], sc["median_burden_rank"]).statistic), 3)
    out["meii_b_mean"] = round(float(sc["meii_burden"].mean()), 4)
    sc["topk_count"] = topk_change(sc)
    out["topk_dependent"] = int(sc["topk_count"].isin([1, 2, 3]).sum())
    for mode in ("wo", "year_block"):
        f = floors(w, cfg, wq, wy, mode, B, jobs, cfg["seed"])
        sc = sc.merge(f, on=["campus", "entity"])
        out[f"share_flag_{mode}"] = round(float(sc[f"sig_{mode}"].mean()), 4)
        out[f"n_flag_{mode}"] = int(sc[f"sig_{mode}"].sum())
    sel = SL.rule_selections(sc, 0.20)
    out["K"] = sel["K"]
    out["worst_regret"] = {r: sel[r]["worst_regret"] for r in SL.RULE_ORDER}
    out["_sel"] = {r: [(int(sc.at[i, "campus"]), str(sc.at[i, "entity"]))
                       for i in sel[r]["_sel"]] for r in ("single_cost", "minimax")}
    return out, sc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=500)
    ap.add_argument("--jobs", type=int, default=8)
    args = ap.parse_args()
    cfg = load_config()
    wq = {int(k): (v[1] - v[0] + 1) * 4 for k, v in cfg["campus_valid_window"].items()}
    wy = {int(k): tuple(v) for k, v in cfg["campus_valid_window"].items()}
    wo = comparable_wo(load_clean(cfg), "system")
    base = pd.read_csv(results_path("p2b_ledgers_entity.csv"))
    prof = pd.read_csv(results_path("p3d_entity.csv"))
    out = {}
    # entity preventive share and its association with the diagnostics
    share = (wo.assign(ppm=wo["wo_type"] == "PPM")
             .groupby(["campus", "entity"], observed=True)["ppm"].mean()
             .rename("ppm_share").reset_index())
    ent = prof.merge(share, on=["campus", "entity"])
    out["entity_ppm_share_median"] = round(float(ent["ppm_share"].median()), 3)
    out["rho_ppm_share_vs_meii_b"] = round(float(spearmanr(
        ent["ppm_share"], ent["meii_burden"]).statistic), 3)
    out["rho_ppm_share_vs_gap"] = round(float(spearmanr(
        ent["ppm_share"], ent["gap_risk"]).statistic), 3)
    out["ppm_share_by_profile"] = {
        k: round(float(v), 3) for k, v in ent.groupby("profile")["ppm_share"].median().items()}
    ent.to_csv(results_path("p10c_entity.csv"), index=False)

    full_sel = SL.rule_selections(base, 0.20)
    full = {r: {(int(base.at[i, "campus"]), str(base.at[i, "entity"]))
                for i in full_sel[r]["_sel"]} for r in ("single_cost", "minimax")}
    for stream in ("PPM", "UPM"):
        res, sc = stream_analysis(wo[wo["wo_type"] == stream], cfg, wq, wy,
                                  args.B, args.jobs)
        common = set(zip(sc["campus"].astype(int), sc["entity"].astype(str)))
        for r in ("single_cost", "minimax"):
            s = set(map(tuple, res["_sel"][r]))
            f = full[r] & common
            res[f"overlap_{r}_with_all_records"] = round(len(s & f) / max(len(s), 1), 3)
        res.pop("_sel")
        out[stream] = res
        print(stream, res, flush=True)
    write_json(out, results_path("p10c_streams.json"))
    print("p10c written")


if __name__ == "__main__":
    main()
