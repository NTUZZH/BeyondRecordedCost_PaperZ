"""Supplementary decomposition and volatility-ledger robustness analyses.

Produces results/p3b_l5_analysis.json with:
  - MEII decomposition: four-ledger vs five-ledger MEII, coupling to the
    L5 gap, share of significant entities that survive a four-ledger floor.
  - L5-variant robustness: N1 significant share, worst-case regret
    (cost vs consensus), H4 ratio, and cost-set captured L5 burden under the
    quarterly and MAD/median volatility definitions.
  - CoV mechanical component: within-campus Spearman of L5 rank vs log
    work-order count, per campus and pooled, plus the L5-vs-volume tau-b after
    partialling out size (size-detrended volatility residual).
  - Dollar exposure: recorded-cost share held by the top-5-by-
    volatility set per campus.
  - Multiplicity: expected false positives and an FDR-adjusted
    significant count for the per-entity MEII tests.

Usage: python p3b_l5_analysis.py [--jobs 8]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, rankdata, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils.io import ROOT, load_config, results_path, write_json
from utils.pipeline import comparability, variant_metrics, variant_n1_share
from p3_stability import RANKS, k_of, topk_sets

LEDGERS = ["L1", "L2", "L3", "L4", "L5"]


def load_wo(cfg):
    wo = pd.read_parquet(ROOT / "data/interim/wo_clean.parquet").rename(
        columns={"labor_hours": "labor"})
    wo = wo[wo["system"].str.strip() != ""].copy()
    wo["entity"] = wo["system"].astype(str)
    return wo


def meii_decomposition(cfg) -> dict:
    m = pd.read_csv(results_path("p3_meii_entity.csv"))
    five = m[m["n_ledgers"] == 5].copy()
    rk = five[RANKS].to_numpy()
    full = five["meii"].to_numpy()
    four = np.nanstd(rk[:, :4], axis=1, ddof=1)
    l5gap = np.abs(rk[:, 4] - rk[:, :4].mean(axis=1))
    disp4 = four
    out = {
        "n_five_ledger": int(len(five)),
        "corr_full_meii_vs_l5gap": round(float(np.corrcoef(full, l5gap)[0, 1]), 3),
        "corr_full_meii_vs_four_dispersion": round(float(np.corrcoef(full, disp4)[0, 1]), 3),
        "four_ledger_meii_median": round(float(np.median(four)), 3),
        "four_ledger_meii_mean": round(float(four.mean()), 3),
        "median_floor_p95": round(float(five["noise_floor_p95"].median()), 3),
        "five_ledger_meii_mean": round(float(full.mean()), 3),
    }
    # four-ledger significance needs a four-ledger floor: bootstrap already
    # stored per-ledger rank arrays; approximate via the same recentred null
    # but over the four volume ledgers only.
    from p3_stability import load_boot
    boot = load_boot()
    n_sig4 = 0
    n_tot = 0
    for c, z in boot.items():
        ranks = z["ranks"][:, :, :4]          # (B, E, 4)
        obs = z["obs_ranks"][:, :4]
        mean_rank = np.nanmean(obs, axis=1)
        dev = ranks - obs[None, :, :]
        recentred = np.clip(mean_rank[None, :, None] + dev, 1e-6, 1.0)
        null4 = np.nanstd(recentred, axis=2, ddof=1)
        floor4 = np.nanpercentile(null4, 95, axis=0)
        obs4 = np.nanstd(obs, axis=1, ddof=1)
        n_sig4 += int((obs4 > floor4).sum())
        n_tot += ranks.shape[1]
    # the four volume-ledger ranks exist for ALL comparable entities
    # (including the two without an L5 rank), so the denominator is the
    # full comparable set, and is reported explicitly to keep the
    # manuscript text consistent with the share.
    out["n_significant_four_ledger_floor"] = n_sig4
    out["n_denominator_four_ledger_floor"] = n_tot
    out["share_significant_four_ledger_floor"] = round(n_sig4 / n_tot, 4)
    # four-ledger entities (no L5)
    four_only = m[m["n_ledgers"] == 4]
    out["four_only_meii"] = [round(float(x), 3) for x in four_only["meii"]]
    return out


def l5_variant_robustness(cfg, wo, jobs) -> dict:
    comp = pd.read_csv(results_path("p2_ledgers_entity.csv"))[["campus", "entity"]]
    windows = {int(k): v for k, v in cfg["campus_valid_window"].items()}
    out = {}
    for tag, kw in (("annual", {}),
                    ("quarterly", {"vol_variant": "cv_quarterly"}),
                    ("mad_median", {"vol_variant": "mad_median"})):
        m, scores = variant_metrics(wo, cfg, comparable=comp, refilter=False, **kw)
        # cost-set captured L5 burden (top 5 per campus)
        caps = []
        for c, g in scores.groupby("campus"):
            g = g.reset_index(drop=True)
            k = k_of(len(g), cfg["topk_abs"])
            sel = topk_sets(g["L1"].to_numpy(dtype=float), k)
            burden = g["L5"].to_numpy(dtype=float)
            tot = np.nansum(burden)
            caps.append(np.nansum(burden[list(sel)]) / tot if tot > 0 else np.nan)
        sig = variant_n1_share(wo, cfg, comp, windows, ("l5", tag), jobs=jobs,
                               vol_variant=kw.get("vol_variant", "cv_annual"))
        out[tag] = {
            "W_pooled": m["W_pooled"],
            "worst_regret_cost": m["worst_regret_cost"],
            "worst_regret_consensus": m["worst_regret_consensus"],
            "regret_ratio": m["regret_ratio"],
            "H4_verdict": m["H4_verdict"],
            "n1_sig_share": round(sig, 4),
            "cost_captured_L5": round(float(np.nanmean(caps)), 4),
            "n_entities_with_L5": int(scores["L5"].notna().sum()),
        }
    return out


def cov_mechanical(cfg, wo) -> dict:
    """How much of the L5-vs-volume inversion is CoV size geometry."""
    led = pd.read_csv(results_path("p2_ledgers_entity.csv"))
    per_campus = {}
    all_l5, all_logn = [], []
    for c, g in led.groupby("campus"):
        g = g.dropna(subset=["r_L5"])
        if len(g) < 4:
            continue
        rho = spearmanr(g["r_L5"], np.log10(g["n_wos"])).statistic
        per_campus[int(c)] = round(float(rho), 3)
        all_l5.append(g["r_L5"].to_numpy())
        all_logn.append(np.log10(g["n_wos"].to_numpy()))
    pooled = spearmanr(np.concatenate(all_l5), np.concatenate(all_logn)).statistic
    # size-detrended volatility: residual of L5 raw on log-n within campus,
    # then re-rank and correlate with cost rank; how much inversion survives.
    resid_rank, cost_rank = [], []
    for c, g in led.groupby("campus"):
        g = g.dropna(subset=["L5"]).copy()
        if len(g) < 4:
            continue
        x = np.log10(g["n_wos"].to_numpy())
        y = g["L5"].to_numpy()
        b1 = np.polyfit(x, y, 1)
        r = y - np.polyval(b1, x)
        resid_rank.append(rankdata(r) / len(r))
        cost_rank.append(g["r_L1"].to_numpy())
    rr = np.concatenate(resid_rank)
    cr = np.concatenate(cost_rank)
    taub_raw = kendalltau(led.dropna(subset=["r_L5"])["r_L1"],
                          led.dropna(subset=["r_L5"])["r_L5"]).statistic
    taub_detrended = kendalltau(cr, rr).statistic
    return {
        "spearman_l5_vs_logn_per_campus": per_campus,
        "spearman_l5_vs_logn_pooled": round(float(pooled), 3),
        "taub_cost_vs_l5_raw": round(float(taub_raw), 3),
        "taub_cost_vs_l5_size_detrended": round(float(taub_detrended), 3),
    }


def volatility_dollar_exposure(cfg) -> dict:
    """Recorded-cost share held by the top-5-by-volatility set."""
    led = pd.read_csv(results_path("p2_ledgers_entity.csv"))
    shares = []
    for c, g in led.groupby("campus"):
        g = g.reset_index(drop=True)
        k = k_of(len(g), cfg["topk_abs"])
        sel = topk_sets(g["L5"].to_numpy(dtype=float), k)
        cost = g["L1"].to_numpy(dtype=float)
        shares.append(np.nansum(cost[list(sel)]) / np.nansum(cost))
    return {
        "vol_top5_recorded_cost_share_mean": round(float(np.mean(shares)), 4),
        "vol_top5_recorded_cost_share_min": round(float(np.min(shares)), 4),
        "vol_top5_recorded_cost_share_max": round(float(np.max(shares)), 4),
    }


def multiplicity(cfg) -> dict:
    """FDR control over the 82 per-entity MEII significance tests."""
    from statsmodels.stats.multitest import multipletests
    m = pd.read_csv(results_path("p3_meii_entity.csv"))
    from p3_stability import load_boot
    boot = load_boot()
    # per-entity p-value = share of null MEII draws >= observed (ledger-matched)
    pvals, keys = [], []
    for c, z in boot.items():
        ranks = z["ranks"]
        obs = z["obs_ranks"]
        ents = list(z["entities"])
        mean_rank = np.nanmean(obs, axis=1)
        dev = ranks - obs[None, :, :]
        recentred = np.clip(mean_rank[None, :, None] + dev, 1e-6, 1.0)
        null_meii = np.nanstd(recentred, axis=2, ddof=1)     # (B, E)
        obs_meii = np.nanstd(obs, axis=1, ddof=1)
        B = ranks.shape[0]
        for ei, e in enumerate(ents):
            p = (1 + int((null_meii[:, ei] >= obs_meii[ei]).sum())) / (B + 1)
            pvals.append(p)
            keys.append((c, e))
    pvals = np.array(pvals)
    n = len(pvals)
    rej_bh, _, _, _ = multipletests(pvals, alpha=0.05, method="fdr_bh")
    return {
        "n_tests": n,
        "expected_false_positives_at_05": round(0.05 * n, 1),
        "n_significant_uncorrected": int((pvals < 0.05).sum()),
        "n_significant_fdr_bh": int(rej_bh.sum()),
        "share_significant_fdr_bh": round(float(rej_bh.mean()), 4),
    }


def h2_fragility(cfg) -> dict:
    """Raw count behind H2."""
    led = pd.read_csv(results_path("p2_ledgers_entity.csv"))
    R = led[RANKS].to_numpy()
    top_cost = R[:, 0] >= 0.9
    others = R[top_cost][:, 1:]
    fell = (others < 0.9).sum(axis=1) >= 2
    return {
        "n_top_decile_cost": int(top_cost.sum()),
        "n_leavers": int(fell.sum()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=8)
    args = ap.parse_args()
    cfg = load_config()
    wo = load_wo(cfg)
    out = {
        "meii_decomposition": meii_decomposition(cfg),
        "l5_variant_robustness": l5_variant_robustness(cfg, wo, args.jobs),
        "cov_mechanical": cov_mechanical(cfg, wo),
        "volatility_dollar_exposure": volatility_dollar_exposure(cfg),
        "multiplicity": multiplicity(cfg),
        "h2_fragility": h2_fragility(cfg),
    }
    write_json(out, results_path("p3b_l5_analysis.json"))
    import json
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
