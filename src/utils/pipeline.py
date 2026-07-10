"""Variant evaluation pipeline shared by Phase 6 robustness checks.

A variant = (WO-level frame, ledger options, campus windows). This module
recomputes entity ledgers, ranks, and the headline stability metrics for any
variant, reusing the exact code paths of Phases 2-3.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.boot import CampusBootstrap, kendalls_w, rank_matrix
from utils.ledgers import PRIMITIVE_LEDGERS, add_ranks, ledger_scores
from p3_stability import RANKS, h2_stat, k_of, topk_sets, RANKINGS

ARCH_RANKS = [f"r_L{j}" for j in range(1, 6)]


def comparability(wo: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Recompute the campus x system comparability filter on a variant frame."""
    g = wo.groupby(["campus", "entity"], observed=True)
    ent = g.agg(
        n_wos=("cost", "size"),
        n_years_active=("year", "nunique"),
        cost_nonmissing=("cost", lambda s: float(s.notna().mean())),
        labor_nonmissing=("labor", lambda s: float(s.notna().mean())),
    ).reset_index()
    ent = ent[ent["entity"].astype(str).str.strip() != ""]
    ok = (
        (ent["n_wos"] >= cfg["min_wo_entity_system"])
        & (ent["n_years_active"] >= cfg["min_active_years"])
        & (ent["cost_nonmissing"] >= cfg["cost_nonmissing_share"])
        & (ent["labor_nonmissing"] >= cfg["labor_nonmissing_share"])
    )
    return ent.loc[ok, ["campus", "entity"]]


def variant_metrics(
    wo: pd.DataFrame,
    cfg: dict,
    windows: dict | None = None,
    refilter: bool = True,
    comparable: pd.DataFrame | None = None,
    tau_pct: float | None = None,
    chron_variant: str = "pf",
    vol_variant: str = "cv_annual",
    shock_stat: str = "sum",
    labor_col: str = "labor",
    with_regret: bool = True,
    with_archetypes: bool = True,
) -> tuple[dict, pd.DataFrame]:
    """Compute headline metrics for a variant. Returns (metrics, scored entities)."""
    windows = windows or {int(k): v for k, v in cfg["campus_valid_window"].items()}
    wq = {c: (v[1] - v[0] + 1) * 4 for c, v in windows.items()}
    wy = {c: v[1] - v[0] + 1 for c, v in windows.items()}

    if comparable is None and refilter:
        comparable = comparability(wo, cfg)
    if comparable is not None:
        wo = wo.merge(comparable, on=["campus", "entity"], how="inner")

    scores = ledger_scores(
        wo, wq, wy,
        tau_pct=tau_pct if tau_pct is not None else cfg["shock_tau_pct"],
        vol_min_years=cfg["vol_min_years"],
        chron_variant=chron_variant, vol_variant=vol_variant,
        shock_stat=shock_stat, labor_col=labor_col,
    )
    scores = add_ranks(scores)
    scores = scores.sort_values(["campus", "entity"]).reset_index(drop=True)

    R = scores[ARCH_RANKS].to_numpy()
    m: dict = {
        "n_entities": int(len(scores)),
        "W_pooled": round(float(kendalls_w(R)), 4),
        "W_per_campus": {
            int(c): round(float(kendalls_w(g[ARCH_RANKS].to_numpy())), 4)
            for c, g in scores.groupby("campus")
        },
        "high_contrast_share": round(float(
            ((np.nanmax(R, axis=1) >= 0.9) & (np.nanmin(R, axis=1) < 0.5)).mean()), 4),
        "h2_stat": round(float(h2_stat(R)), 4),
        "meii_mean": round(float(scores["meii"].mean()), 4),
        "H1_verdict": "supported" if kendalls_w(R) < 0.75 else "not supported",
    }
    m["H2_verdict"] = "supported" if (m["h2_stat"] is not None and m["h2_stat"] >= 0.20) else "not supported"

    if with_regret:
        wr = {}
        for rname, rfun in RANKINGS.items():
            regs = []
            for l_idx, l in enumerate(PRIMITIVE_LEDGERS):
                cbs = []
                for c, g in scores.groupby("campus"):
                    g = g.reset_index(drop=True)
                    k = k_of(len(g), 0.10)
                    burden = g[l].to_numpy(dtype=float)
                    oracle = topk_sets(burden, k)
                    denom = np.nansum(burden[list(oracle)]) if oracle else np.nan
                    if not denom or denom <= 0:
                        continue
                    sel = topk_sets(rfun(g).to_numpy(dtype=float), k)
                    cbs.append(np.nansum(burden[list(sel)]) / denom)
                if cbs:
                    regs.append(1 - float(np.mean(cbs)))
            wr[rname] = max(regs) if regs else np.nan
        m["worst_regret_cost"] = round(wr["recorded_cost"], 4)
        m["worst_regret_consensus"] = round(wr["consensus"], 4)
        m["regret_ratio"] = round(wr["consensus"] / wr["recorded_cost"], 4) if wr["recorded_cost"] > 0 else None
        m["H4_verdict"] = ("supported" if wr["recorded_cost"] > 0
                           and wr["consensus"] <= 0.75 * wr["recorded_cost"] else "not supported")

    if with_archetypes:
        from p4_archetypes import classify
        q3 = float(scores["meii"].quantile(0.75))
        renamed = scores.rename(columns={f"r_{l}": f"r_{l}" for l in PRIMITIVE_LEDGERS})
        m["_archetypes"] = classify(renamed, cfg["theta_high"], cfg["theta_low"], q3)

    return m, scores


def variant_n1_share(wo: pd.DataFrame, cfg: dict, comparable: pd.DataFrame,
                     windows: dict, seed_tag: tuple, B: int = 1000,
                     jobs: int = 8, vol_variant: str = "cv_annual") -> float:
    """N1 ledger-matched significant share for a variant (standard ledger defs,
    or an alternative L5 operationalization via vol_variant)."""
    import zlib
    from joblib import Parallel, delayed

    seed_tag = tuple(
        t if isinstance(t, (int, np.integer)) else zlib.crc32(str(t).encode())
        for t in seed_tag
    )
    wo = wo.merge(comparable, on=["campus", "entity"], how="inner")
    wq = {c: (v[1] - v[0] + 1) * 4 for c, v in windows.items()}
    n_sig = 0
    n_tot = 0
    for c in sorted(wo["campus"].unique()):
        ents = sorted(wo.loc[wo["campus"] == c, "entity"].unique())
        cb = CampusBootstrap(int(c), wo[wo["campus"] == c], ents, wq[int(c)],
                             cfg["vol_min_years"], cfg["shock_tau_pct"],
                             vol_variant=vol_variant)
        obs_rk = rank_matrix(cb.observed_scores())
        bs = 50
        nb = (B + bs - 1) // bs
        batches = Parallel(n_jobs=jobs)(
            delayed(cb.run_batch)((cfg["seed"], *seed_tag, int(c), bi), bs) for bi in range(nb))
        ranks = np.concatenate(batches, axis=0)
        mean_rank = np.nanmean(obs_rk, axis=1)
        dev = ranks - obs_rk[None, :, :]
        recentred = np.clip(mean_rank[None, :, None] + dev, 1e-6, 1.0)
        null_meii = np.nanstd(recentred, axis=2, ddof=1)
        floor95 = np.nanpercentile(null_meii, 95, axis=0)
        obs_meii = np.nanstd(obs_rk, axis=1, ddof=1)
        n_sig += int((obs_meii > floor95).sum())
        n_tot += len(ents)
    return n_sig / n_tot
