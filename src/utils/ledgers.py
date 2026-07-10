"""Ledger definitions as reusable functions.

All computations operate on a cleaned WO-level DataFrame with columns:
  campus, entity (string key), cost, cost2021, labor, quarter, year
plus per-campus metadata (window quarters). Variants for the robustness suite
are exposed as options so Phase 6 reuses the same code paths.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import rankdata

PRIMITIVE_LEDGERS = ["L1", "L2", "L3", "L4", "L5"]


def quantile_type1(values: np.ndarray, q: float, weights: np.ndarray | None = None) -> float:
    """Weighted type-1 quantile: smallest sorted value whose cumulative weight
    reaches q * total. Identical formula in the observed pipeline and in the
    bootstrap machinery (utils/boot.py) so replicate worlds are exactly
    comparable to the observed world."""
    v = np.asarray(values, dtype=float)
    ok = ~np.isnan(v)
    v = v[ok]
    w = np.ones(len(v)) if weights is None else np.asarray(weights, dtype=float)[ok]
    order = np.argsort(v, kind="stable")
    cw = np.cumsum(w[order])
    total = cw[-1]
    idx = np.searchsorted(cw, q * total, side="left")
    return float(v[order][min(idx, len(cw) - 1)])


def campus_tau(wo: pd.DataFrame, tau_pct: float) -> dict:
    """Shock threshold per campus: percentile of pooled per-WO cost (non-missing)."""
    return {
        c: quantile_type1(g["cost"].to_numpy(), tau_pct / 100.0)
        for c, g in wo.groupby("campus", observed=True)
    }


def ledger_scores(
    wo: pd.DataFrame,
    window_quarters: dict,
    window_years: dict,
    tau_pct: float = 95.0,
    vol_min_years: int = 5,
    chron_variant: str = "pf",       # pf | count | active_quarters | median_nonzero
    vol_variant: str = "cv_annual",  # cv_annual | mad_median | cv_quarterly
    shock_stat: str = "sum",         # sum | count
    labor_col: str = "labor",        # labor | labor_cost (R9)
) -> pd.DataFrame:
    """Raw scores L1-L5 (plus L6 ingredients) per (campus, entity)."""
    tau = campus_tau(wo, tau_pct)

    def per_entity(c, g: pd.DataFrame) -> dict:
        cost = g["cost"]
        out = {}
        out["L1"] = float(cost.sum())
        out["L1_2021"] = float(g["cost2021"].sum())
        out["L2"] = float(g[labor_col].sum())
        # L3 chronicity
        tq = window_quarters[int(c)]
        qcounts = g.groupby("quarter", observed=True).size()
        persistence = len(qcounts) / tq
        if chron_variant == "pf":
            out["L3"] = persistence * len(g)
        elif chron_variant == "count":
            out["L3"] = float(len(g))
        elif chron_variant == "active_quarters":
            out["L3"] = float(len(qcounts))
        elif chron_variant == "median_nonzero":
            out["L3"] = float(qcounts.median()) if len(qcounts) else 0.0
        out["persistence"] = persistence
        # L4 shock
        over = g.loc[cost > tau[c], "cost"]
        out["L4"] = float(over.sum()) if shock_stat == "sum" else float(len(over))
        # L5 volatility
        if vol_variant == "cv_quarterly":
            series = g.groupby("quarter", observed=True)["cost"].sum()
            min_obs = vol_min_years * 4
        else:
            series = g.groupby("year", observed=True)["cost"].sum()
            min_obs = vol_min_years
        if len(series) >= min_obs:
            if vol_variant == "mad_median":
                med = float(series.median())
                mad = float((series - med).abs().median())
                out["L5"] = mad / med if med > 0 else np.nan
            else:
                m = float(series.mean())
                out["L5"] = float(series.std(ddof=1)) / m if m > 0 else np.nan
        else:
            out["L5"] = np.nan
        out["n_wos"] = len(g)
        out["n_years_active"] = int(g["year"].nunique())
        return out

    rows = []
    for (c, e), g in wo.groupby(["campus", "entity"], observed=True):
        rows.append({"campus": c, "entity": e, **per_entity(int(c), g)})
    scores = pd.DataFrame(rows)
    scores["tau_campus"] = scores["campus"].map({k: v for k, v in tau.items()})
    return scores


def pct_rank(x: np.ndarray) -> np.ndarray:
    """Within-group percentile rank in (0,1]: average rank / n non-missing."""
    x = np.asarray(x, dtype=float)
    out = np.full(x.shape, np.nan)
    ok = ~np.isnan(x)
    if ok.sum() > 0:
        out[ok] = rankdata(x[ok], method="average") / ok.sum()
    return out


def add_ranks(scores: pd.DataFrame, ledgers: list[str] = PRIMITIVE_LEDGERS) -> pd.DataFrame:
    df = scores.copy()
    for led in ledgers:
        df[f"r_{led}"] = np.nan
        for c, idx in df.groupby("campus", observed=True).groups.items():
            df.loc[idx, f"r_{led}"] = pct_rank(df.loc[idx, led].to_numpy())
    rank_cols = [f"r_{l}" for l in ledgers]
    df["n_ledgers"] = df[rank_cols].notna().sum(axis=1)
    df["mean_rank"] = df[rank_cols].mean(axis=1)
    df["meii"] = df[rank_cols].std(axis=1, ddof=1)
    df["meii_iqr"] = df[rank_cols].quantile(0.75, axis=1) - df[rank_cols].quantile(0.25, axis=1)
    return df


def add_l6(scores: pd.DataFrame, wage_by_campus: dict) -> pd.DataFrame:
    df = scores.copy()
    rho = df["campus"].map(wage_by_campus)
    df["L6"] = df["L1_2021"] + rho * df["L2"]
    return df
