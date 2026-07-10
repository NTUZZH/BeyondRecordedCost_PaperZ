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


REVISED_BURDEN = ["L1", "L2", "L3", "L4x"]
REVISED_ALL = REVISED_BURDEN + ["L5r"]


def _two_way_demean(M: np.ndarray) -> np.ndarray:
    """Residuals of a balanced two-way layout (entity + year effects)."""
    row = M.mean(axis=1, keepdims=True)
    col = M.mean(axis=0, keepdims=True)
    return M - row - col + M.mean()


def annual_panel(wo: pd.DataFrame, campus: int, entities: list[str],
                 window_years: tuple[int, int],
                 cost_col: str = "cost2021") -> np.ndarray:
    """(E, Y) constant-currency annual cost matrix over the FULL campus
    window; years with no recorded work for an entity are true zeros."""
    y0, y1 = window_years
    years = list(range(int(y0), int(y1) + 1))
    E = len(entities)
    M = np.zeros((E, len(years)))
    sub = wo[wo["campus"] == campus]
    grp = sub.groupby(["entity", "year"], observed=True)[cost_col].sum()
    eidx = {e: i for i, e in enumerate(entities)}
    yidx = {y: j for j, y in enumerate(years)}
    for (e, y), v in grp.items():
        if e in eidx and y in yidx:
            M[eidx[e], yidx[y]] = float(np.nan_to_num(v, nan=0.0))
    return M


def revised_ledger_scores(
    wo: pd.DataFrame,
    window_quarters: dict,
    window_years: dict,
    tau_pct: float = 95.0,
    shrink_kappa: float = 3.0,
) -> pd.DataFrame:
    """Revised ledgers, all in constant currency where monetary:

    L1   recorded expenditure (sum of cost2021)
    L2   labor burden (hours)
    L3   persistent work-order volume (active-quarter share x count), with
         the persistence factor also reported separately (L3p)
    L4x  tail-cost exposure: sum of cost above the campus p95 threshold,
         counted in EXCESS of the threshold (excess-over-threshold)
    L5r  budget risk: SD of the two-way-demeaned annual constant-currency
         cost over the full campus window (inactive years = true zeros);
         removes entity level and campus-year common shocks, monetary units
    L5r_shrunk  partial-pooling variant (weight n_y/(n_y+kappa) toward the
         campus median SD)
    L5mae  rolling one-step-ahead mean absolute forecast error (mean of
         previous years as the forecast), monetary units
    L5cv   legacy coefficient of variation (sensitivity only)
    plus L6star ingredients: non-labor recorded cost (constant currency).
    """
    rows = []
    for c, gc in wo.groupby("campus", observed=True):
        c = int(c)
        # campus tau on constant-currency per-WO cost
        tau = quantile_type1(gc["cost2021"].to_numpy(), tau_pct / 100.0)
        entities = sorted(gc["entity"].unique())
        M = annual_panel(wo, c, entities, window_years[c])
        resid = _two_way_demean(M)
        sd = resid.std(axis=1, ddof=1)
        med_sd = float(np.median(sd))
        nY = M.shape[1]
        w_sh = nY / (nY + shrink_kappa)
        for i, e in enumerate(entities):
            g = gc[gc["entity"] == e]
            cost21 = g["cost2021"]
            tq = window_quarters[c]
            qn = g.groupby("quarter", observed=True).size()
            persistence = len(qn) / tq
            excess = (cost21 - tau).clip(lower=0)
            # rolling one-step forecast MAE on the full annual series
            series = M[i]
            errs = [abs(series[t] - series[:t].mean()) for t in range(2, nY)]
            mean_annual = float(series.mean())
            cv = (float(series.std(ddof=1)) / mean_annual
                  if mean_annual > 0 else np.nan)
            lab_cost = g["labor_cost"] if "labor_cost" in g else pd.Series(dtype=float)
            has_lc = lab_cost.notna()
            defl = (g["cost2021"] / g["cost"]).where(g["cost"] > 0)
            nonlab_obs = ((g["cost"] - lab_cost) * defl)[has_lc].clip(lower=0).sum()
            share_nl = (1 - (lab_cost[has_lc].sum() /
                             max(g.loc[has_lc, "cost"].sum(), 1e-9)))
            nonlab_miss = (cost21[~has_lc].sum() * max(share_nl, 0.0))
            rows.append({
                "campus": c, "entity": e,
                "L1": float(cost21.sum()),
                "L2": float(g["labor"].sum()),
                "L3": persistence * len(g),
                "L3p": persistence,
                "L4x": float(excess.sum()),
                "L4freq": float((cost21 > tau).mean()),
                "L4sev": float((cost21[cost21 > tau] - tau).mean())
                         if (cost21 > tau).any() else 0.0,
                "L5r": float(sd[i]),
                "L5r_shrunk": float(w_sh * sd[i] + (1 - w_sh) * med_sd),
                "L5mae": float(np.mean(errs)) if errs else np.nan,
                "L5cv": cv,
                "mean_annual_cost": mean_annual,
                "nonlabor_cost2021": float(nonlab_obs + nonlab_miss),
                "labor_cost_coverage": float(has_lc.mean()),
                "n_wos": len(g),
                "n_years_window": nY,
                "n_years_active": int((M[i] > 0).sum()),
                "tau_campus": tau,
            })
    return pd.DataFrame(rows)


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
