"""Joint-by-campus bootstrap machinery for Phase 3.

Design: per campus, per replicate, every entity's work orders are resampled
with replacement via multinomial weights; all five ledger scores are weighted
aggregates (numpy, never touching the raw table); the shock threshold tau is
re-estimated per replicate from the pooled weighted campus cost distribution;
ranks are recomputed against the other entities' replicates in the same
replicate world. Batches are seeded deterministically via SeedSequence
(seed, campus, batch) so results are independent of parallel scheduling.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata


class CampusBootstrap:
    """Precomputed per-WO arrays for one campus's comparable entities."""

    def __init__(self, campus: int, wo_campus, entity_order: list[str],
                 window_quarters: int, vol_min_years: int, tau_pct: float,
                 vol_variant: str = "cv_annual"):
        self.campus = campus
        self.entities = entity_order
        self.Tq = window_quarters
        self.vol_min_years = vol_min_years
        self.q = tau_pct / 100.0
        # vol_variant governs L5: cv_annual (SD/mean of annual cost, primary),
        # cv_quarterly (SD/mean of quarterly cost), or mad_median (MAD/median
        # of annual cost). Used by the L5-robustness N1 recompute.
        self.vol_variant = vol_variant

        self.cost0 = []      # cost with NaN->0 (for weighted sums)
        self.costraw = []    # cost with NaN (for tau threshold masks)
        self.labor0 = []
        self.qcode = []      # dense quarter codes per entity
        self.nq = []
        self.ycode = []
        self.ny = []
        self.n = []

        for e in entity_order:
            g = wo_campus[wo_campus["entity"] == e]
            cost = g["cost"].to_numpy(dtype=np.float64)
            self.costraw.append(cost)
            self.cost0.append(np.nan_to_num(cost, nan=0.0))
            self.labor0.append(np.nan_to_num(g["labor"].to_numpy(dtype=np.float64), nan=0.0))
            qu = g["quarter"].to_numpy()
            qcats, qc = np.unique(qu, return_inverse=True)
            self.qcode.append(qc.astype(np.int64))
            self.nq.append(len(qcats))
            yr = g["year"].to_numpy()
            ycats, yc = np.unique(yr, return_inverse=True)
            self.ycode.append(yc.astype(np.int64))
            self.ny.append(len(ycats))
            self.n.append(len(g))

        # pooled non-missing costs sorted once, with entity/row mapping
        pool_cost, pool_ent, pool_row = [], [], []
        for ei, cost in enumerate(self.costraw):
            nm = np.flatnonzero(~np.isnan(cost))
            pool_cost.append(cost[nm])
            pool_ent.append(np.full(len(nm), ei, dtype=np.int32))
            pool_row.append(nm.astype(np.int64))
        pc = np.concatenate(pool_cost)
        pe = np.concatenate(pool_ent)
        pr = np.concatenate(pool_row)
        order = np.argsort(pc, kind="stable")
        self.pool_cost_sorted = pc[order]
        # map each pooled sorted row to its position in the concatenated
        # global weight vector (one gather per replicate instead of E masks)
        offsets = np.concatenate([[0], np.cumsum(self.n)])[:-1]
        self.pool_flat_idx = (offsets[pe] + pr)[order]

    # ----- score computation for one weight world -----
    def _tau(self, weights: list[np.ndarray]) -> float:
        wg = np.concatenate(weights)
        cw = np.cumsum(wg[self.pool_flat_idx])
        total = cw[-1]
        if total <= 0:
            return np.inf
        idx = np.searchsorted(cw, self.q * total, side="left")
        idx = min(idx, len(cw) - 1)
        return float(self.pool_cost_sorted[idx])

    def scores_for_weights(self, weights: list[np.ndarray]) -> np.ndarray:
        """Return (n_entities, 5) array of L1..L5 for one replicate world."""
        tau = self._tau(weights)
        out = np.full((len(self.entities), 5), np.nan)
        for ei in range(len(self.entities)):
            w = weights[ei]
            cost0, costraw = self.cost0[ei], self.costraw[ei]
            out[ei, 0] = w @ cost0
            out[ei, 1] = w @ self.labor0[ei]
            qocc = np.bincount(self.qcode[ei], weights=w, minlength=self.nq[ei]) > 0
            out[ei, 2] = (qocc.sum() / self.Tq) * self.n[ei]
            mask = np.nan_to_num(costraw, nan=-np.inf) > tau
            out[ei, 3] = (w * cost0)[mask].sum() if mask.any() else 0.0
            if self.vol_variant == "cv_quarterly":
                csum = np.bincount(self.qcode[ei], weights=w * cost0, minlength=self.nq[ei])
                cocc = np.bincount(self.qcode[ei], weights=w, minlength=self.nq[ei]) > 0
                series = csum[cocc]
                min_obs = self.vol_min_years * 4
            else:
                ysum = np.bincount(self.ycode[ei], weights=w * cost0, minlength=self.ny[ei])
                yocc = np.bincount(self.ycode[ei], weights=w, minlength=self.ny[ei]) > 0
                series = ysum[yocc]
                min_obs = self.vol_min_years
            if len(series) >= min_obs:
                if self.vol_variant == "mad_median":
                    med = np.median(series)
                    out[ei, 4] = np.median(np.abs(series - med)) / med if med > 0 else np.nan
                else:
                    m = series.mean()
                    out[ei, 4] = series.std(ddof=1) / m if m > 0 else np.nan
        return out

    def observed_scores(self) -> np.ndarray:
        return self.scores_for_weights([np.ones(n) for n in self.n])

    def run_batch(self, seed_key: tuple, B: int) -> np.ndarray:
        """Return (B, n_entities, 5) rank array for B replicate worlds."""
        rng = np.random.default_rng(np.random.SeedSequence(seed_key))
        nE = len(self.entities)
        ranks = np.full((B, nE, 5), np.nan)
        for b in range(B):
            weights = [
                rng.multinomial(n, np.full(n, 1.0 / n)).astype(np.float64)
                for n in self.n
            ]
            sc = self.scores_for_weights(weights)
            ranks[b] = rank_matrix(sc)
        return ranks


def rank_matrix(scores: np.ndarray) -> np.ndarray:
    """Within-campus percentile ranks per ledger column, NaN-aware."""
    nE, nL = scores.shape
    out = np.full((nE, nL), np.nan)
    for j in range(nL):
        col = scores[:, j]
        ok = ~np.isnan(col)
        if ok.sum() > 0:
            out[ok, j] = rankdata(col[ok], method="average") / ok.sum()
    return out


def kendalls_w(rank_cols: np.ndarray) -> float:
    """Kendall's W with tie correction. rank_cols: (n_items, m_judges) of
    values whose within-column ordering defines the rankings (complete case)."""
    X = np.asarray(rank_cols, dtype=float)
    ok = ~np.isnan(X).any(axis=1)
    X = X[ok]
    n, m = X.shape
    if n < 3:
        return np.nan
    R = np.zeros((n, m))
    T = 0.0
    for j in range(m):
        R[:, j] = rankdata(X[:, j], method="average")
        _, counts = np.unique(X[:, j], return_counts=True)
        T += float(((counts ** 3) - counts).sum())
    Ri = R.sum(axis=1)
    S = float(((Ri - Ri.mean()) ** 2).sum())
    denom = (m ** 2) * (n ** 3 - n) - m * T
    return 12.0 * S / denom if denom > 0 else np.nan
