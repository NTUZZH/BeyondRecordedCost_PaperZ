"""Independence reference for the burden-only design.

The framework text promises "an independence ceiling from within-campus
permutation of each criterion's ranks". The legacy N2 null (p3_nulls.json)
permutes all five ledgers of the superseded single-layer analysis, so it
does not match the current burden-only rank spread I^B over the four
burden criteria. This script computes the matching reference: within each
campus, permute the four burden percentile-rank columns independently
(destroying cross-criterion association, preserving each marginal), then
record the pooled Kendall's W over the four permuted rankings and the
mean within-burden rank spread I^B across entities. N draws, seeded from
the study seed; deterministic.

Writes results/p3g_independence.json.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils.io import load_config, results_path, write_json  # noqa: E402
from p3_stability import kendalls_w  # noqa: E402

BURDEN_RANKS = ["r_L1", "r_L2", "r_L3", "r_L4x"]


def main() -> None:
    cfg = load_config()
    led = pd.read_csv(results_path("p2b_ledgers_entity.csv"))
    rng = np.random.default_rng(np.random.SeedSequence((cfg["seed"], 993)))
    N = cfg["permutation_N"]
    W_perm, meii_perm = [], []
    campuses = list(led.groupby("campus"))
    for _ in range(N):
        pieces = []
        for _c, g in campuses:
            R = g[BURDEN_RANKS].to_numpy().copy()
            for j in range(R.shape[1]):
                col = R[:, j]
                ok = ~np.isnan(col)
                col[ok] = rng.permutation(col[ok])
                R[:, j] = col
            pieces.append(R)
        Rp = np.vstack(pieces)
        W_perm.append(kendalls_w(Rp))
        meii_perm.append(float(np.nanmean(np.nanstd(Rp, axis=1, ddof=1))))
    out = {
        "n_draws": N,
        "burden_ranks": BURDEN_RANKS,
        "W_perm_mean": round(float(np.mean(W_perm)), 4),
        "W_perm_p95": round(float(np.percentile(W_perm, 95)), 4),
        "meii_b_perm_mean": round(float(np.mean(meii_perm)), 4),
        "meii_b_perm_p5": round(float(np.percentile(meii_perm, 5)), 4),
        "meii_b_perm_p95": round(float(np.percentile(meii_perm, 95)), 4),
    }
    write_json(out, results_path("p3g_independence.json"))
    print("p3g_independence:", out)


if __name__ == "__main__":
    main()
