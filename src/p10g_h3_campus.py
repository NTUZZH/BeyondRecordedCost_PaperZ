"""Phase 10g: H3 with the campus as the unit of inference.

Percentile ranks within a campus depend on one another, so the flags of the
entities of one campus are not independent trials.  The six campuses are
separate organizations with separate records, so they are independent
units.  This script repeats the H3 inference at that level:

  (a) the share of entities above their noise floor at each campus;
  (b) a one-sided sign test across the six campuses of whether the campus
      share exceeds the 10% of H3;
  (c) a campus-cluster bootstrap interval of the pooled share (campuses
      resampled with replacement, 10,000 draws);
  (d) the Benjamini-Yekutieli count of the per-entity bootstrap p-values,
      which controls the false discovery rate under any dependence.

The flags and p-values are the ones of p10b_tests.py (same replicates, same
floor rule), and the script asserts that its pooled counts match.

Output: results/p10g_h3_campus.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from p10b_tests import stacked_replicates
from utils.io import load_config, results_path, write_json


def by_count(p: np.ndarray, q: float = 0.05) -> int:
    """Benjamini-Yekutieli step-up count, valid under arbitrary dependence."""
    p = np.sort(np.asarray(p))
    n = len(p)
    cm = np.sum(1.0 / np.arange(1, n + 1))
    ok = p <= q * np.arange(1, n + 1) / (n * cm)
    return int(np.max(np.flatnonzero(ok)) + 1) if ok.any() else 0


def flags_and_p(df: pd.DataFrame, mode: str):
    """Noise-floor flags and bootstrap p-values, exactly as in p10b.h3."""
    reps, obs = stacked_replicates(df, mode)
    B = reps.shape[0]
    base = np.nanmean(obs[:, :4], axis=1)
    null = np.clip(base[None, :, None] + (reps[:, :, :4] - obs[None, :, :4]),
                   1e-6, 1.0)
    null_sd = np.nanstd(null, axis=2, ddof=1)
    obs_sd = np.nanstd(obs[:, :4], axis=1, ddof=1)
    floor = np.nanpercentile(null_sd, 95, axis=0)
    flag = obs_sd > floor
    pvals = (1 + (null_sd >= obs_sd[None, :]).sum(axis=0)) / (B + 1)
    return flag, pvals


def main() -> None:
    cfg = load_config()
    df = pd.read_csv(results_path("p2b_ledgers_entity.csv"))
    p10b = json.load(open(results_path("p10b_tests.json")))["H3"]
    rng = np.random.default_rng(cfg["seed"])
    campus = df["campus"].to_numpy()
    camps = np.unique(campus)
    out = {"n_campuses": int(len(camps)), "threshold": 0.10}
    for mode in ("wo", "year_block"):
        flag, pvals = flags_and_p(df, mode)
        assert int(flag.sum()) == p10b[mode]["k"], "flags differ from p10b"
        k_c = np.array([int(flag[campus == c].sum()) for c in camps])
        n_c = np.array([int((campus == c).sum()) for c in camps])
        share_c = k_c / n_c
        above = int((share_c > out["threshold"]).sum())
        sign = binomtest(above, len(camps), 0.5, alternative="greater")
        draws = rng.integers(0, len(camps), size=(10_000, len(camps)))
        pooled = k_c[draws].sum(axis=1) / n_c[draws].sum(axis=1)
        out[mode] = {
            "share_by_campus": {str(int(c)): round(float(s), 4)
                                for c, s in zip(camps, share_c)},
            "share_min": round(float(share_c.min()), 4),
            "share_max": round(float(share_c.max()), 4),
            "campuses_above_threshold": above,
            "sign_test_p": float(sign.pvalue),
            "cluster_boot_ci95": [round(float(np.percentile(pooled, 2.5)), 4),
                                  round(float(np.percentile(pooled, 97.5)), 4)],
            "bh_q05_count": p10b[mode]["bh_q05_count"],
            "by_q05_count": by_count(pvals, 0.05),
        }
    write_json(out, results_path("p10g_h3_campus.json"))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
