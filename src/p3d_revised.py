"""Phase 3d: revised stability analysis under the two-layer framework.

Layer 1 (related burden representations L1, L2, L3, L4x):
  concordance (Kendall W, pairwise tau-b), top-k overlap, per-entity
  within-burden instability MEII_b = SD of the four burden ranks,
  significance vs a matched N1 noise floor under BOTH resampling designs
  (work-order multinomial and campus year-block).
Layer 2 (distinct objective L5r budget risk):
  GAP_r = |r_L5r - median burden rank|, significance vs the same floors,
  scale-adjusted association between risk and burden.
Influence/consistency diagnostics: leave-one-ledger-out MEII_b, SD/MAD/IQR/
  range agreement, link between MEII_b and top-k membership changes.
Campus-blocked permutation for the system-category effect on MEII_b.

Outputs: results/p3d_revised.json, results/p3d_entity.csv,
         results/boot2/<campus>_<mode>.npz (replicate rank caches)
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
from p2_ledgers import comparable_wo, load_clean
from utils.boot import RevisedCampusBootstrap, kendalls_w
from utils.io import ROOT, load_config, results_path, write_json
from utils.ledgers import REVISED_ALL, REVISED_BURDEN

BOOT2 = ROOT / "results" / "boot2"
BURDEN_R = [f"r_{l}" for l in REVISED_BURDEN]


def build_boots(cfg, wo, mode: str) -> dict:
    wq = {int(k): (v[1] - v[0] + 1) * 4 for k, v in cfg["campus_valid_window"].items()}
    wy = {int(k): tuple(v) for k, v in cfg["campus_valid_window"].items()}
    boots = {}
    for c, g in wo.groupby("campus", observed=True):
        ents = sorted(g["entity"].unique())
        boots[int(c)] = RevisedCampusBootstrap(
            int(c), g, ents, wq[int(c)], wy[int(c)],
            cfg["shock_tau_pct"], mode=mode)
    return boots


def run_boot(cfg, wo, mode: str, B: int, jobs: int, seed: int) -> None:
    BOOT2.mkdir(exist_ok=True)
    boots = build_boots(cfg, wo, mode)
    batch = max(1, B // max(jobs, 1))
    mode_id = {"wo": 0, "year_block": 1}[mode]
    for c, bt in boots.items():
        keys = []
        b0 = 0
        while b0 < B:
            nb = min(batch, B - b0)
            keys.append(((seed, c, b0, mode_id), nb))
            b0 += nb
        parts = Parallel(n_jobs=jobs)(
            delayed(bt.run_batch)(k, nb) for k, nb in keys)
        ranks = np.concatenate(parts, axis=0)
        obs = bt.observed_scores()
        from utils.boot import rank_matrix
        np.savez_compressed(BOOT2 / f"c{c}_{mode}.npz",
                            ranks=ranks, obs_scores=obs,
                            obs_ranks=rank_matrix(obs),
                            entities=np.array(bt.entities))
        print(f"boot2 campus {c} mode={mode}: {ranks.shape}", flush=True)


def load_boot2(mode: str) -> dict:
    out = {}
    for f in sorted(BOOT2.glob(f"c*_{mode}.npz")):
        z = np.load(f, allow_pickle=True)
        c = int(f.stem.split("_")[0][1:])
        out[c] = {k: z[k] for k in z.files}
    return out


def floors_from_boot(boot: dict, cols: slice, stat: str) -> pd.DataFrame:
    """Per-entity 95th-percentile noise floor for MEII_b (stat='sd' over
    burden cols) or GAP_r (stat='gap': |risk - median(burden)|), using the
    recentred deviation null (same construction as the original N1)."""
    rows = []
    for c, z in boot.items():
        ranks, obs = z["ranks"], z["obs_ranks"]
        dev = ranks - obs[None, :, :]
        if stat == "sd":
            base = np.nanmean(obs[:, cols], axis=1)
            recentred = np.clip(base[None, :, None] + dev[:, :, cols], 1e-6, 1.0)
            null = np.nanstd(recentred, axis=2, ddof=1)
            observed = np.nanstd(obs[:, cols], axis=1, ddof=1)
        else:
            med_b = np.nanmedian(obs[:, :4], axis=1)
            null_b = np.clip(med_b[None, :, None] + dev[:, :, :4], 1e-6, 1.0)
            null_med = np.nanmedian(null_b, axis=2)
            r_obs = obs[:, 4]
            null_r = np.clip(r_obs[None, :] + dev[:, :, 4], 1e-6, 1.0)
            null = np.abs(null_r - null_med)
            observed = np.abs(r_obs - med_b)
        floor = np.nanpercentile(null, 95, axis=0)
        for ei, e in enumerate(z["entities"]):
            rows.append({"campus": c, "entity": str(e),
                         f"{stat}_obs": float(observed[ei]),
                         f"{stat}_floor95": float(floor[ei]),
                         f"{stat}_sig": bool(observed[ei] > floor[ei])})
    return pd.DataFrame(rows)


def influence_analysis(df: pd.DataFrame) -> dict:
    R = df[BURDEN_R].to_numpy()
    out = {}
    # leave-one-ledger-out MEII_b
    loo = {}
    for j, l in enumerate(REVISED_BURDEN):
        keep = [k for k in range(4) if k != j]
        loo[l] = float(np.nanmean(np.nanstd(R[:, keep], axis=1, ddof=1)))
    out["mean_meii_b_leave_one_out"] = {k: round(v, 4) for k, v in loo.items()}
    out["mean_meii_b_all"] = round(float(np.nanmean(np.nanstd(R, axis=1, ddof=1))), 4)
    # dispersion-measure agreement
    sd = np.nanstd(R, axis=1, ddof=1)
    mad = np.nanmedian(np.abs(R - np.nanmedian(R, axis=1, keepdims=True)), axis=1)
    iqr = np.nanpercentile(R, 75, axis=1) - np.nanpercentile(R, 25, axis=1)
    rng_ = np.nanmax(R, axis=1) - np.nanmin(R, axis=1)
    for name, v in (("mad", mad), ("iqr", iqr), ("range", rng_)):
        out[f"spearman_sd_vs_{name}"] = round(
            float(spearmanr(sd, v).statistic), 3)
    # W with each burden ledger removed
    wl = {}
    for j, l in enumerate(REVISED_BURDEN):
        keep = [k for k in range(4) if k != j]
        wl[l] = round(float(kendalls_w(R[:, keep])), 3)
    out["W_burden_leave_one_out"] = wl
    return out


def topk_change(df: pd.DataFrame, q: float = 0.20) -> pd.Series:
    """Number of burden ledgers under which the entity is in the campus
    top-20% (0 or 4 = action-stable; 1-3 = criterion-dependent)."""
    cnt = pd.Series(0, index=df.index)
    for c, g in df.groupby("campus", observed=True):
        k = int(np.ceil(q * len(g)))
        for col in BURDEN_R:
            thr = g[col].nlargest(k).min()
            cnt.loc[g.index] += (g[col] >= thr).astype(int)
    return cnt


def blocked_permutation(df: pd.DataFrame, n_perm: int = 10000,
                        seed: int = 42) -> dict:
    """Campus-blocked permutation test for a system-category effect on
    MEII_b: permute MEII_b within campus, compare between-category variance
    of category means (categories with >=4 entities)."""
    rng = np.random.default_rng(seed)
    d = df.dropna(subset=["meii_burden"]).copy()
    cats = d["entity"].value_counts()
    d = d[d["entity"].isin(cats[cats >= 4].index)]

    def stat(v: np.ndarray) -> float:
        s = pd.Series(v, index=d.index)
        return float(s.groupby(d["entity"]).mean().var(ddof=1))

    obs = stat(d["meii_burden"].to_numpy())
    null = np.empty(n_perm)
    vals = d["meii_burden"].to_numpy().copy()
    campus_idx = d.groupby("campus", observed=True).indices
    for p in range(n_perm):
        v = vals.copy()
        for c, idx in campus_idx.items():
            v[idx] = rng.permutation(v[idx])
        null[p] = stat(v)
    return {"obs_between_category_var": round(obs, 6),
            "p_value_campus_blocked": round(float((null >= obs).mean()), 4),
            "n_categories": int(d["entity"].nunique()),
            "n_entities": int(len(d))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", action="store_true")
    ap.add_argument("--mode", default="both", choices=["wo", "year_block", "both"])
    ap.add_argument("--B", type=int, default=1000)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--analyze", action="store_true")
    args = ap.parse_args()
    cfg = load_config()
    wo = comparable_wo(load_clean(cfg), "system")

    if args.boot:
        modes = ["wo", "year_block"] if args.mode == "both" else [args.mode]
        for m in modes:
            run_boot(cfg, wo, m, args.B, args.jobs, cfg["seed"])
        return

    # ---------- analysis ----------
    df = pd.read_csv(results_path("p2b_ledgers_entity.csv"))
    out = {}

    # Layer 1: burden concordance
    R = df[BURDEN_R].to_numpy()
    out["W_burden_pooled"] = round(float(kendalls_w(R)), 4)
    out["W_burden_by_campus"] = {
        int(c): round(float(kendalls_w(g[BURDEN_R].to_numpy())), 4)
        for c, g in df.groupby("campus", observed=True)}
    taus = {}
    for i in range(4):
        for j in range(i + 1, 4):
            x, y = R[:, i], R[:, j]
            ok = ~(np.isnan(x) | np.isnan(y))
            taus[f"{REVISED_BURDEN[i]}_{REVISED_BURDEN[j]}"] = round(
                float(kendalltau(x[ok], y[ok]).statistic), 3)
    out["pairwise_taub_burden"] = taus

    # Layer 2: risk vs burden, raw and scale-adjusted
    ok = df["r_L5r"].notna()
    out["taub_risk_vs_medburden"] = round(float(kendalltau(
        df.loc[ok, "r_L5r"], df.loc[ok, "median_burden_rank"]).statistic), 3)
    rhos = []
    for c, g in df.groupby("campus", observed=True):
        ln = np.log(g["n_wos"].to_numpy())
        r5 = g["r_L5r"].to_numpy()
        mb = g["median_burden_rank"].to_numpy()
        res5 = r5 - np.polyval(np.polyfit(ln, r5, 1), ln)
        resb = mb - np.polyval(np.polyfit(ln, mb, 1), ln)
        rhos.append(spearmanr(res5, resb).statistic)
    out["spearman_risk_vs_burden_size_adj"] = round(float(np.mean(rhos)), 3)

    # noise floors under both designs
    ent = df[["campus", "entity", "meii_burden", "gap_risk",
              "mean_burden_rank", "median_burden_rank", "n_wos"] + BURDEN_R
             + ["r_L5r"]].copy()
    for mode in ("wo", "year_block"):
        boot = load_boot2(mode)
        if not boot:
            print(f"(no boot2 cache for mode={mode}; run --boot first)")
            continue
        fb = floors_from_boot(boot, slice(0, 4), "sd").rename(columns={
            "sd_obs": f"meii_b_obs_{mode}", "sd_floor95": f"meii_b_floor_{mode}",
            "sd_sig": f"meii_b_sig_{mode}"})
        fg = floors_from_boot(boot, None, "gap").rename(columns={
            "gap_obs": f"gap_obs_{mode}", "gap_floor95": f"gap_floor_{mode}",
            "gap_sig": f"gap_sig_{mode}"})
        ent = ent.merge(fb, on=["campus", "entity"]).merge(
            fg, on=["campus", "entity"])
        out[f"share_meii_b_sig_{mode}"] = round(
            float(ent[f"meii_b_sig_{mode}"].mean()), 4)
        out[f"share_gap_sig_{mode}"] = round(
            float(ent[f"gap_sig_{mode}"].mean()), 4)

    ent["topk_burden_count"] = topk_change(df)
    # two-dimensional profile: within-burden instability (significant
    # MEII_b under the primary work-order design) x risk divergence
    # (GAP_r in the pooled upper quartile; descriptive dimension)
    gap_q3 = float(ent["gap_risk"].quantile(0.75))
    burden_unstable = ent.get("meii_b_sig_wo",
                              pd.Series(False, index=ent.index)).astype(bool)
    risk_divergent = ent["gap_risk"] >= gap_q3
    ent["profile"] = np.select(
        [~burden_unstable & ~risk_divergent,
         ~burden_unstable & risk_divergent,
         burden_unstable & ~risk_divergent,
         burden_unstable & risk_divergent],
        ["stable_aligned", "stable_divergent",
         "unstable_aligned", "unstable_divergent"])
    out["profile_counts"] = ent["profile"].value_counts().to_dict()
    out["gap_upper_quartile"] = round(gap_q3, 4)
    out["influence"] = influence_analysis(df)
    out["blocked_permutation_category"] = blocked_permutation(
        df.assign(entity=df["entity"].str.split("|").str[-1]))
    # MEII_b vs top-k membership instability
    mid = ent["topk_burden_count"].isin([1, 2, 3])
    out["meii_b_mean_topk_contested"] = round(
        float(ent.loc[mid, "meii_burden"].mean()), 4)
    out["meii_b_mean_topk_stable"] = round(
        float(ent.loc[~mid, "meii_burden"].mean()), 4)

    ent.to_csv(results_path("p3d_entity.csv"), index=False)
    write_json(out, results_path("p3d_revised.json"))
    print("p3d analysis written")
    for k, v in out.items():
        if not isinstance(v, dict):
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
