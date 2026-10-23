"""Phase 10b: formal statistical tests for the four pre-specified hypotheses.

H1  burden concordance is substantial but incomplete, with top-20%
    disagreement beyond matched finite-record uncertainty:
      (a) Kendall's W per campus and pooled, chi-square test
          chi2 = m (n - 1) W with n - 1 degrees of freedom (m = 4 measures);
      (b) within-campus permutation test of W (10,000 permutations);
      (c) 95% record-noise interval of pooled W under both resampling designs;
      (d) matched-noise concordance test: the W expected if the four measures
          ranked every entity identically up to the sampling noise of its
          own record (same recentred null as the rank-spread floor), against
          the observed W (one-sided, observed lower = incomplete);
      (e) the same null for the number of entities whose campus top-20%
          membership depends on the burden measure.
H2  budget-risk priorities differ from burden priorities:
      (a) tau-b of risk vs median burden rank, campus-stratified entity
          bootstrap interval;
      (b) difference to the weakest within-burden pair, bootstrap interval;
      (c) size-adjusted Spearman with a within-campus permutation p-value;
      (d) correlation corrected for unreliability (Spearman-Brown split-half
          reliabilities of the risk rank and of the median burden rank);
      (e) overlap of the pooled top-K risk set with the burden top-K sets,
          with a record-noise interval.
H3  at least 10% of entities exceed their noise floor under both designs:
      exact one-sided binomial tests against 10% and against the nominal 5%,
      Clopper-Pearson intervals, and Benjamini-Hochberg control of the
      per-entity bootstrap p-values.
H4  minimax-regret worst-case regret is lower than every single-criterion
    rule and mean-rank consensus:
      (a) the shortlists chosen on the observed records re-scored against
          the criterion rankings of every resampled record (both designs);
      (b) out-of-sample split-half test: shortlists chosen on one random half
          of each entity's work orders, scored against the rankings of the
          other half (200 splits), with Wilcoxon signed-rank tests.

Output: results/p10b_tests.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, chi2, kendalltau, spearmanr, wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p5b_shortlist as SL
from p2_ledgers import comparable_wo, load_clean
from p2b_revised import add_revised_ranks
from p3d_revised import load_boot2
from utils.boot import kendalls_w
from utils.io import load_config, results_path, write_json
from utils.ledgers import REVISED_BURDEN, revised_ledger_scores

BURDEN_R = [f"r_{l}" for l in REVISED_BURDEN]
ALL_R = BURDEN_R + ["r_L5r"]
RULES_H4 = ["single_cost", "single_labor", "single_volume", "single_tail",
            "single_risk", "burden_mean", "burden_median", "all_mean"]


def rnd(x, nd=4):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) \
        else round(float(x), nd)


# --------------------------------------------------------------------------
# replicate assembly
# --------------------------------------------------------------------------
def stacked_replicates(df: pd.DataFrame, mode: str):
    """(B, N, 5) replicate ranks and (N, 5) observed ranks in df row order."""
    boot = load_boot2(mode)
    B = min(z["ranks"].shape[0] for z in boot.values())
    reps = np.full((B, len(df), 5), np.nan)
    obs = np.full((len(df), 5), np.nan)
    for c, z in boot.items():
        ents = [str(e) for e in z["entities"]]
        pos = {e: i for i, e in enumerate(ents)}
        rows = df.index[df["campus"] == c]
        for r in rows:
            j = pos[str(df.at[r, "entity"])]
            reps[:, r, :] = z["ranks"][:B, j, :]
            obs[r, :] = z["obs_ranks"][j, :]
    return reps, obs


def topk_dependent_count(R: np.ndarray, campus: np.ndarray, q=0.20) -> int:
    """Entities whose campus top-q membership holds under 1-3 of the four
    burden measures (same rule as p3d_revised.topk_change)."""
    cnt = np.zeros(len(R), int)
    for c in np.unique(campus):
        idx = np.flatnonzero(campus == c)
        k = int(np.ceil(q * len(idx)))
        for j in range(4):
            v = R[idx, j]
            thr = np.sort(v)[::-1][k - 1]
            cnt[idx] += (v >= thr).astype(int)
    return int(np.isin(cnt, [1, 2, 3]).sum())


# --------------------------------------------------------------------------
# H1
# --------------------------------------------------------------------------
def h1(df: pd.DataFrame, seed: int) -> dict:
    out = {"per_campus": {}}
    m = 4
    for c, g in df.groupby("campus"):
        n = len(g)
        W = kendalls_w(g[BURDEN_R].to_numpy())
        stat = m * (n - 1) * W
        out["per_campus"][int(c)] = {"n": n, "W": rnd(W),
                                     "chi2": rnd(stat, 2), "df": n - 1,
                                     "p": float(chi2.sf(stat, n - 1))}
    R = df[BURDEN_R].to_numpy()
    Wp = kendalls_w(R)
    n = len(df)
    out["pooled"] = {"n": n, "W": rnd(Wp), "chi2": rnd(m * (n - 1) * Wp, 2),
                     "df": n - 1, "p": float(chi2.sf(m * (n - 1) * Wp, n - 1))}
    # within-campus permutation of each measure's ranks
    rng = np.random.default_rng(seed)
    camp = df["campus"].to_numpy()
    groups = [np.flatnonzero(camp == c) for c in np.unique(camp)]
    NP = 10000
    ge = 0
    for _ in range(NP):
        Rp = R.copy()
        for j in range(4):
            for idx in groups:
                Rp[idx, j] = rng.permutation(Rp[idx, j])
        ge += kendalls_w(Rp) >= Wp
    out["pooled"]["perm_p"] = (ge + 1) / (NP + 1)
    out["pooled"]["perm_n"] = NP

    obs_count = topk_dependent_count(R, camp)
    out["topk_dependent_observed"] = obs_count
    for mode in ("wo", "year_block"):
        reps, obs = stacked_replicates(df, mode)
        B = reps.shape[0]
        Wb = np.array([kendalls_w(reps[b][:, :4]) for b in range(B)])
        base = np.nanmean(obs[:, :4], axis=1)
        null = np.clip(base[None, :, None] + (reps[:, :, :4] - obs[None, :, :4]),
                       1e-6, 1.0)
        Wnull = np.array([kendalls_w(null[b]) for b in range(B)])
        cnull = np.array([topk_dependent_count(null[b], camp) for b in range(B)])
        out[mode] = {
            "B": B,
            "W_ci95": [rnd(np.percentile(Wb, 2.5)), rnd(np.percentile(Wb, 97.5))],
            "W_null_mean": rnd(Wnull.mean()),
            "W_null_p2_5": rnd(np.percentile(Wnull, 2.5)),
            "W_null_p97_5": rnd(np.percentile(Wnull, 97.5)),
            # one-sided: observed concordance below the matched-noise null
            "p_incomplete": (int((Wnull <= Wp).sum()) + 1) / (B + 1),
            "topk_null_mean": rnd(cnull.mean(), 2),
            "topk_null_p95": rnd(np.percentile(cnull, 95), 1),
            "p_topk_beyond_noise": (int((cnull >= obs_count).sum()) + 1) / (B + 1),
        }
    return out


# --------------------------------------------------------------------------
# H2
# --------------------------------------------------------------------------
def size_adj_rho(g: pd.DataFrame, risk: np.ndarray | None = None) -> float:
    ln = np.log(g["n_wos"].to_numpy())
    r5 = g["r_L5r"].to_numpy() if risk is None else risk
    mb = g["median_burden_rank"].to_numpy()
    res5 = r5 - np.polyval(np.polyfit(ln, r5, 1), ln)
    resb = mb - np.polyval(np.polyfit(ln, mb, 1), ln)
    return float(spearmanr(res5, resb).statistic)


def pair_taus(d: pd.DataFrame) -> dict:
    out = {}
    for i in range(4):
        for j in range(i + 1, 4):
            out[f"{BURDEN_R[i]}|{BURDEN_R[j]}"] = kendalltau(
                d[BURDEN_R[i]], d[BURDEN_R[j]]).statistic
    out["risk|medburden"] = kendalltau(d["r_L5r"], d["median_burden_rank"]).statistic
    return out


def split_half_reliability(wo, cfg, wq, wy, n_splits=50, seed=20260710) -> dict:
    """Spearman-Brown corrected split-half reliability of the risk rank and
    of the median burden rank (same splitting as p2b_revised)."""
    rng = np.random.default_rng(seed)
    acc = {"r_L5r": [], "median_burden_rank": []}
    for _ in range(n_splits):
        wo2 = wo.copy()
        wo2["half"] = rng.integers(0, 2, size=len(wo2))
        halves = []
        for h in (0, 1):
            sc = revised_ledger_scores(wo2[wo2["half"] == h], wq, wy,
                                       tau_pct=cfg["shock_tau_pct"])
            halves.append(add_revised_ranks(sc).set_index(["campus", "entity"]))
        a, b = halves[0].align(halves[1], join="inner", axis=0)
        for col in acc:
            rhos = [spearmanr(a.loc[c, col], b.loc[c, col]).statistic
                    for c in a.index.get_level_values(0).unique()]
            acc[col].append(float(np.mean(rhos)))
    out = {}
    for col, v in acc.items():
        r = float(np.mean(v))
        out[col] = {"split_half": rnd(r), "spearman_brown": rnd(2 * r / (1 + r))}
    return out


def h2(df: pd.DataFrame, rel: dict, seed: int) -> dict:
    out = {}
    obs = pair_taus(df)
    burden_pairs = [v for k, v in obs.items() if k != "risk|medburden"]
    out["tau_risk_medburden"] = rnd(obs["risk|medburden"], 3)
    out["tau_burden_min_pair"] = rnd(min(burden_pairs), 3)
    out["tau_burden_mean_pair"] = rnd(np.mean(burden_pairs), 3)
    rng = np.random.default_rng(seed)
    groups = {c: np.flatnonzero(df["campus"].to_numpy() == c)
              for c in df["campus"].unique()}
    NB = 5000
    t_risk, d_min, d_mean, sa = [], [], [], []
    for _ in range(NB):
        idx = np.concatenate([rng.choice(ix, size=len(ix), replace=True)
                              for ix in groups.values()])
        d = df.iloc[idx]
        pt = pair_taus(d)
        bp = [v for k, v in pt.items() if k != "risk|medburden"]
        t_risk.append(pt["risk|medburden"])
        d_min.append(min(bp) - pt["risk|medburden"])
        d_mean.append(np.mean(bp) - pt["risk|medburden"])
    ci = lambda v: [rnd(np.percentile(v, 2.5), 3), rnd(np.percentile(v, 97.5), 3)]
    out["tau_risk_medburden_ci95"] = ci(t_risk)
    out["diff_min_pair_minus_risk"] = rnd(min(burden_pairs) - obs["risk|medburden"], 3)
    out["diff_min_pair_minus_risk_ci95"] = ci(d_min)
    out["p_diff_min_pair_le0"] = (int((np.array(d_min) <= 0).sum()) + 1) / (NB + 1)
    out["diff_mean_pair_minus_risk"] = rnd(np.mean(burden_pairs) - obs["risk|medburden"], 3)
    out["diff_mean_pair_minus_risk_ci95"] = ci(d_mean)
    out["p_diff_mean_pair_le0"] = (int((np.array(d_mean) <= 0).sum()) + 1) / (NB + 1)
    # size-adjusted association with permutation test
    sa_obs = float(np.mean([size_adj_rho(g) for _, g in df.groupby("campus")]))
    NP = 10000
    ge = 0
    camp_groups = list(df.groupby("campus"))
    for _ in range(NP):
        vals = [size_adj_rho(g, rng.permutation(g["r_L5r"].to_numpy()))
                for _, g in camp_groups]
        ge += abs(np.mean(vals)) >= abs(sa_obs)
    out["size_adj_rho"] = rnd(sa_obs, 3)
    out["size_adj_perm_p"] = (ge + 1) / (NP + 1)
    for _ in range(2000):
        idx_by = {c: rng.choice(ix, size=len(ix), replace=True)
                  for c, ix in groups.items()}
        vals = []
        for c, ix in idx_by.items():
            g = df.iloc[ix]
            if g["n_wos"].nunique() > 2:
                vals.append(size_adj_rho(g))
        sa.append(np.mean(vals))
    out["size_adj_rho_ci95"] = ci(sa)
    # disattenuated Spearman (mean of within-campus Spearman)
    rho = float(np.mean([spearmanr(g["r_L5r"], g["median_burden_rank"]).statistic
                         for _, g in df.groupby("campus")]))
    rr = rel["r_L5r"]["spearman_brown"]
    rb = rel["median_burden_rank"]["spearman_brown"]
    out["rho_risk_medburden_within_campus"] = rnd(rho, 3)
    out["reliability"] = rel
    out["rho_disattenuated"] = rnd(min(rho / np.sqrt(rr * rb), 1.0), 3)
    # top-K set overlap (pooled K) with record-noise interval
    K = int(np.ceil(0.20 * len(df)))
    def overlaps(R):
        top = {j: set(np.argsort(-R[:, j], kind="stable")[:K]) for j in range(5)}
        mb = np.nanmean(R[:, :4], axis=1)
        topmb = set(np.argsort(-mb, kind="stable")[:K])
        return ([len(top[4] & top[j]) / K for j in range(4)]
                + [len(top[4] & topmb) / K])
    obs_ov = overlaps(df[ALL_R].to_numpy())
    out["K"] = K
    out["overlap_risk_top_with"] = dict(zip(
        ["cost", "labor", "volume", "tail", "burden_mean"],
        [rnd(v, 3) for v in obs_ov]))
    for mode in ("wo", "year_block"):
        reps, _ = stacked_replicates(df, mode)
        ov = np.array([overlaps(reps[b]) for b in range(reps.shape[0])])
        out[f"overlap_cost_ci95_{mode}"] = ci(ov[:, 0])
        out[f"overlap_burden_mean_ci95_{mode}"] = ci(ov[:, 4])
    return out


# --------------------------------------------------------------------------
# H3
# --------------------------------------------------------------------------
def bh_count(p: np.ndarray, q: float = 0.05) -> int:
    p = np.sort(np.asarray(p))
    n = len(p)
    ok = p <= q * np.arange(1, n + 1) / n
    return int(np.max(np.flatnonzero(ok)) + 1) if ok.any() else 0


def h3(df: pd.DataFrame, sim: dict) -> dict:
    out = {}
    n = len(df)
    for mode in ("wo", "year_block"):
        reps, obs = stacked_replicates(df, mode)
        B = reps.shape[0]
        base = np.nanmean(obs[:, :4], axis=1)
        null = np.clip(base[None, :, None] + (reps[:, :, :4] - obs[None, :, :4]),
                       1e-6, 1.0)
        null_sd = np.nanstd(null, axis=2, ddof=1)
        obs_sd = np.nanstd(obs[:, :4], axis=1, ddof=1)
        floor = np.nanpercentile(null_sd, 95, axis=0)
        flag = obs_sd > floor
        k = int(flag.sum())
        pvals = (1 + (null_sd >= obs_sd[None, :]).sum(axis=0)) / (B + 1)
        t10 = binomtest(k, n, 0.10, alternative="greater")
        t05 = binomtest(k, n, 0.05, alternative="greater")
        type1 = max(sim["S0"][mode]["type1_meii_b"],
                    sim["S3"][mode]["type1_meii_b"])
        tsim = binomtest(k, n, max(type1, 0.05), alternative="greater")
        ci = t10.proportion_ci(confidence_level=0.95, method="exact")
        out[mode] = {"k": k, "n": n, "share": rnd(k / n),
                     "ci95": [rnd(ci.low), rnd(ci.high)],
                     "p_vs_10pct": float(t10.pvalue),
                     "p_vs_5pct": float(t05.pvalue),
                     "p_vs_sim_type1": float(tsim.pvalue),
                     "sim_type1_used": rnd(max(type1, 0.05)),
                     "bh_q05_count": bh_count(pvals, 0.05),
                     "min_p": rnd(pvals.min(), 5)}
    return out


# --------------------------------------------------------------------------
# H4
# --------------------------------------------------------------------------
def regret_from_ranks(S_idx: set, R: np.ndarray, K: int) -> float:
    worst = 0.0
    for j in range(5):
        top = set(np.argsort(-R[:, j], kind="stable")[:K])
        worst = max(worst, 1 - len(S_idx & top) / K)
    return worst


def h4_rescore(df: pd.DataFrame) -> dict:
    sel = SL.rule_selections(df, 0.20)
    K = sel["K"]
    pos = {ix: i for i, ix in enumerate(df.index)}
    S = {r: {pos[e] for e in sel[r]["_sel"]} for r in RULES_H4 + ["minimax"]}
    out = {"K": K}
    for mode in ("wo", "year_block"):
        reps, _ = stacked_replicates(df, mode)
        B = reps.shape[0]
        wr = {r: np.array([regret_from_ranks(S[r], reps[b], K) for b in range(B)])
              for r in S}
        res = {"mean_worst": {r: rnd(v.mean(), 3) for r, v in wr.items()}}
        for r in RULES_H4:
            d = wr[r] - wr["minimax"]
            res[r] = {"share_minimax_lower": rnd((d > 0).mean(), 3),
                      "share_minimax_le": rnd((d >= 0).mean(), 3),
                      "mean_diff": rnd(d.mean(), 3),
                      "diff_ci95": [rnd(np.percentile(d, 2.5), 3),
                                    rnd(np.percentile(d, 97.5), 3)]}
        out[mode] = res
    return out


def h4_split_half(wo, cfg, wq, wy, n_splits=200, seed=20261023) -> dict:
    rng = np.random.default_rng(seed)
    rec = {r: [] for r in RULES_H4 + ["minimax"]}
    for s in range(n_splits):
        wo2 = wo.copy()
        wo2["half"] = rng.integers(0, 2, size=len(wo2))
        sc = {}
        for h in (0, 1):
            x = revised_ledger_scores(wo2[wo2["half"] == h], wq, wy,
                                      tau_pct=cfg["shock_tau_pct"])
            sc[h] = add_revised_ranks(x).set_index(["campus", "entity"])
        a, b = sc[0].align(sc[1], join="inner", axis=0)
        a = a.reset_index()
        b = b.reset_index()
        sel = SL.rule_selections(a, 0.20)
        K = sel["K"]
        Rb = b[ALL_R].to_numpy()
        for r in rec:
            rec[r].append(regret_from_ranks(set(sel[r]["_sel"]), Rb, K))
    out = {"n_splits": n_splits,
           "mean_worst": {r: rnd(np.mean(v), 3) for r, v in rec.items()}}
    mm = np.array(rec["minimax"])
    for r in RULES_H4:
        d = np.array(rec[r]) - mm
        try:
            p = float(wilcoxon(d, alternative="greater", zero_method="wilcox").pvalue)
        except ValueError:
            p = None
        out[r] = {"share_minimax_lower": rnd((d > 0).mean(), 3),
                  "share_tie": rnd((d == 0).mean(), 3),
                  "share_minimax_higher": rnd((d < 0).mean(), 3),
                  "mean_diff": rnd(d.mean(), 3),
                  "wilcoxon_p_greater": p}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", type=int, default=200)
    args = ap.parse_args()
    cfg = load_config()
    wq = {int(k): (v[1] - v[0] + 1) * 4 for k, v in cfg["campus_valid_window"].items()}
    wy = {int(k): tuple(v) for k, v in cfg["campus_valid_window"].items()}
    df = pd.read_csv(results_path("p2b_ledgers_entity.csv"))
    sim = __import__("json").load(open(results_path("p3e_simulation.json")))
    wo = comparable_wo(load_clean(cfg), "system")

    out = {}
    out["H1"] = h1(df, cfg["seed"])
    print("H1 done", flush=True)
    rel = split_half_reliability(wo, cfg, wq, wy)
    out["H2"] = h2(df, rel, cfg["seed"])
    print("H2 done", flush=True)
    out["H3"] = h3(df, sim)
    print("H3 done", flush=True)
    out["H4"] = {"rescore": h4_rescore(df),
                 "split_half": h4_split_half(wo, cfg, wq, wy, args.splits)}
    print("H4 done", flush=True)
    write_json(out, results_path("p10b_tests.json"))
    print(__import__("json").dumps(out, indent=1, default=float)[:6000])


if __name__ == "__main__":
    main()
