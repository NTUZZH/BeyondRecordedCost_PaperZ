"""Phase 3: decision-stability testing, MEII, null models, regret.

Stage A (--boot): joint-by-campus cluster bootstrap (N1 machinery), parallel
  across (campus, batch) tasks, checkpointed per campus under results/boot/.
Stage B (--analyze): agreement metrics, MEII + N1 floors, N2 permutation
  ceiling, N3 split-half, structure-in-instability, consensus + regret,
  H1-H4 verdicts.

Usage: python p3_stability.py --boot [--B 1000] [--jobs 20]
       python p3_stability.py --analyze
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import kendalltau, kruskal, rankdata, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils.boot import CampusBootstrap, kendalls_w, rank_matrix
from utils.io import ROOT, load_config, results_path, write_json

LEDGERS = ["L1", "L2", "L3", "L4", "L5"]
RANKS = [f"r_{l}" for l in LEDGERS]
BOOT_DIR = ROOT / "results" / "boot"


# --------------------------------------------------------------------------
# shared loading
# --------------------------------------------------------------------------

def load_inputs(cfg):
    wo = pd.read_parquet(ROOT / "data/interim/wo_clean.parquet").rename(
        columns={"labor_hours": "labor"})
    ent = pd.read_parquet(ROOT / "data/panel/entity_system.parquet")
    comp = ent.loc[ent["comparable"], ["campus", "system"]]
    wo = wo.merge(comp, on=["campus", "system"], how="inner")
    wo["entity"] = wo["system"].astype(str)
    led = pd.read_csv(results_path("p2_ledgers_entity.csv"))
    led = led.sort_values(["campus", "entity"]).reset_index(drop=True)
    return wo, led


def campus_builders(cfg, wo, led):
    wq = {int(k): (v[1] - v[0] + 1) * 4 for k, v in cfg["campus_valid_window"].items()}
    builders = {}
    for c in sorted(led["campus"].unique()):
        order = led.loc[led["campus"] == c, "entity"].tolist()
        builders[int(c)] = CampusBootstrap(
            int(c), wo[wo["campus"] == c], order, wq[int(c)],
            cfg["vol_min_years"], cfg["shock_tau_pct"],
        )
    return builders


# --------------------------------------------------------------------------
# Stage A: bootstrap
# --------------------------------------------------------------------------

def run_boot(cfg, B: int, jobs: int, batch_size: int = 50) -> None:
    wo, led = load_inputs(cfg)
    builders = campus_builders(cfg, wo, led)
    BOOT_DIR.mkdir(parents=True, exist_ok=True)
    seed = cfg["seed"]

    for c, cb in builders.items():
        out = BOOT_DIR / f"campus_{c}.npz"
        if out.exists():
            print(f"campus {c}: checkpoint exists, skip")
            continue
        bs = min(batch_size, B)
        n_batches = (B + bs - 1) // bs
        obs_sc = cb.observed_scores()
        obs_rk = rank_matrix(obs_sc)

        def one_batch(bi):
            return cb.run_batch((seed, c, bi), bs)

        results = Parallel(n_jobs=jobs, verbose=0)(
            delayed(one_batch)(bi) for bi in range(n_batches)
        )
        ranks = np.concatenate(results, axis=0)
        np.savez_compressed(
            out, ranks=ranks, obs_scores=obs_sc, obs_ranks=obs_rk,
            entities=np.array(cb.entities, dtype=object), B=B,
        )
        print(f"campus {c}: B={B} done, ranks shape {ranks.shape}")


# --------------------------------------------------------------------------
# Stage B helpers
# --------------------------------------------------------------------------

def load_boot():
    out = {}
    for f in sorted(BOOT_DIR.glob("campus_*.npz")):
        c = int(f.stem.split("_")[1])
        z = np.load(f, allow_pickle=True)
        out[c] = {k: z[k] for k in z.files}
    return out


def topk_sets(ranks_1d: np.ndarray, k: int) -> set:
    """Indices of the k largest values (NaN excluded)."""
    ok = np.flatnonzero(~np.isnan(ranks_1d))
    if len(ok) == 0:
        return set()
    order = ok[np.argsort(ranks_1d[ok])][::-1]
    return set(order[:k].tolist())


def k_of(n: int, spec) -> int:
    if isinstance(spec, float):
        return max(1, int(round(spec * n)))
    return min(spec, n)


def agreement_stats(led: pd.DataFrame, cfg) -> dict:
    """Kendall's W, tau-b matrix, top-k overlap, high-contrast share."""
    out = {}
    per_campus_w = {}
    for c, g in led.groupby("campus"):
        per_campus_w[int(c)] = round(float(kendalls_w(g[RANKS].to_numpy())), 4)
    out["W_per_campus"] = per_campus_w
    out["W_pooled"] = round(float(kendalls_w(led[RANKS].to_numpy())), 4)

    taub = np.full((5, 5), np.nan)
    for i, j in itertools.combinations(range(5), 2):
        m = led[[RANKS[i], RANKS[j]]].dropna()
        t = kendalltau(m[RANKS[i]], m[RANKS[j]]).statistic
        taub[i, j] = taub[j, i] = t
    np.fill_diagonal(taub, 1.0)
    out["taub_pooled"] = np.round(taub, 4).tolist()
    offdiag = taub[np.triu_indices(5, 1)]
    out["taub_mean_offdiag"] = round(float(np.nanmean(offdiag)), 4)
    out["taub_min_offdiag"] = round(float(np.nanmin(offdiag)), 4)

    overlaps = {}
    cores = {}
    for spec_name, spec in (("pct10", 0.10), ("pct20", 0.20), ("abs5", 5)):
        jacc = {f"{LEDGERS[i]}-{LEDGERS[j]}": [] for i, j in itertools.combinations(range(5), 2)}
        core_sizes = []
        core_shares = []
        for c, g in led.groupby("campus"):
            R = g[RANKS].to_numpy()
            n = len(g)
            k = k_of(n, spec)
            sets = [topk_sets(R[:, j], k) for j in range(5)]
            for i, j in itertools.combinations(range(5), 2):
                a, b = sets[i], sets[j]
                if a or b:
                    jacc[f"{LEDGERS[i]}-{LEDGERS[j]}"].append(len(a & b) / len(a | b))
            core = set.intersection(*sets)
            core_sizes.append(len(core))
            core_shares.append(len(core) / k)
        overlaps[spec_name] = {p: round(float(np.mean(v)), 4) for p, v in jacc.items()}
        overlaps[spec_name]["mean_all_pairs"] = round(
            float(np.mean([np.mean(v) for v in jacc.values()])), 4)
        cores[spec_name] = {
            "total_core_entities": int(np.sum(core_sizes)),
            "mean_core_share_of_k": round(float(np.mean(core_shares)), 4),
        }
    out["topk_jaccard"] = overlaps
    out["consensus_core"] = cores

    R = led[RANKS].to_numpy()
    hi = np.nanmax(R, axis=1) >= 0.9
    lo = np.nanmin(R, axis=1) < 0.5
    out["high_contrast_share"] = round(float((hi & lo).mean()), 4)
    out["high_contrast_count"] = int((hi & lo).sum())
    out["n_entities"] = int(len(led))
    return out


def h2_stat(R: np.ndarray) -> float:
    """Share of top-decile-by-L1 entities not top-decile under >= 2 other ledgers."""
    top_cost = R[:, 0] >= 0.9
    if top_cost.sum() == 0:
        return np.nan
    others = R[top_cost][:, 1:]
    fell = (others < 0.9).sum(axis=1) >= 2
    return float(fell.mean())


def wilson_ci(k: int, n: int, z: float = 1.959964) -> tuple:
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z ** 2 / n
    center = (p + z ** 2 / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / d
    return (round(center - half, 4), round(center + half, 4))


def n1_analysis(cfg, led, boot) -> tuple:
    """Ledger-matched noise floor (primary) + cost-only floor (secondary)."""
    rng = np.random.default_rng(np.random.SeedSequence((cfg["seed"], 991)))
    rows = []
    reps = {}
    for c, z in boot.items():
        ranks = z["ranks"]          # (B, E, 5)
        obs_rk = z["obs_ranks"]     # (E, 5)
        ents = list(z["entities"])
        B, E, _ = ranks.shape
        mean_rank = np.nanmean(obs_rk, axis=1)
        dev = ranks - obs_rk[None, :, :]
        recentred = np.clip(mean_rank[None, :, None] + dev, 1e-6, 1.0)
        null_meii = np.nanstd(recentred, axis=2, ddof=1)      # (B, E)
        floor95 = np.nanpercentile(null_meii, 95, axis=0)     # (E,)

        # cost-only secondary floor: 5 independent replicates of the L1 rank
        dc = dev[:, :, 0]                                     # (B, E)
        draws = rng.integers(0, B, size=(B, 5))
        five = np.clip(mean_rank[None, :, None] + dc[draws].transpose(0, 2, 1), 1e-6, 1.0)
        null_cost = np.nanstd(five, axis=2, ddof=1)
        floor95_cost = np.nanpercentile(null_cost, 95, axis=0)

        same_ledger_sd = np.nanstd(ranks, axis=0, ddof=1)     # (E, 5)
        obs_meii = np.nanstd(obs_rk, axis=1, ddof=1)
        for ei, e in enumerate(ents):
            rows.append({
                "campus": c, "entity": e,
                "meii_obs_boot": float(obs_meii[ei]),
                "mean_rank_obs": float(mean_rank[ei]),
                "noise_floor_p95": float(floor95[ei]),
                "noise_floor_median": float(np.nanmedian(null_meii[:, ei])),
                "significant": bool(obs_meii[ei] > floor95[ei]),
                "cost_floor_p95": float(floor95_cost[ei]),
                "significant_costfloor": bool(obs_meii[ei] > floor95_cost[ei]),
                **{f"rank_sd_{l}": float(same_ledger_sd[ei, j]) for j, l in enumerate(LEDGERS)},
            })
        reps[c] = {"ranks": ranks, "entities": ents}
    n1 = pd.DataFrame(rows)
    return n1, reps


def n2_analysis(cfg, led) -> dict:
    rng = np.random.default_rng(np.random.SeedSequence((cfg["seed"], 992)))
    N = cfg["permutation_N"]
    W_perm, meii_perm_mean, h2_perm = [], [], []
    campuses = list(led.groupby("campus"))
    for _ in range(N):
        pieces = []
        for c, g in campuses:
            R = g[RANKS].to_numpy().copy()
            for j in range(5):
                col = R[:, j]
                ok = ~np.isnan(col)
                col[ok] = rng.permutation(col[ok])
                R[:, j] = col
            pieces.append(R)
        Rp = np.vstack(pieces)
        W_perm.append(kendalls_w(Rp))
        meii_perm_mean.append(float(np.nanmean(np.nanstd(Rp, axis=1, ddof=1))))
        h2_perm.append(h2_stat(Rp))
    return {
        "W_perm_mean": round(float(np.nanmean(W_perm)), 4),
        "W_perm_p95": round(float(np.nanpercentile(W_perm, 95)), 4),
        "W_perm_p99": round(float(np.nanpercentile(W_perm, 99)), 4),
        "meii_perm_mean": round(float(np.nanmean(meii_perm_mean)), 4),
        "meii_perm_p5": round(float(np.nanpercentile(meii_perm_mean, 5)), 4),
        "meii_perm_p95": round(float(np.nanpercentile(meii_perm_mean, 95)), 4),
        "h2_perm_mean": round(float(np.nanmean(h2_perm)), 4),
    }


def n3_analysis(cfg, wo, led, n_splits: int = 100, jobs: int = 8) -> dict:
    builders = campus_builders(cfg, wo, led)

    def one_split(si):
        rng = np.random.default_rng(np.random.SeedSequence((cfg["seed"], 993, si)))
        rhoA_B = {l: [] for l in LEDGERS}
        allA, allB = [], []
        for c, cb in builders.items():
            wA, wB = [], []
            for n in cb.n:
                a = rng.integers(0, 2, size=n).astype(np.float64)
                if a.sum() == 0 or a.sum() == n:
                    a[rng.integers(0, n)] = 1 - a[rng.integers(0, n)]
                wA.append(a)
                wB.append(1.0 - a)
            rA = rank_matrix(cb.scores_for_weights(wA))
            rB = rank_matrix(cb.scores_for_weights(wB))
            allA.append(rA)
            allB.append(rB)
        A = np.vstack(allA)
        Bm = np.vstack(allB)
        out = {}
        for j, l in enumerate(LEDGERS):
            m = ~np.isnan(A[:, j]) & ~np.isnan(Bm[:, j])
            out[l] = float(spearmanr(A[m, j], Bm[m, j]).statistic) if m.sum() > 2 else np.nan
        return out

    res = Parallel(n_jobs=jobs)(delayed(one_split)(si) for si in range(n_splits))
    return {
        l: {
            "spearman_mean": round(float(np.nanmean([r[l] for r in res])), 4),
            "spearman_p5": round(float(np.nanpercentile([r[l] for r in res], 5)), 4),
            "spearman_p95": round(float(np.nanpercentile([r[l] for r in res], 95)), 4),
        }
        for l in LEDGERS
    }


def structure_analysis(cfg, wo, led) -> dict:
    upm = (
        wo.assign(is_upm=wo["wo_type"] == "UPM")
        .groupby(["campus", "entity"], observed=True)["is_upm"].mean()
        .rename("upm_share").reset_index()
    )
    df = led.merge(upm, on=["campus", "entity"], how="left")
    out = {}
    sys_groups = [g["meii"].dropna().to_numpy() for _, g in df.groupby("entity") if g["meii"].notna().sum() >= 4]
    if len(sys_groups) >= 3:
        st = kruskal(*sys_groups)
        out["kruskal_by_system"] = {"H": round(float(st.statistic), 3),
                                    "p": float(st.pvalue), "n_groups": len(sys_groups)}
    cam_groups = [g["meii"].dropna().to_numpy() for _, g in df.groupby("campus")]
    st = kruskal(*cam_groups)
    out["kruskal_by_campus"] = {"H": round(float(st.statistic), 3),
                                "p": float(st.pvalue), "n_groups": len(cam_groups)}
    for var, series in (
        ("log_n_wos", np.log10(df["n_wos"])),
        ("upm_share", df["upm_share"]),
        ("mean_rank", df["mean_rank"]),
    ):
        m = df["meii"].notna() & series.notna()
        r = spearmanr(df.loc[m, "meii"], series[m])
        out[f"spearman_meii_vs_{var}"] = {"rho": round(float(r.statistic), 4), "p": float(r.pvalue)}
    df.to_csv(results_path("p3_structure_entity.csv"), index=False)
    return out


RANKINGS = {
    "recorded_cost": lambda g: g["L1"],
    "wo_count": lambda g: g["n_wos"].astype(float),
    "labor_hours": lambda g: g["L2"],
    "mean_cost_per_wo": lambda g: g["L1"] / g["n_wos"],
    "shock_only": lambda g: g["L4"],
    "consensus": lambda g: g["mean_rank"],
}


def regret_matrix(led: pd.DataFrame, specs=(("pct10", 0.10), ("pct20", 0.20))) -> pd.DataFrame:
    rows = []
    for spec_name, spec in specs:
        for rname, rfun in RANKINGS.items():
            cb_by_ledger = {l: [] for l in LEDGERS}
            for c, g in led.groupby("campus"):
                g = g.reset_index(drop=True)
                k = k_of(len(g), spec)
                sel = topk_sets(rfun(g).to_numpy(dtype=float), k)
                for j, l in enumerate(LEDGERS):
                    burden = g[l].to_numpy(dtype=float)
                    oracle = topk_sets(burden, k)
                    denom = np.nansum(burden[list(oracle)]) if oracle else np.nan
                    num = np.nansum(burden[list(sel)]) if sel else 0.0
                    if denom and denom > 0:
                        cb_by_ledger[l].append(num / denom)
            row = {"k": spec_name, "ranking": rname}
            for l in LEDGERS:
                row[f"CB_{l}"] = round(float(np.mean(cb_by_ledger[l])), 4)
                row[f"regret_{l}"] = round(1 - row[f"CB_{l}"], 4)
            row["worst_regret"] = round(max(row[f"regret_{l}"] for l in LEDGERS), 4)
            rows.append(row)
    return pd.DataFrame(rows)


def boot_ci_stats(led, reps) -> dict:
    """Percentile CIs for headline stats from the joint replicate worlds."""
    order = led[["campus", "entity"]].copy()
    Bs = [reps[c]["ranks"].shape[0] for c in reps]
    B = min(Bs)
    W_b, hc_b, h2_b = [], [], []
    for b in range(B):
        pieces = []
        for c, g in order.groupby("campus"):
            ents = reps[int(c)]["entities"]
            R = reps[int(c)]["ranks"][b]
            eidx = [ents.index(e) for e in g["entity"]]
            pieces.append(R[eidx])
        Rb = np.vstack(pieces)
        W_b.append(kendalls_w(Rb))
        hi = np.nanmax(Rb, axis=1) >= 0.9
        lo = np.nanmin(Rb, axis=1) < 0.5
        hc_b.append(float((hi & lo).mean()))
        h2_b.append(h2_stat(Rb))
    def ci(v):
        return [round(float(np.nanpercentile(v, 2.5)), 4), round(float(np.nanpercentile(v, 97.5)), 4)]
    return {
        "W_pooled_ci95": ci(W_b), "W_boot_mean": round(float(np.nanmean(W_b)), 4),
        "high_contrast_share_ci95": ci(hc_b),
        "h2_stat_ci95": ci(h2_b),
    }


def run_analysis(cfg) -> None:
    wo, led = load_inputs(cfg)
    boot = load_boot()

    # consistency check: boot-internal observed scores vs p2 CSV
    max_rel = {}
    for c, z in boot.items():
        ents = list(z["entities"])
        g = led[led["campus"] == c].set_index("entity").loc[ents]
        p2v = g[LEDGERS].to_numpy(dtype=float)
        bv = z["obs_scores"]
        rel = np.abs(bv - p2v) / np.maximum(np.abs(p2v), 1e-12)
        max_rel[c] = float(np.nanmax(rel))
    assert max(max_rel.values()) < 1e-9, f"boot/p2 mismatch: {max_rel}"

    stab = agreement_stats(led, cfg)
    R = led[RANKS].to_numpy()
    stab["h2_stat"] = round(h2_stat(R), 4)

    n1, reps = n1_analysis(cfg, led, boot)
    meii = led.merge(n1, on=["campus", "entity"], how="left")
    assert (np.abs(meii["meii"] - meii["meii_obs_boot"]) < 1e-9).all()
    q1, q3 = meii["meii"].quantile([0.25, 0.75])
    meii["meii_class"] = np.where(meii["meii"] > q3, "high",
                          np.where(meii["meii"] < q1, "low", "medium"))
    meii.to_csv(results_path("p3_meii_entity.csv"), index=False)

    sig_share = float(meii["significant"].mean())
    nulls = {
        "N1": {
            "share_significant": round(sig_share, 4),
            "n_significant": int(meii["significant"].sum()),
            "n_entities": int(len(meii)),
            "share_significant_wilson_ci95": wilson_ci(int(meii["significant"].sum()), len(meii)),
            "share_significant_costfloor": round(float(meii["significant_costfloor"].mean()), 4),
            "mean_noise_floor_p95": round(float(meii["noise_floor_p95"].mean()), 4),
            "mean_rank_sd_by_ledger": {
                l: round(float(meii[f"rank_sd_{l}"].mean()), 4) for l in LEDGERS},
            "meii_quartiles": [round(float(q1), 4), round(float(q3), 4)],
        },
        "N2": n2_analysis(cfg, led),
        "N3": n3_analysis(cfg, wo, led),
    }
    stab.update(boot_ci_stats(led, reps))

    reg = regret_matrix(led)
    reg.to_csv(results_path("p3_regret_matrix.csv"), index=False)

    struct = structure_analysis(cfg, wo, led)

    wr = {r["ranking"]: r["worst_regret"] for _, r in reg[reg["k"] == "pct10"].iterrows()}
    wr20 = {r["ranking"]: r["worst_regret"] for _, r in reg[reg["k"] == "pct20"].iterrows()}
    hyps = {
        "H1": {
            "statement": "Cross-ledger concordance is moderate at best: pooled Kendall's W < 0.75.",
            "threshold": 0.75, "observed_W_pooled": stab["W_pooled"],
            "observed_W_ci95": stab["W_pooled_ci95"],
            "verdict": "supported" if stab["W_pooled"] < 0.75 else "not supported",
        },
        "H2": {
            "statement": ("At least 20% of top-decile-by-recorded-cost entities fall below "
                          "the top decile under at least two other ledgers."),
            "threshold": 0.20, "observed_share": stab["h2_stat"],
            "observed_ci95": stab["h2_stat_ci95"],
            "verdict": "supported" if stab["h2_stat"] >= 0.20 else "not supported",
        },
        "H3": {
            "statement": ("At least 10% of comparable entities are significantly "
                          "representation-sensitive relative to the N1 noise floor."),
            "threshold": 0.10, "observed_share": round(sig_share, 4),
            "observed_ci95": nulls["N1"]["share_significant_wilson_ci95"],
            "verdict": "supported" if sig_share >= 0.10 else "not supported",
        },
        "H4": {
            "statement": ("The consensus ranking's worst-case regret is at least 25% lower "
                          "than recorded-cost-only ranking's worst-case regret."),
            "threshold_ratio": 0.75,
            "observed_worst_regret_consensus_pct10": wr["consensus"],
            "observed_worst_regret_cost_pct10": wr["recorded_cost"],
            "observed_ratio_pct10": round(wr["consensus"] / wr["recorded_cost"], 4)
                if wr["recorded_cost"] > 0 else None,
            "observed_worst_regret_consensus_pct20": wr20["consensus"],
            "observed_worst_regret_cost_pct20": wr20["recorded_cost"],
            "observed_ratio_pct20": round(wr20["consensus"] / wr20["recorded_cost"], 4)
                if wr20["recorded_cost"] > 0 else None,
            "verdict": "supported" if wr["recorded_cost"] > 0
                and wr["consensus"] <= 0.75 * wr["recorded_cost"] else "not supported",
        },
    }

    write_json(stab, results_path("p3_stability.json"))
    write_json(nulls, results_path("p3_nulls.json"))
    write_json(struct, results_path("p3_structure.json"))
    write_json(hyps, results_path("p3_hypotheses.json"))
    print("Phase 3 analysis complete.")
    print(f"  W pooled = {stab['W_pooled']} (CI {stab['W_pooled_ci95']}), per campus {stab['W_per_campus']}")
    print(f"  high-contrast share = {stab['high_contrast_share']}, H2 stat = {stab['h2_stat']}")
    print(f"  N1 significant share = {sig_share:.3f} (cost-floor {nulls['N1']['share_significant_costfloor']})")
    print(f"  N2 W ceiling p95 = {nulls['N2']['W_perm_p95']}, N3: " +
          ", ".join(f"{l}={nulls['N3'][l]['spearman_mean']}" for l in LEDGERS))
    print("  verdicts:", {h: v["verdict"] for h, v in hyps.items()})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--B", type=int, default=None)
    ap.add_argument("--jobs", type=int, default=20)
    args = ap.parse_args()
    cfg = load_config()
    B = args.B or cfg["bootstrap_B"]
    if args.boot:
        run_boot(cfg, B, args.jobs)
    if args.analyze:
        run_analysis(cfg)
