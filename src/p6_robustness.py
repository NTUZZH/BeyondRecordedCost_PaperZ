"""Phase 6: robustness suite R1-R12.

Each check writes results/p6_R{n}.json (plus CSVs where useful);
results/p6_robustness_summary.json aggregates the manuscript table rows.

Usage: python p6_robustness.py [--only R1,R2,...] [--jobs 8]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils.io import ROOT, load_config, results_path, write_json
from utils.boot import kendalls_w
from utils.pipeline import ARCH_RANKS, comparability, variant_metrics, variant_n1_share
from p3_stability import h2_stat, k_of, topk_sets

LEDGERS = ["L1", "L2", "L3", "L4", "L5"]


def load_wo() -> pd.DataFrame:
    wo = pd.read_parquet(ROOT / "data/interim/wo_clean.parquet").rename(
        columns={"labor_hours": "labor"})
    wo = wo[wo["system"].str.strip() != ""].copy()
    wo["entity"] = wo["system"].astype(str)
    return wo


def main_comparable() -> pd.DataFrame:
    led = pd.read_csv(results_path("p2_ledgers_entity.csv"))
    return led[["campus", "entity"]]


def main_labels() -> pd.Series:
    arch = pd.read_csv(results_path("p4_archetypes.csv"))
    return arch.set_index(["campus", "entity"])["archetype"]


def strip_private(m: dict) -> dict:
    return {k: v for k, v in m.items() if not k.startswith("_")}


def label_agreement(m: dict, base: pd.Series, scores: pd.DataFrame) -> float:
    lab = m.get("_archetypes")
    if lab is None:
        return np.nan
    cur = pd.Series(lab.values, index=pd.MultiIndex.from_frame(scores[["campus", "entity"]]))
    common = cur.index.intersection(base.index)
    return round(float((cur.loc[common] == base.loc[common]).mean()), 4)


# ----------------------------------------------------------------------
def r1_loco(cfg, wo, comp, base_labels, jobs) -> dict:
    """Leave-one-campus-out: pooled metrics on 5-campus subsets."""
    led = pd.read_csv(results_path("p2_ledgers_entity.csv"))
    meii = pd.read_csv(results_path("p3_meii_entity.csv"))
    out = {"per_left_out": {}}
    Ws, hcs, h2s, sig_shares, lab_stab = [], [], [], [], []
    from p4_archetypes import classify
    for c in sorted(led["campus"].unique()):
        sub = led[led["campus"] != c].reset_index(drop=True)
        subm = meii[meii["campus"] != c]
        R = sub[ARCH_RANKS].to_numpy()
        W = float(kendalls_w(R))
        hc = float(((np.nanmax(R, axis=1) >= 0.9) & (np.nanmin(R, axis=1) < 0.5)).mean())
        h2 = float(h2_stat(R))
        sig = float(subm["significant"].mean())
        q3 = float(sub["meii"].quantile(0.75))
        lab = classify(sub, cfg["theta_high"], cfg["theta_low"], q3)
        cur = pd.Series(lab.values, index=pd.MultiIndex.from_frame(sub[["campus", "entity"]]))
        common = cur.index.intersection(base_labels.index)
        agree = float((cur.loc[common] == base_labels.loc[common]).mean())
        Ws.append(W); hcs.append(hc); h2s.append(h2); sig_shares.append(sig); lab_stab.append(agree)
        out["per_left_out"][int(c)] = {
            "W_pooled": round(W, 4), "high_contrast_share": round(hc, 4),
            "h2_stat": round(h2, 4), "n1_sig_share": round(sig, 4),
            "archetype_agreement": round(agree, 4),
        }
    out["W_range"] = [round(min(Ws), 4), round(max(Ws), 4)]
    out["h2_range"] = [round(min(h2s), 4), round(max(h2s), 4)]
    out["high_contrast_range"] = [round(min(hcs), 4), round(max(hcs), 4)]
    out["n1_sig_share_range"] = [round(min(sig_shares), 4), round(max(sig_shares), 4)]
    out["archetype_agreement_min"] = round(min(lab_stab), 4)
    out["verdicts_unchanged"] = bool(
        max(Ws) < 0.75 and min(h2s) >= 0.20 and min(sig_shares) >= 0.10)
    return out


def r2_loyo(cfg, wo, comp, base_labels, jobs) -> dict:
    """Leave-one-year-out: full recompute + N1 bootstrap per left-out year."""
    windows = {int(k): v for k, v in cfg["campus_valid_window"].items()}
    years = sorted({y for v in windows.values() for y in range(v[0], v[1] + 1)})
    out = {"per_left_out_year": {}}
    Ws, h2s, sigs, labs = [], [], [], []
    for y in years:
        sub = wo[wo["year"] != y]
        m, scores = variant_metrics(sub, cfg, refilter=True)
        sig = variant_n1_share(sub, cfg, comparability(sub, cfg), windows, ("r2", y), jobs=jobs)
        agree = label_agreement(m, base_labels, scores)
        Ws.append(m["W_pooled"]); h2s.append(m["h2_stat"]); sigs.append(sig); labs.append(agree)
        out["per_left_out_year"][int(y)] = {
            **{k: m[k] for k in ("W_pooled", "high_contrast_share", "h2_stat",
                                 "regret_ratio", "n_entities")},
            "n1_sig_share": round(sig, 4), "archetype_agreement": agree,
        }
    out["W_range"] = [round(min(Ws), 4), round(max(Ws), 4)]
    out["h2_range"] = [round(min(h2s), 4), round(max(h2s), 4)]
    out["n1_sig_share_range"] = [round(min(sigs), 4), round(max(sigs), 4)]
    out["archetype_agreement_min"] = round(float(np.nanmin(labs)), 4)
    out["verdicts_unchanged"] = bool(
        max(Ws) < 0.75 and min(h2s) >= 0.20 and min(sigs) >= 0.10)
    return out


def r3_system_categories(cfg, wo, comp, base_labels, jobs) -> dict:
    """Cross-ledger agreement within major system categories (pooled entities)."""
    led = pd.read_csv(results_path("p2_ledgers_entity.csv"))
    meii = pd.read_csv(results_path("p3_meii_entity.csv"))[["campus", "entity", "significant"]]
    led = led.merge(meii, on=["campus", "entity"])
    out = {}
    for sys_code, g in led.groupby("entity"):
        if len(g) < 5:
            continue
        R = g[ARCH_RANKS].to_numpy()
        out[sys_code] = {
            "n_entities": int(len(g)),
            "W": round(float(kendalls_w(R)), 4),
            "meii_mean": round(float(g["meii"].mean()), 4),
            "sig_share": round(float(g["significant"].mean()), 4),
        }
    return out


def r4_wage_sweep(cfg, wo, comp, base_labels, jobs) -> dict:
    led = pd.read_csv(results_path("p2_ledgers_entity.csv"))
    lo, hi = cfg["wage_sweep"]
    out = {"sweep": {}}
    base_sel = {}
    for c, g in led.groupby("campus"):
        g = g.reset_index(drop=True)
        base_sel[c] = topk_sets(g["L6"].to_numpy(dtype=float), k_of(len(g), cfg["topk_abs"]))
    for rho in range(lo, hi + 1, 5):
        ov, cap_labor, cap_cost = [], [], []
        for c, g in led.groupby("campus"):
            g = g.reset_index(drop=True)
            l6 = (g["L1_2021"] + rho * g["L2"]).to_numpy(dtype=float)
            k = k_of(len(g), cfg["topk_abs"])
            sel = topk_sets(l6, k)
            ov.append(len(sel & base_sel[c]) / k)
            for col, acc in (("L2", cap_labor), ("L1", cap_cost)):
                burden = g[col].to_numpy(dtype=float)
                acc.append(np.nansum(burden[list(sel)]) / np.nansum(burden))
        out["sweep"][rho] = {
            "overlap_with_default_L6_top5": round(float(np.mean(ov)), 4),
            "captured_labor": round(float(np.mean(cap_labor)), 4),
            "captured_cost": round(float(np.mean(cap_cost)), 4),
        }
    ovs = [v["overlap_with_default_L6_top5"] for v in out["sweep"].values()]
    out["min_overlap"] = round(min(ovs), 4)
    return out


def r5_shock_threshold(cfg, wo, comp, base_labels, jobs) -> dict:
    base = pd.read_csv(results_path("p2_ledgers_entity.csv")).sort_values(
        ["campus", "entity"]).reset_index(drop=True)
    out = {}
    for tag, kw in (("p90", dict(tau_pct=90)), ("p99", dict(tau_pct=99)),
                    ("count_p95", dict(shock_stat="count"))):
        m, scores = variant_metrics(wo, cfg, comparable=comp, refilter=False, **kw)
        merged = base.merge(scores, on=["campus", "entity"], suffixes=("_base", "_var"))
        tb = float(kendalltau(merged["r_L4_base"], merged["r_L4_var"]).statistic)
        dm = float((merged["meii_var"] - merged["meii_base"]).abs().mean())
        out[tag] = {
            **{k: m[k] for k in ("W_pooled", "high_contrast_share", "h2_stat",
                                 "regret_ratio", "H1_verdict", "H2_verdict", "H4_verdict")},
            "taub_L4_vs_base": round(tb, 4),
            "mean_abs_meii_shift": round(dm, 4),
            "archetype_agreement": label_agreement(m, base_labels, scores),
        }
    return out


def r6_zero_rule(cfg, wo, comp, base_labels, jobs) -> dict:
    sub = wo.copy()
    sub.loc[sub["cost"] == 0, "cost"] = np.nan
    sub.loc[sub["cost2021"] == 0, "cost2021"] = np.nan
    m, scores = variant_metrics(sub, cfg, refilter=True)
    windows = {int(k): v for k, v in cfg["campus_valid_window"].items()}
    sig = variant_n1_share(sub, cfg, comparability(sub, cfg), windows, ("r6",), jobs=jobs)
    return {**strip_private(m), "n1_sig_share": round(sig, 4),
            "archetype_agreement": label_agreement(m, base_labels, scores)}


def r7_extremes(cfg, wo, comp, base_labels, jobs) -> dict:
    sub = wo[~wo["extreme_flag"]]
    m, scores = variant_metrics(sub, cfg, comparable=comp, refilter=False)
    windows = {int(k): v for k, v in cfg["campus_valid_window"].items()}
    sig = variant_n1_share(sub, cfg, comp, windows, ("r7",), jobs=jobs)
    return {**strip_private(m), "n1_sig_share": round(sig, 4),
            "archetype_agreement": label_agreement(m, base_labels, scores)}


def r8_chronicity(cfg, wo, comp, base_labels, jobs) -> dict:
    base = pd.read_csv(results_path("p2_ledgers_entity.csv")).sort_values(
        ["campus", "entity"]).reset_index(drop=True)
    out = {}
    for var in ("count", "active_quarters", "median_nonzero"):
        m, scores = variant_metrics(wo, cfg, comparable=comp, refilter=False, chron_variant=var)
        merged = base.merge(scores, on=["campus", "entity"], suffixes=("_base", "_var"))
        out[var] = {
            "taub_L3_vs_base": round(float(kendalltau(
                merged["r_L3_base"], merged["r_L3_var"]).statistic), 4),
            "mean_abs_meii_shift": round(float(
                (merged["meii_var"] - merged["meii_base"]).abs().mean()), 4),
            "W_pooled": m["W_pooled"],
            "archetype_agreement": label_agreement(m, base_labels, scores),
        }
    return out


def r9_labor_cost(cfg, wo, comp, base_labels, jobs) -> dict:
    base = pd.read_csv(results_path("p2_ledgers_entity.csv")).sort_values(
        ["campus", "entity"]).reset_index(drop=True)
    m, scores = variant_metrics(wo, cfg, comparable=comp, refilter=False, labor_col="labor_cost")
    merged = base.merge(scores, on=["campus", "entity"], suffixes=("_base", "_var"))
    return {
        "taub_L2_hours_vs_laborcost": round(float(kendalltau(
            merged["r_L2_base"], merged["r_L2_var"]).statistic), 4),
        "spearman_L2": round(float(spearmanr(
            merged["r_L2_base"], merged["r_L2_var"]).statistic), 4),
        "W_pooled": m["W_pooled"],
        "mean_abs_meii_shift": round(float(
            (merged["meii_var"] - merged["meii_base"]).abs().mean()), 4),
    }


def r10_quarterly_vol(cfg, wo, comp, base_labels, jobs) -> dict:
    base = pd.read_csv(results_path("p2_ledgers_entity.csv")).sort_values(
        ["campus", "entity"]).reset_index(drop=True)
    m, scores = variant_metrics(wo, cfg, comparable=comp, refilter=False, vol_variant="cv_quarterly")
    merged = base.merge(scores, on=["campus", "entity"], suffixes=("_base", "_var"))
    ok = merged[["r_L5_base", "r_L5_var"]].dropna()
    out = {
        "taub_L5_annual_vs_quarterly": round(float(kendalltau(
            ok["r_L5_base"], ok["r_L5_var"]).statistic), 4),
        "W_pooled": m["W_pooled"],
        "mean_abs_meii_shift": round(float(
            (merged["meii_var"] - merged["meii_base"]).abs().mean()), 4),
        "n_with_L5_quarterly": int(scores["L5"].notna().sum()),
    }
    for var in ("mad_median",):
        m2, scores2 = variant_metrics(wo, cfg, comparable=comp, refilter=False, vol_variant=var)
        merged2 = base.merge(scores2, on=["campus", "entity"], suffixes=("_base", "_var"))
        ok2 = merged2[["r_L5_base", "r_L5_var"]].dropna()
        out["taub_L5_annual_vs_mad"] = round(float(kendalltau(
            ok2["r_L5_base"], ok2["r_L5_var"]).statistic), 4)
    return out


def r11_full_window(cfg, wo, comp, base_labels, jobs) -> dict:
    """Full substantial windows (pre-core years back in) for U5 and U9."""
    full_windows = {1: [2014, 2019], 2: [2016, 2020], 4: [2015, 2019],
                    5: [2003, 2020], 8: [2014, 2020], 9: [2013, 2020]}
    # rebuild window-extended WO table from raw clean logic: reuse interim file
    # is impossible (rows outside core were dropped in p1), so rebuild via p1
    from p1_panel import clean_workorders
    cfg_full = dict(cfg)
    cfg_full["campus_valid_window"] = {str(k): v for k, v in full_windows.items()}
    df, _ = clean_workorders(cfg_full)
    df = df.rename(columns={"labor_hours": "labor"})
    df = df[df["system"].str.strip() != ""].copy()
    df["entity"] = df["system"].astype(str)
    m, scores = variant_metrics(df, cfg, windows=full_windows, refilter=True)
    windows = {int(k): v for k, v in full_windows.items()}
    sig = variant_n1_share(df, cfg, comparability(df, cfg), windows, ("r11",), jobs=jobs)
    return {**strip_private(m), "n1_sig_share": round(sig, 4),
            "archetype_agreement": label_agreement(m, base_labels, scores)}


def r12_ppm_upm(cfg, wo, comp, base_labels, jobs) -> dict:
    out = {}
    for sub_name in ("PPM", "UPM"):
        sub = wo[wo["wo_type"] == sub_name]
        m, scores = variant_metrics(sub, cfg, refilter=True)
        windows = {int(k): v for k, v in cfg["campus_valid_window"].items()}
        sig = variant_n1_share(sub, cfg, comparability(sub, cfg), windows,
                               ("r12", sub_name), jobs=jobs)
        out[sub_name] = {**strip_private(m), "n1_sig_share": round(sig, 4)}
    return out


CHECKS = {
    "R1": r1_loco, "R2": r2_loyo, "R3": r3_system_categories, "R4": r4_wage_sweep,
    "R5": r5_shock_threshold, "R6": r6_zero_rule, "R7": r7_extremes,
    "R8": r8_chronicity, "R9": r9_labor_cost, "R10": r10_quarterly_vol,
    "R11": r11_full_window, "R12": r12_ppm_upm,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", type=str, default=None)
    ap.add_argument("--jobs", type=int, default=8)
    args = ap.parse_args()
    cfg = load_config()
    wo = load_wo()
    comp = main_comparable()
    base_labels = main_labels()

    todo = args.only.split(",") if args.only else list(CHECKS)
    summary = {}
    for name in todo:
        print(f"--- {name} ---", flush=True)
        res = CHECKS[name](cfg, wo, comp, base_labels, args.jobs)
        res = {k: v for k, v in res.items() if not k.startswith("_")}
        write_json(res, results_path(f"p6_{name}.json"))
        summary[name] = res
        print(f"{name} done", flush=True)

    # merge with any previously computed checks
    import json
    full = {}
    for name in CHECKS:
        f = results_path(f"p6_{name}.json")
        if f.exists():
            with open(f) as fh:
                full[name] = json.load(fh)
    write_json(full, results_path("p6_robustness_summary.json"))
    print("Phase 6 summary written for:", sorted(full))


if __name__ == "__main__":
    main()
