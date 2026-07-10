"""Phase 7/8: manuscript tables T1-T6.

Each table is written as tables/T{n}.csv (data) and tables/T{n}.tex (booktabs
LaTeX body, \input into the manuscript).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils.io import ROOT, load_config, results_path

TAB = ROOT / "tables"


def jload(name):
    with open(results_path(name)) as fh:
        return json.load(fh)


def write_tex(name: str, body: str) -> None:
    TAB.mkdir(exist_ok=True)
    (TAB / f"{name}.tex").write_text(body)
    print(f"wrote {name}.tex")


# ------------------------------------------------------------------ T1
def t1_dataset():
    cfg = load_config()
    wo = pd.read_parquet(ROOT / "data/interim/wo_clean.parquet")
    ent = pd.read_parquet(ROOT / "data/panel/entity_system.parquet")
    ent = ent[ent["system"].astype(str).str.strip() != ""]
    rows = []
    for c in cfg["campuses_retained"]:
        w = wo[wo["campus"] == c]
        e = ent[ent["campus"] == c]
        y0, y1 = cfg["campus_valid_window"][c]
        rows.append({
            "Campus": f"U{c:02d}",
            "Country": "Canada" if c in cfg["campuses_cad"] else "USA",
            "Window": f"{y0}--{y1}",
            "Work orders": f"{len(w):,}",
            "Cost coverage (%)": f"{100 * w['cost'].notna().mean():.1f}",
            "Labor coverage (%)": f"{100 * w['labor_hours'].notna().mean():.1f}",
            "UPM share (%)": f"{100 * (w['wo_type'] == 'UPM').mean():.1f}",
            "Systems": len(e),
            "Comparable": int(e["comparable"].sum()),
        })
    df = pd.DataFrame(rows)
    n_total = wo[wo["campus"].isin(cfg["campuses_retained"])].shape[0]
    tot = {
        "Campus": "Total", "Country": "", "Window": "",
        "Work orders": f"{n_total:,}",
        "Cost coverage (%)": "", "Labor coverage (%)": "", "UPM share (%)": "",
        "Systems": int(df["Systems"].sum()),
        "Comparable": int(df["Comparable"].sum()),
    }
    df = pd.concat([df, pd.DataFrame([tot])], ignore_index=True)
    df.to_csv(TAB / "T1.csv", index=False)
    body = df.to_latex(index=False, escape=True, column_format="llrrrrrrr")
    write_tex("T1", body)


# ------------------------------------------------------------------ T2
def t2_ledgers():
    rows = [
        ("L1 Recorded cost", r"$\sum_{w \in W(i)} x_w$",
         "Cash expenditure and budget reporting", "Cost field"),
        ("L2 Labor burden", r"$\sum_{w \in W(i)} h_w$",
         "Workforce planning and staffing", "Labor hours"),
        ("L3 Chronicity", r"$P_i \cdot F_i$ (active-quarter share $\times$ WO count)",
         "Recurring nuisance; process improvement", "Dates"),
        ("L4 Shock burden", r"$\sum_{w} x_w \mathbb{1}[x_w > \tau_{c}]$, $\tau_c$ = campus p95",
         "Contingency reserves; budget-risk exposure", "Cost field"),
        ("L5 Budget volatility", r"$\mathrm{SD}(C_{i,y}) / \mathrm{mean}(C_{i,y})$",
         "Budget predictability; financial planning", "Cost + dates, $\\geq 5$ years"),
        ("L6 Total burden (derived)", r"$\mathrm{L1}^{2021} + \rho \cdot \mathrm{L2}$",
         "Decision case only; never in MEII", "Cost + labor + wage rate"),
    ]
    df = pd.DataFrame(rows, columns=["Ledger", "Definition", "Decision view", "Requires"])
    df.to_csv(TAB / "T2.csv", index=False)
    body = df.to_latex(index=False, escape=False, column_format="lp{4.2cm}p{4.2cm}p{2.4cm}")
    write_tex("T2", body)


# ------------------------------------------------------------------ T3
def t3_stability():
    stab = jload("p3_stability.json")
    nulls = jload("p3_nulls.json")
    hyps = jload("p3_hypotheses.json")
    wpc = stab["W_per_campus"]
    rows = [
        ("Kendall's W, pooled (95\\% CI)",
         f"{stab['W_pooled']:.3f} ({stab['W_pooled_ci95'][0]:.3f}, {stab['W_pooled_ci95'][1]:.3f})",
         f"N2 ceiling p95: {nulls['N2']['W_perm_p95']:.3f}",
         f"H1 (< 0.75): {hyps['H1']['verdict']}"),
        ("Kendall's W, campus range",
         f"{min(wpc.values()):.3f}--{max(wpc.values()):.3f}", "", ""),
        ("Mean pairwise tau-b (off-diagonal)",
         f"{stab['taub_mean_offdiag']:.3f}",
         f"min {stab['taub_min_offdiag']:.3f} (L3--L5)", ""),
        ("Top-10\\% Jaccard overlap, mean pair",
         f"{stab['topk_jaccard']['pct10']['mean_all_pairs']:.3f}",
         f"top-20\\%: {stab['topk_jaccard']['pct20']['mean_all_pairs']:.3f}; "
         f"top-5: {stab['topk_jaccard']['abs5']['mean_all_pairs']:.3f}", ""),
        ("Consensus core (share of k, top-10\\%)",
         f"{stab['consensus_core']['pct10']['mean_core_share_of_k']:.3f}",
         f"top-5: {stab['consensus_core']['abs5']['mean_core_share_of_k']:.3f}", ""),
        ("High-contrast entities (95\\% CI)",
         f"{stab['high_contrast_share']*100:.1f}\\% "
         f"({stab['high_contrast_share_ci95'][0]*100:.1f}, {stab['high_contrast_share_ci95'][1]*100:.1f})",
         "top decile somewhere, below median elsewhere", ""),
        ("Cost top-decile leavers (95\\% CI)",
         f"{stab['h2_stat']*100:.1f}\\% "
         f"({stab['h2_stat_ci95'][0]*100:.1f}, {stab['h2_stat_ci95'][1]*100:.1f})",
         "out of top decile under $\\geq 2$ other ledgers",
         f"H2 ($\\geq$ 20\\%): {hyps['H2']['verdict']}"),
        ("MEII significant vs N1 floor (Wilson CI)",
         f"{nulls['N1']['share_significant']*100:.1f}\\% "
         f"({nulls['N1']['share_significant_wilson_ci95'][0]*100:.1f}, "
         f"{nulls['N1']['share_significant_wilson_ci95'][1]*100:.1f})",
         f"cost-only floor: {nulls['N1']['share_significant_costfloor']*100:.1f}\\%",
         f"H3 ($\\geq$ 10\\%): {hyps['H3']['verdict']}"),
        ("Split-half consistency (N3, Spearman)",
         "L1 " + f"{nulls['N3']['L1']['spearman_mean']:.2f}" +
         ", L2 " + f"{nulls['N3']['L2']['spearman_mean']:.2f}" +
         ", L3 " + f"{nulls['N3']['L3']['spearman_mean']:.2f}",
         "L4 " + f"{nulls['N3']['L4']['spearman_mean']:.2f}" +
         ", L5 " + f"{nulls['N3']['L5']['spearman_mean']:.2f}", ""),
    ]
    df = pd.DataFrame(rows, columns=["Metric", "Value", "Reference", "Hypothesis"])
    df.to_csv(TAB / "T3.csv", index=False)
    body = df.to_latex(index=False, escape=False, column_format="p{4.6cm}p{3.6cm}p{3.6cm}p{2.6cm}")
    write_tex("T3", body)


# ------------------------------------------------------------------ T4
def t4_archetypes():
    summ = jload("p4_archetype_summary.json")["system"]
    ex = jload("p4_examples.json")
    label = {
        "stable_priority": "Stable priority", "hidden_burden": "Hidden burden",
        "cash_sink": "Cash sink", "labor_sink": "Labor sink",
        "chronic_drain": "Chronic drain", "shock_generator": "Shock generator",
        "budget_destabiliser": "Budget destabilizer",
        "representation_sensitive": "Representation-sensitive",
        "unremarkable": "Unremarkable",
    }
    rows = []
    for a, lab in label.items():
        n = summ["counts"][a]
        bw = summ["burden_weighted_shares"].get(a, 0.0)
        example = ""
        if a in ex and ex[a]:
            e = ex[a][0]
            pos = e["position_by_ledger"]
            lname = {"L2": "labor", "L3": "chronicity", "L4": "shock", "L5": "volatility"}
            noncost = [(v, k) for k, v in pos.items() if v is not None and k != "L1"]
            bv, bk = min(noncost)
            example = (f"{e['campus']} {e['system_desc']}: "
                       f"{pos['L1']} of {e['n_entities_campus']} by cost, "
                       f"{bv} by {lname[bk]}")
        rows.append({
            "Archetype": lab, "Entities": n,
            "Share (%)": f"{100 * summ['shares'][a]:.1f}",
            "Cost-weighted (%)": f"{100 * bw:.1f}",
            "Example": example,
        })
    df = pd.DataFrame(rows)
    df.to_csv(TAB / "T4.csv", index=False)
    body = df.to_latex(index=False, escape=True, column_format="lrrrp{6.2cm}")
    write_tex("T4", body)


# ------------------------------------------------------------------ T5
def t5_regret():
    reg = pd.read_csv(results_path("p3_regret_matrix.csv"))
    label = {
        "recorded_cost": "Recorded cost", "wo_count": "WO count",
        "labor_hours": "Labor hours", "mean_cost_per_wo": "Mean cost per WO",
        "shock_only": "Shock only", "consensus": "Consensus (mean rank)",
    }
    out = []
    for k in ("pct10", "pct20"):
        sub = reg[reg["k"] == k]
        for _, r in sub.iterrows():
            out.append({
                "k": "top 10\\%" if k == "pct10" else "top 20\\%",
                "Ranking": label[r["ranking"]],
                "Cost": f"{r['regret_L1']:.3f}", "Labor": f"{r['regret_L2']:.3f}",
                "Chronicity": f"{r['regret_L3']:.3f}", "Shock": f"{r['regret_L4']:.3f}",
                "Volatility": f"{r['regret_L5']:.3f}",
                "Worst case": f"\\textbf{{{r['worst_regret']:.3f}}}",
            })
    df = pd.DataFrame(out)
    df.to_csv(TAB / "T5.csv", index=False)
    body = df.to_latex(index=False, escape=False, column_format="llrrrrrr")
    write_tex("T5", body)


# ------------------------------------------------------------------ T6
def t6_robustness():
    summ = jload("p6_robustness_summary.json")
    stab = jload("p3_stability.json")
    nulls = jload("p3_nulls.json")
    rows = [{
        "Check": "Main analysis",
        "W": f"{stab['W_pooled']:.3f}",
        "High-contrast (\\%)": f"{stab['high_contrast_share']*100:.1f}",
        "H2 stat (\\%)": f"{stab['h2_stat']*100:.1f}",
        "N1 sig. (\\%)": f"{jload('p3_nulls.json')['N1']['share_significant']*100:.1f}",
        "Verdicts": "H1+, H2+, H3+, H4$-$",
    }]

    def verd(m):
        v = []
        v.append("H1+" if m.get("H1_verdict") == "supported" else "H1-")
        v.append("H2+" if m.get("H2_verdict") == "supported" else "H2-")
        if "n1_sig_share" in m:
            v.append("H3+" if m["n1_sig_share"] >= 0.10 else "H3-")
        if "H4_verdict" in m:
            v.append("H4+" if m["H4_verdict"] == "supported" else "H4-")
        return ", ".join(v)

    r1, r2 = summ["R1"], summ["R2"]
    rows.append({
        "Check": "R1 leave-one-campus-out (range)",
        "W": f"{r1['W_range'][0]:.3f}--{r1['W_range'][1]:.3f}",
        "High-contrast (\\%)": f"{r1['high_contrast_range'][0]*100:.1f}--{r1['high_contrast_range'][1]*100:.1f}",
        "H2 stat (\\%)": f"{r1['h2_range'][0]*100:.1f}--{r1['h2_range'][1]*100:.1f}",
        "N1 sig. (\\%)": f"{r1['n1_sig_share_range'][0]*100:.1f}--{r1['n1_sig_share_range'][1]*100:.1f}",
        "Verdicts": "unchanged" if r1["verdicts_unchanged"] else "changed",
    })
    rows.append({
        "Check": "R2 leave-one-year-out (range)",
        "W": f"{r2['W_range'][0]:.3f}--{r2['W_range'][1]:.3f}",
        "High-contrast (\\%)": "",
        "H2 stat (\\%)": f"{r2['h2_range'][0]*100:.1f}--{r2['h2_range'][1]*100:.1f}",
        "N1 sig. (\\%)": f"{r2['n1_sig_share_range'][0]*100:.1f}--{r2['n1_sig_share_range'][1]*100:.1f}",
        "Verdicts": "unchanged" if r2["verdicts_unchanged"] else "changed",
    })
    for tag, lab in (("p90", "R5 shock tau = p90"), ("p99", "R5 shock tau = p99"),
                     ("count_p95", "R5 shock count-based")):
        m = summ["R5"][tag]
        rows.append({
            "Check": lab, "W": f"{m['W_pooled']:.3f}",
            "High-contrast (\\%)": f"{m['high_contrast_share']*100:.1f}",
            "H2 stat (\\%)": f"{m['h2_stat']*100:.1f}",
            "N1 sig. (\\%)": "",
            "Verdicts": verd(m),
        })
    for tag, lab in (("R6", "R6 zeros as missing"), ("R7", "R7 extremes excluded")):
        m = summ[tag]
        rows.append({
            "Check": lab, "W": f"{m['W_pooled']:.3f}",
            "High-contrast (\\%)": f"{m['high_contrast_share']*100:.1f}",
            "H2 stat (\\%)": f"{m['h2_stat']*100:.1f}",
            "N1 sig. (\\%)": f"{m['n1_sig_share']*100:.1f}",
            "Verdicts": verd(m),
        })
    for sub in ("PPM", "UPM"):
        m = summ["R12"][sub]
        rows.append({
            "Check": f"R12 {sub} only", "W": f"{m['W_pooled']:.3f}",
            "High-contrast (\\%)": f"{m['high_contrast_share']*100:.1f}",
            "H2 stat (\\%)": f"{m['h2_stat']*100:.1f}",
            "N1 sig. (\\%)": f"{m['n1_sig_share']*100:.1f}",
            "Verdicts": verd(m),
        })
    df = pd.DataFrame(rows)
    df.to_csv(TAB / "T6.csv", index=False)
    body = df.to_latex(index=False, escape=False, column_format="lccccl")
    write_tex("T6", body)


if __name__ == "__main__":
    t1_dataset()
    t2_ledgers()
    t3_stability()
    t4_archetypes()
    t5_regret()
    t6_robustness()
    print("Tables complete.")
