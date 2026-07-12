"""Phase 8b: revised manuscript tables for the two-layer framework.

T2  criteria table (management question, what it measures, units,
    reliability) -- main text
T3  framework validation results with decision interpretation -- main text
T4  two-dimensional profile prevalence -- main text
T5  decision-rule comparison (worst-case and risk-criterion regret,
    additive captured shares) -- main text
A6  simulation validation detail -- Supplementary Table S3
A7  criterion reliability diagnostics -- Supplementary Table S2
A8  completeness/missing-data sweep -- Supplementary Table S4
A9  full criterion decision mapping (stakeholder, decision, additivity,
    size relation) -- Supplementary Table S5
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from p8_tables import TAB, jload, write_tex
from utils.io import results_path

RULE_NAMES = {
    "single_cost": "Recorded cost only", "single_labor": "Labor hours only",
    "single_volume": "Persistent volume only",
    "single_tail": "Tail expenditure only",
    "single_risk": "Budget risk only", "burden_mean": "Burden mean rank",
    "burden_median": "Burden median rank", "all_mean": "All-criteria mean rank",
    "minimax": "Minimax-regret shortlist",
}

PROFILE_LABELS = {
    "stable_aligned": "Metric-stable / risk-aligned",
    "stable_divergent": "Metric-stable / repositioned by risk",
    "unstable_aligned": "Metric-sensitive / risk-aligned",
    "unstable_divergent": "Metric-sensitive / repositioned by risk",
}


def t2_criteria():
    d2 = jload("p2b_diagnostics.json")
    sh = d2["split_half"]
    rows = [
        ("Recorded cost",
         "Where was the greatest expenditure recorded?",
         "Total constant-currency cost", "Currency",
         f"{sh['L1']['mean']:.2f}"),
        ("Labor hours",
         "Which systems consumed the most technician effort?",
         "Total labor hours", "Hours", f"{sh['L2']['mean']:.2f}"),
        ("Persistent volume",
         "Which systems generated sustained work demand?",
         "Volume weighted by temporal persistence", "Count-based",
         f"{sh['L3']['mean']:.2f}"),
        ("Tail expenditure",
         "Which systems accumulated the greatest excess cost from "
         "unusually expensive jobs?",
         "Cost above the campus p95 threshold", "Currency",
         f"{sh['L4x']['mean']:.2f}"),
        ("Budget risk",
         "Which systems produced the largest annual budget surprises?",
         "Common-shock-adjusted annual cost volatility", "Currency",
         f"{sh['L5r']['mean']:.2f}"),
    ]
    body = ["\\begin{tabular}{p{2.2cm}p{4.6cm}p{4.0cm}p{1.7cm}c}",
            "\\toprule",
            "Criterion & Management question & What it measures & Units & "
            "Reliability \\\\", "\\midrule"]
    for r in rows:
        body.append(" & ".join(r) + " \\\\")
    body += ["\\bottomrule", "\\end{tabular}"]
    write_tex("T2", "\n".join(body) + "\n")


def t3_stability_revised():
    d3 = jload("p3d_revised.json")
    sim = jload("p3e_simulation.json")
    sl = jload("p5b_shortlist.json")
    ent = pd.read_csv(results_path("p3d_entity.csv"))
    n = len(ent)
    n_change = int(ent["topk_burden_count"].isin([1, 2, 3]).sum())
    pt = d3["pairwise_taub_burden"]
    wb = list(d3["W_burden_by_campus"].values())
    rows = [
        ("Kendall's $W$ across burden measures (pooled)",
         f"{d3['W_burden_pooled']:.2f}",
         f"campus range {min(wb):.2f}--{max(wb):.2f}; pairwise $\\tau$-b "
         f"{min(pt.values()):.2f}--{max(pt.values()):.2f}",
         "H1: supported",
         "Burden measures are broadly interchangeable at portfolio scale"),
        ("Metric-sensitive entities vs.\\ noise floor (work-order design)",
         f"{100*d3['share_meii_b_sig_wo']:.1f}\\%",
         f"year-block {100*d3['share_meii_b_sig_year_block']:.1f}\\%; "
         f"simulated Type-I {100*sim['S0']['wo']['type1_meii_b']:.1f}\\% / "
         f"{100*sim['S3']['wo']['type1_meii_b']:.1f}\\% (S0/S3)",
         "H3: supported",
         f"Local metric sensitivity beyond finite-record noise; "
         f"{n_change} of {n} change top-20\\% membership"),
        ("Risk vs.\\ burden association ($\\tau$-b)",
         f"{d3['taub_risk_vs_medburden']:.2f}",
         f"size-adjusted $\\rho$ "
         f"{d3['spearman_risk_vs_burden_size_adj']:.2f}",
         "H2: supported",
         "Budget risk cannot be substituted by a burden ranking"),
        ("Minimax vs.\\ cost-only worst-case regret",
         f"{sl['q20']['minimax']['worst_regret']:.3f} vs.\\ "
         f"{sl['q20']['single_cost']['worst_regret']:.3f}",
         "top-20\\% selection",
         "H4: supported",
         "Explicit optimization improves worst-case criterion coverage"),
    ]
    body = ["\\begin{tabular}{p{3.4cm}p{1.5cm}p{3.6cm}p{1.9cm}p{4.2cm}}",
            "\\toprule",
            "Quantity & Value & Reference & Hypothesis & "
            "Decision interpretation \\\\", "\\midrule"]
    for r in rows:
        body.append(" & ".join(r) + " \\\\")
    body += ["\\bottomrule", "\\end{tabular}"]
    write_tex("T3", "\n".join(body) + "\n")


def t4_profiles():
    ent = pd.read_csv(results_path("p3d_entity.csv"))
    led = pd.read_csv(results_path("p2b_ledgers_entity.csv"))
    ent = ent.merge(led[["campus", "entity", "L1"]], on=["campus", "entity"])
    tot_cost = ent.groupby("campus")["L1"].transform("sum")
    ent["cost_w"] = ent["L1"] / tot_cost
    rows = []
    n = len(ent)
    for key, lab in PROFILE_LABELS.items():
        g = ent[ent["profile"] == key]
        contested = int(g["topk_burden_count"].isin([1, 2, 3]).sum())
        rows.append(
            f"{lab} & {len(g)} & {100*len(g)/n:.1f} & "
            f"{100*g['cost_w'].sum()/ent['cost_w'].sum():.1f} & {contested} \\\\")
    body = ["\\begin{tabular}{p{6.4cm}cccc}", "\\toprule",
            "Profile & Entities & Share (\\%) & Cost-weighted (\\%) & "
            "Top-20\\% metric-dependent \\\\", "\\midrule"] + rows + [
            "\\bottomrule", "\\end{tabular}"]
    write_tex("T4", "\n".join(body) + "\n")


def t5_rules():
    sl = jload("p5b_shortlist.json")
    q20, q10 = sl["q20"], sl["q10"]
    body = ["\\begin{tabular}{p{4.2cm}cccc}", "\\toprule",
            "Decision rule & Worst regret (top-20\\%) & Risk regret & "
            "Worst regret (top-10\\%) & Cost captured (\\%) \\\\", "\\midrule"]
    cap_map = {"single_cost": sl["q20"]["captured_cost_rule"],
               "minimax": sl["q20"]["captured_minimax"]}
    for key, name in RULE_NAMES.items():
        cap = (f"{100*cap_map[key]['L1']:.1f}" if key in cap_map else "")
        body.append(
            f"{name} & {q20[key]['worst_regret']:.3f} & "
            f"{q20[key]['mean_regret']['L5r']:.3f} & "
            f"{q10[key]['worst_regret']:.3f} & {cap} \\\\")
    body += ["\\bottomrule", "\\end{tabular}"]
    write_tex("T5", "\n".join(body) + "\n")


def a6_simulation():
    sim = jload("p3e_simulation.json")
    rows = []
    for scen, desc in (("S0", "Shared latent rank (independent)"),
                       ("S1", "Divergent risk dimension"),
                       ("S2", "Within-burden divergence"),
                       ("S3", "AR(1) + common shocks + clustering")):
        for mode, mname in (("wo", "work-order"), ("year_block", "year-block")):
            d = sim[scen][mode]
            extra = ""
            if scen == "S1":
                extra = (f"gap top-3 recall {100*d['gap_top3_recall_divergent']:.0f}\\%")
            if scen == "S2":
                extra = f"$I^B$ power {100*d['power_meii_b_on_divergent']:.0f}\\%"
            rows.append(
                f"{scen} {desc} & {mname} & {100*d['type1_meii_b']:.1f} & "
                f"{100*d['type1_gap']:.1f} & {extra} \\\\")
    body = ["\\begin{tabular}{p{5.2cm}lccp{3.6cm}}", "\\toprule",
            "Scenario & Resampling & Type-I $I^B$ (\\%) & "
            "Type-I gap (\\%) & Power / localization \\\\", "\\midrule"] \
        + rows + ["\\bottomrule", "\\end{tabular}"]
    write_tex("A6", "\n".join(body) + "\n")


def a7_diagnostics():
    d2 = jload("p2b_diagnostics.json")
    names = {"L1": "L1 Recorded cost", "L2": "L2 Labor hours",
             "L3": "L3 Persistent volume", "L4x": "L4 Tail expenditure",
             "L5r": "L5 Budget risk"}
    rows = []
    for l, nm in names.items():
        rows.append(
            f"{nm} & {d2['split_half'][l]['mean']:.2f} & "
            f"{d2['loyo_rank_consistency'][l]['min']:.2f} & "
            f"{d2['size_dependence'][l]:.2f} \\\\")
    rows.append(
        f"Coefficient of variation (conventional) &  &  & "
        f"{d2['size_dependence']['L5cv']:.2f} \\\\")
    body = ["\\begin{tabular}{p{4.6cm}ccc}", "\\toprule",
            "Criterion & Split-half $\\rho$ & LOYO $\\tau$-b (min) & "
            "Size dependence $\\rho$ \\\\", "\\midrule"] + rows + [
            "\\bottomrule", "\\end{tabular}"]
    write_tex("A7", "\n".join(body) + "\n")


def a8_missing():
    mi = jload("p6b_missing.json")
    rows = []
    for tag, d in mi["R13_threshold_sweep"].items():
        ct, lt = tag.replace("cost", "").split("_labor")
        tau = d.get("mean_taub_vs_baseline")
        rows.append(
            f"Thresholds {ct}/{lt} & {d['n_entities']} & {d['W_burden']:.2f} & "
            f"{d['meii_b_mean']:.3f} & {tau if tau else '--'} \\\\")
    for name, d in mi["R14_missing_treatments"].items():
        tau = d.get("mean_taub_vs_baseline")
        rows.append(
            f"Missing cost: {name.replace('_', ' ')} & {d['n_entities']} & "
            f"{d['W_burden']:.2f} & {d['meii_b_mean']:.3f} & "
            f"{tau if tau else '--'} \\\\")
    body = ["\\begin{tabular}{p{4.8cm}cccc}", "\\toprule",
            "Setting & Entities & $W$ & Mean $I^B$ & "
            "$\\tau$-b vs.\\ baseline \\\\", "\\midrule"] + rows + [
            "\\bottomrule", "\\end{tabular}"]
    write_tex("A8", "\n".join(body) + "\n")


def a9_mapping():
    rows = [
        ("L1 Recorded cost", "Cash spent (constant 2021 currency)",
         "Finance / asset manager", "Expenditure review",
         "Currency", "Yes", "Positive"),
        ("L2 Labor hours", "Technician hours consumed",
         "Maintenance supervisor", "Staffing and scheduling",
         "Hours", "Yes", "Positive"),
        ("L3 Persistent volume",
         "Work-order count weighted by temporal persistence",
         "Operations manager", "Process improvement",
         "Count", "Approx.", "Positive"),
        ("L4 Tail expenditure", "Cost in excess of the campus p95 threshold",
         "Financial planner", "Contingency reserves",
         "Currency", "Yes", "Positive"),
        ("L5 Budget risk", "SD of two-way-demeaned annual cost",
         "Budget planner", "Risk control and forecasting",
         "Currency", "No (dispersion)", "Weak"),
    ]
    body = ["\\begin{tabular}{p{2.2cm}p{3.0cm}p{2.0cm}p{2.0cm}p{1.2cm}p{1.4cm}p{1.2cm}}",
            "\\toprule",
            "Criterion & Construct & Primary stakeholder & Decision & Units & "
            "Additive & Size rel. \\\\", "\\midrule"]
    for r in rows:
        body.append(" & ".join(r) + " \\\\")
    body += ["\\bottomrule", "\\end{tabular}"]
    write_tex("A9", "\n".join(body) + "\n")


if __name__ == "__main__":
    t2_criteria()
    t3_stability_revised()
    t4_profiles()
    t5_rules()
    a6_simulation()
    a7_diagnostics()
    a8_missing()
    a9_mapping()
    print("revised tables complete")
