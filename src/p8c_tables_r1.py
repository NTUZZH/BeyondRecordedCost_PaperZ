"""Phase 8c: manuscript and supplemental tables, named as printed.

Main text
  T1  the five criteria (management question, user, measure, reliability)
  T2  retained campuses and their scale
  T3  criterion values per campus (median and interquartile range per year)
  T4  tests of H1-H4
  T5  profiles with prevalence, cost weight, and action
  T6  shortlist rules, scored on the observed records and on records the
      list was not chosen from
  T7  contested systems
Supplemental Materials
  A7  criterion diagnostics (adds the additive budget-risk variant)
  A9  criterion decision mapping
  A10 Kendall's W per campus with chi-square tests
  A11 the framework per maintenance stream (preventive, unplanned)
  A12 year-to-year persistence and the forward check of shortlists

Each LaTeX table has a CSV sibling with the same content.
"""

from __future__ import annotations

import itertools
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from p8_tables import TAB, jload, write_tex
from p8b_revised_tables import PROFILE_LABELS, RULE_NAMES, write_csv
from utils.io import ROOT, load_config, results_path

SYSTEM_NAMES = {
    "B20": "Exterior enclosure", "B30": "Roofing", "C10": "Interior construction",
    "C30": "Interior finishes", "D10": "Conveying", "D20": "Plumbing",
    "D30": "HVAC", "D40": "Fire protection", "D50": "Electrical",
    "D90": "General building services", "E10": "Equipment",
    "E20": "Furnishings", "G20": "Site improvements",
    "G30": "Site mechanical utilities", "G40": "Site electrical utilities",
}


def tabular(spec: str, header: list[str], rows: list[list[str]]) -> str:
    body = [f"\\begin{{tabular}}{{{spec}}}", "\\toprule",
            " & ".join(header) + " \\\\", "\\midrule"]
    body += [" & ".join(r) + " \\\\" for r in rows]
    body += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(body) + "\n"


def plain(cell: str) -> str:
    """CSV text of a LaTeX cell."""
    for a, b in (("\\%", "%"), ("$", ""), ("\\leq", "<="), ("\\geq", ">="),
                 ("\\tau", "tau"), ("\\chi^2", "chi2"), ("--", "-"),
                 ("\\textbf", ""), ("\\ ", " "), ("{", ""), ("}", "")):
        cell = cell.replace(a, b)
    return cell


def emit(name: str, spec: str, header: list[str], rows: list[list[str]]) -> None:
    write_tex(name, tabular(spec, header, rows))
    write_csv(name, [plain(h) for h in header], [[plain(c) for c in r] for r in rows])


def t1_criteria():
    sh = jload("p2b_diagnostics.json")["split_half"]
    rows = [
        ["Recorded cost", "Where was the most money spent?",
         "Finance and budgeting", "Total recorded cost in constant 2021 currency",
         f"{sh['L1']['mean']:.2f}"],
        ["Labor hours", "Which systems took the most technician time?",
         "Workforce planning", "Total labor hours", f"{sh['L2']['mean']:.2f}"],
        ["Persistent volume", "Which systems generate a steady flow of new work?",
         "Operations", "Work-order count weighted by the share of quarters "
         "with work", f"{sh['L3']['mean']:.2f}"],
        ["Tail expenditure", "Which systems produce unusually expensive jobs?",
         "Contingency planning", "Cost above the campus 95th percentile of "
         "per-work-order cost", f"{sh['L4x']['mean']:.2f}"],
        ["Budget risk", "Which systems make annual spending hardest to "
         "predict?", "Budget planning", "Standard deviation of annual cost "
         "around its own mean scaled by the campus spending index",
         f"{sh['L5r']['mean']:.2f}"],
    ]
    emit("T1", "p{2.1cm}p{3.9cm}p{2.3cm}p{4.3cm}c",
         ["Criterion", "Management question", "Main user", "What it measures",
          "Reliability"], rows)


def t2_campuses():
    cfg = load_config()
    cs = pd.read_csv(results_path("p10a_campus_stats.csv"))
    ent = pd.read_parquet(ROOT / "data/panel/entity_system.parquet")
    ent = ent[ent["system"].astype(str).str.strip() != ""]
    rows = []
    for _, r in cs.iterrows():
        c = int(r["campus"][1:])
        e = ent[ent["campus"] == c]
        assert int(e["comparable"].sum()) == int(r["n_comparable"])
        bld = "--" if pd.isna(r["n_buildings"]) else f"{int(r['n_buildings']):,}"
        rows.append([
            r["campus"], "Canada" if c in cfg["campuses_cad"] else "USA",
            r["window"].replace("-", "--"), bld,
            f"{r['wo_per_year']:,.0f}", f"{r['cost_per_year_m']:.1f}",
            f"{r['labor_hours_per_year_k']:.0f}", f"{100 * r['ppm_share']:.1f}",
            str(len(e)), str(int(r["n_comparable"]))])
    emit("T2", "llcrrrrrrr",
         ["Campus", "Country", "Window", "Buildings", "Work orders per year",
          "Cost per year (million)", "Labor hours per year (thousand)",
          "Preventive (\\%)", "Systems", "Comparable"], rows)


def t3_criterion_stats():
    cr = pd.read_csv(results_path("p10a_criteria_stats.csv"))

    def cell(r, k, scale=1.0, nd=0):
        f = f"{{:,.{nd}f}}"
        return (f"{f.format(r[k + '_med'] * scale)} "
                f"({f.format(r[k + '_q1'] * scale)}--{f.format(r[k + '_q3'] * scale)})")
    rows = []
    for _, r in cr.iterrows():
        rows.append([r["campus"], str(int(r["n"])), cell(r, "cost_k"),
                     cell(r, "labor_h"), cell(r, "volume"), cell(r, "tail_k"),
                     cell(r, "risk_k"), cell(r, "risk_rel", 100)])
    emit("T3", "lrcccccc",
         ["Campus", "Systems", "Recorded cost (thousand)", "Labor hours",
          "Persistent volume", "Tail expenditure (thousand)",
          "Budget risk (thousand)", "Relative risk (\\%)"], rows)


def t4_tests():
    t = jload("p10b_tests.json")
    h1, h2, h3, h4 = t["H1"], t["H2"], t["H3"], t["H4"]
    K = h2["K"]
    wo, yb = h1["wo"], h1["year_block"]
    rs, sh = h4["rescore"], h4["split_half"]
    sl = jload("p5b_shortlist.json")["q20"]
    others = {k: v["worst_regret"] for k, v in sl.items()
              if isinstance(v, dict) and "worst_regret" in v and k != "minimax"}
    rs_wo = {k: v for k, v in rs["wo"]["mean_worst"].items() if k != "minimax"}
    rs_yb = {k: v for k, v in rs["year_block"]["mean_worst"].items() if k != "minimax"}
    sh_mw = sh["mean_worst"]
    sh_rest = [v for k, v in sh_mw.items()
               if k not in ("minimax", "single_labor", "burden_median")]
    beaten = ["single_cost", "single_volume", "single_tail", "single_risk",
              "burden_mean", "all_mean"]
    assert all(sh[r]["mean_diff"] > 0 for r in beaten)
    # rounded up, so that the printed bound holds for every listed rule
    p_beaten = math.ceil(max(sh[r]["wilcoxon_p_greater"] for r in beaten) * 1000) / 1000
    ind = jload("p3g_independence.json")
    # top-K overlaps among the burden measures and with budget risk
    e2 = pd.read_csv(results_path("p2b_ledgers_entity.csv"))
    assert int(np.ceil(0.2 * len(e2))) == K
    tops = {c: set(e2.nlargest(K, c).index)
            for c in ("r_L1", "r_L2", "r_L3", "r_L4x", "r_L5r")}
    bp = [len(tops[a] & tops[b]) for a, b in
          itertools.combinations(("r_L1", "r_L2", "r_L3", "r_L4x"), 2)]
    rb = [len(tops["r_L5r"] & tops[c]) for c in ("r_L1", "r_L2", "r_L3", "r_L4x")]
    ov = h2["overlap_risk_top_with"]
    n = lambda x: int(round(x * K))
    rows = [
        ["H1", "Kendall's $W$ of the four burden ranks, chi-square test",
         f"{h1['pooled']['W']:.2f}",
         f"{ind['W_perm_mean']:.2f} for unrelated rankings (95th percentile "
         f"{ind['W_perm_p95']:.2f}); $p < 0.001$", "Supported"],
        ["H1", "$W$ against matched sampling noise",
         f"{h1['pooled']['W']:.2f}",
         f"{wo['W_null_mean']:.3f} ({wo['W_null_p2_5']:.3f}--{wo['W_null_p97_5']:.3f}) / "
         f"{yb['W_null_mean']:.3f} ({yb['W_null_p2_5']:.3f}--{yb['W_null_p97_5']:.3f}); "
         f"$p \\leq {max(wo['p_incomplete'], yb['p_incomplete']):.3f}$", "Supported"],
        ["H1", "Systems changing top-20\\% membership with the measure",
         str(h1["topk_dependent_observed"]),
         f"{wo['topk_null_mean']:.1f} (95th percentile {wo['topk_null_p95']:.0f}) / "
         f"{yb['topk_null_mean']:.1f} ({yb['topk_null_p95']:.0f}); "
         f"$p \\leq {max(wo['p_topk_beyond_noise'], yb['p_topk_beyond_noise']):.3f}$",
         "Supported"],
        ["H2", "$\\tau$-b of budget risk and median burden rank",
         f"{h2['tau_risk_medburden']:.2f} ({h2['tau_risk_medburden_ci95'][0]:.2f}--"
         f"{h2['tau_risk_medburden_ci95'][1]:.2f})",
         f"Mean of burden pairs {h2['tau_burden_mean_pair']:.2f}; difference "
         f"${h2['diff_mean_pair_minus_risk_ci95'][0]:.2f}$ to "
         f"{h2['diff_mean_pair_minus_risk_ci95'][1]:.2f}", "Not supported"],
        ["H2", "Within-campus Spearman correlation: unadjusted; corrected for "
         "reliability; adjusted for work-order count",
         f"{h2['rho_risk_medburden_within_campus']:.2f}; "
         f"{h2['rho_disattenuated']:.2f}; "
         f"{h2['size_adj_rho']:.2f} ({h2['size_adj_rho_ci95'][0]:.2f}--"
         f"{h2['size_adj_rho_ci95'][1]:.2f})",
         "0 for unrelated ranks", "Not supported"],
        ["H2", f"Shared members of two top-{K} sets",
         f"Risk and cost {n(ov['cost'])}; risk and a burden measure "
         f"{min(rb)}--{max(rb)}",
         f"Two burden measures {min(bp)}--{max(bp)}; risk and cost "
         f"{n(h2['overlap_cost_ci95_wo'][0])}--{n(h2['overlap_cost_ci95_wo'][1])} / "
         f"{n(h2['overlap_cost_ci95_year_block'][0])}--"
         f"{n(h2['overlap_cost_ci95_year_block'][1])} under resampling",
         "Shortlist diagnostic"],
    ]
    for mode, lab in (("wo", "work-order"), ("year_block", "year-block")):
        d = h3[mode]
        rows.append(["H3", f"Share above the noise floor, {lab} design",
                     f"{100 * d['share']:.1f}\\% (at least {100 * d['ci95'][0]:.1f}\\%); "
                     f"{d['bh_q05_count']} after false-discovery-rate control",
                     "10\\%; $p < 0.001$", "Supported"])
    g3 = jload("p10g_h3_campus.json")
    rows.append(["H3", "Campuses whose share exceeds 10\\%, both designs",
                 f"{g3['wo']['campuses_above_threshold']} of {g3['n_campuses']} / "
                 f"{g3['year_block']['campuses_above_threshold']} of {g3['n_campuses']}; "
                 f"lowest {100 * g3['wo']['share_min']:.1f}\\% / "
                 f"{100 * g3['year_block']['share_min']:.1f}\\%",
                 f"Sign test $p = {g3['wo']['sign_test_p']:.3f}$; campus bootstrap "
                 f"{100 * g3['wo']['cluster_boot_ci95'][0]:.1f}--"
                 f"{100 * g3['wo']['cluster_boot_ci95'][1]:.1f}\\% / "
                 f"{100 * g3['year_block']['cluster_boot_ci95'][0]:.1f}--"
                 f"{100 * g3['year_block']['cluster_boot_ci95'][1]:.1f}\\%",
                 "Supported"])
    rows += [
        ["H4", "Worst-case regret on the observed records",
         f"{sl['minimax']['worst_regret']:.3f}",
         f"{min(others.values()):.3f} or more for the other rules", "By construction"],
        ["H4", "Mean worst-case regret on resampled records",
         f"{rs['wo']['mean_worst']['minimax']:.3f} / "
         f"{rs['year_block']['mean_worst']['minimax']:.3f}",
         f"{min(rs_wo.values()):.3f} or more / {min(rs_yb.values()):.3f} or more; "
         "every paired interval includes 0",
         "Lowest mean"],
        ["H4", "Mean worst-case regret on the held-out half",
         f"{sh_mw['minimax']:.3f}",
         f"Labor hours {sh_mw['single_labor']:.3f}, median burden "
         f"{sh_mw['burden_median']:.3f}, others {min(sh_rest):.3f}--"
         f"{max(sh_rest):.3f}; lower than six rules at $p \\leq {p_beaten:.3f}$",
         "Partly supported"],
    ]
    emit("T4", "lp{4.4cm}p{3.0cm}p{5.0cm}p{2.0cm}",
         ["Hypothesis", "Quantity", "Observed", "Reference", "Result"], rows)


def t5_profiles():
    ent = pd.read_csv(results_path("p3d_entity.csv"))
    led = pd.read_csv(results_path("p2b_ledgers_entity.csv"))
    ent = ent.merge(led[["campus", "entity", "L1"]], on=["campus", "entity"])
    ent["cost_w"] = ent["L1"] / ent.groupby("campus")["L1"].transform("sum")
    action = {
        "stable_aligned": "Report one burden measure",
        "stable_divergent": "Review budget risk separately",
        "unstable_aligned": "State the measure and inspect systems near the cutoff",
        "unstable_divergent": "Apply an explicit multi-criteria rule",
    }
    rows, n = [], len(ent)
    for key, lab in PROFILE_LABELS.items():
        g = ent[ent["profile"] == key]
        rows.append([lab.replace(" / ", ", "), str(len(g)),
                     f"{100 * len(g) / n:.1f}",
                     f"{100 * g['cost_w'].sum() / ent['cost_w'].sum():.1f}",
                     str(int(g["topk_burden_count"].isin([1, 2, 3]).sum())),
                     action[key]])
    emit("T5", "p{4.6cm}cccc p{4.2cm}",
         ["Profile", "Systems", "Share (\\%)", "Cost-weighted share (\\%)",
          "Top-20\\% membership depends on the measure", "Action"], rows)


def t6_rules():
    sl = jload("p5b_shortlist.json")
    q20, q10 = sl["q20"], sl["q10"]
    t = jload("p10b_tests.json")["H4"]
    rs, sh = t["rescore"], t["split_half"]["mean_worst"]
    cap = {"single_cost": q20["captured_cost_rule"], "minimax": q20["captured_minimax"]}
    vals = {key: [round(q20[key]["worst_regret"], 3),
                  round(q20[key]["mean_regret"]["L5r"], 3),
                  round(q10[key]["worst_regret"], 3),
                  round(rs["wo"]["mean_worst"][key], 3),
                  round(rs["year_block"]["mean_worst"][key], 3),
                  round(sh[key], 3)] for key in RULE_NAMES}
    # Bold marks the best (lowest) value in each regret column across all
    # rules, ties included; the cost-captured column is not ranked.
    best = [min(v[j] for v in vals.values()) for j in range(6)]

    def b(j, x):
        s = f"{x:.3f}"
        return f"\\textbf{{{s}}}" if x == best[j] else s
    rows = []
    for key, name in RULE_NAMES.items():
        v = vals[key]
        rows.append([
            name, b(0, v[0]), b(1, v[1]), b(2, v[2]),
            f"{b(3, v[3])} / {b(4, v[4])}", b(5, v[5]),
            f"{100 * cap[key]['L1']:.1f}" if key in cap else "--"])
    emit("T6", "p{4.0cm}cccccc",
         ["Rule", "Worst-case regret", "Regret on budget risk",
          "Worst-case regret, top 10\\%", "Resampled (work-order / year-block)",
          "Split-half", "Cost captured (\\%)"], rows)


def t7_contested():
    c = pd.read_csv(results_path("p10e_contested.csv"))
    c = c[c["contested"]].sort_values("freq", ascending=False)
    rows = []
    for _, r in c.iterrows():
        rows.append([f"U{int(r['campus']):02d}", SYSTEM_NAMES[r["entity"]]]
                    + [f"{r[k]:.2f}" for k in ("rank_cost", "rank_labor",
                                                "rank_volume", "rank_tail",
                                                "rank_risk")]
                    + [f"{100 * r['freq']:.1f}", "Yes" if r["in_minimax"] else "No"])
    emit("T7", "llccccccc",
         ["Campus", "System", "Cost", "Labor", "Volume", "Tail", "Budget risk",
          "Selected (\\% of weightings)", "Minimax list"], rows)


def a7_diagnostics():
    d2 = jload("p2b_diagnostics.json")
    names = {"L1": "Recorded cost", "L2": "Labor hours",
             "L3": "Persistent volume", "L4x": "Tail expenditure",
             "L5r": "Budget risk"}
    rows = [[nm, f"{d2['split_half'][l]['mean']:.2f}",
             f"{d2['loyo_rank_consistency'][l]['min']:.2f}",
             f"{d2['size_dependence'][l]:.2f}"] for l, nm in names.items()]
    rows.append(["Budget risk, additive adjustment", "n/a", "n/a",
                 f"{d2['size_dependence']['L5r_add']:.2f}"])
    rows.append(["Coefficient of variation of annual cost", "n/a", "n/a",
                 f"${d2['size_dependence']['L5cv']:.2f}$"])
    emit("A7", "p{5.0cm}ccc",
         ["Criterion", "Split-half $\\rho$", "Leave-one-year-out $\\tau$-b (min)",
          "Size dependence $\\rho$"], rows)


def a9_mapping():
    rows = [
        ["Recorded cost", "Cash spent (constant 2021 currency)",
         "Finance, asset manager", "Expenditure review", "Currency", "Yes"],
        ["Labor hours", "Technician hours consumed", "Maintenance supervisor",
         "Staffing and scheduling", "Hours", "Yes"],
        ["Persistent volume", "Work-order count weighted by the share of "
         "active quarters", "Operations manager", "Process improvement",
         "Count", "Approximately"],
        ["Tail expenditure", "Cost above the campus 95th percentile of "
         "the cost per work order", "Financial planner", "Contingency reserves",
         "Currency", "Yes"],
        ["Budget risk", "SD of annual cost around its own mean scaled by the "
         "campus spending index", "Budget planner", "Contingency allowance "
         "and forecasting", "Currency", "No"],
    ]
    emit("A9", "p{2.3cm}p{3.2cm}p{2.3cm}p{2.6cm}p{1.3cm}p{1.9cm}",
         ["Criterion", "Construct", "Primary user", "Decision", "Units",
          "Additive"], rows)


def a10_w_campus():
    h1 = jload("p10b_tests.json")["H1"]
    rows = []
    for c, d in h1["per_campus"].items():
        rows.append([f"U{int(c):02d}", str(d["n"]), f"{d['W']:.2f}",
                     f"{d['chi2']:.1f}", str(d["df"]), f"{d['p']:.1e}"])
    d = h1["pooled"]
    rows.append(["Pooled", str(d["n"]), f"{d['W']:.2f}", f"{d['chi2']:.1f}",
                 str(d["df"]), f"{d['p']:.1e}"])
    emit("A10", "lccccc", ["Campus", "Systems", "$W$", "$\\chi^2$",
                           "Degrees of freedom", "$p$"], rows)


def a11_streams():
    st = jload("p10c_streams.json")
    rows = []
    for k, lab in (("PPM", "Planned preventive"), ("UPM", "Unplanned")):
        d = st[k]
        others = {r: v for r, v in d["worst_regret"].items() if r != "minimax"}
        rows.append([lab, str(d["n_entities"]), f"{d['n_work_orders']:,}",
                     f"{d['W_pooled']:.2f}",
                     f"{d['taub_burden_min']:.2f}--{d['taub_burden_max']:.2f}",
                     f"{d['taub_risk_medburden']:.2f}",
                     f"{100 * d['share_flag_wo']:.1f} / {100 * d['share_flag_year_block']:.1f}",
                     str(d["topk_dependent"]),
                     f"{d['worst_regret']['minimax']:.3f} ({min(others.values()):.3f})"])
    emit("A11", "lcccccccc",
         ["Stream", "Systems", "Work orders", "$W$", "Burden pairs $\\tau$-b",
          "Risk vs burden $\\tau$-b", "Flagged (\\%, work-order / year-block)",
          "Top-20\\% changes", "Minimax worst-case regret (best other)"], rows)


def a12_dynamics():
    dy = jload("p10d_dynamics.json")
    pe = dy["persistence"]
    lab = {"cost": "Recorded cost", "labor": "Labor hours", "volume": "Persistent volume",
           "tail": "Tail expenditure", "mean": "Mean burden rank"}
    rows = [[v, f"{pe['taub_consecutive_mean'][k]:.2f}",
             f"{pe['taub_consecutive_min'][k]:.2f}",
             f"{100 * pe['top20_retained_mean'][k]:.1f}"] for k, v in lab.items()]
    emit("A12", "p{4.0cm}ccc",
         ["Ranking", "Consecutive-year $\\tau$-b (mean)",
          "Consecutive-year $\\tau$-b (min)", "Top 20\\% retained next year (\\%)"],
         rows)
    fw = dy["forward_h4"]
    rows = []
    for key, name in RULE_NAMES.items():
        d = fw[key]
        rows.append([name] + [f"{d[c]:.2f}" for c in ("cost", "labor", "volume",
                                                      "tail", "mean")]
                    + [f"{d['worst_burden']:.2f}"])
    emit("A13", "p{4.0cm}cccccc",
         ["Rule chosen on the first four years", "Cost", "Labor", "Volume",
          "Tail", "Mean burden", "Smallest"], rows)


def a14_h4_detail():
    t = jload("p10b_tests.json")["H4"]
    rs, sh = t["rescore"], t["split_half"]
    rows = []
    for key, name in RULE_NAMES.items():
        if key == "minimax":
            continue
        w, y, h = rs["wo"][key], rs["year_block"][key], sh[key]
        rows.append([name,
                     f"{100 * w['share_minimax_lower']:.1f} / {100 * y['share_minimax_lower']:.1f}",
                     f"{100 * w['share_minimax_le']:.1f} / {100 * y['share_minimax_le']:.1f}",
                     f"{100 * h['share_minimax_lower']:.1f}",
                     f"{100 * h['share_tie']:.1f}",
                     f"{100 * h['share_minimax_higher']:.1f}",
                     f"{h['wilcoxon_p_greater']:.1e}"])
    emit("A14", "p{3.6cm}cccccc",
         ["Rule compared with the minimax list",
          "Minimax lower, resampled (\\%)", "Minimax lower or equal, resampled (\\%)",
          "Minimax lower, split-half (\\%)", "Tie (\\%)", "Minimax higher (\\%)",
          "Wilcoxon $p$"], rows)


def a15_condition():
    c = jload("p10f_condition.json")
    rows = []
    for k, d in c["availability"].items():
        rows.append([f"U{int(k):02d}", f"{d['buildings']:,}", str(d["fci"]),
                     str(d["crv"]), str(d["dmc"])])
    emit("A15", "lcccc",
         ["University", "Buildings with an identifier",
          "Facility condition index", "Current replacement value",
          "Deferred maintenance cost"], rows)
    u = c["U01_buildings"]
    lab = {"cost": "Recorded cost", "labor": "Labor hours", "wos": "Work orders"}
    rows = [[v] + [f"{u[f'rho_{k}_{y}']:.2f}" for y in ("dmc", "crv", "size", "fci")]
            for k, v in lab.items()]
    emit("A16", "p{3.4cm}cccc",
         ["Annual burden at U01", "Deferred maintenance cost",
          "Current replacement value", "Gross size",
          "Facility condition index"], rows)


if __name__ == "__main__":
    a14_h4_detail()
    a15_condition()
    t1_criteria()
    t2_campuses()
    t3_criterion_stats()
    t4_tests()
    t5_profiles()
    t6_rules()
    t7_contested()
    a7_diagnostics()
    a9_mapping()
    a10_w_campus()
    a11_streams()
    a12_dynamics()
    print("revision tables complete")
