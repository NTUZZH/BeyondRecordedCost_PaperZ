"""Phase 9: appendix tables A1-A5 (formerly a separate supplementary doc;
folded into the manuscript appendix so the submission is a single document).

Each table is written as tables/A{n}.tex (booktabs LaTeX body, \\input into
manuscript/sec_appendix.tex). The full per-entity listing (82 rows) stays in
the released results files (results/p3_meii_entity.csv, p4_archetypes.csv)
and is pointed to from the appendix text rather than typeset.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

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


def _tabular(header: str, colspec: str, rows: list[str]) -> str:
    lines = [f"\\begin{{tabular}}{{{colspec}}}", "\\toprule", header + " \\\\",
             "\\midrule"] + [r + " \\\\" for r in rows] + ["\\bottomrule",
                                                           "\\end{tabular}"]
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------ A1
def a1_excluded():
    """Entities excluded by the comparability filter."""
    ent = pd.read_parquet(ROOT / "data/panel/entity_system.parquet")
    ent = ent[ent["system"].astype(str).str.strip() != ""]
    exc = ent[~ent["comparable"]].copy().sort_values(["campus", "system"])
    rows = []
    for _, r in exc.iterrows():
        rows.append(
            f"U{r['campus']:02d} & {r['system']} & {int(r['n_wos']):,} & "
            f"{int(r['n_years_active'])} & {100 * r['cost_nonmissing']:.0f} & "
            f"{100 * r['labor_nonmissing']:.0f}")
    write_tex("A1", _tabular(
        "Campus & System & Work orders & Active years & "
        "Cost recorded (\\%) & Labor recorded (\\%)", "llrrrr", rows))


# ------------------------------------------------------------------ A2/A3
def a2_a3_folds():
    """Per-fold leave-one-campus-out and leave-one-year-out detail."""
    ro = jload("p6_robustness_summary.json")
    rows = []
    for c, d in sorted(ro["R1"]["per_left_out"].items(), key=lambda kv: int(kv[0])):
        rows.append(
            f"U{int(c):02d} & {d['W_pooled']:.3f} & "
            f"{100 * d['high_contrast_share']:.1f} & {100 * d['h2_stat']:.1f} & "
            f"{100 * d['n1_sig_share']:.1f} & {100 * d['archetype_agreement']:.1f}")
    write_tex("A2", _tabular(
        "Campus left out & $W$ & High-contrast (\\%) & H2 stat (\\%) & "
        "N1 sig. (\\%) & Archetype agr. (\\%)", "lccccc", rows))

    rows = []
    for y, d in sorted(ro["R2"]["per_left_out_year"].items(), key=lambda kv: int(kv[0])):
        rows.append(
            f"{y} & {d['W_pooled']:.3f} & {100 * d['high_contrast_share']:.1f} & "
            f"{100 * d['h2_stat']:.1f} & {d['regret_ratio']:.2f} & "
            f"{int(d['n_entities'])} & {100 * d['n1_sig_share']:.1f} & "
            f"{100 * d['archetype_agreement']:.1f}")
    write_tex("A3", _tabular(
        "Year left out & $W$ & High-contrast (\\%) & H2 stat (\\%) & "
        "Regret ratio & Entities & N1 sig. (\\%) & Archetype agr. (\\%)",
        "lccccccc", rows))


# ------------------------------------------------------------------ A4
def a4_within_system():
    """Within-system-category agreement pooled across campuses (R3)."""
    ro = jload("p6_robustness_summary.json")
    arch = pd.read_csv(results_path("p4_archetypes.csv"))
    desc = (arch.groupby("entity")["system_desc"]
            .agg(lambda s: s.mode().iat[0]).to_dict())
    rows = []
    for sysc, d in sorted(ro["R3"].items()):
        name = desc.get(sysc, "")
        rows.append(
            f"{sysc} {name} & {int(d['n_entities'])} & {d['W']:.2f} & "
            f"{d['meii_mean']:.2f} & {100 * d['sig_share']:.0f}")
    write_tex("A4", _tabular(
        "System category & Campuses & $W$ & Mean MEII & N1 sig. (\\%)",
        "lcccc", rows))


# ------------------------------------------------------------------ A5
def a5_chronicity_variants():
    """Chronicity-definition variants (R8)."""
    ro = jload("p6_robustness_summary.json")
    label = {"count": "Raw work-order count",
             "active_quarters": "Active-quarter share",
             "median_nonzero": "Median nonzero quarter rate"}
    rows = []
    for v, d in ro["R8"].items():
        rows.append(
            f"{label.get(v, v)} & {d['taub_L3_vs_base']:.2f} & "
            f"{d['mean_abs_meii_shift']:.3f} & {d['W_pooled']:.2f} & "
            f"{100 * d['archetype_agreement']:.0f}")
    write_tex("A5", _tabular(
        "Chronicity variant & $\\tau$-b vs.\\ base L3 & Mean $|\\Delta$MEII$|$ & "
        "$W$ & Archetype agr. (\\%)", "lcccc", rows))


def main() -> None:
    a1_excluded()
    a2_a3_folds()
    a4_within_system()
    a5_chronicity_variants()


if __name__ == "__main__":
    main()
