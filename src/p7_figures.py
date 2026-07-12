"""Phase 7: manuscript figures F1-F8 (guide Section 8).

All figures are generated from results/ files (plus panel aggregates), saved
as PDF + 300 dpi PNG under figures/. Okabe-Ito palette; ledger colors fixed
across all figures; single-column 90 mm or double-column 190 mm.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import patches
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils.io import ROOT, load_config, results_path

FIG = ROOT / "figures"
MM = 1 / 25.4
W1, W2 = 90 * MM, 190 * MM

# Okabe-Ito
OI = {
    "black": "#000000", "orange": "#E69F00", "sky": "#56B4E9",
    "green": "#009E73", "yellow": "#F0E442", "blue": "#0072B2",
    "vermillion": "#D55E00", "purple": "#CC79A7", "gray": "#999999",
}
LEDGER_COLOR = {
    "L1": OI["blue"], "L2": OI["orange"], "L3": OI["green"],
    "L4": OI["vermillion"], "L5": OI["purple"],
}
LEDGER_NAME = {
    "L1": "Recorded cost", "L2": "Labor hours", "L3": "Chronicity",
    "L4": "Shock burden", "L5": "Budget volatility",
}
ARCH_COLOR = {
    "stable_priority": OI["blue"], "hidden_burden": OI["vermillion"],
    "cash_sink": OI["yellow"], "labor_sink": OI["orange"],
    "chronic_drain": OI["green"], "shock_generator": "#8C510A",
    "budget_destabiliser": OI["purple"], "representation_sensitive": OI["sky"],
    "unremarkable": "#BBBBBB",
}
ARCH_LABEL = {
    "stable_priority": "Stable priority", "hidden_burden": "Hidden burden",
    "cash_sink": "Cash sink", "labor_sink": "Labor sink",
    "chronic_drain": "Chronic drain", "shock_generator": "Shock generator",
    "budget_destabiliser": "Budget destabilizer",
    "representation_sensitive": "Representation-sensitive",
    "unremarkable": "Unremarkable",
}

# One Times face for every character in the paper, figures included. The
# manuscript body (newtx) is set in TeX Gyre Termes, the Times New Roman
# metric-compatible face shipped with TeX, so the figures use the same face
# for text AND math (mathtext "custom" routes rm/it/bf through it), and no
# STIX or DejaVu glyph can leak into a plot. assets/fonts holds TrueType
# copies built by src/utils/make_fonts.py: matplotlib embeds TrueType cleanly
# under pdf.fonttype 42, whereas the stock CFF/OTF trips PDF preflight.
import glob as _glob
import matplotlib.font_manager as _fm
_TTF = _glob.glob(str(ROOT / "assets/fonts/TeXGyreTermes-*.ttf"))
for _p in _TTF + [
        "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSerif-Italic.ttf"]:
    try:
        _fm.fontManager.addfont(_p)
    except Exception:
        pass
_SERIF = ["Times New Roman", "TeX Gyre Termes", "Liberation Serif",
          "Nimbus Roman", "DejaVu Serif"]
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": _SERIF,
    "mathtext.fontset": "custom",
    "mathtext.rm": "serif", "mathtext.it": "serif:italic",
    "mathtext.bf": "serif:bold", "mathtext.sf": "serif",
    "mathtext.cal": "serif:italic", "mathtext.tt": "serif",
    "mathtext.default": "it",
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "grid.linewidth": 0.4, "grid.color": "#DDDDDD", "grid.alpha": 0.8,
    "pdf.fonttype": 42, "savefig.dpi": 300,
})


def save(fig, name: str) -> None:
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIG / f"{name}.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"saved {name}")


def jload(name: str):
    with open(results_path(name)) as fh:
        return json.load(fh)


# ---------------------------------------------------------------- F1
def f1_schematic():
    fig, ax = plt.subplots(figsize=(W2, 2.6))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 34)
    ax.axis("off")

    def box(x, y, w, h, text, fc="#F2F2F2", ec="#555555", fs=7.5, bold=False):
        ax.add_patch(patches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.4", fc=fc, ec=ec, lw=0.8))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, fontweight="bold" if bold else "normal")

    def arrow(x0, y0, x1, y1):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="->", lw=0.9, color="#555555"))

    # module 1: panel construction
    box(0.5, 12, 16.5, 10, "CMMS work-order\nrecords\n(one table)", fc="#E8E8E8", bold=True)
    ax.text(8.5, 8.6, "Module 1\ncleaning, panels,\ncomparability filter",
            ha="center", va="top", fontsize=6.5, color="#444444")
    # module 2: five ledgers
    led_y = [28, 22, 16, 10, 4]
    for (l, y) in zip(["L1", "L2", "L3", "L4", "L5"], led_y):
        box(24, y, 17, 4.6, f"{l}  {LEDGER_NAME[l]}",
            fc="white", ec=LEDGER_COLOR[l], fs=7)
        arrow(16.5, 17, 23.5, y + 2.3)
    ax.text(32.5, 1.2, "Module 2: multi-ledger representation\n(within-campus percentile ranks)",
            ha="center", va="top", fontsize=6.5, color="#444444")
    # module 3: stability testing
    box(48, 19, 17, 9, "Ranking-stability\ntests\nKendall W, tau-b,\ntop-k overlap", fc="#F7F7F7")
    box(48, 6, 17, 9, "Null models\nN1 noise floor\nN2 independence\nN3 split-half", fc="#F7F7F7")
    for y in led_y:
        arrow(41.5, y + 2.3, 47.5, 17)
    ax.text(56.5, 2.8, "Module 3", ha="center", va="top", fontsize=6.5, color="#444444")
    # module 4: MEII + archetypes
    box(71, 19, 15, 9, "MEII\ninstability index\nvs noise floor", fc="#F7F7F7")
    box(71, 6, 15, 9, "Economic\narchetypes\n(ordered rules)", fc="#F7F7F7")
    arrow(65.5, 23.5, 70.5, 23.5)
    arrow(65.5, 10.5, 70.5, 10.5)
    ax.text(78.5, 2.8, "Module 4", ha="center", va="top", fontsize=6.5, color="#444444")
    # module 5: decision case
    box(91, 12, 8.5, 10, "Budget-\nallocation\ndecision\ncase", fc="#E8E8E8", bold=True)
    arrow(86.5, 23.5, 90.5, 18.5)
    arrow(86.5, 10.5, 90.5, 15.5)
    ax.text(95.2, 8.6, "Module 5\ntop-k, captured\nburden, regret",
            ha="center", va="top", fontsize=6.5, color="#444444")
    save(fig, "F1_framework")


# ---------------------------------------------------------------- F2
def coverage_eventdate(cfg) -> pd.DataFrame:
    """Campus x year WO counts using each campus's assigned date column
    (deviation D-02); cached in results/p7_coverage_eventdate.csv."""
    out = results_path("p7_coverage_eventdate.csv")
    if out.exists():
        cov = pd.read_csv(out, index_col=0)
        cov.columns = [int(c) for c in cov.columns]
        return cov
    from utils.io import load_raw
    df = load_raw(usecols=["UniversityID", "WOStartDate", "WOEndDate"])
    date_col = {int(k): v for k, v in cfg["date_column_by_campus"].items()}
    df["event_date"] = df["WOStartDate"]
    for uid, col in date_col.items():
        if col != "WOStartDate":
            m = df["UniversityID"] == uid
            df.loc[m, "event_date"] = df.loc[m, "WOEndDate"]
    df["year"] = df["event_date"].dt.year
    cov = (df.dropna(subset=["year"]).astype({"year": int})
           .groupby(["UniversityID", "year"]).size().unstack(fill_value=0))
    cov = cov.reindex(index=range(1, 13), fill_value=0)
    cov = cov[[y for y in cov.columns if 2002 <= y <= 2021]]
    cov.to_csv(out)
    return cov


def f2_data_overview():
    cfg = load_config()
    cov = coverage_eventdate(cfg)
    wo = pd.read_parquet(ROOT / "data/interim/wo_clean.parquet")

    fig, axes = plt.subplots(1, 2, figsize=(W2, 2.9), width_ratios=[1.55, 1])
    ax = axes[0]
    campuses = list(cov.index)
    years = list(cov.columns)
    M = cov.to_numpy(dtype=float)
    Mlog = np.log10(np.where(M > 0, M, np.nan))
    cmap = LinearSegmentedColormap.from_list("blues", ["#F0F6FB", OI["blue"]])
    im = ax.imshow(Mlog, aspect="auto", cmap=cmap, interpolation="nearest")
    ax.set_xticks(range(len(years)), [str(y) if y % 2 == 0 else "" for y in years], rotation=90)
    retained = cfg["campuses_retained"]
    labels = []
    for c in campuses:
        labels.append(f"U{c:02d}" if c in retained else f"U{c:02d} (excl.)")
    ax.set_yticks(range(len(campuses)), labels)
    # Excluded campuses carry both an "(excl.)" suffix and a gray italic tick,
    # so the exclusion reads at any print size and does not rely on color.
    for tick, c in zip(ax.get_yticklabels(), campuses):
        if c not in retained:
            tick.set_color("#8A8A8A")
            tick.set_style("italic")
    for ci, c in enumerate(campuses):
        if c in retained:
            y0, y1 = cfg["campus_valid_window"][c]
            x0, x1 = years.index(y0), years.index(y1)
            ax.add_patch(patches.Rectangle((x0 - 0.5, ci - 0.5), x1 - x0 + 1, 1,
                                           fill=False, ec=OI["vermillion"], lw=1.4))
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("log10 work orders / year", fontsize=6.5)
    cb.ax.tick_params(labelsize=6.5)
    ax.set_title("(a) Coverage and analysis windows", loc="left")

    ax = axes[1]
    rows = []
    for c in retained:
        w = wo[wo["campus"] == c]
        rows.append({
            "campus": f"U{c:02d}",
            "cost": w["cost"].notna().mean(),
            "labor": w["labor_hours"].notna().mean(),
        })
    dfc = pd.DataFrame(rows)
    ypos = np.arange(len(dfc))
    ax.barh(ypos - 0.19, dfc["cost"], height=0.34, color=LEDGER_COLOR["L1"],
            label="Cost recorded")
    ax.barh(ypos + 0.19, dfc["labor"], height=0.34, color=LEDGER_COLOR["L2"],
            label="Labor hours recorded")
    ax.set_yticks(ypos, dfc["campus"])
    ax.invert_yaxis()
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Share of work orders (analysis window)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, frameon=False)
    ax.set_title("(b) Field coverage, retained campuses", loc="left")
    fig.tight_layout()
    save(fig, "F2_data_overview")


# ---------------------------------------------------------------- F3
def f3_agreement():
    stab = jload("p3_stability.json")
    taub = np.array(stab["taub_pooled"])
    nulls = jload("p3_nulls.json")
    # Panel (a) = per-campus concordance (cited first in the text); panel (b) =
    # pairwise tau-b heatmap (cited second).
    fig, axes = plt.subplots(1, 2, figsize=(W2, 2.9), width_ratios=[1, 1.15])
    ax = axes[0]
    Wc = stab["W_per_campus"]
    camp = sorted(Wc, key=lambda k: int(k))
    x = np.arange(len(camp))
    ax.axhspan(0, nulls["N2"]["W_perm_p95"], color="#EEEEEE", zorder=0)
    ax.text(len(camp) - 0.4, nulls["N2"]["W_perm_p95"] + 0.015,
            "N2 independence ceiling (p95)", fontsize=6, ha="right", color="#666666")
    ax.axhline(stab["W_pooled"], color=OI["black"], lw=0.9, ls="--")
    ax.text(-0.35, stab["W_pooled"] + 0.03, f"pooled W = {stab['W_pooled']:.2f}",
            fontsize=6.5, color="#333333")
    ax.scatter(x, [Wc[c] for c in camp], s=26, color=OI["blue"], zorder=3)
    ax.set_xticks(x, [f"U{int(c):02d}" for c in camp])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Kendall's W across five ledgers")
    ax.grid(axis="y")
    ax.set_title("(a) Concordance by campus", loc="left")

    ax = axes[1]
    cmap = LinearSegmentedColormap.from_list(
        "div", [OI["vermillion"], "#FFFFFF", OI["blue"]])
    im = ax.imshow(taub, cmap=cmap, norm=TwoSlopeNorm(vcenter=0, vmin=-1, vmax=1))
    labels = [f"{l}\n{LEDGER_NAME[l].split()[0]}" for l in LEDGER_COLOR]
    ax.set_xticks(range(5), labels, fontsize=6.5)
    ax.set_yticks(range(5), labels, fontsize=6.5)
    for i in range(5):
        for j in range(5):
            v = taub[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.5,
                    color="white" if abs(v) > 0.6 else "black")
    ax.set_title("(b) Pairwise Kendall tau-b (pooled)", loc="left")
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.ax.tick_params(labelsize=6.5)
    fig.tight_layout()
    save(fig, "F3_agreement")


# ---------------------------------------------------------------- F4
def f4_bump():
    led = pd.read_csv(results_path("p2_ledgers_entity.csv"))
    arch = pd.read_csv(results_path("p4_archetypes.csv"))
    led = led.merge(arch[["campus", "entity", "archetype", "system_desc"]],
                    on=["campus", "entity"], how="left")
    stab = jload("p3_stability.json")
    Wc = stab["W_per_campus"]
    picks = [1, 5, 8]  # highest W, lowest W, mid
    fig, axes = plt.subplots(1, 3, figsize=(W2, 3.5), sharey=True)
    xs = np.arange(5)
    present = set()
    for ax, c in zip(axes, picks):
        g = led[led["campus"] == c].copy()
        g["top10"] = g["mean_rank"].rank(ascending=False) <= 10
        for _, row in g[~g["top10"]].iterrows():
            ax.plot(xs, [row[f"r_L{j}"] for j in range(1, 6)],
                    color="#CCCCCC", lw=0.6, zorder=1)
        # collect label positions, then dodge collisions
        lab = []
        for _, row in g[g["top10"]].iterrows():
            col = ARCH_COLOR.get(row["archetype"], "#888888")
            present.add(row["archetype"])
            ax.plot(xs, [row[f"r_L{j}"] for j in range(1, 6)], color=col,
                    lw=1.4, zorder=3, marker="o", ms=2.4)
            y = row["r_L5"] if not np.isnan(row["r_L5"]) else row["r_L4"]
            lab.append([y, str(row["system_desc"])[:18], col])
        lab.sort(key=lambda t: t[0])
        min_gap = 0.052
        for i in range(1, len(lab)):
            if lab[i][0] - lab[i - 1][0] < min_gap:
                lab[i][0] = lab[i - 1][0] + min_gap
        over = lab[-1][0] - 1.01 if lab and lab[-1][0] > 1.01 else 0
        for y, txt, col in lab:
            ax.text(4.12, y - over, txt, fontsize=5.2, va="center", color=col)
        ax.set_xticks(xs, ["Cost", "Labor", "Chron.", "Shock", "Volat."],
                      fontsize=6.5)
        ax.set_xlim(-0.2, 5.6)
        ax.set_ylim(-0.02, 1.06)
        ax.set_title(f"U{c:02d}  (W = {Wc[str(c)]:.2f})", loc="left")
        ax.grid(axis="y")
    axes[0].set_ylabel("Within-campus percentile rank")
    order = [a for a in ARCH_COLOR if a in present and a != "unremarkable"]
    handles = [plt.Line2D([], [], color=ARCH_COLOR[a], lw=1.6, label=ARCH_LABEL[a])
               for a in order]
    handles.append(plt.Line2D([], [], color="#CCCCCC", lw=1.0, label="Other entities"))
    fig.legend(handles=handles, ncol=len(handles), loc="lower center", frameon=False,
               bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    save(fig, "F4_rank_stability")


# ---------------------------------------------------------------- F5
def f5_meii_floor():
    meii = pd.read_csv(results_path("p3_meii_entity.csv"))
    nulls = jload("p3_nulls.json")
    fig, axes = plt.subplots(1, 2, figsize=(W2, 3.0), width_ratios=[1.1, 1])
    ax = axes[0]
    sig = meii["significant"]
    lim = max(meii["meii"].max(), meii["noise_floor_p95"].max()) * 1.08
    ax.plot([0, lim], [0, lim], color="#888888", lw=0.8, ls="--", zorder=1)
    ax.scatter(meii.loc[~sig, "noise_floor_p95"], meii.loc[~sig, "meii"],
               s=16, facecolor="white", edgecolor="#777777", lw=0.7,
               label="Not significant", zorder=2)
    ax.scatter(meii.loc[sig, "noise_floor_p95"], meii.loc[sig, "meii"],
               s=16, color=OI["blue"], label="Significant (MEII > floor p95)",
               zorder=3)
    ax.set_xlabel("Ledger-matched noise floor, 95th percentile")
    ax.set_ylabel("Observed MEII")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    # legend in the empty upper-right triangle (no entity has MEII far above a
    # high floor), clear of the dense significant cluster in the upper left
    ax.legend(loc="upper right", frameon=True, framealpha=0.9, edgecolor="none",
              fontsize=6.5, handletextpad=0.4, borderpad=0.4)
    ax.text(lim * 0.50, lim * 0.44, "MEII = floor", rotation=45, fontsize=6,
            color="#666666", ha="center", va="center", rotation_mode="anchor")
    ax.set_title("(a) MEII vs sampling-noise floor", loc="left")

    ax = axes[1]
    bins = np.linspace(0, max(lim, meii["meii"].max()), 24)
    ax.hist(meii["meii"], bins=bins, color=OI["blue"], alpha=0.85)
    ymax = ax.get_ylim()[1] * 1.14
    ax.set_ylim(0, ymax)
    ax.axvspan(nulls["N2"]["meii_perm_p5"], nulls["N2"]["meii_perm_p95"],
               color="#EEEEEE", zorder=0)
    ax.axvline(nulls["N2"]["meii_perm_mean"], color="#666666", lw=0.9, ls=":")
    ax.text(nulls["N2"]["meii_perm_mean"], ymax * 0.99,
            "N2 independence ceiling\n(mean, p5-p95 band)", fontsize=6,
            color="#555555", va="top", ha="center")
    n1m = meii["noise_floor_p95"].mean()
    ax.axvline(n1m, color=OI["vermillion"], lw=0.9, ls="--")
    ax.text(n1m, ymax * 0.80, "mean N1\nfloor (p95)", fontsize=6,
            color=OI["vermillion"], ha="center", va="top",
            bbox=dict(fc="white", ec="none", alpha=0.8, pad=0.5))
    ax.set_xlabel("MEII (SD of percentile ranks)")
    ax.set_ylabel("Entities")
    ax.set_title("(b) MEII distribution vs null references", loc="left")
    fig.tight_layout()
    save(fig, "F5_meii_floor")


# ---------------------------------------------------------------- F6
def f6_archetypes():
    arch = pd.read_csv(results_path("p4_archetypes.csv"))
    summ = jload("p4_archetype_summary.json")["system"]
    cfg = load_config()
    th, tl = cfg["theta_high"], cfg["theta_low"]
    arch["best_noncost"] = arch[["r_L2", "r_L3", "r_L4", "r_L5"]].max(axis=1)
    fig, axes = plt.subplots(1, 2, figsize=(W2, 3.1), width_ratios=[1.15, 1])
    ax = axes[0]
    for a in ARCH_COLOR:
        g = arch[arch["archetype"] == a]
        if g.empty:
            continue
        ax.scatter(g["r_L1"], g["best_noncost"], s=22,
                   color=ARCH_COLOR[a], label=ARCH_LABEL[a],
                   edgecolor="white", lw=0.4, zorder=3)
    ax.axvline(tl, color="#999999", lw=0.7, ls=":")
    ax.axvline(th, color="#999999", lw=0.7, ls=":")
    ax.axhline(th, color="#999999", lw=0.7, ls=":")
    ax.text(tl, 1.03, r"$\theta_L$", fontsize=7, ha="center", color="#666666")
    ax.text(th, 1.03, r"$\theta_H$", fontsize=7, ha="center", color="#666666")
    ax.set_xlabel("Recorded-cost percentile rank")
    ax.set_ylabel("Best non-cost percentile rank")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.08)
    ax.set_title("(a) Archetype landscape (campus x system)", loc="left")

    ax = axes[1]
    order = [a for a in ARCH_COLOR if summ["counts"].get(a, 0) > 0]
    counts = [summ["counts"][a] for a in order]
    bshares = [summ["burden_weighted_shares"].get(a, 0) for a in order]
    y = np.arange(len(order))
    ax.barh(y - 0.19, np.array(counts) / sum(counts), height=0.34,
            color=[ARCH_COLOR[a] for a in order])
    ax.barh(y + 0.19, bshares, height=0.34, facecolor="white",
            edgecolor=[ARCH_COLOR[a] for a in order], lw=1.1)
    ax.set_yticks(y, [ARCH_LABEL[a] for a in order], fontsize=6.5)
    ax.invert_yaxis()
    ax.set_xlabel("Share")
    hnd = [patches.Patch(fc="#555555", label="Entity share"),
           patches.Patch(fc="white", ec="#555555", label="Cost-weighted share")]
    ax.legend(handles=hnd, frameon=False, loc="upper center",
              bbox_to_anchor=(0.5, -0.16), ncol=2)
    ax.set_title("(b) Prevalence", loc="left")
    ax.grid(axis="x")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, loc="lower center", frameon=False,
               bbox_to_anchor=(0.5, -0.07), fontsize=6.2)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save(fig, "F6_archetypes")


# ---------------------------------------------------------------- F7
def f7_decision():
    ov = pd.read_csv(results_path("p5_overlap.csv"))
    cb = pd.read_csv(results_path("p5_captured_burden.csv"))
    ov5 = ov[ov["k"] == "abs5"].set_index("rule")
    cb5 = cb[cb["k"] == "abs5"].set_index("rule")
    rules = ["labor_hours", "chronicity", "shock", "volatility", "wo_count",
             "mean_cost_per_wo", "total_burden_L6", "consensus"]
    rule_lab = {
        "recorded_cost": "Recorded cost", "labor_hours": "Labor hours",
        "chronicity": "Chronicity", "shock": "Shock", "volatility": "Volatility",
        "wo_count": "WO count", "mean_cost_per_wo": "Mean cost/WO",
        "total_burden_L6": "Total burden (L6)", "consensus": "Consensus",
    }
    fig, axes = plt.subplots(1, 2, figsize=(W2, 3.0), width_ratios=[1, 1.25])
    ax = axes[0]
    y = np.arange(len(rules))
    vals = [ov5.loc[r, "overlap_with_cost_set"] for r in rules]
    ax.barh(y, vals, height=0.6, color=OI["blue"])
    for yi, v in zip(y, vals):
        ax.text(v + 0.015, yi, f"{v:.2f}", va="center", fontsize=6.5)
    ax.set_yticks(y, [rule_lab[r] for r in rules], fontsize=6.5)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Mean overlap with cost-selected top 5")
    ax.set_title("(a) Selection overlap with cost top 5", loc="left")
    ax.grid(axis="x")

    ax = axes[1]
    rules_m = ["recorded_cost"] + rules
    M = np.array([[cb5.loc[r, f"captured_{l}"] for l in LEDGER_COLOR] for r in rules_m])
    cmap = LinearSegmentedColormap.from_list("seq", ["#FDF3EC", OI["vermillion"]])
    im = ax.imshow(M, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(5), [LEDGER_NAME[l].replace(" ", "\n") for l in LEDGER_COLOR],
                  fontsize=6.2)
    ax.set_yticks(range(len(rules_m)), [rule_lab[r] for r in rules_m], fontsize=6.5)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=6,
                    color="white" if M[i, j] > 0.62 else "black")
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Captured share of campus-total burden", fontsize=6.5)
    cbar.ax.tick_params(labelsize=6.5)
    ax.set_title("(b) Captured burden, top 5 per campus", loc="left")
    fig.tight_layout()
    save(fig, "F7_decision")


# ---------------------------------------------------------------- F8
def f8_robustness():
    summ = jload("p6_robustness_summary.json")
    stab = jload("p3_stability.json")
    fig, axes = plt.subplots(1, 2, figsize=(W2, 2.8))
    ax = axes[0]
    items = []
    if "R1" in summ:
        items.append(("LOCO W", summ["R1"]["W_range"], stab["W_pooled"]))
        items.append(("LOCO H2", summ["R1"]["h2_range"], stab["h2_stat"]))
    if "R2" in summ:
        items.append(("LOYO W", summ["R2"]["W_range"], stab["W_pooled"]))
        items.append(("LOYO H2", summ["R2"]["h2_range"], stab["h2_stat"]))
    y = np.arange(len(items))
    for yi, (lab, rng, obs) in zip(y, items):
        color = OI["blue"] if " W" in lab else OI["vermillion"]
        ax.plot(rng, [yi, yi], color=color, lw=2.2, solid_capstyle="round")
        ax.scatter([obs], [yi], marker="D", s=18, color="black", zorder=3)
    ax.set_yticks(y, [i[0] for i in items], fontsize=6.5)
    ax.invert_yaxis()
    # threshold labels sit inside the axes, rotated along their dotted lines
    # (a y above the top spine collides with the panel title).
    mid_y = (len(items) - 1) / 2
    ax.axvline(0.75, color="#999999", lw=0.7, ls=":")
    ax.text(0.74, mid_y, "H1 threshold", fontsize=6, ha="right", va="center",
            rotation=90, color="#666666")
    ax.axvline(0.20, color="#BBBBBB", lw=0.7, ls=":")
    ax.text(0.19, mid_y, "H2 threshold", fontsize=6, ha="right", va="center",
            rotation=90, color="#666666")
    ax.set_xlim(0, 1)
    ax.set_xlabel("Value (diamond = full-sample)")
    ax.set_title("(a) Leave-one-out ranges", loc="left")
    ax.grid(axis="x")

    ax = axes[1]
    if "R5" in summ:
        tags = ["p90", "p99", "count_p95"]
        labs = ["tau = p90", "tau = p99", "count-based"]
        tb = [summ["R5"][t]["taub_L4_vs_base"] for t in tags]
        W = [summ["R5"][t]["W_pooled"] for t in tags]
        x = np.arange(len(tags))
        ax.bar(x - 0.17, tb, width=0.34, color=LEDGER_COLOR["L4"],
               label="tau-b, L4 variant vs main")
        ax.bar(x + 0.17, W, width=0.34, color=OI["blue"], label="pooled W under variant")
        ax.axhline(stab["W_pooled"], color="#555555", lw=0.8, ls="--")
        ax.text(len(tags) - 0.55, stab["W_pooled"] + 0.02, "main W", fontsize=6,
                color="#555555")
        for xi, v in zip(x - 0.17, tb):
            ax.text(xi, v + 0.02, f"{v:.2f}", ha="center", fontsize=6)
        for xi, v in zip(x + 0.17, W):
            ax.text(xi, v + 0.02, f"{v:.2f}", ha="center", fontsize=6)
        ax.set_xticks(x, labs, fontsize=6.5)
        # headroom so the legend clears the tallest bar (p90 tau-b = 0.97)
        ax.set_ylim(0, 1.28)
        ax.legend(frameon=False, loc="upper right", fontsize=6.5)
    ax.set_title("(b) Shock-threshold sensitivity (R5)", loc="left")
    ax.grid(axis="y")
    fig.tight_layout()
    save(fig, "F8_robustness")


if __name__ == "__main__":
    # Only F2 survives into the two-layer manuscript. The other builders in
    # this module belong to the superseded single-layer analysis: they still
    # write files named F1/F3..F7, so calling them here would silently
    # overwrite the revised figures that p7b_revised_figures.py produces.
    # They are kept for provenance and are unreachable from the pipeline.
    f2_data_overview()
    print("Phase 7 figures complete (F2; F1 and F3-F9 come from p7b).")
