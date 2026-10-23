"""Phase 7b: manuscript figures for the two-layer framework.

File names match the printed figure numbers: F1 framework, F3 agreement,
F4 rank stability, F5 noise floors, F6 simulation calibration, F7 entity
profiles, F8 decision rules, plus FS1 (the full simulation panel for the
Supplementary Material). F2 (data overview) comes from p7_figures.py; the
other builders in that module belong to the superseded single-layer
analysis and are no longer run.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from p7_figures import MM, OI, W1, W2, jload, save  # noqa: E402  (rcParams side effect)
import matplotlib.patches as patches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from utils.io import results_path  # noqa: E402

RV_COLOR = {"L1": OI["blue"], "L2": OI["orange"], "L3": OI["green"],
            "L4x": OI["vermillion"], "L5r": OI["purple"]}
RV_NAME = {"L1": "Recorded cost", "L2": "Labor hours",
           "L3": "Persistent volume", "L4x": "Tail expenditure",
           "L5r": "Budget risk"}
BURDEN = ["L1", "L2", "L3", "L4x"]
PROFILE_COLOR = {"stable_aligned": "#BBBBBB",
                 "stable_divergent": OI["purple"],
                 "unstable_aligned": OI["blue"],
                 "unstable_divergent": OI["vermillion"]}
# Profile labels state the diagnosis and the action it supports.
PROFILE_NAME = {
    "stable_aligned":
        "Metric-stable, risk-aligned: report one measure",
    "stable_divergent":
        "Metric-stable, repositioned by risk: review budget risk",
    "unstable_aligned":
        "Metric-sensitive, risk-aligned: state the measure",
    "unstable_divergent":
        "Metric-sensitive, repositioned by risk: multi-criteria rule",
}


def f1_two_layer():
    """Two-layer framework diagram.

    Five aligned stages (records -> competing criteria -> entity
    diagnostics -> decision layer -> output). The two middle diagnostic
    boxes are equal-size structured panels (bold header, hairline rule,
    method detail); a vertical connector shows that crossing the two
    diagnostics yields the four entity profiles. Simple monochrome
    line-art pictograms (database cylinder, per-criterion glyphs,
    balance scales, ranked list) replace part of the box text; no emoji,
    no fills, journal line weight, and all text black.
    """
    fig, ax = plt.subplots(figsize=(W2, 4.35))
    ax.set_xlim(0, 100)
    ax.set_ylim(-8, 51)
    ax.axis("off")
    IC = "#555555"  # icon stroke, matches arrows and box edges

    def box(x, y, w, h, text, fc="#F7F7F7", ec="#555555", fs=7, bold=False,
            lw=0.8):
        ax.add_patch(patches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.4", fc=fc, ec=ec, lw=lw))
        if text:
            ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                    fontsize=fs, fontweight="bold" if bold else "normal")

    def panel(x, y, w, h, header, lines, ec="#555555"):
        """Equal-size diagnostic panel: bold header, hairline rule, detail."""
        ax.add_patch(patches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.4", fc="#F7F7F7", ec=ec,
            lw=0.9))
        ax.text(x + w / 2, y + h - 2.3, header, ha="center", va="center",
                fontsize=7.2, fontweight="bold")
        ax.plot([x + 1.8, x + w - 1.8], [y + h - 4.4, y + h - 4.4],
                color="#999999", lw=0.6)
        ax.text(x + w / 2, y + (h - 5.0) / 2, "\n".join(lines),
                ha="center", va="center", fontsize=6.4, linespacing=1.6)

    def arrow(x0, y0, x1, y1, color="#555555", ls="-"):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="->", lw=0.9, color=color,
                                    linestyle=ls))

    # --- Line-art pictograms (data coords; x and y units print ~equal) ---
    def icon_db(cx, cy, s=2.1):
        """Database cylinder for the records source."""
        top, bot = cy + 0.85 * s, cy - 0.85 * s
        ax.add_patch(patches.Ellipse((cx, top), 2 * s, 0.62 * s, fill=False,
                                     ec=IC, lw=0.9))
        ax.plot([cx - s, cx - s], [bot, top], color=IC, lw=0.9)
        ax.plot([cx + s, cx + s], [bot, top], color=IC, lw=0.9)
        ax.add_patch(patches.Arc((cx, bot), 2 * s, 0.62 * s, theta1=180,
                                 theta2=360, ec=IC, lw=0.9))
        ax.add_patch(patches.Arc((cx, cy), 2 * s, 0.62 * s, theta1=180,
                                 theta2=360, ec=IC, lw=0.9))

    def icon_scales(cx, cy, s=2.0):
        """Balance scales for the minimax trade-off."""
        ax.plot([cx, cx], [cy - s, cy + 0.8 * s], color=IC, lw=0.9)
        ax.plot([cx - 0.9 * s, cx + 0.9 * s], [cy - s, cy - s],
                color=IC, lw=0.9)  # base
        ax.plot([cx - s, cx + s], [cy + 0.8 * s, cy + 0.8 * s],
                color=IC, lw=0.9)  # beam
        for dx in (-s, s):
            ax.plot([cx + dx, cx + dx], [cy + 0.8 * s, cy + 0.25 * s],
                    color=IC, lw=0.7)
            ax.add_patch(patches.Arc((cx + dx, cy + 0.25 * s), 0.9 * s,
                                     0.9 * s, theta1=180, theta2=360,
                                     ec=IC, lw=0.9))

    def icon_list(cx, cy, s=2.0):
        """Ranked list for the priority shortlist."""
        w, h = 1.5 * s, 1.9 * s
        ax.add_patch(patches.FancyBboxPatch(
            (cx - w / 2, cy - h / 2), w, h, boxstyle="round,pad=0.12",
            fc="white", ec=IC, lw=0.9))
        for i, fy in enumerate((0.55, 0.0, -0.55)):
            yy = cy + fy * s
            ax.plot([cx - 0.5 * w + 0.55], [yy], marker="s", ms=1.6,
                    color=IC, lw=0)
            ax.plot([cx - 0.5 * w + 1.1, cx + 0.5 * w - 0.4], [yy, yy],
                    color=IC, lw=0.7)

    def icon_criterion(kind, cx, cy, color, s=1.25):
        """Small per-criterion glyph, drawn in the criterion's hue."""
        if kind == "L1":      # recorded cost: coin
            ax.add_patch(patches.Ellipse((cx, cy), 2 * s, 2 * s, fill=False,
                                         ec=color, lw=0.9))
            ax.text(cx, cy - 0.06, "\\$", ha="center", va="center",
                    fontsize=5.6, color=color)
        elif kind == "L2":    # labor hours: clock
            ax.add_patch(patches.Ellipse((cx, cy), 2 * s, 2 * s, fill=False,
                                         ec=color, lw=0.9))
            ax.plot([cx, cx], [cy, cy + 0.62 * s], color=color, lw=0.8)
            ax.plot([cx, cx + 0.5 * s], [cy, cy], color=color, lw=0.8)
        elif kind == "L3":    # persistent volume: stacked bars
            for fy in (-0.62, 0.0, 0.62):
                ax.plot([cx - s, cx + s], [cy + fy * s, cy + fy * s],
                        color=color, lw=1.5, solid_capstyle="butt")
        elif kind == "L4x":   # tail expenditure: flat series with one spike
            xs = np.array([-1.0, -0.45, -0.1, 0.15, 0.4, 1.0])
            ys = np.array([-0.45, -0.35, -0.45, 0.85, -0.4, -0.5])
            ax.plot(cx + xs * s, cy + ys * s, color=color, lw=0.9)
        elif kind == "L5r":   # budget risk: volatile zigzag
            xs = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
            ys = np.array([-0.5, 0.7, -0.35, 0.85, -0.6])
            ax.plot(cx + xs * s, cy + ys * s, color=color, lw=0.9)

    # Stage labels: one aligned band across the top.
    for x, tag in ((8.0, "Records"), (31.2, "Step 1\nCompute the five criteria"),
                   (58.5, "Steps 2 and 3\nTest agreement, assign profiles"),
                   (80.75, "Step 4\nChoose the shortlist"), (95.5, "Output")):
        ax.text(x, 49.5, tag, ha="center", va="center", fontsize=7.0,
                fontweight="bold", linespacing=1.3)

    # (1) Records: database pictogram over a short label.
    box(0.5, 13.0, 15, 13, "", fc="#E8E8E8")
    icon_db(8.0, 22.3)
    ax.text(8.0, 17.2, "CMMS\nwork-order records", ha="center", va="center",
            fontsize=6.8, fontweight="bold")
    ax.text(8.0, 14.4, "asset, date, cost or hours", ha="center",
            va="center", fontsize=5.8)

    # (2) Layer A: the four alternative burden measures (same question)
    ax.add_patch(patches.FancyBboxPatch((20.5, 9.5), 21.5, 29,
                 boxstyle="round,pad=0.5", fc="none", ec="#888888",
                 lw=1.0, linestyle="--"))
    ax.text(31.2, 43.3, "Four burden measures",
            ha="center", fontsize=7.2, fontweight="bold")
    ax.text(31.2, 41.2, "Which systems use the most maintenance resources?",
            ha="center", fontsize=6.3)
    for l, y in zip(BURDEN, (32.6, 26.1, 19.6, 13.1)):
        box(22, y, 18.5, 5.0, "", fc="white", ec=RV_COLOR[l])
        icon_criterion(l, 24.6, y + 2.5, RV_COLOR[l])
        ax.text(32.2, y + 2.5, RV_NAME[l], ha="center", va="center",
                fontsize=6.9)
        arrow(15.5, 19.5, 21.5, y + 2.5)

    # Layer B: the distinct budget-risk objective (different question)
    ax.add_patch(patches.FancyBboxPatch((20.5, 0.6), 21.5, 7.2,
                 boxstyle="round,pad=0.5", fc="none", ec=OI["purple"],
                 lw=1.0, linestyle="--"))
    ax.text(31.2, -0.7, "Which systems make annual spending\n"
            "hardest to predict?", ha="center", va="top", fontsize=6.3)
    box(22, 1.9, 18.5, 4.6, "", fc="white", ec=RV_COLOR["L5r"])
    icon_criterion("L5r", 24.6, 4.2, RV_COLOR["L5r"])
    ax.text(32.2, 4.2, "Budget risk", ha="center", va="center", fontsize=6.9)
    arrow(15.5, 16.0, 21.5, 4.2)

    # (3) The two diagnostic steps: identical geometry, sparse detail
    # (the running text and the caption carry the method specifics).
    PX, PW, PH = 47.0, 23.0, 12.0
    panel(PX, 25.5, PW, PH, "Step 2: agreement test",
          ["Kendall's $W$ and top-20% overlap,",
           "rank spread $I^B$ of each system",
           "against its sampling-noise floor"])
    panel(PX, 4.0, PW, PH, "Step 3: profiles",
          ["risk gap $G$ of each system,",
           "crossed with the Step 2 result,",
           "gives four profiles and actions"])
    for y in (35.1, 28.6, 22.1, 15.6):
        arrow(41.0, y, PX - 0.5, 31.5)
    arrow(41.0, 4.2, PX - 0.5, 10.0)
    # Crossing the two diagnostics yields the four entity profiles.
    arrow(PX + PW / 2, 24.9, PX + PW / 2, 16.6)

    # (4) Decision layer: balance scales over the rule names.
    box(73.5, 14.5, 14.5, 14, "", fc="#F7F7F7")
    icon_scales(80.75, 24.7)
    ax.text(80.75, 20.4, "Minimax-regret list", ha="center", va="center",
            fontsize=6.9, fontweight="bold")
    ax.text(80.75, 17.2, "robust core and\ncontested systems",
            ha="center", va="center", fontsize=6.2)
    arrow(PX + PW + 0.5, 31.5, 73.0, 25.5)
    arrow(PX + PW + 0.5, 10.0, 73.0, 17.5)

    # (5) Output: ranked-list pictogram over the label.
    box(91.5, 16.5, 8, 10, "", fc="#E8E8E8")
    icon_list(95.5, 23.1)
    ax.text(95.5, 18.6, "Priority\nshortlist", ha="center", va="center",
            fontsize=6.8, fontweight="bold")
    arrow(88.5, 21.5, 91.0, 21.5)
    save(fig, "F1_framework")


def f3_agreement():
    df = pd.read_csv(results_path("p2b_ledgers_entity.csv"))
    d3 = jload("p3d_revised.json")
    fig, axes = plt.subplots(1, 2, figsize=(W2, 3.1),
                             gridspec_kw={"width_ratios": [1, 1.15]})
    ax = axes[0]
    camp = sorted(d3["W_burden_by_campus"])
    vals = [d3["W_burden_by_campus"][c] for c in camp]
    ax.scatter(range(len(camp)), vals, color=OI["blue"], s=26, zorder=3)
    ax.axhline(d3["W_burden_pooled"], color="#333333", ls="--", lw=0.9)
    ax.set_xticks(range(len(camp)), [f"U{int(c):02d}" for c in camp],
                  fontsize=6.5)
    ax.set_ylim(0.4, 1.0)
    ax.set_ylabel("Kendall's $W$ across burden measures", fontsize=7)
    ax.set_title("(a) Burden concordance by campus", loc="left")
    ax.grid(axis="y")

    ax = axes[1]
    cols = BURDEN + ["L5r"]
    R = df[[f"r_{l}" for l in cols]]
    n = len(cols)
    M = np.full((n, n), np.nan)
    from scipy.stats import kendalltau
    for i in range(n):
        for j in range(n):
            x, y = R.iloc[:, i], R.iloc[:, j]
            ok = x.notna() & y.notna()
            M[i, j] = kendalltau(x[ok], y[ok]).statistic
    from matplotlib.colors import LinearSegmentedColormap
    blues = plt.get_cmap("Blues")
    pale = LinearSegmentedColormap.from_list(
        "pale_blues", blues(np.linspace(0.02, 0.62, 256)))
    im = ax.imshow(M, vmin=0, vmax=1, cmap=pale)
    labs = ["Cost", "Labor", "Volume", "Tail", "Budget\nrisk"]
    ax.set_xticks(range(n), labs, fontsize=6.3)
    ax.set_yticks(range(n), labs, fontsize=6.3)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                    fontsize=6, color="black")
    ax.axhline(3.5, color="#333333", lw=1.4)
    ax.axvline(3.5, color="#333333", lw=1.4)
    ax.set_title("(b) Pairwise Kendall $\\tau$-b (pooled)", loc="left")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    fig.tight_layout()
    save(fig, "F3_agreement")


def f4_bump():
    df = pd.read_csv(results_path("p2b_ledgers_entity.csv"))
    d3 = jload("p3d_revised.json")
    Wc = {int(k): v for k, v in d3["W_burden_by_campus"].items()}
    picks = [max(Wc, key=Wc.get), min(Wc, key=Wc.get),
             sorted(Wc, key=Wc.get)[len(Wc) // 2]]
    picks = sorted(set(picks), key=lambda c: -Wc[c])
    fig, axes = plt.subplots(1, len(picks), figsize=(W2, 2.9), sharey=True)
    cols = [f"r_{l}" for l in BURDEN]
    for ax, c in zip(axes, picks):
        g = df[df["campus"] == c].copy()
        g["sysname"] = g["entity"].astype(str)
        top = g.nlargest(min(8, len(g)), "mean_burden_rank")
        x = np.arange(4)
        for _, row in g.iterrows():
            y = row[cols].to_numpy(dtype=float)
            istop = row.name in top.index
            color = OI["blue"] if istop else "#CCCCCC"
            lw = 1.4 if istop else 0.7
            ax.plot(x, y, color=color, lw=lw, alpha=0.9 if istop else 0.6,
                    marker="o", ms=2.4 if istop else 1.4)
        ax.set_xticks(x, ["Cost", "Labor", "Volume", "Tail"], fontsize=6.3)
        ax.set_title(f"U{c:02d}  ($W$ = {Wc[c]:.2f})", fontsize=7.5)
        ax.set_ylim(0, 1.03)
        ax.grid(axis="y")
    axes[0].set_ylabel("Within-campus percentile rank", fontsize=7)
    fig.suptitle("")
    fig.tight_layout()
    save(fig, "FS2_rank_stability")


def f5_meii_floor():
    ent = pd.read_csv(results_path("p3d_entity.csv"))
    fig, axes = plt.subplots(1, 2, figsize=(W2, 2.8), sharey=True)
    for ax, mode, label in ((axes[0], "wo", "work-order resampling"),
                            (axes[1], "year_block", "year-block resampling")):
        f = ent[f"meii_b_floor_{mode}"]
        o = ent["meii_burden"]
        sig = ent[f"meii_b_sig_{mode}"]
        lim = max(f.max(), o.max()) * 1.1
        ax.plot([0, lim], [0, lim], color="#999999", lw=0.8, ls="--")
        ax.scatter(f[~sig], o[~sig], facecolor="none", edgecolor="#777777",
                   s=18, lw=0.8, label="within the noise floor")
        ax.scatter(f[sig], o[sig], color=OI["blue"], s=18,
                   label="above the noise floor")
        share = 100 * sig.mean()
        ax.set_title(f"({'ab'[mode=='year_block']}) {label}: "
                     f"{share:.0f}\\% flagged".replace("\\%", "%"),
                     loc="left", fontsize=7.5)
        ax.set_xlabel("Expected rank spread from matched resampling\n"
                      "(95th percentile)", fontsize=7)
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.grid(True, lw=0.3)
    axes[0].set_ylabel("Observed rank spread $I^B$",
                       fontsize=7)
    axes[1].legend(loc="lower right", fontsize=6, frameon=True,
                   framealpha=0.9)
    fig.tight_layout()
    save(fig, "F5_meii_floor")


def f6_profile_map():
    ent = pd.read_csv(results_path("p3d_entity.csv"))
    d3 = jload("p3d_revised.json")
    fig, ax = plt.subplots(figsize=(W2 * 0.72, 3.4))
    q3 = d3["gap_upper_quartile"]
    for prof in ("stable_aligned", "stable_divergent",
                 "unstable_aligned", "unstable_divergent"):
        g = ent[ent["profile"] == prof]
        contested = g["topk_burden_count"].isin([1, 2, 3])
        ax.scatter(g.loc[~contested, "meii_burden"],
                   g.loc[~contested, "gap_risk"],
                   s=18 + 46 * g.loc[~contested, "mean_burden_rank"],
                   color=PROFILE_COLOR[prof], alpha=0.85, lw=0,
                   label=PROFILE_NAME[prof])
        ax.scatter(g.loc[contested, "meii_burden"],
                   g.loc[contested, "gap_risk"],
                   s=18 + 46 * g.loc[contested, "mean_burden_rank"],
                   facecolor=PROFILE_COLOR[prof], edgecolor="black",
                   lw=0.9, alpha=0.9)
    ax.axhline(q3, color="#555555", lw=0.8, ls=":")
    ax.scatter([], [], s=40, facecolor="white", edgecolor="black", lw=0.9,
               label="Top-20% membership depends on the measure")
    ax.set_xlabel("Rank spread across the burden measures $I^B$",
                  fontsize=7.5)
    ax.set_ylabel("Risk gap $G$", fontsize=7.5)
    leg = ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16),
                    ncol=2, fontsize=6.0, frameon=True, fancybox=False,
                    edgecolor="black", framealpha=1.0)
    leg.get_frame().set_linewidth(0.5)
    ax.grid(True, lw=0.3)
    fig.tight_layout()
    save(fig, "F7_profiles")


def f7_decision():
    """(a) Worst-case regret of the nine rules. (b) Trade-off between the
    smallest coverage of the four burden top sets and the coverage of the
    budget-risk top set (epsilon-constraint curve), with every rule placed
    on the same axes; rules that select identical coverage share a marker."""
    sl = jload("p5b_shortlist.json")
    te = jload("p10e_tradeoff.json")
    q20 = sl["q20"]
    rules = ["single_cost", "single_labor", "single_volume", "single_tail",
             "single_risk", "burden_mean", "burden_median", "all_mean",
             "minimax"]
    names = {"single_cost": "Cost only", "single_labor": "Labor hours only",
             "single_volume": "Volume only", "single_tail": "Tail only",
             "single_risk": "Budget risk only",
             "burden_mean": "Burden mean rank",
             "burden_median": "Burden median rank",
             "all_mean": "All-criteria mean rank",
             "minimax": "Minimax regret"}
    worst = [q20[r]["worst_regret"] for r in rules]
    order = np.argsort(worst)[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(W2, 3.2),
                             gridspec_kw={"width_ratios": [1.0, 1.15]})
    ax = axes[0]
    ypos = np.arange(len(rules))
    colors = [OI["vermillion"] if rules[i] == "minimax" else OI["sky"]
              for i in order]
    ax.barh(ypos, [worst[i] for i in order], color=colors, height=0.62)
    ax.set_yticks(ypos, [names[rules[i]] for i in order], fontsize=6.5)
    ax.set_xlabel("Worst-case regret across the five criteria", fontsize=7)
    ax.set_xlim(0, 0.4)
    ax.set_title("(a) Worst-case regret", loc="left")
    ax.grid(axis="x")

    ax = axes[1]
    fr = np.array(te["frontier"])
    fr = fr[np.argsort(fr[:, 0])]
    ax.plot(fr[:, 0], fr[:, 1], color="black", lw=0.9, zorder=2,
            label="Best achievable trade-off")
    ax.scatter(fr[:, 0], fr[:, 1], s=10, color="black", zorder=3)
    groups = {}
    for r in rules:
        key = tuple(np.round(te["rules"][r], 4))
        groups.setdefault(key, []).append(r)
    style = {"minimax": ("o", OI["vermillion"]),
             "single_cost": ("s", OI["blue"]), "single_labor": ("D", OI["orange"]),
             "single_tail": ("^", OI["green"]), "single_risk": ("v", OI["purple"]),
             "burden_mean": ("P", OI["sky"]), "all_mean": ("X", "#666666")}
    for key, rs in groups.items():
        lead = next(r for r in rs if r in style)
        mk, col = style[lead]
        lab = " and ".join([names[rs[0]]] + [names[r].lower() for r in rs[1:]])
        ax.scatter(*key, marker=mk, s=46, facecolor=col, edgecolor="black",
                   lw=0.5, zorder=4, label=lab)
    ax.set_xlabel("Smallest coverage of the four burden top sets",
                  fontsize=7)
    ax.set_ylabel("Coverage of the budget-risk top set", fontsize=7)
    ax.set_xlim(0.6, 0.86)
    ax.set_ylim(0.6, 1.03)
    ax.set_title("(b) Burden coverage against risk coverage", loc="left")
    ax.grid(True, lw=0.3)
    leg = ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0),
                    fontsize=5.8, frameon=True, fancybox=False,
                    edgecolor="black", framealpha=1.0, borderaxespad=0)
    leg.get_frame().set_linewidth(0.5)
    fig.tight_layout()
    save(fig, "F8_decision")


def _sim_type1_panel(ax, sim):
    scens = ["S0", "S3"]
    x = np.arange(len(scens))
    for off, mode, color, lab in ((-0.17, "wo", OI["blue"],
                                   "work-order"),
                                  (0.17, "year_block", OI["orange"],
                                   "year-block")):
        vals = [100 * sim[s][mode]["type1_meii_b"] for s in scens]
        ax.bar(x + off, vals, width=0.34, color=color, label=lab)
        for xi, v in zip(x + off, vals):
            ax.text(xi, v + 0.25, f"{v:.1f}", ha="center", fontsize=6)
    ax.axhline(5.0, color="#333333", lw=0.9, ls="--")
    ax.text(0.5, 5.25, "nominal 5%", fontsize=6, color="#333333",
            ha="center")
    ax.set_xticks(x, ["S0 independent null", "S3 dependent null"],
                  fontsize=6.5)
    ax.set_ylabel("Per-entity Type-I error (%)", fontsize=7)
    ax.set_ylim(0, 11)
    ax.legend(fontsize=6, frameon=False)
    ax.grid(axis="y")


def f9_simulation():
    """Main-text calibration figure: Type-I error only. The full power and
    localization panel is exported separately for the Supplementary
    Material (FS1)."""
    sim = jload("p3e_simulation.json")
    fig, ax = plt.subplots(figsize=(W2 * 0.55, 2.7))
    _sim_type1_panel(ax, sim)
    ax.set_title("Calibration of the metric-sensitivity flag", loc="left")
    fig.tight_layout()
    save(fig, "F6_simulation")

    # Supplementary version: calibration + power/localization.
    fig, axes = plt.subplots(1, 2, figsize=(W2, 2.7))
    _sim_type1_panel(axes[0], sim)
    axes[0].set_title("(a) Calibration of the metric-sensitivity flag",
                      loc="left")
    ax = axes[1]
    bars = [
        ("$I^B$ power\n(S2 divergent)", 100 * sim["S2"]["wo"]
         ["power_meii_b_on_divergent"], OI["blue"]),
        ("$I^B$ spillover\n(S2 null entities)", 100 * sim["S2"]["wo"]
         ["type1_meii_b"], OI["sky"]),
        ("gap top-3 recall\n(S1 divergent)", 100 * sim["S1"]["wo"]
         ["gap_top3_recall_divergent"], OI["purple"]),
        ("gap flag power\n(S1, per-entity test)", 100 * sim["S1"]["wo"]
         ["power_gap_on_divergent"], "#999999"),
    ]
    x = np.arange(len(bars))
    ax.bar(x, [b[1] for b in bars], color=[b[2] for b in bars], width=0.6)
    for xi, b in zip(x, bars):
        ax.text(xi, b[1] + 1.2, f"{b[1]:.0f}", ha="center", fontsize=6)
    ax.set_xticks(x, [b[0] for b in bars], fontsize=5.6)
    ax.set_ylabel("Rate (%)", fontsize=7)
    ax.set_ylim(0, 70)
    ax.set_title("(b) Power and localization", loc="left")
    ax.grid(axis="y")
    fig.tight_layout()
    save(fig, "FS1_simulation_full")


if __name__ == "__main__":
    f1_two_layer()
    f3_agreement()
    f4_bump()
    f5_meii_floor()
    f6_profile_map()
    f7_decision()
    f9_simulation()
    print("revised figures complete")
