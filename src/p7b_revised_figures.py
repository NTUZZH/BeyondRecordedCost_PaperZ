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
        "metric-stable / risk-aligned: one metric adequate",
    "stable_divergent":
        "metric-stable / repositioned by risk: separate risk review",
    "unstable_aligned":
        "metric-sensitive / risk-aligned: disclose criterion",
    "unstable_divergent":
        "metric-sensitive / repositioned by risk: multi-criteria rule",
}


def f1_two_layer():
    """Compact two-layer framework diagram (the layout the authors approved).

    Each layer carries the management question it answers, so the reader can
    tell the same-question layer from the different-objective layer without
    the caption, but the geometry stays the tight single-row flow that fits
    the column without overflow.
    """
    fig, ax = plt.subplots(figsize=(W2, 3.15))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 41)
    ax.axis("off")

    def box(x, y, w, h, text, fc="#F7F7F7", ec="#555555", fs=7, bold=False,
            lw=0.8):
        ax.add_patch(patches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.4", fc=fc, ec=ec, lw=lw))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, fontweight="bold" if bold else "normal")

    def arrow(x0, y0, x1, y1):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="->", lw=0.9, color="#555555"))

    box(0.5, 14, 15, 10, "CMMS\nwork-order\nrecords", fc="#E8E8E8", bold=True)

    # Layer A: the four alternative burden measures (same question)
    ax.add_patch(patches.FancyBboxPatch((20.5, 9.5), 21.5, 27,
                 boxstyle="round,pad=0.5", fc="none", ec="#888888",
                 lw=1.0, linestyle="--"))
    ax.text(31.2, 40.3, "Layer A: alternative burden measures",
            ha="center", fontsize=6.8, color="#333333", style="italic")
    ax.text(31.2, 38.4, "Which systems create the greatest realized burden?",
            ha="center", fontsize=5.9, color="#555555")
    for l, y in zip(BURDEN, (30.5, 24.5, 18.5, 12.5)):
        box(22, y, 18.5, 4.6, RV_NAME[l], fc="white", ec=RV_COLOR[l], fs=6.8)
        arrow(15.5, 19, 21.5, y + 2.3)

    # Layer B: the distinct budget-risk objective (different question)
    ax.add_patch(patches.FancyBboxPatch((20.5, 1.0), 21.5, 6.4,
                 boxstyle="round,pad=0.5", fc="none", ec=OI["purple"],
                 lw=1.0, linestyle="--"))
    ax.text(31.2, 0.2, "Layer B: distinct management objective. Which systems"
            "\ncreate the largest annual budget surprises?",
            ha="center", va="top", fontsize=5.9, color="#555555")
    box(22, 2.2, 18.5, 4.2, "Budget risk", fc="white",
        ec=RV_COLOR["L5r"], fs=6.8)
    arrow(15.5, 16, 21.5, 4.5)

    box(48, 24, 19, 10, "Metric sensitivity\nKendall $W$, top-$k$ overlap,"
        "\nrank spread $I^B$ vs\nvalidated noise floors", fs=6.6)
    box(48, 10.5, 19, 9, "Risk repositioning\nrisk gap $G$, four\nentity profiles",
        fs=6.6)
    for y in (32.8, 26.8, 20.8, 14.8):
        arrow(40.5 + 0.5, y, 47.5, 29)
    arrow(40.5 + 0.5, 4.3, 47.5, 14)
    box(73, 17, 15, 11, "Robust\nshortlisting\nminimax regret,\nstakeholder\nweights", fs=6.6)
    arrow(67.5, 29, 72.5, 24.5)
    arrow(67.5, 15, 72.5, 20.5)
    box(92, 19, 7.5, 7, "Priority\nshortlist", fc="#E8E8E8", bold=True, fs=6.6)
    arrow(88.5, 22.5, 91.5, 22.5)
    ax.text(57.5, 5.2, "noise calibration validated by simulation\n"
            "(work-order and year-block resampling)",
            ha="center", fontsize=6.2, color="#444444")
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
    ax.text(0.02, d3["W_burden_pooled"] - 0.05,
            f"pooled $W$ = {d3['W_burden_pooled']:.2f}", fontsize=6.5)
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
    im = ax.imshow(M, vmin=-1, vmax=1, cmap="RdBu_r")
    labs = ["Cost", "Labor", "Volume", "Tail", "Budget\nrisk"]
    ax.set_xticks(range(n), labs, fontsize=6.3)
    ax.set_yticks(range(n), labs, fontsize=6.3)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                    fontsize=6,
                    color="white" if abs(M[i, j]) > 0.55 else "black")
    ax.axhline(3.5, color="#333333", lw=1.4)
    ax.axvline(3.5, color="#333333", lw=1.4)
    # Bracket grouping the four burden measures above the heat map.
    ax.annotate("alternative burden measures",
                xy=(1.5, -0.62), xytext=(1.5, -0.62),
                ha="center", fontsize=6.2, color="#333333",
                annotation_clip=False)
    ax.plot([-0.45, 3.45], [-0.48, -0.48], color="#333333", lw=0.9,
            clip_on=False)
    ax.plot([-0.45, -0.45], [-0.48, -0.40], color="#333333", lw=0.9,
            clip_on=False)
    ax.plot([3.45, 3.45], [-0.48, -0.40], color="#333333", lw=0.9,
            clip_on=False)
    ax.set_title("(b) Pairwise Kendall $\\tau$-b (pooled)", loc="left",
                 pad=16)
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
    save(fig, "F4_rank_stability")


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
                   s=18, lw=0.8, label="within noise floor")
        ax.scatter(f[sig], o[sig], color=OI["blue"], s=18,
                   label="metric-sensitive beyond noise floor")
        share = 100 * sig.mean()
        ax.set_title(f"({'ab'[mode=='year_block']}) {label}: "
                     f"{share:.0f}\\% flagged".replace("\\%", "%"),
                     loc="left", fontsize=7.5)
        ax.set_xlabel("Expected rank spread from matched resampling\n"
                      "(95th percentile)", fontsize=7)
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.grid(True, lw=0.3)
    axes[0].set_ylabel("Observed within-burden rank spread $I^B$",
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
    ax.axhline(q3, color="#888888", lw=0.8, ls=":")
    ax.text(ax.get_xlim()[1] * 0.99, q3 + 0.012,
            "repositioned by risk above this line (gap upper quartile)",
            ha="right", fontsize=6, color="#555555")
    ax.set_xlabel("Within-burden rank spread $I^B$", fontsize=7.5)
    ax.set_ylabel("Risk-repositioning gap $G$", fontsize=7.5)
    leg = ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16),
                    ncol=2, fontsize=5.9, frameon=False,
                    title="profile and supported action (black edge = "
                          "top-20% membership metric-dependent)")
    leg.get_title().set_fontsize(5.9)
    ax.grid(True, lw=0.3)
    fig.tight_layout()
    save(fig, "F7_profiles")


def f7_decision():
    sl = jload("p5b_shortlist.json")
    q20 = sl["q20"]
    rules = ["single_cost", "single_labor", "single_volume", "single_tail",
             "single_risk", "burden_mean", "burden_median", "all_mean",
             "minimax"]
    names = ["Cost only", "Labor only", "Volume only", "Tail only",
             "Budget risk only", "Burden mean rank",
             "Burden median rank", "All-criteria mean",
             "Minimax regret"]
    worst = [q20[r]["worst_regret"] for r in rules]
    order = np.argsort(worst)[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(W2, 3.0),
                             gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axes[0]
    ypos = np.arange(len(rules))
    colors = [OI["vermillion"] if rules[i] == "minimax" else
              (OI["blue"] if rules[i].startswith("single") else OI["sky"])
              for i in order]
    ax.barh(ypos, [worst[i] for i in order], color=colors, height=0.62)
    ax.set_yticks(ypos, [names[i] for i in order], fontsize=6.5)
    for y, i in zip(ypos, order):
        ax.text(worst[i] + 0.004, y, f"{worst[i]:.3f}", va="center",
                fontsize=6)
    ax.set_xlabel("Worst-case mean top-20% regret across criteria",
                  fontsize=7)
    ax.set_xlim(0, max(worst) * 1.18)
    ax.set_title("(a) Decision rules compared", loc="left")
    ax.grid(axis="x")

    ax = axes[1]
    scen = sl["weights_q20"]["scenarios"]
    scen_names = ["finance", "workforce", "operations", "risk"]
    crit = ["L1", "L2", "L3", "L4x", "L5r"]
    M = np.array([[scen[s][c] for c in crit] for s in scen_names])
    im = ax.imshow(M, cmap="Oranges", vmin=0, vmax=0.5)
    ax.set_xticks(range(5), ["Cost", "Labor", "Volume", "Tail",
                             "Budget\nrisk"], fontsize=6.3)
    ax.set_yticks(range(4), [s.capitalize() + "-heavy" for s in scen_names],
                  fontsize=6.3)
    for i in range(4):
        for j in range(5):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                    fontsize=6, color="white" if M[i, j] > 0.5 else "black")
    ax.set_title("(b) Regret by stakeholder weighting", loc="left")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
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
