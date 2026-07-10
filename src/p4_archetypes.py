"""Phase 4: economic archetype classification.

Ordered decision list on within-campus percentile ranks; first match wins.
Applied to the primary (campus x system) and secondary (campus x building x
system) entity levels; theta_H sensitivity at 0.75/0.85.

Outputs: results/p4_archetypes.csv, results/p4_archetypes_bldg.csv,
         results/p4_archetype_summary.json, results/p4_examples.json
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils.io import load_config, results_path, write_json

ARCHETYPES = [
    "stable_priority", "hidden_burden", "cash_sink", "labor_sink",
    "chronic_drain", "shock_generator", "budget_destabiliser",
    "representation_sensitive", "unremarkable",
]


def classify(df: pd.DataFrame, th: float, tl: float, meii_q3: float) -> pd.Series:
    """Ordered rules from the methods; NaN ranks never satisfy a condition."""
    r1, r2, r3, r4, r5 = (df[f"r_L{j}"] for j in range(1, 6))
    rank_mat = df[[f"r_L{j}" for j in range(1, 6)]]
    n_high = (rank_mat >= th).sum(axis=1)
    labels = pd.Series("unremarkable", index=df.index)

    rules = [
        ("stable_priority", n_high >= 4),
        ("hidden_burden", (r1 < tl) & ((r2 >= th) | (r3 >= th) | (r4 >= th) | (r5 >= th))),
        ("cash_sink", (r1 >= th) & (r2 < tl) & (r3 < tl)),
        ("labor_sink", (r2 >= th) & (r1 < th)),
        ("chronic_drain", (r3 >= th) & (r4 < tl)),
        ("shock_generator", (r4 >= th) & (r3 < tl)),
        ("budget_destabiliser", r5 >= th),
        ("representation_sensitive", df["meii"] > meii_q3),
    ]
    assigned = pd.Series(False, index=df.index)
    for name, cond in rules:
        m = cond.fillna(False) & ~assigned
        labels[m] = name
        assigned |= m
    return labels


def burden_weighted_shares(df: pd.DataFrame, label_col: str) -> dict:
    """Within-campus cost-share weights, campuses equally weighted."""
    shares = []
    for c, g in df.groupby("campus"):
        w = g["L1"] / g["L1"].sum()
        shares.append(w.groupby(g[label_col]).sum())
    pooled = pd.concat(shares, axis=1).fillna(0.0).mean(axis=1)
    return {k: round(float(v), 4) for k, v in pooled.items()}


def positions(g: pd.DataFrame, col: str) -> pd.Series:
    """1 = highest burden, n = lowest, from percentile ranks."""
    n = g[col].notna().sum()
    return (n - g[col] * n + 1).round().astype("Int64")


def make_examples(df: pd.DataFrame, n_per: int = 5) -> dict:
    out = {}
    for arch in ARCHETYPES:
        g = df[df["archetype"] == arch]
        if arch == "unremarkable" or g.empty:
            continue
        # most extreme examples: largest MEII within archetype, tie-break by n_wos
        g = g.sort_values(["meii", "n_wos"], ascending=False).head(n_per)
        ex = []
        for _, row in g.iterrows():
            camp = df[df["campus"] == row["campus"]]
            n = len(camp)
            pos = {f"L{j}": int(np.round(n - row[f"r_L{j}"] * n + 1))
                   if not np.isnan(row[f"r_L{j}"]) else None
                   for j in range(1, 6)}
            ex.append({
                "campus": f"U{int(row['campus']):02d}",
                "system": row["entity"],
                "system_desc": row.get("system_desc", ""),
                "n_entities_campus": n,
                "position_by_ledger": pos,
                "meii": round(float(row["meii"]), 3),
                "mean_rank": round(float(row["mean_rank"]), 3),
                "n_wos": int(row["n_wos"]),
            })
        out[arch] = ex
    return out


def summarize(df: pd.DataFrame, label_col: str = "archetype") -> dict:
    counts = df[label_col].value_counts()
    return {
        "counts": {k: int(counts.get(k, 0)) for k in ARCHETYPES},
        "shares": {k: round(float(counts.get(k, 0)) / len(df), 4) for k in ARCHETYPES},
        "burden_weighted_shares": burden_weighted_shares(df, label_col),
    }


def main() -> None:
    cfg = load_config()
    th, tl = cfg["theta_high"], cfg["theta_low"]

    summary = {}
    for level, led_file, meii_file, out_file in (
        ("system", "p2_ledgers_entity.csv", "p3_meii_entity.csv", "p4_archetypes.csv"),
        ("bldg", "p2_ledgers_bldg_entity.csv", None, "p4_archetypes_bldg.csv"),
    ):
        df = pd.read_csv(results_path(led_file))
        if meii_file:
            meii = pd.read_csv(results_path(meii_file))[["campus", "entity", "significant"]]
            df = df.merge(meii, on=["campus", "entity"], how="left")
        q3 = float(df["meii"].quantile(0.75))
        df["archetype"] = classify(df, th, tl, q3)
        for th_alt in (0.75, 0.85):
            df[f"archetype_th{int(th_alt*100)}"] = classify(df, th_alt, tl, q3)
        df.to_csv(results_path(out_file), index=False)

        s = summarize(df)
        for th_alt in (0.75, 0.85):
            alt = df[f"archetype_th{int(th_alt*100)}"]
            s[f"share_unchanged_vs_th{int(th_alt*100)}"] = round(
                float((alt == df["archetype"]).mean()), 4)
        if level == "system":
            xtab = pd.crosstab(df["entity"], df["archetype"])
            xtab.to_csv(results_path("p4_archetype_by_system.csv"))
            sysd = pd.read_csv(results_path("p2_ledgers_entity.csv"))
            desc_map = {}
            wo = pd.read_parquet(Path(__file__).resolve().parents[1] / "data/interim/wo_clean.parquet")
            desc_map = (
                wo[wo["system"].str.strip() != ""]
                .groupby("system")["system_desc"]
                .agg(lambda x: x.mode().iloc[0])
                .to_dict()
            )
            df["system_desc"] = df["entity"].map(desc_map)
            examples = make_examples(df)
            write_json(examples, results_path("p4_examples.json"))
            df.to_csv(results_path(out_file), index=False)  # rewrite with desc
        summary[level] = s

    write_json(summary, results_path("p4_archetype_summary.json"))
    print("Phase 4 complete.")
    for level in summary:
        print(f"[{level}]", summary[level]["counts"])


if __name__ == "__main__":
    main()
