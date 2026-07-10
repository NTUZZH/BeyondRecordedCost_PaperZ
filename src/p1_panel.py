"""Phase 1: preprocessing and panel construction.

Cleaning rules (3.1), entity definitions (3.2), comparability filter (3.3).

Inputs:  data/raw/FMUCD.csv, data/raw/cpi/*.csv, config.yaml
Outputs: data/interim/wo_clean.parquet          cleaned WO-level table
         data/panel/entity_system.parquet       campus x system entities + filter flags
         data/panel/entity_bldg_system.parquet  campus x building x system entities
         data/panel/panel_year.parquet          entity x year aggregates
         data/panel/panel_quarter.parquet       entity x quarter aggregates
         results/p1_panel_summary.json
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils.io import ROOT, load_config, load_raw, results_path, write_json


def clean_workorders(cfg: dict) -> tuple[pd.DataFrame, dict]:
    roster = cfg["campuses_retained"]
    date_col = {int(k): v for k, v in cfg["date_column_by_campus"].items()}
    windows = {int(k): v for k, v in cfg["campus_valid_window"].items()}

    df = load_raw()
    stats: dict = {"rows_raw_file": int(len(df))}

    df = df[df["UniversityID"].isin(roster)].copy()
    stats["rows_roster"] = int(len(df))

    # negative costs are credits/corrections, set to missing.
    neg = df["TotalCost"] < 0
    stats["rows_negative_cost_set_missing"] = int(neg.sum())
    df.loc[neg, "TotalCost"] = np.nan

    # drop exact duplicates on the analysis projection
    before = len(df)
    df = df[~df.duplicated()].copy()
    stats["rows_exact_duplicates_dropped"] = int(before - len(df))

    # event date: per-campus date column (most reliable field per campus)
    df["event_date"] = df["WOStartDate"]
    for uid, col in date_col.items():
        if col != "WOStartDate":
            m = df["UniversityID"] == uid
            df.loc[m, "event_date"] = df.loc[m, "WOEndDate"]
    stats["rows_missing_event_date"] = int(df["event_date"].isna().sum())
    df = df[df["event_date"].notna()].copy()
    df["year"] = df["event_date"].dt.year.astype("int16")
    df["quarter"] = (df["year"].astype(int) * 4 + (df["event_date"].dt.quarter - 1)).astype("int32")

    # campus valid window (core window intersected with campus coverage)
    before = len(df)
    keep = pd.Series(False, index=df.index)
    for uid, (y0, y1) in windows.items():
        keep |= (df["UniversityID"] == uid) & (df["year"] >= y0) & (df["year"] <= y1)
    df = df[keep].copy()
    stats["rows_outside_window_dropped"] = int(before - len(df))
    stats["rows_clean"] = int(len(df))

    # near-duplicate flag: same campus, building, system, date, positive cost
    key = ["UniversityID", "BuildingID", "SystemCode", "event_date", "TotalCost"]
    df["near_dup_flag"] = df.duplicated(subset=key, keep=False) & (df["TotalCost"] > 0)
    stats["rows_near_dup_flagged"] = int(df["near_dup_flag"].sum())

    # wo_type harmonization
    t = df["PPM/UPM"].astype("string").str.strip().str.upper()
    df["wo_type"] = np.where(t == "PPM", "PPM", np.where(t == "UPM", "UPM", "UNKNOWN"))
    df["wo_type"] = df["wo_type"].astype("category")
    stats["wo_type_counts"] = {str(k): int(v) for k, v in df["wo_type"].value_counts().items()}

    # CPI deflation to constant 2021 units, within country
    cpi_us = pd.read_csv(ROOT / "data/raw/cpi/us_cpiu_annual.csv").set_index("year")["cpi"]
    cpi_ca = pd.read_csv(ROOT / "data/raw/cpi/canada_cpi_annual.csv").set_index("year")["cpi"]
    base = cfg["cpi_base_year"]
    cad = df["UniversityID"].isin(cfg["campuses_cad"])
    factor = pd.Series(np.nan, index=df.index)
    factor[cad] = (cpi_ca[base] / df.loc[cad, "year"].map(cpi_ca)).astype(float)
    factor[~cad] = (cpi_us[base] / df.loc[~cad, "year"].map(cpi_us)).astype(float)
    df["cost2021"] = df["TotalCost"] * factor

    # extreme flag: above campus-level 99.9th cost percentile (kept in main)
    q = df.groupby("UniversityID", observed=True)["TotalCost"].transform(
        lambda s: s.quantile(cfg["extreme_pct"] / 100.0)
    )
    df["extreme_flag"] = (df["TotalCost"] > q) & df["TotalCost"].notna()
    stats["rows_extreme_flagged"] = int(df["extreme_flag"].sum())

    keep_cols = [
        "UniversityID", "BuildingID", "SystemCode", "SystemDescription",
        "wo_type", "event_date", "year", "quarter",
        "TotalCost", "cost2021", "LaborCost", "LaborHours",
        "near_dup_flag", "extreme_flag",
    ]
    out = df[keep_cols].rename(columns={
        "UniversityID": "campus", "BuildingID": "building",
        "SystemCode": "system", "SystemDescription": "system_desc",
        "TotalCost": "cost", "LaborCost": "labor_cost", "LaborHours": "labor_hours",
    })
    for c in ("building", "system", "system_desc"):
        out[c] = out[c].astype(str)
    return out.reset_index(drop=True), stats


def entity_table(df: pd.DataFrame, keys: list[str], cfg: dict, min_wo: int) -> pd.DataFrame:
    """Aggregate to entities and apply the comparability filter."""
    windows = {int(k): v for k, v in cfg["campus_valid_window"].items()}
    g = df.groupby(keys, observed=True)
    ent = g.agg(
        n_wos=("cost", "size"),
        n_years_active=("year", "nunique"),
        cost_nonmissing=("cost", lambda s: float(s.notna().mean())),
        labor_nonmissing=("labor_hours", lambda s: float(s.notna().mean())),
        cost_total=("cost", "sum"),
        cost2021_total=("cost2021", "sum"),
        labor_total=("labor_hours", "sum"),
    ).reset_index()
    # unclassified records (empty system/building codes) cannot form entities
    for k in keys[1:]:
        ent = ent[ent[k].astype(str).str.strip() != ""]
    labor_ok_campuses = set(cfg["campuses_labor_ok"])
    ent["comparable"] = (
        (ent["n_wos"] >= min_wo)
        & (ent["n_years_active"] >= cfg["min_active_years"])
        & (ent["cost_nonmissing"] >= cfg["cost_nonmissing_share"])
        & (
            ~ent["campus"].isin(labor_ok_campuses)
            | (ent["labor_nonmissing"] >= cfg["labor_nonmissing_share"])
        )
    )
    ent["window_years"] = ent["campus"].map(lambda u: windows[int(u)][1] - windows[int(u)][0] + 1)
    ent["window_quarters"] = ent["window_years"] * 4
    return ent


def main() -> None:
    cfg = load_config()
    df, stats = clean_workorders(cfg)

    (ROOT / "data/interim").mkdir(parents=True, exist_ok=True)
    (ROOT / "data/panel").mkdir(parents=True, exist_ok=True)
    df.to_parquet(ROOT / "data/interim/wo_clean.parquet", index=False)

    ent_sys = entity_table(df, ["campus", "system"], cfg, cfg["min_wo_entity_system"])
    ent_bs = entity_table(df, ["campus", "building", "system"], cfg, cfg["min_wo_entity_building_system"])
    ent_sys.to_parquet(ROOT / "data/panel/entity_system.parquet", index=False)
    ent_bs.to_parquet(ROOT / "data/panel/entity_bldg_system.parquet", index=False)

    # time panels for the primary entity level
    py = (
        df.groupby(["campus", "system", "year"], observed=True)
        .agg(n_wos=("cost", "size"), cost=("cost", "sum"), cost2021=("cost2021", "sum"),
             labor=("labor_hours", "sum"))
        .reset_index()
    )
    pq = (
        df.groupby(["campus", "system", "quarter"], observed=True)
        .agg(n_wos=("cost", "size"), cost=("cost", "sum"), labor=("labor_hours", "sum"))
        .reset_index()
    )
    py.to_parquet(ROOT / "data/panel/panel_year.parquet", index=False)
    pq.to_parquet(ROOT / "data/panel/panel_quarter.parquet", index=False)

    comp = ent_sys[ent_sys["comparable"]]
    total_cost = float(ent_sys["cost_total"].sum())
    stats.update(
        {
            "n_entities_system_all": int(len(ent_sys)),
            "n_entities_system_comparable": int(comp.shape[0]),
            "entities_per_campus_comparable": {
                int(k): int(v) for k, v in comp.groupby("campus").size().items()
            },
            "comparable_cost_share": round(float(comp["cost_total"].sum()) / total_cost, 4),
            "comparable_wo_share": round(float(comp["n_wos"].sum()) / float(ent_sys["n_wos"].sum()), 4),
            "comparable_labor_share": round(float(comp["labor_total"].sum()) / float(ent_sys["labor_total"].sum()), 4),
            "n_entities_bldg_system_all": int(len(ent_bs)),
            "n_entities_bldg_system_comparable": int(ent_bs["comparable"].sum()),
            "bldg_comparable_cost_share": round(
                float(ent_bs.loc[ent_bs["comparable"], "cost_total"].sum())
                / float(ent_bs["cost_total"].sum()), 4),
        }
    )
    write_json(stats, results_path("p1_panel_summary.json"))
    print("Phase 1 complete.")
    for k in ("rows_clean", "n_entities_system_all", "n_entities_system_comparable",
              "comparable_cost_share", "n_entities_bldg_system_comparable"):
        print(f"  {k}: {stats[k]}")
    print("  per campus comparable:", stats["entities_per_campus_comparable"])


if __name__ == "__main__":
    main()
