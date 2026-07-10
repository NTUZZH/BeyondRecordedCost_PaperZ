"""Phase 0: data-quality screening of the raw FMUCD table.

Produces:
  results/p0_schema_inventory.csv     column inventory with dtypes
  results/p0_coverage_matrix.csv      per-campus, per-year record counts
  results/p0_missingness.csv          per-campus field missingness
  results/p0_cost_diagnostics.csv     cost distribution diagnostics
  results/p0_zero_cost_crosstab.csv   zero-cost incidence by campus and type
  results/p0_duplicates.json          duplicate-record analysis
  results/p0_system_taxonomy.csv      system-code taxonomy
  results/p0_audit.json               summary statistics and screening evidence
  results/p0_cost_hist.png            cost histogram (log scale)
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils.io import ROOT, RAW_CSV, load_config, load_raw, results_path, write_json


def schema_inventory() -> pd.DataFrame:
    """Item 1: every column, dtype, missing share, example values (100k-row sample)."""
    sample = pd.read_csv(RAW_CSV, nrows=100_000)
    rows = []
    for col in sample.columns:
        s = sample[col]
        examples = s.dropna().unique()[:3]
        rows.append(
            {
                "column": col,
                "pandas_dtype": str(s.dtype),
                "missing_share_sample": round(float(s.isna().mean()), 4),
                "n_unique_sample": int(s.nunique()),
                "examples": "; ".join(str(x)[:60] for x in examples),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    cfg = load_config()
    audit: dict = {"n_rows_file": None}

    inv = schema_inventory()
    inv.to_csv(results_path("p0_schema_inventory.csv"), index=False)

    df = load_raw()
    audit["n_rows_file"] = int(len(df))
    df["year"] = df["WOStartDate"].dt.year

    # --- Item 2: coverage matrix (campus x year WO counts) ---
    cov = (
        df.groupby(["UniversityID", "year"], observed=True)
        .size()
        .unstack(fill_value=0)
        .sort_index()
    )
    cov.to_csv(results_path("p0_coverage_matrix.csv"))
    substantial = cov >= 200
    windows = {}
    for uid in cov.index:
        yrs = [int(y) for y in cov.columns[substantial.loc[uid]] if not np.isnan(y)]
        windows[int(uid)] = {
            "first_substantial_year": min(yrs) if yrs else None,
            "last_substantial_year": max(yrs) if yrs else None,
            "n_substantial_years": len(yrs),
            "total_wos": int(cov.loc[uid].sum()),
        }
    audit["campus_windows"] = windows

    # --- Item 3: missingness per campus ---
    miss_rows = []
    for uid, g in df.groupby("UniversityID", observed=True):
        n = len(g)
        miss_rows.append(
            {
                "UniversityID": int(uid),
                "n_wos": n,
                "cost_missing": round(float(g["TotalCost"].isna().mean()), 4),
                "cost_zero": round(float((g["TotalCost"] == 0).mean()), 4),
                "cost_negative": round(float((g["TotalCost"] < 0).mean()), 4),
                "labor_missing": round(float(g["LaborHours"].isna().mean()), 4),
                "labor_zero": round(float((g["LaborHours"] == 0).mean()), 4),
                "labor_negative": round(float((g["LaborHours"] < 0).mean()), 4),
                "laborcost_missing": round(float(g["LaborCost"].isna().mean()), 4),
                "laborcost_zero": round(float((g["LaborCost"] == 0).mean()), 4),
                "startdate_missing": round(float(g["WOStartDate"].isna().mean()), 4),
                "enddate_missing": round(float(g["WOEndDate"].isna().mean()), 4),
                "system_missing": round(float(g["SystemCode"].isna().mean()), 4),
                "ppm_upm_missing": round(float(g["PPM/UPM"].isna().mean()), 4),
                "building_missing": round(float(g["BuildingID"].isna().mean()), 4),
            }
        )
    miss = pd.DataFrame(miss_rows)
    miss.to_csv(results_path("p0_missingness.csv"), index=False)

    # --- Item 4: cost distribution diagnostics per campus ---
    diag_rows = []
    for uid, g in df.groupby("UniversityID", observed=True):
        x = g["TotalCost"].dropna()
        xp = x[x > 0]
        diag_rows.append(
            {
                "UniversityID": int(uid),
                "n_cost_obs": int(len(x)),
                "min": float(x.min()),
                "p50": float(x.quantile(0.50)),
                "p90": float(x.quantile(0.90)),
                "p95": float(x.quantile(0.95)),
                "p99": float(x.quantile(0.99)),
                "p999": float(x.quantile(0.999)),
                "max": float(x.max()),
                "share_gt_10k": round(float((x > 10_000).mean()), 5),
                "share_gt_100k": round(float((x > 100_000).mean()), 6),
                "n_gt_100k": int((x > 100_000).sum()),
                "share_positive": round(float((x > 0).mean()), 4),
                "mean_positive": float(xp.mean()) if len(xp) else np.nan,
            }
        )
    diag = pd.DataFrame(diag_rows)
    diag.to_csv(results_path("p0_cost_diagnostics.csv"), index=False)

    # histogram of positive costs, log x-scale, per campus overlay
    fig, ax = plt.subplots(figsize=(9, 5))
    pos = df.loc[df["TotalCost"] > 0, ["UniversityID", "TotalCost"]]
    bins = np.logspace(np.log10(max(pos["TotalCost"].min(), 0.01)), np.log10(pos["TotalCost"].max()), 80)
    for uid, g in pos.groupby("UniversityID", observed=True):
        ax.hist(g["TotalCost"], bins=bins, histtype="step", label=f"U{int(uid):02d}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("TotalCost per work order (nominal local currency)")
    ax.set_ylabel("Work orders")
    ax.legend(ncol=4, fontsize=7)
    fig.tight_layout()
    fig.savefig(results_path("p0_cost_hist.png"), dpi=150)
    plt.close(fig)

    # --- Item 5: zero-cost analysis ---
    df["cost_zero"] = df["TotalCost"] == 0
    df["labor_pos"] = df["LaborHours"] > 0
    ct1 = pd.crosstab(df["cost_zero"], df["PPM/UPM"], dropna=False)
    ct2 = pd.crosstab(df["cost_zero"], df["labor_pos"], dropna=False)
    zc = df[df["cost_zero"]]
    zero_detail = {
        "n_zero_cost": int(len(zc)),
        "zero_cost_share": round(float(df["cost_zero"].mean()), 4),
        "zero_cost_with_positive_labor_hours": round(float(zc["labor_pos"].mean()), 4) if len(zc) else None,
        "zero_cost_ppm_share": round(float((zc["PPM/UPM"] == "PPM").mean()), 4) if len(zc) else None,
    }
    ct = pd.concat({"vs_ppm_upm": ct1, "vs_labor_pos": ct2}, axis=1)
    ct.to_csv(results_path("p0_zero_cost_crosstab.csv"))
    audit["zero_cost"] = zero_detail

    # component identity check: TotalCost vs sum of parts
    parts = df[["LaborCost", "MaterialCost", "OtherCost"]].sum(axis=1)
    ok = np.isclose(parts, df["TotalCost"], rtol=1e-4, atol=0.02)
    audit["totalcost_equals_parts_share"] = round(float(ok.mean()), 5)

    # --- Item 6: duplicates / bundling scan ---
    dup_exact = int(df.duplicated(subset=[c for c in df.columns if c not in ("cost_zero", "labor_pos")]).sum())
    df["date_only"] = df["WOStartDate"].dt.date
    near_keys = ["UniversityID", "BuildingID", "SystemCode", "date_only", "TotalCost"]
    near = df[df["TotalCost"] > 0].duplicated(subset=near_keys).sum()
    woid_dup = int(df.duplicated(subset=["UniversityID", "WOID"]).sum())
    audit["duplicates"] = {
        "exact_duplicate_rows": dup_exact,
        "near_duplicates_same_bldg_system_date_cost_positive": int(near),
        "duplicate_woid_within_campus": woid_dup,
        "note": "near-duplicates are flagged, not dropped, in Phase 1",
    }
    write_json(audit["duplicates"], results_path("p0_duplicates.json"))

    # --- Item 7: currency / country determination ---
    country = (
        df.groupby("UniversityID", observed=True)["Country"]
        .agg(lambda s: s.mode().iloc[0] if len(s.mode()) else None)
        .to_dict()
    )
    audit["campus_country"] = {int(k): str(v) for k, v in country.items()}

    # --- Item 8: system taxonomy check ---
    tax = (
        df.groupby(["SystemCode", "SystemDescription"], observed=True)
        .size()
        .reset_index(name="n_wos")
        .sort_values("n_wos", ascending=False)
    )
    tax.to_csv(results_path("p0_system_taxonomy.csv"), index=False)
    sys_per_campus = df.groupby("UniversityID", observed=True)["SystemCode"].nunique()
    audit["system_taxonomy"] = {
        "n_distinct_system_codes": int(tax.shape[0]),
        "systems_per_campus": {int(k): int(v) for k, v in sys_per_campus.items()},
    }

    # end-start lag (justifies assigning years by end date where needed)
    both = df.dropna(subset=["WOStartDate", "WOEndDate"])
    lag = (both["WOEndDate"] - both["WOStartDate"]).dt.days
    audit["end_start_lag_days"] = {
        "n": int(len(lag)),
        "median": float(lag.median()),
        "mean": round(float(lag.mean()), 1),
    }

    # --- Campus screening evidence ---
    labor_bad = (miss["labor_missing"] + miss["labor_zero"]) > cfg["labor_campus_missing_cutoff"]
    audit["labor_missing_or_zero_share"] = {
        int(r.UniversityID): round(float(r.labor_missing + r.labor_zero), 4) for r in miss.itertuples()
    }
    audit["campuses_labor_ok"] = [int(u) for u in miss.loc[~labor_bad, "UniversityID"]]
    cost_bad = (miss["cost_missing"] + miss["cost_zero"]) > 0.50
    audit["cost_missing_or_zero_share"] = {
        int(r.UniversityID): round(float(r.cost_missing + r.cost_zero), 4) for r in miss.itertuples()
    }
    audit["campuses_cost_ok"] = [int(u) for u in miss.loc[~cost_bad, "UniversityID"]]
    audit["ppm_upm_counts"] = {str(k): int(v) for k, v in df["PPM/UPM"].value_counts(dropna=False).items()}
    audit["year_range_all"] = [int(df["year"].min()), int(df["year"].max())]

    write_json(audit, results_path("p0_audit.json"))
    print("Phase 0 screening complete.")
    print(f"rows={audit['n_rows_file']}, campuses={sorted(audit['campus_country'])}")
    print(f"labor-ok campuses: {audit['campuses_labor_ok']}")
    print(f"cost-ok campuses: {audit['campuses_cost_ok']}")


if __name__ == "__main__":
    main()
