"""Phase 3e: simulation validation of the N1 noise-floor machinery.

Scenarios (synthetic campuses with known ground truth):
  S0  one latent burden rank; every ledger a noisy measurement of it; no
      real representation divergence  -> per-entity Type-I error of the
      MEII_b and GAP_r flags should be near the nominal 5%.
  S1  a subset of entities carries a genuinely divergent budget-risk
      dimension (inflated idiosyncratic annual variance) -> power and
      localization of the GAP_r flag.
  S2  a subset of entities diverges INSIDE the burden family (labor rate
      decoupled from cost) -> power of the MEII_b flag.
  S3  temporally dependent world: AR(1) annual multipliers plus campus-year
      common shocks and within-year WO clustering -> compares the
      work-order bootstrap against the year-block bootstrap; the block
      design should keep Type-I closer to nominal.

Output: results/p3e_simulation.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils.boot import RevisedCampusBootstrap, rank_matrix
from utils.io import load_config, results_path, write_json

E, Y, TQ = 12, 7, 28
TAU_PCT = 95.0


def synth_campus(rng, scenario: str):
    """Return (wo DataFrame, truth dict) for one synthetic campus.

    In the null world every burden ledger's ENTITY-LEVEL expectation is a
    monotone function of one latent burden theta (work-order volume, per-WO
    cost scale, labor rate, and hence tail exposure all driven by theta), so
    observed cross-ledger rank divergence arises from finite sampling only.
    """
    theta = rng.uniform(0, 1, E)                    # latent burden
    # volume driven by the same latent construct (log-linear in theta)
    n_i = np.round(np.exp(np.log(100) + theta * np.log(20.0))).astype(int)
    risk_div = np.zeros(E, bool)
    fam_div = np.zeros(E, bool)
    if scenario == "S1":
        risk_div[rng.choice(E, 3, replace=False)] = True
    if scenario == "S2":
        fam_div[rng.choice(E, 3, replace=False)] = True

    rows = []
    ar = 0.6 if scenario == "S3" else 0.0
    common = np.zeros(Y)
    if scenario == "S3":
        common = rng.normal(0, 0.35, Y)             # campus-year shocks
    for i in range(E):
        # annual activity multipliers
        eps = rng.normal(0, 0.25, Y)
        m = np.zeros(Y)
        for t in range(Y):
            m[t] = ar * (m[t - 1] if t else 0) + eps[t]
        if risk_div[i]:
            m += rng.normal(0, 1.2, Y)              # divergent risk dimension
        year_w = np.exp(m + common)
        year_w /= year_w.sum()
        yrs = rng.choice(Y, size=n_i[i], p=year_w)
        if scenario == "S3":                        # within-year clustering
            q_of_year = rng.integers(0, 4, Y)
            qs = (q_of_year[yrs] + (rng.random(n_i[i]) < 0.25)
                  * rng.integers(0, 4, n_i[i])) % 4
        else:
            qs = rng.integers(0, 4, n_i[i])
        # per-WO cost: lognormal, scale tied to the same theta
        mu = 3.0 + 1.2 * theta[i]
        cost = rng.lognormal(mu, 1.1, n_i[i])
        # labor hours: identical rate for every entity in the null world
        # (per-WO noise averages out); S2 decouples a subset
        lab_rate = 0.08 * (3.0 if fam_div[i] else 1.0)
        labor = cost * lab_rate * rng.lognormal(0, 0.35, n_i[i])
        rows.append(pd.DataFrame({
            "entity": f"E{i:02d}", "year": 2014 + yrs,
            "quarter": (2014 + yrs) * 4 + qs,
            "cost2021": cost, "labor": labor}))
    wo = pd.concat(rows, ignore_index=True)
    return wo, {"theta": theta, "risk_div": risk_div, "fam_div": fam_div}


def one_world(seed: int, scenario: str, B: int):
    rng = np.random.default_rng(seed)
    wo, truth = synth_campus(rng, scenario)
    res = {}
    for mode in ("wo", "year_block"):
        bt = RevisedCampusBootstrap(1, wo, sorted(wo["entity"].unique()),
                                    TQ, (2014, 2014 + Y - 1), TAU_PCT,
                                    mode=mode)
        obs = bt.observed_scores()
        obs_r = rank_matrix(obs)
        ranks = bt.run_batch((seed, 1, 0, hash(mode) % 97), B)
        dev = ranks - obs_r[None, :, :]
        # MEII_b flag
        base = np.nanmean(obs_r[:, :4], axis=1)
        rec = np.clip(base[None, :, None] + dev[:, :, :4], 1e-6, 1.0)
        floor = np.nanpercentile(np.nanstd(rec, axis=2, ddof=1), 95, axis=0)
        obs_sd = np.nanstd(obs_r[:, :4], axis=1, ddof=1)
        sig_b = obs_sd > floor
        # GAP_r flag
        med_b = np.nanmedian(obs_r[:, :4], axis=1)
        nb = np.clip(med_b[None, :, None] + dev[:, :, :4], 1e-6, 1.0)
        nm = np.nanmedian(nb, axis=2)
        nr = np.clip(obs_r[:, 4][None, :] + dev[:, :, 4], 1e-6, 1.0)
        gfloor = np.nanpercentile(np.abs(nr - nm), 95, axis=0)
        gobs = np.abs(obs_r[:, 4] - med_b)
        sig_g = gobs > gfloor
        # localization: are the truly risk-divergent entities the top-3 by
        # observed GAP? (recall of the descriptive ranking, no test)
        top3 = np.zeros(E, bool)
        top3[np.argsort(-gobs)[:3]] = True
        res[mode] = {"sig_b": sig_b, "sig_g": sig_g, "gap_top3": top3}
    return truth, res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worlds", type=int, default=200)
    ap.add_argument("--B", type=int, default=200)
    ap.add_argument("--jobs", type=int, default=4)
    args = ap.parse_args()
    cfg = load_config()
    out = {"worlds": args.worlds, "B": args.B}

    for scen in ("S0", "S1", "S2", "S3"):
        nw = args.worlds if scen in ("S0", "S3") else max(args.worlds // 2, 50)
        results = Parallel(n_jobs=args.jobs)(
            delayed(one_world)(cfg["seed"] * 1000 + scen_id * 100000 + w,
                               scen, args.B)
            for scen_id, w in [(int(scen[1]), w) for w in range(nw)])
        agg = {}
        for mode in ("wo", "year_block"):
            fb = np.concatenate([r[1][mode]["sig_b"] for r in results])
            fg = np.concatenate([r[1][mode]["sig_g"] for r in results])
            risk = np.concatenate([r[0]["risk_div"] for r in results])
            fam = np.concatenate([r[0]["fam_div"] for r in results])
            null_mask = ~(risk | fam)
            agg[mode] = {
                "type1_meii_b": round(float(fb[null_mask].mean()), 4),
                "type1_gap": round(float(fg[null_mask].mean()), 4),
            }
            if scen == "S1":
                agg[mode]["power_gap_on_divergent"] = round(
                    float(fg[risk].mean()), 4)
                top3 = np.concatenate([r[1][mode]["gap_top3"] for r in results])
                agg[mode]["gap_top3_recall_divergent"] = round(
                    float(top3[risk].mean()), 4)
                agg[mode]["false_meii_b_on_divergent"] = round(
                    float(fb[risk].mean()), 4)
            if scen == "S2":
                agg[mode]["power_meii_b_on_divergent"] = round(
                    float(fb[fam].mean()), 4)
        out[scen] = agg
        print(scen, agg, flush=True)

    write_json(out, results_path("p3e_simulation.json"))
    print("p3e simulation written")


if __name__ == "__main__":
    main()
