#!/usr/bin/env bash
# Regenerate every number, figure, and table of the CMERF study from raw data.
# Prerequisites: data/raw/FMUCD.csv (see data/raw/README.md for hash),
# data/raw/cpi/*.csv (committed), Python env with pandas/numpy/scipy/
# statsmodels/pyyaml/joblib/pyarrow/matplotlib. Deterministic: all seeds
# live in config.yaml.
set -euo pipefail
cd "$(dirname "$0")"

PY=${PY:-python}
# The bootstrap batches are fixed at 250 replicates, so the released caches
# reproduce for any JOBS value. The released numbers were produced with
# Python 3.11, numpy 1.26, scipy 1.17, and pandas 3.0.
JOBS=${JOBS:-4}

echo "== Phase 0: data screening =="
$PY src/p0_download_audit.py

echo "== Phase 1: panels =="
$PY src/p1_panel.py

echo "== Phase 2: ledgers =="
$PY src/p2_ledgers.py

echo "== Phase 3: bootstrap + stability (heaviest step) =="
$PY src/p3_stability.py --boot --jobs "$JOBS"
$PY src/p3_stability.py --analyze

echo "== Phase 3b: MEII decomposition + L5-variant robustness =="
$PY src/p3b_l5_analysis.py --jobs "$JOBS"

echo "== Phase 2b/3d: revised two-layer ledgers + stability =="
$PY src/p2b_revised.py
$PY src/p3d_revised.py --boot --B 1000 --jobs "$JOBS"
$PY src/p3d_revised.py --analyze

echo "== Phase 3g: independence reference =="
$PY src/p3g_independence.py

echo "== Phase 3e: simulation validation =="
$PY src/p3e_simulation.py --worlds 200 --B 200 --jobs "$JOBS"

echo "== Phase 5b: priority shortlisting =="
$PY src/p5b_shortlist.py

echo "== Phase 6b: missing-data robustness =="
$PY src/p6b_missing.py

echo "== Phase 10: campus statistics, hypothesis tests, streams, dynamics, trade-off, condition data =="
$PY src/p10a_describe.py
$PY src/p10e_tradeoff.py
$PY src/p10d_dynamics.py
$PY src/p10c_streams.py --B 500 --jobs "$JOBS"
$PY src/p10b_tests.py --splits 200
$PY src/p10g_h3_campus.py
$PY src/p10f_condition.py

echo "== Phase 4: archetypes =="
$PY src/p4_archetypes.py

echo "== Phase 5: decision case =="
$PY src/p5_decision_case.py

echo "== Phase 6: robustness suite =="
$PY src/p6_robustness.py --jobs "$JOBS"

echo "== Phase 7: figures =="
$PY src/p7_figures.py
$PY src/p7b_revised_figures.py

echo "== Tables (manuscript + supplemental) =="
$PY src/p8_tables.py
$PY src/p8b_revised_tables.py
$PY src/p8c_tables_r1.py

echo "run_all complete."
