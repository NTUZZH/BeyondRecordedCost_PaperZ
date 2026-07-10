#!/usr/bin/env bash
# Regenerate every number, figure, and table of Paper Z (CMERF) from raw data.
# Prerequisites: data/raw/FMUCD.csv (see data/raw/README.md for hash),
# data/raw/cpi/*.csv (committed), Python env with pandas/numpy/scipy/
# statsmodels/pyyaml/joblib/pyarrow/matplotlib. Deterministic: all seeds
# live in config.yaml.
set -euo pipefail
cd "$(dirname "$0")"

PY=${PY:-python}
JOBS=${JOBS:-8}

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

echo "== Phase 4: archetypes =="
$PY src/p4_archetypes.py

echo "== Phase 5: decision case =="
$PY src/p5_decision_case.py

echo "== Phase 6: robustness suite =="
$PY src/p6_robustness.py --jobs "$JOBS"

echo "== Phase 7: figures =="
$PY src/p7_figures.py

echo "== Tables (manuscript + appendix) =="
$PY src/p8_tables.py
$PY src/p9_supplementary.py

echo "run_all complete."
