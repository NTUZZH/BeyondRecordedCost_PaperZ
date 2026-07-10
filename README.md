# Beyond Recorded Cost: CMERF

Analysis code and machine-generated results for the paper *Beyond Recorded
Cost: A Multi-Ledger Representation-Robustness Framework for Maintenance
Economic Reasoning from CMMS Work Orders*.

CMERF (CMMS Multi-ledger Economic Representation Framework) re-represents
maintenance work-order records through five economic ledgers (recorded cost,
labor hours, chronicity, shock burden, budget volatility), converts each to
within-campus percentile ranks, tests whether the resulting priority rankings
agree, calibrates a per-entity Maintenance Economic Instability Index (MEII)
against a bootstrap noise floor, classifies entities into economic archetypes,
and quantifies the decision consequences of single-metric prioritization.

## Reproduce

```bash
python -m venv .venv && source .venv/bin/activate     # or use conda
pip install -r requirements.txt
# place the raw dataset first (see data/raw/README.md), then:
bash run_all.sh
```

`run_all.sh` runs the full pipeline end to end and regenerates every file
under `results/`, `tables/`, and the paper figures. All randomness is seeded
from `config.yaml` (single seed), so a matching environment reproduces the
results byte-for-byte. Environment knobs: `JOBS=<n>` sets the parallel worker
count, `PY=<python>` selects the interpreter.

## Data

The raw FMUCD dataset is not redistributed; download it from Mendeley Data
(doi:10.17632/cb8d2nsjss.1) and place `FMUCD.csv` at `data/raw/FMUCD.csv`
(see `data/raw/README.md` for the SHA-256 checksum). The CPI series used by
the secondary constant-currency analyses are included under `data/raw/cpi/`.

## Layout

| Path | Contents |
|---|---|
| `src/` | pipeline stages `p0`..`p9`, `p3b`, and `src/utils/` helpers |
| `config.yaml` | every analysis parameter and the random seed |
| `run_all.sh` | one-command reproduction |
| `results/` | machine-generated JSON/CSV, the single source of truth for every reported number |
| `tables/` | tables T1..T6 and appendix tables A1..A5 (CSV + LaTeX) |

## Pipeline stages

`p0` raw-data screening; `p1` cleaning and panel construction; `p2` ledgers
and within-campus ranks; `p3` stability, MEII, null models, regret; `p3b` MEII
decomposition and volatility-ledger robustness; `p4` archetypes; `p5` decision
case; `p6` robustness suite; `p7` figures; `p8`/`p9` tables. Utilities cover
the bootstrap machinery, ledger definitions, and the variant pipeline.

## License

Code is released under the MIT License (`LICENSE`). The FMUCD dataset is
distributed by its authors under its own license on Mendeley Data.
