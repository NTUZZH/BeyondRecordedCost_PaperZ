# Beyond Recorded Cost: CMERF

Analysis code and machine-generated results for the paper *Beyond Recorded
Cost: A Noise-Calibrated Multi-Criteria Framework for Priority Stability in
CMMS Work Orders*.

CMERF (CMMS Multi-criteria Economic Robustness Framework) reads one CMMS
work-order table on two layers: four alternative measures of realized
maintenance burden (recorded cost, labor hours, persistent work-order
volume, tail expenditure; coded L1--L4) and a distinct budget-risk
objective (common-shock-adjusted annual spending volatility; coded L5).
Each criterion becomes a within-campus percentile rank. The pipeline tests
portfolio-level agreement among the burden measures, calibrates each
entity's within-burden rank spread against bootstrap noise floors under
two resampling designs (validated by simulation), measures how far the
risk objective repositions each entity, classifies entities into four
decision profiles, and compares priority-shortlisting rules including an
exact minimax-regret shortlist under uncertain stakeholder weights.

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
the constant-currency deflation are included under `data/raw/cpi/`.

## Layout

| Path | Contents |
|---|---|
| `src/` | pipeline stages `p0`..`p9` plus revised stages (`p2b`, `p3d`, `p3e`, `p5b`, `p6b`, `p7b`, `p8b`) and `src/utils/` helpers |
| `config.yaml` | every analysis parameter and the random seed |
| `run_all.sh` | one-command reproduction |
| `results/` | machine-generated JSON/CSV, the single source of truth for every reported number |
| `tables/` | manuscript tables T1..T5 and supplementary tables A1, A6..A9 (LaTeX; legacy tables retained) |

## Pipeline stages

`p0` raw-data screening; `p1` cleaning and panel construction; `p2` legacy
five-ledger layer; `p2b` revised two-layer criteria (burden measures +
monetary budget risk); `p3`/`p3b` legacy stability and decomposition; `p3d`
revised two-layer stability with dual-design noise floors; `p3e` simulation
validation; `p4` legacy archetypes; `p5` legacy decision case; `p5b`
priority shortlisting (exact minimax regret, stakeholder weights);
`p6`/`p6b` robustness suites; `p7`/`p7b` figures; `p8`/`p8b`/`p9` tables.
Utilities cover the bootstrap machinery and criterion definitions.

## License

Code is released under the MIT License (`LICENSE`). The FMUCD dataset is
distributed by its authors under its own license on Mendeley Data.
