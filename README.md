# CMERF: CMMS Multi-criteria Economic Robustness Framework

Analysis code and machine-generated results for the paper *Which Assets
Deserve Attention First? Metric Choice and Budget Risk in Maintenance
Prioritization from Infrastructure Work-Order Records*.

CMERF (CMMS Multi-criteria Economic Robustness Framework) reads one CMMS
work-order table on two layers: four alternative measures of realized
maintenance burden (recorded cost, labor hours, persistent work-order
volume, tail expenditure; coded L1--L4) and a budget-risk criterion (the
standard deviation of a system's annual cost around its own mean scaled by
the campus spending index of the other systems; coded L5).
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
| `src/` | pipeline stages `p0`..`p8`, the two-layer stages (`p2b`, `p3d`, `p3e`, `p3g`, `p5b`, `p6b`, `p7b`, `p8b`, `p8c`), the `p10` analyses, and `src/utils/` helpers |
| `config.yaml` | every analysis parameter and the random seed |
| `run_all.sh` | one-command reproduction |
| `results/` | machine-generated JSON/CSV, the single source of truth for every reported number |
| `tables/` | manuscript tables T1..T7 (LaTeX + CSV) and the tables of the Supplemental Materials (`A*` files) |
| `assets/fonts/` | TrueType build of TeX Gyre Termes (GUST Font License), so figures render in the same Times face as the paper; rebuild with `python src/utils/make_fonts.py` |

## Pipeline stages

`p0` raw-data screening; `p1` cleaning and panel construction; `p2` legacy
five-ledger layer; `p2b` revised two-layer criteria (burden measures +
monetary budget risk); `p3`/`p3b` legacy stability and decomposition; `p3d`
revised two-layer stability with dual-design noise floors; `p3e` simulation
validation; `p4` legacy archetypes; `p5` legacy decision case; `p5b`
priority shortlisting (exact minimax regret, stakeholder weights);
`p6`/`p6b` robustness suites; `p3g` independence reference; `p10a`
campus and criterion statistics; `p10b` tests of the four hypotheses,
including the resampled and split-half tests of the shortlist rule; `p10g`
the test of H3 with the campus as the unit; `p10c`
the framework on preventive and unplanned work orders separately; `p10d`
year-to-year persistence and the forward check of the shortlists; `p10e`
the burden versus budget-risk trade-off curve and the contested systems;
`p10f` record-derived burden against building condition data; `p7`/`p7b`
figures; `p8`/`p8b`/`p8c` tables. Utilities cover the bootstrap machinery
and criterion definitions.

## Releases

- **v1.1** (October 2026). Budget risk scales each system's expected cost
  by the campus spending index of the other systems (proportional form),
  and the additive form of v1.0 is kept as a sensitivity variant. The
  minimax-regret shortlist breaks ties among equally good lists in a stated
  order. The release adds the `p3g` and `p10` analyses and the tables
  T1..T7 and A10..A16. The Supplemental Materials PDF is attached to this
  release.
- **v1.0** (August 2026). Release cited by the first submission.

## License

Code is released under the MIT License (`LICENSE`). The FMUCD dataset is
distributed by its authors under its own license on Mendeley Data.
