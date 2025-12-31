# Vacuum-Class Toy Model (Reproductive Universes — Paper 2 Support)

This subpackage implements a toy ensemble model of **vacuum-class fragmentation dynamics** with a **two-regime branching rate** (field vs cluster) and a simple selection mechanism via "intelligence"-mediated lifetime and cluster-residence optimisation.

## How to run

From the repo root:

```bash
python -m repro_universe_toy_model.run_experiments
```

Quick/smoke mode (writes `_quick` variants):

```bash
python -m repro_universe_toy_model.run_experiments --mode quick
```

## Outputs

All outputs are written under `repro_universe_toy_model/results/`:

- `sweep_r/`
  - `sweep_r_summary.csv` (or `_quick`)
  - `sweep_r_seed_diagnostics.csv` (or `_quick`)
- `sweep_s/`
  - `sweep_s_summary.csv` (or `_quick`)
  - `sweep_s_seed_diagnostics.csv` (or `_quick`)
- `sweep_pfail/`
  - `sweep_pfail_summary.csv` (or `_quick`)
  - `sweep_pfail_seed_diagnostics.csv` (or `_quick`)
- `scatter_sample/`
  - `scatter_r2.0_s0.5_pfail0.5.csv` (or `_quick`)
- `diagnostics.log` (or `diagnostics_quick.log`)

## Resume behavior

Runs are **resumable**:

- Seed-level diagnostics are persisted per sweep.
- If interrupted, rerunning will **skip** parameter points that already have `n_seeds >= N_SEEDS` (or the quick-mode equivalent).
- If only some seeds are present, it will run the missing ones and then **recompute** and **upsert** the aggregated summary row.

All CSV writes are **atomic** (temp file → rename), so interrupted writes do not corrupt output files.

