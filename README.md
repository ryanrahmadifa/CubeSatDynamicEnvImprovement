# Performance Prediction of CPS-PL in Dynamic Environments: Improvements on Sampling and Feature Engineering

An independent extension of the 29th ACM International Systems and Software Product Line Conference 2025 paper:

> **"Performance Prediction of Cyber-Physical Systems Product Lines in Dynamic Environments"**
> Marco Wijaya, Sami Lazreg, Tagir Fabarisov, Andreas Hein, Maxime Cordy
> , SnT, University of Luxembourg
> DOI: https://doi.org/10.1145/3744915

The original paper establishes baseline performance prediction results for a CubeSat product line using random sampling and standard regression models (regression tree, random forest, linear regression). 

This repository extends that work with active learning for more effective configuration sampling, **specifically in medium (350 configurations) sample size, achieving 0.9655 Spearman ρ** when using 8-orbit scores as training target for predicting 15-orbit configuration performance ranking.

## Active Learning (AL) Approach

The **Tree-Based Representativity** sampling (Bi et al., 2025, repository: https://github.com/bjhtud/Benchmark-AL-Mat.git `Benchmark-AL-Mat`) is used as the AL strategy. Each iteration, it:

1. Fits a Breiman regression tree on all labeled configurations so far;
2. Computes leaf node proportions across the combined labeled + unlabeled pool; and
3. Selects the next batch from underrepresented leaf regions, balancing informativeness (uncertain predictions) with representativity (coverage of the config space).

This is run iteratively in batches of 35, starting from 35 random seed configs (**no bias from AL**), until the simulation budget (3,500) is reached. The AL guidance signal is the **8-orbit mission score**, making the selection process cheaper than using full 15-orbit simulations.

## Subscores as Features

The original work states a composite score used for ranking the performance of configurations, instead of just using the ultimate score as ground truth we will also extract the subscores of energy subsystems (power), thermal control (thermal), OBC (on-board computing), communication (comm), and payload. The subscores will be used as features for the final regression-based prediction. 

**To ensure no leakage, only subscores and/or the score of the 4-orbit simulation may be used when predicting the 8-orbit score as the target.**

## Key findings (vs.  CPS-PL baseline, 8 Orbits Score, Random Forest Regression)

| Method | Features | Spearman ρ vs 15-orbit GT |
|---|---|---|
| Random sampling 350 configs (CPS-PL baseline) | config only | 0.8313 ± 0.0302 |
| AL sampling 350 configs (8-orbit score guided) | **config only** | **0.8588 ± 0.0386** |
| Random sampling 350 configs | config + 4-orbit telemetry (subscores + score) | 0.9519 ± 0.0125 |
| AL sampling 350 configs (8-orbit score guided) | config + 4-orbit subscores | 0.9621 ± 0.0075 |
| AL sampling 350 configs (8-orbit score guided) | config + 4-orbit scores | 0.9641 ± 0.0077 |
| AL sampling 350 configs (8-orbit score guided) | **config + 4-orbit telemetry (subscores + score)** | **0.9655 ± 0.0084** |

Two orthogonal improvements over the CPS-PL baseline:
1. **Active learning sampling**: selects more informative configurations, improving ρ with the same simulation budget
2. **Short-sim telemetry as features**: 4-orbit simulation outputs augment config parameters, capturing dynamic behavior that static features miss when predicting 8-orbit ranking

## Conclusion

The CPS-PL baseline achieves **ρ = 0.8313** using random sampling and config-only features. This work shows two improvements:

- **AL sampling alone** (8-orbit guided, config-only features): **ρ = 0.8588**, a modest but consistent gain from smarter configuration selection with the same simulation budget
- **4-orbit telemetry as features** (random sampling): **ρ = 0.9519**, short-duration telemetry captures dynamic behavior that static config parameters miss
- **AL + telemetry combined** (best): **ρ = 0.9655**, combining both improvements yields a +16.2% gain in ranking reliability over the CPS-PL baseline

CPS performance prediction difficulty stems not only from sample size and the sampling method, but from feature expressiveness where short-duration simulation telemetry may encode the physical dynamics that ultimately determine mission ranking.

## Key Files

- `outputs/` : generated outputs via running `gen_dataset.py` and `gen_dataset_al.py`, can be used to directly run prediction pipeline
- `DataBase.py` : generates `cubesat_productline.csv` — the full ~354,200 configuration product line
- `gen_dataset.py` : systematic sampling of full product line (original paper implementation, every 100th config)
- `gen_dataset_al.py` : active learning dataset generation (contribution, configurable budget by active learning sampling)
- `predictors.py` : train and evaluate regression models across sample sizes, targets, and feature sets (added args for datasets, enabling telemetry data )
- `compute_score.py` : mission score computation from simulation telemetry

## Quickstart

```bash
uv sync

# Generate product line data
uv run python DataBase.py

# Generate datasets
uv run python gen_dataset.py                  # full product line, every 100th config (CPS-PL baseline data)
uv run python gen_dataset_al.py               # AL-guided 3500 configs, 8-orbit guided (edit script for 15-orbit)

# CPS-PL baseline (random sampling, config features only)
uv run python predictors.py --model forest --dataset mid3 --target score_8orbits --features config_only --active_learning none

# AL + config only
uv run python predictors.py --model forest --dataset mid3 --target score_8orbits --features config_only --active_learning al_8orbits

# AL + full 4-orbit telemetry (best result)
uv run python predictors.py --model forest --dataset mid3 --target score_8orbits --features with_all --active_learning al_8orbits
```

---
Muhammad Ryanrahmadifa, 2026
