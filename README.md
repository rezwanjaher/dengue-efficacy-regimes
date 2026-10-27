# Efficacy Regimes in Vector-Borne Epidemic Control

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23683522.svg)](https://doi.org/10.5281/zenodo.23683522)

Code, data, trained agents and results for the paper

> Md Rezwan Jaher, Md Sumon Hossain and Md Ahsanul Haque.
> *Efficacy Regimes in Vector-Borne Epidemic Control: Timing, Targeting, and the Price of Information.* 2026.

The model is a ward-level vector-host simulation of dengue in Dhaka North City
Corporation (54 wards), with a fixed seasonal budget split across fogging,
larviciding, case management and surveillance. The paper sweeps the field
efficacy of fogging and compares nine policies, including a reinforcement-learning
agent, under matched budgets.

## Quick start: rebuild the tables

```
pip install -r requirements.txt
python reproduce.py
```

This rebuilds the paper's tables and figures from the shipped results in a few
minutes, without a GPU, and checks each table against the copy the paper uses.
Output goes to `reproduced/`; nothing shipped is changed. It ends with a line
such as `12 of 12 tables match the paper's copies`.

Tested with Python 3.11.

## What is here

| Folder | Contents |
|---|---|
| `src/env/` | The simulator: dynamics (`physics.py`) and the environment (`smip_env.py`) |
| `src/common/policy.py` | The agent's network |
| `src/train.py` | Training (proximal policy optimisation, three-stage curriculum) |
| `src/evaluate_dengue.py` | The heuristic policies |
| `src/run_all_evals.py` | The evaluation harness: every policy, budget and seed |
| `src/` (other) | Policy-free experiments, calibration, surveillance sweeps, and the table and figure scripts |
| `configs/archetypes.json` | Every parameter, with its source or derivation |
| `data/` | The ten ward graphs and the 2022 census populations |
| `rl_models/` | The eight trained agents in the paper (54 and 10 wards, four seeds each) |
| `evaluation_dengue/` | The results every table and figure is built from |
| `report/` | The tables and figures exactly as the paper uses them |

`RELEASE_MANIFEST.csv` lists every file with its MD5. Comments were removed from
the scripts for release; the column `code_tree_verified` records that each
script's code is identical to the version that produced the results.

## Which script makes which table

| Table or figure | Script | Reads |
|---|---|---|
| Timing, delay, budget and protocol tables and figures | `make_report_panels.py` (via `make_figures.py`) | `evaluation_dengue/v18_*`, `exp_delay_*`, `learning_curve.csv` |
| Crossings figure, information figure | `make_result_figures_panels.py` (via `make_figures.py`) | `exp_efficacy_*_full.csv`, `capture_curve.csv`, `survey_sweep.csv`, `voi_curve.csv` |
| Crossings by instance | `make_crossings_table.py` | `exp_efficacy_*_full.csv`, `data/*_graph.json` |
| Ward listing | `make_ward_table.py` | `data/`, `t/ward_thana_crosswalk_bbs2022.csv` |
| Surveillance sensitivity | `make_surveillance_table.py` | `evaluation_dengue/sweeps_ls08/` |
| Schedules and best start week | `exp_schedules.py` | runs the simulation |
| Combined appendix tables | `compact_tables.py` | the tables above |

## Rerunning the simulations

Run every command from the repository root. Each script refuses to replace an
existing output, so write to a new name or delete the old file first.

### 1. Policy evaluations (uses the trained agents)

Every policy, budget multiplier and seed on both instances, about one to two
minutes per run on a laptop. Each CSV in `evaluation_dengue/` has a
`.meta.json` beside it with the exact command that wrote it.

```
python src/run_all_evals.py --all --version_tag V18 --lambda_sigma 0.8 --gamma 0.99 --free-heuristics --survey-share 0.25 --csv evaluation_dengue/v18_free_cap25.csv
```

This is the matched protocol behind the headline tables. The protocol matrix
also needs the three other cells: drop `--survey-share 0.25` (`v18_free`),
drop `--free-heuristics` (`v18_paced_cap25`), or drop both (`v18_paced`).

For the efficacy sweep, add `--eta-F` with each of 0.10, 0.30, 0.40, 0.45,
0.50, 0.54, 0.56, 0.58, 0.62, 0.75 and 0.90 (`v18_eta*.csv`). For the
surveillance cap at efficacy 0.45, use `--eta-F 0.45` with `--survey-share`
set to 0.10, 0.40, 0.60 or 1.0 (`v18_eta045_free_cap*.csv`).

Always pass `--lambda_sigma 0.8` and `--version_tag V18`; the defaults differ.

### 2. Policy-free experiments

Coverage is written straight into the dynamics; no agent is used.

```
python src/coverage_threshold.py --all --control 8000 --csv evaluation_dengue/v18_coverage_threshold.csv
python src/coverage_threshold.py --all --timing --burst_len 6 --control 8000 --csv evaluation_dengue/v18_threshold_timing.csv
python src/coverage_threshold.py --all --horizons 20,30,40,52 --control 8000 --csv evaluation_dengue/v18_threshold_horizon.csv

python src/exp_all.py delay    --dncc
python src/exp_all.py delay    --dncc --seasonal
python src/exp_all.py prereg   --dncc
python src/exp_all.py channels --dncc
```

Repeat the `exp_all.py` lines with `--dncc10` for the 10-ward instance.

The efficacy sweep on all ten instances, behind the crossings table:

```
python src/exp_multi.py efficacy --all-instances --eta-grid "0.10,0.20,0.30,0.40,0.42,0.44,0.46,0.48,0.50,0.52,0.54,0.56,0.58,0.60,0.62,0.75,0.90"
```

`dncc10` is the same graph as `contiguous10` and gives identical results, so
the table lists ten instances.

Schedules at a fixed budget, and the calibration of K:

```
python src/exp_schedules.py --all-instances --tex-dir report/tables
python src/calibrate_k.py --graph data/dncc_graph.json --econ DNCC_Seasonal_Budget --k-grid 3.5,4.0,4.5,5.0,5.5,6.0 --episodes 30 --out evaluation_dengue/calibration_k_full.csv
```

### 3. Surveillance sensitivity (uses the trained agents)

Each value runs the evaluation harness on a copy of the configuration. Logs go
to `evaluation_dengue/sweeps_ls08/`.

```
python src/sweep_surveillance.py --param eta_S
python src/sweep_surveillance.py --param eta_S     --eta-f 0.40
python src/sweep_surveillance.py --param eta_S     --eta-f 0.45
python src/sweep_surveillance.py --param eta_S     --eta-f 0.54
python src/sweep_surveillance.py --param surv_cost --eta-f 0.40
python src/sweep_surveillance.py --param surv_cost --eta-f 0.45
python src/sweep_surveillance.py --param surv_cost --eta-f 0.54
python src/sweep_surveillance.py --param eta_C     --eta-f 0.40
python src/sweep_surveillance.py --param eta_C     --eta-f 0.45
python src/sweep_surveillance.py --param eta_C     --eta-f 0.54
```

### 4. Summaries of the runs above

These read the results files and run no simulation:

```
python src/voi_curve.py --out evaluation_dengue/voi_curve.csv
python src/capture_curve.py --out evaluation_dengue/capture_curve.csv
python src/survey_sweep.py --eta 0.45 --out evaluation_dengue/survey_sweep.csv
```

Then `python reproduce.py` rebuilds the tables and figures.

`evaluation_dengue/learning_curve.csv` holds the agents' evaluation scores at
checkpoints saved during training. Those checkpoints are not shipped; only the
final agents are.

## Retraining the agents

One seed takes about 4 hours on an NVIDIA T4 GPU (54 wards; slightly less for
10 wards). The eight published agents took about 32 GPU-hours. They were
trained on Google Colab.

```
python src/train.py --seed 0 --graph data/dncc_graph.json --archetype Dengue_Dhaka_Advanced --econ DNCC_Seasonal_Budget --lambda_sigma 0.8 --rho 0.2 --regime estimated --history 5 --horizon 20 --steps 4000000 --reward_mode weighted
```

The published agents use seeds 0 to 3. For the 10-ward instance use
`--graph data/dncc10_graph.json --econ DNCC10_Seasonal_Budget`. Training runs
the pre-training checks in `src/preflight_epidemic.py` first and stops if any
fails.

New runs are saved under `rl_models/`, or under
`/content/drive/MyDrive/DRL_POMDP_DENGUE/rl_models/` when run on Colab. Their
folder names carry the tag `V19`, so they never mix with the published `V18`
agents in `rl_models/test/`. A run whose folder already exists resumes from
its latest checkpoint. To evaluate new agents, pass
`--models_root rl_models --version_tag V19` to `run_all_evals.py`.

## Robustness checks

The appendix reports further one-off checks (uneven breeding capacity, isolated
wards, joint parameter draws, alternative spatial mixing, numerical scheme,
prior immunity, seeding) and the correlation of each policy's
allocation with true ward infection. Their scripts are available from the corresponding
author on request.

## Licence

Code: MIT (`LICENSE`). Data, results and trained weights: CC BY 4.0
(`LICENSE-data`). Please cite the paper (`CITATION.cff`). The archived release
of this repository is at https://doi.org/10.5281/zenodo.23683522.

## Contact

Md Rezwan Jaher, rezwan.jaher.262@northsouth.edu,
ORCID [0009-0002-1692-5967](https://orcid.org/0009-0002-1692-5967).
