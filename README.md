# AG-VS source and experiments

This repository implements AG-VS-BO and AG-VS-TuRBO, comparison algorithms,
synthetic benchmarks, and MOPTA08. MOPTA08 is the only real-life experiment
in the released registry. Existing results of removed tasks are preserved locally.

## Installation

The reference environment is Windows x64, CPython 3.13.7. From the repository root:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
```

Use the virtual environment's `python` for all commands below. `requirements.txt`
pins all dependencies in one file, including optional algorithms (CMA-ES),
test tools (pytest), and transitive dependencies. JAX/NumPyro are needed
by the shared MCTS-VS import path, including its BO/TuRBO variants.
The development PyTorch build is `2.11.0+cu130`; the
portable requirement is `2.11.0`. CPU experiments do not require CUDA.
Record resolved versions with `python -m pip freeze > environment.txt`.

Third-party reference implementations retain their licences and provenance in
`Experiment/benchmark/baseline/UPSTREAM_SOURCES.md` and `upstream_sources.json`.
Their historical requirements are not the requirements of the maintained adapters.

## Inspect plans without running trials

```powershell
python -m Experiment.benchmark.mopta --list
python -m Experiment.benchmark.runner_synthetic --list
python -m Experiment.grid_search_nbad --list
python -m Experiment.grid_search_tseq --list
python -m Experiment.version_comparison --list
```

Direct script execution also works: `python Experiment/benchmark/mopta.py --list`.
All budgets count scalar objective evaluations, including initial sampling.
Trial `i` uses optimizer and benchmark seed `2021 + i`; the default 25 trials
use IDs 0–24 and seeds 2021–2045. Compatible complete traces are reused. Worker
count is not a scientific cache key. Changed scientific settings can require
reruns; `--rerun-complete` forces them. Partial JSON traces are not a universal
checkpoint of the optimizer state.

## Run MOPTA08

```powershell
python -m Experiment.benchmark.mopta
```

This starts 25 trials per algorithm, with 150 evaluations and one worker.
The algorithm order is MCTS-VS-TuRBO, MCTS-VS-BO, AG-VS-TuRBO, TuRBO,
AG-VS-BO, GTBO, HeSBO, ALEBO, LA-MCTS-TuRBO, LA-MCTS-BO.
AG-VS uses `d_eff=4`, `N_split=5`, `N_bad=N_split=5`, `t_seq=2`, and
selection batch size `B=3`. This batch control is separate from backend TuRBO
settings. HeSBO and ALEBO use embedding dimension 10. Settings and algorithm
order are defined at the top of `Experiment/benchmark/mopta.py`.
Use `--workers N` to change parallelism. MOPTA orchestration is entirely in this
file; there is no separate `runner_real_life.py`.

MOPTA08 is a 124-dimensional automotive design **executable evaluator**, not a
training dataset. The external Windows executable is acquired from the official
Edinburgh MOPTA 2008 Jones Benchmark deposit, DOI `10.7488/ds/7894`.
On first use, the adapter downloads the PC archive, verifies its checksum and
executable SHA-256, and caches it in `%LOCALAPPDATA%\AG_VS\mopta08`.
Later trials reuse this executable. The executable is not part of the source
release and requires Windows. A missing dependency raises an error; there is
no replacement objective.

Set `MOPTA08_EXECUTABLE` to an existing official executable, or set
`MOPTA08_CACHE_DIR` to choose its cache directory. `--list` and `--plots-only`
do not download or execute it. Inputs are in `[0,1]^124`; the evaluator returns
mass and 68 constraints. Optimizers maximize
`-(mass + 10 * sum(max(0, constraint_i)))`.
Reports convert this to positive penalized cost (lower is better), so negative
optimizer output is expected. `d_eff` is a search parameter, not a known true
active-coordinate mask.

## Run synthetic experiments

For direct use on your own maximization objective:

```python
import numpy as np
from ag_vs import AGVSOptimizer

optimizer = AGVSOptimizer(
    objective=lambda x: -float(np.sum(x[:2] ** 2)),
    bounds=np.tile([-1.0, 1.0], (10, 1)),
    eff_dim=2, n_split=3, n_init=6, max_iter=40,
    t_seq=2, batch_size=3, N_bad=3, seed=2021,
    ipt_solver="vanillabo",  # use "turbo" for AG-VS-TuRBO
    printer_on=False, print_tree_each_iter=False,
)
result = optimizer.optimize()
print("Best observed value:", max(optimizer.y))
```

For minimization, negate your objective. `noise_variance` controls the BO
surrogate's assumed variance (API default `1e-4`); observation noise itself
belongs to the objective/runner. The study runners supply their own settings.

```powershell
python -m Experiment.benchmark.runner_synthetic
python -m Experiment.benchmark.runner_synthetic --benchmarks rastrigin3_300 --algorithms AG-VS-BO --trials 25 --workers 1
```

The default families are Levy, Rastrigin, Ackley, Rosenbrock, and Sphere, each
at `(d_eff,D)=(3,300)` and `(6,1000)`. Each has 700 evaluations, 25 trials,
and one worker. Active coordinates are fixed by the registry's deterministic
placement seed. Observation noise variance is `1e-4`. Objective figures and
tables use noiseless objective values and aggregate across trials, rather than
plotting GP posterior means. Historical Hartmann definitions remain readable
but are excluded from the runnable default scope.

Default algorithms: AG-VS-BO, MCTS-VS-BO, MCTS-VS-TuRBO, TuRBO, Dropout-BO,
Dropout-TuRBO, LA-MCTS-TuRBO, VAE-BO. AG-VS uses `B=3`, `t_seq=2`, and
`N_bad=4*N_split`. Its normal synthetic GP variance follows the configured
observation noise. Fixed BO surrogate noise variance is `1e-4` (standard
deviation `0.01`), including MCTS-VS-BO, Dropout-BO, VAE-BO, and LA-MCTS-BO.
Real-life learned-noise models continue to estimate noise.
Non-default algorithms may have smaller evaluation caps;
see `algorithm_evaluation_budget` in the shared runner. For more than 25 trials,
all selected synthetic conditions finish a block of 25 before the next block.

## Sensitivity and version comparisons

```powershell
python -m Experiment.grid_search_nbad
python -m Experiment.grid_search_tseq
python -m Experiment.version_comparison
```

The grid searches compare both AG-VS backends on Rastrigin3_300 and
Rastrigin6_1000, with 25 trials and 25 workers per cell.
`N_bad` values are `6,10,14,18,20,28`; `t_seq` values are `1,2,4,6,8,10`.
The current t_seq figures omit 1, although it remains in the run plan.
The version comparison runs only Rastrigin6_1000, comparing merge-enabled
and merge-disabled AG-VS with its explicit GP variance `1e-4`.
Each command's `--help` documents available overrides.

## Refresh reports and validate

```powershell
python -m Experiment.benchmark.mopta --plots-only
python -m Experiment.benchmark.runner_synthetic --plots-only
python -m Experiment.grid_search_nbad --plots-only
python -m Experiment.grid_search_tseq --plots-only
python -m Experiment.version_comparison --plots-only
python -m pytest -q
python -m Experiment.utils smoke --suite synthetic --algorithms AG-VS-BO
```

MOPTA refreshes current figures before optimization and figures/tables after its
algorithm sequence finishes. Its table covers the active MOPTA condition and
uses AG-VS-TuRBO as the comparison reference. Raw traces live under
`Experiment/Result/{synthetic,real_life}`, figures under `Experiment/Result/Figures`,
and tables under `Experiment/tables`. Grid/ablation outputs have separate
subdirectories. Plot-only commands require existing traces; a source-only
checkout contains no experimental results.

The smoke command runs a small bounded check outside study results and writes
`.cache/smoke.json`. `--suite mopta` explicitly runs the physical evaluator and
requires its executable. Smoke GTBO uses one group test; smoke output is not
scientific comparison data.

## What each file does

| File or directory | Responsibility |
| --- | --- |
| `ag_vs/optimizer.py` | AG-VS loop, local models, candidates, scores and restarts. |
| `ag_vs/helper_functions.py` | Tree traversal, updates, splitting and merging. |
| `ag_vs/utils.py` | Node/data structures, numerical helpers, isolated random streams and grid cache identity. |
| `ag_vs/GaussianProcess.py` | GP fitting and prediction. |
| `ag_vs/inner_optimizer/` | BO and TuRBO candidate generation. |
| `ag_vs/EarlyStopMechanism.py` | Optional stopping calculations. |
| `ag_vs/Figure.py` | Optimizer and tree visualizations. |
| `ag_vs/helper.py`, `ag_vs/__init__.py` | Optional command dispatch and lazy public imports. |
| `Experiment/utils.py` | Benchmark registry, objectives, seeds, configuration tables, worker entry point and smoke checks. |
| `Experiment/benchmark/runner_synthetic.py` | Synthetic entry point and shared scheduling, optimizer dispatch, cache, figures and tables. |
| `Experiment/benchmark/mopta.py` | MOPTA settings, algorithm order, scheduling and reporting. |
| `Experiment/benchmark/real_life/tasks.py` | Lazy, multiprocessing-safe MOPTA objective factory and configuration. |
| `Experiment/benchmark/real_life/mopta08_backend.py` | Executable acquisition, verification and evaluation. |
| `Experiment/grid_search_nbad.py`, `Experiment/grid_search_tseq.py` | Sensitivity studies and figures. |
| `Experiment/version_comparison.py` | Merge-policy ablation. |
| `Experiment/benchmark/baseline/` | Comparison adapters, native backends, shared settings and attributed reference code. |
| `Experiment/benchmark/tests/`, `Experiment/benchmark/real_life/tests/` | Algorithm, cache/seed, reporting, entry-point and MOPTA tests. |

The former `run.py`, `run_real_Life.py`, `runner_real_life.py`, `utils_exp.py`,
`native_smoke.py`, `ag_vs/randomness.py` and `ag_vs/grid_support.py` are replaced
by the modules above.

## Anonymous distribution

Share source, requirements, documentation and third-party licences. Exclude
`.git` (including vendored Git metadata), `.cache`, `.venv`, `.vscode`, bytecode,
logs, local backups, `environment.txt`,
`Experiment/Result`, and generated `Experiment/tables`: these can contain user
paths and historical data. Retain attribution to upstream authors. Inspect the
actual upload and its hosting URL for submission-author identifiers. Matching
seeds does not guarantee identical wall-clock times or bitwise trajectories
across hardware, library versions or execution environments.
