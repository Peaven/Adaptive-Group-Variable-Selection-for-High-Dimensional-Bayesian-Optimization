# Native benchmark implementations

The active runners execute project-owned Python modules in the main environment.
They do not launch external optimizers, Ax strategies, or another interpreter.
`upstream_sources.json` records source revisions, supplied-file SHA-256 hashes,
local modules, license notices, and the validated numerical-library versions.

| Methods | Project modules | Reference revision |
|---|---|---|
| MCTS-VS, Dropout, VAE-BO, SAASBO | `native_optimizers.py`, `native_baselines.py`, `native/` | MCTS-VS `29d32b6977d37f5e257d359373497e375befa96b` and supplied Song scripts |
| TuRBO, inner TuRBO runs | `native/turbo/` | Uber `de0db39f481d9505bb3610b7b7aa0ebf7702e4a5` |
| LA-MCTS | `native/lamcts/`, `native_baselines.py` | LA-MCTS copy in the pinned MCTS-VS snapshot |
| HeSBO, ALEBO | `embeddings.py`, `native/alebo_model.py` | Ax 0.2.2 source and supplied scripts; no Ax runtime |
| GTBO | `gtbo.py`, `native/group_testing/` | `fe3dca9d179e46b4d9cc7bc6954b09e1078919e8` |
| LassoBO | `lassobo_adapter.py` | Existing Hoang et al. implementation retained |

REMBO is reference material only. Existing historical result files are preserved;
current reports require the native implementation identity and matching settings.

## Effective settings and compatibility changes

- The maintained BO adapters use fixed GP observation variance `1e-4`
  (standard deviation `0.01`), overriding the upstream `alpha=0.01` setting.
  This applies to MCTS-VS-BO, Dropout-BO, VAE-BO, Vanilla BO, and LA-MCTS-BO;
  the AG-VS API default and merge comparison also use `1e-4`.
  Real-life learned-noise models retain noise estimation. Upstream reference
  snapshots and stored experimental results are unchanged.
- Every trial ID 0?49 uses seed 2021?2070 for both benchmark and optimizer.
  Python, NumPy, Torch, Sobol, and JAX streams initialize with that seed and
  advance within the trial. Numerical work runs inside an isolated RNG context.
  NAS sampling advances its own persistent stream initialized with the same seed.
  Synthetic active-coordinate construction seeds are unchanged.
- AG-VS-TuRBO uses three Thompson samples per stage and two sequential stages
  per visit (or the selected t_seq sweep value), including replacement roots,
  merges, and trust-region restarts. Baseline TuRBO retains batch 10; applicable
  inner runs retain their 50-evaluation limit and BestK-20 imputation.
- SAASBO retains Song's RBF SAAS GP, half-Cauchy sparsity scale 0.1, NUTS with
  512 warmup and 256 posterior draws, thinning 16, depth 6, 10 initial points,
  and the three largest integrated pointwise EI values among 1,024 Sobol points.
  GP fitting receives the actual trial seed; JAX keys advance between fits.
  Coordinates normalize before fitting and unnormalize before evaluation.
- HeSBO uses a single sparse signed hash projection and two initial points
  (the supplied k=1 strategy), Mat?rn-5/2 ARD embedded GP, and analytic EI.
- ALEBO normalizes projection columns, initializes inside the feasible polytope,
  learns the upper-triangular Mahalanobis metric, and integrates 25 diagonal
  Laplace samples by moment matching. It uses 10 feasible initial points,
  10 initial MAP restarts, and constrained SLSQP EI optimization with 1,000
  feasible starts reduced to 10 restarts. Modern BoTorch fits the MAP GP;
  autograd replaces finite differences for diagonal curvature. The source
  1e-3 curvature nugget is also the floor for numerical negative curvature.
  Modern Torch linear algebra and GPyTorch posterior APIs replace retired APIs.
- GTBO follows the [paper](https://arxiv.org/html/2310.03515v1) and pinned
  authors' code. It uses 10,000 SMC particles, prior activity 0.05, three initial
  groups, classification threshold 0.5, and convergence thresholds 0.001/0.9.
  The lower threshold resolves the malformed paper value using the code value.
  Full-dimensional Mat?rn-5/2 BO uses sequential qLogNEI on CPU float64,
  active LN(0,1) and inactive LN(7,1) priors (LN(3,1) on NAS-Bench-101).
  Code-derived controls are at most 100 group tests, at most five groups per
  selection, MI-loss tolerance 0.01, 4,096 raw acquisition samples, 16 restarts,
  and GP reset every 10 steps. It retains 1,024 acquisition MC samples and
  reference nonconverged-selection cutoffs 1e-5/0.95.
  All 10 default measurements, variance probes, group tests, 10 BO initial
  points, and subsequent proposals count against the budget. Deduplication
  affects GP training only. Constant-observation standard deviations are
  floored; partial final phases stop at the objective boundary.
  A benchmark default takes precedence; otherwise midpoint is used, except
  centered-optimum tasks use normalized 0.25. No active coordinates are supplied.
- Local imports replace upstream sys.path manipulation. TuRBO final batches
  truncate to the remaining budget. Modern JAX mapping replaces the retired
  chunk helper. The scalar Bernoulli log-prior normalization uses the number
  of dimensions instead of the upstream particle-count typo; normalized SMC
  weights are unaffected by this constant correction.

Licenses and retained notices are in `native/licenses/`. The MCTS-VS and SAASBO
repositories do not include a repository license file; no license is inferred.
The original upstream snapshots remain reference material; active imports use
`native/`. Optional non-roster methods keep their separately documented gates.

## Sweeps and outputs

`python -m Experiment.grid_search_nbad --list` and
`python -m Experiment.grid_search_tseq --list` display the complete plans without
launching trials. Both use AG-VS-BO and AG-VS-TuRBO on rastrigin3_300 and
rastrigin6_1000, 25 shared seeds (2021–2045), batch 3, and noise variance 1e-4.
The absolute N_bad sweep is 6,10,14,18,20,28 with t_seq=2. The t_seq sweep is
1,2,4,6,8,10 with N_bad multiplier 4. Each has 24 cells / 600 trials.
Each sweep exports four separate six-curve figures, with consistent parameter
colors, evaluation-count axes, staggered 95% intervals, PDF/PNG, and curve and
summary CSVs. Figures require compatible completed traces; validation never
fabricates study results or launches the complete study.

Bounded validation can be repeated with `python -m Experiment.utils smoke
--suite synthetic` or `--suite mopta`. The smoke harness records its GTBO
group-test cap explicitly. MOPTA checks require the official Windows evaluator.
