"""Single source of truth for benchmark identities and experiment-level seeds.

Optimizer settings belong to the optimizer adapters.  This module owns only the
experiment matrix and the randomness that must be shared across optimizers:

* the benchmark identity;
* the physical/effective dimensions and evaluation budget;
* the fixed synthetic active-coordinate placement; and
* the ordered algorithm/benchmark seed rules for ``trial_id``.

The registry intentionally has no dependency on runner or baseline modules, so
benchmark wrappers and report generators can import it without loading optional
optimization packages.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import sys
from types import MappingProxyType
from typing import Iterable, Mapping

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


SYNTHETIC_D_EFFS = (3, 6)
SYNTHETIC_AMBIENT_DIMS = (300, 1000)
SYNTHETIC_TRIALS = 25
SYNTHETIC_BENCHMARK_NOISE_VARIANCE = 1e-4
SYNTHETIC_BENCHMARK_NOISE_STD = float(
    np.sqrt(SYNTHETIC_BENCHMARK_NOISE_VARIANCE)
)
# Run only the two paired synthetic scales requested by the experiment design.
# These are not a Cartesian product: 3-1000 and 6-300 are intentionally absent.
SYNTHETIC_DIMENSION_MATRIX = (
    (3, 300),
    (6, 1000),
)
SYNTHETIC_OBJECTIVES = (
    "levy",
    "rastrigin",
    "ackley",
    "hartmann",
    "rosenbrock",
    "sphere",
)

# Exact project/reference algorithm roster for the dedicated synthetic table.
# Ordering is presentation/execution ordering only; it is deliberately not a
# seed offset (all methods consume the experiment's ordered SeedBundle).
SYNTHETIC_ALGORITHMS = (
    "AG-VS-BO",
      "MCTS-VS-BO",
      "MCTS-VS-TuRBO",
      "MCTS-VS-SAASBO",
      "TuRBO",
      "CMA-ES",
      "Dropout-BO",
    "Dropout-TuRBO",
    "LassoBO",
    "LA-MCTS-TuRBO",
    "SAASBO",
    "GTBO",
    "HeSBO",
    "ALEBO",
    "VAE-BO",
)

REAL_LIFE_D_EFFS = (3, 4)
REAL_LIFE_D_EFFS_BY_TASK = MappingProxyType({"mopta08": REAL_LIFE_D_EFFS})
REAL_LIFE_EVALUATION_BUDGETS = MappingProxyType({"mopta08": 150})

# Algorithms with retained real-life implementations/metadata. VAE-BO remains
# readable for historical result files but is excluded from new real-life runs.
REAL_LIFE_REFERENCE_ALGORITHMS = (
    "AG-VS-BO", "AG-VS-TuRBO", "MCTS-VS-BO", "MCTS-VS-TuRBO", "TuRBO",
    "LA-MCTS-BO", "LA-MCTS-TuRBO", "HeSBO", "ALEBO", "GTBO",
)
REAL_LIFE_ALGORITHMS = REAL_LIFE_REFERENCE_ALGORITHMS

# ``--seed`` in the runner is an optional additive offset. Synthetic optimizer
# and benchmark randomness share the same paper-style trial sequence. The
# current 25-trial synthetic plan uses ids 0, ..., 24 and seeds 2021, ..., 2045;
# the seed helpers retain the longer historical sequence for old traces.
BASE_ALGORITHM_SEED = 2021
BASE_BENCHMARK_SEED = 2021


def _normalise_component(value: object) -> str:
    if isinstance(value, str):
        return value.strip().lower()
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return str(value).strip().lower()


def deterministic_seed(*components: object, namespace: str = "experiment") -> int:
    """Return a stable unsigned 64-bit seed for ordered identity components.

    Python's built-in ``hash`` is process-randomised, so it must not be used for
    persisted experiment identities.  Length-prefixing makes the encoding
    unambiguous (for example, ``("ab", "c")`` differs from ``("a", "bc")``).
    """

    encoded_parts = [_normalise_component(namespace), *map(_normalise_component, components)]
    payload = bytearray()
    for part in encoded_parts:
        raw = part.encode("utf-8")
        payload.extend(len(raw).to_bytes(4, byteorder="big", signed=False))
        payload.extend(raw)
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def effective_dimension_seed(objective_name: str, d_eff: int, ambient_dim: int) -> int:
    """Seed owned by one exact synthetic ``(objective, d_eff, D)`` identity."""

    return deterministic_seed(
        str(objective_name).strip().lower(),
        int(d_eff),
        int(ambient_dim),
        namespace="effective_dimension",
    )


def effective_dimension_indices(
    objective_name: str,
    d_eff: int,
    ambient_dim: int,
) -> tuple[int, ...]:
    """Choose fixed active coordinates exactly once from the benchmark identity.

    The returned order is the order produced by ``Generator.choice``.  It is not
    sorted because coordinate order is part of how a low-dimensional objective
    is embedded (notably for Rosenbrock and Hartmann).
    """

    d_eff = int(d_eff)
    ambient_dim = int(ambient_dim)
    if d_eff <= 0:
        raise ValueError("d_eff must be positive")
    if ambient_dim < d_eff:
        raise ValueError("ambient_dim must be greater than or equal to d_eff")
    seed = effective_dimension_seed(objective_name, d_eff, ambient_dim)
    rng = np.random.default_rng(seed)
    return tuple(
        int(index)
        for index in rng.choice(ambient_dim, size=d_eff, replace=False).tolist()
    )


def algorithm_seed(trial_id: int, *, base_seed_offset: int = 0) -> int:
    """Optimizer-internal seed: ``BASE_ALGORITHM_SEED + offset + trial_id``."""

    if int(base_seed_offset) != 0:
        raise ValueError("Trial seeds are fixed at 2021..2070; seed offset must be zero")
    trial_id = _validate_trial_id(trial_id)
    return int(BASE_ALGORITHM_SEED + int(base_seed_offset) + trial_id)


def benchmark_seed(trial_id: int, *, base_seed_offset: int = 0) -> int:
    """Benchmark/noise seed: ``BASE_BENCHMARK_SEED + offset + trial_id``."""

    if int(base_seed_offset) != 0:
        raise ValueError("Trial seeds are fixed at 2021..2070; seed offset must be zero")
    trial_id = _validate_trial_id(trial_id)
    return int(BASE_BENCHMARK_SEED + int(base_seed_offset) + trial_id)


def _validate_trial_id(trial_id: int) -> int:
    trial_id = int(trial_id)
    if not 0 <= trial_id < 50:
        raise ValueError("trial_id must use the ordered range 0, ..., 49")
    return trial_id


@dataclass(frozen=True)
class SeedBundle:
    """The explicitly separated seeds for one benchmark/trial execution."""

    trial_id: int
    algorithm_seed: int
    benchmark_seed: int
    effective_dimension_seed: int | None


@dataclass(frozen=True)
class SyntheticExperiment:
    objective: str
    d_eff: int
    ambient_dim: int
    budget: int
    effective_dimension_seed: int
    effective_dimension_indices: tuple[int, ...]

    @property
    def key(self) -> str:
        """Short CLI/result key retained for compatibility with existing traces."""

        return f"{self.objective}{self.d_eff}_{self.ambient_dim}"

    @property
    def experiment_id(self) -> str:
        return f"synthetic/{self.objective}/deff{self.d_eff}/D{self.ambient_dim}"

    @property
    def D(self) -> int:
        return int(self.ambient_dim)

    @property
    def selection_batch_size(self) -> int:
        return 3

    @property
    def q_batch(self) -> None:
        return None

    @property
    def t_seq(self) -> int:
        return 2

    def seeds(self, trial_id: int, *, base_seed_offset: int = 0) -> SeedBundle:
        trial_id = _validate_trial_id(trial_id)
        return SeedBundle(
            trial_id=trial_id,
            algorithm_seed=algorithm_seed(trial_id, base_seed_offset=base_seed_offset),
            benchmark_seed=benchmark_seed(trial_id, base_seed_offset=base_seed_offset),
            effective_dimension_seed=int(self.effective_dimension_seed),
        )

    def metadata(self) -> dict[str, object]:
        budget_source = (
            "MCTS-VS official Hartmann reference max_samples=600"
            if self.objective == "hartmann"
            else "pre-existing project synthetic-suite default"
        )
        return {
            "experiment_id": self.experiment_id,
            "experiment_kind": "synthetic",
            "objective": self.objective,
            "D": int(self.ambient_dim),
            "d_eff": int(self.d_eff),
            "budget": int(self.budget),
            "trials": SYNTHETIC_TRIALS,
            "budget_source": budget_source,
            "benchmark_noise_std": SYNTHETIC_BENCHMARK_NOISE_STD,
            "benchmark_noise_variance": SYNTHETIC_BENCHMARK_NOISE_VARIANCE,
            "benchmark_noise_distribution": "Gaussian(0, 1e-4)",
            "effective_dimension_seed": int(self.effective_dimension_seed),
            "effective_dimension_indices": list(self.effective_dimension_indices),
            "q_batch": self.q_batch,
            "t_seq": int(self.t_seq),
            "selection_batch_size": self.selection_batch_size,
            "batch_size": self.selection_batch_size,
        }


@dataclass(frozen=True)
class RealLifeTask:
    key: str
    display_name: str
    upstream_name: str
    physical_dim: int
    bounds: tuple[float, float]
    budget: int
    metric_name: str
    paper_scope: str
    reference_embedding_dimensions: tuple[tuple[str, int], ...] = ()
    paper_omitted_algorithms: tuple[str, ...] = ()

    def reference_embedding_dimension(self, algorithm: str) -> int | None:
        dimensions = dict(self.reference_embedding_dimensions)
        value = dimensions.get(str(algorithm))
        return None if value is None else int(value)


@dataclass(frozen=True)
class RealLifeExperiment:
    task: RealLifeTask
    configured_d_eff: int
    trials: int = 25

    @property
    def key(self) -> str:
        return f"{self.task.key}_deff{self.configured_d_eff}"

    @property
    def experiment_id(self) -> str:
        return f"real_life/{self.task.display_name}/deff{self.configured_d_eff}"

    @property
    def benchmark(self) -> str:
        return self.task.display_name

    @property
    def d_eff(self) -> int:
        return int(self.configured_d_eff)

    @property
    def physical_dim(self) -> int:
        return int(self.task.physical_dim)

    @property
    def budget(self) -> int:
        return int(self.task.budget)

    @property
    def selection_batch_size(self) -> int:
        return 3

    @property
    def q_batch(self) -> None:
        return None

    @property
    def t_seq(self) -> int:
        return 2

    def seeds(self, trial_id: int, *, base_seed_offset: int = 0) -> SeedBundle:
        trial_id = _validate_trial_id(trial_id)
        paper_seed = benchmark_seed(
            trial_id, base_seed_offset=base_seed_offset
        )
        return SeedBundle(
            trial_id=trial_id,
            # Real-life repetitions extend the paper's seed sequence end-to-end:
            # optimizer and benchmark randomness both use 2021 + trial_id.
            algorithm_seed=paper_seed,
            benchmark_seed=paper_seed,
            # These controlled conditions do not fabricate a ground-truth active
            # subset for the physical NAS/control objective.
            effective_dimension_seed=None,
        )

    def metadata(self) -> dict[str, object]:
        return {
            "experiment_id": self.experiment_id,
            "experiment_kind": "real_life",
            "benchmark": self.task.display_name,
            "task_key": self.task.key,
            "upstream_name": self.task.upstream_name,
            "physical_dim": int(self.task.physical_dim),
            "configured_d_eff": int(self.configured_d_eff),
            "bounds": tuple(float(value) for value in self.task.bounds),
            "budget": int(self.task.budget),
            "trials": int(self.trials),
            "paper_reference_trial_seeds": [2021, 2022, 2023, 2024, 2025],
            "protocol_scope": (
                f"{self.trials} sequential seeds "
                f"(2021--{2020 + self.trials}) and configured d_eff variants"
            ),
            "metric_name": self.task.metric_name,
            "paper_scope": self.task.paper_scope,
            "reference_embedding_dimensions": dict(
                self.task.reference_embedding_dimensions
            ),
            "paper_omitted_algorithms": list(self.task.paper_omitted_algorithms),
            "q_batch": self.q_batch,
            "t_seq": int(self.t_seq),
            "selection_batch_size": self.selection_batch_size,
            "batch_size": self.selection_batch_size,
            "effective_dimension_seed": None,
            "effective_dimension_indices": None,
        }


_REAL_LIFE_TASKS = {"mopta08": RealLifeTask(
            key="mopta08",
            display_name="MOPTA08",
            upstream_name="MOPTA 2008 Jones automotive benchmark",
            physical_dim=124,
            bounds=(0.0, 1.0),
            budget=REAL_LIFE_EVALUATION_BUDGETS["mopta08"],
            metric_name="negative_soft_constrained_vehicle_mass",
            paper_scope="GTBO real-world benchmark",
            # Keep the embedding baselines identical to the NASBench setup.
            reference_embedding_dimensions=(("HeSBO", 10), ("ALEBO", 10)),
        )}
REAL_LIFE_TASKS: Mapping[str, RealLifeTask] = MappingProxyType(_REAL_LIFE_TASKS)


def _synthetic_budget(objective: str) -> int:
    # The official Hartmann comparison uses max_samples=600.  Other existing
    # project objectives retain their established 700-evaluation budget.
    return 600 if objective == "hartmann" else 700


def _build_synthetic_experiments() -> dict[str, SyntheticExperiment]:
    experiments: dict[str, SyntheticExperiment] = {}
    # Insertion order is the bare runner's execution order. Complete all
    # configured scales of Levy first, then Rastrigin, then Ackley; remaining
    # objective families follow the declared SYNTHETIC_OBJECTIVES order.
    for objective in SYNTHETIC_OBJECTIVES:
        for d_eff, ambient_dim in SYNTHETIC_DIMENSION_MATRIX:
            seed = effective_dimension_seed(objective, d_eff, ambient_dim)
            experiment = SyntheticExperiment(
                objective=objective,
                d_eff=d_eff,
                ambient_dim=ambient_dim,
                budget=_synthetic_budget(objective),
                effective_dimension_seed=seed,
                effective_dimension_indices=effective_dimension_indices(
                    objective,
                    d_eff,
                    ambient_dim,
                ),
            )
            experiments[experiment.key] = experiment
    return experiments


def _build_real_life_experiments() -> dict[str, RealLifeExperiment]:
    return {
        experiment.key: experiment
        for task_key, configured_dimensions in REAL_LIFE_D_EFFS_BY_TASK.items()
        for task in (REAL_LIFE_TASKS[task_key],)
        for configured_d_eff in configured_dimensions
        for experiment in (
            RealLifeExperiment(
                task,
                configured_d_eff,
                trials=25,
            ),
        )
    }


_SYNTHETIC_EXPERIMENTS = _build_synthetic_experiments()
_REAL_LIFE_EXPERIMENTS = _build_real_life_experiments()
SYNTHETIC_EXPERIMENTS: Mapping[str, SyntheticExperiment] = MappingProxyType(
    _SYNTHETIC_EXPERIMENTS
)
REAL_LIFE_EXPERIMENTS: Mapping[str, RealLifeExperiment] = MappingProxyType(
    _REAL_LIFE_EXPERIMENTS
)


def get_synthetic_experiment(identifier: str) -> SyntheticExperiment:
    key = str(identifier).strip()
    lowered = key.lower()
    if lowered in SYNTHETIC_EXPERIMENTS:
        return SYNTHETIC_EXPERIMENTS[lowered]
    for experiment in SYNTHETIC_EXPERIMENTS.values():
        if experiment.experiment_id.lower() == lowered:
            return experiment
    raise KeyError(
        f"Unknown synthetic experiment {identifier!r}; "
        f"available={tuple(SYNTHETIC_EXPERIMENTS)}"
    )


def get_real_life_task(task: str) -> RealLifeTask:
    key = str(task).strip().lower().replace("-", "")
    if key in {"mopta", "mopta08"}:
        return REAL_LIFE_TASKS["mopta08"]
    raise KeyError(f"Unknown real-life task {task!r}; available={tuple(REAL_LIFE_TASKS)}")


def get_real_life_experiment(identifier: str, configured_d_eff: int | None = None) -> RealLifeExperiment:
    if configured_d_eff is not None:
        task = get_real_life_task(identifier)
        key = f"{task.key}_deff{int(configured_d_eff)}"
        try:
            return REAL_LIFE_EXPERIMENTS[key]
        except KeyError as exc:
            allowed = REAL_LIFE_D_EFFS_BY_TASK.get(task.key, (3, 6))
            raise KeyError(
                f"{task.key} configured_d_eff must be one of {allowed}; "
                f"got {configured_d_eff}"
            ) from exc
    key = str(identifier).strip().lower()
    if key in REAL_LIFE_EXPERIMENTS:
        return REAL_LIFE_EXPERIMENTS[key]
    for experiment in REAL_LIFE_EXPERIMENTS.values():
        if experiment.experiment_id.lower() == key:
            return experiment
    raise KeyError(
        f"Unknown real-life experiment {identifier!r}; "
        f"available={tuple(REAL_LIFE_EXPERIMENTS)}"
    )


def _coerce_real_life_experiment(
    experiment: RealLifeExperiment | str,
) -> RealLifeExperiment:
    if isinstance(experiment, RealLifeExperiment):
        return experiment
    return get_real_life_experiment(str(experiment))


def real_life_algorithm_is_applicable(
    experiment: RealLifeExperiment | str,
    algorithm: str,
) -> bool:
    """Whether ``algorithm`` belongs in this reference real-life condition.

    The controlled configured ``d_eff`` remains part of the experiment
    identity. Historical paper omissions are retained as task metadata, but do
    not remove methods from the project's complete cross-benchmark comparison.
    """

    experiment = _coerce_real_life_experiment(experiment)
    algorithm = str(algorithm)
    return algorithm in REAL_LIFE_ALGORITHMS


def iter_applicable_real_life_algorithms(
    experiment: RealLifeExperiment | str,
) -> tuple[str, ...]:
    """Return the ordered reference algorithms applicable to one condition."""

    experiment = _coerce_real_life_experiment(experiment)
    return tuple(
        algorithm
        for algorithm in REAL_LIFE_ALGORITHMS
        if real_life_algorithm_is_applicable(experiment, algorithm)
    )


def real_life_reference_embedding_dimension(
    experiment: RealLifeExperiment | str,
    algorithm: str,
) -> int | None:
    """Return an explicit paper embedding override, or ``None`` if absent."""

    experiment = _coerce_real_life_experiment(experiment)
    return experiment.task.reference_embedding_dimension(str(algorithm))


def real_life_optimizer_dimension(
    experiment: RealLifeExperiment | str,
    algorithm: str,
) -> int:
    """Resolve the runtime search/embedding dimension for an applicable method.

    HeSBO/ALEBO reference exceptions override the configured condition for that
    method only.  No coordinate mask is fabricated and the objective's physical
    dimension is unchanged.
    """

    experiment = _coerce_real_life_experiment(experiment)
    algorithm = str(algorithm)
    if not real_life_algorithm_is_applicable(experiment, algorithm):
        raise ValueError(
            f"{algorithm} is not applicable to reference condition {experiment.key}"
        )
    reference_dimension = real_life_reference_embedding_dimension(
        experiment,
        algorithm,
    )
    if reference_dimension is not None:
        return int(reference_dimension)
    return int(experiment.configured_d_eff)


def get_experiment(identifier: str) -> SyntheticExperiment | RealLifeExperiment:
    try:
        return get_synthetic_experiment(identifier)
    except KeyError:
        return get_real_life_experiment(identifier)


def iter_synthetic_experiments() -> Iterable[SyntheticExperiment]:
    return iter(SYNTHETIC_EXPERIMENTS.values())


def iter_real_life_experiments() -> Iterable[RealLifeExperiment]:
    return iter(REAL_LIFE_EXPERIMENTS.values())


def validate_registry() -> None:
    expected_synthetic = {
        (objective, d_eff, ambient_dim)
        for objective in SYNTHETIC_OBJECTIVES
        for d_eff, ambient_dim in SYNTHETIC_DIMENSION_MATRIX
    }
    actual_synthetic = {
        (experiment.objective, experiment.d_eff, experiment.ambient_dim)
        for experiment in SYNTHETIC_EXPERIMENTS.values()
    }
    if actual_synthetic != expected_synthetic:
        raise AssertionError("Synthetic registry does not match the required dimension matrix")

    identities: set[str] = set()
    for experiment in SYNTHETIC_EXPERIMENTS.values():
        if experiment.experiment_id in identities:
            raise AssertionError(f"Duplicate experiment identity: {experiment.experiment_id}")
        identities.add(experiment.experiment_id)
        indices = experiment.effective_dimension_indices
        if len(indices) != experiment.d_eff or len(set(indices)) != experiment.d_eff:
            raise AssertionError(f"Invalid active-coordinate subset for {experiment.key}")
        if any(index < 0 or index >= experiment.ambient_dim for index in indices):
            raise AssertionError(f"Out-of-range active coordinate for {experiment.key}")
        if indices != effective_dimension_indices(
            experiment.objective,
            experiment.d_eff,
            experiment.ambient_dim,
        ):
            raise AssertionError(f"Non-deterministic active-coordinate subset for {experiment.key}")

    expected_real_life = {
        (task, d_eff)
        for task, dimensions in REAL_LIFE_D_EFFS_BY_TASK.items()
        for d_eff in dimensions
    }
    actual_real_life = {
        (experiment.task.key, experiment.configured_d_eff)
        for experiment in REAL_LIFE_EXPERIMENTS.values()
    }
    if actual_real_life != expected_real_life:
        raise AssertionError("Real-life registry does not match its task-specific d_eff matrix")
    if len(REAL_LIFE_REFERENCE_ALGORITHMS) != len(
        set(REAL_LIFE_REFERENCE_ALGORITHMS)
    ):
        raise AssertionError("Duplicate algorithm in real-life reference scope")
    if len(SYNTHETIC_ALGORITHMS) != len(set(SYNTHETIC_ALGORITHMS)):
        raise AssertionError("Duplicate algorithm in synthetic reference scope")
    for task in REAL_LIFE_TASKS.values():
        expected_budget = int(REAL_LIFE_EVALUATION_BUDGETS[task.key])
        if int(task.budget) != expected_budget:
            raise AssertionError(
                f"Real-life task {task.key} must use its registered "
                f"{expected_budget}-evaluation budget"
            )
        reference_dimensions = dict(task.reference_embedding_dimensions)
        if any(
            algorithm not in REAL_LIFE_REFERENCE_ALGORITHMS
            for algorithm in reference_dimensions
        ):
            raise AssertionError(f"Unknown reference embedding algorithm for {task.key}")
        if any(int(dimension) <= 0 for dimension in reference_dimensions.values()):
            raise AssertionError(f"Invalid reference embedding dimension for {task.key}")
        if any(
            algorithm not in REAL_LIFE_REFERENCE_ALGORITHMS
            for algorithm in task.paper_omitted_algorithms
        ):
            raise AssertionError(f"Unknown paper omission for {task.key}")


validate_registry()


__all__ = [
    "BASE_ALGORITHM_SEED",
    "BASE_BENCHMARK_SEED",
    "REAL_LIFE_D_EFFS",
    "REAL_LIFE_D_EFFS_BY_TASK",
    "REAL_LIFE_EVALUATION_BUDGETS",
    "REAL_LIFE_ALGORITHMS",
    "REAL_LIFE_REFERENCE_ALGORITHMS",
    "REAL_LIFE_EXPERIMENTS",
    "REAL_LIFE_TASKS",
    "SYNTHETIC_AMBIENT_DIMS",
    "SYNTHETIC_ALGORITHMS",
    "SYNTHETIC_D_EFFS",
    "SYNTHETIC_DIMENSION_MATRIX",
    "SYNTHETIC_EXPERIMENTS",
    "SYNTHETIC_OBJECTIVES",
    "SYNTHETIC_TRIALS",
    "RealLifeExperiment",
    "RealLifeTask",
    "SeedBundle",
    "SyntheticExperiment",
    "algorithm_seed",
    "benchmark_seed",
    "deterministic_seed",
    "effective_dimension_indices",
    "effective_dimension_seed",
    "get_experiment",
    "get_real_life_experiment",
    "get_real_life_task",
    "get_synthetic_experiment",
    "iter_real_life_experiments",
    "iter_applicable_real_life_algorithms",
    "iter_synthetic_experiments",
    "real_life_algorithm_is_applicable",
    "real_life_optimizer_dimension",
    "real_life_reference_embedding_dimension",
    "validate_registry",
]

# Synthetic objective construction

import math
from typing import Callable

import numpy as np

Array = np.ndarray
Objective = Callable[[Array], float]


def _vector(x: object, ambient_dim: int) -> Array:
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    vector = np.asarray(x, dtype=float).reshape(-1)
    if vector.size != int(ambient_dim):
        raise ValueError(
            f"Synthetic objective expected a candidate with D={ambient_dim}; "
            f"received shape {np.asarray(x).shape} ({vector.size} values)."
        )
    return vector


def _unit_active_coordinates(
    x: object,
    ambient_dim: int,
    active_indices: Array,
    active_bounds: Array,
) -> Array:
    vector = _vector(x, ambient_dim)
    lower = active_bounds[:, 0]
    upper = active_bounds[:, 1]
    values = lower + vector[active_indices] * (upper - lower)
    return np.clip(values, lower, upper)


def _unit_optimum(
    ambient_dim: int,
    active_indices: Array,
    active_bounds: Array,
    active_optimum: Array,
) -> Array:
    optimum = np.zeros(int(ambient_dim), dtype=float)
    lower = active_bounds[:, 0]
    upper = active_bounds[:, 1]
    optimum[active_indices] = (active_optimum - lower) / (upper - lower)
    return optimum


def _ackley(experiment: SyntheticExperiment, active: Array) -> dict[str, object]:
    active_bounds = np.repeat([[-5.0, 10.0]], experiment.d_eff, axis=0)
    bounds = np.repeat([[0.0, 1.0]], experiment.ambient_dim, axis=0)

    def score(x: object) -> float:
        z = _unit_active_coordinates(
            x,
            experiment.ambient_dim,
            active,
            active_bounds,
        )
        value = (
            -20.0 * np.exp(-0.2 * np.sqrt(np.sum(z**2) / z.size))
            - np.exp(np.sum(np.cos(2.0 * np.pi * z)) / z.size)
            + 20.0
            + math.e
        )
        return -float(value)

    lipschitz = (
        (4.0 / np.sqrt(experiment.d_eff) + 2.0 * np.pi * np.e)
        * np.sqrt(experiment.d_eff)
        * 15.0
    )
    return {
        "func": score,
        "bounds": bounds,
        "active_bounds": active_bounds,
        "L": float(lipschitz),
        "optimum_x": _unit_optimum(
            experiment.ambient_dim,
            active,
            active_bounds,
            np.zeros(experiment.d_eff, dtype=float),
        ),
    }


def _rastrigin(experiment: SyntheticExperiment, active: Array) -> dict[str, object]:
    active_bounds = np.repeat([[-3.0, 4.0]], experiment.d_eff, axis=0)
    bounds = np.repeat([[0.0, 1.0]], experiment.ambient_dim, axis=0)

    def score(x: object) -> float:
        z = _unit_active_coordinates(
            x,
            experiment.ambient_dim,
            active,
            active_bounds,
        )
        value = 10.0 * z.size + np.sum(z**2 - 10.0 * np.cos(2.0 * np.pi * z))
        return -float(value)

    return {
        "func": score,
        "bounds": bounds,
        "active_bounds": active_bounds,
        "L": float((8.0 + 20.0 * np.pi) * np.sqrt(experiment.d_eff) * 7.0),
        "optimum_x": _unit_optimum(
            experiment.ambient_dim,
            active,
            active_bounds,
            np.zeros(experiment.d_eff, dtype=float),
        ),
    }


def _rosenbrock(experiment: SyntheticExperiment, active: Array) -> dict[str, object]:
    active_bounds = np.repeat([[-2.0, 2.0]], experiment.d_eff, axis=0)
    bounds = np.repeat([[0.0, 1.0]], experiment.ambient_dim, axis=0)

    def score(x: object) -> float:
        z = _unit_active_coordinates(
            x,
            experiment.ambient_dim,
            active,
            active_bounds,
        )
        value = np.sum(
            100.0 * (z[1:] - z[:-1] ** 2) ** 2
            + (1.0 - z[:-1]) ** 2
        )
        return -float(value)

    # On z in [-2, 2]^d, the first, interior, and last gradient components
    # have respective sharp absolute bounds 4806, 6006, and 1200.  All bounds
    # are attained simultaneously at z=(-2, ..., -2).  Candidates live in the
    # unit cube, so the affine active-coordinate map contributes a factor of 4.
    if experiment.d_eff < 2:
        lipschitz = 0.0
    else:
        first_gradient_bound = 400.0 * 2.0 * (2.0 + 2.0**2) + 2.0 * (1.0 + 2.0)
        preceding_term_bound = 200.0 * (2.0 + 2.0**2)
        interior_gradient_bound = first_gradient_bound + preceding_term_bound
        gradient_norm_bound = np.sqrt(
            first_gradient_bound**2
            + max(0, experiment.d_eff - 2) * interior_gradient_bound**2
            + preceding_term_bound**2
        )
        lipschitz = 4.0 * gradient_norm_bound
    return {
        "func": score,
        "bounds": bounds,
        "active_bounds": active_bounds,
        "L": float(lipschitz),
        "optimum_x": _unit_optimum(
            experiment.ambient_dim,
            active,
            active_bounds,
            np.ones(experiment.d_eff, dtype=float),
        ),
    }


def _sphere(experiment: SyntheticExperiment, active: Array) -> dict[str, object]:
    active_bounds = np.repeat([[-2.0, 2.0]], experiment.d_eff, axis=0)
    bounds = np.repeat([[0.0, 1.0]], experiment.ambient_dim, axis=0)

    def score(x: object) -> float:
        z = _unit_active_coordinates(
            x,
            experiment.ambient_dim,
            active,
            active_bounds,
        )
        return -float(np.sum(z**2))

    return {
        "func": score,
        "bounds": bounds,
        "active_bounds": active_bounds,
        "L": float(4.0 * np.sqrt(experiment.d_eff) * 4.0),
        "optimum_x": _unit_optimum(
            experiment.ambient_dim,
            active,
            active_bounds,
            np.zeros(experiment.d_eff, dtype=float),
        ),
    }


def _levy(experiment: SyntheticExperiment, active: Array) -> dict[str, object]:
    bounds = np.repeat([[-10.0, 10.0]], experiment.ambient_dim, axis=0)
    active_bounds = np.repeat([[-10.0, 10.0]], experiment.d_eff, axis=0)

    def score(x: object) -> float:
        vector = _vector(x, experiment.ambient_dim)
        z = np.clip(vector[active], -10.0, 10.0)
        w = 1.0 + (z - 1.0) / 4.0
        first = np.sin(np.pi * w[0]) ** 2
        # Use the standard Levy sum over i=1,...,d-1 (zero-based w[:-1]).
        # The vendored MCTS-VS snapshot starts at w[1] and accidentally omits
        # the first quadratic term; benchmark objectives intentionally follow
        # the canonical mathematical definition rather than that source typo.
        middle = np.sum(
            (w[:-1] - 1.0) ** 2
            * (1.0 + 10.0 * np.sin(np.pi * w[:-1] + 1.0) ** 2)
        )
        last = (w[-1] - 1.0) ** 2 * (
            1.0 + np.sin(2.0 * np.pi * w[-1]) ** 2
        )
        return -float(first + middle + last)

    optimum = np.zeros(experiment.ambient_dim, dtype=float)
    optimum[active] = 1.0
    return {
        "func": score,
        "bounds": bounds,
        "active_bounds": active_bounds,
        "L": 500.0,
        "optimum_x": optimum,
    }


_HARTMANN_ALPHA = np.array([1.0, 1.2, 3.0, 3.2], dtype=float)
_HARTMANN_PARAMETERS = {
    3: (
        np.array(
            [
                [3.0, 10.0, 30.0],
                [0.1, 10.0, 35.0],
                [3.0, 10.0, 30.0],
                [0.1, 10.0, 35.0],
            ],
            dtype=float,
        ),
        1.0e-4
        * np.array(
            [
                [3689.0, 1170.0, 2673.0],
                [4699.0, 4387.0, 7470.0],
                [1091.0, 8732.0, 5547.0],
                [381.0, 5743.0, 8828.0],
            ],
            dtype=float,
        ),
        np.array([0.114614, 0.555649, 0.852547], dtype=float),
    ),
    6: (
        np.array(
            [
                [10.0, 3.0, 17.0, 3.5, 1.7, 8.0],
                [0.05, 10.0, 17.0, 0.1, 8.0, 14.0],
                [3.0, 3.5, 1.7, 10.0, 17.0, 8.0],
                [17.0, 8.0, 0.05, 10.0, 0.1, 14.0],
            ],
            dtype=float,
        ),
        1.0e-4
        * np.array(
            [
                [1312.0, 1696.0, 5569.0, 124.0, 8283.0, 5886.0],
                [2329.0, 4135.0, 8307.0, 3736.0, 1004.0, 9991.0],
                [2348.0, 1451.0, 3522.0, 2883.0, 3047.0, 6650.0],
                [4047.0, 8828.0, 8732.0, 5743.0, 1091.0, 381.0],
            ],
            dtype=float,
        ),
        np.array(
            [0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573],
            dtype=float,
        ),
    ),
}


def _hartmann(experiment: SyntheticExperiment, active: Array) -> dict[str, object]:
    try:
        a_matrix, p_matrix, active_optimum = _HARTMANN_PARAMETERS[experiment.d_eff]
    except KeyError as exc:
        raise ValueError(
            "Hartmann is scientifically defined here only for d_eff in {3, 6}"
        ) from exc

    bounds = np.repeat([[0.0, 1.0]], experiment.ambient_dim, axis=0)
    active_bounds = np.repeat([[0.0, 1.0]], experiment.d_eff, axis=0)

    def score(x: object) -> float:
        z = _vector(x, experiment.ambient_dim)[active]
        inner = np.sum(a_matrix * (z.reshape(1, -1) - p_matrix) ** 2, axis=1)
        return float(np.sum(_HARTMANN_ALPHA * np.exp(-inner)))

    optimum = np.zeros(experiment.ambient_dim, dtype=float)
    optimum[active] = active_optimum
    return {
        "func": score,
        "bounds": bounds,
        "active_bounds": active_bounds,
        "L": 50.0,
        "optimum_x": optimum,
    }


_OBJECTIVE_BUILDERS = {
    "ackley": _ackley,
    "hartmann": _hartmann,
    "levy": _levy,
    "rastrigin": _rastrigin,
    "rosenbrock": _rosenbrock,
    "sphere": _sphere,
}


def make_synthetic_config(
    experiment: str | SyntheticExperiment,
) -> dict[str, object]:
    """Build one runner configuration without consuming trial/algorithm RNG."""

    if not isinstance(experiment, SyntheticExperiment):
        experiment = get_synthetic_experiment(experiment)
    active = np.asarray(experiment.effective_dimension_indices, dtype=int)
    config = _OBJECTIVE_BUILDERS[experiment.objective](experiment, active)
    config.update(
        {
            "experiment_id": experiment.experiment_id,
            "experiment_kind": "synthetic",
            "objective": experiment.objective,
            "D": int(experiment.ambient_dim),
            "d_eff": int(experiment.d_eff),
            "effective_dim": int(experiment.d_eff),
            "effective_dim_role": "ground_truth",
            "active_indices": active.copy(),
            "effective_dimension_indices": active.copy(),
            "effective_dimension_seed": int(experiment.effective_dimension_seed),
            "max_iter": int(experiment.budget),
            "trials": SYNTHETIC_TRIALS,
            "budget_source": (
                "MCTS-VS official Hartmann reference max_samples=600"
                if experiment.objective == "hartmann"
                else "pre-existing project synthetic-suite default"
            ),
            "benchmark_noise_std": SYNTHETIC_BENCHMARK_NOISE_STD,
            "benchmark_noise_variance": SYNTHETIC_BENCHMARK_NOISE_VARIANCE,
            "benchmark_noise_distribution": "Gaussian(0, 1e-4)",
            "q_batch": experiment.q_batch,
            "t_seq": int(experiment.t_seq),
            "selection_batch_size": experiment.selection_batch_size,
            "batch_size": experiment.selection_batch_size,
            "metric_name": "objective",
            "experiment_spec": experiment,
            "seed_bundle_factory": experiment.seeds,
        }
    )
    return config


def make_all_synthetic_configs() -> dict[str, dict[str, object]]:
    return {
        name: make_synthetic_config(experiment)
        for name, experiment in SYNTHETIC_EXPERIMENTS.items()
    }


__all__ = ["make_all_synthetic_configs", "make_synthetic_config"]

# Synthetic configuration-table generation

import csv
import io
import json
from pathlib import Path
from typing import Callable, Iterable, Mapping

NOT_SPECIFIED = "not explicitly specified"
NOT_APPLICABLE = "not applicable"

SYNTHETIC_TABLE_COLUMNS = (
    "Algorithm",
    "Objective",
    "d_eff",
    "D",
    "Effective-dimension indices",
    "Effective-dimension seed",
    "Budget",
    "Initial samples",
    "GP",
    "Kernel",
    "Kernel nu",
    "ARD",
    "Initial lengthscale",
    "Noise",
    "Hyperparameter fitting",
    "Acquisition",
    "Acquisition parameters",
    "Candidate generation",
    "Candidate selection",
    "Acquisition optimizer",
    "Batch size",
    "q_batch",
    "t_seq",
    "AG-VS selection batch",
    "MCTS settings",
    "TuRBO settings",
    "Variable-selection settings",
    "Algorithm seed rule",
    "Benchmark seed rule",
    "Special settings",
)


def _text(value, *, missing=NOT_SPECIFIED) -> str:
    if value is None:
        return missing
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return str(value)


def _first(settings: Mapping[str, object], *keys: str, missing=NOT_SPECIFIED):
    for key in keys:
        if key in settings and settings[key] is not None:
            return settings[key]
    return missing


def _subset(settings: Mapping[str, object], keys: Iterable[str]) -> str:
    values = {key: settings[key] for key in keys if key in settings and settings[key] is not None}
    return _text(values) if values else NOT_APPLICABLE


def _is_non_gp(settings: Mapping[str, object]) -> bool:
    return (
        settings.get("kernel") is None
        and settings.get("gp_kernel") is None
        and str(settings.get("acquisition", "")).lower() in {"none", ""}
    )


def _kernel_nu(settings: Mapping[str, object]) -> object:
    if settings.get("kernel_nu") is not None:
        return settings["kernel_nu"]
    kernel = str(_first(settings, "kernel", "gp_kernel", missing=""))
    if "nu=2.5" in kernel or "5/2" in kernel:
        return 2.5
    return NOT_SPECIFIED


def _noise(settings: Mapping[str, object], experiment: SyntheticExperiment) -> str:
    benchmark_part = (
        f"benchmark Gaussian noise std={SYNTHETIC_BENCHMARK_NOISE_STD:g}, "
        "variance=1e-4 (common seeded stream by evaluation index)"
    )
    if _is_non_gp(settings):
        return benchmark_part
    if settings.get("observation_noise_std") is not None:
        gp_part = (
            f"GP assumed std={settings['observation_noise_std']}"
            + (
                f", alpha={settings['alpha']}"
                if settings.get("alpha") is not None
                else ""
            )
        )
    elif settings.get("noise_variance") is not None:
        gp_part = f"GP noise variance={settings['noise_variance']}"
    elif settings.get("noise") is not None:
        gp_part = str(settings["noise"])
    else:
        gp_part = NOT_SPECIFIED
    return f"{benchmark_part}; {gp_part}"


def _acquisition(settings: Mapping[str, object]) -> object:
    acquisition = settings.get("acquisition")
    if acquisition is not None:
        return acquisition
    selection = str(settings.get("candidate_selection", ""))
    if "Thompson" in selection:
        return "GP posterior Thompson sampling"
    return NOT_SPECIFIED


def _acquisition_parameters(settings: Mapping[str, object]) -> str:
    return _subset(
        settings,
        (
            "xi",
            "ucb_beta",
            "beta",
            "candidate_count",
            "qei_mc_samples",
            "group_size",
            "num_acquisition_evaluations",
            "decomposition_candidates",
            "direct_budget",
        ),
    )


def _gp(settings: Mapping[str, object]) -> object:
    if _is_non_gp(settings):
        return NOT_APPLICABLE
    if settings.get("surrogate_family") is not None:
        return settings["surrogate_family"]
    if settings.get("gp_kernel") is not None:
        return "native TuRBO-1 GP"
    if settings.get("kernel") is not None:
        return "Gaussian process (exact class not explicitly specified)"
    return NOT_SPECIFIED


def _kernel(settings: Mapping[str, object]) -> object:
    if _is_non_gp(settings):
        return NOT_APPLICABLE
    return _first(settings, "kernel", "gp_kernel")


def _ard(settings: Mapping[str, object]) -> object:
    if _is_non_gp(settings):
        return NOT_APPLICABLE
    return _first(settings, "ard", "use_ard")


def _mcts_settings(algorithm: str, settings: Mapping[str, object]) -> str:
    if not algorithm.startswith("MCTS-VS"):
        return NOT_APPLICABLE
    return _subset(
        settings,
        (
            "feature_batch_size",
            "sample_batch_size",
            "min_num_variables",
            "select_right_threshold",
            "k",
            "cp",
            "split_type",
            "uipt_solver",
            "ipt_solver",
        ),
    )


def _turbo_settings(algorithm: str, settings: Mapping[str, object]) -> str:
    if "turbo" not in algorithm.lower():
        return NOT_APPLICABLE
    return _subset(
        settings,
        (
            "implementation",
            "gp_kernel",
            "use_ard",
            "internal_batch_size",
            "n_training_steps",
            "max_cholesky_size",
            "min_cuda",
            "device",
            "dtype",
            "candidate_pool",
            "candidate_generation",
            "candidate_selection",
            "trust_region_logic",
            "turbo_max_evals",
        ),
    )


def _variable_selection(algorithm: str, settings: Mapping[str, object]) -> object:
    if settings.get("variable_selection") is not None:
        return settings["variable_selection"]
    if algorithm.startswith("MCTS-VS"):
        return "MCTS-VS coordinate tree selection"
    if settings.get("active_dims") is not None:
        return f"active_dims={settings['active_dims']} (configured synthetic d_eff)"
    if settings.get("latent_dim") is not None:
        return f"latent dimension={settings['latent_dim']} (configured synthetic d_eff)"
    if settings.get("embedding_dim") is not None:
        return f"embedding dimension={settings['embedding_dim']} (configured synthetic d_eff)"
    if settings.get("active_dim") is not None:
        return f"active/embedding dimension={settings['active_dim']}"
    if algorithm.startswith("Dropout"):
        return "Dropout retained-variable mechanism; active_dims=d_eff"
    return NOT_APPLICABLE


def synthetic_configuration_rows(
    algorithms: Iterable[str],
    settings_provider: Callable[[str, str], Mapping[str, object]],
    experiments: Iterable[SyntheticExperiment] | None = None,
) -> list[dict[str, object]]:
    """Return one row per Algorithm x Objective x d_eff x D."""

    algorithms = tuple(algorithms)
    experiments = tuple(iter_synthetic_experiments() if experiments is None else experiments)
    rows: list[dict[str, object]] = []
    for experiment in experiments:
        for algorithm in algorithms:
            settings = dict(settings_provider(algorithm, experiment.key))
            non_gp = _is_non_gp(settings)
            q_batch = settings.get("q_batch", NOT_APPLICABLE)
            t_seq = settings.get("t_seq", NOT_APPLICABLE)
            selection_batch = (
                experiment.selection_batch_size
                if algorithm.startswith("AG-VS")
                else NOT_APPLICABLE
            )
            budget_source = (
                "MCTS-VS official Hartmann reference max_samples=600"
                if experiment.objective == "hartmann"
                else "pre-existing project synthetic-suite default"
            )
            special_keys = (
                "active_dim",
                "active_dims",
                "embedding_dim",
                "latent_dim",
                "active_dimension_policy",
                "vae_learning_rate",
                "vae_update_interval",
                "latent_training_bounds",
                "latent_acquisition_bounds",
                "latent_bounds_policy",
                "population_size",
                "sigma",
                "group_size",
                "decomposition_strategy",
                "decomposition_candidates",
                "hyperparameter_relearn_interval",
                "low_variance_relearn_trigger",
                "early_relearn_bandwidth_rule",
                "mean_model",
                "duplicate_avoidance",
                "direct_budget",
                "bandwidth_bounds",
                "kernel_scale_bounds",
                "trust_region_center",
                "noise",
                "runtime_status",
                "reference_environment",
                "launch_command",
            )
            algorithm_special = {
                key: settings[key]
                for key in special_keys
                if key in settings and settings[key] is not None
            }
            rows.append(
                {
                    "Algorithm": algorithm,
                    "Objective": experiment.objective,
                    "d_eff": experiment.d_eff,
                    "D": experiment.ambient_dim,
                    "Effective-dimension indices": json.dumps(
                        list(experiment.effective_dimension_indices),
                        separators=(",", ":"),
                    ),
                    "Effective-dimension seed": experiment.effective_dimension_seed,
                    "Budget": experiment.budget,
                    "Initial samples": _first(settings, "n_init"),
                    "GP": _gp(settings),
                    "Kernel": _kernel(settings),
                    "Kernel nu": NOT_APPLICABLE if non_gp else _kernel_nu(settings),
                    "ARD": _ard(settings),
                    "Initial lengthscale": (
                        NOT_APPLICABLE
                        if non_gp
                        else _first(settings, "initial_lengthscale")
                    ),
                    "Noise": _noise(settings, experiment),
                    "Hyperparameter fitting": (
                        NOT_APPLICABLE
                        if non_gp
                        else _first(settings, "hyperparameter_fitting")
                    ),
                    "Acquisition": _acquisition(settings),
                    "Acquisition parameters": _acquisition_parameters(settings),
                    "Candidate generation": _first(
                        settings,
                        "candidate_generation",
                        "acquisition_sampling",
                        "acquisition_sampling_strategy",
                    ),
                    "Candidate selection": _first(settings, "candidate_selection"),
                    "Acquisition optimizer": _first(settings, "acquisition_optimizer"),
                    "Batch size": _first(settings, "batch_size", "internal_batch_size"),
                    "q_batch": q_batch,
                    "t_seq": t_seq,
                    "AG-VS selection batch": selection_batch,
                    "MCTS settings": _mcts_settings(algorithm, settings),
                    "TuRBO settings": _turbo_settings(algorithm, settings),
                    "Variable-selection settings": _variable_selection(algorithm, settings),
                    "Algorithm seed rule": (
                        f"{BASE_ALGORITHM_SEED} + base_seed_offset + trial_id"
                    ),
                    "Benchmark seed rule": (
                        f"{BASE_BENCHMARK_SEED} + base_seed_offset + trial_id"
                    ),
                    "Special settings": _text(
                        {
                            "benchmark_noise_std": SYNTHETIC_BENCHMARK_NOISE_STD,
                            "benchmark_noise_distribution": "Gaussian(0, 1e-4)",
                            "budget_source": budget_source,
                            "experiment_id": experiment.experiment_id,
                            "optimization_direction": "maximize project score",
                            "settings_source": settings.get("source", NOT_SPECIFIED),
                            "algorithm_specific": algorithm_special,
                        }
                    ),
                }
            )
    return rows


def synthetic_configuration_csv_text(
    algorithms: Iterable[str],
    settings_provider: Callable[[str, str], Mapping[str, object]],
) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=SYNTHETIC_TABLE_COLUMNS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(synthetic_configuration_rows(algorithms, settings_provider))
    return buffer.getvalue()


def write_synthetic_configuration_table(
    output_path: str | Path | None = None,
    algorithms: Iterable[str] = SYNTHETIC_ALGORITHMS,
) -> Path:
    """Write the live runner/registry cross product to a dedicated CSV file."""

    from Experiment.benchmark import runner

    if output_path is None:
        output_path = runner.SYNTHETIC_TABLE_ROOT / "synthetic_configuration_matrix.csv"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        synthetic_configuration_csv_text(
            algorithms,
            runner.algorithm_settings_summary,
        ),
        encoding="utf-8",
    )
    return output_path


def write_synthetic_configuration_tables(
    table_root: str | Path | None = None,
) -> tuple[Path, ...]:
    """Write separate BO- and TuRBO-family configuration matrices."""

    from Experiment.benchmark import runner

    root = runner.SYNTHETIC_TABLE_ROOT if table_root is None else Path(table_root)
    outputs = []
    for family in runner.ALGORITHM_OUTPUT_FAMILIES:
        algorithms = [
            algorithm
            for algorithm in SYNTHETIC_ALGORITHMS
            if runner.algorithm_output_family(algorithm) == family
        ]
        outputs.append(
            write_synthetic_configuration_table(
                root / family / "synthetic_configuration_matrix.csv",
                algorithms=algorithms,
            )
        )
    (root / "synthetic_configuration_matrix.csv").unlink(missing_ok=True)
    return tuple(outputs)



__all__ = [
    "NOT_APPLICABLE",
    "NOT_SPECIFIED",
    "SYNTHETIC_TABLE_COLUMNS",
    "synthetic_configuration_csv_text",
    "synthetic_configuration_rows",
    "write_synthetic_configuration_table",
    "write_synthetic_configuration_tables",
]

def experiment_entrypoint(argv: list[str] | None = None) -> None:
    """Run any registered synthetic or real-life benchmark from one entrypoint."""

    import argparse

    from Experiment.benchmark import runner

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--benchmark", required=True)
    selected, remaining = parser.parse_known_args(argv)
    experiment = get_experiment(selected.benchmark)
    if isinstance(experiment, SyntheticExperiment):
        config = make_synthetic_config(experiment)
        benchmark_root = Path(__file__).resolve().parent / "benchmark"
    else:
        from Experiment.benchmark.real_life.tasks import make_real_life_config

        config = make_real_life_config(
            experiment.task.key,
            experiment.configured_d_eff,
        )
        benchmark_root = Path(__file__).resolve().parent / "benchmark" / "real_life"
    runner.register_benchmark(experiment.key, config)
    runner.run_benchmark(experiment.key, config, benchmark_root, argv=remaining)


def main(argv: list[str] | None = None) -> None:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] == "smoke":
        smoke_entrypoint(arguments[1:])
    else:
        experiment_entrypoint(arguments)


def run_smoke_checks(suite="synthetic", algorithms=None):
    """Exercise optimizer adapters within small budgets, outside study results."""
    import platform
    import torch
    from Experiment.benchmark import runner, mopta
    from Experiment.benchmark.real_life.tasks import make_real_life_config

    if suite not in {"synthetic", "mopta"}:
        raise ValueError(f"Unknown smoke suite {suite!r}")
    torch.set_num_threads(1)
    if suite == "mopta":
        benchmark = mopta.configure_runner()
        config = runner.apply_runner_agvs_batch_policy(make_real_life_config("mopta08", 4))
        roster = mopta.ALGORITHMS
    else:
        benchmark = "smoke_sphere"
        config = dict(bounds=np.tile([-1., 1.], (6, 1)), effective_dim=2,
                      n_init=4, batch_size=3, q_batch=None, t_seq=2,
                      experiment_kind="synthetic", objective="sphere",
                      active_indices=None, n_bad_multiplier=4, n_bad_enabled=True)
        roster = runner.DEFAULT_ALGORITHMS
    selected = ["AG-VS-BO"] if algorithms is None else algorithms
    if not set(selected).issubset(roster):
        raise ValueError(f"Choose algorithms from {tuple(roster)}")
    reports = []
    for algorithm in selected:
        budget = {"GTBO": 60, "HeSBO": 4, "ALEBO": 11,
                  "TuRBO": 31, "LA-MCTS-TuRBO": 41,
                  "LA-MCTS-BO": 24}.get(algorithm, 23)
        objective = (config["objective_factory"](2021) if suite == "mopta"
                     else lambda x: -float(np.square(np.asarray(x)[:2]).sum()))
        try:
            if hasattr(objective, "prepare"):
                objective.prepare()
            recorder = runner.NoisyObjectiveRecorder(
                objective, noise_std=.01 if suite == "synthetic" else 0., seed=2021)
            optimizer = runner.build_optimizer(
                algorithm, config, recorder, 2021, budget,
                benchmark_name=benchmark, device="cpu")
            if algorithm == "GTBO":
                optimizer.settings["gt_iterations"] = 1
            print(f"START {suite} {algorithm} budget={budget}", flush=True)
            runner.run_optimizer(algorithm, optimizer, recorder, max_iter=budget)
            if len(recorder.observed_history) != budget:
                raise AssertionError(f"{algorithm}: expected {budget} evaluations")
            reports.append(dict(algorithm=algorithm, seed=2021, evaluations=budget,
                                status="passed", settings_override=(
                                    {"gt_iterations": 1} if algorithm == "GTBO" else {})))
            print(f"PASS {suite} {algorithm}", flush=True)
        finally:
            if hasattr(objective, "close"):
                objective.close()
    return dict(suite=suite, python=platform.python_version(), trials=reports)


def smoke_entrypoint(argv=None):
    """Run an explicitly requested small check, never the full trial matrix."""
    import argparse
    import json
    parser = argparse.ArgumentParser(description=run_smoke_checks.__doc__)
    parser.add_argument("--suite", choices=("synthetic", "mopta"), default="synthetic")
    parser.add_argument("--algorithms", nargs="+")
    parser.add_argument("--output", type=Path, default=Path(".cache/smoke.json"))
    args = parser.parse_args(argv)
    report = run_smoke_checks(args.suite, args.algorithms)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
