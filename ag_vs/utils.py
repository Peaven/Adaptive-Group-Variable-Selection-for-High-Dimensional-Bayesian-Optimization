from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np

Array = np.ndarray


class SharedObservationIndices:
    """A zero-copy view over the optimizer's global observation indices.

    A live view tracks every global observation. ``frozen`` creates a shared
    prefix plus a small owned suffix, which lets root children inherit the
    restart dataset without materializing ``list(range(n))`` repeatedly.
    """

    def __init__(
        self,
        owner: Any,
        *,
        prefix_start: int = 0,
        prefix_stop: Optional[int] = None,
        indices: Optional[Sequence[int]] = None,
    ) -> None:
        self.owner = owner
        self.prefix_start = max(0, int(prefix_start))
        self.prefix_stop = None if prefix_stop is None else int(prefix_stop)
        self._shared_indices = (
            None
            if indices is None
            else tuple(dict.fromkeys(int(index) for index in indices))
        )
        self._extra: List[int] = []

    @property
    def prefix_length(self) -> int:
        if self._shared_indices is not None:
            return len(self._shared_indices)
        observations = (
            getattr(self.owner, "y")
            if hasattr(self.owner, "y")
            else self.owner
        )
        if self.prefix_stop is None:
            stop = int(len(observations))
        else:
            stop = min(int(self.prefix_stop), int(len(observations)))
        return max(0, stop - self.prefix_start)

    @property
    def is_live(self) -> bool:
        return self._shared_indices is None and self.prefix_stop is None

    @property
    def shared_indices(self) -> Optional[Sequence[int]]:
        return self._shared_indices

    def frozen(self) -> "SharedObservationIndices":
        if self._shared_indices is not None:
            frozen = SharedObservationIndices(
                self.owner,
                indices=self._shared_indices,
            )
            # The immutable base tuple is safe to share across the root and
            # every replacement child; only each view's small suffix is owned.
            frozen._shared_indices = self._shared_indices
        else:
            frozen = SharedObservationIndices(
                self.owner,
                prefix_start=self.prefix_start,
                prefix_stop=self.prefix_start + self.prefix_length,
            )
        frozen._extra = list(self._extra)
        return frozen

    def append(self, index: int) -> None:
        index = int(index)
        if index in self:
            return
        self._extra.append(index)

    def __len__(self) -> int:
        return self.prefix_length + len(self._extra)

    def __iter__(self):
        if self._shared_indices is not None:
            yield from self._shared_indices
        else:
            yield from range(self.prefix_start, self.prefix_start + self.prefix_length)
        yield from self._extra

    def __contains__(self, index: object) -> bool:
        try:
            value = int(index)
        except (TypeError, ValueError, OverflowError):
            return False
        if self._shared_indices is not None:
            return value in self._shared_indices or value in self._extra
        return self.prefix_start <= value < self.prefix_start + self.prefix_length or value in self._extra

    def __getitem__(self, item):
        if isinstance(item, slice):
            return list(self)[item]
        index = int(item)
        if index < 0:
            index += len(self)
        if 0 <= index < self.prefix_length:
            if self._shared_indices is not None:
                return self._shared_indices[index]
            return self.prefix_start + index
        return self._extra[index - self.prefix_length]


def _require_rng(rng: Optional[np.random.Generator]) -> np.random.Generator:
    if rng is None:
        raise ValueError(
            "A caller-owned numpy.random.Generator is required; stochastic "
            "benchmark helpers must not create entropy-seeded RNG state."
        )
    return rng


def latin_hypercube_sample(bounds: Array, n_points: int, rng: Optional[np.random.Generator] = None) -> Array:
    rng = _require_rng(rng)
    bounds = np.asarray(bounds, dtype=float)
    d = bounds.shape[0]
    n_points = int(n_points)
    if n_points <= 0:
        return np.empty((0, d), dtype=float)

    u = np.zeros((n_points, d), dtype=float)
    for j in range(d):
        perm = rng.permutation(n_points)
        u[:, j] = (perm + rng.random(n_points)) / float(n_points)
    return bounds[:, 0] + u * (bounds[:, 1] - bounds[:, 0])


def uniform_random_sample(bounds: Array, n_points: int, rng: Optional[np.random.Generator] = None) -> Array:
    rng = _require_rng(rng)
    bounds = np.asarray(bounds, dtype=float)
    n_points = int(n_points)
    if n_points <= 0:
        return np.empty((0, bounds.shape[0]), dtype=float)
    return rng.uniform(bounds[:, 0], bounds[:, 1], size=(n_points, bounds.shape[0]))


def center_sample(bounds: Array, n_points: int, rng: Optional[np.random.Generator] = None) -> Array:
    del rng
    bounds = np.asarray(bounds, dtype=float)
    n_points = int(n_points)
    if n_points <= 0:
        return np.empty((0, bounds.shape[0]), dtype=float)
    center = 0.5 * (bounds[:, 0] + bounds[:, 1])
    return np.repeat(center.reshape(1, -1), n_points, axis=0)


def grid_sample(bounds: Array, n_points: int, rng: Optional[np.random.Generator] = None) -> Array:
    rng = _require_rng(rng)
    bounds = np.asarray(bounds, dtype=float)
    n_points = int(n_points)
    dim = int(bounds.shape[0])
    if n_points <= 0:
        return np.empty((0, dim), dtype=float)
    levels = max(2, int(np.ceil(float(n_points) ** (1.0 / max(dim, 1)))))
    axes = [np.linspace(bounds[j, 0], bounds[j, 1], levels) for j in range(dim)]
    mesh = np.meshgrid(*axes, indexing="ij")
    points = np.stack([item.reshape(-1) for item in mesh], axis=1)
    if len(points) > n_points:
        indices = rng.choice(len(points), size=n_points, replace=False)
        points = points[np.sort(indices)]
    return np.asarray(points[:n_points], dtype=float)


def halton_sample(bounds: Array, n_points: int, rng: Optional[np.random.Generator] = None) -> Array:
    rng = _require_rng(rng)
    bounds = np.asarray(bounds, dtype=float)
    n_points = int(n_points)
    dim = int(bounds.shape[0])
    if n_points <= 0:
        return np.empty((0, dim), dtype=float)

    primes = []
    candidate = 2
    while len(primes) < dim:
        is_prime = all(candidate % p for p in primes if p * p <= candidate)
        if is_prime:
            primes.append(candidate)
        candidate += 1

    def radical_inverse(index: int, base: int) -> float:
        value = 0.0
        factor = 1.0 / float(base)
        while index > 0:
            value += factor * float(index % base)
            index //= base
            factor /= float(base)
        return value

    start = int(rng.integers(1, 10000))
    unit = np.empty((n_points, dim), dtype=float)
    for i in range(n_points):
        for j, base in enumerate(primes):
            unit[i, j] = radical_inverse(start + i, base)
    return bounds[:, 0] + unit * (bounds[:, 1] - bounds[:, 0])


def sobol_sample(bounds: Array, n_points: int, rng: Optional[np.random.Generator] = None) -> Array:
    """Scrambled Sobol samples using the MCTS-VS engine/seed policy.

    PyTorch is the reference engine.  SciPy is used only as an explicit
    compatibility fallback when PyTorch is absent; the streams are not claimed
    to be identical.
    """
    rng = _require_rng(rng)
    bounds = np.asarray(bounds, dtype=float)
    n_points = int(n_points)
    dim = int(bounds.shape[0])
    if n_points <= 0:
        return np.empty((0, dim), dtype=float)
    # Use the trial-owned NumPy generator to seed a fresh Sobol engine for each
    # acquisition.  This is the historical MCTS-VS/AG-VS benchmark protocol:
    # it is deterministic for a trial while preserving the optimizer's exact
    # advancing RNG sequence across successive acquisitions.
    seed = int(rng.integers(0, 500000))
    try:
        import torch
        from torch.quasirandom import SobolEngine

        engine = SobolEngine(dimension=dim, scramble=True, seed=seed)
        unit = engine.draw(n_points).to(dtype=torch.float64).cpu().numpy()
    except ImportError:
        from scipy.stats import qmc

        engine = qmc.Sobol(d=dim, scramble=True, seed=seed)
        unit = engine.random(n=n_points)
    return bounds[:, 0] + np.asarray(unit, dtype=float) * (bounds[:, 1] - bounds[:, 0])


def sample_initial_points(
    bounds: Array,
    n_points: int,
    rng: Optional[np.random.Generator] = None,
    strategy: str = "lhs",
) -> Array:
    strategy = str(strategy).strip().lower().replace("-", "_")
    if strategy in {"lhs", "latin_hypercube", "latin_hypercube_sample"}:
        return latin_hypercube_sample(bounds, n_points=n_points, rng=rng)
    if strategy in {"uniform", "random", "uniform_random"}:
        return uniform_random_sample(bounds, n_points=n_points, rng=rng)
    if strategy in {"center", "centre", "midpoint"}:
        return center_sample(bounds, n_points=n_points, rng=rng)
    if strategy in {"grid", "cartesian_grid"}:
        return grid_sample(bounds, n_points=n_points, rng=rng)
    if strategy in {"halton", "quasi_random", "qmc"}:
        return halton_sample(bounds, n_points=n_points, rng=rng)
    if strategy in {"sobol", "sobol_scrambled"}:
        return sobol_sample(bounds, n_points=n_points, rng=rng)
    raise ValueError(
        "initial sampling strategy must be one of: "
        "lhs, uniform, center, grid, halton, sobol"
    )


def sample_acquisition_candidates(
    lows: Array,
    highs: Array,
    n_points: int,
    rng: Optional[np.random.Generator] = None,
    strategy: str = "lhs",
) -> Array:
    """Candidate samples used to optimize acquisition functions."""
    bounds = np.column_stack(
        [np.asarray(lows, dtype=float), np.asarray(highs, dtype=float)]
    )
    strategy = str(strategy).strip().lower().replace("-", "_")
    if strategy in {"sobol", "sobol_scrambled"}:
        return sobol_sample(bounds, n_points=int(n_points), rng=rng)
    if strategy in {"lhs", "latin_hypercube", "latin_hypercube_sample"}:
        return latin_hypercube_sample(bounds, n_points=int(n_points), rng=rng)
    if strategy in {"uniform", "random", "uniform_random"}:
        return uniform_random_sample(bounds, n_points=int(n_points), rng=rng)
    if strategy in {"halton", "quasi_random", "qmc"}:
        return halton_sample(bounds, n_points=int(n_points), rng=rng)
    raise ValueError("acquisition sampling strategy must be one of: lhs, sobol, uniform, halton")


def build_node_score_history(
    score_indices: Iterable[int],
    X: Sequence[Array],
    mu_values: Optional[Sequence[float]] = None,
    historical_mu_best_by_index: Optional[Mapping[int, float]] = None,
    n_init: int = 0,
    *,
    observed_values: Optional[Sequence[float]] = None,
    historical_best_observed_by_index: Optional[Mapping[int, float]] = None,
) -> List[Dict]:
    """Build scores using the reference saved before each proposal batch.

    The historical ``best_observed_y_i_minus_1`` field name is retained for
    compatibility; all observations in one batch share its pre-batch value.
    Legacy mu input is also supported.
    """
    node_score_history: List[Dict] = []
    uses_observations = observed_values is not None
    values = np.asarray(
        observed_values if uses_observations else mu_values,
        dtype=float,
    ).reshape(-1)
    historical_best = (
        historical_best_observed_by_index
        if uses_observations
        else historical_mu_best_by_index
    ) or {}
    for evaluation_index in score_indices:
        evaluation_index = int(evaluation_index)
        if evaluation_index < int(n_init) or evaluation_index >= len(X):
            continue
        if evaluation_index >= values.size:
            continue
        value_i = float(values[evaluation_index])
        best_before = historical_best.get(evaluation_index)
        if best_before is None or not np.isfinite(value_i) or not np.isfinite(float(best_before)):
            continue
        record = {
            "evaluation_index": evaluation_index,
            "x_i": np.asarray(X[evaluation_index], dtype=float).copy(),
        }
        if uses_observations:
            record.update(
                {
                    "observed_y_i": value_i,
                    "best_observed_y_i_minus_1": float(best_before),
                }
            )
        else:
            record.update(
                {
                    "mu_i": value_i,
                    "mu_best_i_minus_1": float(best_before),
                }
            )
        node_score_history.append(record)
    return node_score_history


def calculate_score(
    node_score_history: Sequence[Dict],
    epsilon: float = 1e-12,
    abs_mode: bool = True,
) -> float:
    """Calculate an absolute or signed unscaled mean observation/value gap."""
    del epsilon  # Retained only for backwards-compatible callers.
    terms: List[float] = []
    for record in node_score_history:
        if "observed_y_i" in record:
            value_i = float(record["observed_y_i"])
            best_before = float(record["best_observed_y_i_minus_1"])
        else:
            # Backwards compatibility for already serialized score histories.
            value_i = float(record["mu_i"])
            best_before = float(record["mu_best_i_minus_1"])
        if bool(abs_mode):
            terms.append(abs(value_i - best_before))
        else:
            terms.append(value_i - best_before)
    if not terms:
        return float("inf")
    return float(np.mean(np.asarray(terms, dtype=float)))


def weighted_score(score_and_count: Iterable[tuple[float, int]]) -> float:
    """Return the pseudo-code count-weighted parent/merged-node score."""
    weighted_sum = 0.0
    total_count = 0
    for score, count in score_and_count:
        count = int(count)
        score = float(score)
        if count <= 0 or not np.isfinite(score):
            continue
        weighted_sum += score * count
        total_count += count
    return float(weighted_sum / total_count) if total_count else float("inf")


def format_plot_value(value: float, digits: int = 2) -> str:
    value = float(value)
    if np.isposinf(value):
        return "inf"
    if np.isneginf(value):
        return "-inf"
    if not np.isfinite(value):
        return "nan"
    return f"{value:.{digits}g}"


@dataclass
class Node:
    dims: List[int]
    h: int
    idx: int
    parent: Optional["Node"] = None

    local_bounds: Optional[Array] = None
    split_type: str = "root"

    local_indices: List[int] = field(default_factory=list)
    score_indices: List[int] = field(default_factory=list)
    children: List["Node"] = field(default_factory=list)

    selection_score: float = np.inf
    lengthscale_score: float = np.inf
    lengthscale_values: List[float] = field(default_factory=list)
    gp_best_mean: float = -np.inf

    y_best: float = -np.inf

    initial_selection_priority: bool = False
    selected_count: int = 0

    diameter: float = 0.0

    expanded: bool = False
    model: Optional[Any] = None
    lengthscales: Optional[Array] = None
    gp_kernel_name: str = "none"
    gp_kernel_score: float = -np.inf
    lengthscale_gp_kernel_name: str = "none"
    lengthscale_gp_kernel_score: float = -np.inf

    split_ready: bool = False
    split_frozen: bool = False
    plot_archived: bool = False

    # Pseudo-code D_N. Each entry stores x_i, the latest local-GP mu_i, and
    # the immutable global mu_best captured immediately before evaluation i.
    node_score_history: List[Dict] = field(default_factory=list)

    # Pseudo-code state variables e_N, m_N and g_N.
    exclusion_state: int = 0
    merge_disabled: int = 0
    evaluation_completed: int = 0

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0

    @property
    def n(self) -> int:
        return len(self.local_indices)

    @property
    def D_hi(self) -> int:
        return len(self.local_indices)

    @property
    def Dim_hi(self) -> int:
        return len(self.dims)

    @property
    def active_dimensions(self) -> List[int]:
        return self.dims

    @active_dimensions.setter
    def active_dimensions(self, value: List[int]) -> None:
        self.dims = [int(dim) for dim in value]

    @property
    def node_dataset(self) -> List[int]:
        return self.local_indices

    @node_dataset.setter
    def node_dataset(self, value: List[int]) -> None:
        self.local_indices = [int(index) for index in value]

    @property
    def gp_model(self) -> Optional[Any]:
        return self.model

    @gp_model.setter
    def gp_model(self, value: Optional[Any]) -> None:
        self.model = value


@dataclass
class AGVSResult:
    best_x: Array
    best_y: float
    best_observed_y: float
    X: Array
    y: Array
    true_y: Array
    stopped: bool
    stop_iteration: Optional[int]
    history: List[Dict]
    root: Node
    threshold_trace: List[Dict]
    important_dim: Optional[List[int]] = None
    best_pure_x_coordinate: Optional[Array] = None
    returned_posterior_mean: Optional[float] = None
    posterior_mean_history: Optional[List[Optional[float]]] = None
    best_true_y: Optional[float] = None
    posterior_dataset: Optional[List[Dict]] = None


def find_optimum_B_and_cost(
    D: int,
    d_eff: int,
    n_init: int,
    q_batch: Optional[int],
    batch_size: Optional[int],
    t_seq: int,
) -> tuple[int, int]:
    """
    Finds the optimal B (N_split) and expected iteration using a ball-finding simulation 
    modeled on our tree structure until we reach the final terminal children.
    
    Parameters:
    D: Total balls (D)
    d_eff: Heavy balls (d)
    n_init: Initial random points
    q_batch: Joint-acquisition point count, mutually exclusive with batch_size
    batch_size: Independently scored point count, mutually exclusive with q_batch
    t_seq: Number of sequential acquisition/refit stages on the selected node
    """
    if (q_batch is None) == (batch_size is None):
        raise ValueError("Exactly one of q_batch and batch_size must be provided")
    point_batch_size = q_batch if q_batch is not None else batch_size
    node_trial_cost = max(1, int(point_batch_size)) * max(1, int(t_seq))
    
    if d_eff <= 0:
        d_eff = 1
    if D <= d_eff:
        return 2, n_init

    best_B = None
    min_cost = float('inf')

    # N_split groups are evaluated; N_split - d_eff groups are eliminated.
    for B in range(max(2, d_eff + 1), D + 1):
        cost = n_init
        balls = D
        G = B
        eliminate = max(1, G - d_eff)
        survivors = max(1, G - eliminate)
        
        # Traverse the merge path until the final terminal children are created.
        while balls > d_eff:
            # If remaining balls can fit into the groups as single balls (terminal children)
            if balls <= G:
                # We split them into single terminal child nodes (size 1)
                cost += balls * node_trial_cost
                break
                
            q = balls // G
            r = balls % G
            
            # Worst-case balls merged into the next parent node
            balls_next = q * survivors + min(r, survivors)
            
            if balls_next == balls:
                # Cannot divide further, split remaining into terminal child nodes
                cost += balls * node_trial_cost
                break
                
            # Cost to try the G groups in this split step
            cost += G * node_trial_cost
            balls = balls_next

        # Once the d_eff terminal children exist, the real optimizer selects
        # them round-robin. Include one complete initial round: each terminal
        # child receives one full sequential trial block before simulation ends.
        cost += int(d_eff) * node_trial_cost
            
        if cost < min_cost:
            min_cost = cost
            best_B = B

    if best_B is None:
        best_B = 2
        min_cost = n_init
        
    return best_B, int(min_cost)


from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
import random
import numpy as np

_trial = ContextVar("agvs_trial_randomness", default=None)


def trial_seed():
    state = _trial.get()
    return 2021 if state is None else state[0]


def sobol_engine(dimension, *, role="candidates"):
    from torch.quasirandom import SobolEngine
    state = _trial.get()
    if state is None:
        return SobolEngine(dimension, scramble=True, seed=trial_seed())
    key = (role, int(dimension))
    if key not in state[1]:
        state[1][key] = SobolEngine(dimension, scramble=True, seed=state[0])
    return state[1][key]


@contextmanager
def seeded_trial(seed):
    import torch
    seed = 2021 if seed is None else int(seed)
    previous = _trial.get()
    if previous is not None and previous[0] == seed:
        yield
        return
    python_state, numpy_state = random.getstate(), np.random.get_state()
    token = _trial.set((seed, {}))
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    try:
        with torch.random.fork_rng(devices=devices):
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        _trial.reset(token)


def isolated_trial(method):
    @wraps(method)
    def run(self, *args, **kwargs):
        with seeded_trial(self.seed):
            return method(self, *args, **kwargs)
    return run


def jax_key(seed=None, *, role="inference"):
    """Split an advancing JAX stream initialized with the unmodified trial seed."""
    from jax import random as jrandom
    selected = trial_seed() if seed is None else int(seed)
    state = _trial.get()
    if state is None:
        return jrandom.PRNGKey(selected)
    if selected != state[0]:
        raise ValueError("JAX seed must match the active trial seed")
    key = ("jax", role)
    if key not in state[1]:
        state[1][key] = jrandom.PRNGKey(selected)
    state[1][key], draw = jrandom.split(state[1][key])
    return draw


def acquisition_initial_conditions(acquisition, bounds, num_restarts, raw_samples, *, role):
    """BoTorch restart selection with a persistent Sobol proposal stream."""
    from botorch.optim.initializers import gen_batch_initial_conditions
    def generator(n, q, seed):
        if seed != trial_seed():
            raise ValueError("Acquisition seed must match the trial seed")
        unit = sobol_engine(bounds.shape[-1], role=role).draw(n*q).to(bounds)
        return (bounds[0] + unit*(bounds[1]-bounds[0])).reshape(n, q, -1)
    return gen_batch_initial_conditions(acquisition, bounds, q=1,
        num_restarts=num_restarts, raw_samples=raw_samples,
        options={"seed": trial_seed(), "init_batch_limit": 64}, generator=generator)


def cell_identity(configure, value, benchmark, algorithm, trial, budget):
    import os
    from Experiment.benchmark import runner
    from Experiment.utils import make_synthetic_config

    # Plotting visits several cells in one process. Restore the caller's runner
    # controls and worker environment after deriving each cell's exact identity.
    names = [name for name in vars(runner) if name.startswith(("SYNTHETIC_", "AGVS_TURBO_"))]
    names.append("AGVS_SHOW_FINAL_TREE")
    saved = {name: getattr(runner, name) for name in names}
    environment = {k: v for k, v in os.environ.items() if k.startswith("AGVS_")}
    previous_config = runner.BENCHMARKS.get(benchmark)
    try:
        configure(value)
        config = runner.apply_runner_agvs_batch_policy(make_synthetic_config(benchmark))
        runner.BENCHMARKS[benchmark] = config
        return runner.requested_trace_cache_identity(
            config, benchmark, algorithm, trial, budget, device="cpu")
    finally:
        if previous_config is None:
            runner.BENCHMARKS.pop(benchmark, None)
        else:
            runner.BENCHMARKS[benchmark] = previous_config
        for name, setting in saved.items():
            setattr(runner, name, setting)
        for name in list(os.environ):
            if name.startswith("AGVS_") and name not in environment:
                del os.environ[name]
        os.environ.update(environment)
