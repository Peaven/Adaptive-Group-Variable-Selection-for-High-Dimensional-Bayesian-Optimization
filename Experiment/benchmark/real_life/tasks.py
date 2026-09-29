"""Dependency-lazy MOPTA08 objective and benchmark configuration."""
from __future__ import annotations
from dataclasses import dataclass
from types import MappingProxyType
import numpy as np
from Experiment.utils import REAL_LIFE_D_EFFS, REAL_LIFE_TASKS, get_real_life_experiment, get_real_life_task
from Experiment.benchmark.real_life.errors import RealLifeBenchmarkError
REAL_LIFE_TRIALS = 25
REAL_LIFE_EFFECTIVE_DIMS = REAL_LIFE_D_EFFS
TASKS = MappingProxyType({key: MappingProxyType({'upstream_name': task.upstream_name, 'dimension': task.physical_dim, 'bounds': task.bounds, 'max_iter': task.budget, 'metric_name': task.metric_name, 'paper_scope': task.paper_scope}) for key, task in REAL_LIFE_TASKS.items()})
METRIC_TRANSFORMS = MappingProxyType({'mopta08': 'maximize negative soft-constrained mass: -(objective + 10 * sum(max(0, constraint_i)))'})

def metric_to_maximize(task: str, raw_value: object) -> float:
    """Put every real-life metric on the runner's maximize convention."""
    task = get_real_life_task(task).key
    value = float(raw_value)
    if not np.isfinite(value):
        raise FloatingPointError(f'{task} returned non-finite objective {value}')
    return value

@dataclass
class MoptaObjective:
    """One seeded physical objective, constructed only on first evaluation."""
    task: str
    seed: int
    dataset_path: object | None = None

    def __post_init__(self):
        try:
            task = get_real_life_task(self.task)
        except KeyError as exc:
            raise ValueError(str(exc)) from exc
        self.task = task.key
        self.seed = int(self.seed)
        self._task_spec = task
        self._objective = None
        self._closed = False
        self._evaluation_index = 0
        self.last_projection = None

    def _build(self):
        from Experiment.benchmark.real_life.mopta08_backend import Mopta08Objective
        return Mopta08Objective()

    def prepare(self):
        """Construct the physical backend without consuming an evaluation.

        The runner calls this before starting its algorithm wall-clock timer so
        dataset deserialization and environment construction are not charged to
        an optimizer.  Candidate/evaluation RNG streams are unchanged because
        preparation does not advance ``_evaluation_index``.
        """
        if self._closed:
            raise RuntimeError(f'Real-life task {self.task!r} is already closed')
        if self._objective is None:
            self._objective = self._build()
        return self

    def _project_candidate(self, candidate: object) -> np.ndarray:
        vector = np.asarray(candidate, dtype=np.float64)
        if vector.ndim != 1:
            raise ValueError(f'{self.task} candidate must be one-dimensional; got shape {vector.shape}')
        if vector.size != int(self._task_spec.physical_dim):
            raise ValueError(f'{self.task} candidate has {vector.size} values; expected physical dimension {self._task_spec.physical_dim}')
        if not np.all(np.isfinite(vector)):
            raise ValueError(f'{self.task} candidate must contain only finite values')
        low, high = self._task_spec.bounds
        vector = np.clip(vector, float(low), float(high))
        return vector

    def __call__(self, x):
        if self._closed:
            raise RuntimeError(f'Real-life task {self.task!r} is already closed')
        raw_candidate = np.asarray(x, dtype=np.float64)
        if raw_candidate.ndim != 1:
            raise ValueError(f'{self.task} candidate must be one-dimensional; got shape {raw_candidate.shape}')
        if raw_candidate.size != int(self._task_spec.physical_dim):
            raise ValueError(f'{self.task} candidate has {raw_candidate.size} values; expected physical dimension {self._task_spec.physical_dim}')
        if self._objective is None:
            self._objective = self._build()
        candidate = self._project_candidate(raw_candidate)
        try:
            value = self._objective(candidate)
        except RealLifeBenchmarkError:
            raise
        except Exception as exc:
            raise RuntimeError(f"{self.task}: official backend rejected projected candidate {getattr(self.last_projection, 'architecture_key', None)!r}: {type(exc).__name__}: {exc}") from exc
        if isinstance(value, tuple):
            value = value[0]
        value = metric_to_maximize(self.task, value)
        self._evaluation_index += 1
        return value

    def close(self):
        if self._closed:
            return
        self._closed = True
        objective = self._objective
        if objective is None:
            return
        close = getattr(objective, 'close', None)
        if callable(close):
            close()

    def runtime_metadata(self):
        metadata = {'task': self.task, 'benchmark_seed': int(self.seed), 'metric_transform': METRIC_TRANSFORMS[self.task]}
        from Experiment.benchmark.real_life.mopta08_backend import MOPTA08_DATASET_DOI, MOPTA08_EXECUTABLE_SHA256
        metadata.update({'backend_family': 'official_mopta08_windows_executable', 'source': 'MOPTA 2008 Jones Benchmark PC archive', 'dataset_doi': MOPTA08_DATASET_DOI, 'executable': str(getattr(self._objective, 'executable', '')), 'executable_sha256': MOPTA08_EXECUTABLE_SHA256, 'physical_dimension': 124, 'normalized_domain': '[0,1]^124', 'constraint_count': 68, 'soft_constraint_penalty': 10.0})
        return metadata

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

@dataclass(frozen=True)
class ObjectiveFactory:
    task: str
    dataset_path: object | None = None

    def __call__(self, seed):
        return MoptaObjective(self.task, int(seed), dataset_path=self.dataset_path)

def _factory_only(_x):
    raise RuntimeError('Real-life objectives must be constructed through objective_factory.')

def real_life_benchmark_name(task, effective_dim):
    try:
        return get_real_life_experiment(task, int(effective_dim)).key
    except KeyError as exc:
        raise ValueError(str(exc)) from exc

def make_real_life_config(task, effective_dim, *, dataset_path: object | None=None):
    """Build a runner config without changing the physical benchmark dimension."""
    try:
        experiment = get_real_life_experiment(task, int(effective_dim))
    except KeyError as exc:
        raise ValueError(str(exc)) from exc
    task_spec = experiment.task
    low, high = task_spec.bounds
    reference_embeddings = dict(task_spec.reference_embedding_dimensions)
    paper_omissions = tuple(task_spec.paper_omitted_algorithms)
    return {
        'func': _factory_only,
        'objective_factory': ObjectiveFactory(task_spec.key, dataset_path),
        'bounds': np.repeat([[float(low), float(high)]], int(task_spec.physical_dim), axis=0),
        'active_indices': None,
        'effective_dim': int(experiment.configured_d_eff),
        'configured_d_eff': int(experiment.configured_d_eff),
        'physical_dim': int(task_spec.physical_dim),
        'effective_dim_role': 'configured_optimizer_search_dimension_not_ground_truth',
        'effective_dimension_seed': None,
        'real_life_task': task_spec.key,
        'real_life_variant': experiment.key,
        'experiment_id': experiment.experiment_id,
        'L': None,
        'experiment_kind': 'real_life',
        'metric_name': str(task_spec.metric_name),
        'metric_transform': METRIC_TRANSFORMS[task_spec.key],
        'dataset_path': None,
        'dataset_identifier': 'MOPTA08 Jones Benchmark; DOI 10.7488/ds/7894',
        'objective_protocol': {'implementation': 'official MOPTA08 Windows executable', 'dimension': 124, 'constraints': 68, 'domain': '[0,1]^124', 'maximize': '-(objective + 10 * sum(max(0, constraint_i)))', 'archive_md5': 'aa10adbdc3eb119d5da643ba70157c36', 'executable_sha256': '35b1d27ca1e91df646612c2d96bd1cb2e435d7c919d898e6343a2de4e6b25c48'},
        'optimization_direction': 'maximize',
        'paper_scope': str(task_spec.paper_scope),
        'upstream_name': str(task_spec.upstream_name),
        'max_iter': int(experiment.budget),
        'trials': int(experiment.trials),
        'paper_reference_trial_seeds': [2021, 2022, 2023, 2024, 2025],
        'protocol_scope': f'{experiment.trials} sequential seeds (2021--{2020 + experiment.trials}) and controlled configured_d_eff variants',
        'experiment_spec': experiment,
        'seed_bundle_factory': experiment.seeds,
        'trial_id_policy': 'zero_based_0_to_N_minus_1',
        'algorithm_seed_policy': 'registry algorithm_seed = 2021 + zero_based trial_id',
        'benchmark_seed_policy': 'registry benchmark_seed = 2021 + zero_based trial_id',
        'rollout_seed_policy': None,
        'nas_dataset_seed_policy': None,
        'nas_dataset_seed_protocol_version': None,
        'num_rollouts': None,
        'mujoco_protocol_track': None,
        'reference_environment_id': None,
        'environment_version_policy': None,
        'candidate_projection': 'clip_to_[0,1]^124+official_MOPTA08_executable',
        'nasbench101_edge_policy': None,
        'duplicate_architecture_policy': None,
        'reference_embedding_dimensions': reference_embeddings,
        'reference_embedding_exception': 'Reference baseline dimensions override configured d_eff for the listed algorithms only; configured d_eff remains the experiment identity.' if reference_embeddings else None,
        'paper_omitted_algorithms': paper_omissions,
        'reference_mcts_cp': 50.0,
        'q_batch': experiment.q_batch,
        't_seq': int(experiment.t_seq),
        'selection_batch_size': experiment.selection_batch_size,
        'batch_size': experiment.selection_batch_size,
        'upstream_repository': 'https://github.com/gtboauthors/gtbo',
        'wall_clock_timing_policy': 'exclude_backend_initialization; start_immediately_before_optimizer_construction',
    }
__all__ = ['METRIC_TRANSFORMS', 'ObjectiveFactory', 'MoptaObjective', 'REAL_LIFE_EFFECTIVE_DIMS', 'REAL_LIFE_TRIALS', 'TASKS', 'make_real_life_config', 'metric_to_maximize', 'real_life_benchmark_name']
