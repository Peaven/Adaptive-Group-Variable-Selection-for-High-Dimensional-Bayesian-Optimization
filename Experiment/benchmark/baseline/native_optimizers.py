from __future__ import annotations

import contextlib
import functools
import importlib.util
import os
import random
import sys
import types
import time
import webbrowser
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
from scipy.stats import norm
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
import warnings

from Experiment.benchmark.baseline.reference_infrastructure import (
    BO_REFERENCE_SPEC,
    TURBO_REFERENCE_SPEC,
    bo_reference_dict,
    classical_expected_improvement,
    make_reference_bo_gp,
    sklearn_learned_noise_variance,
    mcts_cp_for_benchmark,
    sample_reference_bo_candidates,
    turbo_reference_dict,
)


BASELINE_ROOT = Path(__file__).resolve().parent


class _PosteriorStopSignal(RuntimeError):
    """Internal control-flow signal raised only at a completed batch boundary."""


class _EvaluationBudgetStopSignal(RuntimeError):
    """Internal signal preventing a vendored batch loop from overshooting budget."""


class NativeRandomSearchOptimizer:
    """Import-safe adapter for MCTS-VS ``baseline/random_search.py``."""

    def __init__(
        self,
        objective,
        bounds,
        max_iter,
        seed=None,
        noise_variance=1e-4,
        risk_tolerances=None,
        stopping_threshold=None,
        enable_posterior_stopping=False,
        stopping_min_points=1,
        **_,
    ):
        self.objective = objective
        self.bounds = np.asarray(bounds, dtype=float)
        self.max_iter = int(max_iter)
        self.rng = np.random.default_rng(seed)
        self.X = np.empty((0, self.bounds.shape[0]), dtype=float)
        self.y = np.empty((0,), dtype=float)
        self.history = []
        self.noise_variance = max(float(noise_variance), 1e-12)
        self.risk_tolerances = [0.05] if risk_tolerances is None else [float(x) for x in risk_tolerances]
        self.stopping_threshold = None if stopping_threshold is None else float(stopping_threshold)
        self.enable_posterior_stopping = bool(enable_posterior_stopping)
        if self.enable_posterior_stopping:
            raise ValueError("Random Search has no GP posterior; posterior stopping is not defined")
        self.stopping_min_points = max(1, int(stopping_min_points))
        self.threshold_trace = []
        self.best_stoppable_threshold_by_delta = {
            _risk_key(delta): -float("inf") for delta in self.risk_tolerances
        }
        self.stopped = False
        self.stop_iteration = None

    def optimize(self):
        points, values = [], []
        while len(values) < self.max_iter:
            x = self.rng.uniform(self.bounds[:, 0], self.bounds[:, 1])
            y = float(self.objective(x))
            points.append(x.copy())
            values.append(y)
            self.X = np.asarray(points, dtype=float)
            self.y = np.asarray(values, dtype=float)
            stopping_info = None
            if self.enable_posterior_stopping and len(values) >= self.stopping_min_points:
                stopping_info = _posterior_threshold_snapshot(
                    self.X,
                    self.y,
                    noise_variance=self.noise_variance,
                    risk_tolerances=self.risk_tolerances,
                    best_by_delta=self.best_stoppable_threshold_by_delta,
                    use_ard=False,
                    candidate_indices=[len(values) - 1],
                    stopping_threshold=self.stopping_threshold,
                )
                if stopping_info is not None:
                    self.threshold_trace.append(stopping_info)
                    self.stopped = bool(stopping_info.get("stopped", False))
                    if self.stopped:
                        self.stop_iteration = int(len(values))
            self.history.append({
                "n": len(values),
                "best_y": float(np.max(values)),
                "stopping_info": stopping_info,
            })
            if self.stopped:
                break
        return self


class OfficialCMAESOptimizer:
    """Adapter for the exact CMA-ES settings in MCTS-VS ``baseline/cmaes.py``."""

    def __init__(
        self,
        objective,
        bounds,
        max_iter,
        initial_points=None,
        seed=None,
        sigma=0.8,
        population_size=20,
        noise_variance=1e-4,
        risk_tolerances=None,
        stopping_threshold=None,
        enable_posterior_stopping=False,
        stopping_min_points=3,
        **_,
    ):
        self.objective = objective
        self.bounds = np.asarray(bounds, dtype=float)
        self.max_iter = int(max_iter)
        self.initial_points = None if initial_points is None else np.asarray(initial_points, dtype=float)
        self.seed = seed
        self.sigma = float(sigma)
        self.population_size = int(population_size)
        self.X = np.empty((0, self.bounds.shape[0]), dtype=float)
        self.y = np.empty((0,), dtype=float)
        self.history = []
        self.noise_variance = max(float(noise_variance), 1e-12)
        self.risk_tolerances = [0.05] if risk_tolerances is None else [float(x) for x in risk_tolerances]
        self.stopping_threshold = None if stopping_threshold is None else float(stopping_threshold)
        self.enable_posterior_stopping = bool(enable_posterior_stopping)
        if self.enable_posterior_stopping:
            raise ValueError("CMA-ES has no GP posterior; posterior stopping is not defined")
        self.stopping_min_points = max(2, int(stopping_min_points))
        self.threshold_trace = []
        self.best_stoppable_threshold_by_delta = {
            _risk_key(delta): -float("inf") for delta in self.risk_tolerances
        }
        self.stopped = False
        self.stop_iteration = None

    def optimize(self):
        try:
            import cma
        except ImportError as exc:
            raise ImportError(
                "Reference CMA-ES requires `cma==3.1.0` in the pinned historical "
                "environment."
            ) from exc
        cma_version = str(getattr(cma, "__version__", "unknown"))
        if cma_version != "3.1.0":
            raise RuntimeError(
                "Reference CMA-ES requires exactly `cma==3.1.0`; found "
                f"{cma_version}. Use the pinned Python/NumPy environment in "
                "Experiment/benchmark/baseline/REFERENCE_ENVIRONMENTS.md."
            )
        rng = np.random.default_rng(self.seed)
        x0 = (
            self.initial_points[0].copy()
            if self.initial_points is not None and len(self.initial_points)
            else rng.uniform(self.bounds[:, 0], self.bounds[:, 1])
        )
        options = {
            "popsize": self.population_size,
            "bounds": [self.bounds[:, 0].tolist(), self.bounds[:, 1].tolist()],
            "seed": self.seed,
            "verbose": -9,
        }
        strategy = cma.CMAEvolutionStrategy(x0, self.sigma, options)
        points, values = [], []
        while len(values) < self.max_iter:
            batch = strategy.ask()
            remaining = self.max_iter - len(values)
            evaluated = [np.clip(np.asarray(x, dtype=float), self.bounds[:, 0], self.bounds[:, 1]) for x in batch[:remaining]]
            observed = [float(self.objective(x)) for x in evaluated]
            # The official script minimizes -f with CMA-ES.
            strategy.tell(evaluated, [-value for value in observed])
            points.extend(x.copy() for x in evaluated)
            values.extend(observed)
            self.X = np.asarray(points, dtype=float)
            self.y = np.asarray(values, dtype=float)
            stopping_info = None
            if self.enable_posterior_stopping and len(values) >= self.stopping_min_points:
                batch_start = len(values) - len(observed)
                stopping_info = _posterior_threshold_snapshot(
                    self.X,
                    self.y,
                    noise_variance=self.noise_variance,
                    risk_tolerances=self.risk_tolerances,
                    best_by_delta=self.best_stoppable_threshold_by_delta,
                    use_ard=False,
                    candidate_indices=list(range(batch_start, len(values))),
                    stopping_threshold=self.stopping_threshold,
                )
                if stopping_info is not None:
                    self.threshold_trace.append(stopping_info)
                    self.stopped = bool(stopping_info.get("stopped", False))
                    if self.stopped:
                        self.stop_iteration = int(len(values))
            for index in range(len(observed)):
                n = len(values) - len(observed) + index + 1
                self.history.append({
                    "n": n,
                    "best_y": float(np.max(values[:n])),
                    "stopping_info": stopping_info if index == len(observed) - 1 else None,
                })
            if self.stopped:
                break
        return self


def _load_native_turbo1():
    from .native.turbo.turbo_1 import Turbo1
    return Turbo1


def _load_native_mcts():
    from .native.mctsvs.MCTS import MCTS
    return MCTS


def _risk_key(value: float) -> str:
    return f"{float(value):.12g}"


def _fixed_noise_sklearn_gp(
    noise_variance: float,
    dim: Optional[int] = None,
    *,
    learn_noise: bool = False,
) -> GaussianProcessRegressor:
    return make_reference_bo_gp(
        alpha=max(float(noise_variance), 1e-12),
        learn_noise=learn_noise,
        dim=dim,
    )


from ag_vs.utils import isolated_trial as _isolates_legacy_rng, seeded_trial as _isolated_legacy_rng


def _posterior_threshold_trace(
    X: np.ndarray,
    y: np.ndarray,
    noise_variance: float,
    risk_tolerances,
    batch_size: int,
    min_points: int,
    use_ard: bool = True,
    verbose: bool = False,
    learn_noise: bool = False,
) -> list[dict[str, Any]]:
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    if X.size == 0 or y.size == 0:
        return []

    risk_tolerances = [float(delta) for delta in risk_tolerances]
    best_by_delta = {_risk_key(delta): -float("inf") for delta in risk_tolerances}
    trace: list[dict[str, Any]] = []
    batch_size = max(1, int(batch_size))
    min_points = max(2, int(min_points))
    endpoints = list(range(min_points, len(y) + 1, batch_size))
    if endpoints and endpoints[-1] != len(y):
        endpoints.append(len(y))
    if not endpoints and len(y) >= min_points:
        endpoints = [len(y)]

    for trace_index, n in enumerate(endpoints, start=1):
        if verbose:
            print(
                f"[posterior-trace] n={int(n)}/{len(y)} "
                f"endpoint={trace_index}/{len(endpoints)} dim={X.shape[1]}",
                flush=True,
            )
        Xn = X[:n]
        yn = y[:n]
        query_start = max(0, int(n) - int(batch_size))
        X_query = Xn[query_start:n]
        try:
            gp = _fixed_noise_sklearn_gp(
                noise_variance,
                X.shape[1] if use_ard else None,
                learn_noise=learn_noise,
            )
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=ConvergenceWarning)
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                gp.fit(Xn, yn)
            mean, std = gp.predict(X_query, return_std=True)
        except Exception:
            continue

        mean = np.asarray(mean, dtype=float).reshape(-1)
        std = np.maximum(np.asarray(std, dtype=float).reshape(-1), 1e-12)
        current = {}
        best = {}
        threshold_argmax_by_delta = {}
        for delta in risk_tolerances:
            key = _risk_key(delta)
            threshold_values = mean + std * norm.ppf(delta)
            j = int(np.argmax(threshold_values))
            threshold = float(threshold_values[j])
            threshold_argmax_by_delta[key] = int(query_start + j)
            current[key] = threshold
            best_by_delta[key] = max(float(best_by_delta[key]), threshold)
            best[key] = float(best_by_delta[key])
        first_key = _risk_key(risk_tolerances[0])
        center_idx = int(threshold_argmax_by_delta.get(first_key, query_start + int(np.argmax(mean))))
        center_local = int(center_idx - query_start)
        effective_noise_variance = sklearn_learned_noise_variance(
            gp, fallback=noise_variance
        )
        trace.append(
            {
                "n": int(n),
                "posterior_mean_at_best": float(mean[center_local]),
                "posterior_std_at_best": float(std[center_local]),
                "best_observed_value": float(np.max(yn)),
                "best_index": center_idx,
                "posterior_candidate_indices": list(range(query_start, int(n))),
                "threshold_argmax_index_by_delta": threshold_argmax_by_delta,
                "current_stoppable_threshold_by_delta": current,
                "best_stoppable_threshold_by_delta": best,
                "noise_variance": float(effective_noise_variance),
                "learned_noise_variance": (
                    float(effective_noise_variance) if learn_noise else None
                ),
                "noise_model": (
                    "learned from observations by marginal likelihood"
                    if learn_noise
                    else "fixed known variance; not learned from observations"
                ),
                "posterior_center_rule": "current batch points evaluated by GP; threshold is max over batch points",
            }
        )
    return trace


def _posterior_threshold_snapshot(
    X: np.ndarray,
    y: np.ndarray,
    noise_variance: float,
    risk_tolerances,
    best_by_delta: dict[str, float],
    use_ard: bool = True,
    candidate_indices: Optional[list[int]] = None,
    stopping_threshold: Optional[float] = None,
    stopping_delta: Optional[float] = None,
    stopping_direction: str = "max",
    learn_noise: bool = False,
) -> Optional[dict[str, Any]]:
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    if X.size == 0 or y.size == 0:
        return None

    gp = _fixed_noise_sklearn_gp(
        noise_variance,
        X.shape[1] if use_ard else None,
        learn_noise=learn_noise,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        gp.fit(X, y)
    if candidate_indices is None:
        candidate_indices = list(range(len(y)))
    candidate_indices = [int(idx) for idx in candidate_indices if 0 <= int(idx) < len(y)]
    if not candidate_indices:
        candidate_indices = list(range(len(y)))
    X_query = X[candidate_indices]
    mean, std = gp.predict(X_query, return_std=True)
    mean = np.asarray(mean, dtype=float).reshape(-1)
    std = np.maximum(np.asarray(std, dtype=float).reshape(-1), 1e-12)

    current = {}
    best = {}
    threshold_argmax_by_delta = {}
    for delta in risk_tolerances:
        key = _risk_key(delta)
        threshold_values = mean + std * norm.ppf(float(delta))
        j = int(np.argmax(threshold_values))
        threshold = float(threshold_values[j])
        threshold_argmax_by_delta[key] = int(candidate_indices[j])
        current[key] = threshold
        best_by_delta[key] = max(float(best_by_delta.get(key, -float("inf"))), threshold)
        best[key] = float(best_by_delta[key])

    first_key = _risk_key(risk_tolerances[0])
    center_idx = int(threshold_argmax_by_delta.get(first_key, candidate_indices[int(np.argmax(mean))]))
    center_local = candidate_indices.index(center_idx) if center_idx in candidate_indices else int(np.argmax(mean))
    stop_probability = None
    stop_index = None
    required_stop_probability = None
    stopped = False
    if stopping_threshold is not None:
        delta = float(risk_tolerances[0] if stopping_delta is None else stopping_delta)
        required_stop_probability = 1.0 - delta
        if str(stopping_direction).strip().lower() == "min":
            probabilities = norm.cdf((float(stopping_threshold) - mean) / std)
        else:
            probabilities = norm.sf((float(stopping_threshold) - mean) / std)
        local_stop_index = int(np.argmax(probabilities))
        stop_probability = float(probabilities[local_stop_index])
        stop_index = int(candidate_indices[local_stop_index])
        stopped = bool(stop_probability >= required_stop_probability)
    effective_noise_variance = sklearn_learned_noise_variance(
        gp, fallback=noise_variance
    )
    return {
        "n": int(len(y)),
        "posterior_mean_at_best": float(mean[center_local]),
        "posterior_std_at_best": float(std[center_local]),
        "best_observed_value": float(np.max(y)),
        "best_index": center_idx,
        "posterior_candidate_indices": list(candidate_indices),
        "threshold_argmax_index_by_delta": threshold_argmax_by_delta,
        "current_stoppable_threshold_by_delta": current,
        "best_stoppable_threshold_by_delta": best,
        "noise_variance": float(effective_noise_variance),
        "learned_noise_variance": (
            float(effective_noise_variance) if learn_noise else None
        ),
        "noise_model": (
            "learned from observations by marginal likelihood"
            if learn_noise
            else "fixed known variance; not learned from observations"
        ),
        "stop_probability": stop_probability,
        "required_stop_probability": required_stop_probability,
        "stopping_threshold": None if stopping_threshold is None else float(stopping_threshold),
        "stop_index": stop_index,
        "stopped": stopped,
        "posterior_center_rule": "current batch points evaluated by GP; threshold is max over batch points",
    }


class NativeTurbo1Optimizer:
    """Thin adapter around uber-research/TuRBO's official TuRBO-1 code.

    The upstream implementation minimizes, while the local benchmark scores are
    maximized. This adapter passes ``-f(x)`` upstream and exposes maximization
    scores through ``y``.
    """

    def __init__(
        self,
        f: Callable[[np.ndarray], float],
        lb,
        ub,
        n_init: int = 20,
        max_evals: int = 500,
        batch_size: int = 10,
        verbose: bool = False,
        use_ard: bool = True,
        max_cholesky_size: int = 2000,
        n_training_steps: int = 50,
        min_cuda: int = 1024,
        device: str = "cpu",
        dtype: str = TURBO_REFERENCE_SPEC.dtype,
        seed: Optional[int] = None,
        initial_points: Optional[np.ndarray] = None,
        noise_variance: float = BO_REFERENCE_SPEC.alpha,
        learn_noise: bool = False,
        risk_tolerances=None,
        stopping_threshold: Optional[float] = None,
        enable_posterior_stopping: bool = True,
        **_: Any,
    ) -> None:
        self.f = f
        self.lb = np.asarray(lb, dtype=float)
        self.ub = np.asarray(ub, dtype=float)
        self.dim = int(self.lb.size)
        self.max_evals = int(max_evals)
        self.batch_size = max(1, int(batch_size))
        self.n_init = max(1, int(n_init))
        if self.n_init >= self.max_evals:
            self.n_init = max(1, self.max_evals - self.batch_size)
        self.verbose = bool(verbose)
        self.seed = seed
        if initial_points is None:
            self.initial_points = None
        else:
            initial_points = np.asarray(initial_points, dtype=float)
            if initial_points.ndim != 2 or initial_points.shape[1] != self.dim:
                raise ValueError("initial_points must have shape (n_points, dim)")
            self.initial_points = np.clip(initial_points, self.lb, self.ub)
        self.noise_variance = max(float(noise_variance), 1e-12)
        self.learn_noise = bool(learn_noise)
        self.risk_tolerances = [0.05] if risk_tolerances is None else [float(x) for x in risk_tolerances]
        self.stopping_threshold = None if stopping_threshold is None else float(stopping_threshold)
        self.enable_posterior_stopping = bool(enable_posterior_stopping)

        self._eval_counter = 0
        Turbo1 = _load_native_turbo1()
        self._turbo = Turbo1(
            f=self._evaluate_for_official_turbo,
            lb=self.lb,
            ub=self.ub,
            n_init=self.n_init,
            max_evals=self.max_evals,
            batch_size=self.batch_size,
            verbose=self.verbose,
            use_ard=bool(use_ard),
            max_cholesky_size=int(max_cholesky_size),
            n_training_steps=int(n_training_steps),
            min_cuda=int(min_cuda),
            device=device,
            dtype=dtype,
        )
        self.turbo_infrastructure = turbo_reference_dict(device=device)
        self.turbo_infrastructure["turbo_max_evals"] = int(self.max_evals)

        self.X = np.zeros((0, self.dim), dtype=float)
        self.y = np.zeros((0,), dtype=float)
        self.best_x = None
        self.best_y = -np.inf
        self.history = []
        self.threshold_trace = []
        self.best_stoppable_threshold_by_delta = {
            _risk_key(delta): -float("inf") for delta in self.risk_tolerances
        }
        self.stopped = False
        self.stop_iteration = None

    def _evaluate_for_official_turbo(self, x):
        value = -float(self.f(np.asarray(x, dtype=float)))
        self._eval_counter += 1
        if self.verbose and (
            self._eval_counter == self.n_init
            or self._eval_counter >= self.max_evals
            or (
                self._eval_counter > self.n_init
                and (self._eval_counter - self.n_init) % self.batch_size == 0
            )
        ):
            print(f"[TuRBO] eval {self._eval_counter}/{self.max_evals}", flush=True)
        return value

    @_isolates_legacy_rng
    def optimize(self):
        self._eval_counter = 0
        self.threshold_trace = []
        self.stopped = False
        self.stop_iteration = None
        self.best_stoppable_threshold_by_delta = {
            _risk_key(delta): -float("inf") for delta in self.risk_tolerances
        }

        self._optimize_turbo_with_batch_threshold_trace()
        self.X = np.asarray(self._turbo.X, dtype=float).copy()
        self.y = -np.asarray(self._turbo.fX, dtype=float).reshape(-1)
        if self.y.size:
            best_idx = int(np.argmax(self.y))
            self.best_y = float(self.y[best_idx])
            self.best_x = self.X[best_idx].copy()
        self.history = [
            {
                "n": int(i + 1),
                "best_y": float(np.max(self.y[: i + 1])),
                "trust_region_length": None,
            }
            for i in range(len(self.y))
        ]
        return self

    def _record_turbo_threshold_snapshot(self, candidate_indices=None):
        """Record thresholds from TuRBO's already-fitted native GPyTorch GP."""

        if not self.enable_posterior_stopping:
            return None

        X = np.asarray(self._turbo.X, dtype=float)
        y = -np.asarray(self._turbo.fX, dtype=float).reshape(-1)
        if len(y) < self.n_init:
            return None
        gp = getattr(self._turbo, "_last_native_gp", None)
        if gp is None:
            return None
        if candidate_indices is None:
            candidate_indices = list(range(len(y)))
        candidate_indices = [
            int(index)
            for index in candidate_indices
            if 0 <= int(index) < len(y)
        ]
        if not candidate_indices:
            return None

        import gpytorch
        import torch
        import warnings

        target_median = float(self._turbo._last_native_target_median)
        target_scale = float(self._turbo._last_native_target_scale)
        X_query = np.clip(
            (X[candidate_indices] - self.lb) / (self.ub - self.lb),
            0.0,
            1.0,
        )
        parameter = next(gp.parameters())
        with warnings.catch_warnings():
            # Reporting intentionally evaluates points from the GP training set.
            warnings.filterwarnings(
                "ignore",
                message="The input matches the stored training data.*",
            )
            with torch.no_grad(), gpytorch.settings.max_cholesky_size(
                self._turbo.max_cholesky_size
            ):
                query = torch.as_tensor(
                    X_query,
                    device=parameter.device,
                    dtype=parameter.dtype,
                )
                posterior = gp.likelihood(gp(query))
            # Native TuRBO models the standardized minimization objective -f.
            # Convert its posterior back to the experiment's maximization scale.
            mean = -(
                target_median
                + target_scale * posterior.mean.detach().cpu().numpy()
            )
            std = target_scale * (
                posterior.variance.clamp_min(1e-24).sqrt().detach().cpu().numpy()
            )
        mean = np.asarray(mean, dtype=float).reshape(-1)
        std = np.maximum(np.asarray(std, dtype=float).reshape(-1), 1e-12)

        current = {}
        best = {}
        threshold_argmax_by_delta = {}
        for delta in self.risk_tolerances:
            key = _risk_key(delta)
            threshold_values = mean + std * norm.ppf(float(delta))
            local_index = int(np.argmax(threshold_values))
            threshold = float(threshold_values[local_index])
            threshold_argmax_by_delta[key] = int(candidate_indices[local_index])
            current[key] = threshold
            self.best_stoppable_threshold_by_delta[key] = max(
                float(self.best_stoppable_threshold_by_delta.get(key, -float("inf"))),
                threshold,
            )
            best[key] = float(self.best_stoppable_threshold_by_delta[key])

        first_key = _risk_key(self.risk_tolerances[0])
        center_idx = int(threshold_argmax_by_delta[first_key])
        center_local = candidate_indices.index(center_idx)
        stop_probability = None
        stop_index = None
        required_stop_probability = None
        stopped = False
        if self.enable_posterior_stopping and self.stopping_threshold is not None:
            required_stop_probability = 1.0 - float(self.risk_tolerances[0])
            probabilities = norm.sf(
                (float(self.stopping_threshold) - mean) / std
            )
            local_stop_index = int(np.argmax(probabilities))
            stop_probability = float(probabilities[local_stop_index])
            stop_index = int(candidate_indices[local_stop_index])
            stopped = bool(stop_probability >= required_stop_probability)

        learned_noise_variance = float(
            gp.likelihood.noise.detach().cpu().reshape(-1)[0]
        ) * (target_scale**2)
        entry = {
            "n": int(len(y)),
            "posterior_mean_at_best": float(mean[center_local]),
            "posterior_std_at_best": float(std[center_local]),
            "best_observed_value": float(np.max(y)),
            "best_index": center_idx,
            "posterior_candidate_indices": list(candidate_indices),
            "threshold_argmax_index_by_delta": threshold_argmax_by_delta,
            "current_stoppable_threshold_by_delta": current,
            "best_stoppable_threshold_by_delta": best,
            "noise_variance": learned_noise_variance,
            "learned_noise_variance": learned_noise_variance,
            "noise_model": "native TuRBO GPyTorch likelihood learned by marginal likelihood",
            "stop_probability": stop_probability,
            "required_stop_probability": required_stop_probability,
            "stopping_threshold": self.stopping_threshold,
            "stop_index": stop_index,
            "stopped": stopped,
            "posterior_center_rule": "current batch points evaluated by the native TuRBO GP",
            "posterior_source": "same native GPyTorch GP fit used for TuRBO candidate generation",
            "gp_device": str(parameter.device),
            "gp_dtype": str(parameter.dtype),
        }
        if entry is not None:
            self.threshold_trace.append(entry)
            self.stopped = bool(entry.get("stopped", False))
            if self.stopped:
                self.stop_iteration = int(entry["n"])
            if self.verbose:
                best_threshold = max(entry["best_stoppable_threshold_by_delta"].values())
                print(
                    f"[TuRBO stopping-trace] n={entry['n']}/{self.max_evals} "
                    f"best_stoppable_threshold={best_threshold:.6g}",
                    flush=True,
                )
        return entry

    def _fit_final_native_turbo_reporting_gp(self):
        """Fit the one terminal native GP not followed by another proposal."""

        turbo = self._turbo
        if len(turbo._fX) == 0:
            return None
        import gpytorch
        import torch

        gp_module = sys.modules.get("Experiment.benchmark.baseline.native.turbo.gp")
        if gp_module is None:
            _load_native_turbo1()
            gp_module = sys.modules["Experiment.benchmark.baseline.native.turbo.gp"]
        utils_module = sys.modules["Experiment.benchmark.baseline.native.turbo.utils"]
        X_unit = utils_module.to_unit_cube(
            np.asarray(turbo._X, dtype=float), turbo.lb, turbo.ub
        )
        fX = np.asarray(turbo._fX, dtype=float).reshape(-1)
        target_median = float(np.median(fX))
        target_scale = float(np.std(fX))
        if target_scale < 1e-6:
            target_scale = 1.0
        standardized = (fX - target_median) / target_scale
        if len(X_unit) < turbo.min_cuda:
            device, dtype = torch.device("cpu"), torch.float64
        else:
            device, dtype = turbo.device, turbo.dtype
        with gpytorch.settings.max_cholesky_size(turbo.max_cholesky_size):
            gp = gp_module.train_gp(
                train_x=torch.as_tensor(X_unit, device=device, dtype=dtype),
                train_y=torch.as_tensor(standardized, device=device, dtype=dtype),
                use_ard=turbo.use_ard,
                num_steps=turbo.n_training_steps,
                hypers={},
            )
        turbo._last_native_gp = gp
        turbo._last_native_target_median = target_median
        turbo._last_native_target_scale = target_scale
        turbo._last_native_training_size = int(len(X_unit))
        return gp

    def _optimize_turbo_with_batch_threshold_trace(self):
        turbo = self._turbo
        utils_module = sys.modules.get("Experiment.benchmark.baseline.native.turbo.utils")
        if utils_module is None:
            _load_native_turbo1()
            utils_module = sys.modules["Experiment.benchmark.baseline.native.turbo.utils"]
        latin_hypercube = utils_module.latin_hypercube
        from_unit_cube = utils_module.from_unit_cube
        to_unit_cube = utils_module.to_unit_cube

        while turbo.n_evals < turbo.max_evals:
            if len(turbo._fX) > 0 and turbo.verbose:
                n_evals, fbest = turbo.n_evals, turbo._fX.min()
                print(f"{n_evals}) Restarting with fbest = {fbest:.4}")
                sys.stdout.flush()

            turbo._restart()

            remaining = int(turbo.max_evals - turbo.n_evals)
            if remaining <= 0:
                break
            init_count = min(int(turbo.n_init), remaining)
            if (
                self.initial_points is not None
                and turbo.n_evals == 0
                and len(self.initial_points) >= init_count
            ):
                X_init = self.initial_points[:init_count].copy()
            else:
                X_init = latin_hypercube(init_count, turbo.dim)
                X_init = from_unit_cube(X_init, turbo.lb, turbo.ub)
            fX_init = np.array([[turbo.f(x)] for x in X_init])

            turbo.n_evals += init_count
            turbo._X = deepcopy(X_init)
            turbo._fX = deepcopy(fX_init)
            turbo.X = np.vstack((turbo.X, deepcopy(X_init)))
            turbo.fX = np.vstack((turbo.fX, deepcopy(fX_init)))

            if turbo.verbose:
                fbest = turbo._fX.min()
                print(f"Starting from fbest = {fbest:.4}")
                sys.stdout.flush()

            pending_threshold_indices = list(
                range(len(turbo.fX) - init_count, len(turbo.fX))
            )

            while turbo.n_evals < turbo.max_evals and turbo.length >= turbo.length_min and not self.stopped:
                X = to_unit_cube(deepcopy(turbo._X), turbo.lb, turbo.ub)
                fX = deepcopy(turbo._fX).ravel()

                X_cand, y_cand, _ = turbo._create_candidates(
                    X,
                    fX,
                    length=turbo.length,
                    n_training_steps=turbo.n_training_steps,
                    hypers={},
                )
                self._record_turbo_threshold_snapshot(
                    candidate_indices=pending_threshold_indices
                )
                if self.stopped:
                    break
                X_next = turbo._select_candidates(X_cand, y_cand)
                remaining = int(turbo.max_evals - turbo.n_evals)
                X_next = X_next[: min(int(turbo.batch_size), remaining)]
                X_next = from_unit_cube(X_next, turbo.lb, turbo.ub)
                fX_next = np.array([[turbo.f(x)] for x in X_next])

                turbo._adjust_length(fX_next)
                turbo.n_evals += int(len(X_next))
                turbo._X = np.vstack((turbo._X, X_next))
                turbo._fX = np.vstack((turbo._fX, fX_next))

                if turbo.verbose and fX_next.min() < turbo.fX.min():
                    n_evals, fbest = turbo.n_evals, fX_next.min()
                    print(f"{n_evals}) New best: {fbest:.4}")
                    sys.stdout.flush()

                turbo.X = np.vstack((turbo.X, deepcopy(X_next)))
                turbo.fX = np.vstack((turbo.fX, deepcopy(fX_next)))
                pending_threshold_indices = list(
                    range(len(turbo.fX) - len(X_next), len(turbo.fX))
                )

            last_recorded_n = (
                int(self.threshold_trace[-1].get("n", 0))
                if self.threshold_trace
                else 0
            )
            if (
                self.enable_posterior_stopping
                and not self.stopped
                and last_recorded_n < len(turbo.fX)
            ):
                self._fit_final_native_turbo_reporting_gp()
                self._record_turbo_threshold_snapshot(
                    candidate_indices=pending_threshold_indices
                )


class NativeMCTSVSOptimizer:
    """Thin adapter around lamda-bbo/MCTS-VS official MCTS implementation."""

    def __init__(
        self,
        f: Optional[Callable[[np.ndarray], float]] = None,
        lb=None,
        ub=None,
        feature_batch_size: int = 2,
        sample_batch_size: int = 3,
        min_num_variables: int = 3,
        select_right_threshold: int = 5,
        cp: Optional[float] = None,
        Cp: Optional[float] = None,
        k: int = 20,
        seed: Optional[int] = None,
        ipt_solver: str = "bo",
        uipt_solver: str = "bestk",
        turbo_max_evals: int = 50,
        split_type: str = "mean",
        max_iter: Optional[int] = None,
        noise_variance: float = BO_REFERENCE_SPEC.alpha,
        learn_noise: bool = False,
        turbo_device: str = TURBO_REFERENCE_SPEC.device,
        turbo_min_cuda: int = TURBO_REFERENCE_SPEC.min_cuda,
        benchmark_name: Optional[str] = None,
        risk_tolerances=None,
        stopping_threshold: Optional[float] = None,
        enable_posterior_stopping: bool = True,
        tree_plot_on: bool = False,
        tree_plot_path: Optional[str] = None,
        tree_plot_html_path: Optional[str] = None,
        tree_plot_open_browser: bool = False,
        tree_plot_pause: float = 0.001,
        active_indices=None,
        initial_points: Optional[np.ndarray] = None,
        **_: Any,
    ) -> None:
        self.f = f
        self.lb = np.asarray(lb, dtype=float)
        self.ub = np.asarray(ub, dtype=float)
        self.dim = int(self.lb.size)
        self.feature_batch_size = int(feature_batch_size)
        self.sample_batch_size = int(sample_batch_size)
        self.min_num_variables = int(min_num_variables)
        self.select_right_threshold = int(select_right_threshold)
        explicit_cp = Cp if Cp is not None else cp
        self.cp = float(
            mcts_cp_for_benchmark(benchmark_name)
            if explicit_cp is None
            else explicit_cp
        )
        self.cp_source = (
            "Appendix Table 3 benchmark mapping"
            if explicit_cp is None
            else "explicit adapter argument"
        )
        self.k = int(k)
        self.seed = seed
        self.ipt_solver = str(ipt_solver)
        self.uipt_solver = str(uipt_solver)
        self.turbo_max_evals = int(turbo_max_evals)
        self.split_type = str(split_type)
        self.max_iter = None if max_iter is None else int(max_iter)
        self.noise_variance = max(float(noise_variance), 1e-12)
        self.learn_noise = bool(learn_noise)
        if self.ipt_solver == "bo" and not np.isclose(
            self.noise_variance, BO_REFERENCE_SPEC.alpha
        ):
            raise ValueError("MCTS-VS fair BO infrastructure requires GP alpha=1e-4.")
        self.rng = np.random.default_rng(self.seed)
        self.turbo_device = str(turbo_device).strip().lower()
        self.turbo_min_cuda = int(turbo_min_cuda)
        if self.turbo_min_cuda < 0:
            raise ValueError("turbo_min_cuda must be non-negative.")
        if self.turbo_device not in {"cpu", "cuda"}:
            raise ValueError("turbo_device must be 'cpu' or 'cuda'.")
        if self.turbo_device == "cuda":
            try:
                import torch
            except ImportError as exc:
                raise ImportError("MCTS-VS-TuRBO CUDA requires PyTorch with CUDA support.") from exc
            if not torch.cuda.is_available():
                raise ValueError("turbo_device='cuda' requested, but CUDA is unavailable.")
        self.bo_infrastructure = bo_reference_dict() if self.ipt_solver == "bo" else None
        self.turbo_infrastructure = turbo_reference_dict(device=self.turbo_device)
        self.turbo_infrastructure["min_cuda"] = int(self.turbo_min_cuda)
        if self.turbo_device == "cuda" and self.turbo_min_cuda == 0:
            self.turbo_infrastructure["small_data_runtime"] = (
                "CUDA float32 from the first local GP fit (min_cuda=0)"
            )
        self.turbo_infrastructure["turbo_max_evals"] = int(self.turbo_max_evals)
        self.risk_tolerances = [0.05] if risk_tolerances is None else [float(x) for x in risk_tolerances]
        self.stopping_threshold = None if stopping_threshold is None else float(stopping_threshold)
        self.enable_posterior_stopping = bool(enable_posterior_stopping)
        self.tree_plot_on = bool(tree_plot_on)
        self.tree_plot_path = tree_plot_path
        self.tree_plot_html_path = tree_plot_html_path
        self.tree_plot_open_browser = bool(tree_plot_open_browser)
        self.tree_plot_pause = float(tree_plot_pause)
        self.active_indices = set() if active_indices is None else {int(dim) for dim in active_indices}
        if initial_points is None:
            self.initial_points = None
        else:
            initial_points = np.asarray(initial_points, dtype=float)
            if initial_points.ndim != 2 or initial_points.shape[1] != self.dim:
                raise ValueError("initial_points must have shape (n_points, dim)")
            self.initial_points = np.clip(initial_points, self.lb, self.ub)
        self._tree_plot_fig = None
        self._tree_plot_ax = None
        self._tree_plot_browser_opened = False

        self._agent = None
        self.X = np.zeros((0, self.dim), dtype=float)
        self.y = np.zeros((0,), dtype=float)
        self.best_x = None
        self.best_y = -np.inf
        self.trace = []
        self.threshold_trace = []
        self.best_stoppable_threshold_by_delta = {
            _risk_key(delta): -float("inf") for delta in self.risk_tolerances
        }
        self._mcts_eval_counter = 0
        self._mcts_eval_budget = None
        self._mcts_live_X = []
        self._mcts_live_y = []
        self.value_trace = []
        self.selected_variables = []
        self.stopped = False
        self.stop_iteration = None

    def _format_mcts_plot_value(self, value: float, digits: int = 4) -> str:
        value = float(value)
        if np.isposinf(value):
            return "inf"
        if np.isneginf(value):
            return "-inf"
        if not np.isfinite(value):
            return "nan"
        return f"{value:.{digits}g}"

    def _mcts_tree_nodes(self, agent) -> list:
        nodes_by_id = {}
        for node in list(getattr(agent, "nodes", []) or []):
            nodes_by_id[int(getattr(node, "id", len(nodes_by_id)))] = node
            for kid in list(getattr(node, "kids", []) or []):
                nodes_by_id[int(getattr(kid, "id", len(nodes_by_id)))] = kid
        return sorted(nodes_by_id.values(), key=lambda node: int(getattr(node, "id", 0)))

    def _mcts_tree_positions(self, root) -> dict:
        positions = {}
        leaf_counter = 0
        x_spacing = 7.0
        y_spacing = 4.4

        def visit(node, depth):
            nonlocal leaf_counter
            kids = list(getattr(node, "kids", []) or [])
            if not kids:
                x = float(leaf_counter) * x_spacing
                leaf_counter += 1
            else:
                child_x = [visit(kid, depth + 1) for kid in kids]
                x = float(np.mean(child_x)) if child_x else float(leaf_counter)
            positions[int(getattr(node, "id", len(positions)))] = (x, -float(depth) * y_spacing)
            return x

        visit(root, 0)
        return positions

    def _write_mcts_tree_plot_html(self, iteration) -> None:
        if not self.tree_plot_path or not self.tree_plot_html_path:
            return
        html_path = Path(self.tree_plot_html_path).expanduser().resolve()
        image_path = Path(self.tree_plot_path).expanduser().resolve()
        html_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            image_src = os.path.relpath(str(image_path), start=str(html_path.parent)).replace("\\", "/")
        except ValueError:
            image_src = str(image_path)
        title_iter = "init" if iteration == "init" else f"iteration {iteration}"
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        html_path.write_text(
            f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>MCTS-VS Live Tree</title>
  <style>
    body {{ margin: 0; font-family: Arial, sans-serif; background: #f6f7f9; color: #111827; }}
    header {{ padding: 10px 14px; background: #ffffff; border-bottom: 1px solid #d8dde6; }}
    h1 {{ margin: 0; font-size: 16px; }}
    p {{ margin: 4px 0 0; font-size: 12px; color: #4b5563; }}
    img {{ display: block; width: min(100vw, 1800px); height: auto; margin: 0 auto; }}
  </style>
  <script>
    function refreshImage() {{
      const img = document.getElementById("tree");
      img.src = "{image_src}?t=" + Date.now();
    }}
    window.setInterval(refreshImage, 400);
  </script>
</head>
<body>
  <header>
    <h1>MCTS-VS Live Tree - {title_iter}</h1>
    <p>Last update: {stamp}. The tree image refreshes after each saved iteration.</p>
  </header>
  <img id="tree" src="{image_src}?t={time.time():.6f}" alt="MCTS-VS live tree">
</body>
</html>
""",
            encoding="utf-8",
        )
        if self.tree_plot_open_browser and not self._tree_plot_browser_opened:
            try:
                webbrowser.open(f"file://{html_path}", new=1, autoraise=True)
                self._tree_plot_browser_opened = True
            except Exception:
                pass

    def _plot_mcts_tree(self, agent, iteration="init") -> None:
        if not self.tree_plot_on:
            return
        root = getattr(agent, "ROOT", None)
        if root is None:
            return
        try:
            import matplotlib.pyplot as plt
        except Exception:
            return

        if self._tree_plot_fig is None or self._tree_plot_ax is None:
            self._tree_plot_fig, self._tree_plot_ax = plt.subplots(figsize=(38, 22))
            try:
                plt.ion()
                self._tree_plot_fig.show()
            except Exception:
                pass

        ax = self._tree_plot_ax
        ax.clear()
        nodes = self._mcts_tree_nodes(agent)
        positions = self._mcts_tree_positions(root)
        current = getattr(agent, "CURT", None)
        current_id = None if current is None else int(getattr(current, "id", -1))

        for node in nodes:
            node_id = int(getattr(node, "id", -1))
            if node_id not in positions:
                continue
            x, y = positions[node_id]
            for kid in list(getattr(node, "kids", []) or []):
                kid_id = int(getattr(kid, "id", -1))
                if kid_id not in positions:
                    continue
                x1, y1 = positions[kid_id]
                ax.plot([x, x1], [y, y1], color="#8a8f98", linewidth=1.2, zorder=1)

        for node in nodes:
            node_id = int(getattr(node, "id", -1))
            if node_id not in positions:
                continue
            x, y = positions[node_id]
            active_dims = [int(dim) for dim in list(getattr(node, "active_dims_idx", []) or [])]
            has_true_active = any(dim in self.active_indices for dim in active_dims)
            is_leaf = len(list(getattr(node, "kids", []) or [])) == 0
            is_current = node_id == current_id
            face = "#22c55e" if has_true_active else ("#d8e8ff" if is_leaf else "#ead7ff")
            edge = "#d62728" if is_current else "#1f2937"
            linewidth = 3.2 if is_current else 1.2
            ax.scatter([x], [y], s=2100 if is_current else 1700, c=face, edgecolors=edge, linewidths=linewidth, zorder=3)

            values = [float(y_value) for _, y_value in list(getattr(node, "samples", []) or [])]
            max_value = max(values) if values else float("-inf")
            try:
                uct = node.get_uct(self.cp)
            except Exception:
                uct = float("nan")
            shown_dims = ",".join(str(dim) for dim in active_dims[:8])
            if len(active_dims) > 8:
                shown_dims += ",..."
            label_lines = [
                f"{'CURT ' if is_current else ''}node{node_id}",
                f"n={int(getattr(node, 'n', 0))}, samples={len(values)}",
                f"max={self._format_mcts_plot_value(max_value)}",
                f"uct={self._format_mcts_plot_value(uct)}",
                f"Dim={len(active_dims)}",
            ]
            if shown_dims:
                label_lines.append(f"dims: {shown_dims}")
            ax.text(x, y, "\n".join(label_lines), ha="center", va="center", fontsize=7, color="#111827", zorder=4)

        title_iter = "init" if iteration == "init" else f"iteration {iteration}"
        ax.set_title(f"MCTS-VS tree - {title_iter} | nodes={len(nodes)}", fontsize=12)
        ax.axis("off")
        ax.margins(x=0.35, y=0.35)
        self._tree_plot_fig.tight_layout()

        if self.tree_plot_path:
            output_path = Path(self.tree_plot_path).expanduser().resolve()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            self._tree_plot_fig.savefig(output_path, dpi=160)
            self._write_mcts_tree_plot_html(iteration)

        try:
            self._tree_plot_fig.canvas.draw()
            self._tree_plot_fig.canvas.flush_events()
            plt.pause(self.tree_plot_pause)
        except Exception:
            pass

    def _record_mcts_threshold_snapshot(self, candidate_indices=None):
        if not self.enable_posterior_stopping:
            return None
        if len(self._mcts_live_y) < max(2, self.feature_batch_size * 2 * self.sample_batch_size):
            return None
        X = np.vstack(self._mcts_live_X)
        y = np.asarray(self._mcts_live_y, dtype=float)
        entry = _posterior_threshold_snapshot(
            X,
            y,
            noise_variance=self.noise_variance,
            risk_tolerances=self.risk_tolerances,
            best_by_delta=self.best_stoppable_threshold_by_delta,
            use_ard=False,
            candidate_indices=candidate_indices,
            stopping_threshold=(self.stopping_threshold if self.enable_posterior_stopping else None),
            learn_noise=self.learn_noise,
        )
        if entry is not None:
            self.threshold_trace.append(entry)
            self.stopped = bool(entry.get("stopped", False))
            if self.stopped:
                self.stop_iteration = int(entry["n"])
        return entry

    def _evaluate_for_official_mcts(self, x):
        if (
            self._mcts_eval_budget is not None
            and self._mcts_eval_counter >= int(self._mcts_eval_budget)
        ):
            raise _EvaluationBudgetStopSignal(
                "MCTS-VS reached the requested scalar objective-call budget"
            )
        value = float(self.f(np.asarray(x, dtype=float)))
        self._mcts_eval_counter += 1
        self._mcts_live_X.append(np.asarray(x, dtype=float).reshape(1, -1))
        self._mcts_live_y.append(value)
        if self._mcts_eval_counter % max(1, self.sample_batch_size) == 0:
            end = len(self._mcts_live_y)
            start = max(0, end - int(self.sample_batch_size))
            entry = self._record_mcts_threshold_snapshot(candidate_indices=list(range(start, end)))
            if entry is not None and bool(entry.get("stopped", False)):
                raise _PosteriorStopSignal("MCTS-VS posterior stopping condition reached")
        return value

    @_isolates_legacy_rng
    def optimize(self, max_samples=None, verbose=False, black_box_function=None):
        if black_box_function is not None:
            self.f = black_box_function
        if self.f is None:
            raise ValueError("Objective function is required for official MCTS-VS.")
        max_samples = int(max_samples if max_samples is not None else self.max_iter)
        if max_samples <= 0:
            raise ValueError("max_samples must be positive.")

        self.rng = np.random.default_rng(self.seed)
        self.threshold_trace = []
        self.best_stoppable_threshold_by_delta = {
            _risk_key(delta): -float("inf") for delta in self.risk_tolerances
        }
        self._mcts_eval_counter = 0
        self._mcts_eval_budget = int(max_samples)
        self._mcts_live_X = []
        self._mcts_live_y = []
        self.stopped = False
        self.stop_iteration = None
        MCTS = _load_native_mcts()
        def candidates(dims, gpr, X_sample, Y_sample, n, lb, ub):
            bounds = np.column_stack([lb, ub])
            points = sample_reference_bo_candidates(bounds, rng=self.rng)
            values = classical_expected_improvement(gpr, X_sample, points)
            indices = np.argsort(values)[-int(n):]
            return points[indices], values[indices]

        try:
            self._agent = MCTS(
                func=self._evaluate_for_official_mcts, dims=self.dim,
                lb=self.lb, ub=self.ub, feature_batch_size=self.feature_batch_size,
                sample_batch_size=self.sample_batch_size, Cp=self.cp,
                min_num_variables=self.min_num_variables,
                select_right_threshold=self.select_right_threshold, k=self.k,
                split_type=self.split_type, ipt_solver=self.ipt_solver,
                uipt_solver=self.uipt_solver, turbo_max_evals=self.turbo_max_evals,
                gp_factory=lambda: _fixed_noise_sklearn_gp(
                    self.noise_variance, dim=None, learn_noise=self.learn_noise),
                candidate_func=candidates, turbo_device=self.turbo_device,
                turbo_min_cuda=self.turbo_min_cuda,
            )
            self._plot_mcts_tree(self._agent, iteration="init")
            self._agent.search(max_samples=max_samples, verbose=bool(verbose))
        except (_PosteriorStopSignal, _EvaluationBudgetStopSignal):
            pass

        # The callback records complete sample batches as they arrive.  When
        # max_samples is not divisible by sample_batch_size, the budget guard
        # stops the upstream loop before another callback can flush the final
        # partial batch, so record that endpoint explicitly.
        final_n = len(self._mcts_live_y)
        last_snapshot_n = (
            int(self.threshold_trace[-1].get("n", 0))
            if self.threshold_trace
            else 0
        )
        if (
            self.enable_posterior_stopping
            and final_n == int(max_samples)
            and final_n % max(1, self.sample_batch_size) != 0
            and last_snapshot_n != final_n
        ):
            partial_size = final_n % max(1, self.sample_batch_size)
            self._record_mcts_threshold_snapshot(
                candidate_indices=list(range(final_n - partial_size, final_n))
            )

        self._plot_mcts_tree(self._agent, iteration=len(self._mcts_live_y))

        samples = list(getattr(self._agent, "samples", []))
        if self._mcts_live_y:
            self.X = np.vstack(self._mcts_live_X)
            self.y = np.asarray(self._mcts_live_y, dtype=float)
            best_idx = int(np.argmax(self.y))
            self.best_x = self.X[best_idx].copy()
            self.best_y = float(self.y[best_idx])
        elif samples:
            self.X = np.vstack([np.asarray(x, dtype=float).reshape(1, -1) for x, _ in samples])
            self.y = np.asarray([float(y) for _, y in samples], dtype=float)
            best_idx = int(np.argmax(self.y))
            self.best_x = self.X[best_idx].copy()
            self.best_y = float(self.y[best_idx])
        self.value_trace = list(getattr(self._agent, "value_trace", []))
        self.selected_variables = list(getattr(self._agent, "selected_variables", []))
        self.trace = [
            {"n": int(n), "best_y": float(best)}
            for n, best in self.value_trace
        ]
        return self


# Thin public variants live beside the implementation so the runner has one
# authoritative MCTS-VS adapter module instead of four forwarding files.
class MCTSVSBOOptimizer(NativeMCTSVSOptimizer):
    def __init__(self, *args, **kwargs):
        kwargs["ipt_solver"] = "bo"
        super().__init__(*args, **kwargs)


class MCTSVSTuRBOOptimizer(NativeMCTSVSOptimizer):
    def __init__(self, *args, **kwargs):
        kwargs["ipt_solver"] = "turbo"
        super().__init__(*args, **kwargs)


class MCTSVSRandomSearchOptimizer(NativeMCTSVSOptimizer):
    def __init__(self, *args, **kwargs):
        kwargs.update(
            ipt_solver="rs",
            enable_posterior_stopping=False,
            stopping_threshold=None,
        )
        super().__init__(*args, **kwargs)


class MCTSVSSAASBOOptimizer(NativeMCTSVSOptimizer):
    def __init__(self, *args, **kwargs):
        kwargs["ipt_solver"] = "saasbo"
        super().__init__(*args, **kwargs)
