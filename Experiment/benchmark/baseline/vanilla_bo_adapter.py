from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional
import math
import warnings

import numpy as np
from scipy.stats import norm
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor

from Experiment.benchmark.baseline.reference_infrastructure import (
    BO_REFERENCE_SPEC,
    bo_reference_dict,
    classical_expected_improvement,
    make_reference_bo_gp,
    sample_reference_bo_candidates,
)

def latin_hypercube_sample(bounds, n_points, rng=None):
    """Reference baseline LHS using only the optimizer-owned generator."""

    if rng is None:
        raise ValueError(
            "Reference Vanilla BO requires a caller-owned numpy.random.Generator."
        )
    bounds = np.asarray(bounds, dtype=float)
    n_points = int(n_points)
    dimension = int(bounds.shape[0])
    if n_points <= 0:
        return np.empty((0, dimension), dtype=float)
    centers = (1.0 + 2.0 * np.arange(n_points, dtype=float)) / float(2 * n_points)
    unit = np.empty((n_points, dimension), dtype=float)
    for index in range(dimension):
        unit[:, index] = centers[rng.permutation(n_points)]
    unit += rng.uniform(-1.0, 1.0, size=(n_points, dimension)) / float(2 * n_points)
    return bounds[:, 0] + unit * (bounds[:, 1] - bounds[:, 0])


@dataclass
class VanillaBOResult:
    best_x: np.ndarray
    best_y: float
    X: np.ndarray
    y: np.ndarray
    history: list[dict[str, Any]] = field(default_factory=list)
    dimension_report_path: Optional[str] = None
    stopping_results: dict[str, dict[str, dict[str, Any]]] = field(default_factory=dict)
    stopped: bool = False
    returned_posterior_mean: Optional[float] = None
    best_observed_y: Optional[float] = None


class VanillaBOOptimizer:
    """Vanilla full-dimensional BO with known-noise GP posterior stopping."""

    def __init__(
        self,
        objective: Callable[[np.ndarray], float],
        bounds,
        max_iter: int = 500,
        n_init: int = 20,
        q_batch: int = 1,
        acquisition: str = "ei",
        noise_variance: float = BO_REFERENCE_SPEC.alpha,
        candidate_count: int = BO_REFERENCE_SPEC.candidate_count,
        beta: float = 2.0,
        stopping_threshold: Optional[float] = None,
        stopping_delta: float = 0.05,
        tau_values: Optional[list[float] | np.ndarray] = None,
        delta_values: Optional[list[float] | np.ndarray] = None,
        enable_posterior_stopping: bool = True,
        stopping_min_points: Optional[int] = None,
        random_state: Optional[int] = None,
        active_indices: Optional[list[int] | np.ndarray] = None,
        dimension_report_path: Optional[str | Path] = None,
        benchmark_name: Optional[str] = None,
        trial: Optional[int] = None,
        initial_points: Optional[np.ndarray] = None,
        verbose: bool = True,
    ):
        self.objective = objective
        self.bounds = np.asarray(bounds, dtype=float)
        if self.bounds.ndim != 2 or self.bounds.shape[1] != 2:
            raise ValueError("bounds must have shape (dim, 2).")
        if np.any(self.bounds[:, 1] <= self.bounds[:, 0]):
            raise ValueError("Each upper bound must be greater than lower bound.")

        self.dim = int(self.bounds.shape[0])
        self.max_iter = int(max_iter)
        self.n_init = int(n_init)
        self.q_batch = max(1, int(q_batch))
        self.acquisition = str(acquisition).lower()
        self.noise_variance = max(float(noise_variance), 1e-12)
        self.candidate_count = int(candidate_count)
        del beta
        if self.acquisition not in {"ei", "expected_improvement"}:
            raise ValueError("Vanilla BO uses classical analytic Expected Improvement only.")
        if not np.isclose(self.noise_variance, BO_REFERENCE_SPEC.alpha):
            raise ValueError("Reference Vanilla BO requires GP alpha=0.01 (assumed noise std=0.1).")
        if self.candidate_count != BO_REFERENCE_SPEC.candidate_count:
            raise ValueError("Reference Vanilla BO requires exactly 1024 acquisition candidates.")
        self.bo_infrastructure = bo_reference_dict()
        del stopping_threshold
        self.stopping_threshold = None
        self.stopping_delta = float(stopping_delta)
        if delta_values is None:
            delta_values = [float(stopping_delta)]
        self.tau_values = [] if tau_values is None else [float(value) for value in np.asarray(tau_values, dtype=float).reshape(-1)]
        self.delta_values = [float(value) for value in np.asarray(delta_values, dtype=float).reshape(-1)]
        self.enable_posterior_stopping = bool(enable_posterior_stopping)
        self.stopping_min_points = int(stopping_min_points or max(self.n_init + 1, 3))
        self.rng = np.random.default_rng(random_state)
        self.active_indices = set(int(x) for x in np.asarray(active_indices if active_indices is not None else [], dtype=int).reshape(-1))
        self.dimension_report_path = Path(dimension_report_path) if dimension_report_path is not None else None
        self.benchmark_name = benchmark_name
        self.trial = trial
        if initial_points is None:
            self.initial_points = latin_hypercube_sample(
                self.bounds,
                self.n_init,
                rng=self.rng,
            )
        else:
            initial_points = np.asarray(initial_points, dtype=float)
            if initial_points.ndim != 2 or initial_points.shape[1] != self.dim:
                raise ValueError("initial_points must have shape (n_points, dim)")
            if len(initial_points) < self.n_init:
                raise ValueError(
                    "initial_points must contain the complete reference initial "
                    f"design of {self.n_init} points."
                )
            self.initial_points = np.clip(
                initial_points[: self.n_init],
                self.bounds[:, 0],
                self.bounds[:, 1],
            )
        self.verbose = bool(verbose)

        self.X = np.empty((0, self.dim), dtype=float)
        self.y = np.empty((0,), dtype=float)
        self.best_x: Optional[np.ndarray] = None
        self.best_y = -float("inf")
        self.history: list[dict[str, Any]] = []
        self.stopped = False
        self.stopping_info: Optional[dict[str, Any]] = None
        self.stopping_results = self._init_stopping_results()
        self.best_stoppable_threshold_by_delta = {
            self._key(delta): -float("inf") for delta in self.delta_values
        }
        self._last_gp: Optional[GaussianProcessRegressor] = None
        self._last_lengthscales: Optional[np.ndarray] = None

        if self.dimension_report_path is not None:
            self.dimension_report_path.parent.mkdir(parents=True, exist_ok=True)
            self.dimension_report_path.write_text(self._report_header(), encoding="utf-8")

    def _report_header(self) -> str:
        active = sorted(self.active_indices)
        return (
            "Vanilla BO top-10 dimension report\n"
            f"benchmark={self.benchmark_name} trial={self.trial}\n"
            f"important/active dimensions={active}\n"
            "The reference GP is non-ARD; per-dimension importance is not defined.\n"
            "ANSI green marks true important dimensions when the viewer supports terminal colors.\n\n"
        )

    @staticmethod
    def _key(value: float) -> str:
        return f"{float(value):.12g}"

    def _init_stopping_results(self) -> dict[str, dict[str, dict[str, Any]]]:
        results: dict[str, dict[str, dict[str, Any]]] = {}
        if not self.tau_values:
            return results
        for tau in self.tau_values:
            tau_key = self._key(tau)
            results[tau_key] = {}
            for delta in self.delta_values:
                delta_key = self._key(delta)
                results[tau_key][delta_key] = {
                    "tau": float(tau),
                    "delta": float(delta),
                    "stopped": False,
                    "success": False,
                    "iterations": None,
                    "optimizer_iteration": None,
                    "probability": None,
                    "recommended_x": None,
                    "recommended_index": None,
                    "recommended_value": None,
                    "true_score_at_stop": None,
                    "selected_coordinate_subset": [],
                    "num_coordinate_switches": 0,
                    "num_turbo_restarts": 0,
                    "global_dataset_size": None,
                    "posterior_mean_at_recommendation": None,
                    "posterior_std_at_recommendation": None,
                }
        return results

    def _fit_gp(self) -> Optional[GaussianProcessRegressor]:
        if self.y.size < 2:
            return None
        gp = make_reference_bo_gp(alpha=self.noise_variance)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            gp.fit(self.X, self.y)
        self._last_gp = gp
        self._last_lengthscales = self._extract_lengthscales(gp)
        return gp

    def _extract_lengthscales(self, gp: GaussianProcessRegressor) -> np.ndarray:
        try:
            values = np.asarray(gp.kernel_.k2.length_scale, dtype=float).reshape(-1)
        except Exception:
            values = np.ones(self.dim, dtype=float)
        if values.size == 1:
            values = np.full(self.dim, float(values[0]), dtype=float)
        if values.size != self.dim:
            values = np.resize(values, self.dim).astype(float)
        return np.clip(values, 1e-12, np.inf)

    def _sample_candidates(self, n_points: int) -> np.ndarray:
        if int(n_points) != BO_REFERENCE_SPEC.candidate_count:
            raise ValueError(
                "Reference Vanilla BO acquisition requires exactly "
                f"{BO_REFERENCE_SPEC.candidate_count} candidates."
            )
        return sample_reference_bo_candidates(
            self.bounds,
            rng=self.rng,
            count=BO_REFERENCE_SPEC.candidate_count,
        )

    def _expected_improvement(self, gp: GaussianProcessRegressor, candidates: np.ndarray) -> np.ndarray:
        return classical_expected_improvement(gp, self.X, candidates)

    def _propose_batch(self, q: int, gp: Optional[GaussianProcessRegressor]) -> np.ndarray:
        if self.y.size < self.n_init or gp is None:
            start = int(self.y.size)
            stop = min(start + int(q), int(self.n_init))
            points = self.initial_points[start:stop].copy()
            if len(points) != int(q):
                raise RuntimeError(
                    "Vanilla BO attempted to mix initial-design and acquisition "
                    "points in one batch."
                )
            return points

        candidates = self._sample_candidates(max(self.candidate_count, q))
        scores = self._expected_improvement(gp, candidates)
        # Match baseline/vanilia_bo.py exactly: retain the top-k indices in the
        # ascending order returned by np.argsort(...)[-k:].
        order = np.argsort(np.asarray(scores, dtype=float))[-int(q) :]
        return candidates[order].copy()

    def _update_best(self) -> None:
        if self.y.size == 0:
            return
        idx = int(np.argmax(self.y))
        self.best_y = float(self.y[idx])
        self.best_x = self.X[idx].copy()

    def _posterior_stopping_snapshot(
        self,
        gp: Optional[GaussianProcessRegressor],
        optimizer_iteration: int,
        candidate_indices: Optional[np.ndarray] = None,
    ) -> Optional[dict[str, Any]]:
        if not self.enable_posterior_stopping:
            return None
        if gp is None or self.y.size < self.stopping_min_points:
            return None
        if candidate_indices is None:
            candidate_indices = np.arange(self.y.size, dtype=int)
        candidate_indices = np.asarray(candidate_indices, dtype=int).reshape(-1)
        candidate_indices = candidate_indices[(candidate_indices >= 0) & (candidate_indices < self.y.size)]
        if candidate_indices.size == 0:
            candidate_indices = np.arange(self.y.size, dtype=int)
        x_query = self.X[candidate_indices]
        mean, std = gp.predict(x_query, return_std=True)
        mean = np.asarray(mean, dtype=float).reshape(-1)
        std = np.maximum(np.asarray(std, dtype=float).reshape(-1), 1e-12)
        probabilities = {}
        current_thresholds = {}
        best_thresholds = {}
        threshold_argmax_by_delta = {}
        newly_stopped = []
        for delta in self.delta_values:
            delta_key = self._key(delta)
            threshold_values = mean + std * float(norm.ppf(float(delta)))
            j = int(np.argmax(threshold_values))
            threshold = float(threshold_values[j])
            threshold_argmax_by_delta[delta_key] = int(candidate_indices[j])
            current_thresholds[delta_key] = threshold
            previous = float(self.best_stoppable_threshold_by_delta.get(delta_key, -float("inf")))
            best_threshold = max(previous, threshold)
            self.best_stoppable_threshold_by_delta[delta_key] = best_threshold
            best_thresholds[delta_key] = best_threshold
        for tau in self.tau_values:
            tau_key = self._key(tau)
            prob_values = norm.sf((float(tau) - mean) / std)
            j_prob = int(np.argmax(prob_values))
            prob = float(prob_values[j_prob])
            probabilities[tau_key] = prob
            for delta in self.delta_values:
                delta_key = self._key(delta)
                result = self.stopping_results[tau_key][delta_key]
                if result["stopped"]:
                    continue
                if prob >= 1.0 - float(delta):
                    result.update(
                        {
                            "stopped": True,
                            "iterations": int(self.y.size),
                            "optimizer_iteration": int(optimizer_iteration),
                            "probability": prob,
                            "recommended_x": self.X[int(candidate_indices[j_prob])].copy(),
                            "recommended_index": int(candidate_indices[j_prob]),
                            "recommended_value": None,
                            "true_score_at_stop": None,
                            "selected_coordinate_subset": [],
                            "num_coordinate_switches": 0,
                            "num_turbo_restarts": 0,
                            "global_dataset_size": int(self.y.size),
                            "posterior_mean_at_recommendation": float(mean[j_prob]),
                            "posterior_std_at_recommendation": float(std[j_prob]),
                        }
                    )
                    newly_stopped.append({"tau": float(tau), "delta": float(delta), "probability": prob})
        first_delta_key = self._key(self.delta_values[0])
        center_idx = int(threshold_argmax_by_delta.get(first_delta_key, candidate_indices[int(np.argmax(mean))]))
        center_local = int(np.where(candidate_indices == center_idx)[0][0]) if np.any(candidate_indices == center_idx) else int(np.argmax(mean))
        primary_tau_key = self._key(self.tau_values[0]) if self.tau_values else None
        primary_stop_probability = None if primary_tau_key is None else probabilities.get(primary_tau_key)
        return {
            "n": int(self.y.size),
            "probabilities_by_tau": probabilities,
            "current_stoppable_threshold_by_delta": current_thresholds,
            "best_stoppable_threshold_by_delta": best_thresholds,
            "newly_stopped": newly_stopped,
            "posterior_mean_at_best": float(mean[center_local]),
            "posterior_std_at_best": float(std[center_local]),
            "best_observed_value": float(np.max(self.y)),
            "best_index": center_idx,
            "best_x": self.X[center_idx].copy(),
            "posterior_candidate_indices": [int(idx) for idx in candidate_indices.tolist()],
            "threshold_argmax_index_by_delta": threshold_argmax_by_delta,
            "noise_variance": float(self.noise_variance),
            "noise_model": "fixed known variance; not learned from observations",
            "stop_probability": primary_stop_probability,
            "required_stop_probability": 1.0 - float(self.delta_values[0]),
            "stopping_threshold": None if not self.tau_values else float(self.tau_values[0]),
            "stopped": bool(newly_stopped),
            "posterior_center_rule": "current batch points evaluated by fixed-noise GP; threshold is max over batch points",
        }

    def _format_dim(self, dim: int, lengthscale: float, rank: int) -> str:
        is_active = dim in self.active_indices
        label = "ACTIVE" if is_active else "inactive"
        text = f"{rank:2d}. dim={dim:3d} lengthscale={lengthscale:.8g} {label}"
        if is_active:
            return f"\033[32m{text}\033[0m"
        return text

    def _append_dimension_report(self, iteration: int, stopping_info: Optional[dict[str, Any]]) -> None:
        if self.dimension_report_path is None:
            return
        if self._last_lengthscales is None:
            body = "GP not fit yet; no lengthscale ranking available."
        else:
            order = np.argsort(self._last_lengthscales)[: min(10, self.dim)]
            body = "\n".join(
                self._format_dim(int(dim), float(self._last_lengthscales[int(dim)]), rank)
                for rank, dim in enumerate(order, start=1)
            )
        stop_text = "None"
        if stopping_info is not None:
            thresholds = stopping_info.get("best_stoppable_threshold_by_delta", {})
            stop_text = (
                "thresholds="
                + ", ".join(f"delta={key}: {value:.6g}" for key, value in thresholds.items())
                if thresholds
                else "None"
            )
        with open(self.dimension_report_path, "a", encoding="utf-8") as handle:
            handle.write(
                f"iteration={iteration} n_eval={self.y.size} best_y={self.best_y:.8g} "
                f"stopping={stop_text}\n{body}\n\n"
            )

    def optimize(self, max_iter: Optional[int] = None, verbose: Optional[bool] = None) -> VanillaBOResult:
        if max_iter is not None:
            self.max_iter = int(max_iter)
        if verbose is not None:
            self.verbose = bool(verbose)

        iteration = 1
        while self.y.size < self.max_iter and not self.stopped:
            remaining_budget = self.max_iter - int(self.y.size)
            remaining_initial = max(0, self.n_init - int(self.y.size))
            if remaining_initial:
                gp_before = None
                q = min(self.q_batch, remaining_budget, remaining_initial)
            else:
                gp_before = self._fit_gp()
                q = min(self.q_batch, remaining_budget)
            X_next = self._propose_batch(q, gp_before)
            y_next = np.asarray([float(self.objective(x)) for x in X_next], dtype=float)
            batch_start = int(self.y.size)
            self.X = np.vstack([self.X, X_next])
            self.y = np.concatenate([self.y, y_next])
            self._update_best()

            gp_after = self._fit_gp() if batch_start >= self.n_init else None
            batch_indices = np.arange(batch_start, int(self.y.size), dtype=int)
            stopping_info = self._posterior_stopping_snapshot(
                gp_after,
                optimizer_iteration=iteration,
                candidate_indices=batch_indices,
            )
            if stopping_info is not None and stopping_info.get("newly_stopped"):
                self.stopping_info = stopping_info
                self.stopped = True

            self._append_dimension_report(iteration, stopping_info)

            record = {
                "iteration": int(iteration),
                "n_eval": int(self.y.size),
                "x_batch": X_next.copy(),
                "y_batch": y_next.copy(),
                "best_y": float(self.best_y),
                "best_x": None if self.best_x is None else self.best_x.copy(),
                "lengthscales": None if self._last_lengthscales is None else self._last_lengthscales.copy(),
                "top10_dimensions": []
                if self._last_lengthscales is None
                else [int(dim) for dim in np.argsort(self._last_lengthscales)[: min(10, self.dim)]],
                "stopping_info": stopping_info,
                "stopped": False,
                "stopping_results": self.stopping_results,
            }
            self.history.append(record)

            if self.verbose:
                thresholds = None if stopping_info is None else stopping_info.get("best_stoppable_threshold_by_delta", {})
                threshold_text = "None" if not thresholds else max(thresholds.values())
                print(
                    f"[VanillaBO] iteration {iteration}/{math.ceil(self.max_iter / self.q_batch)} "
                    f"q={len(y_next)} n={self.y.size} best={self.best_y:.6g} "
                    f"best_stoppable_threshold={threshold_text}",
                    flush=True,
                )
            iteration += 1

        posterior_entries = [
            record.get("stopping_info") for record in self.history
            if isinstance(record, dict) and isinstance(record.get("stopping_info"), dict)
        ]
        posterior_entries = [
            entry for entry in posterior_entries
            if entry.get("posterior_mean_at_best") is not None
        ]
        if posterior_entries:
            returned_entry = posterior_entries[-1] if self.stopped else max(
                posterior_entries, key=lambda entry: float(entry["posterior_mean_at_best"])
            )
            returned_mean = float(returned_entry["posterior_mean_at_best"])
            returned_idx = int(returned_entry.get("best_index", int(np.argmax(self.y))))
            returned_x = self.X[returned_idx].copy()
        else:
            returned_mean = float("nan")
            returned_x = np.asarray(self.best_x, dtype=float) if self.best_x is not None else np.empty((0,), dtype=float)

        return VanillaBOResult(
            best_x=returned_x,
            best_y=returned_mean,
            X=self.X.copy(),
            y=self.y.copy(),
            history=self.history,
            dimension_report_path=None if self.dimension_report_path is None else str(self.dimension_report_path),
            stopping_results=self.stopping_results,
            stopped=bool(self.stopped),
            returned_posterior_mean=returned_mean,
            best_observed_y=float(self.best_y),
        )
