from __future__ import annotations

from statistics import NormalDist
from typing import Dict, List, Optional, Tuple

import numpy as np

from ag_vs.utils import Node
from Experiment.benchmark.baseline.reference_infrastructure import (
    sklearn_learned_noise_variance,
)

_STANDARD_NORMAL = NormalDist()


class EarlyStopMechanism:
    def __init__(
        self,
        *,
        tau: Optional[float],
        risk_tolerances: List[float],
        n_init: int,
        noise_std: float,
        learn_noise: bool = False,
    ) -> None:
        self.tau = tau
        self.risk_tolerances = [float(value) for value in risk_tolerances]
        self.n_init = int(n_init)
        self.noise_std = float(noise_std)
        self.learn_noise = bool(learn_noise)
        self.best_threshold_by_delta = {
            f"{float(delta):.12g}": -float("inf") for delta in self.risk_tolerances
        }
        self.trace: List[Dict] = []

    @staticmethod
    def candidate_indices(X: List[np.ndarray], y: List[float], candidate_indices: Optional[List[int]] = None) -> List[int]:
        if candidate_indices is not None:
            valid = [int(idx) for idx in candidate_indices if 0 <= int(idx) < len(X)]
            if valid:
                return valid
        if len(y) == 0:
            return []
        return [int(np.argmax(np.asarray(y, dtype=float)))]

    def stop_probability(
        self,
        *,
        node: Node,
        X: List[np.ndarray],
        y: List[float],
        candidate_indices: Optional[List[int]] = None,
    ) -> Tuple[float, float, float, Optional[int]]:
        if self.tau is None or node.model is None or len(y) == 0:
            return 0.0, float("nan"), float("nan"), None
        query_indices = self.candidate_indices(X, y, candidate_indices)
        if not query_indices:
            return 0.0, float("nan"), float("nan"), None
        x_query = np.asarray(X, dtype=float)[query_indices][:, node.dims]
        mu, std = node.model.predict(x_query, return_std=True)
        mu = np.asarray(mu, dtype=float).reshape(-1)
        std = np.maximum(np.asarray(std, dtype=float).reshape(-1), 1e-12)
        z = (float(self.tau) - mu) / std
        probs = np.array([
            1.0 - _STANDARD_NORMAL.cdf(float(value))
            for value in z
        ])
        j = int(np.argmax(probs))
        return float(probs[j]), float(mu[j]), float(std[j]), int(query_indices[j])

    def record_thresholds(
        self,
        *,
        node: Node,
        X: List[np.ndarray],
        y: List[float],
        candidate_indices: Optional[List[int]] = None,
    ) -> Optional[Dict]:
        if node.model is None or len(y) == 0 or not self.risk_tolerances:
            return None
        if len(y) <= self.n_init:
            return None
        query_indices = self.candidate_indices(X, y, candidate_indices)
        if not query_indices:
            return None
        x_query = np.asarray(X, dtype=float)[query_indices][:, node.dims]
        mu, std = node.model.predict(x_query, return_std=True)
        mu = np.asarray(mu, dtype=float).reshape(-1)
        std = np.maximum(np.asarray(std, dtype=float).reshape(-1), 1e-12)

        current = {}
        best = {}
        threshold_argmax_by_delta = {}
        for delta in self.risk_tolerances:
            key = f"{float(delta):.12g}"
            threshold_values = mu + std * float(_STANDARD_NORMAL.inv_cdf(float(delta)))
            j = int(np.argmax(threshold_values))
            threshold = float(threshold_values[j])
            threshold_argmax_by_delta[key] = int(query_indices[j])
            current[key] = threshold
            best_threshold = max(float(self.best_threshold_by_delta.get(key, -float("inf"))), threshold)
            self.best_threshold_by_delta[key] = best_threshold
            best[key] = best_threshold

        first_delta_key = f"{float(self.risk_tolerances[0]):.12g}"
        center_idx = int(threshold_argmax_by_delta.get(first_delta_key, query_indices[int(np.argmax(mu))]))
        center_local = query_indices.index(center_idx) if center_idx in query_indices else int(np.argmax(mu))
        learned_noise_variance = None
        if self.learn_noise:
            learned_noise_variance = getattr(node.model, "noise_variance", None)
            if learned_noise_variance is None:
                learned_noise_variance = sklearn_learned_noise_variance(node.model)
        effective_noise_variance = (
            float(learned_noise_variance)
            if learned_noise_variance is not None
            else float(self.noise_std) ** 2
        )
        entry = {
            "n": int(len(y)),
            "node": (int(node.h), int(node.idx)),
            "posterior_mean_at_best": float(mu[center_local]),
            "posterior_std_at_best": float(std[center_local]),
            "best_observed_value": float(np.max(np.asarray(y, dtype=float))),
            "best_index": center_idx,
            "posterior_candidate_indices": list(query_indices),
            "threshold_argmax_index_by_delta": threshold_argmax_by_delta,
            "current_stoppable_threshold_by_delta": current,
            "best_stoppable_threshold_by_delta": best,
            "noise_variance": effective_noise_variance,
            "learned_noise_variance": (
                effective_noise_variance if self.learn_noise else None
            ),
            "noise_model": (
                "learned from observations by marginal likelihood"
                if self.learn_noise
                else "fixed known variance; not learned from observations"
            ),
            "posterior_center_rule": "current batch points evaluated by selected-node GP; threshold is max over batch points",
        }
        self.trace.append(entry)
        return entry
