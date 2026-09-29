"""ADD-GP-UCB adapter based on the upstream kirthevasank/add-gp-bandits implementation.

The original implementation is MATLAB.  This project-owned Python adapter
retains its additive squared-exponential GP, partial decomposition search,
component-wise UCB schedule, and independent DIRECT acquisition solves.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.linalg import cho_factor, cho_solve, orth
from scipy.optimize import direct
from scipy.stats import norm


class ADDGPUCBOptimizer:
    """Additive GP-UCB for box-constrained maximization."""

    OPT_PT_STD_THRESHOLD = 0.002
    MAX_THRESHOLD_EXCEEDS = 5
    BANDWIDTH_LOWER_BOUND = 1e-5
    BANDWIDTH_UPPER_BOUND = 5.0
    DUPLICATE_DISTANCE_THRESHOLD = 1e-10
    DUPLICATE_RETRY_LIMIT = 10_000

    def __init__(
        self,
        objective,
        bounds,
        max_iter,
        initial_points=None,
        seed=None,
        n_init=10,
        group_size=4,
        noise_variance=None,
        num_acq_evals=None,
        hyperparameter_relearn_interval=25,
        decomposition_candidates=5,
        risk_tolerances=None,
        stopping_threshold=None,
        enable_posterior_stopping=True,
    ):
        self.objective = objective
        self.bounds = np.asarray(bounds, dtype=float)
        self.dim = int(self.bounds.shape[0])
        self.max_iter = int(max_iter)
        if self.max_iter < 10:
            raise ValueError("ADD-GP-UCB requires its ten-point reference initial design.")
        self.initial_points = None if initial_points is None else np.asarray(initial_points, dtype=float)
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        if int(n_init) != 10:
            raise ValueError("ADD-GP-UCB reference numInitPts is 10.")
        self.n_init = 10
        if int(group_size) != 4:
            raise ValueError("ADD-GP-UCB reference maximum group size is 4.")
        self.group_size = min(4, self.dim)
        if noise_variance is not None:
            raise ValueError(
                "ADD-GP-UCB derives commonNoise=0.01*std(initial values); "
                "a fixed shared noise variance is not reference-compatible."
            )
        self.noise_variance = None
        expected_acq_evals = min(5000, 100 * self.dim)
        self.num_acq_evals = int(num_acq_evals or expected_acq_evals)
        if self.num_acq_evals != expected_acq_evals:
            raise ValueError(
                f"ADD-GP-UCB reference numDiRectEvals is {expected_acq_evals} for D={self.dim}."
            )
        self.hyperparameter_relearn_interval = max(1, int(hyperparameter_relearn_interval))
        if self.hyperparameter_relearn_interval != 25:
            raise ValueError("ADD-GP-UCB reference hyperparameter relearn interval is 25.")
        self.decomposition_candidates = max(1, int(decomposition_candidates))
        if self.decomposition_candidates != 5:
            raise ValueError("ADD-GP-UCB partialLearn requires exactly five permutation trials.")
        self.risk_tolerances = [0.05] if risk_tolerances is None else [float(x) for x in risk_tolerances]
        self.stopping_threshold = None if stopping_threshold is None else float(stopping_threshold)
        self.enable_posterior_stopping = bool(enable_posterior_stopping)

        self.X = np.empty((0, self.dim), dtype=float)
        self.y = np.empty((0,), dtype=float)
        self.history = []
        self.groups = self._random_partition()
        self.bandwidth = self.BANDWIDTH_UPPER_BOUND
        self.bandwidth_upper_bound = self.BANDWIDTH_UPPER_BOUND
        self.kernel_scale = 1.0
        self._chol = None
        self._alpha = None
        self._low_std_counter = 0
        self.early_relearn_events = []
        self.reference_settings = {
            "kernel": "additive squared-exponential (RBF) components",
            "mean": (
                "combined posterior uses the data mean; individual component "
                "posteriors use zero mean"
            ),
            "common_noise": "0.01 * std(initial values)",
            "num_initial_points": 10,
            "group_size": 4,
            "decomposition_strategy": "partialLearn; five random permutations",
            "hyperparameter_bounds": (
                "bandwidth=[1e-5,5], scale=[0.03,30]*std(initial values)"
            ),
            "translation_detail": (
                "The MATLAB sequence is retained: an optimizer-owned random orthogonal "
                "provisional transform, five optimizer-owned random permutation trials, "
                "then a post-selection DIRECT refit."
            ),
            "hyperparameter_relearn_interval": 25,
            "hyperparameter_relearn_schedule": (
                "every 25 iterations, or after five consecutive selected-point "
                "combined posterior std values below 0.002; early refits set "
                "bandwidth upper bound=max(1e-5, 0.9*current bandwidth)"
            ),
            "acquisition": "independent per-component UCB maximization with DIRECT",
            "beta_schedule": "beta_t = component_dimension * log(2*t) / 5",
            "direct_evaluations": "ceil(min(5000, 100*D) / number_of_groups) per component",
            "duplicate_avoidance": (
                "while min distance/current bandwidth < 1e-10, add "
                "0.1*bandwidth Gaussian perturbations and project to bounds; "
                "a finite retry limit is a defensive guard"
            ),
            "gpu_support": "not explicitly specified; original MATLAB/DIRECT stack is CPU",
        }
        self.threshold_trace = []
        self.best_stoppable_threshold_by_delta = {
            f"{float(delta):.12g}": -float("inf") for delta in self.risk_tolerances
        }
        self.stopped = False
        self.stop_iteration = None

    def _random_partition(self):
        permutation = self.rng.permutation(self.dim)
        return [
            np.sort(permutation[start : start + self.group_size]).astype(int)
            for start in range(0, self.dim, self.group_size)
        ]

    @staticmethod
    def _sqdist(X1, X2):
        X1 = np.asarray(X1, dtype=float)
        X2 = np.asarray(X2, dtype=float)
        return np.maximum(
            np.sum(X1 * X1, axis=1)[:, None]
            + np.sum(X2 * X2, axis=1)[None, :]
            - 2.0 * X1 @ X2.T,
            0.0,
        )

    def _group_kernel(self, X1, X2, group, bandwidth=None, scale=None):
        bandwidth = max(float(self.bandwidth if bandwidth is None else bandwidth), 1e-8)
        scale = max(float(self.kernel_scale if scale is None else scale), 1e-12)
        X1 = np.asarray(X1, dtype=float)
        X2 = np.asarray(X2, dtype=float)
        X1_group = X1 if X1.shape[1] == len(group) else X1[:, group]
        X2_group = X2 if X2.shape[1] == len(group) else X2[:, group]
        squared_distance = self._sqdist(X1_group, X2_group)
        return scale * np.exp(-0.5 * squared_distance / (bandwidth * bandwidth))

    def _additive_kernel(self, X1, X2, groups=None, bandwidth=None, scale=None):
        groups = self.groups if groups is None else groups
        kernel = np.zeros((len(X1), len(X2)), dtype=float)
        for group in groups:
            kernel += self._group_kernel(X1, X2, group, bandwidth=bandwidth, scale=scale)
        return kernel

    def _negative_log_marginal_likelihood(self, log_params, groups, X, centered_y):
        bandwidth = float(np.exp(log_params[0]))
        scale = float(np.exp(log_params[1]))
        K = self._additive_kernel(X, X, groups=groups, bandwidth=bandwidth, scale=scale)
        K.flat[:: len(K) + 1] += float(self.noise_variance or 0.0) + 1e-9
        try:
            chol = cho_factor(K, lower=True, check_finite=False)
            alpha = cho_solve(chol, centered_y, check_finite=False)
        except np.linalg.LinAlgError:
            return 1e100
        log_det = 2.0 * np.sum(np.log(np.maximum(np.diag(chol[0]), 1e-300)))
        return float(0.5 * centered_y @ alpha + 0.5 * log_det + 0.5 * len(X) * math.log(2.0 * math.pi))

    def _direct_hyperparameters(self, groups, target_y, *, X=None):
        training_X = self.X if X is None else np.asarray(X, dtype=float)
        scale_reference = max(float(getattr(self, "_initial_std", 0.0)), 1e-12)
        scale_bounds = (0.03 * scale_reference, 30.0 * scale_reference)
        result = direct(
            self._negative_log_marginal_likelihood,
            bounds=[
                (
                    math.log(self.BANDWIDTH_LOWER_BOUND),
                    math.log(self.bandwidth_upper_bound),
                ),
                (math.log(scale_bounds[0]), math.log(scale_bounds[1])),
            ],
            args=(groups, training_X, target_y),
            maxiter=20,
            locally_biased=False,
        )
        if not np.all(np.isfinite(result.x)):
            raise RuntimeError("ADD-GP-UCB DIRECT hyperparameter fitting returned non-finite values.")
        return np.asarray(result.x, dtype=float)

    def _learn_structure_and_hyperparameters(self):
        if len(self.y) < 2:
            return
        self._y_mean = float(np.mean(self.y))
        target_y = self.y - self._y_mean

        # Official partialLearn first fits shared h/scale for an initial
        # transform, scores exactly five random permutation decompositions with
        # those values, then refits h/scale for the selected decomposition.
        group_sizes = [len(group) for group in self.groups]
        canonical_groups = []
        start = 0
        for group_size in group_sizes:
            canonical_groups.append(np.arange(start, start + group_size, dtype=int))
            start += group_size
        provisional_transform = orth(self.rng.normal(size=(self.dim, self.dim)))
        provisional_X = self.X @ provisional_transform
        provisional = self._direct_hyperparameters(
            canonical_groups,
            target_y,
            X=provisional_X,
        )
        candidate_groups = [self._random_partition() for _ in range(self.decomposition_candidates)]
        scored = [
            self._negative_log_marginal_likelihood(provisional, groups, self.X, target_y)
            for groups in candidate_groups
        ]
        self.groups = candidate_groups[int(np.argmin(scored))]
        fitted = self._direct_hyperparameters(self.groups, target_y, X=self.X)
        self.bandwidth = float(np.exp(fitted[0]))
        self.kernel_scale = float(np.exp(fitted[1]))

    def _fit(self):
        self._y_mean = float(np.mean(self.y))
        centered_y = self.y - self._y_mean
        K = self._additive_kernel(self.X, self.X)
        jitter = float(self.noise_variance or 0.0) + 1e-9
        for _ in range(8):
            try:
                matrix = K.copy()
                matrix.flat[:: len(matrix) + 1] += jitter
                self._chol = cho_factor(matrix, lower=True, check_finite=False)
                self._alpha = cho_solve(self._chol, centered_y, check_finite=False)
                return
            except np.linalg.LinAlgError:
                jitter *= 10.0
        raise np.linalg.LinAlgError("ADD-GP-UCB additive kernel is not positive definite.")

    def _component_posterior(self, candidates, group):
        cross = self._group_kernel(candidates, self.X, group)
        # addGPRegression.m supplies zero-valued meanFuncs{k} to
        # computeIndGPOutput.m.  The common data mean belongs only to the
        # combined posterior, not to every additive component.
        mean = cross @ self._alpha
        solved = cho_solve(self._chol, cross.T, check_finite=False)
        variance = np.maximum(self.kernel_scale - np.sum(cross * solved.T, axis=1), 1e-12)
        return mean, np.sqrt(variance)

    def _combined_posterior(self, candidates):
        candidates = np.asarray(candidates, dtype=float)
        cross = self._additive_kernel(candidates, self.X)
        # The empty commonMeanFunc is converted to mean(y) downstream, in
        # addGPDecompMargLikelihood.m/addGPRegression.m.  This is the behavior
        # implemented by computeCombGPOutput.m.
        mean = self._y_mean + cross @ self._alpha
        solved = cho_solve(self._chol, cross.T, check_finite=False)
        prior_variance = float(len(self.groups)) * float(self.kernel_scale)
        variance = np.maximum(prior_variance - np.sum(cross * solved.T, axis=1), 1e-12)
        return mean, np.sqrt(variance)

    def _avoid_duplicate(self, point):
        """Apply boAddGP.m's normalized-distance perturbation rule."""

        point = np.clip(np.asarray(point, dtype=float), self.bounds[:, 0], self.bounds[:, 1])
        if not len(self.X):
            return point
        bandwidth = float(self.bandwidth)

        def normalized_min_distance(candidate):
            distances = np.linalg.norm(self.X - candidate.reshape(1, -1), axis=1)
            return float(np.min(distances) / bandwidth)

        if normalized_min_distance(point) >= self.DUPLICATE_DISTANCE_THRESHOLD:
            return point
        for _ in range(self.DUPLICATE_RETRY_LIMIT):
            point = np.clip(
                point + 0.1 * bandwidth * self.rng.normal(size=self.dim),
                self.bounds[:, 0],
                self.bounds[:, 1],
            )
            if normalized_min_distance(point) >= self.DUPLICATE_DISTANCE_THRESHOLD:
                return point
        raise RuntimeError(
            "ADD-GP-UCB could not perturb a duplicate proposal inside the bounds "
            f"after {self.DUPLICATE_RETRY_LIMIT} attempts."
        )

    def _propose(self):
        self._fit()
        point = np.empty(self.dim, dtype=float)
        per_group_budget = max(1, int(math.ceil(self.num_acq_evals / len(self.groups))))
        t = len(self.y) + 1
        for group in self.groups:
            beta_t = self.beta_t(len(group), t)

            def negative_ucb(values):
                candidates = np.asarray(values, dtype=float).reshape(1, -1)
                mean, std = self._component_posterior(candidates, group)
                return -float(mean[0] + math.sqrt(beta_t) * std[0])

            result = direct(
                negative_ucb,
                bounds=list(zip(self.bounds[group, 0], self.bounds[group, 1])),
                maxfun=per_group_budget,
                maxiter=max(1, per_group_budget),
                locally_biased=False,
            )
            point[group] = np.asarray(result.x, dtype=float)
        point = np.clip(point, self.bounds[:, 0], self.bounds[:, 1])
        # getNextQueryPt.m evaluates the combined posterior standard deviation
        # before boAddGP.m applies duplicate avoidance.  That unperturbed std is
        # what drives the consecutive-low-variance relearning trigger.
        _, combined_std = self._combined_posterior(point.reshape(1, -1))
        return self._avoid_duplicate(point), float(combined_std[0])

    @staticmethod
    def beta_t(component_dimension, t):
        return float(component_dimension) * math.log(2.0 * float(t)) / 5.0

    def _record_stopping_snapshot(self, candidate_indices):
        if not self.enable_posterior_stopping or len(self.y) < max(2, self.n_init):
            return None
        self._fit()
        candidate_indices = [int(index) for index in candidate_indices if 0 <= int(index) < len(self.y)]
        if not candidate_indices:
            return None
        query = self.X[candidate_indices]
        mean, std = self._combined_posterior(query)
        current, best, argmax_by_delta = {}, {}, {}
        for delta in self.risk_tolerances:
            key = f"{float(delta):.12g}"
            values = mean + std * float(norm.ppf(delta))
            local_index = int(np.argmax(values))
            threshold = float(values[local_index])
            current[key] = threshold
            argmax_by_delta[key] = int(candidate_indices[local_index])
            best[key] = max(float(self.best_stoppable_threshold_by_delta[key]), threshold)
            self.best_stoppable_threshold_by_delta[key] = best[key]
        stop_probability = None
        required_probability = None
        stop_index = None
        stopped = False
        if self.stopping_threshold is not None:
            probabilities = norm.sf((float(self.stopping_threshold) - mean) / std)
            local_index = int(np.argmax(probabilities))
            stop_probability = float(probabilities[local_index])
            required_probability = 1.0 - float(self.risk_tolerances[0])
            stop_index = int(candidate_indices[local_index])
            stopped = bool(stop_probability >= required_probability)
        first_key = f"{float(self.risk_tolerances[0]):.12g}"
        center_index = int(argmax_by_delta[first_key])
        center_local = candidate_indices.index(center_index)
        entry = {
            "n": int(len(self.y)),
            "posterior_mean_at_best": float(mean[center_local]),
            "posterior_std_at_best": float(std[center_local]),
            "best_observed_value": float(np.max(self.y)),
            "best_index": center_index,
            "posterior_candidate_indices": candidate_indices,
            "threshold_argmax_index_by_delta": argmax_by_delta,
            "current_stoppable_threshold_by_delta": current,
            "best_stoppable_threshold_by_delta": best,
            "noise_variance": float(self.noise_variance or 0.0),
            "noise_model": "commonNoise=0.01*std(initial values), per official MATLAB defaults",
            "stop_probability": stop_probability,
            "required_stop_probability": required_probability,
            "stopping_threshold": self.stopping_threshold,
            "stop_index": stop_index,
            "stopped": stopped,
            "posterior_center_rule": "current point evaluated by the fixed-noise additive GP",
        }
        self.threshold_trace.append(entry)
        self.stopped = stopped
        if stopped:
            self.stop_iteration = int(len(self.y))
        return entry

    def _initial_design(self):
        if self.initial_points is not None and len(self.initial_points) >= self.n_init:
            return self.initial_points[: self.n_init].copy()
        if self.n_init == 1:
            return ((self.bounds[:, 0] + self.bounds[:, 1]) / 2.0).reshape(1, -1)
        return self.rng.uniform(self.bounds[:, 0], self.bounds[:, 1], size=(self.n_init, self.dim))

    def optimize(self):
        points = [np.asarray(x, dtype=float).copy() for x in self._initial_design()]
        values = [float(self.objective(x)) for x in points]
        self.X = np.asarray(points, dtype=float)
        self.y = np.asarray(values, dtype=float)
        self._initial_std = float(np.std(self.y, ddof=1)) if len(self.y) > 1 else 0.0
        self.noise_variance = 0.01 * self._initial_std
        self.kernel_scale = max(self._initial_std, 1e-12)
        stopping_by_n = {}
        initial_entry = self._record_stopping_snapshot(range(len(self.y)))
        if initial_entry is not None:
            stopping_by_n[int(initial_entry["n"])] = initial_entry
        while len(self.y) < self.max_iter and not self.stopped:
            bo_iteration = len(self.y) - self.n_init + 1
            low_std_relearn = self._low_std_counter == self.MAX_THRESHOLD_EXCEEDS
            if low_std_relearn:
                previous_upper_bound = float(self.bandwidth_upper_bound)
                self.bandwidth_upper_bound = max(
                    self.BANDWIDTH_LOWER_BOUND,
                    0.9 * float(self.bandwidth),
                )
                self._low_std_counter = 0
                self.early_relearn_events.append(
                    {
                        "bo_iteration": int(bo_iteration),
                        "previous_bandwidth_upper_bound": previous_upper_bound,
                        "bandwidth_upper_bound": float(self.bandwidth_upper_bound),
                    }
                )
            if (
                bo_iteration == 1
                or (bo_iteration - 1) % self.hyperparameter_relearn_interval == 0
                or low_std_relearn
            ):
                self._learn_structure_and_hyperparameters()
            x_next, selected_point_std = self._propose()
            y_next = float(self.objective(x_next))
            self.X = np.vstack([self.X, x_next])
            self.y = np.append(self.y, y_next)
            if selected_point_std < self.OPT_PT_STD_THRESHOLD:
                self._low_std_counter += 1
            else:
                self._low_std_counter = 0
            entry = self._record_stopping_snapshot([len(self.y) - 1])
            if entry is not None:
                stopping_by_n[int(entry["n"])] = entry
        self.history = [
            {
                "n": idx + 1,
                "best_y": float(np.max(self.y[: idx + 1])),
                "stopping_info": stopping_by_n.get(idx + 1),
            }
            for idx in range(len(self.y))
        ]
        return self


__all__ = ["ADDGPUCBOptimizer"]
