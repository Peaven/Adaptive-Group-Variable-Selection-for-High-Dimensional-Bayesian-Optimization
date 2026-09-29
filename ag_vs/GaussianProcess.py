from __future__ import annotations

import warnings
from copy import deepcopy
from typing import Optional, Tuple

import numpy as np
from scipy.linalg import cho_solve, solve_triangular
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.base import clone
from ag_vs.utils import Array, Node
from Experiment.benchmark.baseline.reference_infrastructure import (
    BO_REFERENCE_SPEC,
    make_reference_bo_gp,
)


class GaussianProcessManager:
    def __init__(
        self,
        noise_std: Optional[float] = None,
        noise_variance: Optional[float] = None,
        learn_noise: bool = False,
        kernel_selection: str = "fixed",
        default_kernel: str = "matern52",
    ) -> None:
        if noise_variance is None:
            if noise_std is None:
                noise_variance = float(BO_REFERENCE_SPEC.alpha)
            else:
                parsed_std = float(noise_std)
                noise_variance = (
                    float(BO_REFERENCE_SPEC.alpha)
                    if parsed_std == float(BO_REFERENCE_SPEC.observation_noise_std)
                    else parsed_std**2
                )
        self.noise_variance = float(noise_variance)
        if not np.isfinite(self.noise_variance) or self.noise_variance < 0.0:
            raise ValueError("noise_variance must be finite and non-negative")
        self.noise_std = float(np.sqrt(self.noise_variance))
        self.learn_noise = bool(learn_noise)
        self.kernel_selection = str(kernel_selection).strip().lower()
        self.default_kernel = str(default_kernel).strip().lower()
        if self.kernel_selection not in {"fixed", "default"} or self.default_kernel != "matern52":
            raise ValueError("AG-VS supports only its fixed reference Matern-5/2 GP.")

    def make_gp(
        self,
        dim: int,
        kernel_name: str = "matern52",
        initial_kernel=None,
    ) -> GaussianProcessRegressor:
        if kernel_name != "matern52":
            raise ValueError("AG-VS supports only its fixed reference Matern-5/2 GP.")
        dim = int(dim)
        if dim <= 0:
            raise ValueError("AG-VS GP dimension must be positive")
        model = make_reference_bo_gp(
            alpha=self.noise_variance,
            learn_noise=self.learn_noise,
            # AG-VS uses one shared length scale in every local subspace.  Do
            # not pass ``dim`` here: make_reference_bo_gp interprets it as a
            # request for a per-coordinate (ARD) length-scale vector.
            dim=None,
        )
        if initial_kernel is not None:
            model.kernel = clone(initial_kernel)
        return model

    @staticmethod
    def kernel_score(model: GaussianProcessRegressor) -> float:
        score = getattr(model, "log_marginal_likelihood_value_", -np.inf)
        try:
            score = float(score)
        except Exception:
            score = -np.inf
        return score if np.isfinite(score) else -np.inf

    @staticmethod
    def extract_lengthscales(model: GaussianProcessRegressor, dim: int) -> Array:
        candidates = []
        try:
            candidates.append(model.kernel_.k2.length_scale)
        except Exception:
            pass
        try:
            candidates.append(model.kernel_.k1.k2.length_scale)
        except Exception:
            pass
        for value in candidates:
            arr = np.asarray(value, dtype=float).reshape(-1)
            if arr.size == 1 and np.isfinite(arr[0]) and arr[0] > 0.0:
                # Keep the node diagnostic shape stable while making it clear
                # that every coordinate shares the same fitted length scale.
                return np.full(int(dim), float(arr[0]), dtype=float)
            if arr.size == int(dim) and np.all(np.isfinite(arr)) and np.all(arr > 0.0):
                return arr
        return np.empty(0, dtype=float)

    def fit_best_gp(
        self,
        X: Array,
        y: Array,
        dim: int,
        initial_kernel=None,
    ) -> Tuple[Optional[GaussianProcessRegressor], str, float]:
        try:
            model = self.make_gp(
                dim,
                kernel_name=self.default_kernel,
                initial_kernel=initial_kernel,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                warnings.simplefilter("ignore", RuntimeWarning)
                model.fit(X, y)
            return model, self.default_kernel, self.kernel_score(model)
        except Exception:
            return None, "none", -np.inf

    @staticmethod
    def _unwrap_sklearn_model(model):
        return getattr(model, "_sklearn_model", model)

    def _incremental_fixed_kernel_fit(
        self,
        source_model,
        X: Array,
        y: Array,
    ) -> Optional[GaussianProcessRegressor]:
        """Extend an exact sklearn GP Cholesky factor for appended observations."""
        source = self._unwrap_sklearn_model(source_model)
        required = ("kernel_", "X_train_", "y_train_", "L_", "alpha_")
        if source is None or any(not hasattr(source, name) for name in required):
            return None
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        old_X = np.asarray(source.X_train_, dtype=float)
        old_y = np.asarray(source.y_train_, dtype=float).reshape(-1)
        old_n = int(len(old_X))
        if (
            old_n < 2
            or old_n > len(X)
            or old_X.shape[1:] != X.shape[1:]
            or not np.allclose(old_X, X[:old_n], rtol=0.0, atol=0.0)
            or not np.allclose(old_y, y[:old_n], rtol=0.0, atol=0.0)
        ):
            return None
        alpha_setting = np.asarray(getattr(source, "alpha", self.noise_variance))
        if alpha_setting.ndim != 0:
            return None
        alpha_value = float(alpha_setting)
        if not np.isfinite(alpha_value) or alpha_value < 0.0:
            return None

        model = deepcopy(source)
        if old_n == len(X):
            model.gp_incremental_added = 0
            return model
        try:
            new_X = X[old_n:]
            cross = np.asarray(source.kernel_(new_X, old_X), dtype=float)
            solved = solve_triangular(
                np.asarray(source.L_, dtype=float),
                cross.T,
                lower=True,
                check_finite=False,
            )
            schur = np.asarray(source.kernel_(new_X), dtype=float)
            schur.flat[:: len(new_X) + 1] += alpha_value
            schur -= solved.T @ solved
            schur = 0.5 * (schur + schur.T)
            new_factor = np.linalg.cholesky(schur)
            full_factor = np.zeros((len(X), len(X)), dtype=float)
            full_factor[:old_n, :old_n] = np.asarray(source.L_, dtype=float)
            full_factor[old_n:, :old_n] = solved.T
            full_factor[old_n:, old_n:] = new_factor
            alpha_vector = cho_solve(
                (full_factor, True),
                y,
                check_finite=False,
            )
        except (ValueError, TypeError, np.linalg.LinAlgError):
            return None

        model.X_train_ = X.copy()
        model.y_train_ = y.copy()
        model.L_ = full_factor
        model.alpha_ = np.asarray(alpha_vector, dtype=float)
        model.n_features_in_ = int(X.shape[1])
        model.log_marginal_likelihood_value_ = float(
            -0.5 * y.dot(model.alpha_)
            - np.log(np.diag(full_factor)).sum()
            - 0.5 * len(X) * np.log(2.0 * np.pi)
        )
        model.gp_incremental_added = int(len(X) - old_n)
        return model

    def fit_node_gp(
        self,
        node: Node,
        X_all: list[Array],
        y_all: list[float],
    ):
        """Fit the node's sole isotropic GP for acquisition and diagnostics."""
        node.model = None
        node.gp_kernel_name = "none"
        node.gp_kernel_score = -np.inf
        node._gp_fit_signature = None
        node.lengthscales = None
        node.lengthscale_gp_kernel_name = "none"
        node.lengthscale_gp_kernel_score = -np.inf
        local_indices = [
            int(idx)
            for idx in getattr(node, "local_indices", []) or []
            if 0 <= int(idx) < len(X_all) and 0 <= int(idx) < len(y_all)
        ]
        # A newly merged/reinitialized node may initially own one unique
        # observation. sklearn's exact GP supports that seed fit; the node is
        # then evaluated for a full t_seq point-batch block before splitting.
        if len(local_indices) < 1 or len(node.dims) == 0:
            return None
        X_all_arr = np.asarray(X_all, dtype=float)
        y_all_arr = np.asarray(y_all, dtype=float)
        X = X_all_arr[np.asarray(local_indices, dtype=int)][:, node.dims]
        y = y_all_arr[np.asarray(local_indices, dtype=int)]
        incremental_source = getattr(node, "gp_incremental_source_model", None)
        model = None
        fit_mode = "full_hyperparameter_fit"
        if bool(getattr(node, "gp_hold_hyperparameters", False)) and incremental_source is not None:
            model = self._incremental_fixed_kernel_fit(incremental_source, X, y)
            if model is not None:
                fit_mode = "incremental_fixed_kernel"
        warm_source = getattr(node, "gp_warm_start_model", None)
        if warm_source is None:
            warm_source = incremental_source
        initial_kernel = None
        if warm_source is not None:
            warm_source = self._unwrap_sklearn_model(warm_source)
            initial_kernel = getattr(warm_source, "kernel_", None)
        if model is None:
            model, kernel_name, kernel_score = self.fit_best_gp(
                X,
                y,
                len(node.dims),
                initial_kernel=initial_kernel,
            )
            if model is not None and initial_kernel is not None:
                fit_mode = "warm_started_hyperparameter_fit"
        else:
            kernel_name = self.default_kernel
            kernel_score = self.kernel_score(model)
        for attribute in ("gp_incremental_source_model", "gp_warm_start_model"):
            if hasattr(node, attribute):
                delattr(node, attribute)
        if model is None:
            return None
        node.gp_fit_mode = str(fit_mode)
        node.model = model
        node.gp_kernel_name = kernel_name
        node.gp_kernel_score = float(kernel_score)

        lengthscales = self.extract_lengthscales(model, len(node.dims))
        if lengthscales.size == len(node.dims):
            node.lengthscales = lengthscales
            node.lengthscale_gp_kernel_name = "acquisition_gp_matern52_isotropic"
            node.lengthscale_gp_kernel_score = float(kernel_score)
        return model
