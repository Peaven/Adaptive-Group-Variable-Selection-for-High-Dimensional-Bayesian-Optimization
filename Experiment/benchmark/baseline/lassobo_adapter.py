"""Published-setting adapter for Hoang et al.'s AISTATS 2025 LassoBO.

This is a clean, runner-compatible implementation of Algorithm 1 in
``High Dimensional Bayesian Optimization using Lasso Variable Selection``.
The paper is authoritative for the experiment settings: 30 LHS observations,
lambda=1e-3, M_t=ceil(t**(1/3)), an ARD squared-exponential GP, and GP-UCB.
Where the paper does not specify a numerical optimizer detail, this adapter
uses the authors' public Fsoft-AIC/LassoBO implementation (BoTorch 0.5.1).
"""

from __future__ import annotations

import copy
import math

import botorch
import gpytorch
import numpy as np
import torch
from botorch.acquisition import UpperConfidenceBound
from botorch.optim import optimize_acqf
from botorch.utils.transforms import normalize, unnormalize
from gpytorch.constraints import Interval
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.models import ExactGP

from Experiment.benchmark.baseline.native_optimizers import (
    _isolates_legacy_rng,
)


class _SparseARDGP(ExactGP, botorch.models.gpytorch.GPyTorchModel):
    """The ARD RBF GP used to estimate LassoBO inverse length scales."""

    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self._num_outputs = 1
        self.mean_module = ConstantMean()
        base_kernel = RBFKernel(
            ard_num_dims=int(train_x.shape[-1]),
            lengthscale_constraint=Interval(1e-3, 1e6),
        )
        self.covar_module = ScaleKernel(
            base_kernel,
            outputscale_constraint=Interval(0.1, 10.0),
        )

    def forward(self, x):
        return MultivariateNormal(self.mean_module(x), self.covar_module(x))


class LassoBOOptimizer:
    """Sequential LassoBO using the paper's published experimental defaults."""

    def __init__(
        self,
        objective,
        bounds,
        max_iter,
        initial_points=None,
        seed=None,
        n_init=30,
        lasso_lambda=1e-3,
        subspace_root=3,
        ucb_beta_low=0.75,
        ucb_beta_high=1.5,
        num_restarts=8,
        raw_samples=1000,
        gp_training_steps=250,
        gp_learning_rate=0.1,
        gp_lr_gamma=0.9,
        gp_early_stop_patience=33,
    ):
        self.objective = objective
        self.bounds = np.asarray(bounds, dtype=float)
        if self.bounds.ndim != 2 or self.bounds.shape[1] != 2:
            raise ValueError("bounds must have shape (D, 2)")
        if np.any(self.bounds[:, 0] >= self.bounds[:, 1]):
            raise ValueError("every lower bound must be smaller than its upper bound")
        self.dim = int(self.bounds.shape[0])
        self.max_iter = int(max_iter)
        self.n_init = min(int(n_init), self.max_iter)
        self.initial_points = (
            None
            if initial_points is None
            else np.asarray(initial_points, dtype=float)
        )
        self.seed = None if seed is None else int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.lasso_lambda = float(lasso_lambda)
        self.subspace_root = int(subspace_root)
        self.ucb_beta_low = float(ucb_beta_low)
        self.ucb_beta_high = float(ucb_beta_high)
        self.num_restarts = int(num_restarts)
        self.raw_samples = int(raw_samples)
        self.gp_training_steps = int(gp_training_steps)
        self.gp_learning_rate = float(gp_learning_rate)
        self.gp_lr_gamma = float(gp_lr_gamma)
        self.gp_early_stop_patience = int(gp_early_stop_patience)
        if self.subspace_root <= 0:
            raise ValueError("subspace_root must be positive")
        if self.lasso_lambda < 0:
            raise ValueError("lasso_lambda must be non-negative")

        self.X = np.empty((0, self.dim), dtype=float)
        self.y = np.empty((0,), dtype=float)
        self.history = []
        self.selected_dimensions_history = []
        self.inverse_lengthscale_history = []
        self.subspace_count_history = []
        self.ucb_beta_history = []
        self.threshold_trace = []
        self.stopped = False
        self.stop_iteration = None
        self._warm_state = None

    def _lhs(self, n):
        n = int(n)
        centers = (1.0 + 2.0 * np.arange(n, dtype=float)) / (2.0 * n)
        unit = np.empty((n, self.dim), dtype=float)
        for index in range(self.dim):
            unit[:, index] = centers[self.rng.permutation(n)]
        unit += self.rng.uniform(-1.0, 1.0, size=(n, self.dim)) / (2.0 * n)
        return self.bounds[:, 0] + unit * (self.bounds[:, 1] - self.bounds[:, 0])

    def _initial_design(self):
        if self.initial_points is not None:
            if len(self.initial_points) < self.n_init:
                raise ValueError("initial_points contains fewer rows than n_init")
            return np.asarray(self.initial_points[: self.n_init], dtype=float).copy()
        return self._lhs(self.n_init)

    def subspace_count(self, iteration):
        """Published M_t=ceil(t^(1/n)), with n=3 in every experiment."""

        iteration = max(1, int(iteration))
        return int(math.ceil(iteration ** (1.0 / self.subspace_root)))

    @staticmethod
    def important_dimensions(inverse_lengthscales):
        """Apply Section 3.1's published above-mean importance rule."""

        scores = np.asarray(inverse_lengthscales, dtype=float).reshape(-1)
        finite = np.where(np.isfinite(scores), scores, 0.0)
        selected = np.flatnonzero(finite > float(np.mean(finite)))
        if selected.size == 0:
            selected = np.asarray([int(np.argmax(finite))], dtype=int)
        return selected.astype(int)

    def _standardized_training_data(self):
        train_x = torch.as_tensor(self.X, dtype=torch.double)
        torch_bounds = torch.as_tensor(self.bounds.T, dtype=torch.double)
        train_x = normalize(train_x, torch_bounds)
        train_y = torch.as_tensor(self.y, dtype=torch.double)
        center = train_y.median()
        scale = train_y.std()
        train_y = (train_y - center) / (scale + 1e-6)
        return train_x, train_y, torch_bounds

    def _fit_sparse_gp(self, reset=False):
        train_x, train_y, _ = self._standardized_training_data()
        likelihood = GaussianLikelihood(noise_constraint=Interval(0.0, 5e-2)).to(
            dtype=torch.double
        )
        model = _SparseARDGP(train_x, train_y, likelihood).to(dtype=torch.double)
        if self._warm_state is not None and not reset:
            model.load_state_dict(self._warm_state)
        else:
            model.initialize(
                **{
                    "covar_module.outputscale": 1.1,
                    "covar_module.base_kernel.lengthscale": float(
                        self.rng.choice(np.asarray([1.0, 0.5, 0.25]))
                    ),
                    "likelihood.noise": 1e-2,
                }
            )

        model.train()
        likelihood.train()
        marginal_likelihood = ExactMarginalLogLikelihood(likelihood, model)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.gp_learning_rate)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=self.gp_lr_gamma,
        )
        best_loss = float("inf")
        best_state = None
        stale_steps = 0
        for _ in range(self.gp_training_steps):
            optimizer.zero_grad()
            with gpytorch.settings.cholesky_jitter(1e-4):
                output = model(train_x)
                lengthscales = model.covar_module.base_kernel.lengthscale.reshape(-1)
                # rho_i is the inverse squared length scale in Eq. (7).
                sparse_penalty = torch.sum(1.0 / (lengthscales.square() + 1e-6))
                loss = (
                    -marginal_likelihood(output, train_y)
                    + self.lasso_lambda * sparse_penalty
                )
            loss.backward()
            optimizer.step()
            value = float(loss.detach())
            if np.isfinite(value) and value < best_loss:
                best_loss = value
                best_state = copy.deepcopy(model.state_dict())
                stale_steps = 0
            else:
                scheduler.step()
                stale_steps += 1
                if stale_steps > self.gp_early_stop_patience:
                    break
        if best_state is not None:
            model.load_state_dict(best_state)
        self._warm_state = copy.deepcopy(model.state_dict())
        model.eval()
        likelihood.eval()
        lengthscales = (
            model.covar_module.base_kernel.lengthscale.detach().cpu().numpy().reshape(-1)
        )
        inverse_lengthscales = 1.0 / np.maximum(lengthscales, 1e-12)
        return model, inverse_lengthscales

    def _random_backgrounds(self, count):
        return self.rng.uniform(0.0, 1.0, size=(int(count), self.dim))

    def _propose(self, model, active_dimensions, iteration):
        active_dimensions = np.asarray(active_dimensions, dtype=int)
        train_x, train_y, torch_bounds = self._standardized_training_data()
        # The fitted sparse ARD model is itself a BoTorch-compatible model.
        # Reusing it also avoids a version-sensitive SingleTaskGP constructor.
        gp = model
        gp.eval()
        beta = float(self.rng.uniform(self.ucb_beta_low, self.ucb_beta_high))
        acquisition = UpperConfidenceBound(gp, beta=beta)
        unit_bounds = torch.stack(
            [torch.zeros(self.dim, dtype=torch.double), torch.ones(self.dim, dtype=torch.double)]
        )

        count = self.subspace_count(iteration)
        random_backgrounds = self._random_backgrounds(count)
        observed_unit = normalize(train_x, unit_bounds)
        best_background = observed_unit[int(torch.argmax(train_y))].detach().cpu().numpy()
        backgrounds = np.vstack([random_backgrounds, best_background])
        inactive = np.setdiff1d(np.arange(self.dim), active_dimensions)

        best_candidate = None
        best_acquisition = -float("inf")
        for background in backgrounds:
            fixed_features = {int(i): float(background[i]) for i in inactive}
            candidate, value = optimize_acqf(
                acquisition,
                bounds=unit_bounds,
                q=1,
                fixed_features=fixed_features,
                num_restarts=self.num_restarts,
                raw_samples=self.raw_samples,
            )
            scalar = float(value.detach().reshape(-1)[0])
            if scalar > best_acquisition:
                best_acquisition = scalar
                best_candidate = candidate.detach()
        if best_candidate is None:
            raise RuntimeError("LassoBO acquisition optimization produced no candidate")
        proposed = unnormalize(best_candidate, torch_bounds)[0].cpu().numpy()
        self.subspace_count_history.append(count)
        self.ucb_beta_history.append(beta)
        return np.clip(proposed, self.bounds[:, 0], self.bounds[:, 1])

    @_isolates_legacy_rng
    def optimize(self):
        for point in self._initial_design():
            self.X = np.vstack([self.X, point])
            self.y = np.append(self.y, float(self.objective(point)))

        no_improvement = 0
        reset_after = 11  # M=5 in the public code gives max_tol=2*M+1.
        reset_model = True
        iteration = 1
        while len(self.y) < self.max_iter:
            model, scores = self._fit_sparse_gp(reset=reset_model)
            reset_model = False
            active = self.important_dimensions(scores)
            candidate = self._propose(model, active, iteration)
            previous_best = float(np.max(self.y))
            value = float(self.objective(candidate))
            self.X = np.vstack([self.X, candidate])
            self.y = np.append(self.y, value)
            self.inverse_lengthscale_history.append(scores.copy())
            self.selected_dimensions_history.append(active.copy())
            if value > previous_best:
                no_improvement = 0
            else:
                no_improvement += 1
                if no_improvement > reset_after:
                    no_improvement = 0
                    reset_model = True
                    self._warm_state = None
            self.history.append(
                {
                    "n": int(len(self.y)),
                    "best_y": float(np.max(self.y)),
                    "active_dimensions": active.tolist(),
                    "M_t": int(self.subspace_count_history[-1]),
                    "ucb_beta": float(self.ucb_beta_history[-1]),
                    "stopping_info": None,
                }
            )
            iteration += 1
        return self


__all__ = ["LassoBOOptimizer"]
