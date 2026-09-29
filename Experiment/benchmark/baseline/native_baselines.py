"""Project-owned baseline loops derived from lamda-bbo/MCTS-VS.

Methods that cannot be executed faithfully in this runtime are rejected with
an actionable dependency/environment message.  They are never replaced by a
similarly named sklearn approximation.
"""

from __future__ import annotations

import importlib
from importlib import metadata as importlib_metadata

import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern

from Experiment.benchmark.baseline.native_optimizers import (
    _load_native_mcts,
    _isolates_legacy_rng,
    _posterior_threshold_snapshot,
    _risk_key,
)
from Experiment.benchmark.baseline.reference_infrastructure import (
    BO_REFERENCE_SPEC,
    classical_expected_improvement,
    make_reference_bo_gp,
    sample_reference_bo_candidates,
)


class _ExactBudgetReached(RuntimeError):
    """Private control-flow signal raised before an over-budget evaluation."""


class _BudgetedReferenceObjective:
    """Callable objective facade expected by the vendored LA-MCTS implementation."""

    def __init__(self, objective, bounds, budget):
        self.objective = objective
        bounds = np.asarray(bounds, dtype=float)
        self.lb = bounds[:, 0].copy()
        self.ub = bounds[:, 1].copy()
        self.dims = int(bounds.shape[0])
        self.budget = int(budget)
        self.points = []
        self.values = []

    def __call__(self, candidate):
        if len(self.values) >= self.budget:
            raise _ExactBudgetReached
        point = np.asarray(candidate, dtype=float).reshape(-1)
        if point.shape != (self.dims,):
            raise ValueError(
                "Reference optimizer proposed a point with shape "
                f"{point.shape}; expected ({self.dims},)."
            )
        value = float(self.objective(point))
        self.points.append(point.copy())
        self.values.append(value)
        return value


class _BestKImputationStrategy:
    """Import-safe equivalent of the vendored ``UiptBestKStrategy``.

    The original module cannot be imported directly without adding its
    repository root to ``sys.path``.  Keeping this tiny state machine here also
    makes clear that it is part of the Dropout baseline, not TuRBO.  Its
    NumPy draws intentionally use the legacy global generator because the
    upstream TuRBO component does too; ``optimize`` isolates and restores that
    state with ``_isolates_legacy_rng``.
    """

    def __init__(self, dims, k=20):
        self.dims = int(dims)
        self.k = int(k)
        self.best_xs = []
        self.best_ys = []

    def init_strategy(self, xs, ys):
        for x, y in zip(xs, ys):
            self.update(x, y)

    def get_full_variable(self, fixed_variables, lb, ub):
        del lb, ub  # BestK fills from observations; only active values use bounds.
        if len(self.best_xs) == 0:
            raise RuntimeError("BestK imputation requires at least one observation.")
        best_xs = np.asarray(self.best_xs, dtype=float)
        new_x = np.zeros(self.dims, dtype=float)
        for dim in range(self.dims):
            if dim in fixed_variables:
                new_x[dim] = float(fixed_variables[dim])
            else:
                new_x[dim] = float(np.random.choice(best_xs[:, dim]))
        return new_x

    def update(self, x, y):
        x = np.asarray(x, dtype=float).copy()
        y = float(y)
        if len(self.best_xs) < self.k:
            self.best_xs.append(x)
            self.best_ys.append(y)
            if len(self.best_xs) == self.k:
                self.best_xs = np.vstack(self.best_xs)
                self.best_ys = np.asarray(self.best_ys, dtype=float)
        else:
            minimum = float(np.min(self.best_ys))
            if y > minimum:
                tied = np.argwhere(self.best_ys == minimum).reshape(-1)
                index = int(np.random.choice(tied))
                self.best_xs[index] = x
                self.best_ys[index] = y
        assert len(self.best_xs) <= self.k


def _load_native_turbo_vs_component():
    from .native.turbo.turbo_1_component import Turbo1_VS_Component
    return Turbo1_VS_Component


def _load_native_lamcts():
    from .native.lamcts.MCTS import MCTS
    return MCTS


def lamcts_local_turbo_posterior_trace(
    X,
    y,
    *,
    outer_initial_count=20,
    turbo_epoch_budget=50,
    turbo_batch_size=10,
    noise_variance=BO_REFERENCE_SPEC.alpha,
    risk_tolerances=None,
    learn_noise=False,
):
    """Reconstruct batch-end posteriors for each LA-MCTS leaf TuRBO epoch.

    The vendored LA-MCTS API returns evaluated points but discards each
    short-lived leaf TuRBO object.  Its evaluation order is nevertheless
    explicit: 20 outer initial points followed by independent TuRBO epochs of
    at most 50 evaluations, in native batches of 10. Refit a reporting GP on
    the current epoch prefix after every completed batch. Real-life runs learn
    its observation noise; synthetic runs retain the configured fixed noise.
    This is reporting-only and cannot influence proposals or stopping.
    """

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    if X.ndim != 2 or len(X) != len(y):
        raise ValueError("LA-MCTS posterior reconstruction requires aligned 2-D X and 1-D y.")
    if len(y) <= int(outer_initial_count):
        return []

    deltas = [0.05] if risk_tolerances is None else [float(x) for x in risk_tolerances]
    best_by_delta = {_risk_key(delta): -float("inf") for delta in deltas}
    trace = []
    epoch_start = int(outer_initial_count)
    epoch_budget = max(1, int(turbo_epoch_budget))
    batch_size = max(1, int(turbo_batch_size))

    while epoch_start < len(y):
        epoch_end = min(len(y), epoch_start + epoch_budget)
        epoch_X = X[epoch_start:epoch_end]
        epoch_y = y[epoch_start:epoch_end]
        endpoints = list(range(batch_size, len(epoch_y) + 1, batch_size))
        if not endpoints or endpoints[-1] != len(epoch_y):
            endpoints.append(len(epoch_y))

        for local_end in endpoints:
            query_start = max(0, int(local_end) - batch_size)
            entry = _posterior_threshold_snapshot(
                epoch_X[:local_end],
                epoch_y[:local_end],
                noise_variance=float(noise_variance),
                risk_tolerances=deltas,
                best_by_delta=best_by_delta,
                use_ard=True,
                candidate_indices=list(range(query_start, int(local_end))),
                stopping_threshold=None,
                learn_noise=bool(learn_noise),
            )
            if entry is None:
                continue

            def global_index(value):
                return None if value is None else int(epoch_start + int(value))

            entry["n"] = int(epoch_start + local_end)
            entry["best_index"] = global_index(entry.get("best_index"))
            entry["stop_index"] = global_index(entry.get("stop_index"))
            entry["posterior_candidate_indices"] = [
                global_index(index) for index in entry.get("posterior_candidate_indices", [])
            ]
            entry["threshold_argmax_index_by_delta"] = {
                key: global_index(index)
                for key, index in entry.get("threshold_argmax_index_by_delta", {}).items()
            }
            entry["best_observed_value"] = float(np.max(y[: entry["n"]]))
            entry["local_turbo_epoch_start"] = int(epoch_start)
            entry["local_turbo_epoch_end"] = int(epoch_end)
            entry["local_turbo_training_count"] = int(local_end)
            entry["posterior_center_rule"] = (
                "current LA-MCTS leaf TuRBO batch points evaluated by the "
                "batch-end local reporting GP with learned observation noise"
                if learn_noise
                else (
                    "current LA-MCTS leaf TuRBO batch points evaluated by the "
                    "batch-end fixed-noise local reporting GP"
                )
            )
            trace.append(entry)
        epoch_start = epoch_end

    return trace


class NativeBaselineOptimizer:
    def __init__(
        self,
        objective,
        bounds,
        max_iter,
        method,
        initial_points=None,
        seed=None,
        active_dim=6,
        n_init=10,
        batch_size=3,
        candidate_count=1024,
        noise_variance=BO_REFERENCE_SPEC.alpha,
        learn_noise=False,
        turbo_max_evals=50,
        turbo_device="cpu",
        turbo_min_cuda=1024,
        la_mcts_cp=0.1,
        experiment_name="registered_objective",
        vae_update_interval=20,
        vae_learning_rate=0.001,
        risk_tolerances=None,
        stopping_threshold=None,
        enable_posterior_stopping=True,
    ):
        self.objective = objective
        self.bounds = np.asarray(bounds, dtype=float)
        self.dim = int(self.bounds.shape[0])
        self.max_iter = int(max_iter)
        self.method = str(method).lower()
        self.initial_points = None if initial_points is None else np.asarray(initial_points, dtype=float)
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.active_dim = min(int(active_dim), self.dim)
        self.n_init = min(int(n_init), self.max_iter)
        self.batch_size = int(batch_size)
        self.candidate_count = int(candidate_count)
        self.noise_variance = max(float(noise_variance), 1e-12)
        self.learn_noise = bool(learn_noise)
        if (
            self.method in {"dropout-bo", "vae-bo"}
            and self.candidate_count != BO_REFERENCE_SPEC.candidate_count
        ):
            raise ValueError("MCTS-VS paper BO adapters require 1024 acquisition candidates.")
        if self.method in {"dropout-bo", "vae-bo"} and not np.isclose(
            self.noise_variance, BO_REFERENCE_SPEC.alpha
        ):
            raise ValueError("MCTS-VS paper classical-BO adapters require GP alpha=1e-4.")
        self.turbo_max_evals = int(turbo_max_evals)
        self.turbo_min_cuda = int(turbo_min_cuda)
        if self.turbo_min_cuda < 0:
            raise ValueError("turbo_min_cuda must be non-negative.")
        self.la_mcts_cp = float(la_mcts_cp)
        self.experiment_name = str(experiment_name)
        if self.method in {"dropout-turbo"} and (
            self.turbo_max_evals <= 10 or self.turbo_max_evals % 10
        ):
            raise ValueError(
                "Native variable-subspace TuRBO requires turbo_max_evals to be a "
                "multiple of its batch size 10 and greater than 10."
            )
        self.turbo_device = str(turbo_device).strip().lower()
        if self.method in {"dropout-turbo", "la-mcts-turbo"} and self.turbo_device not in {
            "cpu",
            "cuda",
        }:
            raise ValueError("turbo_device must be 'cpu' or 'cuda'.")
        if self.method in {"dropout-turbo", "la-mcts-turbo"} and self.turbo_device == "cuda":
            try:
                import torch
            except ImportError as exc:
                raise ImportError("TuRBO-backed baseline CUDA execution requires PyTorch.") from exc
            if not torch.cuda.is_available():
                raise ValueError("turbo_device='cuda' requested, but CUDA is unavailable.")
        self.vae_update_interval = int(vae_update_interval)
        if self.method == "vae-bo" and self.vae_update_interval <= 0:
            raise ValueError("vae_update_interval must be positive.")
        self.vae_learning_rate = float(vae_learning_rate)
        expected_vae_learning_rate = 0.01 if self.learn_noise else 0.001
        if self.method == "vae-bo" and not np.isclose(
            self.vae_learning_rate, expected_vae_learning_rate
        ):
            raise ValueError(
                "VAE-BO learning rate must be 0.01 for real-life learned-noise "
                "runs and 0.001 for synthetic fixed-noise runs."
            )
        self.vae_latent_training_bounds = (
            float(self.bounds[0, 0]),
            float(self.bounds[0, 1]),
        )
        # Preserve baseline/vae_bo.py literally: encoded training values are
        # clipped to [lb[0], ub[0]], while optimize_acqf receives the scalar
        # products lb[0]*active_dims and ub[0]*active_dims. NumPy broadcasts
        # those scalars over every latent coordinate.
        self.vae_latent_acquisition_bounds = (
            float(self.bounds[0, 0]) * self.active_dim,
            float(self.bounds[0, 1]) * self.active_dim,
        )
        self.risk_tolerances = [0.05] if risk_tolerances is None else [float(x) for x in risk_tolerances]
        self.stopping_threshold = None if stopping_threshold is None else float(stopping_threshold)
        self.enable_posterior_stopping = bool(enable_posterior_stopping)
        self.X = np.empty((0, self.dim), dtype=float)
        self.y = np.empty((0,), dtype=float)
        self.history = []
        self._vae = None
        self._vae_optimizer = None
        self._uipt_solver = None
        self.threshold_trace = []
        self.best_stoppable_threshold_by_delta = {
            _risk_key(delta): -float("inf") for delta in self.risk_tolerances
        }
        self.stopped = False
        self.stop_iteration = None
        if self.method in {"dropout-turbo"}:
            if self.batch_size != 10:
                raise ValueError("Dropout-TuRBO require native batch_size=10.")

    def _lhs(self, n):
        n = int(n)
        centers = (1.0 + 2.0 * np.arange(n, dtype=float)) / float(2 * n)
        unit = np.empty((n, self.dim), dtype=float)
        for dim in range(self.dim):
            unit[:, dim] = centers[self.rng.permutation(n)]
        unit += self.rng.uniform(-1.0, 1.0, size=(n, self.dim)) / float(2 * n)
        return self.bounds[:, 0] + unit * (self.bounds[:, 1] - self.bounds[:, 0])

    def _initial_design(self):
        if self.initial_points is not None and len(self.initial_points) >= self.n_init:
            return self.initial_points[: self.n_init].copy()
        return self._lhs(self.n_init)

    def _sobol(self, lows, highs, n=None):
        lows, highs = np.asarray(lows), np.asarray(highs)
        count = int(n or self.candidate_count)
        if count != BO_REFERENCE_SPEC.candidate_count:
            raise ValueError(
                "MCTS-VS paper BO adapters require exactly "
                f"{BO_REFERENCE_SPEC.candidate_count} acquisition candidates."
            )
        return sample_reference_bo_candidates(
            np.column_stack([lows, highs]),
            rng=self.rng,
            count=count,
        )

    def _fit_gp(self, dims):
        dims = np.asarray(dims, dtype=int)
        gp = make_reference_bo_gp(
            alpha=self.noise_variance,
            learn_noise=self.learn_noise,
            dim=None,
        )
        gp.fit(self.X[:, dims], self.y)
        return gp

    def _ei_points(self, dims, q):
        dims = np.asarray(dims, dtype=int)
        gp = self._fit_gp(dims)
        candidates = self._sobol(self.bounds[dims, 0], self.bounds[dims, 1])
        scores = classical_expected_improvement(gp, self.X[:, dims], candidates)
        chosen = np.argsort(scores)[-int(q) :]
        return candidates[chosen]

    def _selected_dims(self):
        if self.method.startswith("dropout"):
            return np.sort(self.rng.choice(self.dim, self.active_dim, replace=False))
        return np.arange(self.dim, dtype=int)

    def _fill(self, dims, selected):
        dims = np.asarray(dims, dtype=int)
        if self._uipt_solver is None:
            raise RuntimeError("BestK imputation strategy has not been initialized.")
        fixed = {int(dim): float(value) for dim, value in zip(dims, selected)}
        return self._uipt_solver.get_full_variable(
            fixed,
            self.bounds[:, 0],
            self.bounds[:, 1],
        )

    def _native_turbo_epoch(self, dims, evaluation_budget):
        """Run one source-faithful Dropout variable-subspace TuRBO epoch."""

        dims = np.asarray(dims, dtype=int)
        evaluation_budget = int(evaluation_budget)
        if evaluation_budget <= 0:
            raise ValueError("A native TuRBO epoch budget must be positive.")
        component_class = _load_native_turbo_vs_component()
        # The upstream constructor rejects max_evals == batch_size even though
        # its optimize loop supports exactly one batch.  Construct at 11 and
        # restore 10 before optimize for a final one-batch epoch.
        constructor_budget = max(evaluation_budget, 11)
        turbo = component_class(
            f=lambda x: -float(self.objective(np.asarray(x, dtype=float))),
            lb=self.bounds[dims, 0].copy(),
            ub=self.bounds[dims, 1].copy(),
            n_init=1,  # Deliberately unused by Turbo1_VS_Component.
            max_evals=constructor_budget,
            batch_size=10,
            verbose=False,
            use_ard=True,
            max_cholesky_size=2000,
            n_training_steps=50,
            min_cuda=self.turbo_min_cuda,
            device=self.turbo_device,
            dtype="float32",
        )
        turbo.max_evals = evaluation_budget
        sampled_x, sampled_loss = turbo.optimize(
            self.X[:, dims],
            -self.y,
            dims,
            self._uipt_solver,
            n=1,
        )
        sampled_x = np.asarray(sampled_x, dtype=float)
        sampled_y = -np.asarray(sampled_loss, dtype=float).reshape(-1)
        if sampled_x.shape != (evaluation_budget, self.dim) or sampled_y.shape != (
            evaluation_budget,
        ):
            raise RuntimeError(
                "Vendored Turbo1_VS_Component returned an unexpected batch shape: "
                f"X={sampled_x.shape}, y={sampled_y.shape}."
            )
        return sampled_x, sampled_y

    def _saas_points(self, q):
        from .native.saasbo_component import run_saasbo_one_epoch
        unit = (self.X - self.bounds[:, 0]) / (self.bounds[:, 1] - self.bounds[:, 0])
        points = run_saasbo_one_epoch(
            unit, -self.y, self.dim, int(q), None,
            np.zeros(self.dim), np.ones(self.dim), seed=self.seed,
            kernel="rbf", observation_variance=(0.0 if self.learn_noise else 1e-6),
        )
        return self.bounds[:, 0] + np.asarray(points) * (self.bounds[:, 1] - self.bounds[:, 0])

    def _ensure_vae(self):
        if self._vae is not None:
            return
        import torch
        import torch.nn as nn

        class VAE(nn.Module):
            def __init__(inner, input_dim, latent_dim):
                super().__init__()
                inner.encoder = nn.Sequential(nn.Linear(input_dim, 256), nn.ReLU(), nn.Linear(256, 128))
                inner.mu = nn.Linear(128, latent_dim)
                inner.logvar = nn.Linear(128, latent_dim)
                inner.latent_mapping = nn.Linear(latent_dim, 128)
                inner.decoder = nn.Sequential(nn.Linear(128, 256), nn.ReLU(), nn.Linear(256, input_dim))

            def encode(inner, x):
                encoded = inner.encoder(x)
                return inner.mu(encoded), inner.logvar(encoded)

            def sample_z(inner, mu, logvar):
                # The upstream float32 VAE can overflow exp(logvar) on wide
                # synthetic domains (notably Levy [-10, 10]), producing NaN
                # latent inputs for sklearn's GP. Clamping only the numerical
                # exponent preserves the reference sampling equation.
                safe_mu = torch.nan_to_num(mu, nan=0.0, posinf=1e6, neginf=-1e6)
                safe_logvar = torch.nan_to_num(
                    logvar,
                    nan=0.0,
                    posinf=20.0,
                    neginf=-20.0,
                ).clamp(-20.0, 20.0)
                return safe_mu + torch.randn_like(safe_logvar) * torch.exp(
                    0.5 * safe_logvar
                )

            def decode(inner, z):
                return inner.decoder(inner.latent_mapping(z))

            def forward(inner, x):
                mu, logvar = inner.encode(x)
                return inner.decode(inner.sample_z(mu, logvar)), mu, logvar

        self._vae = VAE(self.dim, self.active_dim)
        self._vae_optimizer = torch.optim.Adam(
            self._vae.parameters(),
            lr=self.vae_learning_rate,
        )

    def _train_vae(self, epochs=30):
        import torch
        from torch.nn import functional as F

        self._ensure_vae()
        data = torch.tensor(self.X, dtype=torch.float32)
        self._vae.train()
        for _ in range(int(epochs)):
            for index in range(len(data)):
                x = data[index : index + 1]
                reconstructed, mu, logvar = self._vae(x)
                if not (
                    torch.isfinite(reconstructed).all()
                    and torch.isfinite(mu).all()
                    and torch.isfinite(logvar).all()
                ):
                    self._vae_optimizer.zero_grad()
                    continue
                reconstruction = F.mse_loss(reconstructed, x)
                safe_logvar = logvar.clamp(-20.0, 20.0)
                kld = torch.mean(
                    -0.5
                    * torch.sum(
                        1 + safe_logvar - mu**2 - safe_logvar.exp(),
                        dim=1,
                    )
                )
                loss = reconstruction + 0.5 * kld
                self._vae_optimizer.zero_grad()
                if not torch.isfinite(loss):
                    continue
                loss.backward()
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    self._vae.parameters(),
                    max_norm=10.0,
                    error_if_nonfinite=False,
                )
                if torch.isfinite(gradient_norm):
                    self._vae_optimizer.step()

    def _vae_points(self, q):
        import torch

        if self._vae is None or len(self.y) % self.vae_update_interval == 0:
            self._train_vae(epochs=30)
        self._vae.eval()
        with torch.no_grad():
            mu, logvar = self._vae.encode(torch.tensor(self.X, dtype=torch.float32))
            latent = self._vae.sample_z(mu, logvar).cpu().numpy()
        training_low = np.full(
            self.active_dim,
            self.vae_latent_training_bounds[0],
        )
        training_high = np.full(
            self.active_dim,
            self.vae_latent_training_bounds[1],
        )
        latent_midpoint = (training_low + training_high) / 2.0
        latent = np.where(np.isfinite(latent), latent, latent_midpoint)
        latent = np.clip(latent, training_low, training_high)
        gp = make_reference_bo_gp(
            alpha=self.noise_variance,
            learn_noise=self.learn_noise,
            dim=None,
        ).fit(latent, self.y)
        acquisition_low = np.full(
            self.active_dim,
            self.vae_latent_acquisition_bounds[0],
        )
        acquisition_high = np.full(
            self.active_dim,
            self.vae_latent_acquisition_bounds[1],
        )
        candidates = self._sobol(acquisition_low, acquisition_high)
        mu_c, std = gp.predict(candidates, return_std=True)
        std = np.maximum(std, 1e-12)
        improvement = mu_c - float(np.max(gp.predict(latent))) - 1e-4
        z = improvement / std
        scores = improvement * norm.cdf(z) + std * norm.pdf(z)
        selected = candidates[np.argsort(scores)[-int(q) :]]
        with torch.no_grad():
            decoded = self._vae.decode(torch.tensor(selected, dtype=torch.float32)).cpu().numpy()
        decoded_midpoint = (self.bounds[:, 0] + self.bounds[:, 1]) / 2.0
        decoded = np.where(np.isfinite(decoded), decoded, decoded_midpoint)
        return np.clip(decoded, self.bounds[:, 0], self.bounds[:, 1])

    def _propose(self, q):
        if self.method == "saasbo":
            return self._saas_points(q)
        if self.method == "vae-bo":
            return self._vae_points(q)
        dims = self._selected_dims()
        if self.method.endswith("-rs"):
            selected = self.rng.uniform(self.bounds[dims, 0], self.bounds[dims, 1], size=(q, len(dims)))
        else:
            selected = self._ei_points(dims, q)
        return np.asarray([self._fill(dims, row) for row in selected], dtype=float)

    def _record_stopping_snapshot(self, candidate_indices):
        if not self.enable_posterior_stopping or len(self.y) < max(2, self.n_init):
            return None
        entry = _posterior_threshold_snapshot(
            self.X,
            self.y,
            noise_variance=self.noise_variance,
            risk_tolerances=self.risk_tolerances,
            best_by_delta=self.best_stoppable_threshold_by_delta,
            use_ard=False,
            candidate_indices=[int(index) for index in candidate_indices],
            stopping_threshold=self.stopping_threshold,
            learn_noise=self.learn_noise,
        )
        if entry is not None:
            self.threshold_trace.append(entry)
            self.stopped = bool(entry.get("stopped", False))
            if self.stopped:
                self.stop_iteration = int(entry["n"])
        return entry

    def _finalize_reference_evaluations(self, budgeted_objective):
        """Publish live reference evaluations through the common result contract."""

        if budgeted_objective.points:
            self.X = np.asarray(budgeted_objective.points, dtype=float).reshape(
                -1, self.dim
            )
        else:
            self.X = np.empty((0, self.dim), dtype=float)
        self.y = np.asarray(budgeted_objective.values, dtype=float).reshape(-1)
        if len(self.y) != self.max_iter:
            raise RuntimeError(
                f"{self.method} completed {len(self.y)} objective evaluations; "
                f"the caller requested exactly {self.max_iter}."
            )
        self.history = [
            {
                "n": index + 1,
                "best_y": float(np.max(self.y[: index + 1])),
                "stopping_info": None,
            }
            for index in range(len(self.y))
        ]
        return self

    def _optimize_lamcts(self):
        """Execute vendored LA-MCTS-BO/TuRBO with an exact budget guard."""

        lamcts_class = _load_native_lamcts()
        uses_turbo = self.method == "la-mcts-turbo"
        budgeted_objective = _BudgetedReferenceObjective(
            self.objective,
            self.bounds,
            self.max_iter,
        )
        agent = None
        try:
            try:
                agent = lamcts_class(
                    lb=budgeted_objective.lb,
                    ub=budgeted_objective.ub,
                    dims=self.dim,
                    ninits=20,
                    func=budgeted_objective,
                    Cp=self.la_mcts_cp,
                    leaf_size=20,
                    solver_type="turbo" if uses_turbo else "bo",
                    turbo_max_evals=50,
                    kernel_type="rbf",
                    gamma_type="auto",
                )
                # Upstream checks max_samples only after a complete leaf epoch.
                # The objective facade prevents an over-budget evaluation.
                if len(budgeted_objective.values) < self.max_iter:
                    agent.search(max_samples=self.max_iter)
            except _ExactBudgetReached:
                pass
        finally:
            pass
        self.la_mcts_agent = agent
        self._finalize_reference_evaluations(budgeted_objective)
        if uses_turbo:
            self.threshold_trace = lamcts_local_turbo_posterior_trace(
                self.X,
                self.y,
                outer_initial_count=20,
                turbo_epoch_budget=self.turbo_max_evals,
                turbo_batch_size=10,
                noise_variance=self.noise_variance,
                risk_tolerances=self.risk_tolerances,
                learn_noise=self.learn_noise,
            )
        return self

    @_isolates_legacy_rng
    def optimize(self):
        if self.method in {"la-mcts-bo", "la-mcts-turbo"}:
            return self._optimize_lamcts()
        points, values = [], []
        for x in self._initial_design():
            points.append(np.asarray(x, dtype=float).copy())
            values.append(float(self.objective(x)))
        self.X, self.y = np.asarray(points), np.asarray(values)
        if self.method.startswith(("dropout",)):
            self._uipt_solver = _BestKImputationStrategy(self.dim, k=20)
            self._uipt_solver.init_strategy(self.X, self.y)
        stopping_by_n = {}
        initial_entry = self._record_stopping_snapshot(range(len(self.y)))
        if initial_entry is not None:
            stopping_by_n[int(initial_entry["n"])] = initial_entry
        while len(self.y) < self.max_iter and not self.stopped:
            batch_start = len(self.y)
            if self.method in {"dropout-turbo"}:
                epoch_budget = min(self.turbo_max_evals, self.max_iter - len(self.y))
                dims = self._selected_dims()
                proposed, observed = self._native_turbo_epoch(dims, epoch_budget)
                self.X = np.vstack([self.X, proposed])
                self.y = np.concatenate([self.y, observed])
            elif self.method in {"dropout-bo", "dropout-rs"}:
                q = min(self.batch_size, self.max_iter - len(self.y))
                dims = self._selected_dims()
                if self.method.endswith("-rs"):
                    selected = self.rng.uniform(
                        self.bounds[dims, 0],
                        self.bounds[dims, 1],
                        size=(q, len(dims)),
                    )
                else:
                    selected = self._ei_points(dims, q)
                proposed_rows, observed_values = [], []
                for row in selected:
                    point = self._fill(dims, row)
                    value = float(self.objective(point))
                    proposed_rows.append(point)
                    observed_values.append(value)
                    self.X = np.vstack([self.X, point])
                    self.y = np.append(self.y, value)
                    self._uipt_solver.update(point, value)
                proposed = np.asarray(proposed_rows, dtype=float)
                observed = np.asarray(observed_values, dtype=float)
            else:
                q = min(self.batch_size, self.max_iter - len(self.y))
                proposed = self._propose(q)
                observed = np.asarray([float(self.objective(x)) for x in proposed], dtype=float)
                self.X = np.vstack([self.X, proposed])
                self.y = np.concatenate([self.y, observed])
            entry = self._record_stopping_snapshot(range(batch_start, len(self.y)))
            if entry is not None:
                stopping_by_n[int(entry["n"])] = entry
        self.history = [
            {
                "n": i + 1,
                "best_y": float(np.max(self.y[: i + 1])),
                "stopping_info": stopping_by_n.get(i + 1),
            }
            for i in range(len(self.y))
        ]
        return self


# Public algorithm variants are intentionally colocated with their shared
# implementation. Their constructor settings are unchanged from the former
# one-class forwarding modules.
class DropoutBOOptimizer(NativeBaselineOptimizer):
    def __init__(self, *args, **kwargs):
        kwargs.update(method="dropout-bo", n_init=10, batch_size=3)
        super().__init__(*args, **kwargs)


class DropoutTuRBOOptimizer(NativeBaselineOptimizer):
    def __init__(self, *args, **kwargs):
        kwargs.update(method="dropout-turbo", n_init=10, batch_size=10)
        super().__init__(*args, **kwargs)


class DropoutRandomSearchOptimizer(NativeBaselineOptimizer):
    def __init__(self, *args, **kwargs):
        kwargs.update(
            method="dropout-rs",
            n_init=10,
            batch_size=3,
            enable_posterior_stopping=False,
            stopping_threshold=None,
        )
        super().__init__(*args, **kwargs)






class LAMCTSBOOptimizer(NativeBaselineOptimizer):
    def __init__(self, *args, **kwargs):
        kwargs.update(method="la-mcts-bo", n_init=20, batch_size=1)
        super().__init__(*args, **kwargs)


class LAMCTSTuRBOOptimizer(NativeBaselineOptimizer):
    def __init__(self, *args, **kwargs):
        kwargs.update(method="la-mcts-turbo", n_init=20, batch_size=10)
        super().__init__(*args, **kwargs)


