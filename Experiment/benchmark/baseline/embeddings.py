"""Native HeSBO and ALEBO loops, derived from the supplied Ax 0.2.2 sources.

No Ax strategies or managed-loop APIs are used. ALEBO retains its learned
Mahalanobis metric, Laplace samples, moment matching, and feasible polytope.
"""
from __future__ import annotations
import copy
import warnings

import numpy as np
import torch

from ag_vs.utils import isolated_trial, acquisition_initial_conditions


# GPyTorch already rounds the former 1e-7 HeSBO variance up to 1e-6.  NAS-Bench
# can still produce ill-conditioned fits because its continuous embedding is
# decoded onto a discrete architecture space.  Keep that effective reference
# floor first, then retry only failed fits with progressively larger numerical
# nuggets.  The reported objective-noise model remains fixed/known.
HESBO_NUMERICAL_TRAIN_YVAR_SCHEDULE = (1e-6, 1e-4, 1e-3)


class EmbeddingOptimizer:
    def __init__(self, objective, bounds, max_iter, seed, active_dim,
                 method="hesbo", fit_restarts=10, laplace_nsamp=25,
                 raw_samples=1000, num_restarts=10, **unused):
        self.objective, self.seed = objective, int(seed)
        self.bounds = np.asarray(bounds, dtype=float)
        self.dim, self.active_dim = len(self.bounds), min(int(active_dim), len(bounds))
        self.max_iter, self.method = int(max_iter), method
        self.learn_noise = False
        self.n_init = 2 if method == "hesbo" else 10
        self.fit_restarts, self.laplace_nsamp = fit_restarts, laplace_nsamp
        self.raw_samples, self.num_restarts = raw_samples, num_restarts
        self.X, self.y = np.empty((0, self.dim)), np.empty(0)
        self.history, self.threshold_trace = [], []
        self.stopped, self.stop_iteration = False, None
        self.rng = np.random.default_rng(self.seed)
        self._map_state = None
        self.hesbo_fit_retry_count = 0
        self.hesbo_unfitted_fallback_count = 0
        self.hesbo_numerical_train_yvar_history = []
        self.hesbo_last_fit_error = None
        self._warm_start_low = None
        self.warm_start_metadata = None

    def warm_start_alebo_from_ambient_history(self, X, y, *, rank_tolerance=1e-6):
        """Recover ALEBO's search subspace from stored ambient observations.

        Historical ALEBO traces did not persist their random projection matrix.
        Their ambient points nevertheless lie in the same ``active_dim``-rank
        linear subspace. An SVD recovers an equivalent orthonormal basis, which
        lets the optimizer retain every old observation and continue without
        querying the physical objective again.
        """

        if self.method != "alebo":
            raise ValueError("ambient-history warm start is supported only for ALEBO")
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        if X.ndim != 2 or X.shape[1] != self.dim:
            raise ValueError(
                f"ALEBO warm-start X must have shape (n, {self.dim}); got {X.shape}"
            )
        if X.shape[0] != y.size or y.size < self.n_init:
            raise ValueError(
                "ALEBO warm-start X/y lengths must match and include the initial design"
            )
        if y.size >= self.max_iter:
            raise ValueError("ALEBO warm-start history must be below max_iter")
        if not np.all(np.isfinite(X)) or not np.all(np.isfinite(y)):
            raise ValueError("ALEBO warm-start history must contain only finite values")

        span = self.bounds[:, 1] - self.bounds[:, 0]
        if np.any(span <= 0.0):
            raise ValueError("ALEBO bounds must have positive width")
        ambient = 2.0 * (X - self.bounds[:, 0]) / span - 1.0
        if np.max(np.abs(ambient)) > 1.00001:
            raise ValueError("ALEBO warm-start points fall outside the ambient bounds")

        _u, singular_values, vh = np.linalg.svd(ambient, full_matrices=False)
        if vh.shape[0] < self.active_dim:
            raise ValueError("ALEBO warm-start history cannot identify the search subspace")
        basis = vh[: self.active_dim]
        low = ambient @ basis.T
        reconstructed = low @ basis
        reconstruction_error = float(np.max(np.abs(reconstructed - ambient)))
        if reconstruction_error > float(rank_tolerance):
            numerical_rank = int(np.sum(singular_values > float(rank_tolerance)))
            raise ValueError(
                "ALEBO warm-start points do not lie in the configured embedding "
                f"subspace: rank={numerical_rank}, active_dim={self.active_dim}, "
                f"max_abs_reconstruction_error={reconstruction_error}"
            )

        self.B = torch.tensor(basis, dtype=torch.double)
        self.Binv = torch.linalg.pinv(self.B)
        self.X = X.copy()
        self.y = y.copy()
        self._warm_start_low = torch.tensor(low, dtype=torch.double)
        self.low_X = self._warm_start_low.clone()
        running_best = np.maximum.accumulate(self.y)
        self.history = [
            {"n": index + 1, "best_y": float(value)}
            for index, value in enumerate(running_best)
        ]

        # Restore the NumPy generator position used by feasible-set sampling.
        # Restore the exact number of uniform variates consumed by the old
        # run: the initial design requests n_init feasible points, while each
        # completed post-initial acquisition requests raw_samples candidates.
        post_initial_calls = max(0, int(y.size) - int(self.n_init))
        initial_draws = max(10000, 10 * int(self.n_init)) * int(self.dim)
        acquisition_draws = (
            post_initial_calls
            * max(10000, 10 * int(self.raw_samples))
            * int(self.dim)
        )
        advance = getattr(self.rng.bit_generator, "advance", None)
        if callable(advance):
            advance(initial_draws + acquisition_draws)

        self.warm_start_metadata = {
            "mode": "svd_recovered_ambient_subspace",
            "source_evaluations": int(y.size),
            "active_dim": int(self.active_dim),
            "numerical_rank": int(
                np.sum(singular_values > float(rank_tolerance))
            ),
            "rank_tolerance": float(rank_tolerance),
            "max_abs_reconstruction_error": reconstruction_error,
            "rng_feasible_calls_advanced": int(1 + post_initial_calls),
            "rng_uniform_variates_advanced": int(
                initial_draws + acquisition_draws
            ),
        }
        return dict(self.warm_start_metadata)

    def _initialize_projection(self):
        d, D = self.active_dim, self.dim
        if self.method == "hesbo":
            self.A = torch.zeros(D, d, dtype=torch.double)
            self.hash_indices = torch.randint(d, (D,))
            self.hash_signs = 2 * torch.randint(2, (D,)) - 1
            self.A[torch.arange(D), self.hash_indices] = self.hash_signs.double()
        else:
            self.B = torch.randn(d, D, dtype=torch.double)
            self.B /= self.B.norm(dim=0)
            self.Binv = torch.linalg.pinv(self.B)

    def _feasible_alebo(self, n):
        # Source initialization: project a large ambient uniform sample then
        # repeatedly halve the sampling box until enough points are feasible.
        ambient = torch.tensor(self.rng.uniform(-1, 1, (max(10000, 10*n), self.dim)), dtype=torch.double)
        low = ambient @ self.B.T
        high = low @ self.Binv.T
        scale = 16.0
        while True:
            feasible = (high.abs() * scale <= 1).all(dim=1)
            if int(feasible.sum()) >= n:
                return low[feasible][:n] * scale
            scale /= 2

    def _evaluate(self, low):
        high = low @ (self.A.T if self.method == "hesbo" else self.Binv.T)
        if high.abs().max() > 1.00001:
            raise RuntimeError("Embedding candidate violates ambient bounds")
        unit = (high.clamp(-1, 1).numpy() + 1) / 2
        x = self.bounds[:, 0] + unit * (self.bounds[:, 1]-self.bounds[:, 0])
        value = float(self.objective(x))
        self.X = np.vstack((self.X, x))
        self.y = np.append(self.y, value)
        self.history.append(dict(n=len(self.y), best_y=float(self.y.max())))

    @isolated_trial
    def optimize(self):
        from torch.quasirandom import SobolEngine
        if self._warm_start_low is not None:
            low = self._warm_start_low.clone()
        else:
            self._initialize_projection()
            count = min(self.n_init, self.max_iter)
            if self.method == "hesbo":
                low = 2 * SobolEngine(self.active_dim, scramble=True, seed=self.seed).draw(count).double()-1
            else:
                low = self._feasible_alebo(count)
            for point in low:
                self._evaluate(point)
        while len(self.y) < self.max_iter:
            candidate = self._candidate(low).detach().reshape(-1)
            self._evaluate(candidate)
            low = torch.cat((low, candidate[None]))
        self.low_X = low
        return self

    def _fit_alebo(self, x, y):
        from .native.alebo_model import ALEBOGP
        from botorch.fit import fit_gpytorch_mll
        from gpytorch.mlls import ExactMarginalLogLikelihood
        best_loss, model = float("inf"), None
        for _ in range(self.fit_restarts if self._map_state is None else 1):
            current = ALEBOGP(self.B, x, y, torch.full_like(y, 1e-7))
            if self._map_state is not None:
                current.load_state_dict(self._map_state)
            mll = ExactMarginalLogLikelihood(current.likelihood, current)
            fit_gpytorch_mll(mll)
            current.train()
            loss = -mll(current(x), current.train_targets)
            if float(loss.detach()) < best_loss:
                best_loss, model = float(loss.detach()), current
        self._map_state = copy.deepcopy(model.state_dict())
        # Diagonal Laplace approximation of U, holding mean and scale at MAP.
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        model.train()
        U = model.covar_module.base_kernel.Uvec
        loss = -mll(model(x), model.train_targets)
        gradient = torch.autograd.grad(loss, U, create_graph=True)[0]
        diagonal = torch.stack([
            torch.autograd.grad(gradient[i], U, retain_graph=True)[0][i]
            for i in range(U.numel())
        ]).detach()
        # Upstream adds a 1e-3 nugget before inversion. Clamp negative numerical
        # curvature to that nugget to keep the covariance positive definite.
        std = (diagonal + 1e-3).clamp_min(1e-3).rsqrt()
        samples = U.detach() + torch.randn(self.laplace_nsamp, U.numel(), dtype=torch.double) * std
        samples[0] = U.detach()
        ns = self.laplace_nsamp
        batch = ALEBOGP(self.B, x.repeat(ns, 1, 1), y.repeat(ns, 1, 1), torch.full_like(y, 1e-7).repeat(ns, 1, 1))
        with torch.no_grad():
            batch.covar_module.base_kernel.Uvec.copy_(samples)
            batch.mean_module.constant.copy_(model.mean_module.constant.expand_as(batch.mean_module.constant))
            batch.covar_module.raw_outputscale.copy_(model.covar_module.raw_outputscale.expand_as(batch.covar_module.raw_outputscale))
        return batch.eval()

    def _fit_hesbo(self, x, y):
        """Fit HeSBO with a deterministic numerical-nugget fallback ladder."""

        from botorch.models import SingleTaskGP
        from botorch.fit import fit_gpytorch_mll
        from gpytorch.kernels import MaternKernel, ScaleKernel
        from gpytorch.mlls import ExactMarginalLogLikelihood

        last_model = None
        failures = []
        for index, variance in enumerate(HESBO_NUMERICAL_TRAIN_YVAR_SCHEDULE):
            model = SingleTaskGP(
                x,
                y,
                train_Yvar=torch.full_like(y, float(variance)),
                covar_module=ScaleKernel(
                    MaternKernel(nu=2.5, ard_num_dims=self.active_dim)
                ),
                outcome_transform=None,
            )
            last_model = model
            try:
                fit_gpytorch_mll(
                    ExactMarginalLogLikelihood(model.likelihood, model)
                )
            except Exception as exc:
                failures.append(f"{type(exc).__name__}: {exc}")
                continue

            self.hesbo_fit_retry_count += index
            self.hesbo_numerical_train_yvar_history.append(float(variance))
            self.hesbo_last_fit_error = failures[-1] if failures else None
            return model.eval()

        # A positive 1e-3 fixed nugget makes the initialized GP posterior valid
        # even if SciPy rejects every hyperparameter fit.  Continuing with those
        # initialized hyperparameters is preferable to discarding a partially
        # completed 150-evaluation trial.
        self.hesbo_fit_retry_count += len(HESBO_NUMERICAL_TRAIN_YVAR_SCHEDULE) - 1
        self.hesbo_unfitted_fallback_count += 1
        self.hesbo_numerical_train_yvar_history.append(
            float(HESBO_NUMERICAL_TRAIN_YVAR_SCHEDULE[-1])
        )
        self.hesbo_last_fit_error = failures[-1] if failures else None
        warnings.warn(
            "HeSBO GP hyperparameter fitting failed at every numerical noise "
            "level; using the initialized GP with train_Yvar=1e-3 for this "
            "acquisition step.",
            RuntimeWarning,
        )
        return last_model.eval()

    def _candidate(self, x):
        from botorch.acquisition.analytic import ExpectedImprovement
        from botorch.optim import optimize_acqf
        y = torch.tensor(self.y, dtype=torch.double).unsqueeze(-1)
        y = (y-y.mean())/y.std(unbiased=False).clamp_min(1e-8)
        if self.method == "hesbo":
            model = self._fit_hesbo(x, y)
            acquisition = ExpectedImprovement(model, best_f=y.max())
            low_bounds = torch.tensor([[-1.0]*self.active_dim, [1.0]*self.active_dim], dtype=torch.double)
            initial = acquisition_initial_conditions(acquisition, low_bounds,
                self.num_restarts, self.raw_samples, role="hesbo_acquisition")
            candidate, _ = optimize_acqf(acquisition,
                bounds=low_bounds, batch_initial_conditions=initial,
                q=1, num_restarts=self.num_restarts, raw_samples=None,
                options={"seed": self.seed})
        else:
            model = self._fit_alebo(x, y)
            acq = ExpectedImprovement(model, best_f=y.max())
            initial = self._feasible_alebo(self.raw_samples).unsqueeze(1)
            with torch.no_grad():
                scores = acq(initial)
            initial = initial[torch.topk(scores, min(self.num_restarts, len(scores))).indices]
            indices = torch.arange(self.active_dim)
            constraints = [(indices, sign*row, -1.0) for row in self.Binv for sign in (-1, 1)]
            extent = self.B.abs().sum(dim=1)
            candidate, _ = optimize_acqf(acq, bounds=torch.stack((-extent, extent)),
                q=1, num_restarts=len(initial), raw_samples=None,
                batch_initial_conditions=initial, inequality_constraints=constraints,
                options={"method": "SLSQP", "batch_limit": 1, "seed": self.seed})
        return candidate


class HeSBOOptimizer(EmbeddingOptimizer):
    def __init__(self, *args, **kwargs):
        kwargs["method"] = "hesbo"
        super().__init__(*args, **kwargs)


class ALEBOOptimizer(EmbeddingOptimizer):
    def __init__(self, *args, **kwargs):
        kwargs["method"] = "alebo"
        super().__init__(*args, **kwargs)
