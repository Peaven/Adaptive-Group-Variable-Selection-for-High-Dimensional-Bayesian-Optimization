"""Native GTBO, following Hellsten et al. and gtboauthors/gtbo.

SMC and group selection live in native/group_testing. Every objective query,
including calibration and duplicate observations, consumes the shared budget.
"""
from __future__ import annotations

import warnings

import numpy as np
import torch
from torch.quasirandom import SobolEngine
from ag_vs.utils import isolated_trial, acquisition_initial_conditions

GTBO_SETTINGS = dict(
    n_particles=10000, prior_activeness_rate=0.05, n_initial_groups=3,
    activeness_threshold=0.5, lower_convergence_threshold=0.001,
    upper_convergence_threshold=0.9, n_default_samples=10, n_init=10,
    gt_iterations=100, max_groups_per_iteration=5,
    maximum_mutual_information_loss=0.01, raw_samples=4096,
    num_restarts=16, retrain_gp_from_scratch_every=10,
    active_prior_parameters=(0.0, 1.0), dtype="float64", device="cpu",
    acquisition="qLogNEI", batch_size=1, mc_samples=1024,
    n_normal_samples=1000, test_only_non_converged_lb=1e-5,
    test_only_non_converged_ub=0.95,
)


class _BudgetComplete(Exception):
    pass


class GTBOOptimizer:
    def __init__(self, objective, bounds, max_iter, seed, *,
                 experiment_kind="synthetic", objective_name="", default=None,
                 **settings):
        unknown = settings.keys() - GTBO_SETTINGS.keys()
        if unknown:
            raise TypeError(f"Unknown GTBO settings: {sorted(unknown)}")
        self.settings = {**GTBO_SETTINGS, **settings}
        self.settings["inactive_prior_parameters"] = (
            (7.0, 1.0) if experiment_kind == "synthetic" else (3.0, 1.0))
        self.objective, self.seed = objective, int(seed)
        self.bounds = np.asarray(bounds, dtype=float)
        self.dim, self.max_iter = len(self.bounds), int(max_iter)
        if self.dim < 6 or self.max_iter < 1:
            raise ValueError("GTBO needs D >= 6 and a positive budget")
        self.lb_vec = torch.tensor(self.bounds[:, 0], dtype=torch.double)
        self.ub_vec = torch.tensor(self.bounds[:, 1], dtype=torch.double)
        if default is None:
            fraction = 0.25 if objective_name in {"sphere", "ackley", "rastrigin", "griewank"} else 0.5
            self.default = self.lb_vec + fraction * (self.ub_vec - self.lb_vec)
        else:
            self.default = torch.as_tensor(default, dtype=torch.double)
        self.returns_noiseless = False
        self.X = np.empty((0, self.dim))
        self.y = np.empty(0)
        self.history, self.threshold_trace = [], []
        self.stopped, self.stop_iteration = False, None
        self.phase = "calibration"
        self.active_dimensions = np.empty(0, dtype=int)
        self.marginals = np.full(self.dim, 0.05)
        self._model_state = None
        self.gp_state_reset_recovery_count = 0
        self.gp_unfitted_fallback_count = 0
        self.gp_fit_recovery_history = []
        self.gp_last_fit_error = None

    def __call__(self, x):
        if len(self.y) >= self.max_iter:
            raise _BudgetComplete
        point = np.asarray(torch.as_tensor(x).detach().cpu(), dtype=float).reshape(-1)
        if point.shape != (self.dim,) or not np.isfinite(point).all():
            raise ValueError("Invalid GTBO objective point")
        point = np.clip(point, self.bounds[:, 0], self.bounds[:, 1])
        value = float(self.objective(point))
        if not np.isfinite(value):
            raise ValueError("GTBO objective returned a nonfinite value")
        self.X = np.vstack((self.X, point))
        self.y = np.append(self.y, value)
        self.history.append(dict(n=len(self.y), best_y=float(self.y.max()), phase=self.phase))
        # Group testing uses the reference minimization convention.
        return torch.tensor(-value, dtype=torch.double)

    @isolated_trial
    def optimize(self):
        from .native.group_testing.ground_truth_evaluator import GroundTruthEvaluator
        from .native.group_testing.tester import GroupTester
        cfg = self.settings
        try:
            evaluator = GroundTruthEvaluator(
                self, self.default, n_default_samples=cfg["n_default_samples"])
            if len(self.y) >= self.max_iter:
                return self
            self.phase = "group_testing"
            keys = ("n_particles", "prior_activeness_rate", "n_initial_groups",
                    "activeness_threshold", "lower_convergence_threshold",
                    "upper_convergence_threshold", "max_groups_per_iteration",
                    "maximum_mutual_information_loss", "n_normal_samples",
                    "test_only_non_converged_lb", "test_only_non_converged_ub")
            tester = GroupTester(
                evaluator=evaluator,
                noise_std=max(float(evaluator.noise_std), 1e-6),
                signal_std=max(float(torch.sqrt(evaluator.f_std**2 + evaluator.pure_noise_std**2)), 1e-6),
                gt_iterations=min(cfg["gt_iterations"], max(0, self.max_iter-len(self.y)-cfg["n_init"])),
                **{key: cfg[key] for key in keys},
            )
            state = tester.optimize()
            self.marginals = state.marginals[0].detach().cpu().numpy()
            self.active_dimensions = np.flatnonzero(self.marginals >= cfg["activeness_threshold"])
            # Deduplicate only training data. The evaluation ledger is immutable.
            first = min(cfg["n_default_samples"], len(self.y))
            signs = np.sign((self.X[first:] - self.default.numpy())[:, self.active_dimensions])
            _, indices = np.unique(signs, axis=0, return_index=True)
            training_indices = (first + np.sort(indices)).tolist()
            self.phase = "bo_initialization"
            engine = SobolEngine(self.dim, scramble=True, seed=self.seed)
            for unit in engine.draw(min(cfg["n_init"], self.max_iter-len(self.y))).double():
                training_indices.append(len(self.y))
                self(self.lb_vec + unit * (self.ub_vec-self.lb_vec))
            self.phase = "bayesian_optimization"
            iteration = 0
            while len(self.y) < self.max_iter:
                if iteration % cfg["retrain_gp_from_scratch_every"] == 0:
                    self._model_state = None
                candidate = self._candidate(training_indices)
                training_indices.append(len(self.y))
                self(candidate)
                iteration += 1
        except _BudgetComplete:
            pass
        return self

    def _make_gp(self, x, y, *, load_state):
        from botorch.models import SingleTaskGP
        from gpytorch.kernels import MaternKernel, ScaleKernel
        from gpytorch.priors import LogNormalPrior

        cfg = self.settings
        loc = torch.full((self.dim,), cfg["inactive_prior_parameters"][0], dtype=torch.double)
        loc[self.active_dimensions] = cfg["active_prior_parameters"][0]
        kernel = ScaleKernel(MaternKernel(
            nu=2.5, ard_num_dims=self.dim,
            lengthscale_prior=LogNormalPrior(loc, torch.ones_like(loc)),
        )).double()
        kernel.base_kernel.initialize(lengthscale=loc.exp().reshape(1, -1))
        model = SingleTaskGP(x, y, covar_module=kernel, outcome_transform=None)
        if load_state and self._model_state is not None:
            model.load_state_dict(self._model_state)
        return model

    def _fit_gp(self, x, y):
        """Fit the learned-noise GP, resetting a failed warm state if needed."""

        from botorch.exceptions.errors import ModelFittingError
        from botorch.fit import fit_gpytorch_mll
        from gpytorch.mlls import ExactMarginalLogLikelihood

        used_warm_state = self._model_state is not None
        failures = []
        for load_state in ([True, False] if used_warm_state else [False]):
            model = self._make_gp(x, y, load_state=load_state)
            try:
                fit_gpytorch_mll(
                    ExactMarginalLogLikelihood(model.likelihood, model)
                )
            except ModelFittingError as exc:
                failures.append(f"{type(exc).__name__}: {exc}")
                if load_state:
                    self._model_state = None
                    continue
                break

            recovery = "fresh_refit" if failures else "none"
            if recovery == "fresh_refit":
                self.gp_state_reset_recovery_count += 1
            self.gp_last_fit_error = failures[-1] if failures else None
            if failures:
                self.gp_fit_recovery_history.append(
                    {
                        "n_evaluations": int(len(self.y)),
                        "used_warm_state": bool(used_warm_state),
                        "recovery": recovery,
                        "fit_error": self.gp_last_fit_error,
                    }
                )
            self._model_state = {
                key: value.detach().clone()
                for key, value in model.state_dict().items()
            }
            return model.eval()

        # The learned-noise GP has a positive initialized noise and therefore
        # remains usable when SciPy rejects every hyperparameter fit.  Do not
        # persist this state: the next iteration should attempt a fresh fit.
        self._model_state = None
        self.gp_unfitted_fallback_count += 1
        self.gp_last_fit_error = failures[-1]
        self.gp_fit_recovery_history.append(
            {
                "n_evaluations": int(len(self.y)),
                "used_warm_state": bool(used_warm_state),
                "recovery": "initialized_model",
                "fit_error": self.gp_last_fit_error,
            }
        )
        warnings.warn(
            "GTBO GP hyperparameter fitting failed after resetting its cached "
            "state; using the initialized learned-noise GP for this acquisition "
            "step.",
            RuntimeWarning,
        )
        return model.eval()

    def _candidate(self, indices):
        from botorch.acquisition import qLogNoisyExpectedImprovement
        from botorch.sampling.normal import SobolQMCNormalSampler
        from botorch.optim import optimize_acqf

        cfg = self.settings
        x = torch.tensor(self.X[indices], dtype=torch.double)
        x = (x-self.lb_vec)/(self.ub_vec-self.lb_vec)
        y = torch.tensor(self.y[indices], dtype=torch.double).unsqueeze(-1)
        y = (y-y.mean()) / y.std(unbiased=False).clamp_min(1e-8)
        model = self._fit_gp(x, y)
        acquisition = qLogNoisyExpectedImprovement(
            model, X_baseline=x, prune_baseline=True,
            sampler=SobolQMCNormalSampler(torch.Size([cfg["mc_samples"]]), seed=self.seed))
        unit_bounds = torch.stack((torch.zeros(self.dim), torch.ones(self.dim))).double()
        initial = acquisition_initial_conditions(acquisition, unit_bounds,
            cfg["num_restarts"], cfg["raw_samples"], role="gtbo_acquisition")
        candidate, _ = optimize_acqf(
            acquisition, bounds=unit_bounds, batch_initial_conditions=initial,
            q=1, num_restarts=cfg["num_restarts"], raw_samples=None,
            options={"seed": self.seed, "batch_limit": 16})
        return self.lb_vec + candidate.detach().reshape(-1) * (self.ub_vec-self.lb_vec)
