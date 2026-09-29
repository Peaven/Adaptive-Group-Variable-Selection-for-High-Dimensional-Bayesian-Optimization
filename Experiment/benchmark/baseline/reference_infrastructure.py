"""Verified inner-optimizer settings shared by the fair benchmark variants.

This module contains project-owned adapters only.  The vendored MCTS-VS and
TuRBO trees remain byte-for-byte at their recorded upstream revisions.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional

from ag_vs.utils import trial_seed
import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel


@dataclass(frozen=True)
class BOReferenceSpec:
    surrogate_family: str = "sklearn.gaussian_process.GaussianProcessRegressor"
    kernel: str = "ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5)"
    kernel_nu: float = 2.5
    ard: bool = False
    initial_output_scale: float = 1.0
    initial_lengthscale: float = 1.0
    observation_noise_std: float = 0.01
    alpha: float = 1e-4
    normalize_y: bool = False
    hyperparameter_fitting: str = "marginal likelihood (sklearn fmin_l_bfgs_b)"
    acquisition: str = "classical analytic Expected Improvement"
    xi: float = 1e-4
    candidate_count: int = 1024
    candidate_generation: str = (
        "PyTorch scrambled Sobol; SciPy is a compatibility fallback only when "
        "PyTorch is absent and is not claimed to produce the identical stream"
    )
    candidate_seed_policy: str = (
        "fresh scrambled-Sobol engine per acquisition; seed drawn from the "
        "advancing trial-owned NumPy Generator"
    )
    candidate_selection: str = (
        "top-k highest pointwise classical analytic EI (k=optimizer point "
        "batch size; not joint qEI)"
    )


@dataclass(frozen=True)
class TuRBOReferenceSpec:
    implementation: str = "vendored uber-research/TuRBO TuRBO-1"
    surrogate_family: str = "GPyTorch ExactGP"
    gp_library: str = "GPyTorch on PyTorch"
    gp_kernel: str = "ScaleKernel(MaternKernel(nu=2.5, ARD=True))"
    use_ard: bool = True
    internal_batch_size: int = 10
    final_batch_budget_rule: str = (
        "generate the native batch of 10, then evaluate only the remaining "
        "prefix to preserve the exact scalar objective-call budget; this "
        "terminal partial batch cannot affect a later proposal"
    )
    n_training_steps: int = 50
    max_cholesky_size: int = 2000
    min_cuda: int = 1024
    device: str = "cpu"
    dtype: str = "float32"
    small_data_runtime: str = "CPU float64 below min_cuda=1024 (native TuRBO behavior)"
    candidate_pool: str = "min(100 * selected_dim, 5000)"
    candidate_generation: str = "TuRBO trust-region scrambled Sobol perturbations"
    candidate_selection: str = "GP posterior Thompson sampling"
    trust_region_logic: str = "native TuRBO-1 expansion, contraction, and restart"
    turbo_max_evals: int = 50


BO_REFERENCE_SPEC = BOReferenceSpec()
TURBO_REFERENCE_SPEC = TuRBOReferenceSpec()

# Real-life objectives own their stochasticity (dataset repetitions or seeded
# environment rollouts), so their sklearn GPs estimate observation noise from
# the collected responses.  Synthetic reference runs use the configured
# fixed alpha unless ``learn_noise`` is requested explicitly.
REAL_LIFE_GP_JITTER = 1e-10
REAL_LIFE_GP_NOISE_BOUNDS = (1e-8, 1e1)


def mcts_cp_for_benchmark(benchmark_name: Optional[str]) -> float:
    """Appendix Table 3 MCTS exploration constants.

    A generic/unknown benchmark uses the algorithm default 0.1; its
    benchmark-specific value is otherwise not explicitly specified.
    """

    name = "" if benchmark_name is None else str(benchmark_name).strip().lower()
    if "levy" in name:
        return 10.0
    if any(token in name for token in ("hopper", "walker", "mujoco")):
        return 50.0
    return 0.1  # Hartmann and NAS; generic default when not explicitly specified.


def bo_reference_dict() -> dict:
    return asdict(BO_REFERENCE_SPEC)


def turbo_reference_dict(*, device: Optional[str] = None) -> dict:
    values = asdict(TURBO_REFERENCE_SPEC)
    if device is not None:
        values["device"] = str(device)
    return values


def make_reference_bo_gp(
    *,
    alpha: float = BO_REFERENCE_SPEC.alpha,
    learn_noise: bool = False,
    dim: Optional[int] = None,
) -> GaussianProcessRegressor:
    """Construct the MCTS-VS-style GP with the configured benchmark variance.

    Kernel bounds and output normalization intentionally use sklearn's defaults,
    matching ``mcts_vs_official/baseline/vanilia_bo.py``.
    """

    length_scale = (
        BO_REFERENCE_SPEC.initial_lengthscale
        if dim is None
        else np.full(int(dim), BO_REFERENCE_SPEC.initial_lengthscale, dtype=float)
    )
    signal_kernel = ConstantKernel(BO_REFERENCE_SPEC.initial_output_scale) * Matern(
        length_scale=length_scale,
        nu=BO_REFERENCE_SPEC.kernel_nu,
    )
    kernel = signal_kernel
    if learn_noise:
        kernel = signal_kernel + WhiteKernel(
            noise_level=max(float(alpha), REAL_LIFE_GP_NOISE_BOUNDS[0]),
            noise_level_bounds=REAL_LIFE_GP_NOISE_BOUNDS,
        )
    return GaussianProcessRegressor(
        kernel=kernel,
        alpha=(REAL_LIFE_GP_JITTER if learn_noise else float(alpha)),
        normalize_y=BO_REFERENCE_SPEC.normalize_y,
        n_restarts_optimizer=0,
        random_state=trial_seed(),
    )


def sklearn_learned_noise_variance(
    model: GaussianProcessRegressor,
    *,
    fallback: Optional[float] = None,
) -> Optional[float]:
    """Return the fitted WhiteKernel variance from a sklearn GP, if present."""

    fitted = getattr(model, "kernel_", None)
    stack = [fitted] if fitted is not None else []
    while stack:
        kernel = stack.pop()
        if isinstance(kernel, WhiteKernel):
            value = float(kernel.noise_level)
            if np.isfinite(value) and value >= 0.0:
                return value
        for name in ("k1", "k2"):
            child = getattr(kernel, name, None)
            if child is not None:
                stack.append(child)
    return None if fallback is None else float(fallback)


def sample_reference_bo_candidates(
    bounds: np.ndarray,
    *,
    rng: np.random.Generator,
    count: int = BO_REFERENCE_SPEC.candidate_count,
) -> np.ndarray:
    """Draw the MCTS-VS scrambled-Sobol candidate pool without RNG leakage.

    PyTorch's ``SobolEngine`` is the reference engine.  SciPy QMC is an
    explicit compatibility fallback only when PyTorch cannot be imported; its
    scrambled stream is not claimed to be identical.
    """

    bounds = np.asarray(bounds, dtype=float)
    if bounds.ndim != 2 or bounds.shape[1] != 2:
        raise ValueError("bounds must have shape (dimension, 2)")
    if int(count) != BO_REFERENCE_SPEC.candidate_count:
        raise ValueError(
            f"Reference BO requires exactly {BO_REFERENCE_SPEC.candidate_count} candidates."
        )
    seed = int(rng.integers(0, 500000))
    try:
        import torch
        from torch.quasirandom import SobolEngine

        unit = (
            SobolEngine(bounds.shape[0], scramble=True, seed=seed)
            .draw(int(count))
            .to(dtype=torch.float64)
            .cpu()
            .numpy()
        )
    except ImportError:
        from scipy.stats import qmc

        unit = qmc.Sobol(bounds.shape[0], scramble=True, seed=seed).random(int(count))
    return bounds[:, 0] + np.asarray(unit, dtype=float) * (bounds[:, 1] - bounds[:, 0])


def classical_expected_improvement(
    model: GaussianProcessRegressor,
    X_sample: np.ndarray,
    candidates: np.ndarray,
    *,
    xi: float = BO_REFERENCE_SPEC.xi,
) -> np.ndarray:
    """Pointwise analytic EI as implemented by the original MCTS-VS baseline."""

    X_sample = np.asarray(X_sample, dtype=float)
    candidates = np.asarray(candidates, dtype=float)
    mean, std = model.predict(candidates, return_std=True)
    std = np.asarray(std, dtype=float).reshape(-1)
    sample_mean = np.asarray(model.predict(X_sample), dtype=float).reshape(-1)
    incumbent = float(np.max(sample_mean))
    improvement = np.asarray(mean, dtype=float).reshape(-1) - incumbent - float(xi)
    values = np.zeros_like(improvement)
    positive = std > 0.0
    z = np.zeros_like(improvement)
    z[positive] = improvement[positive] / std[positive]
    values[positive] = (
        improvement[positive] * norm.cdf(z[positive])
        + std[positive] * norm.pdf(z[positive])
    )
    return values


def assert_bo_fairness(left: dict, right: dict) -> None:
    """Assert every user-requested BO-fairness field is identical."""

    keys = (
        "surrogate_family",
        "kernel",
        "kernel_nu",
        "ard",
        "initial_output_scale",
        "initial_lengthscale",
        "observation_noise_std",
        "alpha",
        "normalize_y",
        "hyperparameter_fitting",
        "acquisition",
        "xi",
        "candidate_count",
        "candidate_generation",
        "candidate_selection",
    )
    assert {key: left[key] for key in keys} == {key: right[key] for key in keys}


def assert_turbo_fairness(left: dict, right: dict) -> None:
    """Assert every user-requested TuRBO-fairness field is identical."""

    keys = (
        "implementation",
        "gp_kernel",
        "use_ard",
        "internal_batch_size",
        "final_batch_budget_rule",
        "n_training_steps",
        "max_cholesky_size",
        "min_cuda",
        "device",
        "dtype",
        "small_data_runtime",
        "candidate_pool",
        "candidate_generation",
        "candidate_selection",
        "trust_region_logic",
        "turbo_max_evals",
    )
    assert {key: left[key] for key in keys} == {key: right[key] for key in keys}


__all__ = [
    "BO_REFERENCE_SPEC",
    "TURBO_REFERENCE_SPEC",
    "assert_bo_fairness",
    "assert_turbo_fairness",
    "bo_reference_dict",
    "classical_expected_improvement",
    "make_reference_bo_gp",
    "sklearn_learned_noise_variance",
    "REAL_LIFE_GP_JITTER",
    "REAL_LIFE_GP_NOISE_BOUNDS",
    "mcts_cp_for_benchmark",
    "sample_reference_bo_candidates",
    "turbo_reference_dict",
]
