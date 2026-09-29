"""Native TuRBO-1 proposals for AG-VS selected subspaces."""

from __future__ import annotations

from copy import deepcopy
import warnings

import numpy as np

from Experiment.benchmark.baseline.reference_infrastructure import TURBO_REFERENCE_SPEC


def _configured_point_batch_size(optimizer):
    q_batch = getattr(optimizer, "q_batch", None)
    batch_size = getattr(optimizer, "batch_size", None)
    if (q_batch is None) == (batch_size is None):
        raise ValueError(
            "AG-VS-TuRBO requires exactly one of q_batch or batch_size"
        )
    return int(q_batch if q_batch is not None else batch_size)


_NUMERICAL_RETRY_JITTERS = (1e-3, 1e-2)


def _train_gp_with_numerical_retry(train_gp, **kwargs):
    """Retry only GPyTorch non-PSD failures with explicit extra jitter."""

    import gpytorch

    try:
        return train_gp(**kwargs), 0.0
    except Exception as exc:
        if type(exc).__name__ != "NotPSDError":
            raise
        last_error = exc

    for jitter in _NUMERICAL_RETRY_JITTERS:
        try:
            with gpytorch.settings.cholesky_jitter(
                float_value=float(jitter),
                double_value=float(jitter),
            ):
                return train_gp(**kwargs), float(jitter)
        except Exception as exc:
            if type(exc).__name__ != "NotPSDError":
                raise
            last_error = exc
    raise last_error


class TurboNodeGP:
    """Sklearn-like view of the sole native TuRBO acquisition GP."""

    def __init__(
        self,
        gp,
        lows,
        highs,
        target_median,
        target_scale,
        X_train,
        cholesky_jitter=0.0,
    ):
        self.gp = gp
        self.lows = np.asarray(lows, dtype=float)
        self.highs = np.asarray(highs, dtype=float)
        self.target_median = float(target_median)
        self.target_scale = float(target_scale)
        self.X_train_ = np.asarray(X_train, dtype=float)
        self.cholesky_jitter = float(cholesky_jitter)

    @property
    def lengthscales(self):
        raw = (
            self.gp.covar_module.base_kernel.lengthscale.detach()
            .cpu()
            .numpy()
            .reshape(-1)
            .astype(float)
        )
        if raw.size == 1:
            return np.full(self.X_train_.shape[1], float(raw[0]), dtype=float)
        return raw

    @property
    def noise_variance(self):
        return float(self.gp.likelihood.noise.detach().cpu().reshape(-1)[0]) * (
            self.target_scale ** 2
        )

    def predict(self, X, return_std=False):
        import gpytorch
        import torch
        from gpytorch.utils.warnings import GPInputWarning

        X = np.asarray(X, dtype=float)
        X_unit = np.clip((X - self.lows) / (self.highs - self.lows), 0.0, 1.0)
        parameter = next(self.gp.parameters())
        X_torch = torch.as_tensor(X_unit, device=parameter.device, dtype=parameter.dtype)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", GPInputWarning)
            with (
                torch.no_grad(),
                gpytorch.settings.fast_pred_var(),
                gpytorch.settings.cholesky_jitter(
                    float_value=max(self.cholesky_jitter, 1e-6),
                    double_value=max(self.cholesky_jitter, 1e-8),
                ),
            ):
                posterior = self.gp.likelihood(self.gp(X_torch))
        mean = -(self.target_median + self.target_scale * posterior.mean.detach().cpu().numpy())
        if not return_std:
            return np.asarray(mean, dtype=float)
        std = self.target_scale * posterior.variance.clamp_min(0.0).sqrt().detach().cpu().numpy()
        return np.asarray(mean, dtype=float), np.asarray(std, dtype=float)


def fit_turbo_node_gp(optimizer, node):
    """Fit the single native isotropic GP used by the selected subspace."""
    X_local, y_local = _training_data(optimizer, node)
    if X_local is None:
        return None

    import torch
    from Experiment.benchmark.baseline.native.turbo.gp import train_gp

    bounds = np.asarray(optimizer._node_bounds(node), dtype=float)
    lows, highs = bounds[:, 0], bounds[:, 1]
    X_unit = np.clip((X_local - lows) / (highs - lows), 0.0, 1.0)
    fX = -np.asarray(y_local, dtype=float).reshape(-1)
    target_median = float(np.median(fX))
    target_scale = float(np.std(fX))
    if target_scale < 1e-6:
        target_scale = 1.0
    standardized = (fX - target_median) / target_scale
    if len(X_unit) < optimizer.turbo_min_cuda:
        # Native TuRBO forces small-data GP training to CPU float64.
        device, dtype = torch.device("cpu"), torch.float64
    else:
        device = torch.device(optimizer.turbo_device)
        dtype = torch.float64 if str(TURBO_REFERENCE_SPEC.dtype).lower() == "float64" else torch.float32
    warm_source = getattr(node, "gp_warm_start_model", None)
    warm_hypers = {}
    if isinstance(warm_source, TurboNodeGP):
        try:
            warm_hypers = deepcopy(warm_source.gp.state_dict())
        except Exception:
            warm_hypers = {}
    gp, cholesky_jitter = _train_gp_with_numerical_retry(
        train_gp,
        train_x=torch.as_tensor(X_unit, device=device, dtype=dtype),
        train_y=torch.as_tensor(standardized, device=device, dtype=dtype),
        use_ard=False,
        num_steps=TURBO_REFERENCE_SPEC.n_training_steps,
        hypers=warm_hypers,
    )
    model = TurboNodeGP(
        gp,
        lows,
        highs,
        target_median,
        target_scale,
        X_local,
        cholesky_jitter=cholesky_jitter,
    )
    lengthscales = model.lengthscales
    if lengthscales.size != len(node.dims):
        return None
    node.model = model
    node.gp_kernel_name = "turbo_scalekernel_matern52_isotropic"
    node.gp_kernel_score = float("nan")
    node.lengthscales = lengthscales.copy()
    node.lengthscale_gp_kernel_name = "acquisition_gp_turbo_matern52_isotropic"
    node.lengthscale_gp_kernel_score = float("nan")
    node.gp_cholesky_jitter = float(cholesky_jitter)
    node.gp_fit_mode = (
        "numerical_jitter_retry"
        if cholesky_jitter > 0.0
        else (
            "warm_started_hyperparameter_fit"
            if warm_hypers
            else "full_hyperparameter_fit"
        )
    )
    for attribute in ("gp_incremental_source_model", "gp_warm_start_model"):
        if hasattr(node, attribute):
            delattr(node, attribute)
    return model


def _turbo_engine(optimizer, leaf):
    dims = tuple(int(dim) for dim in getattr(leaf, "dims", []) or [])
    node_bounds = np.asarray(optimizer._node_bounds(leaf), dtype=float)
    signature = (
        dims,
        tuple(node_bounds.reshape(-1).tolist()),
        optimizer.turbo_device,
        optimizer.turbo_min_cuda,
        _configured_point_batch_size(optimizer),
    )
    engine = getattr(leaf, "_native_turbo_engine", None)
    if engine is not None and getattr(leaf, "_native_turbo_signature", None) == signature:
        return engine

    try:
        from Experiment.benchmark.baseline.native_optimizers import _load_native_turbo1

        Turbo1 = _load_native_turbo1()
    except Exception as exc:
        raise ImportError(
            "AG-VS-TuRBO requires the vendored uber-research/TuRBO dependencies "
            "(torch and gpytorch) to load successfully."
        ) from exc

    engine = Turbo1(
        f=lambda _x: 0.0,
        lb=node_bounds[:, 0],
        ub=node_bounds[:, 1],
        n_init=1,
        max_evals=int(optimizer.turbo_max_evals),
        batch_size=_configured_point_batch_size(optimizer),
        verbose=False,
        use_ard=False,
        max_cholesky_size=TURBO_REFERENCE_SPEC.max_cholesky_size,
        n_training_steps=TURBO_REFERENCE_SPEC.n_training_steps,
        min_cuda=optimizer.turbo_min_cuda,
        device=optimizer.turbo_device,
        dtype=TURBO_REFERENCE_SPEC.dtype,
    )
    leaf._native_turbo_engine = engine
    leaf._native_turbo_signature = signature
    leaf.turbo_restart_count = 0
    return engine


def _training_data(optimizer, leaf):
    dims = [int(dim) for dim in getattr(leaf, "dims", []) or []]
    valid = sorted(
        {
            int(index)
            for index in getattr(leaf, "local_indices", []) or []
            if 0 <= int(index) < len(optimizer.X)
        }
    )
    # Merge/restart nodes are allowed a one-observation seed GP. They receive
    # their own full node trial before any random split.
    if len(valid) < 1:
        return None, None
    X = np.asarray(optimizer.X, dtype=float)[valid][:, dims]
    y = np.asarray(optimizer.y, dtype=float)[valid]
    return X, y


def _seeded_native_candidates(optimizer, engine, model, X_unit, fX):
    """Run an upstream routine with isolated legacy global RNG state.

    The vendored TuRBO code draws from NumPy's and Torch's process-global
    generators.  Saving and restoring every affected state keeps those legacy
    implementation details from leaking into the surrounding benchmark.
    """
    import gpytorch
    import torch
    from ag_vs.utils import sobol_engine

    standardized = (fX - model.target_median) / model.target_scale
    x_center = X_unit[int(np.argmin(standardized))][None, :]
    weights = model.lengthscales.copy()
    weights /= weights.mean()
    weights /= np.prod(np.power(weights, 1.0 / len(weights)))
    lb = np.clip(x_center - weights * float(engine.length) / 2.0, 0.0, 1.0)
    ub = np.clip(x_center + weights * float(engine.length) / 2.0, 0.0, 1.0)
    if engine.n_cand < engine.min_cuda:
        candidate_device, candidate_dtype = torch.device("cpu"), torch.float64
    else:
        candidate_device, candidate_dtype = engine.device, engine.dtype
    # ExactGP keeps its training data outside the usual module buffers, so
    # ``gp.to(...)`` alone can leave float64 training inputs behind while
    # the kernel parameters and candidates become float32.
    train_x = model.gp.train_inputs[0].to(
        device=candidate_device, dtype=candidate_dtype
    )
    train_y = model.gp.train_targets.to(
        device=candidate_device, dtype=candidate_dtype
    )
    model.gp = model.gp.to(device=candidate_device, dtype=candidate_dtype)
    model.gp.set_train_data(inputs=train_x, targets=train_y, strict=False)
    parameter = next(model.gp.parameters())
    sobol = sobol_engine(len(weights), role="agvs_turbo")
    pert = sobol.draw(engine.n_cand).to(device=parameter.device, dtype=parameter.dtype).cpu().numpy()
    pert = lb + (ub - lb) * pert
    mask = np.random.rand(engine.n_cand, len(weights)) <= min(20.0 / len(weights), 1.0)
    empty = np.where(mask.sum(axis=1) == 0)[0]
    mask[empty, np.random.randint(0, len(weights), size=len(empty))] = True
    X_cand = np.repeat(x_center, engine.n_cand, axis=0)
    X_cand[mask] = pert[mask]
    X_cand_torch = torch.as_tensor(X_cand, device=parameter.device, dtype=parameter.dtype)
    with (
        torch.no_grad(),
        gpytorch.settings.max_cholesky_size(engine.max_cholesky_size),
        gpytorch.settings.cholesky_jitter(
            float_value=max(model.cholesky_jitter, 1e-6),
            double_value=max(model.cholesky_jitter, 1e-8),
        ),
    ):
        y_cand = model.gp.likelihood(model.gp(X_cand_torch)).sample(
            torch.Size([engine.batch_size])
        ).t().cpu().numpy()
    return X_cand, model.target_median + model.target_scale * y_cand


def propose_turbo_q(optimizer, leaf, q):
    """Generate one isotropic TuRBO Thompson batch and its budget-safe prefix."""

    dims = [int(dim) for dim in getattr(leaf, "dims", []) or []]
    q = int(q)
    configured_batch_size = _configured_point_batch_size(optimizer)
    if q <= 0 or q > configured_batch_size:
        raise ValueError(
            "AG-VS-TuRBO proposal count must be in [1, "
            f"{configured_batch_size}]; got q={q}."
        )
    if not dims:
        raise RuntimeError("AG-VS-TuRBO cannot optimize a leaf with no active dimensions.")

    node_bounds = np.asarray(optimizer._node_bounds(leaf), dtype=float)
    lows, highs = node_bounds[:, 0], node_bounds[:, 1]
    X_local, y_local = _training_data(optimizer, leaf)
    if X_local is None:
        out = []
        for _ in range(q):
            point = optimizer._parent_best_base(leaf)
            point[dims] = optimizer.rng.uniform(lows, highs)
            out.append((np.clip(point, optimizer.bounds[:, 0], optimizer.bounds[:, 1]), float("inf")))
        optimizer._last_joint_q_acquisition = q > 1
        return out

    model = optimizer._fit_local_gp(leaf)
    if model is None or not isinstance(model, TurboNodeGP):
        cause = getattr(leaf, "_last_gp_fit_error", None)
        if cause is None:
            cause = getattr(optimizer, "_last_gp_fit_error", None)
        detail = "" if cause is None else f" Cause: {type(cause).__name__}: {cause}"
        raise RuntimeError(
            "AG-VS-TuRBO could not fit its shared acquisition GP." + detail
        ) from cause
    engine = _turbo_engine(optimizer, leaf)
    if engine.n_evals >= int(optimizer.turbo_max_evals) or engine.length < engine.length_min:
        engine._restart()
        engine.n_evals = 0
        leaf.turbo_restart_count = int(getattr(leaf, "turbo_restart_count", 0)) + 1

    X_unit = np.clip((X_local - lows) / (highs - lows), 0.0, 1.0)
    fX = -np.asarray(y_local, dtype=float).reshape(-1)
    engine._X = deepcopy(X_unit)
    engine._fX = deepcopy(fX.reshape(-1, 1))

    X_cand, y_cand = _seeded_native_candidates(optimizer, engine, model, X_unit, fX)
    selected_unit = engine._select_candidates(X_cand, np.asarray(y_cand, dtype=float).copy())
    selected_unit = np.asarray(selected_unit, dtype=float)
    if selected_unit.shape[0] != int(engine.batch_size):
        raise RuntimeError(
            "Vendored TuRBO-1 returned an unexpected q-batch: "
            f"expected {int(engine.batch_size)}, "
            f"got {selected_unit.shape[0]}."
        )
    selected = lows + (highs - lows) * selected_unit

    leaf._native_turbo_pending = {
        "selected_unit": selected_unit.copy(),
        "baseline_X": X_unit.copy(),
        "baseline_fX": fX.reshape(-1, 1).copy(),
    }
    out = []
    for row, sampled_column in zip(selected[:q], range(q)):
        point = optimizer._parent_best_base(leaf)
        point[dims] = row
        sampled_value = float(np.min(np.asarray(y_cand)[:, sampled_column]))
        out.append((np.clip(point, optimizer.bounds[:, 0], optimizer.bounds[:, 1]), -sampled_value))
    optimizer._last_joint_q_acquisition = q > 1
    return out


def update_turbo_after_batch(optimizer, leaf, evaluated_indices):
    """Apply native TuRBO-1 success/failure and trust-region state updates."""

    pending = getattr(leaf, "_native_turbo_pending", None)
    engine = getattr(leaf, "_native_turbo_engine", None)
    if not pending or engine is None:
        return
    indices = [int(index) for index in evaluated_indices if 0 <= int(index) < len(optimizer.y)]
    if not indices:
        return
    if len(indices) > int(engine.batch_size):
        raise RuntimeError(
            "AG-VS-TuRBO trust-region updates cannot exceed the AG-VS "
            "scheduler proposal count B="
            f"{int(engine.batch_size)}; got {len(indices)}."
        )
    fX_next = -np.asarray(optimizer.y, dtype=float)[indices].reshape(-1, 1)
    engine._X = np.asarray(pending["baseline_X"], dtype=float).copy()
    engine._fX = np.asarray(pending["baseline_fX"], dtype=float).copy()
    engine._adjust_length(fX_next)
    engine.n_evals += len(indices)
    engine._X = np.vstack([engine._X, np.asarray(pending["selected_unit"], dtype=float)[: len(indices)]])
    engine._fX = np.vstack([engine._fX, fX_next])
    leaf._native_turbo_pending = None


def propose_turbo_one(optimizer, leaf):
    return propose_turbo_q(optimizer, leaf, 1)[0]


__all__ = ["TurboNodeGP", "fit_turbo_node_gp", "propose_turbo_one", "propose_turbo_q", "update_turbo_after_batch"]
