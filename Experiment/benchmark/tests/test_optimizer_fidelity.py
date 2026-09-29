from __future__ import annotations

import math
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from Experiment.benchmark.baseline.embeddings import (
    ALEBOOptimizer,
    HESBO_NUMERICAL_TRAIN_YVAR_SCHEDULE,
    HeSBOOptimizer,
)
from scipy.linalg import cho_factor

from ag_vs.optimizer import AGVSOptimizer
from ag_vs.inner_optimizer.turbo.proposal import propose_turbo_q, update_turbo_after_batch
from ag_vs.utils import calculate_score, latin_hypercube_sample, sobol_sample, weighted_score
from Experiment.benchmark.baseline import native_baselines as paper_adapters
from Experiment.benchmark.baseline import native_optimizers as official_adapters
from Experiment.benchmark.baseline.add_gp_ucb import ADDGPUCBOptimizer
from Experiment.benchmark.baseline.native_baselines import (
    LAMCTSBOOptimizer,
    LAMCTSTuRBOOptimizer,
)
from Experiment.benchmark.baseline.native_optimizers import (
    NativeMCTSVSOptimizer,
    NativeTurbo1Optimizer,
)
from Experiment.benchmark.baseline.native_optimizers import OfficialCMAESOptimizer
from Experiment.benchmark.baseline.native_optimizers import _isolated_legacy_rng
from Experiment.benchmark.baseline.reference_infrastructure import (
    BO_REFERENCE_SPEC,
    assert_bo_fairness,
    assert_turbo_fairness,
    make_reference_bo_gp,
    mcts_cp_for_benchmark,
    sample_reference_bo_candidates,
)
from Experiment.benchmark.baseline.native_baselines import (
    lamcts_local_turbo_posterior_trace,
)
from Experiment.benchmark.baseline.vanilla_bo_adapter import VanillaBOOptimizer


ROOT = Path(__file__).resolve().parents[3]


def _bounds(dimension=10):
    return np.repeat([[-1.0, 1.0]], int(dimension), axis=0)


def test_hesbo_retries_failed_gp_fit_with_stable_numerical_nugget(monkeypatch):
    import botorch.fit
    import torch

    calls = []

    def fail_once_then_fit(mll, *_args, **_kwargs):
        calls.append(float(mll.model.likelihood.noise.mean()))
        if len(calls) == 1:
            raise RuntimeError("simulated NAS-Bench numerical fit failure")
        mll.eval()
        return mll

    monkeypatch.setattr(botorch.fit, "fit_gpytorch_mll", fail_once_then_fit)
    optimizer = HeSBOOptimizer(
        lambda x: -float(np.sum(np.asarray(x) ** 2)),
        _bounds(6),
        4,
        2021,
        2,
    )
    train_x = torch.tensor(
        [[-0.8, -0.2], [0.1, 0.4], [0.7, 0.9]], dtype=torch.double
    )
    train_y = torch.tensor([[-1.0], [0.0], [1.0]], dtype=torch.double)

    model = optimizer._fit_hesbo(train_x, train_y)

    assert model.training is False
    assert calls == pytest.approx(
        list(HESBO_NUMERICAL_TRAIN_YVAR_SCHEDULE[:2])
    )
    assert optimizer.hesbo_fit_retry_count == 1
    assert optimizer.hesbo_unfitted_fallback_count == 0
    assert optimizer.hesbo_numerical_train_yvar_history == [1e-4]
    assert "simulated NAS-Bench" in optimizer.hesbo_last_fit_error


def test_plain_turbo_reuses_native_gp_for_threshold_reporting(monkeypatch):
    def fail_auxiliary_sklearn_fit(*_args, **_kwargs):
        raise AssertionError("plain TuRBO must not fit an auxiliary sklearn GP")

    monkeypatch.setattr(
        official_adapters,
        "_posterior_threshold_snapshot",
        fail_auxiliary_sklearn_fit,
    )
    optimizer = NativeTurbo1Optimizer(
        f=lambda x: -float(np.sum((np.asarray(x, dtype=float) - 0.25) ** 2)),
        lb=np.zeros(3),
        ub=np.ones(3),
        n_init=20,
        max_evals=30,
        batch_size=10,
        use_ard=True,
        n_training_steps=30,
        min_cuda=1024,
        device="cpu",
        dtype="float64",
        seed=7,
        risk_tolerances=[1 / 256],
        stopping_threshold=1e100,
        enable_posterior_stopping=True,
    ).optimize()

    assert [entry["n"] for entry in optimizer.threshold_trace] == [20, 30]
    assert all(
        entry["posterior_source"].startswith("same native GPyTorch GP fit")
        for entry in optimizer.threshold_trace
    )
    assert all(entry["gp_device"] == "cpu" for entry in optimizer.threshold_trace)


def test_lamcts_reports_local_turbo_posterior_at_each_batch_end():
    rng = np.random.default_rng(7)
    X = rng.random((50, 3))
    y = -np.sum((X - 0.3) ** 2, axis=1)

    trace = lamcts_local_turbo_posterior_trace(
        X,
        y,
        outer_initial_count=20,
        turbo_epoch_budget=50,
        turbo_batch_size=10,
        risk_tolerances=[1 / 256],
    )

    assert [entry["n"] for entry in trace] == [30, 40, 50]
    assert all(np.isfinite(entry["posterior_mean_at_best"]) for entry in trace)
    assert trace[-1]["local_turbo_training_count"] == 30
    assert trace[-1]["posterior_candidate_indices"] == list(range(40, 50))


def _agvs(effective_dimension, backend="vanillabo", **overrides):
    settings = dict(
        objective=lambda x: -float(np.sum(np.asarray(x) ** 2)),
        bounds=_bounds(10),
        eff_dim=int(effective_dimension),
        max_iter=30,
        n_init=10,
        noise_std=0.01,
        tau=0.0,
        risk_tolerances=[0.05],
        sim_mode=False,
        n_split=int(effective_dimension) + 1,
        verbose=False,
        printer_on=False,
        print_tree_each_iter=False,
        ipt_solver=backend,
        q_batch=None,
        batch_size=3,
        t_seq=3,
    )
    settings.update(overrides)
    return AGVSOptimizer(**settings)


def test_reference_bo_is_the_requested_classical_ei_stack():
    gp = make_reference_bo_gp()
    assert gp.alpha == pytest.approx(1e-4)
    assert gp.normalize_y is False
    assert gp.kernel.k1.constant_value == pytest.approx(1.0)
    assert gp.kernel.k2.length_scale == pytest.approx(1.0)
    assert gp.kernel.k2.nu == pytest.approx(2.5)
    assert BO_REFERENCE_SPEC.ard is False
    assert BO_REFERENCE_SPEC.xi == pytest.approx(1e-4)
    assert BO_REFERENCE_SPEC.candidate_count == 1024
    assert "pointwise classical analytic EI" in BO_REFERENCE_SPEC.candidate_selection
    assert "not joint qEI" in BO_REFERENCE_SPEC.candidate_selection


def test_default_gp_variance_reaches_agvs_and_native_bo_models():
    from ag_vs.GaussianProcess import GaussianProcessManager
    from Experiment.benchmark.baseline.native.vanilla import get_gpr_model
    from Experiment.benchmark.baseline.native.lamcts.Classifier import Classifier
    from Experiment.benchmark import runner

    models = [
        GaussianProcessManager().make_gp(3),
        get_gpr_model(),
        Classifier([], 3, "rbf").gpr,
    ]
    for model in models:
        assert model.alpha == pytest.approx(1e-4)
    assert BO_REFERENCE_SPEC.observation_noise_std == pytest.approx(0.01)
    assert BO_REFERENCE_SPEC.observation_noise_std**2 == pytest.approx(BO_REFERENCE_SPEC.alpha)
    settings = runner.algorithm_settings_summary("LA-MCTS-BO", "mopta08_deff4")
    assert settings["alpha"] == pytest.approx(1e-4)
    # An explicit API variance remains an override, rather than being squared.
    assert GaussianProcessManager(noise_variance=0.02).make_gp(3).alpha == pytest.approx(0.02)


def test_reference_candidate_stream_uses_rng_seeded_pytorch_scrambled_sobol():
    torch = pytest.importorskip("torch")
    bounds = _bounds(3)
    seed_rng = np.random.default_rng(23)
    torch_seed = int(seed_rng.integers(0, 500000))
    expected_unit = (
        torch.quasirandom.SobolEngine(3, scramble=True, seed=torch_seed)
        .draw(1024)
        .to(dtype=torch.float64)
        .cpu()
        .numpy()
    )
    expected = bounds[:, 0] + expected_unit * (bounds[:, 1] - bounds[:, 0])
    actual = sample_reference_bo_candidates(
        bounds,
        rng=np.random.default_rng(23),
    )
    assert np.array_equal(actual, expected)


@pytest.mark.parametrize(
    "sampler",
    [
        lambda: sample_reference_bo_candidates(_bounds(2), rng=np.random.default_rng(5)),
        lambda: sobol_sample(_bounds(2), 8, rng=np.random.default_rng(5)),
    ],
)
def test_sobol_fallback_does_not_hide_broken_pytorch(monkeypatch, sampler):
    import builtins

    original_import = builtins.__import__

    def broken_torch_import(name, *args, **kwargs):
        if name == "torch" or name.startswith("torch."):
            raise RuntimeError("simulated broken PyTorch installation")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", broken_torch_import)
    with pytest.raises(RuntimeError, match="broken PyTorch"):
        sampler()


@pytest.mark.parametrize(
    "sampler",
    [
        lambda: latin_hypercube_sample(_bounds(2), 4),
        lambda: sobol_sample(_bounds(2), 8),
    ],
)
def test_stochastic_helpers_require_caller_owned_rng(sampler):
    with pytest.raises(ValueError, match="caller-owned"):
        sampler()


@pytest.mark.parametrize("effective_dimension", [3, 6])
def test_agvs_experiment_settings_use_batch_size_mode(effective_dimension):
    optimizer = _agvs(effective_dimension)
    assert optimizer.q_batch is None
    assert optimizer.t_seq == 3
    assert optimizer.selection_batch_size == 3
    assert optimizer.batch_size == 3
    assert optimizer.acquisition_batch_mode == "batch_size"
    assert optimizer.node_backend == "vanillabo"


@pytest.mark.parametrize(
    "override,match",
    [
        ({"candidate_count": 2048}, "1024"),
        ({"noise_variance": -0.01}, "finite and non-negative"),
        ({"node_backend": "random"}, "vanillabo.*turbo"),
    ],
)
def test_agvs_rejects_retired_or_unfair_surfaces(override, match):
    with pytest.raises(ValueError, match=match):
        _agvs(3, **override)


def test_agvs_uses_supplied_noise_variance_and_defaults_to_one_e_minus_four():
    default_optimizer = _agvs(3, noise_std=None)
    configured_optimizer = _agvs(
        3,
        noise_std=None,
        noise_variance=1.0e-2,
    )

    assert default_optimizer.noise_variance == pytest.approx(1.0e-4)
    assert default_optimizer.noise_std == pytest.approx(1.0e-2)
    assert configured_optimizer.noise_variance == pytest.approx(1.0e-2)
    assert configured_optimizer.noise_std == pytest.approx(1.0e-1)
    assert configured_optimizer.gp_manager.make_gp(3).alpha == pytest.approx(1.0e-2)


def test_agvs_uses_the_mcts_acquisition_for_each_backend():
    ag_bo = _agvs(3)
    ag_turbo = _agvs(3, backend="turbo")
    mcts = NativeMCTSVSOptimizer(
        f=lambda x: 0.0,
        lb=np.zeros(5),
        ub=np.ones(5),
        max_iter=30,
        benchmark_name="hartmann6",
        enable_posterior_stopping=False,
    )
    assert ag_bo.bo_infrastructure["ard"] is False
    assert ag_bo.bo_infrastructure["isotropic"] is True
    assert "shared length_scale" in ag_bo.bo_infrastructure["kernel"]
    assert ag_bo.bo_infrastructure["acquisition"] == mcts.bo_infrastructure["acquisition"]
    assert ag_turbo.turbo_infrastructure["gp_kernel"] != mcts.turbo_infrastructure["gp_kernel"]
    assert ag_turbo.turbo_infrastructure["use_ard"] is False
    assert ag_turbo.turbo_infrastructure["anisotropic"] is False
    assert ag_turbo.turbo_infrastructure["isotropic"] is True
    assert ag_turbo.turbo_infrastructure["internal_batch_size"] == 3
    assert mcts.turbo_infrastructure["internal_batch_size"] == 10


def test_agvs_bo_proposes_three_independently_scored_ei_candidates():
    optimizer = _agvs(3, max_iter=13, seed=19)
    optimizer.X = [
        np.linspace(-0.9, 0.9, optimizer.D),
        np.linspace(0.8, -0.8, optimizer.D),
        np.zeros(optimizer.D),
    ]
    optimizer.y = [-1.0, -0.7, 0.0]
    optimizer.root.local_indices = [0, 1, 2]
    optimizer.root.score_indices = [0, 1, 2]

    proposals = optimizer._propose_q_from_leaf(optimizer.root, optimizer.batch_size)

    assert len(proposals) == 3
    assert len({tuple(np.asarray(point)) for point, _value in proposals}) == 3
    assert optimizer._last_joint_q_acquisition is False
    assert all(np.isfinite(value) and value >= 0.0 for _point, value in proposals)


def test_agvs_batch_size_mode_is_independent_and_mutually_exclusive_with_q_batch():
    optimizer = _agvs(
        3,
        batch_size=3,
        t_seq=2,
        max_iter=16,
        seed=23,
    )
    result = optimizer.optimize()
    records = result.history[-6:]
    assert optimizer.acquisition_batch_mode == "batch_size"
    assert all(record["q_batch"] is None for record in records)
    assert all(record["batch_size"] == 3 for record in records)
    assert all(record["joint_q_acquisition"] is False for record in records)
    assert {record["sequence_index"] for record in records} == {1, 2}

    with pytest.raises(ValueError, match="exactly one"):
        _agvs(3, q_batch=3, batch_size=3)
    with pytest.raises(ValueError, match="exactly one"):
        _agvs(3, q_batch=None, batch_size=None)


def test_agvs_turbo_evaluates_and_updates_one_complete_native_batch(monkeypatch):
    optimizer = _agvs(3, backend="turbo", max_iter=21)
    assert optimizer.selection_batch_size == 3
    assert optimizer._evaluation_batch_size(optimizer.root, remaining_budget=100) == 3
    assert optimizer._evaluation_batch_size(optimizer.root, remaining_budget=1) == 1

    optimizer.X = np.array(
        [
            [-0.8, -0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [-0.4, -0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.2, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.6, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    optimizer.y = np.array([-1.0, -0.4, 0.2, 0.6], dtype=float)
    monkeypatch.setattr(optimizer, "_parent_best_base", lambda _leaf: np.zeros(10))

    class RecordingEngine:
        batch_size = 3
        n_evals = 0
        length = 0.8
        length_min = 0.5**7

        def _select_candidates(self, candidates, _draws):
            return np.asarray(candidates, dtype=float)[:3]

        def _adjust_length(self, values):
            self.adjusted_values = np.asarray(values, dtype=float).copy()

    engine = RecordingEngine()
    leaf = SimpleNamespace(
        dims=[0, 1],
        local_bounds=None,
        local_indices=[0, 1, 2, 3],
        _native_turbo_engine=engine,
        turbo_restart_count=0,
    )
    node_bounds = optimizer._node_bounds(leaf)
    leaf._native_turbo_signature = (
        (0, 1),
        tuple(node_bounds.reshape(-1).tolist()),
        optimizer.turbo_device,
        optimizer.turbo_min_cuda,
        optimizer.batch_size,
    )
    unit_candidates = np.linspace(0.05, 0.95, 40).reshape(20, 2)
    posterior_draws = np.tile(np.linspace(1.0, 0.0, 20).reshape(-1, 1), (1, 3))
    from ag_vs.inner_optimizer.turbo.proposal import TurboNodeGP

    shared_model = object.__new__(TurboNodeGP)
    shared_model.target_median = 0.0
    shared_model.target_scale = 1.0
    monkeypatch.setattr(optimizer, "_fit_local_gp", lambda _leaf: shared_model)
    monkeypatch.setattr(
        "ag_vs.inner_optimizer.turbo.proposal._seeded_native_candidates",
        lambda *_args, **_kwargs: (unit_candidates, posterior_draws),
    )

    proposals = propose_turbo_q(optimizer, leaf, 3)
    assert len(proposals) == 3
    evaluated_points = np.asarray([point for point, _ in proposals], dtype=float)
    evaluated_values = np.asarray(
        [float(optimizer.objective(point)) for point in evaluated_points],
        dtype=float,
    )
    first_new_index = len(optimizer.y)
    optimizer.X = np.vstack([optimizer.X, evaluated_points])
    optimizer.y = np.concatenate([optimizer.y, evaluated_values])
    update_turbo_after_batch(
        optimizer,
        leaf,
        range(first_new_index, first_new_index + 3),
    )

    assert engine.n_evals == 3
    assert engine.adjusted_values.shape == (3, 1)
    assert np.array_equal(engine.adjusted_values[:, 0], -evaluated_values)
    assert engine._X.shape[0] == 7
    assert engine._fX.shape[0] == 7

    final_proposal = propose_turbo_q(optimizer, leaf, 1)
    assert len(final_proposal) == 1
    final_point = np.asarray(final_proposal[0][0], dtype=float)
    final_value = float(optimizer.objective(final_point))
    final_index = len(optimizer.y)
    optimizer.X = np.vstack([optimizer.X, final_point])
    optimizer.y = np.append(optimizer.y, final_value)
    update_turbo_after_batch(optimizer, leaf, [final_index])
    assert engine.n_evals == 4
    assert engine.adjusted_values.shape == (1, 1)
    assert engine.adjusted_values[0, 0] == pytest.approx(-final_value)


def test_agvs_turbo_final_partial_batch_respects_exact_objective_budget(monkeypatch):
    objective_calls = []

    def objective(x):
        objective_calls.append(np.asarray(x, dtype=float).copy())
        return -float(np.sum(np.asarray(x, dtype=float) ** 2))

    optimizer = _agvs(3, backend="turbo", max_iter=21, objective=objective)
    observed_batch_sizes = []
    proposal = np.zeros(optimizer.D, dtype=float)

    monkeypatch.setattr(
        optimizer,
        "_propose_q_from_leaf",
        lambda _leaf, q: [(proposal.copy(), 0.0) for _ in range(int(q))],
    )
    monkeypatch.setattr(
        optimizer,
        "_propose_from_leaf",
        lambda _leaf: (proposal.copy(), 0.0),
    )
    monkeypatch.setattr(
        optimizer,
        "_after_proposal_batch",
        lambda _leaf, indices: observed_batch_sizes.append(len(indices)),
    )
    result = optimizer.optimize()

    assert len(objective_calls) == 21
    assert len(result.y) == 21
    assert observed_batch_sizes == [3, 3, 3, 2]


def test_local_and_isolated_rng_ownership_prevents_global_leakage():
    random.seed(1234)
    np.random.seed(5678)
    expected_python = random.random()
    expected_numpy = float(np.random.random())

    random.seed(1234)
    np.random.seed(5678)
    _agvs(3, seed=99)
    assert random.random() == pytest.approx(expected_python)
    assert float(np.random.random()) == pytest.approx(expected_numpy)

    random.seed(4321)
    np.random.seed(8765)
    expected_python = random.random()
    expected_numpy = float(np.random.random())
    random.seed(4321)
    np.random.seed(8765)
    with _isolated_legacy_rng(9):
        random.random()
        np.random.random()
    assert random.random() == pytest.approx(expected_python)
    assert float(np.random.random()) == pytest.approx(expected_numpy)


def test_vanilla_bo_and_agvs_bo_share_gp_noise_and_acquisition():
    vanilla = VanillaBOOptimizer(
        objective=lambda x: 0.0,
        bounds=_bounds(3),
        max_iter=10,
        n_init=10,
        enable_posterior_stopping=False,
        verbose=False,
    )
    agvs = _agvs(3)
    assert vanilla.bo_infrastructure["alpha"] == agvs.bo_infrastructure["alpha"]
    assert vanilla.bo_infrastructure["kernel_nu"] == agvs.bo_infrastructure["kernel_nu"]
    assert vanilla.bo_infrastructure["acquisition"] == agvs.bo_infrastructure["acquisition"]
    assert agvs.bo_infrastructure["ard"] is False
    assert agvs.bo_infrastructure["isotropic"] is True


def test_vae_bo_wide_domain_numerical_guard_keeps_gp_inputs_finite():
    optimizer = paper_adapters.NativeBaselineOptimizer(
        objective=lambda x: -float(np.sum(np.asarray(x, dtype=float) ** 2)),
        bounds=np.repeat([[-10.0, 10.0]], 4, axis=0),
        max_iter=20,
        method="vae-bo",
        seed=7,
        active_dim=2,
        n_init=10,
        batch_size=3,
        enable_posterior_stopping=False,
    )
    optimizer.X = np.linspace(-9.0, 9.0, 40).reshape(10, 4)
    optimizer.y = -np.sum(optimizer.X**2, axis=1)
    optimizer._ensure_vae()
    for parameter in optimizer._vae.parameters():
        parameter.data.fill_(float("nan"))

    points = optimizer._vae_points(3)

    assert points.shape == (3, 4)
    assert np.all(np.isfinite(points))
    assert np.all(points >= -10.0)
    assert np.all(points <= 10.0)


def test_vanilla_bo_completes_exactly_ten_lhs_points_before_first_ei_batch(monkeypatch):
    initial = np.linspace(-0.9, 0.9, 30).reshape(10, 3)
    optimizer = VanillaBOOptimizer(
        objective=lambda x: float(np.sum(x)),
        bounds=_bounds(3),
        max_iter=13,
        n_init=10,
        q_batch=3,
        initial_points=initial,
        random_state=31,
        enable_posterior_stopping=False,
        verbose=False,
    )
    fitted_gp = object()
    calls = []

    def fake_fit():
        return fitted_gp if len(optimizer.y) >= optimizer.n_init else None

    def fake_propose(q, gp):
        calls.append((len(optimizer.y), int(q), gp))
        if gp is None:
            start = len(optimizer.y)
            return optimizer.initial_points[start : start + q].copy()
        return np.full((q, optimizer.dim), 0.75)

    monkeypatch.setattr(optimizer, "_fit_gp", fake_fit)
    monkeypatch.setattr(optimizer, "_propose_batch", fake_propose)
    result = optimizer.optimize()

    assert [(n, q) for n, q, _ in calls] == [(0, 3), (3, 3), (6, 3), (9, 1), (10, 3)]
    assert all(gp is None for _, _, gp in calls[:4])
    assert calls[4][2] is fitted_gp
    assert np.array_equal(result.X[:10], initial)
    assert np.array_equal(result.X[10:], np.full((3, 3), 0.75))


def test_mcts_reference_defaults_and_appendix_cp_mapping():
    assert mcts_cp_for_benchmark("levy10") == pytest.approx(10.0)
    assert mcts_cp_for_benchmark("hartmann6") == pytest.approx(0.1)
    assert mcts_cp_for_benchmark("nasbench101") == pytest.approx(0.1)
    assert mcts_cp_for_benchmark("hopper") == pytest.approx(50.0)
    assert mcts_cp_for_benchmark("walker2d") == pytest.approx(50.0)
    optimizer = NativeMCTSVSOptimizer(
        f=lambda x: 0.0,
        lb=np.zeros(5),
        ub=np.ones(5),
        max_iter=30,
        benchmark_name="levy10",
        enable_posterior_stopping=False,
    )
    assert (optimizer.feature_batch_size, optimizer.sample_batch_size) == (2, 3)
    assert optimizer.min_num_variables == 3
    assert optimizer.select_right_threshold == 5
    assert optimizer.k == 20
    assert optimizer.split_type == "mean"
    assert optimizer.uipt_solver == "bestk"
    assert optimizer.cp == pytest.approx(10.0)


@pytest.mark.parametrize("inner_solver", ["bo", "rs", "turbo"])
def test_mcts_objective_callback_enforces_exact_scalar_budget(inner_solver):
    calls = []

    def objective(x):
        calls.append(np.asarray(x, dtype=float).copy())
        return -float(np.sum(np.asarray(x, dtype=float) ** 2))

    noise_variance = 1e-4
    optimizer = NativeMCTSVSOptimizer(
        f=objective,
        lb=np.zeros(5),
        ub=np.ones(5),
        max_iter=13,
        ipt_solver=inner_solver,
        benchmark_name="hartmann6",
        noise_variance=noise_variance,
        turbo_device="cpu",
        enable_posterior_stopping=False,
    ).optimize(max_samples=13)

    assert len(calls) == 13
    assert len(optimizer.y) == 13


def test_mcts_flushes_posterior_snapshot_for_final_partial_sample_batch():
    optimizer = NativeMCTSVSOptimizer(
        f=lambda x: -float(np.sum(np.asarray(x, dtype=float) ** 2)),
        lb=np.zeros(5),
        ub=np.ones(5),
        max_iter=13,
        ipt_solver="rs",
        benchmark_name="hartmann6",
        noise_variance=1.0e-4,
        enable_posterior_stopping=True,
    ).optimize(max_samples=13)

    assert [entry["n"] for entry in optimizer.threshold_trace] == [12, 13]
    assert optimizer.threshold_trace[-1]["posterior_candidate_indices"] == [12]


def test_score_formula_empty_and_weighted_frozen_baseline_behavior():
    assert calculate_score([]) == math.inf
    records = [
        {"mu_i": 12.0, "mu_best_i_minus_1": 10.0},
        {"mu_i": 5.0, "mu_best_i_minus_1": -10.0},
    ]
    assert calculate_score(records) == pytest.approx(8.5)
    assert calculate_score(records, abs_mode=False) == pytest.approx(8.5)
    assert calculate_score(
        [{"mu_i": 5.0, "mu_best_i_minus_1": 10.0}],
        abs_mode=True,
    ) == pytest.approx(5.0)
    assert calculate_score(
        [{"mu_i": 5.0, "mu_best_i_minus_1": 10.0}],
        abs_mode=False,
    ) == pytest.approx(-5.0)
    assert weighted_score([(20.0, 2), (50.0, 3)]) == pytest.approx(38.0)
    assert weighted_score([(math.inf, 2), (50.0, 0)]) == math.inf


def test_add_gp_uses_rbf_beta_schedule_and_component_direct(monkeypatch):
    optimizer = ADDGPUCBOptimizer(
        objective=lambda x: 0.0,
        bounds=_bounds(5),
        max_iter=10,
        seed=7,
        enable_posterior_stopping=False,
    )
    optimizer.X = np.linspace(-0.9, 0.9, 50).reshape(10, 5)
    optimizer.y = np.linspace(-1.0, 1.0, 10)
    optimizer.noise_variance = 0.01
    optimizer.bandwidth = 0.7
    optimizer.kernel_scale = 1.3
    expected = 1.3 * math.exp(-0.5 * (0.4**2) / (0.7**2))
    actual = optimizer._group_kernel(
        np.array([[0.0]]),
        np.array([[0.4]]),
        np.array([0]),
    )[0, 0]
    assert actual == pytest.approx(expected)
    assert optimizer.beta_t(4, 11) == pytest.approx(4 * math.log(22) / 5)

    calls = []

    def fake_direct(func, bounds, **kwargs):
        midpoint = np.asarray([(low + high) / 2 for low, high in bounds])
        assert np.isfinite(func(midpoint))
        calls.append(kwargs)
        return SimpleNamespace(x=midpoint, success=True)

    monkeypatch.setattr("Experiment.benchmark.baseline.add_gp_ucb.direct", fake_direct)
    point, selected_point_std = optimizer._propose()
    assert point.shape == (5,)
    assert selected_point_std >= 0.0
    assert len(calls) == len(optimizer.groups)
    expected_budget = math.ceil(min(5000, 100 * optimizer.dim) / len(optimizer.groups))
    assert all(call["maxfun"] == expected_budget for call in calls)
    assert all(call["locally_biased"] is False for call in calls)


def test_add_gp_derives_reference_common_noise_from_initial_values():
    initial = np.linspace(-1.0, 1.0, 50).reshape(10, 5)
    optimizer = ADDGPUCBOptimizer(
        objective=lambda x: float(np.sum(x)),
        bounds=_bounds(5),
        max_iter=10,
        initial_points=initial,
        seed=2,
        enable_posterior_stopping=False,
    ).optimize()
    assert optimizer.noise_variance == pytest.approx(0.01 * np.std(optimizer.y, ddof=1))
    assert "combined posterior uses the data mean" in optimizer.reference_settings["mean"]


def test_add_gp_individual_and_combined_means_match_matlab_output_functions():
    """Regress computeIndGPOutput.m versus computeCombGPOutput.m semantics."""

    optimizer = ADDGPUCBOptimizer(
        objective=lambda x: 0.0,
        bounds=_bounds(2),
        max_iter=10,
        seed=3,
        enable_posterior_stopping=False,
    )
    optimizer.X = np.array([[0.0, 0.0]])
    optimizer.y = np.array([7.0])
    optimizer.groups = [np.array([0]), np.array([1])]
    optimizer.bandwidth = 1.0
    optimizer.kernel_scale = 1.0
    optimizer._y_mean = 7.0
    optimizer._alpha = np.array([2.0])
    optimizer._chol = cho_factor(np.array([[2.0]]), lower=True)

    component_mean, _ = optimizer._component_posterior(
        np.array([[0.0]]), optimizer.groups[0]
    )
    combined_mean, _ = optimizer._combined_posterior(np.array([[0.0, 0.0]]))
    assert component_mean[0] == pytest.approx(2.0)
    assert combined_mean[0] == pytest.approx(11.0)


def test_add_gp_duplicate_avoidance_uses_owned_rng_and_bounded_projection():
    optimizer = ADDGPUCBOptimizer(
        objective=lambda x: 0.0,
        bounds=_bounds(2),
        max_iter=10,
        seed=17,
        enable_posterior_stopping=False,
    )
    optimizer.X = np.zeros((1, 2))
    optimizer.bandwidth = 0.5
    perturbed = optimizer._avoid_duplicate(np.zeros(2))
    assert np.linalg.norm(perturbed) / optimizer.bandwidth >= 1e-10
    assert np.all(perturbed >= optimizer.bounds[:, 0])
    assert np.all(perturbed <= optimizer.bounds[:, 1])


def test_add_gp_five_low_std_proposals_trigger_early_refit_and_bw_shrink(monkeypatch):
    optimizer = ADDGPUCBOptimizer(
        objective=lambda x: float(np.sum(x)),
        bounds=_bounds(2),
        max_iter=16,
        seed=19,
        enable_posterior_stopping=False,
    )
    learned_upper_bounds = []

    def fake_learn():
        learned_upper_bounds.append(float(optimizer.bandwidth_upper_bound))

    def fake_propose():
        return np.array([0.25, -0.25]), 0.001

    monkeypatch.setattr(optimizer, "_learn_structure_and_hyperparameters", fake_learn)
    monkeypatch.setattr(optimizer, "_propose", fake_propose)
    optimizer.optimize()

    assert learned_upper_bounds == pytest.approx([5.0, 4.5])
    assert optimizer.early_relearn_events == [
        {
            "bo_iteration": 6,
            "previous_bandwidth_upper_bound": 5.0,
            "bandwidth_upper_bound": 4.5,
        }
    ]


def test_add_gp_partial_learn_starts_with_random_orthogonal_fit_then_refits_raw(monkeypatch):
    optimizer = ADDGPUCBOptimizer(
        objective=lambda x: 0.0,
        bounds=_bounds(5),
        max_iter=10,
        seed=29,
        enable_posterior_stopping=False,
    )
    optimizer.X = np.linspace(-0.8, 0.8, 50).reshape(10, 5)
    optimizer.y = np.linspace(-2.0, 1.0, 10)
    optimizer._initial_std = float(np.std(optimizer.y, ddof=1))
    optimizer.noise_variance = 0.01 * optimizer._initial_std
    calls = []

    def fake_direct(groups, target_y, *, X=None):
        calls.append((tuple(tuple(group) for group in groups), np.asarray(X).copy()))
        return np.log(np.array([0.7, 1.2]))

    monkeypatch.setattr(optimizer, "_direct_hyperparameters", fake_direct)
    optimizer._learn_structure_and_hyperparameters()

    assert len(calls) == 2
    assert calls[0][0] == ((0, 1, 2, 3), (4,))
    assert not np.allclose(calls[0][1], optimizer.X)
    assert np.array_equal(calls[1][1], optimizer.X)
    assert len(optimizer.groups) == 2






def test_lamcts_uses_vendored_settings_and_never_overshoots_partial_epoch(monkeypatch):
    captured = {}

    class FakeLAMCTS:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.func = kwargs["func"]
            for index in range(kwargs["ninits"]):
                self.func(np.full(kwargs["dims"], -0.9 + 0.01 * index))

        def search(self, max_samples):
            captured["max_samples"] = max_samples
            while True:
                self.func(np.zeros(captured["dims"]))

    monkeypatch.setattr(paper_adapters, "_load_native_lamcts", lambda: FakeLAMCTS)
    objective_calls = []
    optimizer = LAMCTSTuRBOOptimizer(
        objective=lambda x: objective_calls.append(np.asarray(x).copy())
        or float(np.sum(x)),
        bounds=_bounds(5),
        max_iter=23,
        seed=31,
        la_mcts_cp=10.0,
        enable_posterior_stopping=False,
    ).optimize()

    assert captured["ninits"] == 20
    assert captured["Cp"] == pytest.approx(10.0)
    assert captured["leaf_size"] == 20
    assert captured["solver_type"] == "turbo"
    assert captured["turbo_max_evals"] == 50
    assert captured["kernel_type"] == "rbf"
    assert captured["gamma_type"] == "auto"
    assert captured["max_samples"] == 23
    assert captured["func"].lb.tolist() == [-1.0] * 5
    assert captured["func"].ub.tolist() == [1.0] * 5
    assert len(objective_calls) == len(optimizer.X) == len(optimizer.y) == 23


def test_lamcts_bo_uses_official_bo_solver_and_exact_budget(monkeypatch):
    captured = {}

    class FakeLAMCTS:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.func = kwargs["func"]
            for index in range(kwargs["ninits"]):
                self.func(np.full(kwargs["dims"], 0.01 * index))

        def search(self, max_samples):
            captured["max_samples"] = max_samples
            while True:
                self.func(np.zeros(captured["dims"]))

    monkeypatch.setattr(paper_adapters, "_load_native_lamcts", lambda: FakeLAMCTS)
    objective_calls = []
    optimizer = LAMCTSBOOptimizer(
        objective=lambda x: objective_calls.append(np.asarray(x).copy())
        or float(np.sum(x)),
        bounds=_bounds(5),
        max_iter=23,
        seed=31,
        la_mcts_cp=10.0,
        enable_posterior_stopping=False,
    ).optimize()

    assert captured["solver_type"] == "bo"
    assert captured["ninits"] == 20
    assert captured["leaf_size"] == 20
    assert captured["kernel_type"] == "rbf"
    assert captured["gamma_type"] == "auto"
    assert captured["max_samples"] == 23
    assert len(objective_calls) == len(optimizer.y) == 23


def test_cmaes_rejects_modern_pycma_instead_of_silently_changing_defaults(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "cma", SimpleNamespace(__version__="4.4.0"))
    optimizer = OfficialCMAESOptimizer(
        objective=lambda x: 0.0,
        bounds=_bounds(3),
        max_iter=20,
        seed=5,
        enable_posterior_stopping=False,
    )
    with pytest.raises(RuntimeError, match="cma==3.1.0"):
        optimizer.optimize()


def test_retired_duplicate_and_variant_surfaces_are_absent():
    removed = [
        "Experiment/baseline/builders_init_legacy.py",
        "Experiment/baseline/vanilia_bo.py",
        "Experiment/baseline/vanilla_bo_builder.py",
        "Experiment/baseline/turbo_builder.py",
        "ag_vs/acquisition.py",
        "ag_vs/inner_optimizer/turbo/strategy.py",
    ]
    assert all(not (ROOT / relative).exists() for relative in removed)
    assert not (
        ROOT
        / "Experiment/baseline/add_gp_bandits_official/add-gp-bandits-master/BOLibkky/'"
    ).exists()
