from __future__ import annotations

import typing
from types import SimpleNamespace

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.gaussian_process import GaussianProcessRegressor

from ag_vs.EarlyStopMechanism import EarlyStopMechanism
from ag_vs.GaussianProcess import GaussianProcessManager
from ag_vs.helper_functions import merge, split
from ag_vs.inner_optimizer.vanillabo_cuda import CudaSklearnGPPredictor
from ag_vs.inner_optimizer.turbo.proposal import _train_gp_with_numerical_retry
from ag_vs.optimizer import AGVSOptimizer
from ag_vs.utils import Node, SharedObservationIndices, find_optimum_B_and_cost
from Experiment.benchmark.runner_synthetic import (
    REAL_LIFE_N_BAD_ENABLED,
    REAL_LIFE_N_BAD_MULTIPLIER,
    SYNTHETIC_N_BAD_ENABLED,
    SYNTHETIC_N_BAD_MULTIPLIER,
    _batch_sizes_from_result,
    algorithm_settings_summary,
    apply_runner_agvs_batch_policy,
    batch_parallel_wall_clock_history,
    posterior_mean_history,
    posterior_threshold_history_by_delta,
    reported_posterior_mean_history,
    risk_label,
    sync_agvs_result_to_recorder,
)


def _bounds(dimension: int = 10) -> np.ndarray:
    return np.repeat([[-1.0, 1.0]], int(dimension), axis=0)


def _optimizer(**overrides) -> AGVSOptimizer:
    settings = {
        "objective": lambda x: -float(np.sum(np.asarray(x, dtype=float) ** 2)),
        "bounds": _bounds(),
        "eff_dim": 3,
        "max_iter": 12,
        "n_init": 10,
        "noise_std": 0.1,
        "tau": 0.0,
        "risk_tolerances": [0.05],
        "sim_mode": False,
        "n_split": 4,
        "seed": 7,
        "verbose": False,
        "printer_on": False,
        "print_tree_each_iter": False,
        "q_batch": None,
        "batch_size": 3,
        "t_seq": 3,
    }
    settings.update(overrides)
    return AGVSOptimizer(**settings)


@pytest.mark.parametrize("backend", ["vanillabo", "turbo"])
@pytest.mark.parametrize("batch_size", [1, 3])
@pytest.mark.parametrize("has_training_data", [False, True])
def test_candidate_completion_uses_global_best_observation(
    monkeypatch, backend, batch_size, has_training_data
):
    optimizer = _optimizer(node_backend=backend, batch_size=batch_size)
    optimizer.X = [np.full(optimizer.D, value) for value in (-0.8, 0.7, -0.2)]
    optimizer.y = [-10.0, -1.0, -4.0]
    optimizer.global_best_record = {
        "x": optimizer.X[0].copy(), "mu": 100.0, "evaluation_index": 0,
    }
    # The global observed incumbent is absent from this leaf's training data.
    leaf = Node(
        dims=[1, 4], h=1, idx=20, parent=optimizer.root,
        local_indices=[0, 2] if has_training_data else [],
    )
    if has_training_data:
        if backend == "turbo":
            from ag_vs.inner_optimizer.turbo.proposal import TurboNodeGP

            model = object.__new__(TurboNodeGP)
            engine = SimpleNamespace(
                batch_size=batch_size, n_evals=0, length=0.8, length_min=0.01,
                _select_candidates=lambda candidates, _draws: candidates[:batch_size],
            )
            monkeypatch.setattr(
                "ag_vs.inner_optimizer.turbo.proposal._turbo_engine",
                lambda *_args: engine,
            )
            monkeypatch.setattr(
                "ag_vs.inner_optimizer.turbo.proposal._seeded_native_candidates",
                lambda *_args: (np.full((3, 2), 0.25), np.zeros((3, batch_size))),
            )
        else:
            model = SimpleNamespace(X_train_=np.asarray(optimizer.X)[[0, 2]][:, leaf.dims])
            monkeypatch.setattr(
                optimizer, "_sample_acquisition_candidates",
                lambda *_args: np.full((3, 2), -0.5),
            )
            for module in ("ag_vs.optimizer", "ag_vs.inner_optimizer.vanillabo"):
                monkeypatch.setattr(
                    f"{module}.classical_expected_improvement",
                    lambda _model, _training, candidates: np.ones(len(candidates)),
                )
        monkeypatch.setattr(optimizer, "_fit_local_gp", lambda _leaf: model)
    else:
        monkeypatch.setattr(optimizer, "_fit_local_gp", lambda _leaf: None)

    original_X = np.asarray(optimizer.X).copy()
    proposals = (
        [optimizer._propose_from_leaf(leaf)] if batch_size == 1
        else optimizer._propose_q_from_leaf(leaf, batch_size)
    )
    inactive_dims = [dim for dim in range(optimizer.D) if dim not in leaf.dims]
    assert len(proposals) == batch_size
    for point, _score in proposals:
        np.testing.assert_array_equal(point[inactive_dims], original_X[1, inactive_dims])
        if has_training_data:
            np.testing.assert_allclose(point[leaf.dims], -0.5)
    np.testing.assert_array_equal(optimizer.X, original_X)


def test_observed_completion_ignores_nonfinite_values_keeps_first_tie_and_returns_copy():
    optimizer = _optimizer()
    optimizer.X = [np.full(optimizer.D, value) for value in (-0.8, -0.4, 0.0, 0.4, 0.8)]
    optimizer.y = [np.nan, -2.0, -2.0, np.inf, -np.inf]
    base = optimizer._base_fill_for_leaf(optimizer.root)
    np.testing.assert_array_equal(base, optimizer.X[1])
    base[:] = 0.9
    np.testing.assert_array_equal(optimizer.X[1], np.full(optimizer.D, -0.4))

    optimizer.X.append(np.full(optimizer.D, 0.6))
    optimizer.y.append(-1.0)
    np.testing.assert_array_equal(
        optimizer._base_fill_for_leaf(optimizer.root), optimizer.X[-1]
    )


@pytest.mark.parametrize("values", [[], [np.nan, np.inf, -np.inf]])
def test_observed_completion_without_finite_history_uses_seeded_bounded_fallback(values):
    optimizer = _optimizer()
    reference = _optimizer()
    optimizer.y = values
    optimizer.X = [np.zeros(optimizer.D) for _ in values]
    expected = reference.rng.uniform(reference.bounds[:, 0], reference.bounds[:, 1])
    np.testing.assert_array_equal(optimizer._base_fill_for_leaf(optimizer.root), expected)


def test_real_life_mode_requires_tau_only_when_early_stopping_is_enabled():
    optimizer = _optimizer(
        experiment_mode=False,
        early_stop_enabled=False,
        tau=None,
    )
    assert optimizer.tau is None

    with pytest.raises(ValueError, match="requires tau when early stopping is enabled"):
        _optimizer(
            experiment_mode=False,
            early_stop_enabled=True,
            tau=None,
        )


def test_n_bad_must_be_a_positive_integer():
    with pytest.raises(ValueError, match="N_bad"):
        _optimizer(N_bad=0)
    with pytest.raises(ValueError, match="N_bad"):
        _optimizer(N_bad=1.5)


def test_turbo_gp_retries_only_non_psd_failures_with_extra_jitter():
    NotPSDError = type("NotPSDError", (RuntimeError,), {})
    calls = []

    def train_gp(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise NotPSDError("matrix is not positive definite")
        return "fitted"

    model, jitter = _train_gp_with_numerical_retry(train_gp, marker=7)

    assert model == "fitted"
    assert jitter == pytest.approx(1e-3)
    assert calls == [{"marker": 7}, {"marker": 7}]


def test_turbo_gp_does_not_retry_unrelated_failures():
    calls = []

    def train_gp(**kwargs):
        calls.append(kwargs)
        raise ValueError("bad training data")

    with pytest.raises(ValueError, match="bad training data"):
        _train_gp_with_numerical_retry(train_gp)
    assert len(calls) == 1


def test_n_bad_defaults_to_three_times_n_split():
    optimizer = _optimizer(n_split=5)
    assert optimizer.N_bad == 3 * optimizer.split_k == 15
    assert optimizer.turbo_device == "cpu"


def test_runner_uses_family_specific_n_bad_multipliers():
    synthetic = apply_runner_agvs_batch_policy({"experiment_kind": "synthetic"})
    real_life = apply_runner_agvs_batch_policy({"experiment_kind": "real_life"})

    assert synthetic["n_bad_multiplier"] == SYNTHETIC_N_BAD_MULTIPLIER
    assert real_life["n_bad_multiplier"] == REAL_LIFE_N_BAD_MULTIPLIER
    assert synthetic["n_bad_enabled"] is SYNTHETIC_N_BAD_ENABLED
    assert real_life["n_bad_enabled"] is REAL_LIFE_N_BAD_ENABLED


def test_disabled_n_bad_never_counts_or_reinitializes(monkeypatch):
    optimizer = _optimizer(N_bad=1, N_bad_enabled=False)
    optimizer.global_best_record = {"mu": 1.0}
    monkeypatch.setattr(
        optimizer,
        "_restart_tree_with_shared_data",
        lambda **_kwargs: pytest.fail("N_bad-disabled optimizer restarted the tree"),
    )

    for _ in range(10):
        assert optimizer._record_node_selection_outcome(
            best_observed_before=1.0,
            selected_node=optimizer.root,
        ) is False
    assert optimizer.consecutive_bad_node_selections == 0
    assert optimizer.tree_restart_count == 0


def test_shared_observation_indices_use_live_root_and_frozen_child_prefixes():
    observations = [0.1, 0.2, 0.3]
    root_indices = SharedObservationIndices(observations)
    child_indices = root_indices.frozen()
    observations.append(0.4)
    assert list(root_indices) == [0, 1, 2, 3]
    assert list(child_indices) == [0, 1, 2]
    child_indices.append(3)
    assert list(child_indices) == [0, 1, 2, 3]


def test_n_bad_improvement_resets_consecutive_selection_failures(monkeypatch):
    optimizer = _optimizer(N_bad=2)
    optimizer.y = [1.0]
    monkeypatch.setattr(optimizer, "_restart_tree_with_shared_data", lambda **_kwargs: pytest.fail("unexpected restart"))

    assert optimizer._record_node_selection_outcome(
        best_observed_before=1.0,
        selected_node=optimizer.root,
    ) is False
    assert optimizer.consecutive_bad_node_selections == 1

    optimizer.y.append(1.25)
    assert optimizer._record_node_selection_outcome(
        best_observed_before=1.0,
        selected_node=optimizer.root,
    ) is False
    assert optimizer.consecutive_bad_node_selections == 0


def test_n_bad_one_defers_restart_until_child_state_exists(monkeypatch):
    optimizer = _optimizer(N_bad=1)
    optimizer.y = [1.0]
    monkeypatch.setattr(
        optimizer,
        "_restart_tree_with_shared_data",
        lambda **_kwargs: pytest.fail("root-only tree cannot be reinitialized yet"),
    )

    assert optimizer._record_node_selection_outcome(
        best_observed_before=1.0,
        selected_node=optimizer.root,
    ) is False
    assert optimizer.consecutive_bad_node_selections == 1


def test_n_bad_one_defers_restart_until_child_state_exists():
    optimizer = _optimizer(
        objective=lambda _x: 0.0,
        bounds=_bounds(4),
        eff_dim=1,
        max_iter=12,
        n_init=4,
        n_split=2,
        N_bad=1,
        batch_size=1,
        t_seq=1,
        early_stop_enabled=False,
    )

    result = optimizer.optimize()

    assert len(result.y) == 12
    assert optimizer.tree_restart_count >= 1


def test_n_bad_restart_uses_child_data_and_defers_replacement_root_gp(monkeypatch):
    optimizer = _optimizer(N_bad=2)
    optimizer.X = [
        np.full(optimizer.D, 0.1),
        np.full(optimizer.D, 0.2),
        np.full(optimizer.D, 0.3),
    ]
    optimizer.y = [0.1, 0.2, 0.3]
    optimizer.true_y = list(optimizer.y)
    optimizer.X = [
        np.full(optimizer.D, 0.1),
        np.full(optimizer.D, 0.2),
        np.full(optimizer.D, 0.2),
    ]
    optimizer.n_init = 2
    optimizer.global_best_record = {"mu": 1.0}
    shared_dataset = {(1,): {"mu": 1.0}}
    optimizer.global_dataset = shared_dataset

    old_root = optimizer.root
    child_groups = [[0, 1, 2], [3, 4, 5], [6, 7], [8, 9]]
    children = []
    for child_idx, dims in enumerate(child_groups, start=1):
        child = Node(dims=dims, h=1, idx=child_idx, parent=old_root)
        child.local_indices = [0, 1, 2]
        child.score_indices = [1] if child_idx == 1 else ([2] if child_idx == 2 else [])
        child.lengthscales = np.asarray(
            [float(dim + 1) for dim in dims], dtype=float
        )
        child._last_gp_fit_id = child_idx
        children.append(child)
    selected = children[0]
    optimizer._next_node_idx = len(children) + 1
    old_root.children = children
    old_root.expanded = True
    optimizer._all_nodes = [old_root, *children]
    optimizer._plot_nodes = [old_root, *children]
    optimizer._source_leaf_by_index = {1: children[0], 2: children[1]}

    monkeypatch.setattr(optimizer, "_plot_group_tree", lambda **_kwargs: None)

    assert optimizer._record_node_selection_outcome(
        best_observed_before=0.3,
        selected_node=selected,
    ) is False
    assert optimizer._record_node_selection_outcome(
        best_observed_before=0.3,
        selected_node=selected,
    ) is True

    assert optimizer.root is not old_root
    assert optimizer.roots == [optimizer.root]
    assert optimizer._all_nodes == [optimizer.root]
    assert optimizer._plot_nodes == [optimizer.root]
    assert optimizer.root.parent is None
    assert optimizer.root.idx == 0
    assert optimizer._next_node_idx == 1
    assert optimizer.root.dims == list(range(optimizer.D))
    assert isinstance(optimizer.root.local_indices, SharedObservationIndices)
    assert optimizer.root.score_indices == []
    # The same x with different observations represents two distinct noisy
    # evaluations and both must survive restart deduplication.
    assert list(optimizer.root.local_indices) == [0, 1, 2]
    assert np.isposinf(optimizer.root.ave_diff_score)
    assert np.isneginf(optimizer.root.selection_score)
    assert optimizer.root.score_value_source == "awaiting_reinitialized_root_own_visit"
    assert optimizer.root.expanded is False
    assert optimizer.root.children == []
    assert optimizer.root.model is None
    assert optimizer.root.restart_requires_own_trial is True
    assert optimizer.global_dataset is shared_dataset
    assert optimizer.global_best_record == {"mu": 1.0}
    assert optimizer._source_leaf_by_index == {}
    assert optimizer.consecutive_bad_node_selections == 0
    assert optimizer.tree_restart_count == 1
    assert optimizer.tree_restart_history[-1]["transferred_evaluation_count"] == 3
    assert optimizer.tree_restart_history[-1]["excluded_parent_evaluation_count"] == 0
    assert optimizer.tree_restart_history[-1]["duplicate_x_y_observations_removed"] == 0
    assert optimizer.tree_restart_history[-1]["observation_storage"] == "shared_active_child_training_union"
    assert optimizer.tree_restart_history[-1]["restart_data_source"] == "active_leaf_local_indices_union"
    assert optimizer.tree_restart_history[-1]["restart_split_source"] == "seeded_optimizer_rng_random_permutation"
    assert optimizer.tree_restart_history[-1]["root_direct_split"] is False
    assert optimizer.tree_restart_history[-1]["node_index_reset"] is True


def test_restart_deduplicates_x_y_pairs_with_componentwise_epsilon():
    optimizer = _optimizer()
    repeated_x = np.full(optimizer.D, 0.25)
    epsilon = optimizer.restart_pair_epsilon
    optimizer.X = [
        repeated_x.copy(),
        repeated_x + 0.5 * epsilon,
        repeated_x + 2.0 * epsilon,
        repeated_x + 0.5 * epsilon,
    ]
    optimizer.y = [
        0.5,
        0.5 + 0.5 * epsilon,
        0.5 + 0.5 * epsilon,
        0.5 + 2.0 * epsilon,
    ]
    optimizer.true_y = list(optimizer.y)

    child = Node(dims=list(range(optimizer.D)), h=1, idx=1, parent=optimizer.root)
    child.local_indices = [0, 1, 2, 3]
    child.score_indices = [0, 1, 2, 3]
    optimizer.root.children = [child]
    optimizer.root.expanded = True
    optimizer._all_nodes = [optimizer.root, child]

    _children, retained_indices, stats = optimizer._restart_child_snapshot()

    # Index 1 is within epsilon in every x coordinate and y, so index 0 wins.
    # Index 2 exceeds x epsilon; index 3 exceeds y epsilon, so both remain.
    assert retained_indices == [0, 2, 3]
    assert stats["duplicate_x_y_observations_removed"] == 1


def test_incremental_fixed_kernel_factor_matches_full_factorization():
    rng = np.random.default_rng(42)
    X = rng.uniform(-1.0, 1.0, size=(22, 5))
    y = np.sin(4.0 * X[:, 0]) + 0.3 * X[:, 1] - X[:, 2] ** 2
    manager = GaussianProcessManager(noise_std=0.1)
    source, _kernel_name, _score = manager.fit_best_gp(X[:11], y[:11], 5)
    assert source is not None

    incremental = manager._incremental_fixed_kernel_fit(source, X, y)
    assert incremental is not None
    full = GaussianProcessRegressor(
        kernel=clone(source.kernel_),
        alpha=source.alpha,
        normalize_y=False,
        optimizer=None,
    ).fit(X, y)
    query = rng.uniform(-1.0, 1.0, size=(13, 5))
    incremental_mean, incremental_std = incremental.predict(query, return_std=True)
    full_mean, full_std = full.predict(query, return_std=True)
    np.testing.assert_allclose(incremental_mean, full_mean, rtol=2e-9, atol=2e-9)
    np.testing.assert_allclose(incremental_std, full_std, rtol=2e-8, atol=2e-9)
    np.testing.assert_allclose(incremental.L_, full.L_, rtol=2e-9, atol=2e-9)
    assert incremental.gp_incremental_added == 11


def test_restart_root_gets_own_trial_before_its_random_dimension_split():
    optimizer = _optimizer(
        objective=lambda _x: 0.0,
        bounds=_bounds(4),
        eff_dim=1,
        max_iter=8,
        n_init=4,
        n_split=2,
        N_bad=3,
        batch_size=1,
        t_seq=1,
        incremental_root_gp=True,
        warm_start_root_gp=True,
        early_stop_enabled=False,
    )
    optimizer.optimize()
    assert optimizer.tree_restart_count == 1
    assert isinstance(optimizer.root.local_indices, SharedObservationIndices)
    # The replacement root inherits the union of the active children's
    # training data, then appends observations from its own post-restart visits.
    assert list(optimizer.root.local_indices) == list(range(8))
    assert optimizer.root.model is not None
    assert optimizer.root.expanded is True
    assert len(optimizer.root.children) == 2
    restart = optimizer.tree_restart_history[-1]
    assert restart["root_gp_fit_mode"] == "deferred_until_selected"
    assert restart["root_gp_incremental_added"] == 0
    assert restart["root_direct_split"] is False
    assert restart["required_own_trial_evaluations"] == 1


def test_gp_is_not_refitted_after_t_seq_batch(monkeypatch):
    optimizer = _optimizer(
        bounds=_bounds(4),
        eff_dim=1,
        max_iter=7,
        n_init=4,
        n_split=2,
        N_bad=99,
        batch_size=1,
        t_seq=3,
        early_stop_enabled=False,
    )
    monkeypatch.setattr(
        optimizer,
        "_refit_after_t_seq_step",
        lambda *_args, **_kwargs: pytest.fail("batch-end GP refit must be disabled"),
    )

    result = optimizer.optimize()

    # Initial n=4 fit, then stale-node fits only before acquisitions at n=5
    # and n=6. There is deliberately no terminal fit at n=7.
    assert [entry["after_iteration"] for entry in optimizer.gp_fit_history] == [4, 5, 6]
    assert all(
        entry["gp_refit_after_t_seq_step"] is False
        for entry in result.history[-3:]
    )


def test_t_seq_one_has_no_post_evaluation_gp_refit(monkeypatch):
    optimizer = _optimizer(
        bounds=_bounds(4),
        eff_dim=1,
        max_iter=5,
        n_init=4,
        n_split=2,
        N_bad=99,
        batch_size=1,
        t_seq=1,
        early_stop_enabled=False,
    )
    monkeypatch.setattr(
        optimizer,
        "_refit_after_t_seq_step",
        lambda *_args, **_kwargs: pytest.fail("t_seq=1 must not trigger a trailing refit"),
    )

    result = optimizer.optimize()

    assert [entry["after_iteration"] for entry in optimizer.gp_fit_history] == [4]
    assert result.history[-1]["gp_refit_after_t_seq_step"] is False


def test_agvs_bo_gp_uses_one_isotropic_lengthscale():
    rng = np.random.default_rng(17)
    X = rng.uniform(-1.0, 1.0, size=(50, 8))
    y = np.sin(7.0 * X[:, 7]) + 0.05 * X[:, 2]
    node = Node(
        dims=[7, 2, 5],
        h=0,
        idx=0,
        local_indices=list(range(len(X))),
    )
    manager = GaussianProcessManager(noise_std=0.1)

    model = manager.fit_node_gp(node, list(X), list(y))
    assert model is not None
    fitted_lengthscale = np.asarray(model.kernel_.k2.length_scale).reshape(-1)
    assert fitted_lengthscale.size == 1
    assert node.lengthscale_gp_kernel_name == "acquisition_gp_matern52_isotropic"
    assert node.lengthscales is not None
    assert node.lengthscales.shape == (3,)
    np.testing.assert_allclose(node.lengthscales, fitted_lengthscale[0])
    assert np.all(np.isfinite(node.lengthscales))
    assert np.all(node.lengthscales > 0.0)
    assert np.allclose(node.lengthscales, node.lengthscales[0])


@pytest.mark.parametrize("learn_noise", [False, True])
def test_cuda_bo_predictor_matches_fitted_sklearn_posterior(learn_noise):
    rng = np.random.default_rng(29)
    X = rng.uniform(-1.0, 1.0, size=(18, 4))
    y = np.sin(3.0 * X[:, 0]) + X[:, 1]
    node = Node(dims=[0, 1, 2, 3], h=0, idx=0, local_indices=list(range(len(X))))
    manager = GaussianProcessManager(noise_std=0.1, learn_noise=learn_noise)
    sklearn_model = manager.fit_node_gp(node, list(X), list(y))
    assert sklearn_model is not None

    # The CPU device exercises the same torch posterior implementation on CI;
    # production construction selects CUDA and uses identical float64 math.
    predictor = CudaSklearnGPPredictor(sklearn_model, device="cpu")
    query = rng.uniform(-1.0, 1.0, size=(9, 4))
    expected_mean, expected_std = sklearn_model.predict(query, return_std=True)
    actual_mean, actual_std = predictor.predict(query, return_std=True)
    np.testing.assert_allclose(actual_mean, expected_mean, rtol=1e-7, atol=1e-9)
    np.testing.assert_allclose(actual_std, expected_std, rtol=3e-6, atol=5e-9)

    expected_mean, expected_cov = sklearn_model.predict(query[:3], return_cov=True)
    actual_mean, actual_cov = predictor.predict(query[:3], return_cov=True)
    np.testing.assert_allclose(actual_mean, expected_mean, rtol=1e-7, atol=1e-9)
    np.testing.assert_allclose(actual_cov, expected_cov, rtol=3e-6, atol=5e-9)


def test_failed_local_refit_clears_stale_model_signature_and_stopping_state(monkeypatch):
    optimizer = _optimizer()
    node = optimizer.root
    optimizer.X = [np.zeros(optimizer.D), np.ones(optimizer.D)]
    optimizer.y = [0.0, 1.0]
    node.local_indices = [0, 1]

    fitted = optimizer._fit_local_gp(node)
    assert fitted is not None
    assert optimizer._node_gp_is_current(node)

    # An unrelated global observation does not invalidate this node-local fit.
    optimizer.X.append(np.full(optimizer.D, 0.5))
    optimizer.y.append(0.5)
    monkeypatch.setattr(
        optimizer.gp_manager,
        "fit_node_gp",
        lambda *_args, **_kwargs: pytest.fail("unchanged sibling-local data must use the cache"),
    )
    assert optimizer._fit_local_gp(node) is fitted
    assert optimizer._node_gp_is_current(node)

    # Once the observation becomes node-owned, a failed refit must not leave
    # the old posterior or its signature available to scoring/stopping.
    node.local_indices.append(2)
    monkeypatch.setattr(optimizer.gp_manager, "fit_node_gp", lambda *_args, **_kwargs: None)
    assert optimizer._fit_local_gp(node) is None
    assert node.model is None
    assert node._gp_fit_signature is None
    assert node.gp_kernel_name == "none"
    assert node.lengthscales is None
    assert not hasattr(node, "_posterior_mean_cache")
    assert not optimizer._node_gp_is_current(node)

    monkeypatch.setattr(
        optimizer.early_stop,
        "record_thresholds",
        lambda **_kwargs: pytest.fail("stale model reached stopping thresholds"),
    )
    monkeypatch.setattr(
        optimizer.early_stop,
        "stop_probability",
        lambda **_kwargs: pytest.fail("stale model reached stop probability"),
    )
    assert optimizer._record_stoppable_thresholds(node, [2]) is None
    probability, mean, std, index = optimizer._posterior_stop_probability(node, [2])
    assert probability == 0.0
    assert np.isnan(mean)
    assert np.isnan(std)
    assert index is None

    optimizer.n_init = 0
    node.score_indices = [2]
    assert np.isinf(optimizer._node_percent_best_diff(node))
    assert node.node_score_history == []


class _CountingMeanPosterior:
    def __init__(self):
        self.batch_sizes = []

    def predict(self, X, return_std=False):
        X = np.asarray(X, dtype=float)
        self.batch_sizes.append(int(len(X)))
        mean = np.sum(X, axis=1)
        if return_std:
            return mean, np.full(len(X), 0.1, dtype=float)
        return mean


def test_node_score_uses_observations_without_an_extra_posterior_call():
    optimizer = _optimizer()
    optimizer.n_init = 0
    optimizer.X = [np.full(optimizer.D, value) for value in (0.1, 0.2, 0.3, 0.4)]
    optimizer.y = [0.0, 0.1, 0.2, 0.3]
    optimizer.historical_best_observed_by_index = {2: 0.15, 3: 0.25}
    model = _CountingMeanPosterior()
    node = Node(
        dims=[0, 1],
        h=1,
        idx=20,
        local_indices=[0, 1, 2, 3],
        score_indices=[2, 3],
        model=model,
    )
    node._gp_fit_signature = optimizer._node_gp_fit_signature(node)

    optimizer._update_posterior_dataset_from_node(node, model)
    score = optimizer._node_percent_best_diff(node)

    assert model.batch_sizes == [4]
    assert node.score_posterior_source == "not_used"
    assert node.score_value_source == "observed_y_vs_historical_best_observed_y"
    assert score == pytest.approx((abs(0.2 - 0.15) + abs(0.3 - 0.25)) / 2.0)


def test_node_score_never_requires_a_gp_prediction():
    optimizer = _optimizer()
    optimizer.n_init = 0
    optimizer.X = [np.full(optimizer.D, value) for value in (0.1, 0.2, 0.3, 0.4)]
    optimizer.y = [0.0, 0.1, 0.2, 0.3]
    optimizer.historical_best_observed_by_index = {2: 0.15, 3: 0.25}
    model = _CountingMeanPosterior()
    node = Node(
        dims=[0, 1],
        h=1,
        idx=21,
        local_indices=[0, 1, 2, 3],
        score_indices=[2, 3],
        model=model,
    )
    node._gp_fit_signature = optimizer._node_gp_fit_signature(node)

    score = optimizer._node_percent_best_diff(node)

    assert model.batch_sizes == []
    assert node.score_posterior_source == "not_used"
    assert score == pytest.approx((abs(0.2 - 0.15) + abs(0.3 - 0.25)) / 2.0)


@pytest.mark.parametrize("terminal_flag", ["sad_terminal_branch", "sad_never_split"])
def test_terminal_or_non_splittable_leaf_uses_its_isotropic_acquisition_gp(terminal_flag):
    optimizer = _optimizer()
    optimizer.X = [np.zeros(optimizer.D), np.ones(optimizer.D)]
    optimizer.y = [0.0, 1.0]
    node = Node(dims=[0], h=1, idx=20, local_indices=[0, 1])
    setattr(node, terminal_flag, True)
    assert optimizer._fit_local_gp(node) is not None
    assert optimizer._node_gp_is_current(node)
    assert node.lengthscales.shape == (1,)
    assert node.lengthscale_gp_kernel_name == "acquisition_gp_matern52_isotropic"


def test_singleton_node_uses_shared_isotropic_acquisition_gp_lengthscale():
    optimizer = _optimizer()
    optimizer.X = [np.zeros(optimizer.D), np.ones(optimizer.D)]
    optimizer.y = [0.0, 1.0]
    node = Node(dims=[0], h=1, idx=21, local_indices=[0, 1])
    assert optimizer._fit_local_gp(node) is not None
    assert node.lengthscale_gp_kernel_name == "acquisition_gp_matern52_isotropic"
    assert node.lengthscales is not None
    assert node.lengthscales.shape == (1,)
    assert np.isfinite(node.lengthscales[0])
    assert node.lengthscales[0] > 0.0


def test_split_does_not_require_a_lengthscale_vector():
    node = Node(dims=[7, 2, 5], h=0, idx=0)
    optimizer = _optimizer()

    expected_order = np.random.default_rng(7).permutation(node.dims).tolist()
    assert split(optimizer, node) is True
    assert node.sad_split_permutation == expected_order
    assert [child.dims for child in node.children] == [
        [dimension] for dimension in expected_order
    ]


def test_split_uses_seeded_random_order_not_gp_lengthscales(monkeypatch):
    optimizer = _optimizer()
    node = Node(
        dims=[0, 1, 2],
        h=1,
        idx=50,
        local_bounds=_bounds(3),
        local_indices=[0, 1],
    )
    optimizer.X = [np.zeros(optimizer.D), np.ones(optimizer.D)]
    optimizer.y = [0.0, 1.0]

    monkeypatch.setattr(
        optimizer,
        "_fit_local_gp",
        lambda *_args, **_kwargs: pytest.fail(
            "random dimension splitting must not fit or inspect a GP"
        ),
    )

    expected_order = np.random.default_rng(7).permutation(node.dims).tolist()
    assert split(optimizer, node) is True
    assert node.sad_split_permutation == expected_order
    assert [child.dims for child in node.children] == [
        [dimension] for dimension in expected_order
    ]


def test_public_merge_defers_gp_until_own_trial(monkeypatch):
    optimizer = _optimizer()
    parent = optimizer.root
    source_specs = [
        ([0, 1, 2], [8.0, 9.0, 10.0], 4.0),
        ([3, 4, 5], [1.0, 2.0, 3.0], 3.0),
        ([6, 7], [6.0, 7.0], 2.0),
        ([8, 9], [4.0, 5.0], 1.0),
    ]
    siblings = []
    for idx, (dims, lengthscales, score) in enumerate(source_specs, start=1):
        child = Node(
            dims=list(dims),
            h=1,
            idx=idx,
            parent=parent,
            local_bounds=optimizer.bounds[dims],
            lengthscales=np.asarray(lengthscales),
        )
        child.ave_diff_score = float(score)
        child.sad_cohort_id = "cohort"
        child.sad_trial_done = True
        child.local_indices = [0, 1]
        siblings.append(child)

    parent.children = siblings
    parent.expanded = True
    parent.sad_eliminate_for_cohort = 1
    optimizer._all_nodes = [parent, *siblings]
    optimizer._plot_nodes = [parent, *siblings]
    optimizer._next_node_idx = 5

    optimizer.X = [np.zeros(optimizer.D), np.ones(optimizer.D)]
    optimizer.y = [0.0, 1.0]
    fitted_nodes = []

    def fit_shared_gp(node):
        fitted_nodes.append(node)
        node.model = object()
        node._gp_fit_signature = optimizer._node_gp_fit_signature(node)
        node.lengthscales = np.asarray([80.0, 90.0, 100.0, 10.0, 20.0, 30.0, 60.0, 70.0])
        node.lengthscale_gp_kernel_name = "acquisition_gp_matern52_isotropic"
        return node.model

    monkeypatch.setattr(optimizer, "_fit_local_gp", fit_shared_gp)

    merged = optimizer._merge_sad_cohort(parent, "cohort", siblings)

    assert merged is not None
    assert fitted_nodes == []
    assert merged.model is None
    assert merged.local_indices == [0, 1]
    assert merged.score_indices == []
    assert np.isposinf(merged.ave_diff_score)
    assert np.isneginf(merged.selection_score)
    assert merged.score_value_source == "awaiting_merged_node_own_visit"
    assert merged.sad_requires_own_trial is True
    assert merged.expanded is False
    assert merged.children == []
    assert optimizer.sad_merge_history[-1]["merged_immediate_split"] is False
    assert optimizer.sad_merge_history[-1]["merged_requires_own_trial"] is True
    assert optimizer.sad_merge_history[-1]["required_own_trial_evaluations"] == 9


def test_runtime_merge_inherits_child_data_but_starts_unscored(monkeypatch):
    optimizer = _optimizer()
    parent = optimizer.root
    optimizer.X = [np.zeros(optimizer.D), np.ones(optimizer.D)]
    optimizer.y = [0.0, 1.0]
    siblings = []
    for idx, dims in enumerate(([0, 1, 2], [3, 4, 5], [6, 7], [8, 9]), start=1):
        child = Node(
            dims=list(dims),
            h=1,
            idx=idx,
            parent=parent,
            local_bounds=optimizer.bounds[list(dims)],
        )
        child.local_indices = [0] if idx % 2 else [1]
        child.score_indices = list(child.local_indices)
        child.ave_diff_score = float(5 - idx)
        siblings.append(child)

    parent.children = siblings
    parent.expanded = True
    optimizer._all_nodes = [parent, *siblings]
    optimizer._plot_nodes = [parent, *siblings]
    optimizer._next_node_idx = 5

    fitted_nodes = []

    def fit_shared_gp(node):
        fitted_nodes.append(node)
        node.model = object()
        node._gp_fit_signature = optimizer._node_gp_fit_signature(node)
        node.lengthscales = np.ones(len(node.dims), dtype=float)
        return node.model

    monkeypatch.setattr(optimizer, "_fit_local_gp", fit_shared_gp)
    monkeypatch.setattr(optimizer, "_plot_group_tree", lambda **_kwargs: None)

    merged = merge(optimizer, parent, siblings, d_eff=optimizer.eff_dim, cohort_id="cohort")

    assert merged is not None
    assert fitted_nodes == []
    assert merged.model is None
    assert merged.local_indices == [0, 1]
    assert merged.score_indices == []
    assert np.isposinf(merged.ave_diff_score)
    assert np.isneginf(merged.selection_score)
    assert merged.sad_requires_own_trial is True
    assert merged.expanded is False
    assert merged.children == []
    assert optimizer.sad_merge_history[-1]["merged_immediate_split"] is False


@pytest.mark.parametrize("backend", ["vanillabo", "turbo"])
@pytest.mark.parametrize("merge_entrypoint", ["runtime", "optimizer"])
def test_target_dimension_merge_and_revisits_fit_all_observations(backend, merge_entrypoint):
    optimizer = _optimizer(
        bounds=_bounds(4), eff_dim=2, n_split=3, n_init=1, node_backend=backend,
    )
    # Include a repeated observation and an evaluation owned only by the
    # excluded child: neither may disappear from the full training history.
    optimizer.X = [np.full(4, value) for value in (-0.8, -0.3, 0.4, -0.8)]
    optimizer.y = [0.0, 0.2, 0.4, 0.0]
    parent = optimizer.root
    siblings = []
    for index, dims in enumerate(([0], [1], [2, 3]), start=1):
        child = Node(
            dims=list(dims), h=1, idx=index, parent=parent,
            local_indices=[0, index], score_indices=[index],
        )
        child.ave_diff_score = float(4 - index)
        child.sad_cohort_id = "target-cohort"
        child.sad_trial_done = True
        siblings.append(child)
    parent.children = siblings
    parent.expanded = True
    parent.sad_eliminate_for_cohort = 1
    optimizer._all_nodes = [parent, *siblings]
    optimizer._plot_nodes = [parent, *siblings]
    optimizer._next_node_idx = 4

    if merge_entrypoint == "runtime":
        retained = merge(optimizer, parent, siblings, cohort_id="target-cohort")
    else:
        retained = optimizer._merge_sad_cohort(parent, "target-cohort", siblings)

    assert retained.dims == [0, 1]
    assert retained.local_indices == [0, 1, 2, 3]
    assert retained.score_indices == []
    assert retained.node_score_history == []
    assert retained.model is None
    assert np.isposinf(retained.ave_diff_score)
    assert optimizer._prepare_selected_node_for_evaluation(retained) is retained
    model = optimizer._fit_local_gp(retained)
    assert model is not None
    np.testing.assert_array_equal(model.X_train_, np.asarray(optimizer.X)[:, retained.dims])
    assert optimizer._node_gp_is_current(retained)

    excluded = siblings[-1]
    optimizer.X.append(np.array([0.9, 0.7, -0.4, 0.6]))
    optimizer.y.append(0.8)
    optimizer._assign_new_index_to_leaf_tree(4, excluded)
    assert optimizer._node_gp_is_current(retained)

    optimizer._prepare_selected_node_for_evaluation(retained)
    assert retained.local_indices == [0, 1, 2, 3, 4]
    assert retained.score_indices == []
    assert retained.node_score_history == []
    assert not optimizer._node_gp_is_current(retained)
    refitted = optimizer._fit_local_gp(retained)
    assert refitted is not None and refitted is not model
    np.testing.assert_array_equal(refitted.X_train_, np.asarray(optimizer.X)[:, retained.dims])
    # Preparing a visit without new observations must keep the GP cache valid.
    optimizer._prepare_selected_node_for_evaluation(retained)
    assert optimizer._fit_local_gp(retained) is refitted

    optimizer.X.append(np.zeros(4))
    optimizer.y.append(0.3)
    optimizer._assign_new_index_to_leaf_tree(5, retained)
    optimizer.X.append(np.ones(4))
    optimizer.y.append(0.9)
    optimizer._assign_new_index_to_leaf_tree(6, excluded)
    optimizer._prepare_selected_node_for_evaluation(retained)
    assert retained.local_indices == list(range(7))
    assert retained.score_indices == [5]


@pytest.mark.parametrize("node_kind", ["larger_retained", "excluded", "ordinary"])
def test_full_history_sync_is_limited_to_retained_nodes_at_target_dimension(node_kind):
    optimizer = _optimizer()
    optimizer.X = [np.full(optimizer.D, value) for value in (-0.5, 0.0, 0.5)]
    optimizer.y = [0.1, 0.2, 0.3]
    node = Node(
        dims=list(range(optimizer.eff_dim + (node_kind == "larger_retained"))),
        h=1, idx=20, parent=optimizer.root, local_indices=[0, 1], score_indices=[1],
    )
    node.sad_merged_node = node_kind != "ordinary"
    node.sad_no_merge_branch = node_kind == "excluded"
    node.exclusion_state = int(node_kind == "excluded")
    optimizer._prepare_selected_node_for_evaluation(node)
    assert node.local_indices == [0, 1]
    assert node.score_indices == [1]


def test_stale_lengthscales_are_replaced_by_isotropic_acquisition_gp():
    optimizer = _optimizer()
    optimizer.X = [np.zeros(optimizer.D), np.ones(optimizer.D)]
    optimizer.y = [0.0, 1.0]
    node = Node(
        dims=[0, 1, 2],
        h=1,
        idx=50,
        local_indices=[0, 1],
        lengthscales=np.asarray([99.0, 99.0, 99.0]),
    )
    assert optimizer._fit_local_gp(node) is not None
    assert node.lengthscales.shape == (3,)
    assert node.lengthscale_gp_kernel_name == "acquisition_gp_matern52_isotropic"
    assert not np.array_equal(node.lengthscales, [99.0, 99.0, 99.0])
    assert np.allclose(node.lengthscales, node.lengthscales[0])
    assert optimizer._node_gp_is_current(node)


class _FixedPosterior:
    def predict(self, X, return_std=False):
        assert len(X) == 2
        mean = np.asarray([1.7, 21.0])
        std = np.asarray([0.1, 10.0])
        return (mean, std) if return_std else mean


def test_early_stop_probability_returns_its_own_triggering_candidate_index():
    mechanism = EarlyStopMechanism(
        tau=1.0,
        risk_tolerances=[0.05],
        n_init=0,
        noise_std=0.1,
    )
    node = Node(dims=[0], h=0, idx=0, model=_FixedPosterior())
    X = [np.asarray([0.0]), np.asarray([1.0])]
    y = [0.0, 0.0]

    threshold = mechanism.record_thresholds(
        node=node,
        X=X,
        y=y,
        candidate_indices=[0, 1],
    )
    probability, mean, std, stop_index = mechanism.stop_probability(
        node=node,
        X=X,
        y=y,
        candidate_indices=[0, 1],
    )

    assert threshold is not None
    assert threshold["best_index"] == 1
    assert stop_index == 0
    assert probability > 0.999
    assert mean == pytest.approx(1.7)
    assert std == pytest.approx(0.1)


def test_batch_does_not_run_stopping_or_threshold_checks(monkeypatch):
    optimizer = _optimizer(eff_dim=6, n_split=7, early_stop_enabled=True)
    first = np.full(optimizer.D, 0.25)
    second = np.full(optimizer.D, -0.25)

    monkeypatch.setattr(
        optimizer,
        "_propose_q_from_leaf",
        lambda _leaf, q: [(first.copy(), 1.0), (second.copy(), 0.5)][: int(q)],
    )

    monkeypatch.setattr(
        optimizer,
        "_record_stoppable_thresholds",
        lambda *_args, **_kwargs: pytest.fail("threshold calculation must be disabled"),
    )
    monkeypatch.setattr(
        optimizer,
        "_posterior_stop_probability",
        lambda *_args, **_kwargs: pytest.fail("stopping calculation must be disabled"),
    )

    result = optimizer.optimize()

    assert result.stopped is False
    assert len(result.y) == 12
    assert result.stop_iteration is None
    assert [record["iteration"] for record in result.history] == [11, 12]
    assert [record["batch_endpoint"] for record in result.history] == [False, True]
    assert all(record["batch_iteration"] == 11 for record in result.history)
    assert all(record["selection_batch_size"] == 3 for record in result.history)
    assert all(record["batch_size"] == 3 for record in result.history)
    assert all(record["acquisition_batch_mode"] == "batch_size" for record in result.history)
    assert all(record["q_batch"] is None for record in result.history)
    assert all(record["optimizer_point_batch_size"] == 2 for record in result.history)
    assert all(record["stop_index"] is None for record in result.history)
    assert all(record["threshold_entry"] is None for record in result.history)
    assert result.threshold_trace == []
    assert _batch_sizes_from_result("AG-VS-BO", result, 12) == [1] * 10 + [2]


@pytest.mark.parametrize("backend", ["vanillabo", "turbo"])
@pytest.mark.parametrize("batch_mode", ["batch_size", "q_batch"])
@pytest.mark.parametrize(
    "point_batch_size, values, references, first_visit_score",
    [
        (3, [12, 11, 14, 13, 16, 15, 18, 17], [10, 10, 10, 14, 14, 14, 16, 16], 11 / 6),
        (3, [12, 11, 14, 13, 16, 15, 18], [10, 10, 10, 14, 14, 14, 16], 11 / 6),
        (1, [12, 11, 14], [10, 12, 12], 3 / 2),
    ],
)
def test_score_reference_is_frozen_per_batch_and_refreshed_between_batches(
    monkeypatch, backend, batch_mode, point_batch_size, values, references, first_visit_score
):
    observations = iter([10.0, 9.0, *values])
    optimizer = _optimizer(
        objective=lambda _x: float(next(observations)),
        n_init=2, max_iter=2 + len(values), t_seq=2, node_backend=backend,
        batch_size=point_batch_size if batch_mode == "batch_size" else None,
        q_batch=point_batch_size if batch_mode == "q_batch" else None,
    )
    # Exercise the real evaluation/scoring loop with deterministic responses;
    # candidate generation and GP fitting are independent of this policy.
    monkeypatch.setattr(optimizer, "_fit_local_gp", lambda _node: None)
    monkeypatch.setattr(
        optimizer, "_propose_q_from_leaf",
        lambda _node, q: [(np.zeros(optimizer.D), 0.0) for _ in range(q)],
    )
    monkeypatch.setattr(
        optimizer, "_propose_from_leaf", lambda _node: (np.zeros(optimizer.D), 0.0),
    )

    result = optimizer.optimize()

    assert len(result.y) == optimizer.max_iter
    assert optimizer.historical_best_observed_by_index == dict(enumerate(references, start=2))
    root_visits = [entry for entry in result.history if entry["selected_node"] == optimizer.root.idx]
    assert root_visits[-1]["node_ave_diff_score"] == pytest.approx(first_visit_score)
    for node in optimizer._all_nodes:
        for record in node.node_score_history:
            assert record["best_observed_y_i_minus_1"] == references[record["evaluation_index"] - 2]
    # Progress reporting still shows every new observed maximum immediately.
    np.testing.assert_array_equal(
        [entry["best_y"] for entry in result.history],
        np.maximum.accumulate([10.0, *values])[1:],
    )


def test_legacy_agvs_batch_metadata_uses_actual_optimizer_batch_size():
    history = [
        {
            "iteration": iteration,
            "q_iteration": 11,
            "sequence_index": 1,
            "batch_size": 2,
            "optimizer_point_batch_size": 10,
        }
        for iteration in range(11, 21)
    ]
    history.append(
        {
            "iteration": 21,
            "q_iteration": 21,
            "sequence_index": 1,
            "batch_size": 2,
            "optimizer_point_batch_size": 1,
        }
    )

    sizes = _batch_sizes_from_result("AG-VS-TuRBO", {"history": history}, 21)

    assert sizes == [1] * 10 + [10, 1]


def test_batch_reconstruction_handles_initial_only_and_metadata_free_histories():
    assert _batch_sizes_from_result("AG-VS-BO", {"history": []}, 10) == [1] * 10

    metadata_free = {"history": [{"n": index} for index in range(1, 21)]}
    assert _batch_sizes_from_result("TuRBO", metadata_free, 20) == []
    reconstructed = batch_parallel_wall_clock_history(
        list(range(1, 21)),
        algorithm="TuRBO",
        result=metadata_free,
        n_evaluations=20,
        wall_clock=20.0,
        q_batch=10,
    )
    assert reconstructed == [10.0] * 10 + [20.0] * 10


def test_simulation_expected_iteration_uses_finalized_initial_design_size():
    optimizer = _optimizer(sim_mode=True, n_split=None, n_init=None, max_iter=20)
    _, expected = find_optimum_B_and_cost(
        optimizer.D,
        optimizer.eff_dim,
        optimizer.n_init,
        optimizer.q_batch,
        optimizer.selection_batch_size,
        optimizer.t_seq,
    )

    assert optimizer.split_k == 10
    assert optimizer.n_init == 20
    assert optimizer.expected_iteration == expected == 137


def test_turbo_simulation_cost_uses_batch_size_mode():
    optimizer = _optimizer(
        sim_mode=True,
        n_split=None,
        n_init=20,
        max_iter=20,
        node_backend="turbo",
    )
    _, expected = find_optimum_B_and_cost(
        optimizer.D,
        optimizer.eff_dim,
        optimizer.n_init,
        optimizer.q_batch,
        optimizer.selection_batch_size,
        optimizer.t_seq,
    )

    assert optimizer.selection_batch_size == 3
    assert optimizer.expected_iteration == expected


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"max_iter": 0}, "max_iter"),
        ({"max_iter": 1.5}, "max_iter"),
        ({"n_init": 0}, "n_init"),
        ({"n_init": np.nan}, "n_init"),
        ({"eff_dim": 0}, "eff_dim"),
        ({"eff_dim": 10}, "eff_dim"),
        ({"eff_dim": 2.5}, "eff_dim"),
        ({"n_split": 3}, "n_split"),
        ({"n_split": 11}, "n_split"),
        ({"n_split": 4.5}, "n_split"),
        ({"q_batch": None, "batch_size": None}, "exactly one"),
        ({"batch_size": np.nan}, "batch_size"),
        ({"t_seq": 1.5}, "t_seq"),
        ({"candidate_count": np.inf}, "candidate_count"),
        ({"tau": np.nan}, "tau"),
        ({"beta": np.inf}, "beta"),
        ({"tree_plot_pause": np.nan}, "tree_plot_pause"),
        ({"bounds": np.empty((0, 2))}, "bounds"),
        ({"bounds": np.asarray([[0.0, np.nan], [0.0, 1.0]])}, "bounds"),
        ({"bounds": np.asarray([[0.0, np.inf], [0.0, 1.0]])}, "bounds"),
        ({"bounds": np.asarray([[1.0, 1.0], [0.0, 1.0]])}, "upper"),
        ({"bounds": np.asarray([[2.0, 1.0], [0.0, 1.0]])}, "upper"),
    ],
)
def test_constructor_rejects_invalid_core_configuration(overrides, message):
    with pytest.raises(ValueError, match=message):
        _optimizer(**overrides)


def test_constructor_rejects_nonfinite_initial_points():
    points = np.zeros((2, 10))
    points[0, 0] = np.nan

    with pytest.raises(ValueError, match="initial_points"):
        _optimizer(initial_points=points)


def test_optimizer_type_hints_resolve_sequence_annotation():
    hints = typing.get_type_hints(AGVSOptimizer._after_proposal_batch)

    assert hints["evaluated_indices"] == typing.Sequence[int]


def test_algorithm_settings_disclose_isotropic_gp_and_random_splitting():
    settings = algorithm_settings_summary("AG-VS-BO", "hartmann3_300")

    assert settings["ard"] is False
    assert settings["isotropic"] is True
    assert settings["split_ard"] is False
    assert settings["split_gp_role"] == "seeded random permutation; no GP-based dimension ranking"


def test_threshold_history_deduplicates_sources_and_keeps_terminal_endpoint():
    delta = 1 / 256
    key = f"{delta:.12g}"

    def snapshot(n_eval, value):
        return {
            "n": n_eval,
            "current_stoppable_threshold_by_delta": {key: value},
            "best_stoppable_threshold_by_delta": {key: value},
        }

    snapshots = [snapshot(1, 1.0), snapshot(2, 2.0)]
    result = SimpleNamespace(
        threshold_trace=snapshots,
        history=[
            {"n_eval": entry["n"], "stopping_info": dict(entry)}
            for entry in snapshots
        ],
    )

    history = posterior_threshold_history_by_delta(result, n_evaluations=2)

    assert history[risk_label(delta)] == [1.0, 2.0]


def test_sync_agvs_result_preserves_real_timestamps_and_only_fills_missing_values():
    result = SimpleNamespace(
        X=np.asarray([[0.0], [0.5], [1.0]]),
        true_y=np.asarray([0.0, 0.5, 1.0]),
        y=np.asarray([0.0, 0.5, 1.0]),
    )
    recorder = SimpleNamespace(
        x_history=[],
        true_history=[],
        observed_history=[],
        eval_wall_clock_history=[0.1, 0.7, 0.9],
    )

    sync_agvs_result_to_recorder(recorder, result, wall_clock=1.0)
    assert recorder.eval_wall_clock_history == [0.1, 0.7, 0.9]

    recorder.eval_wall_clock_history = [0.1]
    sync_agvs_result_to_recorder(recorder, result, wall_clock=1.0)
    assert recorder.eval_wall_clock_history[0] == 0.1
    assert recorder.eval_wall_clock_history[-1] == 1.0
