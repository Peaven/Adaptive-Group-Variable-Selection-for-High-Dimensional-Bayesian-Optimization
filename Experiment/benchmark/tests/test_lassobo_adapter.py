from __future__ import annotations

import numpy as np
import pytest

from Experiment.benchmark import runner
from Experiment.benchmark.baseline.lassobo_adapter import LassoBOOptimizer


def make_optimizer(**kwargs):
    return LassoBOOptimizer(
        objective=lambda x: -float(np.sum(np.asarray(x, dtype=float) ** 2)),
        bounds=np.asarray([[0.0, 1.0]] * 4),
        max_iter=35,
        seed=17,
        **kwargs,
    )


def test_published_lassobo_defaults_are_registered_separately_from_mcts_lasso():
    assert runner.canonical_algorithm_name("lassobo") == "LassoBO"
    with pytest.raises(ValueError):
        runner.canonical_algorithm_name("lasso-bo")
    settings = runner.algorithm_settings_summary("LassoBO", "ackley3_300")
    assert settings["n_init"] == 30
    assert settings["batch_size"] == 1
    assert settings["lasso_lambda"] == 1e-3
    assert settings["subspace_count"] == "M_t=ceil(t^(1/3))"
    assert settings["kernel"] == "ARD squared-exponential (RBF)"
    assert settings["acquisition"] == "GP-UCB"


def test_published_dynamic_subspace_schedule_uses_cube_root():
    optimizer = make_optimizer()
    assert [optimizer.subspace_count(t) for t in (1, 2, 8, 9, 27, 28)] == [
        1,
        2,
        2,
        3,
        3,
        4,
    ]


def test_above_mean_variable_importance_rule_never_returns_empty_set():
    assert LassoBOOptimizer.important_dimensions([1.0, 1.0, 4.0, 2.0]).tolist() == [2]
    assert LassoBOOptimizer.important_dimensions([0.0, 0.0, 0.0]).tolist() == [0]


def test_lassobo_lhs_has_exact_published_initial_size_and_respects_bounds():
    optimizer = make_optimizer()
    points = optimizer._initial_design()
    assert points.shape == (30, 4)
    assert np.all(points >= 0.0)
    assert np.all(points <= 1.0)

