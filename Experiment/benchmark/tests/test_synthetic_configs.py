from __future__ import annotations

import numpy as np
import pytest

from Experiment.utils import (
    SYNTHETIC_BENCHMARK_NOISE_STD,
    SYNTHETIC_BENCHMARK_NOISE_VARIANCE,
    SYNTHETIC_DIMENSION_MATRIX,
    SYNTHETIC_EXPERIMENTS,
)
from Experiment.utils import make_all_synthetic_configs, make_synthetic_config


def test_all_configs_are_registry_derived_and_metadata_complete():
    configs = make_all_synthetic_configs()
    assert set(configs) == set(SYNTHETIC_EXPERIMENTS)
    for name, config in configs.items():
        experiment = SYNTHETIC_EXPERIMENTS[name]
        assert config["objective"] == experiment.objective
        assert config["D"] == experiment.ambient_dim == config["bounds"].shape[0]
        assert config["d_eff"] == experiment.d_eff == config["effective_dim"]
        assert tuple(config["active_indices"]) == experiment.effective_dimension_indices
        assert tuple(config["effective_dimension_indices"]) == experiment.effective_dimension_indices
        assert config["effective_dimension_seed"] == experiment.effective_dimension_seed
        assert config["benchmark_noise_std"] == SYNTHETIC_BENCHMARK_NOISE_STD
        assert config["benchmark_noise_variance"] == SYNTHETIC_BENCHMARK_NOISE_VARIANCE
        assert config["benchmark_noise_distribution"] == "Gaussian(0, 1e-4)"
        assert config["max_iter"] == experiment.budget
        assert config["q_batch"] is None
        assert config["t_seq"] == 2
        assert config["selection_batch_size"] == 3
        assert config["batch_size"] == 3
        assert config["selection_batch_size"] == experiment.selection_batch_size


def test_factory_does_not_resample_active_coordinates():
    for name in SYNTHETIC_EXPERIMENTS:
        first = make_synthetic_config(name)
        second = make_synthetic_config(name)
        np.testing.assert_array_equal(first["active_indices"], second["active_indices"])
        assert first["effective_dimension_seed"] == second["effective_dimension_seed"]


def test_objectives_depend_only_on_fixed_active_coordinates_and_are_finite():
    rng = np.random.default_rng(7)
    for name, config in make_all_synthetic_configs().items():
        bounds = np.asarray(config["bounds"], dtype=float)
        active = np.asarray(config["active_indices"], dtype=int)
        inactive = np.setdiff1d(np.arange(len(bounds)), active)
        x = rng.uniform(bounds[:, 0], bounds[:, 1])
        baseline = float(config["func"](x))
        changed_inactive = x.copy()
        changed_inactive[inactive] = rng.uniform(
            bounds[inactive, 0],
            bounds[inactive, 1],
        )
        assert float(config["func"](changed_inactive)) == baseline, name
        assert np.isfinite(float(config["func"](config["optimum_x"])))


def test_known_optimum_scores_preserve_objective_equations():
    expected = {
        "ackley": 0.0,
        "hartmann3": 3.8627797869493365,
        "hartmann6": 3.322368011391339,
        "levy": 0.0,
        "rastrigin": 0.0,
        "rosenbrock": 0.0,
        "sphere": 0.0,
    }
    for name, config in make_all_synthetic_configs().items():
        objective = str(config["objective"])
        key = f"hartmann{config['d_eff']}" if objective == "hartmann" else objective
        score = float(config["func"](config["optimum_x"]))
        assert np.isclose(score, expected[key], atol=1e-12), name


def test_levy_penalizes_a_nonoptimal_first_active_coordinate():
    config = make_synthetic_config("levy3_300")
    active = np.asarray(config["active_indices"], dtype=int)
    candidate = np.asarray(config["optimum_x"], dtype=float).copy()
    candidate[active] = np.asarray([5.0, 1.0, 1.0])

    w0 = 1.0 + (5.0 - 1.0) / 4.0
    expected_loss = (
        np.sin(np.pi * w0) ** 2
        + (w0 - 1.0) ** 2
        * (1.0 + 10.0 * np.sin(np.pi * w0 + 1.0) ** 2)
    )

    assert float(config["func"](candidate)) == pytest.approx(-expected_loss)
    assert float(config["func"](candidate)) < -1.0


@pytest.mark.parametrize("d_eff,ambient_dim", SYNTHETIC_DIMENSION_MATRIX)
def test_rosenbrock_lipschitz_bound_covers_all_active_gradient_components(
    d_eff, ambient_dim
):
    config = make_synthetic_config(f"rosenbrock{d_eff}_{ambient_dim}")

    # The gradient bounds are attained together at z=(-2,...,-2).  The factor
    # four converts derivatives from the configured z domain to the unit cube.
    gradient_z = np.full(d_eff, -6006.0)
    gradient_z[0] = -4806.0
    gradient_z[-1] = -1200.0
    exact_domain_lipschitz = 4.0 * np.linalg.norm(gradient_z)
    assert float(config["L"]) == pytest.approx(exact_domain_lipschitz)

    # Numerically sample just inside that corner so centered differences do not
    # cross the objective's clipping boundary.
    active = np.asarray(config["active_indices"], dtype=int)
    candidate = np.zeros(int(config["D"]), dtype=float)
    candidate[active] = 1.0e-4
    step = 1.0e-6
    numerical_gradient = []
    for dim in active:
        plus = candidate.copy()
        minus = candidate.copy()
        plus[dim] += step
        minus[dim] -= step
        numerical_gradient.append(
            (float(config["func"](plus)) - float(config["func"](minus)))
            / (2.0 * step)
        )
    assert np.linalg.norm(numerical_gradient) <= float(config["L"])
