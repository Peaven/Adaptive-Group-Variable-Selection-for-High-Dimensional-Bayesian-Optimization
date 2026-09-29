from __future__ import annotations

import numpy as np
import pytest

from Experiment.utils import (
    BASE_ALGORITHM_SEED,
    BASE_BENCHMARK_SEED,
    REAL_LIFE_ALGORITHMS,
    REAL_LIFE_EXPERIMENTS,
    SYNTHETIC_DIMENSION_MATRIX,
    SYNTHETIC_EXPERIMENTS,
    SYNTHETIC_OBJECTIVES,
    SYNTHETIC_TRIALS,
    algorithm_seed,
    benchmark_seed,
    effective_dimension_indices,
    get_real_life_experiment,
    iter_applicable_real_life_algorithms,
    real_life_algorithm_is_applicable,
    real_life_optimizer_dimension,
    real_life_reference_embedding_dimension,
)


REQUIRED_MATRIX = {(3, 300), (6, 1000)}


def test_every_synthetic_family_has_exact_required_matrix_only():
    assert set(SYNTHETIC_DIMENSION_MATRIX) == REQUIRED_MATRIX
    assert set(SYNTHETIC_OBJECTIVES) == {
        "ackley",
        "hartmann",
        "levy",
        "rastrigin",
        "rosenbrock",
        "sphere",
    }
    for objective in SYNTHETIC_OBJECTIVES:
        actual = {
            (experiment.d_eff, experiment.ambient_dim)
            for experiment in SYNTHETIC_EXPERIMENTS.values()
            if experiment.objective == objective
        }
        assert actual == REQUIRED_MATRIX
    assert len(SYNTHETIC_EXPERIMENTS) == (
        len(SYNTHETIC_OBJECTIVES) * len(SYNTHETIC_DIMENSION_MATRIX)
    )


def test_synthetic_execution_order_completes_objective_families_in_priority_order():
    expected = [
        f"{objective}{d_eff}_{ambient_dim}"
        for objective in SYNTHETIC_OBJECTIVES
        for d_eff, ambient_dim in ((3, 300), (6, 1000))
    ]
    assert list(SYNTHETIC_EXPERIMENTS) == expected


def test_synthetic_names_dimensions_and_budgets_match_identity():
    for name, experiment in SYNTHETIC_EXPERIMENTS.items():
        assert name == f"{experiment.objective}{experiment.d_eff}_{experiment.ambient_dim}"
        assert f"deff{experiment.d_eff}" in experiment.experiment_id
        assert experiment.experiment_id.endswith(f"D{experiment.ambient_dim}")
        assert experiment.budget == (600 if experiment.objective == "hartmann" else 700)
        assert experiment.metadata()["trials"] == 25 == SYNTHETIC_TRIALS


def test_active_coordinates_are_exact_choice_result_and_fixed_by_identity():
    seen_seeds = set()
    seen_identity_subsets = set()
    for experiment in SYNTHETIC_EXPERIMENTS.values():
        expected = effective_dimension_indices(
            experiment.objective,
            experiment.d_eff,
            experiment.ambient_dim,
        )
        assert experiment.effective_dimension_indices == expected
        assert len(expected) == experiment.d_eff
        assert len(set(expected)) == experiment.d_eff
        assert expected != tuple(range(experiment.d_eff))
        assert experiment.effective_dimension_seed not in seen_seeds
        seen_seeds.add(experiment.effective_dimension_seed)
        identity_subset = (experiment.ambient_dim, expected)
        assert identity_subset not in seen_identity_subsets
        seen_identity_subsets.add(identity_subset)


def test_seed_ownership_is_ordered_and_effective_seed_ignores_trial_and_algorithm():
    experiment = SYNTHETIC_EXPERIMENTS["rosenbrock3_300"]
    plans = [experiment.seeds(trial_id) for trial_id in range(5)]
    assert [plan.trial_id for plan in plans] == list(range(5))
    assert [plan.algorithm_seed for plan in plans] == [
        BASE_ALGORITHM_SEED + trial_id for trial_id in range(5)
    ]
    assert [plan.benchmark_seed for plan in plans] == [
        BASE_BENCHMARK_SEED + trial_id for trial_id in range(5)
    ]
    assert len({plan.algorithm_seed for plan in plans}) == len(plans)
    assert len({plan.benchmark_seed for plan in plans}) == len(plans)
    assert all(
        plan.algorithm_seed == plan.benchmark_seed
        for plan in plans
    )
    assert [
        experiment.seeds(trial_id).algorithm_seed
        for trial_id in range(SYNTHETIC_TRIALS)
    ] == list(range(2021, 2046))
    assert (
        SYNTHETIC_EXPERIMENTS["ackley6_1000"].seeds(4).algorithm_seed
        == experiment.seeds(4).algorithm_seed
    )
    assert {plan.effective_dimension_seed for plan in plans} == {
        experiment.effective_dimension_seed
    }
    for seed_function in (algorithm_seed, benchmark_seed):
        with pytest.raises(ValueError):
            seed_function(3, base_seed_offset=11)
    with pytest.raises(ValueError, match="trial_id"):
        experiment.seeds(-1)


def test_real_life_conditions_are_separate_without_fabricated_active_subsets():
    assert set(REAL_LIFE_EXPERIMENTS) == {"mopta08_deff3", "mopta08_deff4"}
    for experiment in REAL_LIFE_EXPERIMENTS.values():
        assert experiment.seeds(0).effective_dimension_seed is None
        assert experiment.physical_dim == 124



def test_agvs_synthetic_uses_batch3_for_two_sequential_stages():
    for experiment in SYNTHETIC_EXPERIMENTS.values():
        assert experiment.selection_batch_size == 3
        assert experiment.q_batch is None
        assert experiment.t_seq == 2


def test_agvs_real_life_uses_batch3_for_two_sequential_stages():
    for experiment in REAL_LIFE_EXPERIMENTS.values():
        assert experiment.trials == 25
        assert experiment.selection_batch_size == 3
        assert experiment.q_batch is None
        assert experiment.t_seq == 2


def test_real_life_task_specific_evaluation_budgets():
    for experiment in REAL_LIFE_EXPERIMENTS.values():
        assert experiment.budget == 150
        assert experiment.task.bounds == (0.0, 1.0)
        assert real_life_reference_embedding_dimension(experiment, "HeSBO") == 10
        assert real_life_reference_embedding_dimension(experiment, "ALEBO") == 10



def test_removed_real_life_tasks_are_not_registered():
    for task in ("rover60", "nasbench101", "hopper", "walker2d", "dna", "diabetes"):
        with pytest.raises(KeyError, match="Unknown real-life task"):
            get_real_life_experiment(task, 3)



def test_real_life_reference_applicability_and_embedding_exceptions_are_central():
    from Experiment.benchmark import mopta
    assert set(REAL_LIFE_ALGORITHMS) == set(mopta.ALGORITHMS)
    for condition in REAL_LIFE_EXPERIMENTS.values():
        for algorithm in ("HeSBO", "ALEBO"):
            assert real_life_optimizer_dimension(condition, algorithm) == 10
            assert real_life_algorithm_is_applicable(condition, algorithm)
        assert real_life_optimizer_dimension(condition, "AG-VS-BO") == condition.configured_d_eff
        for algorithm in ("VAE-BO", "LassoBO", "SAASBO"):
            assert not real_life_algorithm_is_applicable(condition, algorithm)

