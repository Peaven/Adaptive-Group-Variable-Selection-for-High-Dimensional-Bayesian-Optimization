from __future__ import annotations

import pytest

from Experiment import grid_search_nbad
from Experiment.benchmark import runner
from Experiment.utils import make_synthetic_config


def _preserve_grid_mutated_runner_state(monkeypatch):
    for name in (
        "SYNTHETIC_Q_BATCH",
        "SYNTHETIC_BATCH_SIZE",
        "SYNTHETIC_T_SEQ",
        "SYNTHETIC_N_BAD_MULTIPLIER",
        "SYNTHETIC_N_BAD_ABSOLUTE",
        "SYNTHETIC_N_BAD_ENABLED",
        "SYNTHETIC_RESULT_ROOT",
        "SYNTHETIC_FIGURE_ROOT",
        "SYNTHETIC_TABLE_ROOT",
        "AGVS_SHOW_FINAL_TREE",
    ):
        monkeypatch.setattr(runner, name, getattr(runner, name))
    for name in (
        "AGVS_SYNTHETIC_Q_BATCH",
        "AGVS_SYNTHETIC_BATCH_SIZE",
        "AGVS_SYNTHETIC_T_SEQ",
        "AGVS_SYNTHETIC_N_BAD_MULTIPLIER",
        "AGVS_SYNTHETIC_N_BAD_ABSOLUTE",
        "AGVS_SYNTHETIC_N_BAD_ENABLED",
        "AGVS_SHOW_FINAL_TREE",
    ):
        monkeypatch.delenv(name, raising=False)


def test_default_plan_covers_only_two_synthetic_rastrigin_benchmarks():
    assert grid_search_nbad.parse_args([]).workers == 25
    assert grid_search_nbad.parse_args([]).trials == 25
    benchmarks = grid_search_nbad.selected_benchmarks(None)
    assert benchmarks == ("rastrigin3_300", "rastrigin6_1000")
    assert grid_search_nbad.N_BAD_VALUES == (6, 10, 14, 18, 20, 28)
    assert grid_search_nbad.ALGORITHMS == ("AG-VS-BO", "AG-VS-TuRBO")
    assert len(benchmarks) * len(grid_search_nbad.N_BAD_VALUES) == 12
    assert 12 * len(grid_search_nbad.ALGORITHMS) * grid_search_nbad.TRIALS == 600


@pytest.mark.parametrize("node_backend", ["vanillabo", "turbo"])
def test_absolute_n_bad_reaches_the_optimizer_unchanged(monkeypatch, node_backend):
    _preserve_grid_mutated_runner_state(monkeypatch)
    grid_search_nbad.configure_runner_cell(7)
    config = runner.apply_runner_agvs_batch_policy(
        make_synthetic_config("rastrigin3_300")
    )
    optimizer = runner.make_agvs(
        config,
        recorder=None,
        node_backend=node_backend,
        seed=2021,
        max_iter=25,
        device="cpu",
    )
    assert config["n_bad_is_absolute"] is True
    assert config["n_bad_multiplier"] == 7
    assert optimizer.N_bad == 7
    assert optimizer.batch_size == 3
    assert optimizer.t_seq == 2
    assert runner.AGVS_SHOW_FINAL_TREE is False


def test_normal_synthetic_n_bad_remains_multiplier_times_split(monkeypatch):
    monkeypatch.setattr(runner, "SYNTHETIC_Q_BATCH", None)
    monkeypatch.setattr(runner, "SYNTHETIC_BATCH_SIZE", 3)
    monkeypatch.setattr(runner, "SYNTHETIC_T_SEQ", 2)
    monkeypatch.setattr(runner, "SYNTHETIC_N_BAD_MULTIPLIER", 4)
    monkeypatch.setattr(runner, "SYNTHETIC_N_BAD_ABSOLUTE", None)
    monkeypatch.setattr(runner, "SYNTHETIC_N_BAD_ENABLED", True)
    config = runner.apply_runner_agvs_batch_policy(
        make_synthetic_config("rastrigin3_300")
    )
    optimizer = runner.make_agvs(
        config,
        recorder=None,
        seed=2021,
        max_iter=25,
        device="cpu",
    )
    assert "n_bad_is_absolute" not in config
    assert optimizer.N_bad == 4 * (3 + 1)


def test_iteration_outputs_draw_every_n_bad_curve(tmp_path, monkeypatch):
    result_root = tmp_path / "results"
    figure_root = tmp_path / "figures"
    table_root = tmp_path / "tables"
    monkeypatch.setattr(grid_search_nbad, "RESULT_ROOT", result_root)
    monkeypatch.setattr(grid_search_nbad, "FIGURE_ROOT", figure_root)
    monkeypatch.setattr(grid_search_nbad, "TABLE_ROOT", table_root)
    benchmark = "rastrigin3_300"
    for algorithm in grid_search_nbad.ALGORITHMS:
        for n_bad in (1, 2):
            for trial in (0, 1):
                trace = {
                    "benchmark": benchmark,
                    "algorithm": algorithm,
                    "trial_id": trial,
                    "algorithm_seed": 2021 + trial,
                    "benchmark_seed": 2021 + trial,
                    "n_evaluations": 3,
                    "best_observed_history": [
                        float(n_bad),
                        float(n_bad + trial + 1),
                        float(n_bad + trial + 2),
                    ],
                    "complete": True, "cache_identity": grid_search_nbad.cell_identity(grid_search_nbad.configure_runner_cell, n_bad, benchmark, algorithm, trial, 3),
                    "settings": {
                        "metric_name": "test objective",
                        "benchmark_noise_variance": 1e-4,
                        "algorithm_settings": {
                            "N_bad": n_bad,
                            "N_bad_mode": "absolute",
                            "batch_size": 3,
                            "t_seq": 2,
                        },
                    },
                }
                path = runner.trace_path_for(
                    grid_search_nbad.cell_roots(n_bad)[0],
                    benchmark,
                    algorithm,
                    trial,
                )
                runner.write_json(path, trace)

    curves = grid_search_nbad.all_curves(benchmark, (1, 2), 2)
    assert [curve["n_bad"] for curve in curves] == [1, 2, 1, 2]
    assert {curve["algorithm"] for curve in curves} == set(grid_search_nbad.ALGORITHMS)
    assert all(len(curve["x"]) == 3 for curve in curves)
    assert all(curve["completed_trials"] == 2 for curve in curves)

    individual = grid_search_nbad.save_benchmark_plot(benchmark, (1, 2), 2)
    summary, iterations = grid_search_nbad.write_csv_outputs(
        (benchmark,), (1, 2), 2
    )
    assert len(individual) == 2
    assert all(path.is_file() for path in individual)
    assert summary.is_file()
    assert iterations.is_file()
