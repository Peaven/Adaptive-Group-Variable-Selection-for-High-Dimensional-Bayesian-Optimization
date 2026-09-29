from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from Experiment.benchmark.baseline.reference_infrastructure import (
    assert_bo_fairness,
    assert_turbo_fairness,
)
from Experiment.benchmark import runner
from Experiment.utils import (
    BASE_ALGORITHM_SEED,
    BASE_BENCHMARK_SEED,
    SYNTHETIC_BENCHMARK_NOISE_VARIANCE,
    SYNTHETIC_BENCHMARK_NOISE_STD,
    SYNTHETIC_EXPERIMENTS,
)
from Experiment.benchmark.real_life.tasks import make_real_life_config


def test_runner_registers_only_central_synthetic_matrix():
    assert runner.SYNTHETIC_BENCHMARKS == [
        name for name in SYNTHETIC_EXPERIMENTS if not name.startswith("hartmann")
    ]
    assert set(SYNTHETIC_EXPERIMENTS).issubset(runner.BENCHMARKS)
    scripts = runner.discover_benchmark_scripts()
    assert set(name for name in scripts if name in SYNTHETIC_EXPERIMENTS) == set(
        runner.SYNTHETIC_BENCHMARKS
    )
    assert {scripts[name].name for name in runner.SYNTHETIC_BENCHMARKS} == {
        "utils.py"
    }


def test_ordered_runner_seed_semantics_are_algorithm_independent():
    for trial_id in range(4):
        seeds = {
            runner.seed_bundle_for_config(
                runner.BENCHMARKS["ackley3_300"],
                trial_id,
            ).algorithm_seed
            for _algorithm in runner.SYNTHETIC_ALGORITHM_ORDER
        }
        assert seeds == {BASE_ALGORITHM_SEED + trial_id}
        bundle = runner.seed_bundle_for_config(
            runner.BENCHMARKS["ackley3_300"],
            trial_id,
        )
        assert bundle.algorithm_seed == BASE_ALGORITHM_SEED + trial_id
        assert bundle.benchmark_seed == BASE_BENCHMARK_SEED + trial_id
        assert bundle.effective_dimension_seed == SYNTHETIC_EXPERIMENTS[
            "ackley3_300"
        ].effective_dimension_seed


def test_only_two_canonical_agvs_variants_exist():
    assert runner.AGVS_ALGORITHMS == {"AG-VS-BO", "AG-VS-TuRBO"}
    assert runner.benchmark_q_batch(runner.BENCHMARKS["ackley3_300"]) is None
    assert (
        runner.benchmark_batch_size(runner.BENCHMARKS["ackley3_300"])
        == runner.SYNTHETIC_BATCH_SIZE
    )
    assert (
        runner.benchmark_batch_size(runner.BENCHMARKS["ackley6_1000"])
        == runner.SYNTHETIC_BATCH_SIZE
    )
    with pytest.raises(ValueError, match="Unknown algorithm"):
        runner.canonical_algorithm_name("unknown-agvs-variant")


def test_runner_applies_absolute_unscaled_observation_score_to_both_agvs_variants():
    for algorithm in ("AG-VS-BO", "AG-VS-TuRBO"):
        settings = runner.algorithm_settings_summary(algorithm, "ackley3_300")
        assert settings["abs_mode"] is True
        assert settings["score_formula"] == "mean(abs(y_obs_i - best_y_obs_i_minus_1))"
        assert settings["score_value_source"] == "observations; no posterior prediction"


def test_synthetic_algorithm_roster_is_exact_and_reference_backed():
    assert runner.SYNTHETIC_ALGORITHM_ORDER == [
        "AG-VS-BO",
        "MCTS-VS-BO",
        "MCTS-VS-TuRBO",
        "MCTS-VS-SAASBO",
        "TuRBO",
        "CMA-ES",
        "Dropout-BO",
        "Dropout-TuRBO",
        "LassoBO",
        "LA-MCTS-TuRBO",
        "SAASBO",
        "GTBO",
        "HeSBO",
        "ALEBO",
        "VAE-BO",
    ]
    assert runner.DEFAULT_ALGORITHMS == [
        "AG-VS-BO",
        "MCTS-VS-BO",
        "MCTS-VS-TuRBO",
        "TuRBO",
        "Dropout-BO",
        "Dropout-TuRBO",
        "LA-MCTS-TuRBO",
        "VAE-BO",
    ]
    assert runner.NONDEFAULT_REFERENCE_ALGORITHMS == {
        "MCTS-VS-SAASBO",
        "MCTS-VS-RS",
        "Dropout-RS",
        "CMA-ES",
        "GTBO",
        "LassoBO",
        "SAASBO",
        "HeSBO",
        "ALEBO",
        "ADD-GP-UCB",
        "VanillaBO",
        "RandomSearch",
        "LA-MCTS-BO",
    }
    assert runner.DEFAULT_BENCHMARKS == runner.SYNTHETIC_BENCHMARKS


def test_plot_top_five_uses_final_mean_noiseless_and_styles_are_global():
    by_algorithm = {
        f"algorithm-{index}": [
            {"history": [-10.0, float(index)]},
            {"history": [-9.0, float(index) + 0.2]},
        ]
        for index in range(6)
    }
    selected, scores = runner.select_top_algorithms_by_final_mean_noiseless(
        by_algorithm,
        lambda item: item["history"],
        limit=5,
    )

    assert selected == [
        "algorithm-5",
        "algorithm-4",
        "algorithm-3",
        "algorithm-2",
        "algorithm-1",
    ]
    assert scores["algorithm-5"] == pytest.approx(5.1)

    short_style = runner.algorithm_plot_style("TuRBO", np.arange(10))
    long_style = runner.algorithm_plot_style("TuRBO", np.arange(700))
    for key in ("color", "marker", "linestyle"):
        assert short_style[key] == long_style[key]


def test_agvs_and_mcts_settings_report_their_actual_differences():
    ag_bo = runner.algorithm_settings_summary("AG-VS-BO", "ackley3_300")
    mcts_bo = runner.algorithm_settings_summary("MCTS-VS-BO", "ackley3_300")
    ag_turbo = runner.algorithm_settings_summary("AG-VS-TuRBO", "ackley3_300")
    mcts_turbo = runner.algorithm_settings_summary("MCTS-VS-TuRBO", "ackley3_300")
    assert (ag_bo["batch_size"], ag_bo["q_batch"], ag_bo["t_seq"]) == (
        runner.SYNTHETIC_BATCH_SIZE,
        runner.SYNTHETIC_Q_BATCH,
        runner.SYNTHETIC_T_SEQ,
    )
    assert ag_bo["acquisition"] == "classical analytic Expected Improvement"
    assert ag_bo["alpha"] == pytest.approx(SYNTHETIC_BENCHMARK_NOISE_VARIANCE)
    assert ag_bo["noise_variance"] == pytest.approx(
        SYNTHETIC_BENCHMARK_NOISE_VARIANCE
    )
    assert ag_bo["observation_noise_std"] == pytest.approx(
        SYNTHETIC_BENCHMARK_NOISE_STD
    )
    assert ag_turbo["use_ard"] is False
    assert ag_turbo["anisotropic"] is False
    assert ag_turbo["isotropic"] is True
    assert ag_turbo["internal_batch_size"] == 3
    assert ag_turbo["batch_size"] == 3
    assert ag_turbo["t_seq"] == 2
    assert "no independent TuRBO batch setting" in ag_turbo["turbo_backend_batch_policy"]
    assert "then traverse the tree again" in ag_turbo["node_visit_policy"]
    assert mcts_turbo["internal_batch_size"] == 10


def test_algorithm_output_families_and_styles_are_stable():
    assert runner.STAT_TABLE_REFERENCE_BY_FAMILY == {
        "bo": "AG-VS-BO",
        "turbo": "AG-VS-TuRBO",
        "united": "AG-VS-BO",
    }
    assert runner.stat_table_reference_by_family("united") == "AG-VS-BO"
    assert (
        runner.stat_table_reference_by_family("united", contains_real_life=True)
        == "AG-VS-TuRBO"
    )
    assert (
        runner.stat_table_reference_by_family("bo", contains_real_life=True)
        == "AG-VS-BO"
    )
    assert (
        runner.stat_table_reference_by_family("turbo", contains_real_life=True)
        == "AG-VS-TuRBO"
    )
    assert runner.algorithm_output_family("AG-VS-TuRBO") == "turbo"
    assert runner.algorithm_output_family("MCTS-VS-TuRBO") == "turbo"
    assert runner.algorithm_output_family("TuRBO") == "turbo"
    assert runner.algorithm_output_family("LA-MCTS-TuRBO") == "turbo"
    assert runner.algorithm_output_family("AG-VS-BO") == "bo"
    assert runner.algorithm_output_family("VanillaBO") == "bo"
    assert runner.algorithm_output_family("VAE-BO") == "bo"

    bo_only_traces = [{"algorithm": "AG-VS-BO", "complete": True}]
    assert runner.figure_output_family_is_available(bo_only_traces, "bo") is True
    assert runner.figure_output_family_is_available(bo_only_traces, "turbo") is False
    assert runner.figure_output_family_is_available(bo_only_traces, "united") is True

    turbo_only_traces = [{"algorithm": "AG-VS-TuRBO", "complete": True}]
    assert runner.figure_output_family_is_available(turbo_only_traces, "bo") is False
    assert runner.figure_output_family_is_available(turbo_only_traces, "turbo") is True
    assert runner.figure_output_family_is_available(turbo_only_traces, "united") is True

    bo_algorithms = {"AG-VS-BO", "MCTS-VS-BO", "VanillaBO", "Dropout-BO", "VAE-BO"}
    turbo_algorithms = {
        "AG-VS-TuRBO",
        "MCTS-VS-TuRBO",
        "TuRBO",
        "Dropout-TuRBO",
    }
    assert all(runner.comparison_output_family_includes(name, "bo") for name in bo_algorithms)
    assert not any(runner.comparison_output_family_includes(name, "turbo") for name in bo_algorithms)
    assert all(runner.comparison_output_family_includes(name, "turbo") for name in turbo_algorithms)
    assert not any(runner.comparison_output_family_includes(name, "bo") for name in turbo_algorithms)
    assert all(
        runner.comparison_output_family_includes(name, "united")
        for name in bo_algorithms | turbo_algorithms
    )

    for algorithm in runner.DEFAULT_ALGORITHMS:
        short = runner.algorithm_plot_style(algorithm, x_axis=[1, 2])
        long = runner.algorithm_plot_style(algorithm, x_axis=list(range(100)))
        assert (short["color"], short["marker"], short["linestyle"]) == (
            long["color"],
            long["marker"],
            long["linestyle"],
        )


def test_synthetic_objective_noise_is_seeded_gaussian_and_common_by_evaluation():
    first = runner.NoisyObjectiveRecorder(
        lambda _x: 10.0,
        noise_std=SYNTHETIC_BENCHMARK_NOISE_STD,
        seed=BASE_BENCHMARK_SEED + 7,
    )
    second = runner.NoisyObjectiveRecorder(
        lambda _x: -2.0,
        noise_std=SYNTHETIC_BENCHMARK_NOISE_STD,
        seed=BASE_BENCHMARK_SEED + 7,
    )
    for index in range(8):
        first([index])
        second([index])

    first_noise = np.asarray(first.observed_history) - np.asarray(first.true_history)
    second_noise = np.asarray(second.observed_history) - np.asarray(second.true_history)
    np.testing.assert_allclose(first_noise, second_noise, rtol=0.0, atol=1e-15)
    assert np.any(first_noise != 0.0)


def test_objective_recorder_replays_stored_prefix_without_objective_calls():
    objective_calls = []
    source = {
        "n_evaluations": 2,
        "x_history": [[0.1, 0.2], [0.3, 0.4]],
        "true_y_history": [1.0, 2.0],
        "observed_y_history": [1.1, 2.1],
        "evaluation_wall_clock_history": [3.0, 5.0],
        "wall_clock_seconds": 6.0,
    }
    recorder = runner.NoisyObjectiveRecorder(
        lambda x: objective_calls.append(np.asarray(x).copy()) or 3.0,
        benchmark_name="mopta08_deff4",
        algorithm="HeSBO",
        trial=0,
        max_evals=3,
        replay_trace=source,
    )

    assert recorder([0.1, 0.2]) == pytest.approx(1.1)
    assert recorder([0.3, 0.4]) == pytest.approx(2.1)
    assert objective_calls == []
    assert recorder([0.5, 0.6]) == pytest.approx(3.0)
    assert len(objective_calls) == 1
    assert recorder.replayed_evaluations == 2
    assert recorder.eval_wall_clock_history[:2] == [3.0, 5.0]
    assert recorder.eval_wall_clock_history[2] >= 6.0


def test_objective_recorder_accepts_optimizer_native_warm_start_without_calls():
    objective_calls = []
    source = {
        "n_evaluations": 2,
        "x_history": [[0.1, 0.2], [0.3, 0.4]],
        "true_y_history": [1.0, 2.0],
        "observed_y_history": [1.1, 2.1],
        "evaluation_wall_clock_history": [3.0, 5.0],
        "wall_clock_seconds": 6.0,
    }
    recorder = runner.NoisyObjectiveRecorder(
        lambda x: objective_calls.append(np.asarray(x).copy()) or 3.0,
        benchmark_name="mopta08_deff4",
        algorithm="ALEBO",
        trial=0,
        max_evals=3,
        replay_trace=source,
    )

    recorder.accept_replay_history_as_warm_start()

    assert objective_calls == []
    assert recorder.replayed_evaluations == 2
    assert recorder.x_history == [pytest.approx([0.1, 0.2]), pytest.approx([0.3, 0.4])]
    assert recorder.true_history == [1.0, 2.0]
    assert recorder.observed_history == [1.1, 2.1]
    assert recorder.eval_wall_clock_history == [3.0, 5.0]
    assert recorder.current_wall_clock_seconds() >= 6.0


def test_extract_embedding_matrix_accepts_alebo_B():
    matrix = np.arange(12, dtype=float).reshape(3, 4)
    assert runner.extract_embedding_matrix(SimpleNamespace(B=matrix)) == matrix.tolist()


def test_objective_recorder_rejects_divergent_resume_point():
    recorder = runner.NoisyObjectiveRecorder(
        lambda _x: pytest.fail("stored evaluations must not query the objective"),
        benchmark_name="mopta08_deff4",
        algorithm="HeSBO",
        trial=0,
        max_evals=2,
        replay_trace={
            "n_evaluations": 1,
            "x_history": [[0.1, 0.2]],
            "true_y_history": [1.0],
            "observed_y_history": [1.0],
            "evaluation_wall_clock_history": [1.0],
            "wall_clock_seconds": 1.0,
        },
    )
    with pytest.raises(RuntimeError, match="Resume replay diverged"):
        recorder([0.1, 0.25])


def test_under_budget_resume_source_is_backed_up_outside_trial_glob(tmp_path):
    benchmark = "ackley3_300"
    algorithm = "RandomSearch"
    config = runner.BENCHMARKS[benchmark]
    stored_identity = runner.requested_trace_cache_identity(
        config, benchmark, algorithm, trial=0, max_iter=2
    )
    target_identity = runner.requested_trace_cache_identity(
        config, benchmark, algorithm, trial=0, max_iter=4
    )
    path = runner.trace_path_for(tmp_path, benchmark, algorithm, 0)
    runner.write_json(
        path,
        {
            "benchmark": benchmark,
            "algorithm": algorithm,
            "trial_id": 0,
            "complete": True,
            "error": None,
            "n_evaluations": 2,
            "cache_identity": stored_identity,
            "x_history": [[0.1], [0.2]],
            "true_y_history": [1.0, 2.0],
            "observed_y_history": [1.0, 2.0],
            "evaluation_wall_clock_history": [1.0, 2.0],
            "wall_clock_seconds": 2.5,
        },
    )

    source = runner.load_under_budget_trace_for_resume(
        tmp_path,
        benchmark,
        algorithm,
        0,
        target_max_iter=4,
        expected_identity=target_identity,
    )

    assert source is not None
    assert source["source_evaluations"] == 2
    assert source["backup_path"].exists()
    assert not source["backup_path"].name.startswith("trial_")
    assert path.read_bytes() == source["backup_path"].read_bytes()


def test_objective_tables_are_split_into_bo_and_turbo_folders(monkeypatch, tmp_path):
    monkeypatch.setattr(runner, "save_objective_table_png", lambda *_args, **_kwargs: None)
    grouped = {
        "ackley3_300": {
            "VanillaBO": [
                {
                    "trial": 0,
                    "final_best_noiseless_y_at_max_iter": -2.0,
                    "final_best_observed_at_max_iter": 999.0,
                    "final_best_posterior_mean_at_max_iter": -2.1,
                    "final_best_stoppable_threshold_at_max_iter_1_256": -2.2,
                }
            ],
            "TuRBO": [
                {
                    "trial": 0,
                    "final_best_noiseless_y_at_max_iter": -1.0,
                    "final_best_observed_at_max_iter": 888.0,
                    "final_best_posterior_mean_at_max_iter": -1.1,
                    "final_best_stoppable_threshold_at_max_iter_1_256": -1.2,
                }
            ],
        }
    }

    runner.save_objective_tables(grouped=grouped, table_dir=tmp_path)

    assert (tmp_path / "bo" / "best_noiseless_wilcoxon.tex").exists()
    assert (tmp_path / "turbo" / "best_noiseless_wilcoxon.tex").exists()
    assert (tmp_path / "united" / "best_noiseless_wilcoxon.tex").exists()
    assert not (tmp_path / "best_noiseless_wilcoxon.tex").exists()

    bo_table = (tmp_path / "bo" / "best_noiseless_wilcoxon.tex").read_text(encoding="utf-8")
    turbo_table = (tmp_path / "turbo" / "best_noiseless_wilcoxon.tex").read_text(encoding="utf-8")
    assert "VanillaBO" in bo_table
    assert "TuRBO" not in bo_table
    assert "TuRBO" in turbo_table
    assert "VanillaBO" not in turbo_table
    assert "999.000" not in bo_table
    assert "888.000" not in turbo_table
    assert "-2.000" in bo_table
    assert "-1.000" in turbo_table


@pytest.mark.parametrize("preserve_cached_rows", [False, True])
def test_objective_table_refresh_can_replace_previously_cached_benchmarks(
    monkeypatch, tmp_path, preserve_cached_rows
):
    monkeypatch.setattr(runner, "save_objective_table_png", lambda *_args, **_kwargs: None)

    def traces(value):
        return [{
            "trial": 0,
            "settings": {"experiment_kind": "real_life"},
            "final_best_noiseless_y_at_max_iter": value,
            "final_best_observed_at_max_iter": value,
        }]

    runner.save_objective_tables(
        grouped={
            "nasbench101_deff2": {"LassoBO": traces(0.938), "TuRBO": traces(0.939)},
            "rover60_deff2": {"TuRBO": traces(1.5)},
            "mopta08_deff4": {"LassoBO": traces(-320.0), "TuRBO": traces(-310.0)},
        },
        table_dir=tmp_path,
    )
    options = {} if preserve_cached_rows else {"preserve_cached_rows": False}
    runner.save_objective_tables(
        grouped={"mopta08_deff4": {
            "AG-VS-BO": traces(-290.0), "AG-VS-TuRBO": traces(-280.0),
        }},
        table_dir=tmp_path,
        **options,
    )

    for family in ("bo", "turbo", "united"):
        cache = json.loads((tmp_path / family / "wilcoxon_table_rows.json").read_text())
        for metric in ("best_observed", "best_noiseless"):
            rows = cache[metric]
            benchmarks = {row["benchmark"] for row in rows}
            assert "mopta08_deff4" in benchmarks
            assert ("nasbench101_deff2" in benchmarks) == preserve_cached_rows
            if not preserve_cached_rows:
                assert benchmarks == {"mopta08_deff4"}
                assert all("LassoBO" not in row["algorithms"] for row in rows)
        for table in (tmp_path / family).glob("*.tex"):
            contents = table.read_text(encoding="utf-8")
            assert ("nasbench101" in contents) == preserve_cached_rows
            if not preserve_cached_rows:
                assert "rover60" not in contents
                assert "LassoBO" not in contents
                assert "mopta08" in contents


def test_mcts_cp_is_objective_specific():
    assert runner.mcts_cp_for_benchmark("levy3_300") == 10.0
    assert runner.mcts_cp_for_benchmark("hartmann6_1000") == 0.1
    assert runner.mcts_cp_for_benchmark("mopta08_deff3") == 0.1
    assert runner.mcts_cp_for_benchmark("hopper_deff6") == 50.0
    for benchmark_name, expected in (
        ("levy3_300", 10.0),
        ("hartmann6_1000", 0.1),
        ("mopta08_deff3", 0.1),
        ("hopper_deff6", 50.0),
    ):
        settings = runner.algorithm_settings_summary(
            "LA-MCTS-TuRBO",
            benchmark_name,
        )
        assert settings["cp"] == expected
        assert settings["leaf_size"] == 20
        assert settings["turbo_max_evals"] == 50
        assert settings["posterior_stopping_enabled"] is False
    levy_optimizer = runner.make_mcts_paper_baseline(
        "LA-MCTS-TuRBO",
        runner.BENCHMARKS["levy3_300"],
        recorder=lambda _x: 0.0,
        seed=13,
        max_iter=20,
        benchmark_name="levy3_300",
    )
    assert levy_optimizer.la_mcts_cp == 10.0
    assert levy_optimizer.enable_posterior_stopping is False
    assert levy_optimizer.stopping_threshold is None


@pytest.mark.parametrize("inner_solver", ["bo", "turbo", "rs", "saasbo"])
def test_runner_builds_every_registered_mcts_variant(inner_solver):
    benchmark_name = "ackley3_300"
    optimizer = runner.make_mcts_vs(
        runner.BENCHMARKS[benchmark_name],
        recorder=lambda _x: 0.0,
        seed=11,
        max_iter=30,
        ipt_solver=inner_solver,
        benchmark_name=benchmark_name,
    )
    assert optimizer.ipt_solver == inner_solver
    expected_noise = (
        runner.BO_GP_ASSUMED_NOISE_VAR
        if inner_solver in {"bo", "turbo"}
        else runner.REFERENCE_DEFAULT_GP_NOISE_VAR
    )
    assert optimizer.noise_variance == pytest.approx(expected_noise)


def test_runner_threads_single_device_and_objective_specific_vae_controls():
    ackley_name = "ackley3_300"
    for algorithm in ("Dropout-TuRBO",):
        optimizer = runner.make_mcts_paper_baseline(
            algorithm,
            runner.BENCHMARKS[ackley_name],
            recorder=lambda _x: 0.0,
            seed=13,
            max_iter=60,
            benchmark_name=ackley_name,
        )
        assert optimizer.turbo_max_evals == runner.MCTS_TURBO_MAX_EVALS == 50
        assert optimizer.turbo_device == runner.DEVICE

    hartmann = runner.make_mcts_paper_baseline(
        "VAE-BO",
        runner.BENCHMARKS["hartmann3_300"],
        recorder=lambda _x: 0.0,
        seed=13,
        max_iter=10,
        benchmark_name="hartmann3_300",
    )
    ackley = runner.make_mcts_paper_baseline(
        "VAE-BO",
        runner.BENCHMARKS[ackley_name],
        recorder=lambda _x: 0.0,
        seed=13,
        max_iter=10,
        benchmark_name=ackley_name,
    )
    levy = runner.make_mcts_paper_baseline(
        "VAE-BO",
        runner.BENCHMARKS["levy6_1000"],
        recorder=lambda _x: 0.0,
        seed=13,
        max_iter=10,
        benchmark_name="levy6_1000",
    )
    assert hartmann.vae_update_interval == 30
    assert ackley.vae_update_interval == 20
    assert hartmann.vae_learning_rate == ackley.vae_learning_rate == pytest.approx(0.001)
    assert hartmann.vae_latent_training_bounds == (0.0, 1.0)
    assert hartmann.vae_latent_acquisition_bounds == (0.0, 3.0)
    assert levy.vae_latent_training_bounds == (-10.0, 10.0)
    assert levy.vae_latent_acquisition_bounds == (-60.0, 60.0)
    hartmann_settings = runner.algorithm_settings_summary("VAE-BO", "hartmann3_300")
    assert hartmann_settings["vae_update_interval"] == 30
    assert hartmann_settings["alpha"] == pytest.approx(0.01)
    assert hartmann_settings["xi"] == pytest.approx(1e-4)
    assert hartmann_settings["candidate_count"] == 1024
    assert hartmann_settings["latent_training_bounds"] == [0.0, 1.0]
    assert hartmann_settings["latent_acquisition_bounds"] == [0.0, 3.0]
    assert runner.algorithm_settings_summary("VAE-BO", ackley_name)[
        "vae_update_interval"
    ] == 20
    real_life_settings = runner.algorithm_settings_summary(
        "VAE-BO", "mopta08_deff3"
    )
    assert real_life_settings["vae_update_interval"] == 30
    assert real_life_settings["vae_learning_rate"] == pytest.approx(0.01)


def test_single_device_flag_is_forwarded_to_every_child_process(monkeypatch):
    calls = []
    monkeypatch.setattr(
        runner.subprocess,
        "run",
        lambda command, **kwargs: calls.append((list(command), dict(kwargs))),
    )

    runner.main(
        [
            "--benchmarks",
            "ackley3_300",
            "--algorithms",
            "AG-VS-BO",
            "--trials",
            "1",
            "--max-iter",
            "1",
            "--workers",
            "1",
            "--device",
            "cpu",
            "--skip-plot",
        ]
    )

    assert len(calls) == 1
    command, kwargs = calls[0]
    assert command[command.index("--device") + 1] == "cpu"
    assert command[command.index("--workers") + 1] == "1"
    assert kwargs["env"]["AGVS_DEVICE"] == "cpu"
    assert kwargs["env"]["AGVS_WORKERS"] == "1"
    assert kwargs["cwd"] == str(runner.ROOT)
    assert kwargs["env"]["PYTHONPATH"].split(os.pathsep)[0] == str(runner.ROOT)


def test_turbo_benchmark_child_keeps_current_interpreter(monkeypatch):
    calls = []
    monkeypatch.setattr(
        runner.subprocess,
        "run",
        lambda command, **kwargs: calls.append((list(command), dict(kwargs))),
    )

    runner.main(
        [
            "--benchmarks",
            "ackley3_300",
            "--algorithms",
            "TuRBO",
            "--trials",
            "1",
            "--max-iter",
            "30",
            "--workers",
            "1",
            "--skip-plot",
        ]
    )

    assert len(calls) == 1
    assert Path(calls[0][0][0]) == Path(sys.executable)


def test_synthetic_entrypoint_imports_project_from_an_arbitrary_directory(tmp_path):
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = runner.subprocess.run(
        [
            sys.executable,
            str(runner.ROOT / "Experiment" / "utils.py"),
            "--benchmark",
            "ackley3_300",
            "--help",
        ],
        cwd=str(tmp_path),
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--device {cpu}" in completed.stdout


@pytest.mark.parametrize("rerun", [False, True])
@pytest.mark.parametrize("trial_override", [None, 55])
def test_synthetic_batches_refresh_reports_between_all_function_rounds(monkeypatch, rerun, trial_override):
    events = []
    names = ["ackley3_300", "sphere3_300"]

    def run_child(command, **kwargs):
        assert "--defer-reports" in command
        assert ("--rerun-complete" in command) == rerun
        assert command[command.index("--workers") + 1] == str(runner.WORKERS)
        events.append((
            command[command.index("--benchmark") + 1],
            int(command[command.index("--trial-start") + 1]),
            int(command[command.index("--trials") + 1]),
        ))

    monkeypatch.setattr(runner.subprocess, "run", run_child)
    monkeypatch.setattr(
        runner, "refresh_requested_reports",
        lambda benchmarks, skip_plot=False: events.append(("reports", tuple(benchmarks))),
    )
    runner.main([
        "--benchmarks", *names, "--algorithms", "RandomSearch",
        *(["--trials", str(trial_override)] if trial_override is not None else []),
        *(["--rerun-complete"] if rerun else []),
    ])
    expected = [
        ("ackley3_300", 0, 25), ("sphere3_300", 0, 25), ("reports", tuple(names)),
    ]
    if trial_override == 55:
        expected.extend([
            ("ackley3_300", 25, 50), ("sphere3_300", 25, 50), ("reports", tuple(names)),
            ("ackley3_300", 50, 55), ("sphere3_300", 50, 55), ("reports", tuple(names)),
        ])
    assert events == expected


@pytest.mark.parametrize("skip_complete", [False, True])
def test_trial_batch_excludes_previous_ids_and_resumes_current_batch(
    monkeypatch, tmp_path, skip_complete
):
    calls = []
    benchmark, algorithm = "ackley3_300", "RandomSearch"
    monkeypatch.setattr(runner, "existing_complete_traces", lambda *args, **kwargs: {
        (0, benchmark, algorithm): True, (25, benchmark, algorithm): True,
    })

    def fake_trial(args):
        calls.append(args[0])
        return [{"benchmark": benchmark, "algorithm": algorithm,
                 "trial_id": args[0], "complete": True, "error": None}]

    monkeypatch.setattr(runner, "_run_round_robin_trial", fake_trial)
    status = runner.run_round_robin(
        benchmarks=[benchmark], algorithms=[algorithm], trials=27, trial_start=25,
        max_iter=1, output_dir=str(tmp_path), figure_dir=str(tmp_path / "figures"),
        seed=0, update_plots=False, skip_complete=skip_complete, workers=1,
    )
    assert calls == ([26] if skip_complete else [25, 26])
    assert status["complete"] is True
    assert status["expected_run_count"] == status["completed_run_count"] == 2
    assert status["trial_ids"] == [25, 26]


def test_bare_runner_uses_only_runnable_synthetic_defaults(monkeypatch):
    calls = []
    monkeypatch.setattr(
        runner.subprocess,
        "run",
        lambda command, **kwargs: calls.append((list(command), dict(kwargs))),
    )

    runner.main(
        [
            "--trials",
            "1",
            "--max-iter",
            "1",
            "--workers",
            "1",
            "--device",
            "cpu",
            "--skip-plot",
        ]
    )

    assert len(runner.SYNTHETIC_BENCHMARKS) == 10
    assert len(calls) == len(runner.DEFAULT_BENCHMARKS) == 10
    assert all(
        not any("hartmann" in str(part).lower() for part in command)
        for command, _kwargs in calls
    )
    for command, _kwargs in calls:
        selected = command[command.index("--algorithms") + 1 : command.index("--benchmark")]
        assert selected == runner.DEFAULT_ALGORITHMS
        assert not set(selected) & runner.NONDEFAULT_REFERENCE_ALGORITHMS


def test_user_editable_runner_selections_apply_without_cli_selectors(monkeypatch):
    calls = []
    monkeypatch.setattr(runner, "RUNNER_BENCHMARKS", ("sphere3_300",))
    monkeypatch.setattr(runner, "RUNNER_ALGORITHMS", ("RandomSearch",))
    monkeypatch.setattr(
        runner.subprocess,
        "run",
        lambda command, **kwargs: calls.append((list(command), dict(kwargs))),
    )

    runner.main(["--trials", "1", "--max-iter", "1", "--workers", "1", "--skip-plot"])

    assert len(calls) == 1
    command, _kwargs = calls[0]
    assert command[command.index("--benchmark") + 1] == "sphere3_300"
    selected = command[command.index("--algorithms") + 1 : command.index("--benchmark")]
    assert selected == ["RandomSearch"]


def test_explicit_real_task_defaults_match_active_synthetic_result_algorithms(monkeypatch):
    calls = []
    monkeypatch.setattr(
        runner.subprocess,
        "run",
        lambda command, **kwargs: calls.append((list(command), dict(kwargs))),
    )

    runner.main(
        [
            "--benchmarks",
            "mopta08_deff3",
            "--trials",
            "1",
            "--max-iter",
            "1",
            "--workers",
            "1",
            "--device",
            "cpu",
            "--skip-plot",
        ]
    )

    assert len(calls) == 1
    command, _kwargs = calls[0]
    selected = command[command.index("--algorithms") + 1 : command.index("--max-iter")]
    assert selected == list(runner.REAL_LIFE_ALGORITHMS)
    assert selected[:2] == ["AG-VS-BO", "AG-VS-TuRBO"]


def test_mopta_algorithms_share_the_registered_budget():
    from Experiment.benchmark import mopta
    for algorithm in mopta.ALGORITHMS:
        assert runner.algorithm_evaluation_budget(algorithm, 150, "mopta08_deff4") == 150



def test_all_algorithms_use_the_same_cpu_device_for_fair_timing():
    config = runner.BENCHMARKS["ackley3_300"]
    optimizer = runner.build_optimizer(
        "RandomSearch",
        config,
        recorder=lambda _x: 0.0,
        seed=3,
        max_iter=2,
        benchmark_name="ackley3_300",
        device="cpu",
    )
    assert optimizer.configured_device == "cpu"
    for algorithm in runner.DEFAULT_ALGORITHMS:
        actual = runner.algorithm_settings_summary(
            algorithm,
            "ackley3_300",
            device="cpu",
        )["device"]
        assert actual == "cpu"

    for algorithm in runner.TURBO_BACKEND_ALGORITHMS:
        turbo_identity = runner.requested_trace_cache_identity(
            config,
            "ackley3_300",
            algorithm,
            trial=0,
            max_iter=700,
            device="cpu",
        )
        assert turbo_identity["device"] == "cpu"
        assert turbo_identity["algorithm_settings"]["device"] == "cpu"
        assert turbo_identity["algorithm_settings"]["min_cuda"] == 0
        assert turbo_identity["algorithm_settings"]["dtype"] == "float32"


@pytest.mark.filterwarnings(
    "ignore:This process .* is multi-threaded, use of fork\\(\\) may lead to deadlocks:DeprecationWarning"
)
def test_single_device_reaches_parallel_workers_and_traces(tmp_path):
    runner.run_round_robin(
        benchmarks=["ackley3_300"],
        algorithms=["RandomSearch"],
        trials=2,
        max_iter=2,
        output_dir=str(tmp_path),
        figure_dir=str(tmp_path / "figures"),
        seed=0,
        update_plots=False,
        skip_complete=False,
        workers=2,
        device="cpu",
        table_dir=str(tmp_path / "tables"),
    )

    traces = runner.load_traces(tmp_path)
    assert len(traces) == 2
    assert {trace["settings"]["device"] for trace in traces} == {"cpu"}
    status = json.loads(
        (tmp_path / "ackley3_300" / "run_status.json").read_text(
            encoding="utf-8"
        )
    )
    assert status["workers"] == 2
    assert status["device"] == "cpu"
    assert set(status["experiment_kinds"]) == {"ackley3_300"}
    assert set(status["active_indices"]) == {"ackley3_300"}
    assert set(status["effective_dims"]) == {"ackley3_300"}


def test_algorithms_run_in_benchmark_local_barrier_phases(monkeypatch, tmp_path):
    calls = []

    def fake_trial(args):
        pairs = list(args[10])
        calls.append(pairs)
        return [
            {
                "benchmark": benchmark,
                "algorithm": algorithm,
                "trial_id": args[0],
                "complete": True,
                "error": None,
            }
            for benchmark, algorithm in pairs
        ]

    monkeypatch.setattr(runner, "_run_round_robin_trial", fake_trial)
    monkeypatch.setattr(runner, "preflight_algorithm_devices", lambda _algorithms: None)
    status = runner.run_round_robin(
        benchmarks=["ackley3_300"],
        algorithms=["RandomSearch", "AG-VS-TuRBO", "TuRBO", "Dropout-TuRBO"],
        trials=1,
        max_iter=2,
        output_dir=str(tmp_path),
        figure_dir=str(tmp_path / "figures"),
        seed=0,
        update_plots=False,
        skip_complete=False,
        workers=1,
        device="cpu",
        table_dir=str(tmp_path / "tables"),
    )

    assert calls == [
        [("ackley3_300", "AG-VS-TuRBO")],
        [("ackley3_300", "RandomSearch")],
        [("ackley3_300", "TuRBO")],
        [("ackley3_300", "Dropout-TuRBO")],
    ]
    assert status["algorithm_devices"] == {
        "RandomSearch": "cpu",
        "AG-VS-TuRBO": "cpu",
        "TuRBO": "cpu",
        "Dropout-TuRBO": "cpu",
    }
    assert status["requested_workers"] == 1
    assert status["cpu_workers"] == 1
    assert status["order"] == (
        "for benchmark -> for algorithm -> parallel trials; algorithm barrier"
    )
    assert status["algorithm_phase_order"] == [
        "AG-VS-TuRBO",
        "RandomSearch",
        "TuRBO",
        "Dropout-TuRBO",
    ]


def test_fair_cpu_policy_disables_gpu_worker_pool():
    assert runner.algorithm_uses_gpu_worker_pool("AG-VS-BO", "cuda") is False
    assert runner.algorithm_uses_gpu_worker_pool("AG-VS-BO", "cpu") is False
    assert runner.algorithm_uses_gpu_worker_pool("TuRBO", "cpu") is False
    assert runner.algorithm_uses_gpu_worker_pool("RandomSearch", "cuda") is False
    assert runner.algorithm_device("AG-VS-BO", "cuda") == "cpu"
    assert runner.algorithm_device("TuRBO", "cuda") == "cpu"
    assert runner.algorithm_worker_limit("AG-VS-BO", 25, "cuda") == 25
    assert runner.algorithm_worker_limit("AG-VS-BO", 3, "cuda") == 3
    assert runner.algorithm_worker_limit("AG-VS-BO", 25, "cpu") == 25


def test_parallel_worker_failure_recovers_written_json_and_retries_only_missing(
    monkeypatch, tmp_path
):
    scans = 0
    sequential_trials = []
    executor_settings = []

    def fake_existing(*_args, **_kwargs):
        nonlocal scans
        scans += 1
        if scans == 1:
            return {}
        return {
            (0, "ackley3_300", "RandomSearch"): {
                "complete": True,
                "error": None,
            }
        }

    class BrokenExecutor:
        def __init__(self, **kwargs):
            executor_settings.append(dict(kwargs))

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def map(self, *_args, **_kwargs):
            raise RuntimeError("worker disappeared")

    def fake_trial(args):
        sequential_trials.append(int(args[0]))
        return [{
            "benchmark": "ackley3_300",
            "algorithm": "RandomSearch",
            "trial_id": int(args[0]),
            "complete": True,
            "error": None,
        }]

    monkeypatch.setattr(runner, "existing_complete_traces", fake_existing)
    monkeypatch.setattr(runner.concurrent.futures, "ProcessPoolExecutor", BrokenExecutor)
    monkeypatch.setattr(runner, "_run_round_robin_trial", fake_trial)

    status = runner.run_round_robin(
        benchmarks=["ackley3_300"],
        algorithms=["RandomSearch"],
        trials=2,
        max_iter=2,
        output_dir=str(tmp_path),
        figure_dir=str(tmp_path / "figures"),
        seed=0,
        update_plots=False,
        skip_complete=True,
        workers=2,
        device="cpu",
        table_dir=str(tmp_path / "tables"),
    )

    assert sequential_trials == [1]
    assert executor_settings[0]["max_workers"] == 2
    assert status["complete"] is True
    assert status["completed_run_count"] == 2


def test_trace_cache_requires_exact_identity_and_requested_budget(tmp_path):
    benchmark = "ackley3_300"
    algorithm = "RandomSearch"
    config = runner.BENCHMARKS[benchmark]
    identity = runner.requested_trace_cache_identity(
        config,
        benchmark,
        algorithm,
        trial=0,
        max_iter=10,
        base_seed_offset=0,
        device="cpu",
    )
    path = runner.trace_path_for(tmp_path, benchmark, algorithm, 0)
    trace = {
        "benchmark": benchmark,
        "algorithm": algorithm,
        "trial_id": 0,
        "complete": True,
        "error": None,
        "n_evaluations": 10,
        "stopped_early": False,
        "stop_iteration": None,
        "cache_identity": identity,
    }
    runner.write_json(path, trace)

    assert runner.load_existing_trace_if_complete(
        tmp_path,
        benchmark,
        algorithm,
        0,
        max_iter=10,
        cache_identity=identity,
    ) is not None

    different_seed_identity = json.loads(json.dumps(identity))
    different_seed_identity["algorithm_seed"] += 1
    assert runner.load_existing_trace_if_complete(
        tmp_path,
        benchmark,
        algorithm,
        0,
        max_iter=10,
        cache_identity=different_seed_identity,
    ) is None

    legacy_trace = dict(trace)
    legacy_trace.pop("cache_identity")
    runner.write_json(path, legacy_trace)
    assert runner.load_existing_trace_if_complete(
        tmp_path,
        benchmark,
        algorithm,
        0,
        max_iter=10,
        cache_identity=identity,
    ) is None

    trace.update(n_evaluations=6, stopped_early=True, stop_iteration=6)
    runner.write_json(path, trace)
    assert runner.load_existing_trace_if_complete(
        tmp_path,
        benchmark,
        algorithm,
        0,
        max_iter=10,
        cache_identity=identity,
    ) is None

    trace.update(
        complete=False,
        error="optimizer raised after its final objective evaluation",
        interrupted=True,
        n_evaluations=10,
        stopped_early=False,
        stop_iteration=None,
    )
    runner.write_json(path, trace)
    recovered = runner.load_existing_trace_if_complete(
        tmp_path,
        benchmark,
        algorithm,
        0,
        max_iter=10,
        cache_identity=identity,
    )
    assert recovered is not None
    assert recovered["complete"] is True
    assert recovered["error"] is None
    assert recovered["interrupted"] is False
    assert recovered["budget_completion_recovery"]["previous_complete"] is False
    assert recovered["budget_completion_recovery"]["requested_max_iter"] == 10


def test_real_life_cache_still_requires_matching_seeds():
    benchmark = "mopta08_deff4"
    config = make_real_life_config("mopta08", 4)
    expected = runner.requested_trace_cache_identity(
        config, benchmark, "AG-VS-BO", trial=0, max_iter=12
    )
    stored = json.loads(json.dumps(expected))
    stored["algorithm_seed"] += 1

    assert runner.cache_identities_match(stored, expected) is False


def test_synthetic_cache_accepts_legacy_missing_null_provenance_fields():
    identity = runner.requested_trace_cache_identity(
        runner.BENCHMARKS["ackley3_300"],
        "ackley3_300",
        "AG-VS-BO",
        trial=0,
        max_iter=700,
    )
    legacy = json.loads(json.dumps(identity))
    legacy["benchmark_config"].pop("dataset_identifier")
    legacy["benchmark_config"].pop("objective_protocol")

    assert runner.cache_identities_match(legacy, identity)

    configured_protocol = json.loads(json.dumps(identity))
    configured_protocol["benchmark_config"]["objective_protocol"] = "changed"
    assert not runner.cache_identities_match(legacy, configured_protocol)


def test_non_agvs_cache_ignores_historical_n_bad_reporting_fields():
    identity = runner.requested_trace_cache_identity(
        runner.BENCHMARKS["ackley3_300"],
        "ackley3_300",
        "TuRBO",
        trial=0,
        max_iter=700,
    )
    legacy = json.loads(json.dumps(identity))
    legacy["benchmark_config"]["n_bad_multiplier"] = 4
    legacy["benchmark_config"]["n_bad_enabled"] = True

    assert runner.cache_identities_match(legacy, identity)


@pytest.mark.parametrize("algorithm", ["SAASBO", "ALEBO", "GTBO"])
@pytest.mark.parametrize("benchmark", ["ackley3_300", "mopta08_deff3"])
def test_capped_reference_budget_is_reused_by_scheduler(
    monkeypatch, tmp_path, algorithm, benchmark
):
    if benchmark == "mopta08_deff3":
        monkeypatch.setitem(
            runner.BENCHMARKS, benchmark, make_real_life_config("mopta08", 3)
        )
    config = runner.BENCHMARKS[benchmark]
    identity = runner.requested_trace_cache_identity(
        config, benchmark, algorithm, trial=0, max_iter=700
    )
    assert identity["requested_max_iter"] == 200
    path = runner.trace_path_for(tmp_path, benchmark, algorithm, 0)
    trace = {
        "benchmark": benchmark,
        "algorithm": algorithm,
        "trial_id": 0,
        "complete": True,
        "error": None,
        "n_evaluations": 199,
        "cache_identity": identity,
    }
    runner.write_json(path, trace)
    assert not runner.existing_complete_traces(
        tmp_path, [benchmark], [algorithm], 1, max_iter=700
    )
    trace["n_evaluations"] = 200
    runner.write_json(path, trace)
    assert (0, benchmark, algorithm) in runner.existing_complete_traces(
        tmp_path, [benchmark], [algorithm], 1, max_iter=700
    )
    assert runner.algorithms_requiring_runtime_for_request(
        output_dir=tmp_path, benchmark=benchmark, algorithms=[algorithm],
        trials=1, max_iter=700,
    ) == []


@pytest.mark.parametrize("algorithm", ["SAASBO", "ALEBO", "GTBO"])
@pytest.mark.parametrize("requested, expected", [(700, 200), (150, 150)])
def test_reference_trial_passes_capped_budget_to_optimizer(
    monkeypatch, tmp_path, algorithm, requested, expected
):
    def check_optimizer_budget(*args, **kwargs):
        assert kwargs["max_iter"] == expected
        assert args[2].max_evals == expected
        raise RuntimeError("budget verified before optimizer startup")

    monkeypatch.setattr(runner, "build_optimizer", check_optimizer_budget)
    with pytest.raises(RuntimeError, match="budget verified"):
        runner.run_single_trial(
            "ackley3_300", algorithm, 0, requested, str(tmp_path), 0,
            skip_complete=False,
        )


def test_agvs_cache_still_tracks_n_bad_settings():
    identity = runner.requested_trace_cache_identity(
        runner.BENCHMARKS["ackley3_300"],
        "ackley3_300",
        "AG-VS-BO",
        trial=0,
        max_iter=700,
    )
    changed = json.loads(json.dumps(identity))
    changed["benchmark_config"]["n_bad_multiplier"] += 1

    assert not runner.cache_identities_match(changed, identity)


@pytest.mark.parametrize(
    "algorithm",
    ["AG-VS-BO", "MCTS-VS-BO", "TuRBO", "HeSBO", "ALEBO", "VAE-BO"],
)
def test_budget_complete_cache_recovery_is_algorithm_independent(
    tmp_path, algorithm
):
    benchmark = "ackley3_300"
    identity = runner.requested_trace_cache_identity(
        runner.BENCHMARKS[benchmark], benchmark, algorithm, trial=0, max_iter=10
    )
    path = runner.trace_path_for(tmp_path, benchmark, algorithm, 0)
    runner.write_json(
        path,
        {
            "benchmark": benchmark,
            "algorithm": algorithm,
            "trial_id": 0,
            "complete": False,
            "interrupted": False,
            "error": "stale adapter completion error",
            "n_evaluations": 10,
            "cache_identity": identity,
        },
    )

    reused = runner.load_existing_trace_if_complete(
        tmp_path,
        benchmark,
        algorithm,
        0,
        max_iter=10,
        cache_identity=identity,
    )
    assert reused is not None
    assert reused["complete"] is True
    assert reused["error"] is None


def test_agvs_legacy_n_bad_cache_is_migrated_only_from_recorded_settings(tmp_path):
    benchmark = "ackley3_300"
    algorithm = "AG-VS-BO"
    identity = runner.requested_trace_cache_identity(
        runner.BENCHMARKS[benchmark],
        benchmark,
        algorithm,
        trial=0,
        max_iter=10,
    )
    legacy_identity = json.loads(json.dumps(identity))
    legacy_identity["benchmark_config"].pop("n_bad_enabled")
    legacy_identity["benchmark_config"].pop("n_bad_multiplier")
    path = runner.trace_path_for(tmp_path, benchmark, algorithm, 0)
    runner.write_json(
        path,
        {
            "benchmark": benchmark,
            "algorithm": algorithm,
            "trial_id": 0,
            "complete": True,
            "error": None,
            "n_evaluations": 10,
            "cache_identity": legacy_identity,
            "settings": {
                "algorithm_settings": {
                    "N_bad_enabled": identity["benchmark_config"]["n_bad_enabled"],
                    "N_bad_multiplier": identity["benchmark_config"]["n_bad_multiplier"],
                }
            },
        },
    )

    assert runner.load_existing_trace_if_complete(
        tmp_path,
        benchmark,
        algorithm,
        0,
        max_iter=10,
        cache_identity=identity,
    ) is not None
    migrated = json.loads(path.read_text(encoding="utf-8"))
    assert migrated["cache_identity"]["benchmark_config"]["n_bad_enabled"] is True
    assert migrated["cache_identity"]["benchmark_config"]["n_bad_multiplier"] == 4
    assert migrated["cache_identity_migrations"] == [
        "n_bad_from_recorded_algorithm_settings_v1"
    ]

    migrated["cache_identity"]["benchmark_config"].pop("n_bad_enabled")
    migrated["cache_identity"]["benchmark_config"].pop("n_bad_multiplier")
    migrated["cache_identity_migrations"] = []
    migrated["settings"]["algorithm_settings"]["N_bad_multiplier"] = 99
    runner.write_json(path, migrated)
    assert runner.load_existing_trace_if_complete(
        tmp_path,
        benchmark,
        algorithm,
        0,
        max_iter=10,
        cache_identity=identity,
    ) is None
    rejected = json.loads(path.read_text(encoding="utf-8"))
    # The migration preserves the trace's recorded value, but the strict cache
    # comparison above correctly refuses it for a current multiplier-4 request.
    assert rejected["cache_identity"]["benchmark_config"]["n_bad_multiplier"] == 99


def test_turbo_cache_ignores_reporting_metadata_but_tracks_hyperparameters():
    identity = runner.requested_trace_cache_identity(
        runner.BENCHMARKS["ackley3_300"],
        "ackley3_300",
        "TuRBO",
        trial=0,
        max_iter=700,
        base_seed_offset=0,
        device="cpu",
    )
    older_identity = json.loads(json.dumps(identity))
    for key in ("surrogate_family", "gp_library", "small_data_runtime"):
        older_identity["algorithm_settings"].pop(key, None)
    older_identity["algorithm_settings"]["posterior_stopping_check"] = (
        "older reporting-only wording"
    )
    older_identity["benchmark_config"]["q_batch"] = 1
    older_identity["benchmark_config"]["selection_batch_size"] = 2
    older_identity["benchmark_config"].pop("wall_clock_timing_policy", None)

    assert runner.cache_identities_match(older_identity, identity)

    changed_hyperparameter = json.loads(json.dumps(older_identity))
    changed_hyperparameter["algorithm_settings"]["batch_size"] = 1
    assert not runner.cache_identities_match(changed_hyperparameter, identity)


def test_cache_ignores_stopping_description_but_tracks_stopping_behavior():
    identity = runner.requested_trace_cache_identity(
        runner.BENCHMARKS["ackley3_300"],
        "ackley3_300",
        "Dropout-BO",
        trial=0,
        max_iter=700,
        base_seed_offset=0,
        device="cpu",
    )
    older_wording = json.loads(json.dumps(identity))
    older_wording["algorithm_settings"]["posterior_stopping_check"] = (
        "end of every optimizer iteration/batch after the fixed-noise GP refit"
    )
    assert runner.cache_identities_match(older_wording, identity)

    changed_behavior = json.loads(json.dumps(older_wording))
    changed_behavior["algorithm_settings"]["posterior_stopping_enabled"] = True
    assert not runner.cache_identities_match(changed_behavior, identity)


def test_agvs_cache_ignores_scheduler_descriptions_but_tracks_scheduler_values():
    identity = runner.requested_trace_cache_identity(
        runner.BENCHMARKS["ackley3_300"],
        "ackley3_300",
        "AG-VS-BO",
        trial=0,
        max_iter=700,
    )
    older_identity = json.loads(json.dumps(identity))
    for key in (
        "node_visit_policy",
        "turbo_backend_batch_policy",
        "final_batch_budget_rule",
    ):
        older_identity["algorithm_settings"].pop(key, None)
    assert runner.cache_identities_match(older_identity, identity)

    changed_scheduler = json.loads(json.dumps(older_identity))
    changed_scheduler["algorithm_settings"]["batch_size"] = 2
    assert not runner.cache_identities_match(changed_scheduler, identity)


def test_benchmark_entrypoint_refreshes_figures_only_after_running(
    monkeypatch, tmp_path
):
    results = tmp_path / "results"
    figures = tmp_path / "figures"
    tables = tmp_path / "tables"
    monkeypatch.setattr(runner, "SYNTHETIC_RESULT_ROOT", results)
    monkeypatch.setattr(runner, "SYNTHETIC_FIGURE_ROOT", figures)
    monkeypatch.setattr(runner, "SYNTHETIC_TABLE_ROOT", tables)
    calls = []
    monkeypatch.setattr(
        runner,
        "plot_results",
        lambda *_args, **_kwargs: calls.append("plot") or [],
    )
    monkeypatch.setattr(
        runner,
        "run_round_robin",
        lambda **_kwargs: calls.append("run")
        or {"complete": True, "status_path": str(tmp_path / "run_status.json")},
    )

    runner.run_benchmark(
        "ackley3_300",
        runner.BENCHMARKS["ackley3_300"],
        tmp_path,
        argv=["--trials", "1", "--algorithms", "RandomSearch"],
    )

    assert calls == ["run", "plot"]


def test_plot_policy_uses_sparse_95ci_error_bars_without_filled_regions(monkeypatch):
    assert runner.TRIALS == 25
    assert runner.PLOT_CONFIDENCE_ERROR_BARS is True
    assert runner.PLOT_POSTERIOR_MEAN_DIAGNOSTICS is False
    assert runner.PLOT_TERMINAL_DIAGNOSTICS is False
    monkeypatch.setattr(
        runner.plt,
        "fill_between",
        lambda *_args, **_kwargs: pytest.fail("CI plots must not draw filled bands"),
    )
    calls = []
    monkeypatch.setattr(
        runner.plt,
        "errorbar",
        lambda *args, **kwargs: calls.append((args, kwargs)) or object(),
    )
    result_a = runner.plot_confidence_band(
        np.arange(701),
        np.zeros(701),
        np.ones(701),
        color="red",
        series_index=0,
        series_count=6,
    )
    result_b = runner.plot_confidence_band(
        np.arange(701),
        np.zeros(701),
        np.ones(701),
        color="blue",
        series_index=1,
        series_count=6,
    )
    assert result_a is not None and result_b is not None
    assert len(calls) == 2
    assert list(calls[0][0][0][:3]) == [10.0, 70.0, 130.0]
    assert list(calls[1][0][0][:3]) == [20.0, 80.0, 140.0]
    assert set(calls[0][0][0]).isdisjoint(set(calls[1][0][0]))
    assert np.allclose(calls[0][1]["yerr"], 0.5)
    assert calls[0][1]["fmt"] == "none"


def test_short_nas_grid_gives_every_algorithm_two_non_overlapping_ci_whiskers(
    monkeypatch,
):
    calls = []
    monkeypatch.setattr(
        runner.plt,
        "errorbar",
        lambda *args, **kwargs: calls.append((args, kwargs)) or object(),
    )
    for series_index in range(11):
        runner.plot_confidence_band(
            np.arange(151),
            np.zeros(151),
            np.ones(151),
            series_index=series_index,
            series_count=11,
        )

    positions = [set(call[0][0]) for call in calls]
    assert all(len(series_positions) == 2 for series_positions in positions)
    assert all(
        positions[left].isdisjoint(positions[right])
        for left in range(len(positions))
        for right in range(left + 1, len(positions))
    )
    assert sum(map(len, positions)) == 22
    assert runner.algorithm_plot_style("AG-VS-BO")["marker"] is None


def test_short_wall_clock_grid_gives_every_algorithm_two_ci_whiskers(monkeypatch):
    calls = []
    monkeypatch.setattr(
        runner.plt,
        "errorbar",
        lambda *args, **kwargs: calls.append((args, kwargs)) or object(),
    )

    for series_index in range(11):
        runner.plot_confidence_band(
            np.linspace(0.0, 40.0, 81),
            np.zeros(81),
            np.ones(81),
            series_index=series_index,
            series_count=11,
        )

    positions = [set(call[0][0]) for call in calls]
    assert all(len(series_positions) == 2 for series_positions in positions)
    assert all(
        positions[left].isdisjoint(positions[right])
        for left in range(len(positions))
        for right in range(left + 1, len(positions))
    )


def test_synthetic_algorithms_use_unique_colors_without_markers():
    styles = [
        runner.algorithm_plot_style(algorithm)
        for algorithm in runner.DEFAULT_ALGORITHMS
    ]

    assert all(style["marker"] is None for style in styles)
    assert len({style["color"] for style in styles}) == len(styles)
    assert all(style["linestyle"] in {"-", "--", "-.", ":"} for style in styles)


def test_terminal_wall_clock_summary_uses_each_trials_final_finite_value():
    mean, ci, n = runner.terminal_mean_ci_from_histories(
        ([1.0, 3.0], [2.0, float("nan"), 5.0], [])
    )

    assert n == 2
    assert mean == pytest.approx(4.0)
    assert ci == pytest.approx(1.96)


def test_wall_clock_objective_grid_is_reported_in_milliseconds():
    items = [
        {
            "algorithm": "AG-VS-BO",
            "n_evaluations": 2,
            "wall_clock_seconds": 0.003,
            "evaluation_wall_clock_history": [0.001, 0.003],
            "best_observed_history": [1.0, 2.0],
            "settings": {"batch_size": 1},
        }
        for _ in range(2)
    ]

    x_axis, mean, _ci, n = runner.wall_clock_mean_ci_from_traces(
        items,
        lambda item: item["best_observed_history"],
        "AG-VS-BO",
    )

    assert n == 2
    assert x_axis.tolist() == [1.0, 2.0, 3.0]
    assert mean.tolist() == [1.0, 1.0, 2.0]


def test_wall_clock_objective_grid_supports_500_millisecond_steps():
    items = [
        {
            "algorithm": "AG-VS-BO",
            "n_evaluations": 2,
            "wall_clock_seconds": 1.5,
            "evaluation_wall_clock_history": [0.5, 1.5],
            "best_observed_history": [1.0, 2.0],
            "settings": {"batch_size": 1},
        }
        for _ in range(2)
    ]

    x_axis, mean, _ci, n = runner.wall_clock_mean_ci_from_traces(
        items,
        lambda item: item["best_observed_history"],
        "AG-VS-BO",
        minimum_grid_step_milliseconds=500,
        maximum_time_milliseconds=10000,
    )

    assert n == 2
    assert x_axis.tolist() == [500.0, 1000.0, 1500.0]
    assert mean.tolist() == [1.0, 1.0, 2.0]


def test_wall_clock_objective_grid_can_extend_completed_trials_to_terminal_mean():
    items = [
        {
            "algorithm": "AG-VS-TuRBO",
            "n_evaluations": 2,
            "wall_clock_seconds": terminal_time,
            "evaluation_wall_clock_history": [0.5, terminal_time],
            "best_observed_history": [1.0, terminal_value],
            "settings": {"batch_size": 1},
        }
        for terminal_time, terminal_value in ((1.5, 2.0), (2.6, 4.0))
    ]

    x_axis, mean, _ci, n = runner.wall_clock_mean_ci_from_traces(
        items,
        lambda item: item["best_observed_history"],
        "AG-VS-TuRBO",
        minimum_grid_step_milliseconds=500,
        extend_completed_trials=True,
    )

    assert n == 2
    assert x_axis.tolist() == [500.0, 1000.0, 1500.0, 2000.0, 2500.0, 2600.0]
    assert mean[-1] == pytest.approx(3.0)


def test_wall_clock_objective_grid_excludes_n_init_duration():
    items = [
        {
            "algorithm": "TuRBO",
            "n_evaluations": 3,
            "wall_clock_seconds": 0.7,
            "evaluation_wall_clock_history": [0.1, 0.2, 0.7],
            "best_observed_history": [1.0, 2.0, 3.0],
            "settings": {"algorithm_settings": {"n_init": 2, "batch_size": 1}},
        }
        for _ in range(2)
    ]

    x_axis, mean, _ci, n = runner.wall_clock_mean_ci_from_traces(
        items,
        lambda item: item["best_observed_history"],
        "TuRBO",
        minimum_grid_step_milliseconds=250,
    )

    assert n == 2
    assert x_axis.tolist() == [0.0, 250.0, 500.0]
    assert mean.tolist() == [2.0, 2.0, 3.0]


def test_wall_clock_objective_grid_can_include_n_init_duration():
    items = [
        {
            "algorithm": "TuRBO",
            "n_evaluations": 3,
            "wall_clock_seconds": 0.7,
            "evaluation_wall_clock_history": [0.1, 0.2, 0.7],
            "best_observed_history": [1.0, 2.0, 3.0],
            "settings": {"algorithm_settings": {"n_init": 2, "batch_size": 1}},
        }
        for _ in range(2)
    ]

    x_axis, mean, _ci, n = runner.wall_clock_mean_ci_from_traces(
        items,
        lambda item: item["best_observed_history"],
        "TuRBO",
        minimum_grid_step_milliseconds=250,
        exclude_initial_design=False,
        extend_completed_trials=True,
    )

    assert n == 2
    assert x_axis.tolist() == [100.0, 350.0, 600.0, 700.0]
    assert mean.tolist() == [1.0, 2.0, 2.0, 3.0]


def test_plot_results_creates_final_wall_clock_ranking_bar(tmp_path):
    traces = []
    for algorithm, terminal_seconds in (("TuRBO", 8.0), ("RandomSearch", 3.0)):
        traces.append(
            {
                "benchmark": "ackley3_300",
                "algorithm": algorithm,
                "trial_id": 0,
                "complete": True,
                "n_evaluations": 2,
                "wall_clock_seconds": terminal_seconds,
                "evaluation_wall_clock_history": [
                    terminal_seconds / 2.0,
                    terminal_seconds,
                ],
                "best_observed_history": [-2.0, -1.0],
                "best_noiseless_y_history": [-2.0, -1.0],
                "settings": {"experiment_kind": "synthetic"},
            }
        )

    figure_root = tmp_path / "figures"
    paths = runner.plot_results(
        tmp_path / "results",
        figure_root,
        benchmark_filter="ackley3_300",
        update_objective_tables=False,
        algorithm_family="united",
        _loaded_traces=traces,
        _top_algorithms_by_benchmark={
            "ackley3_300": ["RandomSearch", "TuRBO"]
        },
        _generate_first_n_figures=False,
    )

    expected = (
        figure_root
        / "Ackley3_300"
        / "final_iteration_wall_clock_bar_sorted.pdf"
    )
    assert expected.is_file()
    assert str(expected) in paths


def test_synthetic_comparison_pdf_pairs_noiseless_and_wall_clock_panels(
    tmp_path, monkeypatch
):
    result_root = tmp_path / "results"
    figure_root = tmp_path / "figures"
    noiseless_history_calls = []
    original_noiseless_history = runner.trace_noiseless_history_for_plot

    def tracked_noiseless_history(item):
        noiseless_history_calls.append(item["benchmark"])
        return original_noiseless_history(item)

    monkeypatch.setattr(
        runner,
        "trace_noiseless_history_for_plot",
        tracked_noiseless_history,
    )
    for benchmark, lower in (("ackley3_300", -4.0), ("ackley6_1000", -8.0)):
        for algorithm, seconds in (("VanillaBO", 2.0), ("TuRBO", 4.0)):
            trace = {
                "benchmark": benchmark,
                "algorithm": algorithm,
                "trial_id": 0,
                "algorithm_seed": 2021,
                "benchmark_seed": 2021,
                "cache_identity": runner.requested_trace_cache_identity(
                    runner.apply_runner_agvs_batch_policy(runner.BENCHMARKS[benchmark]),
                    benchmark, algorithm, trial=0, max_iter=2),
                "trial": 0,
                "complete": True,
                "n_evaluations": 2,
                "wall_clock_seconds": seconds,
                "evaluation_wall_clock_history": [seconds / 2.0, seconds],
                "best_observed_history": [-100.0, -99.0],
                "best_noiseless_y_history": [lower, -1.0],
                "settings": {
                    "experiment_kind": "synthetic",
                    "max_iter": 2,
                    "original_dim": 300 if benchmark.endswith("300") else 1000,
                },
            }
            runner.write_json(
                runner.trace_path_for(result_root, benchmark, algorithm, 0),
                trace,
            )

    paths = runner.save_synthetic_comparison_pdfs(
        benchmark_root=result_root,
        figure_root=figure_root,
        benchmark_names=("ackley3_300", "ackley6_1000"),
    )

    for family in runner.FIGURE_OUTPUT_FAMILIES:
        assert (figure_root / family / "ackley_graphs.pdf").is_file()
        assert (
            figure_root / "first_5_trials" / family / "ackley_graphs.pdf"
        ).is_file()
    # Three family comparison PDFs plus the united wall-clock PDF are emitted
    # for both the all-trial and first-five-trial views.
    assert len(paths) == 8
    assert noiseless_history_calls
    assert set(noiseless_history_calls) == {"ackley3_300", "ackley6_1000"}


def test_round_robin_reruns_identity_matching_finished_under_budget_trial(
    monkeypatch, tmp_path
):
    benchmark = "ackley3_300"
    algorithm = "RandomSearch"
    max_iter = 10
    identity = runner.requested_trace_cache_identity(
        runner.BENCHMARKS[benchmark], benchmark, algorithm, trial=0, max_iter=max_iter
    )
    runner.write_json(
        runner.trace_path_for(tmp_path, benchmark, algorithm, 0),
        {
            "benchmark": benchmark,
            "algorithm": algorithm,
            "trial_id": 0,
            "complete": True,
            "error": None,
            "n_evaluations": 4,
            "stopped_early": False,
            "cache_identity": identity,
        },
    )
    submitted = []
    monkeypatch.setattr(
        runner,
        "run_single_trial",
        lambda **kwargs: submitted.append(kwargs),
    )

    status = runner.run_round_robin(
        benchmarks=[benchmark],
        algorithms=[algorithm],
        trials=1,
        max_iter=max_iter,
        output_dir=str(tmp_path),
        figure_dir=str(tmp_path / "figures"),
        seed=0,
        update_plots=False,
        skip_complete=True,
        workers=1,
        device="cpu",
        table_dir=str(tmp_path / "tables"),
    )

    assert len(submitted) == 1
    assert submitted[0]["trial"] == 0
    assert status["complete"] is False
    assert status["completed_run_count"] == 0


def test_trace_cache_ignores_duplicated_agvs_benchmark_settings_for_baselines(tmp_path):
    benchmark = "ackley3_300"
    algorithm = "RandomSearch"
    config = runner.BENCHMARKS[benchmark]
    identity = runner.requested_trace_cache_identity(
        config, benchmark, algorithm, trial=0, max_iter=10
    )
    stored_identity = json.loads(json.dumps(identity))
    stored_identity["benchmark_config"]["selection_batch_size"] = 2

    path = runner.trace_path_for(tmp_path, benchmark, algorithm, 0)
    runner.write_json(
        path,
        {
            "complete": True,
            "error": None,
            "n_evaluations": 10,
            "stopped_early": False,
            "stop_iteration": None,
            "cache_identity": stored_identity,
        },
    )

    assert runner.load_existing_trace_if_complete(
        tmp_path,
        benchmark,
        algorithm,
        0,
        max_iter=10,
        cache_identity=identity,
    ) is not None

    changed_algorithm_identity = json.loads(json.dumps(identity))
    changed_algorithm_identity["algorithm_settings"]["n_init"] += 1
    assert runner.load_existing_trace_if_complete(
        tmp_path,
        benchmark,
        algorithm,
        0,
        max_iter=10,
        cache_identity=changed_algorithm_identity,
    ) is None


def test_imported_budget_complete_json_can_be_reused_with_legacy_metadata(tmp_path):
    benchmark = "ackley3_300"
    algorithm = "RandomSearch"
    identity = runner.requested_trace_cache_identity(
        runner.BENCHMARKS[benchmark], benchmark, algorithm, trial=0, max_iter=10
    )
    stored_identity = json.loads(json.dumps(identity))
    stored_identity.pop("implementation_version")
    stored_identity.pop("seed_policy")
    stored_identity["algorithm_settings"]["legacy_report_label"] = "old runner"

    trace = {
        "benchmark": benchmark,
        "algorithm": algorithm,
        "trial_id": 0,
        "complete": True,
        "error": None,
        "n_evaluations": 10,
        "algorithm_seed": identity["algorithm_seed"],
        "benchmark_seed": identity["benchmark_seed"],
        "initial_design_seed": identity["initial_design_seed"],
        "cache_identity": stored_identity,
    }
    runner.write_json(
        runner.trace_path_for(tmp_path, benchmark, algorithm, 0), trace
    )

    assert runner.load_existing_trace_if_complete(
        tmp_path,
        benchmark,
        algorithm,
        0,
        max_iter=10,
        cache_identity=identity,
    ) is None
    assert runner.load_existing_trace_if_complete(
        tmp_path,
        benchmark,
        algorithm,
        0,
        max_iter=10,
        cache_identity=identity,
        reuse_complete_json=True,
    ) is not None
    assert runner.load_traces(tmp_path) == []
    assert len(runner.load_traces(tmp_path, reuse_complete_json=True)) == 1

    trace["cache_identity"]["algorithm_seed"] += 1
    runner.write_json(
        runner.trace_path_for(tmp_path, benchmark, algorithm, 0), trace
    )
    assert runner.load_existing_trace_if_complete(
        tmp_path,
        benchmark,
        algorithm,
        0,
        max_iter=10,
        cache_identity=identity,
        reuse_complete_json=True,
    ) is None


def test_imported_agvs_json_with_different_n_bad_is_not_reused(tmp_path):
    benchmark = "ackley3_300"
    algorithm = "AG-VS-BO"
    requested_config = dict(runner.BENCHMARKS[benchmark])
    requested_config.update(
        {
            "n_bad_enabled": True,
            "n_bad_multiplier": 28,
            "n_bad_is_absolute": True,
        }
    )
    requested_identity = runner.requested_trace_cache_identity(
        requested_config,
        benchmark,
        algorithm,
        trial=0,
        max_iter=10,
    )
    stored_identity = json.loads(json.dumps(requested_identity))
    stored_identity.pop("implementation_version")
    stored_identity["benchmark_config"]["n_bad_multiplier"] = 3

    trace = {
        "benchmark": benchmark,
        "algorithm": algorithm,
        "trial_id": 0,
        "complete": True,
        "error": None,
        "n_evaluations": 10,
        "algorithm_seed": requested_identity["algorithm_seed"],
        "benchmark_seed": requested_identity["benchmark_seed"],
        "initial_design_seed": requested_identity["initial_design_seed"],
        "cache_identity": stored_identity,
    }
    runner.write_json(
        runner.trace_path_for(tmp_path, benchmark, algorithm, 0), trace
    )

    assert runner.load_existing_trace_if_complete(
        tmp_path,
        benchmark,
        algorithm,
        0,
        max_iter=10,
        cache_identity=requested_identity,
        reuse_complete_json=True,
    ) is None


def test_round_robin_prefilters_all_fifty_completed_trials(
    monkeypatch, tmp_path
):
    scanned_trials = []
    submitted_runs = []

    def completed_loader(
        _output_dir,
        benchmark_name,
        algorithm,
        trial,
        **_kwargs,
    ):
        scanned_trials.append(int(trial))
        return {
            "benchmark": benchmark_name,
            "algorithm": algorithm,
            "trial_id": int(trial),
            "complete": True,
            "error": None,
        }

    monkeypatch.setattr(runner, "load_existing_trace_if_complete", completed_loader)
    monkeypatch.setattr(
        runner,
        "run_single_trial",
        lambda **kwargs: submitted_runs.append(kwargs),
    )

    status = runner.run_round_robin(
        benchmarks=["ackley3_300"],
        algorithms=["RandomSearch"],
        trials=50,
        max_iter=2,
        output_dir=str(tmp_path),
        figure_dir=str(tmp_path / "figures"),
        seed=0,
        update_plots=False,
        skip_complete=True,
        workers=4,
        device="cpu",
        table_dir=str(tmp_path / "tables"),
    )

    assert scanned_trials == list(range(50))
    assert submitted_runs == []
    assert status["complete"] is True
    assert status["returned_run_count"] == 50


def test_synthetic_plot_loader_uses_only_the_active_25_trial_scope(tmp_path):
    benchmark = "ackley3_300"
    algorithm = "RandomSearch"
    config = runner.BENCHMARKS[benchmark]
    for trial in (24, 25):
        identity = runner.requested_trace_cache_identity(
            config,
            benchmark,
            algorithm,
            trial=trial,
            max_iter=2,
        )
        runner.write_json(
            runner.trace_path_for(tmp_path, benchmark, algorithm, trial),
            {
                "benchmark": benchmark,
                "algorithm": algorithm,
                "trial_id": trial,
                "trial": trial,
                "algorithm_seed": 2021 + trial,
                "benchmark_seed": 2021 + trial,
                "complete": True,
                "error": None,
                "n_evaluations": 2,
                "cache_identity": identity,
            },
        )

    traces = runner.load_traces(tmp_path)
    assert [trace["trial_id"] for trace in traces] == [24]
    traces = runner.load_traces(tmp_path, trial_limits={benchmark: 50})
    assert [trace["trial_id"] for trace in traces] == [24, 25]


def test_real_life_cache_identity_tracks_executable_and_protocol(monkeypatch, tmp_path):
    config = make_real_life_config("mopta08", 4)
    def identity(c):
        return runner.requested_trace_cache_identity(c, "mopta08_deff4", "AG-VS-BO", 0, 150)
    monkeypatch.setenv("MOPTA08_EXECUTABLE", str(tmp_path / "first.exe"))
    first = identity(config)
    monkeypatch.setenv("MOPTA08_EXECUTABLE", str(tmp_path / "second.exe"))
    assert first != identity(config)
    changed = dict(config, objective_protocol={**config["objective_protocol"], "maximize": "changed"})
    assert identity(config) != identity(changed)



def test_round_robin_rejects_zero_work(tmp_path):
    common = {
        "benchmarks": ["ackley3_300"],
        "algorithms": ["RandomSearch"],
        "trials": 1,
        "max_iter": 2,
        "output_dir": str(tmp_path),
        "figure_dir": str(tmp_path / "figures"),
        "seed": 0,
        "update_plots": False,
        "workers": 1,
    }
    with pytest.raises(ValueError, match="trials"):
        runner.run_round_robin(**{**common, "trials": 0})
    with pytest.raises(ValueError, match="algorithm"):
        runner.run_round_robin(**{**common, "algorithms": []})
    with pytest.raises(ValueError, match="benchmark"):
        runner.run_round_robin(**{**common, "benchmarks": []})


def test_real_life_registration_rejects_wrong_condition_identity():
    wrong = make_real_life_config("mopta08", 3)
    with pytest.raises(ValueError, match="configured d_eff"):
        runner.register_benchmark("mopta08_deff4", wrong)



def test_spawned_worker_reconstructs_real_life_registration(monkeypatch, tmp_path):
    benchmark = "mopta08_deff3"
    monkeypatch.delitem(runner.BENCHMARKS, benchmark, raising=False)

    def stop_after_lookup(_algorithm, config, benchmark_name):
        assert benchmark_name == benchmark
        assert config["real_life_task"] == "mopta08"
        raise RuntimeError("registration-reconstructed")

    monkeypatch.setattr(runner, "validate_algorithm_applicability", stop_after_lookup)
    with pytest.raises(RuntimeError, match="registration-reconstructed"):
        runner.run_single_trial(
            benchmark,
            "AG-VS-BO",
            trial=0,
            max_iter=150,
            output_dir=tmp_path,
            base_seed=0,
        )


def test_real_life_primary_table_value_is_best_observed():
    trace = {
        "trial": 0,
        "best_observed_history": [0.2, 0.7, 0.7],
        "reported_posterior_mean_history": [],
    }
    assert runner.final_values_by_trial(
        [trace], value_type="best_observed"
    ) == {0: 0.7}


def test_round_robin_status_reports_failed_trial(monkeypatch, tmp_path):
    monkeypatch.setattr(
        runner,
        "run_single_trial",
        lambda **kwargs: {
            "benchmark": kwargs["benchmark_name"],
            "algorithm": kwargs["algorithm"],
            "trial_id": kwargs["trial"],
            "complete": False,
            "error": "simulated optimizer failure",
        },
    )

    runner.run_round_robin(
        benchmarks=["ackley3_300"],
        algorithms=["RandomSearch"],
        trials=1,
        max_iter=2,
        output_dir=str(tmp_path),
        figure_dir=str(tmp_path / "figures"),
        seed=0,
        update_plots=False,
        skip_complete=False,
        workers=1,
        device="cpu",
        table_dir=str(tmp_path / "tables"),
    )

    status = json.loads(
        (tmp_path / "ackley3_300" / "run_status.json").read_text(encoding="utf-8")
    )
    assert status["complete"] is False
    assert status["failed_run_count"] == 1
    assert status["failed_runs"][0]["error"] == "simulated optimizer failure"


def test_benchmark_entrypoint_raises_when_round_robin_is_incomplete(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(runner, "SYNTHETIC_RESULT_ROOT", tmp_path / "results")
    monkeypatch.setattr(runner, "SYNTHETIC_FIGURE_ROOT", tmp_path / "figures")
    monkeypatch.setattr(runner, "SYNTHETIC_TABLE_ROOT", tmp_path / "tables")
    monkeypatch.setattr(
        runner,
        "run_round_robin",
        lambda **_kwargs: {
            "complete": False,
            "status_path": str(tmp_path / "run_status.json"),
        },
    )
    with pytest.raises(RuntimeError, match="failed or incomplete"):
        runner.run_benchmark(
            "ackley3_300",
            runner.BENCHMARKS["ackley3_300"],
            tmp_path,
            argv=[
                "--trials",
                "1",
                "--algorithms",
                "RandomSearch",
                "--skip-plot",
            ],
        )


def test_batched_child_defers_reports_and_forwards_trial_range(monkeypatch, tmp_path):
    for attr, folder in (
        ("SYNTHETIC_RESULT_ROOT", "results"),
        ("SYNTHETIC_FIGURE_ROOT", "figures"),
        ("SYNTHETIC_TABLE_ROOT", "tables"),
    ):
        monkeypatch.setattr(runner, attr, tmp_path / folder)
    calls = []
    monkeypatch.setattr(
        runner, "run_round_robin",
        lambda **kwargs: calls.append(kwargs) or {"complete": True},
    )

    def unexpected_report(*args, **kwargs):
        pytest.fail("A batched child must defer reports until every function is done")

    for name in ("plot_results", "save_synthetic_comparison_pdfs", "save_all_benchmark_objective_tables"):
        monkeypatch.setattr(runner, name, unexpected_report)
    runner.run_benchmark(
        "ackley3_300", runner.BENCHMARKS["ackley3_300"], tmp_path,
        argv=["--trials", "50", "--trial-start", "25", "--defer-reports",
              "--algorithms", "RandomSearch"],
    )
    assert calls[0]["trial_start"] == 25
    assert calls[0]["trials"] == 50
    assert calls[0]["update_plots"] is False


def test_benchmark_entrypoint_updates_tables_when_plots_are_skipped(
    monkeypatch, tmp_path
):
    results = tmp_path / "results"
    figures = tmp_path / "figures"
    tables = tmp_path / "tables"
    monkeypatch.setattr(runner, "SYNTHETIC_RESULT_ROOT", results)
    monkeypatch.setattr(runner, "SYNTHETIC_FIGURE_ROOT", figures)
    monkeypatch.setattr(runner, "SYNTHETIC_TABLE_ROOT", tables)
    monkeypatch.setattr(
        runner,
        "run_round_robin",
        lambda **_kwargs: {"complete": True, "status_path": str(tmp_path / "run_status.json")},
    )
    calls = []
    monkeypatch.setattr(
        runner,
        "save_all_benchmark_objective_tables",
        lambda benchmark_root, table_dir: calls.append((benchmark_root, table_dir)) or [],
    )

    runner.run_benchmark(
        "ackley3_300",
        runner.BENCHMARKS["ackley3_300"],
        tmp_path,
        argv=[
            "--trials",
            "1",
            "--algorithms",
            "RandomSearch",
            "--skip-plot",
        ],
    )

    assert calls == [(results, tables)]


def test_round_robin_worker_returns_compact_status_not_full_trace(monkeypatch, tmp_path):
    monkeypatch.setattr(
        runner,
        "run_single_trial",
        lambda **kwargs: {
            "benchmark": kwargs["benchmark_name"],
            "algorithm": kwargs["algorithm"],
            "trial_id": kwargs["trial"],
            "complete": True,
            "error": None,
            "x_history": [[0.0]] * 10_000,
        },
    )
    summaries = runner._run_round_robin_trial(
        (
            0,
            1,
            ["ackley3_300"],
            ["RandomSearch"],
            2,
            str(tmp_path),
            0,
            False,
            str(tmp_path / "figures"),
            "cpu",
        )
    )

    assert summaries == [
        {
            "benchmark": "ackley3_300",
            "algorithm": "RandomSearch",
            "trial_id": 0,
            "complete": True,
            "error": None,
        }
    ]


def test_round_robin_worker_converts_construction_exception_to_failure_status(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(
        runner,
        "run_single_trial",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("construction failed")),
    )
    summaries = runner._run_round_robin_trial(
        (
            0,
            1,
            ["ackley3_300"],
            ["RandomSearch"],
            2,
            str(tmp_path),
            0,
            False,
            str(tmp_path / "figures"),
            "cpu",
        )
    )
    assert summaries[0]["complete"] is False
    assert summaries[0]["error"] == "RuntimeError: construction failed"


def test_cli_exposes_only_device_flag_after_workers(capsys):
    with pytest.raises(SystemExit) as caught:
        runner.main(["--help"])
    assert caught.value.code == 0
    help_text = capsys.readouterr().out
    assert "--device {cpu}" in help_text
    assert "--turbo-device" not in help_text
    assert help_text.index("--workers") < help_text.index("--device")


def test_cli_rejects_disallowed_device_and_nonpositive_workers():
    with pytest.raises(SystemExit) as cuda_error:
        runner.main(["--device", "cuda", "--skip-plot"])
    assert cuda_error.value.code == 2

    with pytest.raises(SystemExit) as worker_error:
        runner.main(["--workers", "0", "--skip-plot"])
    assert worker_error.value.code == 2


def test_cma_table_semantics_distinguish_center_from_objective_samples():
    settings = runner.algorithm_settings_summary("CMA-ES", "hartmann3_300")
    assert settings["n_init"] == 0
    assert settings["batch_size"] == settings["population_size"] == 20
    assert "center" in settings["initial_sampling"]


def test_close_objective_hook_is_safe_and_observable():
    class Objective:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    objective = Objective()
    assert runner.close_objective(objective) is None
    assert objective.closed
    assert runner.close_objective(lambda _x: 0.0) is None


def test_real_life_reference_dimensions_feed_runtime_and_report_settings():
    config = make_real_life_config("mopta08", 4)
    for algorithm in ("HeSBO", "ALEBO"):
        assert runner.optimizer_active_dimension(algorithm, config, "mopta08_deff4") == 10
    assert runner.optimizer_active_dimension("AG-VS-BO", config, "mopta08_deff4") == 4
    settings = runner.algorithm_settings_summary("HeSBO", "mopta08_deff4")
    assert settings["configured_d_eff"] == 4
    assert settings["physical_dim"] == 124
    assert settings["reference_embedding_dimension"] == 10
    assert settings["optimizer_active_dimension"] == 10
    assert settings["learn_observation_noise"] is False
    assert settings["fixed_noise_variance"] == 0.0



@pytest.mark.parametrize("algorithm", ["HeSBO", "ALEBO"])
def test_real_life_ax_embedding_builder_uses_official_known_zero_variance(
    monkeypatch, algorithm
):
    from Experiment.benchmark.baseline import native_baselines

    benchmark_name = "mopta08_deff4"
    config = make_real_life_config("mopta08", 4)
    runner.register_benchmark(benchmark_name, config)

    optimizer = runner.make_mcts_paper_baseline(
        algorithm,
        config,
        recorder=lambda _x: 0.0,
        seed=7,
        max_iter=12,
        benchmark_name=benchmark_name,
    )

    assert optimizer.learn_noise is False


def test_real_life_active_roster_excludes_vae_bo():
    config = {
        "experiment_kind": "real_life",
        "real_life_variant": "mopta08_deff3",
        "effective_dim": 3,
    }
    assert runner.validate_algorithm_applicability(
        "ALEBO",
        config,
        "mopta08_deff3",
    ) == "ALEBO"
    with pytest.raises(ValueError, match="VAE-BO is not applicable"):
        runner.validate_algorithm_applicability(
            "VAE-BO",
            config,
            "mopta08_deff3",
        )
    assert runner.validate_algorithm_applicability(
        "HeSBO",
        config,
        "mopta08_deff3",
    ) == "HeSBO"


def test_reference_runtime_preflight_fails_once_before_any_evaluation():
    runner.preflight_reference_runtime(
        ["AG-VS-BO", "MCTS-VS-BO", "LA-MCTS-TuRBO"]
    )
    assert runner.reference_runtime_metadata("LA-MCTS-TuRBO")[
        "runtime_status"
    ] == "READY"
    for algorithm in (
        "ADD-GP-UCB",
        "HeSBO",
        "ALEBO",
        "LA-MCTS-TuRBO",
        "CMA-ES",
        "SAASBO",
        "MCTS-VS-SAASBO",
    ):
        launch_command = runner.reference_runtime_metadata(algorithm)[
            "launch_command"
        ]
        assert "Experiment.benchmark.runner" in launch_command
        assert f"--algorithms {algorithm}" in launch_command
        assert "registered objective arguments" not in launch_command
        assert "<python" not in launch_command
    runner.preflight_reference_runtime(["AG-VS-BO", "HeSBO", "ALEBO", "GTBO", "SAASBO"])


@pytest.mark.parametrize("algorithm", ["HeSBO", "ALEBO", "LA-MCTS-TuRBO"])
def test_reference_managed_loops_do_not_claim_project_posterior_stopping(algorithm):
    settings = runner.algorithm_settings_summary(algorithm, "hartmann6_1000")
    assert settings["posterior_stopping_enabled"] is False
    assert settings["posterior_stopping_threshold"] is None
    assert "reference-managed" in settings["posterior_stopping_check"]


@pytest.mark.parametrize("source_d_eff,target_d_eff", [(3, 4), (4, 3)])
@pytest.mark.parametrize("algorithm", ["TuRBO", "HeSBO"])
def test_real_life_dimension_invariant_trace_is_materialized_from_sibling(
    tmp_path,
    source_d_eff,
    target_d_eff,
    algorithm,
):
    source_name = f"mopta08_deff{source_d_eff}"
    target_name = f"mopta08_deff{target_d_eff}"
    source_config = make_real_life_config("mopta08", source_d_eff)
    target_config = make_real_life_config("mopta08", target_d_eff)
    runner.register_benchmark(source_name, source_config)
    runner.register_benchmark(target_name, target_config)

    source_identity = runner.requested_trace_cache_identity(
        source_config,
        source_name,
        algorithm,
        trial=0,
        max_iter=12,
    )
    runner.write_json(
        runner.trace_path_for(tmp_path, source_name, algorithm, 0),
        {
            "benchmark": source_name,
            "algorithm": algorithm,
            "trial_id": 0,
            "complete": True,
            "error": None,
            "n_evaluations": 12,
            "stopped_early": False,
            "cache_identity": source_identity,
            "settings": {
                "effective_dim": source_d_eff,
                "d_eff": source_d_eff,
                "real_life_variant": source_name,
                "algorithm_settings": source_identity["algorithm_settings"],
            },
            "observed_y_history": [1.0, 2.0],
        },
    )

    created = runner.materialize_dimension_invariant_real_life_traces(
        output_dir=tmp_path,
        benchmarks=[target_name],
        algorithms=[algorithm, "VAE-BO", "AG-VS-BO"],
        trials=1,
        max_iter=12,
        base_seed_offset=0,
        device="cpu",
    )
    assert created == [(target_name, algorithm, 0, source_name)]

    target_identity = runner.requested_trace_cache_identity(
        target_config,
        target_name,
        algorithm,
        trial=0,
        max_iter=12,
    )
    reused = runner.load_existing_trace_if_complete(
        tmp_path,
        target_name,
        algorithm,
        0,
        max_iter=12,
        cache_identity=target_identity,
    )
    assert reused is not None
    assert reused["benchmark"] == target_name
    assert reused["settings"]["d_eff"] == target_d_eff
    assert reused["observed_y_history"] == [1.0, 2.0]
    assert reused["dimension_reuse_provenance"]["source_benchmark"] == source_name


def test_real_life_d_eff_dependency_classification_respects_reference_embeddings():
    experiment = runner.get_real_life_experiment("mopta08", 3)
    for algorithm in ("AG-VS-BO", "AG-VS-TuRBO", "VAE-BO"):
        assert runner.real_life_algorithm_uses_configured_d_eff(
            algorithm,
            experiment,
        )
    for algorithm in (
        "MCTS-VS-BO",
        "MCTS-VS-TuRBO",
        "TuRBO",
        "LA-MCTS-BO",
        "LA-MCTS-TuRBO",
        "HeSBO",
        "ALEBO",
    ):
        assert not runner.real_life_algorithm_uses_configured_d_eff(
            algorithm,
            experiment,
        )
