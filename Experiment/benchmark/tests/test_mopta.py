"""MOPTA orchestration checks without starting an experiment."""
from types import SimpleNamespace
import os
import pytest
from Experiment.benchmark import mopta, runner
from Experiment.utils import REAL_LIFE_EXPERIMENTS


def test_mopta_scope_and_order():
    assert mopta.BENCHMARKS == ("mopta08_deff4",)
    assert mopta.ALGORITHMS == (
        "MCTS-VS-TuRBO", "MCTS-VS-BO", "AG-VS-TuRBO", "TuRBO", "AG-VS-BO",
        "GTBO", "HeSBO", "ALEBO", "LA-MCTS-TuRBO", "LA-MCTS-BO")
    assert mopta.TRIALS == 25
    assert mopta.WORKERS == 1


def test_mopta_settings_reach_worker_environment(monkeypatch):
    for name in ("Q_BATCH", "BATCH_SIZE", "T_SEQ", "N_BAD_MULTIPLIER", "N_BAD_ABSOLUTE", "N_BAD_ENABLED"):
        monkeypatch.delenv("AGVS_REAL_LIFE_" + name, raising=False)
    mopta.configure_runner()
    assert runner._validated_agvs_batch_policy("real_life") == (None, 3, 2, 1, True)
    assert os.environ["AGVS_REAL_LIFE_N_BAD_ABSOLUTE"] == "none"
    assert os.environ["AGVS_REAL_LIFE_N_BAD_MULTIPLIER"] == "1"
    for algorithm in ("AG-VS-BO", "AG-VS-TuRBO"):
        settings = runner.algorithm_settings_summary(algorithm, "mopta08_deff4")
        assert settings["effective_dim"] == 4
        assert settings["batch_size"] == 3
        assert settings["t_seq"] == 2
        optimizer = runner.build_optimizer(
            algorithm, runner.BENCHMARKS["mopta08_deff4"], lambda x: 0.0,
            2021, 150, benchmark_name="mopta08_deff4", device="cpu")
        assert optimizer.N_bad == 5


@pytest.mark.parametrize("algorithm", mopta.ALGORITHMS)
def test_all_mopta_algorithms_use_150_evaluations(algorithm):
    assert runner.algorithm_evaluation_budget(algorithm, 150, "mopta08_deff4") == 150
    assert not mopta.trace_reaches_registered_budget({"algorithm": algorithm, "n_evaluations": 149}, "mopta08_deff4")
    assert mopta.trace_reaches_registered_budget({"algorithm": algorithm, "n_evaluations": 150}, "mopta08_deff4")


def test_reporting_crop_preserves_source():
    trace = {"algorithm": "LA-MCTS-BO", "n_evaluations": 200,
             "best_observed_history": list(range(200)),
             "evaluation_wall_clock_history": list(range(1, 201))}
    report = mopta.trace_for_reporting_at_registered_budget(trace, "mopta08_deff4")
    assert report["n_evaluations"] == 150
    assert report["best_observed"] == 149
    assert report["wall_clock_seconds"] == 150
    assert len(trace["best_observed_history"]) == 200


def test_runner_arguments_keep_budget_seed_and_reuse():
    args = mopta.parse_args(["--workers", "7"])
    command = mopta.runner_arguments("mopta08_deff4", args)
    assert command[command.index("--workers") + 1] == "7"
    assert command[command.index("--trials") + 1] == "25"
    assert command[command.index("--seed") + 1] == "0"
    assert "--max-iter" not in command
    assert "--skip-plot" in command
    assert "--reuse-complete-json" in command
    assert "--resume-under-budget" not in command


def test_main_refreshes_before_algorithms_and_after_all_finish(monkeypatch):
    events = []
    monkeypatch.setattr(mopta, "configure_runner", lambda: "mopta08_deff4")
    monkeypatch.setattr(mopta, "refresh_benchmark_figures", lambda b: events.append(("before", b)))
    monkeypatch.setattr(mopta, "refresh_figures", lambda b: events.append(("after", b)))
    monkeypatch.setattr(runner, "main", lambda args: events.append(("run", args[args.index("--algorithms") + 1])))
    assert mopta.main([]) == 0
    assert events == [("before", "mopta08_deff4"), *[("run", a) for a in mopta.ALGORITHMS], ("after", ("mopta08_deff4",))]


def test_plots_only_and_list_never_start_trials(monkeypatch):
    monkeypatch.setattr(mopta, "configure_runner", lambda: "mopta08_deff4")
    monkeypatch.setattr(runner, "main", lambda _: pytest.fail("Unexpected optimization"))
    reports = []
    monkeypatch.setattr(mopta, "refresh_figures", lambda b: reports.append(b))
    assert mopta.main(["--list"]) == 0
    assert reports == []
    assert mopta.main(["--plots-only"]) == 0
    assert reports == [("mopta08_deff4",)]
