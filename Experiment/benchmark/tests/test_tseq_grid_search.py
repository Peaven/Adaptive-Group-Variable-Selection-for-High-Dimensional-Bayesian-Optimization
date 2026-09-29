from __future__ import annotations

import pytest

from Experiment import grid_search_tseq, grid_search_nbad
from Experiment.benchmark import runner
from Experiment.utils import make_synthetic_config


def _preserve_runner_state(monkeypatch):
    for name in (
        "SYNTHETIC_Q_BATCH",
        "SYNTHETIC_BATCH_SIZE",
        "SYNTHETIC_T_SEQ",
        "SYNTHETIC_N_BAD_MULTIPLIER",
        "SYNTHETIC_N_BAD_ENABLED",
        "SYNTHETIC_N_BAD_ABSOLUTE",
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
        "AGVS_SYNTHETIC_N_BAD_ENABLED",
        "AGVS_SYNTHETIC_N_BAD_ABSOLUTE",
        "AGVS_SHOW_FINAL_TREE",
    ):
        monkeypatch.delenv(name, raising=False)


def test_default_tseq_plan_is_fixed():
    assert grid_search_tseq.parse_args([]).workers == 25
    assert grid_search_tseq.selected_benchmarks(None) == (
        "rastrigin3_300",
        "rastrigin6_1000",
    )
    assert grid_search_tseq.T_SEQ_VALUES == (1, 2, 4, 6, 8, 10)
    assert grid_search_tseq.figure_t_seq_values(
        grid_search_tseq.T_SEQ_VALUES
    ) == (2, 4, 6, 8, 10)
    assert grid_search_tseq.ALGORITHMS == ("AG-VS-BO", "AG-VS-TuRBO")
    assert grid_search_tseq.BATCH_SIZE == 3
    assert grid_search_tseq.TRIALS == 25
    assert (
        2
        * len(grid_search_tseq.T_SEQ_VALUES)
        * len(grid_search_tseq.ALGORITHMS)
        * grid_search_tseq.TRIALS
        == 600
    )


def test_tseq_combined_figure_uses_benchmark_rows_algorithm_columns_and_top_legend(
    monkeypatch, tmp_path
):
    import numpy as np
    from matplotlib.figure import Figure

    def curves_for(benchmark, values, _trials):
        assert tuple(values) == (2, 4)
        return [
            {
                "algorithm": algorithm,
                "t_seq_value": value,
                "completed_trials": 25,
                "x": np.arange(1, 5),
                "mean": np.arange(4, dtype=float) + value,
                "ci95": np.full(4, 0.1),
                "metric": "value",
            }
            for algorithm in grid_search_tseq.ALGORITHMS
            for value in values
        ]

    monkeypatch.setattr(grid_search_tseq, "all_curves", curves_for)
    monkeypatch.setattr(grid_search_tseq, "FIGURE_ROOT", tmp_path)
    original_savefig = Figure.savefig
    saved = []

    def check_figure(figure, path, **kwargs):
        assert len(figure.axes) == 4
        assert figure._suptitle is None
        assert [axis.get_title() for axis in figure.axes[:2]] == list(
            grid_search_tseq.ALGORITHMS
        )
        assert all(axis.get_title() == "" for axis in figure.axes[2:])
        assert len(figure.legends) == 1
        labels = [text.get_text() for text in figure.legends[0].get_texts()]
        assert labels == [r"$t_{\mathrm{seq}}=2$", r"$t_{\mathrm{seq}}=4$"]
        assert all("=1" not in label for label in labels)
        saved.append(path.suffix)
        return original_savefig(figure, path, **kwargs)

    monkeypatch.setattr(Figure, "savefig", check_figure)
    outputs = grid_search_tseq.save_combined_benchmark_plot(
        grid_search_tseq.BENCHMARKS,
        (1, 2, 4),
        25,
    )
    assert saved == [".pdf", ".png"]
    assert all(path.is_file() for path in outputs)


def test_nbad_combined_figure_matches_tseq_layout_and_uses_available_rows(
    monkeypatch, tmp_path
):
    import numpy as np
    from matplotlib.figure import Figure

    def curves_for(benchmark, values, _trials):
        if benchmark == "rastrigin6_1000":
            return []
        return [
            {
                "algorithm": algorithm,
                "n_bad": value,
                "completed_trials": 25,
                "x": np.arange(1, 5),
                "mean": np.arange(4, dtype=float) + value,
                "ci95": np.full(4, 0.1),
                "metric": "value",
            }
            for algorithm in grid_search_nbad.ALGORITHMS
            for value in values
        ]

    monkeypatch.setattr(grid_search_nbad, "all_curves", curves_for)
    monkeypatch.setattr(grid_search_nbad, "FIGURE_ROOT", tmp_path)
    original_savefig = Figure.savefig
    saved = []

    def check_figure(figure, path, **kwargs):
        assert len(figure.axes) == 2
        assert figure._suptitle is None
        assert [axis.get_title() for axis in figure.axes] == list(
            grid_search_nbad.ALGORITHMS
        )
        assert len(figure.legends) == 1
        labels = [text.get_text() for text in figure.legends[0].get_texts()]
        assert labels == [r"$N_{\mathrm{bad}}=6$", r"$N_{\mathrm{bad}}=10$"]
        saved.append(path.suffix)
        return original_savefig(figure, path, **kwargs)

    monkeypatch.setattr(Figure, "savefig", check_figure)
    outputs = grid_search_nbad.save_combined_benchmark_plot(
        grid_search_nbad.BENCHMARKS,
        (6, 10),
        25,
    )
    assert saved == [".pdf", ".png"]
    assert all(path.is_file() for path in outputs)


@pytest.mark.parametrize("grid, parameter", [
    (grid_search_tseq, "t_seq_value"), (grid_search_nbad, "n_bad"),
])
def test_grid_pdf_separates_algorithms_in_adjacent_panels(monkeypatch, tmp_path, grid, parameter):
    import numpy as np
    from matplotlib.figure import Figure
    curves = [
        {"algorithm": algorithm, parameter: value, "completed_trials": 25,
         "x": np.arange(1, 5), "mean": np.arange(4) + value,
         "ci95": np.full(4, 0.1)}
        for algorithm in grid.ALGORITHMS for value in (1, 2)
    ]
    monkeypatch.setattr(grid, "all_curves", lambda *args: curves)
    monkeypatch.setattr(grid, "FIGURE_ROOT", tmp_path)
    original_savefig = Figure.savefig
    saved = []

    def check_panels(figure, path, **kwargs):
        assert len(figure.axes) == 2
        left, right = figure.axes
        assert left.get_position().x1 < right.get_position().x0
        assert left.get_ylim() == right.get_ylim()
        for axis, algorithm in zip(figure.axes, grid.ALGORITHMS):
            assert axis.get_title() == algorithm
            labels = axis.get_legend_handles_labels()[1]
            assert len(labels) == 2
            assert all(label.startswith(algorithm + " (") for label in labels)
        saved.append(path.suffix)
        return original_savefig(figure, path, **kwargs)

    monkeypatch.setattr(Figure, "savefig", check_panels)
    outputs = grid.save_benchmark_plot("rastrigin3_300", (1, 2), 25)
    assert saved == [".pdf", ".png"]
    assert all(path.is_file() for path in outputs)


@pytest.mark.parametrize("node_backend", ["vanillabo", "turbo"])
def test_tseq_and_batch_size_reach_optimizer(monkeypatch, node_backend):
    _preserve_runner_state(monkeypatch)
    grid_search_tseq.configure_runner_cell(9)
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
    assert config["t_seq"] == 9
    assert optimizer.noise_variance == pytest.approx(1.0e-4)
    assert config["batch_size"] == 3
    assert "n_bad_is_absolute" not in config
    assert optimizer.t_seq == 9
    assert optimizer.batch_size == 3
    assert optimizer.N_bad == 4 * (3 + 1)


def test_tseq_loader_rejects_wrong_seed_or_noise(tmp_path, monkeypatch):
    monkeypatch.setattr(grid_search_tseq, "RESULT_ROOT", tmp_path / "results")
    benchmark = "rastrigin3_300"

    def write_trial(trial, *, algorithm_seed, noise_variance):
        trace = {
            "benchmark": benchmark,
            "algorithm": "AG-VS-BO",
            "trial_id": trial,
            "algorithm_seed": algorithm_seed,
            "benchmark_seed": 2021 + trial,
            "n_evaluations": 3,
            "best_observed_history": [1.0, 2.0, 3.0],
            "complete": True, "cache_identity": grid_search_tseq.cell_identity(grid_search_tseq.configure_runner_cell, 3, benchmark, "AG-VS-BO", trial, 3),
            "settings": {
                "benchmark_noise_variance": noise_variance,
                "algorithm_settings": {"batch_size": 3, "t_seq": 3},
            },
        }
        path = runner.trace_path_for(
            grid_search_tseq.cell_roots(3)[0], benchmark, "AG-VS-BO", trial
        )
        runner.write_json(path, trace)

    write_trial(0, algorithm_seed=2021, noise_variance=1e-4)
    write_trial(1, algorithm_seed=9999, noise_variance=1e-4)
    write_trial(2, algorithm_seed=2023, noise_variance=1e-3)

    traces, _ = grid_search_tseq.load_cell_traces(benchmark, 3, 50)
    assert [trace["trial_id"] for trace in traces] == [0]
