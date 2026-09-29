from __future__ import annotations

import copy
import numpy as np

from ag_vs.optimizer import AGVSOptimizer
from Experiment.benchmark import runner
from Experiment import version_comparison
from Experiment.utils import make_synthetic_config


def _preserve_runner_state(monkeypatch):
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
        "AGVS_MERGE_ENABLED_OVERRIDE",
        "AGVS_GP_NOISE_VARIANCE_OVERRIDE",
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
        "AGVS_MERGE_ENABLED",
        "AGVS_GP_NOISE_VARIANCE",
        "AGVS_SHOW_FINAL_TREE",
    ):
        monkeypatch.delenv(name, raising=False)
    for benchmark in version_comparison.BENCHMARKS:
        monkeypatch.setitem(runner.BENCHMARKS, benchmark, runner.BENCHMARKS[benchmark])


def test_default_version_comparison_plan():
    args = version_comparison.parse_args([])
    assert args.trials == 25
    assert args.workers == 25
    assert version_comparison.BENCHMARKS == (
        "rastrigin6_1000",
    )
    assert version_comparison.ALGORITHMS == ("AG-VS-BO", "AG-VS-TuRBO")
    assert version_comparison.VERSIONS == ("merge_enabled", "merge_disabled")
    assert 4 * 2 * 2 * args.trials == 400


def test_merge_setting_reaches_both_agvs_backends(monkeypatch):
    _preserve_runner_state(monkeypatch)
    for version, expected in version_comparison.VERSION_ENABLED.items():
        config = version_comparison.make_cell_config("rastrigin3_300", version)
        assert config["agvs_merge_enabled"] is expected
        assert config["agvs_gp_noise_variance"] == 1e-4
        assert runner.benchmark_agvs_merge_enabled(config) is expected
        for backend in ("vanillabo", "turbo"):
            optimizer = runner.make_agvs(
                config,
                recorder=None,
                node_backend=backend,
                seed=2021,
                max_iter=25,
                device="cpu",
            )
            assert optimizer.merge_enabled is expected
            assert optimizer.root.sad_no_merge_branch is (not expected)
            assert optimizer.root.merge_disabled == int(not expected)


def test_merge_disabled_keeps_siblings_then_splits_highest_scoring_child():
    optimizer = AGVSOptimizer(
        objective=lambda x: float(-np.sum(np.asarray(x, dtype=float) ** 2)),
        bounds=np.repeat([[-5.0, 5.0]], 6, axis=0),
        eff_dim=2,
        max_iter=20,
        n_init=4,
        noise_variance=1e-4,
        q_batch=None,
        batch_size=1,
        t_seq=1,
        seed=7,
        n_split=3,
        N_bad=12,
        merge_enabled=False,
        risk_tolerances=[1 / 256],
        verbose=False,
        printer_on=False,
    )
    root = optimizer.root
    assert optimizer._split_node(root)
    children = list(root.children)
    assert len(children) == 3
    cohort_id = children[0].sad_cohort_id
    for index, child in enumerate(children):
        child.selected_count = 1
        child.sad_trial_done = True
        child.ave_diff_score = float(index + 1)
    optimizer._maybe_complete_deferred_cohort(root, cohort_id)

    assert root.children == children
    assert optimizer.sad_merge_history == []
    assert all(not hasattr(child, "sad_cohort_id") for child in children)
    selected = optimizer._select_node()
    assert selected is children[-1]
    assert optimizer._split_node(selected)
    assert selected.children
    assert all(child.sad_no_merge_branch for child in selected.children)


def test_cache_identity_separates_merge_versions_without_changing_ordinary_runs(
    monkeypatch,
):
    _preserve_runner_state(monkeypatch)
    enabled = version_comparison.expected_identity(
        "rastrigin3_300", "merge_enabled", "AG-VS-BO", 0
    )
    disabled = version_comparison.expected_identity(
        "rastrigin3_300", "merge_disabled", "AG-VS-BO", 0
    )
    assert enabled["benchmark_config"]["agvs_merge_enabled"] is True
    assert disabled["benchmark_config"]["agvs_merge_enabled"] is False
    assert not runner.cache_identities_match(enabled, disabled)

    monkeypatch.setattr(runner, "AGVS_MERGE_ENABLED_OVERRIDE", None)
    monkeypatch.setattr(runner, "AGVS_GP_NOISE_VARIANCE_OVERRIDE", None)
    monkeypatch.delenv("AGVS_MERGE_ENABLED", raising=False)
    monkeypatch.delenv("AGVS_GP_NOISE_VARIANCE", raising=False)
    ordinary = runner.apply_runner_agvs_batch_policy(
        make_synthetic_config("rastrigin3_300")
    )
    runner.register_benchmark("rastrigin3_300", ordinary)
    identity = runner.requested_trace_cache_identity(
        ordinary, "rastrigin3_300", "AG-VS-BO", 0, ordinary["max_iter"]
    )
    assert "agvs_merge_enabled" not in identity["benchmark_config"]


def test_legacy_canonical_bo_trace_is_copied_into_merge_enabled_cell(
    monkeypatch, tmp_path
):
    _preserve_runner_state(monkeypatch)
    source_root = tmp_path / "source"
    result_root = tmp_path / "comparison"
    monkeypatch.setattr(version_comparison, "SOURCE_SYNTHETIC_RESULT_ROOT", source_root)
    monkeypatch.setattr(version_comparison, "RESULT_ROOT", result_root)
    benchmark = "rastrigin3_300"
    target_identity = version_comparison.expected_identity(
        benchmark, "merge_enabled", "AG-VS-BO", 0
    )
    legacy_identity = copy.deepcopy(target_identity)
    legacy_identity["benchmark_config"].pop("agvs_merge_enabled")
    legacy_identity["benchmark_config"].pop("agvs_gp_noise_variance")
    legacy_identity["algorithm_settings"].pop("merge_enabled")
    legacy_identity["algorithm_settings"].pop("merge_policy")
    trace = {
        "benchmark": benchmark,
        "algorithm": "AG-VS-BO",
        "trial_id": 0,
        "algorithm_seed": 2021,
        "benchmark_seed": 2021,
        "complete": True,
        "error": None,
        "n_evaluations": 700,
        "cache_identity": legacy_identity,
        "settings": {
            "benchmark_noise_variance": 1e-4,
            "algorithm_settings": {
                "batch_size": 3,
                "t_seq": 2,
                "N_bad_multiplier": 4,
                "N_bad_enabled": True,
                "alpha": 1e-4,
            },
        },
    }
    source_path = runner.trace_path_for(
        source_root, benchmark, "AG-VS-BO", 0
    )
    runner.write_json(source_path, trace)

    assert version_comparison.seed_merge_enabled_bo_results(benchmark, 1) == (1, 0)
    destination = runner.trace_path_for(
        version_comparison.cell_roots("merge_enabled")[0],
        benchmark,
        "AG-VS-BO",
        0,
    )
    assert destination.is_file()
    assert source_path.read_bytes() == destination.read_bytes()


def test_combined_figure_has_algorithm_headers_top_legend_and_requested_labels(
    monkeypatch, tmp_path
):
    from matplotlib.figure import Figure

    def curves_for(_benchmark, versions, _trials):
        return [
            {
                "algorithm": algorithm,
                "version": version,
                "completed_trials": 25,
                "x": np.arange(1, 6),
                "mean": np.arange(5, dtype=float),
                "ci95": np.full(5, 0.1),
                "metric": "best noiseless objective value",
            }
            for algorithm in version_comparison.ALGORITHMS
            for version in versions
        ]

    monkeypatch.setattr(version_comparison, "all_curves", curves_for)
    monkeypatch.setattr(version_comparison, "FIGURE_ROOT", tmp_path)
    original_savefig = Figure.savefig
    saved = []

    def check_figure(figure, path, **kwargs):
        assert len(figure.axes) == (
            len(version_comparison.BENCHMARKS)
            * len(version_comparison.ALGORITHMS)
        )
        assert figure._suptitle is None
        assert [axis.get_title() for axis in figure.axes[:2]] == list(
            version_comparison.ALGORITHMS
        )
        assert all(axis.get_title() == "" for axis in figure.axes[2:])
        assert figure._supxlabel.get_text() == "Number of evaluations"
        assert figure._supylabel.get_text() == "Value"
        assert [text.get_text() for text in figure.legends[0].get_texts()] == [
            "Merge enabled",
            "Merge disabled",
        ]
        saved.append(path.suffix)
        return original_savefig(figure, path, **kwargs)

    monkeypatch.setattr(Figure, "savefig", check_figure)
    outputs = version_comparison.save_combined_figure(
        version_comparison.BENCHMARKS,
        version_comparison.VERSIONS,
        25,
    )
    assert saved == [".pdf", ".png"]
    assert all(path.is_file() for path in outputs)
