r"""Run the dedicated MOPTA08 comparison with the MCTS-VS variants first.

From the repository root::

    py -3.13 Experiment/benchmark/mopta.py

All algorithms use 150 evaluations per trial. Trial IDs 0--24 use optimizer
and benchmark seeds 2021--2045. Complete compatible results are reused.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from collections import defaultdict
import os
from Experiment.benchmark import runner
from Experiment.utils import REAL_LIFE_EXPERIMENTS


TRIALS = 25
WORKERS = 1
DEVICE = "cpu"
SEED = 0

# AG-VS controls for the active MOPTA08 experiment. Exactly one of Q_BATCH
# and BATCH_SIZE must be None. N_bad = N_split by default.
Q_BATCH = None
BATCH_SIZE = 3
T_SEQ = 2
N_BAD_MULTIPLIER = 1
N_BAD_ABSOLUTE = None
N_BAD_ENABLED = True

# The released real-life experiment scope contains MOPTA08 only.
EXCLUDED_TASKS = frozenset()
BENCHMARKS = ("mopta08_deff4",)

# Keep the dedicated MOPTA runner and shared engine in the same order.
ALGORITHMS = ("MCTS-VS-TuRBO", "MCTS-VS-BO", "AG-VS-TuRBO", "TuRBO", "AG-VS-BO", "GTBO", "HeSBO", "ALEBO", "LA-MCTS-TuRBO", "LA-MCTS-BO")

REPORT_ALGORITHMS = ALGORITHMS

CONTINUE_ON_BENCHMARK_ERROR = True
GENERATE_FIGURES_AFTER_EACH_BENCHMARK = True
GENERATE_FINAL_TABLES = True
# Imported historical real-life JSON files are authoritative once they are
# error-free, reach the registered budget, and match the requested benchmark,
# algorithm, trial, seed bundle, and device. Worker count remains irrelevant.
REUSE_COMPLETE_JSON = True
# Incomplete trials are rerun; compatible complete results are reused.
RESUME_UNDER_BUDGET = False


AGVS_BATCH_SIZE = 3
AGVS_T_SEQ = 2
AGVS_N_BAD_MULTIPLIER = 1


D_EFF = 4


def apply_real_life_settings() -> None:
    """Apply SETTINGS locally and export the same values to child processes."""

    values = (Q_BATCH, BATCH_SIZE)
    if (values[0] is None) == (values[1] is None):
        raise ValueError("Exactly one of Q_BATCH and BATCH_SIZE must be None")
    for name, value in (
        ("BATCH_SIZE", Q_BATCH if Q_BATCH is not None else BATCH_SIZE),
        ("T_SEQ", T_SEQ),
        ("N_BAD_MULTIPLIER", N_BAD_MULTIPLIER),
    ):
        if isinstance(value, bool) or int(value) <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if N_BAD_ABSOLUTE is not None and (
        isinstance(N_BAD_ABSOLUTE, bool) or int(N_BAD_ABSOLUTE) <= 0
    ):
        raise ValueError("N_BAD_ABSOLUTE must be None or a positive integer")

    import os

    environment = {
        "AGVS_REAL_LIFE_Q_BATCH": "none" if Q_BATCH is None else str(int(Q_BATCH)),
        "AGVS_REAL_LIFE_BATCH_SIZE": (
            "none" if BATCH_SIZE is None else str(int(BATCH_SIZE))
        ),
        "AGVS_REAL_LIFE_T_SEQ": str(int(T_SEQ)),
        "AGVS_REAL_LIFE_N_BAD_MULTIPLIER": str(int(N_BAD_MULTIPLIER)),
        "AGVS_REAL_LIFE_N_BAD_ABSOLUTE": (
            "none" if N_BAD_ABSOLUTE is None else str(int(N_BAD_ABSOLUTE))
        ),
        "AGVS_REAL_LIFE_N_BAD_ENABLED": "1" if N_BAD_ENABLED else "0",
    }
    os.environ.update(environment)

    # runner_synthetic.py is already imported in this parent process; update its live
    # constants as well as exporting the child-process environment above.
    runner.REAL_LIFE_Q_BATCH = None if Q_BATCH is None else int(Q_BATCH)
    runner.REAL_LIFE_BATCH_SIZE = None if BATCH_SIZE is None else int(BATCH_SIZE)
    runner.REAL_LIFE_T_SEQ = int(T_SEQ)
    runner.REAL_LIFE_N_BAD_MULTIPLIER = int(N_BAD_MULTIPLIER)
    runner.REAL_LIFE_N_BAD_ABSOLUTE = (
        None if N_BAD_ABSOLUTE is None else int(N_BAD_ABSOLUTE)
    )
    runner.REAL_LIFE_N_BAD_ENABLED = bool(N_BAD_ENABLED)
    runner._validated_agvs_batch_policy("real_life")

    # Plot-only mode does not enter runner.main(), which normally performs this
    # dynamic registration before scheduling. Register the active real-life
    # configs here so cached JSON identities can also be validated for plots.
    from Experiment.benchmark.real_life.tasks import make_real_life_config

    for benchmark in BENCHMARKS:
        experiment = REAL_LIFE_EXPERIMENTS[benchmark]
        runner.register_benchmark(
            benchmark,
            runner.apply_runner_agvs_batch_policy(
                make_real_life_config(
                    experiment.task.key,
                    experiment.configured_d_eff,
                )
            ),
        )


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def selected_benchmarks(raw_names=None) -> tuple[str, ...]:
    if raw_names is None:
        return BENCHMARKS
    names = tuple(str(name).strip().lower() for name in raw_names)
    unknown = tuple(name for name in names if name not in BENCHMARKS)
    if unknown:
        raise ValueError(
            f"Unknown real-life benchmark(s): {unknown}; available={BENCHMARKS}"
        )
    return names


def runner_arguments(
    benchmark: str,
    args,
    algorithms: tuple[str, ...] = ALGORITHMS,
) -> list[str]:
    command = [
        "--benchmarks",
        benchmark,
        "--algorithms",
        *algorithms,
        "--trials",
        str(int(args.trials)),
        "--workers",
        str(int(args.workers)),
        "--device",
        str(args.device),
        "--seed",
        str(int(args.seed)),
        # Suppress per-trial refreshes. This module refreshes once per benchmark.
        "--skip-plot",
    ]
    if args.rerun_complete:
        command.append("--rerun-complete")
    if REUSE_COMPLETE_JSON:
        command.append("--reuse-complete-json")
    if RESUME_UNDER_BUDGET:
        command.append("--resume-under-budget")
    return command


def run_benchmark_algorithms(benchmark: str, args) -> None:
    """All algorithms execute directly in the main Python environment."""
    runner.main(runner_arguments(benchmark, args, ALGORITHMS))


def refresh_figures(benchmarks: tuple[str, ...]) -> None:
    """Regenerate selected figures and aggregate tables (plots-only mode)."""

    for benchmark in benchmarks:
        refresh_benchmark_figures(benchmark)
    if benchmarks:
        refresh_aggregate_tables()
        figure_root, _table_root = runner.output_roots_for_experiment_kind(
            "real_life"
        )
        runner.remove_legacy_millisecond_figure_aliases(figure_root)


def trace_reaches_registered_budget(trace: dict, benchmark: str) -> bool:
    try:
        return int(trace.get("n_evaluations", 0)) >= runner.algorithm_evaluation_budget(
            trace.get("algorithm"),
            REAL_LIFE_EXPERIMENTS[benchmark].budget,
            benchmark,
        )
    except (KeyError, TypeError, ValueError):
        return False


def trace_for_reporting_at_registered_budget(trace: dict, benchmark: str) -> dict:
    """Return a non-destructive reporting view capped at the algorithm budget."""

    limit = runner.algorithm_evaluation_budget(
        trace.get("algorithm"),
        REAL_LIFE_EXPERIMENTS[benchmark].budget,
        benchmark,
    )
    try:
        recorded = int(trace.get("n_evaluations", 0) or 0)
    except (TypeError, ValueError):
        return trace
    if recorded <= int(limit):
        return trace

    # A shallow copy is sufficient because crop_trace_payload_to_max_iter
    # replaces every sequence/dictionary it truncates instead of mutating the
    # original JSON-derived containers.
    cropped = runner.crop_trace_payload_to_max_iter(dict(trace), int(limit))

    def last_finite(values):
        for value in reversed(values if isinstance(values, list) else []):
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            if runner.np.isfinite(numeric):
                return numeric
        return None

    observed = last_finite(cropped.get("best_observed_history"))
    if observed is not None:
        cropped["final_best_observed_at_max_iter"] = observed
        cropped["best_observed"] = observed

    noiseless = last_finite(cropped.get("best_noiseless_y_history"))
    if noiseless is not None:
        cropped["final_best_noiseless_y_at_max_iter"] = noiseless
        cropped["best_noiseless_y"] = noiseless

    posterior = last_finite(cropped.get("reported_posterior_mean_history"))
    if posterior is None:
        posterior = last_finite(cropped.get("best_posterior_mean_history"))
    if posterior is not None:
        cropped["final_best_posterior_mean_at_max_iter"] = posterior
        cropped["best_posterior_mean"] = posterior
        cropped["returned_posterior_mean"] = posterior

    threshold_histories = cropped.get("best_threshold_history_by_delta", {})
    if isinstance(threshold_histories, dict):
        threshold = last_finite(threshold_histories.get("1_256"))
        if threshold is not None:
            cropped["final_best_stoppable_threshold_at_max_iter_1_256"] = threshold
            if "final_best_stoppable_threshold_at_max_iter" in cropped:
                cropped["final_best_stoppable_threshold_at_max_iter"] = threshold

    elapsed = last_finite(cropped.get("evaluation_wall_clock_history"))
    if elapsed is not None:
        cropped["wall_clock_seconds"] = elapsed
    cropped["reporting_evaluation_limit"] = int(limit)
    return cropped


def refresh_benchmark_figures(benchmark: str) -> None:
    """Generate one completed benchmark's figures before starting the next."""

    result_root = runner.result_root_for_experiment_kind("real_life")
    figure_root, table_root = runner.output_roots_for_experiment_kind("real_life")
    active_algorithms = set(REPORT_ALGORITHMS)
    traces = [
        trace_for_reporting_at_registered_budget(trace, benchmark)
        for trace in runner.load_traces(
            result_root,
            reuse_complete_json=REUSE_COMPLETE_JSON,
            trial_limits={benchmark: TRIALS},
        )
        if trace.get("complete", False)
        and str(trace.get("benchmark", "")).strip().lower() == benchmark
        and trace.get("algorithm") in active_algorithms
        and trace_reaches_registered_budget(trace, benchmark)
    ]
    runner.plot_results(
        result_root,
        figure_root,
        benchmark_filter=benchmark,
        update_objective_tables=False,
        table_dir=table_root,
        _loaded_traces=traces,
    )


def refresh_aggregate_tables() -> None:
    """Refresh active-suite tables without excluded benchmark traces."""

    result_root = runner.result_root_for_experiment_kind("real_life")
    _figure_root, table_root = runner.output_roots_for_experiment_kind("real_life")
    grouped = defaultdict(lambda: defaultdict(list))
    active = set(BENCHMARKS)
    active_algorithms = set(REPORT_ALGORITHMS)
    for trace in runner.load_traces(
        result_root,
        reuse_complete_json=REUSE_COMPLETE_JSON,
        trial_limits={benchmark: TRIALS for benchmark in BENCHMARKS},
    ):
        benchmark = str(trace.get("benchmark", "")).strip().lower()
        algorithm = trace.get("algorithm")
        if (
            trace.get("complete", False)
            and benchmark in active
            and algorithm in active_algorithms
            and trace_reaches_registered_budget(trace, benchmark)
        ):
            grouped[benchmark][algorithm].append(
                trace_for_reporting_at_registered_budget(trace, benchmark)
            )
    for output_path in runner.save_objective_tables(
        grouped,
        result_root,
        write_per_benchmark=False,
        table_dir=table_root,
        preserve_cached_rows=False,
    ):
        print(output_path, flush=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run the 25-trial MOPTA08 comparison in ALGORITHMS order."
    )

    parser.set_defaults(device="cpu", seed=0, trials=TRIALS)
    parser.add_argument(
        "--workers",
        type=positive_int,
        default=WORKERS,
        help="Number of concurrent trials for each algorithm (default: 1).",
    )
    parser.add_argument("--list", action="store_true", help="Show settings only.")
    parser.add_argument(
        "--plots-only",
        action="store_true",
        help="Refresh MOPTA08 figures and tables from completed JSON files.",
    )
    parser.add_argument(
        "--skip-final-plots",
        action="store_true",
        help="Skip the figure and table refresh after the runs.",
    )
    parser.add_argument(
        "--rerun-complete",
        action="store_true",
        help="Rerun completed trials instead of reusing their JSON results.",
    )
    return parser.parse_args(argv)


def configure_runner() -> str:
    global BENCHMARKS, REPORT_ALGORITHMS, BATCH_SIZE, T_SEQ, N_BAD_MULTIPLIER
    benchmark = f"mopta08_deff{D_EFF}"
    if benchmark not in REAL_LIFE_EXPERIMENTS:
        raise ValueError(f"MOPTA08 condition is not registered: {benchmark}")
    BENCHMARKS = (benchmark,)
    REPORT_ALGORITHMS = ALGORITHMS
    BATCH_SIZE = AGVS_BATCH_SIZE
    T_SEQ = AGVS_T_SEQ
    N_BAD_MULTIPLIER = AGVS_N_BAD_MULTIPLIER
    apply_real_life_settings()
    return benchmark


def main(argv=None) -> int:
    args = parse_args(argv)
    benchmark = configure_runner()
    experiment = REAL_LIFE_EXPERIMENTS[benchmark]

    if args.list:
        n_split = experiment.configured_d_eff + 1
        print(
            f"{benchmark}: D={experiment.physical_dim}, "
            f"d_eff={experiment.configured_d_eff}, "
            f"evaluations={experiment.budget}, trials={TRIALS}, "
            f"seeds=2021--2045, workers={args.workers}"
        )
        print("algorithm order: " + " -> ".join(ALGORITHMS))
        print(
            f"AG-VS: B={AGVS_BATCH_SIZE}, t_seq={AGVS_T_SEQ}, "
            f"N_split={n_split}, "
            f"N_bad={AGVS_N_BAD_MULTIPLIER}*N_split="
            f"{AGVS_N_BAD_MULTIPLIER * n_split}"
        )
        return 0

    if args.plots_only:
        refresh_figures((benchmark,))
        return 0

    print(
        f"MOPTA08: {TRIALS} trials, seeds 2021--2045, "
        f"{experiment.budget} evaluations; "
        + " -> ".join(ALGORITHMS),
        flush=True,
    )
    if not args.skip_final_plots:
        print("\n=== Refreshing current MOPTA08 figures ===", flush=True)
        refresh_benchmark_figures(benchmark)
    for algorithm in ALGORITHMS:
        print(f"\n=== MOPTA08 algorithm: {algorithm} ===", flush=True)
        runner.main(
            runner_arguments(benchmark, args, (algorithm,))
        )
    if not args.skip_final_plots:
        print("\n=== Refreshing completed MOPTA08 figures ===", flush=True)
        refresh_figures((benchmark,))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
