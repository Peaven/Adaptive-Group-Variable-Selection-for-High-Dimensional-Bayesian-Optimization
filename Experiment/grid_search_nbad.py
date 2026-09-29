#!/usr/bin/env python3
r"""Sweep absolute AG-VS-BO and AG-VS-TuRBO N_bad values on two synthetic Rastrigin benchmarks.

The fixed policy is batch_size=3, t_seq=2, Gaussian objective-noise variance
1e-4, and 25 trials with seeds 2021,...,2045. N_bad in {6,10,14,18,20,28} is passed as the absolute
node visit threshold (never multiplied by N_split). Available benchmark results
share one paper-facing PDF arranged by benchmark rows and algorithm columns,
with staggered vertical 95% confidence intervals.

Examples from the repository root::

    py -3.13 -m Experiment.grid_search_nbad --list
    py -3.13 -m Experiment.grid_search_nbad
    py -3.13 -m Experiment.grid_search_nbad --plots-only
    py -3.13 -m Experiment.grid_search_nbad --benchmarks rastrigin3_300
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import sys
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ag_vs.utils import cell_identity
from Experiment.benchmark import runner
from Experiment.utils import (
    SYNTHETIC_BENCHMARK_NOISE_VARIANCE,
    make_synthetic_config,
)


# =============================================================================
# SETTINGS -- command-line arguments may override trial/worker/sweep selections
# =============================================================================

TRIALS = 25
WORKERS = 25
DEVICE = "cpu"
SEED_OFFSET = 0

BATCH_SIZE = 3
T_SEQ = 2
Q_BATCH = None
N_BAD_VALUES = (6, 10, 14, 18, 20, 28)
ALGORITHMS = ("AG-VS-BO", "AG-VS-TuRBO")
BENCHMARKS = ("rastrigin3_300", "rastrigin6_1000")

RESULT_ROOT = ROOT / "Experiment" / "Result" / "grid_search_nbad" / "synthetic"
FIGURE_ROOT = (
    ROOT / "Experiment" / "Result" / "Figures" / "grid_search_nbad" / "synthetic"
)
TABLE_ROOT = ROOT / "Experiment" / "tables" / "grid_search_nbad" / "synthetic"


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def normalized_positive_values(values: Iterable[int]) -> tuple[int, ...]:
    normalized = tuple(sorted(dict.fromkeys(int(value) for value in values)))
    if not normalized or any(value <= 0 for value in normalized):
        raise ValueError("N_bad values must be non-empty positive integers")
    return normalized


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep absolute N_bad values for AG-VS-BO and AG-VS-TuRBO with batch_size=3 and "
            "t_seq=2 on the two configured synthetic Rastrigin benchmarks."
        )
    )
    parser.add_argument("--trials", type=positive_int, default=TRIALS)
    parser.add_argument("--workers", type=positive_int, default=WORKERS)
    parser.add_argument(
        "--seed",
        type=int,
        choices=(SEED_OFFSET,),
        default=SEED_OFFSET,
        help="Fixed at 0; trial ID i uses seed 2021 + i.",
    )
    parser.add_argument("--device", choices=(DEVICE,), default=DEVICE)
    parser.add_argument(
        "--n-bad-values",
        nargs="+",
        type=positive_int,
        default=list(N_BAD_VALUES),
    )
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=None,
        help="Optional subset of rastrigin3_300 and rastrigin6_1000.",
    )
    parser.add_argument(
        "--plots-only",
        action="store_true",
        help="Regenerate CSV/figures from existing traces without optimization.",
    )
    parser.add_argument(
        "--rerun-complete",
        action="store_true",
        help="Rerun cache-compatible, budget-complete traces.",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop at the first failed N_bad cell instead of continuing.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print the complete experiment plan without running it.",
    )
    return parser.parse_args(argv)


def selected_benchmarks(raw_names: list[str] | None) -> tuple[str, ...]:
    if raw_names is None:
        return BENCHMARKS
    requested = tuple(dict.fromkeys(str(name).strip().lower() for name in raw_names))
    unknown = tuple(name for name in requested if name not in BENCHMARKS)
    if unknown:
        raise ValueError(
            f"Unknown synthetic benchmark(s): {unknown}; available={BENCHMARKS}"
        )
    return requested


def cell_name(n_bad: int) -> str:
    return f"n_bad_{int(n_bad):03d}"


def cell_roots(n_bad: int) -> tuple[Path, Path, Path]:
    name = cell_name(n_bad)
    return (
        RESULT_ROOT / name,
        FIGURE_ROOT / "runner_artifacts" / name,
        TABLE_ROOT / "runner_artifacts" / name,
    )


def configure_runner_cell(n_bad: int) -> tuple[Path, Path, Path]:
    """Configure an exact absolute N_bad value in parent and spawned workers."""

    n_bad = int(n_bad)
    result_root, figure_root, table_root = cell_roots(n_bad)

    runner.SYNTHETIC_Q_BATCH = Q_BATCH
    runner.SYNTHETIC_BATCH_SIZE = BATCH_SIZE
    runner.SYNTHETIC_T_SEQ = T_SEQ
    runner.SYNTHETIC_N_BAD_MULTIPLIER = n_bad
    runner.SYNTHETIC_N_BAD_ABSOLUTE = n_bad
    runner.SYNTHETIC_N_BAD_ENABLED = True
    # Sensitivity plots contain only the six N_bad performance curves. Tree
    # snapshots are intentionally disabled for this grid-search workflow.
    runner.AGVS_SHOW_FINAL_TREE = False

    # Windows workers use spawn and therefore import runner_synthetic.py afresh. Export
    # every setting that must survive that interpreter boundary.
    os.environ["AGVS_SYNTHETIC_Q_BATCH"] = "none"
    os.environ["AGVS_SYNTHETIC_BATCH_SIZE"] = str(BATCH_SIZE)
    os.environ["AGVS_SYNTHETIC_T_SEQ"] = str(T_SEQ)
    os.environ["AGVS_SYNTHETIC_N_BAD_MULTIPLIER"] = str(n_bad)
    os.environ["AGVS_SYNTHETIC_N_BAD_ABSOLUTE"] = str(n_bad)
    os.environ["AGVS_SYNTHETIC_N_BAD_ENABLED"] = "1"
    os.environ["AGVS_SHOW_FINAL_TREE"] = "0"

    runner.SYNTHETIC_RESULT_ROOT = result_root
    runner.SYNTHETIC_FIGURE_ROOT = figure_root
    runner.SYNTHETIC_TABLE_ROOT = table_root
    runner._validated_agvs_batch_policy("synthetic")
    return result_root, figure_root, table_root


def run_cell(
    benchmark: str,
    n_bad: int,
    args: argparse.Namespace,
) -> None:
    configure_runner_cell(n_bad)
    config = make_synthetic_config(benchmark)
    config = runner.apply_runner_agvs_batch_policy(config)
    if not bool(config.get("n_bad_is_absolute")):
        raise RuntimeError("absolute N_bad override was not applied")
    if int(config.get("n_bad_multiplier")) != int(n_bad):
        raise RuntimeError("configured absolute N_bad does not match the grid cell")

    command = [
        "--algorithms",
        *ALGORITHMS,
        "--trials",
        str(int(args.trials)),
        "--workers",
        str(int(args.workers)),
        "--device",
        str(args.device),
        "--seed",
        str(int(args.seed)),
        "--skip-plot",
        "--reuse-complete-json",
    ]
    if args.rerun_complete:
        command.append("--rerun-complete")
    runner.run_benchmark(
        benchmark,
        config,
        ROOT / "Experiment" / "benchmark" / "synthetic",
        argv=command,
    )


def benchmark_result_paths(root: Path, benchmark: str, algorithm: str):
    if not root.is_dir():
        return
    benchmark_key = str(benchmark).lower()
    for benchmark_dir in root.iterdir():
        if not benchmark_dir.is_dir() or benchmark_dir.name.lower() != benchmark_key:
            continue
        algorithm_slug = str(algorithm).strip().lower()
        for algorithm_dir in benchmark_dir.iterdir():
            if not algorithm_dir.is_dir() or algorithm_dir.name.lower() != algorithm_slug:
                continue
            yield from algorithm_dir.glob(f"trial_*_{algorithm_slug}.json")


def finite_history(trace: dict) -> list[float]:
    raw = trace.get("best_observed_history", [])
    if not isinstance(raw, list):
        return []
    values: list[float] = []
    for value in raw:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return []
        if not math.isfinite(numeric):
            return []
        values.append(numeric)
    return values


def load_cell_traces(
    benchmark: str,
    n_bad: int,
    trials: int,
    algorithm: str = ALGORITHMS[0],
) -> tuple[list[dict], str]:
    """Load only complete traces whose recorded runtime settings match the cell."""

    result_root, _, _ = cell_roots(n_bad)
    traces_by_trial: dict[int, dict] = {}
    metric_name = "best observed value"
    for path in benchmark_result_paths(result_root, benchmark, algorithm):
        try:
            with path.open("r", encoding="utf-8") as handle:
                trace = json.load(handle)
        except (OSError, ValueError, TypeError):
            continue
        if str(trace.get("benchmark", "")).lower() != str(benchmark).lower():
            continue
        if str(trace.get("algorithm", "")).upper() != str(algorithm).upper():
            continue
        identity = trace.get("cache_identity", {})
        if not isinstance(identity, dict):
            continue
        if trace.get("complete") is not True or trace.get("error"):
            continue
        try:
            expected = cell_identity(configure_runner_cell, n_bad, benchmark,
                algorithm, int(trace.get("trial_id", -1)), int(identity["requested_max_iter"]))
            if (
                not runner.cache_identities_match(identity, expected)
                and not runner.completed_json_matches_request(
                    trace,
                    expected_identity=expected,
                    benchmark_name=benchmark,
                    algorithm=algorithm,
                    trial=int(trace.get("trial_id", -1)),
                )
            ):
                continue
        except (KeyError, ValueError, TypeError):
            continue
        settings = trace.get("settings", {})
        algorithm_settings = (
            settings.get("algorithm_settings", {})
            if isinstance(settings, dict)
            else {}
        )
        try:
            recorded_n_bad = int(algorithm_settings.get("N_bad"))
            recorded_batch = int(algorithm_settings.get("batch_size"))
            recorded_t_seq = int(algorithm_settings.get("t_seq"))
            trial_id = int(trace.get("trial_id", trace.get("trial", -1)))
        except (TypeError, ValueError):
            continue
        if recorded_n_bad != int(n_bad):
            continue
        if str(algorithm_settings.get("N_bad_mode", "")).lower() != "absolute":
            continue
        if recorded_batch != BATCH_SIZE or recorded_t_seq != T_SEQ:
            continue
        if trial_id < 0 or trial_id >= int(trials):
            continue
        try:
            correct_seed_pair = (
                int(trace.get("algorithm_seed")) == 2021 + trial_id
                and int(trace.get("benchmark_seed")) == 2021 + trial_id
            )
            recorded_noise_variance = float(
                settings.get("benchmark_noise_variance")
            )
        except (TypeError, ValueError):
            continue
        if not correct_seed_pair or not math.isclose(
            recorded_noise_variance,
            float(SYNTHETIC_BENCHMARK_NOISE_VARIANCE),
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            continue
        cache_identity = trace.get("cache_identity", {})
        requested_budget = (
            cache_identity.get("requested_max_iter")
            if isinstance(cache_identity, dict)
            else None
        )
        try:
            complete = int(trace.get("n_evaluations", 0)) >= int(requested_budget)
        except (TypeError, ValueError):
            complete = False
        history = finite_history(trace)
        if not complete or not history:
            continue
        traces_by_trial.setdefault(trial_id, trace)
        if isinstance(settings, dict) and settings.get("metric_name"):
            metric_name = str(settings["metric_name"])
    return [traces_by_trial[key] for key in sorted(traces_by_trial)], metric_name


def cell_is_complete(benchmark: str, n_bad: int, trials: int) -> bool:
    """Return true only when every requested paired trial is cache-complete."""

    return all(
        len(load_cell_traces(benchmark, n_bad, trials, algorithm)[0]) == int(trials)
        for algorithm in ALGORITHMS
    )


def mean_curve(
    benchmark: str,
    n_bad: int,
    trials: int,
    algorithm: str = ALGORITHMS[0],
):
    import numpy as np

    traces, metric = load_cell_traces(benchmark, n_bad, trials, algorithm)
    histories = [np.asarray(finite_history(trace), dtype=float) for trace in traces]
    if not histories:
        return None, None, None, 0, metric
    common_length = min(history.size for history in histories)
    matrix = np.vstack([history[:common_length] for history in histories])
    mean = np.mean(matrix, axis=0)
    if len(histories) > 1:
        sem = np.std(matrix, axis=0, ddof=1) / math.sqrt(len(histories))
        ci95 = 1.96 * sem
    else:
        ci95 = np.zeros(common_length, dtype=float)
    x_axis = np.arange(1, common_length + 1)
    return x_axis, mean, ci95, len(histories), metric


def all_curves(
    benchmark: str,
    n_bad_values: tuple[int, ...],
    trials: int,
) -> list[dict]:
    curves = []
    for algorithm in ALGORITHMS:
        for n_bad in n_bad_values:
            x_axis, mean, ci95, count, metric = mean_curve(
                benchmark, n_bad, trials, algorithm
            )
            if x_axis is None:
                continue
            curves.append(
                {
                    "algorithm": algorithm,
                    "n_bad": int(n_bad),
                    "x": x_axis,
                    "mean": mean,
                    "ci95": ci95,
                    "completed_trials": int(count),
                    "metric": metric,
                }
            )
    return curves


def color_map(n_bad_values: tuple[int, ...]):
    import matplotlib.pyplot as plt
    import numpy as np

    keys = [(algorithm, value) for algorithm in ALGORITHMS for value in n_bad_values]
    colors = plt.get_cmap("turbo")(np.linspace(0.04, 0.96, len(n_bad_values)))
    return {(algorithm, value): colors[index] for algorithm in ALGORITHMS for index, value in enumerate(n_bad_values)}


def draw_iteration_panel(
    axis,
    benchmark: str,
    curves: list[dict],
    colors: dict,
    *,
    show_ci: bool,
    compact: bool,
    include_algorithm_in_label: bool = True,
) -> dict[str, object]:
    import numpy as np

    handles: dict[str, object] = {}
    if not curves:
        axis.text(0.5, 0.5, "No matching complete traces", ha="center", va="center")
        axis.set_axis_off()
        return handles
    series_count = len(curves)
    for series_index, curve in enumerate(curves):
        n_bad = int(curve["n_bad"])
        algorithm = str(curve["algorithm"])
        key = (algorithm, n_bad)
        label = (
            f"{algorithm} (N_bad={n_bad}, n={curve['completed_trials']})"
            if include_algorithm_in_label
            else rf"$N_{{\mathrm{{bad}}}}={n_bad}$"
        )
        line, = axis.plot(
            curve["x"],
            curve["mean"],
            color=colors[key],
            linestyle="-" if algorithm == "AG-VS-BO" else "--",
            linewidth=1.25 if compact else 1.8,
            label=label,
        )
        handles[label] = line
        if show_ci:
            # Four sparse, algorithm-staggered vertical CI positions. Since
            # each series owns different slots, whiskers never share an x.
            slot_count = min(
                max(1, len(curve["x"]) - 1),
                max(series_count, 4 * series_count),
            )
            slot_positions = np.unique(
                np.linspace(1, len(curve["x"]) - 1, slot_count, dtype=int)
            )
            positions = slot_positions[series_index::series_count]
            if positions.size:
                axis.errorbar(
                    curve["x"][positions],
                    curve["mean"][positions],
                    yerr=curve["ci95"][positions],
                    fmt="none",
                    ecolor=colors[key],
                    elinewidth=0.8,
                    capsize=2.0,
                    alpha=0.65,
                    label="_nolegend_",
                )
    axis.set_xlabel("Iteration (objective evaluations)", fontsize=8 if compact else 10)
    axis.set_ylabel("Mean best observed value", fontsize=8 if compact else 10)
    axis.grid(True, alpha=0.22)
    axis.set_xlim(left=1)
    axis.tick_params(labelsize=7 if compact else 9)
    return handles


def save_benchmark_plot(benchmark, n_bad_values, trials):
    """Save BO and TuRBO sensitivity panels side by side in one PDF and PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    curves = all_curves(benchmark, n_bad_values, trials)
    if not curves:
        return []
    colors = color_map(n_bad_values)
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, len(ALGORITHMS), figsize=(20, 6.4), sharey=True, squeeze=False)
    for axis, algorithm in zip(axes[0], ALGORITHMS):
        selected = [curve for curve in curves if curve["algorithm"] == algorithm]
        draw_iteration_panel(axis, benchmark, selected, colors, show_ci=True, compact=False)
        if selected:
            axis.legend(fontsize=8, ncol=2, title="Mean; vertical bars: 95% CI")
        axis.set_title(algorithm)
    figure.suptitle(
        f"{benchmark} | nbad sensitivity\nbatch_size={BATCH_SIZE}, "
        f"noise variance=1e-4, {int(trials)} trials, seeds 2021-{2020 + int(trials)}"
    )
    figure.tight_layout(rect=(0, 0, 1, 0.92))
    outputs = []
    stem = FIGURE_ROOT / f"{benchmark}_nbad_iteration"
    for extension in (".pdf", ".png"):
        output = stem.with_suffix(extension)
        figure.savefig(output, dpi=220, bbox_inches="tight")
        outputs.append(output)
    plt.close(figure)
    return outputs


def benchmark_figure_label(benchmark: str) -> str:
    labels = {
        "rastrigin3_300": r"Rastrigin ($d_{\mathrm{eff}}=3$, $D=300$)",
        "rastrigin6_1000": r"Rastrigin ($d_{\mathrm{eff}}=6$, $D=1000$)",
    }
    return labels.get(str(benchmark).lower(), str(benchmark))


def save_combined_benchmark_plot(benchmarks, n_bad_values, trials):
    """Save every available N_bad benchmark in the t_seq paper-style layout."""

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    benchmarks = tuple(benchmarks)
    n_bad_values = normalized_positive_values(n_bad_values)
    curves_by_benchmark = {
        benchmark: all_curves(benchmark, n_bad_values, trials)
        for benchmark in benchmarks
    }
    available_benchmarks = tuple(
        benchmark for benchmark in benchmarks if curves_by_benchmark[benchmark]
    )
    if not available_benchmarks:
        return []

    colors = color_map(n_bad_values)
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(
        len(available_benchmarks),
        len(ALGORITHMS),
        figsize=(15.5, 5.5 * len(available_benchmarks)),
        sharex=True,
        sharey="row",
        squeeze=False,
    )
    legend_handles: dict[str, object] = {}
    for row_index, benchmark in enumerate(available_benchmarks):
        curves = curves_by_benchmark[benchmark]
        for column_index, algorithm in enumerate(ALGORITHMS):
            axis = axes[row_index, column_index]
            selected = [
                curve for curve in curves if curve["algorithm"] == algorithm
            ]
            handles = draw_iteration_panel(
                axis,
                benchmark,
                selected,
                colors,
                show_ci=True,
                compact=False,
                include_algorithm_in_label=False,
            )
            legend_handles.update(handles)
            axis.set_xlabel("")
            axis.set_ylabel("")
            if row_index == 0:
                axis.set_title(algorithm, fontsize=14, pad=10)
            else:
                axis.set_title("")
            if column_index == 0:
                axis.text(
                    -0.16,
                    0.5,
                    benchmark_figure_label(benchmark),
                    transform=axis.transAxes,
                    rotation=90,
                    ha="center",
                    va="center",
                    fontsize=12,
                    fontweight="semibold",
                )

    figure.supxlabel("Evaluation", fontsize=13, y=0.025)
    figure.supylabel("Value", fontsize=13, x=0.018)
    if legend_handles:
        ordered_labels = [
            rf"$N_{{\mathrm{{bad}}}}={value}$" for value in n_bad_values
        ]
        available_labels = [
            label for label in ordered_labels if label in legend_handles
        ]
        figure.legend(
            [legend_handles[label] for label in available_labels],
            available_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.995),
            ncol=max(1, len(available_labels)),
            frameon=False,
            fontsize=11,
        )
    figure.tight_layout(rect=(0.055, 0.07, 0.995, 0.89), h_pad=2.0, w_pad=1.2)

    outputs = []
    stem = FIGURE_ROOT / "rastrigin_nbad_value_vs_evaluation"
    for extension in (".pdf", ".png"):
        output = stem.with_suffix(extension)
        figure.savefig(output, dpi=220, bbox_inches="tight")
        outputs.append(output)
    plt.close(figure)

    for stale_pattern in (
        "rastrigin*_nbad_iteration.pdf",
        "rastrigin*_nbad_iteration.png",
        "rastrigin*_ag-vs-*_nbad_iteration.pdf",
        "rastrigin*_ag-vs-*_nbad_iteration.png",
    ):
        for stale_path in FIGURE_ROOT.glob(stale_pattern):
            stale_path.unlink(missing_ok=True)
    return outputs



def write_csv_outputs(
    benchmarks: tuple[str, ...],
    n_bad_values: tuple[int, ...],
    trials: int,
) -> tuple[Path, Path]:
    TABLE_ROOT.mkdir(parents=True, exist_ok=True)
    summary_path = TABLE_ROOT / "all_synthetic_nbad_summary.csv"
    curves_path = TABLE_ROOT / "all_synthetic_nbad_iteration_curves.csv"
    summary_fields = (
        "benchmark",
        "algorithm",
        "batch_size",
        "t_seq",
        "n_bad",
        "requested_trials",
        "completed_trials",
        "iterations",
        "mean_final_best_observed",
        "ci95_final_best_observed",
        "metric",
    )
    curve_fields = (
        "benchmark",
        "algorithm",
        "batch_size",
        "t_seq",
        "n_bad",
        "iteration",
        "requested_trials",
        "completed_trials",
        "mean_best_observed",
        "ci95_best_observed",
        "metric",
    )
    with summary_path.open("w", encoding="utf-8", newline="") as summary_handle, curves_path.open(
        "w", encoding="utf-8", newline=""
    ) as curves_handle:
        summary_writer = csv.DictWriter(summary_handle, fieldnames=summary_fields)
        curve_writer = csv.DictWriter(curves_handle, fieldnames=curve_fields)
        summary_writer.writeheader()
        curve_writer.writeheader()
        for benchmark in benchmarks:
            for curve in all_curves(benchmark, n_bad_values, trials):
                summary_writer.writerow(
                    {
                        "benchmark": benchmark,
                        "algorithm": curve["algorithm"],
                        "batch_size": BATCH_SIZE,
                        "t_seq": T_SEQ,
                        "n_bad": curve["n_bad"],
                        "requested_trials": int(trials),
                        "completed_trials": curve["completed_trials"],
                        "iterations": len(curve["x"]),
                        "mean_final_best_observed": float(curve["mean"][-1]),
                        "ci95_final_best_observed": float(curve["ci95"][-1]),
                        "metric": curve["metric"],
                    }
                )
                for iteration, mean, ci95 in zip(
                    curve["x"], curve["mean"], curve["ci95"]
                ):
                    curve_writer.writerow(
                        {
                            "benchmark": benchmark,
                            "algorithm": curve["algorithm"],
                            "batch_size": BATCH_SIZE,
                            "t_seq": T_SEQ,
                            "n_bad": curve["n_bad"],
                            "iteration": int(iteration),
                            "requested_trials": int(trials),
                            "completed_trials": curve["completed_trials"],
                            "mean_best_observed": float(mean),
                            "ci95_best_observed": float(ci95),
                            "metric": curve["metric"],
                        }
                    )
    return summary_path, curves_path


def print_plan(
    benchmarks: tuple[str, ...],
    n_bad_values: tuple[int, ...],
    trials: int,
) -> None:
    cells = len(benchmarks) * len(n_bad_values)
    print(f"Algorithms: {list(ALGORITHMS)}")
    print(f"Fixed settings: batch_size={BATCH_SIZE}, t_seq={T_SEQ}, q_batch={Q_BATCH}")
    print(f"Synthetic benchmarks ({len(benchmarks)}): {list(benchmarks)}")
    print(f"Absolute N_bad values ({len(n_bad_values)}): {list(n_bad_values)}")
    print(f"Algorithm/benchmark/parameter combinations: {cells * len(ALGORITHMS)}")
    print(f"Trials per cell: {int(trials)}")
    print(f"Seeds: 2021..{2020 + int(trials)} (base seed offset 0, trial IDs 0..{int(trials) - 1})")
    print("Gaussian objective-noise variance: 1e-4")
    print(f"Optimizer trials requested: {cells * len(ALGORITHMS) * int(trials)}")
    print(f"Result root: {RESULT_ROOT}")
    print(f"Figure root: {FIGURE_ROOT}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    benchmarks = selected_benchmarks(args.benchmarks)
    n_bad_values = normalized_positive_values(args.n_bad_values)
    print_plan(benchmarks, n_bad_values, args.trials)
    if args.list:
        return 0

    failures: list[str] = []
    if not args.plots_only:
        total_cells = len(benchmarks) * len(n_bad_values)
        current = 0
        for benchmark in benchmarks:
            for n_bad in n_bad_values:
                current += 1
                label = f"{benchmark} | absolute N_bad={n_bad}"
                print(f"\n=== N_bad grid {current}/{total_cells}: {label} ===", flush=True)
                if not args.rerun_complete and cell_is_complete(
                    benchmark,
                    n_bad,
                    int(args.trials),
                ):
                    print(
                        f"[grid-skip] {label}: all {int(args.trials)} trials complete",
                        flush=True,
                    )
                    continue
                try:
                    run_cell(benchmark, n_bad, args)
                except Exception as exc:
                    message = f"{label}: {type(exc).__name__}: {exc}"
                    failures.append(message)
                    print(f"[grid-error] {message}", file=sys.stderr, flush=True)
                    if args.stop_on_error:
                        raise
    # Always regenerate summaries and plots so interrupted/resumed sweeps expose
    # every complete cell currently available.
    summary_path, curves_path = write_csv_outputs(
        benchmarks,
        n_bad_values,
        int(args.trials),
    )
    combined_outputs = save_combined_benchmark_plot(
        benchmarks,
        n_bad_values,
        int(args.trials),
    )
    print(f"\nSummary CSV: {summary_path}")
    print(f"Iteration CSV: {curves_path}")
    print(f"Combined figure: {combined_outputs}")
    if failures:
        print(f"Completed with {len(failures)} failed grid cell(s):", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
