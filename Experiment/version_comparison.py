#!/usr/bin/env python3
"""Compare canonical AG-VS merging with a no-merge tree ablation.

The experiment runs AG-VS-BO and AG-VS-TuRBO only on the registered
Rastrigin ``d_eff=6,D=1000`` benchmark.  Both
versions retain score-based traversal and recursive splitting.  The disabled
version changes one operation only: after every child cohort has been scored,
the siblings remain separate instead of being merged.  The highest-scoring
active child is then selected and split again while it has at least two
dimensions.

Defaults: 25 trials, 25 workers, seeds 2021--2045, batch_size=3, t_seq=2,
and synthetic observation-noise variance 1e-4.  Figures and CSV files report
the running-best noiseless objective value.

Examples from the repository root::

    py -3.13 Experiment/version_comparison.py --list
    py -3.13 Experiment/version_comparison.py
    py -3.13 Experiment/version_comparison.py --plots-only
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import shutil
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Experiment.benchmark import runner
from Experiment.utils import (
    SYNTHETIC_BENCHMARK_NOISE_VARIANCE,
    make_synthetic_config,
)


TRIALS = 25
WORKERS = 25
DEVICE = "cpu"
SEED_OFFSET = 0

BATCH_SIZE = 3
T_SEQ = 2
Q_BATCH = None
N_BAD_MULTIPLIER = 4
AGVS_GP_NOISE_VARIANCE = 1e-2

ALGORITHMS = ("AG-VS-BO", "AG-VS-TuRBO")
BENCHMARKS = (
    "rastrigin6_1000",
)
VERSIONS = ("merge_enabled", "merge_disabled")
VERSION_ENABLED = {
    "merge_enabled": True,
    "merge_disabled": False,
}
VERSION_LABELS = {
    "merge_enabled": "Merge enabled",
    "merge_disabled": "Merge disabled",
}

RESULT_ROOT = ROOT / "Experiment" / "Result" / "version_comparison" / "synthetic"
FIGURE_ROOT = (
    ROOT
    / "Experiment"
    / "Result"
    / "Figures"
    / "version_comparison"
    / "synthetic"
)
TABLE_ROOT = ROOT / "Experiment" / "tables" / "version_comparison" / "synthetic"
SOURCE_SYNTHETIC_RESULT_ROOT = ROOT / "Experiment" / "Result" / "synthetic"


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare merge-enabled and merge-disabled AG-VS-BO/TuRBO on "
            "Rastrigin d_eff=6,D=1000."
        )
    )
    parser.add_argument("--trials", type=positive_int, default=TRIALS)
    parser.add_argument("--workers", type=positive_int, default=WORKERS)
    parser.add_argument("--seed", type=int, choices=(SEED_OFFSET,), default=SEED_OFFSET)
    parser.add_argument("--device", choices=(DEVICE,), default=DEVICE)
    parser.add_argument("--benchmarks", nargs="+", default=None)
    parser.add_argument("--versions", nargs="+", choices=VERSIONS, default=None)
    parser.add_argument(
        "--plots-only",
        action="store_true",
        help="Regenerate CSV and figures from complete matching JSON traces.",
    )
    parser.add_argument(
        "--rerun-complete",
        action="store_true",
        help="Rerun cache-compatible, budget-complete traces.",
    )
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument("--list", action="store_true")
    return parser.parse_args(argv)


def selected_benchmarks(raw: list[str] | None) -> tuple[str, ...]:
    if raw is None:
        return BENCHMARKS
    selected = tuple(dict.fromkeys(str(value).strip().lower() for value in raw))
    unknown = tuple(value for value in selected if value not in BENCHMARKS)
    if unknown:
        raise ValueError(f"Unknown benchmark(s) {unknown}; available={BENCHMARKS}")
    return selected


def selected_versions(raw: list[str] | None) -> tuple[str, ...]:
    if raw is None:
        return VERSIONS
    return tuple(dict.fromkeys(str(value).strip().lower() for value in raw))


def cell_roots(version: str) -> tuple[Path, Path, Path]:
    if version not in VERSION_ENABLED:
        raise ValueError(f"Unknown merge version {version!r}")
    return (
        RESULT_ROOT / version,
        FIGURE_ROOT / "runner_artifacts" / version,
        TABLE_ROOT / "runner_artifacts" / version,
    )


def configure_runner_cell(version: str) -> tuple[Path, Path, Path]:
    """Set parent and Windows-spawned worker controls for one version."""

    merge_enabled = VERSION_ENABLED[version]
    result_root, figure_root, table_root = cell_roots(version)

    runner.SYNTHETIC_Q_BATCH = Q_BATCH
    runner.SYNTHETIC_BATCH_SIZE = BATCH_SIZE
    runner.SYNTHETIC_T_SEQ = T_SEQ
    runner.SYNTHETIC_N_BAD_MULTIPLIER = N_BAD_MULTIPLIER
    runner.SYNTHETIC_N_BAD_ABSOLUTE = None
    runner.SYNTHETIC_N_BAD_ENABLED = True
    runner.AGVS_MERGE_ENABLED_OVERRIDE = merge_enabled
    runner.AGVS_GP_NOISE_VARIANCE_OVERRIDE = AGVS_GP_NOISE_VARIANCE
    runner.AGVS_SHOW_FINAL_TREE = False

    os.environ["AGVS_SYNTHETIC_Q_BATCH"] = "none"
    os.environ["AGVS_SYNTHETIC_BATCH_SIZE"] = str(BATCH_SIZE)
    os.environ["AGVS_SYNTHETIC_T_SEQ"] = str(T_SEQ)
    os.environ["AGVS_SYNTHETIC_N_BAD_MULTIPLIER"] = str(N_BAD_MULTIPLIER)
    os.environ["AGVS_SYNTHETIC_N_BAD_ABSOLUTE"] = "none"
    os.environ["AGVS_SYNTHETIC_N_BAD_ENABLED"] = "1"
    os.environ["AGVS_MERGE_ENABLED"] = "1" if merge_enabled else "0"
    os.environ["AGVS_GP_NOISE_VARIANCE"] = str(AGVS_GP_NOISE_VARIANCE)
    os.environ["AGVS_SHOW_FINAL_TREE"] = "0"

    runner.SYNTHETIC_RESULT_ROOT = result_root
    runner.SYNTHETIC_FIGURE_ROOT = figure_root
    runner.SYNTHETIC_TABLE_ROOT = table_root
    runner._validated_agvs_batch_policy("synthetic")
    return result_root, figure_root, table_root


def make_cell_config(benchmark: str, version: str) -> dict:
    configure_runner_cell(version)
    config = runner.apply_runner_agvs_batch_policy(make_synthetic_config(benchmark))
    config["agvs_merge_enabled"] = VERSION_ENABLED[version]
    # Existing canonical AG-VS-BO traces use alpha=1e-2.  The ablation keeps
    # this surrogate setting fixed across both merge versions so copied and
    # newly executed cells differ only in the merge operation.
    config["agvs_gp_noise_variance"] = AGVS_GP_NOISE_VARIANCE
    runner.register_benchmark(benchmark, config)
    return config


def expected_identity(
    benchmark: str,
    version: str,
    algorithm: str,
    trial: int,
) -> dict:
    config = make_cell_config(benchmark, version)
    return runner.requested_trace_cache_identity(
        config,
        benchmark,
        algorithm,
        int(trial),
        int(config["max_iter"]),
        base_seed_offset=SEED_OFFSET,
        device=DEVICE,
    )


def run_cell(benchmark: str, version: str, args: argparse.Namespace) -> None:
    config = make_cell_config(benchmark, version)
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
    algorithm_key = str(algorithm).lower()
    for benchmark_dir in root.iterdir():
        if not benchmark_dir.is_dir() or benchmark_dir.name.lower() != benchmark_key:
            continue
        for algorithm_dir in benchmark_dir.iterdir():
            if not algorithm_dir.is_dir() or algorithm_dir.name.lower() != algorithm_key:
                continue
            yield from algorithm_dir.glob(f"trial_*_{algorithm_key}.json")


def seed_merge_enabled_bo_results(benchmark: str, trials: int) -> tuple[int, int]:
    """Copy compatible canonical AG-VS-BO traces into the enabled cell.

    Source files are never edited.  Historical native traces predate the
    explicit merge flag; that schema necessarily means canonical merging was
    enabled.  Every scientific setting and seed is validated against the
    requested comparison identity before copying.
    """

    version = "merge_enabled"
    config = make_cell_config(benchmark, version)
    budget = int(config["max_iter"])
    destination_root, _, _ = cell_roots(version)
    copied = 0
    reused = 0
    for trial in range(int(trials)):
        source = runner.trace_path_for(
            SOURCE_SYNTHETIC_RESULT_ROOT, benchmark, "AG-VS-BO", trial
        )
        if not source.is_file():
            continue
        try:
            trace = runner.load_plot_trace(source)
        except (OSError, ValueError, TypeError):
            continue
        expected = runner.requested_trace_cache_identity(
            config,
            benchmark,
            "AG-VS-BO",
            trial,
            budget,
            base_seed_offset=SEED_OFFSET,
            device=DEVICE,
        )
        if not runner.completed_json_matches_request(
            trace,
            expected_identity=expected,
            benchmark_name=benchmark,
            algorithm="AG-VS-BO",
            trial=trial,
        ):
            continue
        if int(trace.get("n_evaluations", 0) or 0) < budget:
            continue
        settings = trace.get("settings", {})
        algorithm_settings = (
            settings.get("algorithm_settings", {}) if isinstance(settings, dict) else {}
        )
        try:
            compatible = (
                int(algorithm_settings.get("batch_size")) == BATCH_SIZE
                and int(algorithm_settings.get("t_seq")) == T_SEQ
                and int(algorithm_settings.get("N_bad_multiplier")) == N_BAD_MULTIPLIER
                and bool(algorithm_settings.get("N_bad_enabled"))
                and math.isclose(
                    float(algorithm_settings.get("alpha")),
                    AGVS_GP_NOISE_VARIANCE,
                    rel_tol=0.0,
                    abs_tol=1e-15,
                )
                and math.isclose(
                    float(settings.get("benchmark_noise_variance")),
                    float(SYNTHETIC_BENCHMARK_NOISE_VARIANCE),
                    rel_tol=0.0,
                    abs_tol=1e-15,
                )
            )
        except (TypeError, ValueError, OverflowError):
            compatible = False
        if not compatible:
            continue
        destination = runner.trace_path_for(
            destination_root, benchmark, "AG-VS-BO", trial
        )
        if destination.is_file():
            try:
                existing = runner.load_plot_trace(destination)
            except (OSError, ValueError, TypeError):
                existing = {}
            if runner.completed_json_matches_request(
                existing,
                expected_identity=expected,
                benchmark_name=benchmark,
                algorithm="AG-VS-BO",
                trial=trial,
            ) and int(existing.get("n_evaluations", 0) or 0) >= budget:
                reused += 1
                continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        copied += 1
    return copied, reused


def finite_noiseless_history(trace: dict) -> list[float]:
    raw = runner.trace_noiseless_history_for_plot(trace)
    if not isinstance(raw, list):
        return []
    values: list[float] = []
    for value in raw:
        try:
            numeric = float(value)
        except (TypeError, ValueError, OverflowError):
            return []
        if not math.isfinite(numeric):
            return []
        values.append(numeric)
    return values


def load_cell_traces(
    benchmark: str,
    version: str,
    trials: int,
    algorithm: str,
) -> tuple[list[dict], str]:
    """Load complete traces that exactly match the requested merge condition."""

    config = make_cell_config(benchmark, version)
    budget = int(config["max_iter"])
    result_root, _, _ = cell_roots(version)
    traces_by_trial: dict[int, dict] = {}
    metric = "best noiseless objective value"
    for path in benchmark_result_paths(result_root, benchmark, algorithm):
        try:
            trace = runner.load_plot_trace(path)
            trial_id = int(trace.get("trial_id", trace.get("trial", -1)))
        except (OSError, ValueError, TypeError):
            continue
        if not (0 <= trial_id < int(trials)):
            continue
        if str(trace.get("benchmark", "")).lower() != benchmark.lower():
            continue
        try:
            if runner.canonical_algorithm_name(trace.get("algorithm")) != algorithm:
                continue
        except (TypeError, ValueError):
            continue
        if trace.get("complete") is not True or trace.get("error"):
            continue
        identity = trace.get("cache_identity")
        if not isinstance(identity, dict):
            continue
        expected = runner.requested_trace_cache_identity(
            config,
            benchmark,
            algorithm,
            trial_id,
            budget,
            base_seed_offset=SEED_OFFSET,
            device=DEVICE,
        )
        if not runner.cache_identities_match(identity, expected):
            if not runner.completed_json_matches_request(
                trace,
                expected_identity=expected,
                benchmark_name=benchmark,
                algorithm=algorithm,
                trial=trial_id,
            ):
                continue
        settings = trace.get("settings", {})
        algorithm_settings = (
            settings.get("algorithm_settings", {}) if isinstance(settings, dict) else {}
        )
        recorded_merge = algorithm_settings.get("merge_enabled")
        if recorded_merge is not VERSION_ENABLED[version]:
            legacy_canonical_merge = (
                version == "merge_enabled"
                and recorded_merge is None
                and identity.get("benchmark_config", {}).get("agvs_merge_enabled") is None
            )
            if not legacy_canonical_merge:
                continue
        try:
            gp_settings_match = True
            if algorithm == "AG-VS-BO":
                gp_settings_match = math.isclose(
                    float(
                        algorithm_settings.get(
                            "noise_variance", algorithm_settings.get("alpha")
                        )
                    ),
                    AGVS_GP_NOISE_VARIANCE,
                    rel_tol=0.0,
                    abs_tol=1e-15,
                )
            valid_settings = (
                int(algorithm_settings.get("batch_size")) == BATCH_SIZE
                and int(algorithm_settings.get("t_seq")) == T_SEQ
                and gp_settings_match
                and int(trace.get("algorithm_seed")) == 2021 + trial_id
                and int(trace.get("benchmark_seed")) == 2021 + trial_id
                and int(trace.get("n_evaluations", 0)) >= budget
                and math.isclose(
                    float(settings.get("benchmark_noise_variance")),
                    float(SYNTHETIC_BENCHMARK_NOISE_VARIANCE),
                    rel_tol=0.0,
                    abs_tol=1e-15,
                )
            )
        except (TypeError, ValueError, OverflowError):
            valid_settings = False
        history = finite_noiseless_history(trace)
        if not valid_settings or len(history) < budget:
            continue
        traces_by_trial.setdefault(trial_id, trace)
    return [traces_by_trial[key] for key in sorted(traces_by_trial)], metric


def cell_is_complete(benchmark: str, version: str, trials: int) -> bool:
    return all(
        len(load_cell_traces(benchmark, version, trials, algorithm)[0]) == int(trials)
        for algorithm in ALGORITHMS
    )


def mean_curve(benchmark: str, version: str, trials: int, algorithm: str):
    import numpy as np

    traces, metric = load_cell_traces(benchmark, version, trials, algorithm)
    histories = [np.asarray(finite_noiseless_history(trace), dtype=float) for trace in traces]
    if not histories:
        return None, None, None, 0, metric
    common_length = min(history.size for history in histories)
    matrix = np.vstack([history[:common_length] for history in histories])
    mean = np.mean(matrix, axis=0)
    if len(histories) > 1:
        ci95 = 1.96 * np.std(matrix, axis=0, ddof=1) / math.sqrt(len(histories))
    else:
        ci95 = np.zeros(common_length, dtype=float)
    return np.arange(1, common_length + 1), mean, ci95, len(histories), metric


def all_curves(benchmark: str, versions: tuple[str, ...], trials: int) -> list[dict]:
    curves: list[dict] = []
    for algorithm in ALGORITHMS:
        for version in versions:
            x_axis, mean, ci95, count, metric = mean_curve(
                benchmark, version, trials, algorithm
            )
            if x_axis is None:
                continue
            curves.append(
                {
                    "algorithm": algorithm,
                    "version": version,
                    "x": x_axis,
                    "mean": mean,
                    "ci95": ci95,
                    "completed_trials": count,
                    "metric": metric,
                }
            )
    return curves


def benchmark_label(benchmark: str) -> str:
    return {
        "rastrigin3_300": r"Rastrigin ($d_{\mathrm{eff}}=3$, $D=300$)",
        "rastrigin6_1000": r"Rastrigin ($d_{\mathrm{eff}}=6$, $D=1000$)",
        "rosenbrock3_300": r"Rosenbrock ($d_{\mathrm{eff}}=3$, $D=300$)",
        "rosenbrock6_1000": r"Rosenbrock ($d_{\mathrm{eff}}=6$, $D=1000$)",
    }.get(benchmark, benchmark)


def save_combined_figure(
    benchmarks: tuple[str, ...],
    versions: tuple[str, ...],
    trials: int,
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    curves_by_benchmark = {
        benchmark: all_curves(benchmark, versions, trials) for benchmark in benchmarks
    }
    available = tuple(
        benchmark for benchmark in benchmarks if curves_by_benchmark[benchmark]
    )
    if not available:
        return []

    colors = {"merge_enabled": "#1f77b4", "merge_disabled": "#d95f02"}
    linestyles = {"merge_enabled": "-", "merge_disabled": "--"}
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(
        len(available),
        len(ALGORITHMS),
        figsize=(15.5, 5.5 * len(available)),
        sharex=True,
        sharey="row",
        squeeze=False,
    )
    legend_handles: dict[str, object] = {}
    for row, benchmark in enumerate(available):
        curves = curves_by_benchmark[benchmark]
        for column, algorithm in enumerate(ALGORITHMS):
            axis = axes[row, column]
            selected = [curve for curve in curves if curve["algorithm"] == algorithm]
            for series_index, curve in enumerate(selected):
                version = curve["version"]
                label = VERSION_LABELS[version]
                line, = axis.plot(
                    curve["x"],
                    curve["mean"],
                    color=colors[version],
                    linestyle=linestyles[version],
                    linewidth=1.9,
                    label=label,
                )
                legend_handles[label] = line
                slot_count = min(
                    max(1, len(curve["x"]) - 1),
                    max(len(selected), 4 * len(selected)),
                )
                positions = np.unique(
                    np.linspace(1, len(curve["x"]) - 1, slot_count, dtype=int)
                )[series_index::max(1, len(selected))]
                if positions.size:
                    axis.errorbar(
                        curve["x"][positions],
                        curve["mean"][positions],
                        yerr=curve["ci95"][positions],
                        fmt="none",
                        ecolor=colors[version],
                        elinewidth=0.8,
                        capsize=2.0,
                        alpha=0.65,
                        label="_nolegend_",
                    )
            axis.grid(True, alpha=0.22)
            axis.set_xlim(left=1)
            axis.set_xlabel("")
            axis.set_ylabel("")
            if row == 0:
                axis.set_title(algorithm, fontsize=14, pad=10)
            if column == 0:
                axis.text(
                    -0.16,
                    0.5,
                    benchmark_label(benchmark),
                    transform=axis.transAxes,
                    rotation=90,
                    ha="center",
                    va="center",
                    fontsize=12,
                    fontweight="semibold",
                )

    figure.supxlabel("Number of evaluations", fontsize=13, y=0.025)
    figure.supylabel("Value", fontsize=13, x=0.018)
    ordered_labels = [VERSION_LABELS[version] for version in versions]
    available_labels = [label for label in ordered_labels if label in legend_handles]
    if available_labels:
        figure.legend(
            [legend_handles[label] for label in available_labels],
            available_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.995),
            ncol=len(available_labels),
            frameon=False,
            fontsize=11,
        )
    figure.tight_layout(rect=(0.055, 0.07, 0.995, 0.89), h_pad=2.0, w_pad=1.2)

    stem_name = (
        f"{available[0]}_merge_version_comparison_value_vs_evaluation"
        if len(available) == 1
        else "merge_version_comparison_value_vs_evaluation"
    )
    stem = FIGURE_ROOT / stem_name
    outputs: list[Path] = []
    for suffix in (".pdf", ".png"):
        output = stem.with_suffix(suffix)
        figure.savefig(output, dpi=220, bbox_inches="tight")
        outputs.append(output)
    plt.close(figure)
    return outputs


def write_csv_outputs(
    benchmarks: tuple[str, ...],
    versions: tuple[str, ...],
    trials: int,
) -> tuple[Path, Path]:
    TABLE_ROOT.mkdir(parents=True, exist_ok=True)
    summary_path = TABLE_ROOT / "version_comparison_summary.csv"
    curves_path = TABLE_ROOT / "version_comparison_curves.csv"
    common_fields = (
        "benchmark",
        "algorithm",
        "version",
        "merge_enabled",
        "batch_size",
        "t_seq",
        "requested_trials",
        "completed_trials",
    )
    with summary_path.open("w", encoding="utf-8", newline="") as summary_handle, curves_path.open(
        "w", encoding="utf-8", newline=""
    ) as curves_handle:
        summary_writer = csv.DictWriter(
            summary_handle,
            fieldnames=common_fields
            + ("evaluations", "mean_final_best_noiseless", "ci95_final_best_noiseless", "metric"),
        )
        curves_writer = csv.DictWriter(
            curves_handle,
            fieldnames=common_fields
            + ("evaluation", "mean_best_noiseless", "ci95_best_noiseless", "metric"),
        )
        summary_writer.writeheader()
        curves_writer.writeheader()
        for benchmark in benchmarks:
            for curve in all_curves(benchmark, versions, trials):
                common = {
                    "benchmark": benchmark,
                    "algorithm": curve["algorithm"],
                    "version": curve["version"],
                    "merge_enabled": VERSION_ENABLED[curve["version"]],
                    "batch_size": BATCH_SIZE,
                    "t_seq": T_SEQ,
                    "requested_trials": int(trials),
                    "completed_trials": int(curve["completed_trials"]),
                }
                summary_writer.writerow(
                    {
                        **common,
                        "evaluations": len(curve["x"]),
                        "mean_final_best_noiseless": float(curve["mean"][-1]),
                        "ci95_final_best_noiseless": float(curve["ci95"][-1]),
                        "metric": curve["metric"],
                    }
                )
                for evaluation, mean, ci95 in zip(
                    curve["x"], curve["mean"], curve["ci95"]
                ):
                    curves_writer.writerow(
                        {
                            **common,
                            "evaluation": int(evaluation),
                            "mean_best_noiseless": float(mean),
                            "ci95_best_noiseless": float(ci95),
                            "metric": curve["metric"],
                        }
                    )
    return summary_path, curves_path


def print_plan(
    benchmarks: tuple[str, ...], versions: tuple[str, ...], trials: int, workers: int
) -> None:
    combinations = len(benchmarks) * len(versions) * len(ALGORITHMS)
    print(f"Benchmarks: {list(benchmarks)}")
    print(f"Algorithms: {list(ALGORITHMS)}")
    print(f"Versions: {list(versions)}")
    print(
        f"Fixed settings: batch_size={BATCH_SIZE}, t_seq={T_SEQ}, "
        f"N_bad={N_BAD_MULTIPLIER}*N_split, AG-VS GP variance={AGVS_GP_NOISE_VARIANCE:g}"
    )
    print(f"Trials per algorithm/version/benchmark: {int(trials)}")
    print(f"Workers: {int(workers)}")
    print(f"Seeds: 2021..{2020 + int(trials)}")
    print("Performance metric: running-best noiseless objective value")
    print(f"Total optimizer trials: {combinations * int(trials)}")
    print(f"Result root: {RESULT_ROOT}")
    print(f"Figure root: {FIGURE_ROOT}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    benchmarks = selected_benchmarks(args.benchmarks)
    versions = selected_versions(args.versions)
    print_plan(benchmarks, versions, args.trials, args.workers)
    if args.list:
        return 0

    failures: list[str] = []
    if not args.plots_only:
        if "merge_enabled" in versions:
            for benchmark in benchmarks:
                copied, reused = seed_merge_enabled_bo_results(
                    benchmark, int(args.trials)
                )
                print(
                    f"[source-copy] {benchmark} | AG-VS-BO | Merge enabled: "
                    f"copied={copied}, already_present={reused}",
                    flush=True,
                )
        total_cells = len(benchmarks) * len(versions)
        current = 0
        for benchmark in benchmarks:
            for version in versions:
                current += 1
                label = f"{benchmark} | {VERSION_LABELS[version]}"
                print(f"\n=== Version comparison {current}/{total_cells}: {label} ===", flush=True)
                if not args.rerun_complete and cell_is_complete(
                    benchmark, version, int(args.trials)
                ):
                    print(
                        f"[version-skip] {label}: all {int(args.trials)} paired trials complete",
                        flush=True,
                    )
                    continue
                try:
                    run_cell(benchmark, version, args)
                except Exception as exc:
                    message = f"{label}: {type(exc).__name__}: {exc}"
                    failures.append(message)
                    print(f"[version-error] {message}", file=sys.stderr, flush=True)
                    if args.stop_on_error:
                        raise

    summary_path, curves_path = write_csv_outputs(benchmarks, versions, int(args.trials))
    figure_paths = save_combined_figure(benchmarks, versions, int(args.trials))
    print(f"\nSummary CSV: {summary_path}")
    print(f"Curve CSV: {curves_path}")
    print(f"Combined figure: {figure_paths}")
    if failures:
        print(f"Completed with {len(failures)} failed cell(s):", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
