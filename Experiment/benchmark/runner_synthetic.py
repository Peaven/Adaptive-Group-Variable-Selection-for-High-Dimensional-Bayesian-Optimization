#!/usr/bin/env python3
"""
runner_synthetic.py

Shared execution and plotting engine for the registered AG-VS benchmarks.
"""

import argparse
import builtins
import contextlib
import concurrent.futures
import copy
import gc
import hashlib
import importlib.metadata
import importlib.util
import inspect
import json
import math
import mmap
import multiprocessing
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
import warnings
from collections import defaultdict
from pathlib import Path
from statistics import NormalDist

# Ensure root directories are in sys.path
HERE = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[2]
RESULT_ROOT = ROOT / "Experiment" / "Result"
SYNTHETIC_RESULT_ROOT = RESULT_ROOT / "synthetic"
REAL_LIFE_RESULT_ROOT = RESULT_ROOT / "real_life"
FIGURE_ROOT = RESULT_ROOT / "Figures"
SYNTHETIC_FIGURE_ROOT = FIGURE_ROOT / "synthetic"
REAL_LIFE_FIGURE_ROOT = FIGURE_ROOT / "real_life"
TABLE_ROOT = ROOT / "Experiment" / "tables"
SYNTHETIC_TABLE_ROOT = TABLE_ROOT / "synthetic"
REAL_LIFE_TABLE_ROOT = TABLE_ROOT / "real_life"
# Existing benchmark scripts are synthetic; keep the old constant as an alias.
RESULT_FIGURE_ROOT = SYNTHETIC_FIGURE_ROOT


def output_roots_for_experiment_kind(kind):
    """Return isolated figure/table roots for synthetic or real-life runs."""
    normalized = str(kind).strip().lower().replace("-", "_")
    if normalized == "synthetic":
        return SYNTHETIC_FIGURE_ROOT, SYNTHETIC_TABLE_ROOT
    if normalized in {"real_life", "realworld", "real_world"}:
        return REAL_LIFE_FIGURE_ROOT, REAL_LIFE_TABLE_ROOT
    raise ValueError("experiment kind must be 'synthetic' or 'real_life'")


def result_root_for_experiment_kind(kind):
    normalized = str(kind).strip().lower().replace("-", "_")
    if normalized == "synthetic":
        return SYNTHETIC_RESULT_ROOT
    if normalized in {"real_life", "realworld", "real_world"}:
        return REAL_LIFE_RESULT_ROOT
    raise ValueError("experiment kind must be 'synthetic' or 'real_life'")


def remove_legacy_millisecond_figure_aliases(figure_root):
    """Remove obsolete generated aliases whose names advertise millisecond units."""

    removed = []
    for path in Path(figure_root).rglob("*milliseconds*.pdf"):
        path.unlink(missing_ok=True)
        removed.append(str(path))
    return removed
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

for p in [
    ROOT,
    ROOT / "Experiment" / "benchmark" / "baseline",
]:
    if p.exists():
        p_str = str(p)
        if p_str not in sys.path:
            sys.path.insert(0, p_str)

# Keep the project package directory ahead of vendored baseline source paths.
# In particular, baseline modules must not shadow the project-owned ag_vs package.
root_path = str(ROOT)
if root_path in sys.path:
    sys.path.remove(root_path)
sys.path.insert(0, root_path)

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "ag-vs-matplotlib")
)
# Independent trials are parallelized at the process level.  Keeping numerical
# libraries single-threaded prevents each worker from spawning another full set
# of CPU threads (severe oversubscription on many-core servers).
for _thread_env_var in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_thread_env_var, "1")

def _env_bool(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return value.strip().lower() not in {"", "0", "false", "no", "off"}


def _env_optional_bool(name):
    value = os.environ.get(name)
    if value is None or value.strip().lower() in {"", "none", "null"}:
        return None
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean or 'none'")


def _env_optional_positive_int(name, default):
    """Read a positive integer or the literal ``none`` from the environment."""

    value = os.environ.get(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"", "none", "null"}:
        return None
    parsed = int(normalized)
    if parsed <= 0:
        raise ValueError(f"{name} must be a positive integer or 'none'")
    return parsed


def _env_optional_nonnegative_float(name):
    value = os.environ.get(name)
    if value is None or value.strip().lower() in {"", "none", "null"}:
        return None
    parsed = float(value)
    if not np.isfinite(parsed) or parsed < 0.0:
        raise ValueError(f"{name} must be a finite non-negative number or 'none'")
    return parsed


_MPL_LIVE_PLOT_ON = _env_bool("AGVS_PYTHON_LIVE_PLOT_ON", False)
_MEMORY_DIAGNOSTICS_ON = _env_bool("AGVS_MEMORY_DIAGNOSTICS", False)
_RECYCLE_WORKERS_ON = _env_bool("AGVS_RECYCLE_WORKERS", True)
try:
    EVALUATION_PROGRESS_EVERY = max(
        1,
        int(os.environ.get("AGVS_PROGRESS_EVERY", "10")),
    )
except (TypeError, ValueError):
    EVALUATION_PROGRESS_EVERY = 10

import matplotlib
if not _MPL_LIVE_PLOT_ON:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator
import numpy as np
from Experiment.utils import (
    BASE_ALGORITHM_SEED,
    BASE_BENCHMARK_SEED,
    REAL_LIFE_ALGORITHMS,
    REAL_LIFE_EXPERIMENTS,
    SYNTHETIC_ALGORITHMS,
    SYNTHETIC_BENCHMARK_NOISE_STD,
    SYNTHETIC_BENCHMARK_NOISE_VARIANCE,
    SYNTHETIC_EXPERIMENTS,
    SYNTHETIC_TRIALS,
    get_real_life_experiment,
    iter_applicable_real_life_algorithms,
    real_life_algorithm_is_applicable,
    real_life_optimizer_dimension,
    real_life_reference_embedding_dimension,
)
from Experiment.utils import make_all_synthetic_configs
try:
    from scipy.stats import wilcoxon
except Exception:
    wilcoxon = None

warnings.filterwarnings("ignore")


def process_memory_mb():
    """Return current process memory in MB for optional diagnostics."""
    try:
        import psutil

        process = psutil.Process(os.getpid())
        return float(process.memory_info().rss) / (1024.0 ** 2)
    except Exception:
        try:
            import resource

            usage = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            if os.name == "posix":
                usage /= 1024.0
            return usage
        except Exception:
            pass
    if os.name == "nt":
        try:
            import ctypes
            from ctypes import wintypes

            class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            counters = PROCESS_MEMORY_COUNTERS()
            counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
            handle = ctypes.windll.kernel32.GetCurrentProcess()
            get_process_memory_info = ctypes.WinDLL("Psapi.dll").GetProcessMemoryInfo
            get_process_memory_info.argtypes = [
                wintypes.HANDLE,
                ctypes.POINTER(PROCESS_MEMORY_COUNTERS),
                wintypes.DWORD,
            ]
            get_process_memory_info.restype = wintypes.BOOL
            ok = get_process_memory_info(
                handle,
                ctypes.byref(counters),
                counters.cb,
            )
            if ok:
                return float(counters.WorkingSetSize) / (1024.0 ** 2)
        except Exception:
            pass
    return None


def print_memory_diagnostic(label, benchmark_name=None, algorithm=None, trial=None):
    if not _MEMORY_DIAGNOSTICS_ON:
        return
    memory_mb = process_memory_mb()
    if memory_mb is None:
        return
    parts = [f"[memory] {label}", f"rss_mb={memory_mb:.1f}"]
    if benchmark_name is not None:
        parts.append(f"benchmark={benchmark_name}")
    if algorithm is not None:
        parts.append(f"algorithm={algorithm}")
    if trial is not None:
        parts.append(f"trial={int(trial):03d}")
    print(" ".join(parts), flush=True)


def release_trial_memory(optimizer=None, result=None, recorder=None):
    """Drop large per-trial model/data references after JSON artifacts are written."""
    seen_nodes = set()

    def visit_node(node):
        if node is None:
            return
        node_id = id(node)
        if node_id in seen_nodes:
            return
        seen_nodes.add(node_id)
        for attr in (
            "model",
            "cached_ei_x",
            "_gp_fit_signature",
            "turbo_best_x",
            "turbo_best_y",
        ):
            if hasattr(node, attr):
                try:
                    setattr(node, attr, None)
                except Exception:
                    pass
        for child in list(getattr(node, "children", []) or []):
            visit_node(child)

    for owner in (optimizer, result):
        for attr in ("root",):
            visit_node(getattr(owner, attr, None))
        for attr in ("roots", "_all_nodes", "_plot_nodes"):
            for node in list(getattr(owner, attr, []) or []):
                visit_node(node)

    if optimizer is not None:
        for attr in (
            "tree_plot_fig",
            "tree_plot_ax",
        ):
            if hasattr(optimizer, attr):
                try:
                    setattr(optimizer, attr, None)
                except Exception:
                    pass
        for attr in ("X", "y", "true_y", "history", "gp_fit_history", "threshold_trace", "sad_merge_history"):
            value = getattr(optimizer, attr, None)
            if isinstance(value, list):
                value.clear()

    if recorder is not None:
        for attr in ("x_history", "true_history", "observed_history", "eval_wall_clock_history"):
            value = getattr(recorder, attr, None)
            if isinstance(value, list):
                value.clear()

    try:
        plt.close("all")
    except Exception:
        pass
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    gc.collect()


@contextlib.contextmanager
def suppress_all_prints():
    """Suppress stdout/stderr and Python print calls from noisy optimizer internals."""
    original_print = builtins.print

    def _silent_print(*args, **kwargs):
        return None

    with open(os.devnull, "w") as devnull:
        try:
            builtins.print = _silent_print
            with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                yield
        finally:
            builtins.print = original_print


def run_benchmark(benchmark, config, script_dir, argv=None):
    """Run one benchmark script and store every output in the central Result tree."""
    del script_dir
    config = apply_runner_agvs_batch_policy(config)
    # Re-register the normalized copy so all downstream consumers observe the
    # same runner-owned batch settings.
    register_benchmark(benchmark, config)
    experiment_kind = str(config.get("experiment_kind", "synthetic"))
    if experiment_kind == "real_life":
        real_experiment = real_life_experiment_for_config(config, benchmark)
        default_algorithms = list(iter_applicable_real_life_algorithms(real_experiment))
    else:
        default_algorithms = list(DEFAULT_ALGORITHMS)
    result_dir = result_root_for_experiment_kind(experiment_kind)
    json_dir = result_dir
    figure_dir, table_dir = output_roots_for_experiment_kind(experiment_kind)
    parser = argparse.ArgumentParser(description=f"Run AG-VS benchmark: {benchmark}")
    parser.add_argument("--trials", type=int, default=int(config.get("trials", TRIALS)))
    parser.add_argument("--trial-start", type=int, default=0)
    parser.add_argument("--defer-reports", action="store_true")
    parser.add_argument(
        "--max-iter",
        type=int,
        default=int(config.get("max_iter", default_max_iter_for_effective_dim(benchmark_effective_dim(config)))),
    )
    parser.add_argument("--algorithms", nargs="*", default=default_algorithms)
    parser.add_argument("--seed", type=int, choices=(0,), default=0)
    parser.add_argument("--skip-plot", action="store_true")
    parser.add_argument("--rerun-complete", action="store_true")
    parser.add_argument(
        "--reuse-complete-json",
        action="store_true",
        help=(
            "Treat an error-free JSON that reached the requested budget as "
            "complete even when its cache metadata comes from an older runner. "
            "The benchmark, algorithm, trial, seeds, and device must still match."
        ),
    )
    parser.add_argument(
        "--resume-under-budget",
        action="store_true",
        help=(
            "Continue a compatible completed JSON that is below the requested "
            "budget by deterministically replaying its stored evaluations, then "
            "evaluating only the missing suffix."
        ),
    )
    parser.add_argument(
        "--workers",
        type=positive_int,
        default=WORKERS,
        help=f"Worker count (runner SETTINGS default: {WORKERS}).",
    )
    parser.add_argument(
        "--device",
        choices=(DEVICE,),
        default=DEVICE,
        help=f"Fixed at the runner SETTINGS value ({DEVICE}) for fair comparison.",
    )
    add_n_bad_cli_flags(parser)
    args = parser.parse_args(argv)
    args.device = normalize_device(args.device)
    if args.n_bad_enabled is not None:
        config = dict(config)
        config["n_bad_enabled"] = bool(args.n_bad_enabled)
        # apply_runner_agvs_batch_policy owns defaults; an explicit CLI value
        # is the final per-run override stored in the runtime registry.
        BENCHMARKS[str(benchmark).lower()] = config
    if args.trials is not None and int(args.trials) <= 0:
        raise ValueError("--trials must be a positive integer")
    if args.max_iter is not None and int(args.max_iter) <= 0:
        raise ValueError("--max-iter must be a positive integer")
    if int(args.trials) <= 0:
        raise ValueError("--trials must be a positive integer")
    if not 0 <= args.trial_start < args.trials:
        raise ValueError("--trial-start must be between zero and --trials - 1")
    if not args.algorithms:
        raise ValueError("At least one algorithm must be selected")

    json_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    algorithms_needing_runtime = algorithms_requiring_runtime_for_request(
        output_dir=json_dir,
        benchmark=benchmark,
        algorithms=args.algorithms,
        trials=args.trials,
        max_iter=args.max_iter,
        base_seed_offset=args.seed,
        device=args.device,
        reuse_complete_json=args.reuse_complete_json,
    )
    preflight_reference_runtime(
        algorithms_needing_runtime,
        benchmark_name=benchmark,
    )
    preflight_algorithm_devices(algorithms_needing_runtime)
    preflight_real_life_config(config)

    print(f"Benchmark: {benchmark}")
    print(f"Algorithms: {normalize_algorithm_args(args.algorithms)}")
    print(f"Trials: {args.trials}")
    print(f"Max evaluations per run: {args.max_iter}")
    print(f"Device: {args.device}")
    print(f"N_bad reinitialization enabled: {bool(config.get('n_bad_enabled', False))}")
    print(f"JSON directory: {json_dir}")
    print(f"Figure directory: {figure_dir}")
    run_status = run_round_robin(
        benchmarks=[benchmark], algorithms=args.algorithms, trials=args.trials,
        max_iter=args.max_iter, output_dir=str(json_dir), figure_dir=str(figure_dir),
        seed=args.seed, update_plots=(FIGURE_MODE and not args.skip_plot and not args.defer_reports),
        skip_complete=(not args.rerun_complete),
        workers=args.workers,
        device=args.device,
        table_dir=str(table_dir),
        trial_start=args.trial_start,
        reuse_complete_json=args.reuse_complete_json,
        resume_under_budget=args.resume_under_budget,
    )
    run_complete = bool(run_status.get("complete", False))
    if args.defer_reports:
        pass  # The parent refreshes reports after every function finishes its batch.
    elif FIGURE_MODE and not args.skip_plot:
        plot_results(
            str(json_dir),
            str(figure_dir),
            benchmark_filter=benchmark,
            table_dir=str(table_dir),
        )
        if benchmark in SYNTHETIC_EXPERIMENTS:
            save_synthetic_comparison_pdfs(
                benchmark_root=json_dir,
                figure_root=figure_dir,
            )
    else:
        # Tables are aggregate experiment artifacts, not optional figures. Keep
        # them current after every successfully completed benchmark even when
        # figure generation is disabled with --skip-plot.
        save_all_benchmark_objective_tables(
            benchmark_root=json_dir,
            table_dir=table_dir,
        )
    if not run_complete:
        raise RuntimeError(
            "Benchmark finished with failed or incomplete trials; figures were "
            "updated from every complete trace before raising this error; see "
            f"{run_status.get('status_path', 'run_status.json')}"
        )


def _import_first(candidates, symbol):
    errors = []
    for statement in candidates:
        ns = {}
        try:
            exec(statement, ns)
            return ns[symbol]
        except Exception as exc:
            errors.append(f"{statement} -> {exc}")
    raise ImportError(f"Could not import {symbol}. Tried:\n" + "\n".join(errors))


try:
    MCTSVSOptimizer = _import_first(
        [
            "from Experiment.benchmark.baseline.native_optimizers import NativeMCTSVSOptimizer as MCTSVSOptimizer",
        ],
        "MCTSVSOptimizer",
    )
except ImportError as exc:
    MCTSVSOptimizer = None
    MCTSVS_IMPORT_ERROR = exc
else:
    MCTSVS_IMPORT_ERROR = None


try:
    Turbo1 = _import_first(
        [
            "from Experiment.benchmark.baseline.native_optimizers import NativeTurbo1Optimizer as Turbo1",
        ],
        "Turbo1",
    )
except ImportError as exc:
    Turbo1 = None
    TURBO_IMPORT_ERROR = exc
else:
    TURBO_IMPORT_ERROR = None


try:
    VanillaBOOptimizer = _import_first(
        [
            "from Experiment.benchmark.baseline.vanilla_bo_adapter import VanillaBOOptimizer",
        ],
        "VanillaBOOptimizer",
    )
except ImportError as exc:
    VanillaBOOptimizer = None
    VANILLA_BO_IMPORT_ERROR = exc
else:
    VANILLA_BO_IMPORT_ERROR = None

from Experiment.benchmark.baseline.native_optimizers import (
    OfficialCMAESOptimizer,
    NativeRandomSearchOptimizer,
    MCTSVSBOOptimizer,
    MCTSVSRandomSearchOptimizer,
    MCTSVSSAASBOOptimizer,
    MCTSVSTuRBOOptimizer,
)
from Experiment.benchmark.baseline.native_baselines import (
    DropoutBOOptimizer,
    DropoutRandomSearchOptimizer,
    DropoutTuRBOOptimizer,
    LAMCTSBOOptimizer,
    LAMCTSTuRBOOptimizer,
)
from Experiment.benchmark.baseline.lassobo_adapter import LassoBOOptimizer
from Experiment.benchmark.baseline.embeddings import HeSBOOptimizer, ALEBOOptimizer
from Experiment.benchmark.baseline.gtbo import GTBOOptimizer, GTBO_SETTINGS
from Experiment.utils import REAL_LIFE_ALGORITHMS
try:
    from Experiment.benchmark.baseline.add_gp_ucb import ADDGPUCBOptimizer
except ImportError as exc:
    ADDGPUCBOptimizer = None
    ADD_GP_IMPORT_ERROR = exc
else:
    ADD_GP_IMPORT_ERROR = None
from Experiment.benchmark.baseline.native_baselines import (
    NativeBaselineOptimizer,
)
from Experiment.benchmark.baseline.reference_infrastructure import (
    BO_REFERENCE_SPEC,
    REAL_LIFE_GP_JITTER,
    REAL_LIFE_GP_NOISE_BOUNDS,
    TURBO_REFERENCE_SPEC,
    bo_reference_dict,
    turbo_reference_dict,
)


try:
    AGVSOptimizer = _import_first(
        [
            "from ag_vs import AGVSOptimizer",
            "from ag_vs.optimizer import AGVSOptimizer",
        ],
        "AGVSOptimizer",
    )
except ImportError as exc:
    AGVSOptimizer = None
    AGVS_IMPORT_ERROR = exc
else:
    AGVS_IMPORT_ERROR = None


# =============================================================================
# SETTINGS — edit experiment/runtime controls only in this block
# =============================================================================

def normalize_device(device):
    """Normalize the single runner-wide compute-device setting."""
    value = str(device).strip().lower()
    if value not in {"cpu", "cuda"}:
        raise ValueError("device must be 'cpu' or 'cuda'.")
    return value


def positive_int(value):
    """Argparse type for positive runner resource counts."""
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


# Run selection and resources.
TRIALS = SYNTHETIC_TRIALS
WORKERS = 1
SYNTHETIC_TRIAL_BATCH_SIZE = 25
DEVICE = normalize_device(os.environ.get("AGVS_DEVICE", "cpu"))
MAX_EXPERIMENT = 700
# Per-trial evaluation caps apply to both synthetic and real-life benchmarks.
ALGORITHM_EVALUATION_CAPS = {"SAASBO": 200, "ALEBO": 200, "GTBO": 200}
NASBENCH101_ALGORITHM_EVALUATION_CAPS = {
    "AG-VS-BO": 200,
    "AG-VS-TuRBO": 200,
    "TuRBO": 200,
    "LassoBO": 200,
    "LA-MCTS-BO": 200,
    "LA-MCTS-TuRBO": 200,
    "GTBO": 200,
    "HeSBO": 200,
    "ALEBO": 200,
}
NASBENCH101_PLOT_MAX_EVALUATION = 200
NASBENCH101_VALUE_Y_LIMITS = (0.934, 0.940)
ROVER60_PLOT_MAX_EVALUATION = 500
ROVER60_ALGORITHM_EVALUATION_CAPS = {
    "AG-VS-BO": 500,
    "AG-VS-TuRBO": 500,
    "TuRBO": 500,
    "MCTS-VS-BO": 500,
    "MCTS-VS-TuRBO": 500,
}
WALKER2D_ALGORITHM_EVALUATION_CAPS = {
    "GTBO": 150,
    "HeSBO": 150,
    "ALEBO": 150,
}
D_ORIG = 300
D_EMBEDDING = 6
FIRST_N_TRIAL_FIGURES = 5

# AG-VS controls. Exactly one of Q_BATCH/BATCH_SIZE must be None per family.
SYNTHETIC_Q_BATCH = _env_optional_positive_int("AGVS_SYNTHETIC_Q_BATCH", None)
SYNTHETIC_BATCH_SIZE = _env_optional_positive_int("AGVS_SYNTHETIC_BATCH_SIZE", 3)
SYNTHETIC_T_SEQ = _env_optional_positive_int("AGVS_SYNTHETIC_T_SEQ", 2)
SYNTHETIC_N_BAD_MULTIPLIER = _env_optional_positive_int(
    "AGVS_SYNTHETIC_N_BAD_MULTIPLIER", 4
)
SYNTHETIC_N_BAD_ENABLED = _env_bool("AGVS_SYNTHETIC_N_BAD_ENABLED", True)
SYNTHETIC_N_BAD_ABSOLUTE = _env_optional_positive_int(
    "AGVS_SYNTHETIC_N_BAD_ABSOLUTE", None
)
# Dedicated comparison runners may set this before spawning workers. Ordinary
# benchmark runs leave it unset and retain canonical merging.
AGVS_MERGE_ENABLED_OVERRIDE = _env_optional_bool("AGVS_MERGE_ENABLED")
AGVS_GP_NOISE_VARIANCE_OVERRIDE = _env_optional_nonnegative_float(
    "AGVS_GP_NOISE_VARIANCE"
)

REAL_LIFE_Q_BATCH = _env_optional_positive_int("AGVS_REAL_LIFE_Q_BATCH", None)
REAL_LIFE_BATCH_SIZE = _env_optional_positive_int("AGVS_REAL_LIFE_BATCH_SIZE", 3)
REAL_LIFE_T_SEQ = _env_optional_positive_int("AGVS_REAL_LIFE_T_SEQ", 2)
REAL_LIFE_N_BAD_MULTIPLIER = _env_optional_positive_int(
    "AGVS_REAL_LIFE_N_BAD_MULTIPLIER", 4
)
REAL_LIFE_N_BAD_ENABLED = _env_bool("AGVS_REAL_LIFE_N_BAD_ENABLED", True)
# Optional sensitivity-study override. Ordinary experiments leave this unset
# and continue to use N_bad = multiplier * N_split.
REAL_LIFE_N_BAD_ABSOLUTE = _env_optional_positive_int(
    "AGVS_REAL_LIFE_N_BAD_ABSOLUTE", None
)

# AG-VS optimizer settings.
AGVS_ACQUISITION_SAMPLING_STRATEGY = "sobol"
AGVS_ABS_MODE = True
AGVS_TURBO_N_INIT = 1
AGVS_TURBO_MAX_EVALS = 50
AGVS_TURBO_USE_ARD = False
AGVS_TURBO_MAX_CHOLESKY_SIZE = 2000
AGVS_TURBO_N_TRAINING_STEPS = 50
AGVS_TURBO_MIN_CUDA = 0
AGVS_TURBO_DTYPE = "float32"

# Reference TuRBO settings.
TURBO_N_INIT = 20
TURBO_BATCH_SIZE = 10
TURBO_USE_ARD = True
TURBO_MAX_CHOLESKY_SIZE = 2000
TURBO_N_TRAINING_STEPS = 50
TURBO_MIN_CUDA = 0
TURBO_DTYPE = "float32"
TURBO_VERBOSE = True

# MCTS-VS settings.
MCTS_FEATURE_BATCH_SIZE = 2
MCTS_SAMPLE_BATCH_SIZE = 3
MCTS_CP = 0.1
MCTS_K = 20
MCTS_TURBO_N_INIT = 1
MCTS_TURBO_MAX_EVALS = 50
MCTS_TURBO_BATCH_SIZE = 10
MCTS_TURBO_USE_ARD = True
MCTS_TURBO_MAX_CHOLESKY_SIZE = 2000
MCTS_TURBO_N_TRAINING_STEPS = 50
MCTS_TURBO_MIN_CUDA = 0
MCTS_TURBO_DTYPE = "float32"
MCTS_PAPER_BO_N_INIT = 10
MCTS_PAPER_BO_BATCH_SIZE = 3

# Shared acquisition defaults.
STANDARD_BATCH_ACQUISITION = "ei"

# BO, stopping, figures, and diagnostics.
TURBO_GP_ASSUMED_NOISE_VAR = 1e-4
REFERENCE_DEFAULT_GP_NOISE_VAR = 1e-4
RISK_TOLERANCES = [1 / 256]
POSTERIOR_STOPPING_THRESHOLD = 1e100
POSTERIOR_STOPPING_ENABLED = False
PLOT_CONFIDENCE_ERROR_BARS = True
# At most 70 shared CI slots are used.  On short real-life axes the grid grows
# just enough to give every plotted algorithm at least two staggered whiskers;
# on long axes adjacent slots remain at least 10 x-axis units apart.  Algorithms
# occupy different slots, so their vertical 95% CI bars do not overlap.
PLOT_CONFIDENCE_GRID_DIVISIONS = 70
PLOT_CONFIDENCE_MIN_GRID_STEP = 10.0
PLOT_CONFIDENCE_MIN_ROUNDS_PER_ALGORITHM = 2
PLOT_CONFIDENCE_ERROR_BAR_ALPHA = 0.65
PLOT_POSTERIOR_MEAN_DIAGNOSTICS = False
PLOT_TERMINAL_DIAGNOSTICS = False

AGVS_VERBOSE = _env_bool("AGVS_VERBOSE", False)
MCTS_TREE_SHOW = False
AGVS_SHOW_FINAL_TREE = _env_bool("AGVS_SHOW_FINAL_TREE", True)
FIGURE_MODE = True
AGVS_PYTHON_LIVE_PLOT_ON = _MPL_LIVE_PLOT_ON
AGVS_TREE_OPEN_BROWSER = _env_bool("AGVS_TREE_OPEN_BROWSER", False)

MCTS_TREE_PLOT_PATH = str(RESULT_FIGURE_ROOT / "bo" / "Live" / "mcts_vs_live_tree.png")
MCTS_TREE_PLOT_HTML_PATH = str(RESULT_FIGURE_ROOT / "bo" / "Live" / "mcts_vs_live_tree.html")
MCTS_TURBO_TREE_PLOT_PATH = str(RESULT_FIGURE_ROOT / "turbo" / "Live" / "mcts_vs_turbo_live_tree.png")
MCTS_TURBO_TREE_PLOT_HTML_PATH = str(RESULT_FIGURE_ROOT / "turbo" / "Live" / "mcts_vs_turbo_live_tree.html")

# Every synthetic observation receives common-random-number Gaussian noise.
# For a fixed benchmark/trial, every algorithm recreates the same seeded noise
# stream, so evaluation number n receives the same epsilon across algorithms.
BO_GP_ASSUMED_NOISE_STD = BO_REFERENCE_SPEC.observation_noise_std
BO_GP_ASSUMED_NOISE_VAR = BO_REFERENCE_SPEC.alpha
TURBO_GP_ASSUMED_NOISE_STD = TURBO_GP_ASSUMED_NOISE_VAR ** 0.5
STANDARD_NORMAL = NormalDist(mu=0.0, sigma=1.0)

AGVS_BO_ALGORITHMS = {"AG-VS-BO"}
AGVS_TURBO_ALGORITHMS = {"AG-VS-TuRBO"}
AGVS_ALGORITHMS = AGVS_BO_ALGORITHMS | AGVS_TURBO_ALGORITHMS
AGVS_VARIANT_CONFIG = {
    "AG-VS-BO": {"node_backend": "vanillabo"},
    "AG-VS-TuRBO": {"node_backend": "turbo"},
}
SYNTHETIC_ALGORITHM_ORDER = list(SYNTHETIC_ALGORITHMS)
# Keep the complete scientific roster above for reports and explicit historical
# runs. The bare runner uses only algorithms available in the normal runtime.
DEFAULT_ALGORITHMS = [
    "AG-VS-BO",
    # AG-VS-TuRBO remains available to real-life runners and historical reports.
    "MCTS-VS-BO",
    "MCTS-VS-TuRBO",
    # "MCTS-VS-SAASBO",  # blocked: exact JAX/NumPyro versions are unavailable
    # "ADD-GP-UCB",      # excluded from the active experiment by user request
    "TuRBO",
    # "VanillaBO",       # excluded from the active experiment by user request
    # "CMA-ES",          # blocked here: requires cma 3.1.0 and NumPy 1.x
    "Dropout-BO",
    "Dropout-TuRBO",
    "LA-MCTS-TuRBO",
    "VAE-BO",
    # LassoBO, GTBO, SAASBO, HeSBO, and ALEBO run in the real-life roster only.
    # "RandomSearch",    # excluded from the active experiment by user request
]
NONDEFAULT_REFERENCE_ALGORITHMS = frozenset(
    (set(SYNTHETIC_ALGORITHM_ORDER) - set(DEFAULT_ALGORITHMS))
    | {
        "MCTS-VS-RS",
        "Dropout-RS",
        "ADD-GP-UCB",
        "VanillaBO",
        "RandomSearch",
        "LA-MCTS-BO",
    }
)
RUNNER_ALGORITHM_CHOICES = tuple(
    dict.fromkeys(
        [
            *SYNTHETIC_ALGORITHM_ORDER,
            *REAL_LIFE_ALGORITHMS,
            "ADD-GP-UCB",
            "VanillaBO",
            "MCTS-VS-RS",
            "RandomSearch",
            "Dropout-RS",
            "LA-MCTS-BO",
            "LA-MCTS-TuRBO",
        ]
    )
)
# Hartmann remains registered only so historical traces and objective metadata
# stay readable. It is excluded from the runnable synthetic benchmark scope.
DEFERRED_SYNTHETIC_BENCHMARKS = {
    name for name in SYNTHETIC_EXPERIMENTS if name.startswith("hartmann")
}
DEFERRED_BENCHMARKS = {
    *DEFERRED_SYNTHETIC_BENCHMARKS,
    *(name for name in REAL_LIFE_EXPERIMENTS if name.startswith("rover60_")),
}
REAL_LIFE_BENCHMARKS = [
    name for name in REAL_LIFE_EXPERIMENTS if name not in DEFERRED_BENCHMARKS
]

# Active, runnable synthetic experiments. Registered Hartmann definitions are
# intentionally retained outside this list for historical-result compatibility.
SYNTHETIC_BENCHMARKS = [
    name
    for name in SYNTHETIC_EXPERIMENTS
    if name not in DEFERRED_SYNTHETIC_BENCHMARKS
]

DEFAULT_BENCHMARKS = [
    *SYNTHETIC_BENCHMARKS,
    # Real-life tasks require external datasets, packages, or simulators and
    # therefore remain explicit via --benchmarks or --all-benchmarks.
    # *REAL_LIFE_BENCHMARKS,
]

# Optional user-editable defaults for running this module without CLI selectors.
# Keep either value as None to use the project defaults above. CLI
# --benchmarks/--algorithms always take precedence.
RUNNER_BENCHMARKS = None  # Example: ("ackley3_300", "sphere6_1000")
RUNNER_ALGORITHMS = None  # Example: ("AG-VS-BO", "MCTS-VS-BO")

REAL_LIFE_BENCHMARK_DEFAULTS = {
    name: {"max_iter": experiment.budget, "trials": experiment.trials}
    for name, experiment in REAL_LIFE_EXPERIMENTS.items()
}

def seed_bundle_for_config(config, trial_id, *, base_seed_offset=0):
    """Consume the central experiment's ordered seed bundle for one trial."""

    factory = config.get("seed_bundle_factory") if isinstance(config, dict) else None
    if not callable(factory):
        experiment = config.get("experiment_spec") if isinstance(config, dict) else None
        factory = getattr(experiment, "seeds", None)
    if not callable(factory):
        raise ValueError("Benchmark config must expose its central seed_bundle_factory")
    bundle = factory(int(trial_id), base_seed_offset=int(base_seed_offset))
    configured_effective_seed = config.get("effective_dimension_seed")
    if bundle.effective_dimension_seed != configured_effective_seed:
        raise ValueError(
            "Seed bundle effective_dimension_seed does not match benchmark identity: "
            f"bundle={bundle.effective_dimension_seed}, "
            f"config={configured_effective_seed}"
        )
    return bundle


# Backward-compatible fallbacks for callers without a registered config.
AGVS_Q_BATCH = SYNTHETIC_Q_BATCH
AGVS_BATCH_SIZE = SYNTHETIC_BATCH_SIZE
AGVS_T_SEQ = SYNTHETIC_T_SEQ
AGVS_N_BAD_MULTIPLIER = SYNTHETIC_N_BAD_MULTIPLIER
AGVS_N_BAD_ENABLED = SYNTHETIC_N_BAD_ENABLED
LEGACY_TRACE_BATCH_FALLBACK = max(1, int(D_EMBEDDING / 2))
AGVS_N_INIT = D_EMBEDDING * 2
# False: mean(y_obs_i - historical_best_y_obs).
# True: mean(abs(y_obs_i - historical_best_y_obs)); no denominator or x100 scaling.


def add_n_bad_cli_flags(parser):
    """Add Python-3.8-compatible AG-VS restart override flags."""
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--n-bad-enabled",
        dest="n_bad_enabled",
        action="store_true",
        default=None,
        help="Enable AG-VS N_bad tree reinitialization for this run.",
    )
    group.add_argument(
        "--no-n-bad-enabled",
        dest="n_bad_enabled",
        action="store_false",
        help="Disable AG-VS N_bad tree reinitialization for this run.",
    )


TURBO_BACKEND_ALGORITHMS = frozenset(
    {
        "AG-VS-TuRBO",
        "MCTS-VS-TuRBO",
        "TuRBO",
        "Dropout-TuRBO",
        "LA-MCTS-TuRBO",
    }
)


def algorithm_device(algorithm, runner_device=DEVICE):
    """Use one common CPU device for fair cross-algorithm timing."""

    del algorithm, runner_device
    return "cpu"


def algorithm_uses_gpu_worker_pool(algorithm, runner_device=DEVICE):
    """GPU phases are disabled by the fair CPU-only experiment policy."""

    del algorithm, runner_device
    return False


def algorithm_worker_limit(algorithm, requested_workers, runner_device=DEVICE):
    """Use the runner-requested worker count without algorithm-specific caps."""

    del algorithm, runner_device
    return max(1, int(requested_workers))


def preflight_algorithm_devices(algorithms):
    """CPU-only runs require no CUDA preflight."""

    del algorithms

def _validated_agvs_batch_policy(experiment_kind):
    """Return the one runner-owned AG-VS policy for an experiment family."""

    kind = str(experiment_kind).strip().lower().replace("-", "_")
    if kind == "synthetic":
        q_batch = SYNTHETIC_Q_BATCH
        batch_size = SYNTHETIC_BATCH_SIZE
        t_seq = SYNTHETIC_T_SEQ
        n_bad_multiplier = SYNTHETIC_N_BAD_MULTIPLIER
        n_bad_enabled = SYNTHETIC_N_BAD_ENABLED
    elif kind == "real_life":
        q_batch = REAL_LIFE_Q_BATCH
        batch_size = REAL_LIFE_BATCH_SIZE
        t_seq = REAL_LIFE_T_SEQ
        n_bad_multiplier = REAL_LIFE_N_BAD_MULTIPLIER
        n_bad_enabled = REAL_LIFE_N_BAD_ENABLED
    else:
        raise ValueError(f"Unknown experiment kind for AG-VS batch policy: {kind!r}")

    if (q_batch is None) == (batch_size is None):
        raise ValueError(
            f"{kind}: exactly one of Q_BATCH and BATCH_SIZE must be an integer; "
            f"got q_batch={q_batch!r}, batch_size={batch_size!r}"
        )
    active_value = q_batch if q_batch is not None else batch_size
    if isinstance(active_value, bool) or int(active_value) <= 0:
        raise ValueError(f"{kind}: active AG-VS batch value must be a positive integer")
    if isinstance(t_seq, bool) or int(t_seq) <= 0:
        raise ValueError(f"{kind}: T_SEQ must be a positive integer")
    if isinstance(n_bad_multiplier, bool) or int(n_bad_multiplier) <= 0:
        raise ValueError(f"{kind}: N_BAD_MULTIPLIER must be a positive integer")
    return (
        None if q_batch is None else int(q_batch),
        None if batch_size is None else int(batch_size),
        int(t_seq),
        int(n_bad_multiplier),
        bool(n_bad_enabled),
    )


def apply_runner_agvs_batch_policy(config):
    """Copy a benchmark config and apply the centralized runner controls."""

    configured = dict(config)
    kind = str(configured.get("experiment_kind", "synthetic"))
    q_batch, batch_size, t_seq, n_bad_multiplier, n_bad_enabled = (
        _validated_agvs_batch_policy(kind)
    )
    configured.update(
        {
            "q_batch": q_batch,
            "batch_size": batch_size,
            "selection_batch_size": batch_size,
            "t_seq": t_seq,
            "n_bad_multiplier": n_bad_multiplier,
            "n_bad_enabled": n_bad_enabled,
            "agvs_batch_settings_source": "Experiment.benchmark.runner_synthetic central controls",
        }
    )
    normalized_kind = kind.strip().lower().replace("-", "_")
    n_bad_absolute = (
        SYNTHETIC_N_BAD_ABSOLUTE
        if normalized_kind == "synthetic"
        else REAL_LIFE_N_BAD_ABSOLUTE
        if normalized_kind == "real_life"
        else None
    )
    if n_bad_absolute is not None:
        configured.update(
            {
                "n_bad_multiplier": int(n_bad_absolute),
                "n_bad_is_absolute": True,
                "agvs_n_bad_settings_source": (
                    f"AGVS_{normalized_kind.upper()}_N_BAD_ABSOLUTE sensitivity override"
                ),
            }
        )
    if AGVS_MERGE_ENABLED_OVERRIDE is not None:
        configured["agvs_merge_enabled"] = bool(AGVS_MERGE_ENABLED_OVERRIDE)
    if AGVS_GP_NOISE_VARIANCE_OVERRIDE is not None:
        configured["agvs_gp_noise_variance"] = float(
            AGVS_GP_NOISE_VARIANCE_OVERRIDE
        )
    return configured


def mcts_cp_for_benchmark(benchmark_name):
    """Appendix C, Table 3: Levy=10, Hartmann/NAS=0.1, MuJoCo=50."""
    name = str(benchmark_name or "").lower()
    if name.startswith("levy"):
        return 10.0
    if name.startswith(("hopper", "walker", "mujoco")):
        return 50.0
    return 0.1


def cma_sigma_for_benchmark(benchmark_name):
    """Appendix C CMA-ES step sizes; retain 0.8 for unlisted synthetic tasks."""
    name = str(benchmark_name or "").lower()
    if name.startswith("levy"):
        return 10.0
    if name.startswith("hartmann"):
        return 0.8
    if name.startswith("nas"):
        return 0.1
    if name.startswith(("hopper", "walker", "mujoco")):
        return 0.01
    return 0.8


def vae_update_interval_for_benchmark(benchmark_name):
    """Return the published update cadence for paper real-life conditions."""

    name = str(benchmark_name or "").lower()
    return 30 if name.startswith("hartmann") or name in REAL_LIFE_EXPERIMENTS else 20


def vae_learning_rate_for_benchmark(benchmark_name):
    """Return the published real-life VAE-BO rate without changing synthetic runs."""

    name = str(benchmark_name or "").lower()
    return 0.01 if name in REAL_LIFE_EXPERIMENTS else 0.001


def reference_runtime_metadata(algorithm, benchmark_name=None):
    """Describe exact historical-runtime blockers without substituting methods."""

    algorithm = canonical_algorithm_name(algorithm)
    pinned = (
        "reference target Python 3.8.8 (validated compatibility runtime "
        "Python 3.8.20 x86_64); PyTorch 1.10.1; BoTorch 0.5.1; GPyTorch 1.5.1; "
        "Ax 0.2.2; NumPy 1.21.6; SciPy 1.7.3; scikit-learn 1.0.2"
    )
    benchmark_target = str(benchmark_name or "hartmann3_300").lower()
    launchers = {
        "ADD-GP-UCB": (
            f"python3 -m Experiment.benchmark.runner_synthetic --benchmarks {benchmark_target} "
            "--algorithms ADD-GP-UCB --trials 1 --workers 1 --skip-plot"
        ),
        "HeSBO": (
            "python -m Experiment.benchmark.runner_synthetic --benchmarks "
            f"{benchmark_target} --algorithms HeSBO --trials 1 --workers 1 --skip-plot"
        ),
        "ALEBO": (
            "python -m Experiment.benchmark.runner_synthetic --benchmarks "
            f"{benchmark_target} --algorithms ALEBO --trials 1 --workers 1 --skip-plot"
        ),
        "LA-MCTS-TuRBO": (
            f"python3 -m Experiment.benchmark.runner_synthetic --benchmarks {benchmark_target} "
            "--algorithms LA-MCTS-TuRBO --trials 1 --workers 1 --skip-plot"
        ),
        "LA-MCTS-BO": (
            f"python3 -m Experiment.benchmark.runner_synthetic --benchmarks {benchmark_target} "
            "--algorithms LA-MCTS-BO --trials 1 --workers 1 --skip-plot"
        ),
        "CMA-ES": (
            "python -m Experiment.benchmark.runner_synthetic --benchmarks "
            f"{benchmark_target} --algorithms CMA-ES --trials 1 --workers 1 --skip-plot"
        ),
        "SAASBO": (
            "python -m Experiment.benchmark.runner_synthetic --benchmarks "
            f"{benchmark_target} --algorithms SAASBO --trials 1 --workers 1 --skip-plot"
        ),
        "MCTS-VS-SAASBO": (
            "python -m Experiment.benchmark.runner_synthetic --benchmarks "
            f"{benchmark_target} --algorithms MCTS-VS-SAASBO --trials 1 --workers 1 "
            "--skip-plot"
        ),
    }
    if algorithm in {"HeSBO", "ALEBO", "GTBO", "LA-MCTS-BO", "LA-MCTS-TuRBO"}:
        return {"runtime_status": "READY", "reference_environment": "native main Python environment",
                "launch_command": f"python -m Experiment.benchmark.runner_synthetic --benchmarks {benchmark_target} --algorithms {algorithm} --trials 1 --workers 1 --skip-plot"}
    if algorithm == "ADD-GP-UCB":
        direct_available = getattr(
            importlib.import_module("scipy.optimize"),
            "direct",
            None,
        ) is not None
        return {
            "runtime_status": (
                "READY"
                if direct_available
                else "BLOCKED_REFERENCE_ENVIRONMENT: requires scipy.optimize.direct (SciPy >= 1.9)"
            ),
            "reference_environment": (
                "Python environment with SciPy >= 1.9 for the documented Python "
                "translation; the retained upstream implementation is MATLAB"
            ),
            "launch_command": launchers[algorithm],
        }
    if algorithm in {"SAASBO", "MCTS-VS-SAASBO"}:
        missing = [
            package
            for package in ("jax", "numpyro")
            if importlib.util.find_spec(package) is None
        ]
        return {
            "runtime_status": (
                "READY"
                if not missing
                else "BLOCKED_REFERENCE_ENVIRONMENT: missing " + ", ".join(missing)
            ),
            "reference_environment": (
                "native main Python environment; JAX and NumPyro"
            ),
            "launch_command": launchers[algorithm],
        }
    if algorithm == "CMA-ES":
        try:
            cma_version = importlib.metadata.version("cma")
        except importlib.metadata.PackageNotFoundError:
            cma_version = "not installed"
        compatible = cma_version == "3.1.0" and int(np.__version__.split(".", 1)[0]) < 2
        return {
            "runtime_status": (
                "READY"
                if compatible
                else (
                    "BLOCKED_REFERENCE_ENVIRONMENT: requires cma==3.1.0 with "
                    f"NumPy 1.x; found cma={cma_version}, NumPy={np.__version__}"
                )
            ),
            "reference_environment": pinned + "; cma 3.1.0",
            "launch_command": launchers[algorithm],
        }
    return {"runtime_status": "READY"}


def preflight_reference_runtime(algorithms, benchmark_name=None):
    """Fail once, before evaluations, for exact methods unavailable in-process."""

    blocked = {}
    for algorithm in normalize_algorithm_args(algorithms):
        metadata = reference_runtime_metadata(algorithm, benchmark_name=benchmark_name)
        if str(metadata.get("runtime_status", "")).startswith(
            "BLOCKED_REFERENCE_ENVIRONMENT"
        ):
            blocked[algorithm] = metadata
    if not blocked:
        return
    lines = [
        "Exact reference baselines are unavailable in this runtime; no objective "
        "evaluations were started."
    ]
    for algorithm, metadata in blocked.items():
        lines.append(
            f"- {algorithm}: {metadata['runtime_status']}; "
            f"reference={metadata.get('reference_environment', 'not explicitly specified')}; "
            f"launcher={metadata.get('launch_command', 'not explicitly specified')}"
        )
    lines.append(
        "Run the recorded historical environment/launcher, or explicitly select "
        "only READY algorithms with --algorithms. No approximation will be used."
    )
    raise RuntimeError("\n".join(lines))


def preflight_real_life_config(config):
    """Check the physical backend before scheduling MOPTA trials."""
    if str(config.get("experiment_kind", "synthetic")) != "real_life":
        return None
    if config.get("real_life_task") != "mopta08":
        raise ValueError(f"Unknown real-life task {config.get('real_life_task')!r}")
    from Experiment.benchmark.real_life.mopta08_backend import preflight_mopta08_backend
    return preflight_mopta08_backend()


def benchmark_effective_dim(config):
    if isinstance(config, dict) and config.get("effective_dim") is not None:
        return int(config["effective_dim"])
    if isinstance(config, dict) and config.get("active_indices") is not None:
        return int(len(np.asarray(config["active_indices"], dtype=int).reshape(-1)))
    return int(D_EMBEDDING)


def benchmark_agvs_effective_dim(config):
    if isinstance(config, dict) and config.get("agvs_effective_dim") is not None:
        value = config.get("agvs_effective_dim")
        if isinstance(value, bool) or int(value) <= 0:
            raise ValueError("agvs_effective_dim must be a positive integer")
        return int(value)
    return benchmark_effective_dim(config)


def benchmark_agvs_merge_enabled(config):
    """Return whether AG-VS sibling cohorts are merged after scoring.

    The default preserves the canonical AG-VS behavior.  Dedicated ablations
    can disable merging while retaining ordinary score-based child selection
    and recursive splitting.
    """

    if not isinstance(config, dict) or "agvs_merge_enabled" not in config:
        return True
    value = config.get("agvs_merge_enabled")
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError("agvs_merge_enabled must be a boolean")
    return bool(value)


def real_life_experiment_for_config(config, benchmark_name=None):
    """Resolve a registered real-life identity without inferring an active mask."""

    if not isinstance(config, dict) or str(config.get("experiment_kind", "synthetic")) != "real_life":
        return None
    embedded_spec = config.get("experiment_spec")
    if embedded_spec is not None and hasattr(embedded_spec, "task"):
        return embedded_spec
    identifier = config.get("real_life_variant") or benchmark_name
    if identifier is None:
        raise ValueError("Real-life benchmark config is missing its registry identity")
    try:
        return get_real_life_experiment(str(identifier))
    except KeyError as exc:
        raise ValueError(str(exc)) from exc


def validate_algorithm_applicability(algorithm, config, benchmark_name=None):
    """Reject methods outside a real task's explicit reference comparison."""

    algorithm = canonical_algorithm_name(algorithm)
    experiment = real_life_experiment_for_config(config, benchmark_name)
    if experiment is None:
        return algorithm
    if not real_life_algorithm_is_applicable(experiment, algorithm):
        allowed = iter_applicable_real_life_algorithms(experiment)
        raise ValueError(
            f"{algorithm} is not applicable to reference real-life condition "
            f"{experiment.key}; applicable algorithms={allowed}"
        )
    return algorithm


def optimizer_active_dimension(algorithm, config, benchmark_name=None):
    """Resolve algorithm search dimension while keeping real physical D intact."""

    algorithm = canonical_algorithm_name(algorithm)
    experiment = real_life_experiment_for_config(config, benchmark_name)
    if experiment is None:
        return benchmark_effective_dim(config)
    validate_algorithm_applicability(algorithm, config, benchmark_name)
    return real_life_optimizer_dimension(experiment, algorithm)


def default_max_iter_for_effective_dim(effective_dim):
    """Compatibility fallback; registered experiments own their exact budget."""
    del effective_dim
    return int(MAX_EXPERIMENT)


def algorithm_evaluation_budget(algorithm, max_iter, benchmark_name=None):
    """Cap a canonical algorithm's budget without increasing smaller requests."""
    requested = int(max_iter)
    algorithm = canonical_algorithm_name(algorithm)
    cap = ALGORITHM_EVALUATION_CAPS.get(algorithm, requested)
    benchmark = str(benchmark_name or "").strip().lower()
    experiment = REAL_LIFE_EXPERIMENTS.get(benchmark)
    if experiment is not None and experiment.task.key == "nasbench101":
        cap = NASBENCH101_ALGORITHM_EVALUATION_CAPS.get(algorithm, cap)
    elif experiment is not None and experiment.task.key == "walker2d":
        cap = WALKER2D_ALGORITHM_EVALUATION_CAPS.get(algorithm, cap)
    elif experiment is not None and experiment.task.key == "rover60":
        cap = ROVER60_ALGORITHM_EVALUATION_CAPS.get(algorithm, cap)
    return min(requested, int(cap))


def benchmark_budget_by_name(benchmark_name, fallback=MAX_EXPERIMENT):
    config = BENCHMARKS.get(str(benchmark_name or "").lower())
    if config is None:
        return int(fallback)
    return int(config.get("max_iter", fallback))


def effective_dim_from_benchmark_name(name, fallback=D_EMBEDDING):
    """Extract d_eff from names such as rosenbrock10_1000."""
    match = re.fullmatch(r".+?(\d+)_\d+", str(name).lower())
    return int(match.group(1)) if match is not None else int(fallback)


def benchmark_n_init(config):
    if isinstance(config, dict) and config.get("agvs_n_init") is not None:
        value = config.get("agvs_n_init")
        if isinstance(value, bool) or int(value) <= 0:
            raise ValueError("agvs_n_init must be a positive integer")
        return int(value)
    return int(benchmark_agvs_effective_dim(config) * 2)


def benchmark_agvs_gp_noise_variance(config):
    """Resolve the AG-VS surrogate variance from the benchmark configuration."""
    default = float(BO_REFERENCE_SPEC.alpha)
    if not isinstance(config, dict):
        return default

    value = None
    for key in ("agvs_gp_noise_variance", "gp_noise_variance"):
        if config.get(key) is not None:
            value = config[key]
            break

    experiment_kind = str(config.get("experiment_kind", "synthetic")).strip().lower()
    if value is None and experiment_kind == "synthetic":
        if config.get("benchmark_noise_variance") is not None:
            value = config["benchmark_noise_variance"]
        elif config.get("benchmark_noise_std") is not None:
            value = float(config["benchmark_noise_std"]) ** 2

    if value is None:
        return default
    try:
        variance = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("AG-VS GP noise variance must be finite and non-negative") from exc
    if not np.isfinite(variance) or variance < 0.0:
        raise ValueError("AG-VS GP noise variance must be finite and non-negative")
    return variance


def benchmark_q_batch(config):
    if isinstance(config, dict) and "q_batch" in config:
        value = config.get("q_batch")
        return None if value is None else int(value)
    return AGVS_Q_BATCH


def benchmark_batch_size(config):
    if isinstance(config, dict):
        if "batch_size" in config:
            value = config.get("batch_size")
            return None if value is None else int(value)
        if "selection_batch_size" in config:
            value = config.get("selection_batch_size")
            return None if value is None else int(value)
    return AGVS_BATCH_SIZE


def benchmark_q_batch_by_name(benchmark_name, fallback=AGVS_Q_BATCH):
    """Resolve the AG-VS q-batch policy from a registered benchmark."""
    config = BENCHMARKS.get(str(benchmark_name or "").lower())
    if config is not None:
        return benchmark_q_batch(config)
    return None if fallback is None else int(fallback)


def benchmark_batch_size_by_name(benchmark_name, fallback=AGVS_BATCH_SIZE):
    config = BENCHMARKS.get(str(benchmark_name or "").lower())
    if config is not None:
        return benchmark_batch_size(config)
    del benchmark_name
    return None if fallback is None else int(fallback)






def validate_active_indices_for_config(config, benchmark_name="benchmark"):
    if not isinstance(config, dict) or config.get("active_indices") is None:
        return
    bounds = np.asarray(config["bounds"], dtype=float)
    original_dim = int(bounds.shape[0])
    active = np.asarray(config["active_indices"], dtype=int).reshape(-1)
    if active.size == 0:
        raise ValueError(f"{benchmark_name}: active_indices must contain at least one dimension.")
    invalid = [int(dim) for dim in active if int(dim) < 0 or int(dim) >= original_dim]
    if invalid:
        raise ValueError(
            f"{benchmark_name}: active_indices must be valid extrinsic dimension indices "
            f"in [0, {original_dim - 1}]. Invalid active_indices={invalid}."
        )


# Runtime registry.  Synthetic configurations are materialised from the central
# immutable identity registry at import; real-life wrappers register lazily.
BENCHMARKS = {
    name: apply_runner_agvs_batch_policy(spec)
    for name, spec in make_all_synthetic_configs().items()
}

def register_benchmark(name, spec):
    """Register a benchmark specification (bounds, objective, etc.) at runtime."""
    key = str(name).lower()
    spec = apply_runner_agvs_batch_policy(spec)
    validate_active_indices_for_config(spec, benchmark_name=key)
    if key in SYNTHETIC_EXPERIMENTS:
        registered = SYNTHETIC_EXPERIMENTS[key]
        active = tuple(int(value) for value in np.asarray(spec["active_indices"]).reshape(-1))
        if active != registered.effective_dimension_indices:
            raise ValueError(
                f"{key}: active_indices must come from the central registry; "
                f"expected={registered.effective_dimension_indices}, got={active}"
            )
        if int(np.asarray(spec["bounds"]).shape[0]) != registered.ambient_dim:
            raise ValueError(f"{key}: D does not match the central registry")
        if benchmark_effective_dim(spec) != registered.d_eff:
            raise ValueError(f"{key}: d_eff does not match the central registry")
    elif key in REAL_LIFE_EXPERIMENTS:
        registered = REAL_LIFE_EXPERIMENTS[key]
        if spec.get("active_indices") is not None:
            raise ValueError(
                f"{key}: real-life configured d_eff is an optimizer condition; "
                "a ground-truth active_indices mask must not be fabricated"
            )
        if int(np.asarray(spec["bounds"]).shape[0]) != registered.physical_dim:
            raise ValueError(f"{key}: physical D does not match the central registry")
        if benchmark_effective_dim(spec) != registered.configured_d_eff:
            raise ValueError(
                f"{key}: configured d_eff does not match the central registry"
            )
        if spec.get("effective_dimension_seed") is not None:
            raise ValueError(
                f"{key}: effective_dimension_seed must be None without an active-subset protocol"
            )
        if not callable(spec.get("seed_bundle_factory")):
            raise ValueError(f"{key}: missing central seed_bundle_factory")
        expected_fields = {
            "experiment_kind": "real_life",
            "experiment_id": registered.experiment_id,
            "real_life_task": registered.task.key,
            "real_life_variant": registered.key,
            "physical_dim": registered.physical_dim,
            "configured_d_eff": registered.configured_d_eff,
            "max_iter": registered.budget,
            "metric_name": registered.task.metric_name,
        }
        if registered.task.key == "nasbenchasr":
            expected_fields["metric_name"] = "negative_timit_validation_per"
        mismatches = {
            field: {"expected": expected, "got": spec.get(field)}
            for field, expected in expected_fields.items()
            if spec.get(field) != expected
        }
        expected_bounds = np.repeat(
            [[float(registered.task.bounds[0]), float(registered.task.bounds[1])]],
            registered.physical_dim,
            axis=0,
        )
        if not np.array_equal(np.asarray(spec["bounds"], dtype=float), expected_bounds):
            mismatches["bounds"] = {
                "expected": expected_bounds.tolist(),
                "got": np.asarray(spec["bounds"], dtype=float).tolist(),
            }
        embedded = spec.get("experiment_spec")
        if embedded is None or getattr(embedded, "key", None) != registered.key:
            mismatches["experiment_spec"] = {
                "expected": registered.key,
                "got": getattr(embedded, "key", None),
            }
        objective_factory = spec.get("objective_factory")
        if getattr(objective_factory, "task", None) != registered.task.key:
            mismatches["objective_factory.task"] = {
                "expected": registered.task.key,
                "got": getattr(objective_factory, "task", None),
            }
        if mismatches:
            raise ValueError(
                f"{key}: config identity does not match the central registry: "
                f"{mismatches}"
            )
    BENCHMARKS[key] = spec


# =============================================================================
# Recorder
# =============================================================================

class NoisyObjectiveRecorder:
    """Stores both noiseless and noisy observations. Optimizers see noisy score."""

    def __init__(
        self,
        func,
        noise_std=0.0,
        seed=0,
        *,
        benchmark_name=None,
        algorithm=None,
        trial=None,
        max_evals=None,
        progress_every=EVALUATION_PROGRESS_EVERY,
        replay_trace=None,
        replay_x_atol=1e-8,
    ):
        self.func = func
        self.noise_std = float(noise_std)
        self.rng = np.random.default_rng(seed)
        self.benchmark_name = None if benchmark_name is None else str(benchmark_name)
        self.algorithm = None if algorithm is None else str(algorithm)
        self.trial = None if trial is None else int(trial)
        self.max_evals = None if max_evals is None else int(max_evals)
        self.progress_every = max(1, int(progress_every))
        self.x_history = []
        self.true_history = []
        self.observed_history = []
        self.start_time = None
        self.eval_wall_clock_history = []
        self.replay_x_atol = float(replay_x_atol)
        self.replayed_evaluations = 0
        self._replay_x_history = []
        self._replay_true_history = []
        self._replay_observed_history = []
        self._replay_wall_clock_history = []
        self._replay_base_wall_clock_seconds = 0.0
        self._continuation_start_time = None
        if replay_trace is not None:
            self._configure_replay(replay_trace)

    def _configure_replay(self, trace):
        if not isinstance(trace, dict):
            raise TypeError("replay_trace must be a JSON trace dictionary")
        try:
            count = int(trace.get("n_evaluations", 0) or 0)
        except (TypeError, ValueError) as exc:
            raise ValueError("replay_trace has an invalid n_evaluations") from exc
        if count <= 0:
            raise ValueError("replay_trace must contain at least one evaluation")

        histories = {
            "x_history": trace.get("x_history"),
            "true_y_history": trace.get("true_y_history"),
            "observed_y_history": trace.get("observed_y_history"),
            "evaluation_wall_clock_history": trace.get(
                "evaluation_wall_clock_history"
            ),
        }
        for name, values in histories.items():
            if not isinstance(values, list) or len(values) < count:
                raise ValueError(
                    f"replay_trace {name} must contain at least {count} entries"
                )

        replay_x = [
            np.asarray(row, dtype=float).reshape(-1)
            for row in histories["x_history"][:count]
        ]
        if any(row.size == 0 or not np.all(np.isfinite(row)) for row in replay_x):
            raise ValueError("replay_trace x_history contains an invalid point")
        widths = {int(row.size) for row in replay_x}
        if len(widths) != 1:
            raise ValueError("replay_trace x_history has inconsistent dimensions")

        replay_true = np.asarray(
            histories["true_y_history"][:count], dtype=float
        ).reshape(-1)
        replay_observed = np.asarray(
            histories["observed_y_history"][:count], dtype=float
        ).reshape(-1)
        replay_times = np.asarray(
            histories["evaluation_wall_clock_history"][:count], dtype=float
        ).reshape(-1)
        if not (
            np.all(np.isfinite(replay_true))
            and np.all(np.isfinite(replay_observed))
            and np.all(np.isfinite(replay_times))
        ):
            raise ValueError("replay_trace contains non-finite response or time values")
        if np.any(replay_times < 0.0) or np.any(np.diff(replay_times) < -1e-9):
            raise ValueError(
                "replay_trace evaluation_wall_clock_history must be nonnegative "
                "and nondecreasing"
            )

        try:
            source_wall_clock = float(trace.get("wall_clock_seconds", replay_times[-1]))
        except (TypeError, ValueError) as exc:
            raise ValueError("replay_trace has an invalid wall_clock_seconds") from exc
        if not np.isfinite(source_wall_clock) or source_wall_clock < 0.0:
            raise ValueError("replay_trace wall_clock_seconds must be finite and nonnegative")

        self._replay_x_history = replay_x
        self._replay_true_history = replay_true.tolist()
        self._replay_observed_history = replay_observed.tolist()
        self._replay_wall_clock_history = replay_times.tolist()
        self._replay_base_wall_clock_seconds = max(
            source_wall_clock,
            float(replay_times[-1]),
        )

    @property
    def replay_evaluation_count(self):
        return len(self._replay_observed_history)

    def accept_replay_history_as_warm_start(self):
        """Populate histories for an optimizer-native warm start without calls."""

        if self.replay_evaluation_count <= 0:
            raise ValueError("No replay trace was configured for warm start")
        if self.x_history or self.true_history or self.observed_history:
            raise RuntimeError("Recorder history is already populated")
        self.x_history = [row.copy() for row in self._replay_x_history]
        self.true_history = list(self._replay_true_history)
        self.observed_history = list(self._replay_observed_history)
        self.eval_wall_clock_history = list(self._replay_wall_clock_history)
        self.replayed_evaluations = self.replay_evaluation_count
        if self.noise_std > 0:
            self.rng.normal(
                0.0,
                self.noise_std,
                size=self.replay_evaluation_count,
            )
        if self.start_time is None:
            self.start_time = time.time()
        self._continuation_start_time = time.time()
        terminal_stream = getattr(sys, "__stdout__", None) or sys.stdout
        terminal_stream.write(
            f"[RESUME-WARM-START] benchmark={self.benchmark_name} "
            f"algorithm={self.algorithm} trial={self.trial:03d} "
            f"stored_evals={self.replay_evaluation_count} status=VERIFIED\n"
        )
        terminal_stream.flush()

    def current_wall_clock_seconds(self):
        """Return old recorded time plus only newly executed continuation time."""

        if self.replay_evaluation_count:
            if self.replayed_evaluations < self.replay_evaluation_count:
                return 0.0
            if self._continuation_start_time is None:
                return float(self._replay_base_wall_clock_seconds)
            return float(
                self._replay_base_wall_clock_seconds
                + max(0.0, time.time() - self._continuation_start_time)
            )
        if self.start_time is None:
            return 0.0
        return float(max(0.0, time.time() - self.start_time))

    def report_progress(self, status="RUNNING", force=False):
        if (
            self.benchmark_name is None
            or self.algorithm is None
            or self.trial is None
            or self.max_evals is None
        ):
            return
        n_eval = len(self.observed_history)
        should_print = (
            bool(force)
            or n_eval == 1
            or n_eval >= self.max_evals
            or n_eval % self.progress_every == 0
        )
        if not should_print:
            return
        elapsed = self.current_wall_clock_seconds()
        message = (
            f"[EVAL-PROGRESS] benchmark={self.benchmark_name} "
            f"algorithm={self.algorithm} trial={self.trial:03d} "
            f"eval={n_eval}/{self.max_evals} status={status} "
            f"elapsed={elapsed:.1f}s"
        )
        rover60_progress_values = (
            self.benchmark_name.lower().startswith("rover60_")
            and self.observed_history
        )
        if rover60_progress_values and self.algorithm == "TuRBO":
            message += f" best_value={max(self.observed_history):.6f}"
        elif rover60_progress_values and self.algorithm in {
            "MCTS-VS-BO",
            "MCTS-VS-TuRBO",
        }:
            message += (
                f" current_value={self.observed_history[-1]:.6f}"
                f" best_value={max(self.observed_history):.6f}"
            )
        elif (
            self.benchmark_name.lower().startswith("dna_")
            and self.algorithm in AGVS_ALGORITHMS
            and self.observed_history
        ):
            message += (
                f" current_value={self.observed_history[-1]:.6f}"
                f" best_value={max(self.observed_history):.6f}"
            )
        elif (
            self.benchmark_name.lower().startswith("mopta08_")
            and self.observed_history
        ):
            message += (
                f" current_value={self.observed_history[-1]:.6f}"
                f" best_value={max(self.observed_history):.6f}"
            )
        # MCTS-VS runs inside ``suppress_all_prints`` so the very noisy
        # upstream implementation stays quiet.  Write progress directly to
        # the process' original stdout; otherwise that context also swallows
        # our benchmark-level evaluation counter.
        terminal_stream = getattr(sys, "__stdout__", None) or sys.stdout
        terminal_stream.write(message + "\n")
        terminal_stream.flush()

    def __call__(self, x):
        if hasattr(x, "detach"):
            x = x.detach().cpu().numpy()
        x = np.asarray(x, dtype=float).reshape(-1)
        if self.start_time is None:
            self.start_time = time.time()
        replay_index = len(self.observed_history)
        if replay_index < self.replay_evaluation_count:
            expected_x = self._replay_x_history[replay_index]
            same_shape = x.shape == expected_x.shape
            matches = same_shape and np.allclose(
                x,
                expected_x,
                rtol=0.0,
                atol=self.replay_x_atol,
                equal_nan=False,
            )
            if not matches:
                max_difference = None
                if same_shape:
                    max_difference = float(np.max(np.abs(x - expected_x)))
                raise RuntimeError(
                    "Resume replay diverged from the stored NASBench trial at "
                    f"evaluation {replay_index + 1}: max_abs_difference="
                    f"{max_difference}, tolerance={self.replay_x_atol}. The stored "
                    "JSON was left unchanged."
                )

            # Consume the same synthetic-noise draw so a later continuation
            # keeps the original random stream. Real-life NASBench uses zero
            # recorder noise, but this makes the replay primitive general.
            if self.noise_std > 0:
                self.rng.normal(0.0, self.noise_std)
            self.x_history.append(expected_x.copy())
            self.true_history.append(self._replay_true_history[replay_index])
            self.observed_history.append(self._replay_observed_history[replay_index])
            self.eval_wall_clock_history.append(
                self._replay_wall_clock_history[replay_index]
            )
            self.replayed_evaluations += 1
            if self.replayed_evaluations == 1:
                terminal_stream = getattr(sys, "__stdout__", None) or sys.stdout
                terminal_stream.write(
                    f"[RESUME-REPLAY] benchmark={self.benchmark_name} "
                    f"algorithm={self.algorithm} trial={self.trial:03d} "
                    f"stored_evals={self.replay_evaluation_count} status=START\n"
                )
                terminal_stream.flush()
            if self.replayed_evaluations == self.replay_evaluation_count:
                self._continuation_start_time = time.time()
                terminal_stream = getattr(sys, "__stdout__", None) or sys.stdout
                terminal_stream.write(
                    f"[RESUME-REPLAY] benchmark={self.benchmark_name} "
                    f"algorithm={self.algorithm} trial={self.trial:03d} "
                    f"stored_evals={self.replay_evaluation_count} status=VERIFIED\n"
                )
                terminal_stream.flush()
            return self._replay_observed_history[replay_index]

        true_value = float(self.func(x))
        observed_value = true_value
        if self.noise_std > 0:
            observed_value += float(self.rng.normal(0.0, self.noise_std))

        self.x_history.append(x.copy())
        self.true_history.append(true_value)
        self.observed_history.append(observed_value)
        self.eval_wall_clock_history.append(self.current_wall_clock_seconds())
        self.report_progress(status="RUNNING")
        return observed_value


def close_objective(objective):
    """Close a persistent benchmark objective and return a diagnostic, if any."""

    close = getattr(objective, "close", None)
    if not callable(close):
        return None
    try:
        close()
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"
    return None


def prepare_objective(objective):
    """Eagerly initialize a physical backend and return preparation seconds."""

    prepare = getattr(objective, "prepare", None)
    if not callable(prepare):
        return 0.0
    started = time.time()
    prepare()
    return float(time.time() - started)


# =============================================================================
# Utility functions
# =============================================================================

def canonical_algorithm_name(name):
    key = "-".join(str(name).strip().lower().replace("_", "-").split())
    aliases = {
        "mcts": "MCTS-VS-BO",
        "mcts-vs": "MCTS-VS-BO",
        "mcts-vs-bo": "MCTS-VS-BO",
        "mctsvs": "MCTS-VS-BO",
        "mctsvs-bo": "MCTS-VS-BO",
        "mcts-vs-turbo": "MCTS-VS-TuRBO",
        "mctsvs-turbo": "MCTS-VS-TuRBO",
        "mcts-turbo": "MCTS-VS-TuRBO",
        "mcts-vs-rs": "MCTS-VS-RS",
        "mctsvs-rs": "MCTS-VS-RS",
        "mcts-vs-saasbo": "MCTS-VS-SAASBO",
        "mctsvs-saasbo": "MCTS-VS-SAASBO",
        "add-gp-ucb": "ADD-GP-UCB",
        "addgpucb": "ADD-GP-UCB",
        "add-gp": "ADD-GP-UCB",
        "turbo": "TuRBO",
        "turbo1": "TuRBO",
        "turbo-1": "TuRBO",
        "vanillabo": "VanillaBO",
        "vanilla-bo": "VanillaBO",
        "vanilla": "VanillaBO",
        "bo": "VanillaBO",
        "standard-bo": "VanillaBO",
        "random": "RandomSearch",
        "random-search": "RandomSearch",
        "randomsearch": "RandomSearch",
        "rs": "RandomSearch",
        "cma": "CMA-ES",
        "cma-es": "CMA-ES",
        "cmaes": "CMA-ES",
        "dropout-bo": "Dropout-BO",
        "dropout-turbo": "Dropout-TuRBO",
        "dropout-turbo1": "Dropout-TuRBO",
        "dropout-rs": "Dropout-RS",
        "lassobo": "LassoBO",
        "hoang-lassobo": "LassoBO",
        "lamcts": "LA-MCTS-TuRBO",
        "la-mcts": "LA-MCTS-TuRBO",
        "lamcts-bo": "LA-MCTS-BO",
        "la-mcts-bo": "LA-MCTS-BO",
        "lamcts-turbo": "LA-MCTS-TuRBO",
        "la-mcts-turbo": "LA-MCTS-TuRBO",
        "saasbo": "SAASBO",
        "gtbo": "GTBO",
        "hesbo": "HeSBO",
        "alebo": "ALEBO",
        "vae-bo": "VAE-BO",
        "vaebo": "VAE-BO",
        "ag-vs-bo": "AG-VS-BO",
        "agvs-bo": "AG-VS-BO",
        "ag-vs-turbo": "AG-VS-TuRBO",
        "agvs-turbo": "AG-VS-TuRBO",
    }
    if key not in aliases:
        raise ValueError(f"Unknown algorithm name: {name}")
    return aliases[key]


def algorithm_output_slug(algorithm):
    algorithm = canonical_algorithm_name(algorithm)
    if algorithm == "AG-VS-BO":
        return "ag-vs-bo"
    if algorithm == "AG-VS-TuRBO":
        return "ag-vs-turbo"
    if algorithm == "MCTS-VS-BO":
        return "mcts-vs-bo"
    if algorithm == "MCTS-VS-TuRBO":
        return "mcts-vs-turbo"
    if algorithm == "MCTS-VS-RS":
        return "mcts-vs-rs"
    if algorithm == "MCTS-VS-SAASBO":
        return "mcts-vs-saasbo"
    if algorithm == "ADD-GP-UCB":
        return "add-gp-ucb"
    if algorithm == "RandomSearch":
        return "random-search"
    if algorithm == "CMA-ES":
        return "cma-es"
    if algorithm == "Dropout-TuRBO":
        return "dropout-turbo"
    if algorithm == "LassoBO":
        return "lassobo"
    if algorithm == "LA-MCTS-TuRBO":
        return "la-mcts-turbo"
    if algorithm == "LA-MCTS-BO":
        return "la-mcts-bo"
    if algorithm == "VAE-BO":
        return "vae-bo"
    return algorithm


def normalize_algorithm_args(algorithms):
    return [canonical_algorithm_name(name) for name in algorithms]


def ensure_jsonable(value):
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    try:
        return np.asarray(value, dtype=float).tolist()
    except Exception:
        return None


def make_jsonable(value):
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): make_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_jsonable(v) for v in value]
    return value


def crop_sequence_to_max_iter(values, max_iter):
    if values is None:
        return None
    limit = max(0, int(max_iter))
    if isinstance(values, np.ndarray):
        return values[:limit]
    try:
        return list(values)[:limit]
    except Exception:
        return values


def crop_history_records_to_max_iter(records, max_iter):
    if records is None:
        return None
    limit = max(0, int(max_iter))
    out = []
    for index, record in enumerate(list(records)):
        if isinstance(record, dict) and "iteration" in record:
            try:
                if int(record.get("iteration")) > limit:
                    continue
            except Exception:
                pass
        elif index >= limit:
            break
        out.append(record)
    return out


def crop_threshold_histories_to_max_iter(threshold_by_delta, max_iter):
    if not isinstance(threshold_by_delta, dict):
        return threshold_by_delta
    limit = max(0, int(max_iter))
    return {key: crop_sequence_to_max_iter(values, limit) for key, values in threshold_by_delta.items()}


def crop_trace_payload_to_max_iter(trace, max_iter):
    if not isinstance(trace, dict):
        return trace
    limit = max(0, int(max_iter))
    sequence_keys = [
        "evaluation_wall_clock_history",
        "best_observed_history",
        "posterior_mean_history",
        "best_posterior_mean_history",
        "reported_posterior_mean_history",
        "best_noiseless_y_history",
        "recommended_true_y_history",
        "observed_y_history",
        "true_y_history",
        "x_history",
        "z_history",
    ]
    for key in sequence_keys:
        if key in trace:
            trace[key] = make_jsonable(crop_sequence_to_max_iter(trace.get(key), limit))
    for key in [
        "best_threshold_history_by_delta",
        "stopping_wall_clock_by_delta",
    ]:
        if key in trace:
            trace[key] = make_jsonable(crop_threshold_histories_to_max_iter(trace.get(key), limit))
    for key in ["agvs_history", "agvs_merge_history", "agvs_sel_merge_history", "vanilla_bo_history"]:
        if key in trace:
            trace[key] = make_jsonable(crop_history_records_to_max_iter(trace.get(key), limit))
    trace["n_evaluations"] = min(int(trace.get("n_evaluations", 0) or 0), limit)
    return trace


def extract_embedding_matrix(optimizer):
    for attr in (
        "A",
        "B",
        "A_np",
        "projection_matrix",
        "embedding_matrix",
        "random_embedding_matrix",
    ):
        if hasattr(optimizer, attr):
            out = ensure_jsonable(getattr(optimizer, attr))
            if out is not None:
                return out
    return None


def best_so_far(values):
    values = np.asarray(values, dtype=float).reshape(-1)
    out = []
    best = -np.inf
    for v in values:
        best = max(best, float(v))
        out.append(best)
    return out


def posterior_mean_history(result, n_evaluations):
    """Return model posterior means aligned to evaluation count.

    A value is emitted only when an algorithm has fitted its GP and recorded the
    posterior at its recommended point.  Pre-GP evaluations deliberately remain
    ``None``; observed objective values are never used as a fallback.
    """
    traces = []
    threshold_trace = getattr(result, "threshold_trace", None)
    if isinstance(threshold_trace, list):
        traces.extend(entry for entry in threshold_trace if isinstance(entry, dict))

    history = getattr(result, "history", None)
    if isinstance(history, list):
        for entry in history:
            if not isinstance(entry, dict):
                continue
            info = entry.get("stopping_info")
            if isinstance(info, dict) and info.get("posterior_mean_at_best") is not None:
                traces.append({
                    "n": int(info.get("n", entry.get("n_eval", 0))),
                    "posterior_mean_at_best": info.get("posterior_mean_at_best"),
                })
            threshold_entry = entry.get("threshold_entry")
            if isinstance(threshold_entry, dict) and threshold_entry.get("posterior_mean_at_best") is not None:
                traces.append(threshold_entry)

    out = [None] * max(0, int(n_evaluations))
    for entry in sorted(traces, key=lambda item: int(item.get("n", 0))):
        try:
            n_eval = int(entry.get("n", 0))
            value = float(entry.get("posterior_mean_at_best"))
        except (TypeError, ValueError):
            continue
        if 1 <= n_eval <= len(out) and np.isfinite(value):
            out[n_eval - 1] = value

    native_history = getattr(result, "posterior_mean_history", None)
    if isinstance(native_history, (list, tuple, np.ndarray)):
        for index, value in enumerate(list(native_history)[: len(out)]):
            if value is None:
                continue
            try:
                value = float(value)
            except (TypeError, ValueError):
                continue
            if np.isfinite(value):
                out[index] = value

    if result_stopped_early(result):
        returned = getattr(result, "returned_posterior_mean", None)
        stop_iteration = getattr(result, "stop_iteration", None)
        try:
            returned = float(returned)
            stop_index = int(stop_iteration) - 1
        except (TypeError, ValueError):
            returned = float("nan")
            stop_index = -1
        if np.isfinite(returned) and 0 <= stop_index < len(out):
            out[stop_index] = returned
    return out


def best_posterior_mean_so_far(values):
    """Running maximum of available posterior means, preserving pre-GP nulls."""
    out = []
    best = None
    for value in values or []:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            numeric = float("nan")
        if np.isfinite(numeric):
            best = numeric if best is None else max(best, numeric)
        out.append(best)
    return out


def result_stopped_early(result):
    if result is None:
        return False
    if bool(getattr(result, "stopped", False)):
        return True
    stopping_info = getattr(result, "stopping_info", None)
    if isinstance(stopping_info, dict) and stopping_info.get("newly_stopped"):
        return True
    stopping_results = getattr(result, "stopping_results", None)
    if isinstance(stopping_results, dict):
        return any(
            bool(delta_result.get("stopped", False))
            for by_delta in stopping_results.values() if isinstance(by_delta, dict)
            for delta_result in by_delta.values() if isinstance(delta_result, dict)
        )
    return False


def reported_posterior_mean_history(values, stopped_early=False):
    """Performance history with current-point semantics at an early stop."""
    reported = best_posterior_mean_so_far(values)
    if stopped_early:
        for index in range(len(values) - 1, -1, -1):
            value = values[index]
            try:
                value = float(value)
            except (TypeError, ValueError):
                continue
            if np.isfinite(value):
                reported[index] = value
                break
    return reported


def apply_posterior_return_contract(result, n_evaluations):
    """Normalize every benchmark adapter to the same posterior-return API."""
    if result is None:
        return result
    # AG-VS already selects its normal return from the cross-node max-mu
    # dataset (and its early-stop return from the triggering current point).
    if getattr(result, "posterior_dataset", None) is not None:
        return result
    values = posterior_mean_history(result, n_evaluations)
    reported = reported_posterior_mean_history(values, result_stopped_early(result))
    returned = next((value for value in reversed(reported) if value is not None), None)
    try:
        result.posterior_mean_history = values
        result.reported_posterior_mean_history = reported
        result.returned_posterior_mean = returned
        if returned is not None:
            result.best_y = float(returned)
            result.current_y = float(returned)
    except (AttributeError, TypeError):
        pass
    return result


def best_true_recommended(observed, true_values):
    observed = np.asarray(observed, dtype=float).reshape(-1)
    true_values = np.asarray(true_values, dtype=float).reshape(-1)
    out = []
    best_idx = None
    best_obs = -np.inf
    for idx, obs in enumerate(observed):
        if best_idx is None or obs > best_obs:
            best_idx = idx
            best_obs = float(obs)
        out.append(float(true_values[best_idx]))
    return out


def stopping_wall_clock_history(best_threshold_history, true_or_observed_history, eval_wall_clock_history):
    thresholds = np.asarray(best_threshold_history, dtype=float).reshape(-1)
    scores = np.asarray(true_or_observed_history, dtype=float).reshape(-1)
    times = np.asarray(eval_wall_clock_history, dtype=float).reshape(-1)

    if thresholds.size == 0 or scores.size == 0 or times.size == 0:
        return []

    usable_len = min(scores.size, times.size)
    scores = scores[:usable_len]
    times = times[:usable_len]
    best_scores = np.maximum.accumulate(scores)
    final_time = float(times[-1])

    out = []
    for tau in thresholds:
        reached = np.flatnonzero(best_scores >= float(tau))
        if reached.size > 0:
            out.append(float(times[int(reached[0])]))
        else:
            out.append(final_time)
    return out


def risk_label(delta):
    delta = float(delta)
    if np.isclose(delta, 1 / 256):
        return "1_256"
    if np.isclose(delta, 1 / 64):
        return "1_64"
    if np.isclose(delta, 1 / 16):
        return "1_16"
    if np.isclose(delta, 1 / 4):
        return "1_4"
    return str(delta).replace(".", "p").replace("/", "_")


def risk_label_pretty(delta):
    delta = float(delta)
    if np.isclose(delta, 1 / 256):
        return "1/256"
    if np.isclose(delta, 1 / 64):
        return "1/64"
    if np.isclose(delta, 1 / 16):
        return "1/16"
    if np.isclose(delta, 1 / 4):
        return "1/4"
    return f"{delta:g}"


def primary_risk_label():
    return risk_label(RISK_TOLERANCES[0])


def primary_risk_value():
    return float(RISK_TOLERANCES[0])


def _risk_key_to_label(key):
    key_text = str(key)
    known = {
        "1_256": "1_256",
        "1_64": "1_64",
        "1_16": "1_16",
        "1_4": "1_4",
        f"{float(1 / 256):.12g}": "1_256",
        f"{float(1 / 64):.12g}": "1_64",
        f"{float(1 / 16):.12g}": "1_16",
        f"{float(1 / 4):.12g}": "1_4",
        str(float(1 / 256)): "1_256",
        str(float(1 / 64)): "1_64",
        str(float(1 / 16)): "1_16",
        str(float(1 / 4)): "1_4",
    }
    return known.get(key_text)


def sanitize_risk_delta_json(obj):
    """Keep only the primary risk tolerance in JSON-like structures."""
    keep_label = primary_risk_label()
    keep_value = primary_risk_value()

    if isinstance(obj, dict):
        out = {}
        for key, value in obj.items():
            key_text = str(key)
            mapped_label = _risk_key_to_label(key_text)
            if mapped_label is not None:
                if mapped_label == keep_label:
                    out[keep_label] = sanitize_risk_delta_json(value)
                continue

            if key_text in {"risk_tolerances", "delta_values", "DELTA_VALUES", "RISK_TOLERANCES"}:
                if isinstance(value, dict):
                    out[key] = {keep_label: keep_value}
                elif isinstance(value, (list, tuple)):
                    out[key] = [keep_value]
                else:
                    out[key] = keep_value
                continue

            out[key] = sanitize_risk_delta_json(value)
        return out

    if isinstance(obj, list):
        return [sanitize_risk_delta_json(value) for value in obj]

    return obj


def active_z_history(benchmark_name, x_history, config):
    x_history = np.asarray(x_history, dtype=float)
    if x_history.size == 0:
        return []
    if x_history.ndim == 1:
        x_history = x_history.reshape(1, -1)
    active_indices = np.asarray(config["active_indices"], dtype=int)
    return [
        np.asarray(x, dtype=float).reshape(-1)[active_indices].tolist()
        for x in x_history
    ]


def agvs_dimension_selection_counts_from_history(history, original_dim):
    counts = np.zeros(int(original_dim), dtype=int)
    if not isinstance(history, list):
        return counts.tolist()

    for entry in history:
        if not isinstance(entry, dict):
            continue
        for dim in entry.get("selected_dims", []):
            dim = int(dim)
            if 0 <= dim < int(original_dim):
                counts[dim] += 1

    return counts.tolist()


def agvs_final_terminal_leaf_dims_from_trace(item, original_dim):
    """Recover the final terminal-leaf cohort by scanning history backwards."""
    if not isinstance(item, dict):
        return []
    stored = item.get("agvs_final_terminal_leaf_dims")
    if stored is not None:
        return sorted({int(dim) for dim in stored if 0 <= int(dim) < int(original_dim)})

    history = item.get("agvs_history")
    settings = item.get("settings", {})
    expected_dim = max(1, int(settings.get("effective_dim", len(item.get("important_dim") or []))))
    if isinstance(history, list) and history:
        depths = [int(entry.get("node_depth", -1)) for entry in history if isinstance(entry, dict)]
        terminal_depth = max(depths, default=-1)
        found = []
        for entry in reversed(history):
            if not isinstance(entry, dict) or int(entry.get("node_depth", -1)) != terminal_depth:
                continue
            dims = [int(dim) for dim in entry.get("selected_dims", [])]
            for dim in dims:
                if 0 <= dim < int(original_dim) and dim not in found:
                    found.append(dim)
            if len(found) >= expected_dim:
                break
        if found:
            return sorted(found[:expected_dim])

    return sorted({
        int(dim) for dim in (item.get("important_dim") or [])
        if 0 <= int(dim) < int(original_dim)
    })


def write_json(path, payload):
    def json_default(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, Path):
            return str(obj)
        raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    payload = sanitize_risk_delta_json(payload)
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=json_default)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


# =============================================================================
# Trace helpers
# =============================================================================

def benchmark_output_slug(name):
    text = str(name).strip()
    if not text:
        return text
    return text[:1].upper() + text[1:]


def trace_path_for(output_dir, benchmark_name, algorithm, trial):
    algorithm = canonical_algorithm_name(algorithm)
    slug = algorithm_output_slug(algorithm)
    return Path(output_dir) / benchmark_output_slug(benchmark_name) / slug / f"trial_{int(trial):03d}_{slug}.json"


def agvs_node_lengthscale_report_path(output_dir, benchmark_name, trial, algorithm="AG-VS-BO"):
    slug = algorithm_output_slug(algorithm)
    return (
        Path(output_dir)
        / benchmark_output_slug(benchmark_name)
        / slug
        / f"trial_{int(trial):03d}_ag_vs_node_lengthscales.txt"
    )


def agvs_final_tree_png_path(output_dir, benchmark_name, trial, algorithm="AG-VS-BO"):
    slug = algorithm_output_slug(algorithm)
    return (
        Path(output_dir)
        / benchmark_output_slug(benchmark_name)
        / slug
        / f"trial_{int(trial):03d}_{slug}_final_tree.png"
    )


def save_agvs_final_tree_png(optimizer, output_dir, benchmark_name, trial, algorithm="AG-VS-BO"):
    if optimizer is None or not hasattr(optimizer, "_plot_group_tree"):
        return None

    path = agvs_final_tree_png_path(output_dir, benchmark_name, trial, algorithm=algorithm)
    path.parent.mkdir(parents=True, exist_ok=True)

    selected = None
    history = getattr(optimizer, "history", None)
    if isinstance(history, list) and history:
        last = history[-1]
        if isinstance(last, dict) and last.get("selected_node") is not None:
            selected_idx = int(last.get("selected_node"))
            for node in getattr(optimizer, "_all_nodes", []) or []:
                if int(getattr(node, "idx", -1)) == selected_idx:
                    selected = node
                    break

    old_tree_plot_path = getattr(optimizer, "tree_plot_path", None)
    old_tree_plot_html_path = getattr(optimizer, "tree_plot_html_path", None)
    try:
        optimizer.tree_plot_path = str(path)
        optimizer.tree_plot_html_path = None
        optimizer._plot_group_tree(
            selected=selected,
            iteration=len(getattr(optimizer, "y", []) or []),
            force=True,
        )
    finally:
        optimizer.tree_plot_path = old_tree_plot_path
        optimizer.tree_plot_html_path = old_tree_plot_html_path

    return str(path)


def _format_active_dim_for_txt(dim, active_indices):
    dim = int(dim)
    text = str(dim)
    if dim in active_indices:
        return f"\033[32m{text}\033[0m"
    return text


def write_agvs_node_lengthscale_report(optimizer, output_dir, benchmark_name, trial, active_indices, algorithm="AG-VS-BO"):
    nodes = list(getattr(optimizer, "_all_nodes", []) or [])
    gp_fit_history = list(getattr(optimizer, "gp_fit_history", []) or [])
    path = agvs_node_lengthscale_report_path(output_dir, benchmark_name, trial, algorithm=algorithm)
    path.parent.mkdir(parents=True, exist_ok=True)
    active_set = {int(dim) for dim in np.asarray(active_indices, dtype=int).reshape(-1)}

    lines = [
        "AG-VS isotropic node GP lengthscale report",
        f"benchmark={benchmark_name} trial={int(trial):03d}",
        f"active_dimensions={sorted(active_set)}",
        "Every coordinate in a node shares one fitted length scale.",
        "Dimension grouping is random and does not use this diagnostic value.",
        "ANSI green marks active dimensions.",
        "",
        "All fitted/used local GPs, in fit order:",
        "",
    ]

    if gp_fit_history:
        for entry in gp_fit_history:
            lines.append(
                f"fit_id={int(entry.get('fit_id', 0)):04d} "
                f"after_iteration={int(entry.get('after_iteration', 0))} "
                f"node={entry.get('node', '?')} depth={int(entry.get('node_depth', 0))} "
                f"n={int(entry.get('local_n', 0))} Dim_h_i={int(entry.get('Dim_h_i', 0))} "
                f"acquisition_kernel={entry.get('kernel', 'none')} "
                f"acquisition_kernel_score={float(entry.get('kernel_score', float('nan'))):.8g} "
                f"lengthscale_kernel={entry.get('lengthscale_kernel', 'none')} "
                f"lengthscale_kernel_score="
                f"{float(entry.get('lengthscale_kernel_score', float('nan'))):.8g} "
                f"shared_lengthscale="
                f"{float(entry.get('isotropic_lengthscale', float('nan'))):.10g}"
            )
    else:
        lines.append("No AG-VS GP fit history was available.")

    lines.extend(["", "Final fitted node GP snapshots:", ""])

    fitted_count = 0
    for node in sorted(nodes, key=lambda item: (int(getattr(item, "h", 0)), int(getattr(item, "idx", 0)))):
        dims = [int(dim) for dim in getattr(node, "active_dimensions", [])]
        lengthscales = getattr(node, "lengthscales", None)
        model = getattr(node, "gp_model", None)
        if model is None or lengthscales is None or len(dims) == 0:
            continue
        lengthscales = np.asarray(lengthscales, dtype=float).reshape(-1)
        if lengthscales.size != len(dims):
            continue
        finite_pairs = [
            (int(dim), float(ls))
            for dim, ls in zip(dims, lengthscales)
            if np.isfinite(float(ls))
        ]
        if not finite_pairs:
            continue
        fitted_count += 1
        active_in_node = [dim for dim in dims if dim in active_set]
        node_label = f"{int(getattr(node, 'h', 0))},{int(getattr(node, 'idx', 0))}"
        lines.extend(
            [
                (
                    f"node={node_label} depth={int(getattr(node, 'h', 0))} "
                    f"n={int(getattr(node, 'n', len(getattr(node, 'node_dataset', []))))} "
                    f"Dim_h_i={len(dims)} "
                    f"acquisition_kernel={getattr(node, 'gp_kernel_name', 'none')} "
                    f"acquisition_kernel_score="
                    f"{float(getattr(node, 'gp_kernel_score', float('nan'))):.8g} "
                    f"lengthscale_kernel={getattr(node, 'lengthscale_gp_kernel_name', 'none')} "
                    f"lengthscale_kernel_score="
                    f"{float(getattr(node, 'lengthscale_gp_kernel_score', float('nan'))):.8g}"
                ),
                "active_dims_in_node=["
                + ", ".join(_format_active_dim_for_txt(dim, active_set) for dim in active_in_node)
                + "]",
                f"shared_lengthscale={float(finite_pairs[0][1]):.10g}",
            ]
        )
        lines.append("")

    if fitted_count == 0:
        lines.append("No fitted AG-VS node GP lengthscales were available.")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


def requested_trace_cache_identity(
    config,
    benchmark_name,
    algorithm,
    trial,
    max_iter,
    base_seed_offset=0,
    device=DEVICE,
    seed_bundle=None,
):
    """Return the complete, JSON-stable identity required for trace reuse."""

    benchmark_name = str(benchmark_name).lower()
    algorithm = canonical_algorithm_name(algorithm)
    trial = int(trial)
    max_iter = algorithm_evaluation_budget(algorithm, max_iter, benchmark_name)
    device = algorithm_device(algorithm, device)
    if seed_bundle is None:
        seed_bundle = seed_bundle_for_config(
            config,
            trial,
            base_seed_offset=int(base_seed_offset),
        )

    bounds = np.asarray(config["bounds"], dtype=float)
    active_indices = config.get("active_indices")
    active_bounds = config.get("active_bounds")
    optimum_x = config.get("optimum_x")
    objective_factory = config.get("objective_factory")
    configured_dataset = config.get("dataset_path")
    if configured_dataset is None:
        configured_dataset = getattr(objective_factory, "dataset_path", None)
    def path_fingerprint(raw_path):
        if raw_path is None:
            return None
        path = Path(raw_path).expanduser().resolve()
        payload = {
            "strategy": "resolved_path_size_mtime_v1",
            "path": str(path),
            "exists": path.exists(),
        }
        if not path.exists():
            return payload
        if path.is_file():
            stat = path.stat()
            payload.update(
                {
                    "kind": "file",
                    "size": int(stat.st_size),
                    "mtime_ns": int(stat.st_mtime_ns),
                }
            )
            return payload
        entries = []
        for child in sorted(path.glob("nb-asr-e40-*.pickle")):
            stat = child.stat()
            entries.append((child.name, int(stat.st_size), int(stat.st_mtime_ns)))
        manifest = json.dumps(entries, separators=(",", ":"), ensure_ascii=True)
        payload.update(
            {
                "kind": "directory",
                "matching_entry_count": len(entries),
                "manifest_sha256": hashlib.sha256(manifest.encode("utf-8")).hexdigest(),
            }
        )
        return payload

    def package_signature(distribution_names):
        versions = {}
        for distribution in distribution_names:
            try:
                versions[distribution] = importlib.metadata.version(distribution)
            except importlib.metadata.PackageNotFoundError:
                versions[distribution] = None
        return versions

    real_life_task = config.get("real_life_task")
    runtime_signature = None
    if str(config.get("experiment_kind", "synthetic")) == "real_life":
        packages = (
            "numpy",
            "scipy",
            "scikit-learn",
            "torch",
            "botorch",
            "gpytorch",
            "ax-platform",
            "cma",
            "jax",
            "numpyro",
        )
        if real_life_task == "mopta08":
            from Experiment.benchmark.real_life.mopta08_backend import (
                MOPTA08_EXECUTABLE_SHA256, configured_mopta08_executable,
            )
            runtime_signature = {
                "package_versions": package_signature(packages),
                "mopta08_executable": str(configured_mopta08_executable()),
                "mopta08_executable_sha256": MOPTA08_EXECUTABLE_SHA256,
            }
        else:
            raise ValueError(f"Unknown real-life task {real_life_task!r}")

    # Worker count and process/worker IDs are runtime scheduling details.  They
    # are intentionally excluded from both persisted trial data and cache
    # identity so changing parallelism does not rerun compatible trials.
    config_signature = {
        "experiment_id": config.get("experiment_id"),
        "experiment_kind": str(config.get("experiment_kind", "synthetic")),
        "objective": config.get("objective"),
        "real_life_task": real_life_task,
        "real_life_variant": config.get("real_life_variant"),
        "upstream_name": config.get("upstream_name"),
        "metric_name": config.get("metric_name", "objective"),
        "metric_transform": config.get("metric_transform"),
        "optimization_direction": config.get("optimization_direction"),
        "budget_source": config.get("budget_source"),
        "registered_max_iter": config.get("max_iter"),
        "original_dim": int(bounds.shape[0]),
        "effective_dim": int(benchmark_effective_dim(config)),
        "effective_dim_role": config.get("effective_dim_role", "ground_truth"),
        "effective_dimension_seed": config.get("effective_dimension_seed"),
        "active_indices": (
            None
            if active_indices is None
            else np.asarray(active_indices, dtype=int).reshape(-1).tolist()
        ),
        "bounds": bounds.tolist(),
        "active_bounds": (
            None
            if active_bounds is None
            else np.asarray(active_bounds, dtype=float).tolist()
        ),
        "optimum_x": (
            None
            if optimum_x is None
            else np.asarray(optimum_x, dtype=float).reshape(-1).tolist()
        ),
        "lipschitz_constant": config.get("L"),
        "benchmark_noise_std": float(config.get("benchmark_noise_std", 0.0)),
        "q_batch": config.get("q_batch"),
        "selection_batch_size": config.get("selection_batch_size"),
        "t_seq": config.get("t_seq"),
        # N_bad is an AG-VS tree-reinitialization control. Keeping it out of
        # unrelated algorithms' identities preserves valid dimension-invariant
        # reuse for methods such as TuRBO.
        "n_bad_multiplier": (
            config.get("n_bad_multiplier")
            if str(algorithm).startswith("AG-VS-")
            else None
        ),
        "n_bad_enabled": (
            config.get("n_bad_enabled")
            if str(algorithm).startswith("AG-VS-")
            else None
        ),
        "candidate_projection": config.get("candidate_projection"),
        "nasbench101_edge_policy": config.get("nasbench101_edge_policy"),
        "dataset": path_fingerprint(configured_dataset),
        "dataset_identifier": config.get("dataset_identifier"),
        "objective_protocol": config.get("objective_protocol"),
        "num_rollouts": config.get("num_rollouts"),
        "rollout_seed_policy": config.get("rollout_seed_policy"),
        "nas_dataset_seed_policy": config.get("nas_dataset_seed_policy"),
        "nas_dataset_seed_protocol_version": config.get(
            "nas_dataset_seed_protocol_version"
        ),
        "reference_environment_id": config.get("reference_environment_id"),
        "environment_version_policy": config.get("environment_version_policy"),
        "mujoco_protocol_track": config.get("mujoco_protocol_track"),
        "wall_clock_timing_policy": config.get("wall_clock_timing_policy"),
        "runtime_signature": runtime_signature,
    }
    if str(algorithm).startswith("AG-VS-") and bool(
        config.get("n_bad_is_absolute", False)
    ):
        # Preserve ordinary cache identities: this discriminator is emitted
        # only by an explicit absolute-N_bad sensitivity experiment.
        config_signature["n_bad_is_absolute"] = True
    if str(algorithm).startswith("AG-VS-") and "agvs_merge_enabled" in config:
        # Preserve existing experiment identities.  This discriminator is
        # emitted only by an explicit merge ablation.
        config_signature["agvs_merge_enabled"] = benchmark_agvs_merge_enabled(config)
    if str(algorithm).startswith("AG-VS-") and "agvs_gp_noise_variance" in config:
        config_signature["agvs_gp_noise_variance"] = benchmark_agvs_gp_noise_variance(config)
    if str(algorithm).startswith("AG-VS-") and "agvs_n_init" in config:
        config_signature["agvs_n_init"] = benchmark_n_init(config)
    if str(algorithm).startswith("AG-VS-") and "agvs_effective_dim" in config:
        config_signature["agvs_effective_dim"] = benchmark_agvs_effective_dim(config)
        config_signature["agvs_n_init"] = benchmark_n_init(config)
    return sanitize_risk_delta_json(make_jsonable(
        {
            "schema_version": 2,
            "benchmark": benchmark_name,
            "algorithm": algorithm,
            "trial_id": trial,
            "requested_max_iter": max_iter,
            "implementation_version": "native-v1",
            "seed_policy": "shared_trial_seed_v1",
            "base_seed_offset": int(base_seed_offset),
            "algorithm_seed": int(seed_bundle.algorithm_seed),
            "benchmark_seed": int(seed_bundle.benchmark_seed),
            "effective_dimension_seed": seed_bundle.effective_dimension_seed,
            "initial_design_seed": int(seed_bundle.algorithm_seed),
            "device": device,
            "benchmark_config": config_signature,
            "algorithm_settings": algorithm_settings_summary(
                algorithm,
                benchmark_name,
                device=device,
            ),
        }
    ))


def load_existing_trace_if_complete(
    output_dir,
    benchmark_name,
    algorithm,
    trial,
    max_iter=None,
    cache_identity=None,
    reuse_complete_json=False,
):
    """Load a cache-compatible trace only after it reached the requested budget.

    Budget completion is the reusable-cache contract for every algorithm.  A
    stale ``complete=False``/``error`` pair can be left behind when a managed
    optimizer raises while returning after its final objective call.  Such a
    trace must not repeat expensive evaluations once ``n_evaluations`` already
    reached ``max_iter``.  Conversely, ``complete=True`` alone does not make a
    short trace reusable.
    """

    path = trace_path_for(output_dir, benchmark_name, algorithm, trial)
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            trace = json.load(handle)
    except Exception:
        return None

    migrated_n_bad_identity = migrate_legacy_n_bad_cache_identity(trace)
    if migrated_n_bad_identity:
        # Old AG-VS traces already recorded the actual N_bad values under
        # settings.algorithm_settings. Persisting those same values in the
        # cache signature is a provenance-preserving schema migration, not an
        # assumption about an old run. write_json uses fsync + os.replace.
        try:
            write_json(path, trace)
        except OSError as exc:
            warnings.warn(
                f"Could not persist the verified N_bad cache migration for {path}: {exc}",
                RuntimeWarning,
            )

    expected_identity = make_jsonable(cache_identity)
    stored_identity = trace.get("cache_identity")
    strict_identity_match = (
        expected_identity is not None
        and cache_identities_match(stored_identity, expected_identity)
    )
    relaxed_identity_match = (
        bool(reuse_complete_json)
        and expected_identity is not None
        and completed_json_matches_request(
            trace,
            expected_identity=expected_identity,
            benchmark_name=benchmark_name,
            algorithm=algorithm,
            trial=trial,
        )
    )
    if not strict_identity_match and not relaxed_identity_match:
        return None

    requested_budget = max_iter
    if requested_budget is None and isinstance(expected_identity, dict):
        requested_budget = expected_identity.get("requested_max_iter")
    try:
        requested_budget = algorithm_evaluation_budget(
            canonical_algorithm_name(algorithm), requested_budget, benchmark_name
        )
        recorded_evaluations = int(trace.get("n_evaluations", 0) or 0)
    except (TypeError, ValueError):
        return None
    if requested_budget <= 0 or recorded_evaluations < requested_budget:
        return None

    if trace.get("complete", False) is not True or trace.get("error") not in (
        None,
        "",
        False,
    ):
        # Preserve why the old writer marked the trace unsuccessful, while
        # normalizing the public status so plotting/table loaders include this
        # already budget-complete result.  The exact cache identity check above
        # remains mandatory, so an unrelated/stale experiment is never reused.
        trace["budget_completion_recovery"] = {
            "schema_version": 1,
            "reason": "n_evaluations reached requested_max_iter",
            "requested_max_iter": requested_budget,
            "n_evaluations": recorded_evaluations,
            "previous_complete": trace.get("complete"),
            "previous_error": trace.get("error"),
            "previous_interrupted": trace.get("interrupted"),
        }
        trace["complete"] = True
        trace["error"] = None
        trace["interrupted"] = False
        try:
            write_json(path, trace)
        except OSError as exc:
            warnings.warn(
                f"Could not persist budget-complete cache recovery for {path}: {exc}",
                RuntimeWarning,
            )
    return trace


def completed_json_matches_request(
    trace,
    *,
    expected_identity,
    benchmark_name,
    algorithm,
    trial,
):
    """Validate the immutable run coordinates of an imported completed JSON.

    This is the deliberately narrow compatibility path used by
    ``--reuse-complete-json``. It permits old reporting/runtime metadata and
    optimizer-version fields to differ, while still rejecting a result from a
    different benchmark, algorithm, trial, seed bundle, or device. Budget
    completion is checked separately by ``load_existing_trace_if_complete``.
    """

    if not isinstance(trace, dict) or not isinstance(expected_identity, dict):
        return False
    if trace.get("complete") is not True or trace.get("error") not in (
        None,
        "",
        False,
    ):
        return False

    stored_identity = trace.get("cache_identity")
    if not isinstance(stored_identity, dict):
        return False

    requested_benchmark = str(benchmark_name).strip().lower()
    requested_algorithm = canonical_algorithm_name(algorithm)
    try:
        trace_trial = int(trace.get("trial_id", trace.get("trial")))
        stored_trial = int(stored_identity.get("trial_id"))
        expected_trial = int(expected_identity.get("trial_id"))
    except (TypeError, ValueError):
        return False

    if str(trace.get("benchmark", "")).strip().lower() != requested_benchmark:
        return False
    try:
        trace_algorithm = canonical_algorithm_name(trace.get("algorithm"))
        stored_algorithm = canonical_algorithm_name(stored_identity.get("algorithm"))
        expected_algorithm = canonical_algorithm_name(expected_identity.get("algorithm"))
    except (KeyError, TypeError, ValueError):
        return False
    if not (
        trace_algorithm
        == stored_algorithm
        == expected_algorithm
        == requested_algorithm
    ):
        return False
    if not (trace_trial == stored_trial == expected_trial == int(trial)):
        return False
    if str(stored_identity.get("benchmark", "")).strip().lower() != requested_benchmark:
        return False
    if str(expected_identity.get("benchmark", "")).strip().lower() != requested_benchmark:
        return False

    # These values define the random stream and execution target. Workers are
    # intentionally absent: parallelism does not change trial identity.
    for key in (
        "base_seed_offset",
        "algorithm_seed",
        "benchmark_seed",
        "effective_dimension_seed",
        "initial_design_seed",
        "device",
    ):
        expected_value = expected_identity.get(key)
        stored_value = stored_identity.get(key)
        if stored_value != expected_value:
            return False
        if key in trace and trace.get(key) != expected_value:
            return False

    # Relaxed legacy-result reuse must still preserve the scientific AG-VS
    # reinitialization condition.  In particular, changing an absolute N_bad
    # value must schedule a new trial even when old imported JSON metadata is
    # otherwise accepted.
    if requested_algorithm in AGVS_ALGORITHMS:
        stored_config = stored_identity.get("benchmark_config")
        expected_config = expected_identity.get("benchmark_config")
        if not isinstance(stored_config, dict) or not isinstance(
            expected_config, dict
        ):
            return False
        for key in (
            "agvs_effective_dim",
            "agvs_n_init",
            "n_bad_enabled",
            "n_bad_multiplier",
            "n_bad_is_absolute",
        ):
            if stored_config.get(key) != expected_config.get(key):
                return False
        expected_merge = expected_config.get("agvs_merge_enabled")
        stored_merge = stored_config.get("agvs_merge_enabled")
        if stored_merge != expected_merge:
            # Before merge ablations existed, every native AG-VS run used the
            # canonical merge path.  Missing legacy metadata therefore proves
            # merge-enabled behavior, but can never satisfy merge_disabled.
            settings = trace.get("settings", {})
            algorithm_settings = (
                settings.get("algorithm_settings", {})
                if isinstance(settings, dict)
                else {}
            )
            legacy_runtime_merge = algorithm_settings.get("merge_enabled")
            if not (
                stored_merge is None
                and expected_merge is True
                and legacy_runtime_merge in (None, True)
            ):
                return False
        expected_gp_variance = expected_config.get("agvs_gp_noise_variance")
        stored_gp_variance = stored_config.get("agvs_gp_noise_variance")
        if stored_gp_variance != expected_gp_variance:
            settings = trace.get("settings", {})
            algorithm_settings = (
                settings.get("algorithm_settings", {})
                if isinstance(settings, dict)
                else {}
            )
            legacy_gp_variance = algorithm_settings.get(
                "noise_variance", algorithm_settings.get("alpha")
            )
            try:
                legacy_gp_matches = (
                    stored_gp_variance is None
                    and expected_gp_variance is not None
                    and legacy_gp_variance is not None
                    and math.isclose(
                        float(legacy_gp_variance),
                        float(expected_gp_variance),
                        rel_tol=0.0,
                        abs_tol=1e-15,
                    )
                )
            except (TypeError, ValueError, OverflowError):
                legacy_gp_matches = False
            if not legacy_gp_matches:
                return False
    return True


def load_under_budget_trace_for_resume(
    output_dir,
    benchmark_name,
    algorithm,
    trial,
    *,
    target_max_iter,
    expected_identity,
):
    """Load and preserve a compatible completed trace below a larger budget.

    Optimizer state is not serialized by every baseline. Continuation therefore
    rebuilds the optimizer and deterministically replays the stored evaluations.
    The recorder verifies every proposed x before returning its stored y, so an
    implementation or environment change cannot silently splice two trajectories.
    """

    source_path = trace_path_for(output_dir, benchmark_name, algorithm, trial)
    if not source_path.exists():
        return None
    try:
        source_bytes = source_path.read_bytes()
        trace = json.loads(source_bytes.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not completed_json_matches_request(
        trace,
        expected_identity=expected_identity,
        benchmark_name=benchmark_name,
        algorithm=algorithm,
        trial=trial,
    ):
        return None
    try:
        source_evaluations = int(trace.get("n_evaluations", 0) or 0)
        target_max_iter = int(target_max_iter)
    except (TypeError, ValueError):
        return None
    if not 0 < source_evaluations < target_max_iter:
        return None

    source_sha256 = hashlib.sha256(source_bytes).hexdigest()
    algorithm_slug = algorithm_output_slug(canonical_algorithm_name(algorithm))
    backup_path = (
        Path(output_dir)
        / benchmark_output_slug(benchmark_name)
        / "_resume_sources"
        / algorithm_slug
        / (
            f"source_trial_{int(trial):03d}_evals_{source_evaluations}_"
            f"{source_sha256[:12]}.json"
        )
    )
    if not backup_path.exists():
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = backup_path.with_suffix(backup_path.suffix + ".tmp")
        temporary_path.write_bytes(source_bytes)
        os.replace(temporary_path, backup_path)

    return {
        "trace": trace,
        "source_path": source_path,
        "backup_path": backup_path,
        "source_sha256": source_sha256,
        "source_evaluations": source_evaluations,
        "target_evaluations": target_max_iter,
    }


def migrate_legacy_n_bad_cache_identity(trace):
    """Copy recorded legacy AG-VS N_bad settings into its cache signature.

    Returns ``True`` only when a complete, error-free AG-VS trace contains
    explicit legacy runtime evidence and one or both cache fields were absent.
    Conflicting or incomplete metadata is never rewritten or accepted.
    """

    if not isinstance(trace, dict):
        return False
    if trace.get("complete", False) is not True:
        return False
    if trace.get("error") not in (None, "", False):
        return False

    cache_identity = trace.get("cache_identity")
    if not isinstance(cache_identity, dict):
        return False
    if cache_identity.get("implementation_version") != "native-v1":
        return False
    algorithm = str(
        trace.get("algorithm") or cache_identity.get("algorithm") or ""
    )
    if not algorithm.startswith("AG-VS-"):
        return False

    benchmark_config = cache_identity.get("benchmark_config")
    settings = trace.get("settings")
    if not isinstance(benchmark_config, dict) or not isinstance(settings, dict):
        return False
    legacy_settings = settings.get("algorithm_settings")
    if not isinstance(legacy_settings, dict):
        return False

    enabled = legacy_settings.get("N_bad_enabled")
    multiplier = legacy_settings.get("N_bad_multiplier")
    if not isinstance(enabled, bool):
        return False
    if (
        isinstance(multiplier, bool)
        or not isinstance(multiplier, int)
        or multiplier <= 0
    ):
        return False

    recorded = {
        "n_bad_enabled": enabled,
        "n_bad_multiplier": multiplier,
    }
    for key, value in recorded.items():
        if key in benchmark_config and benchmark_config[key] != value:
            return False

    changed = False
    for key, value in recorded.items():
        if key not in benchmark_config:
            benchmark_config[key] = value
            changed = True
    if changed:
        migrations = trace.setdefault("cache_identity_migrations", [])
        marker = "n_bad_from_recorded_algorithm_settings_v1"
        if isinstance(migrations, list) and marker not in migrations:
            migrations.append(marker)
    return changed


def migrate_legacy_n_bad_cache_identities(result_roots=None):
    """Atomically migrate every evidence-bearing legacy AG-VS result JSON.

    The bulk path preserves the original JSON bytes and inserts only the two
    missing cache fields. This matters because historical AG-VS traces contain
    very large numeric histories; parsing and re-serializing those arrays would
    make a metadata-only migration unnecessarily slow and noisy.
    """

    roots = (
        (SYNTHETIC_RESULT_ROOT, REAL_LIFE_RESULT_ROOT)
        if result_roots is None
        else tuple(Path(root) for root in result_roots)
    )
    stats = {
        "scanned": 0,
        "migrated": 0,
        "already_current_or_ineligible": 0,
        "unreadable": 0,
    }
    for root in roots:
        if not Path(root).exists():
            continue
        candidates = sorted(
            path
            for path in Path(root).glob("*/*/*.json")
            if path.parent.name.lower() in {"ag-vs-bo", "ag-vs-turbo"}
        )
        for path in candidates:
            stats["scanned"] += 1
            try:
                payload = path.read_bytes()
            except OSError:
                stats["unreadable"] += 1
                continue

            if b'"n_bad_enabled"' in payload or b'"n_bad_multiplier"' in payload:
                stats["already_current_or_ineligible"] += 1
                continue

            enabled_match = re.search(
                rb'"N_bad_enabled"\s*:\s*(true|false)', payload
            )
            multiplier_match = re.search(
                rb'"N_bad_multiplier"\s*:\s*([1-9][0-9]*)', payload
            )
            if (
                b'"complete": true' not in payload
                or b'"error": null' not in payload
                or enabled_match is None
                or multiplier_match is None
            ):
                stats["already_current_or_ineligible"] += 1
                continue

            cache_start = payload.find(b'"cache_identity": {')
            benchmark_start = payload.find(b'"benchmark_config": {', cache_start)
            insertion_marker = b'      "candidate_projection":'
            insertion_at = payload.find(insertion_marker, benchmark_start)
            if cache_start < 0 or benchmark_start < 0 or insertion_at < 0:
                stats["unreadable"] += 1
                continue

            enabled_json = enabled_match.group(1)
            multiplier_json = multiplier_match.group(1)
            newline = b"\r\n" if b"\r\n" in payload else b"\n"
            insertion = (
                b'      "n_bad_multiplier": '
                + multiplier_json
                + b','
                + newline
                + b'      "n_bad_enabled": '
                + enabled_json
                + b','
                + newline
            )
            migrated_payload = (
                payload[:insertion_at] + insertion + payload[insertion_at:]
            )
            tmp = path.with_suffix(path.suffix + ".tmp")
            try:
                with tmp.open("wb") as handle:
                    handle.write(migrated_payload)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(tmp, path)
            except OSError:
                stats["unreadable"] += 1
                continue
            stats["migrated"] += 1
    return stats


def cache_identities_match(stored_identity, expected_identity):
    """Compare run identities without letting reporting-only config invalidate runs.

    q_batch, batch_size/selection_batch_size, and t_seq are duplicated in the benchmark
    signature for reporting, but they control only AG-VS.  AG-VS identities
    already carry their effective values in ``algorithm_settings``.  Ignoring
    the duplicated benchmark-level copies therefore preserves strict AG-VS
    invalidation while allowing unrelated algorithms to reuse valid traces.

    Human-readable stopping-check/device-policy wording is reporting metadata for every
    algorithm.  The behavior-bearing stopping flag, threshold, probability,
    and risk tolerances remain part of the identity. Plain TuRBO also records
    a few human-readable surrogate/runtime labels. Keep cache invalidation tied
    to behavior and actual hyperparameters rather than reporting text.
    """
    stored = copy.deepcopy(make_jsonable(stored_identity))
    expected = copy.deepcopy(make_jsonable(expected_identity))
    if not isinstance(stored, dict) or not isinstance(expected, dict):
        return stored == expected

    stored_config = stored.get("benchmark_config")
    expected_config = expected.get("benchmark_config")
    synthetic_cache = (
        isinstance(stored_config, dict)
        and isinstance(expected_config, dict)
        and str(stored_config.get("experiment_kind", "synthetic")).lower()
        == "synthetic"
        and str(expected_config.get("experiment_kind", "synthetic")).lower()
        == "synthetic"
    )
    duplicated_agvs_keys = (
        "q_batch",
        "batch_size",
        "selection_batch_size",
        "t_seq",
    )
    for identity in (stored, expected):
        benchmark_config = identity.get("benchmark_config")
        if isinstance(benchmark_config, dict):
            for key in duplicated_agvs_keys:
                benchmark_config.pop(key, None)

            # A missing field and an explicit null both mean that no timing
            # policy was configured in these synthetic traces.
            if benchmark_config.get("wall_clock_timing_policy") is None:
                benchmark_config.pop("wall_clock_timing_policy", None)

            # These provenance fields were added after the first synthetic
            # result set was written.  Missing and explicit null both mean
            # that no dataset/protocol override was configured.  A populated
            # value remains behavior-bearing and must still match exactly.
            if synthetic_cache:
                for key in ("dataset_identifier", "objective_protocol"):
                    if benchmark_config.get(key) is None:
                        benchmark_config.pop(key, None)

            # N_bad controls only AG-VS tree reinitialization.  Historical
            # identities duplicated the synthetic defaults into every
            # algorithm's benchmark signature; those values cannot affect a
            # non-AG-VS run and must not invalidate its completed trace.
            if identity.get("algorithm") not in AGVS_ALGORITHMS:
                benchmark_config.pop("n_bad_multiplier", None)
                benchmark_config.pop("n_bad_enabled", None)

        algorithm_settings = identity.get("algorithm_settings")
        if isinstance(algorithm_settings, dict):
            for key in (
                "posterior_stopping_check",
                "device_policy",
                # These sentences document behavior already represented by
                # batch_size/q_batch/t_seq and the TuRBO hyperparameters.
                # Adding or rewording them must not invalidate completed runs.
                "node_visit_policy",
                "turbo_backend_batch_policy",
                "final_batch_budget_rule",
            ):
                algorithm_settings.pop(key, None)

        if identity.get("algorithm") == "TuRBO":
            if isinstance(algorithm_settings, dict):
                for key in (
                    "surrogate_family",
                    "gp_library",
                    "small_data_runtime",
                ):
                    algorithm_settings.pop(key, None)
    return stored == expected


def count_existing_complete_traces(
    output_dir,
    benchmarks,
    algorithms,
    trials,
    max_iter=None,
    base_seed_offset=0,
    device=DEVICE,
    reuse_complete_json=False,
):
    traces = existing_complete_traces(
        output_dir,
        benchmarks,
        algorithms,
        trials,
        max_iter=max_iter,
        base_seed_offset=base_seed_offset,
        device=device,
        reuse_complete_json=reuse_complete_json,
    )
    counts = {}
    for benchmark in benchmarks:
        benchmark = str(benchmark).lower()
        for algorithm in algorithms:
            algorithm = canonical_algorithm_name(algorithm)
            count = sum(
                (int(trial), benchmark, algorithm) in traces
                for trial in range(int(trials))
            )
            counts[(benchmark, algorithm)] = count
    return len(traces), counts


def existing_complete_traces(
    output_dir,
    benchmarks,
    algorithms,
    trials,
    max_iter=None,
    base_seed_offset=0,
    device=DEVICE,
    reuse_complete_json=False,
):
    """Return every reusable trace that reached the requested budget."""

    traces = {}
    for benchmark in benchmarks:
        benchmark = str(benchmark).lower()
        config = BENCHMARKS.get(benchmark)
        if config is None:
            continue
        for algorithm in algorithms:
            algorithm = canonical_algorithm_name(algorithm)
            for trial in range(int(trials)):
                requested_max_iter = (
                    int(config.get("max_iter"))
                    if max_iter is None
                    else int(max_iter)
                )
                cache_identity = requested_trace_cache_identity(
                    config,
                    benchmark,
                    algorithm,
                    trial,
                    requested_max_iter,
                    base_seed_offset=base_seed_offset,
                    device=device,
                )
                trace = load_existing_trace_if_complete(
                    output_dir,
                    benchmark,
                    algorithm,
                    trial,
                    max_iter=requested_max_iter,
                    cache_identity=cache_identity,
                    reuse_complete_json=reuse_complete_json,
                )
                if trace is not None:
                    # Scheduling needs only membership, not multi-megabyte history
                    # arrays from every skipped trial.
                    traces[(int(trial), benchmark, algorithm)] = {
                        "complete": True,
                        "n_evaluations": int(trace.get("n_evaluations", 0) or 0),
                    }
    return traces


# =============================================================================
# Algorithm builders
# =============================================================================

def make_agvs(
    config,
    recorder,
    seed,
    max_iter,
    node_backend="vanillabo",
    initial_points=None,
    device=DEVICE,
):
    if AGVSOptimizer is None:
        raise ImportError(
            "AG-VS seçildi ama AGVSOptimizer import edilemedi. "
            f"Original import error: {AGVS_IMPORT_ERROR}"
        )

    bounds = np.asarray(config["bounds"], dtype=float)
    node_backend = str(node_backend).strip().lower()
    if node_backend not in {"vanillabo", "turbo"}:
        raise ValueError("AG-VS node_backend must be 'vanillabo' or 'turbo'")
    tree_slug = f"ag-vs-{'bo' if node_backend == 'vanillabo' else 'turbo'}"
    tree_family = "bo" if node_backend == "vanillabo" else "turbo"
    tree_plot_path = str(
        RESULT_FIGURE_ROOT / tree_family / "Live" / f"{tree_slug}_live_tree.png"
    )
    tree_plot_html_path = str(
        RESULT_FIGURE_ROOT / tree_family / "Live" / f"{tree_slug}_live_tree.html"
    )

    if AGVS_VERBOSE and tree_plot_path and tree_plot_html_path:
        print(
            f"AG-VS live tree enabled: {tree_plot_html_path} "
            f"(png: {tree_plot_path}, open_browser={AGVS_TREE_OPEN_BROWSER}, "
            f"python_live_plot={AGVS_PYTHON_LIVE_PLOT_ON})",
            flush=True,
        )

    effective_dim = benchmark_agvs_effective_dim(config)
    # Exactly one of q_batch and batch_size is configured. q_batch uses joint
    # acquisition, while batch_size uses independent pointwise acquisition.
    effective_q_batch = benchmark_q_batch(config)
    effective_batch_size = benchmark_batch_size(config)
    effective_t_seq = int(config.get("t_seq", AGVS_T_SEQ))
    effective_n_split = int(effective_dim + 1)
    is_real_life = (
        str(config.get("experiment_kind", "synthetic")).strip().lower()
        == "real_life"
    )
    effective_n_bad_multiplier = int(
        config.get(
            "n_bad_multiplier",
            REAL_LIFE_N_BAD_MULTIPLIER
            if is_real_life
            else SYNTHETIC_N_BAD_MULTIPLIER,
        )
    )
    # Ordinary experiments express N_bad as a multiplier of N_split. Dedicated
    # sensitivity studies may instead request this input as the exact absolute
    # patience while retaining the central runner and cache machinery.
    n_bad_is_absolute = bool(config.get("n_bad_is_absolute", False))
    effective_n_bad = int(
        effective_n_bad_multiplier
        if n_bad_is_absolute
        else effective_n_bad_multiplier * effective_n_split
    )
    learn_noise = is_real_life
    gp_noise_variance = benchmark_agvs_gp_noise_variance(config)
    experiment_mode = (not is_real_life) and config.get("active_indices") is not None
    return AGVSOptimizer(
        objective=recorder if recorder is not None else config["func"],
        bounds=bounds,
        eff_dim=effective_dim,
        max_iter=max_iter,
        n_init=benchmark_n_init(config),
        abs_mode=AGVS_ABS_MODE,
        noise_variance=gp_noise_variance,
        learn_noise=learn_noise,
        tau=None,
        acquisition=STANDARD_BATCH_ACQUISITION,
        candidate_count=1024,
        acquisition_sampling_strategy=AGVS_ACQUISITION_SAMPLING_STRATEGY,
        q_batch=effective_q_batch,
        batch_size=effective_batch_size,
        t_seq=effective_t_seq,
        N_bad=effective_n_bad,
        N_bad_enabled=bool(config.get("n_bad_enabled", AGVS_N_BAD_ENABLED)),
        merge_enabled=benchmark_agvs_merge_enabled(config),
        sim_mode=False,
        n_split=effective_n_split,
        initial_sampling_strategy="lhs",
        initial_points=initial_points,
        seed=seed,
        verbose=AGVS_VERBOSE,
        printer_on=True,
        active_indices=config.get("active_indices"),
        tree_plot_on=AGVS_VERBOSE,
        tree_plot_path=tree_plot_path,
        tree_plot_html_path=tree_plot_html_path,
        tree_plot_open_browser=AGVS_TREE_OPEN_BROWSER,
        tree_plot_pause=0.05 if AGVS_PYTHON_LIVE_PLOT_ON else 0.001,
        risk_tolerances=RISK_TOLERANCES,
        kernel_selection="fixed",
        default_kernel="matern52",
        node_backend=node_backend,
        early_stop_enabled=POSTERIOR_STOPPING_ENABLED,
        experiment_mode=experiment_mode,
        figure_mode=FIGURE_MODE,
        important_indices=config.get("active_indices"),
        ipt_solver=node_backend,
        turbo_max_evals=AGVS_TURBO_MAX_EVALS,
        turbo_device=normalize_device(device),
        turbo_min_cuda=AGVS_TURBO_MIN_CUDA,
    )


def make_vanilla_bo(config, recorder, seed, max_iter, benchmark_name=None, trial=None, output_dir=None, initial_points=None):
    if VanillaBOOptimizer is None:
        raise ImportError(
            "VanillaBO seçildi ama VanillaBOOptimizer import edilemedi. "
            f"Original import error: {VANILLA_BO_IMPORT_ERROR}"
        )

    bounds = np.asarray(config["bounds"], dtype=float)
    report_path = None
    if output_dir is not None and benchmark_name is not None and trial is not None:
        report_path = (
            Path(output_dir)
            / str(benchmark_name).lower()
            / "VanillaBO"
            / f"trial_{int(trial):03d}_top10_dimensions.txt"
        )

    return VanillaBOOptimizer(
        objective=recorder,
        bounds=bounds,
        max_iter=max_iter,
        n_init=MCTS_PAPER_BO_N_INIT,
        q_batch=MCTS_PAPER_BO_BATCH_SIZE,
        acquisition="ei",
        noise_variance=BO_GP_ASSUMED_NOISE_VAR,
        candidate_count=1024,
        beta=2.0,
        stopping_delta=RISK_TOLERANCES[0],
        tau_values=[POSTERIOR_STOPPING_THRESHOLD],
        delta_values=RISK_TOLERANCES,
        enable_posterior_stopping=POSTERIOR_STOPPING_ENABLED,
        stopping_min_points=1,
        random_state=seed,
        active_indices=config.get("active_indices"),
        dimension_report_path=report_path,
        benchmark_name=benchmark_name,
        trial=trial,
        initial_points=initial_points,
        verbose=True,
    )


def make_mcts_vs(
    config,
    recorder,
    seed,
    max_iter,
    ipt_solver="bo",
    initial_points=None,
    benchmark_name=None,
    device=DEVICE,
):
    if MCTSVSOptimizer is None:
        raise ImportError(
            "MCTS-VS seçildi ama MCTSVSOptimizer import edilemedi. "
            f"Original import error: {MCTSVS_IMPORT_ERROR}"
        )

    bounds = np.asarray(config["bounds"], dtype=float)
    ipt_solver = str(ipt_solver).strip().lower()
    tree_plot_path = MCTS_TURBO_TREE_PLOT_PATH if ipt_solver == "turbo" else MCTS_TREE_PLOT_PATH
    tree_plot_html_path = MCTS_TURBO_TREE_PLOT_HTML_PATH if ipt_solver == "turbo" else MCTS_TREE_PLOT_HTML_PATH
    optimizer_class = {
        "bo": MCTSVSBOOptimizer,
        "turbo": MCTSVSTuRBOOptimizer,
        "rs": MCTSVSRandomSearchOptimizer,
        "saasbo": MCTSVSSAASBOOptimizer,
    }[ipt_solver]
    uses_gp = ipt_solver != "rs"
    learn_noise = str(config.get("experiment_kind", "synthetic")) == "real_life"
    if ipt_solver in {"bo", "turbo"}:
        assumed_noise_std = BO_GP_ASSUMED_NOISE_STD
        assumed_noise_variance = BO_GP_ASSUMED_NOISE_VAR
    else:
        assumed_noise_variance = REFERENCE_DEFAULT_GP_NOISE_VAR
        assumed_noise_std = float(np.sqrt(assumed_noise_variance))
    return optimizer_class(
        f=recorder,
        lb=bounds[:, 0],
        ub=bounds[:, 1],
        feature_batch_size=MCTS_FEATURE_BATCH_SIZE,
        sample_batch_size=MCTS_SAMPLE_BATCH_SIZE,
        min_num_variables=3,
        select_right_threshold=5,
        cp=mcts_cp_for_benchmark(benchmark_name),
        k=MCTS_K,
        seed=seed,
        initial_points=initial_points,
        ipt_solver=ipt_solver,
        uipt_solver="bestk",
        split_type="mean",
        turbo_max_evals=MCTS_TURBO_MAX_EVALS,
        turbo_batch_size=MCTS_TURBO_BATCH_SIZE,
        turbo_use_ard=MCTS_TURBO_USE_ARD,
        turbo_max_cholesky_size=MCTS_TURBO_MAX_CHOLESKY_SIZE,
        turbo_n_training_steps=MCTS_TURBO_N_TRAINING_STEPS,
        turbo_device=normalize_device(device),
        turbo_min_cuda=MCTS_TURBO_MIN_CUDA,
        turbo_dtype=MCTS_TURBO_DTYPE,
        stopping_direction="max",
        noise_std=assumed_noise_std,
        noise_variance=assumed_noise_variance,
        learn_noise=learn_noise,
        stopping_threshold=None,
        enable_posterior_stopping=False,
        risk_tolerances=RISK_TOLERANCES,
        max_iter=max_iter,
        tree_plot_on=MCTS_TREE_SHOW,
        tree_plot_path=tree_plot_path,
        tree_plot_html_path=tree_plot_html_path,
        tree_plot_open_browser=AGVS_TREE_OPEN_BROWSER,
        tree_plot_pause=0.001,
        active_indices=config.get("active_indices"),
    )


def make_turbo(config, recorder, seed, max_iter, initial_points=None, device=DEVICE):
    if Turbo1 is None:
        raise ImportError(
            "TuRBO seçildi ama Turbo1 import edilemedi. "
            f"Original import error: {TURBO_IMPORT_ERROR}"
        )

    device = algorithm_device("TuRBO", device)
    preflight_algorithm_devices(["TuRBO"])
    bounds = np.asarray(config["bounds"], dtype=float)
    return Turbo1(
        f=recorder,
        lb=bounds[:, 0],
        ub=bounds[:, 1],
        n_init=TURBO_N_INIT,
        max_evals=max_iter,
        batch_size=TURBO_BATCH_SIZE,
        verbose=TURBO_VERBOSE,
        use_ard=TURBO_USE_ARD,
        max_cholesky_size=TURBO_MAX_CHOLESKY_SIZE,
        n_training_steps=TURBO_N_TRAINING_STEPS,
        min_cuda=TURBO_MIN_CUDA,
        device=device,
        dtype=TURBO_DTYPE,
        seed=seed,
        initial_points=initial_points,
        risk_tolerances=RISK_TOLERANCES,
        maximize=True,
        noise_variance=TURBO_GP_ASSUMED_NOISE_VAR,
        learn_noise=(
            str(config.get("experiment_kind", "synthetic")) == "real_life"
        ),
        stopping_threshold=None,
        enable_posterior_stopping=False,
    )


def make_random_search(config, recorder, seed, max_iter):
    return NativeRandomSearchOptimizer(
        objective=recorder,
        bounds=np.asarray(config["bounds"], dtype=float),
        max_iter=max_iter,
        seed=seed,
        noise_variance=REFERENCE_DEFAULT_GP_NOISE_VAR,
        risk_tolerances=RISK_TOLERANCES,
        stopping_threshold=None,
        enable_posterior_stopping=False,
        stopping_min_points=1,
    )


def make_cmaes(config, recorder, seed, max_iter, initial_points=None, benchmark_name=None):
    return OfficialCMAESOptimizer(
        objective=recorder,
        bounds=np.asarray(config["bounds"], dtype=float),
        max_iter=max_iter,
        initial_points=initial_points,
        seed=seed,
        sigma=cma_sigma_for_benchmark(benchmark_name),
        population_size=20,
        noise_variance=REFERENCE_DEFAULT_GP_NOISE_VAR,
        risk_tolerances=RISK_TOLERANCES,
        stopping_threshold=None,
        enable_posterior_stopping=False,
        stopping_min_points=3,
    )


def make_add_gp_ucb(config, recorder, seed, max_iter):
    if ADDGPUCBOptimizer is None:
        raise ImportError(
            "ADD-GP-UCB requires scipy.optimize.direct (SciPy >= 1.9) in its "
            "execution environment."
        ) from ADD_GP_IMPORT_ERROR
    return ADDGPUCBOptimizer(
        objective=recorder,
        bounds=np.asarray(config["bounds"], dtype=float),
        max_iter=max_iter,
        seed=seed,
        n_init=10,
        group_size=4,
        num_acq_evals=min(5000, 100 * int(np.asarray(config["bounds"]).shape[0])),
        hyperparameter_relearn_interval=25,
        decomposition_candidates=5,
        risk_tolerances=RISK_TOLERANCES,
        stopping_threshold=None,
        enable_posterior_stopping=False,
    )


MCTS_PAPER_ADAPTER_SETTINGS = {
    "Dropout-BO": {"method": "dropout-bo", "n_init": 10, "batch_size": 3},
    "Dropout-TuRBO": {"method": "dropout-turbo", "n_init": 10, "batch_size": 10},
    "Dropout-RS": {"method": "dropout-rs", "n_init": 10, "batch_size": 3},
    "LA-MCTS-BO": {"method": "la-mcts-bo", "n_init": 20, "batch_size": 1},
    "LA-MCTS-TuRBO": {"method": "la-mcts-turbo", "n_init": 20, "batch_size": 10},
    "SAASBO": {"method": "saasbo", "n_init": 10, "batch_size": 3},
    # Supplied HeSBO strategy: one signed projection, two initialization points.
    "HeSBO": {"method": "hesbo", "n_init": 2, "batch_size": 1},
    "ALEBO": {"method": "alebo", "n_init": 10, "batch_size": 1},
    "VAE-BO": {"method": "vae-bo", "n_init": 10, "batch_size": 3},
}

MCTS_PAPER_OPTIMIZER_CLASSES = {
    "Dropout-BO": DropoutBOOptimizer,
    "Dropout-TuRBO": DropoutTuRBOOptimizer,
    "Dropout-RS": DropoutRandomSearchOptimizer,
    "LA-MCTS-BO": LAMCTSBOOptimizer,
    "LA-MCTS-TuRBO": LAMCTSTuRBOOptimizer,
    "HeSBO": HeSBOOptimizer,
    "ALEBO": ALEBOOptimizer,
}

CONFIGURED_ACTIVE_DIMENSION_ALGORITHMS = {
    "Dropout-BO",
    "Dropout-TuRBO",
    "Dropout-RS",
    "HeSBO",
    "ALEBO",
    "VAE-BO",
}


def make_mcts_paper_baseline(
    algorithm,
    config,
    recorder,
    seed,
    max_iter,
    initial_points=None,
    benchmark_name=None,
    device=DEVICE,
):
    device = normalize_device(device)
    settings = MCTS_PAPER_ADAPTER_SETTINGS[algorithm]
    uses_gp = not str(settings["method"]).endswith("-rs")
    supports_project_posterior_stopping = False
    optimizer_class = MCTS_PAPER_OPTIMIZER_CLASSES.get(algorithm, NativeBaselineOptimizer)
    assumed_noise_variance = (
        BO_GP_ASSUMED_NOISE_VAR
        if algorithm in {"Dropout-BO", "VAE-BO"}
        else REFERENCE_DEFAULT_GP_NOISE_VAR
    )
    is_real_life = str(config.get("experiment_kind", "synthetic")) == "real_life"
    # The official MCTS-VS Ax embedding script supplies (value, 0.0) for both
    # HeSBO and ALEBO, including real-life objectives. Unknown-noise mode is
    # not merely a metadata difference in Ax 0.2.2: ALEBO receives NaN Yvars
    # and stops after its ten initialization trials.
    learn_noise = is_real_life and algorithm not in {"HeSBO", "ALEBO"}
    algorithm_specific_settings = {}
    if algorithm in {"HeSBO", "ALEBO"}:
        algorithm_specific_settings["experiment_name"] = str(
            benchmark_name or "registered_objective"
        )
    if algorithm == "VAE-BO":
        algorithm_specific_settings.update(
            {
                "vae_update_interval": vae_update_interval_for_benchmark(
                    benchmark_name
                ),
                "vae_learning_rate": vae_learning_rate_for_benchmark(
                    benchmark_name
                ),
            }
        )
    if algorithm in {"Dropout-TuRBO", "LA-MCTS-TuRBO"}:
        algorithm_specific_settings.update(
            {
                "turbo_max_evals": MCTS_TURBO_MAX_EVALS,
                "turbo_device": device,
                "turbo_min_cuda": MCTS_TURBO_MIN_CUDA,
            }
        )
    if algorithm in {"LA-MCTS-BO", "LA-MCTS-TuRBO"}:
        algorithm_specific_settings["la_mcts_cp"] = mcts_cp_for_benchmark(
            benchmark_name
        )
    active_dim = (
        optimizer_active_dimension(algorithm, config, benchmark_name)
        if algorithm in CONFIGURED_ACTIVE_DIMENSION_ALGORITHMS
        else int(np.asarray(config["bounds"]).shape[0])
    )
    return optimizer_class(
        objective=recorder,
        bounds=np.asarray(config["bounds"], dtype=float),
        max_iter=max_iter,
        initial_points=initial_points,
        seed=seed,
        active_dim=active_dim,
        noise_variance=assumed_noise_variance,
        learn_noise=learn_noise,
        candidate_count=1024,
        risk_tolerances=RISK_TOLERANCES,
        stopping_threshold=(
            POSTERIOR_STOPPING_THRESHOLD
            if supports_project_posterior_stopping
            else None
        ),
        enable_posterior_stopping=supports_project_posterior_stopping,
        **algorithm_specific_settings,
        **settings,
    )


def build_optimizer(
    algorithm,
    config,
    recorder,
    seed,
    max_iter,
    benchmark_name=None,
    trial=None,
    output_dir=None,
    device=DEVICE,
):
    algorithm = canonical_algorithm_name(algorithm)
    device = algorithm_device(algorithm, device)
    validate_algorithm_applicability(algorithm, config, benchmark_name)

    def _configured(optimizer):
        # Every optimizer receives the same runner-wide setting. Implementations
        # backed by NumPy/sklearn remain CPU-native; Torch/TuRBO paths consume
        # this value as their actual tensor device.
        optimizer.configured_device = device
        return optimizer

    def _make_agvs(**kwargs):
        return make_agvs(
            config,
            recorder,
            seed,
            max_iter,
            initial_points=None,
            device=device,
            **kwargs,
        )

    if algorithm == "VanillaBO":
        return _configured(make_vanilla_bo(
            config,
            recorder,
            seed,
            max_iter,
            benchmark_name=benchmark_name,
            trial=trial,
            output_dir=output_dir,
            initial_points=None,
        ))
    if algorithm in AGVS_ALGORITHMS:
        return _configured(_make_agvs(**AGVS_VARIANT_CONFIG[algorithm]))
    if algorithm == "MCTS-VS-BO":
        return _configured(make_mcts_vs(config, recorder, seed, max_iter, ipt_solver="bo", initial_points=None, benchmark_name=benchmark_name, device=device))
    if algorithm == "MCTS-VS-TuRBO":
        return _configured(make_mcts_vs(config, recorder, seed, max_iter, ipt_solver="turbo", initial_points=None, benchmark_name=benchmark_name, device=device))
    if algorithm == "MCTS-VS-RS":
        return _configured(make_mcts_vs(config, recorder, seed, max_iter, ipt_solver="rs", initial_points=None, benchmark_name=benchmark_name, device=device))
    if algorithm == "MCTS-VS-SAASBO":
        return _configured(make_mcts_vs(config, recorder, seed, max_iter, ipt_solver="saasbo", initial_points=None, benchmark_name=benchmark_name, device=device))
    if algorithm == "TuRBO":
        return _configured(make_turbo(config, recorder, seed, max_iter, initial_points=None, device=device))
    if algorithm == "RandomSearch":
        return _configured(make_random_search(config, recorder, seed, max_iter))
    if algorithm == "CMA-ES":
        return _configured(make_cmaes(config, recorder, seed, max_iter, initial_points=None, benchmark_name=benchmark_name))
    if algorithm == "ADD-GP-UCB":
        return _configured(make_add_gp_ucb(config, recorder, seed, max_iter))
    if algorithm == "GTBO":
        return _configured(GTBOOptimizer(
            objective=recorder, bounds=config["bounds"], max_iter=max_iter, seed=seed,
            experiment_kind=config.get("experiment_kind", "synthetic"),
            objective_name=config.get("objective", str(benchmark_name).split("_")[0].rstrip("0123456789")),
            default=config.get("default_point"),
        ))
    if algorithm == "LassoBO":
        return _configured(LassoBOOptimizer(
            objective=recorder,
            bounds=np.asarray(config["bounds"], dtype=float),
            max_iter=max_iter,
            initial_points=None,
            seed=seed,
            n_init=30,
            lasso_lambda=1e-3,
            subspace_root=3,
        ))
    if algorithm in MCTS_PAPER_ADAPTER_SETTINGS:
        return _configured(make_mcts_paper_baseline(
            algorithm,
            config,
            recorder,
            seed,
            max_iter,
            initial_points=None,
            benchmark_name=benchmark_name,
            device=device,
        ))
    raise ValueError(f"Unknown algorithm: {algorithm}")


def run_optimizer(algorithm, optimizer, recorder, max_iter):
    algorithm = canonical_algorithm_name(algorithm)

    if algorithm == "VanillaBO":
        return optimizer.optimize(max_iter=max_iter)

    if algorithm in AGVS_ALGORITHMS:
        return optimizer.optimize()

    if algorithm in {"MCTS-VS-BO", "MCTS-VS-TuRBO"}:
        with suppress_all_prints():
            return optimizer.optimize(max_samples=max_iter, verbose=False, black_box_function=recorder)

    if algorithm in {"MCTS-VS-RS", "MCTS-VS-SAASBO"}:
        with suppress_all_prints():
            return optimizer.optimize(max_samples=max_iter, verbose=False, black_box_function=recorder)

    if algorithm == "TuRBO":
        return optimizer.optimize()

    if algorithm in {"RandomSearch", "CMA-ES", "ADD-GP-UCB"}:
        return optimizer.optimize()
    if algorithm in {"LassoBO", "GTBO"}:
        return optimizer.optimize()
    if algorithm in MCTS_PAPER_ADAPTER_SETTINGS:
        return optimizer.optimize()

    raise ValueError(f"Unknown algorithm: {algorithm}")


def sync_agvs_result_to_recorder(recorder, result, wall_clock):
    X = np.asarray(result.X, dtype=float)
    true_y = np.asarray(result.true_y, dtype=float).reshape(-1)
    observed_y = np.asarray(result.y, dtype=float).reshape(-1)

    recorder.x_history = [np.asarray(x, dtype=float).copy() for x in X]
    recorder.true_history = [float(y) for y in true_y]
    recorder.observed_history = [float(y) for y in observed_y]
    n_evaluations = int(observed_y.size)
    if n_evaluations <= 0:
        recorder.eval_wall_clock_history = []
        return

    raw_recorded_times = getattr(recorder, "eval_wall_clock_history", None)
    if raw_recorded_times is None:
        raw_recorded_times = []
    recorded_times = np.asarray(raw_recorded_times, dtype=float).reshape(-1)
    if recorded_times.size >= n_evaluations:
        recorder.eval_wall_clock_history = recorded_times[:n_evaluations].tolist()
        return

    if recorded_times.size == 0:
        recorder.eval_wall_clock_history = np.linspace(
            float(wall_clock) / n_evaluations,
            float(wall_clock),
            n_evaluations,
        ).tolist()
        return

    # Preserve every real timestamp and synthesize only the missing suffix.
    final_time = max(float(wall_clock), float(recorded_times[-1]))
    suffix = np.linspace(
        float(recorded_times[-1]),
        final_time,
        n_evaluations - int(recorded_times.size) + 1,
    )[1:]
    recorder.eval_wall_clock_history = np.concatenate([recorded_times, suffix]).tolist()


def _trace_q_batch(trace_or_settings, default=1):
    if not isinstance(trace_or_settings, dict):
        return int(default)
    candidates = [
        trace_or_settings.get("batch_size"),
        trace_or_settings.get("q_batch"),
        trace_or_settings.get("q_batch_per_embedding"),
    ]
    algorithm_settings = trace_or_settings.get("algorithm_settings", {})
    if isinstance(algorithm_settings, dict):
        candidates.extend(
            [
                algorithm_settings.get("batch_size"),
                algorithm_settings.get("q_batch"),
                algorithm_settings.get("q_batch_per_embedding"),
            ]
        )
    settings = trace_or_settings.get("settings", {})
    if isinstance(settings, dict):
        candidates.extend(
            [
                settings.get("batch_size"),
                settings.get("q_batch"),
                settings.get("q_batch_per_embedding"),
            ]
        )
        algorithm_settings = settings.get("algorithm_settings", {})
        if isinstance(algorithm_settings, dict):
            candidates.extend(
                [
                    algorithm_settings.get("batch_size"),
                    algorithm_settings.get("q_batch"),
                    algorithm_settings.get("q_batch_per_embedding"),
                ]
            )

    for value in candidates:
        if isinstance(value, (int, np.integer)) and int(value) > 0:
            return int(value)
        if isinstance(value, float) and np.isfinite(value) and int(value) > 0:
            return int(value)
        if isinstance(value, str) and value.strip().isdigit() and int(value.strip()) > 0:
            return int(value.strip())
    return int(default)


def _batch_sizes_from_result(algorithm, result, n_evaluations):
    history = _history_from_result(algorithm, result)
    sizes = []

    if algorithm in AGVS_ALGORITHMS and not history:
        return [1] * max(0, int(n_evaluations))

    if algorithm == "VanillaBO" and isinstance(history, list):
        previous_n = 0
        for record in history:
            if not isinstance(record, dict):
                continue
            n_eval = record.get("n_eval")
            if isinstance(n_eval, (int, np.integer, float)) and np.isfinite(float(n_eval)):
                size = int(n_eval) - previous_n
                previous_n = int(n_eval)
            else:
                y_batch = record.get("y_batch", [])
                size = int(np.asarray(y_batch).reshape(-1).size)
            if size > 0:
                sizes.append(size)

    elif isinstance(history, list):
        seen_batches = set()
        for record in history:
            if not isinstance(record, dict):
                continue
            if not any(
                key in record
                for key in ("optimizer_point_batch_size", "batch_size", "q_batch")
            ):
                continue
            batch_size = record.get(
                "optimizer_point_batch_size",
                record.get("batch_size", record.get("q_batch", 1)),
            )
            if isinstance(batch_size, (int, np.integer, float)) and np.isfinite(float(batch_size)):
                batch_size = max(1, int(batch_size))
            else:
                batch_size = 1
            if batch_size <= 1:
                sizes.append(1)
                continue
            key = (
                record.get("batch_iteration", record.get("q_iteration")),
                record.get("sequence_index"),
                batch_size,
            )
            if key in seen_batches:
                continue
            seen_batches.add(key)
            sizes.append(batch_size)

    total = int(sum(sizes))
    if total > int(n_evaluations):
        trimmed = []
        remaining = int(n_evaluations)
        for size in sizes:
            if remaining <= 0:
                break
            trimmed_size = min(int(size), remaining)
            if trimmed_size > 0:
                trimmed.append(trimmed_size)
            remaining -= trimmed_size
        sizes = trimmed
    elif total < int(n_evaluations):
        remaining = int(n_evaluations) - total
        if sizes:
            sizes = [1] * remaining + sizes

    return sizes


def _history_from_result(algorithm, result):
    if isinstance(result, dict):
        for key in [
            "history",
            "agvs_history",
            "vanilla_bo_history",
            "turbo_history",
            "mcts_history",
            "mcts_vs_history",
        ]:
            history = result.get(key)
            if isinstance(history, list):
                return history
        for value in result.values():
            if (
                isinstance(value, list)
                and value
                and isinstance(value[0], dict)
                and any(
                    key in value[0]
                    for key in ("elapsed_sec", "batch_size", "q_batch", "n_eval", "y_batch")
                )
            ):
                return value
        return None
    return getattr(result, "history", None)


def _elapsed_wall_clock_times_from_history(algorithm, result, n_evaluations, wall_clock):
    history = _history_from_result(algorithm, result)
    if not isinstance(history, list):
        return None

    elapsed = []
    for record in history:
        if not isinstance(record, dict):
            continue
        value = record.get("elapsed_sec")
        try:
            value = float(value)
        except Exception:
            continue
        if np.isfinite(value) and value >= 0:
            elapsed.append(value)
    if not elapsed:
        return None

    elapsed = np.asarray(elapsed, dtype=float)
    elapsed = elapsed[: max(0, int(n_evaluations))]
    missing_initial = max(0, int(n_evaluations) - int(elapsed.size))
    if missing_initial > 0:
        first_elapsed = float(elapsed[0])
        if first_elapsed > 0:
            prefix = np.linspace(
                first_elapsed / (missing_initial + 1),
                first_elapsed * missing_initial / (missing_initial + 1),
                missing_initial,
            )
        else:
            prefix = np.zeros(missing_initial, dtype=float)
        elapsed = np.concatenate([prefix, elapsed])

    if elapsed.size < int(n_evaluations):
        fill = np.linspace(
            float(elapsed[-1]),
            float(wall_clock),
            int(n_evaluations) - int(elapsed.size) + 1,
        )[1:]
        elapsed = np.concatenate([elapsed, fill])
    elif elapsed.size > int(n_evaluations):
        elapsed = elapsed[: int(n_evaluations)]

    return elapsed


def batch_parallel_wall_clock_history(raw_times, algorithm, result, n_evaluations, wall_clock, q_batch):
    n_evaluations = int(n_evaluations)
    if n_evaluations <= 0:
        return []

    times = _elapsed_wall_clock_times_from_history(algorithm, result, n_evaluations, wall_clock)
    if times is None:
        times = np.asarray(raw_times, dtype=float).reshape(-1)
    if times.size < n_evaluations:
        if times.size > 0:
            fill = np.linspace(float(times[-1]), float(wall_clock), n_evaluations - times.size + 1)[1:]
            times = np.concatenate([times, fill])
        else:
            times = np.linspace(float(wall_clock) / n_evaluations, float(wall_clock), n_evaluations)
    times = times[:n_evaluations]

    sizes = _batch_sizes_from_result(algorithm, result, n_evaluations)
    if not sizes:
        q = max(1, int(q_batch))
        sizes = [min(q, n_evaluations - start) for start in range(0, n_evaluations, q)]

    batch_times = np.empty(n_evaluations, dtype=float)
    cursor = 0
    for size in sizes:
        size = int(size)
        if size <= 0 or cursor >= n_evaluations:
            continue
        end = min(cursor + size, n_evaluations)
        batch_end_time = float(times[end - 1])
        batch_times[cursor:end] = batch_end_time
        cursor = end

    if cursor < n_evaluations:
        q = max(1, int(q_batch))
        while cursor < n_evaluations:
            end = min(cursor + q, n_evaluations)
            batch_times[cursor:end] = float(times[end - 1])
            cursor = end

    return batch_times.tolist()


def posterior_threshold_history_by_delta(result, n_evaluations):
    traces_by_n = {}

    threshold_trace = getattr(result, "threshold_trace", None)
    if isinstance(threshold_trace, list):
        for entry in threshold_trace:
            if not isinstance(entry, dict):
                continue
            n_eval = int(entry.get("n", 0))
            if n_eval > 0:
                # The explicit trace is authoritative. If it contains multiple
                # snapshots for one endpoint, retain the last completed one.
                traces_by_n[n_eval] = entry

    history = getattr(result, "history", None)
    if isinstance(history, list):
        for entry in history:
            if not isinstance(entry, dict):
                continue
            info = entry.get("stopping_info")
            if isinstance(info, dict) and "best_stoppable_threshold_by_delta" in info:
                n_eval = int(info.get("n", entry.get("n_eval", 0)))
                if n_eval > 0 and n_eval not in traces_by_n:
                    traces_by_n[n_eval] = {
                        "n": n_eval,
                        "current_stoppable_threshold_by_delta": info.get(
                            "current_stoppable_threshold_by_delta",
                            {},
                        ),
                        "best_stoppable_threshold_by_delta": info["best_stoppable_threshold_by_delta"],
                    }

    if not traces_by_n:
        return None

    traces = [traces_by_n[n_eval] for n_eval in sorted(traces_by_n)]
    out = {risk_label(delta): [] for delta in RISK_TOLERANCES}
    last = {risk_label(delta): None for delta in RISK_TOLERANCES}
    previous_n = 0

    for entry in traces:
        n_eval = max(previous_n, int(entry.get("n", previous_n)))
        gap_size = max(0, n_eval - previous_n - 1)
        for delta in RISK_TOLERANCES:
            label = risk_label(delta)
            out[label].extend([last[label]] * gap_size)
        current_thresholds = entry.get("current_stoppable_threshold_by_delta", {})
        best_thresholds = entry.get("best_stoppable_threshold_by_delta", {})
        for delta in RISK_TOLERANCES:
            label = risk_label(delta)
            key = f"{float(delta):.12g}"
            current_value = current_thresholds.get(key, current_thresholds.get(label))
            if current_value is not None and np.isfinite(float(current_value)):
                last[label] = (
                    float(current_value)
                    if last[label] is None
                    else max(float(last[label]), float(current_value))
                )
            else:
                best_value = best_thresholds.get(key, best_thresholds.get(label))
                if best_value is not None and np.isfinite(float(best_value)):
                    last[label] = (
                        float(best_value)
                        if last[label] is None
                        else max(float(last[label]), float(best_value))
                    )
            out[label].append(last[label])
        previous_n = n_eval

    for delta in RISK_TOLERANCES:
        label = risk_label(delta)
        if len(out[label]) < int(n_evaluations):
            out[label].extend([last[label]] * (int(n_evaluations) - len(out[label])))
        out[label] = out[label][: int(n_evaluations)]
        out[label] = [
            None
            if value is None or not np.isfinite(float(value))
            else float(value)
            for value in out[label]
        ]

    return out


def _algorithm_settings_summary_impl(algorithm, benchmark_name=None, device=DEVICE):
    algorithm = canonical_algorithm_name(algorithm)
    device = normalize_device(device)

    if algorithm == "VanillaBO":
        return {
            "paper_style": "MCTS-VS official Vanilla BO baseline",
            "source": (
                "Experiment/benchmark/baseline/mcts_vs_official/baseline/vanilia_bo.py "
                "via Experiment/benchmark/baseline/vanilla_bo_adapter.py"
            ),
            **bo_reference_dict(),
            "n_init": MCTS_PAPER_BO_N_INIT,
            "batch_size": MCTS_PAPER_BO_BATCH_SIZE,
            "initial_sampling": "independent LHS (Appendix C / official implementation)",
            "acquisition_optimizer": "exhaustive ranking of the Sobol candidate pool",
            "risk_tolerances": RISK_TOLERANCES,
            "stoppable_threshold_graph": "For each risk tolerance, plot the highest threshold certified so far by GP posterior mu + sigma * Phi^{-1}(risk_tolerance).",
            "dimension_importance": "top-10 smallest ARD lengthscales written after every BO iteration",
        }

    if algorithm in AGVS_ALGORITHMS:
        variant_config = AGVS_VARIANT_CONFIG[algorithm]
        node_backend = variant_config.get("node_backend", "vanillabo")
        registered_config = BENCHMARKS.get(str(benchmark_name or "").lower())
        q_batch = benchmark_q_batch_by_name(benchmark_name)
        batch_size = benchmark_batch_size_by_name(benchmark_name)
        t_seq = (
            int(registered_config.get("t_seq", AGVS_T_SEQ))
            if registered_config is not None
            else AGVS_T_SEQ
        )
        point_batch_size = q_batch if q_batch is not None else batch_size
        n_init = benchmark_n_init(registered_config) if registered_config is not None else AGVS_N_INIT
        effective_dim = (
            benchmark_agvs_effective_dim(registered_config)
            if registered_config is not None
            else D_EMBEDDING
        )
        gp_noise_variance = benchmark_agvs_gp_noise_variance(registered_config)
        gp_noise_std = float(math.sqrt(gp_noise_variance))
        merge_enabled = benchmark_agvs_merge_enabled(registered_config)
        merge_metadata = (
            {
                "merge_enabled": merge_enabled,
                "merge_policy": (
                    "score every split cohort, merge retained siblings, then continue splitting"
                    if merge_enabled
                    else "keep every scored child separate; repeatedly select and split the highest-scoring active child"
                ),
            }
            if registered_config is not None
            and "agvs_merge_enabled" in registered_config
            else {}
        )
        infrastructure = (
            bo_reference_dict()
            if node_backend == "vanillabo"
            else turbo_reference_dict(device=device)
        )
        if node_backend == "vanillabo":
            infrastructure.update(
                {
                    "alpha": gp_noise_variance,
                    "noise_variance": gp_noise_variance,
                    "observation_noise_std": gp_noise_std,
                }
            )
        if node_backend == "turbo":
            infrastructure.update(
                {
                    "min_cuda": AGVS_TURBO_MIN_CUDA,
                    "device": device,
                    "dtype": AGVS_TURBO_DTYPE,
                    "small_data_runtime": "CUDA float32 from the first local GP fit (min_cuda=0)",
                    "gp_kernel": "ScaleKernel(MaternKernel(nu=2.5, ARD=False))",
                    "use_ard": False,
                    "anisotropic": False,
                    "isotropic": True,
                    "internal_batch_size": point_batch_size,
                    "turbo_backend_batch_policy": (
                        "no independent TuRBO batch setting; the proposal count is "
                        "supplied by the AG-VS scheduler B"
                    ),
                    "reference_internal_batch_size": TURBO_REFERENCE_SPEC.internal_batch_size,
                    "final_batch_budget_rule": (
                        f"AG-VS requests B={point_batch_size} Thompson candidates and evaluates only "
                        "the remaining budget-safe prefix in the terminal batch"
                    ),
                }
            )
        return {
            "paper_style": "Adaptive Grouping and Variable Selection (AG-VS) Bayesian optimization",
            **infrastructure,
            "q_batch": q_batch,
            "q_batch_policy": (
                "active joint-acquisition mode from runner central controls"
                if q_batch is not None
                else "disabled by runner central controls; batch_size mode is active"
            ),
            "batch_size": batch_size,
            "selection_batch_size": batch_size,
            "acquisition_batch_mode": (
                "q_batch" if q_batch is not None else "batch_size"
            ),
            "batch_parameter_policy": "exactly one of q_batch or batch_size is non-null",
            "t_seq": t_seq,
            "node_visit_policy": (
                f"select one node, execute t_seq={t_seq} AG-VS stages with B={point_batch_size}, "
                "then traverse the tree again to select the next node"
            ),
            "n_init": n_init,
            "effective_dim": effective_dim,
            "abs_mode": bool(AGVS_ABS_MODE),
            "score_formula": (
                "mean(abs(y_obs_i - best_y_obs_i_minus_1))"
                if AGVS_ABS_MODE
                else "mean(y_obs_i - best_y_obs_i_minus_1)"
            ),
            "score_value_source": "observations; no posterior prediction",
            "node_backend": node_backend,
            **merge_metadata,
            "variable_selection": "AG-VS adaptive grouping/variable-selection score",
            "split_policy": (
                "canonical merged-leaf mechanism"
                if merge_enabled
                else "recursive highest-score child splitting with sibling merging disabled"
            ),
            "gp_sharing_policy": (
                "one node-local isotropic GP used by acquisition; dimension groups are "
                "random and the node score uses observations"
            ),
            "gp_refit_policy": (
                "fit a stale node immediately before acquisition when it is selected; "
                "no trailing refit after its point batch"
            ),
            "split_surrogate_family": "none; splitting does not use a GP",
            "split_kernel": "not applicable",
            "split_ard": False,
            "split_gp_role": "seeded random permutation; no GP-based dimension ranking",
            "dimension_split_policy": "seeded random permutation into balanced groups",
            "acquisition_optimizer": (
                "exhaustive ranking of the scrambled-Sobol candidate pool"
                if node_backend == "vanillabo"
                else "native TuRBO trust-region batch Thompson selection"
            ),
            "acquisition": (
                "classical analytic Expected Improvement"
                if node_backend == "vanillabo"
                else "GP posterior Thompson sampling"
            ),
            "candidate_selection": (
                "top-k independently scored pointwise analytic EI candidates"
                if node_backend == "vanillabo"
                else f"native TuRBO trust-region Thompson batch of {batch_size} points"
            ),
            "surrogate_family": (
                "sklearn.gaussian_process.GaussianProcessRegressor"
                if node_backend == "vanillabo"
                else infrastructure.get("surrogate_family")
            ),
            "kernel": (
                "ConstantKernel(1.0) * Matern(shared length_scale, nu=2.5)"
                if node_backend == "vanillabo"
                else infrastructure.get("gp_kernel")
            ),
            "ard": False,
            "isotropic": True,
            "noise": (
                f"alpha={gp_noise_variance:g} (observation noise std={gp_noise_std:g})"
                if node_backend == "vanillabo"
                else infrastructure.get("noise")
            ),
            "hyperparameter_fitting": (
                "marginal likelihood (sklearn fmin_l_bfgs_b)"
                if node_backend == "vanillabo"
                else infrastructure.get("hyperparameter_fitting")
            ),
            "candidate_generation": (
                "1024 scrambled-Sobol candidates"
                if node_backend == "vanillabo"
                else infrastructure.get("candidate_generation")
            ),
            "candidate_seed_policy": (
                "fresh scrambled-Sobol engine per acquisition; seed drawn from "
                "the advancing trial-owned NumPy Generator"
                if node_backend == "vanillabo"
                else None
            ),
        }

    if algorithm == "MCTS-VS-BO":
        return {
            "paper_style": "MCTS-VS Coordinate tree selection BO with Vanilla BO leaf optimization",
            **bo_reference_dict(),
            "n_init": 2 * MCTS_FEATURE_BATCH_SIZE * MCTS_SAMPLE_BATCH_SIZE,
            "initial_sampling": "independently generated LHS points per feature mask",
            "batch_size": MCTS_SAMPLE_BATCH_SIZE,
            "cp": mcts_cp_for_benchmark(benchmark_name),
            "feature_batch_size": MCTS_FEATURE_BATCH_SIZE,
            "sample_batch_size": MCTS_SAMPLE_BATCH_SIZE,
            "min_num_variables": 3,
            "select_right_threshold": 5,
            "k": MCTS_K,
            "split_type": "mean",
            "uipt_solver": "bestk",
            "ipt_solver": "bo",
            "acquisition_optimizer": "exhaustive ranking of the Sobol candidate pool",
        }

    if algorithm == "MCTS-VS-TuRBO":
        return {
            "paper_style": "MCTS-VS Coordinate tree selection BO with TuRBO local optimization",
            **turbo_reference_dict(device=device),
            "min_cuda": MCTS_TURBO_MIN_CUDA,
            "device": device,
            "dtype": MCTS_TURBO_DTYPE,
            "small_data_runtime": "CUDA float32 from the first local GP fit (min_cuda=0)",
            "n_init": 2 * MCTS_FEATURE_BATCH_SIZE * MCTS_SAMPLE_BATCH_SIZE,
            "initial_sampling": "independently generated LHS points per feature mask",
            "cp": mcts_cp_for_benchmark(benchmark_name),
            "feature_batch_size": MCTS_FEATURE_BATCH_SIZE,
            "sample_batch_size": MCTS_SAMPLE_BATCH_SIZE,
            "min_num_variables": 3,
            "select_right_threshold": 5,
            "k": MCTS_K,
            "split_type": "mean",
            "uipt_solver": "bestk",
            "ipt_solver": "turbo",
        }

    if algorithm == "MCTS-VS-RS":
        return {
            "paper_style": "MCTS-VS variable selection with uniform random inner optimization",
            "source": "Experiment/benchmark/baseline/mcts_vs_official",
            "n_init": 2 * MCTS_FEATURE_BATCH_SIZE * MCTS_SAMPLE_BATCH_SIZE,
            "initial_sampling": "independently generated LHS points per feature mask",
            "cp": mcts_cp_for_benchmark(benchmark_name),
            "feature_batch_size": MCTS_FEATURE_BATCH_SIZE,
            "sample_batch_size": MCTS_SAMPLE_BATCH_SIZE,
            "min_num_variables": 3,
            "select_right_threshold": 5,
            "k": MCTS_K,
            "split_type": "mean",
            "uipt_solver": "bestk",
            "ipt_solver": "rs",
            "acquisition": "none",
            "acquisition_sampling": "uniform random on selected coordinates",
            "kernel": None,
        }

    if algorithm == "MCTS-VS-SAASBO":
        return {
            "paper_style": "MCTS-VS variable selection with SAASBO inner optimization",
            "source": "Experiment/benchmark/baseline/mcts_vs_official",
            "n_init": 2 * MCTS_FEATURE_BATCH_SIZE * MCTS_SAMPLE_BATCH_SIZE,
            "initial_sampling": "independently generated LHS points per feature mask",
            "cp": mcts_cp_for_benchmark(benchmark_name),
            "feature_batch_size": MCTS_FEATURE_BATCH_SIZE,
            "sample_batch_size": MCTS_SAMPLE_BATCH_SIZE,
            "min_num_variables": 3,
            "select_right_threshold": 5,
            "k": MCTS_K,
            "split_type": "mean",
            "uipt_solver": "bestk",
            "ipt_solver": "saasbo",
            "acquisition": "classical analytic Expected Improvement",
            "candidate_count": 1024,
            "batch_size": MCTS_SAMPLE_BATCH_SIZE,
            "acquisition_sampling": "1024 scrambled Sobol candidates ranked by EI; select best 3",
            "candidate_selection": "top-3 pointwise classical EI values",
            "acquisition_optimizer": "exhaustive ranking of the Sobol candidate pool",
            "kernel": "SAAS RBF",
        }

    if algorithm == "ADD-GP-UCB":
        return {
            "paper_style": "Additive Gaussian Process UCB (Kandasamy et al., ICML 2015)",
            "source": "Experiment/benchmark/baseline/add_gp_ucb.py; official MATLAB source retained under add_gp_bandits_official",
            "n_init": 10,
            "batch_size": 1,
            "initial_sampling": "10 uniform random points (official implementation)",
            "group_size": 4,
            "decomposition_strategy": (
                "partialLearn: orthogonal provisional DIRECT fit; score five random "
                "permutations; DIRECT-refit the selected decomposition"
            ),
            "decomposition_candidates": 5,
            "acquisition": "ucb",
            "ucb_beta": "0.2 * d_tilde * log(2*t)",
            "candidate_generation": "independent DIRECT optimization for each additive component",
            "candidate_selection": "concatenate independent per-component UCB maximizers",
            "acquisition_optimizer": "DIRECT per additive component",
            "direct_budget": "min(5000, 100*D) per acquisition round",
            "surrogate_family": "additive Gaussian process with independent component GPs",
            "kernel": "additive sum of component RBF kernels",
            "hyperparameter_fitting": (
                "component RBF bandwidth/kernel scale by marginal likelihood; refit "
                "every 25 iterations or after five consecutive selected-point combined "
                "posterior std values below 0.002"
            ),
            "hyperparameter_relearn_interval": 25,
            "bandwidth_bounds": [1e-5, 5.0],
            "kernel_scale_bounds": "[0.03, 30] * std(initial objective values)",
            "low_variance_relearn_trigger": "5 consecutive combined posterior std < 0.002",
            "early_relearn_bandwidth_rule": "upper=max(1e-5, 0.9*current bandwidth)",
            "mean_model": "combined posterior mean(y); zero mean for each component posterior",
            "duplicate_avoidance": (
                "while normalized distance < 1e-10, add 0.1*bandwidth Gaussian "
                "perturbation and project to bounds"
            ),
            "noise": "commonNoise = 0.01 * std(initial objective values)",
        }

    if algorithm == "GTBO":
        return {**GTBO_SETTINGS, "implementation": "native-v1",
                "source": "gtboauthors/gtbo@fe3dca9d179e46b4d9cc7bc6954b09e1078919e8",
                "kernel": "Mat?rn-5/2 ARD",
                "default_point_policy": "benchmark default; else midpoint; centered-optimum synthetic: normalized 0.25",
                "gp_fit_recovery_policy": (
                    "fit cached learned-noise GP; after failure reset cached state "
                    "and refit; if that also fails use a freshly initialized GP "
                    "for one acquisition"
                ),
                "inactive_prior_parameters": (3.0, 1.0) if str(benchmark_name).startswith("nas") else (7.0, 1.0)}
    if algorithm == "LassoBO":
        return {
            "paper_style": "LassoBO (Hoang, Tran, Gupta, and Nguyen; AISTATS 2025)",
            "source": (
                "Experiment/benchmark/baseline/lassobo_adapter.py; algorithm and "
                "published settings from arXiv:2504.01743; unspecified optimizer "
                "details from github.com/Fsoft-AIC/LassoBO"
            ),
            "n_init": 30,
            "batch_size": 1,
            "initial_sampling": "30-point Latin hypercube design",
            "surrogate_family": "exact all-D Gaussian process",
            "kernel": "ARD squared-exponential (RBF)",
            "variable_importance": "inverse GP length scale",
            "important_dimension_rule": "rho_i above the all-dimension mean (Section 3.1)",
            "lasso_lambda": 1e-3,
            "subspace_count": "M_t=ceil(t^(1/3))",
            "unimportant_dimension_imputation": (
                "M_t uniform backgrounds plus the best observed point"
            ),
            "acquisition": "GP-UCB",
            "ucb_beta": "Uniform[0.75, 1.5] per iteration (official code detail)",
            "acquisition_optimizer": "BoTorch optimize_acqf",
            "num_restarts": 8,
            "raw_samples": 1000,
            "gp_optimizer": "Adam(lr=0.1), at most 250 steps",
            "gp_lr_scheduler": "ExponentialLR(gamma=0.9) after non-improvement",
            "gp_early_stop_patience": 33,
            "paper_code_discrepancy": (
                "paper defaults lambda=1e-3 and dynamic M_t; public code contains "
                "1e-4 and fixed M=5, so this adapter follows the publication"
            ),
            "reference_repetitions": 10,
            "project_repetitions": TRIALS,
        }

    if algorithm == "TuRBO":
        return {
            "paper_style": "TuRBO local trust-region Bayesian optimization",
            "implementation": "project-owned TuRBO-1 derived from uber-research/TuRBO",
            "surrogate_family": "GPyTorch ExactGP",
            "gp_library": "GPyTorch on PyTorch",
            "n_init": TURBO_N_INIT,
            "n_init_policy": "MCTS-VS official baseline/turbo.py: fixed 20-point LHS",
            "batch_size": TURBO_BATCH_SIZE,
            "use_ard": TURBO_USE_ARD,
            "internal_batch_size": TURBO_BATCH_SIZE,
            "n_training_steps": TURBO_N_TRAINING_STEPS,
            "max_cholesky_size": TURBO_MAX_CHOLESKY_SIZE,
            "min_cuda": TURBO_MIN_CUDA,
            "device": device,
            "dtype": TURBO_DTYPE,
            "small_data_runtime": (
                "project TuRBO-backend CUDA policy: GPU float32 from the first GP fit "
                "(min_cuda=0)"
            ),
            "initial_sampling": "independent LHS (TuRBO default)",
            "acquisition": "GP posterior Thompson sampling",
            "candidate_generation": "TuRBO trust-region scrambled Sobol perturbations",
            "candidate_selection": "GP posterior Thompson sampling",
            "gp_kernel": "ScaleKernel(MaternKernel(nu=2.5, ARD=True))",
            "trust_region_logic": "native TuRBO-1 expansion, contraction, and restart",
            "maximize": True,
            "noise": (
                "native TuRBO GaussianLikelihood noise learned jointly with GP "
                "hyperparameters; standardized constraint [5e-4, 0.2]"
            ),
            "posterior_reporting": (
                "posterior stopping/threshold reporting disabled; no terminal reporting-only fit"
            ),
            "auxiliary_sklearn_gp_fits": 0,
            "internal_objective": "-objective is minimized internally so experiment traces remain in maximize convention",
            "trust_region_center": "training point with best observed standardized response",
        }

    if algorithm == "RandomSearch":
        return {
            "paper_style": "MCTS-VS official random-search baseline",
            "source": (
                "Experiment/benchmark/baseline/mcts_vs_official/baseline/random_search.py "
                "via Experiment/benchmark/baseline/native_optimizers.py"
            ),
            "n_init": 0,
            "initial_sampling": "not separate",
            "acquisition": "none",
            "acquisition_sampling": "uniform random",
            "kernel": None,
        }

    if algorithm == "CMA-ES":
        return {
            "paper_style": "MCTS-VS official CMA-ES baseline",
            "source": (
                "Experiment/benchmark/baseline/mcts_vs_official/baseline/cmaes.py "
                "via Experiment/benchmark/baseline/native_optimizers.py"
            ),
            "n_init": 0,
            "initial_sampling": (
                "no separately evaluated initial design; uniformly sampled CMA center"
            ),
            "acquisition": "none",
            "acquisition_sampling": "CMA multivariate Gaussian population",
            "kernel": None,
            "population_size": 20,
            "batch_size": 20,
            "sigma": cma_sigma_for_benchmark(benchmark_name),
        }

    if algorithm in MCTS_PAPER_ADAPTER_SETTINGS:
        settings = MCTS_PAPER_ADAPTER_SETTINGS[algorithm]
        method = settings["method"]
        is_turbo = method.endswith("turbo")
        is_random = method.endswith("-rs")
        registered_config = BENCHMARKS.get(str(benchmark_name or "").lower())
        configured_d_eff = (
            benchmark_effective_dim(registered_config)
            if registered_config is not None
            else effective_dim_from_benchmark_name(benchmark_name)
        )
        if is_random:
            acquisition = "none"
        elif is_turbo:
            acquisition = "GP posterior Thompson sampling"
        else:
            acquisition = "classical analytic Expected Improvement"
        if is_random:
            kernel = None
        elif is_turbo:
            kernel = "ARD Matern-5/2"
        elif algorithm == "SAASBO":
            kernel = "SAAS RBF"
        elif algorithm == "ALEBO":
            kernel = "ALEBO embedded Mahalanobis/ARD GP"
        else:
            kernel = "Matern-5/2"
        uses_configured_active_dim = (
            algorithm in CONFIGURED_ACTIVE_DIMENSION_ALGORITHMS
        )
        result = {
            "paper_style": f"MCTS-VS paper baseline: {algorithm}",
            "source": "baseline",
            "n_init": int(settings["n_init"]),
            "batch_size": int(settings["batch_size"]),
            "initial_sampling": (
                "native embedded initialization" if algorithm in {"HeSBO", "ALEBO"} else "independent LHS"
            ),
            "acquisition": acquisition,
            "acquisition_sampling": (
                "uniform random" if is_random else (
                    "TuRBO trust-region scrambled Sobol perturbations"
                    if is_turbo
                    else "reference acquisition candidate generation"
                )
            ),
            "candidate_selection": (
                "uniform random"
                if is_random
                else "GP posterior Thompson sampling"
                if is_turbo
                else "highest classical EI candidate(s)"
            ),
            "kernel": kernel,
            "active_dim": configured_d_eff if uses_configured_active_dim else None,
            "active_dimension_policy": (
                "configured synthetic d_eff"
                if uses_configured_active_dim
                else "not applicable"
            ),
            "vae_learning_rate": (
                vae_learning_rate_for_benchmark(benchmark_name)
                if algorithm == "VAE-BO"
                else None
            ),
            "vae_update_interval": (
                vae_update_interval_for_benchmark(benchmark_name)
                if algorithm == "VAE-BO"
                else None
            ),
        }
        if algorithm in {"Dropout-BO", "VAE-BO"}:
            result.update(bo_reference_dict())
            result["batch_size"] = int(settings["batch_size"])
            result["candidate_selection"] = (
                f"top-{int(settings['batch_size'])} pointwise EI values"
            )
            result["acquisition_optimizer"] = (
                "exhaustive ranking of the Sobol candidate pool"
            )
        if algorithm in {"Dropout-TuRBO", "LA-MCTS-TuRBO"}:
            result.update(turbo_reference_dict(device=device))
            result.update(
                {
                    "min_cuda": MCTS_TURBO_MIN_CUDA,
                    "device": device,
                    "dtype": MCTS_TURBO_DTYPE,
                    "small_data_runtime": (
                        "CUDA float32 from the first local GP fit (min_cuda=0)"
                    ),
                }
            )
        if algorithm in {"LA-MCTS-BO", "LA-MCTS-TuRBO"}:
            result.update(
                {
                    "cp": mcts_cp_for_benchmark(benchmark_name),
                    "leaf_size": 20,
                    "solver_type": "turbo" if algorithm.endswith("TuRBO") else "bo",
                    "turbo_max_evals": 50,
                    "kernel_type": "rbf",
                    "gamma_type": "auto",
                    "budget_guard": (
                        "objective-boundary guard; no over-budget evaluation "
                        "during a partial final TuRBO epoch"
                    ),
                    "posterior_reporting": (
                        "batch-end fixed-noise local GP refit on the current "
                        "LA-MCTS leaf TuRBO epoch; reporting-only"
                    ) if algorithm.endswith("TuRBO") else "none; official LA-MCTS-BO loop",
                }
            )
        if algorithm == "LA-MCTS-BO":
            result["alpha"] = float(BO_REFERENCE_SPEC.alpha)
            result["observation_noise_std"] = float(BO_REFERENCE_SPEC.observation_noise_std)
        if algorithm == "VAE-BO":
            result["latent_dim"] = configured_d_eff
            if registered_config is not None:
                first_bounds = np.asarray(registered_config["bounds"], dtype=float)[0]
                result.update(
                    {
                        "latent_training_bounds": [
                            float(first_bounds[0]),
                            float(first_bounds[1]),
                        ],
                        "latent_acquisition_bounds": [
                            float(first_bounds[0]) * configured_d_eff,
                            float(first_bounds[1]) * configured_d_eff,
                        ],
                        "latent_bounds_policy": (
                            "literal vendored vae_bo.py: clip encoded training z to "
                            "[lb[0], ub[0]], then pass scalar lb[0]*active_dims and "
                            "ub[0]*active_dims to Sobol EI generation"
                        ),
                    }
                )
        if algorithm in {"HeSBO", "ALEBO"}:
            result["embedding_dim"] = configured_d_eff
            result["projections"] = 1
            if algorithm == "HeSBO":
                result.update(
                    numerical_train_yvar_schedule=[1e-6, 1e-4, 1e-3],
                    numerical_fit_policy=(
                        "fit with the effective reference 1e-6 floor; retry failed "
                        "NAS-Bench fits at 1e-4 and 1e-3; if all fits fail, use "
                        "the initialized 1e-3-nugget GP for that acquisition"
                    ),
                )
            if algorithm == "ALEBO":
                result.update(fit_restarts=10, laplace_nsamp=25, raw_samples=1000, num_restarts=10)
        if algorithm == "SAASBO":
            result.update(alpha=0.1, num_warmup=512, num_samples=256, thinning=16,
                          candidate_count=1024, max_tree_depth=6, num_chains=1,
                          coordinate_policy="normalize training X; unnormalize candidates")
        if algorithm.startswith("Dropout"):
            result["active_dims"] = configured_d_eff
        return result

    return {"algorithm": algorithm}


def algorithm_settings_summary(algorithm, benchmark_name=None, device=DEVICE):
    """Return verified method settings plus the runner stopping contract."""
    canonical_algorithm = canonical_algorithm_name(algorithm)
    device = algorithm_device(canonical_algorithm, device)
    settings = _algorithm_settings_summary_impl(
        canonical_algorithm,
        benchmark_name,
        device=device,
    )
    settings.update(
        reference_runtime_metadata(
            canonical_algorithm,
            benchmark_name=benchmark_name,
        )
    )
    non_gp_algorithms = {"RandomSearch", "CMA-ES", "MCTS-VS-RS", "Dropout-RS"}
    uses_gp = canonical_algorithm not in non_gp_algorithms
    # These reference-managed implementations own their complete execution
    # loops. Injecting the project's posterior-threshold refit/check would
    # change the historical method, so they always consume the caller's exact
    # budget and report that the extra stopping layer is intentionally absent.
    reference_managed = {"HeSBO", "ALEBO", "LA-MCTS-BO", "LA-MCTS-TuRBO"}
    posterior_stopping_enabled = False
    settings.update(
        {
            "posterior_stopping_enabled": posterior_stopping_enabled,
            "posterior_stopping_check": (
                "end of every optimizer iteration/batch after the GP refit"
                if posterior_stopping_enabled
                else (
                    "disabled by the runner; reference-managed execution loop is unchanged"
                    if canonical_algorithm in reference_managed
                    else "disabled by the runner"
                    if uses_gp
                    else "not applicable: this algorithm has no GP posterior"
                )
            ),
            "posterior_stopping_threshold": (
                POSTERIOR_STOPPING_THRESHOLD if posterior_stopping_enabled else None
            ),
            "posterior_stopping_required_probability": (
                1.0 - float(RISK_TOLERANCES[0])
                if posterior_stopping_enabled
                else None
            ),
            "risk_tolerances": list(RISK_TOLERANCES),
            "device": device,
            "device_policy": (
                "every TuRBO backend uses dedicated CUDA; other Torch paths use the "
                "runner device, while NumPy/sklearn methods remain CPU-native"
            ),
        }
    )
    settings["implementation_version"] = "native-v1"
    settings["seed_policy"] = "shared_trial_seed_v1"
    settings.setdefault("benchmark_noise_std", SYNTHETIC_BENCHMARK_NOISE_STD)
    try:
        real_experiment = get_real_life_experiment(str(benchmark_name))
    except (KeyError, TypeError):
        real_experiment = None
    if real_experiment is not None:
        applicable = real_life_algorithm_is_applicable(
            real_experiment,
            canonical_algorithm,
        )
        reference_dimension = real_life_reference_embedding_dimension(
            real_experiment,
            canonical_algorithm,
        )
        settings.update(
            {
                "paper_applicable": bool(applicable),
                "configured_d_eff": int(real_experiment.configured_d_eff),
                "physical_dim": int(real_experiment.physical_dim),
                "reference_embedding_dimension": reference_dimension,
                "paper_omitted_algorithms": list(
                    real_experiment.task.paper_omitted_algorithms
                ),
            }
        )
        if applicable:
            settings["optimizer_active_dimension"] = (
                benchmark_agvs_effective_dim(
                    BENCHMARKS.get(str(benchmark_name or "").lower())
                )
                if canonical_algorithm in AGVS_ALGORITHMS
                else real_life_optimizer_dimension(
                    real_experiment,
                    canonical_algorithm,
                )
            )
        if reference_dimension is not None:
            settings.update(
                {
                    "active_dim": int(reference_dimension),
                    "embedding_dim": int(reference_dimension),
                    "active_dimension_policy": (
                        "explicit real-life reference embedding override; configured "
                        "d_eff remains the condition identity and physical D is unchanged"
                    ),
                }
            )
        if uses_gp:
            if canonical_algorithm in {"HeSBO", "ALEBO"}:
                settings.update(
                    {
                        "learn_observation_noise": False,
                        "observation_noise_mode": "known fixed observation variance",
                        "noise_estimation_method": "fixed observation variance following supplied embedding reference",
                        "learned_noise_trace_source": None,
                        "fixed_noise_variance": 0.0,
                        "noise": "known zero observation variance",
                    }
                )
                noise_method = None
                noise_trace_source = None
            elif canonical_algorithm == "LA-MCTS-BO":
                settings.update(
                    {
                        "learn_observation_noise": False,
                        "observation_noise_mode": "known fixed observation variance",
                        "noise_estimation_method": "fixed sklearn GP alpha",
                        "learned_noise_trace_source": None,
                        "fixed_noise_variance": float(BO_REFERENCE_SPEC.alpha),
                        "noise": "fixed sklearn GP alpha",
                    }
                )
                noise_method = None
                noise_trace_source = None
            elif canonical_algorithm in {"SAASBO", "MCTS-VS-SAASBO"}:
                noise_method = "Bayesian SAAS likelihood noise inference"
                noise_trace_source = "Bayesian SAAS acquisition likelihood"
            elif "TuRBO" in canonical_algorithm:
                noise_method = "GPyTorch GaussianLikelihood marginal-likelihood optimization"
                noise_trace_source = (
                    "AG-VS node acquisition GaussianLikelihood"
                    if canonical_algorithm == "AG-VS-TuRBO"
                    else (
                        "learned-noise posterior reporting GP; the native TuRBO "
                        "acquisition GaussianLikelihood is also optimized but transient"
                    )
                )
            else:
                noise_method = "sklearn WhiteKernel marginal-likelihood optimization"
                noise_trace_source = "acquisition and posterior-reporting GP"
            if canonical_algorithm not in {"HeSBO", "ALEBO", "LA-MCTS-BO"}:
                settings.update(
                    {
                        "learn_observation_noise": True,
                        "observation_noise_mode": "estimated_from_real_life_observations",
                        "noise_estimation_method": noise_method,
                        "learned_noise_trace_source": noise_trace_source,
                        "fixed_noise_variance": None,
                        "noise": noise_method,
                    }
                )
            # Values inherited from the synthetic reference specification must
            # not make a real-life report look like a fixed-noise experiment.
            if canonical_algorithm not in {"HeSBO", "ALEBO", "LA-MCTS-BO"}:
                for fixed_noise_field in (
                    "observation_noise_std",
                    "alpha",
                    "noise_variance",
                ):
                    if fixed_noise_field in settings:
                        settings[fixed_noise_field] = None
            if noise_method and noise_method.startswith("sklearn WhiteKernel"):
                signal_kernel = str(
                    settings.get("kernel")
                    or "ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5)"
                )
                settings.update(
                    {
                        "kernel": signal_kernel + " + WhiteKernel(learned noise variance)",
                        "initial_noise_variance": float(BO_REFERENCE_SPEC.alpha),
                        "noise_variance_bounds": [
                            float(REAL_LIFE_GP_NOISE_BOUNDS[0]),
                            float(REAL_LIFE_GP_NOISE_BOUNDS[1]),
                        ],
                        "gp_numerical_jitter": float(REAL_LIFE_GP_JITTER),
                        "hyperparameter_fitting": (
                            "signal, length-scale, and observation-noise parameters "
                            "by marginal likelihood (sklearn fmin_l_bfgs_b)"
                        ),
                    }
                )
            if canonical_algorithm == "LA-MCTS-TuRBO":
                settings["posterior_reporting"] = (
                    "batch-end learned-noise local GP refit on the current "
                    "LA-MCTS leaf TuRBO epoch; reporting-only"
                )
    return settings


def run_single_trial(
    benchmark_name,
    algorithm,
    trial,
    max_iter,
    output_dir,
    base_seed,
    device=DEVICE,
    skip_complete=True,
    artifact_dir=None,
    show_final_tree=AGVS_SHOW_FINAL_TREE,
    resume_under_budget=False,
):
    trial = int(trial)
    if trial < 0:
        raise ValueError("trial_id must be in the ordered range 0, ..., N_TRIALS - 1")
    benchmark_name = str(benchmark_name).lower()
    algorithm = canonical_algorithm_name(algorithm)
    max_iter = algorithm_evaluation_budget(algorithm, max_iter, benchmark_name)
    requested_runner_device = normalize_device(device)
    device = algorithm_device(algorithm, requested_runner_device)
    # Windows ProcessPoolExecutor uses ``spawn``. Child interpreters import
    # this module but do not execute real_life_entrypoint.main(), so its
    # parent-only dynamic registration is absent. Reconstruct the immutable
    # registry-backed config in the child before looking it up. This is also
    # safe for recycled workers and direct single-trial calls.
    if benchmark_name not in BENCHMARKS and benchmark_name in REAL_LIFE_EXPERIMENTS:
        from Experiment.benchmark.real_life.tasks import make_real_life_config

        experiment = REAL_LIFE_EXPERIMENTS[benchmark_name]
        register_benchmark(
            benchmark_name,
            apply_runner_agvs_batch_policy(
                make_real_life_config(
                    experiment.task.key,
                    experiment.configured_d_eff,
                )
            ),
        )
    if benchmark_name not in BENCHMARKS:
        raise ValueError(f"Benchmark spec for {benchmark_name} must be registered first.")

    config = BENCHMARKS[benchmark_name]
    validate_algorithm_applicability(algorithm, config, benchmark_name)
    experiment_kind = str(config.get("experiment_kind", "synthetic"))
    experiment_mode = (
        experiment_kind.strip().lower() != "real_life"
        and config.get("active_indices") is not None
    )
    print_memory_diagnostic("trial-start", benchmark_name, algorithm, trial)

    seed_bundle = seed_bundle_for_config(
        config,
        trial,
        base_seed_offset=base_seed,
    )
    seed = int(seed_bundle.algorithm_seed)
    benchmark_seed_value = int(seed_bundle.benchmark_seed)
    cache_identity = requested_trace_cache_identity(
        config,
        benchmark_name,
        algorithm,
        trial,
        max_iter,
        base_seed_offset=base_seed,
        device=device,
        seed_bundle=seed_bundle,
    )

    if skip_complete:
        existing_trace = load_existing_trace_if_complete(
            output_dir,
            benchmark_name,
            algorithm,
            trial,
            max_iter=max_iter,
            cache_identity=cache_identity,
        )
        if existing_trace is not None:
            existing_trace["_skipped_existing"] = True
            print(
                f"[trial {trial:03d}] {benchmark_name:10s} | {algorithm:8s} | "
                f"SKIP existing complete JSON | evals={int(existing_trace.get('n_evaluations', 0)):4d}",
                flush=True,
            )
            return existing_trace

    resume_source = None
    if bool(resume_under_budget):
        resume_source = load_under_budget_trace_for_resume(
            output_dir,
            benchmark_name,
            algorithm,
            trial,
            target_max_iter=max_iter,
            expected_identity=cache_identity,
        )
        if resume_source is not None:
            print(
                f"[resume-under-budget] {benchmark_name} | {algorithm} | "
                f"trial={trial:03d} | stored={resume_source['source_evaluations']} "
                f"-> target={max_iter} | backup={resume_source['backup_path']}",
                flush=True,
            )

    # Appendix C uses each method's own initialization strategy and seed.
    initial_design_seed = seed
    task_seed = int(benchmark_seed_value)
    recorder_seed = int(benchmark_seed_value)
    recorder_noise_std = float(config.get("benchmark_noise_std", 0.0))
    objective_factory = config.get("objective_factory")
    objective = objective_factory(task_seed) if callable(objective_factory) else config["func"]
    objective_preparation_wall_clock_seconds = 0.0
    if experiment_kind.strip().lower() == "real_life":
        preparation_started = time.time()
        print(
            f"[BACKEND-PREP] benchmark={benchmark_name} algorithm={algorithm} "
            f"trial={trial:03d} status=START",
            flush=True,
        )
        try:
            objective_preparation_wall_clock_seconds = prepare_objective(objective)
        except BaseException:
            close_objective(objective)
            raise
        print(
            f"[BACKEND-PREP] benchmark={benchmark_name} algorithm={algorithm} "
            f"trial={trial:03d} status=DONE "
            f"elapsed={time.time() - preparation_started:.1f}s",
            flush=True,
        )

    # Paper-style algorithm wall-clock begins only after the physical backend
    # is ready, but includes optimizer construction and all acquisition work.
    start = time.time()
    recorder = NoisyObjectiveRecorder(
        objective,
        noise_std=recorder_noise_std,
        seed=recorder_seed,
        benchmark_name=benchmark_name,
        algorithm=algorithm,
        trial=trial,
        max_evals=max_iter,
        # Real-life evaluations are expensive and comparatively sparse, so
        # expose every completed evaluation in the terminal. Synthetic runs
        # retain the configurable (default: every 10 evaluations) interval.
        progress_every=(
            1
            if str(experiment_kind).strip().lower() == "real_life"
            else EVALUATION_PROGRESS_EVERY
        ),
        replay_trace=(None if resume_source is None else resume_source["trace"]),
    )
    recorder.start_time = start
    recorder.report_progress(status="START", force=True)

    try:
        optimizer = build_optimizer(
            algorithm,
            config,
            recorder,
            seed=seed,
            max_iter=max_iter,
            benchmark_name=benchmark_name,
            trial=trial,
            output_dir=output_dir,
            device=device,
        )
        if resume_source is not None and algorithm == "ALEBO":
            warm_start = getattr(
                optimizer,
                "warm_start_alebo_from_ambient_history",
                None,
            )
            if not callable(warm_start):
                raise RuntimeError(
                    "ALEBO optimizer does not support stored-history warm start"
                )
            source_trace = resume_source["trace"]
            warm_start(
                source_trace["x_history"][: resume_source["source_evaluations"]],
                source_trace["observed_y_history"][
                    : resume_source["source_evaluations"]
                ],
            )
            recorder.accept_replay_history_as_warm_start()
    except BaseException:
        # Objective factories may own persistent dataset/environment handles.
        # Do not leak one if optional optimizer construction fails before the
        # normal optimization finally block is entered.
        close_objective(objective)
        raise
    print_memory_diagnostic("after-build", benchmark_name, algorithm, trial)
    if algorithm in AGVS_ALGORITHMS:
        runtime_eliminate = getattr(optimizer, "sad_eliminate", None)
        if runtime_eliminate is None:
            runtime_eliminate = max(
                1,
                int(getattr(optimizer, "split_k", benchmark_effective_dim(config) + 1))
                - int(benchmark_effective_dim(config)),
            )
        expected_iteration = getattr(optimizer, "expected_iteration", None)
        setting_parts = [
            f"[trial {trial:03d}][AGVS][calculated-settings]",
            f"benchmark={benchmark_name}",
            f"algorithm={algorithm}",
            f"d_eff={int(benchmark_effective_dim(config))}",
            f"q_batch={getattr(optimizer, 'q_batch', benchmark_q_batch(config))}",
            f"batch_size={getattr(optimizer, 'batch_size', benchmark_batch_size(config))}",
            f"t_seq={int(getattr(optimizer, 't_seq', AGVS_T_SEQ))}",
            f"sim_mode={bool(getattr(optimizer, 'sim_mode', False))}",
            f"N_split={int(getattr(optimizer, 'split_k', benchmark_effective_dim(config) + 1))}",
            f"Eliminate={int(runtime_eliminate)}",
        ]
        if bool(getattr(optimizer, "sim_mode", False)) and expected_iteration is not None:
            setting_parts.append(f"expected_iter={int(expected_iteration)}")
        setting_parts.append(f"max_iter={int(max_iter)}")
        print(
            " ".join(setting_parts),
            flush=True,
        )

    error = None
    objective_close_error = None
    interrupted = False
    result = None
    try:
        result = run_optimizer(algorithm, optimizer, recorder, max_iter=max_iter)
        result_n_evaluations = len(recorder.observed_history)
        if result_n_evaluations == 0 and result is not None and hasattr(result, "y"):
            result_n_evaluations = len(np.asarray(result.y).reshape(-1))
        result = apply_posterior_return_contract(result, result_n_evaluations)
    except KeyboardInterrupt:
        interrupted = True
        error = "KeyboardInterrupt: run interrupted by user"
    except Exception as exc:
        error = str(exc)
    finally:
        objective_close_error = close_objective(objective)
        if objective_close_error is not None:
            close_message = f"objective.close() failed: {objective_close_error}"
            error = close_message if error is None else f"{error}; {close_message}"
    wall_clock = recorder.current_wall_clock_seconds()
    recorder.report_progress(
        status="FAILED" if error is not None else "DONE",
        force=True,
    )
    print_memory_diagnostic("after-optimize", benchmark_name, algorithm, trial)

    agvs_final_tree_png = None
    family_artifact_dir = (
        Path(output_dir)
        if artifact_dir is None
        else Path(artifact_dir) / algorithm_output_family(algorithm)
    )
    if (
        bool(show_final_tree)
        and algorithm in AGVS_ALGORITHMS
        and error is None
        and result is not None
    ):
        agvs_final_tree_png = save_agvs_final_tree_png(
            optimizer,
            family_artifact_dir,
            benchmark_name,
            trial,
            algorithm=algorithm,
        )

    if algorithm in AGVS_ALGORITHMS and error is None and result is not None:
        sync_agvs_result_to_recorder(recorder, result, wall_clock)

    agvs_node_lengthscale_report = None
    if experiment_mode and algorithm in AGVS_ALGORITHMS and error is None and result is not None:
        agvs_node_lengthscale_report = write_agvs_node_lengthscale_report(
            optimizer,
            family_artifact_dir,
            benchmark_name,
            trial,
            config.get("active_indices", []),
            algorithm=algorithm,
        )

    x_history = np.asarray(recorder.x_history, dtype=float)
    original_dim = int(np.asarray(config["bounds"]).shape[0])
    true_history = np.asarray(recorder.true_history, dtype=float).reshape(-1)
    observed_history = np.asarray(recorder.observed_history, dtype=float).reshape(-1)
    eval_wall_clock_history = np.asarray(recorder.eval_wall_clock_history, dtype=float).reshape(-1)
    max_iter_limit = max(0, int(max_iter))
    x_history = crop_sequence_to_max_iter(x_history, max_iter_limit)
    true_history = crop_sequence_to_max_iter(true_history, max_iter_limit)
    observed_history = crop_sequence_to_max_iter(observed_history, max_iter_limit)
    eval_wall_clock_history = crop_sequence_to_max_iter(eval_wall_clock_history, max_iter_limit)
    if algorithm in AGVS_ALGORITHMS:
        configured_q = getattr(optimizer, "q_batch", benchmark_q_batch(config))
        configured_batch = getattr(
            optimizer,
            "batch_size",
            benchmark_batch_size(config),
        )
        q_batch_for_wall_clock = int(
            configured_q if configured_q is not None else configured_batch
        )
    else:
        q_batch_for_wall_clock = _trace_q_batch(
            {
                "algorithm_settings": algorithm_settings_summary(
                    algorithm,
                    benchmark_name,
                    device=device,
                )
            },
            default=LEGACY_TRACE_BATCH_FALLBACK,
        )
    eval_wall_clock_history = np.asarray(
        batch_parallel_wall_clock_history(
            eval_wall_clock_history,
            algorithm=algorithm,
            result=result,
            n_evaluations=len(observed_history),
            wall_clock=wall_clock,
            q_batch=q_batch_for_wall_clock,
        ),
        dtype=float,
    )
    z_history = active_z_history(benchmark_name, x_history, config) if experiment_mode else None

    best_observed_history = best_so_far(observed_history)
    best_noiseless_y_history = best_so_far(true_history)
    recommended_true_y_history = best_true_recommended(observed_history, true_history)
    posterior_means = posterior_mean_history(result, len(observed_history))
    best_posterior_means = best_posterior_mean_so_far(posterior_means)
    stopped_early = result_stopped_early(result)
    reported_posterior_means = reported_posterior_mean_history(posterior_means, stopped_early)

    posterior_threshold_history = posterior_threshold_history_by_delta(result, len(observed_history))
    if posterior_threshold_history is not None:
        threshold_history_by_delta = crop_threshold_histories_to_max_iter(posterior_threshold_history, max_iter_limit)
    else:
        threshold_history_by_delta = {
            risk_label(delta): [None] * len(observed_history)
            for delta in RISK_TOLERANCES
        }
    threshold_history_by_delta = mask_threshold_history_before_n(
        threshold_history_by_delta,
        initial_gp_start_count_for_algorithm(algorithm, config),
    )
    stopping_wall_clock_by_delta = {
        risk_label(delta): stopping_wall_clock_history(
            threshold_history_by_delta[risk_label(delta)],
            observed_history,
            eval_wall_clock_history,
        )
        for delta in RISK_TOLERANCES
    }
    primary_delta_key = primary_risk_label()
    final_best_noiseless_y_at_max_iter = None
    if len(best_noiseless_y_history) > 0:
        final_best_noiseless_y_at_max_iter = float(best_noiseless_y_history[-1])
    final_best_posterior_mean_at_max_iter = next(
        (float(value) for value in reversed(reported_posterior_means) if value is not None),
        None,
    )
    final_best_stoppable_threshold_at_max_iter = None
    primary_threshold_history = threshold_history_by_delta.get(primary_delta_key, [])
    if len(primary_threshold_history) > 0:
        for value in reversed(primary_threshold_history):
            if value is None:
                continue
            try:
                value = float(value)
            except Exception:
                continue
            if np.isfinite(value):
                final_best_stoppable_threshold_at_max_iter = float(value)
                break

    agvs_history = (
        crop_history_records_to_max_iter(getattr(result, "history", None), max_iter_limit)
        if algorithm in AGVS_ALGORITHMS
        else None
    )
    agvs_merge_history = (
        make_jsonable(
            crop_history_records_to_max_iter(
                getattr(optimizer, "sad_merge_history", None),
                max_iter_limit,
            )
        )
        if algorithm in AGVS_ALGORITHMS
        else None
    )
    agvs_tree_restart_history = (
        make_jsonable(
            crop_history_records_to_max_iter(
                getattr(optimizer, "tree_restart_history", None),
                max_iter_limit,
            )
        )
        if algorithm in AGVS_ALGORITHMS
        else None
    )
    agvs_sel_merge_history = None
    vanilla_bo_history = (
        make_jsonable(crop_history_records_to_max_iter(getattr(result, "history", None), max_iter_limit))
        if algorithm == "VanillaBO"
        else None
    )
    vanilla_bo_dimension_report_path = (
        getattr(result, "dimension_report_path", None) if algorithm == "VanillaBO" else None
    )
    algorithm_settings = algorithm_settings_summary(
        algorithm,
        benchmark_name,
        device=device,
    )
    effective_dim = (
        benchmark_agvs_effective_dim(config)
        if algorithm in AGVS_ALGORITHMS
        else benchmark_effective_dim(config)
    )
    if algorithm in AGVS_ALGORITHMS:
        calculated_eliminate = getattr(optimizer, "sad_eliminate", None)
        if calculated_eliminate is None:
            calculated_eliminate = max(
                1,
                int(getattr(optimizer, "split_k", effective_dim + 1)) - int(effective_dim),
            )
        algorithm_settings.update(
            {
                "calculated_n_split": getattr(optimizer, "split_k", None),
                "calculated_eliminate": int(calculated_eliminate),
                "sim_mode": bool(getattr(optimizer, "sim_mode", False)),
                "n_init": benchmark_n_init(config),
                "effective_dim": effective_dim,
                "q_batch": (
                    None
                    if getattr(optimizer, "q_batch", benchmark_q_batch(config)) is None
                    else int(getattr(optimizer, "q_batch", benchmark_q_batch(config)))
                ),
                "q_batch_policy": (
                    "runner central controls; exactly one of q_batch/batch_size is active"
                ),
                "batch_settings_source": config.get("agvs_batch_settings_source"),
                "batch_size": (
                    None
                    if getattr(optimizer, "batch_size", benchmark_batch_size(config)) is None
                    else int(getattr(optimizer, "batch_size", benchmark_batch_size(config)))
                ),
                "selection_batch_size": (
                    None
                    if getattr(optimizer, "selection_batch_size", benchmark_batch_size(config)) is None
                    else int(getattr(optimizer, "selection_batch_size", benchmark_batch_size(config)))
                ),
                "acquisition_batch_mode": getattr(
                    optimizer,
                    "acquisition_batch_mode",
                    "q_batch" if benchmark_q_batch(config) is not None else "batch_size",
                ),
                "batch_parameter_policy": "exactly one of q_batch or batch_size is non-null",
                "t_seq": int(getattr(optimizer, "t_seq", 1)),
                "N_bad": int(
                    getattr(
                        optimizer,
                        "N_bad",
                        int(config.get("n_bad_multiplier", AGVS_N_BAD_MULTIPLIER))
                        * (
                            1
                            if bool(config.get("n_bad_is_absolute", False))
                            else int(getattr(optimizer, "split_k", effective_dim + 1))
                        ),
                    )
                ),
                "N_bad_multiplier": int(
                    config.get("n_bad_multiplier", AGVS_N_BAD_MULTIPLIER)
                ),
                "N_bad_mode": (
                    "absolute"
                    if bool(config.get("n_bad_is_absolute", False))
                    else "N_bad_multiplier * N_split"
                ),
                "N_bad_enabled": bool(
                    getattr(
                        optimizer,
                        "N_bad_enabled",
                        config.get("n_bad_enabled", AGVS_N_BAD_ENABLED),
                    )
                ),
                "merge_enabled": bool(
                    getattr(
                        optimizer,
                        "merge_enabled",
                        benchmark_agvs_merge_enabled(config),
                    )
                ),
                "tree_restart_count": int(getattr(optimizer, "tree_restart_count", 0)),
                "warm_start_root_gp": bool(getattr(optimizer, "warm_start_root_gp", True)),
                "incremental_root_gp": bool(getattr(optimizer, "incremental_root_gp", True)),
                "default_compute_device": "cpu",
            }
        )
        if bool(getattr(optimizer, "sim_mode", False)) and getattr(optimizer, "expected_iteration", None) is not None:
            algorithm_settings["expected_iter"] = int(optimizer.expected_iteration)
    elif algorithm == "VanillaBO":
        algorithm_settings["n_init"] = MCTS_PAPER_BO_N_INIT
        algorithm_settings["q_batch"] = MCTS_PAPER_BO_BATCH_SIZE
        algorithm_settings["effective_dim"] = effective_dim
    elif algorithm == "HeSBO":
        algorithm_settings.update(
            {
                "numerical_fit_retry_count": int(
                    getattr(optimizer, "hesbo_fit_retry_count", 0)
                ),
                "unfitted_fallback_count": int(
                    getattr(optimizer, "hesbo_unfitted_fallback_count", 0)
                ),
                "numerical_train_yvar_history": make_jsonable(
                    getattr(optimizer, "hesbo_numerical_train_yvar_history", [])
                ),
                "last_fit_error_before_recovery": getattr(
                    optimizer, "hesbo_last_fit_error", None
                ),
            }
        )
    elif algorithm == "ALEBO":
        algorithm_settings["warm_start"] = make_jsonable(
            getattr(optimizer, "warm_start_metadata", None)
        )
    elif algorithm == "GTBO":
        algorithm_settings.update(
            {
                "gp_state_reset_recovery_count": int(
                    getattr(optimizer, "gp_state_reset_recovery_count", 0)
                ),
                "gp_unfitted_fallback_count": int(
                    getattr(optimizer, "gp_unfitted_fallback_count", 0)
                ),
                "gp_fit_recovery_history": make_jsonable(
                    getattr(optimizer, "gp_fit_recovery_history", [])
                ),
                "last_fit_error_before_recovery": getattr(
                    optimizer, "gp_last_fit_error", None
                ),
            }
        )

    posterior_stopping_trace = []
    raw_stopping_trace = getattr(result, "threshold_trace", None)
    if isinstance(raw_stopping_trace, list):
        posterior_stopping_trace.extend(
            entry
            for entry in raw_stopping_trace
            if isinstance(entry, dict) and int(entry.get("n", 0)) <= max_iter_limit
        )
    if not posterior_stopping_trace:
        result_history = getattr(result, "history", None)
        if isinstance(result_history, list):
            posterior_stopping_trace.extend(
                entry["stopping_info"]
                for entry in result_history
                if isinstance(entry, dict)
                and isinstance(entry.get("stopping_info"), dict)
                and int(entry["stopping_info"].get("n", entry.get("n_eval", 0))) <= max_iter_limit
            )
    posterior_stopping_trace = make_jsonable(posterior_stopping_trace)
    learned_noise_variance_history = []
    if experiment_kind == "real_life":
        for entry in posterior_stopping_trace:
            if not isinstance(entry, dict):
                continue
            value = entry.get("learned_noise_variance")
            if value is None and "learned" in str(entry.get("noise_model", "")).lower():
                value = entry.get("noise_variance")
            try:
                value = float(value)
            except (TypeError, ValueError, OverflowError):
                continue
            if np.isfinite(value) and value >= 0.0:
                learned_noise_variance_history.append(value)
        algorithm_settings["learned_noise_variance_history"] = list(
            learned_noise_variance_history
        )
        algorithm_settings["final_learned_noise_variance"] = (
            float(learned_noise_variance_history[-1])
            if learned_noise_variance_history
            else None
        )

    runtime_metadata = {}
    metadata_factory = getattr(objective, "runtime_metadata", None)
    if callable(metadata_factory):
        try:
            runtime_metadata = make_jsonable(metadata_factory())
        except Exception as exc:
            runtime_metadata = {"metadata_error": f"{type(exc).__name__}: {exc}"}
    request_benchmark_config = cache_identity.get("benchmark_config", {})
    runtime_metadata["dataset_fingerprint"] = request_benchmark_config.get(
        "dataset"
    )
    request_runtime_signature = request_benchmark_config.get("runtime_signature")
    if isinstance(request_runtime_signature, dict):
        runtime_metadata["package_versions"] = request_runtime_signature.get(
            "package_versions", {}
        )

    final_best_observed_at_max_iter = (
        None if len(best_observed_history) == 0 else float(best_observed_history[-1])
    )
    trace = {
        "final_best_observed_at_max_iter": final_best_observed_at_max_iter,
        "final_best_posterior_mean_at_max_iter": final_best_posterior_mean_at_max_iter,
        "returned_posterior_mean": final_best_posterior_mean_at_max_iter,
        "returned_posterior_mean_rule": "running-best available diagnostic posterior mean; posterior stopping is disabled",
        "final_best_noiseless_y_at_max_iter": final_best_noiseless_y_at_max_iter,
        "final_best_stoppable_threshold_at_max_iter_1_256": final_best_stoppable_threshold_at_max_iter,
        "algorithm": algorithm,
        "benchmark": benchmark_name,
        "trial_id": int(trial),
        "trial": int(trial),
        "algorithm_seed": int(seed),
        "benchmark_seed": int(benchmark_seed_value),
        "effective_dimension_seed": config.get("effective_dimension_seed"),
        "seed": int(seed),
        "objective_noise_seed": int(recorder_seed),
        "initial_design_seed": int(initial_design_seed if experiment_mode else seed),
        "cache_identity": cache_identity,
        "complete": error is None,
        "interrupted": interrupted,
        "error": error,
        "settings": {
            "original_dim": original_dim,
            "D": original_dim,
            "effective_dim": effective_dim,
            "d_eff": effective_dim,
            "effective_dim_role": config.get("effective_dim_role", "ground_truth"),
            "experiment_kind": experiment_kind,
            "real_life_task": config.get("real_life_task"),
            "real_life_variant": config.get("real_life_variant"),
            "metric_transform": config.get("metric_transform"),
            "optimization_direction": config.get("optimization_direction", "maximize"),
            "runtime_provenance": runtime_metadata,
            "device": device,
            "runner_requested_device": requested_runner_device,
            "metric_name": config.get("metric_name", "objective"),
            "task_seed": int(task_seed),
            "experiment_mode": bool(experiment_mode),
            "active_indices": np.asarray(config["active_indices"], dtype=int).tolist() if experiment_mode else None,
            "effective_dimension_indices": (
                np.asarray(config["active_indices"], dtype=int).tolist()
                if experiment_mode
                else None
            ),
            "effective_dimension_seed": config.get("effective_dimension_seed"),
            "max_iter": int(max_iter),
            "trial_id": int(trial),
            "algorithm_seed": int(seed),
            "algorithm_seed_rule": (
                f"{BASE_BENCHMARK_SEED} + base_seed_offset + trial_id"
                if experiment_kind == "real_life"
                else f"{BASE_ALGORITHM_SEED} + base_seed_offset + trial_id"
            ),
            "benchmark_seed": int(benchmark_seed_value),
            "benchmark_seed_rule": (
                f"{BASE_BENCHMARK_SEED} + base_seed_offset + trial_id"
            ),
            "seed": int(seed),
            "objective_noise_seed": int(recorder_seed),
            "objective_noise_seed_policy": "benchmark_seed; shared across algorithms",
            "objective_noise_seed_formula": "objective_noise_seed = benchmark_seed",
            "initial_design_seed": int(initial_design_seed if experiment_mode else seed),
            "initial_design_seed_policy": (
                "shared trial seed; independent of algorithm and benchmark identity"
            ),
            "initial_design_seed_formula": "same deterministic seed as the algorithm trial",
            "initial_design_strategy": "algorithm-specific native LHS, Sobol, feasible embedding, or default-point design",
            "benchmark_noise_std": float(recorder_noise_std),
            "benchmark_noise_variance": float(recorder_noise_std**2),
            "benchmark_noise_mean": 0.0,
            "benchmark_noise_distribution": str(
                config.get(
                    "benchmark_noise_distribution",
                    "Gaussian" if recorder_noise_std > 0.0 else "none",
                )
            ),
            "common_random_numbers_by_evaluation_index": bool(
                str(config.get("experiment_kind", "synthetic")) == "synthetic"
            ),
            "noise_generation_policy": (
                "Synthetic observations add seeded Gaussian N(0, 1e-4) noise. "
                "The objective-noise seed equals benchmark_seed, so all algorithms "
                "in the same benchmark/trial consume the same noise stream by evaluation "
                "index. GP assumed noise is algorithm infrastructure and is reported separately."
                if str(config.get("experiment_kind", "synthetic")) == "synthetic"
                else "Real-life wrappers own environment/dataset randomness through benchmark_seed."
            ),
            "objective_close_error": objective_close_error,
            "wall_clock_timing_policy": config.get("wall_clock_timing_policy"),
            "objective_preparation_wall_clock_seconds": float(
                objective_preparation_wall_clock_seconds
            ),
            "posterior_threshold_policy": (
                "disabled; no posterior stopping or stoppable-threshold calculation"
            ),
            "risk_tolerances": {risk_label(delta): float(delta) for delta in RISK_TOLERANCES},
            "round_robin_order": DEFAULT_ALGORITHMS,
            "algorithm_settings": algorithm_settings,
            "threshold_formula": None,
            "reported_objective_policy": (
                "Real-life algorithm performance is the running-best observed benchmark "
                "metric (accuracy or reward). GP posterior values are diagnostics only."
                if experiment_kind == "real_life"
                else "Synthetic posterior means remain diagnostic only; stopping is disabled "
                "and noiseless objective values remain the independent performance metric."
            ),
            "wall_clock_history_definition": "evaluation_wall_clock_history[t] is batch-parallel cumulative algorithm seconds measured after backend preparation; all evaluations in the same q_batch share the batch completion time",
        },
        "objective_preparation_wall_clock_seconds": float(
            objective_preparation_wall_clock_seconds
        ),
        "wall_clock_seconds": float(wall_clock),
        "evaluation_wall_clock_history": eval_wall_clock_history.tolist(),
        "n_evaluations": int(len(observed_history)),
        "best_observed": None if observed_history.size == 0 else float(np.max(observed_history)),
        "best_posterior_mean": final_best_posterior_mean_at_max_iter,
        "best_noiseless_y": None if true_history.size == 0 else float(np.max(true_history)),
        "best_true_recommended": None if len(recommended_true_y_history) == 0 else float(np.max(recommended_true_y_history)),
        "embedding_matrix_A": extract_embedding_matrix(optimizer),
        "agvs_history": agvs_history,
        "agvs_merge_history": agvs_merge_history,
        "agvs_tree_restart_history": agvs_tree_restart_history,
        "agvs_sel_merge_history": agvs_sel_merge_history,
        "agvs_final_tree_png_path": agvs_final_tree_png,
        "agvs_node_lengthscale_report_path": agvs_node_lengthscale_report,
        "vanilla_bo_history": vanilla_bo_history,
        "vanilla_bo_dimension_report_path": vanilla_bo_dimension_report_path,
        "posterior_stopping_trace": posterior_stopping_trace,
        "posterior_stopping_checked_at_n": [
            int(entry.get("n", 0))
            for entry in posterior_stopping_trace
            if isinstance(entry, dict) and int(entry.get("n", 0)) > 0
        ],
        "stopped_early": bool(stopped_early),
        "stop_iteration": getattr(result, "stop_iteration", None),
        "important_dim": getattr(result, "important_dim", None) if algorithm in AGVS_ALGORITHMS else None,
        "best_pure_x_coordinate": make_jsonable(getattr(result, "best_pure_x_coordinate", None)) if algorithm in AGVS_ALGORITHMS else None,
        "agvs_posterior_dataset": make_jsonable(getattr(result, "posterior_dataset", None)) if algorithm in AGVS_ALGORITHMS else None,
        "best_observed_history": best_observed_history,
        "posterior_mean_history": posterior_means,
        "best_posterior_mean_history": best_posterior_means,
        "reported_posterior_mean_history": reported_posterior_means,
        "best_noiseless_y_history": best_noiseless_y_history,
        "recommended_true_y_history": recommended_true_y_history,
        "best_threshold_history_by_delta": threshold_history_by_delta,
        "stopping_wall_clock_by_delta": stopping_wall_clock_by_delta,
        "observed_y_history": observed_history.tolist(),
        "true_y_history": true_history.tolist(),
        "x_history": x_history.tolist(),
        "z_history": z_history,
    }
    if resume_source is not None:
        resume_mode = (
            "svd_recovered_subspace_warm_start"
            if algorithm == "ALEBO"
            else "deterministic_optimizer_replay"
        )
        trace["resume"] = {
            "schema_version": 1,
            "mode": resume_mode,
            "source_path": str(resume_source["source_path"]),
            "backup_path": str(resume_source["backup_path"]),
            "source_sha256": str(resume_source["source_sha256"]),
            "source_evaluations": int(resume_source["source_evaluations"]),
            "target_evaluations": int(resume_source["target_evaluations"]),
            "replayed_evaluations": int(recorder.replayed_evaluations),
            "new_evaluations": max(
                0,
                int(len(observed_history))
                - int(resume_source["source_evaluations"]),
            ),
            "x_match_rtol": 0.0 if algorithm != "ALEBO" else None,
            "x_match_atol": (
                float(recorder.replay_x_atol)
                if algorithm != "ALEBO"
                else None
            ),
            "source_wall_clock_seconds": float(
                recorder._replay_base_wall_clock_seconds
            ),
            "source_json_preserved_on_failure": True,
        }

    if algorithm in AGVS_ALGORITHMS:
        configured_q_batch = benchmark_q_batch(config)
        configured_batch_size = benchmark_batch_size(config)
        expected_q_batch = (
            None if configured_q_batch is None else int(configured_q_batch)
        )
        expected_batch_size = (
            None if configured_batch_size is None else int(configured_batch_size)
        )
        expected_t_seq = int(config.get("t_seq", AGVS_T_SEQ))
        raw_runtime_q_batch = getattr(optimizer, "q_batch", expected_q_batch)
        raw_runtime_batch_size = getattr(
            optimizer,
            "batch_size",
            expected_batch_size,
        )
        runtime_q_batch = (
            None if raw_runtime_q_batch is None else int(raw_runtime_q_batch)
        )
        runtime_batch_size = (
            None if raw_runtime_batch_size is None else int(raw_runtime_batch_size)
        )
        runtime_t_seq = int(getattr(optimizer, "t_seq", expected_t_seq))
        if (runtime_q_batch is None) == (runtime_batch_size is None):
            raise RuntimeError(
                "Refusing to write an AG-VS trace unless exactly one of "
                "q_batch and batch_size is non-null"
            )
        runtime_point_batch_size = (
            runtime_q_batch
            if runtime_q_batch is not None
            else runtime_batch_size
        )
        runtime_batch_settings = {
            "q_batch": runtime_q_batch,
            "batch_size": runtime_batch_size,
            "selection_batch_size": runtime_batch_size,
            "t_seq": runtime_t_seq,
            "points_per_selected_node_cycle": (
                runtime_point_batch_size * runtime_t_seq
            ),
            "q_batch_semantics": (
                "mutually exclusive joint-acquisition alternative; null in this run"
            ),
            "batch_size_semantics": (
                "AG-VS-BO: top-k independent pointwise EI; AG-VS-TuRBO: native "
                "trust-region Thompson batch"
            ),
            "tree_update_policy": (
                "repeat acquisition/evaluate t_seq times; refit only before a later acquisition "
                "when the selected node model is stale, then update/split the selected node"
            ),
            "protocol": (
                f"synthetic: q_batch={SYNTHETIC_Q_BATCH}, "
                f"batch_size={SYNTHETIC_BATCH_SIZE}, t_seq={SYNTHETIC_T_SEQ}, "
                f"N_bad={SYNTHETIC_N_BAD_MULTIPLIER}*N_split, "
                f"N_bad_enabled={SYNTHETIC_N_BAD_ENABLED}; "
                f"real-life: q_batch={REAL_LIFE_Q_BATCH}, "
                f"batch_size={REAL_LIFE_BATCH_SIZE}, t_seq={REAL_LIFE_T_SEQ}, "
                f"N_bad="
                f"{REAL_LIFE_N_BAD_ABSOLUTE if REAL_LIFE_N_BAD_ABSOLUTE is not None else f'{REAL_LIFE_N_BAD_MULTIPLIER}*N_split'}, "
                f"N_bad_enabled={REAL_LIFE_N_BAD_ENABLED}"
            ),
        }
        expected_runtime_settings = (
            expected_q_batch,
            expected_batch_size,
            expected_t_seq,
        )
        actual_runtime_settings = (
            runtime_q_batch,
            runtime_batch_size,
            runtime_t_seq,
        )
        if actual_runtime_settings != expected_runtime_settings:
            raise RuntimeError(
                "Refusing to write an AG-VS trace with inconsistent batch settings: "
                f"runtime={actual_runtime_settings}, "
                f"benchmark_config={expected_runtime_settings}"
            )
        expected_merge_enabled = benchmark_agvs_merge_enabled(config)
        runtime_merge_enabled = bool(
            getattr(optimizer, "merge_enabled", expected_merge_enabled)
        )
        if runtime_merge_enabled != expected_merge_enabled:
            raise RuntimeError(
                "Refusing to write an AG-VS trace with inconsistent merge settings: "
                f"runtime={runtime_merge_enabled}, "
                f"benchmark_config={expected_merge_enabled}"
            )
        runtime_batch_settings["merge_enabled"] = runtime_merge_enabled
        trace["settings"]["agvs_batch_settings"] = runtime_batch_settings

    if algorithm in AGVS_ALGORITHMS and error is None and result is not None:
        trace["agvs_dimension_selection_counts"] = agvs_dimension_selection_counts_from_history(
            agvs_history,
            original_dim,
        )
        trace["agvs_final_terminal_leaf_dims"] = agvs_final_terminal_leaf_dims_from_trace(
            trace,
            original_dim,
        )

    trace = crop_trace_payload_to_max_iter(trace, max_iter)
    trace_path = trace_path_for(output_dir, benchmark_name, algorithm, trial)
    if resume_source is not None and error is not None:
        # Keep failed-resume diagnostics outside the ``trial_*.json`` glob used
        # by report generation. They are diagnostics rather than trial results.
        failure_path = trace_path.with_name(
            f"resume_failure_{trace_path.stem}.json"
        )
        write_json(failure_path, trace)
        print(
            f"[resume-failed] preserved source JSON at {trace_path}; "
            f"details={failure_path}",
            flush=True,
        )
    else:
        write_json(trace_path, trace)
    print_memory_diagnostic("after-json", benchmark_name, algorithm, trial)
    release_trial_memory(optimizer=optimizer, result=result, recorder=recorder)
    print_memory_diagnostic("after-cleanup", benchmark_name, algorithm, trial)
    return trace


def real_life_algorithm_uses_configured_d_eff(algorithm, experiment):
    """Whether a real-life optimizer changes across configured d_eff variants.

    Algorithms with a benchmark-specific dimension reported by the comparison
    paper use that reference value.  A d_eff-aware algorithm without such a
    same-benchmark value falls back to the configured condition (deff2/deff3).
    Algorithms that do not consume d_eff are invariant as well.
    """

    algorithm = canonical_algorithm_name(algorithm)
    if algorithm in AGVS_ALGORITHMS:
        return True
    if algorithm not in CONFIGURED_ACTIVE_DIMENSION_ALGORITHMS:
        return False
    # HeSBO/ALEBO use their task-specific paper embedding override whenever
    # applicable, so their runtime dimension is identical in both conditions.
    return real_life_reference_embedding_dimension(experiment, algorithm) is None


def materialize_dimension_invariant_real_life_traces(
    *,
    output_dir,
    benchmarks,
    algorithms,
    trials,
    max_iter,
    base_seed_offset=0,
    device=DEVICE,
    reuse_complete_json=False,
):
    """Reuse a valid sibling d_eff trace when the optimizer is invariant.

    The objective, trial seed, physical bounds, and budget are identical across
    the registered conditions. A target JSON is rebuilt with the requested
    cache identity and condition metadata; the source JSON is never modified.
    """

    from Experiment.benchmark.real_life.tasks import make_real_life_config

    created = []
    for target_name in [str(name).lower() for name in benchmarks]:
        try:
            target_experiment = get_real_life_experiment(target_name)
        except (KeyError, TypeError):
            continue
        target_d_eff = int(target_experiment.configured_d_eff)
        target_config = BENCHMARKS[target_name]
        source_experiments = sorted(
            (
                experiment
                for experiment in REAL_LIFE_EXPERIMENTS.values()
                if experiment.task.key == target_experiment.task.key
                and int(experiment.configured_d_eff) != target_d_eff
            ),
            # Prefer the paper's original deff3 condition when several sibling
            # variants exist, while still permitting reuse in either direction.
            key=lambda experiment: (
                int(experiment.configured_d_eff) != 3,
                int(experiment.configured_d_eff),
            ),
        )

        for raw_algorithm in algorithms:
            algorithm = canonical_algorithm_name(raw_algorithm)
            if real_life_algorithm_uses_configured_d_eff(
                algorithm,
                target_experiment,
            ):
                continue
            for trial in range(int(trials)):
                source_name = None
                source_trace = None
                for source_experiment in source_experiments:
                    candidate_name = source_experiment.key
                    if candidate_name not in BENCHMARKS:
                        register_benchmark(
                            candidate_name,
                            make_real_life_config(
                                target_experiment.task.key,
                                int(source_experiment.configured_d_eff),
                            ),
                        )
                    source_config = BENCHMARKS[candidate_name]
                    source_identity = requested_trace_cache_identity(
                        source_config,
                        candidate_name,
                        algorithm,
                        trial,
                        max_iter,
                        base_seed_offset=base_seed_offset,
                        device=device,
                    )
                    source_trace = load_existing_trace_if_complete(
                        output_dir,
                        candidate_name,
                        algorithm,
                        trial,
                        max_iter=max_iter,
                        cache_identity=source_identity,
                        reuse_complete_json=reuse_complete_json,
                    )
                    if source_trace is not None:
                        source_name = candidate_name
                        break
                if source_trace is None:
                    continue

                target_identity = requested_trace_cache_identity(
                    target_config,
                    target_name,
                    algorithm,
                    trial,
                    max_iter,
                    base_seed_offset=base_seed_offset,
                    device=device,
                )
                if load_existing_trace_if_complete(
                    output_dir,
                    target_name,
                    algorithm,
                    trial,
                    max_iter=max_iter,
                    cache_identity=target_identity,
                    reuse_complete_json=reuse_complete_json,
                ) is not None:
                    continue

                target_trace = copy.deepcopy(source_trace)
                target_trace.pop("_skipped_existing", None)
                target_trace["benchmark"] = target_name
                target_trace["cache_identity"] = target_identity
                target_trace["dimension_reuse_provenance"] = {
                    "reused": True,
                    "source_benchmark": source_name,
                    "target_benchmark": target_name,
                    "reason": "algorithm runtime is invariant to configured real-life d_eff",
                    "source_trace": str(
                        trace_path_for(output_dir, source_name, algorithm, trial)
                    ),
                }
                settings = target_trace.setdefault("settings", {})
                settings.update(
                    {
                        "effective_dim": target_d_eff,
                        "d_eff": target_d_eff,
                        "configured_d_eff": target_d_eff,
                        "real_life_variant": target_name,
                        "algorithm_settings": copy.deepcopy(
                            target_identity["algorithm_settings"]
                        ),
                        "dimension_reuse_provenance": copy.deepcopy(
                            target_trace["dimension_reuse_provenance"]
                        ),
                    }
                )
                write_json(
                    trace_path_for(output_dir, target_name, algorithm, trial),
                    target_trace,
                )
                created.append((target_name, algorithm, int(trial), source_name))
    return created


def algorithms_requiring_runtime_for_request(
    *,
    output_dir,
    benchmark,
    algorithms,
    trials,
    max_iter,
    base_seed_offset=0,
    device=DEVICE,
    reuse_complete_json=False,
):
    """Return algorithms for which at least one requested trial must execute."""

    benchmark = str(benchmark).lower()
    target_config = BENCHMARKS[benchmark]
    try:
        target_experiment = get_real_life_experiment(benchmark)
    except (KeyError, TypeError):
        target_experiment = None

    source_configs = []
    if target_experiment is not None:
        from Experiment.benchmark.real_life.tasks import make_real_life_config

        target_d_eff = int(target_experiment.configured_d_eff)
        source_experiments = sorted(
            (
                experiment
                for experiment in REAL_LIFE_EXPERIMENTS.values()
                if experiment.task.key == target_experiment.task.key
                and int(experiment.configured_d_eff) != target_d_eff
            ),
            key=lambda experiment: (
                int(experiment.configured_d_eff) != 3,
                int(experiment.configured_d_eff),
            ),
        )
        for source_experiment in source_experiments:
            source_name = source_experiment.key
            if source_name not in BENCHMARKS:
                register_benchmark(
                    source_name,
                    make_real_life_config(
                        target_experiment.task.key,
                        int(source_experiment.configured_d_eff),
                    ),
                )
            source_configs.append((source_name, BENCHMARKS[source_name]))

    required = []
    for raw_algorithm in algorithms:
        algorithm = canonical_algorithm_name(raw_algorithm)
        needs_execution = False
        for trial in range(int(trials)):
            target_identity = requested_trace_cache_identity(
                target_config,
                benchmark,
                algorithm,
                trial,
                max_iter,
                base_seed_offset=base_seed_offset,
                device=device,
            )
            if load_existing_trace_if_complete(
                output_dir,
                benchmark,
                algorithm,
                trial,
                max_iter=max_iter,
                cache_identity=target_identity,
                reuse_complete_json=reuse_complete_json,
            ) is not None:
                continue

            reusable = (
                target_experiment is not None
                and bool(source_configs)
                and not real_life_algorithm_uses_configured_d_eff(
                    algorithm,
                    target_experiment,
                )
            )
            if reusable:
                sibling_trace_exists = False
                for source_name, source_config in source_configs:
                    source_identity = requested_trace_cache_identity(
                        source_config,
                        source_name,
                        algorithm,
                        trial,
                        max_iter,
                        base_seed_offset=base_seed_offset,
                        device=device,
                    )
                    if load_existing_trace_if_complete(
                        output_dir,
                        source_name,
                        algorithm,
                        trial,
                        max_iter=max_iter,
                        cache_identity=source_identity,
                        reuse_complete_json=reuse_complete_json,
                    ) is not None:
                        sibling_trace_exists = True
                        break
                if sibling_trace_exists:
                    continue
            needs_execution = True
            break
        if needs_execution:
            required.append(algorithm)
    return required


def run_round_robin(
    benchmarks,
    algorithms,
    trials,
    max_iter,
    output_dir,
    figure_dir,
    seed,
    update_plots=True,
    skip_complete=True,
    workers=None,
    device=DEVICE,
    table_dir=None,
    trial_start=0,
    reuse_complete_json=False,
    resume_under_budget=False,
):
    benchmarks = [str(b).lower() for b in benchmarks]
    algorithms = normalize_algorithm_args(algorithms)
    algorithm_phase_order = sorted(
        algorithms,
        key=lambda algorithm: (
            0
            if algorithm in AGVS_ALGORITHMS
            else 1
            if algorithm.startswith("MCTS-VS-")
            else 2,
            algorithms.index(algorithm),
        ),
    )
    device = normalize_device(device)
    if not benchmarks:
        raise ValueError("At least one benchmark must be selected")
    if not algorithms:
        raise ValueError("At least one algorithm must be selected")
    if int(trials) <= 0:
        raise ValueError("trials must be a positive integer")
    trial_start = int(trial_start)
    if not 0 <= trial_start < int(trials):
        raise ValueError("trial_start must be between zero and trials - 1")
    if int(max_iter) <= 0:
        raise ValueError("max_iter must be a positive integer")

    if update_plots:
        print(
            "[tables] Updating aggregate Wilcoxon tables before benchmark execution...",
            flush=True,
        )
        table_paths = save_all_benchmark_objective_tables(
            benchmark_root=output_dir,
            table_dir=table_dir,
        )
        if table_paths:
            print(
                f"[tables] Updated {len(table_paths)} aggregate table file(s) at "
                f"{Path(table_dir) if table_dir is not None else SYNTHETIC_TABLE_ROOT}.",
                flush=True,
            )

    for b in benchmarks:
        if b not in BENCHMARKS:
            raise ValueError(f"Unknown benchmark: {b}. Available: {sorted(BENCHMARKS)}")
        spec = BENCHMARKS[b]
        for algorithm in algorithm_phase_order:
            validate_algorithm_applicability(algorithm, spec, b)
        effective_dim = benchmark_effective_dim(spec)
        active = spec.get("active_indices")
        if active is None:
            print(
                f"[search-dims] {b:10s} | assumed_dim={effective_dim} | "
                "ground_truth_active_indices=unknown",
                flush=True,
            )
        else:
            active_indices = np.asarray(active, dtype=int).tolist()
            print(
                f"[active-dims] {b:10s} | d_eff={effective_dim} | "
                f"effective_dimension_seed={spec.get('effective_dimension_seed')} | "
                f"indices={active_indices}",
                flush=True,
            )

    reused_dimension_traces = materialize_dimension_invariant_real_life_traces(
        output_dir=output_dir,
        benchmarks=benchmarks,
        algorithms=algorithms,
        trials=trials,
        max_iter=max_iter,
        base_seed_offset=seed,
        device=device,
        reuse_complete_json=reuse_complete_json,
    )
    if reused_dimension_traces:
        print(
            "[d-eff-reuse] Materialized "
            f"{len(reused_dimension_traces)} trace(s) from valid sibling d_eff "
            "results for dimension-invariant algorithms.",
            flush=True,
        )

    completed_traces = {}
    if skip_complete:
        completed_traces = existing_complete_traces(
            output_dir,
            benchmarks,
            algorithms,
            trials,
            max_iter=max_iter,
            base_seed_offset=seed,
            device=device,
            reuse_complete_json=reuse_complete_json,
        )
        completed_traces = {
            key: trace for key, trace in completed_traces.items()
            if key[0] >= trial_start
        }
        existing_total = len(completed_traces)
        existing_counts = {
            (benchmark, algorithm): sum(
                (trial, benchmark, algorithm) in completed_traces
                for trial in range(trial_start, int(trials))
            )
            for benchmark in benchmarks
            for algorithm in algorithms
        }
        expected_total = (int(trials) - trial_start) * len(benchmarks) * len(algorithms)
        print(
            f"[skip-existing] found {existing_total}/{expected_total} cache-compatible, budget-complete JSON trace(s).",
            flush=True,
        )
        for (benchmark, algorithm), count in sorted(existing_counts.items()):
            if count:
                print(
                    f"[skip-existing] {benchmark:16s} | {algorithm:16s} | {count}/{int(trials)}",
                    flush=True,
                )

    if workers is None:
        workers = int(os.environ.get("AGVS_WORKERS", str(WORKERS)))
    requested_workers = max(1, int(workers))
    workers = requested_workers
    pending_trials_by_pair = {
        (benchmark, algorithm): [
            trial
            for trial in range(trial_start, int(trials))
            if (trial, benchmark, algorithm) not in completed_traces
        ]
        for benchmark in benchmarks
        for algorithm in algorithms
    }
    trial_results = [
        {
            "benchmark": benchmark,
            "algorithm": algorithm,
            "trial_id": trial,
            "complete": True,
            "error": None,
            "skipped_existing": True,
        }
        for trial, benchmark, algorithm in sorted(completed_traces)
    ]
    pending_count = sum(len(items) for items in pending_trials_by_pair.values())
    if pending_count == 0:
        print(
            "[skip-existing] all requested runs reached their budgets; no worker tasks submitted.",
            flush=True,
        )
    scheduled_turbo_algorithms = sorted(
        {
            algorithm
            for (benchmark, algorithm), pending_trials in pending_trials_by_pair.items()
            if pending_trials and algorithm in TURBO_BACKEND_ALGORITHMS
        }
    )
    if scheduled_turbo_algorithms:
        preflight_algorithm_devices(scheduled_turbo_algorithms)

    # Benchmark-local algorithm barriers: finish every pending trial of the
    # current algorithm before advancing to the next algorithm. This gives all
    # workers to AG-VS first, then MCTS-VS, then the remaining requested methods
    # in their configured order instead of interleaving one run from each.
    for benchmark in benchmarks:
        for algorithm in algorithm_phase_order:
            pending_trials = pending_trials_by_pair[(benchmark, algorithm)]
            if not pending_trials:
                continue
            uses_gpu = algorithm_uses_gpu_worker_pool(algorithm, device)
            worker_limit = algorithm_worker_limit(algorithm, workers, device)
            phase_workers = max(1, min(int(worker_limit), len(pending_trials)))
            phase_args = [
                (
                    trial,
                    int(trials),
                    benchmarks,
                    algorithms,
                    algorithm_evaluation_budget(algorithm, max_iter, benchmark),
                    output_dir,
                    seed,
                    skip_complete,
                    figure_dir,
                    device,
                    [(benchmark, algorithm)],
                    bool(resume_under_budget),
                )
                for trial in pending_trials
            ]
            phase_label = "cuda-gpu" if uses_gpu else "parallel"
            print(
                f"[{phase_label}] benchmark={benchmark} algorithm={algorithm}: "
                f"finishing {len(phase_args)} trial(s) with {phase_workers} worker(s).",
                flush=True,
            )
            if phase_workers == 1:
                for args in phase_args:
                    trial_results.extend(_run_round_robin_trial(args))
                continue
            if _MPL_LIVE_PLOT_ON:
                raise ValueError(
                    "Parallel trials cannot be combined with AGVS_PYTHON_LIVE_PLOT_ON."
                )
            context = (
                multiprocessing.get_context("fork") if os.name == "posix" else None
            )
            executor_kwargs = {"max_workers": phase_workers}
            if context is not None:
                executor_kwargs["mp_context"] = context
            if _RECYCLE_WORKERS_ON and os.name != "posix" and (
                "max_tasks_per_child"
                in inspect.signature(concurrent.futures.ProcessPoolExecutor).parameters
            ):
                executor_kwargs["max_tasks_per_child"] = 1
            try:
                with concurrent.futures.ProcessPoolExecutor(**executor_kwargs) as executor:
                    for results in executor.map(_run_round_robin_trial, phase_args):
                        trial_results.extend(results)
            except Exception as exc:
                # A Windows worker can disappear after finishing optimization
                # while serializing/writing a large trace. Preserve every JSON
                # that reached disk, then finish only the genuinely missing
                # trials in-process instead of aborting the whole benchmark.
                print(
                    f"[parallel-recovery] benchmark={benchmark} algorithm={algorithm}: "
                    f"{type(exc).__name__}: {exc}; rescanning JSON and retrying "
                    "missing trials sequentially.",
                    flush=True,
                )
                recovered = existing_complete_traces(
                    output_dir,
                    [benchmark],
                    [algorithm],
                    trials,
                    max_iter=max_iter,
                    base_seed_offset=seed,
                    device=device,
                    reuse_complete_json=reuse_complete_json,
                )
                pending_ids = set(pending_trials)
                trial_results = [
                    result
                    for result in trial_results
                    if not (
                        result.get("benchmark") == benchmark
                        and result.get("algorithm") == algorithm
                        and result.get("trial_id") in pending_ids
                    )
                ]
                for phase_arg in phase_args:
                    trial = int(phase_arg[0])
                    if (trial, benchmark, algorithm) in recovered:
                        trial_results.append(
                            {
                                "benchmark": benchmark,
                                "algorithm": algorithm,
                                "trial_id": trial,
                                "complete": True,
                                "error": None,
                                "recovered_after_worker_failure": True,
                            }
                        )
                    else:
                        trial_results.extend(_run_round_robin_trial(phase_arg))

    expected_run_count = (int(trials) - trial_start) * len(benchmarks) * len(algorithms)
    failed_runs = []
    for result in trial_results:
        if (
            not isinstance(result, dict)
            or result.get("complete") is not True
            or result.get("error") not in (None, "", False)
        ):
            failed_runs.append(
                {
                    "benchmark": result.get("benchmark") if isinstance(result, dict) else None,
                    "algorithm": result.get("algorithm") if isinstance(result, dict) else None,
                    "trial_id": result.get("trial_id") if isinstance(result, dict) else None,
                    "error": result.get("error") if isinstance(result, dict) else "missing trial result",
                }
            )
    missing_run_count = max(0, expected_run_count - len(trial_results))
    aggregate_complete = not failed_runs and missing_run_count == 0

    status_path = (
        Path(output_dir) / benchmark_output_slug(benchmarks[0]) / "run_status.json"
        if len(benchmarks) == 1
        else Path(output_dir) / "Status" / "run_status.json"
    )
    requested_specs = {name: BENCHMARKS[name] for name in benchmarks}
    status_payload = {
            "complete": bool(aggregate_complete),
            "message": (
                "Round-robin experiment finished successfully."
                if aggregate_complete
                else "Round-robin experiment finished with failed or incomplete trials."
            ),
            "expected_run_count": int(expected_run_count),
            "returned_run_count": int(len(trial_results)),
            "completed_run_count": int(len(trial_results) - len(failed_runs)),
            "failed_run_count": int(len(failed_runs)),
            "missing_run_count": int(missing_run_count),
            "failed_runs": failed_runs,
            "trials": int(trials),
            "trial_start": trial_start,
            "benchmarks": benchmarks,
            "algorithms": algorithms,
            "algorithm_phase_order": algorithm_phase_order,
            "order": "for benchmark -> for algorithm -> parallel trials; algorithm barrier",
            "trial_ids": list(range(trial_start, int(trials))),
            "workers": workers,
            "requested_workers": requested_workers,
            "resume_under_budget": bool(resume_under_budget),
            "cpu_workers": workers,
            "device": device,
            "algorithm_devices": {
                algorithm: algorithm_device(algorithm, device)
                for algorithm in algorithms
            },
            "dimension_invariant_reuse_count": len(reused_dimension_traces),
            "dimension_invariant_reuse": [
                {
                    "target_benchmark": target,
                    "algorithm": algorithm,
                    "trial_id": trial_id,
                    "source_benchmark": source,
                }
                for target, algorithm, trial_id, source in reused_dimension_traces
            ],
            "benchmark_noise_std": {
                name: float(BENCHMARKS[name].get("benchmark_noise_std", 0.0))
                for name in benchmarks
            },
            "algorithm_seed_rule": (
                f"synthetic and real_life: {BASE_ALGORITHM_SEED} + "
                "base_seed_offset + trial_id"
            ),
            "benchmark_seed_rule": (
                f"{BASE_BENCHMARK_SEED} + base_seed_offset + trial_id"
            ),
            "risk_tolerances": {risk_label(delta): float(delta) for delta in RISK_TOLERANCES},
            "experiment_kinds": {
                name: str(spec.get("experiment_kind", "synthetic"))
                for name, spec in requested_specs.items()
            },
            "active_coordinate_model": (
                "Synthetic benchmarks record ground-truth active indices. Real-life benchmarks "
                "record only an algorithmic search-dimension assumption."
            ),
            "active_indices": {
                name: (
                    np.asarray(spec["active_indices"], dtype=int).tolist()
                    if spec.get("active_indices") is not None
                    else None
                )
                for name, spec in requested_specs.items()
            },
            "effective_dimension_seeds": {
                name: spec.get("effective_dimension_seed")
                for name, spec in requested_specs.items()
            },
            "effective_dims": {name: int(benchmark_effective_dim(spec)) for name, spec in requested_specs.items()},
            "effective_dim_roles": {
                name: str(spec.get("effective_dim_role", "ground_truth"))
                for name, spec in requested_specs.items()
            },
            "lipschitz_constants": {
                name: (float(spec["L"]) if spec.get("L") is not None else None)
                for name, spec in requested_specs.items()
            },
            "ag_vs_q_batch_policy": (
                "runner central controls; exactly one of q_batch/batch_size is active"
            ),
            "ag_vs_q_batch_by_benchmark": {
                name: (
                    None
                    if benchmark_q_batch(spec) is None
                    else int(benchmark_q_batch(spec))
                )
                for name, spec in requested_specs.items()
            },
            "ag_vs_batch_parameter_policy": "exactly one of q_batch or batch_size is non-null",
            "ag_vs_batch_size_by_benchmark": {
                name: (
                    None
                    if benchmark_batch_size(spec) is None
                    else int(benchmark_batch_size(spec))
                )
                for name, spec in requested_specs.items()
            },
            "ag_vs_t_seq_by_benchmark": {
                name: int(spec.get("t_seq", AGVS_T_SEQ))
                for name, spec in requested_specs.items()
            },
            "ag_vs_n_init_by_benchmark": {
                name: int(benchmark_n_init(spec)) for name, spec in requested_specs.items()
            },
            "ag_vs_abs_mode": bool(AGVS_ABS_MODE),
            "ag_vs_score_formula": (
                "mean(abs(y_obs_i - best_y_obs_i_minus_1))"
                if AGVS_ABS_MODE
                else "mean(y_obs_i - best_y_obs_i_minus_1)"
            ),
            "ag_vs_acquisition": STANDARD_BATCH_ACQUISITION,
            "ag_vs_acquisition_sampling_strategy": AGVS_ACQUISITION_SAMPLING_STRATEGY,
            "ag_vs_turbo_settings": {
                "n_init": AGVS_TURBO_N_INIT,
                "turbo_max_evals": AGVS_TURBO_MAX_EVALS,
                "use_ard": AGVS_TURBO_USE_ARD,
                "max_cholesky_size": AGVS_TURBO_MAX_CHOLESKY_SIZE,
                "n_training_steps": AGVS_TURBO_N_TRAINING_STEPS,
                "min_cuda": AGVS_TURBO_MIN_CUDA,
                "device": algorithm_device("AG-VS-TuRBO", device),
                "dtype": AGVS_TURBO_DTYPE,
            },
            "mcts_vs_turbo_settings": {
                "n_init": MCTS_TURBO_N_INIT,
                "turbo_max_evals": MCTS_TURBO_MAX_EVALS,
                "batch_size": MCTS_TURBO_BATCH_SIZE,
                "use_ard": MCTS_TURBO_USE_ARD,
                "max_cholesky_size": MCTS_TURBO_MAX_CHOLESKY_SIZE,
                "n_training_steps": MCTS_TURBO_N_TRAINING_STEPS,
                "min_cuda": MCTS_TURBO_MIN_CUDA,
                "device": algorithm_device("MCTS-VS-TuRBO", device),
                "dtype": MCTS_TURBO_DTYPE,
            },
            "turbo_n_init": TURBO_N_INIT,
            "turbo_batch_size": TURBO_BATCH_SIZE,
            "turbo_use_ard": TURBO_USE_ARD,
            "turbo_max_cholesky_size": TURBO_MAX_CHOLESKY_SIZE,
            "turbo_n_training_steps": TURBO_N_TRAINING_STEPS,
            "turbo_min_cuda": TURBO_MIN_CUDA,
            "turbo_device": algorithm_device("TuRBO", device),
            "turbo_dtype": TURBO_DTYPE,
            "turbo_verbose": TURBO_VERBOSE,
        }
    status_payload["status_path"] = str(status_path)
    write_json(status_path, status_payload)
    return status_payload


def _run_round_robin_trial(args):
    if len(args) == 10:
        (
            trial,
            trials,
            benchmarks,
            algorithms,
            max_iter,
            output_dir,
            seed,
            skip_complete,
            figure_dir,
            device,
        ) = args
        requested_pairs = [
            (benchmark, algorithm)
            for benchmark in benchmarks
            for algorithm in algorithms
        ]
        resume_under_budget = False
    elif len(args) == 11:
        (
            trial,
            trials,
            benchmarks,
            algorithms,
            max_iter,
            output_dir,
            seed,
            skip_complete,
            figure_dir,
            device,
            requested_pairs,
        ) = args
        resume_under_budget = False
    else:
        (
            trial,
            trials,
            benchmarks,
            algorithms,
            max_iter,
            output_dir,
            seed,
            skip_complete,
            figure_dir,
            device,
            requested_pairs,
            resume_under_budget,
        ) = args
    print("\n" + "=" * 90)
    print(f"ROUND-ROBIN trial_id={trial} ({trial + 1}/{trials})")
    print("=" * 90)
    results = []
    for benchmark, algorithm in requested_pairs:
        try:
            trace = run_single_trial(
                benchmark_name=benchmark,
                algorithm=algorithm,
                trial=trial,
                max_iter=max_iter,
                output_dir=output_dir,
                base_seed=seed,
                skip_complete=skip_complete,
                artifact_dir=figure_dir,
                show_final_tree=AGVS_SHOW_FINAL_TREE,
                device=device,
                resume_under_budget=bool(resume_under_budget),
            )
        except KeyboardInterrupt:
            raise
        except BaseException as exc:
            trace = {
                "benchmark": benchmark,
                "algorithm": algorithm,
                "trial_id": trial,
                "complete": False,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
                "failure_stage": "trial_setup_or_result_processing",
            }
            # A constructor/import failure occurs outside the normal trace
            # writer. Preserve its diagnosis instead of silently advancing to
            # the next algorithm. Never replace an existing evaluation trace.
            failure_path = trace_path_for(output_dir, benchmark, algorithm, trial)
            if failure_path.exists():
                failure_path = failure_path.with_suffix(".failure.json")
            try:
                write_json(failure_path, trace)
                details = f"; details={failure_path}"
            except OSError as write_error:
                details = f"; could not save failure details: {write_error}"
            print(
                f"[TRIAL-FAILED] benchmark={benchmark} algorithm={algorithm} "
                f"trial={int(trial):03d}: {trace['error']}{details}",
                flush=True,
            )
        results.append(
            {
                "benchmark": trace.get("benchmark", benchmark),
                "algorithm": trace.get("algorithm", algorithm),
                "trial_id": trace.get("trial_id", trial),
                "complete": trace.get("complete") is True,
                "error": trace.get("error"),
            }
            if isinstance(trace, dict)
            else {
                "benchmark": benchmark,
                "algorithm": algorithm,
                "trial_id": trial,
                "complete": False,
                "error": "missing trial trace",
            }
        )
    return results


# =============================================================================
# Plotting
# =============================================================================

PLOT_TRACE_KEYS = frozenset(
    {
        "algorithm",
        "benchmark",
        "cache_identity",
        "algorithm_seed",
        "benchmark_seed",
        "trial_id",
        "trial",
        "complete",
        "error",
        "settings",
        "wall_clock_seconds",
        "evaluation_wall_clock_history",
        "n_evaluations",
        "best_observed_history",
        "observed_y_history",
        "best_noiseless_y_history",
        "true_y_history",
        "reported_posterior_mean_history",
        "best_posterior_mean_history",
        "best_threshold_history_by_delta",
        "agvs_dimension_selection_counts",
        "final_best_observed_at_max_iter",
        "best_observed",
        "final_best_noiseless_y_at_max_iter",
        "best_noiseless_y",
        "final_best_posterior_mean_at_max_iter",
        "best_posterior_mean",
        "final_best_stoppable_threshold_at_max_iter_1_256",
        "final_best_stoppable_threshold_at_max_iter",
    }
)
_TOP_LEVEL_PRETTY_JSON_KEY = re.compile(br'^  "([^"\\]+)": ', re.MULTILINE)


def load_plot_trace(path):
    """Read only plot/table fields, skipping multi-gigabyte candidate arrays."""

    path = Path(path)
    with path.open("rb") as handle:
        if path.stat().st_size == 0:
            return {}
        with mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as mapped:
            matches = _TOP_LEVEL_PRETTY_JSON_KEY.finditer(mapped)
            previous = next(matches, None)
            if previous is None:
                mapped.seek(0)
                return json.loads(mapped.read().decode("utf-8"))

            trace = {}
            for current in matches:
                key = previous.group(1).decode("utf-8")
                if key in PLOT_TRACE_KEYS:
                    raw_value = mapped[previous.end():current.start()].rstrip()
                    if raw_value.endswith(b","):
                        raw_value = raw_value[:-1].rstrip()
                    trace[key] = json.loads(raw_value)
                previous = current

            key = previous.group(1).decode("utf-8")
            if key in PLOT_TRACE_KEYS:
                raw_value = mapped[previous.end():].rstrip()
                if raw_value.endswith(b"}"):
                    raw_value = raw_value[:-1].rstrip()
                if raw_value.endswith(b","):
                    raw_value = raw_value[:-1].rstrip()
                trace[key] = json.loads(raw_value)
            return trace


def load_traces(results_dir, reuse_complete_json=False, trial_limits=None):
    trial_limits = {
        str(name).strip().lower(): int(limit)
        for name, limit in (trial_limits or {}).items()
    }
    traces = []
    for path in Path(results_dir).rglob("trial_*.json"):
        trace = load_plot_trace(path)
        identity = trace.get("cache_identity", {})
        if (
            identity.get("implementation_version") != "native-v1"
            and not reuse_complete_json
        ):
            continue
        try:
            trial = int(trace.get("trial_id", trace.get("trial", -1)))
            benchmark = str(trace["benchmark"]).strip().lower()
            if benchmark in trial_limits:
                trial_limit = trial_limits[benchmark]
            elif benchmark in SYNTHETIC_EXPERIMENTS:
                trial_limit = int(SYNTHETIC_TRIALS)
            elif benchmark in REAL_LIFE_EXPERIMENTS:
                trial_limit = int(REAL_LIFE_EXPERIMENTS[benchmark].trials)
            else:
                continue
            if not (0 <= trial < trial_limit) or any(int(trace.get(key, -1)) != 2021 + trial for key in ("algorithm_seed", "benchmark_seed")):
                continue
            algorithm = canonical_algorithm_name(trace.get("algorithm"))
            config = apply_runner_agvs_batch_policy(BENCHMARKS[benchmark])
            requested_max_iter = int(identity["requested_max_iter"])
            expected = requested_trace_cache_identity(config, benchmark, algorithm,
                trial, requested_max_iter, device=DEVICE)
            if not cache_identities_match(identity, expected):
                if not reuse_complete_json or not completed_json_matches_request(
                    trace,
                    expected_identity=expected,
                    benchmark_name=benchmark,
                    algorithm=algorithm,
                    trial=trial,
                ):
                    continue
                requested_budget = algorithm_evaluation_budget(
                    algorithm, requested_max_iter, benchmark
                )
                if int(trace.get("n_evaluations", 0) or 0) < requested_budget:
                    continue
        except (KeyError, TypeError, ValueError):
            continue
        traces.append(trace)
    return traces


def pad_histories(histories):
    histories = [h for h in histories if h is not None and len(h) > 0]
    if not histories:
        return np.empty((0, 0), dtype=float)
    max_len = max(len(h) for h in histories)
    arr = np.full((len(histories), max_len), np.nan, dtype=float)
    for i, h in enumerate(histories):
        h = np.asarray(h, dtype=float)
        arr[i, : len(h)] = h
        if len(h) < max_len and len(h) > 0:
            arr[i, len(h):] = h[-1]
    return arr


def mean_ci_from_histories(histories, start_at_zero=True, zero_value="first"):
    data = pad_histories(histories)
    if data.size == 0:
        return None, None, None, 0
    if start_at_zero:
        if zero_value == "zero":
            first_col = np.zeros((data.shape[0], 1), dtype=float)
        else:
            first_col = data[:, :1].copy()
        data = np.hstack([first_col, data])
    n = data.shape[0]
    mean = np.nanmean(data, axis=0)
    std = np.nanstd(data, axis=0, ddof=1) if n > 1 else np.zeros_like(mean)
    ci = 1.96 * std / math.sqrt(max(n, 1))
    x_axis = np.arange(0, len(mean)) if start_at_zero else np.arange(1, len(mean) + 1)
    return x_axis, mean, ci, n


def terminal_mean_ci_from_histories(histories):
    """Return mean and 95% CI of each trial's final finite wall-clock value."""

    terminal_values = []
    for history in histories or []:
        values = np.asarray(history, dtype=float).reshape(-1)
        finite_values = values[np.isfinite(values)]
        if finite_values.size:
            terminal_values.append(float(finite_values[-1]))
    if not terminal_values:
        return None, None, 0
    data = np.asarray(terminal_values, dtype=float)
    n = int(data.size)
    mean = float(np.mean(data))
    std = float(np.std(data, ddof=1)) if n > 1 else 0.0
    ci = 1.96 * std / math.sqrt(max(n, 1))
    return mean, float(ci), n


def wall_clock_mean_ci_from_traces(
    items,
    history_getter,
    algorithm,
    minimum_grid_step_milliseconds=1,
    maximum_time_milliseconds=None,
    maximum_grid_points=2000,
    exclude_initial_design=True,
    extend_completed_trials=False,
):
    """Aggregate running-best traces on a common millisecond wall-clock grid.

    Each trial is treated as a right-continuous step function.  The common
    horizon is normally the shortest completed trial for the algorithm. When
    ``extend_completed_trials`` is true, completed trials retain their final
    incumbent until the longest trial finishes; this makes an anytime curve's
    endpoint equal the fixed-budget terminal mean. By default, time spent
    producing ``n_init`` is removed and the running-best value after the
    initial design is placed at post-initialization time zero.
    """

    curves = []
    for item in items or []:
        if not isinstance(item, dict):
            continue
        values = np.asarray(history_getter(item), dtype=float).reshape(-1)
        raw_times = item.get("evaluation_wall_clock_history", [])
        n_evaluations = int(item.get("n_evaluations", len(values)))
        if n_evaluations <= 0 or values.size == 0:
            continue
        total_time = float(item.get("wall_clock_seconds", 0.0))
        times = np.asarray(
            batch_parallel_wall_clock_history(
                raw_times,
                algorithm=str(item.get("algorithm", algorithm)),
                result=item,
                n_evaluations=n_evaluations,
                wall_clock=total_time,
                q_batch=_trace_q_batch(item, default=LEGACY_TRACE_BATCH_FALLBACK),
            ),
            dtype=float,
        ).reshape(-1)
        usable = min(values.size, times.size, n_evaluations)
        if usable <= 0:
            continue
        values = values[:usable]
        times = times[:usable]
        valid = np.isfinite(values) & np.isfinite(times) & (times >= 0.0)
        if not np.any(valid):
            continue
        values = values[valid]
        times = np.maximum.accumulate(times[valid])
        if values.size == 0 or times.size == 0 or float(times[-1]) <= 0.0:
            continue
        if exclude_initial_design:
            n_init = max(0, int(plot_marker_start_after_n_init(item)))
            if n_init > 0:
                initial_end_index = min(n_init, values.size) - 1
                initial_end_time = float(times[initial_end_index])
                values = values[initial_end_index:]
                times = times[initial_end_index:] - initial_end_time
                if values.size < 2 or float(times[-1]) <= 0.0:
                    continue
        curves.append((times, values))

    if not curves:
        return None, None, None, 0

    # Persisted timing stays in seconds. Convert only the plot grid to
    # milliseconds so fast methods do not collapse to a single integer-second
    # point and appear to be missing from the comparison.
    curves = [(times * 1000.0, values) for times, values in curves]
    terminal_times = [float(times[-1]) for times, _values in curves]
    horizon = (
        max(terminal_times) if extend_completed_trials else min(terminal_times)
    )
    if maximum_time_milliseconds is not None:
        horizon = min(horizon, float(maximum_time_milliseconds))
    # Start only once every trial has completed at least one evaluation. This
    # prevents assigning a future first observation to an earlier wall-clock
    # millisecond. End at the shortest completed trial so every grid point uses the
    # same trial set in the mean and confidence interval.
    first_common_millisecond = int(
        math.ceil(max(float(times[0]) for times, _values in curves))
    )
    # Elapsed seconds often produce values such as 499.99999999999994 ms.
    # Keep the mathematically intended terminal millisecond on the grid.
    last_common_millisecond = int(math.floor(horizon + 1e-6))
    if (
        not np.isfinite(horizon)
        or horizon <= 0.0
        or last_common_millisecond < first_common_millisecond
    ):
        return None, None, None, 0
    # Preserve millisecond detail for short NAS runs but bound memory for long
    # MuJoCo runs.
    grid_step_milliseconds = max(
        max(1, int(minimum_grid_step_milliseconds)),
        int(
            math.ceil(
                (last_common_millisecond - first_common_millisecond)
                / max(1, int(maximum_grid_points))
            )
        ),
    )
    grid = np.arange(
        first_common_millisecond,
        last_common_millisecond + 1,
        grid_step_milliseconds,
        dtype=float,
    )
    if extend_completed_trials and grid[-1] < last_common_millisecond:
        grid = np.append(grid, float(last_common_millisecond))
    sampled = np.empty((len(curves), len(grid)), dtype=float)
    for row, (times, values) in enumerate(curves):
        indices = np.searchsorted(times, grid, side="right") - 1
        # At millisecond t, use exactly the last running-best observation whose
        # evaluation completed at or before t.
        indices = np.clip(indices, 0, len(values) - 1)
        sampled[row] = values[indices]

    n = sampled.shape[0]
    mean = np.mean(sampled, axis=0)
    std = np.std(sampled, axis=0, ddof=1) if n > 1 else np.zeros_like(mean)
    ci = 1.96 * std / math.sqrt(max(n, 1))
    return grid, mean, ci, n


def bernoulli_wilson_ci_from_histories(histories, start_at_zero=True, z=1.96):
    data = pad_histories(histories)
    if data.size == 0:
        return None, None, None, None, 0
    data = np.where(np.isfinite(data), data, np.nan)
    if start_at_zero:
        first_col = data[:, :1].copy()
        data = np.hstack([first_col, data])
    valid = np.isfinite(data)
    successes = np.nansum(data, axis=0)
    n = np.sum(valid, axis=0).astype(float)
    rate = np.divide(successes, n, out=np.full_like(successes, np.nan, dtype=float), where=n > 0)

    z2 = float(z) ** 2
    denom = 1.0 + z2 / np.maximum(n, 1.0)
    center = (rate + z2 / (2.0 * np.maximum(n, 1.0))) / denom
    radius = (
        float(z)
        * np.sqrt((rate * (1.0 - rate) / np.maximum(n, 1.0)) + z2 / (4.0 * np.maximum(n, 1.0) ** 2))
        / denom
    )
    lower = np.clip(center - radius, 0.0, 1.0)
    upper = np.clip(center + radius, 0.0, 1.0)
    lower[n <= 0] = np.nan
    upper[n <= 0] = np.nan
    x_axis = np.arange(0, len(rate)) if start_at_zero else np.arange(1, len(rate) + 1)
    return x_axis, rate, lower, upper, int(np.nanmax(n)) if n.size else 0


def success_history_from_trace(item, delta_key):
    if not isinstance(item, dict):
        return []
    noiseless = item.get("best_noiseless_y_history")
    if not noiseless:
        true_hist = item.get("true_y_history", [])
        noiseless = best_so_far(true_hist) if true_hist else []
    threshold_by_delta = item.get("best_threshold_history_by_delta", {})
    threshold = None
    if isinstance(threshold_by_delta, dict):
        threshold = threshold_by_delta.get(delta_key)
        if threshold is None and threshold_by_delta:
            threshold = next(iter(threshold_by_delta.values()))
    if not noiseless or not threshold:
        return []
    usable_len = min(len(noiseless), len(threshold))
    if usable_len <= 0:
        return []
    noiseless_arr = np.asarray(noiseless[:usable_len], dtype=float)
    threshold_arr = np.asarray(threshold[:usable_len], dtype=float)
    valid = np.isfinite(noiseless_arr) & np.isfinite(threshold_arr)
    out = np.full(usable_len, np.nan, dtype=float)
    out[valid] = (noiseless_arr[valid] >= threshold_arr[valid]).astype(float)
    return out.tolist()


def agvs_terminal_merge_iteration(item):
    if not isinstance(item, dict):
        return None
    records = item.get("agvs_merge_history")
    if not isinstance(records, list):
        return None
    starts = []
    for record in records:
        if not isinstance(record, dict):
            continue
        if not bool(record.get("merged_terminal_after_child_trials", False)):
            continue
        try:
            iteration = int(record.get("after_iteration"))
        except Exception:
            continue
        if iteration > 0:
            starts.append(iteration)
    return min(starts) if starts else None


def terminal_start_by_trial_for_items(items):
    out = {}
    for item in items or []:
        iteration = agvs_terminal_merge_iteration(item)
        if iteration is None:
            continue
        try:
            trial = int(item.get("trial"))
        except Exception:
            continue
        out[trial] = int(iteration)
    return out


def current_threshold_history_from_trace(item, delta_key):
    if not isinstance(item, dict):
        return []
    settings = item.get("settings", {})
    try:
        n_evaluations = int(item.get("n_evaluations", len(item.get("true_y_history", []))))
    except Exception:
        n_evaluations = len(item.get("true_y_history", []))

    traces = []
    for history_key in ("agvs_history", "vanilla_bo_history", "history"):
        history = item.get(history_key)
        if not isinstance(history, list):
            continue
        for record in history:
            if not isinstance(record, dict):
                continue
            entry = record.get("threshold_entry")
            if entry is None:
                entry = record.get("stopping_info")
            if not isinstance(entry, dict):
                continue
            current_by_delta = entry.get("current_stoppable_threshold_by_delta", {})
            if not isinstance(current_by_delta, dict):
                continue
            value = current_by_delta.get(delta_key)
            if value is None:
                for delta in RISK_TOLERANCES:
                    if risk_label(delta) != delta_key:
                        continue
                    value = current_by_delta.get(f"{float(delta):.12g}")
                    if value is None:
                        value = current_by_delta.get(str(float(delta)))
                    if value is not None:
                        break
            try:
                value = float(value)
                n_eval = int(entry.get("n", record.get("iteration", record.get("n_eval"))))
            except Exception:
                continue
            if np.isfinite(value) and n_eval > 0:
                traces.append((n_eval, value))

    if not traces:
        return []

    out = [None] * max(0, int(n_evaluations))
    for n_eval, value in sorted(traces):
        idx = min(max(int(n_eval) - 1, 0), len(out) - 1)
        out[idx] = float(value)
    return hide_threshold_before_initial_gp(out, item)


def post_terminal_best_noiseless_history(item, terminal_start_iteration):
    true_hist = item.get("true_y_history", []) if isinstance(item, dict) else []
    if not true_hist:
        return []
    try:
        start_index = max(0, int(terminal_start_iteration) - 1)
    except Exception:
        return []
    values = np.asarray(true_hist, dtype=float)
    if start_index >= values.size:
        return []
    out = np.full(values.size, np.nan, dtype=float)
    tail = values[start_index:]
    valid = np.isfinite(tail)
    if not np.any(valid):
        return []
    clean = np.where(valid, tail, -np.inf)
    best_tail = np.maximum.accumulate(clean)
    best_tail[~np.isfinite(best_tail)] = np.nan
    out[start_index:] = best_tail
    return out.tolist()


def post_terminal_best_threshold_history(item, delta_key, terminal_start_iteration):
    current = current_threshold_history_from_trace(item, delta_key)
    if not current:
        current = threshold_history_from_trace(item, delta_key)
    if not current:
        return []
    try:
        start_index = max(0, int(terminal_start_iteration) - 1)
    except Exception:
        return []
    values = np.asarray([np.nan if value is None else value for value in current], dtype=float)
    if start_index >= values.size:
        return []
    out = np.full(values.size, np.nan, dtype=float)
    tail = values[start_index:]
    valid = np.isfinite(tail)
    if not np.any(valid):
        return []
    clean = np.where(valid, tail, -np.inf)
    best_tail = np.maximum.accumulate(clean)
    best_tail[~np.isfinite(best_tail)] = np.nan
    out[start_index:] = best_tail
    return out.tolist()


def terminate_success_history_from_trace(item, delta_key, terminal_start_iteration):
    noiseless = post_terminal_best_noiseless_history(item, terminal_start_iteration)
    threshold = post_terminal_best_threshold_history(item, delta_key, terminal_start_iteration)
    if not noiseless or not threshold:
        return []
    usable_len = min(len(noiseless), len(threshold))
    if usable_len <= 0:
        return []
    noiseless_arr = np.asarray(noiseless[:usable_len], dtype=float)
    threshold_arr = np.asarray(threshold[:usable_len], dtype=float)
    valid = np.isfinite(noiseless_arr) & np.isfinite(threshold_arr)
    out = np.full(usable_len, np.nan, dtype=float)
    out[valid] = (noiseless_arr[valid] >= threshold_arr[valid]).astype(float)
    return out.tolist()


def expected_iter_from_trace(item):
    if not isinstance(item, dict):
        return None
    settings = item.get("settings", {})
    algorithm_settings = settings.get("algorithm_settings", {}) if isinstance(settings, dict) else {}
    if isinstance(algorithm_settings, dict):
        if item.get("algorithm") in AGVS_ALGORITHMS:
            if not bool(algorithm_settings.get("sim_mode", False)):
                return None
            try:
                from ag_vs.utils import find_optimum_B_and_cost

                D = int(settings.get("original_dim"))
                d_eff = int(algorithm_settings.get("effective_dim", settings.get("effective_dim")))
                n_init = int(algorithm_settings.get("n_init", benchmark_n_init(settings)))
                q_value = algorithm_settings.get("q_batch")
                batch_value = algorithm_settings.get("batch_size")
                q_batch = None if q_value is None else int(q_value)
                batch_size = None if batch_value is None else int(batch_value)
                t_seq = int(algorithm_settings.get("t_seq", AGVS_T_SEQ))
                _split, cost = find_optimum_B_and_cost(
                    D, d_eff, n_init, q_batch, batch_size, t_seq
                )
                return int(cost)
            except Exception:
                return None
    return None


def expected_iter_text_for_items(items):
    values = []
    for item in items or []:
        value = expected_iter_from_trace(item)
        if value is not None:
            values.append(int(value))
    if not values:
        return None
    unique = sorted(set(values))
    if len(unique) == 1:
        return f"AG-VS expected_iter={unique[0]}"
    shown = ",".join(str(value) for value in unique[:4])
    if len(unique) > 4:
        shown += ",..."
    return f"AG-VS expected_iter={shown}"


def terminal_iter_text_from_starts(terminal_start_by_trial):
    starts = [int(value) for value in (terminal_start_by_trial or {}).values() if value is not None]
    if not starts:
        return None
    starts = sorted(starts)
    median = int(np.median(np.asarray(starts, dtype=float)))
    if starts[0] == starts[-1]:
        return f"terminate_iter={starts[0]}"
    return f"terminate_iter median={median} range={starts[0]}-{starts[-1]}"


def annotate_plot_corner(ax, lines):
    lines = [line for line in (lines or []) if line]
    if not lines:
        return
    ax.text(
        0.99,
        0.03,
        "\n".join(lines),
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.8, "edgecolor": "0.75"},
    )


def threshold_history_from_trace(item, delta_key):
    if not isinstance(item, dict):
        return []
    threshold_by_delta = item.get("best_threshold_history_by_delta", {})
    if not isinstance(threshold_by_delta, dict) or delta_key not in threshold_by_delta:
        return []
    return hide_threshold_before_initial_gp(threshold_by_delta[delta_key], item)


PLOT_MAX_EVALUATION = MAX_EXPERIMENT
PLOT_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">", "h", "p"]
PLOT_LINESTYLES = ["-", "--", "-.", ":"]
PLOT_COLORS = [
    "#1f77b4",  # blue
    "#ff7f0e",  # orange
    "#2ca02c",  # green
    "#d62728",  # red
    "#9467bd",  # purple
    "#8c564b",  # brown
    "#008c8c",  # teal
    "#7f7f7f",  # gray
    "#bcbd22",  # olive
    "#17becf",  # cyan
    "#003f5c",  # navy
    "#ffa600",  # gold
    "#e377c2",  # magenta
    "#111111",  # black
    "#f95d6a",  # coral
    "#6b705c",  # muted green-gray
    "#665191",  # indigo
    "#a05195",  # plum
    "#2f4b7c",  # steel blue
    "#00a087",  # emerald
]

ALGORITHM_STYLE_OVERRIDES = {
    # Keep AG-VS in blue and LA-MCTS in orange so the two method families
    # remain easy to distinguish, including for common color-vision deficits.
    "AG-VS-BO": {"linestyle": "-", "color": "#0072b2"},
    "AG-VS-TuRBO": {"linestyle": "--", "color": "#56b4e9"},
    "MCTS-VS-BO": {"linestyle": "-.", "color": "#9467bd"},
    "MCTS-VS-TuRBO": {"linestyle": "-.", "color": "#6b705c"},
    "MCTS-VS-SAASBO": {"linestyle": ":", "color": "#00a087"},
    "ADD-GP-UCB": {"linestyle": ":", "color": "#003f5c"},
    "TuRBO": {"linestyle": "-.", "color": "#d62728"},
    "VanillaBO": {"linestyle": "-", "color": "#1f77b4"},
    "Dropout-BO": {"linestyle": "--", "color": "#ff7f0e"},
    "Dropout-TuRBO": {"linestyle": "--", "color": "#17becf"},
    "LassoBO": {"linestyle": "--", "color": "#8c564b"},
    "LA-MCTS-BO": {"linestyle": "-", "color": "#e69f00"},
    "LA-MCTS-TuRBO": {"linestyle": ":", "color": "#d55e00"},
    "CMA-ES": {"linestyle": "-", "color": "#665191"},
    "SAASBO": {"linestyle": ":", "color": "#f95d6a"},
    "HeSBO": {"linestyle": "--", "color": "#bcbd22"},
    "ALEBO": {"linestyle": "-", "color": "#a05195"},
    "VAE-BO": {"linestyle": "--", "color": "#e377c2"},
}

TURBO_BASED_ALGORITHMS = frozenset(
    {
        "AG-VS-TuRBO",
        "MCTS-VS-TuRBO",
        "TuRBO",
        "Dropout-TuRBO",
        "LA-MCTS-TuRBO",
    }
)
ALGORITHM_OUTPUT_FAMILIES = ("bo", "turbo")
FIGURE_OUTPUT_FAMILIES = (*ALGORITHM_OUTPUT_FAMILIES, "united")

# Paper-style paired synthetic summary figures.  The two configured experiment
# scales are shown side by side with their terminal wall-clock comparisons.
SYNTHETIC_GRAPH_Y_LIMITS = {
    "ackley3_300": (-14.0, 0.0),
    "ackley6_1000": (-14.0, 0.0),
    "levy3_300": (-2.0, 0.0),
    "levy6_1000": (-10.0, 0.0),
    "rastrigin3_300": (-10.0, 0.0),
    "rastrigin6_1000": (-80.0, 0.0),
    "rosenbrock3_300": (-5.0, 0.0),
    "rosenbrock6_1000": (-100.0, 0.0),
    "sphere3_300": (-0.5, 0.0),
    "sphere6_1000": (-2.0, 0.0),
}
SYNTHETIC_GRAPH_OBJECTIVES = (
    "ackley",
    "levy",
    "rastrigin",
    "rosenbrock",
    "sphere",
)


def figure_output_family_is_available(traces, family):
    """Return whether a backend-specific comparison can be drawn."""

    family = str(family).strip().lower()
    if family not in FIGURE_OUTPUT_FAMILIES:
        raise ValueError(f"Unknown algorithm output family: {family}")
    if not traces:
        return False
    if family == "united":
        return True
    return any(
        comparison_output_family_includes(trace.get("algorithm"), family)
        for trace in traces
    )


def algorithm_output_family(algorithm):
    """Return the stable output family used by tables and figures."""

    try:
        canonical = canonical_algorithm_name(str(algorithm))
    except Exception:
        canonical = str(algorithm)
    return "turbo" if canonical in TURBO_BASED_ALGORITHMS else "bo"


def comparison_output_family_includes(algorithm, family):
    """Return whether an algorithm belongs in a comparison output folder.

    ``bo`` contains only BO-backend algorithms, ``turbo`` contains only
    TuRBO-backend algorithms, and ``united`` contains every algorithm.
    """

    family = str(family).strip().lower()
    if family not in FIGURE_OUTPUT_FAMILIES:
        raise ValueError(f"Unknown algorithm output family: {family}")
    if family in ALGORITHM_OUTPUT_FAMILIES:
        return algorithm_output_family(algorithm) == family
    return True


def grouped_for_algorithm_family(grouped, family):
    family = str(family).strip().lower()
    if family not in ALGORITHM_OUTPUT_FAMILIES:
        raise ValueError(f"Unknown algorithm output family: {family}")
    filtered = defaultdict(lambda: defaultdict(list))
    for benchmark, by_algorithm in grouped.items():
        for algorithm, items in by_algorithm.items():
            if comparison_output_family_includes(algorithm, family):
                filtered[benchmark][algorithm].extend(list(items))
    return filtered


def algorithm_plot_order():
    ordered = []
    for algorithm in list(DEFAULT_ALGORITHMS) + list(SYNTHETIC_ALGORITHM_ORDER):
        try:
            canonical = canonical_algorithm_name(str(algorithm))
        except Exception:
            canonical = str(algorithm)
        if canonical not in ordered:
            ordered.append(canonical)
    return ordered


def algorithm_plot_index(algorithm):
    try:
        canonical = canonical_algorithm_name(str(algorithm))
    except Exception:
        canonical = str(algorithm)
    plot_order = algorithm_plot_order()
    if canonical in plot_order:
        return int(plot_order.index(canonical))
    fallback = sum((idx + 1) * ord(char) for idx, char in enumerate(canonical))
    return int(fallback % 1000) + len(plot_order)


def algorithm_plot_style(algorithm, x_axis=None):
    idx = algorithm_plot_index(algorithm)
    try:
        canonical = canonical_algorithm_name(str(algorithm))
    except Exception:
        canonical = str(algorithm)
    style = {
        "color": PLOT_COLORS[idx % len(PLOT_COLORS)],
        "linestyle": PLOT_LINESTYLES[(idx // len(PLOT_COLORS)) % len(PLOT_LINESTYLES)],
        # CI whiskers already identify sampled uncertainty locations. Repeated
        # hollow markers made dense multi-algorithm plots visually collide with
        # their caps, especially over the first NASBench evaluations.
        "marker": None,
    }
    style.update(ALGORITHM_STYLE_OVERRIDES.get(canonical, {}))
    # Overrides retain each method's color and line style, but comparison
    # curves intentionally have no point markers; they collided with CI caps.
    style["marker"] = None
    return style


def plot_algorithm_series(x_axis, y_axis, algorithm, label):
    line, = plt.plot(
        x_axis,
        y_axis,
        label=label,
        **algorithm_plot_style(algorithm, x_axis=x_axis),
    )
    return line


def plot_confidence_band(
    x_axis,
    lower,
    upper,
    *,
    color=None,
    series_index=0,
    series_count=1,
    **_kwargs,
):
    """Draw staggered 95% CI whiskers on a shared plot grid.

    The legacy function name is retained for callers, but the rendering is an
    unfilled error-bar layer. Different algorithms occupy alternating grid
    slots, so their confidence intervals never overlap at the same x value.
    """

    if not PLOT_CONFIDENCE_ERROR_BARS:
        return None
    x_values = np.asarray(x_axis, dtype=float).reshape(-1)
    lower_values = np.asarray(lower, dtype=float).reshape(-1)
    upper_values = np.asarray(upper, dtype=float).reshape(-1)
    usable = min(x_values.size, lower_values.size, upper_values.size)
    if usable <= 0:
        return None
    x_values = x_values[:usable]
    lower_values = lower_values[:usable]
    upper_values = upper_values[:usable]
    valid = np.isfinite(x_values) & np.isfinite(lower_values) & np.isfinite(upper_values)
    valid_indices = np.flatnonzero(valid)
    if valid_indices.size == 0:
        return None
    series_count = max(1, int(series_count))
    series_index = int(series_index) % series_count
    valid_x = x_values[valid_indices]
    x_min = float(valid_x[0])
    x_max = float(valid_x[-1])
    if x_max <= x_min:
        # A single shared x position cannot display multiple algorithms'
        # intervals without overlap. Wait until the trace has enough extent.
        return None
    else:
        span = x_max - x_min
        max_slots = max(series_count, int(PLOT_CONFIDENCE_GRID_DIVISIONS))
        natural_slots = max(
            1,
            int(np.floor(span / max(float(PLOT_CONFIDENCE_MIN_GRID_STEP), 1e-12))),
        )
        required_slots = (
            series_count * max(1, int(PLOT_CONFIDENCE_MIN_ROUNDS_PER_ALGORITHM))
        )
        slot_count = min(max_slots, max(natural_slots, required_slots))
        grid_step = span / float(slot_count)
        first_target = x_min + grid_step * (series_index + 1)
        targets = np.arange(
            first_target,
            x_max + 0.5 * grid_step,
            grid_step * series_count,
        )
        targets = targets[targets <= x_max + 1e-12]
        if targets.size == 0:
            return None
        positions = np.searchsorted(valid_x, targets, side="left")
        positions = np.clip(positions, 0, valid_indices.size - 1)
        indices = np.unique(valid_indices[positions])
    centers = 0.5 * (lower_values[indices] + upper_values[indices])
    errors = 0.5 * np.maximum(0.0, upper_values[indices] - lower_values[indices])
    return plt.errorbar(
        x_values[indices],
        centers,
        yerr=errors,
        fmt="none",
        ecolor=color,
        elinewidth=0.8,
        capsize=2.0,
        capthick=0.8,
        alpha=float(PLOT_CONFIDENCE_ERROR_BAR_ALPHA),
        label="_nolegend_",
        zorder=1,
    )


def select_top_algorithms_by_final_mean_noiseless(
    by_algorithm,
    history_getter,
    limit=5,
):
    """Rank algorithms by the last finite point of their mean noiseless curve."""

    scores = {}
    for algorithm, items in by_algorithm.items():
        histories = [history_getter(item) for item in items]
        _x_axis, mean, _ci, n = mean_ci_from_histories(histories)
        if n == 0:
            continue
        finite = np.asarray(mean, dtype=float)
        finite = finite[np.isfinite(finite)]
        if finite.size:
            scores[algorithm] = float(finite[-1])
    ranked = sorted(
        scores,
        key=lambda algorithm: (
            -scores[algorithm],
            algorithm_plot_index(algorithm),
            str(algorithm),
        ),
    )
    selected = ranked if limit is None else ranked[: max(0, int(limit))]
    return selected, scores


def trace_noiseless_history_for_plot(item):
    hist = item.get("best_noiseless_y_history") if isinstance(item, dict) else None
    if isinstance(hist, list):
        return hist
    true_hist = item.get("true_y_history", []) if isinstance(item, dict) else []
    return best_so_far(true_hist) if true_hist else []


def legend_sorted_by_last_y(descending=True, *, outside_top=False, ncol=None):
    ax = plt.gca()
    handles, labels = ax.get_legend_handles_labels()
    items = []
    for order, (handle, label) in enumerate(zip(handles, labels)):
        if not label or str(label).startswith("_"):
            continue
        last_y = float("-inf")
        try:
            y_data = np.asarray(handle.get_ydata(), dtype=float).reshape(-1)
            finite = y_data[np.isfinite(y_data)]
            if finite.size:
                last_y = float(finite[-1])
        except Exception:
            pass
        items.append((last_y, order, handle, label))
    items.sort(key=lambda item: (item[0], -item[1]), reverse=bool(descending))
    if items:
        handles = [item[2] for item in items]
        labels = [item[3] for item in items]
        if outside_top:
            return ax.figure.legend(
                handles,
                labels,
                loc="upper center",
                bbox_to_anchor=(0.5, 0.995),
                ncol=int(ncol or 4),
                frameon=False,
            )
        return ax.legend(handles, labels)
    return None


def crop_plot_series(x_axis, *series, max_evaluation=PLOT_MAX_EVALUATION):
    if x_axis is None:
        return (x_axis, *series)
    x_axis = np.asarray(x_axis)
    mask = x_axis <= int(max_evaluation)
    cropped = [np.asarray(values)[mask] if values is not None else None for values in series]
    return (x_axis[mask], *cropped)


def plot_max_evaluation_for_items(items):
    values = []
    for item in items:
        if not isinstance(item, dict):
            continue
        settings = item.get("settings", {})
        if not isinstance(settings, dict):
            continue
        value = settings.get("max_iter")
        try:
            value = int(value)
        except Exception:
            continue
        if value > 0:
            values.append(value)
    return max(values) if values else int(MAX_EXPERIMENT)


def plot_max_evaluation_for_benchmark(benchmark, items):
    maximum = plot_max_evaluation_for_items(items)
    if str(benchmark).lower().startswith("nasbench101"):
        return min(maximum, int(NASBENCH101_PLOT_MAX_EVALUATION))
    if str(benchmark).lower().startswith("rover60"):
        return min(maximum, int(ROVER60_PLOT_MAX_EVALUATION))
    return maximum


def apply_nasbench101_value_y_limits(benchmark, ax=None):
    if not str(benchmark).lower().startswith("nasbench101"):
        return False
    target = plt.gca() if ax is None else ax
    target.set_ylim(*NASBENCH101_VALUE_Y_LIMITS)
    return True


def plot_marker_start_after_n_init(item):
    settings = item.get("settings", {}) if isinstance(item, dict) else {}
    algorithm = item.get("algorithm") if isinstance(item, dict) else None
    candidates = [
        settings.get("n_init"),
        settings.get("n_init_per_embedding"),
        settings.get("initial_design_size"),
    ]
    algorithm_settings = settings.get("algorithm_settings", {})
    if isinstance(algorithm_settings, dict):
        if algorithm in {"MCTS-VS-BO", "MCTS-VS-TuRBO"}:
            feature_batch_size = algorithm_settings.get("feature_batch_size")
            sample_batch_size = algorithm_settings.get("sample_batch_size")
            try:
                return 2 * int(feature_batch_size) * int(sample_batch_size)
            except Exception:
                pass
        candidates.extend(
            [
                algorithm_settings.get("n_init"),
                algorithm_settings.get("n_init_per_embedding"),
                algorithm_settings.get("initial_design_size"),
            ]
        )

    for value in candidates:
        if isinstance(value, (int, np.integer)):
            return int(value)
        if isinstance(value, float) and np.isfinite(value):
            return int(value)
        if isinstance(value, str) and value.strip().isdigit():
            return int(value.strip())
    return 0


def hide_threshold_before_initial_gp(history, item):
    values = list(history or [])
    marker_start = int(plot_marker_start_after_n_init(item))
    hidden = max(0, min(marker_start, len(values)))
    for idx in range(hidden):
        values[idx] = None
    return values


def initial_gp_start_count_for_algorithm(algorithm, config=None):
    algorithm = canonical_algorithm_name(algorithm)
    if algorithm in AGVS_ALGORITHMS:
        return benchmark_n_init(config) if config is not None else int(AGVS_N_INIT)
    if algorithm == "VanillaBO":
        return int(MCTS_PAPER_BO_N_INIT)
    if algorithm in {"MCTS-VS-BO", "MCTS-VS-TuRBO", "MCTS-VS-RS", "MCTS-VS-SAASBO"}:
        return int(2 * MCTS_FEATURE_BATCH_SIZE * MCTS_SAMPLE_BATCH_SIZE)
    if algorithm == "TuRBO":
        return int(TURBO_N_INIT)
    if algorithm == "CMA-ES":
        return 1
    if algorithm == "ADD-GP-UCB":
        return 10
    if algorithm == "GTBO":
        return 10
    if algorithm == "LassoBO":
        return 30
    if algorithm in MCTS_PAPER_ADAPTER_SETTINGS:
        return int(MCTS_PAPER_ADAPTER_SETTINGS[algorithm]["n_init"])
    return 0


def mask_threshold_history_before_n(threshold_history_by_delta, n_initial):
    if not isinstance(threshold_history_by_delta, dict):
        return threshold_history_by_delta
    n_initial = max(0, int(n_initial))
    out = {}
    for key, values in threshold_history_by_delta.items():
        values = list(values or [])
        for idx in range(min(n_initial, len(values))):
            values[idx] = None
        out[key] = values
    return out


STAT_TABLE_REFERENCE_ALGORITHM = "AG-VS-BO"
REAL_LIFE_STAT_TABLE_REFERENCE_ALGORITHM = "AG-VS-TuRBO"
STAT_TABLE_REFERENCE_BY_FAMILY = {
    "bo": "AG-VS-BO",
    "turbo": "AG-VS-TuRBO",
    "united": "AG-VS-BO",
}


def stat_table_reference_by_family(algorithm_family, contains_real_life=False):
    """Choose the comparison reference without changing synthetic tables."""

    family = str(algorithm_family).strip().lower()
    if family not in STAT_TABLE_REFERENCE_BY_FAMILY:
        raise ValueError(f"Unknown algorithm output family: {algorithm_family}")
    if bool(contains_real_life) and family == "united":
        return REAL_LIFE_STAT_TABLE_REFERENCE_ALGORITHM
    return STAT_TABLE_REFERENCE_BY_FAMILY[family]
LEGACY_STAT_TABLE_PREFIX_BY_FAMILY = {
    "bo": "ag-vs-bo_vs_all_except_ag-vs-turbo",
    "turbo": "ag-vs-turbo_vs_all_except_ag-vs-bo",
}
STAT_TABLE_ALGORITHM_ORDER = (
    "AG-VS-TuRBO",
    "AG-VS-BO",
    "MCTS-VS-TuRBO",
    "MCTS-VS-BO",
    "TuRBO",
    "VanillaBO",
)
STAT_TABLE_ALPHA = 0.05
STAT_TABLE_EXCLUDED_BENCHMARK_PREFIXES = ("hartmann",)
STAT_TABLE_ROW_CACHE_PATH = SYNTHETIC_TABLE_ROOT / "wilcoxon_table_rows.json"


def benchmark_table_sort_key(name):
    """Sort as family, effective dimension, then ambient dimension."""
    text = str(name).lower()
    match = re.fullmatch(r"(.+?)(\d+)_(\d+)", text)
    if match is None:
        return text, math.inf, math.inf
    family, effective_dim, original_dim = match.groups()
    return family, int(effective_dim), int(original_dim)


def latex_escape(text):
    return (
        str(text)
        .replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("≈", "$\\approx$")
    )


def trace_final_stoppable_threshold_value(item):
    if not isinstance(item, dict):
        return None
    primary_key = primary_risk_label()
    keys = [
        f"final_best_stoppable_threshold_at_max_iter_{primary_key}",
        "final_best_stoppable_threshold_at_max_iter_1_256",
        "final_best_stoppable_threshold_at_max_iter",
    ]
    for key in keys:
        value = item.get(key)
        try:
            value = float(value)
        except Exception:
            continue
        if np.isfinite(value):
            return float(value)

    threshold_by_delta = item.get("best_threshold_history_by_delta", {})
    if isinstance(threshold_by_delta, dict):
        hist = threshold_by_delta.get(primary_key)
        if hist is None and threshold_by_delta:
            hist = next(iter(threshold_by_delta.values()))
        if hist is not None:
            values = np.asarray([value for value in hist if value is not None], dtype=float)
            values = values[np.isfinite(values)]
            if values.size:
                return float(values[-1])
    return None


def trace_final_best_noiseless_value(item):
    if not isinstance(item, dict):
        return None
    keys = [
        "final_best_noiseless_y_at_max_iter",
        "best_noiseless_y",
    ]
    for key in keys:
        value = item.get(key)
        try:
            value = float(value)
        except Exception:
            continue
        if np.isfinite(value):
            return float(value)
    hist = item.get("best_noiseless_y_history")
    if hist is not None:
        values = np.asarray([value for value in hist if value is not None], dtype=float)
        values = values[np.isfinite(values)]
        if values.size:
            return float(values[-1])
    return None


def trace_final_best_observed_value(item):
    if not isinstance(item, dict):
        return None
    for key in ("final_best_observed_at_max_iter", "best_observed"):
        try:
            value = float(item.get(key))
        except (TypeError, ValueError):
            continue
        if np.isfinite(value):
            return value
    history = item.get("best_observed_history", [])
    for value in reversed(history if isinstance(history, list) else []):
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(value):
            return value
    return None


def trace_final_best_posterior_mean(item):
    if not isinstance(item, dict):
        return None
    for key in ("final_best_posterior_mean_at_max_iter", "best_posterior_mean"):
        try:
            value = float(item.get(key))
        except (TypeError, ValueError):
            continue
        if np.isfinite(value):
            return value
    history = item.get("reported_posterior_mean_history", item.get("best_posterior_mean_history", []))
    for value in reversed(history if isinstance(history, list) else []):
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(value):
            return value
    return None


def final_values_by_trial(items, value_type="stoppable_threshold"):
    out = {}
    for item in items or []:
        if value_type == "stoppable_threshold":
            value = trace_final_stoppable_threshold_value(item)
        elif value_type == "posterior_mean":
            value = trace_final_best_posterior_mean(item)
        elif value_type == "best_noiseless":
            value = trace_final_best_noiseless_value(item)
        elif value_type == "best_observed":
            value = trace_final_best_observed_value(item)
        else:
            value = None
        if value is None:
            continue
        try:
            trial = int(item.get("trial"))
        except Exception:
            trial = len(out) + 1
        out[trial] = float(value)
    return out


def wilcoxon_symbol(reference_by_trial, other_by_trial, alpha=STAT_TABLE_ALPHA):
    if wilcoxon is None:
        return "≈"
    common = sorted(set(reference_by_trial).intersection(other_by_trial))
    if len(common) < 2:
        return "≈"
    ref = np.asarray([reference_by_trial[idx] for idx in common], dtype=float)
    other = np.asarray([other_by_trial[idx] for idx in common], dtype=float)
    diff = ref - other
    if not np.any(np.abs(diff) > 1e-12):
        return "≈"
    try:
        greater = wilcoxon(ref, other, alternative="greater", zero_method="wilcox").pvalue
        less = wilcoxon(ref, other, alternative="less", zero_method="wilcox").pvalue
    except Exception:
        return "≈"
    if np.isfinite(greater) and float(greater) < float(alpha):
        return "+"
    if np.isfinite(less) and float(less) < float(alpha):
        return "-"
    return "≈"


def stat_table_reference_algorithm(stats, preferred=None):
    reference = preferred or STAT_TABLE_REFERENCE_ALGORITHM
    if reference in stats:
        return reference
    return None


def stat_table_algorithm_sort_key(algorithm):
    try:
        return (0, STAT_TABLE_ALGORITHM_ORDER.index(algorithm))
    except ValueError:
        return (1, algorithm_plot_index(algorithm))


def objective_table_rows(
    grouped,
    value_type="stoppable_threshold",
    reference_algorithm=None,
):
    rows = []
    algorithms = []
    table_grouped = {
        benchmark: by_algorithm
        for benchmark, by_algorithm in grouped.items()
        if not str(benchmark).lower().startswith(STAT_TABLE_EXCLUDED_BENCHMARK_PREFIXES)
    }
    for by_algorithm in table_grouped.values():
        for algorithm in by_algorithm:
            if algorithm not in algorithms:
                algorithms.append(algorithm)
    algorithms = sorted(
        algorithms,
        key=lambda algorithm: (
            0 if algorithm == reference_algorithm else 1,
            stat_table_algorithm_sort_key(algorithm),
        ),
    )

    plus_minus_counts = {algorithm: {"+": 0, "-": 0, "≈": 0} for algorithm in algorithms}
    for benchmark, by_algorithm in sorted(
        table_grouped.items(), key=lambda item: benchmark_table_sort_key(item[0])
    ):
        values_by_algorithm = {
            algorithm: final_values_by_trial(items, value_type=value_type)
            for algorithm, items in by_algorithm.items()
        }
        stats = {}
        for algorithm, values_by_trial in values_by_algorithm.items():
            values = np.asarray(list(values_by_trial.values()), dtype=float)
            values = values[np.isfinite(values)]
            if values.size == 0:
                continue
            stats[algorithm] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
                "n": int(values.size),
            }
        if not stats:
            continue

        reference = stat_table_reference_algorithm(stats, preferred=reference_algorithm)
        best_mean = max(item["mean"] for item in stats.values())
        symbols = {}
        for algorithm in algorithms:
            if algorithm not in stats:
                symbols[algorithm] = ""
            elif reference is None or algorithm == reference:
                symbols[algorithm] = ""
            else:
                symbol = wilcoxon_symbol(values_by_algorithm[reference], values_by_algorithm[algorithm])
                symbols[algorithm] = symbol
                if algorithm in plus_minus_counts and symbol in plus_minus_counts[algorithm]:
                    plus_minus_counts[algorithm][symbol] += 1
        rows.append(
            {
                "benchmark": benchmark,
                "algorithms": algorithms,
                "reference": reference,
                "best_mean": float(best_mean),
                "stats": stats,
                "symbols": symbols,
            }
        )
    return algorithms, rows, plus_minus_counts


def merge_objective_table_row_cache(
    value_type,
    algorithms,
    rows,
    cache_path=None,
    reference_algorithm=None,
    preserve_cached_rows=True,
):
    """Update table rows, optionally replacing the entire cached benchmark set."""
    cache_path = Path(cache_path) if cache_path is not None else STAT_TABLE_ROW_CACHE_PATH
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache = {}
    if cache_path.exists():
        try:
            cache = json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception:
            cache = {}

    cached_rows = {
        str(row["benchmark"]): row
        for row in (cache.get(value_type, []) if preserve_cached_rows else [])
        if isinstance(row, dict) and row.get("benchmark") is not None
    }
    fresh_rows_by_benchmark = {
        str(row["benchmark"]): row
        for row in rows
        if isinstance(row, dict) and row.get("benchmark") is not None
    }
    if reference_algorithm is not None:
        for benchmark, row in cached_rows.items():
            if benchmark in fresh_rows_by_benchmark:
                continue
            if row.get("reference") != reference_algorithm:
                row["reference"] = (
                    reference_algorithm
                    if reference_algorithm in row.get("stats", {})
                    else None
                )
                row["symbols"] = {
                    algorithm: "" for algorithm in row.get("stats", {})
                }
    cached_rows.update(
        fresh_rows_by_benchmark
    )

    cached_rows = {
        benchmark: row
        for benchmark, row in cached_rows.items()
        if not benchmark.lower().startswith(STAT_TABLE_EXCLUDED_BENCHMARK_PREFIXES)
    }
    try:
        cache_path.resolve().relative_to(SYNTHETIC_TABLE_ROOT.resolve())
        active_synthetic_rows = {name.lower() for name in DEFAULT_BENCHMARKS}
        cached_rows = {
            benchmark: row
            for benchmark, row in cached_rows.items()
            if benchmark.lower() in active_synthetic_rows
        }
    except ValueError:
        pass
    merged_rows = [
        cached_rows[key]
        for key in sorted(cached_rows, key=benchmark_table_sort_key)
    ]

    merged_algorithms = set(algorithms)
    for row in merged_rows:
        merged_algorithms.update(row.get("stats", {}).keys())
    merged_algorithms = sorted(
        merged_algorithms,
        key=lambda algorithm: (
            0 if algorithm == reference_algorithm else 1,
            stat_table_algorithm_sort_key(algorithm),
        ),
    )

    plus_minus_counts = {
        algorithm: {"+": 0, "-": 0, "≈": 0}
        for algorithm in merged_algorithms
    }
    for row in merged_rows:
        for algorithm, symbol in row.get("symbols", {}).items():
            if algorithm in plus_minus_counts and symbol in plus_minus_counts[algorithm]:
                plus_minus_counts[algorithm][symbol] += 1
        row["algorithms"] = merged_algorithms

    cache[value_type] = merged_rows
    cache_path.write_text(
        json.dumps(cache, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return merged_algorithms, merged_rows, plus_minus_counts


def format_stat_cell(mean, std, symbol="", bold=False, latex=False):
    value = f"{float(mean):.3f}({float(std):.3f})"
    if latex and bold:
        value = f"\\textbf{{{value}}}"
    elif (not latex) and bold:
        value = f"$\\mathbf{{{value}}}$"
    if symbol:
        if latex and symbol == "≈":
            symbol = "$\\approx$"
        value = f"{value} {symbol}"
    return value


def objective_table_latex(algorithms, rows, plus_minus_counts, caption):
    cols = "l" + "c" * len(algorithms)
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        f"\\begin{{tabular}}{{{cols}}}",
        "\\toprule",
        "Problem & " + " & ".join(latex_escape(algorithm) for algorithm in algorithms) + " \\\\",
        "\\midrule",
    ]
    previous_family = None
    for row in rows:
        family = benchmark_table_sort_key(row["benchmark"])[0]
        if previous_family is not None and family != previous_family:
            lines.append("\\midrule")
        cells = [latex_escape(row["benchmark"])]
        for algorithm in algorithms:
            stat = row["stats"].get(algorithm)
            if stat is None:
                cells.append("--")
                continue
            bold = np.isclose(float(stat["mean"]), float(row["best_mean"]))
            cells.append(
                format_stat_cell(
                    stat["mean"],
                    stat["std"],
                    symbol=row["symbols"].get(algorithm, ""),
                    bold=bold,
                    latex=True,
                )
            )
        lines.append(" & ".join(cells) + " \\\\")
        previous_family = family
    lines.extend(["\\midrule"])
    summary = ["$+/-/\\approx$"]
    for algorithm in algorithms:
        counts = plus_minus_counts.get(algorithm, {})
        if not counts or sum(counts.values()) == 0:
            summary.append("/")
        else:
            summary.append(f"{counts.get('+', 0)}/{counts.get('-', 0)}/{counts.get('≈', 0)}")
    lines.append(" & ".join(summary) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    return "\n".join(lines)


def save_objective_table_png(
    algorithms,
    rows,
    plus_minus_counts,
    path,
    title,
    metric_description,
    reference_algorithm=STAT_TABLE_REFERENCE_ALGORITHM,
):
    if not rows:
        return None
    header = ["Problem"] + algorithms
    body = []
    for row in rows:
        cells = [row["benchmark"]]
        for algorithm in algorithms:
            stat = row["stats"].get(algorithm)
            if stat is None:
                cells.append("--")
                continue
            bold = np.isclose(float(stat["mean"]), float(row["best_mean"]))
            cells.append(
                format_stat_cell(
                    stat["mean"],
                    stat["std"],
                    symbol=row["symbols"].get(algorithm, ""),
                    bold=bold,
                    latex=False,
                )
            )
        body.append(cells)

    summary = ["+/-/≈"]
    for algorithm in algorithms:
        counts = plus_minus_counts.get(algorithm, {})
        summary.append("/" if not counts or sum(counts.values()) == 0 else f"{counts.get('+', 0)}/{counts.get('-', 0)}/{counts.get('≈', 0)}")
    body.append(summary)

    width = max(11.0, 2.0 + 2.2 * len(algorithms))
    height = max(3.5, 1.3 + 0.45 * len(body))
    fig, ax = plt.subplots(figsize=(width, height))
    ax.axis("off")
    table = ax.table(cellText=body, colLabels=header, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.3)
    family_boundary_rows = set()
    previous_family = None
    for data_index, row in enumerate(rows, start=1):
        family = benchmark_table_sort_key(row["benchmark"])[0]
        if previous_family is not None and family != previous_family:
            family_boundary_rows.add(data_index)
        previous_family = family
    for (row_idx, col_idx), cell in table.get_celld().items():
        cell.visible_edges = ""
        cell.set_linewidth(0.0)
        if row_idx == 0:
            cell.visible_edges += "TB"
            cell.set_linewidth(1.0)
        if row_idx == len(body):
            cell.visible_edges += "TB"
            cell.set_linewidth(1.0)
        if row_idx in family_boundary_rows:
            cell.visible_edges += "T"
            cell.set_linewidth(0.8)
        if col_idx == 0:
            cell.visible_edges += "R"
            cell.set_linewidth(max(cell.get_linewidth(), 0.8))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.savefig(path.with_suffix(".pdf"), dpi=220, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def save_objective_tables(
    grouped=None,
    figure_dir=None,
    write_per_benchmark=False,
    table_dir=None,
    algorithm_family=None,
    preserve_cached_rows=True,
):
    # Keep every aggregate table artifact outside the algorithm package and in
    # one stable experiment output directory.
    target_dir = Path(table_dir) if table_dir is not None else SYNTHETIC_TABLE_ROOT
    target_dir.mkdir(parents=True, exist_ok=True)
    legacy_prefix = LEGACY_STAT_TABLE_PREFIX_BY_FAMILY.get(algorithm_family)
    if legacy_prefix:
        for stale_path in target_dir.glob(f"{legacy_prefix}__*"):
            if stale_path.is_file() and stale_path.suffix.lower() in {".png", ".tex"}:
                stale_path.unlink(missing_ok=True)
    
    # Callers that already loaded/grouped traces should not pay for another full
    # recursive scan of the benchmark tree.  The old implementation ignored the
    # supplied value and re-read every trial JSON on every plot refresh.
    if grouped is None:
        traces = [t for t in load_traces(HERE) if t.get("complete", False)]
        all_grouped = defaultdict(lambda: defaultdict(list))
        for trace in traces:
            benchmark = trace.get("benchmark")
            algorithm = trace.get("algorithm")
            if benchmark and algorithm:
                all_grouped[benchmark][algorithm].append(trace)
    else:
        all_grouped = grouped

    if algorithm_family is None:
        family_paths = []
        aggregate_artifact_names = (
            "wilcoxon_table_rows.json",
            "best_observed_objective_wilcoxon.tex",
            "best_observed_objective_wilcoxon.png",
            "stoppable_threshold_wilcoxon.tex",
            "stoppable_threshold_wilcoxon.png",
            "best_posterior_mean_wilcoxon.tex",
            "best_posterior_mean_wilcoxon.png",
            "best_noiseless_wilcoxon.tex",
            "best_noiseless_wilcoxon.png",
        )
        for family in ALGORITHM_OUTPUT_FAMILIES:
            family_grouped = grouped_for_algorithm_family(all_grouped, family)
            family_reference = STAT_TABLE_REFERENCE_BY_FAMILY[family]
            if not any(family_grouped.values()):
                # A full regeneration from the current JSON source of truth
                # must not leave aggregate tables from a previously populated
                # algorithm family in place.
                for stale_name in aggregate_artifact_names:
                    (target_dir / family / stale_name).unlink(missing_ok=True)
                continue
            family_paths.extend(
                save_objective_tables(
                    family_grouped,
                    figure_dir=figure_dir,
                    write_per_benchmark=write_per_benchmark,
                    table_dir=target_dir / family,
                    algorithm_family=family,
                    preserve_cached_rows=preserve_cached_rows,
                )
            )

        # Publish the backend-only tables and additionally the combined table.
        if any(all_grouped.values()):
            family_paths.extend(
                save_objective_tables(
                    all_grouped,
                    figure_dir=figure_dir,
                    write_per_benchmark=write_per_benchmark,
                    table_dir=target_dir / "united",
                    algorithm_family="united",
                    preserve_cached_rows=preserve_cached_rows,
                )
            )

        # Mixed-family aggregate artifacts were produced by older revisions.
        # They must not coexist with the family-specific source of truth.
        for legacy_name in aggregate_artifact_names:
            (target_dir / legacy_name).unlink(missing_ok=True)
        return family_paths

    out_paths = []
    contains_real_life = any(
        str(item.get("settings", {}).get("experiment_kind", "synthetic"))
        == "real_life"
        for by_algorithm in all_grouped.values()
        for items in by_algorithm.values()
        for item in items
        if isinstance(item, dict)
    )
    family_reference = stat_table_reference_by_family(
        algorithm_family,
        contains_real_life=contains_real_life,
    )

    # The paper's real-life figures compare measured accuracy/reward.
    if contains_real_life:
        algorithms_ob, rows_ob, counts_ob = objective_table_rows(
            all_grouped,
            value_type="best_observed",
            reference_algorithm=family_reference,
        )
        algorithms_ob, rows_ob, counts_ob = merge_objective_table_row_cache(
            "best_observed",
            algorithms_ob,
            rows_ob,
            target_dir / "wilcoxon_table_rows.json",
            reference_algorithm=family_reference,
            preserve_cached_rows=preserve_cached_rows,
        )
        if rows_ob:
            caption_ob = (
                "Best observed real-life objective values obtained by the compared "
                "optimizers. Each result is the mean and standard deviation across "
                "trials of final running-best benchmark accuracy or reward."
            )
            tex_path_ob = target_dir / "best_observed_objective_wilcoxon.tex"
            tex_path_ob.write_text(
                objective_table_latex(
                    algorithms_ob, rows_ob, counts_ob, caption_ob
                ),
                encoding="utf-8",
            )
            out_paths.append(str(tex_path_ob))
            png_path_ob = save_objective_table_png(
                algorithms_ob,
                rows_ob,
                counts_ob,
                target_dir / "best_observed_objective_wilcoxon.png",
                "Best Observed Real-Life Objective With Wilcoxon Signed-Rank Test",
                "the final running-best observed accuracy or reward",
                reference_algorithm=family_reference,
            )
            if png_path_ob:
                out_paths.append(png_path_ob)

    # Posterior stopping and its threshold artifacts are disabled runner-wide.
    if POSTERIOR_STOPPING_ENABLED:
        algorithms_st, rows_st, plus_minus_counts_st = objective_table_rows(
            all_grouped,
            value_type="stoppable_threshold",
            reference_algorithm=family_reference,
        )
        algorithms_st, rows_st, plus_minus_counts_st = merge_objective_table_row_cache(
            "stoppable_threshold",
            algorithms_st,
            rows_st,
            target_dir / "wilcoxon_table_rows.json",
            reference_algorithm=family_reference,
            preserve_cached_rows=preserve_cached_rows,
        )
    else:
        algorithms_st, rows_st, plus_minus_counts_st = [], [], {}
    if rows_st:
        caption_st = (
            "Best stoppable threshold values obtained by the compared optimizers. Each result consists of the mean and "
            "standard deviation of the final best stoppable threshold across runs. The best mean value on each problem is bolded. "
            "The symbols '+', '-' and '≈' indicate that the reference method is significantly superior to, "
            "inferior to, and statistically indistinguishable from the corresponding method, respectively, "
            "according to the Wilcoxon signed-rank test with confidence level 0.05."
        )
        tex_path_st = target_dir / "stoppable_threshold_wilcoxon.tex"
        tex_path_st.write_text(
            objective_table_latex(algorithms_st, rows_st, plus_minus_counts_st, caption_st),
            encoding="utf-8",
        )
        out_paths.append(str(tex_path_st))
        png_path_st = save_objective_table_png(
            algorithms_st,
            rows_st,
            plus_minus_counts_st,
            target_dir / "stoppable_threshold_wilcoxon.png",
            "Best Stoppable Threshold With Wilcoxon Signed-Rank Test",
            "the final best GP posterior credible threshold",
            reference_algorithm=family_reference,
        )
        if png_path_st:
            out_paths.append(png_path_st)
    else:
        for stale_path in (
            target_dir / "stoppable_threshold_wilcoxon.tex",
            target_dir / "stoppable_threshold_wilcoxon.png",
        ):
            stale_path.unlink(missing_ok=True)

    # Posterior-mean diagnostic plots/tables are intentionally omitted.
    if PLOT_POSTERIOR_MEAN_DIAGNOSTICS:
        algorithms_bn, rows_bn, plus_minus_counts_bn = objective_table_rows(
            all_grouped,
            value_type="posterior_mean",
            reference_algorithm=family_reference,
        )
        algorithms_bn, rows_bn, plus_minus_counts_bn = merge_objective_table_row_cache(
            "posterior_mean",
            algorithms_bn,
            rows_bn,
            target_dir / "wilcoxon_table_rows.json",
            reference_algorithm=family_reference,
            preserve_cached_rows=preserve_cached_rows,
        )
    else:
        algorithms_bn, rows_bn, plus_minus_counts_bn = [], [], {}
    if rows_bn:
        caption_bn = (
            "Best Gaussian-process posterior mean values returned by the compared optimizers. Each result consists of the mean and "
            "standard deviation across runs of the returned posterior mean: the triggering current point when early stopping occurs, "
            "otherwise the final running-best posterior mean at a model-recommended evaluated point. "
            "Raw measured and noiseless objective values are not used for this metric. The best mean value on each problem is bolded. "
            "The symbols '+', '-' and '≈' indicate that the reference method is significantly superior to, "
            "inferior to, and statistically indistinguishable from the corresponding method, respectively, "
            "according to the Wilcoxon signed-rank test with confidence level 0.05."
        )
        tex_path_bn = target_dir / "best_posterior_mean_wilcoxon.tex"
        tex_path_bn.write_text(
            objective_table_latex(algorithms_bn, rows_bn, plus_minus_counts_bn, caption_bn),
            encoding="utf-8",
        )
        out_paths.append(str(tex_path_bn))
        png_path_bn = save_objective_table_png(
            algorithms_bn,
            rows_bn,
            plus_minus_counts_bn,
            target_dir / "best_posterior_mean_wilcoxon.png",
            "Best GP Posterior Mean With Wilcoxon Signed-Rank Test",
            "the returned GP posterior mean (current point at early stop; otherwise running best)",
            reference_algorithm=family_reference,
        )
        if png_path_bn:
            out_paths.append(png_path_bn)
    else:
        for stale_path in (
            target_dir / "best_posterior_mean_wilcoxon.tex",
            target_dir / "best_posterior_mean_wilcoxon.png",
        ):
            stale_path.unlink(missing_ok=True)

    # 3. Best noiseless objective table & plot (independent validation metric).
    algorithms_nl, rows_nl, plus_minus_counts_nl = objective_table_rows(
        all_grouped,
        value_type="best_noiseless",
        reference_algorithm=family_reference,
    )
    algorithms_nl, rows_nl, plus_minus_counts_nl = merge_objective_table_row_cache(
        "best_noiseless",
        algorithms_nl,
        rows_nl,
        target_dir / "wilcoxon_table_rows.json",
        reference_algorithm=family_reference,
        preserve_cached_rows=preserve_cached_rows,
    )
    if rows_nl:
        caption_nl = (
            "Best noiseless objective values obtained by the compared optimizers. Each result consists of the mean and "
            "standard deviation of the final best noiseless value across runs. This is an independent ground-truth "
            "validation metric and does not alter the posterior mean returned by an optimizer. The best mean value on "
            "each problem is bolded. The symbols '+', '-' and '≈' compare the reference method with each corresponding "
            "method using the Wilcoxon signed-rank test at confidence level 0.05."
        )
        tex_path_nl = target_dir / "best_noiseless_wilcoxon.tex"
        tex_path_nl.write_text(
            objective_table_latex(algorithms_nl, rows_nl, plus_minus_counts_nl, caption_nl),
            encoding="utf-8",
        )
        out_paths.append(str(tex_path_nl))
        png_path_nl = save_objective_table_png(
            algorithms_nl,
            rows_nl,
            plus_minus_counts_nl,
            target_dir / "best_noiseless_wilcoxon.png",
            "Best Noiseless Objective With Wilcoxon Signed-Rank Test",
            "the final best noiseless objective value",
            reference_algorithm=family_reference,
        )
        if png_path_nl:
            out_paths.append(png_path_nl)
    else:
        for stale_path in (
            target_dir / "best_noiseless_wilcoxon.tex",
            target_dir / "best_noiseless_wilcoxon.png",
        ):
            stale_path.unlink(missing_ok=True)

    explicit_prefixes = {
        "bo": "only_bo_algorithms",
        "turbo": "only_turbo_algorithms",
    }
    explicit_prefix = explicit_prefixes.get(algorithm_family)
    if explicit_prefix is not None:
        explicit_paths = []
        for source_name in list(out_paths):
            source = Path(source_name)
            if source.suffix.lower() not in {".png", ".tex"} or not source.exists():
                continue
            explicit_path = source.with_name(
                f"{explicit_prefix}__{source.name}"
            )
            shutil.copy2(source, explicit_path)
            explicit_paths.append(str(explicit_path))
        out_paths.extend(explicit_paths)

    return out_paths


_AGGREGATE_GROUPED_CACHE = None
_AGGREGATE_GROUPED_CACHE_ROOT = None


def update_aggregate_objective_tables(changed_grouped, benchmark_root=None, table_dir=None):
    """Update changed benchmark rows without re-reading every JSON each time."""
    global _AGGREGATE_GROUPED_CACHE, _AGGREGATE_GROUPED_CACHE_ROOT
    benchmark_root = Path(benchmark_root) if benchmark_root is not None else RESULT_ROOT
    cache_root = str(benchmark_root.resolve())

    if _AGGREGATE_GROUPED_CACHE is None or _AGGREGATE_GROUPED_CACHE_ROOT != cache_root:
        _AGGREGATE_GROUPED_CACHE = defaultdict(lambda: defaultdict(list))
        _AGGREGATE_GROUPED_CACHE_ROOT = cache_root
        for trace in load_traces(benchmark_root):
            if not trace.get("complete", False):
                continue
            benchmark = trace.get("benchmark")
            algorithm = trace.get("algorithm")
            if benchmark and algorithm:
                _AGGREGATE_GROUPED_CACHE[benchmark][algorithm].append(trace)

    # plot_results has already loaded the current benchmark's traces. Replace
    # only those rows in the aggregate cache; all other benchmark rows remain.
    for benchmark, by_algorithm in changed_grouped.items():
        replacement = defaultdict(list)
        for algorithm, items in by_algorithm.items():
            replacement[algorithm] = list(items)
        _AGGREGATE_GROUPED_CACHE[benchmark] = replacement

    return save_objective_tables(
        _AGGREGATE_GROUPED_CACHE,
        benchmark_root,
        write_per_benchmark=False,
        table_dir=table_dir,
    )


def save_all_benchmark_objective_tables(benchmark_root=None, table_dir=None):
    """Write aggregate tables from every JSON under the central Result tree."""
    benchmark_root = Path(benchmark_root) if benchmark_root is not None else RESULT_ROOT
    traces = [t for t in load_traces(benchmark_root) if t.get("complete", False)]
    grouped = defaultdict(lambda: defaultdict(list))
    for trace in traces:
        benchmark = trace.get("benchmark")
        algorithm = trace.get("algorithm")
        if benchmark and algorithm:
            grouped[benchmark][algorithm].append(trace)
    return save_objective_tables(
        grouped,
        benchmark_root,
        write_per_benchmark=False,
        table_dir=table_dir,
    )


def _synthetic_wall_clock_histories_seconds(algorithm, items):
    histories = []
    for item in items:
        raw_times = item.get("evaluation_wall_clock_history", [])
        n_evaluations = int(item.get("n_evaluations", len(raw_times)))
        total_time = float(item.get("wall_clock_seconds", 0.0))
        if raw_times:
            seconds = batch_parallel_wall_clock_history(
                raw_times,
                algorithm=str(item.get("algorithm", algorithm)),
                result=item,
                n_evaluations=n_evaluations,
                wall_clock=total_time,
                q_batch=_trace_q_batch(
                    item,
                    default=LEGACY_TRACE_BATCH_FALLBACK,
                ),
            )
        elif n_evaluations > 0:
            seconds = np.linspace(
                total_time / n_evaluations,
                total_time,
                n_evaluations,
            )
        else:
            continue
        histories.append(np.asarray(seconds, dtype=float).tolist())
    return histories


def save_synthetic_comparison_pdfs(
    benchmark_root=None,
    figure_root=None,
    benchmark_names=None,
    reuse_complete_json=True,
):
    """Create one four-panel noiseless-performance/wall-clock PDF per objective."""

    benchmark_root = Path(benchmark_root or SYNTHETIC_RESULT_ROOT)
    figure_root = Path(figure_root or SYNTHETIC_FIGURE_ROOT)
    selected = (
        set(SYNTHETIC_GRAPH_Y_LIMITS)
        if benchmark_names is None
        else {str(name).strip().lower() for name in benchmark_names}
    )
    output_paths = []

    def render(objective, trace_subset, target_root):
        grouped = defaultdict(lambda: defaultdict(list))
        for trace in trace_subset:
            grouped[str(trace["benchmark"]).lower()][trace["algorithm"]].append(trace)

        for family in FIGURE_OUTPUT_FAMILIES:
            family_root = Path(target_root) / family
            family_root.mkdir(parents=True, exist_ok=True)
            benchmark_pair = (
                f"{objective}3_300",
                f"{objective}6_1000",
            )
            if not all(name in grouped for name in benchmark_pair):
                continue

            algorithms = sorted(
                {
                    algorithm
                    for name in benchmark_pair
                    for algorithm in grouped[name]
                    if comparison_output_family_includes(algorithm, family)
                },
                key=lambda name: (algorithm_plot_index(name), str(name)),
            )
            if not algorithms:
                continue

            fig, axes = plt.subplots(1, 4, figsize=(23.0, 6.2))
            legend_handles = {}
            for pair_index, benchmark in enumerate(benchmark_pair):
                performance_ax = axes[2 * pair_index]
                wall_clock_ax = axes[2 * pair_index + 1]
                benchmark_max = plot_max_evaluation_for_items(
                    [
                        item
                        for algorithm in algorithms
                        for item in grouped[benchmark].get(algorithm, [])
                    ]
                )

                for algorithm_index, algorithm in enumerate(algorithms):
                    items = grouped[benchmark].get(algorithm, [])
                    histories = [
                        trace_noiseless_history_for_plot(item)
                        for item in items
                    ]
                    x_axis, mean, ci, n = mean_ci_from_histories(histories)
                    if n == 0:
                        continue
                    x_axis, mean, ci = crop_plot_series(
                        x_axis, mean, ci, max_evaluation=benchmark_max
                    )
                    line, = performance_ax.plot(
                        x_axis,
                        mean,
                        label=algorithm,
                        **algorithm_plot_style(algorithm, x_axis=x_axis),
                    )
                    legend_handles.setdefault(algorithm, line)
                    if PLOT_CONFIDENCE_ERROR_BARS and len(x_axis) > 1:
                        x_values = np.asarray(x_axis, dtype=float)
                        # Give each algorithm a distinct five-evaluation grid
                        # slot: 10, 15, 20, ... and repeat only after every
                        # selected algorithm has received one slot.
                        grid_step = max(
                            5.0,
                            float(benchmark_max) / 140.0,
                        )
                        targets = np.arange(
                            10.0 + grid_step * algorithm_index,
                            float(x_values[-1]) + 0.5 * grid_step,
                            max(
                                150.0,
                                grid_step * len(algorithms) * 2.0,
                            ),
                        )
                        positions = np.unique(
                            np.clip(
                                np.searchsorted(x_values, targets, side="left"),
                                0,
                                len(x_values) - 1,
                            )
                        )
                        if positions.size == 0:
                            continue
                        performance_ax.errorbar(
                            x_values[positions],
                            np.asarray(mean)[positions],
                            yerr=np.asarray(ci)[positions],
                            fmt="none",
                            ecolor=line.get_color(),
                            elinewidth=0.75,
                            capsize=2.0,
                            alpha=float(PLOT_CONFIDENCE_ERROR_BAR_ALPHA),
                        )

                performance_ax.set_xlabel("Number of evaluations")
                performance_ax.set_ylabel("Value")
                performance_ax.set_xlim(0, benchmark_max)
                performance_ax.set_ylim(*SYNTHETIC_GRAPH_Y_LIMITS[benchmark])
                performance_ax.grid(True, alpha=0.25)

                bar_rows = []
                for algorithm in algorithms:
                    mean_ms, ci_ms, n = terminal_mean_ci_from_histories(
                        _synthetic_wall_clock_histories_seconds(
                            algorithm,
                            grouped[benchmark].get(algorithm, []),
                        )
                    )
                    if n > 0:
                        bar_rows.append((algorithm, mean_ms, ci_ms))
                if bar_rows:
                    x_pos = np.arange(len(bar_rows))
                    wall_clock_ax.bar(
                        x_pos,
                        [row[1] for row in bar_rows],
                        yerr=[row[2] for row in bar_rows],
                        capsize=3,
                        color=[algorithm_plot_style(row[0])["color"] for row in bar_rows],
                        edgecolor="black",
                        linewidth=0.45,
                    )
                    wall_clock_ax.set_xticks(
                        x_pos,
                        [row[0] for row in bar_rows],
                        rotation=62,
                        ha="right",
                        fontsize=7,
                    )
                wall_clock_ax.set_xlabel("Algorithms")
                wall_clock_ax.set_ylabel("Wall-clock time (s)")
                wall_clock_ax.yaxis.set_major_formatter(
                    FuncFormatter(lambda value, _position: f"{value:,.0f}")
                )
                wall_clock_ax.grid(True, axis="y", alpha=0.25)
                fig.text(
                    0.255 if pair_index == 0 else 0.745,
                    0.865,
                    benchmark,
                    ha="center",
                    va="bottom",
                    fontsize=11,
                )

            if legend_handles:
                ordered_labels = [
                    algorithm for algorithm in algorithms if algorithm in legend_handles
                ]
                fig.legend(
                    [legend_handles[label] for label in ordered_labels],
                    ordered_labels,
                    loc="upper center",
                    bbox_to_anchor=(0.5, 0.99),
                    ncol=min(8, len(ordered_labels)),
                    frameon=False,
                )
            fig.tight_layout(rect=(0.01, 0.02, 0.99, 0.82), w_pad=2.0)
            output_path = family_root / f"{objective}_graphs.pdf"
            fig.savefig(output_path, dpi=220, bbox_inches="tight")
            plt.close(fig)
            output_paths.append(str(output_path))

        # Keep the paper-facing dimension-selection result compact: one PDF per
        # objective, directly under ``united``, with both d_eff conditions and
        # both AG-VS backends arranged as panels.
        dimension_algorithms = ("AG-VS-BO", "AG-VS-TuRBO")
        dimension_panels = []
        for benchmark in (f"{objective}3_300", f"{objective}6_1000"):
            for algorithm in dimension_algorithms:
                items = grouped[benchmark].get(algorithm, [])
                count_rows = []
                benchmark_dim = None
                for item in items:
                    settings = item.get("settings", {})
                    if benchmark_dim is None:
                        try:
                            benchmark_dim = int(settings.get("original_dim"))
                        except (TypeError, ValueError):
                            benchmark_dim = None
                    counts = item.get("agvs_dimension_selection_counts")
                    if counts is None and benchmark_dim is not None:
                        counts = agvs_dimension_selection_counts_from_history(
                            item.get("agvs_history"), benchmark_dim
                        )
                    row = np.asarray(counts if counts is not None else [], dtype=float)
                    if benchmark_dim is None and row.size:
                        benchmark_dim = int(row.size)
                    if benchmark_dim is not None and row.size == benchmark_dim:
                        count_rows.append(row)
                if not count_rows or benchmark_dim is None:
                    continue

                mean_counts = np.mean(np.vstack(count_rows), axis=0)
                settings = items[0].get("settings", {})
                active_indices = {
                    int(dim) for dim in (settings.get("active_indices") or [])
                }
                order = sorted(
                    range(benchmark_dim),
                    key=lambda dim: (-mean_counts[dim], dim),
                )
                dimension_panels.append(
                    {
                        "benchmark": benchmark,
                        "algorithm": algorithm,
                        "order": order,
                        "counts": mean_counts[order],
                        "colors": [
                            "#2ca02c" if dim in active_indices else "#1f77b4"
                            for dim in order
                        ],
                        "benchmark_dim": benchmark_dim,
                    }
                )

        if dimension_panels:
            fig, axes = plt.subplots(
                1,
                len(dimension_panels),
                figsize=(5.5 * len(dimension_panels), 5.2),
                squeeze=False,
            )
            axes = axes.reshape(-1)
            benchmark_centers = defaultdict(list)
            for panel_index, (ax, panel) in enumerate(zip(axes, dimension_panels)):
                x_pos = np.arange(panel["benchmark_dim"])
                ax.bar(x_pos, panel["counts"], color=panel["colors"], width=0.9)
                tick_step = max(1, panel["benchmark_dim"] // 20)
                tick_positions = x_pos[::tick_step]
                ax.set_xticks(
                    tick_positions,
                    [str(panel["order"][int(pos)]) for pos in tick_positions],
                    rotation=90,
                    fontsize=6,
                )
                ax.set_xlabel("Dimension index")
                if panel_index == 0:
                    ax.set_ylabel("Selection Count")
                ax.set_title(panel["algorithm"], fontsize=10)
                ax.grid(True, axis="y", alpha=0.25)
                benchmark_centers[panel["benchmark"]].append(panel_index)

            for benchmark, indices in benchmark_centers.items():
                center = np.mean([(index + 0.5) / len(dimension_panels) for index in indices])
                fig.text(
                    center,
                    0.88,
                    benchmark,
                    ha="center",
                    va="bottom",
                    fontsize=11,
                )
            fig.legend(
                handles=[
                    plt.Rectangle((0, 0), 1, 1, color="#2ca02c", label="Active"),
                    plt.Rectangle((0, 0), 1, 1, color="#1f77b4", label="Inactive"),
                ],
                loc="upper center",
                bbox_to_anchor=(0.5, 0.995),
                ncol=2,
                frameon=False,
            )
            fig.tight_layout(rect=(0.01, 0.02, 0.99, 0.82), w_pad=1.4)
            united_root = Path(target_root) / "united"
            united_root.mkdir(parents=True, exist_ok=True)
            dimension_path = united_root / f"{objective}_dimension_selection.pdf"
            fig.savefig(dimension_path, dpi=220, bbox_inches="tight")
            plt.close(fig)
            output_paths.append(str(dimension_path))

        # One compact evaluation-versus-wall-clock PDF per objective. The two
        # dimensional settings are panels in the same paper-facing artifact,
        # matching the layout and naming convention of ``levy_graphs.pdf``.
        benchmark_pair = (f"{objective}3_300", f"{objective}6_1000")
        wall_clock_algorithms = sorted(
            {
                algorithm
                for benchmark in benchmark_pair
                for algorithm in grouped[benchmark]
            },
            key=lambda name: (algorithm_plot_index(name), str(name)),
        )
        if wall_clock_algorithms:
            fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8), squeeze=False)
            axes = axes.reshape(-1)
            legend_handles = {}
            plotted_panel = False
            for panel_index, (ax, benchmark) in enumerate(zip(axes, benchmark_pair)):
                benchmark_max = plot_max_evaluation_for_items(
                    [
                        item
                        for algorithm in wall_clock_algorithms
                        for item in grouped[benchmark].get(algorithm, [])
                    ]
                )
                for algorithm in wall_clock_algorithms:
                    histories = _synthetic_wall_clock_histories_seconds(
                        algorithm,
                        grouped[benchmark].get(algorithm, []),
                    )
                    x_axis, mean, _ci, n = mean_ci_from_histories(
                        histories,
                        start_at_zero=True,
                        zero_value="zero",
                    )
                    if n == 0:
                        continue
                    x_axis, mean, _ci = crop_plot_series(
                        x_axis,
                        mean,
                        _ci,
                        max_evaluation=benchmark_max,
                    )
                    line, = ax.plot(
                        x_axis,
                        mean,
                        label=algorithm,
                        **algorithm_plot_style(algorithm, x_axis=x_axis),
                    )
                    legend_handles.setdefault(algorithm, line)
                    plotted_panel = True
                ax.set_title(benchmark)
                ax.set_xlabel("Number of evaluations")
                if panel_index == 0:
                    ax.set_ylabel("Wall-clock time (s)")
                ax.set_xlim(0, benchmark_max)
                ax.set_ylim(bottom=0)
                ax.grid(True, alpha=0.25)

            if plotted_panel:
                ordered_labels = [
                    algorithm
                    for algorithm in wall_clock_algorithms
                    if algorithm in legend_handles
                ]
                fig.legend(
                    [legend_handles[label] for label in ordered_labels],
                    ordered_labels,
                    loc="upper center",
                    bbox_to_anchor=(0.5, 0.99),
                    ncol=min(8, len(ordered_labels)),
                    frameon=False,
                    fontsize=8,
                )
                fig.tight_layout(rect=(0.01, 0.02, 0.99, 0.84), w_pad=1.8)
                united_root = Path(target_root) / "united"
                united_root.mkdir(parents=True, exist_ok=True)
                wall_clock_path = (
                    united_root / f"{objective}_iteration_vs_wall_clock.pdf"
                )
                fig.savefig(wall_clock_path, dpi=220, bbox_inches="tight")
                output_paths.append(str(wall_clock_path))
            plt.close(fig)

    for objective in SYNTHETIC_GRAPH_OBJECTIVES:
        benchmark_pair = (f"{objective}3_300", f"{objective}6_1000")
        if not all(name in selected for name in benchmark_pair):
            continue
        objective_traces = []
        for benchmark in benchmark_pair:
            benchmark_dir = benchmark_root / benchmark_output_slug(benchmark)
            trace_root = benchmark_dir if benchmark_dir.is_dir() else benchmark_root
            objective_traces.extend(
                trace
                for trace in load_traces(
                    trace_root,
                    reuse_complete_json=bool(reuse_complete_json),
                )
                if trace.get("complete", False)
                and str(trace.get("benchmark", "")).strip().lower() == benchmark
            )
        render(objective, objective_traces, figure_root)
        if int(FIRST_N_TRIAL_FIGURES) > 0:
            first_n_traces = []
            for trace in objective_traces:
                trial_id = trace.get("trial_id", trace.get("trial"))
                try:
                    include = 0 <= int(trial_id) < int(FIRST_N_TRIAL_FIGURES)
                except (TypeError, ValueError):
                    include = False
                if include:
                    first_n_traces.append(trace)
            render(
                objective,
                first_n_traces,
                figure_root / f"first_{int(FIRST_N_TRIAL_FIGURES)}_trials",
            )
    return output_paths


def plot_results(
    results_dir,
    figure_dir,
    benchmark_filter=None,
    update_objective_tables=True,
    table_dir=None,
    algorithm_family=None,
    _loaded_traces=None,
    _top_algorithms_by_benchmark=None,
    _generate_first_n_figures=True,
):
    if _loaded_traces is None:
        trace_root = Path(results_dir)
        if benchmark_filter is not None:
            benchmark_dir = trace_root / benchmark_output_slug(benchmark_filter)
            if benchmark_dir.is_dir():
                trace_root = benchmark_dir
        traces = [t for t in load_traces(trace_root) if t.get("complete", False)]
    else:
        traces = list(_loaded_traces)
    if benchmark_filter is not None:
        benchmark_filter = str(benchmark_filter).lower()
        traces = [
            trace for trace in traces
            if str(trace.get("benchmark", "")).lower() == benchmark_filter
        ]

    if algorithm_family is not None:
        family = str(algorithm_family).strip().lower()
        if family not in FIGURE_OUTPUT_FAMILIES:
            raise ValueError(f"Unknown algorithm output family: {family}")
        if family != "united":
            traces = [
                trace
                for trace in traces
                if comparison_output_family_includes(trace.get("algorithm"), family)
            ]

    if _top_algorithms_by_benchmark is None:
        ranking_groups = defaultdict(lambda: defaultdict(list))
        for trace in traces:
            ranking_groups[trace["benchmark"]][trace["algorithm"]].append(trace)
        _top_algorithms_by_benchmark = {}
        for benchmark, ranking_by_algorithm in ranking_groups.items():
            selected, _scores = select_top_algorithms_by_final_mean_noiseless(
                ranking_by_algorithm,
                trace_noiseless_history_for_plot,
                limit=None,
            )
            _top_algorithms_by_benchmark[benchmark] = selected

    if algorithm_family is None and traces:
        grouped_all = defaultdict(lambda: defaultdict(list))
        for trace in traces:
            grouped_all[trace["benchmark"]][trace["algorithm"]].append(trace)

        paths = []
        for family in FIGURE_OUTPUT_FAMILIES:
            if not figure_output_family_is_available(traces, family):
                continue
            paths.extend(
                plot_results(
                    results_dir,
                    Path(figure_dir) / family,
                    benchmark_filter=benchmark_filter,
                    update_objective_tables=False,
                    table_dir=table_dir,
                    algorithm_family=family,
                    _loaded_traces=traces,
                    # Select the top five independently within BO and TuRBO.
                    _top_algorithms_by_benchmark=None,
                )
            )
        if update_objective_tables:
            paths.extend(
                update_aggregate_objective_tables(
                    grouped_all,
                    benchmark_root=results_dir,
                    table_dir=table_dir,
                )
            )
        if _generate_first_n_figures and int(FIRST_N_TRIAL_FIGURES) > 0:
            first_n_traces = []
            for trace in traces:
                trial_id = trace.get("trial_id", trace.get("trial"))
                try:
                    include = 0 <= int(trial_id) < int(FIRST_N_TRIAL_FIGURES)
                except (TypeError, ValueError):
                    include = False
                if include:
                    first_n_traces.append(trace)
            if first_n_traces:
                paths.extend(
                    plot_results(
                        results_dir,
                        Path(figure_dir) / f"first_{int(FIRST_N_TRIAL_FIGURES)}_trials",
                        benchmark_filter=benchmark_filter,
                        update_objective_tables=False,
                        table_dir=None,
                        algorithm_family=None,
                        _loaded_traces=first_n_traces,
                        _top_algorithms_by_benchmark=None,
                        _generate_first_n_figures=False,
                    )
                )
        return paths

    if not traces:
        return []

    def trace_experiment_mode(item):
        settings = item.get("settings", {}) if isinstance(item, dict) else {}
        if "experiment_mode" in settings:
            return bool(settings.get("experiment_mode"))
        return settings.get("active_indices") is not None

    def trace_posterior_mean_history(item):
        hist = item.get("reported_posterior_mean_history", item.get("best_posterior_mean_history"))
        return hist if isinstance(hist, list) else []

    def trace_noiseless_history(item):
        return trace_noiseless_history_for_plot(item)

    def trace_observed_history(item):
        hist = item.get("best_observed_history")
        if isinstance(hist, list):
            return hist
        observed = item.get("observed_y_history", [])
        return best_so_far(observed) if observed else []

    grouped = defaultdict(lambda: defaultdict(list))
    for trace in traces:
        grouped[trace["benchmark"]][trace["algorithm"]].append(trace)

    Path(figure_dir).mkdir(parents=True, exist_ok=True)
    paths = []

    for benchmark, by_algorithm in sorted(grouped.items()):
        benchmark_dir = Path(figure_dir) / benchmark_output_slug(benchmark)
        benchmark_dir.mkdir(parents=True, exist_ok=True)

        family_ranked_algorithms, final_noiseless_mean_by_algorithm = (
            select_top_algorithms_by_final_mean_noiseless(
                by_algorithm,
                trace_noiseless_history,
                limit=None,
            )
        )
        top_algorithms = [
            algorithm
            for algorithm in _top_algorithms_by_benchmark.get(
                benchmark,
                family_ranked_algorithms,
            )
            if algorithm in by_algorithm
        ]
        by_algorithm = {
            algorithm: by_algorithm[algorithm]
            for algorithm in top_algorithms
        }
        if not by_algorithm:
            continue

        ci_algorithm_order = sorted(
            by_algorithm,
            key=lambda name: (algorithm_plot_index(name), str(name)),
        )
        ci_slot_by_algorithm = {
            name: index for index, name in enumerate(ci_algorithm_order)
        }

        confidence_errorbar_renderer = globals()["plot_confidence_band"]

        def plot_confidence_band(x_axis, lower, upper, **kwargs):
            """Bind existing plot calls to the current algorithm's CI slot."""
            return confidence_errorbar_renderer(
                x_axis,
                lower,
                upper,
                series_index=ci_slot_by_algorithm[algorithm],
                series_count=len(ci_algorithm_order),
                **kwargs,
            )
        
        # Look up dimension robustly
        if benchmark in BENCHMARKS:
            benchmark_dim = int(np.asarray(BENCHMARKS[benchmark]["bounds"]).shape[0])
        elif traces and "settings" in traces[0] and "original_dim" in traces[0]["settings"]:
            benchmark_dim = int(traces[0]["settings"]["original_dim"])
        else:
            benchmark_dim = D_ORIG

        benchmark_plot_max = plot_max_evaluation_for_benchmark(
            benchmark,
            [item for items in by_algorithm.values() for item in items]
        )
        real_life_benchmark = any(
            str(item.get("settings", {}).get("experiment_kind", "synthetic"))
            == "real_life"
            for items in by_algorithm.values()
            for item in items
            if isinstance(item, dict)
        )
        nasbench101_plot = str(benchmark).lower().startswith("nasbench101")
        rover60_plot = str(benchmark).lower().startswith("rover60")
        mopta08_plot = str(benchmark).lower().startswith("mopta08")
        paper_style_plot = nasbench101_plot or rover60_plot or mopta08_plot
        paper_style_title = (
            "NAS-Bench-101"
            if nasbench101_plot
            else "Rover60"
            if rover60_plot
            else "MOPTA08"
        )
        paper_style_value_label = (
            "Accuracy"
            if nasbench101_plot
            else "Reward"
            if rover60_plot
            else "Function value"
        )
        if real_life_benchmark:
            plt.figure(figsize=(9, 7.5) if paper_style_plot else (9, 5.5))
            plotted_observed = False
            for algorithm, items in sorted(by_algorithm.items()):
                histories = [trace_observed_history(item) for item in items]
                x_axis, mean, ci, n = mean_ci_from_histories(histories)
                if n == 0:
                    continue
                x_axis, mean, ci = crop_plot_series(
                    x_axis, mean, ci, max_evaluation=benchmark_plot_max
                )
                line = plot_algorithm_series(
                    x_axis,
                    mean,
                    algorithm,
                    (
                        canonical_algorithm_name(algorithm)
                        if paper_style_plot
                        else f"{algorithm} best observed objective (n={n})"
                    ),
                )
                plot_confidence_band(
                    x_axis,
                    mean - ci,
                    mean + ci,
                    color=line.get_color(),
                    alpha=0.18,
                )
                plotted_observed = True
            if plotted_observed:
                plt.xlabel("Evaluation")
                plt.ylabel(
                    paper_style_value_label
                    if paper_style_plot
                    else "Mean running-best observed accuracy/reward"
                )
                plt.title(
                    paper_style_title
                    if paper_style_plot
                    else f"{benchmark} | mean best observed real-life objective"
                )
                plt.grid(True, alpha=0.25)
                plt.xlim(0, benchmark_plot_max)
                apply_nasbench101_value_y_limits(benchmark)
                legend_sorted_by_last_y(
                    outside_top=paper_style_plot,
                    ncol=4 if paper_style_plot else None,
                )
                out = benchmark_dir / "best_observed_objective_mean.pdf"
                plt.tight_layout(
                    rect=(0.0, 0.0, 1.0, 0.84)
                    if paper_style_plot
                    else None
                )
                plt.savefig(out, dpi=200)
                paths.append(str(out))
                alias_out = benchmark_dir / "value_vs_iteration_all_algorithms_mean.pdf"
                plt.savefig(alias_out, dpi=200)
                paths.append(str(alias_out))
            plt.close()

            # Paper-style real-life comparison: running-best observed metric
            # against actual elapsed wall-clock time. Never use the unbounded
            # GP posterior mean as an accuracy/reward curve.
            plt.figure(
                figsize=(9, 7.5)
                if str(benchmark).lower().startswith("nasbench101")
                else (9, 5.5)
            )
            plotted_time_objective = False
            real_life_task = next(
                (
                    str(item.get("settings", {}).get("real_life_task", ""))
                    for items in by_algorithm.values()
                    for item in items
                    if isinstance(item, dict)
                    and item.get("settings", {}).get("real_life_task")
                ),
                "",
            ).strip().lower()
            for algorithm, items in sorted(by_algorithm.items()):
                time_axis, mean, ci, n = wall_clock_mean_ci_from_traces(
                    items,
                    trace_observed_history,
                    algorithm,
                    minimum_grid_step_milliseconds=(
                        1_000
                        if str(benchmark).lower().startswith("nasbench101")
                        else 1
                    ),
                    maximum_time_milliseconds=(
                        20_000
                        if str(benchmark).lower().startswith("nasbench101")
                        else None
                    ),
                    maximum_grid_points=(
                        20
                        if str(benchmark).lower().startswith("nasbench101")
                        else 2000
                    ),
                    # Real-life wall-clock comparisons include the time spent
                    # generating and evaluating the initial design.  Those
                    # timestamps are persisted in every trial JSON, and the
                    # final-time ranking bar uses the same full runtime.
                    exclude_initial_design=False,
                    extend_completed_trials=(
                        str(benchmark).lower().startswith("nasbench101")
                    ),
                )
                if n == 0:
                    continue
                plot_time_axis = time_axis / 1000.0
                line = plot_algorithm_series(
                    plot_time_axis,
                    mean,
                    algorithm,
                    (
                        canonical_algorithm_name(algorithm)
                        if str(benchmark).lower().startswith("nasbench101")
                        else f"{algorithm} best observed (n={n})"
                    ),
                )
                plot_confidence_band(
                    plot_time_axis,
                    mean - ci,
                    mean + ci,
                    color=line.get_color(),
                    alpha=0.18,
                )
                plotted_time_objective = True
            if plotted_time_objective:
                if real_life_task in {
                    "nasbench101",
                    "nasbench201",
                    "nasbench1shot1",
                    "transnasbench101",
                }:
                    objective_label = "Mean running-best observed accuracy"
                elif real_life_task == "nasbenchasr":
                    objective_label = "Mean running-best negative validation PER"
                elif real_life_task in {"hopper", "walker2d", "rover60"}:
                    objective_label = "Mean running-best observed reward"
                else:
                    objective_label = "Mean running-best observed objective"
                if str(benchmark).lower().startswith("nasbench101"):
                    objective_label = "Accuracy"
                elif rover60_plot:
                    objective_label = "Reward"
                elif mopta08_plot:
                    objective_label = "Function value"
                plt.xlabel("Wall-clock time (s)")
                plt.ylabel(objective_label)
                plt.title(
                    paper_style_title
                    if paper_style_plot
                    else f"{benchmark} | mean best observed objective vs wall-clock time"
                )
                plt.grid(True, alpha=0.25)
                if str(benchmark).lower().startswith("nasbench101"):
                    plt.xscale("linear")
                    plt.xlim(0.0, 20.0)
                    apply_nasbench101_value_y_limits(benchmark)
                    plt.xlabel("Wall-clock time (s)")
                    plt.gca().xaxis.set_major_locator(MultipleLocator(1.0))
                else:
                    plt.xlim(left=0.0)
                legend_sorted_by_last_y(
                    outside_top=paper_style_plot,
                    ncol=4,
                )
                out = benchmark_dir / "best_observed_vs_wall_clock_mean.pdf"
                plt.tight_layout(
                    rect=(0.0, 0.0, 1.0, 0.84)
                    if paper_style_plot
                    else None
                )
                plt.savefig(out, dpi=200)
                paths.append(str(out))
                alias_out = benchmark_dir / "value_vs_time_seconds_all_algorithms_mean.pdf"
                plt.savefig(alias_out, dpi=200)
                paths.append(str(alias_out))
                (
                    benchmark_dir
                    / "value_vs_time_milliseconds_all_algorithms_mean.pdf"
                ).unlink(missing_ok=True)
            plt.close()

        plt.figure(figsize=(9, 5.5))
        plotted_posterior = False
        posterior_means_by_algorithm = {}

        posterior_plot_items = (
            by_algorithm if PLOT_POSTERIOR_MEAN_DIAGNOSTICS else {}
        )
        for algorithm, items in sorted(posterior_plot_items.items()):
            histories = []
            for item in items:
                histories.append(trace_posterior_mean_history(item))

            x_axis, mean, ci, n = mean_ci_from_histories(histories)
            if n == 0:
                continue
            x_axis, mean, ci = crop_plot_series(x_axis, mean, ci, max_evaluation=benchmark_plot_max)
            posterior_means_by_algorithm[algorithm] = mean

            line = plot_algorithm_series(x_axis, mean, algorithm, f"{algorithm} best posterior mean (n={n})")
            plot_confidence_band(x_axis, mean - ci, mean + ci, color=line.get_color(), alpha=0.18)
            plotted_posterior = True

        if plotted_posterior:
            plt.xlabel("Evaluation")
            plt.ylabel("Mean running-best GP posterior mean")
            plt.title(f"{benchmark} | model-recommended posterior mean with 95% CI")
            plt.grid(True, alpha=0.25)
            plt.xlim(0, benchmark_plot_max)
            legend_sorted_by_last_y()
            out = benchmark_dir / "best_posterior_mean_95ci.pdf"
            plt.tight_layout()
            plt.savefig(out, dpi=200)
            plt.ylim(-2, 0)
            plt.title(
                f"{benchmark} | model-recommended posterior mean with 95% CI "
                "| y in [-2, 0]"
            )
            zoom_out = benchmark_dir / "best_posterior_mean_95ci_y_-2_0.pdf"
            plt.tight_layout()
            plt.savefig(zoom_out, dpi=200)
            plt.ylim(-1, 0)
            plt.title(
                f"{benchmark} | model-recommended posterior mean with 95% CI "
                "| y in [-1, 0]"
            )
            zoom_minus_one = benchmark_dir / "best_posterior_mean_95ci_y_-1_0.pdf"
            plt.tight_layout()
            plt.savefig(zoom_minus_one, dpi=200)
            plt.close()
            paths.append(str(out))
            paths.append(str(zoom_out))
            paths.append(str(zoom_minus_one))
        else:
            plt.close()

        if plotted_posterior and str(benchmark).lower().startswith("rosenbrock"):
            plt.figure(figsize=(9, 5.5))
            for algorithm, items in sorted(by_algorithm.items()):
                histories = []
                for item in items:
                    histories.append(trace_posterior_mean_history(item))
                x_axis, mean, ci, n = mean_ci_from_histories(histories)
                if n == 0:
                    continue
                x_axis, mean, ci = crop_plot_series(
                    x_axis,
                    mean,
                    ci,
                    max_evaluation=benchmark_plot_max,
                )
                line = plot_algorithm_series(x_axis, mean, algorithm, f"{algorithm} best posterior mean (n={n})")
                plot_confidence_band(x_axis, mean - ci, mean + ci, color=line.get_color(), alpha=0.18)

            plt.xlabel("Evaluation")
            plt.ylabel("Mean running-best GP posterior mean")
            plt.title(f"{benchmark} | model-recommended posterior mean with 95% CI | y in [-100, 0]")
            plt.grid(True, alpha=0.25)
            plt.xlim(0, benchmark_plot_max)
            plt.ylim(-100, 0)
            legend_sorted_by_last_y()
            zoom_out = benchmark_dir / "best_posterior_mean_95ci_y_-100_0.pdf"
            plt.tight_layout()
            plt.savefig(zoom_out, dpi=200)
            plt.close()
            paths.append(str(zoom_out))

        if plotted_posterior and str(benchmark).lower().startswith("rastrigin"):
            plt.figure(figsize=(9, 5.5))
            for algorithm, items in sorted(by_algorithm.items()):
                histories = []
                for item in items:
                    histories.append(trace_posterior_mean_history(item))
                x_axis, mean, ci, n = mean_ci_from_histories(histories)
                if n == 0:
                    continue
                x_axis, mean, ci = crop_plot_series(
                    x_axis,
                    mean,
                    ci,
                    max_evaluation=benchmark_plot_max,
                )
                line = plot_algorithm_series(x_axis, mean, algorithm, f"{algorithm} best posterior mean (n={n})")
                plot_confidence_band(x_axis, mean - ci, mean + ci, color=line.get_color(), alpha=0.18)

            plt.xlabel("Evaluation")
            plt.ylabel("Mean running-best GP posterior mean")
            plt.title(f"{benchmark} | model-recommended posterior mean with 95% CI | y in [-10, 0]")
            plt.grid(True, alpha=0.25)
            plt.xlim(0, benchmark_plot_max)
            plt.ylim(-10, 0)
            legend_sorted_by_last_y()
            zoom_out = benchmark_dir / "best_posterior_mean_95ci_y_-10_0.pdf"
            plt.tight_layout()
            plt.savefig(zoom_out, dpi=200)
            plt.close()
            paths.append(str(zoom_out))

        if plotted_posterior and str(benchmark).lower().startswith("levy"):
            plt.figure(figsize=(9, 5.5))
            for algorithm, items in sorted(by_algorithm.items()):
                histories = []
                for item in items:
                    histories.append(trace_posterior_mean_history(item))
                x_axis, mean, ci, n = mean_ci_from_histories(histories)
                if n == 0:
                    continue
                x_axis, mean, ci = crop_plot_series(
                    x_axis,
                    mean,
                    ci,
                    max_evaluation=benchmark_plot_max,
                )
                line = plot_algorithm_series(x_axis, mean, algorithm, f"{algorithm} reported posterior mean (n={n})")
                plot_confidence_band(x_axis, mean - ci, mean + ci, color=line.get_color(), alpha=0.18)

            plt.xlabel("Evaluation")
            plt.ylabel("Mean reported GP posterior mean")
            plt.title(f"{benchmark} | reported posterior mean with 95% CI | y in [-10, 0]")
            plt.grid(True, alpha=0.25)
            plt.xlim(0, benchmark_plot_max)
            plt.ylim(-10, 0)
            legend_sorted_by_last_y()
            zoom_out = benchmark_dir / "best_posterior_mean_95ci_y_-10_0.pdf"
            plt.tight_layout()
            plt.savefig(zoom_out, dpi=200)
            plt.close()
            paths.append(str(zoom_out))

        # Ground-truth/noiseless performance is plotted separately from the
        # posterior-return curve; neither metric replaces the other.
        plt.figure(figsize=(9, 7.5) if paper_style_plot else (9, 5.5))
        plotted_noiseless = False
        noiseless_means = {}
        for algorithm, items in sorted(by_algorithm.items()):
            histories = [trace_noiseless_history(item) for item in items]
            x_axis, mean, ci, n = mean_ci_from_histories(histories)
            if n == 0:
                continue
            x_axis, mean, ci = crop_plot_series(
                x_axis, mean, ci, max_evaluation=benchmark_plot_max
            )
            noiseless_means[algorithm] = mean
            line = plot_algorithm_series(
                x_axis,
                mean,
                algorithm,
                (
                    canonical_algorithm_name(algorithm)
                    if paper_style_plot
                    else f"{algorithm} noiseless best-y (n={n})"
                ),
            )
            plot_confidence_band(x_axis, mean - ci, mean + ci, color=line.get_color(), alpha=0.18)
            plotted_noiseless = True

        if plotted_noiseless:
            plt.xlabel(
                "Evaluation" if real_life_benchmark else "Number of evaluations"
            )
            plt.ylabel(
                paper_style_value_label
                if paper_style_plot
                else "Mean best noiseless score"
                if real_life_benchmark
                else "Value"
            )
            plt.title(
                paper_style_title
                if paper_style_plot
                else f"{benchmark} | mean best noiseless y"
                if real_life_benchmark
                else str(benchmark)
            )
            plt.grid(True, alpha=0.25)
            plt.xlim(0, benchmark_plot_max)
            apply_nasbench101_value_y_limits(benchmark)
            legend_sorted_by_last_y(
                outside_top=paper_style_plot,
                ncol=4 if paper_style_plot else None,
            )
            out = benchmark_dir / "best_noiseless_y_mean.pdf"
            plt.tight_layout(
                rect=(0.0, 0.0, 1.0, 0.84)
                if paper_style_plot
                else None
            )
            plt.savefig(out, dpi=200)
            benchmark_lower = str(benchmark).lower()
            if not real_life_benchmark:
                for stale_pattern in (
                    "best_noiseless_y_mean_y_*.pdf",
                    "best_posterior_mean_95ci_y_*.pdf",
                    "inverse_cdf_threshold_delta_*_95ci_y_*.pdf",
                ):
                    for stale_path in benchmark_dir.glob(stale_pattern):
                        stale_path.unlink(missing_ok=True)
                zoom_paths = []
            elif benchmark_lower.startswith("nasbench101"):
                for stale_name in (
                    "best_noiseless_y_mean_y_-2_0.pdf",
                    "best_noiseless_y_mean_y_-1_0.pdf",
                    "best_noiseless_y_mean_y_0.9_0.95.pdf",
                    "best_noiseless_y_mean_y_0.9_0.94.pdf",
                    "best_noiseless_y_mean_y_0.93_0.94.pdf",
                ):
                    (benchmark_dir / stale_name).unlink(missing_ok=True)
                apply_nasbench101_value_y_limits(benchmark)
                plt.title("NAS-Bench-101")
                precise_zoom_out = (
                    benchmark_dir / "best_noiseless_y_mean_y_0.934_0.940.pdf"
                )
                plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.84))
                plt.savefig(precise_zoom_out, dpi=200)
                zoom_paths = [precise_zoom_out]
            elif rover60_plot:
                # Rover reward has no meaningful fixed [-2, 0] zoom; retain
                # the auto-scaled primary graph, matching the paper layout.
                zoom_paths = []
            else:
                plt.ylim(-2, 0)
                plt.title(
                    f"{benchmark} | mean best noiseless y | y in [-2, 0]"
                )
                zoom_out = benchmark_dir / "best_noiseless_y_mean_y_-2_0.pdf"
                plt.tight_layout()
                plt.savefig(zoom_out, dpi=200)
                plt.ylim(-1, 0)
                plt.title(
                    f"{benchmark} | mean best noiseless y | y in [-1, 0]"
                )
                zoom_minus_one = benchmark_dir / "best_noiseless_y_mean_y_-1_0.pdf"
                plt.tight_layout()
                plt.savefig(zoom_minus_one, dpi=200)
                zoom_paths = [zoom_out, zoom_minus_one]
            plt.close()
            paths.append(str(out))
            paths.extend(str(path) for path in zoom_paths)
        else:
            plt.close()

        zoom_specs = []
        benchmark_lower = str(benchmark).lower()
        if benchmark_lower.startswith("rosenbrock"):
            zoom_specs.append((-100, 0))
        elif benchmark_lower.startswith(("rastrigin", "levy")):
            zoom_specs.append((-10, 0))
        if benchmark_lower == "levy6_1000" or benchmark_lower.startswith("rosenbrock"):
            zoom_specs.append((-5, 0))
        for low, high in (
            zoom_specs if plotted_noiseless and real_life_benchmark else ()
        ):
            plt.figure(figsize=(9, 5.5))
            for algorithm, items in sorted(by_algorithm.items()):
                histories = [trace_noiseless_history(item) for item in items]
                x_axis, mean, ci, n = mean_ci_from_histories(histories)
                if n == 0:
                    continue
                x_axis, mean, ci = crop_plot_series(
                    x_axis, mean, ci, max_evaluation=benchmark_plot_max
                )
                line = plot_algorithm_series(
                    x_axis, mean, algorithm, f"{algorithm} noiseless best-y (n={n})"
                )
                plot_confidence_band(x_axis, mean - ci, mean + ci, color=line.get_color(), alpha=0.18)
            plt.xlabel("Evaluation")
            plt.ylabel("Mean best noiseless score")
            plt.title(f"{benchmark} | mean best noiseless y | y in [{low}, {high}]")
            plt.grid(True, alpha=0.25)
            plt.xlim(0, benchmark_plot_max)
            plt.ylim(low, high)
            legend_sorted_by_last_y()
            zoom_out = benchmark_dir / f"best_noiseless_y_mean_y_{low}_{high}.pdf"
            plt.tight_layout()
            plt.savefig(zoom_out, dpi=200)
            plt.close()
            paths.append(str(zoom_out))

        if not PLOT_POSTERIOR_MEAN_DIAGNOSTICS:
            for stale_path in benchmark_dir.glob("best_posterior_mean*.pdf"):
                stale_path.unlink(missing_ok=True)

        if not POSTERIOR_STOPPING_ENABLED:
            for stale_pattern in (
                "inverse_cdf_threshold_delta_*.pdf",
                "success_rate_delta_*_wilson_95ci.pdf",
            ):
                for stale_path in benchmark_dir.glob(stale_pattern):
                    stale_path.unlink(missing_ok=True)
            terminate_dir = benchmark_dir / "terminate"
            if terminate_dir.exists():
                for stale_path in terminate_dir.glob(
                    "success_rate_after_terminal_merge_delta_*_wilson_95ci.pdf"
                ):
                    stale_path.unlink(missing_ok=True)

        for delta in (RISK_TOLERANCES if POSTERIOR_STOPPING_ENABLED else []):
            delta_key = risk_label(delta)
            delta_pretty = risk_label_pretty(delta)
            plt.figure(figsize=(9, 5.5))

            for algorithm, items in sorted(by_algorithm.items()):
                threshold_histories = [
                    threshold_history_from_trace(item, delta_key)
                    for item in items
                ]

                x_axis, threshold_mean, threshold_ci, n = mean_ci_from_histories(threshold_histories)
                if n == 0:
                    continue
                x_axis, threshold_mean, threshold_ci = crop_plot_series(
                    x_axis,
                    threshold_mean,
                    threshold_ci,
                    max_evaluation=benchmark_plot_max,
                )

                threshold_line = plot_algorithm_series(x_axis, threshold_mean, algorithm, f"{algorithm} threshold (n={n})")
                plot_confidence_band(
                    x_axis,
                    threshold_mean - threshold_ci,
                    threshold_mean + threshold_ci,
                    color=threshold_line.get_color(),
                    alpha=0.15,
                )

                mean_best_noiseless_y = noiseless_means.get(algorithm)
                if mean_best_noiseless_y is not None:
                    usable_len = min(len(mean_best_noiseless_y), len(threshold_mean))
                    if usable_len > 0:
                        bad_mask = mean_best_noiseless_y[:usable_len] < threshold_mean[:usable_len]
                        n_init_values = [plot_marker_start_after_n_init(item) for item in items]
                        marker_start = max(n_init_values) if n_init_values else 0
                        bad_mask &= x_axis[:usable_len] > marker_start
                        if np.any(bad_mask):
                            bad_x = x_axis[:usable_len][bad_mask]
                            bad_y = threshold_mean[:usable_len][bad_mask]
                            if len(bad_x) > 80:
                                pick = np.linspace(0, len(bad_x) - 1, 80).astype(int)
                                bad_x = bad_x[pick]
                                bad_y = bad_y[pick]
                            plt.scatter(
                                bad_x,
                                bad_y,
                                marker="x",
                                color=threshold_line.get_color(),
                                s=34,
                                linewidths=1.2,
                                label="_nolegend_",
                            )

            plt.xlabel("Evaluation")
            plt.ylabel("Mean best stoppable threshold")
            plt.title(f"{benchmark} | inverse-CDF threshold with 95% CI | δ={delta_pretty}")
            plt.grid(True, alpha=0.25)
            plt.xlim(0, benchmark_plot_max)
            legend_sorted_by_last_y()
            out = benchmark_dir / f"inverse_cdf_threshold_delta_{delta_key}_95ci.pdf"
            plt.tight_layout()
            plt.savefig(out, dpi=200)
            plt.ylim(-2, 0)
            plt.title(
                f"{benchmark} | inverse-CDF threshold with 95% CI | "
                f"Î´={delta_pretty} | y in [-2, 0]"
            )
            zoom_out = (
                benchmark_dir
                / f"inverse_cdf_threshold_delta_{delta_key}_95ci_y_-2_0.pdf"
            )
            plt.tight_layout()
            plt.savefig(zoom_out, dpi=200)
            plt.ylim(-1, 0)
            plt.title(
                f"{benchmark} | inverse-CDF threshold with 95% CI | "
                f"delta={delta_pretty} | y in [-1, 0]"
            )
            zoom_minus_one = (
                benchmark_dir
                / f"inverse_cdf_threshold_delta_{delta_key}_95ci_y_-1_0.pdf"
            )
            plt.tight_layout()
            plt.savefig(zoom_minus_one, dpi=200)
            plt.close()
            paths.append(str(out))
            paths.append(str(zoom_out))
            paths.append(str(zoom_minus_one))

            plt.figure(figsize=(9, 5.5))
            plotted_success = False
            for algorithm, items in sorted(by_algorithm.items()):
                success_histories = [
                    success_history_from_trace(item, delta_key)
                    for item in items
                ]
                x_axis, success_rate, success_lower, success_upper, n = bernoulli_wilson_ci_from_histories(
                    success_histories,
                    start_at_zero=True,
                )
                if n == 0:
                    continue
                x_axis, success_rate, success_lower, success_upper = crop_plot_series(
                    x_axis,
                    success_rate,
                    success_lower,
                    success_upper,
                    max_evaluation=benchmark_plot_max,
                )
                line = plot_algorithm_series(
                    x_axis,
                    success_rate,
                    algorithm,
                    f"{algorithm} success rate (n={n})",
                )
                plot_confidence_band(
                    x_axis,
                    success_lower,
                    success_upper,
                    color=line.get_color(),
                    alpha=0.15,
                )
                plotted_success = True

            if plotted_success:
                plt.xlabel("Evaluation")
                plt.ylabel("Success rate")
                plt.title(
                    f"{benchmark} | success rate with Wilson 95% CI | "
                    f"best noiseless y >= best stoppable threshold | δ={delta_pretty}"
                )
                plt.grid(True, alpha=0.25)
                plt.xlim(0, benchmark_plot_max)
                plt.ylim(-0.02, 1.02)
                legend_sorted_by_last_y()
                success_out = benchmark_dir / f"success_rate_delta_{delta_key}_wilson_95ci.pdf"
                plt.tight_layout()
                plt.savefig(success_out, dpi=200)
                plt.close()
                paths.append(str(success_out))
            else:
                plt.close()

            terminal_start_by_trial = {}
            expected_iter_items = []
            for agvs_algorithm in AGVS_ALGORITHMS:
                agvs_items = by_algorithm.get(agvs_algorithm, [])
                expected_iter_items.extend(agvs_items)
                for trial, start_iter in terminal_start_by_trial_for_items(agvs_items).items():
                    terminal_start_by_trial.setdefault(trial, start_iter)

            if terminal_start_by_trial:
                terminate_dir = benchmark_dir / "terminate"
                terminate_dir.mkdir(parents=True, exist_ok=True)
                plt.figure(figsize=(9, 5.5))
                plotted_terminal_success = False
                for algorithm, items in sorted(by_algorithm.items()):
                    terminal_histories = []
                    used_trials = 0
                    for item in items:
                        try:
                            trial = int(item.get("trial"))
                        except Exception:
                            continue
                        terminal_start = terminal_start_by_trial.get(trial)
                        if terminal_start is None:
                            continue
                        history = terminate_success_history_from_trace(item, delta_key, terminal_start)
                        if history:
                            terminal_histories.append(history)
                            used_trials += 1

                    x_axis, success_rate, success_lower, success_upper, n = bernoulli_wilson_ci_from_histories(
                        terminal_histories,
                        start_at_zero=False,
                    )
                    if n == 0:
                        continue
                    x_axis, success_rate, success_lower, success_upper = crop_plot_series(
                        x_axis,
                        success_rate,
                        success_lower,
                        success_upper,
                        max_evaluation=benchmark_plot_max,
                    )
                    line = plot_algorithm_series(
                        x_axis,
                        success_rate,
                        algorithm,
                        f"{algorithm} terminal success rate (n={used_trials})",
                    )
                    plot_confidence_band(
                        x_axis,
                        success_lower,
                        success_upper,
                        color=line.get_color(),
                        alpha=0.15,
                    )
                    plotted_terminal_success = True

                if plotted_terminal_success:
                    plt.xlabel("Evaluation")
                    plt.ylabel("Success rate after terminal merge")
                    plt.title(
                        f"{benchmark} | after terminal merge success rate with Wilson 95% CI | "
                        f"best noiseless y >= best stoppable threshold | δ={delta_pretty}"
                    )
                    plt.grid(True, alpha=0.25)
                    plt.xlim(0, benchmark_plot_max)
                    plt.ylim(-0.02, 1.02)
                    legend_sorted_by_last_y()
                    annotate_plot_corner(
                        plt.gca(),
                        [
                            terminal_iter_text_from_starts(terminal_start_by_trial),
                            expected_iter_text_for_items(expected_iter_items),
                        ],
                    )
                    terminal_success_out = (
                        terminate_dir
                        / f"success_rate_after_terminal_merge_delta_{delta_key}_wilson_95ci.pdf"
                    )
                    plt.tight_layout()
                    plt.savefig(terminal_success_out, dpi=200)
                    plt.close()
                    paths.append(str(terminal_success_out))
                else:
                    plt.close()

            if str(benchmark).lower().startswith("rosenbrock") and delta_key == risk_label(1 / 256):
                plt.figure(figsize=(9, 5.5))
                for algorithm, items in sorted(by_algorithm.items()):
                    threshold_histories = [
                        threshold_history_from_trace(item, delta_key)
                        for item in items
                    ]

                    x_axis, threshold_mean, threshold_ci, n = mean_ci_from_histories(threshold_histories)
                    if n == 0:
                        continue
                    x_axis, threshold_mean, threshold_ci = crop_plot_series(
                        x_axis,
                        threshold_mean,
                        threshold_ci,
                        max_evaluation=benchmark_plot_max,
                    )
                    line = plot_algorithm_series(x_axis, threshold_mean, algorithm, f"{algorithm} threshold (n={n})")
                    plot_confidence_band(
                        x_axis,
                        threshold_mean - threshold_ci,
                        threshold_mean + threshold_ci,
                        color=line.get_color(),
                        alpha=0.15,
                    )

                plt.xlabel("Evaluation")
                plt.ylabel("Mean best stoppable threshold")
                plt.title(f"{benchmark} | inverse-CDF threshold with 95% CI | δ={delta_pretty} | y in [-1000, 0]")
                plt.grid(True, alpha=0.25)
                plt.xlim(0, benchmark_plot_max)
                plt.ylim(-1000, 0)
                legend_sorted_by_last_y()
                zoom_out = benchmark_dir / f"inverse_cdf_threshold_delta_{delta_key}_95ci_y_-1000_0.pdf"
                plt.tight_layout()
                plt.savefig(zoom_out, dpi=200)
                plt.close()
                paths.append(str(zoom_out))

                plt.figure(figsize=(9, 5.5))
                for algorithm, items in sorted(by_algorithm.items()):
                    threshold_histories = [
                        threshold_history_from_trace(item, delta_key)
                        for item in items
                    ]

                    x_axis, threshold_mean, threshold_ci, n = mean_ci_from_histories(threshold_histories)
                    if n == 0:
                        continue
                    x_axis, threshold_mean, threshold_ci = crop_plot_series(
                        x_axis,
                        threshold_mean,
                        threshold_ci,
                        max_evaluation=benchmark_plot_max,
                    )
                    line = plot_algorithm_series(x_axis, threshold_mean, algorithm, f"{algorithm} threshold (n={n})")
                    plot_confidence_band(
                        x_axis,
                        threshold_mean - threshold_ci,
                        threshold_mean + threshold_ci,
                        color=line.get_color(),
                        alpha=0.15,
                    )

                plt.xlabel("Evaluation")
                plt.ylabel("Mean best stoppable threshold")
                plt.title(f"{benchmark} | inverse-CDF threshold with 95% CI | δ={delta_pretty} | y in [-100, 0]")
                plt.grid(True, alpha=0.25)
                plt.xlim(0, benchmark_plot_max)
                plt.ylim(-100, 0)
                legend_sorted_by_last_y()
                zoom_out = benchmark_dir / f"inverse_cdf_threshold_delta_{delta_key}_95ci_y_-100_0.pdf"
                plt.tight_layout()
                plt.savefig(zoom_out, dpi=200)
                plt.close()
                paths.append(str(zoom_out))

            if (
                (
                    str(benchmark).lower().startswith("levy")
                    or str(benchmark).lower().startswith("rastrigin")
                )
                and delta_key == risk_label(1 / 256)
            ):
                plt.figure(figsize=(9, 5.5))
                for algorithm, items in sorted(by_algorithm.items()):
                    threshold_histories = [
                        threshold_history_from_trace(item, delta_key)
                        for item in items
                    ]

                    x_axis, threshold_mean, threshold_ci, n = mean_ci_from_histories(threshold_histories)
                    if n == 0:
                        continue
                    x_axis, threshold_mean, threshold_ci = crop_plot_series(
                        x_axis,
                        threshold_mean,
                        threshold_ci,
                        max_evaluation=benchmark_plot_max,
                    )
                    line = plot_algorithm_series(x_axis, threshold_mean, algorithm, f"{algorithm} threshold (n={n})")
                    plot_confidence_band(
                        x_axis,
                        threshold_mean - threshold_ci,
                        threshold_mean + threshold_ci,
                        color=line.get_color(),
                        alpha=0.15,
                    )

                plt.xlabel("Evaluation")
                plt.ylabel("Mean best stoppable threshold")
                plt.title(f"{benchmark} | inverse-CDF threshold with 95% CI | δ={delta_pretty} | y in [-10, 0]")
                plt.grid(True, alpha=0.25)
                plt.xlim(0, benchmark_plot_max)
                plt.ylim(-10, 0)
                legend_sorted_by_last_y()
                zoom_out = benchmark_dir / f"inverse_cdf_threshold_delta_{delta_key}_95ci_y_-10_0.pdf"
                plt.tight_layout()
                plt.savefig(zoom_out, dpi=200)
                plt.close()
                paths.append(str(zoom_out))

        def wall_clock_histories_for_items(algorithm, items):
            wall_clock_histories = []
            for item in items:
                eval_times = item.get("evaluation_wall_clock_history", [])
                q_batch_for_plot = _trace_q_batch(
                    item,
                    default=LEGACY_TRACE_BATCH_FALLBACK,
                )
                n_eval = int(item.get("n_evaluations", len(eval_times)))
                total_time = float(item.get("wall_clock_seconds", 0.0))
                if eval_times:
                    wall_clock_histories.append(
                        batch_parallel_wall_clock_history(
                            eval_times,
                            algorithm=str(item.get("algorithm", algorithm)),
                            result=item,
                            n_evaluations=n_eval,
                            wall_clock=total_time,
                            q_batch=q_batch_for_plot,
                        )
                    )
                elif n_eval > 0:
                    synthetic_times = np.linspace(total_time / n_eval, total_time, n_eval)
                    wall_clock_histories.append(
                        batch_parallel_wall_clock_history(
                            synthetic_times,
                            algorithm=str(item.get("algorithm", algorithm)),
                            result=None,
                            n_evaluations=n_eval,
                            wall_clock=total_time,
                            q_batch=q_batch_for_plot,
                        )
                    )
            return wall_clock_histories

        def save_wall_clock_plot(output_name, aliases=()):
            plt.figure(figsize=(9, 7.5) if paper_style_plot else (9, 5.5))
            plotted = False
            for algorithm, items in sorted(by_algorithm.items()):
                wall_clock_histories = wall_clock_histories_for_items(
                    algorithm, items
                )

                x_axis, wall_clock_mean, wall_clock_ci, n = mean_ci_from_histories(
                    wall_clock_histories,
                    start_at_zero=True,
                    zero_value="zero",
                )
                if n == 0:
                    continue
                x_axis, wall_clock_mean, wall_clock_ci = crop_plot_series(
                    x_axis,
                    wall_clock_mean,
                    wall_clock_ci,
                    max_evaluation=benchmark_plot_max,
                )

                plotted = True
                line = plot_algorithm_series(
                    x_axis,
                    wall_clock_mean,
                    algorithm,
                    (
                        canonical_algorithm_name(algorithm)
                        if paper_style_plot
                        else f"{algorithm} cumulative wall-clock (n={n})"
                    ),
                )
                plot_confidence_band(
                    wall_clock_mean * 0 + x_axis,
                    wall_clock_mean - wall_clock_ci,
                    wall_clock_mean + wall_clock_ci,
                    color=line.get_color(),
                    alpha=0.15,
                )

            plt.xlabel("Evaluation")
            plt.ylabel("Mean cumulative wall-clock time (s) at evaluation")
            plt.title(
                paper_style_title
                if paper_style_plot
                else f"{benchmark} | mean batch-parallel wall-clock"
            )
            plt.grid(True, alpha=0.25)
            plt.xlim(0, benchmark_plot_max)
            if plotted:
                legend_sorted_by_last_y(
                    outside_top=paper_style_plot,
                    ncol=4 if paper_style_plot else None,
                )
            out = benchmark_dir / output_name
            plt.tight_layout(
                rect=(0.0, 0.0, 1.0, 0.84)
                if paper_style_plot
                else None
            )
            plt.savefig(out, dpi=200)
            paths.append(str(out))
            for alias in aliases:
                alias_out = benchmark_dir / str(alias)
                plt.savefig(alias_out, dpi=200)
                paths.append(str(alias_out))
            plt.close()

        save_wall_clock_plot(
            "per_iteration_wall_clock_mean.pdf",
            aliases=("wall_clock_seconds_vs_iteration_all_algorithms_mean.pdf",),
        )
        (
            benchmark_dir
            / "wall_clock_milliseconds_vs_iteration_all_algorithms_mean.pdf"
        ).unlink(missing_ok=True)

        terminal_wall_clock_rows = []
        for algorithm, items in by_algorithm.items():
            mean_seconds, ci_seconds, n = terminal_mean_ci_from_histories(
                wall_clock_histories_for_items(algorithm, items)
            )
            if n > 0:
                terminal_wall_clock_rows.append(
                    (
                        float(mean_seconds),
                        str(algorithm),
                        float(ci_seconds),
                        int(n),
                    )
                )
        terminal_wall_clock_rows.sort(key=lambda row: (row[0], row[1]))
        if terminal_wall_clock_rows:
            means = np.asarray([row[0] for row in terminal_wall_clock_rows], dtype=float)
            algorithms = [row[1] for row in terminal_wall_clock_rows]
            cis = np.asarray([row[2] for row in terminal_wall_clock_rows], dtype=float)
            colors = [algorithm_plot_style(name)["color"] for name in algorithms]
            x_pos = np.arange(len(algorithms))
            plt.figure(figsize=(max(9.0, 0.9 * len(algorithms)), 5.8))
            plt.bar(
                x_pos,
                means,
                yerr=cis,
                capsize=4,
                color=colors,
                edgecolor="black",
                linewidth=0.5,
            )
            plt.xticks(x_pos, algorithms, rotation=35, ha="right")
            if real_life_benchmark:
                plt.xlabel("Algorithm (fastest to slowest)")
                plt.ylabel("Mean cumulative wall-clock time (s) at final iteration")
                plt.title(
                    paper_style_title
                    if paper_style_plot
                    else f"{benchmark} | final-iteration wall-clock ranking"
                )
            else:
                plt.xlabel("Algorithms")
                plt.ylabel("Wall-clock time (s)")
                plt.title("Algorithms vs Wall-clock")
                plt.gca().yaxis.set_major_formatter(
                    FuncFormatter(lambda value, _position: f"{value:,.0f}")
                )
            plt.grid(axis="y", alpha=0.25)
            upper = float(np.max(means + cis)) if means.size else 0.0
            plt.ylim(0.0, max(1.0, upper * 1.08))
            out = benchmark_dir / "final_iteration_wall_clock_bar_sorted.pdf"
            plt.tight_layout()
            plt.savefig(out, dpi=200)
            plt.close()
            paths.append(str(out))

        if real_life_benchmark and str(algorithm_family).lower() == "united":
            for algorithm, items in sorted(by_algorithm.items()):
                algorithm_dir = (
                    benchmark_dir / "per_algorithm" / algorithm_output_slug(algorithm)
                )
                algorithm_dir.mkdir(parents=True, exist_ok=True)

                histories = [trace_observed_history(item) for item in items]
                x_axis, mean, ci, n = mean_ci_from_histories(histories)
                if n > 0:
                    x_axis, mean, ci = crop_plot_series(
                        x_axis, mean, ci, max_evaluation=benchmark_plot_max
                    )
                    plt.figure(figsize=(9, 6.5) if paper_style_plot else (9, 5.5))
                    line = plot_algorithm_series(
                        x_axis, mean, algorithm, f"{algorithm} (n={n})"
                    )
                    plot_confidence_band(
                        x_axis, mean - ci, mean + ci,
                        color=line.get_color(), alpha=0.18,
                    )
                    plt.xlabel("Iteration / evaluation")
                    plt.ylabel(
                        paper_style_value_label
                        if paper_style_plot
                        else "Mean running-best observed value"
                    )
                    plt.title(
                        paper_style_title
                        if paper_style_plot
                        else f"{benchmark} | {algorithm} | value vs iteration"
                    )
                    plt.grid(True, alpha=0.25)
                    plt.xlim(0, benchmark_plot_max)
                    apply_nasbench101_value_y_limits(benchmark)
                    plt.legend(
                        loc="lower center" if paper_style_plot else "best",
                        bbox_to_anchor=(0.5, 1.01) if paper_style_plot else None,
                        frameon=False if paper_style_plot else True,
                    )
                    out = algorithm_dir / "value_vs_iteration_mean.pdf"
                    plt.tight_layout(
                        rect=(0.0, 0.0, 1.0, 0.90)
                        if paper_style_plot
                        else None
                    )
                    plt.savefig(out, dpi=200)
                    plt.close()
                    paths.append(str(out))

                time_axis, mean, ci, n = wall_clock_mean_ci_from_traces(
                    items,
                    trace_observed_history,
                    algorithm,
                    minimum_grid_step_milliseconds=(
                        1_000
                        if str(benchmark).lower().startswith("nasbench101")
                        else 1
                    ),
                    maximum_time_milliseconds=(
                        20_000
                        if str(benchmark).lower().startswith("nasbench101")
                        else None
                    ),
                    maximum_grid_points=(
                        20
                        if str(benchmark).lower().startswith("nasbench101")
                        else 2000
                    ),
                    exclude_initial_design=False,
                )
                if n > 0:
                    plt.figure(figsize=(9, 6.5) if paper_style_plot else (9, 5.5))
                    plot_time_axis = time_axis / 1000.0
                    line = plot_algorithm_series(
                        plot_time_axis, mean, algorithm, f"{algorithm} (n={n})"
                    )
                    plot_confidence_band(
                        plot_time_axis, mean - ci, mean + ci,
                        color=line.get_color(), alpha=0.18,
                    )
                    plt.xlabel("Wall-clock time (s)")
                    plt.ylabel(
                        paper_style_value_label
                        if paper_style_plot
                        else "Mean running-best observed value"
                    )
                    plt.title(
                        paper_style_title
                        if paper_style_plot
                        else f"{benchmark} | {algorithm} | value vs time"
                    )
                    plt.grid(True, alpha=0.25)
                    if str(benchmark).lower().startswith("nasbench101"):
                        plt.xlim(0.0, 20.0)
                        apply_nasbench101_value_y_limits(benchmark)
                        plt.gca().xaxis.set_major_locator(MultipleLocator(1.0))
                    else:
                        plt.xlim(left=0.0)
                    plt.legend(
                        loc="lower center" if paper_style_plot else "best",
                        bbox_to_anchor=(0.5, 1.01) if paper_style_plot else None,
                        frameon=False if paper_style_plot else True,
                    )
                    out = algorithm_dir / "value_vs_time_seconds_mean.pdf"
                    plt.tight_layout(
                        rect=(0.0, 0.0, 1.0, 0.90)
                        if paper_style_plot
                        else None
                    )
                    plt.savefig(out, dpi=200)
                    plt.close()
                    paths.append(str(out))
                    (algorithm_dir / "value_vs_time_milliseconds_mean.pdf").unlink(
                        missing_ok=True
                    )

                wall_histories = wall_clock_histories_for_items(algorithm, items)
                x_axis, mean, ci, n = mean_ci_from_histories(
                    wall_histories, start_at_zero=True, zero_value="zero"
                )
                if n > 0:
                    x_axis, mean, ci = crop_plot_series(
                        x_axis, mean, ci, max_evaluation=benchmark_plot_max
                    )
                    plt.figure(figsize=(9, 6.5) if paper_style_plot else (9, 5.5))
                    line = plot_algorithm_series(
                        x_axis, mean, algorithm, f"{algorithm} (n={n})"
                    )
                    plot_confidence_band(
                        x_axis, mean - ci, mean + ci,
                        color=line.get_color(), alpha=0.18,
                    )
                    plt.xlabel("Iteration / evaluation")
                    plt.ylabel("Mean cumulative wall-clock time (s)")
                    plt.title(
                        paper_style_title
                        if paper_style_plot
                        else f"{benchmark} | {algorithm} | wall-clock vs iteration"
                    )
                    plt.grid(True, alpha=0.25)
                    plt.xlim(0, benchmark_plot_max)
                    plt.legend(
                        loc="lower center" if paper_style_plot else "best",
                        bbox_to_anchor=(0.5, 1.01) if paper_style_plot else None,
                        frameon=False if paper_style_plot else True,
                    )
                    out = algorithm_dir / "wall_clock_seconds_vs_iteration_mean.pdf"
                    plt.tight_layout(
                        rect=(0.0, 0.0, 1.0, 0.90)
                        if paper_style_plot
                        else None
                    )
                    plt.savefig(out, dpi=200)
                    plt.close()
                    paths.append(str(out))
                    (
                        algorithm_dir / "wall_clock_milliseconds_vs_iteration_mean.pdf"
                    ).unlink(missing_ok=True)

        for agvs_algorithm in (
            "AG-VS-BO",
            "AG-VS-TuRBO",
        ):
            agvs_items = by_algorithm.get(agvs_algorithm, [])
            agvs_count_rows = []
            for item in agvs_items:
                counts = item.get("agvs_dimension_selection_counts")
                if counts is None:
                    counts = agvs_dimension_selection_counts_from_history(item.get("agvs_history"), benchmark_dim)
                row = np.asarray(counts, dtype=float)
                if row.size == benchmark_dim:
                    agvs_count_rows.append(row)

            if agvs_count_rows:
                dim_select_dir = benchmark_dir / f"{algorithm_output_slug(agvs_algorithm)}-dim-select"
                dim_select_dir.mkdir(parents=True, exist_ok=True)
                counts = np.mean(np.vstack(agvs_count_rows), axis=0)
                agvs_experiment_mode = trace_experiment_mode(agvs_items[0])
                if agvs_experiment_mode:
                    trace_active = agvs_items[0].get("settings", {}).get("active_indices")
                    if trace_active is not None:
                        active_indices = set(int(dim) for dim in trace_active)
                    elif benchmark in BENCHMARKS:
                        active_indices = set(int(dim) for dim in BENCHMARKS[benchmark]["active_indices"])
                    else:
                        active_indices = set()
                else:
                    active_indices = set()
                order = sorted(range(benchmark_dim), key=lambda dim: (-counts[dim], dim))
                sorted_counts = counts[order]
                colors = ["#2ca02c" if agvs_experiment_mode and dim in active_indices else "#1f77b4" for dim in order]
                x_pos = np.arange(benchmark_dim)

                plt.figure(figsize=(10.5, 4.8))
                plt.bar(x_pos, sorted_counts, color=colors, width=0.9)
                tick_step = max(1, benchmark_dim // 30)
                tick_positions = x_pos[::tick_step]
                tick_labels = [str(order[int(pos)]) for pos in tick_positions]
                plt.xticks(tick_positions, tick_labels, rotation=90, fontsize=7)
                plt.xlabel("Dimension index")
                plt.ylabel("Selection Count")
                plt.title(paper_style_title if paper_style_plot else benchmark)
                plt.grid(True, axis="y", alpha=0.25)
                if agvs_experiment_mode:
                    active_patch = plt.Rectangle((0, 0), 1, 1, color="#2ca02c", label="Active")
                    inactive_patch = plt.Rectangle((0, 0), 1, 1, color="#1f77b4", label="Inactive")
                    plt.legend(handles=[active_patch, inactive_patch])
                out_name = f"{algorithm_output_slug(agvs_algorithm)}_dimension_selection_counts_sorted.pdf"
                out = dim_select_dir / out_name
                plt.tight_layout()
                plt.savefig(out, dpi=200)
                plt.close()
                paths.append(str(out))

                if not PLOT_TERMINAL_DIAGNOSTICS:
                    for stale_path in dim_select_dir.glob(
                        f"{algorithm_output_slug(agvs_algorithm)}_final_terminal_*.pdf"
                    ):
                        stale_path.unlink(missing_ok=True)
                    continue

                terminal_trial_counts = np.zeros(benchmark_dim, dtype=int)
                for item in agvs_items:
                    for dim in agvs_final_terminal_leaf_dims_from_trace(item, benchmark_dim):
                        terminal_trial_counts[int(dim)] += 1

                terminal_order = sorted(
                    range(benchmark_dim),
                    key=lambda dim: (-terminal_trial_counts[dim], dim),
                )
                terminal_sorted = terminal_trial_counts[terminal_order]
                terminal_colors = [
                    "#2ca02c" if agvs_experiment_mode and dim in active_indices else "#1f77b4"
                    for dim in terminal_order
                ]
                x_pos = np.arange(benchmark_dim)
                plt.figure(figsize=(15, 5.5))
                plt.bar(x_pos, terminal_sorted, color=terminal_colors, width=0.9)
                tick_step = max(1, benchmark_dim // 30)
                tick_positions = x_pos[::tick_step]
                tick_labels = [str(terminal_order[int(pos)]) for pos in tick_positions]
                plt.xticks(tick_positions, tick_labels, rotation=90, fontsize=7)
                plt.xlabel("Dimension index, sorted by final terminal-leaf trial count descending")
                plt.ylabel(f"Trials containing dimension in final terminal leaf set (out of {len(agvs_items)})")
                plt.title(
                    f"{benchmark} | {agvs_algorithm} final terminal-leaf dimension frequency "
                    f"across {len(agvs_items)} trial(s)"
                )
                plt.ylim(0, max(len(agvs_items), 1))
                plt.grid(True, axis="y", alpha=0.25)
                if agvs_experiment_mode:
                    active_patch = plt.Rectangle((0, 0), 1, 1, color="#2ca02c", label="true active dimension")
                    inactive_patch = plt.Rectangle((0, 0), 1, 1, color="#1f77b4", label="inactive dimension")
                    plt.legend(handles=[active_patch, inactive_patch])
                terminal_out = benchmark_dir / (
                    f"{algorithm_output_slug(agvs_algorithm)}_final_terminal_leaf_dimension_trial_counts_sorted.pdf"
                )
                terminal_out = dim_select_dir / terminal_out.name
                plt.tight_layout()
                plt.savefig(terminal_out, dpi=200)
                plt.close()
                paths.append(str(terminal_out))

    if update_objective_tables:
        paths.extend(
            update_aggregate_objective_tables(
                grouped,
                benchmark_root=results_dir,
                table_dir=table_dir,
            )
        )
    return paths


def discover_benchmark_scripts():
    unified_entrypoint = Path(__file__).resolve().parents[1] / "utils.py"
    scripts = {
        name: unified_entrypoint
        for name in SYNTHETIC_BENCHMARKS
    }
    scripts.update(
        {name: unified_entrypoint for name in REAL_LIFE_EXPERIMENTS}
    )
    return scripts


def benchmark_trial_rounds(benchmarks, trials=None):
    """Visit every synthetic function before advancing to its next 25 trials."""
    totals = {
        name: int(trials) if trials is not None else int(
            BENCHMARKS[name].get("trials", TRIALS)
            if name in SYNTHETIC_EXPERIMENTS else REAL_LIFE_EXPERIMENTS[name].trials
        )
        for name in benchmarks
    }
    synthetic_total = max(
        (totals[name] for name in benchmarks if name in SYNTHETIC_EXPERIMENTS),
        default=1,
    )
    for start in range(0, synthetic_total, SYNTHETIC_TRIAL_BATCH_SIZE):
        batch = []
        for name in benchmarks:
            if name in SYNTHETIC_EXPERIMENTS:
                if start < totals[name]:
                    batch.append((name, start, min(start + SYNTHETIC_TRIAL_BATCH_SIZE, totals[name])))
            elif start == 0:
                batch.append((name, 0, totals[name]))
        yield batch


def refresh_requested_reports(benchmarks, skip_plot=False):
    """Refresh cumulative reports at a barrier between trial batches."""
    kinds = {
        "real_life" if name in REAL_LIFE_EXPERIMENTS else "synthetic"
        for name in benchmarks
    }
    if not skip_plot:
        for name in benchmarks:
            kind = "real_life" if name in REAL_LIFE_EXPERIMENTS else "synthetic"
            figure_root, table_root = output_roots_for_experiment_kind(kind)
            plot_results(
                result_root_for_experiment_kind(kind), figure_root,
                benchmark_filter=name, update_objective_tables=False,
                table_dir=table_root,
            )
    for kind in sorted(kinds):
        result_root = result_root_for_experiment_kind(kind)
        figure_root, table_root = output_roots_for_experiment_kind(kind)
        if not skip_plot and kind == "synthetic":
            for path in save_synthetic_comparison_pdfs(
                benchmark_root=result_root, figure_root=figure_root,
                benchmark_names=[name for name in benchmarks if name in SYNTHETIC_EXPERIMENTS],
            ):
                print(path, flush=True)
        print(f"\n=== Updating {kind} aggregate benchmark objective table ===", flush=True)
        for path in save_all_benchmark_objective_tables(
            benchmark_root=result_root, table_dir=table_root,
        ):
            print(path, flush=True)
        if not skip_plot:
            remove_legacy_millisecond_figure_aliases(figure_root)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run AG-VS benchmark scripts.")
    parser.add_argument(
        "--resume-under-budget",
        action="store_true",
        help=(
            "Replay compatible completed traces below the requested budget and "
            "evaluate only their missing suffix."
        ),
    )
    parser.add_argument(
        "--benchmarks",
        nargs="*",
        default=None,
        help="Benchmark script names to run. Default: runner.DEFAULT_BENCHMARKS.",
    )
    parser.add_argument(
        "--all-benchmarks",
        action="store_true",
        help="Run every non-deferred experiment in the central registry.",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=None,
        help="Override trial count; defaults to each benchmark script's configured value.",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=None,
        help=(
            "Override the evaluation budget for every requested benchmark. "
            "Defaults to the exact per-identity budget in the central registry."
        ),
    )
    parser.add_argument(
        "--algorithms",
        nargs="*",
        default=None,
        help=(
            "Explicit algorithms for every requested benchmark. By default, synthetic "
            "tasks use the locally runnable DEFAULT_ALGORITHMS and each explicitly "
            "requested real task uses its locally runnable applicable subset."
        ),
    )
    parser.add_argument("--seed", type=int, choices=(0,), default=0)
    parser.add_argument("--skip-plot", action="store_true")
    parser.add_argument(
        "--plots-only",
        action="store_true",
        help="Regenerate bo/turbo/united figures and tables from existing JSON traces without running trials.",
    )
    parser.add_argument("--rerun-complete", action="store_true")
    parser.add_argument(
        "--reuse-complete-json",
        action="store_true",
        help=(
            "Reuse error-free, budget-complete imported JSON files after "
            "validating benchmark, algorithm, trial, seed bundle, and device."
        ),
    )
    parser.add_argument(
        "--workers",
        type=positive_int,
        default=WORKERS,
        help=f"Worker count (runner SETTINGS default: {WORKERS}).",
    )
    parser.add_argument(
        "--device",
        choices=(DEVICE,),
        default=DEVICE,
        help=f"Fixed at the runner SETTINGS value ({DEVICE}) for fair comparison.",
    )
    add_n_bad_cli_flags(parser)
    parser.add_argument("--list", action="store_true", help="List discovered benchmark scripts and exit.")
    parser.add_argument(
        "--list-algorithms",
        action="store_true",
        help="List registered algorithm names and exit.",
    )
    parser.add_argument(
        "--agvs-live-tree",
        action="store_true",
        help="Enable AG-VS live tree PNG/HTML generation in child benchmark scripts.",
    )
    parser.add_argument(
        "--agvs-open-tree",
        action="store_true",
        help="Open the AG-VS live tree HTML in a browser from child benchmark scripts.",
    )
    parser.add_argument(
        "--agvs-python-live-plot",
        action="store_true",
        help="Use an interactive Matplotlib backend for the live tree instead of only writing HTML/PNG.",
    )
    args = parser.parse_args(argv)
    args.device = normalize_device(args.device)

    scripts = discover_benchmark_scripts()
    if args.list:
        for name in sorted(scripts):
            print(name)
        return
    if args.list_algorithms:
        for algorithm in RUNNER_ALGORITHM_CHOICES:
            print(algorithm)
        return

    # Plot-only with no explicit filter means every discoverable active result
    # set. This branch returns before any child process or optimizer task can be
    # created.
    if args.plots_only and args.benchmarks is None:
        requested = sorted(
            scripts,
            key=lambda name: (
                0 if name in SYNTHETIC_EXPERIMENTS else 1,
                name,
            ),
        )
    elif args.all_benchmarks:
        requested = sorted(
            (name for name in scripts if name not in DEFERRED_BENCHMARKS),
            key=lambda name: (
                0 if name in REAL_LIFE_EXPERIMENTS else 1,
                name,
            ),
        )
    elif args.benchmarks is not None:
        requested = [str(name).lower() for name in args.benchmarks]
    elif RUNNER_BENCHMARKS is not None:
        requested = [str(name).lower() for name in RUNNER_BENCHMARKS]
    else:
        requested = [str(name).lower() for name in DEFAULT_BENCHMARKS]

    if not requested:
        raise ValueError("At least one benchmark must be selected")

    missing = [name for name in requested if name not in scripts]
    if missing:
        raise ValueError(f"Unknown benchmark script(s): {missing}. Available: {sorted(scripts)}")

    configured_algorithms = (
        args.algorithms
        if args.algorithms is not None
        else RUNNER_ALGORITHMS
    )
    explicit_algorithms = (
        None
        if configured_algorithms is None
        else normalize_algorithm_args(configured_algorithms)
    )
    algorithms_by_benchmark = {}
    for name in requested:
        if name in REAL_LIFE_EXPERIMENTS:
            experiment = REAL_LIFE_EXPERIMENTS[name]
            if explicit_algorithms is None:
                selected = list(iter_applicable_real_life_algorithms(experiment))
            else:
                selected = list(explicit_algorithms)
                invalid = [
                    algorithm
                    for algorithm in selected
                    if not real_life_algorithm_is_applicable(experiment, algorithm)
                ]
                if invalid:
                    raise ValueError(
                        f"Algorithms {invalid} are not applicable to reference real-life "
                        f"condition {name}; applicable algorithms="
                        f"{iter_applicable_real_life_algorithms(experiment)}"
                    )
        else:
            selected = list(
                DEFAULT_ALGORITHMS
                if explicit_algorithms is None
                else explicit_algorithms
            )
        algorithms_by_benchmark[name] = selected

    empty_algorithm_benchmarks = [
        name for name, selected in algorithms_by_benchmark.items() if not selected
    ]
    if empty_algorithm_benchmarks:
        raise ValueError(
            "At least one algorithm must be selected for every benchmark; empty for "
            f"{empty_algorithm_benchmarks}"
        )

    if args.plots_only:
        if args.skip_plot:
            raise ValueError("--plots-only and --skip-plot cannot be used together")
        requested_kinds = set()
        for name in requested:
            kind = "real_life" if name in REAL_LIFE_EXPERIMENTS else "synthetic"
            requested_kinds.add(kind)
            result_root = result_root_for_experiment_kind(kind)
            figure_root, table_root = output_roots_for_experiment_kind(kind)
            plot_results(
                result_root,
                figure_root,
                benchmark_filter=name,
                update_objective_tables=False,
                table_dir=table_root,
            )
        for kind in sorted(requested_kinds):
            result_root = result_root_for_experiment_kind(kind)
            figure_root, table_root = output_roots_for_experiment_kind(kind)
            if kind == "synthetic":
                for output_path in save_synthetic_comparison_pdfs(
                    benchmark_root=result_root,
                    figure_root=figure_root,
                    benchmark_names=[
                        name for name in requested if name in SYNTHETIC_EXPERIMENTS
                    ],
                ):
                    print(output_path, flush=True)
            for output_path in save_all_benchmark_objective_tables(
                benchmark_root=result_root,
                table_dir=table_root,
            ):
                print(output_path, flush=True)
            remove_legacy_millisecond_figure_aliases(figure_root)
        return

    # Each benchmark child performs a result-aware preflight. This allows a
    # higher-d_eff children to reuse complete dimension-invariant deff3 traces without
    # requiring an optimizer runtime that will not be launched.

    base_cmd = [
        sys.executable,
        "",
        "--seed",
        str(int(args.seed)),
        "--device",
        args.device,
        "--workers",
        str(max(1, int(args.workers))),
    ]
    if args.trials is not None:
        base_cmd.extend(["--trials", str(int(args.trials))])
    if args.skip_plot:
        base_cmd.append("--skip-plot")
    if args.rerun_complete:
        base_cmd.append("--rerun-complete")
    if args.reuse_complete_json:
        base_cmd.append("--reuse-complete-json")
    if args.resume_under_budget:
        base_cmd.append("--resume-under-budget")
    if args.n_bad_enabled is True:
        base_cmd.append("--n-bad-enabled")
    elif args.n_bad_enabled is False:
        base_cmd.append("--no-n-bad-enabled")
    if args.agvs_live_tree:
        os.environ["AGVS_VERBOSE"] = "1"
    if args.agvs_open_tree:
        os.environ["AGVS_VERBOSE"] = "1"
        os.environ["AGVS_TREE_OPEN_BROWSER"] = "1"
    if args.agvs_python_live_plot:
        os.environ["AGVS_VERBOSE"] = "1"
        os.environ["AGVS_PYTHON_LIVE_PLOT_ON"] = "1"

    print("Runner will execute benchmark scripts in order:")
    for name in requested:
        print(f"  - {name}: {scripts[name]}")
    if explicit_algorithms is None:
        print("Algorithms: per-benchmark locally runnable defaults")
    else:
        print(f"Algorithms: {explicit_algorithms}")
    print(
        f"Trials: {int(args.trials)} (explicit override)"
        if args.trials is not None
        else "Trials: per-benchmark defaults"
    )
    if args.max_iter is None:
        print("Max evaluations per run: exact per-identity registry budget")
    else:
        print(f"Max evaluations per run (explicit override): {int(args.max_iter)}")
    print(f"Execution resources: device={args.device}, workers={int(args.workers)}")
    print(
        "N_bad reinitialization: "
        + (
            "enabled (CLI override)"
            if args.n_bad_enabled is True
            else "disabled (CLI override)"
            if args.n_bad_enabled is False
            else "runner family default"
        )
    )

    child_env = os.environ.copy()
    child_env["AGVS_WORKERS"] = str(max(1, int(args.workers)))
    child_env["AGVS_DEVICE"] = args.device
    inherited_pythonpath = child_env.get("PYTHONPATH", "").strip()
    child_env["PYTHONPATH"] = (
        os.pathsep.join((str(ROOT), inherited_pythonpath))
        if inherited_pythonpath
        else str(ROOT)
    )
    try:
        for batch in benchmark_trial_rounds(requested, args.trials):
            for name, trial_start, trial_end in batch:
                cmd = list(base_cmd)
                if "--trials" in cmd:
                    trial_flag = cmd.index("--trials")
                    del cmd[trial_flag:trial_flag + 2]
                cmd.extend(["--trials", str(trial_end)])
                if name in SYNTHETIC_EXPERIMENTS:
                    cmd.extend(["--trial-start", str(trial_start), "--defer-reports"])
                    print(f"[trial-batch] {name}: trials {trial_start + 1}-{trial_end}", flush=True)
                cmd[1] = str(scripts[name])
                cmd.extend(["--algorithms", *algorithms_by_benchmark[name]])
                if name in SYNTHETIC_EXPERIMENTS:
                    # Keep nargs='*' for --algorithms bounded by the selector.
                    cmd.extend(["--benchmark", name])
                if args.max_iter is not None:
                    child_max_iter = int(args.max_iter)
                elif name in REAL_LIFE_BENCHMARK_DEFAULTS:
                    child_max_iter = int(REAL_LIFE_BENCHMARK_DEFAULTS[name]["max_iter"])
                else:
                    child_max_iter = int(SYNTHETIC_EXPERIMENTS[name].budget)
                cmd.extend(["--max-iter", str(child_max_iter)])
                if name in REAL_LIFE_EXPERIMENTS:
                    # Legacy real-life commands bounded --algorithms with
                    # --max-iter. Retain that stable ordering for callers/tests.
                    cmd.extend(["--benchmark", name])
                print(f"\n=== Running {name} ===", flush=True)
                print(f"Algorithms: {algorithms_by_benchmark[name]}", flush=True)
                print(f"Max evaluations: {child_max_iter}", flush=True)
                subprocess.run(cmd, check=True, env=child_env, cwd=str(ROOT))
            refresh_requested_reports(
                [name for name, _start, _end in batch], skip_plot=args.skip_plot
            )
    except BaseException:
        # Preserve reports from completed trials even if a child fails or is interrupted.
        refresh_requested_reports(requested, skip_plot=args.skip_plot)
        raise


if __name__ == "__main__":
    main()
