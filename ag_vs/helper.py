"""Unified public helpers and command entry points for the AG-VS project.

This module centralizes the user-facing functions that were previously spread
across the top-level runner, synthetic/real-life entry points, registry/table
helpers, and plot scripts. The original scientific implementation modules stay
small and importable; their command files now delegate here.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys
import time

from Experiment.benchmark import runner_synthetic as runner
from Experiment.benchmark import mopta
from Experiment.utils import (
    synthetic_configuration_rows,
    write_synthetic_configuration_table,
    write_synthetic_configuration_tables,
)
from Experiment.utils import (
    REAL_LIFE_ALGORITHMS,
    REAL_LIFE_EXPERIMENTS,
    SYNTHETIC_ALGORITHMS,
    SYNTHETIC_EXPERIMENTS,
    get_experiment,
    get_real_life_experiment,
    get_synthetic_experiment,
    experiment_entrypoint,
)
from Experiment.utils import (
    make_all_synthetic_configs,
    make_synthetic_config,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

USAGE = """\
Usage:
  python -m ag_vs.helper synthetic [runner options]
  python -m ag_vs.helper real [real-life runner options]
  python -m ag_vs.helper plot [--benchmarks NAME ...]
  python -m ag_vs.helper nbad [grid options]
  python -m ag_vs.helper tseq [grid options]
  python -m ag_vs.helper config-table
  python -m ag_vs.helper migrate-cache
  python -m ag_vs.helper migrate-synthetic-seeds
  python -m ag_vs.helper list
"""


_SYNTHETIC_TRIAL_PATTERN = re.compile(r'"trial_id"\s*:\s*(\d+)')
_SYNTHETIC_TRIAL_FILENAME_PATTERN = re.compile(r"^trial_(\d+)_", re.IGNORECASE)
_SYNTHETIC_SEED_KEYS = (
    "algorithm_seed",
    "benchmark_seed",
    "seed",
    "objective_noise_seed",
    "initial_design_seed",
    "task_seed",
)
_SYNTHETIC_SEED_VALUE_PATTERN = re.compile(
    r'("(?:' + "|".join(map(re.escape, _SYNTHETIC_SEED_KEYS)) + r')"\s*:\s*)-?\d+'
)


def migrate_synthetic_json_seeds(*, dry_run: bool = False):
    """Historical trajectories cannot acquire new seeds by editing metadata."""
    raise ValueError("Seed migration is disabled: rerun with seeds 2021?2070; historical traces remain unchanged.")


def synthetic_entrypoint(argv: list[str] | None = None) -> None:
    experiment_entrypoint(argv)


def real_life_entrypoint(argv: list[str] | None = None) -> None:
    experiment_entrypoint(argv)


def plot_all_results() -> None:
    """Regenerate project figures/tables from JSON without optimization."""

    runner.main(["--plots-only"])








def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments or arguments[0] in {"-h", "--help", "help"}:
        print(USAGE)
        return 0

    command, *remaining = arguments
    command = command.strip().lower()
    if command == "synthetic":
        runner.main(["--benchmarks", *runner.SYNTHETIC_BENCHMARKS, *remaining])
        return 0
    if command in {"real", "real-life", "real_life"}:
        return int(mopta.main(remaining))
    if command in {"plot", "plots"}:
        runner.main(["--plots-only", *remaining])
        return 0
    if command == "nbad":
        from Experiment.grid_search_nbad import main as nbad_main

        return int(nbad_main(remaining))
    if command in {"tseq", "grid-tseq", "grid_tseq"}:
        from Experiment.grid_search_tseq import main as grid_tseq_main

        return int(grid_tseq_main(remaining))
    if command == "config-table":
        if remaining:
            raise SystemExit("config-table accepts no extra arguments")
        for output in write_synthetic_configuration_tables():
            print(output)
        return 0
    if command == "migrate-cache":
        if remaining:
            raise SystemExit("migrate-cache accepts no extra arguments")
        stats = runner.migrate_legacy_n_bad_cache_identities()
        print("N_bad cache migration: " + ", ".join(f"{k}={v}" for k, v in stats.items()))
        return 0
    if command == "migrate-synthetic-seeds":
        if remaining:
            raise SystemExit("migrate-synthetic-seeds accepts no extra arguments")
        stats = migrate_synthetic_json_seeds()
        print(
            "Synthetic seed migration: "
            + ", ".join(f"{key}={value}" for key, value in stats.items())
        )
        return 0
    if command == "list":
        runner.main(["--list"])
        return 0
    raise SystemExit(f"Unknown command {command!r}.\n\n{USAGE}")


__all__ = [
    "REAL_LIFE_ALGORITHMS",
    "REAL_LIFE_EXPERIMENTS",
    "SYNTHETIC_ALGORITHMS",
    "SYNTHETIC_EXPERIMENTS",
    "get_experiment",
    "get_real_life_experiment",
    "get_synthetic_experiment",
    "make_all_synthetic_configs",
    "make_synthetic_config",
    "migrate_synthetic_json_seeds",
    "plot_all_results",
    "real_life_entrypoint",
    "runner",
    "synthetic_configuration_rows",
    "synthetic_entrypoint",
    "write_synthetic_configuration_table",
    "write_synthetic_configuration_tables",
]


if __name__ == "__main__":
    raise SystemExit(main())
