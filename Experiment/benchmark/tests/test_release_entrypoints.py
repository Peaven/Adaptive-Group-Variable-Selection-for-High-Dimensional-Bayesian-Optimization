"""Entry-point/layout regressions: these never launch the full studies."""
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from Experiment.benchmark import runner
from Experiment.utils import make_synthetic_config

ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize("script", [
    "Experiment/benchmark/mopta.py",
    "Experiment/benchmark/runner_synthetic.py",
    "Experiment/grid_search_nbad.py",
    "Experiment/grid_search_tseq.py",
    "Experiment/version_comparison.py",
])
def test_script_entrypoints_work_from_another_directory(script, tmp_path):
    result = subprocess.run(
        [sys.executable, str(ROOT / script), "--list"],
        cwd=tmp_path, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Traceback" not in result.stderr


@pytest.mark.parametrize("name", runner.SYNTHETIC_BENCHMARKS)
def test_every_active_synthetic_objective_is_runnable(name):
    config = make_synthetic_config(name)
    bounds = config["bounds"]
    for fraction in (0.0, 0.25, 0.5, 1.0):
        point = bounds[:, 0] + fraction * (bounds[:, 1] - bounds[:, 0])
        assert np.isfinite(config["func"](point))


def test_shared_worker_entrypoint_moved_with_registry():
    assert {p.name for p in runner.discover_benchmark_scripts().values()} == {"utils.py"}
    from ag_vs.utils import cell_identity, seeded_trial
    from Experiment import grid_search_nbad, grid_search_tseq, version_comparison
    assert callable(cell_identity) and callable(seeded_trial)
    assert grid_search_nbad.ROOT == ROOT
    assert grid_search_tseq.ROOT == ROOT
    assert version_comparison.ROOT == ROOT


def test_smoke_command_help_works_without_loading_real_backend():
    result = subprocess.run(
        [sys.executable, "-m", "Experiment.utils", "smoke", "--help"],
        cwd=ROOT, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "mopta" in result.stdout
