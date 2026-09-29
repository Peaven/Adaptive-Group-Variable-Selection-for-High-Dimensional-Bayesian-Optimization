"""Keep runner tests independent of a developer's saved experiment data."""
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def isolated_runner_output_roots(monkeypatch, tmp_path):
    from Experiment.benchmark import runner

    experiment_root = Path(runner.ROOT) / "Experiment"
    data_roots = (experiment_root / "Result", experiment_root / "tables")
    for name, value in vars(runner).copy().items():
        if isinstance(value, Path) and any(value.is_relative_to(root) for root in data_roots):
            monkeypatch.setattr(runner, name, tmp_path / value.relative_to(experiment_root))
    monkeypatch.setattr(runner, "BENCHMARKS", dict(runner.BENCHMARKS))
