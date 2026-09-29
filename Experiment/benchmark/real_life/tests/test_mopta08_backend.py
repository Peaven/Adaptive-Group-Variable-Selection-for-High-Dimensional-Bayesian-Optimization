from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from Experiment.benchmark.real_life import mopta08_backend


def test_mopta08_soft_constraint_value_uses_runner_maximization():
    output = np.zeros(69, dtype=float)
    output[0] = 100.0
    output[1:5] = [2.0, -5.0, 3.0, 0.0]
    assert mopta08_backend.penalized_mopta08_value(output) == pytest.approx(-150.0)


def test_mopta08_objective_writes_124_values_and_reads_69_outputs(
    tmp_path, monkeypatch
):
    executable = tmp_path / "mopta08.exe"
    executable.touch()

    def fake_run(command, *, cwd, **kwargs):
        assert command == [str(executable)]
        candidate = (Path(cwd) / "input.txt").read_text(encoding="ascii").splitlines()
        assert len(candidate) == 124
        assert all(float(value) == pytest.approx(0.5) for value in candidate)
        values = [300.0, 0.2, -0.5] + [0.0] * 66
        (Path(cwd) / "output.txt").write_text(
            "\n".join(str(value) for value in values) + "\n",
            encoding="ascii",
        )
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(mopta08_backend.subprocess, "run", fake_run)
    objective = mopta08_backend.Mopta08Objective(executable)
    try:
        assert objective(np.full(124, 0.5)) == pytest.approx(-302.0)
    finally:
        objective.close()
