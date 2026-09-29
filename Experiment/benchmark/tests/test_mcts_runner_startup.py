"""Fresh-process checks: pytest's already-loaded modules can conceal import cycles."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[3]


def run_fresh_python(script, *args):
    environment = dict(os.environ, PYTHONWARNINGS="ignore", OMP_NUM_THREADS="1",
                       MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    result = subprocess.run([sys.executable, "-c", script, *map(str, args)],
                            cwd=ROOT, env=environment, capture_output=True,
                            text=True, timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr
    return result


@pytest.mark.parametrize("first_import", [
    "from Experiment.benchmark.baseline import reference_infrastructure",
    "from Experiment.benchmark.baseline import native_optimizers",
    "from ag_vs import AGVSOptimizer",
])
def test_mcts_available_regardless_of_first_import(first_import):
    run_fresh_python(first_import + "\n" + """
from Experiment.benchmark import runner
assert runner.MCTSVSOptimizer is not None, runner.MCTSVS_IMPORT_ERROR
assert runner.MCTSVS_IMPORT_ERROR is None
""")


def test_randomness_import_does_not_eagerly_load_optimizer():
    run_fresh_python("""
import sys
import ag_vs.utils
assert 'ag_vs.optimizer' not in sys.modules
from ag_vs import AGVSOptimizer
from ag_vs.optimizer import AGVSOptimizer as direct
assert AGVSOptimizer is direct
""")


def test_fresh_runner_saves_both_mcts_variants_beyond_initialization(tmp_path):
    run_fresh_python("""
import sys
import torch
from Experiment.benchmark import runner
torch.set_num_threads(1)
assert runner.MCTSVSOptimizer is not None, runner.MCTSVS_IMPORT_ERROR
for algorithm in ('MCTS-VS-BO', 'MCTS-VS-TuRBO'):
    trace = runner.run_single_trial('levy3_300', algorithm, 0, 13, sys.argv[1], 0,
        device='cpu', skip_complete=False, show_final_tree=False)
    assert trace['complete'], trace['error']
    assert trace['n_evaluations'] == 13
    assert trace['algorithm_seed'] == trace['benchmark_seed'] == 2021
    path = runner.trace_path_for(sys.argv[1], 'levy3_300', algorithm, 0)
    assert path.is_file(), path
    assert runner.load_existing_trace_if_complete(sys.argv[1], 'levy3_300',
        algorithm, 0, max_iter=13, cache_identity=trace['cache_identity']) is not None
""", tmp_path)


@pytest.mark.parametrize("existing_trace", [False, True])
def test_worker_persists_and_prints_startup_failure(monkeypatch, tmp_path, capsys, existing_trace):
    from Experiment.benchmark import runner
    path = runner.trace_path_for(tmp_path, 'levy3_300', 'MCTS-VS-BO', 0)
    original = {'complete': True, 'observed_y_history': [1., 2.]}
    if existing_trace:
        runner.write_json(path, original)
    def fail(**kwargs):
        raise ImportError('test startup failure')
    monkeypatch.setattr(runner, 'run_single_trial', fail)
    statuses = runner._run_round_robin_trial((0, 1, ['levy3_300'], ['MCTS-VS-BO'],
        13, str(tmp_path), 0, False, str(tmp_path/'figures'), 'cpu'))
    assert statuses[0]['complete'] is False
    failure_path = path.with_suffix('.failure.json') if existing_trace else path
    failure = json.loads(failure_path.read_text(encoding='utf-8'))
    assert failure['error'] == 'ImportError: test startup failure'
    assert 'raise ImportError' in failure['traceback']
    assert failure['complete'] is False
    assert '[TRIAL-FAILED]' in capsys.readouterr().out
    if existing_trace:
        assert json.loads(path.read_text(encoding='utf-8')) == original
