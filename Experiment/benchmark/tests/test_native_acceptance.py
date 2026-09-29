"""Numerical and integration contracts for native-v1 benchmarks."""
import random
import os
from collections import Counter
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from ag_vs.utils import seeded_trial, sobol_engine, jax_key
from Experiment import grid_search_nbad, grid_search_tseq
from Experiment.benchmark import runner
from Experiment.benchmark.baseline.embeddings import HeSBOOptimizer, ALEBOOptimizer
from Experiment.benchmark.baseline.gtbo import GTBOOptimizer, GTBO_SETTINGS
from Experiment.benchmark.baseline.native.alebo_model import ALEBOKernel
from Experiment.benchmark.tests.test_optimizer_fidelity import _agvs
from Experiment.benchmark.tests.test_nbad_grid_search import _preserve_grid_mutated_runner_state


def bounds(d=6):
    return np.tile([-1., 1.], (d, 1))


def sphere(x):
    return -float(np.square(x).sum())


def _repeatable_hesbo(seed):
    import warnings
    torch.set_num_threads(1)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        optimizer = HeSBOOptimizer(sphere, bounds(), 4, seed, 2,
                                   raw_samples=32, num_restarts=2).optimize()
    return optimizer.X


def test_native_trajectory_is_independent_of_process_schedule():
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context
    expected = {seed: _repeatable_hesbo(seed) for seed in (2021, 2022)}
    with ProcessPoolExecutor(max_workers=2, mp_context=get_context('spawn')) as pool:
        actual = list(pool.map(_repeatable_hesbo, (2022, 2021)))
    for seed, points in zip((2022, 2021), actual):
        np.testing.assert_allclose(points, expected[seed], rtol=0, atol=1e-10)
    assert not np.allclose(expected[2021], expected[2022])


def test_projection_reference_equations_and_feasible_initialization():
    hesbo = HeSBOOptimizer(sphere, bounds(17), 3, 2021, 3)
    with seeded_trial(2021):
        hesbo._initialize_projection()
    with seeded_trial(2021):
        hashes = torch.randint(3, (17,))
        signs = 2 * torch.randint(2, (17,)) - 1
    low = torch.tensor([.2, -.7, .6], dtype=torch.double)
    torch.testing.assert_close(hesbo.A @ low, signs * low[hashes])
    assert torch.all(torch.count_nonzero(hesbo.A, dim=1) == 1)
    alebo = ALEBOOptimizer(sphere, bounds(17), 11, 2021, 3)
    with seeded_trial(2021):
        alebo._initialize_projection()
        feasible = alebo._feasible_alebo(30)
    torch.testing.assert_close(alebo.B.norm(dim=0), torch.ones(17, dtype=torch.double))
    assert (feasible @ alebo.Binv.T).abs().max() <= 1


def test_alebo_recovers_stored_ambient_subspace_for_warm_start():
    rng = np.random.default_rng(17)
    ambient_dim, active_dim, count = 36, 10, 150
    basis = np.linalg.qr(rng.normal(size=(ambient_dim, active_dim)))[0].T
    low = rng.normal(scale=0.12, size=(count, active_dim))
    ambient = low @ basis
    assert np.max(np.abs(ambient)) < 1.0
    stored_x = (ambient + 1.0) / 2.0
    stored_y = np.linspace(0.1, 0.9, count)

    optimizer = ALEBOOptimizer(
        sphere,
        np.tile([0.0, 1.0], (ambient_dim, 1)),
        200,
        2021,
        active_dim,
    )
    metadata = optimizer.warm_start_alebo_from_ambient_history(
        stored_x,
        stored_y,
    )

    np.testing.assert_allclose(optimizer.X, stored_x, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(optimizer.y, stored_y, rtol=0.0, atol=0.0)
    assert optimizer.low_X.shape == (count, active_dim)
    assert tuple(optimizer.B.shape) == (active_dim, ambient_dim)
    reconstructed = optimizer.low_X @ optimizer.B
    torch.testing.assert_close(
        reconstructed,
        torch.tensor(ambient, dtype=torch.double),
        rtol=0.0,
        atol=1e-6,
    )
    assert metadata["numerical_rank"] == active_dim
    assert metadata["max_abs_reconstruction_error"] <= 1e-6


def test_alebo_kernel_matches_mahalanobis_equation():
    with seeded_trial(2021):
        kernel = ALEBOKernel(torch.randn(2, 6, dtype=torch.double), torch.Size())
    with torch.no_grad():
        kernel.Uvec.copy_(torch.tensor([1.2, -.4, .8]))
    x = torch.tensor([[.1, .2], [-.7, .8], [.9, -.1]], dtype=torch.double)
    upper = torch.tensor([[1.2, -.4], [0., .8]], dtype=torch.double)
    delta = x[:, None, :] - x[None, :, :]
    expected = torch.exp(-.5 * ((delta @ upper.T)**2).sum(-1))
    torch.testing.assert_close(kernel(x).to_dense(), expected)


def test_trial_streams_advance_repeat_and_restore_outside_state():
    def draw(seed):
        with seeded_trial(seed):
            first = sobol_engine(4).draw(3)
            second = sobol_engine(4).draw(3)
            keys = (np.asarray(jax_key(seed)), np.asarray(jax_key(seed)))
            return first, second, keys, random.random(), np.random.random(), torch.rand(2)
    before = (random.getstate(), np.random.get_state(), torch.random.get_rng_state())
    first = draw(2021)
    draw(2070)  # Simulate a different worker scheduling order.
    second = draw(2021)
    assert not torch.equal(first[0], first[1])
    assert not np.array_equal(first[2][0], first[2][1])
    for a, b in zip(first[:2], second[:2]):
        torch.testing.assert_close(a, b)
    assert first[3:5] == second[3:5]
    torch.testing.assert_close(first[5], second[5])
    assert random.getstate() == before[0]
    np.testing.assert_equal(np.random.get_state(), before[1])
    assert torch.equal(torch.random.get_rng_state(), before[2])


@pytest.mark.parametrize("algorithm", grid_search_nbad.ALGORITHMS)
@pytest.mark.parametrize("value", grid_search_nbad.N_BAD_VALUES)
def test_every_absolute_threshold_reaches_both_optimizers(monkeypatch, algorithm, value):
    _preserve_grid_mutated_runner_state(monkeypatch)
    grid_search_nbad.configure_runner_cell(value)
    assert os.environ["AGVS_SYNTHETIC_N_BAD_ABSOLUTE"] == str(value)
    assert os.environ["AGVS_SYNTHETIC_BATCH_SIZE"] == "3"
    config = runner.apply_runner_agvs_batch_policy(runner.BENCHMARKS['rastrigin3_300'])
    opt = runner.build_optimizer(algorithm, config, sphere, 2021, 25, device='cpu')
    assert (opt.N_bad, opt.t_seq, opt.batch_size) == (value, 2, 3)


@pytest.mark.parametrize("stages", grid_search_tseq.T_SEQ_VALUES)
@pytest.mark.parametrize("backend", ['vanillabo', 'turbo'])
def test_all_visit_lengths_evaluate_three_per_stage(monkeypatch, stages, backend):
    _preserve_grid_mutated_runner_state(monkeypatch)
    grid_search_tseq.configure_runner_cell(stages)
    assert os.environ["AGVS_SYNTHETIC_T_SEQ"] == str(stages)
    assert os.environ["AGVS_SYNTHETIC_BATCH_SIZE"] == "3"
    config = runner.apply_runner_agvs_batch_policy(runner.BENCHMARKS['rastrigin3_300'])
    name = 'AG-VS-TuRBO' if backend == 'turbo' else 'AG-VS-BO'
    configured = runner.build_optimizer(name, config, sphere, 2021, 40, device='cpu')
    assert (configured.t_seq, configured.batch_size) == (stages, 3)
    opt = _agvs(1, backend=backend, bounds=bounds(), n_init=4,
                t_seq=stages, max_iter=4+3*stages, seed=2021, early_stop_enabled=False)
    calls = []
    def propose(node, q):
        calls.append((node.idx, q, len(opt.y)))
        return [(np.full(6, len(opt.y)/100), 0.) for _ in range(q)]
    monkeypatch.setattr(opt, '_propose_q_from_leaf', propose)
    result = opt.optimize()
    assert [x[1] for x in calls] == [3]*stages
    assert len({x[0] for x in calls}) == 1
    assert [x[2] for x in calls] == list(range(4, 4+3*stages, 3))
    assert len(result.y) == 4+3*stages


def test_real_thompson_batches_survive_tree_and_trust_region_restarts(monkeypatch):
    from ag_vs.inner_optimizer.turbo import proposal
    opt = _agvs(1, backend='turbo', bounds=bounds(), n_init=4, max_iter=29,
                t_seq=2, N_bad=1, seed=2021, early_stop_enabled=False,
                objective=lambda x: 0.)
    records = []
    original = proposal._seeded_native_candidates
    def observe(optimizer, engine, model, x, y):
        records.append((engine.batch_size, engine.failtol, len(x)))
        return original(optimizer, engine, model, x, y)
    monkeypatch.setattr(proposal, '_seeded_native_candidates', observe)
    update = opt._after_proposal_batch
    def force_restart(node, indices):
        update(node, indices)
        node._native_turbo_engine.length = 0.
    monkeypatch.setattr(opt, '_after_proposal_batch', force_restart)
    result = opt.optimize()
    assert len(result.y) == 29
    assert opt.tree_restart_count > 0
    assert records and all(batch == 3 for batch, _, _ in records)
    batches = [row['acquisition_batch_size'] for row in result.history
               if row.get('q_index') == 1]
    assert batches == [3]*8+[1]


@pytest.mark.parametrize('budget', [1, 9, 16, 23, 30])
def test_gtbo_constant_data_phases_and_exact_ledger(monkeypatch, budget):
    calls = []
    opt = GTBOOptimizer(lambda x: calls.append(x.copy()) or 1., bounds(), budget,
                        2021, n_particles=128, gt_iterations=1)
    training_sizes = []
    def candidate(indices):
        training_sizes.append(len(indices))
        return opt.default
    monkeypatch.setattr(opt, '_candidate', candidate)
    opt.optimize()
    assert len(calls) == len(opt.X) == len(opt.y) == len(opt.history) == budget
    np.testing.assert_allclose(opt.X[:min(10,budget)], np.tile(opt.default, (min(10,budget),1)))
    assert np.isfinite(opt.marginals).all()
    if budget == 30:
        counts = Counter(h['phase'] for h in opt.history)
        assert counts['calibration'] == 16  # Ten default repeats plus six probes.
        assert counts['bo_initialization'] == 10
        assert counts['bayesian_optimization'] >= 3
        assert training_sizes[0] < 27  # Deduplication never erases ledger entries.


def test_gtbo_priors_defaults_and_no_oracle_inputs():
    expected = dict(n_particles=10000, prior_activeness_rate=.05, n_initial_groups=3,
                    lower_convergence_threshold=.001, upper_convergence_threshold=.9,
                    activeness_threshold=.5, raw_samples=4096, num_restarts=16)
    for key, value in expected.items():
        assert GTBO_SETTINGS[key] == value
    for kind, prior in [('synthetic', (7., 1.)), ('real_life', (3., 1.))]:
        opt = GTBOOptimizer(sphere, bounds(), 30, 2021, experiment_kind=kind)
        assert opt.settings['inactive_prior_parameters'] == prior
        assert opt.settings['active_prior_parameters'] == (0., 1.)
        assert not len(opt.active_dimensions)
    opt = GTBOOptimizer(sphere, bounds(), 30, 2021, objective_name='rastrigin')
    np.testing.assert_allclose(opt.default, -.5)
    with pytest.raises(TypeError):
        GTBOOptimizer(sphere, bounds(), 30, 2021, active_dimensions=[0, 1])


def test_gtbo_resets_failed_warm_gp_state_before_aborting(monkeypatch):
    import botorch.fit
    from botorch.exceptions.errors import ModelFittingError

    opt = GTBOOptimizer(sphere, bounds(), 30, 2021)
    opt.active_dimensions = np.array([0, 2])
    x = torch.rand(12, 6, dtype=torch.double)
    y = torch.rand(12, 1, dtype=torch.double)
    opt._model_state = opt._make_gp(x, y, load_state=False).state_dict()
    calls = []

    def fail_warm_then_fit(mll):
        calls.append(mll)
        if len(calls) == 1:
            raise ModelFittingError("bad cached hyperparameters")
        return mll

    monkeypatch.setattr(botorch.fit, "fit_gpytorch_mll", fail_warm_then_fit)
    model = opt._fit_gp(x, y)

    assert len(calls) == 2
    assert not model.training
    assert opt._model_state is not None
    assert opt.gp_state_reset_recovery_count == 1
    assert opt.gp_unfitted_fallback_count == 0
    assert opt.gp_fit_recovery_history[-1]["recovery"] == "fresh_refit"


def test_gtbo_uses_initialized_gp_when_all_fits_fail(monkeypatch):
    import botorch.fit
    from botorch.exceptions.errors import ModelFittingError

    opt = GTBOOptimizer(sphere, bounds(), 30, 2021)
    x = torch.rand(12, 6, dtype=torch.double)
    y = torch.rand(12, 1, dtype=torch.double)

    def fail(_mll):
        raise ModelFittingError("optimizer rejected parameters")

    monkeypatch.setattr(botorch.fit, "fit_gpytorch_mll", fail)
    with pytest.warns(RuntimeWarning, match="initialized learned-noise GP"):
        model = opt._fit_gp(x, y)

    assert not model.training
    assert opt._model_state is None
    assert opt.gp_state_reset_recovery_count == 0
    assert opt.gp_unfitted_fallback_count == 1
    assert opt.gp_fit_recovery_history[-1]["recovery"] == "initialized_model"


def test_gtbo_particle_probabilities_match_enumerated_gaussians():
    from itertools import product
    from Experiment.benchmark.baseline.native.group_testing.bayes import log_likelihood, log_prior
    particles = torch.tensor(list(product([0., 1.], repeat=3)), dtype=torch.double)
    groups = torch.tensor([[1., 0., 1.], [0., 1., 0.]], dtype=torch.double)
    observations = torch.tensor([.2, 1.3], dtype=torch.double)
    expected = []
    for particle in particles:
        variance = torch.where(groups @ particle > 0, 4., .1).double()
        expected.append(torch.distributions.Normal(0., variance.sqrt()).log_prob(observations).sum())
    torch.testing.assert_close(log_likelihood(particles, observations, groups, .1, 4.), torch.stack(expected))
    expected_prior = (particles * np.log(.05) + (1-particles) * np.log(.95)).sum(-1)
    torch.testing.assert_close(log_prior(particles, .05), expected_prior)


@pytest.mark.parametrize('marginals,expected_tests', [
    ([.001, .9, 0., .95, .0001, 1.], 1),
    ([.002, .899, 0., .95, .0001, 1.], 2),
])
def test_gtbo_convergence_uses_inclusive_selected_thresholds(monkeypatch, marginals, expected_tests):
    from Experiment.benchmark.baseline.native.group_testing.tester import GroupTester
    class Evaluator:
        dim, samples = 6, None
        def __call__(self, *args):
            return torch.tensor(0.), torch.tensor(0.), torch.tensor(0.), torch.zeros(6)
    tester = GroupTester(Evaluator(), 1., 2., gt_iterations=2,
                         lower_convergence_threshold=.001, upper_convergence_threshold=.9)
    monkeypatch.setattr(tester, 'resample', lambda: setattr(tester.state, 'marginals',
        [torch.tensor(marginals, dtype=torch.double)]))
    monkeypatch.setattr(tester, 'select_next_groups', lambda *args: torch.tensor([[True, False, False, False, False, False]]))
    tester.optimize()
    assert len(tester.groups) == expected_tests


def test_gtbo_group_activity_and_mutual_information():
    from Experiment.benchmark.baseline.native.group_testing.tester import GroupTester
    tester = GroupTester(SimpleNamespace(dim=6, samples=None), 1., 1., gt_iterations=1)
    tester.state._particles_long = torch.tensor([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]])
    tester.state.particle_weights = torch.tensor([.4, .2, .3, .1], dtype=torch.double)
    group = torch.tensor([1, 0, 0])
    assert tester.estimate_gamma(group) == pytest.approx(.3)
    # Equal conditional observation laws carry no information about activity.
    assert float(tester.calculate_mutual_information(group)) == pytest.approx(0., abs=.01)


@pytest.mark.parametrize('removed', ['LASSO-VS-BO', 'LASSO-VS-TuRBO', 'lasso-bo', 'lasso-turbo'])
def test_removed_names_cannot_launch_or_become_lassobo(removed):
    with pytest.raises(ValueError):
        runner.canonical_algorithm_name(removed)
    assert runner.canonical_algorithm_name('LassoBO') == 'LassoBO'


def test_merged_node_gets_two_real_batches_and_refits():
    from ag_vs.helper_functions import merge
    from ag_vs.utils import Node
    opt = _agvs(1, backend='turbo', bounds=bounds(), n_init=4, max_iter=20,
                t_seq=2, seed=2021, early_stop_enabled=False)
    opt.X = list(np.random.default_rng(2021).uniform(-1, 1, (4, 6)))
    opt.y = [sphere(x) for x in opt.X]
    opt.true_y = list(opt.y)
    children = [Node(dims=dims, h=1, idx=i+1, parent=opt.root,
                     local_bounds=opt.bounds[dims])
                for i, dims in enumerate([[0, 1], [2, 3], [4], [5]])]
    for child in children:
        child.local_indices = list(range(4))
        child.score_indices = list(range(4))
        child.ave_diff_score = float(child.idx)
    opt.root.children, opt.root.expanded = children, True
    opt._all_nodes = opt._plot_nodes = [opt.root, *children]
    opt._next_node_idx = 5
    merged = merge(opt, opt.root, children, d_eff=1, cohort_id='test')
    assert merged is not None
    counts = []
    with seeded_trial(2021):
        for _ in range(2):
            points = opt._propose_q_from_leaf(merged, 3)
            counts.append(len(merged.model.X_train_))
            start = len(opt.y)
            for x, _ in points:
                opt.X.append(x)
                opt.y.append(sphere(x))
                merged.local_indices.append(len(opt.y)-1)
            opt._after_proposal_batch(merged, list(range(start, start+3)))
            assert merged._native_turbo_engine.batch_size == 3
    assert counts == [4, 7]
    assert merged._native_turbo_engine.n_evals == 6


@pytest.mark.parametrize('grid,values,key', [
    (grid_search_nbad, grid_search_nbad.N_BAD_VALUES, 'N_bad'),
    (grid_search_tseq, grid_search_tseq.T_SEQ_VALUES, 't_seq'),
])
def test_two_panel_figures_six_curves_and_strict_cell_cache(tmp_path, monkeypatch, grid, values, key):
    import matplotlib.figure
    monkeypatch.setattr(grid, 'RESULT_ROOT', tmp_path/'results')
    monkeypatch.setattr(grid, 'FIGURE_ROOT', tmp_path/'figures')
    monkeypatch.setattr(grid, 'TABLE_ROOT', tmp_path/'tables')
    plotted = []
    original = matplotlib.figure.Figure.savefig
    def record(figure, filename, **kwargs):
        plotted.append((str(filename), figure.axes[0].get_legend_handles_labels()[1],
                        figure.axes[0].get_title()))
        return original(figure, filename, **kwargs)
    monkeypatch.setattr(matplotlib.figure.Figure, 'savefig', record)
    paths = []
    for benchmark in grid.BENCHMARKS:
        for algorithm in grid.ALGORITHMS:
            for value in values:
                identity = grid.cell_identity(grid.configure_runner_cell, value,
                    benchmark, algorithm, 0, 3)
                trace = dict(benchmark=benchmark, algorithm=algorithm, trial_id=0,
                    algorithm_seed=2021, benchmark_seed=2021, n_evaluations=3,
                    best_observed_history=[1., 2., 3.], complete=True,
                    cache_identity=identity, settings={
                        'benchmark_noise_variance': 1e-4,
                        'algorithm_settings': {**identity['algorithm_settings'],
                            **({'N_bad': value, 'N_bad_mode': 'absolute'} if key == 'N_bad' else {})}})
                path = runner.trace_path_for(grid.cell_roots(value)[0], benchmark, algorithm, 0)
                runner.write_json(path, trace)
        curves = grid.all_curves(benchmark, values, 1)
        assert len(curves) == 12
        paths.extend(grid.save_benchmark_plot(benchmark, values, 1))
    assert len(paths) == 4  # One two-algorithm figure per benchmark, in two formats.
    assert all(p.is_file() for p in paths)
    assert sum(p.suffix == '.pdf' for p in paths) == 2
    assert all(len(labels) == 6 for _, labels, _ in plotted)
    assert all(title == "AG-VS-BO" for _, _, title in plotted)
    assert all(p.is_file() for p in grid.write_csv_outputs(grid.BENCHMARKS, values, 1))
    # A trace recorded for a different sensitivity-cell value cannot enter the
    # plot even when benchmark, algorithm, seeds, and budget still match.
    trace['settings']['algorithm_settings'][key] = values[-1] + 1
    runner.write_json(path, trace)
    loaded, _ = grid.load_cell_traces(benchmark, values[-1], 1, algorithm)
    assert loaded == []
