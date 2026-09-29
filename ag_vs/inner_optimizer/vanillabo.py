"""MCTS-VS-style analytic Expected Improvement proposals for AG-VS-BO."""

from __future__ import annotations

import numpy as np

from Experiment.benchmark.baseline.reference_infrastructure import (
    classical_expected_improvement,
)


def propose_vanilla_one(optimizer, leaf):
    return propose_vanilla_independent_batch(optimizer, leaf, 1)[0]


def propose_vanilla_independent_batch(optimizer, leaf, q):
    """Select top-k pointwise EI candidates without modelling batch interaction."""

    dims = list(getattr(leaf, "dims", []))
    q = int(q)
    if q <= 0:
        return []
    if len(dims) == 0:
        base = np.clip(
            optimizer._base_fill_for_leaf(leaf),
            optimizer.bounds[:, 0],
            optimizer.bounds[:, 1],
        )
        optimizer._last_joint_q_acquisition = False
        return [(base.copy(), 0.0) for _ in range(q)]

    node_bounds = optimizer._node_bounds(leaf)
    lows, highs = node_bounds[:, 0], node_bounds[:, 1]
    local_model = optimizer._fit_local_gp(leaf)
    if local_model is None:
        out = []
        for _ in range(q):
            base = optimizer._base_fill_for_leaf(leaf)
            base[dims] = optimizer.rng.uniform(lows, highs)
            out.append(
                (
                    np.clip(
                        base,
                        optimizer.bounds[:, 0],
                        optimizer.bounds[:, 1],
                    ),
                    0.0,
                )
            )
        optimizer._last_joint_q_acquisition = False
        return out

    candidates = optimizer._sample_acquisition_candidates(
        lows,
        highs,
        max(int(optimizer.candidate_count), q),
    )
    scores = classical_expected_improvement(
        local_model,
        np.asarray(local_model.X_train_, dtype=float),
        candidates,
    )
    # Keep the exact MCTS-VS ordering convention from
    # baseline/vanilia_bo.py: np.argsort(scores)[-q:].
    indices = np.argsort(np.asarray(scores, dtype=float))[-q:]
    out = []
    for index in indices:
        base = optimizer._base_fill_for_leaf(leaf)
        base[dims] = candidates[int(index)]
        out.append(
            (
                np.clip(
                    base,
                    optimizer.bounds[:, 0],
                    optimizer.bounds[:, 1],
                ),
                float(scores[int(index)]),
            )
        )
    optimizer._last_joint_q_acquisition = False
    return out


def propose_vanilla_joint_q(optimizer, leaf, q, *, mc_samples=256):
    """Greedily maximize Monte-Carlo qEI using the node GP's joint posterior.

    Each greedy addition is scored as part of the full selected set, so GP
    posterior covariance discourages redundant points.  This is deliberately
    distinct from ``propose_vanilla_independent_batch``, which merely takes the
    top-k pointwise EI scores.
    """

    dims = list(getattr(leaf, "dims", []))
    q = int(q)
    if q <= 0:
        return []
    if q == 1:
        result = propose_vanilla_independent_batch(optimizer, leaf, 1)
        optimizer._last_joint_q_acquisition = False
        return result
    if not dims:
        base = np.clip(
            optimizer._base_fill_for_leaf(leaf),
            optimizer.bounds[:, 0],
            optimizer.bounds[:, 1],
        )
        optimizer._last_joint_q_acquisition = True
        return [(base.copy(), 0.0) for _ in range(q)]

    node_bounds = optimizer._node_bounds(leaf)
    lows, highs = node_bounds[:, 0], node_bounds[:, 1]
    model = optimizer._fit_local_gp(leaf)
    if model is None:
        result = propose_vanilla_independent_batch(optimizer, leaf, q)
        optimizer._last_joint_q_acquisition = True
        return result

    candidates = optimizer._sample_acquisition_candidates(
        lows,
        highs,
        max(int(optimizer.candidate_count), q),
    )
    means, stds = model.predict(candidates, return_std=True)
    means = np.asarray(means, dtype=float).reshape(-1)
    stds = np.maximum(np.asarray(stds, dtype=float).reshape(-1), 1e-12)
    pointwise_ei = classical_expected_improvement(
        model,
        np.asarray(model.X_train_, dtype=float),
        candidates,
    )
    best_f = float(np.max(np.asarray(model.y_train_, dtype=float)))
    selected = [int(np.nanargmax(np.asarray(pointwise_ei, dtype=float)))]
    set_scores = [float(pointwise_ei[selected[0]])]
    sample_count = max(32, int(mc_samples))

    while len(selected) < min(q, len(candidates)):
        selected_x = candidates[np.asarray(selected, dtype=int)]
        selected_mean, selected_cov = model.predict(selected_x, return_cov=True)
        selected_mean = np.asarray(selected_mean, dtype=float).reshape(-1)
        selected_cov = np.asarray(selected_cov, dtype=float)
        selected_cov = 0.5 * (selected_cov + selected_cov.T)
        selected_cov.flat[:: len(selected) + 1] += 1e-10
        selected_samples = optimizer.rng.multivariate_normal(
            selected_mean,
            selected_cov,
            size=sample_count,
            check_valid="ignore",
        )

        # sklearn GPR posterior cross-covariance between every candidate and
        # the already selected set, without forming a 1024x1024 covariance.
        train_x = np.asarray(model.X_train_, dtype=float)
        k_candidate_train = model.kernel_(candidates, train_x)
        k_train_selected = model.kernel_(train_x, selected_x)
        solved = np.linalg.solve(model.L_, k_train_selected)
        solved = np.linalg.solve(model.L_.T, solved)
        cross_cov = model.kernel_(candidates, selected_x) - k_candidate_train @ solved
        inv_selected_cov = np.linalg.pinv(selected_cov, hermitian=True)
        weights = cross_cov @ inv_selected_cov
        conditional_mean = means[:, None] + (
            (selected_samples - selected_mean[None, :]) @ weights.T
        ).T
        conditional_var = np.maximum(
            stds**2 - np.einsum("ij,ij->i", weights, cross_cov),
            1e-12,
        )
        candidate_samples = conditional_mean + np.sqrt(conditional_var)[:, None] * (
            optimizer.rng.standard_normal((len(candidates), sample_count))
        )
        current_max = np.maximum(best_f, np.max(selected_samples, axis=1))
        joint_values = np.mean(
            np.maximum(current_max[None, :], candidate_samples) - best_f,
            axis=1,
        )
        joint_values[np.asarray(selected, dtype=int)] = -np.inf
        next_index = int(np.nanargmax(joint_values))
        selected.append(next_index)
        set_scores.append(float(joint_values[next_index]))

    out = []
    for index, score in zip(selected, set_scores):
        base = optimizer._base_fill_for_leaf(leaf)
        base[dims] = candidates[int(index)]
        out.append(
            (
                np.clip(base, optimizer.bounds[:, 0], optimizer.bounds[:, 1]),
                float(score),
            )
        )
    optimizer._last_joint_q_acquisition = True
    return out


# Backwards-compatible name now means the joint q acquisition path.
propose_vanilla_q = propose_vanilla_joint_q


__all__ = [
    "propose_vanilla_independent_batch",
    "propose_vanilla_joint_q",
    "propose_vanilla_one",
    "propose_vanilla_q",
]
