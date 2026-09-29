"""Benchmark identities and entrypoints for the AG-VS experiment suite."""

import importlib

from Experiment.utils import (
    REAL_LIFE_ALGORITHMS,
    REAL_LIFE_EXPERIMENTS,
    SYNTHETIC_ALGORITHMS,
    SYNTHETIC_EXPERIMENTS,
    get_experiment,
    iter_applicable_real_life_algorithms,
    real_life_algorithm_is_applicable,
    real_life_optimizer_dimension,
    real_life_reference_embedding_dimension,
)

__all__ = [
    "REAL_LIFE_ALGORITHMS",
    "REAL_LIFE_EXPERIMENTS",
    "SYNTHETIC_ALGORITHMS",
    "SYNTHETIC_EXPERIMENTS",
    "get_experiment",
    "iter_applicable_real_life_algorithms",
    "real_life_algorithm_is_applicable",
    "real_life_optimizer_dimension",
    "real_life_reference_embedding_dimension",
    "runner",
]


def __getattr__(name):
    """Load the renamed runner lazily, preserving the historical public alias.

    Eagerly importing it here creates a cycle while ``ag_vs.optimizer`` loads
    benchmark reference constants, because the runner itself imports AG-VS.
    """

    if name == "runner":
        module = importlib.import_module("Experiment.benchmark.runner_synthetic")
        globals()[name] = module
        return module
    raise AttributeError(name)
