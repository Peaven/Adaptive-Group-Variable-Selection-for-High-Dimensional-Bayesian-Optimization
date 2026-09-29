import random

import numpy as np
import pytest

from Experiment.benchmark.baseline.native_baselines import (
    DropoutTuRBOOptimizer,
)
from Experiment.benchmark.baseline import native_baselines as adapters
from Experiment.benchmark.baseline.native_optimizers import _isolated_legacy_rng


def _bounds(dim):
    return np.tile(np.array([[-1.0, 1.0]]), (dim, 1))




def test_clean_vendored_turbo_component_is_importable():
    component = adapters._load_native_turbo_vs_component()
    assert component.__name__ == "Turbo1_VS_Component"
    assert component.__module__ == "Experiment.benchmark.baseline.native.turbo.turbo_1_component"


def test_vae_reference_controls_are_explicit_and_used():
    optimizer = adapters.NativeBaselineOptimizer(
        objective=lambda x: 0.0,
        bounds=_bounds(4),
        max_iter=10,
        method="vae-bo",
        active_dim=2,
        noise_variance=1e-4,
        vae_update_interval=30,
        vae_learning_rate=0.001,
        enable_posterior_stopping=False,
    )
    with _isolated_legacy_rng(0):
        optimizer._ensure_vae()
    assert optimizer.vae_update_interval == 30
    assert optimizer._vae_optimizer.param_groups[0]["lr"] == pytest.approx(0.001)
