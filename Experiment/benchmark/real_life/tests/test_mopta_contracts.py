import pickle

import numpy as np
import pytest

from Experiment.benchmark.real_life.tasks import MoptaObjective, make_real_life_config


def test_config_factory_is_lazy_and_can_be_sent_to_spawned_worker(monkeypatch):
    monkeypatch.setattr(MoptaObjective, "_build", lambda self: pytest.fail("Not lazy"))
    config = make_real_life_config("mopta08", 4)
    factory = pickle.loads(pickle.dumps(config["objective_factory"]))
    objective = factory(2021)
    assert objective._objective is None
    assert objective.seed == 2021
    assert config["bounds"].shape == (124, 2)
    assert config["active_indices"] is None
    assert config["max_iter"] == 150
    assert config["trials"] == 25


def test_prepare_builds_once_and_evaluations_preserve_maximize_sign(monkeypatch):
    calls = []

    class Backend:
        def __call__(self, x):
            calls.append(x.copy())
            return -250.0

        def close(self):
            calls.append("closed")

    builds = []
    def build(self):
        builds.append(self.seed)
        return Backend()

    monkeypatch.setattr(MoptaObjective, "_build", build)
    with MoptaObjective("mopta08", 2022) as objective:
        objective.prepare().prepare()
        assert objective._evaluation_index == 0
        assert objective(np.full(124, 2.0)) == -250.0
        assert objective(np.full(124, -1.0)) == -250.0
        assert objective._evaluation_index == 2
    assert builds == [2022]
    np.testing.assert_array_equal(calls[0], np.ones(124))
    np.testing.assert_array_equal(calls[1], np.zeros(124))
    assert calls[-1] == "closed"


@pytest.mark.parametrize("value", [np.zeros(123), np.zeros((1, 124))])
def test_bad_dimension_is_rejected_before_loading_backend(monkeypatch, value):
    monkeypatch.setattr(MoptaObjective, "_build", lambda self: pytest.fail("Loaded backend"))
    with pytest.raises(ValueError):
        MoptaObjective("mopta08", 2021)(value)
