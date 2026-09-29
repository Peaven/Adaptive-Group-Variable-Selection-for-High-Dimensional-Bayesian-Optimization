"""Public optimizer API without eagerly importing the benchmark dependency graph."""

from importlib import import_module

__all__ = ["AGVSOptimizer"]


def __getattr__(name):
    # Importing ag_vs.utils from reference_infrastructure must not start
    # importing the optimizer while the reference constants are still loading.
    if name == "AGVSOptimizer":
        optimizer = import_module(".optimizer", __name__).AGVSOptimizer
        globals()[name] = optimizer
        return optimizer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
