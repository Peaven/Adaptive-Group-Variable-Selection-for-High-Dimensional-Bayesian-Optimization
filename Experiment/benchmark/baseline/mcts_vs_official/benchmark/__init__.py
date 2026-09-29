"""Vendored MCTS-VS benchmark objectives, importable as a project package."""


def get_problem(*args, **kwargs):
    # Keep the historical convenience import lazy. Importing a standalone
    # objective such as Rover must not pull in every legacy benchmark module.
    from .benchmark_problem import get_problem as _get_problem

    return _get_problem(*args, **kwargs)


__all__ = ["get_problem"]
