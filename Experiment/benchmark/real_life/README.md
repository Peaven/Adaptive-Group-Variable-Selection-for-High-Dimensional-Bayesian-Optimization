# MOPTA08 adapter

This directory contains MOPTA08 (D=124), the released real-life task.
`tasks.py` provides lazy per-trial objectives and registry-derived configurations.
`mopta08_backend.py` acquires and verifies the official Windows executable and
evaluates its objective and 68 constraints.

Run `python -m Experiment.benchmark.mopta --list` to inspect settings, remove
`--list` to run, or use `--plots-only` to regenerate saved-result reports.
See the root README for installation, executable acquisition and seed rules.
