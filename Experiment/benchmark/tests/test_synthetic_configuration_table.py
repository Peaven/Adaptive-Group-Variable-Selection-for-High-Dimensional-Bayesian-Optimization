from __future__ import annotations

import csv
from pathlib import Path

from Experiment.benchmark import runner
from Experiment.utils import (
    SYNTHETIC_TABLE_COLUMNS,
    synthetic_configuration_csv_text,
    synthetic_configuration_rows,
    write_synthetic_configuration_table,
)
from Experiment.utils import SYNTHETIC_EXPERIMENTS


TABLE_ROOT = (
    Path(__file__).resolve().parents[2]
    / "tables"
    / "synthetic"
)


def test_table_has_one_row_per_algorithm_objective_deff_D():
    rows = synthetic_configuration_rows(
        runner.SYNTHETIC_ALGORITHM_ORDER,
        runner.algorithm_settings_summary,
    )
    assert len(rows) == len(runner.SYNTHETIC_ALGORITHM_ORDER) * len(SYNTHETIC_EXPERIMENTS)
    identities = {
        (row["Algorithm"], row["Objective"], row["d_eff"], row["D"])
        for row in rows
    }
    assert len(identities) == len(rows)
    assert all(tuple(row) == SYNTHETIC_TABLE_COLUMNS for row in rows)


def test_configuration_csv_is_generated_without_saved_experiment_data(tmp_path):
    for family in runner.ALGORITHM_OUTPUT_FAMILIES:
        algorithms = [
            algorithm
            for algorithm in runner.SYNTHETIC_ALGORITHM_ORDER
            if runner.algorithm_output_family(algorithm) == family
        ]
        expected = synthetic_configuration_csv_text(
            algorithms,
            runner.algorithm_settings_summary,
        )
        table_path = tmp_path / family / "synthetic_configuration_matrix.csv"
        write_synthetic_configuration_table(table_path, algorithms=algorithms)
        assert table_path.read_text(encoding="utf-8") == expected
        with table_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            assert tuple(reader.fieldnames or ()) == SYNTHETIC_TABLE_COLUMNS

    assert not (tmp_path / "synthetic_configuration_matrix.csv").exists()


def test_controlled_dimensions_and_reference_acquisition_labels_are_exact():
    rows = synthetic_configuration_rows(
        runner.SYNTHETIC_ALGORITHM_ORDER,
        runner.algorithm_settings_summary,
    )
    deff3 = [row for row in rows if row["d_eff"] == 3]
    for algorithm in ("Dropout-BO", "Dropout-TuRBO"):
        assert {
            row["Variable-selection settings"]
            for row in deff3
            if row["Algorithm"] == algorithm
        } == {"active_dims=3 (configured synthetic d_eff)"}
    assert {
        row["Variable-selection settings"]
        for row in deff3
        if row["Algorithm"] == "HeSBO"
    } == {"embedding dimension=3 (configured synthetic d_eff)"}
    assert {
        row["Variable-selection settings"]
        for row in deff3
        if row["Algorithm"] == "ALEBO"
    } == {"embedding dimension=3 (configured synthetic d_eff)"}
    assert {
        row["Variable-selection settings"]
        for row in deff3
        if row["Algorithm"] == "VAE-BO"
    } == {"latent dimension=3 (configured synthetic d_eff)"}

    all_text = "\n".join(str(value) for row in rows for value in row.values())
    assert "classical analytic Expected Improvement" in all_text
    assert "top-k highest pointwise classical analytic EI" in all_text
    assert all("qLog" not in str(row) for row in rows if row["Algorithm"] != "GTBO")
    assert any("qLogNEI" in str(row) for row in rows if row["Algorithm"] == "GTBO")
    assert '"vae_learning_rate":0.001' in all_text
    # ADD-GP-UCB has this setting but is intentionally outside the active
    # synthetic experiment roster, so it must not leak into the active table.
    assert not any(row["Algorithm"] == "ADD-GP-UCB" for row in rows)
    assert "commonNoise = 0.01 * std(initial objective values)" not in all_text
    assert "training point with best observed standardized response" in all_text
