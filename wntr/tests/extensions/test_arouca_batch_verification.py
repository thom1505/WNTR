"""Regression tests for the Arouca batch-verification example."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest


pytestmark = pytest.mark.extensions


def load_arouca_batch_example() -> ModuleType:
    """Load the Arouca batch example as a Python module."""
    repository_root = Path(__file__).resolve().parents[3]

    example_path = (
        repository_root
        / "examples"
        / "design_verification"
        / "arouca_batch_verification.py"
    )

    if not example_path.exists():
        raise FileNotFoundError(
            f"Arouca batch example was not found at: {example_path}"
        )

    specification = importlib.util.spec_from_file_location(
        "arouca_batch_verification",
        example_path,
    )

    if specification is None or specification.loader is None:
        raise ImportError(
            "The Arouca batch example could not be loaded."
        )

    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)

    return module


@pytest.fixture(scope="module")
def batch_example() -> ModuleType:
    """Provide the loaded Arouca batch example."""
    return load_arouca_batch_example()


@pytest.fixture(scope="module")
def reduced_batch(
    batch_example: ModuleType,
):
    """Run four experiments once for the regression tests."""
    cases = batch_example.AROUCA.VALIDATION_CASES[:2]

    all_designs = batch_example.build_batch_designs()

    designs = [
        None,
        all_designs[1],
    ]

    return batch_example.run_arouca_batch(
        cases=cases,
        designs=designs,
        simulator="WNTR",
        continue_on_error=False,
        retain_hydraulic_results=False,
    )


def test_full_experiment_plan_contains_24_runs(
    batch_example: ModuleType,
) -> None:
    """Confirm the planned matrix has three designs and eight cases."""
    plan = batch_example.build_experiment_plan()

    assert len(plan) == 24

    assert plan["case_name"].nunique() == 8

    assert plan["design_label"].nunique() == 3

    assert set(plan["simulator_name"]) == {"WNTR"}


def test_reduced_batch_has_unique_successful_experiments(
    reduced_batch,
) -> None:
    """Confirm the reduced regression batch completes correctly."""
    summary, records = reduced_batch

    assert len(summary) == 4

    assert summary["experiment_id"].is_unique

    assert summary["full_configuration_hash"].is_unique

    assert set(summary["status"]) == {"completed"}

    assert bool(
        summary["original_network_preserved"].all()
    ) is True

    assert set(records) == set(summary["experiment_id"])


def test_optimized_baseline_reproduces_expected_result(
    batch_example: ModuleType,
    reduced_batch,
) -> None:
    """Confirm the optimized baseline remains hydraulically feasible."""
    summary, _ = reduced_batch

    row = summary.loc[
        (
            summary["case_name"]
            == "Baseline design case"
        )
        & (
            summary["design_label"]
            == "Arouca optimized design"
        )
    ].iloc[0]

    assert bool(row["overall_feasible"]) is True

    assert row["minimum_pressure_m"] == pytest.approx(
        15.932,
        abs=0.10,
    )

    assert row["maximum_velocity_mps"] == pytest.approx(
        1.587,
        abs=0.03,
    )

    assert row["critical_pressure_component"] == (
        "Bon_Air_North_Demand"
    )

    assert row["critical_velocity_component"] == (
        "P_Pump_to_Existing"
    )


def test_increased_demand_detects_pressure_failure(
    reduced_batch,
) -> None:
    """Confirm the optimized design fails pressure under 10% demand growth."""
    summary, _ = reduced_batch

    row = summary.loc[
        (
            summary["case_name"]
            == "Demand increased by 10%"
        )
        & (
            summary["design_label"]
            == "Arouca optimized design"
        )
    ].iloc[0]

    assert bool(row["overall_feasible"]) is False

    assert bool(row["pressure_feasible"]) is False

    assert bool(row["velocity_feasible"]) is True

    assert row["pressure_margin_m"] < 0.0

    assert row["velocity_margin_mps"] > 0.0


def test_complete_configuration_hash_is_repeatable(
    batch_example: ModuleType,
) -> None:
    """Confirm that identical experiments receive identical hashes."""
    case = [
        batch_example.AROUCA.VALIDATION_CASES[0]
    ]

    optimized_design = [
        batch_example.build_batch_designs()[1]
    ]

    first_summary, _ = batch_example.run_arouca_batch(
        cases=case,
        designs=optimized_design,
        simulator="WNTR",
        continue_on_error=False,
    )

    second_summary, _ = batch_example.run_arouca_batch(
        cases=case,
        designs=optimized_design,
        simulator="WNTR",
        continue_on_error=False,
    )

    assert (
        first_summary.loc[
            0,
            "full_configuration_hash",
        ]
        == second_summary.loc[
            0,
            "full_configuration_hash",
        ]
    )

    assert (
        first_summary.loc[0, "experiment_id"]
        == second_summary.loc[0, "experiment_id"]
    )
