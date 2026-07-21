"""Regression tests for the Arouca design-verification example.

These tests load the model contained in:

    examples/design_verification/arouca_extension_verification.py

The tests confirm that the model structure is correct, the optimized
baseline design is hydraulically feasible, the original network is
preserved, and increased demand produces the expected pressure failure.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest


pytestmark = pytest.mark.extensions


def load_arouca_example() -> ModuleType:
    """Load the Arouca example script as a Python module."""

    repository_root = Path(__file__).resolve().parents[3]

    example_path = (
        repository_root
        / "examples"
        / "design_verification"
        / "arouca_extension_verification.py"
    )

    if not example_path.exists():
        raise FileNotFoundError(
            f"Arouca example was not found at: {example_path}"
        )

    specification = importlib.util.spec_from_file_location(
        "arouca_extension_verification",
        example_path,
    )

    if specification is None or specification.loader is None:
        raise ImportError(
            "The Arouca example module could not be loaded."
        )

    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)

    return module


@pytest.fixture(scope="module")
def arouca_example() -> ModuleType:
    """Provide the loaded Arouca example module."""

    return load_arouca_example()


def test_arouca_example_contains_expected_components(
    arouca_example: ModuleType,
) -> None:
    """Confirm that the Arouca model contains the expected assets."""

    wn = arouca_example.build_arouca_network()

    assert set(wn.reservoir_name_list) == {
        "Arouca_Source",
    }

    assert set(wn.tank_name_list) == {
        "Windy_Hill_Tank",
        "Bon_Air_North_Tank",
    }

    assert set(wn.pump_name_list) == {
        "Arouca_Highlift_Pump",
    }

    assert set(wn.pipe_name_list) == {
        "P_Proposed_Main",
        "P_Windy_to_BonAirTank",
        "P_Windy_Tank_to_Demand",
        "P_BonAir_Tank_to_Demand",
        "P_Pump_to_Existing",
    }

    assert {
        "Pump_Discharge",
        "Existing_Zone",
        "Windy_Hill_Demand",
        "Bon_Air_North_Demand",
    }.issubset(
        set(wn.junction_name_list)
    )


def test_arouca_baseline_design_is_feasible(
    arouca_example: ModuleType,
    tmp_path: Path,
) -> None:
    """Confirm that the optimized baseline satisfies both constraints."""

    baseline_case = (
        arouca_example.VALIDATION_CASES[0]
    )

    summary, hydraulic_results, assessed_wn = (
        arouca_example.run_case(
            case=baseline_case,
            design=(
                arouca_example.build_optimized_design()
            ),
            simulator="WNTR",
            output_directory=tmp_path,
            export_outputs=False,
        )
    )

    assert summary["Scenario"] == (
        "Baseline design case"
    )

    assert summary["Simulator"] == "WNTR"

    assert summary["Design"] == (
        "Arouca optimized design"
    )

    assert summary["Overall Feasible"] is True

    assert summary["Pressure Feasible"] is True

    assert summary["Velocity Feasible"] is True

    assert summary["Original Network Preserved"] is True

    assert summary["Minimum Pressure (m)"] == pytest.approx(
        15.932,
        abs=0.10,
    )

    assert summary["Maximum Velocity (m/s)"] == pytest.approx(
        1.587,
        abs=0.03,
    )

    assert summary["Critical Pressure Junction"] == (
        "Bon_Air_North_Demand"
    )

    assert summary["Critical Velocity Pipe"] == (
        "P_Pump_to_Existing"
    )

    assert summary["Pressure Compliance (%)"] == pytest.approx(
        100.0
    )

    assert summary["Velocity Compliance (%)"] == pytest.approx(
        100.0
    )

    assert "pressure" in hydraulic_results.node

    assert "velocity" in hydraulic_results.link

    assert set(assessed_wn.pipe_name_list) == {
        "P_Proposed_Main",
        "P_Windy_to_BonAirTank",
        "P_Windy_Tank_to_Demand",
        "P_BonAir_Tank_to_Demand",
        "P_Pump_to_Existing",
    }


def test_arouca_demand_increase_detects_pressure_failure(
    arouca_example: ModuleType,
    tmp_path: Path,
) -> None:
    """Confirm that a 10 percent demand increase causes a pressure failure."""

    increased_demand_case = (
        arouca_example.VALIDATION_CASES[1]
    )

    summary, _, _ = arouca_example.run_case(
        case=increased_demand_case,
        design=arouca_example.build_optimized_design(),
        simulator="WNTR",
        output_directory=tmp_path,
        export_outputs=False,
    )

    assert summary["Scenario"] == (
        "Demand increased by 10%"
    )

    assert summary["Overall Feasible"] is False

    assert summary["Pressure Feasible"] is False

    assert summary["Velocity Feasible"] is True

    assert (
        summary["Minimum Pressure (m)"]
        < arouca_example.MINIMUM_PRESSURE_M
    )

    assert (
        summary["Maximum Velocity (m/s)"]
        < arouca_example.MAXIMUM_VELOCITY_MPS
    )

    assert (
        summary["Pressure Compliance (%)"]
        < 100.0
    )

    assert summary["Velocity Compliance (%)"] == pytest.approx(
        100.0
    )

    assert summary["Original Network Preserved"] is True