"""Tests for design-verification exception types."""

import pandas as pd
import pytest
import wntr

from wntr.extensions.design_verification import (
    DesignVerificationError,
    HydraulicScenario,
    IncompleteHydraulicResultsError,
    InvalidConstraintError,
    InvalidDesignError,
    InvalidScenarioError,
    PipeDesign,
    UnsupportedSimulatorError,
    apply_hydraulic_scenario,
    apply_pipe_design,
    evaluate_minimum_pressure,
    audit_head_pump_curves,
    run_verification_batch,
)

from wntr.extensions.design_verification.runner import (
    _canonical_simulator_name,
    _select_result_columns,
)


pytestmark = pytest.mark.extensions


@pytest.mark.parametrize(
    ("exception_type", "legacy_type"),
    [
        (InvalidDesignError, ValueError),
        (InvalidScenarioError, ValueError),
        (InvalidConstraintError, ValueError),
        (UnsupportedSimulatorError, ValueError),
        (
            IncompleteHydraulicResultsError,
            RuntimeError,
        ),
    ],
)
def test_domain_exception_preserves_legacy_compatibility(
    exception_type,
    legacy_type,
):
    """Domain errors remain compatible with existing handlers."""
    assert issubclass(
        exception_type,
        DesignVerificationError,
    )
    assert issubclass(
        exception_type,
        legacy_type,
    )

    with pytest.raises(DesignVerificationError):
        raise exception_type("test failure")

def test_unsupported_simulator_uses_domain_exception():
    """Unsupported simulator names use the public domain error."""
    with pytest.raises(
        UnsupportedSimulatorError,
        match="WNTR or EPANET",
    ):
        _canonical_simulator_name("unsupported")


def test_missing_required_results_use_domain_exception():
    """Missing required components use the hydraulic-result error."""
    table = pd.DataFrame(
        data=[[20.0]],
        columns=["J1"],
    )

    with pytest.raises(
        IncompleteHydraulicResultsError,
        match="missing",
    ):
        _select_result_columns(
            table=table,
            component_names=["J1", "J2"],
            result_name="pressure",
        )


def test_duplicate_required_results_use_domain_exception():
    """Duplicated required columns use the hydraulic-result error."""
    table = pd.DataFrame(
        data=[[20.0, 21.0]],
        columns=["J1", "J1"],
    )

    with pytest.raises(
        IncompleteHydraulicResultsError,
        match="duplicate",
    ):
        _select_result_columns(
            table=table,
            component_names=["J1"],
            result_name="pressure",
        )


def test_empty_required_components_use_constraint_error():
    """An unassessable component set uses a constraint error."""
    table = pd.DataFrame(
        data=[[20.0]],
        columns=["J1"],
    )

    with pytest.raises(
        InvalidConstraintError,
        match="No applicable components",
    ):
        _select_result_columns(
            table=table,
            component_names=[],
            result_name="pressure",
        )

def test_constraint_validation_uses_domain_exception():
    """Invalid hydraulic limits use InvalidConstraintError."""
    pressure = pd.DataFrame(
        data=[[20.0]],
        columns=["J1"],
    )

    with pytest.raises(
        InvalidConstraintError,
        match="minimum_pressure_m",
    ):
        evaluate_minimum_pressure(
            pressure=pressure,
            minimum_pressure_m=float("nan"),
        )


def test_design_validation_uses_domain_exception():
    """Invalid pipe designs use InvalidDesignError."""
    wn = wntr.network.WaterNetworkModel()

    design = PipeDesign(
        name="Empty design",
        diameters_m={},
    )

    with pytest.raises(
        InvalidDesignError,
        match="at least one pipe",
    ):
        apply_pipe_design(
            wn=wn,
            design=design,
        )


def test_scenario_validation_uses_domain_exception():
    """Invalid operating scenarios use InvalidScenarioError."""
    wn = wntr.network.WaterNetworkModel()

    scenario = HydraulicScenario(
        name="Invalid demand",
        demand_multiplier=0.0,
        demand_model="DD",
        duration_s=0,
        hydraulic_timestep_s=3600,
        report_timestep_s=3600,
    )

    with pytest.raises(
        InvalidScenarioError,
        match="demand_multiplier",
    ):
        apply_hydraulic_scenario(
            wn=wn,
            scenario=scenario,
        )

def _valid_batch_scenario():
    """Return a valid scenario for batch-validation tests."""
    return HydraulicScenario(
        name="Baseline",
        demand_multiplier=1.0,
        demand_model="DD",
        duration_s=0,
        hydraulic_timestep_s=3600,
        report_timestep_s=3600,
    )


def test_empty_design_batch_uses_design_error():
    """An empty design collection uses InvalidDesignError."""
    wn = wntr.network.WaterNetworkModel()

    with pytest.raises(
        InvalidDesignError,
        match="designs must contain",
    ):
        run_verification_batch(
            wn=wn,
            scenarios=[_valid_batch_scenario()],
            designs=[],
            simulators="WNTR",
        )


def test_empty_scenario_batch_uses_scenario_error():
    """An empty scenario collection uses InvalidScenarioError."""
    wn = wntr.network.WaterNetworkModel()

    with pytest.raises(
        InvalidScenarioError,
        match="scenarios must contain",
    ):
        run_verification_batch(
            wn=wn,
            scenarios=[],
            simulators="WNTR",
        )


def test_empty_simulator_batch_uses_simulator_error():
    """An empty simulator collection uses its domain error."""
    wn = wntr.network.WaterNetworkModel()

    with pytest.raises(
        UnsupportedSimulatorError,
        match="simulators must contain",
    ):
        run_verification_batch(
            wn=wn,
            scenarios=[_valid_batch_scenario()],
            simulators=[],
        )


@pytest.mark.parametrize(
    "tolerance",
    [
        -1.0,
        float("nan"),
        float("inf"),
        "invalid",
    ],
)
def test_invalid_pump_tolerance_uses_constraint_error(
    tolerance,
):
    """Invalid pump tolerances use InvalidConstraintError."""
    wn = wntr.network.WaterNetworkModel()
    flowrate = pd.DataFrame(index=[0])

    with pytest.raises(
        InvalidConstraintError,
        match="relative_tolerance",
    ):
        audit_head_pump_curves(
            network=wn,
            flowrate=flowrate,
            relative_tolerance=tolerance,
        )

