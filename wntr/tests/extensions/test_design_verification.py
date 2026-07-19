"""Tests for the design-verification extension."""

import pandas as pd
import pytest
import wntr

from wntr.extensions.design_verification import (
    PipeDesign,
    apply_pipe_design,
    evaluate_maximum_velocity,
    evaluate_minimum_pressure,
)

pytestmark = pytest.mark.extensions


def test_minimum_pressure_summary():
    """Identify the minimum pressure and its location."""
    pressure = pd.DataFrame(
        data=[
            [20.0, 14.0],
            [15.0, 16.0],
        ],
        index=[0, 3600],
        columns=["J1", "J2"],
    )

    result = evaluate_minimum_pressure(
        pressure=pressure,
        minimum_pressure_m=15.0,
    )

    assert result.critical_value == pytest.approx(14.0)
    assert result.compliance_pct == pytest.approx(75.0)
    assert result.feasible is False
    assert result.critical_component == "J2"
    assert result.critical_time == 0


def test_maximum_velocity_uses_absolute_value():
    """Use velocity magnitude when identifying violations."""
    velocity = pd.DataFrame(
        data=[
            [1.0, -2.7],
            [2.5, 0.5],
        ],
        index=[0, 3600],
        columns=["P1", "P2"],
    )

    result = evaluate_maximum_velocity(
        velocity=velocity,
        maximum_velocity_mps=2.5,
    )

    assert result.critical_value == pytest.approx(2.7)
    assert result.compliance_pct == pytest.approx(75.0)
    assert result.feasible is False
    assert result.critical_component == "P2"
    assert result.critical_time == 0


def test_partial_pressure_compliance():
    """Accept a result that meets a reduced compliance requirement."""
    pressure = pd.DataFrame(
        data=[
            [20.0, 14.0],
            [15.0, 16.0],
        ],
        columns=["J1", "J2"],
    )

    result = evaluate_minimum_pressure(
        pressure=pressure,
        minimum_pressure_m=15.0,
        required_compliance_pct=75.0,
    )

    assert result.compliance_pct == pytest.approx(75.0)
    assert result.feasible is True


def test_empty_pressure_table_is_rejected():
    """Reject an empty hydraulic-results table."""
    with pytest.raises(
        ValueError,
        match="pressure cannot be empty",
    ):
        evaluate_minimum_pressure(
            pressure=pd.DataFrame(),
            minimum_pressure_m=15.0,
        )


def test_invalid_velocity_limit_is_rejected():
    """Reject a zero maximum-velocity limit."""
    velocity = pd.DataFrame(
        data=[[1.0]],
        columns=["P1"],
    )

    with pytest.raises(
        ValueError,
        match="greater than zero",
    ):
        evaluate_maximum_velocity(
            velocity=velocity,
            maximum_velocity_mps=0.0,
        )


def test_non_dataframe_pressure_is_rejected():
    """Require pressure results to use a pandas DataFrame."""
    with pytest.raises(
        TypeError,
        match="pandas DataFrame",
    ):
        evaluate_minimum_pressure(
            pressure=[[20.0, 15.0]],
            minimum_pressure_m=15.0,
        )


def test_invalid_compliance_percentage_is_rejected():
    """Reject compliance requirements greater than 100 percent."""
    velocity = pd.DataFrame(
        data=[[1.0]],
        columns=["P1"],
    )

    with pytest.raises(
        ValueError,
        match="between 0 and 100",
    ):
        evaluate_maximum_velocity(
            velocity=velocity,
            maximum_velocity_mps=2.5,
            required_compliance_pct=101.0,
        )

def build_small_network():
    """Create a small network for pipe-design tests."""
    wn = wntr.network.WaterNetworkModel()

    wn.add_reservoir(
        "R1",
        base_head=50.0,
        coordinates=(0.0, 0.0),
    )

    wn.add_junction(
        "J1",
        base_demand=0.01,
        demand_pattern=None,
        elevation=10.0,
        coordinates=(100.0, 0.0),
    )

    wn.add_pipe(
        "P1",
        start_node_name="R1",
        end_node_name="J1",
        length=100.0,
        diameter=0.150,
        roughness=100.0,
        minor_loss=0.0,
    )

    return wn


def test_apply_pipe_design_changes_copied_network():
    """Apply a diameter change to the copied network."""
    wn = build_small_network()

    design = PipeDesign(
        name="Alternative A",
        diameters_m={
            "P1": 0.250,
        },
    )

    trial_wn, audit = apply_pipe_design(
        wn=wn,
        design=design,
    )

    assert trial_wn.get_link("P1").diameter == pytest.approx(
        0.250
    )

    assert len(audit) == 1
    assert audit[0]["pipe_name"] == "P1"
    assert audit[0]["old_diameter_m"] == pytest.approx(
        0.150
    )
    assert audit[0]["new_diameter_m"] == pytest.approx(
        0.250
    )


def test_apply_pipe_design_preserves_original_network():
    """Confirm that the original model is not modified."""
    wn = build_small_network()

    original_diameter = wn.get_link("P1").diameter

    design = PipeDesign(
        name="Alternative A",
        diameters_m={
            "P1": 0.250,
        },
    )

    trial_wn, _ = apply_pipe_design(
        wn=wn,
        design=design,
    )

    assert wn.get_link("P1").diameter == pytest.approx(
        original_diameter
    )

    assert trial_wn.get_link("P1").diameter == pytest.approx(
        0.250
    )

    assert trial_wn is not wn


def test_apply_pipe_design_rejects_missing_pipe():
    """Reject a design containing an unknown pipe name."""
    wn = build_small_network()

    design = PipeDesign(
        name="Invalid alternative",
        diameters_m={
            "MissingPipe": 0.250,
        },
    )

    with pytest.raises(
        KeyError,
        match="does not exist",
    ):
        apply_pipe_design(
            wn=wn,
            design=design,
        )


def test_apply_pipe_design_rejects_negative_diameter():
    """Reject a proposed diameter that is not positive."""
    wn = build_small_network()

    design = PipeDesign(
        name="Invalid alternative",
        diameters_m={
            "P1": -0.250,
        },
    )

    with pytest.raises(
        ValueError,
        match="greater than zero",
    ):
        apply_pipe_design(
            wn=wn,
            design=design,
        )


def test_apply_pipe_design_rejects_empty_design():
    """Reject a design containing no pipe changes."""
    wn = build_small_network()

    design = PipeDesign(
        name="Empty alternative",
        diameters_m={},
    )

    with pytest.raises(
        ValueError,
        match="at least one",
    ):
        apply_pipe_design(
            wn=wn,
            design=design,
        )