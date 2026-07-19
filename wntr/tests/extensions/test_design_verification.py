"""Tests for the design-verification extension."""

import pandas as pd
import pytest

from wntr.extensions.design_verification import (
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