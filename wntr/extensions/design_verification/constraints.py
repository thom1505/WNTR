"""Hydraulic-constraint calculations for design verification."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .exceptions import InvalidConstraintError
from .models import ConstraintResult


def _validate_compliance_percentage(
    required_compliance_pct: float,
) -> float:
    """Validate a required compliance percentage."""
    required = float(required_compliance_pct)

    if not math.isfinite(required):
        raise InvalidConstraintError(
            "required_compliance_pct must be finite."
        )

    if not 0.0 <= required <= 100.0:
        raise InvalidConstraintError(
            "required_compliance_pct must be between 0 and 100."
        )

    return required


def _validate_tolerance(
    value: object,
    *,
    name: str,
) -> float:
    """Return a validated non-negative finite tolerance."""
    try:
        tolerance = float(value)
    except (TypeError, ValueError) as error:
        raise InvalidConstraintError(
            f"{name} must be finite and non-negative."
        ) from error

    if (
        not math.isfinite(tolerance)
        or tolerance < 0.0
    ):
        raise InvalidConstraintError(
            f"{name} must be finite and non-negative."
        )

    return tolerance


def evaluate_minimum_pressure(
    pressure: pd.DataFrame,
    minimum_pressure_m: float,
    required_compliance_pct: float = 100.0,
    pressure_tolerance_m: float = 0.0,
) -> ConstraintResult:
    """Evaluate minimum-pressure compliance.

    Parameters
    ----------
    pressure : pd.DataFrame
        Junction-pressure results. Rows represent simulation times and
        columns represent junction names.
    minimum_pressure_m : float
        Minimum allowable pressure head in metres.
    required_compliance_pct : float, optional
        Percentage of junction-time values required to satisfy the
        pressure limit.
    pressure_tolerance_m : float, optional
        Non-negative numerical tolerance applied below the minimum
        pressure limit. The default of zero preserves strict
        comparison behaviour.

    Returns
    -------
    ConstraintResult
        Pressure-compliance result and critical location.
    """
    values = pressure.to_numpy(dtype=float)

    minimum_pressure = float(minimum_pressure_m)

    if not math.isfinite(minimum_pressure):
        raise InvalidConstraintError(
            "minimum_pressure_m must be finite."
        )

    required = _validate_compliance_percentage(
        required_compliance_pct
    )
    pressure_tolerance = _validate_tolerance(
        pressure_tolerance_m,
        name="pressure_tolerance_m",
    )

    compliant = (
        values
        >= minimum_pressure - pressure_tolerance
    )
    compliance_pct = float(
        compliant.mean() * 100.0
    )

    flat_position = int(np.argmin(values))
    row_position, column_position = np.unravel_index(
        flat_position,
        values.shape,
    )

    return ConstraintResult(
        critical_value=float(
            values[row_position, column_position]
        ),
        compliance_pct=compliance_pct,
        feasible=bool(compliance_pct >= required),
        critical_component=str(
            pressure.columns[column_position]
        ),
        critical_time=pressure.index[row_position],
    )


def evaluate_maximum_velocity(
    velocity: pd.DataFrame,
    maximum_velocity_mps: float,
    required_compliance_pct: float = 100.0,
    velocity_tolerance_mps: float = 0.0,
) -> ConstraintResult:
    """Evaluate maximum absolute pipe-velocity compliance.

    Parameters
    ----------
    velocity : pd.DataFrame
        Pipe-velocity results. Rows represent simulation times and
        columns represent pipe names.
    maximum_velocity_mps : float
        Maximum allowable absolute velocity in metres per second.
    required_compliance_pct : float, optional
        Percentage of pipe-time values required to satisfy the
        velocity limit.
    velocity_tolerance_mps : float, optional
        Non-negative numerical tolerance applied above the maximum
        absolute velocity limit. The default of zero preserves strict
        comparison behaviour.

    Returns
    -------
    ConstraintResult
        Velocity-compliance result and critical location.
    """
    values = velocity.to_numpy(dtype=float)

    maximum_velocity = float(maximum_velocity_mps)

    if (
        not math.isfinite(maximum_velocity)
        or maximum_velocity <= 0.0
    ):
        raise InvalidConstraintError(
            "maximum_velocity_mps must be finite and "
            "greater than zero."
        )

    required = _validate_compliance_percentage(
        required_compliance_pct
    )
    velocity_tolerance = _validate_tolerance(
        velocity_tolerance_mps,
        name="velocity_tolerance_mps",
    )

    # Velocity signs indicate flow direction. Compliance is based on
    # the magnitude of velocity.
    absolute_values = np.abs(values)

    compliant = (
        absolute_values
        <= maximum_velocity + velocity_tolerance
    )
    compliance_pct = float(
        compliant.mean() * 100.0
    )

    flat_position = int(np.argmax(absolute_values))
    row_position, column_position = np.unravel_index(
        flat_position,
        absolute_values.shape,
    )

    return ConstraintResult(
        critical_value=float(
            absolute_values[row_position, column_position]
        ),
        compliance_pct=compliance_pct,
        feasible=bool(compliance_pct >= required),
        critical_component=str(
            velocity.columns[column_position]
        ),
        critical_time=velocity.index[row_position],
    )
