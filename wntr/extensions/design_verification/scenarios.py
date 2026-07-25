"""Apply hydraulic operating scenarios to WNTR models."""

from __future__ import annotations

import copy
import math

from wntr.network import WaterNetworkModel

from .models import HydraulicScenario


def _finite_float(
    value: object,
    parameter_name: str,
) -> float:
    """Convert a value to a finite floating-point number."""
    if isinstance(value, bool):
        raise ValueError(
            f"{parameter_name} must be numeric, not Boolean."
        )

    try:
        converted = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{parameter_name} must be numeric."
        ) from error

    if not math.isfinite(converted):
        raise ValueError(
            f"{parameter_name} must be finite."
        )

    return converted


def _seconds_value(
    value: object,
    parameter_name: str,
    *,
    allow_zero: bool,
) -> int:
    """Validate and convert a simulation time value to seconds."""
    converted = _finite_float(
        value=value,
        parameter_name=parameter_name,
    )

    if not converted.is_integer():
        raise ValueError(
            f"{parameter_name} must be a whole number of seconds."
        )

    converted_int = int(converted)

    if allow_zero:
        if converted_int < 0:
            raise ValueError(
                f"{parameter_name} must be greater than or equal "
                "to zero."
            )
    elif converted_int <= 0:
        raise ValueError(
            f"{parameter_name} must be greater than zero."
        )

    return converted_int


def _canonical_demand_model(value: object) -> str:
    """Return a standard DD or PDD demand-model name."""
    if not isinstance(value, str):
        raise ValueError(
            "demand_model must be a string."
        )

    normalized = value.strip().upper()

    aliases = {
        "DD": "DD",
        "DDA": "DD",
        "PDD": "PDD",
        "PDA": "PDD",
    }

    if normalized not in aliases:
        raise ValueError(
            "demand_model must be one of DD, DDA, PDD, or PDA."
        )

    return aliases[normalized]


def apply_hydraulic_scenario(
    wn: WaterNetworkModel,
    scenario: HydraulicScenario,
) -> tuple[WaterNetworkModel, dict[str, object]]:
    """Apply a hydraulic scenario to an independent network copy.

    The original ``WaterNetworkModel`` is not modified.

    Parameters
    ----------
    wn
        Original WNTR water-distribution network model.
    scenario
        Hydraulic operating conditions to apply.

    Returns
    -------
    scenario_wn
        Independent network copy containing the scenario settings.
    audit
        Dictionary recording the original and applied settings.

    Raises
    ------
    TypeError
        If ``wn`` is not a WaterNetworkModel or ``scenario`` is not a
        HydraulicScenario.
    ValueError
        If the scenario contains invalid names, hydraulic settings,
        pressure settings, durations, or timesteps.
    """
    if not isinstance(wn, WaterNetworkModel):
        raise TypeError(
            "wn must be a WNTR WaterNetworkModel."
        )

    if not isinstance(scenario, HydraulicScenario):
        raise TypeError(
            "scenario must be a HydraulicScenario."
        )

    if (
        not isinstance(scenario.name, str)
        or not scenario.name.strip()
    ):
        raise ValueError(
            "scenario name cannot be empty."
        )

    demand_multiplier = _finite_float(
        value=scenario.demand_multiplier,
        parameter_name="demand_multiplier",
    )

    if demand_multiplier <= 0.0:
        raise ValueError(
            "demand_multiplier must be greater than zero."
        )

    demand_model = _canonical_demand_model(
        scenario.demand_model
    )

    duration_s = _seconds_value(
        value=scenario.duration_s,
        parameter_name="duration_s",
        allow_zero=True,
    )

    hydraulic_timestep_s = _seconds_value(
        value=scenario.hydraulic_timestep_s,
        parameter_name="hydraulic_timestep_s",
        allow_zero=False,
    )

    report_timestep_s = _seconds_value(
        value=scenario.report_timestep_s,
        parameter_name="report_timestep_s",
        allow_zero=False,
    )

    scenario_wn = copy.deepcopy(wn)

    old_settings = {
        "demand_multiplier": float(
            scenario_wn.options.hydraulic.demand_multiplier
        ),
        "demand_model": str(
            scenario_wn.options.hydraulic.demand_model
        ),
        "duration_s": int(
            scenario_wn.options.time.duration
        ),
        "hydraulic_timestep_s": int(
            scenario_wn.options.time.hydraulic_timestep
        ),
        "report_timestep_s": int(
            scenario_wn.options.time.report_timestep
        ),
        "minimum_pressure_m": float(
            scenario_wn.options.hydraulic.minimum_pressure
        ),
        "required_pressure_m": float(
            scenario_wn.options.hydraulic.required_pressure
        ),
        "pressure_exponent": float(
            scenario_wn.options.hydraulic.pressure_exponent
        ),
    }

    if scenario.minimum_pressure_m is None:
        minimum_pressure_m = old_settings[
            "minimum_pressure_m"
        ]
    else:
        minimum_pressure_m = _finite_float(
            value=scenario.minimum_pressure_m,
            parameter_name="minimum_pressure_m",
        )

        if minimum_pressure_m < 0.0:
            raise ValueError(
                "minimum_pressure_m must be greater than or equal "
                "to zero."
            )

    if scenario.required_pressure_m is None:
        required_pressure_m = old_settings[
            "required_pressure_m"
        ]
    else:
        required_pressure_m = _finite_float(
            value=scenario.required_pressure_m,
            parameter_name="required_pressure_m",
        )

        if required_pressure_m < 0.0:
            raise ValueError(
                "required_pressure_m must be greater than or equal "
                "to zero."
            )
    if scenario.pressure_exponent is None:
        pressure_exponent = old_settings[
            "pressure_exponent"
        ]
    else:
        pressure_exponent = _finite_float(
            value=scenario.pressure_exponent,
            parameter_name="pressure_exponent",
        )

        if pressure_exponent <= 0.0:
            raise ValueError(
                "pressure_exponent must be greater than zero."
            )

    if required_pressure_m <= minimum_pressure_m:
        raise ValueError(
            "required_pressure_m must be greater than "
            "minimum_pressure_m."
        )

    scenario_wn.options.hydraulic.demand_multiplier = (
        demand_multiplier
    )
    scenario_wn.options.hydraulic.demand_model = demand_model

    scenario_wn.options.time.duration = duration_s
    scenario_wn.options.time.hydraulic_timestep = (
        hydraulic_timestep_s
    )
    scenario_wn.options.time.report_timestep = (
        report_timestep_s
    )

    scenario_wn.options.hydraulic.minimum_pressure = (
        minimum_pressure_m
    )
    scenario_wn.options.hydraulic.required_pressure = (
        required_pressure_m
    )
    scenario_wn.options.hydraulic.pressure_exponent = (
        pressure_exponent
    )

    new_settings = {
        "demand_multiplier": float(
            scenario_wn.options.hydraulic.demand_multiplier
        ),
        "demand_model": str(
            scenario_wn.options.hydraulic.demand_model
        ),
        "duration_s": int(
            scenario_wn.options.time.duration
        ),
        "hydraulic_timestep_s": int(
            scenario_wn.options.time.hydraulic_timestep
        ),
        "report_timestep_s": int(
            scenario_wn.options.time.report_timestep
        ),
        "minimum_pressure_m": float(
            scenario_wn.options.hydraulic.minimum_pressure
        ),
        "required_pressure_m": float(
            scenario_wn.options.hydraulic.required_pressure
        ),
        "pressure_exponent": float(
            scenario_wn.options.hydraulic.pressure_exponent
        ),
    }

    audit = {
        "scenario_name": scenario.name.strip(),
        "original_settings": old_settings,
        "applied_settings": new_settings,
    }

    return scenario_wn, audit
