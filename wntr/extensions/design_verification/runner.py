"""Run hydraulic simulations and verify design constraints."""

from __future__ import annotations

from dataclasses import asdict
import tempfile
from pathlib import Path
from typing import Any

from wntr.network import WaterNetworkModel
from wntr.sim import EpanetSimulator, WNTRSimulator

from .constraints import (
    evaluate_maximum_velocity,
    evaluate_minimum_pressure,
)
from .exceptions import (
    IncompleteHydraulicResultsError,
    InvalidConstraintError,
    UnsupportedSimulatorError,
)
from .design import apply_pipe_design
from .models import (
    HydraulicScenario,
    PipeDesign,
    VerificationResult,
)
from .pumps import audit_head_pump_curves
from .scenarios import apply_hydraulic_scenario


def _canonical_simulator_name(value: object) -> str:
    """Return a standardized hydraulic-simulator name."""
    if not isinstance(value, str):
        raise UnsupportedSimulatorError(
            "simulator must be a string."
        )

    normalized = value.strip().upper()

    aliases = {
        "WNTR": "WNTR",
        "WNTRSIMULATOR": "WNTR",
        "WNTR SIMULATOR": "WNTR",
        "EPANET": "EPANET",
        "EPANETSIMULATOR": "EPANET",
        "EPANET SIMULATOR": "EPANET",
    }

    if normalized not in aliases:
        raise UnsupportedSimulatorError(
            "simulator must be WNTR or EPANET."
        )

    return aliases[normalized]


def _select_result_columns(
    table: Any,
    component_names: list[str],
    result_name: str,
):
    """Select complete results for all required network components."""
    if not hasattr(table, "columns"):
        raise TypeError(
            f"{result_name} results must be a pandas DataFrame."
        )

    required_names = list(dict.fromkeys(component_names))

    if not required_names:
        raise InvalidConstraintError(
            f"No applicable components exist for {result_name} "
            "verification."
        )

    duplicate_columns = set(
        table.columns[
            table.columns.duplicated(keep=False)
        ]
    )

    duplicated_required = [
        name
        for name in required_names
        if name in duplicate_columns
    ]

    if duplicated_required:
        names = ", ".join(
            repr(name)
            for name in duplicated_required
        )
        raise IncompleteHydraulicResultsError(
            f"The {result_name} results contain duplicate "
            f"columns for required components: {names}."
        )

    missing_names = [
        name
        for name in required_names
        if name not in table.columns
    ]

    if missing_names:
        preview_limit = 10
        preview = ", ".join(
            repr(name)
            for name in missing_names[:preview_limit]
        )
        remaining = len(missing_names) - preview_limit

        if remaining > 0:
            preview += f", and {remaining} more"

        raise IncompleteHydraulicResultsError(
            f"The {result_name} results are missing "
            f"{len(missing_names)} required component(s): "
            f"{preview}."
        )

    return table.loc[:, required_names]


def _simulation_error_code(results: Any) -> int | None:
    """Extract a numeric simulator error code when available."""
    value = getattr(
        results,
        "error_code",
        None,
    )

    if value is None:
        return None

    value = getattr(
        value,
        "value",
        value,
    )

    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def run_design_verification(
    wn: WaterNetworkModel,
    scenario: HydraulicScenario,
    *,
    design: PipeDesign | None = None,
    simulator: str = "WNTR",
    minimum_pressure_m: float = 15.0,
    maximum_velocity_mps: float = 2.5,
    required_compliance_pct: float = 100.0,
    pressure_tolerance_m: float = 0.0,
    velocity_tolerance_mps: float = 0.0,
) -> tuple[
    VerificationResult,
    Any,
    dict[str, object],
]:
    """Run one hydraulic design-verification assessment.

    The original network is protected from modification. Pipe-design
    and hydraulic-scenario changes are applied to independent copies.

    Parameters
    ----------
    wn
        Original WNTR water-distribution network model.
    scenario
        Hydraulic operating conditions to assess.
    design
        Optional pipe-diameter design. If omitted, the unchanged
        baseline network is assessed.
    simulator
        Hydraulic simulation engine. Accepted values are ``WNTR`` and
        ``EPANET``.
    minimum_pressure_m
        Minimum acceptable junction pressure in metres.
    maximum_velocity_mps
        Maximum acceptable absolute pipe velocity in metres per
        second.
    required_compliance_pct
        Required percentage of assessed node-time and pipe-time
        values satisfying each constraint.
    pressure_tolerance_m
        Non-negative numerical tolerance applied below the minimum
        pressure limit. The default of zero preserves strict
        comparison behaviour.
    velocity_tolerance_mps
        Non-negative numerical tolerance applied above the maximum
        absolute velocity limit. The default of zero preserves strict
        comparison behaviour.

    Returns
    -------
    verification
        Structured pressure, velocity and overall feasibility result.
    hydraulic_results
        Raw WNTR hydraulic simulation results.
    audit
        Record of the applied design, scenario, simulator and assessed
        components.

    Raises
    ------
    TypeError
        If network, scenario or design inputs have incorrect types.
    ValueError
        If the simulator is unsupported or no applicable junction or
        pipe results are available.
    RuntimeError
        If expected pressure or velocity results are absent.
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
        design is not None
        and not isinstance(design, PipeDesign)
    ):
        raise TypeError(
            "design must be a PipeDesign or None."
        )

    simulator_name = _canonical_simulator_name(
        simulator
    )

    # Apply a pipe design to an independent copy when one is supplied.
    if design is None:
        design_wn = wn
        design_audit = None
        design_name = None
    else:
        design_wn, design_audit = apply_pipe_design(
            wn=wn,
            design=design,
        )
        design_name = design.name.strip()

    # This creates another independent copy containing the scenario.
    scenario_wn, scenario_audit = (
        apply_hydraulic_scenario(
            wn=design_wn,
            scenario=scenario,
        )
    )

    if simulator_name == "WNTR":
        hydraulic_simulator = WNTRSimulator(
            scenario_wn
        )

        hydraulic_results = (
            hydraulic_simulator.run_sim(
                convergence_error=True,
            )
        )

    else:
        # EPANET creates temporary input, report and binary files.
        # A temporary directory prevents those files from cluttering
        # the project repository.
        with tempfile.TemporaryDirectory(
            prefix="wntr_design_verification_"
        ) as temporary_directory:
            file_prefix = str(
                Path(temporary_directory)
                / "verification"
            )

            hydraulic_simulator = EpanetSimulator(
                scenario_wn
            )

            hydraulic_results = (
                hydraulic_simulator.run_sim(
                    file_prefix=file_prefix,
                    version=2.2,
                    convergence_error=True,
                )
            )

    try:
        pressure = hydraulic_results.node[
            "pressure"
        ]
    except (AttributeError, KeyError) as error:
        raise IncompleteHydraulicResultsError(
            "The simulation did not return node-pressure results."
        ) from error

    try:
        velocity = hydraulic_results.link[
            "velocity"
        ]
    except (AttributeError, KeyError) as error:
        raise IncompleteHydraulicResultsError(
            "The simulation did not return link-velocity results."
        ) from error
    try:
        flowrate = hydraulic_results.link[
            "flowrate"
        ]
    except (AttributeError, KeyError) as error:
        raise IncompleteHydraulicResultsError(
            "The simulation did not return link-flowrate results."
        ) from error

    try:
        status = hydraulic_results.link[
            "status"
        ]
    except (AttributeError, KeyError):
        status = None

    try:
        setting = hydraulic_results.link[
            "setting"
        ]
    except (AttributeError, KeyError):
        setting = None

    # Pressure requirements apply to demand junctions, not reservoirs
    # or tanks.
    junction_pressure = _select_result_columns(
        table=pressure,
        component_names=list(
            scenario_wn.junction_name_list
        ),
        result_name="pressure",
    )

    # Pipe-velocity requirements apply to pipes, not pumps or valves.
    pipe_velocity = _select_result_columns(
        table=velocity,
        component_names=list(
            scenario_wn.pipe_name_list
        ),
        result_name="velocity",
    )

    pressure_result = evaluate_minimum_pressure(
        pressure=junction_pressure,
        minimum_pressure_m=minimum_pressure_m,
        required_compliance_pct=(
            required_compliance_pct
        ),
        pressure_tolerance_m=pressure_tolerance_m,
    )

    velocity_result = evaluate_maximum_velocity(
        velocity=pipe_velocity,
        maximum_velocity_mps=maximum_velocity_mps,
        required_compliance_pct=(
            required_compliance_pct
        ),
        velocity_tolerance_mps=velocity_tolerance_mps,
    )

    pump_result = audit_head_pump_curves(
        network=scenario_wn,
        flowrate=flowrate,
        status=status,
        setting=setting,
    )

    verification = VerificationResult(
        design_name=design_name,
        scenario_name=scenario.name.strip(),
        simulator_name=simulator_name,
        pressure_result=pressure_result,
        velocity_result=velocity_result,
        pump_result=pump_result,
        feasible=bool(
            pressure_result.feasible
            and velocity_result.feasible
            and pump_result.all_pumps_passed
        ),
        simulation_error_code=(
            _simulation_error_code(
                hydraulic_results
            )
        ),
    )

    audit: dict[str, object] = {
        "design": design_audit,
        "scenario": scenario_audit,
        "pump_curve_audit": asdict(
            pump_result
        ),
        "simulator_name": simulator_name,
        "minimum_pressure_m": float(
            minimum_pressure_m
        ),
        "maximum_velocity_mps": float(
            maximum_velocity_mps
        ),
        "required_compliance_pct": float(
            required_compliance_pct
        ),
        "pressure_tolerance_m": float(
            pressure_tolerance_m
        ),
        "velocity_tolerance_mps": float(
            velocity_tolerance_mps
        ),
        "expected_junction_count": len(
            scenario_wn.junction_name_list
        ),
        "assessed_junction_count": len(
            junction_pressure.columns
        ),
        "junctions_assessed": list(
            junction_pressure.columns
        ),
        "expected_pipe_count": len(
            scenario_wn.pipe_name_list
        ),
        "assessed_pipe_count": len(
            pipe_velocity.columns
        ),
        "pipes_assessed": list(
            pipe_velocity.columns
        ),
    }

    return (
        verification,
        hydraulic_results,
        audit,
    )