"""Run reproducible batches of hydraulic design-verification experiments."""

from __future__ import annotations

import hashlib
import json
import platform
import time
from dataclasses import asdict
from datetime import datetime, timezone
from itertools import product
from typing import Any, Sequence

import pandas as pd
import wntr
from wntr.network import WaterNetworkModel

from .models import HydraulicScenario, PipeDesign
from .runner import run_design_verification


def _normalise_designs(
    designs: Sequence[PipeDesign | None] | None,
) -> list[PipeDesign | None]:
    """Validate and return the designs used in a batch."""
    if designs is None:
        return [None]

    if isinstance(designs, (str, bytes)):
        raise TypeError(
            "designs must be a sequence of PipeDesign objects or None."
        )

    values = list(designs)

    if not values:
        raise ValueError("designs must contain at least one item.")

    for design in values:
        if design is not None and not isinstance(design, PipeDesign):
            raise TypeError(
                "Each design must be a PipeDesign object or None."
            )

    return values


def _normalise_scenarios(
    scenarios: Sequence[HydraulicScenario],
) -> list[HydraulicScenario]:
    """Validate and return the scenarios used in a batch."""
    if isinstance(scenarios, (str, bytes)):
        raise TypeError(
            "scenarios must be a sequence of HydraulicScenario objects."
        )

    values = list(scenarios)

    if not values:
        raise ValueError("scenarios must contain at least one item.")

    for scenario in values:
        if not isinstance(scenario, HydraulicScenario):
            raise TypeError(
                "Each scenario must be a HydraulicScenario object."
            )

    return values


def _normalise_simulators(
    simulators: Sequence[str] | str,
) -> list[str]:
    """Return simulator requests as a non-empty list."""
    if isinstance(simulators, str):
        values = [simulators]
    else:
        values = list(simulators)

    if not values:
        raise ValueError("simulators must contain at least one item.")

    for simulator in values:
        if not isinstance(simulator, str):
            raise TypeError(
                "Each simulator must be provided as a string."
            )

    return values


def _safe_name(value: object) -> str:
    """Return a readable name without assuming the value is a string."""
    if isinstance(value, str):
        return value.strip()

    return str(value)


def _design_payload(
    design: PipeDesign | None,
) -> dict[str, object] | None:
    """Return a JSON-serializable pipe-design description."""
    if design is None:
        return None

    sorted_diameters = sorted(
        (
            (str(pipe_name), diameter_m)
            for pipe_name, diameter_m
            in design.diameters_m.items()
        ),
        key=lambda item: item[0],
    )

    return {
        "name": design.name,
        "diameters_m": {
            pipe_name: diameter_m
            for pipe_name, diameter_m in sorted_diameters
        },
    }


def _scenario_payload(
    scenario: HydraulicScenario,
) -> dict[str, object]:
    """Return a JSON-serializable scenario description."""
    return dict(asdict(scenario))


def _canonical_json(value: object) -> str:
    """Serialize configuration data deterministically."""
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=str,
    )


def _configuration_hash(
    *,
    design: PipeDesign | None,
    scenario: HydraulicScenario,
    simulator: str,
    minimum_pressure_m: float,
    maximum_velocity_mps: float,
    required_compliance_pct: float,
) -> tuple[str, str, str]:
    """Return a stable hash and serialized configuration records."""
    design_json = _canonical_json(_design_payload(design))
    scenario_json = _canonical_json(_scenario_payload(scenario))

    payload = {
        "design": json.loads(design_json),
        "scenario": json.loads(scenario_json),
        "simulator": simulator.strip().upper(),
        "minimum_pressure_m": float(minimum_pressure_m),
        "maximum_velocity_mps": float(maximum_velocity_mps),
        "required_compliance_pct": float(
            required_compliance_pct
        ),
    }

    digest = hashlib.sha256(
        _canonical_json(payload).encode("utf-8")
    ).hexdigest()

    return digest, design_json, scenario_json


def _base_row(
    *,
    experiment_id: str,
    configuration_hash: str,
    design: PipeDesign | None,
    scenario: HydraulicScenario,
    simulator: str,
    design_json: str,
    scenario_json: str,
    started_at_utc: str,
    elapsed_s: float,
    minimum_pressure_m: float,
    maximum_velocity_mps: float,
    required_compliance_pct: float,
) -> dict[str, object]:
    """Create fields shared by successful and failed experiments."""
    return {
        "experiment_id": experiment_id,
        "configuration_hash": configuration_hash,
        "status": None,
        "started_at_utc": started_at_utc,
        "elapsed_s": elapsed_s,
        "design_name": (
            None
            if design is None
            else _safe_name(design.name)
        ),
        "scenario_name": _safe_name(scenario.name),
        "simulator_name": simulator.strip(),
        "demand_multiplier": scenario.demand_multiplier,
        "demand_model": scenario.demand_model,
        "duration_s": scenario.duration_s,
        "hydraulic_timestep_s": (
            scenario.hydraulic_timestep_s
        ),
        "report_timestep_s": scenario.report_timestep_s,
        "minimum_pressure_limit_m": float(
            minimum_pressure_m
        ),
        "maximum_velocity_limit_mps": float(
            maximum_velocity_mps
        ),
        "required_compliance_pct": float(
            required_compliance_pct
        ),
        "design_configuration_json": design_json,
        "scenario_configuration_json": scenario_json,
        "python_version": platform.python_version(),
        "wntr_version": getattr(
            wntr,
            "__version__",
            "unknown",
        ),
        "platform": platform.platform(),
    }


def run_verification_batch(
    wn: WaterNetworkModel,
    scenarios: Sequence[HydraulicScenario],
    *,
    designs: Sequence[PipeDesign | None] | None = None,
    simulators: Sequence[str] | str = ("WNTR",),
    minimum_pressure_m: float = 15.0,
    maximum_velocity_mps: float = 2.5,
    required_compliance_pct: float = 100.0,
    continue_on_error: bool = True,
    retain_hydraulic_results: bool = False,
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    """Run every requested design-scenario-simulator combination.

    Parameters
    ----------
    wn
        Original WNTR water-distribution network.
    scenarios
        Hydraulic scenarios to assess.
    designs
        Pipe designs to assess. Use ``None`` for the unchanged
        baseline network. When omitted, only the baseline is assessed.
    simulators
        One or more simulator names accepted by
        :func:`run_design_verification`.
    minimum_pressure_m
        Minimum acceptable junction pressure.
    maximum_velocity_mps
        Maximum acceptable absolute pipe velocity.
    required_compliance_pct
        Required pressure and velocity compliance percentage.
    continue_on_error
        Record failed experiments and continue when ``True``. Re-raise
        the first exception when ``False``.
    retain_hydraulic_results
        Retain raw hydraulic result objects in the returned records.

    Returns
    -------
    summary
        One row per requested experiment, including configuration,
        performance, feasibility, timing, and error information.
    records
        Detailed records keyed by experiment ID. Successful records
        include verification and audit objects. Raw hydraulic results
        are included only when ``retain_hydraulic_results`` is true.
    """
    if not isinstance(wn, WaterNetworkModel):
        raise TypeError("wn must be a WNTR WaterNetworkModel.")

    design_values = _normalise_designs(designs)
    scenario_values = _normalise_scenarios(scenarios)
    simulator_values = _normalise_simulators(simulators)

    rows: list[dict[str, object]] = []
    records: dict[str, dict[str, Any]] = {}

    combinations = product(
        design_values,
        scenario_values,
        simulator_values,
    )

    for experiment_number, (
        design,
        scenario,
        simulator,
    ) in enumerate(combinations, start=1):
        (
            configuration_hash,
            design_json,
            scenario_json,
        ) = _configuration_hash(
            design=design,
            scenario=scenario,
            simulator=simulator,
            minimum_pressure_m=minimum_pressure_m,
            maximum_velocity_mps=maximum_velocity_mps,
            required_compliance_pct=required_compliance_pct,
        )

        experiment_id = (
            f"dv-{experiment_number:05d}-"
            f"{configuration_hash[:12]}"
        )

        started_at_utc = datetime.now(
            timezone.utc
        ).isoformat()

        started = time.perf_counter()

        try:
            (
                verification,
                hydraulic_results,
                audit,
            ) = run_design_verification(
                wn=wn,
                scenario=scenario,
                design=design,
                simulator=simulator,
                minimum_pressure_m=minimum_pressure_m,
                maximum_velocity_mps=maximum_velocity_mps,
                required_compliance_pct=(
                    required_compliance_pct
                ),
            )

            elapsed_s = time.perf_counter() - started

            row = _base_row(
                experiment_id=experiment_id,
                configuration_hash=configuration_hash,
                design=design,
                scenario=scenario,
                simulator=verification.simulator_name,
                design_json=design_json,
                scenario_json=scenario_json,
                started_at_utc=started_at_utc,
                elapsed_s=elapsed_s,
                minimum_pressure_m=minimum_pressure_m,
                maximum_velocity_mps=maximum_velocity_mps,
                required_compliance_pct=(
                    required_compliance_pct
                ),
            )

            pressure_result = verification.pressure_result
            velocity_result = verification.velocity_result

            row.update(
                {
                    "status": "completed",
                    "overall_feasible": verification.feasible,
                    "pressure_feasible": (
                        pressure_result.feasible
                    ),
                    "velocity_feasible": (
                        velocity_result.feasible
                    ),
                    "minimum_pressure_m": (
                        pressure_result.critical_value
                    ),
                    "pressure_margin_m": (
                        pressure_result.critical_value
                        - float(minimum_pressure_m)
                    ),
                    "pressure_compliance_pct": (
                        pressure_result.compliance_pct
                    ),
                    "critical_pressure_component": (
                        pressure_result.critical_component
                    ),
                    "critical_pressure_time": (
                        pressure_result.critical_time
                    ),
                    "maximum_velocity_mps": (
                        velocity_result.critical_value
                    ),
                    "velocity_margin_mps": (
                        float(maximum_velocity_mps)
                        - velocity_result.critical_value
                    ),
                    "velocity_compliance_pct": (
                        velocity_result.compliance_pct
                    ),
                    "critical_velocity_component": (
                        velocity_result.critical_component
                    ),
                    "critical_velocity_time": (
                        velocity_result.critical_time
                    ),
                    "simulation_error_code": (
                        verification.simulation_error_code
                    ),
                    "error_type": None,
                    "error_message": None,
                }
            )

            record: dict[str, Any] = {
                "verification": verification,
                "audit": audit,
                "error": None,
            }

            if retain_hydraulic_results:
                record["hydraulic_results"] = (
                    hydraulic_results
                )

            records[experiment_id] = record

        except Exception as error:
            elapsed_s = time.perf_counter() - started

            if not continue_on_error:
                raise

            row = _base_row(
                experiment_id=experiment_id,
                configuration_hash=configuration_hash,
                design=design,
                scenario=scenario,
                simulator=simulator,
                design_json=design_json,
                scenario_json=scenario_json,
                started_at_utc=started_at_utc,
                elapsed_s=elapsed_s,
                minimum_pressure_m=minimum_pressure_m,
                maximum_velocity_mps=maximum_velocity_mps,
                required_compliance_pct=(
                    required_compliance_pct
                ),
            )

            row.update(
                {
                    "status": "failed",
                    "overall_feasible": None,
                    "pressure_feasible": None,
                    "velocity_feasible": None,
                    "minimum_pressure_m": None,
                    "pressure_margin_m": None,
                    "pressure_compliance_pct": None,
                    "critical_pressure_component": None,
                    "critical_pressure_time": None,
                    "maximum_velocity_mps": None,
                    "velocity_margin_mps": None,
                    "velocity_compliance_pct": None,
                    "critical_velocity_component": None,
                    "critical_velocity_time": None,
                    "simulation_error_code": None,
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                }
            )

            records[experiment_id] = {
                "verification": None,
                "audit": None,
                "error": {
                    "type": type(error).__name__,
                    "message": str(error),
                },
            }

        rows.append(row)

    summary = pd.DataFrame(rows)

    return summary, records
