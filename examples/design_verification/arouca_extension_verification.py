"""Build the Arouca WNTR model and test the custom design-verification extension.

This script reconstructs the conceptual Arouca Highlift, Windy Hill, and
Bon Air North water-distribution model described in Rheal Thomas's MSc
Data Science final project.

It is designed to run against the custom extension on:

    thom1505/WNTR
    branch: feature/design-verification

The workflow:

1. Builds a WaterNetworkModel entirely in Python.
2. Defines the optimized five-pipe design from the project.
3. Applies the design through ``PipeDesign``.
4. Applies operating scenarios through ``HydraulicScenario``.
5. Runs ``run_design_verification``.
6. Evaluates minimum junction pressure and maximum absolute pipe velocity.
7. Runs the baseline and seven stress-test cases.
8. Exports CSV result tables and EPANET INP model files.
9. Confirms that the original WaterNetworkModel is not modified.

Important
---------
This is a conceptual research model based on the assumptions reported in
the MSc project. It is not a field-calibrated operational model. Surveyed
elevations, verified pipe data, calibrated demands, actual pump curves,
SCADA records, valve states, and field pressure/flow measurements are
required before construction or operational decisions are made.
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import wntr

try:
    from wntr.extensions.design_verification import (
        HydraulicScenario,
        PipeDesign,
        apply_hydraulic_scenario,
        apply_pipe_design,
        run_design_verification,
    )
except ImportError as exc:
    raise ImportError(
        "The design-verification extension could not be imported. "
        "Activate the 'wntr-extension' Conda environment and run this "
        "script with the local thom1505/WNTR "
        "'feature/design-verification' branch."
    ) from exc


# =====================================================================
# 1. ENGINEERING UNITS AND ACCEPTANCE CRITERIA
# =====================================================================

UK_GAL_TO_M3 = 0.00454609
IMGD_TO_M3S = 4546.09 / 86400.0

MINIMUM_PRESSURE_M = 15.0
MAXIMUM_VELOCITY_MPS = 2.5
REQUIRED_COMPLIANCE_PCT = 100.0

DURATION_HOURS = 120
HYDRAULIC_TIMESTEP_S = 3600
REPORT_TIMESTEP_S = 3600

PVC_ROUGHNESS_C = 140.0
SOURCE_HEAD_M = 35.0
PUMP_HEAD_BASE_M = 127.0

TANK_MIN_LEVEL_M = 1.0
TANK_INITIAL_LEVEL_M = 7.0
TANK_MAX_LEVEL_M = 10.0

WINDY_HILL_TANK_ELEVATION_M = 127.0
BON_AIR_TANK_ELEVATION_M = 85.0

WINDY_HILL_TANK_VOLUME_M3 = 100_000.0 * UK_GAL_TO_M3
BON_AIR_TANK_VOLUME_M3 = 100_000.0 * UK_GAL_TO_M3

DAILY_PATTERN = [
    0.60, 0.60, 0.60, 0.65, 0.75, 0.90,
    1.10, 1.25, 1.20, 1.10, 1.00, 0.95,
    0.90, 0.90, 0.95, 1.05, 1.15, 1.25,
    1.30, 1.20, 1.05, 0.90, 0.75, 0.65,
]


# =====================================================================
# 2. CASE-STUDY DEMANDS
# =====================================================================

EXISTING_ZONE_DEMAND_IMGD = 0.41
WINDY_HILL_DEMAND_IMGD = 0.26
BON_AIR_NORTH_DEMAND_IMGD = 0.20

EXISTING_ZONE_DEMAND_M3S = (
    EXISTING_ZONE_DEMAND_IMGD * IMGD_TO_M3S
)
WINDY_HILL_DEMAND_M3S = (
    WINDY_HILL_DEMAND_IMGD * IMGD_TO_M3S
)
BON_AIR_NORTH_DEMAND_M3S = (
    BON_AIR_NORTH_DEMAND_IMGD * IMGD_TO_M3S
)

TOTAL_DEMAND_IMGD = (
    EXISTING_ZONE_DEMAND_IMGD
    + WINDY_HILL_DEMAND_IMGD
    + BON_AIR_NORTH_DEMAND_IMGD
)


# =====================================================================
# 3. PIPE DATA AND OPTIMIZED DESIGN
# =====================================================================

PIPE_LENGTHS_M = {
    "P_Proposed_Main": 2700.0,
    "P_Windy_to_BonAirTank": 2200.0,
    "P_Windy_Tank_to_Demand": 700.0,
    "P_BonAir_Tank_to_Demand": 600.0,
    "P_Pump_to_Existing": 1500.0,
}

# The base model is intentionally created with a neutral 0.150 m
# diameter for all five pipes. The custom extension then applies the
# optimized design to a copied model. This directly tests the extension.
BASE_PIPE_DIAMETERS_M = {
    pipe_name: 0.150
    for pipe_name in PIPE_LENGTHS_M
}

OPTIMIZED_PIPE_DIAMETERS_M = {
    "P_Proposed_Main": 0.200,
    "P_Windy_to_BonAirTank": 0.100,
    "P_Windy_Tank_to_Demand": 0.150,
    "P_BonAir_Tank_to_Demand": 0.150,
    "P_Pump_to_Existing": 0.150,
}

# Values reported by the MSc project for the optimized baseline design.
# The script reports differences rather than requiring exact equality,
# because simulator choice and software versions can cause small changes.
REFERENCE_MINIMUM_PRESSURE_M = 15.95
REFERENCE_MAXIMUM_VELOCITY_MPS = 1.59


# =====================================================================
# 4. VALIDATION AND STRESS-TEST CASES
# =====================================================================

VALIDATION_CASES: list[dict[str, float | str]] = [
    {
        "name": "Baseline design case",
        "demand_multiplier": 1.00,
        "roughness_c": 140.0,
        "pump_head_multiplier": 1.00,
        "source_head_m": 35.0,
    },
    {
        "name": "Demand increased by 10%",
        "demand_multiplier": 1.10,
        "roughness_c": 140.0,
        "pump_head_multiplier": 1.00,
        "source_head_m": 35.0,
    },
    {
        "name": "Demand increased by 20%",
        "demand_multiplier": 1.20,
        "roughness_c": 140.0,
        "pump_head_multiplier": 1.00,
        "source_head_m": 35.0,
    },
    {
        "name": "Demand increased by 30%",
        "demand_multiplier": 1.30,
        "roughness_c": 140.0,
        "pump_head_multiplier": 1.00,
        "source_head_m": 35.0,
    },
    {
        "name": "Aged pipes - roughness C=120",
        "demand_multiplier": 1.00,
        "roughness_c": 120.0,
        "pump_head_multiplier": 1.00,
        "source_head_m": 35.0,
    },
    {
        "name": "Pump head reduced by 10%",
        "demand_multiplier": 1.00,
        "roughness_c": 140.0,
        "pump_head_multiplier": 0.90,
        "source_head_m": 35.0,
    },
    {
        "name": "Source head reduced by 3 m",
        "demand_multiplier": 1.00,
        "roughness_c": 140.0,
        "pump_head_multiplier": 1.00,
        "source_head_m": 32.0,
    },
    {
        "name": "Combined adverse case",
        "demand_multiplier": 1.20,
        "roughness_c": 120.0,
        "pump_head_multiplier": 0.95,
        "source_head_m": 35.0,
    },
]


# =====================================================================
# 5. MODEL-BUILDING FUNCTIONS
# =====================================================================

def tank_diameter_from_volume(
    volume_m3: float,
    maximum_level_m: float,
) -> float:
    """Calculate cylindrical tank diameter from volume and water depth."""
    if volume_m3 <= 0.0:
        raise ValueError("Tank volume must be positive.")
    if maximum_level_m <= 0.0:
        raise ValueError("Maximum tank level must be positive.")

    return math.sqrt(
        (4.0 * volume_m3)
        / (math.pi * maximum_level_m)
    )


def build_arouca_network(
    *,
    roughness_c: float = PVC_ROUGHNESS_C,
    pump_head_multiplier: float = 1.0,
    source_head_m: float = SOURCE_HEAD_M,
) -> wntr.network.WaterNetworkModel:
    """Build the conceptual Arouca Highlift network in Python."""

    if roughness_c <= 0.0:
        raise ValueError("Pipe roughness must be positive.")
    if pump_head_multiplier <= 0.0:
        raise ValueError("Pump head multiplier must be positive.")

    wn = wntr.network.WaterNetworkModel()

    # These options make the base model self-describing. The custom
    # HydraulicScenario will apply them again to an independent copy.
    wn.options.time.duration = DURATION_HOURS * 3600
    wn.options.time.hydraulic_timestep = HYDRAULIC_TIMESTEP_S
    wn.options.time.report_timestep = REPORT_TIMESTEP_S
    wn.options.hydraulic.demand_model = "DDA"

    wn.add_pattern("daily", DAILY_PATTERN)

    # Arouca Highlift source.
    wn.add_reservoir(
        "Arouca_Source",
        base_head=float(source_head_m),
        coordinates=(0.0, 0.0),
    )

    # Pump discharge node.
    wn.add_junction(
        "Pump_Discharge",
        elevation=35.0,
        base_demand=0.0,
        demand_pattern=None,
        coordinates=(300.0, 0.0),
    )

    # Highlift pump curve.
    pump_head = PUMP_HEAD_BASE_M * pump_head_multiplier

    wn.add_curve(
        "Highlift_Pump_Curve",
        "HEAD",
        [
            (0.000, pump_head),
            (0.020, pump_head - 2.0),
            (0.028, pump_head - 5.0),
            (0.040, pump_head - 12.0),
        ],
    )

    wn.add_pump(
        "Arouca_Highlift_Pump",
        "Arouca_Source",
        "Pump_Discharge",
        pump_type="HEAD",
        pump_parameter="Highlift_Pump_Curve",
    )

    # Tank geometry is calculated from the assumed 100,000 UK gallon
    # storage volume and 10 m maximum operating depth.
    windy_hill_tank_diameter_m = tank_diameter_from_volume(
        WINDY_HILL_TANK_VOLUME_M3,
        TANK_MAX_LEVEL_M,
    )
    bon_air_tank_diameter_m = tank_diameter_from_volume(
        BON_AIR_TANK_VOLUME_M3,
        TANK_MAX_LEVEL_M,
    )

    wn.add_tank(
        "Windy_Hill_Tank",
        elevation=WINDY_HILL_TANK_ELEVATION_M,
        init_level=TANK_INITIAL_LEVEL_M,
        min_level=TANK_MIN_LEVEL_M,
        max_level=TANK_MAX_LEVEL_M,
        diameter=windy_hill_tank_diameter_m,
        coordinates=(2700.0, 800.0),
    )

    wn.add_tank(
        "Bon_Air_North_Tank",
        elevation=BON_AIR_TANK_ELEVATION_M,
        init_level=TANK_INITIAL_LEVEL_M,
        min_level=TANK_MIN_LEVEL_M,
        max_level=TANK_MAX_LEVEL_M,
        diameter=bon_air_tank_diameter_m,
        coordinates=(2200.0, 1200.0),
    )

    # Demand nodes.
    wn.add_junction(
        "Existing_Zone",
        elevation=65.0,
        base_demand=EXISTING_ZONE_DEMAND_M3S,
        demand_pattern="daily",
        coordinates=(1500.0, -300.0),
    )

    wn.add_junction(
        "Windy_Hill_Demand",
        elevation=110.0,
        base_demand=WINDY_HILL_DEMAND_M3S,
        demand_pattern="daily",
        coordinates=(2800.0, 600.0),
    )

    wn.add_junction(
        "Bon_Air_North_Demand",
        elevation=75.0,
        base_demand=BON_AIR_NORTH_DEMAND_M3S,
        demand_pattern="daily",
        coordinates=(2400.0, 1100.0),
    )

    # Five modelled pipes. Base diameters are intentionally replaced later
    # through the custom PipeDesign extension.
    wn.add_pipe(
        "P_Proposed_Main",
        "Pump_Discharge",
        "Windy_Hill_Tank",
        length=PIPE_LENGTHS_M["P_Proposed_Main"],
        diameter=BASE_PIPE_DIAMETERS_M["P_Proposed_Main"],
        roughness=roughness_c,
        minor_loss=0.0,
        initial_status="OPEN",
    )

    wn.add_pipe(
        "P_Windy_to_BonAirTank",
        "Windy_Hill_Tank",
        "Bon_Air_North_Tank",
        length=PIPE_LENGTHS_M["P_Windy_to_BonAirTank"],
        diameter=BASE_PIPE_DIAMETERS_M["P_Windy_to_BonAirTank"],
        roughness=roughness_c,
        minor_loss=0.0,
        initial_status="OPEN",
    )

    wn.add_pipe(
        "P_Windy_Tank_to_Demand",
        "Windy_Hill_Tank",
        "Windy_Hill_Demand",
        length=PIPE_LENGTHS_M["P_Windy_Tank_to_Demand"],
        diameter=BASE_PIPE_DIAMETERS_M[
            "P_Windy_Tank_to_Demand"
        ],
        roughness=roughness_c,
        minor_loss=0.0,
        initial_status="OPEN",
    )

    wn.add_pipe(
        "P_BonAir_Tank_to_Demand",
        "Bon_Air_North_Tank",
        "Bon_Air_North_Demand",
        length=PIPE_LENGTHS_M[
            "P_BonAir_Tank_to_Demand"
        ],
        diameter=BASE_PIPE_DIAMETERS_M[
            "P_BonAir_Tank_to_Demand"
        ],
        roughness=roughness_c,
        minor_loss=0.0,
        initial_status="OPEN",
    )

    wn.add_pipe(
        "P_Pump_to_Existing",
        "Pump_Discharge",
        "Existing_Zone",
        length=PIPE_LENGTHS_M["P_Pump_to_Existing"],
        diameter=BASE_PIPE_DIAMETERS_M[
            "P_Pump_to_Existing"
        ],
        roughness=roughness_c,
        minor_loss=0.0,
        initial_status="OPEN",
    )

    return wn


def build_optimized_design() -> PipeDesign:
    """Return the optimized pipe design reported in the project."""
    return PipeDesign(
        name="Arouca optimized design",
        diameters_m=OPTIMIZED_PIPE_DIAMETERS_M,
    )


def build_hydraulic_scenario(
    *,
    name: str,
    demand_multiplier: float,
) -> HydraulicScenario:
    """Create one 120-hour demand-driven operating scenario."""
    return HydraulicScenario(
        name=name,
        demand_multiplier=float(demand_multiplier),
        demand_model="DD",
        duration_s=DURATION_HOURS * 3600,
        hydraulic_timestep_s=HYDRAULIC_TIMESTEP_S,
        report_timestep_s=REPORT_TIMESTEP_S,
    )


# =====================================================================
# 6. REPORTING AND EXPORT FUNCTIONS
# =====================================================================

def safe_filename(value: str) -> str:
    """Convert a scenario name into a safe file stem."""
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_") or "scenario"


def time_to_hours(value: Any) -> float | str:
    """Convert a numeric simulation time in seconds to hours."""
    try:
        return float(value) / 3600.0
    except (TypeError, ValueError):
        return str(value)


def record_original_diameters(
    wn: wntr.network.WaterNetworkModel,
) -> dict[str, float]:
    """Record base-network diameters for the mutation-protection check."""
    return {
        pipe_name: float(wn.get_link(pipe_name).diameter)
        for pipe_name in PIPE_LENGTHS_M
    }


def verify_original_network_preserved(
    wn: wntr.network.WaterNetworkModel,
    original_diameters: dict[str, float],
) -> bool:
    """Confirm that extension runs did not modify the supplied network."""
    for pipe_name, original_diameter in original_diameters.items():
        current_diameter = float(
            wn.get_link(pipe_name).diameter
        )
        if not math.isclose(
            current_diameter,
            original_diameter,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            return False

    return True


def create_assessed_network(
    base_wn: wntr.network.WaterNetworkModel,
    design: PipeDesign,
    scenario: HydraulicScenario,
) -> wntr.network.WaterNetworkModel:
    """Recreate the copied network assessed by the extension."""
    design_wn, _ = apply_pipe_design(
        wn=base_wn,
        design=design,
    )
    scenario_wn, _ = apply_hydraulic_scenario(
        wn=design_wn,
        scenario=scenario,
    )
    return scenario_wn


def calculate_tank_levels(
    hydraulic_results: Any,
) -> pd.DataFrame:
    """Return tank levels derived from simulated hydraulic head."""
    head = hydraulic_results.node["head"]

    tank_levels = pd.DataFrame(index=head.index)
    tank_levels.index.name = "time_s"

    tank_levels["time_hr"] = (
        tank_levels.index.astype(float) / 3600.0
    )
    tank_levels["Windy_Hill_Tank_Level_m"] = (
        head["Windy_Hill_Tank"]
        - WINDY_HILL_TANK_ELEVATION_M
    )
    tank_levels["Bon_Air_North_Tank_Level_m"] = (
        head["Bon_Air_North_Tank"]
        - BON_AIR_TANK_ELEVATION_M
    )

    return tank_levels


def create_summary_row(
    *,
    case: dict[str, float | str],
    verification: Any,
    hydraulic_results: Any,
    original_network_preserved: bool,
) -> dict[str, object]:
    """Create one tabular validation summary row."""

    pressure_result = verification.pressure_result
    velocity_result = verification.velocity_result
    tank_levels = calculate_tank_levels(hydraulic_results)

    return {
        "Scenario": str(case["name"]),
        "Demand Multiplier": float(
            case["demand_multiplier"]
        ),
        "Roughness C": float(case["roughness_c"]),
        "Pump Head Multiplier": float(
            case["pump_head_multiplier"]
        ),
        "Source Head (m)": float(case["source_head_m"]),
        "Simulator": verification.simulator_name,
        "Design": verification.design_name,
        "Minimum Pressure (m)": (
            pressure_result.critical_value
        ),
        "Critical Pressure Junction": (
            pressure_result.critical_component
        ),
        "Critical Pressure Time (s)": (
            pressure_result.critical_time
        ),
        "Critical Pressure Time (hr)": time_to_hours(
            pressure_result.critical_time
        ),
        "Pressure Compliance (%)": (
            pressure_result.compliance_pct
        ),
        "Pressure Feasible": pressure_result.feasible,
        "Maximum Velocity (m/s)": (
            velocity_result.critical_value
        ),
        "Critical Velocity Pipe": (
            velocity_result.critical_component
        ),
        "Critical Velocity Time (s)": (
            velocity_result.critical_time
        ),
        "Critical Velocity Time (hr)": time_to_hours(
            velocity_result.critical_time
        ),
        "Velocity Compliance (%)": (
            velocity_result.compliance_pct
        ),
        "Velocity Feasible": velocity_result.feasible,
        "Overall Feasible": verification.feasible,
        "Simulation Error Code": (
            verification.simulation_error_code
        ),
        "Windy Hill Minimum Tank Level (m)": float(
            tank_levels[
                "Windy_Hill_Tank_Level_m"
            ].min()
        ),
        "Windy Hill Maximum Tank Level (m)": float(
            tank_levels[
                "Windy_Hill_Tank_Level_m"
            ].max()
        ),
        "Bon Air Minimum Tank Level (m)": float(
            tank_levels[
                "Bon_Air_North_Tank_Level_m"
            ].min()
        ),
        "Bon Air Maximum Tank Level (m)": float(
            tank_levels[
                "Bon_Air_North_Tank_Level_m"
            ].max()
        ),
        "Original Network Preserved": (
            original_network_preserved
        ),
    }


def export_case_outputs(
    *,
    output_directory: Path,
    case_name: str,
    assessed_wn: wntr.network.WaterNetworkModel,
    hydraulic_results: Any,
    audit: dict[str, object],
) -> None:
    """Export raw results, audit information, and the assessed INP file."""

    case_stem = safe_filename(case_name)
    case_directory = output_directory / case_stem
    case_directory.mkdir(parents=True, exist_ok=True)

    hydraulic_results.node["pressure"].to_csv(
        case_directory / "node_pressure_m.csv"
    )
    hydraulic_results.node["head"].to_csv(
        case_directory / "node_head_m.csv"
    )
    hydraulic_results.node["demand"].to_csv(
        case_directory / "node_demand_m3s.csv"
    )
    hydraulic_results.link["velocity"].to_csv(
        case_directory / "link_velocity_mps.csv"
    )
    hydraulic_results.link["flowrate"].to_csv(
        case_directory / "link_flowrate_m3s.csv"
    )

    tank_levels = calculate_tank_levels(
        hydraulic_results
    )
    tank_levels.to_csv(
        case_directory / "tank_levels_m.csv"
    )

    # Export the actual copied network after the optimized design and
    # scenario have been applied.
    wntr.network.write_inpfile(
        assessed_wn,
        str(case_directory / "assessed_model.inp"),
        units="LPS",
        version=2.2,
    )

    design_audit = audit.get("design")
    if design_audit is not None:
        pd.DataFrame(design_audit).to_csv(
            case_directory / "design_audit.csv",
            index=False,
        )

    scenario_audit = audit.get("scenario")
    if isinstance(scenario_audit, dict):
        pd.DataFrame(
            [scenario_audit]
        ).to_csv(
            case_directory / "scenario_audit.csv",
            index=False,
        )

    audit_summary = {
        key: value
        for key, value in audit.items()
        if key not in {"design", "scenario"}
    }
    pd.DataFrame([audit_summary]).to_csv(
        case_directory / "verification_audit.csv",
        index=False,
    )


def plot_baseline_results(
    *,
    hydraulic_results: Any,
    assessed_wn: wntr.network.WaterNetworkModel,
    output_directory: Path,
) -> None:
    """Save network, pressure, velocity, and tank-level figures."""

    figures_directory = output_directory / "figures"
    figures_directory.mkdir(parents=True, exist_ok=True)

    # Network topology.
    plt.figure(figsize=(11, 7))
    wntr.graphics.plot_network(
        assessed_wn,
        node_labels=True,
        link_labels=True,
        title="Arouca Highlift Conceptual WNTR Model",
    )
    plt.tight_layout()
    plt.savefig(
        figures_directory / "arouca_network_topology.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    pressure = hydraulic_results.node["pressure"]
    time_hr = pressure.index.astype(float) / 3600.0
    demand_nodes = [
        "Existing_Zone",
        "Windy_Hill_Demand",
        "Bon_Air_North_Demand",
    ]

    plt.figure(figsize=(12, 6))
    for node_name in demand_nodes:
        plt.plot(
            time_hr,
            pressure[node_name],
            label=node_name,
        )
    plt.axhline(
        MINIMUM_PRESSURE_M,
        linestyle="--",
        label="Minimum pressure = 15 m",
    )
    plt.xlabel("Time (hours)")
    plt.ylabel("Pressure (m)")
    plt.title(
        "Baseline Pressure Performance over 120 Hours"
    )
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        figures_directory / "baseline_pressure.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    velocity = hydraulic_results.link["velocity"]
    velocity_time_hr = (
        velocity.index.astype(float) / 3600.0
    )

    plt.figure(figsize=(12, 6))
    for link_name in assessed_wn.pipe_name_list:
        if link_name in velocity.columns:
            plt.plot(
                velocity_time_hr,
                velocity[link_name].abs(),
                label=link_name,
            )
    plt.axhline(
        MAXIMUM_VELOCITY_MPS,
        linestyle="--",
        label="Maximum velocity = 2.5 m/s",
    )
    plt.xlabel("Time (hours)")
    plt.ylabel("Absolute velocity (m/s)")
    plt.title(
        "Baseline Pipe-Velocity Performance over 120 Hours"
    )
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        figures_directory / "baseline_velocity.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    tank_levels = calculate_tank_levels(
        hydraulic_results
    )

    plt.figure(figsize=(12, 6))
    plt.plot(
        tank_levels["time_hr"],
        tank_levels["Windy_Hill_Tank_Level_m"],
        label="Windy Hill Tank",
    )
    plt.plot(
        tank_levels["time_hr"],
        tank_levels["Bon_Air_North_Tank_Level_m"],
        label="Bon Air North Tank",
    )
    plt.axhline(
        TANK_MIN_LEVEL_M,
        linestyle="--",
        label="Minimum level",
    )
    plt.axhline(
        TANK_MAX_LEVEL_M,
        linestyle="--",
        label="Maximum level",
    )
    plt.xlabel("Time (hours)")
    plt.ylabel("Tank water level (m)")
    plt.title(
        "Baseline Tank Levels over 120 Hours"
    )
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        figures_directory / "baseline_tank_levels.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


# =====================================================================
# 7. EXTENSION VERIFICATION WORKFLOW
# =====================================================================

def run_case(
    *,
    case: dict[str, float | str],
    design: PipeDesign,
    simulator: str,
    output_directory: Path,
    export_outputs: bool,
) -> tuple[dict[str, object], Any, Any]:
    """Build and assess one baseline or stress-test case."""

    case_name = str(case["name"])

    base_wn = build_arouca_network(
        roughness_c=float(case["roughness_c"]),
        pump_head_multiplier=float(
            case["pump_head_multiplier"]
        ),
        source_head_m=float(case["source_head_m"]),
    )

    original_diameters = record_original_diameters(
        base_wn
    )

    scenario = build_hydraulic_scenario(
        name=case_name,
        demand_multiplier=float(
            case["demand_multiplier"]
        ),
    )

    verification, hydraulic_results, audit = (
        run_design_verification(
            wn=base_wn,
            scenario=scenario,
            design=design,
            simulator=simulator,
            minimum_pressure_m=MINIMUM_PRESSURE_M,
            maximum_velocity_mps=MAXIMUM_VELOCITY_MPS,
            required_compliance_pct=(
                REQUIRED_COMPLIANCE_PCT
            ),
        )
    )

    original_network_preserved = (
        verify_original_network_preserved(
            base_wn,
            original_diameters,
        )
    )

    if not original_network_preserved:
        raise RuntimeError(
            "The original network was modified during verification."
        )

    assessed_wn = create_assessed_network(
        base_wn=base_wn,
        design=design,
        scenario=scenario,
    )

    if export_outputs:
        export_case_outputs(
            output_directory=output_directory,
            case_name=case_name,
            assessed_wn=assessed_wn,
            hydraulic_results=hydraulic_results,
            audit=audit,
        )

    summary_row = create_summary_row(
        case=case,
        verification=verification,
        hydraulic_results=hydraulic_results,
        original_network_preserved=(
            original_network_preserved
        ),
    )

    return summary_row, hydraulic_results, assessed_wn


def print_case_result(
    summary_row: dict[str, object],
) -> None:
    """Print the principal extension result for one scenario."""

    print("\n" + "=" * 72)
    print(str(summary_row["Scenario"]).upper())
    print("=" * 72)
    print(f"Simulator: {summary_row['Simulator']}")
    print(f"Design: {summary_row['Design']}")
    print(
        "Minimum pressure:",
        f"{float(summary_row['Minimum Pressure (m)']):.3f} m",
    )
    print(
        "Critical pressure junction:",
        summary_row["Critical Pressure Junction"],
    )
    print(
        "Pressure compliance:",
        f"{float(summary_row['Pressure Compliance (%)']):.2f}%",
    )
    print(
        "Maximum velocity:",
        f"{float(summary_row['Maximum Velocity (m/s)']):.3f} m/s",
    )
    print(
        "Critical velocity pipe:",
        summary_row["Critical Velocity Pipe"],
    )
    print(
        "Velocity compliance:",
        f"{float(summary_row['Velocity Compliance (%)']):.2f}%",
    )
    print(
        "Overall feasible:",
        summary_row["Overall Feasible"],
    )
    print(
        "Original network preserved:",
        summary_row["Original Network Preserved"],
    )


def print_model_summary(
    wn: wntr.network.WaterNetworkModel,
) -> None:
    """Print the constructed network components and project demand."""

    print("\n" + "=" * 72)
    print("AROUCA WNTR MODEL")
    print("=" * 72)
    print("Reservoirs:", list(wn.reservoir_name_list))
    print("Tanks:", list(wn.tank_name_list))
    print("Junctions:", list(wn.junction_name_list))
    print("Pumps:", list(wn.pump_name_list))
    print("Pipes:", list(wn.pipe_name_list))

    print("\nDemand allocation")
    print(
        f"Existing Arouca zone: "
        f"{EXISTING_ZONE_DEMAND_IMGD:.2f} IMGD "
        f"({EXISTING_ZONE_DEMAND_M3S:.5f} m3/s)"
    )
    print(
        f"Windy Hill: "
        f"{WINDY_HILL_DEMAND_IMGD:.2f} IMGD "
        f"({WINDY_HILL_DEMAND_M3S:.5f} m3/s)"
    )
    print(
        f"Bon Air North: "
        f"{BON_AIR_NORTH_DEMAND_IMGD:.2f} IMGD "
        f"({BON_AIR_NORTH_DEMAND_M3S:.5f} m3/s)"
    )
    print(
        f"Total: {TOTAL_DEMAND_IMGD:.2f} IMGD"
    )

    print("\nBase model pipe diameters")
    for pipe_name in wn.pipe_name_list:
        print(
            f"  {pipe_name}: "
            f"{wn.get_link(pipe_name).diameter:.3f} m"
        )

    print("\nOptimized design applied by the extension")
    for pipe_name, diameter_m in (
        OPTIMIZED_PIPE_DIAMETERS_M.items()
    ):
        print(
            f"  {pipe_name}: {diameter_m:.3f} m"
        )


def build_reference_comparison(
    baseline_summary: dict[str, object],
) -> pd.DataFrame:
    """Compare extension output with the reported project baseline."""

    simulated_minimum_pressure = float(
        baseline_summary["Minimum Pressure (m)"]
    )
    simulated_maximum_velocity = float(
        baseline_summary["Maximum Velocity (m/s)"]
    )

    return pd.DataFrame(
        [
            {
                "Metric": "Minimum pressure (m)",
                "Project reference": (
                    REFERENCE_MINIMUM_PRESSURE_M
                ),
                "Extension result": (
                    simulated_minimum_pressure
                ),
                "Absolute difference": abs(
                    simulated_minimum_pressure
                    - REFERENCE_MINIMUM_PRESSURE_M
                ),
            },
            {
                "Metric": "Maximum velocity (m/s)",
                "Project reference": (
                    REFERENCE_MAXIMUM_VELOCITY_MPS
                ),
                "Extension result": (
                    simulated_maximum_velocity
                ),
                "Absolute difference": abs(
                    simulated_maximum_velocity
                    - REFERENCE_MAXIMUM_VELOCITY_MPS
                ),
            },
        ]
    )


def parse_arguments() -> argparse.Namespace:
    """Parse command-line options."""

    parser = argparse.ArgumentParser(
        description=(
            "Build the conceptual Arouca WNTR model and "
            "test the custom design-verification extension."
        )
    )

    parser.add_argument(
        "--simulator",
        choices=["WNTR", "EPANET"],
        default="WNTR",
        help=(
            "Hydraulic simulator used by the extension. "
            "Default: WNTR."
        ),
    )

    parser.add_argument(
        "--output-dir",
        default="arouca_extension_outputs",
        help=(
            "Directory used for CSV, INP, and PNG outputs."
        ),
    )

    parser.add_argument(
        "--baseline-only",
        action="store_true",
        help=(
            "Run only the baseline case instead of all "
            "eight validation cases."
        ),
    )

    parser.add_argument(
        "--no-export",
        action="store_true",
        help=(
            "Run verification without creating detailed "
            "case files."
        ),
    )

    return parser.parse_args()


def main() -> None:
    """Run the complete Arouca extension-verification workflow."""

    args = parse_arguments()

    output_directory = Path(
        args.output_dir
    ).resolve()
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    example_base_wn = build_arouca_network()
    print_model_summary(example_base_wn)

    design = build_optimized_design()

    cases_to_run = (
        VALIDATION_CASES[:1]
        if args.baseline_only
        else VALIDATION_CASES
    )

    summary_rows: list[dict[str, object]] = []
    baseline_results = None
    baseline_assessed_wn = None

    for case_number, case in enumerate(
        cases_to_run,
        start=1,
    ):
        print(
            f"\nRunning case "
            f"{case_number}/{len(cases_to_run)}: "
            f"{case['name']}"
        )

        (
            summary_row,
            hydraulic_results,
            assessed_wn,
        ) = run_case(
            case=case,
            design=design,
            simulator=args.simulator,
            output_directory=output_directory,
            export_outputs=not args.no_export,
        )

        summary_rows.append(summary_row)
        print_case_result(summary_row)

        if case_number == 1:
            baseline_results = hydraulic_results
            baseline_assessed_wn = assessed_wn

    summary_table = pd.DataFrame(summary_rows)

    summary_file = (
        output_directory
        / "arouca_extension_validation_summary.csv"
    )
    summary_table.to_csv(
        summary_file,
        index=False,
    )

    if summary_rows:
        reference_comparison = build_reference_comparison(
            summary_rows[0]
        )
        reference_comparison.to_csv(
            output_directory
            / "project_reference_comparison.csv",
            index=False,
        )

    if (
        baseline_results is not None
        and baseline_assessed_wn is not None
        and not args.no_export
    ):
        plot_baseline_results(
            hydraulic_results=baseline_results,
            assessed_wn=baseline_assessed_wn,
            output_directory=output_directory,
        )

        # Also export one clearly named optimized baseline model.
        wntr.network.write_inpfile(
            baseline_assessed_wn,
            str(
                output_directory
                / "arouca_optimized_baseline_model.inp"
            ),
            units="LPS",
            version=2.2,
        )

    print("\n" + "=" * 72)
    print("FINAL VALIDATION SUMMARY")
    print("=" * 72)

    display_columns = [
        "Scenario",
        "Minimum Pressure (m)",
        "Maximum Velocity (m/s)",
        "Pressure Compliance (%)",
        "Velocity Compliance (%)",
        "Overall Feasible",
        "Original Network Preserved",
    ]

    print(
        summary_table[
            display_columns
        ].to_string(index=False)
    )

    print("\nSummary file:")
    print(summary_file)

    print("\nOutput directory:")
    print(output_directory)

    print(
        "\nThe script completed successfully. "
        "Review the CSV tables, INP files, and figures "
        "before drawing engineering conclusions."
    )


if __name__ == "__main__":
    main()
