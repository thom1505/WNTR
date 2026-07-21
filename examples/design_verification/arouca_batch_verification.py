"""Run a reproducible Arouca design-and-scenario verification batch.

This example uses the hydraulic model and case-study constants defined
in ``arouca_extension_verification.py``. It evaluates three pipe-design
conditions across eight hydraulic cases using the custom WNTR
design-verification batch engine.

The eight cases include changes in demand, pipe roughness, pump head,
and source head. Network-level changes are applied while constructing
each case-specific WaterNetworkModel. The corresponding
HydraulicScenario applies the demand multiplier and simulation timing.

Outputs
-------
The script writes:

* ``arouca_batch_verification_summary.csv``
* ``arouca_batch_experiment_manifest.json``

The summary contains hydraulic performance, feasibility, timing,
software-version information, network-configuration metadata, and
stable hashes that identify the complete experiment configuration.

Important
---------
This is a conceptual research model. It is not a field-calibrated
operational utility model and must not be used directly for operational
or construction decisions.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any, Sequence

import pandas as pd

from wntr.extensions.design_verification import (
    HydraulicScenario,
    PipeDesign,
    run_verification_batch,
)


BASELINE_DESIGN_LABEL = "Unchanged baseline network"
STRENGTHENED_DESIGN_NAME = "Arouca strengthened design"

STRENGTHENED_PIPE_DIAMETERS_M = {
    "P_Proposed_Main": 0.250,
    "P_Windy_to_BonAirTank": 0.150,
    "P_Windy_Tank_to_Demand": 0.200,
    "P_BonAir_Tank_to_Demand": 0.200,
    "P_Pump_to_Existing": 0.200,
}


def load_arouca_model_module() -> ModuleType:
    """Load the sibling Arouca model example as a Python module."""
    module_path = (
        Path(__file__).resolve().parent
        / "arouca_extension_verification.py"
    )

    if not module_path.exists():
        raise FileNotFoundError(
            f"Arouca model example was not found at: {module_path}"
        )

    specification = importlib.util.spec_from_file_location(
        "arouca_extension_verification",
        module_path,
    )

    if specification is None or specification.loader is None:
        raise ImportError(
            "The Arouca model example could not be loaded."
        )

    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)

    return module


AROUCA = load_arouca_model_module()


def build_batch_designs() -> list[PipeDesign | None]:
    """Return the baseline, optimized, and strengthened designs."""
    return [
        None,
        AROUCA.build_optimized_design(),
        PipeDesign(
            name=STRENGTHENED_DESIGN_NAME,
            diameters_m=STRENGTHENED_PIPE_DIAMETERS_M,
        ),
    ]


def design_label(design: PipeDesign | None) -> str:
    """Return a readable design label for result tables."""
    if design is None:
        return BASELINE_DESIGN_LABEL

    return design.name.strip()


def build_case_scenario(
    case: dict[str, float | str],
) -> HydraulicScenario:
    """Create the hydraulic scenario associated with one case."""
    return HydraulicScenario(
        name=str(case["name"]),
        demand_multiplier=float(
            case["demand_multiplier"]
        ),
        demand_model="DD",
        duration_s=AROUCA.DURATION_HOURS * 3600,
        hydraulic_timestep_s=(
            AROUCA.HYDRAULIC_TIMESTEP_S
        ),
        report_timestep_s=AROUCA.REPORT_TIMESTEP_S,
    )


def network_configuration(
    case: dict[str, float | str],
) -> dict[str, object]:
    """Return reproducible case-specific network parameters."""
    return {
        "case_name": str(case["name"]),
        "roughness_c": float(case["roughness_c"]),
        "pump_head_multiplier": float(
            case["pump_head_multiplier"]
        ),
        "source_head_m": float(case["source_head_m"]),
    }


def canonical_json(value: object) -> str:
    """Return deterministic JSON for hashing and audit records."""
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=str,
    )


def sha256_text(value: str) -> str:
    """Return the SHA-256 digest of text."""
    return hashlib.sha256(
        value.encode("utf-8")
    ).hexdigest()


def network_configuration_metadata(
    case: dict[str, float | str],
) -> tuple[str, str]:
    """Return serialized network configuration and its hash."""
    configuration_json = canonical_json(
        network_configuration(case)
    )

    return (
        configuration_json,
        sha256_text(configuration_json),
    )


def complete_configuration_hash(
    *,
    network_hash: str,
    batch_configuration_hash: str,
) -> str:
    """Hash both network and batch-engine configuration records."""
    combined = canonical_json(
        {
            "network_configuration_hash": network_hash,
            "batch_configuration_hash": (
                batch_configuration_hash
            ),
        }
    )

    return sha256_text(combined)


def capture_network_signature(
    wn: Any,
) -> dict[str, object]:
    """Capture selected values used to confirm model preservation."""
    return {
        "pipe_diameters_m": {
            pipe_name: float(
                wn.get_link(pipe_name).diameter
            )
            for pipe_name in sorted(wn.pipe_name_list)
        },
        "demand_multiplier": float(
            wn.options.hydraulic.demand_multiplier
        ),
        "demand_model": str(
            wn.options.hydraulic.demand_model
        ),
        "duration_s": int(wn.options.time.duration),
        "hydraulic_timestep_s": int(
            wn.options.time.hydraulic_timestep
        ),
        "report_timestep_s": int(
            wn.options.time.report_timestep
        ),
    }


def build_experiment_plan(
    *,
    cases: Sequence[dict[str, float | str]] | None = None,
    designs: Sequence[PipeDesign | None] | None = None,
    simulator: str = "WNTR",
) -> pd.DataFrame:
    """Build the intended experiment matrix without simulating it."""
    case_values = list(
        AROUCA.VALIDATION_CASES
        if cases is None
        else cases
    )

    design_values = list(
        build_batch_designs()
        if designs is None
        else designs
    )

    rows: list[dict[str, object]] = []

    for case_number, case in enumerate(
        case_values,
        start=1,
    ):
        for design_number, design in enumerate(
            design_values,
            start=1,
        ):
            rows.append(
                {
                    "planned_case_number": case_number,
                    "planned_design_number": design_number,
                    "case_name": str(case["name"]),
                    "design_label": design_label(design),
                    "simulator_name": simulator,
                    "demand_multiplier": float(
                        case["demand_multiplier"]
                    ),
                    "roughness_c": float(
                        case["roughness_c"]
                    ),
                    "pump_head_multiplier": float(
                        case["pump_head_multiplier"]
                    ),
                    "source_head_m": float(
                        case["source_head_m"]
                    ),
                }
            )

    return pd.DataFrame(rows)


def run_arouca_batch(
    *,
    cases: Sequence[dict[str, float | str]] | None = None,
    designs: Sequence[PipeDesign | None] | None = None,
    simulator: str = "WNTR",
    continue_on_error: bool = True,
    retain_hydraulic_results: bool = False,
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    """Run the requested Arouca design-and-case experiment matrix."""
    case_values = list(
        AROUCA.VALIDATION_CASES
        if cases is None
        else cases
    )

    design_values = list(
        build_batch_designs()
        if designs is None
        else designs
    )

    if not case_values:
        raise ValueError("cases must contain at least one case.")

    if not design_values:
        raise ValueError("designs must contain at least one design.")

    all_summaries: list[pd.DataFrame] = []
    all_records: dict[str, dict[str, Any]] = {}

    global_experiment_number = 0

    for case_number, case in enumerate(
        case_values,
        start=1,
    ):
        case_name = str(case["name"])

        base_wn = AROUCA.build_arouca_network(
            roughness_c=float(case["roughness_c"]),
            pump_head_multiplier=float(
                case["pump_head_multiplier"]
            ),
            source_head_m=float(case["source_head_m"]),
        )

        before_signature = capture_network_signature(base_wn)

        scenario = build_case_scenario(case)

        case_summary, case_records = (
            run_verification_batch(
                wn=base_wn,
                scenarios=[scenario],
                designs=design_values,
                simulators=[simulator],
                minimum_pressure_m=(
                    AROUCA.MINIMUM_PRESSURE_M
                ),
                maximum_velocity_mps=(
                    AROUCA.MAXIMUM_VELOCITY_MPS
                ),
                required_compliance_pct=(
                    AROUCA.REQUIRED_COMPLIANCE_PCT
                ),
                continue_on_error=continue_on_error,
                retain_hydraulic_results=(
                    retain_hydraulic_results
                ),
            )
        )

        after_signature = capture_network_signature(base_wn)

        original_network_preserved = (
            before_signature == after_signature
        )

        (
            network_configuration_json,
            network_configuration_hash,
        ) = network_configuration_metadata(case)

        case_summary = case_summary.copy()

        case_summary.insert(
            0,
            "case_number",
            case_number,
        )

        case_summary.insert(
            1,
            "case_name",
            case_name,
        )

        case_summary["roughness_c"] = float(
            case["roughness_c"]
        )

        case_summary["pump_head_multiplier"] = float(
            case["pump_head_multiplier"]
        )

        case_summary["source_head_m"] = float(
            case["source_head_m"]
        )

        case_summary["network_configuration_json"] = (
            network_configuration_json
        )

        case_summary["network_configuration_hash"] = (
            network_configuration_hash
        )

        case_summary["original_network_preserved"] = (
            original_network_preserved
        )

        case_summary["design_label"] = (
            case_summary["design_name"]
            .fillna(BASELINE_DESIGN_LABEL)
        )

        original_ids = list(
            case_summary["experiment_id"]
        )

        replacement_ids: list[str] = []

        for row_number, original_id in enumerate(
            original_ids
        ):
            global_experiment_number += 1

            batch_hash = str(
                case_summary.iloc[row_number][
                    "configuration_hash"
                ]
            )

            full_hash = complete_configuration_hash(
                network_hash=network_configuration_hash,
                batch_configuration_hash=batch_hash,
            )

            new_id = (
                f"arouca-{global_experiment_number:03d}-"
                f"{full_hash[:12]}"
            )

            replacement_ids.append(new_id)

            row_index = case_summary.index[row_number]

            case_summary.loc[
                row_index,
                "full_configuration_hash",
            ] = full_hash

            record = case_records[original_id]
            record["case_name"] = case_name
            record["case_number"] = case_number
            record["network_configuration"] = (
                network_configuration(case)
            )
            record["network_configuration_hash"] = (
                network_configuration_hash
            )
            record["full_configuration_hash"] = full_hash
            record["original_network_preserved"] = (
                original_network_preserved
            )

            all_records[new_id] = record

        case_summary["batch_engine_experiment_id"] = (
            original_ids
        )

        case_summary["experiment_id"] = replacement_ids

        all_summaries.append(case_summary)

    summary = pd.concat(
        all_summaries,
        ignore_index=True,
        sort=False,
    )

    return summary, all_records


def summarize_results(summary: pd.DataFrame) -> None:
    """Print principal findings from a completed batch."""
    total = len(summary)

    completed = int(
        (summary["status"] == "completed").sum()
    )

    failed = int(
        (summary["status"] == "failed").sum()
    )

    feasible = int(
        summary["overall_feasible"].fillna(False).sum()
    )

    print("\n" + "=" * 78)
    print("AROUCA BATCH VERIFICATION SUMMARY")
    print("=" * 78)

    print(f"Total experiments: {total}")
    print(f"Completed experiments: {completed}")
    print(f"Failed experiments: {failed}")
    print(f"Hydraulically feasible experiments: {feasible}")

    print(
        "Unique experiment IDs:",
        bool(summary["experiment_id"].is_unique),
    )

    print(
        "Original networks preserved:",
        bool(
            summary[
                "original_network_preserved"
            ].all()
        ),
    )

    completed_rows = summary.loc[
        summary["status"] == "completed"
    ].copy()

    if completed_rows.empty:
        print(
            "\nNo completed experiments were available "
            "for hydraulic interpretation."
        )
        return

    weakest_pressure = completed_rows.loc[
        completed_rows["pressure_margin_m"].idxmin()
    ]

    weakest_velocity = completed_rows.loc[
        completed_rows["velocity_margin_mps"].idxmin()
    ]

    print("\nSmallest pressure margin")

    print(
        f"  Design: {weakest_pressure['design_label']}"
    )
    print(
        f"  Case: {weakest_pressure['case_name']}"
    )
    print(
        "  Margin:",
        f"{float(weakest_pressure['pressure_margin_m']):.3f} m",
    )

    print("\nSmallest velocity margin")

    print(
        f"  Design: {weakest_velocity['design_label']}"
    )
    print(
        f"  Case: {weakest_velocity['case_name']}"
    )
    print(
        "  Margin:",
        f"{float(weakest_velocity['velocity_margin_mps']):.3f} m/s",
    )

    design_summary = (
        completed_rows.groupby("design_label", dropna=False)
        .agg(
            experiments=("experiment_id", "count"),
            feasible=("overall_feasible", "sum"),
            minimum_pressure_m=(
                "minimum_pressure_m",
                "min",
            ),
            maximum_velocity_mps=(
                "maximum_velocity_mps",
                "max",
            ),
            minimum_pressure_margin_m=(
                "pressure_margin_m",
                "min",
            ),
        )
        .reset_index()
    )

    print("\nPerformance by design")
    print(design_summary.to_string(index=False))


def export_batch_outputs(
    *,
    summary: pd.DataFrame,
    records: dict[str, dict[str, Any]],
    output_directory: Path,
) -> tuple[Path, Path]:
    """Export the result table and a reproducibility manifest."""
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_path = (
        output_directory
        / "arouca_batch_verification_summary.csv"
    )

    manifest_path = (
        output_directory
        / "arouca_batch_experiment_manifest.json"
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    manifest = {
        "experiment_count": len(summary),
        "unique_experiment_ids": bool(
            summary["experiment_id"].is_unique
        ),
        "all_original_networks_preserved": bool(
            summary[
                "original_network_preserved"
            ].all()
        ),
        "experiment_ids": list(
            summary["experiment_id"]
        ),
        "full_configuration_hashes": list(
            summary["full_configuration_hash"]
        ),
        "records_available": list(records),
        "experiments": summary.to_dict(
            orient="records"
        ),
    }

    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    return summary_path, manifest_path


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Run the reproducible Arouca design-and-scenario "
            "verification batch."
        )
    )

    parser.add_argument(
        "--simulator",
        choices=["WNTR", "EPANET"],
        default="WNTR",
        help="Hydraulic simulator. Default: WNTR.",
    )

    parser.add_argument(
        "--output-dir",
        default="arouca_batch_outputs",
        help="Directory for the CSV and JSON outputs.",
    )

    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop instead of recording and continuing after an error.",
    )

    return parser.parse_args()


def main() -> None:
    """Run and export the complete 24-experiment matrix."""
    args = parse_arguments()

    plan = build_experiment_plan(
        simulator=args.simulator
    )

    print("\nPlanned experiments:")
    print(plan.to_string(index=False))

    summary, records = run_arouca_batch(
        simulator=args.simulator,
        continue_on_error=not args.stop_on_error,
        retain_hydraulic_results=False,
    )

    summarize_results(summary)

    summary_path, manifest_path = export_batch_outputs(
        summary=summary,
        records=records,
        output_directory=Path(
            args.output_dir
        ).resolve(),
    )

    print("\nSummary CSV:")
    print(summary_path)

    print("\nExperiment manifest:")
    print(manifest_path)


if __name__ == "__main__":
    main()
