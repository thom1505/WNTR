"""Run uncertainty-aware hydraulic verification for the Arouca model.

This example extends the deterministic Arouca batch experiment by using
Latin hypercube sampling to vary demand, pipe roughness, pump head, and
source head. Every pipe-design alternative is tested against the same
realizations, producing paired robustness evidence.

Important
---------
The ranges in this example are illustrative research assumptions. The
model is conceptual and not field calibrated. Replace the ranges with
values justified by utility records, field measurements, calibration,
engineering judgement, or cited literature before formal case-study use.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Mapping

import pandas as pd

from wntr.extensions.design_verification import (
    HydraulicScenario,
    UncertaintyVariable,
    run_robust_verification,
    sample_uncertainty,
    summarize_robustness,
)


def _load_sibling_module(filename: str, module_name: str) -> ModuleType:
    """Load a sibling example module without modifying ``sys.path``."""
    module_path = Path(__file__).resolve().parent / filename
    if not module_path.exists():
        raise FileNotFoundError(f"Required example not found: {module_path}")

    specification = importlib.util.spec_from_file_location(
        module_name,
        module_path,
    )
    if specification is None or specification.loader is None:
        raise ImportError(f"Could not load example module: {module_path}")

    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


AROUCA_BATCH = _load_sibling_module(
    "arouca_batch_verification.py",
    "arouca_batch_verification",
)
AROUCA = AROUCA_BATCH.AROUCA


UNCERTAINTY_VARIABLES = [
    UncertaintyVariable("demand_multiplier", 0.80, 1.30),
    UncertaintyVariable("roughness_c", 110.0, 150.0),
    UncertaintyVariable("pump_head_multiplier", 0.90, 1.05),
    UncertaintyVariable("source_head_m", 32.0, 38.0),
]


def build_network(realization: Mapping[str, object]):
    """Build the realization-specific conceptual Arouca network."""
    return AROUCA.build_arouca_network(
        roughness_c=float(realization["roughness_c"]),
        pump_head_multiplier=float(
            realization["pump_head_multiplier"]
        ),
        source_head_m=float(realization["source_head_m"]),
    )


def build_scenario(realization: Mapping[str, object]) -> HydraulicScenario:
    """Build the realization-specific hydraulic scenario."""
    return HydraulicScenario(
        name=f"Uncertainty realization {realization['realization_id']}",
        demand_multiplier=float(realization["demand_multiplier"]),
        demand_model="DD",
        duration_s=AROUCA.DURATION_HOURS * 3600,
        hydraulic_timestep_s=AROUCA.HYDRAULIC_TIMESTEP_S,
        report_timestep_s=AROUCA.REPORT_TIMESTEP_S,
    )


def export_outputs(
    *,
    samples: pd.DataFrame,
    experiment_summary: pd.DataFrame,
    robustness_summary: pd.DataFrame,
    output_directory: Path,
    seed: int,
    simulator: str,
) -> dict[str, Path]:
    """Export exact inputs, experiment outputs, and aggregate metrics."""
    output_directory.mkdir(parents=True, exist_ok=True)

    paths = {
        "samples": output_directory / "arouca_uncertainty_samples.csv",
        "experiments": (
            output_directory / "arouca_uncertainty_experiments.csv"
        ),
        "robustness": (
            output_directory / "arouca_robustness_summary.csv"
        ),
        "manifest": (
            output_directory / "arouca_uncertainty_manifest.json"
        ),
    }

    samples.to_csv(paths["samples"], index=False)
    experiment_summary.to_csv(paths["experiments"], index=False)
    robustness_summary.to_csv(paths["robustness"], index=False)

    manifest = {
        "study": "Arouca uncertainty-aware design verification",
        "conceptual_model": True,
        "field_calibrated": False,
        "sampling_method": "LATIN_HYPERCUBE",
        "sample_count": len(samples),
        "seed": seed,
        "simulator": simulator,
        "designs": list(
            robustness_summary["design_label"].drop_duplicates()
        ),
        "uncertainty_variables": [
            {
                "name": variable.name,
                "lower": variable.lower,
                "upper": variable.upper,
            }
            for variable in UNCERTAINTY_VARIABLES
        ],
        "requested_experiments": len(experiment_summary),
        "unique_experiment_ids": bool(
            experiment_summary["experiment_id"].is_unique
        ),
        "output_files": {
            name: str(path.name)
            for name, path in paths.items()
            if name != "manifest"
        },
    }
    paths["manifest"].write_text(
        json.dumps(manifest, indent=2, default=str),
        encoding="utf-8",
    )

    return paths


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Run uncertainty-aware hydraulic verification for the "
            "conceptual Arouca network."
        )
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=100,
        help="Number of Latin hypercube realizations. Default: 100.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed. Default: 42.",
    )
    parser.add_argument(
        "--simulator",
        choices=["WNTR", "EPANET"],
        default="WNTR",
        help="Hydraulic simulator. Default: WNTR.",
    )
    parser.add_argument(
        "--output-dir",
        default="arouca_uncertainty_outputs",
        help="Output directory.",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop at the first simulation error.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the uncertainty-aware Arouca experiment."""
    args = parse_arguments()
    samples = sample_uncertainty(
        UNCERTAINTY_VARIABLES,
        args.samples,
        seed=args.seed,
        method="LATIN_HYPERCUBE",
    )

    experiment_summary, _ = run_robust_verification(
        samples=samples,
        network_factory=build_network,
        scenario_factory=build_scenario,
        designs=AROUCA_BATCH.build_batch_designs(),
        simulators=[args.simulator],
        minimum_pressure_m=AROUCA.MINIMUM_PRESSURE_M,
        maximum_velocity_mps=AROUCA.MAXIMUM_VELOCITY_MPS,
        required_compliance_pct=AROUCA.REQUIRED_COMPLIANCE_PCT,
        continue_on_error=not args.stop_on_error,
        retain_hydraulic_results=False,
    )
    robustness_summary = summarize_robustness(experiment_summary)

    print("\nAROUCA UNCERTAINTY-AWARE ROBUSTNESS SUMMARY")
    print("=" * 78)
    print(robustness_summary.to_string(index=False))

    paths = export_outputs(
        samples=samples,
        experiment_summary=experiment_summary,
        robustness_summary=robustness_summary,
        output_directory=Path(args.output_dir).resolve(),
        seed=args.seed,
        simulator=args.simulator,
    )

    print("\nOutputs")
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
