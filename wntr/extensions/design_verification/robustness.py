"""Run and summarize uncertainty-aware hydraulic verification studies."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from statistics import NormalDist
from typing import Any

import numpy as np
import pandas as pd
from wntr.network import WaterNetworkModel

from .batch import run_verification_batch
from .models import HydraulicScenario, PipeDesign


NetworkFactory = Callable[[Mapping[str, object]], WaterNetworkModel]
ScenarioFactory = Callable[[Mapping[str, object]], HydraulicScenario]


def _canonical_json(value: object) -> str:
    """Serialize an experiment configuration deterministically."""
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=str,
    )


def _configuration_hash(value: object) -> tuple[str, str]:
    """Return canonical JSON and its SHA-256 digest."""
    serialized = _canonical_json(value)
    digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    return serialized, digest


def _validate_samples(samples: pd.DataFrame) -> pd.DataFrame:
    """Return a validated copy of the realization table."""
    if not isinstance(samples, pd.DataFrame):
        raise TypeError("samples must be a pandas DataFrame.")

    if samples.empty:
        raise ValueError("samples cannot be empty.")

    values = samples.copy().reset_index(drop=True)

    if "realization_number" not in values.columns:
        values.insert(
            0,
            "realization_number",
            np.arange(1, len(values) + 1, dtype=int),
        )

    if "realization_id" not in values.columns:
        realization_ids: list[str] = []
        for row_number, row in values.iterrows():
            _, digest = _configuration_hash(row.to_dict())
            realization_ids.append(
                f"u-{row_number + 1:05d}-{digest[:10]}"
            )
        values.insert(0, "realization_id", realization_ids)

    if values["realization_id"].isna().any():
        raise ValueError("realization_id values cannot be missing.")

    values["realization_id"] = (
        values["realization_id"].astype(str).str.strip()
    )

    if (values["realization_id"] == "").any():
        raise ValueError("realization_id values cannot be empty.")

    if not values["realization_id"].is_unique:
        raise ValueError("realization_id values must be unique.")

    return values


def run_robust_verification(
    samples: pd.DataFrame,
    network_factory: NetworkFactory,
    scenario_factory: ScenarioFactory,
    *,
    designs: Sequence[PipeDesign | None] | None = None,
    simulators: Sequence[str] | str = ("WNTR",),
    minimum_pressure_m: float = 15.0,
    maximum_velocity_mps: float = 2.5,
    required_compliance_pct: float = 100.0,
    continue_on_error: bool = True,
    retain_hydraulic_results: bool = False,
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    """Run paired design verification across uncertainty realizations.

    Every design and simulator is evaluated against the same realization
    table. This paired structure supports fair robustness comparisons.

    Parameters
    ----------
    samples
        One row per uncertainty realization. Tables returned by
        :func:`sample_uncertainty` can be supplied directly.
    network_factory
        Callable that receives one realization as a mapping and returns
        a case-specific ``WaterNetworkModel``.
    scenario_factory
        Callable that receives the same realization and returns its
        ``HydraulicScenario``.
    designs, simulators, minimum_pressure_m, maximum_velocity_mps,
    required_compliance_pct, continue_on_error, retain_hydraulic_results
        Passed to :func:`run_verification_batch`.

    Returns
    -------
    summary
        One row per realization-design-simulator experiment.
    records
        Detailed records keyed by robust experiment identifier.
    """
    if not callable(network_factory):
        raise TypeError("network_factory must be callable.")

    if not callable(scenario_factory):
        raise TypeError("scenario_factory must be callable.")

    sample_values = _validate_samples(samples)
    all_summaries: list[pd.DataFrame] = []
    all_records: dict[str, dict[str, Any]] = {}
    global_experiment_number = 0

    for _, sample_row in sample_values.iterrows():
        sample_payload = sample_row.to_dict()
        realization_id = str(sample_payload["realization_id"])
        realization_number = int(sample_payload["realization_number"])
        uncertainty_json, uncertainty_hash = _configuration_hash(
            sample_payload
        )

        wn = network_factory(sample_payload)
        if not isinstance(wn, WaterNetworkModel):
            raise TypeError(
                "network_factory must return a WNTR WaterNetworkModel."
            )

        scenario = scenario_factory(sample_payload)
        if not isinstance(scenario, HydraulicScenario):
            raise TypeError(
                "scenario_factory must return a HydraulicScenario."
            )

        realization_summary, realization_records = (
            run_verification_batch(
                wn=wn,
                scenarios=[scenario],
                designs=designs,
                simulators=simulators,
                minimum_pressure_m=minimum_pressure_m,
                maximum_velocity_mps=maximum_velocity_mps,
                required_compliance_pct=required_compliance_pct,
                continue_on_error=continue_on_error,
                retain_hydraulic_results=retain_hydraulic_results,
            )
        )

        realization_summary = realization_summary.copy()
        realization_summary.insert(
            0,
            "realization_number",
            realization_number,
        )
        realization_summary.insert(
            1,
            "realization_id",
            realization_id,
        )
        realization_summary["uncertainty_configuration_json"] = (
            uncertainty_json
        )
        realization_summary["uncertainty_configuration_hash"] = (
            uncertainty_hash
        )

        for column_name, value in sample_payload.items():
            if column_name in {"realization_id", "realization_number"}:
                continue
            realization_summary[
                f"uncertainty_{column_name}"
            ] = value

        original_ids = list(realization_summary["experiment_id"])
        replacement_ids: list[str] = []

        for row_position, original_id in enumerate(original_ids):
            global_experiment_number += 1
            row_index = realization_summary.index[row_position]
            batch_hash = str(
                realization_summary.loc[row_index, "configuration_hash"]
            )
            full_payload = {
                "uncertainty_configuration_hash": uncertainty_hash,
                "batch_configuration_hash": batch_hash,
            }
            _, full_hash = _configuration_hash(full_payload)
            robust_id = (
                f"rdv-{global_experiment_number:06d}-{full_hash[:12]}"
            )
            replacement_ids.append(robust_id)
            realization_summary.loc[
                row_index,
                "full_configuration_hash",
            ] = full_hash

            record = dict(realization_records[original_id])
            record["realization_id"] = realization_id
            record["realization_number"] = realization_number
            record["uncertainty_configuration"] = sample_payload
            record["uncertainty_configuration_hash"] = uncertainty_hash
            record["full_configuration_hash"] = full_hash
            all_records[robust_id] = record

        realization_summary["batch_engine_experiment_id"] = original_ids
        realization_summary["experiment_id"] = replacement_ids
        all_summaries.append(realization_summary)

    summary = pd.concat(
        all_summaries,
        ignore_index=True,
        sort=False,
    )

    if not summary["experiment_id"].is_unique:
        raise RuntimeError("Robust experiment identifiers are not unique.")

    return summary, all_records


def wilson_score_interval(
    successes: int,
    trials: int,
    *,
    confidence_level: float = 0.95,
) -> tuple[float, float]:
    """Return a Wilson confidence interval for a binomial proportion."""
    if isinstance(successes, bool) or isinstance(trials, bool):
        raise ValueError("successes and trials must be whole numbers.")

    successes_int = int(successes)
    trials_int = int(trials)

    if successes_int != successes or trials_int != trials:
        raise ValueError("successes and trials must be whole numbers.")

    if trials_int <= 0:
        raise ValueError("trials must be greater than zero.")

    if not 0 <= successes_int <= trials_int:
        raise ValueError("successes must be between zero and trials.")

    confidence = float(confidence_level)
    if not math.isfinite(confidence) or not 0.0 < confidence < 1.0:
        raise ValueError("confidence_level must be between zero and one.")

    z_value = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    proportion = successes_int / trials_int
    denominator = 1.0 + (z_value**2) / trials_int
    centre = (
        proportion
        + (z_value**2) / (2.0 * trials_int)
    ) / denominator
    half_width = (
        z_value
        * math.sqrt(
            proportion * (1.0 - proportion) / trials_int
            + (z_value**2) / (4.0 * trials_int**2)
        )
        / denominator
    )

    return max(0.0, centre - half_width), min(1.0, centre + half_width)


def _mode_or_none(series: pd.Series) -> object | None:
    """Return the first mode after dropping missing values."""
    values = series.dropna()
    if values.empty:
        return None

    modes = values.mode()
    if modes.empty:
        return None

    return modes.iloc[0]


def summarize_robustness(
    summary: pd.DataFrame,
    *,
    baseline_label: str = "Unchanged baseline network",
    confidence_level: float = 0.95,
) -> pd.DataFrame:
    """Aggregate probability, severity, and critical-component metrics.

    Hydraulic-feasibility probability is conditioned on completed
    simulations. ``overall_success_probability_pct`` additionally treats
    simulator failures as unsuccessful experiments, so both quantities are
    available for transparent interpretation.
    """
    if not isinstance(summary, pd.DataFrame):
        raise TypeError("summary must be a pandas DataFrame.")

    required_columns = {
        "status",
        "overall_feasible",
        "design_name",
        "simulator_name",
        "minimum_pressure_m",
        "maximum_velocity_mps",
        "pressure_margin_m",
        "velocity_margin_mps",
        "pressure_compliance_pct",
        "velocity_compliance_pct",
        "critical_pressure_component",
        "critical_velocity_component",
    }
    missing = sorted(required_columns.difference(summary.columns))
    if missing:
        raise ValueError(
            "summary is missing required columns: " + ", ".join(missing)
        )

    if summary.empty:
        raise ValueError("summary cannot be empty.")

    confidence = float(confidence_level)
    if not math.isfinite(confidence) or not 0.0 < confidence < 1.0:
        raise ValueError("confidence_level must be between zero and one.")

    working = summary.copy()
    working["design_label"] = (
        working["design_name"].fillna(baseline_label).astype(str)
    )

    rows: list[dict[str, object]] = []
    grouped = working.groupby(
        ["design_label", "simulator_name"],
        dropna=False,
        sort=True,
    )

    for (design_label, simulator_name), group in grouped:
        total_count = len(group)
        completed = group.loc[group["status"] == "completed"].copy()
        completed_count = len(completed)
        failed_count = total_count - completed_count

        feasible_count = 0
        ci_lower = math.nan
        ci_upper = math.nan
        feasibility_probability = math.nan

        if completed_count:
            feasible_mask = completed["overall_feasible"].eq(True)
            feasible_count = int(feasible_mask.sum())
            feasibility_probability = feasible_count / completed_count
            ci_lower, ci_upper = wilson_score_interval(
                feasible_count,
                completed_count,
                confidence_level=confidence,
            )

        pressure_margin = pd.to_numeric(
            completed["pressure_margin_m"],
            errors="coerce",
        )
        velocity_margin = pd.to_numeric(
            completed["velocity_margin_mps"],
            errors="coerce",
        )
        minimum_pressure = pd.to_numeric(
            completed["minimum_pressure_m"],
            errors="coerce",
        )
        maximum_velocity = pd.to_numeric(
            completed["maximum_velocity_mps"],
            errors="coerce",
        )

        rows.append(
            {
                "design_label": design_label,
                "simulator_name": simulator_name,
                "requested_experiments": total_count,
                "completed_experiments": completed_count,
                "simulation_failures": failed_count,
                "hydraulically_feasible_experiments": feasible_count,
                "completion_rate_pct": 100.0 * completed_count / total_count,
                "simulation_failure_probability_pct": (
                    100.0 * failed_count / total_count
                ),
                "feasibility_probability_pct": (
                    100.0 * feasibility_probability
                    if completed_count
                    else math.nan
                ),
                "feasibility_ci_lower_pct": (
                    100.0 * ci_lower if completed_count else math.nan
                ),
                "feasibility_ci_upper_pct": (
                    100.0 * ci_upper if completed_count else math.nan
                ),
                "overall_success_probability_pct": (
                    100.0 * feasible_count / total_count
                ),
                "minimum_pressure_p05_m": (
                    float(minimum_pressure.quantile(0.05))
                    if minimum_pressure.notna().any()
                    else math.nan
                ),
                "maximum_velocity_p95_mps": (
                    float(maximum_velocity.quantile(0.95))
                    if maximum_velocity.notna().any()
                    else math.nan
                ),
                "worst_pressure_margin_m": (
                    float(pressure_margin.min())
                    if pressure_margin.notna().any()
                    else math.nan
                ),
                "worst_velocity_margin_mps": (
                    float(velocity_margin.min())
                    if velocity_margin.notna().any()
                    else math.nan
                ),
                "expected_pressure_deficit_m": (
                    float((-pressure_margin).clip(lower=0.0).mean())
                    if pressure_margin.notna().any()
                    else math.nan
                ),
                "expected_velocity_exceedance_mps": (
                    float((-velocity_margin).clip(lower=0.0).mean())
                    if velocity_margin.notna().any()
                    else math.nan
                ),
                "minimum_pressure_compliance_pct": (
                    float(
                        pd.to_numeric(
                            completed["pressure_compliance_pct"],
                            errors="coerce",
                        ).min()
                    )
                    if completed_count
                    else math.nan
                ),
                "minimum_velocity_compliance_pct": (
                    float(
                        pd.to_numeric(
                            completed["velocity_compliance_pct"],
                            errors="coerce",
                        ).min()
                    )
                    if completed_count
                    else math.nan
                ),
                "most_frequent_critical_pressure_component": (
                    _mode_or_none(
                        completed["critical_pressure_component"]
                    )
                    if completed_count
                    else None
                ),
                "most_frequent_critical_velocity_component": (
                    _mode_or_none(
                        completed["critical_velocity_component"]
                    )
                    if completed_count
                    else None
                ),
                "confidence_level": confidence,
            }
        )

    return pd.DataFrame(rows)
