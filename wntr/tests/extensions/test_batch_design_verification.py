"""Tests for batch hydraulic design verification."""

from __future__ import annotations

import pandas as pd

import pytest
import wntr
import wntr.extensions.design_verification.batch as batch_module

from wntr.extensions.design_verification import (
    HydraulicScenario,
    PipeDesign,
    run_verification_batch,
)


pytestmark = pytest.mark.extensions


def build_small_network():
    """Build a small network suitable for fast batch tests."""
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


def build_scenarios():
    """Return two small hydraulic scenarios."""
    return [
        HydraulicScenario(
            name="Baseline",
            demand_multiplier=1.0,
            demand_model="DD",
            duration_s=0,
            hydraulic_timestep_s=3600,
            report_timestep_s=3600,
        ),
        HydraulicScenario(
            name="Higher demand",
            demand_multiplier=1.2,
            demand_model="DD",
            duration_s=0,
            hydraulic_timestep_s=3600,
            report_timestep_s=3600,
        ),
    ]


def test_batch_returns_every_requested_combination():
    """Confirm that every design-scenario combination is assessed."""
    wn = build_small_network()

    designs = [
        None,
        PipeDesign(
            name="Larger pipe",
            diameters_m={"P1": 0.200},
        ),
    ]

    summary, records = run_verification_batch(
        wn=wn,
        scenarios=build_scenarios(),
        designs=designs,
        simulators=["WNTR"],
        minimum_pressure_m=0.0,
        maximum_velocity_mps=10.0,
    )

    assert len(summary) == 4
    assert summary["experiment_id"].is_unique
    assert set(summary["status"]) == {"completed"}
    assert set(summary["scenario_name"]) == {
        "Baseline",
        "Higher demand",
    }
    assert summary["design_name"].isna().any()
    assert "Larger pipe" in set(
        summary["design_name"].dropna()
    )
    assert set(summary["overall_feasible"]) == {True}
    assert set(records) == set(summary["experiment_id"])


def test_batch_preserves_original_network():
    """Confirm that batch assessment does not modify the input model."""
    wn = build_small_network()
    original_diameter = wn.get_link("P1").diameter

    run_verification_batch(
        wn=wn,
        scenarios=build_scenarios(),
        designs=[
            PipeDesign(
                name="Larger pipe",
                diameters_m={"P1": 0.200},
            )
        ],
        simulators="WNTR",
        minimum_pressure_m=0.0,
        maximum_velocity_mps=10.0,
    )

    assert wn.get_link("P1").diameter == pytest.approx(
        original_diameter
    )


def test_batch_records_failure_and_continues():
    """Confirm that one failed experiment does not stop the batch."""
    wn = build_small_network()

    summary, records = run_verification_batch(
        wn=wn,
        scenarios=[build_scenarios()[0]],
        simulators=["WNTR", "unsupported"],
        minimum_pressure_m=0.0,
        maximum_velocity_mps=10.0,
        continue_on_error=True,
    )

    assert len(summary) == 2
    assert set(summary["status"]) == {
        "completed",
        "failed",
    }

    failed_row = summary.loc[
        summary["status"] == "failed"
    ].iloc[0]

    assert failed_row["error_type"] == "ValueError"
    assert "simulator" in failed_row["error_message"]

    failed_record = records[
        failed_row["experiment_id"]
    ]

    assert failed_record["verification"] is None
    assert failed_record["error"]["type"] == "ValueError"



def test_batch_records_hash_failure_and_continues(
    monkeypatch,
):
    """Record a hash failure and continue with the next experiment."""
    wn = build_small_network()

    original_configuration_hash = (
        batch_module._configuration_hash
    )
    call_count = 0

    def fail_first_configuration_hash(**kwargs):
        nonlocal call_count
        call_count += 1

        if call_count == 1:
            raise RuntimeError(
                "configuration hash failed"
            )

        return original_configuration_hash(**kwargs)

    monkeypatch.setattr(
        batch_module,
        "_configuration_hash",
        fail_first_configuration_hash,
    )

    summary, records = run_verification_batch(
        wn=wn,
        scenarios=build_scenarios(),
        simulators="WNTR",
        minimum_pressure_m=0.0,
        maximum_velocity_mps=10.0,
        continue_on_error=True,
    )

    assert len(summary) == 2
    assert list(summary["status"]) == [
        "failed",
        "completed",
    ]

    failed_row = summary.iloc[0]

    assert (
        failed_row["experiment_id"]
        == "dv-00001-unavailable"
    )
    assert (
        failed_row["configuration_hash"]
        == "unavailable"
    )
    assert failed_row["error_type"] == "RuntimeError"
    assert (
        failed_row["error_message"]
        == "configuration hash failed"
    )

    failed_record = records[
        failed_row["experiment_id"]
    ]

    assert failed_record["verification"] is None
    assert failed_record["audit"] is None
    assert (
        failed_record["error"]["type"]
        == "RuntimeError"
    )
    assert (
        failed_record["error"]["message"]
        == "configuration hash failed"
    )

    completed_row = summary.iloc[1]

    assert completed_row["status"] == "completed"
    assert completed_row["experiment_id"].startswith(
        "dv-00002-"
    )
    assert (
        completed_row["experiment_id"]
        != "dv-00002-unavailable"
    )


def test_batch_reraises_hash_failure_when_requested(
    monkeypatch,
):
    """Re-raise configuration-hash failures when requested."""
    wn = build_small_network()

    def fail_configuration_hash(**kwargs):
        raise RuntimeError(
            "configuration hash failed"
        )

    monkeypatch.setattr(
        batch_module,
        "_configuration_hash",
        fail_configuration_hash,
    )

    with pytest.raises(
        RuntimeError,
        match="configuration hash failed",
    ):
        run_verification_batch(
            wn=wn,
            scenarios=[build_scenarios()[0]],
            simulators="WNTR",
            minimum_pressure_m=0.0,
            maximum_velocity_mps=10.0,
            continue_on_error=False,
        )


def test_batch_can_raise_immediately_on_failure():
    """Confirm that continue_on_error=False re-raises an error."""
    wn = build_small_network()

    with pytest.raises(
        ValueError,
        match="simulator",
    ):
        run_verification_batch(
            wn=wn,
            scenarios=[build_scenarios()[0]],
            simulators=["unsupported"],
            minimum_pressure_m=0.0,
            maximum_velocity_mps=10.0,
            continue_on_error=False,
        )


def test_batch_includes_pump_diagnostics():
    """Include pump-audit fields in completed batch rows."""
    wn = build_small_network()

    summary, _ = run_verification_batch(
        wn=wn,
        scenarios=[build_scenarios()[0]],
        simulators="WNTR",
        minimum_pressure_m=0.0,
        maximum_velocity_mps=10.0,
    )

    expected_columns = {
        "pump_feasible",
        "head_pumps_in_network",
        "head_pumps_evaluable",
        "all_head_pumps_evaluable",
        "number_of_pumps_exceeding_curves",
        "total_curve_exceedance_observations",
        "governing_pump_name",
        "maximum_pump_flow_ratio",
    }

    assert expected_columns.issubset(summary.columns)

    row = summary.iloc[0]

    assert pd.isna(row["pump_feasible"])
    assert row["head_pumps_in_network"] == 0
    assert row["head_pumps_evaluable"] == 0
    assert bool(row["all_head_pumps_evaluable"])
    assert row["number_of_pumps_exceeding_curves"] == 0
    assert row["total_curve_exceedance_observations"] == 0
    assert pd.isna(row["governing_pump_name"])
    assert pd.isna(row["maximum_pump_flow_ratio"])


def test_failed_batch_row_has_empty_pump_diagnostics():
    """Leave pump fields empty when verification cannot run."""
    wn = build_small_network()

    summary, _ = run_verification_batch(
        wn=wn,
        scenarios=[build_scenarios()[0]],
        simulators=["unsupported"],
        minimum_pressure_m=0.0,
        maximum_velocity_mps=10.0,
        continue_on_error=True,
    )

    row = summary.iloc[0]

    assert row["status"] == "failed"

    pump_columns = [
        "pump_feasible",
        "head_pumps_in_network",
        "head_pumps_evaluable",
        "all_head_pumps_evaluable",
        "number_of_pumps_exceeding_curves",
        "total_curve_exceedance_observations",
        "governing_pump_name",
        "maximum_pump_flow_ratio",
    ]

    assert all(
        pd.isna(row[column])
        for column in pump_columns
    )
