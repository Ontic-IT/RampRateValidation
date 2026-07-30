"""Unit tests for Phase 6 additive: PhaseConformanceSummary."""

import pytest

from config.constants import ValidationStatus, Comparator
from engine.validation.aggregation import compute_phase_conformance
from models.domain import (
    PhaseConformanceSummary,
    ValidationResult,
    ValidationResults,
)


def _make_result(status: ValidationStatus, region_id: str | None = "R001") -> ValidationResult:
    # Conformance is setpoint-tracking only, so use a SETPOINT_DEVIATION
    # result. Measured deviation is chosen so a PASS earns full credit
    # (deviation 0) and a FAIL earns zero (deviation well past tolerance),
    # keeping the pass-ratio-style percentage assertions valid.
    measured = 0.0 if status == ValidationStatus.PASS else 10.0
    return ValidationResult(
        validation_result_id="VR001",
        requirement_id="SETPOINT_DEVIATION",
        requirement_description="Test requirement",
        measured_value=measured,
        threshold_value=2.0,
        comparator=Comparator.LTE,
        unit="°C",
        method="test",
        region_id=region_id,
        result=status,
        reason="Test reason",
    )


class TestComputePhaseConformance:
    def test_all_pass(self):
        results = ValidationResults(results=[
            _make_result(ValidationStatus.PASS, "R001"),
            _make_result(ValidationStatus.PASS, "R002"),
        ])
        summary = compute_phase_conformance(results)

        assert summary.total_phases == 2
        assert summary.passed_phases == 2
        assert summary.failed_phases == 0
        assert summary.anomaly_phases == 0
        assert summary.conformance_percentage == 100.0
        assert summary.anomaly_phase_ids == []

    def test_one_fail(self):
        results = ValidationResults(results=[
            _make_result(ValidationStatus.PASS, "R001"),
            _make_result(ValidationStatus.FAIL, "R002"),
        ])
        summary = compute_phase_conformance(results)

        assert summary.total_phases == 2
        assert summary.passed_phases == 1
        assert summary.failed_phases == 1
        assert summary.anomaly_phases == 1
        assert summary.conformance_percentage == 50.0
        assert "R002" in summary.anomaly_phase_ids
        assert "R001" not in summary.anomaly_phase_ids

    def test_inconclusive_counts_as_anomaly(self):
        results = ValidationResults(results=[
            _make_result(ValidationStatus.PASS, "R001"),
            _make_result(ValidationStatus.INCONCLUSIVE, "R002"),
        ])
        summary = compute_phase_conformance(results)

        assert summary.anomaly_phases == 1
        assert "R002" in summary.anomaly_phase_ids
        assert summary.passed_phases == 1
        assert summary.failed_phases == 0

    def test_pass_with_warnings_counts_as_anomaly(self):
        results = ValidationResults(results=[
            _make_result(ValidationStatus.PASS, "R001"),
            _make_result(ValidationStatus.PASS_WITH_WARNINGS, "R002"),
        ])
        summary = compute_phase_conformance(results)

        # PASS_WITH_WARNINGS is a pass that also raises an anomaly flag.
        assert summary.passed_phases == 2
        assert summary.anomaly_phases == 1
        assert "R002" in summary.anomaly_phase_ids

    def test_multiple_requirements_same_phase_worst_wins(self):
        """If a phase has both PASS and FAIL, phase is FAILED."""
        results = ValidationResults(results=[
            _make_result(ValidationStatus.PASS, "R001"),
            _make_result(ValidationStatus.FAIL, "R001"),
        ])
        summary = compute_phase_conformance(results)

        assert summary.total_phases == 1
        assert summary.passed_phases == 0
        assert summary.failed_phases == 1
        assert summary.anomaly_phases == 1
        assert "R001" in summary.anomaly_phase_ids

    def test_not_applicable_ignored(self):
        results = ValidationResults(results=[
            _make_result(ValidationStatus.PASS, "R001"),
            _make_result(ValidationStatus.NOT_APPLICABLE, "R002"),
        ])
        summary = compute_phase_conformance(results)

        assert summary.total_phases == 1
        assert summary.passed_phases == 1

    def test_cycle_id_used_when_no_region_id(self):
        results = ValidationResults(results=[
            ValidationResult(
                validation_result_id="VR001",
                requirement_id="CYCLE_COUNT",
                requirement_description="Cycle count",
                measured_value=2.0,
                threshold_value=1.0,
                comparator=Comparator.GTE,
                unit="cycles",
                method="test",
                region_id=None,
                cycle_id="C001",
                result=ValidationStatus.PASS,
                reason="Test",
            ),
        ])
        summary = compute_phase_conformance(results)

        assert summary.total_phases == 1
        assert "C001" in summary.anomaly_phase_ids or summary.passed_phases == 1

    def test_empty_results(self):
        results = ValidationResults(results=[])
        summary = compute_phase_conformance(results)

        assert summary.total_phases == 0
        assert summary.passed_phases == 0
        assert summary.conformance_percentage == 0.0

    def test_conformance_percentage_calculation(self):
        results = ValidationResults(results=[
            _make_result(ValidationStatus.PASS, "R001"),
            _make_result(ValidationStatus.PASS, "R002"),
            _make_result(ValidationStatus.PASS, "R003"),
            _make_result(ValidationStatus.FAIL, "R004"),
        ])
        summary = compute_phase_conformance(results)

        assert summary.total_phases == 4
        assert summary.passed_phases == 3
        assert summary.conformance_percentage == 75.0


class TestPhaseConformanceSummaryModel:
    def test_defaults(self):
        pcs = PhaseConformanceSummary()
        assert pcs.total_phases == 0
        assert pcs.passed_phases == 0
        assert pcs.failed_phases == 0
        assert pcs.anomaly_phases == 0
        assert pcs.conformance_percentage == 0.0
        assert pcs.anomaly_phase_ids == []

    def test_instantiation_with_values(self):
        pcs = PhaseConformanceSummary(
            total_phases=5,
            passed_phases=3,
            failed_phases=1,
            anomaly_phases=2,
            conformance_percentage=60.0,
            anomaly_phase_ids=["R002", "R004"],
        )
        assert pcs.total_phases == 5
        assert pcs.passed_phases == 3
        assert pcs.conformance_percentage == 60.0
