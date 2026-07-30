"""Unit tests for Phase 6: engine/validation/aggregation.py — status aggregation."""

import pytest

from config.constants import OverallValidationStatus, ValidationStatus, Comparator
from engine.validation.aggregation import aggregate_validation_status
from models.domain import (
    AuditLog,
    OverallStatus,
    ValidationDataQualityImpact,
    ValidationResult,
    ValidationResults,
)


def _make_result(status: ValidationStatus, requirement_id: str = "REQ001") -> ValidationResult:
    """Helper to create a validation result."""
    return ValidationResult(
        validation_result_id="VR001",
        requirement_id=requirement_id,
        requirement_description="Test requirement",
        measured_value=5.0,
        threshold_value=5.0,
        comparator=Comparator.GTE,
        unit="test",
        method="test",
        result=status,
        reason="Test reason",
    )


class TestAggregateValidationStatus:
    def test_all_pass_returns_pass(self):
        results = ValidationResults(results=[
            _make_result(ValidationStatus.PASS, "REQ001"),
            _make_result(ValidationStatus.PASS, "REQ002"),
            _make_result(ValidationStatus.PASS, "REQ003"),
        ])
        
        overall = aggregate_validation_status(results)
        
        assert overall.status == OverallValidationStatus.PASS

    def test_any_fail_returns_fail(self):
        results = ValidationResults(results=[
            _make_result(ValidationStatus.PASS, "REQ001"),
            _make_result(ValidationStatus.FAIL, "REQ002"),
            _make_result(ValidationStatus.PASS, "REQ003"),
        ])
        
        overall = aggregate_validation_status(results)
        
        assert overall.status == OverallValidationStatus.FAIL
        assert "REQ002" in overall.reason

    def test_any_inconclusive_returns_inconclusive(self):
        results = ValidationResults(results=[
            _make_result(ValidationStatus.PASS, "REQ001"),
            _make_result(ValidationStatus.INCONCLUSIVE, "REQ002"),
        ])
        
        overall = aggregate_validation_status(results)
        
        assert overall.status == OverallValidationStatus.INCONCLUSIVE

    def test_ramp_rate_fail_forces_overall_fail(self):
        # Requirement hierarchy: ramp rates ARE the test — a ramp-rate FAIL
        # forces overall FAIL regardless of other results.
        results = ValidationResults(results=[
            _make_result(ValidationStatus.FAIL, "HEATING_RAMP_RATE"),
            _make_result(ValidationStatus.INCONCLUSIVE, "REQ002"),
        ])

        overall = aggregate_validation_status(results)

        assert overall.status == OverallValidationStatus.FAIL

    def test_non_ramp_fail_is_maintenance_advisory_not_fail(self):
        # Setpoint-tracking failures indicate chamber health, not test
        # failure: overall status degrades to PASS_WITH_WARNINGS.
        results = ValidationResults(results=[
            _make_result(ValidationStatus.PASS, "HEATING_RAMP_RATE"),
            _make_result(ValidationStatus.FAIL, "SETPOINT_DEVIATION"),
        ])

        overall = aggregate_validation_status(results)

        assert overall.status == OverallValidationStatus.PASS_WITH_WARNINGS
        assert "maintenance" in overall.reason.lower()

    def test_warnings_returns_pass_with_warnings(self):
        results = ValidationResults(results=[
            _make_result(ValidationStatus.PASS, "REQ001"),
            _make_result(ValidationStatus.PASS_WITH_WARNINGS, "REQ002"),
        ])
        
        overall = aggregate_validation_status(results)
        
        assert overall.status == OverallValidationStatus.PASS_WITH_WARNINGS

    def test_quality_blocked_returns_inconclusive(self):
        results = ValidationResults(results=[
            _make_result(ValidationStatus.PASS, "REQ001"),
        ])
        quality_impact = ValidationDataQualityImpact(
            blocks_pass_fail=True,
            reason="Data quality too poor",
        )
        
        overall = aggregate_validation_status(results, quality_impact=quality_impact)
        
        assert overall.status == OverallValidationStatus.INCONCLUSIVE
        assert "Data quality" in overall.reason

    def test_not_applicable_ignored(self):
        results = ValidationResults(results=[
            _make_result(ValidationStatus.PASS, "REQ001"),
            _make_result(ValidationStatus.NOT_APPLICABLE, "REQ002"),
        ])
        
        overall = aggregate_validation_status(results)
        
        assert overall.status == OverallValidationStatus.PASS

    def test_empty_results_returns_pass(self):
        results = ValidationResults(results=[])
        
        overall = aggregate_validation_status(results)
        
        assert overall.status == OverallValidationStatus.PASS

    def test_audit_log_recorded(self):
        results = ValidationResults(results=[
            _make_result(ValidationStatus.PASS, "REQ001"),
        ])
        audit_log = AuditLog()
        
        aggregate_validation_status(results, audit_log=audit_log)
        
        entries = [e for e in audit_log.entries if "aggregate" in e.action]
        assert len(entries) == 1

    def test_reason_lists_failed_requirements(self):
        results = ValidationResults(results=[
            _make_result(ValidationStatus.FAIL, "HEATING_RAMP_RATE"),
            _make_result(ValidationStatus.FAIL, "COOLING_RAMP_RATE"),
            _make_result(ValidationStatus.PASS, "DWELL_DURATION"),
        ])
        
        overall = aggregate_validation_status(results)
        
        assert "HEATING_RAMP_RATE" in overall.reason
        assert "COOLING_RAMP_RATE" in overall.reason
