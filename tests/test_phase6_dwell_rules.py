"""Unit tests for Phase 6: engine/validation/dwell_rules.py — dwell validation rules."""

import pytest
from datetime import datetime, timedelta

from config.constants import RegionType, ValidationStatus, ConfidenceLevel
from engine.validation.dwell_rules import (
    validate_dwell_duration,
    validate_setpoint_deviation,
    validate_overshoot,
    validate_settling_time,
)
from models.domain import (
    AuditLog,
    DwellMetrics,
    Region,
)


def _make_dwell_region(region_type: RegionType, duration: float = 300.0) -> Region:
    """Helper to create a dwell region."""
    base = datetime(2025, 1, 1, 0, 0, 0)
    return Region(
        region_id="R0001",
        start_row=0,
        end_row=100,
        start_time=base,
        end_time=base + timedelta(seconds=duration),
        duration_seconds=duration,
        primary_classification=region_type,
        classification_scores={region_type: 0.9},
        classification_margin=0.5,
        classification_evidence=[],
        classification_confidence=0.9,
    )


class TestValidateDwellDuration:
    def test_pass_when_above_minimum(self):
        region = _make_dwell_region(RegionType.HOT_DWELL, duration=600.0)
        metrics = DwellMetrics(region_id="R0001")
        
        result = validate_dwell_duration(region, metrics, minimum_duration_seconds=300.0)
        
        assert result.result == ValidationStatus.PASS
        assert result.measured_value == 600.0

    def test_fail_when_below_minimum(self):
        region = _make_dwell_region(RegionType.HOT_DWELL, duration=200.0)
        metrics = DwellMetrics(region_id="R0001")
        
        result = validate_dwell_duration(region, metrics, minimum_duration_seconds=300.0)
        
        assert result.result == ValidationStatus.FAIL

    def test_not_applicable_for_ramp(self):
        region = _make_dwell_region(RegionType.HEATING_RAMP, duration=600.0)
        metrics = DwellMetrics(region_id="R0001")
        
        result = validate_dwell_duration(region, metrics, minimum_duration_seconds=300.0)
        
        assert result.result == ValidationStatus.NOT_APPLICABLE

    def test_audit_log_recorded(self):
        region = _make_dwell_region(RegionType.HOT_DWELL, duration=600.0)
        metrics = DwellMetrics(region_id="R0001")
        audit_log = AuditLog()
        
        validate_dwell_duration(region, metrics, minimum_duration_seconds=300.0, audit_log=audit_log)
        
        entries = [e for e in audit_log.entries if "dwell_duration" in e.action]
        assert len(entries) == 1


class TestValidateSetpointDeviation:
    def test_pass_when_within_tolerance(self):
        region = _make_dwell_region(RegionType.HOT_DWELL)
        metrics = DwellMetrics(region_id="R0001", setpoint_deviation_c=2.0)
        
        result = validate_setpoint_deviation(region, metrics, maximum_deviation_c=5.0)
        
        assert result.result == ValidationStatus.PASS

    def test_fail_when_exceeds_tolerance(self):
        region = _make_dwell_region(RegionType.HOT_DWELL)
        metrics = DwellMetrics(region_id="R0001", setpoint_deviation_c=8.0)
        
        result = validate_setpoint_deviation(region, metrics, maximum_deviation_c=5.0)
        
        assert result.result == ValidationStatus.FAIL

    def test_not_applicable_for_ambient(self):
        region = _make_dwell_region(RegionType.AMBIENT_START)
        metrics = DwellMetrics(region_id="R0001", setpoint_deviation_c=2.0)
        
        result = validate_setpoint_deviation(region, metrics, maximum_deviation_c=5.0)
        
        assert result.result == ValidationStatus.NOT_APPLICABLE


class TestValidateOvershoot:
    def test_pass_when_within_limit(self):
        region = _make_dwell_region(RegionType.HOT_DWELL)
        metrics = DwellMetrics(region_id="R0001", overshoot_magnitude_c=3.0)
        
        result = validate_overshoot(region, metrics, maximum_overshoot_c=5.0)
        
        assert result.result == ValidationStatus.PASS

    def test_fail_when_exceeds_limit(self):
        region = _make_dwell_region(RegionType.HOT_DWELL)
        metrics = DwellMetrics(region_id="R0001", overshoot_magnitude_c=8.0)
        
        result = validate_overshoot(region, metrics, maximum_overshoot_c=5.0)
        
        assert result.result == ValidationStatus.FAIL

    def test_pass_when_no_overshoot(self):
        region = _make_dwell_region(RegionType.COLD_DWELL)
        metrics = DwellMetrics(region_id="R0001", overshoot_magnitude_c=0.0)
        
        result = validate_overshoot(region, metrics, maximum_overshoot_c=5.0)
        
        assert result.result == ValidationStatus.PASS


class TestValidateSettlingTime:
    def test_pass_when_within_limit(self):
        region = _make_dwell_region(RegionType.HOT_DWELL)
        metrics = DwellMetrics(region_id="R0001", settling_time_seconds=20.0)
        
        result = validate_settling_time(region, metrics, maximum_settling_time_seconds=30.0)
        
        assert result.result == ValidationStatus.PASS

    def test_fail_when_exceeds_limit(self):
        region = _make_dwell_region(RegionType.HOT_DWELL)
        metrics = DwellMetrics(region_id="R0001", settling_time_seconds=45.0)
        
        result = validate_settling_time(region, metrics, maximum_settling_time_seconds=30.0)
        
        assert result.result == ValidationStatus.FAIL

    def test_result_has_required_fields(self):
        region = _make_dwell_region(RegionType.HOT_DWELL)
        metrics = DwellMetrics(region_id="R0001", settling_time_seconds=20.0)
        
        result = validate_settling_time(region, metrics, maximum_settling_time_seconds=30.0)
        
        assert result.validation_result_id is not None
        assert result.requirement_id == "SETTLING_TIME"
        assert result.method == "tolerance_band_entry"
        assert result.unit == "s"
