"""Unit tests for Phase 6: engine/validation/ramp_rules.py — ramp validation rules."""

import pytest
from datetime import datetime

from config.constants import RampDirection, ValidationStatus
from engine.validation.ramp_rules import (
    validate_heating_ramp_rate,
    validate_cooling_ramp_rate,
    validate_minimum_sustained_ramp_rate,
    validate_data_quality,
)
from models.domain import (
    AuditLog,
    RampMetrics,
    ValidRampRegion,
    ValidationDataQualityImpact,
)


class TestValidateHeatingRampRate:
    @pytest.fixture
    def heating_ramp(self):
        return ValidRampRegion(
            region_id="R0001",
            direction=RampDirection.HEATING,
            start_row=0,
            end_row=100,
            duration_seconds=1200.0,
            included_rows=list(range(100)),
        )

    @pytest.fixture
    def passing_metrics(self):
        return RampMetrics(
            region_id="R0001",
            robust_slope_c_per_min=6.0,
            minimum_sustained_slope_c_per_min=5.0,
        )

    @pytest.fixture
    def failing_metrics(self):
        return RampMetrics(
            region_id="R0001",
            robust_slope_c_per_min=3.0,
            minimum_sustained_slope_c_per_min=2.5,
        )

    def test_pass_when_within_band(self, heating_ramp, passing_metrics):
        # target 5.0, tolerance 1.0 -> 6.0 is at the band edge -> PASS
        result = validate_heating_ramp_rate(
            heating_ramp, passing_metrics,
            required_rate_c_per_min=5.0,
            tolerance_c_per_min=1.0,
        )
        assert result.result == ValidationStatus.PASS
        assert result.measured_value == 6.0

    def test_fail_when_below_band(self, heating_ramp, failing_metrics):
        result = validate_heating_ramp_rate(
            heating_ramp, failing_metrics,
            required_rate_c_per_min=5.0,
            tolerance_c_per_min=1.0,
        )
        assert result.result == ValidationStatus.FAIL

    def test_fail_when_above_band(self, heating_ramp):
        # Ramping much FASTER than commanded is off-profile -> FAIL (band,
        # not floor: a fast ramp is no longer a free pass).
        metrics = RampMetrics(
            region_id="R0001",
            robust_slope_c_per_min=8.0,
            minimum_sustained_slope_c_per_min=7.0,
        )
        result = validate_heating_ramp_rate(
            heating_ramp, metrics,
            required_rate_c_per_min=5.0,
            tolerance_c_per_min=1.0,
        )
        assert result.result == ValidationStatus.FAIL
        assert "faster" in result.reason

    def test_not_applicable_for_cooling_ramp(self, passing_metrics):
        cooling_ramp = ValidRampRegion(
            region_id="R0001",
            direction=RampDirection.COOLING,
            included_rows=list(range(100)),
        )
        result = validate_heating_ramp_rate(
            cooling_ramp, passing_metrics,
            required_rate_c_per_min=5.0,
        )
        assert result.result == ValidationStatus.NOT_APPLICABLE

    def test_audit_log_recorded(self, heating_ramp, passing_metrics):
        audit_log = AuditLog()
        validate_heating_ramp_rate(
            heating_ramp, passing_metrics,
            required_rate_c_per_min=5.0,
            audit_log=audit_log,
        )
        entries = [e for e in audit_log.entries if "heating_ramp" in e.action]
        assert len(entries) == 1

    def test_result_has_required_fields(self, heating_ramp, passing_metrics):
        result = validate_heating_ramp_rate(
            heating_ramp, passing_metrics,
            required_rate_c_per_min=5.0,
        )
        assert result.validation_result_id is not None
        assert result.requirement_id == "HEATING_RAMP_RATE"
        assert result.method == "theil_sen"
        assert result.unit == "°C/min"


class TestValidateCoolingRampRate:
    @pytest.fixture
    def cooling_ramp(self):
        return ValidRampRegion(
            region_id="R0001",
            direction=RampDirection.COOLING,
            start_row=0,
            end_row=100,
            duration_seconds=1980.0,
            included_rows=list(range(100)),
        )

    @pytest.fixture
    def passing_metrics(self):
        return RampMetrics(
            region_id="R0001",
            robust_slope_c_per_min=-4.0,
            minimum_sustained_slope_c_per_min=-3.5,
        )

    def test_pass_when_within_band(self, cooling_ramp):
        # target 3.0, tolerance 0.6 -> |3.2| within band -> PASS
        metrics = RampMetrics(
            region_id="R0001",
            robust_slope_c_per_min=-3.2,
            minimum_sustained_slope_c_per_min=-3.0,
        )
        result = validate_cooling_ramp_rate(
            cooling_ramp, metrics,
            required_rate_c_per_min=3.0,
            tolerance_c_per_min=0.6,
        )
        assert result.result == ValidationStatus.PASS

    def test_fail_when_below_band(self, cooling_ramp):
        metrics = RampMetrics(
            region_id="R0001",
            robust_slope_c_per_min=-2.0,
            minimum_sustained_slope_c_per_min=-1.5,
        )
        result = validate_cooling_ramp_rate(
            cooling_ramp, metrics,
            required_rate_c_per_min=3.0,
            tolerance_c_per_min=0.6,
        )
        assert result.result == ValidationStatus.FAIL

    def test_fail_when_above_band(self, cooling_ramp, passing_metrics):
        # |−4.0| far above the 3.0 ± 0.6 band -> FAIL (too fast is off-profile)
        result = validate_cooling_ramp_rate(
            cooling_ramp, passing_metrics,
            required_rate_c_per_min=3.0,
            tolerance_c_per_min=0.6,
        )
        assert result.result == ValidationStatus.FAIL
        assert "faster" in result.reason


class TestValidateMinimumSustainedRampRate:
    def test_passes_when_above_threshold(self):
        metrics = RampMetrics(
            region_id="R0001",
            minimum_sustained_slope_c_per_min=4.5,
        )
        passes, reason = validate_minimum_sustained_ramp_rate(
            metrics,
            required_rate_c_per_min=5.0,
            minimum_sustained_ratio=0.8,
        )
        assert passes is True
        assert "4.50" in reason

    def test_fails_when_below_threshold(self):
        metrics = RampMetrics(
            region_id="R0001",
            minimum_sustained_slope_c_per_min=3.0,
        )
        passes, reason = validate_minimum_sustained_ramp_rate(
            metrics,
            required_rate_c_per_min=5.0,
            minimum_sustained_ratio=0.8,
        )
        assert passes is False


class TestValidateDataQuality:
    def test_pass_when_not_blocked(self):
        impact = ValidationDataQualityImpact(
            blocks_pass_fail=False,
        )
        status, reason = validate_data_quality(impact, "HEATING_RAMP_RATE")
        assert status == ValidationStatus.PASS

    def test_inconclusive_when_blocked_all(self):
        impact = ValidationDataQualityImpact(
            blocks_pass_fail=True,
            affected_requirement_ids=["ALL"],
            reason="Data quality too poor",
        )
        status, reason = validate_data_quality(impact, "HEATING_RAMP_RATE")
        assert status == ValidationStatus.INCONCLUSIVE

    def test_inconclusive_when_specific_requirement_blocked(self):
        impact = ValidationDataQualityImpact(
            blocks_pass_fail=True,
            affected_requirement_ids=["HEATING_RAMP_RATE"],
            reason="Gaps in heating ramp",
        )
        status, reason = validate_data_quality(impact, "HEATING_RAMP_RATE")
        assert status == ValidationStatus.INCONCLUSIVE

    def test_pass_when_different_requirement_blocked(self):
        impact = ValidationDataQualityImpact(
            blocks_pass_fail=True,
            affected_requirement_ids=["DWELL_DURATION"],
            reason="Gaps in dwell",
        )
        status, reason = validate_data_quality(impact, "HEATING_RAMP_RATE")
        assert status == ValidationStatus.PASS
