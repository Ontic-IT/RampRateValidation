"""Unit tests for Phase 6: engine/validation/cycle_rules.py — cycle validation rules."""

import pytest
from datetime import datetime, timedelta

from config.constants import CycleStatus, RegionType, ValidationStatus, ConfidenceLevel
from engine.validation.cycle_rules import (
    validate_cycle_count,
    validate_profile_sequence,
    _check_sequence_match,
)
from models.domain import (
    AuditLog,
    Cycle,
    CycleList,
    Region,
)


def _make_cycle(cycle_id: str, status: CycleStatus = CycleStatus.COMPLETE) -> Cycle:
    """Helper to create a cycle."""
    base = datetime(2025, 1, 1, 0, 0, 0)
    return Cycle(
        cycle_id=cycle_id,
        cycle_number=1,
        start_row=0,
        end_row=1000,
        start_time=base,
        end_time=base + timedelta(seconds=3600),
        duration_seconds=3600.0,
        status=status,
    )


def _make_region(region_id: str, classification: RegionType) -> Region:
    """Helper to create a region."""
    base = datetime(2025, 1, 1, 0, 0, 0)
    return Region(
        region_id=region_id,
        start_row=0,
        end_row=100,
        start_time=base,
        end_time=base + timedelta(seconds=100),
        duration_seconds=100.0,
        primary_classification=classification,
        classification_scores={classification: 0.9},
        classification_margin=0.5,
        classification_evidence=[],
        classification_confidence=0.9,
    )


class TestValidateCycleCount:
    def test_pass_when_above_minimum(self):
        cycles = CycleList(cycles=[
            _make_cycle("C0001"),
            _make_cycle("C0002"),
            _make_cycle("C0003"),
        ])
        
        result = validate_cycle_count(cycles, minimum_cycles=2)
        
        assert result.result == ValidationStatus.PASS
        assert result.measured_value == 3.0

    def test_fail_when_below_minimum(self):
        cycles = CycleList(cycles=[
            _make_cycle("C0001"),
        ])
        
        result = validate_cycle_count(cycles, minimum_cycles=3)
        
        assert result.result == ValidationStatus.FAIL

    def test_pass_when_within_range(self):
        cycles = CycleList(cycles=[
            _make_cycle("C0001"),
            _make_cycle("C0002"),
        ])
        
        result = validate_cycle_count(cycles, minimum_cycles=1, maximum_cycles=5)
        
        assert result.result == ValidationStatus.PASS

    def test_fail_when_above_maximum(self):
        cycles = CycleList(cycles=[
            _make_cycle("C0001"),
            _make_cycle("C0002"),
            _make_cycle("C0003"),
        ])
        
        result = validate_cycle_count(cycles, minimum_cycles=1, maximum_cycles=2)
        
        assert result.result == ValidationStatus.FAIL

    def test_only_counts_complete_cycles(self):
        cycles = CycleList(cycles=[
            _make_cycle("C0001", CycleStatus.COMPLETE),
            _make_cycle("C0002", CycleStatus.PARTIAL),
            _make_cycle("C0003", CycleStatus.COMPLETE),
        ])
        
        result = validate_cycle_count(cycles, minimum_cycles=3)
        
        assert result.result == ValidationStatus.FAIL
        assert result.measured_value == 2.0

    def test_audit_log_recorded(self):
        cycles = CycleList(cycles=[_make_cycle("C0001")])
        audit_log = AuditLog()
        
        validate_cycle_count(cycles, minimum_cycles=1, audit_log=audit_log)
        
        entries = [e for e in audit_log.entries if "cycle_count" in e.action]
        assert len(entries) == 1


class TestCheckSequenceMatch:
    def test_exact_match(self):
        actual = ["HEATING_RAMP", "HOT_DWELL", "COOLING_RAMP", "COLD_DWELL"]
        expected = ["HEATING_RAMP", "HOT_DWELL", "COOLING_RAMP", "COLD_DWELL"]
        
        matches, details = _check_sequence_match(actual, expected)
        
        assert matches is True

    def test_match_with_extra_regions(self):
        actual = ["HEATING_RAMP", "TRANSIENT", "HOT_DWELL", "HOT_CORRECTION", "COOLING_RAMP", "COLD_DWELL"]
        expected = ["HEATING_RAMP", "HOT_DWELL", "COOLING_RAMP", "COLD_DWELL"]
        
        matches, details = _check_sequence_match(actual, expected)
        
        assert matches is True

    def test_no_match_missing_region(self):
        actual = ["HEATING_RAMP", "HOT_DWELL", "COLD_DWELL"]
        expected = ["HEATING_RAMP", "HOT_DWELL", "COOLING_RAMP", "COLD_DWELL"]
        
        matches, details = _check_sequence_match(actual, expected)
        
        assert matches is False
        assert "COOLING_RAMP" in details

    def test_empty_expected_always_matches(self):
        actual = ["HEATING_RAMP", "HOT_DWELL"]
        expected = []
        
        matches, details = _check_sequence_match(actual, expected)
        
        assert matches is True


class TestValidateProfileSequence:
    def test_pass_when_sequence_matches(self):
        regions = [
            _make_region("R0001", RegionType.HEATING_RAMP),
            _make_region("R0002", RegionType.HOT_DWELL),
            _make_region("R0003", RegionType.COOLING_RAMP),
            _make_region("R0004", RegionType.COLD_DWELL),
        ]
        expected = ["HEATING_RAMP", "HOT_DWELL", "COOLING_RAMP", "COLD_DWELL"]
        
        result = validate_profile_sequence(regions, expected)
        
        assert result.result == ValidationStatus.PASS

    def test_fail_when_sequence_mismatch(self):
        regions = [
            _make_region("R0001", RegionType.HEATING_RAMP),
            _make_region("R0002", RegionType.COLD_DWELL),
        ]
        expected = ["HEATING_RAMP", "HOT_DWELL", "COOLING_RAMP", "COLD_DWELL"]
        
        result = validate_profile_sequence(regions, expected)
        
        assert result.result == ValidationStatus.FAIL

    def test_not_applicable_when_no_expected(self):
        regions = [
            _make_region("R0001", RegionType.HEATING_RAMP),
        ]
        
        result = validate_profile_sequence(regions, [])
        
        assert result.result == ValidationStatus.NOT_APPLICABLE

    def test_result_has_required_fields(self):
        regions = [
            _make_region("R0001", RegionType.HEATING_RAMP),
            _make_region("R0002", RegionType.HOT_DWELL),
        ]
        expected = ["HEATING_RAMP", "HOT_DWELL"]
        
        result = validate_profile_sequence(regions, expected)
        
        assert result.validation_result_id is not None
        assert result.requirement_id == "PROFILE_SEQUENCE"
        assert result.method == "sequence_matching"
