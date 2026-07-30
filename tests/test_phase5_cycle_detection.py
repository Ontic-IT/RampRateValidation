"""Unit tests for Phase 5: workers/cycle_segmentation/ — cycle detection."""

import pytest
from datetime import datetime, timedelta

from config.constants import CycleStatus, RegionType
from workers.cycle_segmentation.cycle_detector import (
    detect_cycles,
    _find_cycle_boundaries,
    _evaluate_cycle_completeness,
    _is_cycle_transition,
)
from models.domain import (
    AuditLog,
    ClassificationEvidence,
    Cycle,
    CycleList,
    Region,
    RegionList,
    ResolvedSetpoints,
    ValidRampRegion,
)


def _make_region(
    region_id: str,
    classification: RegionType,
    start_row: int = 0,
    end_row: int = 100,
) -> Region:
    """Helper to create a region."""
    base = datetime(2025, 1, 1, 0, 0, 0)
    return Region(
        region_id=region_id,
        start_row=start_row,
        end_row=end_row,
        start_time=base + timedelta(seconds=start_row),
        end_time=base + timedelta(seconds=end_row),
        duration_seconds=float(end_row - start_row),
        primary_classification=classification,
        classification_scores={classification: 0.9},
        classification_margin=0.5,
        classification_evidence=[],
        classification_confidence=0.9,
    )


class TestFindCycleBoundaries:
    def test_single_cycle(self):
        regions = [
            _make_region("R0001", RegionType.HEATING_RAMP, 0, 100),
            _make_region("R0002", RegionType.HOT_DWELL, 100, 200),
            _make_region("R0003", RegionType.COOLING_RAMP, 200, 300),
            _make_region("R0004", RegionType.COLD_DWELL, 300, 400),
        ]
        setpoints = ResolvedSetpoints()
        
        boundaries = _find_cycle_boundaries(regions, setpoints)
        
        assert len(boundaries) >= 1
        assert boundaries[0] == (0, 3)

    def test_multiple_cycles(self):
        regions = [
            _make_region("R0001", RegionType.HEATING_RAMP, 0, 100),
            _make_region("R0002", RegionType.HOT_DWELL, 100, 200),
            _make_region("R0003", RegionType.COOLING_RAMP, 200, 300),
            _make_region("R0004", RegionType.COLD_DWELL, 300, 400),
            _make_region("R0005", RegionType.HEATING_RAMP, 400, 500),
            _make_region("R0006", RegionType.HOT_DWELL, 500, 600),
        ]
        setpoints = ResolvedSetpoints()
        
        boundaries = _find_cycle_boundaries(regions, setpoints)
        
        assert len(boundaries) >= 1


class TestEvaluateCycleCompleteness:
    def test_complete_cycle(self):
        regions = [
            _make_region("R0001", RegionType.HEATING_RAMP),
            _make_region("R0002", RegionType.HOT_DWELL),
            _make_region("R0003", RegionType.COOLING_RAMP),
            _make_region("R0004", RegionType.COLD_DWELL),
        ]
        setpoints = ResolvedSetpoints()
        
        status, reason = _evaluate_cycle_completeness(regions, setpoints)
        
        assert status == CycleStatus.COMPLETE
        assert "full" in reason.lower() or "thermal" in reason.lower()

    def test_partial_heating_only(self):
        regions = [
            _make_region("R0001", RegionType.HEATING_RAMP),
            _make_region("R0002", RegionType.HOT_DWELL),
        ]
        setpoints = ResolvedSetpoints()
        
        status, reason = _evaluate_cycle_completeness(regions, setpoints)
        
        assert status == CycleStatus.PARTIAL

    def test_invalid_no_ramps(self):
        regions = [
            _make_region("R0001", RegionType.HOT_DWELL),
            _make_region("R0002", RegionType.COLD_DWELL),
        ]
        setpoints = ResolvedSetpoints()
        
        status, reason = _evaluate_cycle_completeness(regions, setpoints)
        
        assert status == CycleStatus.INVALID


class TestIsCycleTransition:
    def test_cold_dwell_to_heating_is_transition(self):
        prev = _make_region("R0001", RegionType.COLD_DWELL)
        curr = _make_region("R0002", RegionType.HEATING_RAMP)
        
        assert _is_cycle_transition(prev, curr) is True

    def test_heating_to_hot_dwell_not_transition(self):
        prev = _make_region("R0001", RegionType.HEATING_RAMP)
        curr = _make_region("R0002", RegionType.HOT_DWELL)
        
        assert _is_cycle_transition(prev, curr) is False


class TestDetectCycles:
    @pytest.fixture
    def full_cycle_regions(self):
        return RegionList(regions=[
            _make_region("R0001", RegionType.HEATING_RAMP, 0, 100),
            _make_region("R0002", RegionType.HOT_DWELL, 100, 400),
            _make_region("R0003", RegionType.COOLING_RAMP, 400, 500),
            _make_region("R0004", RegionType.COLD_DWELL, 500, 800),
        ])

    @pytest.fixture
    def setpoints(self):
        return ResolvedSetpoints(
            inferred_ambient_c=25.0,
            inferred_hot_setpoint_c=125.0,
            inferred_cold_setpoint_c=-40.0,
        )

    def test_returns_cycle_list(self, full_cycle_regions, setpoints):
        cycles = detect_cycles(full_cycle_regions, [], setpoints)
        assert isinstance(cycles, CycleList)

    def test_detects_complete_cycle(self, full_cycle_regions, setpoints):
        cycles = detect_cycles(full_cycle_regions, [], setpoints)
        assert len(cycles.cycles) >= 1
        assert cycles.cycles[0].status == CycleStatus.COMPLETE

    def test_cycle_has_required_fields(self, full_cycle_regions, setpoints):
        cycles = detect_cycles(full_cycle_regions, [], setpoints)
        cycle = cycles.cycles[0]
        assert cycle.cycle_id is not None
        assert cycle.duration_seconds > 0
        assert len(cycle.region_ids) > 0

    def test_cycle_counts_ramps_and_dwells(self, full_cycle_regions, setpoints):
        cycles = detect_cycles(full_cycle_regions, [], setpoints)
        cycle = cycles.cycles[0]
        assert cycle.hot_dwell_count == 1
        assert cycle.cold_dwell_count == 1

    def test_audit_log_records_detection(self, full_cycle_regions, setpoints):
        audit_log = AuditLog()
        detect_cycles(full_cycle_regions, [], setpoints, audit_log=audit_log)
        entries = [e for e in audit_log.entries if "cycle" in e.action]
        assert len(entries) >= 1

    def test_empty_regions_returns_empty_list(self, setpoints):
        cycles = detect_cycles(RegionList(regions=[]), [], setpoints)
        assert len(cycles.cycles) == 0

    def test_min_regions_enforced(self, setpoints):
        regions = RegionList(regions=[
            _make_region("R0001", RegionType.HEATING_RAMP, 0, 100),
        ])
        cycles = detect_cycles(regions, [], setpoints, min_cycle_regions=2)
        assert len(cycles.cycles) == 0
