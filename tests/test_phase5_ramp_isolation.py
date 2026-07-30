"""Unit tests for Phase 5: workers/ramp_isolation/ — valid ramp extraction."""

import pytest
import numpy as np
from datetime import datetime, timedelta

from config.constants import RegionType, RampDirection
from workers.ramp_isolation.ramp_extractor import (
    isolate_valid_ramps,
    _extract_valid_envelope,
    _compute_monotonicity,
    _compute_reversals_and_stalls,
)
from models.domain import (
    AuditLog,
    CanonicalTraceRow,
    ClassificationEvidence,
    ClassifiedTrace,
    PreprocessingReport,
    Region,
    RegionList,
    ResolvedSetpoints,
    ValidRampRegion,
)


def _make_row(elapsed: float, temp: float, slope: float = 0.0) -> CanonicalTraceRow:
    """Helper to create a classified trace row."""
    base = datetime(2025, 1, 1, 0, 0, 0)
    ts = base + timedelta(seconds=elapsed)
    return CanonicalTraceRow(
        timestamp=ts,
        elapsed_seconds=elapsed,
        elapsed_minutes=elapsed / 60.0,
        temperature_c_raw=temp,
        temperature_c_analysis_signal=temp,
        channel="CH1",
        source_row=int(elapsed),
        source_file="test.csv",
        sample_interval_seconds=1.0,
        rolling_slope_c_per_min=slope,
    )


def _make_region(
    region_id: str,
    start_row: int,
    end_row: int,
    classification: RegionType,
    duration: float = 100.0,
) -> Region:
    """Helper to create a region."""
    base = datetime(2025, 1, 1, 0, 0, 0)
    return Region(
        region_id=region_id,
        start_row=start_row,
        end_row=end_row,
        start_time=base + timedelta(seconds=start_row),
        end_time=base + timedelta(seconds=end_row),
        duration_seconds=duration,
        primary_classification=classification,
        classification_scores={classification: 0.9},
        classification_margin=0.5,
        classification_evidence=[
            ClassificationEvidence(
                evidence_type=classification,
                score=0.9,
                reason="test",
                evidence={"dwell_departure_confidence": 0.8, "dwell_arrival_confidence": 0.8},
                classifier_name="test",
                timestamp=base,
            )
        ],
        classification_confidence=0.9,
    )


class TestExtractValidEnvelope:
    def test_excludes_dwell_tail(self):
        temps = np.array([25.0, 25.0, 25.0, 30.0, 40.0, 50.0])
        slopes = np.array([0.0, 0.0, 0.0, 5.0, 5.0, 5.0])
        elapsed = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        
        start, end, reasons = _extract_valid_envelope(
            temps, slopes, elapsed, is_heating=True,
            noise_floor=0.1, slope_noise_floor=0.5, reversal_tolerance_factor=1.5
        )
        
        assert start >= 2
        assert "dwell tail" in " ".join(reasons).lower() or start > 0

    def test_excludes_overshoot(self):
        temps = np.array([25.0, 50.0, 75.0, 100.0, 105.0, 102.0])
        slopes = np.array([5.0, 5.0, 5.0, 5.0, 1.0, -1.0])
        elapsed = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        
        start, end, reasons = _extract_valid_envelope(
            temps, slopes, elapsed, is_heating=True,
            noise_floor=0.1, slope_noise_floor=0.5, reversal_tolerance_factor=1.5
        )
        
        assert end <= 4


class TestComputeMonotonicity:
    def test_perfect_heating_monotonicity(self):
        slopes = np.array([5.0, 5.0, 5.0, 5.0, 5.0])
        score = _compute_monotonicity(slopes, is_heating=True)
        assert score == 1.0

    def test_perfect_cooling_monotonicity(self):
        slopes = np.array([-5.0, -5.0, -5.0, -5.0, -5.0])
        score = _compute_monotonicity(slopes, is_heating=False)
        assert score == 1.0

    def test_mixed_slopes_lower_monotonicity(self):
        slopes = np.array([5.0, 5.0, -1.0, 5.0, 5.0])
        score = _compute_monotonicity(slopes, is_heating=True)
        assert 0.5 < score < 1.0


class TestComputeReversalsAndStalls:
    def test_no_reversals_in_clean_ramp(self):
        slopes = np.array([5.0, 5.0, 5.0, 5.0, 5.0])
        elapsed = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        
        reversals, stalls = _compute_reversals_and_stalls(
            slopes, elapsed, slope_noise_floor=0.5,
            is_heating=True, reversal_tolerance_factor=1.5, noise_floor=0.1
        )
        
        assert reversals == 0

    def test_stall_duration_computed(self):
        slopes = np.array([5.0, 0.0, 0.0, 5.0, 5.0])
        elapsed = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        
        reversals, stalls = _compute_reversals_and_stalls(
            slopes, elapsed, slope_noise_floor=0.5,
            is_heating=True, reversal_tolerance_factor=1.5, noise_floor=0.1
        )
        
        assert stalls > 0


class TestIsolateValidRamps:
    @pytest.fixture
    def heating_ramp_trace(self):
        rows = []
        for i in range(100):
            temp = 25.0 + (5.0 / 60.0) * i
            slope = 5.0 if i > 5 else 0.0
            rows.append(_make_row(float(i), temp, slope))
        return ClassifiedTrace(rows=rows)

    @pytest.fixture
    def regions(self):
        return RegionList(regions=[
            _make_region("R0001", 0, 99, RegionType.HEATING_RAMP, duration=100.0),
        ])

    @pytest.fixture
    def setpoints(self):
        return ResolvedSetpoints(
            inferred_ambient_c=25.0,
            inferred_hot_setpoint_c=125.0,
            inferred_cold_setpoint_c=-40.0,
        )

    @pytest.fixture
    def preprocessing_report(self):
        return PreprocessingReport(
            noise_floor_c=0.1,
            slope_noise_floor_c_per_min=0.3,
        )

    def test_returns_valid_ramp_list(self, heating_ramp_trace, regions, setpoints, preprocessing_report):
        ramps = isolate_valid_ramps(
            heating_ramp_trace, regions, setpoints, preprocessing_report
        )
        assert isinstance(ramps, list)
        assert all(isinstance(r, ValidRampRegion) for r in ramps)

    def test_heating_ramp_isolated(self, heating_ramp_trace, regions, setpoints, preprocessing_report):
        ramps = isolate_valid_ramps(
            heating_ramp_trace, regions, setpoints, preprocessing_report
        )
        assert len(ramps) >= 1
        assert ramps[0].direction == RampDirection.HEATING

    def test_valid_ramp_has_required_fields(self, heating_ramp_trace, regions, setpoints, preprocessing_report):
        ramps = isolate_valid_ramps(
            heating_ramp_trace, regions, setpoints, preprocessing_report
        )
        ramp = ramps[0]
        assert ramp.region_id is not None
        assert ramp.duration_seconds > 0
        assert len(ramp.included_rows) > 0
        assert ramp.monotonicity_score > 0

    def test_audit_log_records_isolation(self, heating_ramp_trace, regions, setpoints, preprocessing_report):
        audit_log = AuditLog()
        isolate_valid_ramps(
            heating_ramp_trace, regions, setpoints, preprocessing_report,
            audit_log=audit_log
        )
        entries = [e for e in audit_log.entries if "isolate" in e.action]
        assert len(entries) >= 1

    def test_short_ramp_excluded(self, setpoints, preprocessing_report):
        rows = [_make_row(float(i), 25.0 + i * 0.5, 5.0) for i in range(10)]
        trace = ClassifiedTrace(rows=rows)
        regions = RegionList(regions=[
            _make_region("R0001", 0, 9, RegionType.HEATING_RAMP, duration=10.0),
        ])
        
        ramps = isolate_valid_ramps(
            trace, regions, setpoints, preprocessing_report,
            min_ramp_duration_seconds=30.0
        )
        
        assert len(ramps) == 0

    def test_non_ramp_regions_skipped(self, heating_ramp_trace, setpoints, preprocessing_report):
        regions = RegionList(regions=[
            _make_region("R0001", 0, 99, RegionType.HOT_DWELL, duration=100.0),
        ])
        
        ramps = isolate_valid_ramps(
            heating_ramp_trace, regions, setpoints, preprocessing_report
        )
        
        assert len(ramps) == 0
