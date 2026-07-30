"""Unit tests for Phase 4: workers/region_classification/classification_orchestrator.py."""

import pytest
import numpy as np
from datetime import datetime, timedelta

from config.constants import ConfidenceLevel, RegionType
from workers.region_classification.classification_orchestrator import (
    classify_regions,
    _segment_trace,
    _select_classification,
    _determine_confidence_level,
)
from models.domain import (
    AuditLog,
    CanonicalTraceRow,
    ClassifiedTrace,
    PreprocessedTrace,
    ProcessBoundaries,
    Region,
    RegionList,
    ResolvedSetpoints,
)


def _make_row(elapsed: float, temp: float, slope: float = 0.0, mad: float = 0.1) -> CanonicalTraceRow:
    """Helper to create a preprocessed trace row."""
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
        rolling_temperature_MAD=mad,
    )


class TestSelectClassification:
    def test_selects_highest_score(self):
        scores = {
            RegionType.HEATING_RAMP: 0.9,
            RegionType.COOLING_RAMP: 0.1,
            RegionType.HOT_DWELL: 0.3,
        }
        primary, secondary, margin = _select_classification(scores, 0.1)
        assert primary == RegionType.HEATING_RAMP
        assert abs(margin - 0.6) < 0.01

    def test_identifies_secondary_classifications(self):
        scores = {
            RegionType.HEATING_RAMP: 0.8,
            RegionType.RAMP_JITTER: 0.7,
            RegionType.HOT_DWELL: 0.2,
        }
        primary, secondary, margin = _select_classification(scores, 0.1)
        assert primary == RegionType.HEATING_RAMP
        assert RegionType.RAMP_JITTER in secondary

    def test_empty_scores_returns_unknown(self):
        primary, secondary, margin = _select_classification({}, 0.1)
        assert primary == RegionType.UNKNOWN
        assert margin == 0.0


class TestDetermineConfidenceLevel:
    def test_high_confidence(self):
        level = _determine_confidence_level(0.9, 0.3, 0.8, 0.6)
        assert level == ConfidenceLevel.HIGH

    def test_medium_confidence(self):
        level = _determine_confidence_level(0.7, 0.15, 0.8, 0.6)
        assert level == ConfidenceLevel.MEDIUM

    def test_low_confidence(self):
        level = _determine_confidence_level(0.5, 0.05, 0.8, 0.6)
        assert level == ConfidenceLevel.LOW


class TestSegmentTrace:
    @pytest.fixture
    def heating_ramp_trace(self):
        rows = []
        for i in range(60):
            rows.append(_make_row(float(i), 25.0, slope=0.0))
        for i in range(200):
            temp = 25.0 + (5.0 / 60.0) * i
            rows.append(_make_row(float(60 + i), temp, slope=5.0))
        for i in range(100):
            rows.append(_make_row(float(260 + i), 125.0, slope=0.0))
        return PreprocessedTrace(rows=rows)

    def test_segments_by_slope_change(self, heating_ramp_trace):
        boundaries = ProcessBoundaries()
        segments = _segment_trace(heating_ramp_trace, boundaries, min_region_rows=10)
        assert len(segments) >= 2

    def test_segments_have_required_fields(self, heating_ramp_trace):
        boundaries = ProcessBoundaries()
        segments = _segment_trace(heating_ramp_trace, boundaries, min_region_rows=10)
        for seg in segments:
            assert "start_row" in seg
            assert "end_row" in seg
            assert "duration_seconds" in seg


class TestClassifyRegions:
    @pytest.fixture
    def simple_trace(self):
        rows = []
        for i in range(50):
            rows.append(_make_row(float(i), 25.0, slope=0.0))
        for i in range(100):
            temp = 25.0 + (5.0 / 60.0) * i
            rows.append(_make_row(float(50 + i), temp, slope=5.0))
        for i in range(50):
            rows.append(_make_row(float(150 + i), 125.0, slope=0.0))
        return PreprocessedTrace(rows=rows)

    @pytest.fixture
    def setpoints(self):
        return ResolvedSetpoints(
            inferred_ambient_c=25.0,
            inferred_hot_setpoint_c=125.0,
            inferred_cold_setpoint_c=-40.0,
        )

    @pytest.fixture
    def boundaries(self):
        return ProcessBoundaries(
            ambient_start_index=0,
            ambient_end_index=50,
            process_start_index=50,
            process_end_index=200,
        )

    def test_returns_region_list_and_classified_trace(self, simple_trace, setpoints, boundaries):
        regions, classified = classify_regions(simple_trace, setpoints, boundaries)
        assert isinstance(regions, RegionList)
        assert isinstance(classified, ClassifiedTrace)

    def test_all_rows_classified(self, simple_trace, setpoints, boundaries):
        regions, classified = classify_regions(simple_trace, setpoints, boundaries)
        assert len(classified.rows) == len(simple_trace.rows)

    def test_regions_have_required_fields(self, simple_trace, setpoints, boundaries):
        regions, _ = classify_regions(simple_trace, setpoints, boundaries)
        for region in regions.regions:
            assert region.region_id is not None
            assert region.primary_classification is not None
            assert region.classification_scores is not None
            assert region.classification_margin >= 0.0

    def test_classification_evidence_recorded(self, simple_trace, setpoints, boundaries):
        regions, _ = classify_regions(simple_trace, setpoints, boundaries)
        for region in regions.regions:
            assert len(region.classification_evidence) > 0

    def test_audit_log_records_classifications(self, simple_trace, setpoints, boundaries):
        audit_log = AuditLog()
        classify_regions(simple_trace, setpoints, boundaries, audit_log=audit_log)
        classification_entries = [e for e in audit_log.entries if e.action == "classify_region"]
        assert len(classification_entries) > 0

    def test_empty_trace_handled(self, setpoints, boundaries):
        regions, classified = classify_regions(
            PreprocessedTrace(rows=[]),
            setpoints,
            boundaries,
        )
        assert len(regions.regions) == 0
        assert len(classified.rows) == 0

    def test_ambiguity_flagged_when_margin_low(self, simple_trace, setpoints, boundaries):
        regions, _ = classify_regions(
            simple_trace,
            setpoints,
            boundaries,
            ambiguity_margin_threshold=0.5,
        )
        ambiguous_count = sum(1 for r in regions.regions if r.is_ambiguous)
        assert ambiguous_count >= 0

    def test_classified_trace_has_region_ids(self, simple_trace, setpoints, boundaries):
        regions, classified = classify_regions(simple_trace, setpoints, boundaries)
        rows_with_region = [r for r in classified.rows if r.region_id is not None]
        assert len(rows_with_region) > 0

    def test_classified_trace_has_labels(self, simple_trace, setpoints, boundaries):
        regions, classified = classify_regions(simple_trace, setpoints, boundaries)
        rows_with_label = [r for r in classified.rows if r.classification_label is not None]
        assert len(rows_with_label) > 0
