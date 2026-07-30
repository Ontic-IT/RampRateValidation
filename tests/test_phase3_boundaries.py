"""Unit tests for Phase 3: workers/boundary_detection/process_boundaries.py — boundary detection."""

import pytest
import numpy as np
from datetime import datetime

from workers.boundary_detection.process_boundaries import (
    detect_process_boundaries,
    _cusum_process_start,
    _detect_ambient_end,
)
from prereq.preprocessing.signal_processor import preprocess_trace
from models.domain import (
    AuditLog,
    CanonicalTrace,
    CanonicalTraceRow,
    PreprocessedTrace,
    PreprocessingReport,
    ProcessBoundaries,
)


def _make_trace_row(elapsed: float, temp: float, source_row: int = 0) -> CanonicalTraceRow:
    """Helper to create a canonical trace row."""
    return CanonicalTraceRow(
        timestamp=datetime(2025, 1, 1, 0, 0, int(elapsed % 60), int((elapsed * 1000000) % 1000000)),
        elapsed_seconds=elapsed,
        elapsed_minutes=elapsed / 60.0,
        temperature_c_raw=temp,
        channel="CH1",
        source_row=source_row,
        source_file="test.csv",
        sample_interval_seconds=1.0,
    )


class TestCusumProcessStart:
    def test_detects_ramp_start(self):
        slopes = np.concatenate([
            np.zeros(50),
            np.full(100, 5.0),
            np.zeros(50),
        ])
        
        start = _cusum_process_start(
            slopes,
            ambient_end_index=50,
            slope_noise_floor=0.3,
            sample_interval_s=1.0,
            minimum_region_duration_s=5.0,
        )
        
        assert 45 <= start <= 60

    def test_returns_end_if_no_ramp(self):
        slopes = np.zeros(100)
        
        start = _cusum_process_start(
            slopes,
            ambient_end_index=50,
            slope_noise_floor=0.3,
            sample_interval_s=1.0,
            minimum_region_duration_s=5.0,
        )
        
        assert start == 99

    def test_requires_consecutive_rows(self):
        slopes = np.zeros(100)
        slopes[50] = 10.0
        slopes[60] = 10.0
        
        start = _cusum_process_start(
            slopes,
            ambient_end_index=30,
            slope_noise_floor=0.3,
            sample_interval_s=1.0,
            minimum_region_duration_s=5.0,
        )
        
        assert start >= 50


class TestDetectAmbientEnd:
    def test_detects_ambient_end_before_ramp(self):
        slopes = np.concatenate([
            np.random.normal(0, 0.05, 60),
            np.full(100, 5.0),
        ])
        temps = np.concatenate([
            np.full(60, 25.0),
            np.linspace(25, 125, 100),
        ])
        
        end = _detect_ambient_end(slopes, temps, sample_interval=1.0, min_duration=10.0)
        
        assert 40 <= end <= 70


class TestDetectProcessBoundaries:
    @pytest.fixture
    def heating_ramp_trace(self):
        rows = []
        for i in range(60):
            rows.append(_make_trace_row(float(i), 25.0, i))
        for i in range(1200):
            temp = 25.0 + (5.0 / 60.0) * i
            temp = min(temp, 125.0)
            rows.append(_make_trace_row(float(60 + i), temp, 60 + i))
        for i in range(300):
            rows.append(_make_trace_row(float(1260 + i), 125.0, 1260 + i))
        
        return CanonicalTrace(rows=rows)

    @pytest.fixture
    def preprocessed_ramp(self, heating_ramp_trace):
        trace, report = preprocess_trace(heating_ramp_trace)
        return trace, report

    def test_returns_process_boundaries(self, preprocessed_ramp):
        trace, report = preprocessed_ramp
        boundaries = detect_process_boundaries(trace, report)
        
        assert isinstance(boundaries, ProcessBoundaries)

    def test_ambient_end_before_process_start(self, preprocessed_ramp):
        trace, report = preprocessed_ramp
        boundaries = detect_process_boundaries(trace, report)
        
        assert boundaries.ambient_end_index <= boundaries.process_start_index

    def test_process_start_before_process_end(self, preprocessed_ramp):
        trace, report = preprocessed_ramp
        boundaries = detect_process_boundaries(trace, report)
        
        assert boundaries.process_start_index < boundaries.process_end_index

    def test_usable_window_valid(self, preprocessed_ramp):
        trace, report = preprocessed_ramp
        boundaries = detect_process_boundaries(trace, report)
        
        start, end = boundaries.usable_window_row_range
        assert start >= 0
        assert end <= len(trace.rows)
        assert start < end

    def test_detection_method_recorded(self, preprocessed_ramp):
        trace, report = preprocessed_ramp
        boundaries = detect_process_boundaries(trace, report)
        
        assert boundaries.detection_method == "CUSUM_adaptive"

    def test_audit_log_records_detection(self, preprocessed_ramp):
        trace, report = preprocessed_ramp
        audit_log = AuditLog()
        detect_process_boundaries(trace, report, audit_log=audit_log)
        
        entries = [e for e in audit_log.entries if e.action == "detect_process_boundaries"]
        assert len(entries) == 1
        assert entries[0].decision == "SUCCESS"

    def test_empty_trace_handled(self):
        trace = PreprocessedTrace(rows=[])
        report = PreprocessingReport()
        
        boundaries = detect_process_boundaries(trace, report)
        
        assert boundaries.detection_method == "empty_trace"

    def test_process_start_near_expected(self, preprocessed_ramp):
        trace, report = preprocessed_ramp
        boundaries = detect_process_boundaries(trace, report)
        
        assert 40 <= boundaries.process_start_index <= 80
