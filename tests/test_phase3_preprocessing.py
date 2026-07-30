"""Unit tests for Phase 3: prereq/preprocessing/signal_processor.py — signal processing."""

import pytest
import numpy as np
from datetime import datetime

from prereq.preprocessing.signal_processor import (
    preprocess_trace,
    _estimate_sample_interval,
    _hampel_filter,
    _rolling_median,
    _rolling_mad,
    _compute_local_slope,
    _compute_direction_of_travel,
)
from models.domain import (
    AuditLog,
    CanonicalTrace,
    CanonicalTraceRow,
    PreprocessedTrace,
    PreprocessingReport,
)


def _make_trace_row(elapsed: float, temp: float, source_row: int = 0) -> CanonicalTraceRow:
    """Helper to create a canonical trace row."""
    from datetime import timedelta
    base = datetime(2025, 1, 1, 0, 0, 0)
    ts = base + timedelta(seconds=elapsed)
    return CanonicalTraceRow(
        timestamp=ts,
        elapsed_seconds=elapsed,
        elapsed_minutes=elapsed / 60.0,
        temperature_c_raw=temp,
        channel="CH1",
        source_row=source_row,
        source_file="test.csv",
        sample_interval_seconds=1.0,
    )


class TestEstimateSampleInterval:
    def test_regular_interval(self):
        timestamps = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        assert _estimate_sample_interval(timestamps) == 1.0

    def test_irregular_interval_uses_median(self):
        timestamps = np.array([0.0, 1.0, 2.0, 5.0, 6.0])
        assert _estimate_sample_interval(timestamps) == 1.0

    def test_single_point_returns_default(self):
        timestamps = np.array([0.0])
        assert _estimate_sample_interval(timestamps) == 1.0

    def test_empty_returns_default(self):
        timestamps = np.array([])
        assert _estimate_sample_interval(timestamps) == 1.0


class TestHampelFilter:
    def test_no_spikes_detected_in_clean_signal(self):
        values = np.array([25.0, 25.1, 25.0, 24.9, 25.0, 25.1, 25.0])
        spike_flags, cleaned = _hampel_filter(values, k=2, n_sigma=3.0)
        assert not any(spike_flags)
        np.testing.assert_array_almost_equal(cleaned, values)

    def test_spike_detected_and_replaced(self):
        values = np.array([25.0, 25.0, 25.0, 50.0, 25.0, 25.0, 25.0])
        spike_flags, cleaned = _hampel_filter(values, k=2, n_sigma=3.0)
        assert spike_flags[3] is True
        assert cleaned[3] != 50.0
        assert abs(cleaned[3] - 25.0) < 1.0

    def test_multiple_spikes_detected(self):
        values = np.array([25.0, 25.0, 100.0, 25.0, 25.0, -50.0, 25.0, 25.0])
        spike_flags, cleaned = _hampel_filter(values, k=2, n_sigma=3.0)
        assert spike_flags[2] is True
        assert spike_flags[5] is True


class TestRollingStatistics:
    def test_rolling_median(self):
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = _rolling_median(values, window=3)
        assert result[2] == 3.0

    def test_rolling_mad(self):
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = _rolling_mad(values, window=3)
        assert result[2] == 1.0


class TestComputeLocalSlope:
    def test_constant_temperature_zero_slope(self):
        temps = np.array([25.0, 25.0, 25.0, 25.0, 25.0])
        timestamps = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        slopes = _compute_local_slope(temps, timestamps)
        assert all(abs(s) < 0.01 for s in slopes if not np.isnan(s))

    def test_linear_ramp_constant_slope(self):
        temps = np.array([25.0, 25.0833, 25.1667, 25.25, 25.3333])
        timestamps = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        slopes = _compute_local_slope(temps, timestamps)
        for s in slopes[1:-1]:
            assert abs(s - 5.0) < 0.1


class TestComputeDirectionOfTravel:
    def test_heating_direction(self):
        slopes = np.array([5.0, 5.0, 5.0])
        directions = _compute_direction_of_travel(slopes)
        assert all(d == "HEATING" for d in directions)

    def test_cooling_direction(self):
        slopes = np.array([-5.0, -5.0, -5.0])
        directions = _compute_direction_of_travel(slopes)
        assert all(d == "COOLING" for d in directions)

    def test_stable_direction(self):
        slopes = np.array([0.0, 0.05, -0.05])
        directions = _compute_direction_of_travel(slopes)
        assert all(d == "STABLE" for d in directions)


class TestPreprocessTrace:
    @pytest.fixture
    def simple_trace(self):
        rows = [_make_trace_row(float(i), 25.0 + i * 0.0833, i) for i in range(100)]
        return CanonicalTrace(rows=rows)

    def test_returns_preprocessed_trace_and_report(self, simple_trace):
        trace, report = preprocess_trace(simple_trace)
        assert isinstance(trace, PreprocessedTrace)
        assert isinstance(report, PreprocessingReport)

    def test_all_rows_preserved(self, simple_trace):
        trace, _ = preprocess_trace(simple_trace)
        assert len(trace.rows) == 100

    def test_analysis_signal_populated(self, simple_trace):
        trace, _ = preprocess_trace(simple_trace)
        for row in trace.rows:
            assert row.temperature_c_analysis_signal is not None

    def test_rolling_statistics_populated(self, simple_trace):
        trace, _ = preprocess_trace(simple_trace)
        for row in trace.rows[10:-10]:
            assert row.rolling_slope_c_per_min is not None
            assert row.rolling_temperature_median is not None
            assert row.rolling_temperature_MAD is not None

    def test_direction_of_travel_populated(self, simple_trace):
        trace, _ = preprocess_trace(simple_trace)
        for row in trace.rows[10:-10]:
            assert row.direction_of_travel in ("HEATING", "COOLING", "STABLE", None)

    def test_sample_interval_estimated(self, simple_trace):
        _, report = preprocess_trace(simple_trace)
        assert report.estimated_sample_interval_s == 1.0

    def test_noise_floor_estimated(self, simple_trace):
        _, report = preprocess_trace(simple_trace)
        assert report.noise_floor_c >= 0.0

    def test_audit_log_records_preprocessing(self, simple_trace):
        audit_log = AuditLog()
        preprocess_trace(simple_trace, audit_log=audit_log)
        entries = [e for e in audit_log.entries if e.action == "preprocess_trace"]
        assert len(entries) == 1
        assert entries[0].decision == "SUCCESS"

    def test_empty_trace_returns_empty(self):
        trace, report = preprocess_trace(CanonicalTrace(rows=[]))
        assert len(trace.rows) == 0

    def test_spike_detection_flags_spikes(self):
        rows = [_make_trace_row(float(i), 25.0, i) for i in range(20)]
        rows[10] = _make_trace_row(10.0, 100.0, 10)
        trace = CanonicalTrace(rows=rows)
        
        preprocessed, report = preprocess_trace(trace)
        assert len(report.detected_spikes) > 0
        assert "SPIKE" in preprocessed.rows[10].data_quality_flags
