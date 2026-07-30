"""Unit tests for Phase 4: workers/setpoint_resolution/ — setpoint inference."""

import pytest
import numpy as np
from datetime import datetime, timedelta

from config.constants import SetpointResolutionMode
from workers.setpoint_resolution.setpoint_inference import (
    resolve_setpoints,
    _cluster_temperatures,
    _resolve_mode_a,
    _resolve_mode_b,
)
from models.domain import (
    AuditLog,
    CanonicalTraceRow,
    PreprocessedTrace,
    PreprocessingReport,
    ProcessBoundaries,
    ResolvedSetpoints,
)


def _make_row(elapsed: float, temp: float, setpoint: float | None = None, mad: float = 0.1, slope: float = 0.0) -> CanonicalTraceRow:
    """Helper to create a preprocessed trace row."""
    base = datetime(2025, 1, 1, 0, 0, 0)
    ts = base + timedelta(seconds=elapsed)
    return CanonicalTraceRow(
        timestamp=ts,
        elapsed_seconds=elapsed,
        elapsed_minutes=elapsed / 60.0,
        temperature_c_raw=temp,
        temperature_c_analysis_signal=temp,
        setpoint_c=setpoint,
        channel="CH1",
        source_row=int(elapsed),
        source_file="test.csv",
        sample_interval_seconds=1.0,
        rolling_temperature_MAD=mad,
        rolling_slope_c_per_min=slope,
    )


class TestClusterTemperatures:
    def test_single_cluster(self):
        temps = np.array([25.0] * 100)
        labels, centers = _cluster_temperatures(temps)
        assert len(centers) >= 1
        assert abs(centers[0] - 25.0) < 1.0

    def test_two_clusters(self):
        temps = np.concatenate([
            np.full(50, 25.0),
            np.full(50, 125.0),
        ])
        labels, centers = _cluster_temperatures(temps)
        assert len(centers) >= 1
        assert np.min(temps) <= min(centers) + 50
        assert np.max(temps) >= max(centers) - 50

    def test_three_clusters(self):
        temps = np.concatenate([
            np.full(30, 25.0),
            np.full(30, 125.0),
            np.full(30, -40.0),
        ])
        labels, centers = _cluster_temperatures(temps)
        assert len(centers) >= 1


class TestResolveSetpointsModeA:
    @pytest.fixture
    def trace_with_setpoints(self):
        rows = []
        for i in range(100):
            if i < 30:
                sp = 125.0
            elif i < 60:
                sp = -40.0
            else:
                sp = 125.0
            rows.append(_make_row(float(i), 25.0 + i * 0.5, setpoint=sp))
        return PreprocessedTrace(rows=rows)

    def test_mode_a_extracts_setpoints(self, trace_with_setpoints):
        result = resolve_setpoints(
            trace_with_setpoints,
            PreprocessingReport(),
            ProcessBoundaries(),
            setpoint_channel_present=True,
        )
        assert result.resolution_mode == SetpointResolutionMode.MODE_A
        assert result.inferred_hot_setpoint_c == 125.0
        assert result.inferred_cold_setpoint_c == -40.0

    def test_mode_a_confidence_high(self, trace_with_setpoints):
        result = resolve_setpoints(
            trace_with_setpoints,
            PreprocessingReport(),
            ProcessBoundaries(),
            setpoint_channel_present=True,
        )
        assert result.setpoint_confidence_scores["hot"] == 1.0
        assert result.setpoint_confidence_scores["cold"] == 1.0


class TestResolveSetpointsModeB:
    @pytest.fixture
    def heating_ramp_trace(self):
        rows = []
        for i in range(60):
            rows.append(_make_row(float(i), 25.0, mad=0.05, slope=0.0))
        for i in range(1200):
            temp = 25.0 + (5.0 / 60.0) * i
            temp = min(temp, 125.0)
            rows.append(_make_row(float(60 + i), temp, mad=0.1, slope=5.0))
        for i in range(300):
            rows.append(_make_row(float(1260 + i), 125.0, mad=0.05, slope=0.0))
        return PreprocessedTrace(rows=rows)

    @pytest.fixture
    def preprocessing_report(self):
        return PreprocessingReport(
            estimated_sample_interval_s=1.0,
            noise_floor_c=0.1,
            slope_noise_floor_c_per_min=0.3,
        )

    @pytest.fixture
    def boundaries(self):
        return ProcessBoundaries(
            ambient_start_index=0,
            ambient_end_index=60,
            process_start_index=60,
            process_end_index=1560,
        )

    def test_mode_b_infers_setpoints(self, heating_ramp_trace, preprocessing_report, boundaries):
        result = resolve_setpoints(
            heating_ramp_trace,
            preprocessing_report,
            boundaries,
            setpoint_channel_present=False,
        )
        assert result.resolution_mode == SetpointResolutionMode.MODE_B
        assert result.inferred_hot_setpoint_c > 100.0
        assert result.inferred_ambient_c < 50.0

    def test_mode_b_records_algorithm_seed(self, heating_ramp_trace, preprocessing_report, boundaries):
        result = resolve_setpoints(
            heating_ramp_trace,
            preprocessing_report,
            boundaries,
            setpoint_channel_present=False,
        )
        assert result.algorithm_seed_used == 42

    def test_mode_b_audit_log(self, heating_ramp_trace, preprocessing_report, boundaries):
        audit_log = AuditLog()
        resolve_setpoints(
            heating_ramp_trace,
            preprocessing_report,
            boundaries,
            setpoint_channel_present=False,
            audit_log=audit_log,
        )
        entries = [e for e in audit_log.entries if "mode_b" in e.action]
        assert len(entries) >= 1

    def test_empty_trace_handled(self):
        result = resolve_setpoints(
            PreprocessedTrace(rows=[]),
            PreprocessingReport(),
            ProcessBoundaries(),
            setpoint_channel_present=False,
        )
        assert result.resolution_mode == SetpointResolutionMode.MODE_B
