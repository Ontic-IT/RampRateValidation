"""Unit tests for Phase 5: cleaners/metrics/ — metric computation."""

import pytest
import numpy as np
from datetime import datetime, timedelta

from config.constants import RegionType, RampDirection, CycleStatus
from cleaners.metrics.ramp_metrics import (
    compute_ramp_metrics,
    _compute_theil_sen_slope,
    _compute_minimum_sustained_slope,
    _compute_endpoint_slope,
    _compute_jitter_score,
    _compute_taper_score,
    _compute_linearity_score,
)
from cleaners.metrics.dwell_metrics import (
    compute_dwell_metrics,
    _compute_time_in_tolerance_band,
    _compute_overshoot_metrics,
    _compute_oscillation_count,
    _compute_stability_score,
)
from cleaners.metrics.cycle_metrics import compute_cycle_metrics
from models.domain import (
    AuditLog,
    CanonicalTraceRow,
    ClassificationEvidence,
    ClassifiedTrace,
    Cycle,
    CycleMetrics,
    DwellMetrics,
    RampMetrics,
    Region,
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


class TestTheilSenSlope:
    def test_linear_ramp_correct_slope(self):
        temps = np.array([25.0, 25.0833, 25.1667, 25.25, 25.3333])
        elapsed = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        
        slope = _compute_theil_sen_slope(temps, elapsed)
        
        assert abs(slope - 5.0) < 0.5

    def test_cooling_ramp_negative_slope(self):
        temps = np.array([125.0, 124.9167, 124.8333, 124.75, 124.6667])
        elapsed = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        
        slope = _compute_theil_sen_slope(temps, elapsed)
        
        assert slope < 0

    def test_insufficient_data_returns_zero(self):
        temps = np.array([25.0])
        elapsed = np.array([0.0])
        
        slope = _compute_theil_sen_slope(temps, elapsed)
        
        assert slope == 0.0


class TestMinimumSustainedSlope:
    def test_uniform_ramp_returns_consistent_slope(self):
        temps = np.linspace(25, 125, 100)
        elapsed = np.linspace(0, 1200, 100)
        
        min_slope = _compute_minimum_sustained_slope(temps, elapsed, 60.0, 12.0)
        
        assert min_slope > 0

    def test_tapered_ramp_returns_lower_slope(self):
        temps = np.concatenate([
            np.linspace(25, 75, 50),
            np.linspace(75, 85, 50),
        ])
        elapsed = np.linspace(0, 1200, 100)
        
        min_slope = _compute_minimum_sustained_slope(temps, elapsed, 60.0, 12.0)
        
        assert min_slope < 5.0


class TestEndpointSlope:
    def test_endpoint_slope_from_final_rows(self):
        temps = np.linspace(25, 125, 100)
        elapsed = np.linspace(0, 1200, 100)
        
        endpoint = _compute_endpoint_slope(temps, elapsed)
        
        assert endpoint > 0


class TestJitterScore:
    def test_low_mad_low_jitter(self):
        jitter = _compute_jitter_score(0.1, 5.0)
        assert jitter < 0.1

    def test_high_mad_high_jitter(self):
        jitter = _compute_jitter_score(5.0, 5.0)
        assert jitter == 1.0

    def test_zero_slope_returns_one(self):
        jitter = _compute_jitter_score(0.1, 0.0)
        assert jitter == 1.0


class TestTaperScore:
    def test_uniform_slope_low_taper(self):
        slopes = np.array([5.0] * 20)
        taper = _compute_taper_score(slopes)
        assert abs(taper) < 0.1

    def test_decreasing_slope_positive_taper(self):
        slopes = np.concatenate([
            np.full(10, 5.0),
            np.full(10, 2.0),
        ])
        taper = _compute_taper_score(slopes)
        assert taper > 0.3


class TestLinearityScore:
    def test_perfect_linear_high_score(self):
        temps = np.linspace(25, 125, 100)
        elapsed = np.linspace(0, 1200, 100)
        
        linearity = _compute_linearity_score(temps, elapsed, 5.0)
        
        assert linearity > 0.95

    def test_noisy_data_lower_score(self):
        np.random.seed(42)
        temps = np.linspace(25, 125, 100) + np.random.normal(0, 10, 100)
        elapsed = np.linspace(0, 1200, 100)
        
        linearity = _compute_linearity_score(temps, elapsed, 5.0)
        
        assert linearity < 0.99


class TestComputeRampMetrics:
    @pytest.fixture
    def valid_ramp(self):
        return ValidRampRegion(
            region_id="R0001",
            direction=RampDirection.HEATING,
            start_row=0,
            end_row=99,
            duration_seconds=1200.0,
            included_rows=list(range(100)),
            monotonicity_score=0.95,
            reversal_count=0,
            stall_duration_seconds=0.0,
        )

    @pytest.fixture
    def heating_trace(self):
        rows = []
        for i in range(100):
            temp = 25.0 + (100.0 / 100) * i
            slope = 5.0
            rows.append(_make_row(float(i * 12), temp, slope))
        return ClassifiedTrace(rows=rows)

    def test_returns_ramp_metrics(self, valid_ramp, heating_trace):
        metrics = compute_ramp_metrics(valid_ramp, heating_trace)
        assert isinstance(metrics, RampMetrics)

    def test_theil_sen_slope_computed(self, valid_ramp, heating_trace):
        metrics = compute_ramp_metrics(valid_ramp, heating_trace)
        assert metrics.robust_slope_c_per_min > 0
        assert metrics.slope_calculation_method == "theil_sen"

    def test_all_metrics_populated(self, valid_ramp, heating_trace):
        metrics = compute_ramp_metrics(valid_ramp, heating_trace)
        assert metrics.minimum_sustained_slope_c_per_min is not None
        assert metrics.median_rolling_slope_c_per_min is not None
        assert metrics.jitter_score is not None
        assert metrics.taper_score is not None
        assert metrics.linearity_score is not None

    def test_audit_log_records_computation(self, valid_ramp, heating_trace):
        audit_log = AuditLog()
        compute_ramp_metrics(valid_ramp, heating_trace, audit_log=audit_log)
        entries = [e for e in audit_log.entries if "ramp_metrics" in e.action]
        assert len(entries) == 1


class TestDwellMetrics:
    def test_time_in_tolerance_band(self):
        temps = np.array([125.0, 125.5, 124.5, 125.0, 125.2])
        elapsed = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        
        time_in = _compute_time_in_tolerance_band(temps, elapsed, 125.0, 3.0)
        
        assert time_in > 0

    def test_oscillation_count(self):
        temps = np.array([125.0, 126.0, 124.0, 126.0, 124.0, 125.0])
        
        count = _compute_oscillation_count(temps, 125.0, 3.0)
        
        assert count >= 1

    def test_stability_score(self):
        score = _compute_stability_score(0.5, 2.0, 3.0)
        assert 0 <= score <= 1

    def test_compute_dwell_metrics_returns_metrics(self):
        base = datetime(2025, 1, 1, 0, 0, 0)
        region = Region(
            region_id="R0001",
            start_row=0,
            end_row=99,
            start_time=base,
            end_time=base + timedelta(seconds=300),
            duration_seconds=300.0,
            primary_classification=RegionType.HOT_DWELL,
            classification_scores={RegionType.HOT_DWELL: 0.9},
            classification_margin=0.5,
            classification_evidence=[],
            classification_confidence=0.9,
        )
        rows = [_make_row(float(i), 125.0 + np.random.normal(0, 0.5)) for i in range(100)]
        trace = ClassifiedTrace(rows=rows)
        setpoints = ResolvedSetpoints(inferred_hot_setpoint_c=125.0)
        
        metrics = compute_dwell_metrics(region, trace, setpoints)
        
        assert isinstance(metrics, DwellMetrics)
        assert metrics.target_setpoint_c == 125.0


class TestCycleMetrics:
    def test_compute_cycle_metrics(self):
        base = datetime(2025, 1, 1, 0, 0, 0)
        cycle = Cycle(
            cycle_id="C0001",
            cycle_number=1,
            start_row=0,
            end_row=1000,
            start_time=base,
            end_time=base + timedelta(seconds=3600),
            duration_seconds=3600.0,
            region_ids=["R0001", "R0002"],
            regions=[],
            valid_ramps=[],
            heating_ramp_count=1,
            cooling_ramp_count=1,
            status=CycleStatus.COMPLETE,
        )
        ramp_metrics = [
            RampMetrics(region_id="R0001", robust_slope_c_per_min=5.0, jitter_score=0.1, taper_score=0.05),
            RampMetrics(region_id="R0002", robust_slope_c_per_min=-3.0, jitter_score=0.15, taper_score=0.08),
        ]
        dwell_metrics = [
            DwellMetrics(region_id="R0003", stability_score=0.9, overshoot_magnitude_c=2.0),
        ]
        
        metrics = compute_cycle_metrics(cycle, ramp_metrics, dwell_metrics)
        
        assert isinstance(metrics, CycleMetrics)
        assert metrics.average_heating_slope_c_per_min == 5.0
        assert metrics.average_cooling_slope_c_per_min == -3.0
        assert metrics.average_jitter_score > 0
