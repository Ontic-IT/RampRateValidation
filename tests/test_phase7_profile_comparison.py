"""Unit tests for Phase 7: Profile comparison (M12)."""

import pytest
import numpy as np

from workers.profile_comparison.profile_comparator import (
    compare_cycle_metric_distribution,
    detect_cycle_to_cycle_drift,
    compute_profile_consistency_score,
)
from models.domain import (
    AuditLog,
    CycleMetrics,
    DwellMetrics,
    MetricSet,
    RampMetrics,
    Region,
    RegionList,
)
from config.constants import RegionType
from datetime import datetime


def _make_ramp_metrics(region_id: str, slope: float, jitter: float = 0.1, taper: float = 0.05) -> RampMetrics:
    return RampMetrics(
        region_id=region_id,
        robust_slope_c_per_min=slope,
        minimum_sustained_slope_c_per_min=slope * 0.9,
        median_rolling_slope_c_per_min=slope,
        endpoint_slope_c_per_min=slope,
        slope_MAD=0.1,
        jitter_score=jitter,
        taper_score=taper,
        linearity_score=0.95,
        stall_duration_seconds=10.0,
        reversal_count=0,
        monotonicity_score=0.98,
    )


def _make_dwell_metrics(region_id: str, mean_temp: float, overshoot: float = 0.0) -> DwellMetrics:
    return DwellMetrics(
        region_id=region_id,
        target_setpoint_c=mean_temp,
        mean_temperature_c=mean_temp,
        temperature_std_c=0.5,
        temperature_range_c=2.0,
        setpoint_deviation_c=0.2,
        time_inside_tolerance_band_seconds=300.0,
        time_inside_tolerance_band_pct=95.0,
        overshoot_magnitude_c=overshoot,
        overshoot_duration_seconds=10.0 if overshoot > 0 else None,
        settling_time_seconds=30.0,
        oscillation_count=0,
        stability_score=0.95,
    )


def _make_cycle_metrics(cycle_id: str, duration: float, heating_slope: float, cooling_slope: float) -> CycleMetrics:
    return CycleMetrics(
        cycle_id=cycle_id,
        duration_seconds=duration,
        average_heating_slope_c_per_min=heating_slope,
        average_cooling_slope_c_per_min=cooling_slope,
        minimum_heating_slope_c_per_min=heating_slope * 0.9,
        maximum_cooling_slope_c_per_min=cooling_slope * 1.1,
        average_jitter_score=0.1,
        average_taper_score=0.05,
        average_dwell_stability=0.95,
        maximum_overshoot_c=2.0,
        total_ramp_time_seconds=600.0,
        total_dwell_time_seconds=600.0,
        cycle_to_cycle_drift=0.0,
        heating_ramp_count=1,
        cooling_ramp_count=1,
    )


def _make_region(region_id: str, classification: RegionType, duration: float) -> Region:
    return Region(
        region_id=region_id,
        start_row=0,
        end_row=100,
        start_time=datetime(2025, 1, 1, 0, 0, 0),
        end_time=datetime(2025, 1, 1, 0, 5, 0),
        duration_seconds=duration,
        primary_classification=classification,
    )


class TestCompareCycleMetricDistribution:
    def test_all_10_metrics_present(self):
        metric_set = MetricSet(
            ramp_metrics=[_make_ramp_metrics("R001", 5.0)],
            dwell_metrics=[_make_dwell_metrics("D001", 85.0)],
            cycle_metrics=[_make_cycle_metrics("C001", 1200.0, 5.0, -3.0)],
        )
        
        result = compare_cycle_metric_distribution(metric_set)
        
        expected_keys = [
            "ramp_shape", "ramp_duration", "robust_rate", "dwell_duration",
            "dwell_median", "overshoot_magnitude", "settling_time",
            "cycle_duration", "jitter_score", "taper_score"
        ]
        assert set(result.keys()) == set(expected_keys)

    def test_each_metric_has_stats(self):
        metric_set = MetricSet(
            ramp_metrics=[_make_ramp_metrics("R001", 5.0)],
            dwell_metrics=[_make_dwell_metrics("D001", 85.0)],
            cycle_metrics=[_make_cycle_metrics("C001", 1200.0, 5.0, -3.0)],
        )
        
        result = compare_cycle_metric_distribution(metric_set)
        
        for metric_name, stats in result.items():
            assert "mean" in stats
            assert "std" in stats
            assert "min" in stats
            assert "max" in stats

    def test_dwell_duration_from_regions(self):
        regions = RegionList(regions=[
            _make_region("D001", RegionType.HOT_DWELL, 300.0),
            _make_region("D002", RegionType.COLD_DWELL, 320.0),
        ])
        metric_set = MetricSet(
            ramp_metrics=[],
            dwell_metrics=[
                _make_dwell_metrics("D001", 85.0),
                _make_dwell_metrics("D002", 25.0),
            ],
            cycle_metrics=[],
        )
        
        result = compare_cycle_metric_distribution(metric_set, regions)
        
        # Should use region durations (300, 320), not settling_time (30.0)
        assert result["dwell_duration"]["mean"] == pytest.approx(310.0)
        assert result["dwell_duration"]["min"] == pytest.approx(300.0)
        assert result["dwell_duration"]["max"] == pytest.approx(320.0)

    def test_dwell_duration_fallback_without_regions(self):
        metric_set = MetricSet(
            ramp_metrics=[],
            dwell_metrics=[_make_dwell_metrics("D001", 85.0)],
            cycle_metrics=[],
        )
        
        result = compare_cycle_metric_distribution(metric_set)
        
        # Should fall back to settling_time (30.0)
        assert result["dwell_duration"]["mean"] == pytest.approx(30.0)

    def test_empty_metrics(self):
        metric_set = MetricSet()
        
        result = compare_cycle_metric_distribution(metric_set)
        
        # Should return 0.0 for all stats when no data
        for metric_name, stats in result.items():
            assert stats["mean"] == 0.0


class TestDetectCycleToCycleDrift:
    def test_no_drift_consistent_cycles(self):
        cycles = [
            _make_cycle_metrics("C001", 1200.0, 5.0, -3.0),
            _make_cycle_metrics("C002", 1200.0, 5.0, -3.0),
        ]
        
        result = detect_cycle_to_cycle_drift(cycles)
        
        assert result["drift_detected"] is False
        assert result["drift_metrics"]["heating_slope"] == 0.0
        assert result["drift_metrics"]["cooling_slope"] == 0.0

    def test_drift_detected_heating_slope(self):
        cycles = [
            _make_cycle_metrics("C001", 1200.0, 5.0, -3.0),
            _make_cycle_metrics("C002", 1200.0, 4.0, -3.0),  # Heating slope dropped
        ]
        
        result = detect_cycle_to_cycle_drift(cycles)
        
        assert result["drift_detected"] is True
        assert result["drift_metrics"]["heating_slope"] == pytest.approx(1.0)

    def test_less_than_two_cycles_skipped(self):
        cycles = [_make_cycle_metrics("C001", 1200.0, 5.0, -3.0)]
        
        result = detect_cycle_to_cycle_drift(cycles)
        
        assert result["drift_detected"] is False
        assert result["max_drift_cycle_id"] is None


class TestComputeProfileConsistencyScore:
    def test_perfect_consistency(self):
        metric_set = MetricSet(
            cycle_metrics=[
                _make_cycle_metrics("C001", 1200.0, 5.0, -3.0),
                _make_cycle_metrics("C002", 1200.0, 5.0, -3.0),
                _make_cycle_metrics("C003", 1200.0, 5.0, -3.0),
            ]
        )
        
        score = compute_profile_consistency_score(metric_set)
        
        # Perfect consistency → score close to 1.0
        assert score > 0.95

    def test_variable_cycles_lower_score(self):
        metric_set = MetricSet(
            cycle_metrics=[
                _make_cycle_metrics("C001", 1200.0, 5.0, -3.0),
                _make_cycle_metrics("C002", 1000.0, 4.0, -2.5),
                _make_cycle_metrics("C003", 1400.0, 6.0, -3.5),
            ]
        )
        
        score = compute_profile_consistency_score(metric_set)
        
        # Variable cycles → lower score
        assert 0.0 <= score < 0.95

    def test_no_cycles_returns_zero(self):
        metric_set = MetricSet()
        
        score = compute_profile_consistency_score(metric_set)
        
        assert score == 0.0

    def test_single_cycle_returns_one(self):
        metric_set = MetricSet(
            cycle_metrics=[_make_cycle_metrics("C001", 1200.0, 5.0, -3.0)]
        )
        
        score = compute_profile_consistency_score(metric_set)
        
        # Single cycle → perfect consistency
        assert score == 1.0
