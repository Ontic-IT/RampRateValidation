"""Unit tests for Phase 1: config/adaptive_constants.py — adaptive threshold schema and clamping."""

import pytest

from config.adaptive_constants import (
    ADAPTIVE_CONSTANT_BOUNDS,
    AdaptiveConstantBounds,
    AdaptiveConstants,
    AdaptiveThreshold,
    clamp_threshold,
)


class TestAdaptiveConstantBounds:
    """Verify all 9 adaptive constants have canonical bounds defined."""

    def test_all_nine_constants_have_bounds(self):
        expected_names = {
            "noise_floor_c",
            "slope_noise_floor_c_per_min",
            "stable_slope_threshold",
            "stable_variance_threshold",
            "ramp_slope_threshold",
            "overshoot_detection_threshold",
            "correction_oscillation_threshold",
            "dwell_cluster_separation_threshold",
            "minimum_region_duration_seconds",
        }
        assert set(ADAPTIVE_CONSTANT_BOUNDS.keys()) == expected_names

    def test_noise_floor_c_bounds(self):
        b = ADAPTIVE_CONSTANT_BOUNDS["noise_floor_c"]
        assert b.minimum == 0.01
        assert b.maximum == 2.0

    def test_slope_noise_floor_bounds(self):
        b = ADAPTIVE_CONSTANT_BOUNDS["slope_noise_floor_c_per_min"]
        assert b.minimum == 0.05
        assert b.maximum == 5.0

    def test_stable_slope_threshold_bounds(self):
        b = ADAPTIVE_CONSTANT_BOUNDS["stable_slope_threshold"]
        assert b.minimum == 0.1
        assert b.maximum == 8.0

    def test_stable_variance_threshold_bounds(self):
        b = ADAPTIVE_CONSTANT_BOUNDS["stable_variance_threshold"]
        assert b.minimum == 0.02
        assert b.maximum == 3.0

    def test_ramp_slope_threshold_bounds(self):
        b = ADAPTIVE_CONSTANT_BOUNDS["ramp_slope_threshold"]
        assert b.minimum == 0.3
        assert b.maximum == 20.0

    def test_overshoot_detection_threshold_bounds(self):
        b = ADAPTIVE_CONSTANT_BOUNDS["overshoot_detection_threshold"]
        assert b.minimum == 0.1
        assert b.maximum == 10.0

    def test_correction_oscillation_threshold_bounds(self):
        b = ADAPTIVE_CONSTANT_BOUNDS["correction_oscillation_threshold"]
        assert b.minimum == 0.05
        assert b.maximum == 5.0

    def test_dwell_cluster_separation_threshold_bounds(self):
        b = ADAPTIVE_CONSTANT_BOUNDS["dwell_cluster_separation_threshold"]
        assert b.minimum == 0.5
        assert b.maximum == 15.0

    def test_minimum_region_duration_seconds_bounds(self):
        b = ADAPTIVE_CONSTANT_BOUNDS["minimum_region_duration_seconds"]
        assert b.minimum == 5.0
        assert b.maximum == 120.0

    def test_all_bounds_have_minimum_less_than_maximum(self):
        for name, b in ADAPTIVE_CONSTANT_BOUNDS.items():
            assert b.minimum < b.maximum, f"{name}: min={b.minimum} >= max={b.maximum}"


class TestClampThreshold:
    """Verify clamping logic with AuditEntry tracking."""

    def test_value_within_bounds_not_clamped(self):
        t = clamp_threshold("noise_floor_c", 0.5, "MAD of ambient window")
        assert t.value == 0.5
        assert t.was_clamped is False
        assert t.derived_value is None

    def test_value_below_minimum_clamped_up(self):
        t = clamp_threshold("noise_floor_c", 0.001, "MAD of ambient window")
        assert t.value == 0.01
        assert t.was_clamped is True
        assert t.derived_value == 0.001

    def test_value_above_maximum_clamped_down(self):
        t = clamp_threshold("noise_floor_c", 10.0, "MAD of ambient window")
        assert t.value == 2.0
        assert t.was_clamped is True
        assert t.derived_value == 10.0

    def test_value_at_minimum_bound_not_clamped(self):
        t = clamp_threshold("noise_floor_c", 0.01, "MAD of ambient window")
        assert t.value == 0.01
        assert t.was_clamped is False

    def test_value_at_maximum_bound_not_clamped(self):
        t = clamp_threshold("noise_floor_c", 2.0, "MAD of ambient window")
        assert t.value == 2.0
        assert t.was_clamped is False

    def test_derivation_method_preserved(self):
        t = clamp_threshold("stable_slope_threshold", 1.5, "3x slope noise floor")
        assert t.derivation_method == "3x slope noise floor"

    def test_bounds_preserved_in_threshold(self):
        t = clamp_threshold("ramp_slope_threshold", 5.0, "15% of mean achievable rate")
        assert t.minimum_bound == 0.3
        assert t.maximum_bound == 20.0

    def test_unknown_constant_name_raises(self):
        with pytest.raises(KeyError):
            clamp_threshold("nonexistent_constant", 1.0, "test")


class TestAdaptiveConstants:
    """Verify AdaptiveConstants model instantiation and snapshot."""

    @pytest.fixture
    def sample_constants(self):
        """Create a valid AdaptiveConstants instance."""
        return AdaptiveConstants(
            noise_floor_c=clamp_threshold("noise_floor_c", 0.05, "MAD of ambient"),
            slope_noise_floor_c_per_min=clamp_threshold("slope_noise_floor_c_per_min", 0.3, "scaled noise floor"),
            stable_slope_threshold=clamp_threshold("stable_slope_threshold", 0.9, "3x slope noise"),
            stable_variance_threshold=clamp_threshold("stable_variance_threshold", 0.1, "2x ambient MAD"),
            ramp_slope_threshold=clamp_threshold("ramp_slope_threshold", 1.5, "15% mean rate"),
            overshoot_detection_threshold=clamp_threshold("overshoot_detection_threshold", 0.8, "4x dwell MAD"),
            correction_oscillation_threshold=clamp_threshold("correction_oscillation_threshold", 0.4, "2x dwell MAD"),
            dwell_cluster_separation_threshold=clamp_threshold("dwell_cluster_separation_threshold", 5.0, "10% span"),
            minimum_region_duration_seconds=clamp_threshold("minimum_region_duration_seconds", 10.0, "10x interval"),
        )

    def test_instantiation_succeeds(self, sample_constants):
        assert sample_constants.noise_floor_c.value == 0.05

    def test_to_snapshot_returns_flat_dict(self, sample_constants):
        snapshot = sample_constants.to_snapshot()
        assert isinstance(snapshot, dict)
        assert len(snapshot) == 9
        assert snapshot["noise_floor_c"] == 0.05
        assert snapshot["slope_noise_floor_c_per_min"] == 0.3

    def test_all_fields_present_in_snapshot(self, sample_constants):
        snapshot = sample_constants.to_snapshot()
        expected_keys = set(ADAPTIVE_CONSTANT_BOUNDS.keys())
        assert set(snapshot.keys()) == expected_keys
