"""Unit tests for Phase 1: config/classification_weights/ — versioned YAML weight files."""

from pathlib import Path

import yaml
import pytest


class TestWeightFileV100:
    """Verify v1.0.0 weight file loads and has correct structure."""

    @pytest.fixture
    def weight_data(self):
        weight_path = Path(__file__).parent.parent / "config" / "classification_weights" / "v1.0.0.yaml"
        assert weight_path.exists(), "v1.0.0.yaml weight file missing"
        with open(weight_path) as f:
            return yaml.safe_load(f)

    def test_version_present(self, weight_data):
        assert weight_data["version"] == "1.0.0"

    def test_aggregation_method_present(self, weight_data):
        assert "aggregation_method" in weight_data

    def test_ramp_weights_sum_to_one(self, weight_data):
        weights = weight_data["ramp_weights"]
        total = sum(weights.values())
        assert abs(total - 1.0) < 0.01, f"ramp_weights sum={total}"

    def test_ramp_weights_has_15_keys(self, weight_data):
        assert len(weight_data["ramp_weights"]) == 15

    def test_dwell_weights_sum_to_one(self, weight_data):
        weights = weight_data["dwell_weights"]
        total = sum(weights.values())
        assert abs(total - 1.0) < 0.01, f"dwell_weights sum={total}"

    def test_dwell_weights_has_8_keys(self, weight_data):
        assert len(weight_data["dwell_weights"]) == 8

    def test_overshoot_weights_sum_to_one(self, weight_data):
        weights = weight_data["overshoot_weights"]
        total = sum(weights.values())
        assert abs(total - 1.0) < 0.01, f"overshoot_weights sum={total}"

    def test_overshoot_weights_has_7_keys(self, weight_data):
        assert len(weight_data["overshoot_weights"]) == 7

    def test_correction_weights_sum_to_one(self, weight_data):
        weights = weight_data["correction_weights"]
        total = sum(weights.values())
        assert abs(total - 1.0) < 0.01, f"correction_weights sum={total}"

    def test_correction_weights_has_6_keys(self, weight_data):
        assert len(weight_data["correction_weights"]) == 6

    def test_transient_weights_sum_to_one(self, weight_data):
        weights = weight_data["transient_weights"]
        total = sum(weights.values())
        assert abs(total - 1.0) < 0.01, f"transient_weights sum={total}"

    def test_transient_weights_has_5_keys(self, weight_data):
        assert len(weight_data["transient_weights"]) == 5

    def test_unknown_weights_sum_to_one(self, weight_data):
        weights = weight_data["unknown_weights"]
        total = sum(weights.values())
        assert abs(total - 1.0) < 0.01, f"unknown_weights sum={total}"

    def test_unknown_weights_has_4_keys(self, weight_data):
        assert len(weight_data["unknown_weights"]) == 4

    def test_all_weight_values_positive(self, weight_data):
        for group_name in [
            "ramp_weights", "dwell_weights", "overshoot_weights",
            "correction_weights", "transient_weights", "unknown_weights",
        ]:
            for key, value in weight_data[group_name].items():
                assert value > 0, f"{group_name}.{key} is not positive: {value}"

    def test_created_date_present(self, weight_data):
        assert "created_date" in weight_data

    def test_calibration_dataset_version_present(self, weight_data):
        assert "calibration_dataset_version" in weight_data
