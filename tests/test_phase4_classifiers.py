"""Unit tests for Phase 4: workers/region_classification/classifiers/ — evidence-based classifiers."""

import pytest
import numpy as np
from datetime import datetime

from config.constants import RegionType
from workers.region_classification.classifiers import (
    RampClassifier,
    DwellClassifier,
    OvershootClassifier,
    CorrectionClassifier,
    TransientClassifier,
    UnknownClassifier,
)
from models.domain import ClassificationEvidence


class TestRampClassifier:
    @pytest.fixture
    def classifier(self):
        return RampClassifier()

    @pytest.fixture
    def heating_ramp_data(self):
        return {
            "temperatures": list(np.linspace(25, 125, 100)),
            "rolling_slopes": [5.0] * 100,
            "duration_seconds": 1200.0,
        }

    @pytest.fixture
    def cooling_ramp_data(self):
        return {
            "temperatures": list(np.linspace(125, -40, 100)),
            "rolling_slopes": [-5.0] * 100,
            "duration_seconds": 1980.0,
        }

    @pytest.fixture
    def context(self):
        return {
            "hot_setpoint": 125.0,
            "cold_setpoint": -40.0,
            "slope_threshold": 0.5,
            "min_ramp_duration": 30.0,
        }

    def test_heating_ramp_high_score(self, classifier, heating_ramp_data, context):
        evidence = classifier.compute_evidence(heating_ramp_data, context)
        assert evidence["sustained_positive_slope"] > 0.8
        assert evidence["monotonicity_score"] > 0.8

    def test_cooling_ramp_high_score(self, classifier, cooling_ramp_data, context):
        evidence = classifier.compute_evidence(cooling_ramp_data, context)
        assert evidence["sustained_negative_slope"] > 0.8

    def test_classify_returns_evidence_objects(self, classifier, heating_ramp_data, context):
        results = classifier.classify(heating_ramp_data, context)
        assert len(results) == 2
        assert all(isinstance(r, ClassificationEvidence) for r in results)
        assert any(r.evidence_type == RegionType.HEATING_RAMP for r in results)
        assert any(r.evidence_type == RegionType.COOLING_RAMP for r in results)

    def test_weights_sum_to_one(self, classifier):
        weights = classifier.default_weights()
        total = sum(weights.values())
        assert abs(total - 1.0) < 0.01


class TestDwellClassifier:
    @pytest.fixture
    def classifier(self):
        return DwellClassifier()

    @pytest.fixture
    def hot_dwell_data(self):
        return {
            "temperatures": [125.0 + np.random.normal(0, 0.5) for _ in range(300)],
            "rolling_mad": [0.5] * 300,
            "rolling_slopes": [0.0] * 300,
            "duration_seconds": 300.0,
        }

    @pytest.fixture
    def context(self):
        return {
            "hot_setpoint": 125.0,
            "cold_setpoint": -40.0,
            "ambient_temp": 25.0,
            "min_dwell_duration": 300.0,
            "tolerance_band": 3.0,
            "noise_floor": 0.1,
        }

    def test_hot_dwell_high_stability(self, classifier, hot_dwell_data, context):
        evidence = classifier.compute_evidence(hot_dwell_data, context)
        assert evidence["temperature_stability"] > 0.7
        assert evidence["setpoint_proximity"] > 0.7

    def test_classify_returns_three_types(self, classifier, hot_dwell_data, context):
        results = classifier.classify(hot_dwell_data, context)
        assert len(results) == 3
        types = {r.evidence_type for r in results}
        assert RegionType.HOT_DWELL in types
        assert RegionType.COLD_DWELL in types
        assert RegionType.AMBIENT_START in types


class TestOvershootClassifier:
    @pytest.fixture
    def classifier(self):
        return OvershootClassifier()

    @pytest.fixture
    def overshoot_data(self):
        temps = list(np.linspace(120, 130, 15)) + list(np.linspace(130, 125, 15))
        return {
            "temperatures": temps,
            "rolling_slopes": [2.0] * 15 + [-2.0] * 15,
            "duration_seconds": 30.0,
        }

    @pytest.fixture
    def context(self):
        return {
            "hot_setpoint": 125.0,
            "cold_setpoint": -40.0,
            "overshoot_threshold": 5.0,
            "max_overshoot_duration": 60.0,
        }

    def test_overshoot_detected(self, classifier, overshoot_data, context):
        evidence = classifier.compute_evidence(overshoot_data, context)
        assert evidence["peak_magnitude"] > 0.5
        assert evidence["overshoot_direction"] > 0.5


class TestCorrectionClassifier:
    @pytest.fixture
    def classifier(self):
        return CorrectionClassifier()

    @pytest.fixture
    def correction_data(self):
        temps = [130.0, 128.0, 126.0, 124.0, 125.0, 126.0, 125.0, 125.0]
        slopes = [2.0, -2.0, 1.0, -1.0, 0.5, -0.5, 0.0, 0.0]
        return {
            "temperatures": temps,
            "rolling_slopes": slopes,
            "duration_seconds": 40.0,
        }

    @pytest.fixture
    def context(self):
        return {
            "hot_setpoint": 125.0,
            "cold_setpoint": -40.0,
        }

    def test_correction_oscillation_detected(self, classifier, correction_data, context):
        evidence = classifier.compute_evidence(correction_data, context)
        assert evidence["oscillation_count"] > 0.0
        assert evidence["settling_confidence"] > 0.5


class TestTransientClassifier:
    @pytest.fixture
    def classifier(self):
        return TransientClassifier()

    @pytest.fixture
    def transient_data(self):
        return {
            "temperatures": [25.0, 26.0, 24.0, 25.5, 24.5],
            "rolling_slopes": [1.0, -2.0, 1.5, -1.0, 0.5],
            "rolling_mad": [0.5, 0.6, 0.5, 0.4, 0.5],
            "duration_seconds": 5.0,
        }

    @pytest.fixture
    def context(self):
        return {
            "min_region_duration": 10.0,
            "noise_floor": 0.1,
        }

    def test_short_duration_high_score(self, classifier, transient_data, context):
        evidence = classifier.compute_evidence(transient_data, context)
        assert evidence["short_duration"] > 0.3


class TestUnknownClassifier:
    @pytest.fixture
    def classifier(self):
        return UnknownClassifier()

    @pytest.fixture
    def ambiguous_data(self):
        return {
            "temperatures": [50.0] * 10,
            "other_classifier_scores": {
                RegionType.HEATING_RAMP: 0.3,
                RegionType.COOLING_RAMP: 0.25,
                RegionType.HOT_DWELL: 0.2,
            },
            "data_quality_flags": ["SPIKE", "GAP"],
        }

    @pytest.fixture
    def context(self):
        return {}

    def test_low_scores_detected(self, classifier, ambiguous_data, context):
        evidence = classifier.compute_evidence(ambiguous_data, context)
        assert evidence["low_all_scores"] > 0.5

    def test_conflicting_signals_detected(self, classifier, ambiguous_data, context):
        evidence = classifier.compute_evidence(ambiguous_data, context)
        assert evidence["conflicting_signals"] > 0.5

    def test_quality_issues_detected(self, classifier, ambiguous_data, context):
        evidence = classifier.compute_evidence(ambiguous_data, context)
        assert evidence["data_quality_issues"] > 0.0
