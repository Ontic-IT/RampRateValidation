"""Unit tests for Phase 1: models/profile.py — ValidationProfile with 12 sub-objects."""

import pytest
from datetime import datetime
from pathlib import Path

import yaml
from pydantic import ValidationError as PydanticValidationError

from config.constants import AmbiguityHandling, FirstExtreme
from models.profile import (
    ClassificationSettings,
    CycleRules,
    DataQualityRequirements,
    DwellRequirements,
    ExpectedProcessSequence,
    OvershootRequirements,
    ProfileMetadata,
    RampRateRequirements,
    ReportingSettings,
    SetpointDeviationRequirements,
    SettlingRequirements,
    ToleranceResolution,
    ValidationProfile,
    VisualisationSettings,
)


NOW = datetime(2025, 1, 1, 0, 0, 0)


class TestProfileMetadata:
    def test_instantiation(self):
        pm = ProfileMetadata(
            profile_name="Test",
            profile_version="1.0.0",
            created_date=NOW,
            description="Test profile",
            algorithm_version_required="1.0.0",
        )
        assert pm.profile_name == "Test"
        assert pm.algorithm_version_required == "1.0.0"


class TestExpectedProcessSequence:
    def test_hot_first(self):
        eps = ExpectedProcessSequence(first_extreme=FirstExtreme.HOT_FIRST)
        assert eps.first_extreme == FirstExtreme.HOT_FIRST

    def test_cold_first(self):
        eps = ExpectedProcessSequence(first_extreme=FirstExtreme.COLD_FIRST)
        assert eps.first_extreme == FirstExtreme.COLD_FIRST


class TestRampRateRequirements:
    def test_instantiation(self):
        rrr = RampRateRequirements(
            required_heating_ramp_rate_c_per_min=5.0,
            required_cooling_ramp_rate_c_per_min=5.0,
        )
        assert rrr.minimum_sustained_ramp_rate_ratio == 0.8
        assert rrr.sustained_ramp_window_seconds == 60.0


class TestDwellRequirements:
    def test_instantiation(self):
        dr = DwellRequirements(
            minimum_hot_dwell_seconds=300.0,
            minimum_cold_dwell_seconds=300.0,
            allowed_setpoint_deviation_c=3.0,
        )
        assert dr.dwell_stability_window_seconds == 30.0
        assert dr.tolerance_source == "EXPLICIT"

    def test_optional_setpoint_deviation(self):
        dr = DwellRequirements(
            minimum_hot_dwell_seconds=300.0,
            minimum_cold_dwell_seconds=300.0,
        )
        assert dr.allowed_setpoint_deviation_c is None
        assert dr.tolerance_source == "EXPLICIT"


class TestToleranceResolution:
    def test_explicit_user(self):
        tr = ToleranceResolution(
            parameter_name="dwell_setpoint_deviation",
            resolved_value=3.0,
            source="EXPLICIT_USER",
            explicit_value_provided=3.0,
        )
        assert tr.adaptive_value_skipped is True
        assert tr.derivation_method is None

    def test_adaptive_derived(self):
        tr = ToleranceResolution(
            parameter_name="dwell_setpoint_deviation",
            resolved_value=2.5,
            source="ADAPTIVE_DERIVED",
            derivation_method="noise_floor_multiplier",
        )
        assert tr.adaptive_value_skipped is False
        assert tr.derivation_method == "noise_floor_multiplier"


class TestClassificationSettings:
    def test_defaults(self):
        cs = ClassificationSettings()
        assert cs.secondary_classification_threshold == 0.5
        assert cs.ambiguity_margin_threshold == 0.1
        assert cs.high_confidence_threshold == 0.8
        assert cs.medium_confidence_threshold == 0.6
        assert cs.ambiguity_handling == AmbiguityHandling.WARN


class TestValidationProfile:
    @pytest.fixture
    def minimal_profile(self):
        return ValidationProfile(
            profile_metadata=ProfileMetadata(
                profile_name="Test",
                profile_version="1.0.0",
                created_date=NOW,
                description="Test",
                algorithm_version_required="1.0.0",
            ),
            expected_process_sequence=ExpectedProcessSequence(),
            ramp_rate_requirements=RampRateRequirements(
                required_heating_ramp_rate_c_per_min=5.0,
                required_cooling_ramp_rate_c_per_min=5.0,
                allowed_ramp_deviation_c=1.0,
                tolerance_source="EXPLICIT",
            ),
            dwell_requirements=DwellRequirements(
                minimum_hot_dwell_seconds=300.0,
                minimum_cold_dwell_seconds=300.0,
                allowed_setpoint_deviation_c=3.0,
                tolerance_source="EXPLICIT",
            ),
            setpoint_deviation_requirements=SetpointDeviationRequirements(
                allowed_setpoint_deviation_c=3.0,
                setpoint_deviation_warning_c=2.0,
            ),
            overshoot_requirements=OvershootRequirements(
                overshoot_warning_threshold_c=5.0,
                overshoot_failure_threshold_c=10.0,
                max_overshoot_duration_seconds=60.0,
            ),
            settling_requirements=SettlingRequirements(
                settling_time_limit_seconds=120.0,
                settling_tolerance_band_c=2.0,
            ),
        )

    def test_instantiation_with_12_sub_objects(self, minimal_profile):
        assert minimal_profile.profile_metadata.profile_name == "Test"
        assert minimal_profile.data_quality_requirements is not None
        assert minimal_profile.classification_settings is not None
        assert minimal_profile.cycle_rules is not None
        assert minimal_profile.visualisation_settings is not None
        assert minimal_profile.reporting_settings is not None

    def test_all_12_sub_objects_are_populated(self, minimal_profile):
        fields = [
            "profile_metadata",
            "expected_process_sequence",
            "ramp_rate_requirements",
            "dwell_requirements",
            "setpoint_deviation_requirements",
            "overshoot_requirements",
            "settling_requirements",
            "data_quality_requirements",
            "classification_settings",
            "cycle_rules",
            "visualisation_settings",
            "reporting_settings",
        ]
        for field_name in fields:
            assert getattr(minimal_profile, field_name) is not None, f"{field_name} is None"

    def test_profile_yaml_round_trip(self, minimal_profile, tmp_path):
        """Profile can be serialised to YAML and deserialised back."""
        data = minimal_profile.model_dump(mode="json")
        yaml_path = tmp_path / "profile.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(data, f, default_flow_style=False)

        with open(yaml_path) as f:
            loaded = yaml.safe_load(f)

        profile2 = ValidationProfile(**loaded)
        assert profile2.profile_metadata.profile_name == "Test"


class TestExampleProfileYaml:
    """Verify the shipped example_profile.yaml loads and validates."""

    def test_example_profile_loads(self):
        profile_path = Path(__file__).parent.parent / "config" / "profiles" / "example_profile.yaml"
        if not profile_path.exists():
            pytest.skip("example_profile.yaml not found")

        with open(profile_path) as f:
            data = yaml.safe_load(f)

        profile = ValidationProfile(**data)
        assert profile.profile_metadata.profile_name == "Example Thermal Cycling Profile"
        assert profile.profile_metadata.algorithm_version_required == "1.0.0"
        assert len(profile.visualisation_settings.region_colour_map) == 12
        assert profile.tolerance_resolutions == []
