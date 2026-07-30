"""Unit tests for Phase 6 Retrofit 4: tolerance precedence resolver."""

import pytest
from datetime import datetime

from config.constants import AuditCategory
from engine.validation.tolerance_resolver import (
    AdaptiveConstants,
    resolve_tolerance,
)
from models.domain import AuditLog, ToleranceResolution
from models.profile import (
    ValidationProfile,
    ProfileMetadata,
    ExpectedProcessSequence,
    RampRateRequirements,
    DwellRequirements,
    SetpointDeviationRequirements,
    OvershootRequirements,
    SettlingRequirements,
    DataQualityRequirements,
    ClassificationSettings,
    CycleRules,
    VisualisationSettings,
    ReportingSettings,
)


def _make_profile(
    dwell_deviation: float | None = 5.0,
    ramp_deviation: float | None = None,
) -> ValidationProfile:
    return ValidationProfile(
        profile_metadata=ProfileMetadata(
            profile_name="Test",
            profile_version="1.0.0",
            created_date=datetime(2025, 1, 1, 0, 0, 0),
            description="Test",
            algorithm_version_required="1.0.0",
        ),
        expected_process_sequence=ExpectedProcessSequence(),
        ramp_rate_requirements=RampRateRequirements(
            required_heating_ramp_rate_c_per_min=5.0,
            required_cooling_ramp_rate_c_per_min=3.0,
            allowed_ramp_deviation_c=ramp_deviation,
        ),
        dwell_requirements=DwellRequirements(
            minimum_hot_dwell_seconds=300.0,
            minimum_cold_dwell_seconds=300.0,
            allowed_setpoint_deviation_c=dwell_deviation,
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
        data_quality_requirements=DataQualityRequirements(),
        classification_settings=ClassificationSettings(),
        cycle_rules=CycleRules(),
        visualisation_settings=VisualisationSettings(),
        reporting_settings=ReportingSettings(),
    )


class TestResolveToleranceExplicit:
    def test_dwell_setpoint_deviation_explicit(self):
        profile = _make_profile(dwell_deviation=4.0)
        audit_log = AuditLog()

        resolution = resolve_tolerance(
            "dwell_setpoint_deviation", profile, audit_log=audit_log
        )

        assert resolution.source == "EXPLICIT_PROFILE"
        assert resolution.resolved_value == 4.0
        assert resolution.explicit_value_provided == 4.0
        assert resolution.adaptive_value_skipped is True
        assert resolution.derivation_method is None

    def test_ramp_deviation_explicit(self):
        profile = _make_profile(ramp_deviation=2.0)
        audit_log = AuditLog()

        resolution = resolve_tolerance(
            "ramp_deviation", profile, audit_log=audit_log
        )

        assert resolution.source == "EXPLICIT_PROFILE"
        assert resolution.resolved_value == 2.0

    def test_explicit_adaptive_never_invoked(self):
        """Guardrail: adaptive must not run when explicit exists."""
        profile = _make_profile(dwell_deviation=4.0)
        audit_log = AuditLog()

        # Pass adaptive constants that would return a different value
        adaptive = AdaptiveConstants(
            constants={"dwell_setpoint_deviation": 99.0}
        )

        resolution = resolve_tolerance(
            "dwell_setpoint_deviation", profile, adaptive, audit_log
        )

        # Should use explicit (4.0), not adaptive (99.0)
        assert resolution.resolved_value == 4.0
        assert resolution.source == "EXPLICIT_PROFILE"

    def test_audit_entry_logged(self):
        profile = _make_profile(dwell_deviation=4.0)
        audit_log = AuditLog()

        resolve_tolerance("dwell_setpoint_deviation", profile, audit_log=audit_log)

        entries = [e for e in audit_log.entries if e.action == "TOLERANCE_RESOLVED"]
        assert len(entries) == 1
        assert entries[0].category == AuditCategory.VALIDATION


class TestResolveToleranceAdaptive:
    def test_dwell_setpoint_deviation_adaptive(self):
        profile = _make_profile(dwell_deviation=None)
        audit_log = AuditLog()
        adaptive = AdaptiveConstants(
            constants={"dwell_setpoint_deviation": 2.5},
            derivation_methods={"dwell_setpoint_deviation": "noise_floor_multiplier"},
        )

        resolution = resolve_tolerance(
            "dwell_setpoint_deviation", profile, adaptive, audit_log
        )

        assert resolution.source == "ADAPTIVE_DERIVED"
        assert resolution.resolved_value == 2.5
        assert resolution.explicit_value_provided is None
        assert resolution.adaptive_value_skipped is False
        assert resolution.derivation_method == "noise_floor_multiplier"

    def test_ramp_deviation_adaptive(self):
        profile = _make_profile(ramp_deviation=None)
        audit_log = AuditLog()
        adaptive = AdaptiveConstants(
            constants={"ramp_deviation": 1.5},
            derivation_methods={"ramp_deviation": "slope_noise_floor"},
        )

        resolution = resolve_tolerance(
            "ramp_deviation", profile, adaptive, audit_log
        )

        assert resolution.source == "ADAPTIVE_DERIVED"
        assert resolution.resolved_value == 1.5
        assert resolution.derivation_method == "slope_noise_floor"

    def test_adaptive_no_derivation_method(self):
        profile = _make_profile(dwell_deviation=None)
        adaptive = AdaptiveConstants(
            constants={"dwell_setpoint_deviation": 2.5},
        )

        resolution = resolve_tolerance(
            "dwell_setpoint_deviation", profile, adaptive
        )

        assert resolution.derivation_method is None

    def test_no_explicit_and_no_adaptive_raises(self):
        profile = _make_profile(dwell_deviation=None)

        with pytest.raises(ValueError, match="adaptive_constants not provided"):
            resolve_tolerance("dwell_setpoint_deviation", profile)

    def test_no_explicit_and_no_adaptive_value_raises(self):
        profile = _make_profile(dwell_deviation=None)
        adaptive = AdaptiveConstants(constants={})

        with pytest.raises(ValueError, match="No explicit or adaptive tolerance"):
            resolve_tolerance("dwell_setpoint_deviation", profile, adaptive)


class TestAdaptiveConstants:
    def test_get_returns_value(self):
        ac = AdaptiveConstants(constants={"dwell_setpoint_deviation": 2.5})
        assert ac.get("dwell_setpoint_deviation") == 2.5

    def test_get_returns_none_for_missing(self):
        ac = AdaptiveConstants(constants={})
        assert ac.get("dwell_setpoint_deviation") is None

    def test_get_derivation_method(self):
        ac = AdaptiveConstants(
            derivation_methods={"dwell_setpoint_deviation": "noise_floor"}
        )
        assert ac.get_derivation_method("dwell_setpoint_deviation") == "noise_floor"


class TestProfileGetExplicitTolerance:
    def test_dwell_setpoint_deviation_from_dwell_requirements(self):
        profile = _make_profile(dwell_deviation=4.0)
        assert profile.get_explicit_tolerance("dwell_setpoint_deviation") == 4.0

    def test_dwell_setpoint_deviation_falls_back(self):
        """When dwell_requirements is None, fall back to setpoint_deviation_requirements."""
        profile = _make_profile(dwell_deviation=None)
        # setpoint_deviation_requirements.allowed_setpoint_deviation_c = 3.0
        assert profile.get_explicit_tolerance("dwell_setpoint_deviation") == 3.0

    def test_ramp_deviation(self):
        profile = _make_profile(ramp_deviation=2.0)
        assert profile.get_explicit_tolerance("ramp_deviation") == 2.0

    def test_unknown_parameter_returns_none(self):
        profile = _make_profile()
        assert profile.get_explicit_tolerance("unknown") is None
