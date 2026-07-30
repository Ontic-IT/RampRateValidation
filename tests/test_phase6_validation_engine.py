"""Unit tests for Phase 6: engine/validation/validation_engine.py — main validation engine."""

import pytest
from datetime import datetime, timedelta

from config.constants import (
    CycleStatus,
    OverallValidationStatus,
    RampDirection,
    RegionType,
    ValidationStatus,
)
from engine.validation.validation_engine import validate_analysis
from models.domain import (
    AuditLog,
    ClassificationEvidence,
    Cycle,
    CycleList,
    DwellMetrics,
    RampMetrics,
    Region,
    RegionList,
    ResolvedSetpoints,
    ValidRampRegion,
    ValidationDataQualityImpact,
)
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


def _make_profile() -> ValidationProfile:
    """Create a test validation profile."""
    return ValidationProfile(
        profile_metadata=ProfileMetadata(
            profile_name="Test Profile",
            profile_version="1.0.0",
            created_date=datetime(2025, 1, 1, 0, 0, 0),
            description="Test profile",
            algorithm_version_required="1.0.0",
        ),
        expected_process_sequence=ExpectedProcessSequence(
            expected_region_sequence=["HEATING_RAMP", "HOT_DWELL", "COOLING_RAMP", "COLD_DWELL"],
        ),
        ramp_rate_requirements=RampRateRequirements(
            required_heating_ramp_rate_c_per_min=5.0,
            required_cooling_ramp_rate_c_per_min=3.0,
            minimum_sustained_ramp_rate_ratio=0.8,
        ),
        dwell_requirements=DwellRequirements(
            minimum_hot_dwell_seconds=300.0,
            minimum_cold_dwell_seconds=300.0,
            allowed_setpoint_deviation_c=5.0,
        ),
        setpoint_deviation_requirements=SetpointDeviationRequirements(
            allowed_setpoint_deviation_c=5.0,
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
        cycle_rules=CycleRules(
            minimum_complete_cycles_required=1,
        ),
        visualisation_settings=VisualisationSettings(),
        reporting_settings=ReportingSettings(),
    )


def _make_region(region_id: str, classification: RegionType, duration: float = 300.0) -> Region:
    """Helper to create a region."""
    base = datetime(2025, 1, 1, 0, 0, 0)
    return Region(
        region_id=region_id,
        start_row=0,
        end_row=100,
        start_time=base,
        end_time=base + timedelta(seconds=duration),
        duration_seconds=duration,
        primary_classification=classification,
        classification_scores={classification: 0.9},
        classification_margin=0.5,
        classification_evidence=[],
        classification_confidence=0.9,
    )


class TestValidateAnalysis:
    @pytest.fixture
    def profile(self):
        return _make_profile()

    @pytest.fixture
    def passing_regions(self):
        return RegionList(regions=[
            _make_region("R0001", RegionType.HEATING_RAMP, 1200.0),
            _make_region("R0002", RegionType.HOT_DWELL, 600.0),
            _make_region("R0003", RegionType.COOLING_RAMP, 1980.0),
            _make_region("R0004", RegionType.COLD_DWELL, 600.0),
        ])

    @pytest.fixture
    def passing_valid_ramps(self):
        return [
            ValidRampRegion(
                region_id="R0001",
                direction=RampDirection.HEATING,
                duration_seconds=1200.0,
                included_rows=list(range(100)),
            ),
            ValidRampRegion(
                region_id="R0003",
                direction=RampDirection.COOLING,
                duration_seconds=1980.0,
                included_rows=list(range(100)),
            ),
        ]

    @pytest.fixture
    def passing_ramp_metrics(self):
        return [
            RampMetrics(
                region_id="R0001",
                robust_slope_c_per_min=6.0,
                minimum_sustained_slope_c_per_min=5.0,
            ),
            RampMetrics(
                region_id="R0003",
                robust_slope_c_per_min=-4.0,
                minimum_sustained_slope_c_per_min=-3.5,
            ),
        ]

    @pytest.fixture
    def passing_dwell_metrics(self):
        return [
            DwellMetrics(
                region_id="R0002",
                setpoint_deviation_c=2.0,
                overshoot_magnitude_c=3.0,
                settling_time_seconds=20.0,
            ),
            DwellMetrics(
                region_id="R0004",
                setpoint_deviation_c=1.5,
                overshoot_magnitude_c=2.0,
                settling_time_seconds=15.0,
            ),
        ]

    @pytest.fixture
    def passing_cycles(self):
        base = datetime(2025, 1, 1, 0, 0, 0)
        return CycleList(cycles=[
            Cycle(
                cycle_id="C0001",
                cycle_number=1,
                start_row=0,
                end_row=1000,
                start_time=base,
                end_time=base + timedelta(seconds=3600),
                duration_seconds=3600.0,
                status=CycleStatus.COMPLETE,
            ),
        ])

    def test_returns_validation_results_and_overall_status(
        self, profile, passing_regions, passing_valid_ramps,
        passing_ramp_metrics, passing_dwell_metrics, passing_cycles
    ):
        results, overall = validate_analysis(
            profile,
            passing_regions,
            passing_valid_ramps,
            passing_ramp_metrics,
            passing_dwell_metrics,
            passing_cycles,
        )
        
        assert results is not None
        assert overall is not None
        assert len(results.results) > 0

    def test_all_passing_returns_pass(
        self, profile, passing_regions, passing_valid_ramps,
        passing_ramp_metrics, passing_dwell_metrics, passing_cycles
    ):
        results, overall = validate_analysis(
            profile,
            passing_regions,
            passing_valid_ramps,
            passing_ramp_metrics,
            passing_dwell_metrics,
            passing_cycles,
        )
        
        assert overall.status == OverallValidationStatus.PASS

    def test_failing_ramp_rate_returns_fail(
        self, profile, passing_regions, passing_valid_ramps,
        passing_dwell_metrics, passing_cycles
    ):
        failing_metrics = [
            RampMetrics(
                region_id="R0001",
                robust_slope_c_per_min=3.0,
                minimum_sustained_slope_c_per_min=2.5,
            ),
            RampMetrics(
                region_id="R0003",
                robust_slope_c_per_min=-4.0,
                minimum_sustained_slope_c_per_min=-3.5,
            ),
        ]
        
        results, overall = validate_analysis(
            profile,
            passing_regions,
            passing_valid_ramps,
            failing_metrics,
            passing_dwell_metrics,
            passing_cycles,
        )
        
        assert overall.status == OverallValidationStatus.FAIL

    def test_quality_blocked_returns_inconclusive(
        self, profile, passing_regions, passing_valid_ramps,
        passing_ramp_metrics, passing_dwell_metrics, passing_cycles
    ):
        quality_impact = ValidationDataQualityImpact(
            blocks_pass_fail=True,
            affected_requirement_ids=["ALL"],
            reason="Data quality too poor",
        )
        
        results, overall = validate_analysis(
            profile,
            passing_regions,
            passing_valid_ramps,
            passing_ramp_metrics,
            passing_dwell_metrics,
            passing_cycles,
            quality_impact=quality_impact,
        )
        
        assert overall.status == OverallValidationStatus.INCONCLUSIVE

    def test_audit_log_recorded(
        self, profile, passing_regions, passing_valid_ramps,
        passing_ramp_metrics, passing_dwell_metrics, passing_cycles
    ):
        audit_log = AuditLog()
        
        validate_analysis(
            profile,
            passing_regions,
            passing_valid_ramps,
            passing_ramp_metrics,
            passing_dwell_metrics,
            passing_cycles,
            audit_log=audit_log,
        )
        
        entries = [e for e in audit_log.entries if "validate" in e.action]
        assert len(entries) > 0

    def test_validates_sequence(
        self, profile, passing_valid_ramps,
        passing_ramp_metrics, passing_dwell_metrics, passing_cycles
    ):
        wrong_sequence_regions = RegionList(regions=[
            _make_region("R0001", RegionType.COOLING_RAMP, 1980.0),
            _make_region("R0002", RegionType.COLD_DWELL, 600.0),
        ])
        
        results, overall = validate_analysis(
            profile,
            wrong_sequence_regions,
            passing_valid_ramps,
            passing_ramp_metrics,
            passing_dwell_metrics,
            passing_cycles,
        )
        
        sequence_results = [r for r in results.results if r.requirement_id == "PROFILE_SEQUENCE"]
        assert len(sequence_results) == 1
        assert sequence_results[0].result == ValidationStatus.FAIL

    def test_validates_cycle_count(
        self, profile, passing_regions, passing_valid_ramps,
        passing_ramp_metrics, passing_dwell_metrics
    ):
        empty_cycles = CycleList(cycles=[])
        
        results, overall = validate_analysis(
            profile,
            passing_regions,
            passing_valid_ramps,
            passing_ramp_metrics,
            passing_dwell_metrics,
            empty_cycles,
        )
        
        cycle_results = [r for r in results.results if r.requirement_id == "CYCLE_COUNT"]
        assert len(cycle_results) == 1
        assert cycle_results[0].result == ValidationStatus.FAIL
