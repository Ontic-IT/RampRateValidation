"""Unit tests for Phase 1: config/constants.py — all enums and static constants."""

import pytest

from config.constants import (
    AmbiguityHandling,
    AuditCategory,
    AuditSeverity,
    Comparator,
    ConfidenceLevel,
    CycleStatus,
    DataQualityStatus,
    FirstExtreme,
    GapSignificance,
    OverallValidationStatus,
    PipelineStage,
    PipelineStatus,
    RampDirection,
    RegionType,
    SetpointResolutionMode,
    ValidationStatus,
)


class TestRegionType:
    """RegionType enum has exactly 14 values."""

    def test_region_type_count(self):
        assert len(RegionType) == 14

    def test_all_region_types_present(self):
        expected = {
            "HOT_DWELL", "COLD_DWELL", "AMBIENT_START",
            "HEATING_RAMP", "COOLING_RAMP", "RAMP_JITTER", "RAMP_TAPER",
            "HOT_OVERSHOOT", "COLD_OVERSHOOT",
            "HOT_CORRECTION", "COLD_CORRECTION",
            "RECOVERY", "TRANSIENT", "UNKNOWN",
        }
        assert {rt.value for rt in RegionType} == expected

    def test_region_type_is_str_enum(self):
        assert isinstance(RegionType.HEATING_RAMP, str)
        assert RegionType.HEATING_RAMP == "HEATING_RAMP"


class TestValidationStatus:
    """ValidationStatus enum has exactly 5 values."""

    def test_count(self):
        assert len(ValidationStatus) == 5

    def test_all_present(self):
        expected = {"PASS", "PASS_WITH_WARNINGS", "FAIL", "INCONCLUSIVE", "NOT_APPLICABLE"}
        assert {vs.value for vs in ValidationStatus} == expected

    def test_bare_warning_rejected(self):
        with pytest.raises(ValueError):
            ValidationStatus("WARNING")


class TestOverallValidationStatus:
    """OverallValidationStatus enum has exactly 6 values."""

    def test_count(self):
        assert len(OverallValidationStatus) == 6

    def test_all_present(self):
        expected = {"PASS", "PASS_WITH_WARNINGS", "FAIL", "INCONCLUSIVE", "INVALID_INPUT", "ERROR"}
        assert {ovs.value for ovs in OverallValidationStatus} == expected

    def test_bare_warning_rejected(self):
        with pytest.raises(ValueError):
            OverallValidationStatus("WARNING")


class TestDataQualityStatus:
    """DataQualityStatus enum has exactly 4 values."""

    def test_count(self):
        assert len(DataQualityStatus) == 4

    def test_all_present(self):
        expected = {"ACCEPTABLE", "WARNING", "INCONCLUSIVE", "INVALID"}
        assert {dqs.value for dqs in DataQualityStatus} == expected


class TestCycleStatus:
    """CycleStatus enum has exactly 5 values."""

    def test_count(self):
        assert len(CycleStatus) == 5

    def test_all_present(self):
        expected = {"COMPLETE", "PARTIAL", "ABORTED", "AMBIGUOUS", "INVALID"}
        assert {cs.value for cs in CycleStatus} == expected


class TestAuditSeverity:
    def test_all_present(self):
        expected = {"INFO", "WARNING", "ERROR", "CRITICAL"}
        assert {s.value for s in AuditSeverity} == expected


class TestAuditCategory:
    def test_includes_boundary_normalisation(self):
        assert AuditCategory.BOUNDARY_NORMALISATION == "BOUNDARY_NORMALISATION"

    def test_all_present(self):
        expected = {
            "CLASSIFICATION", "VALIDATION", "QUALITY", "PROFILE",
            "METRICS", "PIPELINE", "BOUNDARY_NORMALISATION",
        }
        assert {ac.value for ac in AuditCategory} == expected


class TestConfidenceLevel:
    def test_all_present(self):
        expected = {"HIGH", "MEDIUM", "LOW"}
        assert {cl.value for cl in ConfidenceLevel} == expected


class TestPipelineStage:
    def test_count(self):
        # 15 stages + COMPLETE
        assert len(PipelineStage) == 16

    def test_strict_order_representable(self):
        stages = list(PipelineStage)
        assert stages[0] == PipelineStage.INGESTION
        assert stages[-1] == PipelineStage.COMPLETE


class TestPipelineStatus:
    def test_all_present(self):
        expected = {"RUNNING", "COMPLETED", "FAILED", "SKIPPED"}
        assert {ps.value for ps in PipelineStatus} == expected


class TestComparator:
    def test_all_present(self):
        expected = {"GTE", "LTE", "EQ", "RANGE"}
        assert {c.value for c in Comparator} == expected


class TestFirstExtreme:
    def test_all_present(self):
        expected = {"HOT_FIRST", "COLD_FIRST"}
        assert {fe.value for fe in FirstExtreme} == expected


class TestAmbiguityHandling:
    def test_all_present(self):
        expected = {"WARN", "INCONCLUSIVE", "ALLOW"}
        assert {ah.value for ah in AmbiguityHandling} == expected


class TestSetpointResolutionMode:
    def test_all_present(self):
        expected = {"MODE_A", "MODE_B"}
        assert {m.value for m in SetpointResolutionMode} == expected


class TestGapSignificance:
    def test_all_present(self):
        expected = {"NEGLIGIBLE", "MINOR", "MODERATE", "SEVERE"}
        assert {gs.value for gs in GapSignificance} == expected


class TestRampDirection:
    def test_all_present(self):
        expected = {"HEATING", "COOLING"}
        assert {rd.value for rd in RampDirection} == expected
