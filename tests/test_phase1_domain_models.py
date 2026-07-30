"""Unit tests for Phase 1: models/domain.py — all Pydantic v2 domain models."""

import pytest
from datetime import datetime

from pydantic import ValidationError as PydanticValidationError

from config.constants import (
    AuditCategory,
    AuditSeverity,
    Comparator,
    ConfidenceLevel,
    CycleStatus,
    DataQualityStatus,
    OverallValidationStatus as OverallValidationStatusEnum,
    PipelineStage,
    PipelineStatus,
    RegionType,
    SetpointResolutionMode,
    ValidationStatus,
)
from models.domain import (
    AnalysisContext,
    AnalysisRequest,
    AnalysisResult,
    AuditEntry,
    AuditLog,
    CanonicalTrace,
    CanonicalTraceRow,
    ClassificationEvidence,
    ClassificationWeightVersion,
    ClassifiedTrace,
    CorrectionEvidence,
    Cycle,
    CycleList,
    DwellEvidence,
    FileMetadata,
    MetricSet,
    OvershootEvidence,
    PipelineStageState,
    PreprocessedTrace,
    PreprocessingReport,
    ProcessBoundaries,
    ProfileComparisonResults,
    RampDataQualityStatus,
    RampEvidence,
    RawTraceData,
    Region,
    RegionDataQualityStatus,
    RegionList,
    ReportPackage,
    ResolvedSetpoints,
    RunDataQualityReport,
    RunMetadata,
    TransientEvidence,
    UnknownEvidence,
    ValidRampRegion,
    ValidRampRegionList,
    ValidationDataQualityImpact,
    ValidationResult,
    ValidationResults,
    VisualisationBundle,
)


NOW = datetime(2025, 1, 1, 0, 0, 0)


# ---------------------------------------------------------------------------
# Evidence models
# ---------------------------------------------------------------------------

class TestClassificationEvidence:
    def test_instantiation(self):
        e = ClassificationEvidence(
            evidence_type=RegionType.HEATING_RAMP,
            score=0.85,
            reason="strong slope",
            evidence={"slope_magnitude": 0.9},
            classifier_name="ramp_classifier",
            timestamp=NOW,
        )
        assert e.score == 0.85
        assert e.evidence_type == RegionType.HEATING_RAMP

    def test_score_bounds(self):
        with pytest.raises(PydanticValidationError):
            ClassificationEvidence(
                evidence_type=RegionType.HEATING_RAMP,
                score=1.5,
                reason="invalid",
                evidence={},
                classifier_name="test",
                timestamp=NOW,
            )

    def test_score_negative_rejected(self):
        with pytest.raises(PydanticValidationError):
            ClassificationEvidence(
                evidence_type=RegionType.HEATING_RAMP,
                score=-0.1,
                reason="invalid",
                evidence={},
                classifier_name="test",
                timestamp=NOW,
            )


class TestRampEvidence:
    def test_instantiation_with_all_keys(self):
        evidence_data = {
            "sustained_positive_slope": 0.9,
            "sustained_negative_slope": 0.0,
            "dwell_departure_confidence": 0.8,
            "dwell_arrival_confidence": 0.7,
            "monotonicity_score": 0.85,
            "slope_magnitude": 0.9,
            "overshoot_characteristics": 0.1,
            "stability_score": 0.8,
            "duration_adequacy": 0.95,
            "reversal_tolerance": 0.9,
            "boundary_stability_score": 0.7,
            "transition_persistence_score": 0.8,
            "arrival_certainty_score": 0.75,
            "departure_certainty_score": 0.8,
            "local_gap_resilience_score": 0.95,
        }
        e = RampEvidence(
            evidence_type=RegionType.HEATING_RAMP,
            score=0.88,
            reason="strong heating ramp",
            evidence=evidence_data,
            classifier_name="ramp_classifier",
            timestamp=NOW,
        )
        assert e.evidence["monotonicity_score"] == 0.85


class TestDwellEvidence:
    def test_instantiation(self):
        e = DwellEvidence(
            evidence_type=RegionType.HOT_DWELL,
            score=0.92,
            reason="stable temperature",
            evidence={
                "temperature_stability": 0.95,
                "setpoint_proximity": 0.9,
                "duration_adequacy": 0.88,
                "tolerance_band_persistence": 0.92,
                "cluster_membership": 0.85,
                "boundary_confidence": 0.8,
                "variance_score": 0.9,
                "directionality_absence": 0.95,
            },
            classifier_name="dwell_classifier",
            timestamp=NOW,
        )
        assert e.score == 0.92


class TestOvershootEvidence:
    def test_instantiation(self):
        e = OvershootEvidence(
            evidence_type=RegionType.HOT_OVERSHOOT,
            score=0.75,
            reason="peak above setpoint",
            evidence={
                "peak_magnitude": 0.8,
                "overshoot_direction": 0.9,
                "return_confidence": 0.7,
                "duration_adequacy": 0.6,
                "boundary_confidence": 0.65,
                "oscillation_pattern": 0.5,
                "settling_characteristics": 0.7,
            },
            classifier_name="overshoot_classifier",
            timestamp=NOW,
        )
        assert e.evidence_type == RegionType.HOT_OVERSHOOT


class TestCorrectionEvidence:
    def test_instantiation(self):
        e = CorrectionEvidence(
            evidence_type=RegionType.HOT_CORRECTION,
            score=0.65,
            reason="oscillation after overshoot",
            evidence={
                "return_direction": 0.7,
                "oscillation_count": 0.6,
                "amplitude_decay": 0.8,
                "settling_confidence": 0.7,
                "boundary_confidence": 0.5,
                "duration_adequacy": 0.6,
            },
            classifier_name="correction_classifier",
            timestamp=NOW,
        )
        assert e.score == 0.65


class TestTransientEvidence:
    def test_instantiation(self):
        e = TransientEvidence(
            evidence_type=RegionType.TRANSIENT,
            score=0.55,
            reason="short ambiguous region",
            evidence={
                "short_duration": 0.9,
                "low_slope_magnitude": 0.7,
                "high_variance": 0.6,
                "boundary_ambiguity": 0.8,
                "lack_of_pattern": 0.5,
            },
            classifier_name="transient_classifier",
            timestamp=NOW,
        )
        assert e.evidence_type == RegionType.TRANSIENT


class TestUnknownEvidence:
    def test_instantiation(self):
        e = UnknownEvidence(
            evidence_type=RegionType.UNKNOWN,
            score=0.3,
            reason="no strong signals",
            evidence={
                "low_all_scores": 0.9,
                "conflicting_signals": 0.8,
                "data_quality_issues": 0.5,
                "outlier_characteristics": 0.3,
            },
            classifier_name="unknown_classifier",
            timestamp=NOW,
        )
        assert e.evidence_type == RegionType.UNKNOWN


# ---------------------------------------------------------------------------
# Region model
# ---------------------------------------------------------------------------

class TestRegion:
    def test_instantiation_with_required_fields(self):
        r = Region(
            region_id="R001",
            start_row=0,
            end_row=100,
            start_time=NOW,
            end_time=NOW,
            duration_seconds=60.0,
            primary_classification=RegionType.HEATING_RAMP,
        )
        assert r.region_id == "R001"
        assert r.primary_classification == RegionType.HEATING_RAMP
        assert r.is_ambiguous is False
        assert r.included_in_primary_validation is True

    def test_classification_scores_stored_per_region_type(self):
        scores = {rt: 0.0 for rt in RegionType}
        scores[RegionType.HEATING_RAMP] = 0.88
        scores[RegionType.RAMP_JITTER] = 0.12
        r = Region(
            region_id="R002",
            start_row=0,
            end_row=50,
            start_time=NOW,
            end_time=NOW,
            duration_seconds=30.0,
            primary_classification=RegionType.HEATING_RAMP,
            classification_scores=scores,
            classification_margin=0.76,
        )
        assert r.classification_scores[RegionType.HEATING_RAMP] == 0.88
        assert len(r.classification_scores) == 14

    def test_ambiguity_fields(self):
        r = Region(
            region_id="R003",
            start_row=0,
            end_row=50,
            start_time=NOW,
            end_time=NOW,
            duration_seconds=30.0,
            primary_classification=RegionType.HEATING_RAMP,
            is_ambiguous=True,
            ambiguity_reason="margin < threshold",
        )
        assert r.is_ambiguous is True
        assert r.ambiguity_reason == "margin < threshold"

    def test_boundary_confidence_model(self):
        r = Region(
            region_id="R004",
            start_row=10,
            end_row=90,
            start_time=NOW,
            end_time=NOW,
            duration_seconds=45.0,
            primary_classification=RegionType.HOT_DWELL,
            boundary_start_confidence=0.85,
            boundary_end_confidence=0.9,
            boundary_zone_start_row=8,
            boundary_zone_end_row=92,
            boundary_certainty=ConfidenceLevel.HIGH,
        )
        assert r.boundary_certainty == ConfidenceLevel.HIGH


# ---------------------------------------------------------------------------
# Trace models
# ---------------------------------------------------------------------------

class TestCanonicalTraceRow:
    def test_instantiation_all_18_fields(self):
        row = CanonicalTraceRow(
            timestamp=NOW,
            elapsed_seconds=0.0,
            elapsed_minutes=0.0,
            temperature_c_raw=25.0,
            channel="CH1",
            source_row=0,
            source_file="test.csv",
            sample_interval_seconds=1.0,
        )
        assert row.temperature_c_raw == 25.0
        assert row.temperature_c_analysis_signal is None
        assert row.local_slope_c_per_min is None

    def test_temperature_c_raw_is_frozen(self):
        row = CanonicalTraceRow(
            timestamp=NOW,
            elapsed_seconds=0.0,
            elapsed_minutes=0.0,
            temperature_c_raw=25.0,
            channel="CH1",
            source_row=0,
            source_file="test.csv",
            sample_interval_seconds=1.0,
        )
        with pytest.raises(PydanticValidationError):
            row.temperature_c_raw = 30.0


class TestCanonicalTrace:
    def test_empty_trace(self):
        ct = CanonicalTrace()
        assert len(ct.rows) == 0

    def test_trace_with_rows(self):
        row = CanonicalTraceRow(
            timestamp=NOW,
            elapsed_seconds=0.0,
            elapsed_minutes=0.0,
            temperature_c_raw=25.0,
            channel="CH1",
            source_row=0,
            source_file="test.csv",
            sample_interval_seconds=1.0,
        )
        ct = CanonicalTrace(rows=[row])
        assert len(ct.rows) == 1


class TestRawTraceData:
    def test_instantiation(self):
        rt = RawTraceData(
            columns=["time", "temp"],
            data=[{"time": "00:00", "temp": 25.0}],
            row_count=1,
        )
        assert rt.row_count == 1


# ---------------------------------------------------------------------------
# Data quality models
# ---------------------------------------------------------------------------

class TestPreprocessingReport:
    def test_instantiation(self):
        pr = PreprocessingReport(
            estimated_sample_interval_s=1.0,
            noise_floor_c=0.05,
        )
        assert pr.gap_density_score == 0.0
        assert pr.effective_data_continuity_score == 1.0


class TestRunDataQualityReport:
    def test_instantiation(self):
        r = RunDataQualityReport()
        assert r.overall_status == DataQualityStatus.ACCEPTABLE
        assert r.gap_density_score == 0.0


class TestRegionDataQualityStatus:
    def test_instantiation(self):
        s = RegionDataQualityStatus(region_id="R001")
        assert s.status == DataQualityStatus.ACCEPTABLE
        assert s.data_completeness_pct == 100.0


class TestRampDataQualityStatus:
    def test_instantiation(self):
        s = RampDataQualityStatus(valid_ramp_region_id="VR001")
        assert s.sufficient_for_compliance is True


class TestValidationDataQualityImpact:
    def test_instantiation(self):
        v = ValidationDataQualityImpact()
        assert v.blocks_pass_fail is False


# ---------------------------------------------------------------------------
# Boundary and setpoint models
# ---------------------------------------------------------------------------

class TestProcessBoundaries:
    def test_instantiation(self):
        pb = ProcessBoundaries(
            ambient_start_index=0,
            ambient_end_index=50,
            process_start_index=51,
            process_end_index=1000,
        )
        assert pb.detection_method == ""


class TestResolvedSetpoints:
    def test_instantiation_mode_b(self):
        rs = ResolvedSetpoints(
            resolution_mode=SetpointResolutionMode.MODE_B,
            inferred_ambient_c=25.0,
            inferred_hot_setpoint_c=125.0,
            inferred_cold_setpoint_c=-40.0,
        )
        assert rs.algorithm_seed_used == 42


# ---------------------------------------------------------------------------
# Compliance models
# ---------------------------------------------------------------------------

class TestValidRampRegion:
    def test_all_fields(self):
        vr = ValidRampRegion(
            region_id="VR001",
            start_row=50,
            end_row=170,
            start_time=NOW,
            end_time=NOW,
            duration_seconds=120.0,
            start_temperature_c=25.0,
            end_temperature_c=125.0,
            temperature_delta_c=100.0,
            included_rows=[50, 51, 52],
            monotonicity_score=0.95,
            reversal_count=0,
            stall_duration_seconds=0.0,
            departure_confidence=0.9,
            arrival_confidence=0.85,
        )
        assert vr.region_id == "VR001"
        assert vr.reversal_count == 0
        assert vr.monotonicity_score == 0.95


class TestCycle:
    def test_instantiation(self):
        c = Cycle(
            cycle_id="C001",
            status=CycleStatus.COMPLETE,
            region_ids=["R001", "R002", "R003"],
            region_sequence=["R001", "R002", "R003"],
            start_time=NOW,
            end_time=NOW,
            duration_seconds=600.0,
            cycle_number=1,
        )
        assert c.status == CycleStatus.COMPLETE


# ---------------------------------------------------------------------------
# Validation models
# ---------------------------------------------------------------------------

class TestValidationResult:
    def test_instantiation_with_all_required_fields(self):
        vr = ValidationResult(
            validation_result_id="VR001",
            requirement_id="REQ001",
            requirement_description="Heating ramp rate >= 5 C/min",
            measured_value=5.2,
            threshold_value=5.0,
            comparator=Comparator.GTE,
            unit="C/min",
            method="theil_sen",
        )
        assert vr.comparator == Comparator.GTE
        assert vr.unit == "C/min"
        assert vr.method == "theil_sen"
        assert vr.result == ValidationStatus.NOT_APPLICABLE


# ---------------------------------------------------------------------------
# Audit models
# ---------------------------------------------------------------------------

class TestAuditEntry:
    def test_instantiation(self):
        ae = AuditEntry(
            timestamp=NOW,
            module_name="region_classification",
            action="classify_region",
            severity=AuditSeverity.INFO,
            category=AuditCategory.CLASSIFICATION,
        )
        assert ae.classifier_version is None
        assert ae.weight_version is None
        assert ae.aggregation_method is None

    def test_with_weight_auditability(self):
        ae = AuditEntry(
            timestamp=NOW,
            module_name="region_classification",
            action="classify_region",
            severity=AuditSeverity.INFO,
            category=AuditCategory.CLASSIFICATION,
            classifier_version="ramp_v1",
            weight_version="v1.0.0",
            aggregation_method="weighted_sum",
        )
        assert ae.weight_version == "v1.0.0"


class TestAuditLog:
    def test_add_entry(self):
        log = AuditLog()
        entry = AuditEntry(
            timestamp=NOW,
            module_name="test",
            action="test_action",
        )
        log.add(entry)
        assert len(log.entries) == 1


# ---------------------------------------------------------------------------
# Reproducibility models
# ---------------------------------------------------------------------------

class TestClassificationWeightVersion:
    def test_instantiation(self):
        cwv = ClassificationWeightVersion(
            version="1.0.0",
            weight_file_hash="abc123",
            created_date=NOW,
            calibration_dataset_version="initial",
        )
        assert cwv.version == "1.0.0"


class TestRunMetadata:
    def test_instantiation(self):
        rm = RunMetadata(
            algorithm_version="1.0.0",
            execution_timestamp=NOW,
            random_seed=42,
        )
        assert rm.random_seed == 42
        assert rm.classification_weight_version is None

    def test_with_full_reproducibility_snapshot(self):
        rm = RunMetadata(
            algorithm_version="1.0.0",
            execution_timestamp=NOW,
            profile_hash="hash123",
            input_file_hash="hash456",
            python_version="3.11.0",
            dependency_versions={"pydantic": "2.0.0"},
            classification_weight_hash="whash789",
            adaptive_constants_snapshot={"noise_floor_c": 0.05},
            classifier_configuration_snapshot={"aggregation_method": "weighted_sum"},
            feature_extraction_configuration={"rolling_window_seconds": 30.0},
            random_seed=42,
            calibration_dataset_version="v1.0.0",
        )
        assert rm.adaptive_constants_snapshot["noise_floor_c"] == 0.05


# ---------------------------------------------------------------------------
# File metadata
# ---------------------------------------------------------------------------

class TestFileMetadata:
    def test_instantiation(self):
        fm = FileMetadata(source_file_path="test.csv")
        assert fm.detected_encoding == "utf-8"
        assert fm.detected_temperature_unit == "C"


# ---------------------------------------------------------------------------
# Pipeline state
# ---------------------------------------------------------------------------

class TestPipelineStageState:
    def test_instantiation(self):
        pss = PipelineStageState(
            current_stage=PipelineStage.INGESTION,
            started_at=NOW,
        )
        assert pss.status == PipelineStatus.RUNNING
        assert pss.completed_at is None


# ---------------------------------------------------------------------------
# AnalysisContext
# ---------------------------------------------------------------------------

class TestAnalysisContext:
    def test_instantiation_minimal(self):
        ctx = AnalysisContext(
            request=AnalysisRequest(
                file_path="test.csv",
                profile_path="profile.yaml",
            ),
        )
        assert ctx.raw_trace is None
        assert ctx.quality_blocked is False
        assert ctx.raw_trace_released is False

    def test_memory_release_raw_trace(self):
        ctx = AnalysisContext(
            request=AnalysisRequest(
                file_path="test.csv",
                profile_path="profile.yaml",
            ),
            raw_trace=RawTraceData(columns=["t"], data=[], row_count=0),
        )
        assert ctx.raw_trace is not None
        ctx.release_raw_trace()
        assert ctx.raw_trace is None
        assert ctx.raw_trace_released is True

    def test_memory_release_preprocessed_trace(self):
        ctx = AnalysisContext(
            request=AnalysisRequest(
                file_path="test.csv",
                profile_path="profile.yaml",
            ),
            preprocessed_trace=PreprocessedTrace(),
        )
        ctx.release_preprocessed_trace()
        assert ctx.preprocessed_trace is None
        assert ctx.preprocessed_trace_released is True

    def test_memory_release_classified_trace(self):
        ctx = AnalysisContext(
            request=AnalysisRequest(
                file_path="test.csv",
                profile_path="profile.yaml",
            ),
            classified_trace=ClassifiedTrace(),
        )
        ctx.release_classified_trace()
        assert ctx.classified_trace is None
        assert ctx.classified_trace_released is True

    def test_audit_log_default(self):
        ctx = AnalysisContext(
            request=AnalysisRequest(
                file_path="test.csv",
                profile_path="profile.yaml",
            ),
        )
        assert isinstance(ctx.audit_log, AuditLog)
        assert len(ctx.audit_log.entries) == 0


# ---------------------------------------------------------------------------
# AnalysisResult
# ---------------------------------------------------------------------------

class TestAnalysisResult:
    def test_status_accepts_only_overall_status_enum(self):
        ar = AnalysisResult(
            status=OverallValidationStatusEnum.PASS,
            status_reason="All requirements met",
        )
        assert ar.status == OverallValidationStatusEnum.PASS

    def test_bare_warning_rejected(self):
        with pytest.raises(PydanticValidationError):
            AnalysisResult(status="WARNING")
