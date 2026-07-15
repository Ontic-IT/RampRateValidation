"""Core domain models for the Ramp Rate Validation Tool.

All models use Pydantic v2 BaseModel for validation and serialization.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, ConfigDict

from config.constants import (
    AuditCategory,
    AuditSeverity,
    ConfidenceLevel,
    CycleStatus,
    DataQualityStatus,
    OverallValidationStatus,
    PipelineStage,
    PipelineStatus,
    RegionType,
    SetpointResolutionMode,
    ValidationStatus,
    Comparator,
)


# ---------------------------------------------------------------------------
# Evidence models
# ---------------------------------------------------------------------------

class ClassificationEvidence(BaseModel):
    """Base evidence contract for region classification."""
    evidence_type: RegionType
    score: float = Field(ge=0.0, le=1.0)
    reason: str
    evidence: dict[str, float]
    classifier_name: str
    timestamp: datetime


class RampEvidence(ClassificationEvidence):
    """Evidence from ramp classifiers (heating/cooling)."""
    model_config = ConfigDict(extra="forbid")

    evidence: dict[str, float] = Field(
        description="Expected keys: sustained_positive_slope, sustained_negative_slope, "
        "dwell_departure_confidence, dwell_arrival_confidence, monotonicity_score, "
        "slope_magnitude, overshoot_characteristics, stability_score, duration_adequacy, "
        "reversal_tolerance, boundary_stability_score, transition_persistence_score, "
        "arrival_certainty_score, departure_certainty_score, local_gap_resilience_score"
    )


class DwellEvidence(ClassificationEvidence):
    """Evidence from dwell classifiers (hot/cold)."""
    model_config = ConfigDict(extra="forbid")

    evidence: dict[str, float] = Field(
        description="Expected keys: temperature_stability, setpoint_proximity, "
        "duration_adequacy, tolerance_band_persistence, cluster_membership, "
        "boundary_confidence, variance_score, directionality_absence"
    )


class OvershootEvidence(ClassificationEvidence):
    """Evidence from overshoot classifiers."""
    model_config = ConfigDict(extra="forbid")

    evidence: dict[str, float] = Field(
        description="Expected keys: peak_magnitude, overshoot_direction, "
        "return_confidence, duration_adequacy, boundary_confidence, "
        "oscillation_pattern, settling_characteristics"
    )


class CorrectionEvidence(ClassificationEvidence):
    """Evidence from correction classifiers."""
    model_config = ConfigDict(extra="forbid")

    evidence: dict[str, float] = Field(
        description="Expected keys: return_direction, oscillation_count, "
        "amplitude_decay, settling_confidence, boundary_confidence, duration_adequacy"
    )


class TransientEvidence(ClassificationEvidence):
    """Evidence from transient classifier."""
    model_config = ConfigDict(extra="forbid")

    evidence: dict[str, float] = Field(
        description="Expected keys: short_duration, low_slope_magnitude, "
        "high_variance, boundary_ambiguity, lack_of_pattern"
    )


class UnknownEvidence(ClassificationEvidence):
    """Evidence from unknown classifier."""
    model_config = ConfigDict(extra="forbid")

    evidence: dict[str, float] = Field(
        description="Expected keys: low_all_scores, conflicting_signals, "
        "data_quality_issues, outlier_characteristics"
    )


# ---------------------------------------------------------------------------
# Region and classification models
# ---------------------------------------------------------------------------

class Region(BaseModel):
    """Classification object with evidence-based extensions."""
    region_id: str
    start_row: int
    end_row: int
    start_time: datetime
    end_time: datetime
    duration_seconds: float

    # Evidence-based classification fields
    primary_classification: RegionType
    secondary_classifications: list[RegionType] = Field(default_factory=list)
    classification_scores: dict[RegionType, float] = Field(default_factory=dict)
    classification_margin: float = 0.0
    classification_evidence: list[ClassificationEvidence] = Field(default_factory=list)
    classification_confidence: float = 0.0
    classification_confidence_level: ConfidenceLevel = ConfidenceLevel.LOW
    classification_reason: str = ""

    # Ambiguity handling
    is_ambiguous: bool = False
    ambiguity_reason: str | None = None

    # Boundary confidence model (adaptive transition zones)
    boundary_start_confidence: float = 0.0
    boundary_end_confidence: float = 0.0
    boundary_zone_start_row: int = 0
    boundary_zone_end_row: int = 0
    boundary_certainty: ConfidenceLevel = ConfidenceLevel.LOW

    # Existing fields preserved
    source_rows: list[int] = Field(default_factory=list)
    adaptive_thresholds_used: dict[str, float] = Field(default_factory=dict)
    audit_references: list[str] = Field(default_factory=list)
    included_in_primary_validation: bool = True
    exclusion_reason: str | None = None


class RegionList(BaseModel):
    """Container for all classified regions."""
    regions: list[Region] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Trace models
# ---------------------------------------------------------------------------

class CanonicalTraceRow(BaseModel):
    """Single row of the canonical trace — all 18 fields required."""
    timestamp: datetime
    elapsed_seconds: float
    elapsed_minutes: float
    temperature_c_raw: float = Field(frozen=True)
    temperature_c_analysis_signal: float | None = None
    setpoint_c: float | None = None
    channel: str
    source_row: int
    source_file: str
    sample_interval_seconds: float
    local_slope_c_per_min: float | None = None
    rolling_slope_c_per_min: float | None = None
    rolling_temperature_median: float | None = None
    rolling_temperature_MAD: float | None = None
    second_derivative: float | None = None
    direction_of_travel: str | None = None
    data_quality_flags: list[str] = Field(default_factory=list)
    region_id: str | None = None
    classification_label: str | None = None
    auxiliary_channels: dict[str, float | str] = Field(default_factory=dict)


class AuxiliaryChannelMetadata(BaseModel):
    """Metadata for an auxiliary channel in the trace."""
    channel_name: str
    unit: str | None = None
    data_type: Literal['NUMERIC', 'STATE'] = 'NUMERIC'
    source_column_index: int
    used_in_root_cause: bool = False


class CanonicalTrace(BaseModel):
    """Complete canonical trace data."""
    rows: list[CanonicalTraceRow] = Field(default_factory=list)


class RawTraceData(BaseModel):
    """Raw trace data preserving ALL columns from source file."""
    columns: list[str] = Field(default_factory=list)
    data: list[dict[str, Any]] = Field(default_factory=list)
    row_count: int = 0


class PreprocessedTrace(BaseModel):
    """Trace after preprocessing — analysis signal populated."""
    rows: list[CanonicalTraceRow] = Field(default_factory=list)


class ClassifiedTrace(BaseModel):
    """Trace after classification — region_id and classification_label populated."""
    rows: list[CanonicalTraceRow] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Data quality models
# ---------------------------------------------------------------------------

class PreprocessingReport(BaseModel):
    """Preprocessing outputs and statistics."""
    estimated_sample_interval_s: float = 0.0
    noise_floor_c: float = 0.0
    slope_noise_floor_c_per_min: float = 0.0
    temperature_MAD_baseline: float = 0.0
    rolling_window_seconds_used: float = 0.0
    detected_spikes: list[int] = Field(default_factory=list)
    detected_gaps: list[tuple[int, int]] = Field(default_factory=list)
    duplicate_timestamps: list[int] = Field(default_factory=list)
    out_of_order_rows: list[int] = Field(default_factory=list)

    # Gap density metrics for adaptive impact assessment
    gap_density_score: float = 0.0
    dropout_density_score: float = 0.0
    irregular_sampling_score: float = 0.0
    effective_data_continuity_score: float = 1.0


class RunDataQualityReport(BaseModel):
    """Run-level data quality assessment."""
    overall_status: DataQualityStatus = DataQualityStatus.ACCEPTABLE
    missing_data_pct: float = 0.0
    duplicate_timestamp_pct: float = 0.0
    out_of_order_row_count: int = 0
    irregular_interval_pct: float = 0.0
    large_gap_count: int = 0
    spike_count: int = 0
    dropout_count: int = 0
    process_duration_seconds: float = 0.0
    minimum_ramp_data_available: bool = True
    quality_impact_notes: str = ""

    # Gap density metrics for adaptive impact assessment
    gap_density_score: float = 0.0
    dropout_density_score: float = 0.0
    irregular_sampling_score: float = 0.0
    effective_data_continuity_score: float = 1.0


class RegionDataQualityStatus(BaseModel):
    """Per-region data quality status."""
    region_id: str
    status: DataQualityStatus = DataQualityStatus.ACCEPTABLE
    spike_count_in_region: int = 0
    gap_seconds_in_region: float = 0.0
    data_completeness_pct: float = 100.0
    quality_notes: str = ""


class RampDataQualityStatus(BaseModel):
    """Per-ramp data quality status."""
    valid_ramp_region_id: str
    status: DataQualityStatus = DataQualityStatus.ACCEPTABLE
    spike_count: int = 0
    gap_seconds: float = 0.0
    data_completeness_pct: float = 100.0
    sufficient_for_compliance: bool = True


class ValidationDataQualityImpact(BaseModel):
    """Impact of data quality on validation decisions."""
    blocks_pass_fail: bool = False
    affected_requirement_ids: list[str] = Field(default_factory=list)
    reason: str = ""


# ---------------------------------------------------------------------------
# Boundary and setpoint models
# ---------------------------------------------------------------------------

class ProcessBoundaries(BaseModel):
    """Detected process boundaries."""
    ambient_start_index: int = 0
    ambient_end_index: int = 0
    process_start_index: int = 0
    process_end_index: int = 0
    commissioned_loop_start_index: int = 0
    commissioned_loop_end_index: int = 0
    recovery_start_index: int | None = None
    partial_cycle_edge_indices: list[int] = Field(default_factory=list)
    usable_window_row_range: tuple[int, int] = (0, 0)
    detection_method: str = ""


class ResolvedSetpoints(BaseModel):
    """Resolved setpoint information."""
    resolution_mode: SetpointResolutionMode = SetpointResolutionMode.MODE_B
    inferred_ambient_c: float = 0.0
    inferred_hot_setpoint_c: float = 0.0
    inferred_cold_setpoint_c: float = 0.0
    setpoint_confidence_scores: dict[str, float] = Field(default_factory=dict)
    dwell_MAD_values: dict[str, float] = Field(default_factory=dict)
    cluster_separation_threshold_used: float = 0.0
    setpoint_plateaus: list[dict] | None = None
    algorithm_seed_used: int = 42


# ---------------------------------------------------------------------------
# Compliance models
# ---------------------------------------------------------------------------

class ValidRampRegion(BaseModel):
    """Compliance object — valid ramp region for metrics and validation."""
    region_id: str
    source_region: Any = None  # Reference to source Region
    direction: Any = None  # RampDirection enum
    start_row: int = 0
    end_row: int = 0
    start_time: datetime | None = None
    end_time: datetime | None = None
    duration_seconds: float = 0.0
    start_temperature_c: float = 0.0
    end_temperature_c: float = 0.0
    temperature_delta_c: float = 0.0
    included_rows: list[int] = Field(default_factory=list)
    excluded_rows: list[int] = Field(default_factory=list)
    exclusion_reasons: list[str] = Field(default_factory=list)
    monotonicity_score: float = 0.0
    reversal_count: int = 0
    stall_duration_seconds: float = 0.0
    departure_confidence: float = 0.0
    arrival_confidence: float = 0.0
    data_quality_flags: list[str] = Field(default_factory=list)


class ValidRampRegionList(BaseModel):
    """Container for all valid ramp regions."""
    regions: list[ValidRampRegion] = Field(default_factory=list)


class Cycle(BaseModel):
    """Single cycle definition."""
    cycle_id: str
    cycle_number: int = 0
    start_row: int = 0
    end_row: int = 0
    start_time: datetime | None = None
    end_time: datetime | None = None
    duration_seconds: float = 0.0
    region_ids: list[str] = Field(default_factory=list)
    regions: list[Any] = Field(default_factory=list)  # List of Region objects
    valid_ramps: list[Any] = Field(default_factory=list)  # List of ValidRampRegion
    heating_ramp_count: int = 0
    cooling_ramp_count: int = 0
    hot_dwell_count: int = 0
    cold_dwell_count: int = 0
    status: CycleStatus = CycleStatus.COMPLETE
    completeness_reason: str = ""
    is_complete: bool = True
    cycle_to_cycle_drift: float | None = None


class CycleList(BaseModel):
    """Container for all detected cycles."""
    cycles: list[Cycle] = Field(default_factory=list)


class RampMetrics(BaseModel):
    """Computed metrics for a valid ramp region."""
    region_id: str
    robust_slope_c_per_min: float = 0.0
    minimum_sustained_slope_c_per_min: float = 0.0
    median_rolling_slope_c_per_min: float = 0.0
    endpoint_slope_c_per_min: float = 0.0
    slope_MAD: float | None = None
    jitter_score: float | None = None
    taper_score: float | None = None
    linearity_score: float | None = None
    stall_duration_seconds: float = 0.0
    reversal_count: int = 0
    monotonicity_score: float = 0.0
    slope_calculation_method: str = "theil_sen"
    sustained_window_seconds_used: float = 60.0


class DwellMetrics(BaseModel):
    """Computed metrics for a dwell region."""
    region_id: str
    target_setpoint_c: float | None = None
    mean_temperature_c: float | None = None
    temperature_std_c: float | None = None
    temperature_range_c: float | None = None
    setpoint_deviation_c: float | None = None
    time_inside_tolerance_band_seconds: float = 0.0
    time_inside_tolerance_band_pct: float = 0.0
    overshoot_magnitude_c: float | None = None
    overshoot_duration_seconds: float | None = None
    settling_time_seconds: float | None = None
    oscillation_count: int = 0
    stability_score: float | None = None


class CycleMetrics(BaseModel):
    """Aggregate metrics for a cycle."""
    cycle_id: str
    duration_seconds: float = 0.0
    average_heating_slope_c_per_min: float = 0.0
    average_cooling_slope_c_per_min: float = 0.0
    minimum_heating_slope_c_per_min: float = 0.0
    maximum_cooling_slope_c_per_min: float = 0.0
    average_jitter_score: float = 0.0
    average_taper_score: float = 0.0
    average_dwell_stability: float = 0.0
    maximum_overshoot_c: float = 0.0
    total_ramp_time_seconds: float = 0.0
    total_dwell_time_seconds: float = 0.0
    cycle_to_cycle_drift: float | None = None
    heating_ramp_count: int = 0
    cooling_ramp_count: int = 0


class MetricSet(BaseModel):
    """Container for all computed metrics — populated during Phase 5."""
    ramp_metrics: list[RampMetrics] = Field(default_factory=list)
    dwell_metrics: list[DwellMetrics] = Field(default_factory=list)
    cycle_metrics: list[CycleMetrics] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Validation models
# ---------------------------------------------------------------------------

class ValidationResult(BaseModel):
    """Single validation result evidence contract."""
    validation_result_id: str
    requirement_id: str
    requirement_description: str
    measured_value: float
    threshold_value: float
    comparator: Comparator
    unit: str
    method: str
    region_id: str | None = None
    cycle_id: str | None = None
    included_rows: int = 0
    excluded_regions: list[str] = Field(default_factory=list)
    result: ValidationStatus = ValidationStatus.NOT_APPLICABLE
    reason: str = ""
    audit_references: list[str] = Field(default_factory=list)


class ValidationResults(BaseModel):
    """Container for all validation results."""
    results: list[ValidationResult] = Field(default_factory=list)


class OverallStatus(BaseModel):
    """Overall validation status aggregation."""
    status: OverallValidationStatus
    reason: str = ""


# ---------------------------------------------------------------------------
# Output models
# ---------------------------------------------------------------------------

class ProfileComparisonResults(BaseModel):
    """Profile consistency analysis results — populated during Phase 7."""
    comparisons: dict[str, Any] = Field(default_factory=dict)


class VisualisationBundle(BaseModel):
    """Visualisation payload — populated during Phase 7."""
    charts: dict[str, Any] = Field(default_factory=dict)


class ReportPackage(BaseModel):
    """Report payload — populated during Phase 7."""
    sections: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Audit models
# ---------------------------------------------------------------------------

class AuditEntry(BaseModel):
    """Single audit trail entry."""
    timestamp: datetime
    module_name: str
    action: str
    input_reference: str = ""
    output_reference: str = ""
    decision: str = ""
    reason: str = ""
    thresholds_used: dict[str, float] = Field(default_factory=dict)
    rows_used: int = 0
    rows_excluded: int = 0
    algorithm_version: str = ""
    severity: AuditSeverity = AuditSeverity.INFO
    category: AuditCategory = AuditCategory.PIPELINE
    # Weight auditability (for classification entries)
    classifier_version: str | None = None
    weight_version: str | None = None
    aggregation_method: str | None = None


class AuditLog(BaseModel):
    """Complete audit trail for a run."""
    entries: list[AuditEntry] = Field(default_factory=list)

    def add(self, entry: AuditEntry) -> None:
        """Append an audit entry to the log."""
        self.entries.append(entry)


# ---------------------------------------------------------------------------
# Reproducibility models
# ---------------------------------------------------------------------------

class ClassificationWeightVersion(BaseModel):
    """Weight governance contract."""
    version: str
    weight_file_hash: str
    created_date: datetime
    calibration_dataset_version: str


class RunMetadata(BaseModel):
    """Reproducibility contract — captures all configuration for replay."""
    git_commit: str | None = None
    algorithm_version: str = ""
    profile_hash: str = ""
    input_file_hash: str = ""
    execution_timestamp: datetime
    python_version: str = ""
    dependency_versions: dict[str, str] = Field(default_factory=dict)
    classification_weight_version: ClassificationWeightVersion | None = None
    classification_weight_hash: str = ""
    adaptive_constants_snapshot: dict[str, float] = Field(default_factory=dict)
    classifier_configuration_snapshot: dict[str, Any] = Field(default_factory=dict)
    feature_extraction_configuration: dict[str, Any] = Field(default_factory=dict)
    random_seed: int = 42
    calibration_dataset_version: str = ""


# ---------------------------------------------------------------------------
# File metadata
# ---------------------------------------------------------------------------

class FileMetadata(BaseModel):
    """Detected file metadata from ingestion."""
    source_file_path: str
    detected_delimiter: str = ""
    detected_header_rows: int = 0
    header_row_index: int = 0  # Actual index of header row (may vary by file)
    detected_preamble_line_count: int = 0  # Lines before header
    detected_encoding: str = "utf-8"
    detected_timestamp_format: str = ""
    detected_temperature_unit: str = "C"
    available_channels: list[str] = Field(default_factory=list)
    selected_temperature_channel: str = ""
    selected_setpoint_channel: str | None = None
    raw_row_count: int = 0
    usable_row_count: int = 0
    auxiliary_channel_count: int = 0  # Number of auxiliary channels detected


# ---------------------------------------------------------------------------
# Request model
# ---------------------------------------------------------------------------

class AnalysisRequest(BaseModel):
    """Input request for pipeline execution."""
    file_path: str
    profile_path: str
    output_dir: str = "."
    channel: str | None = None
    setpoint_channel: str | None = None
    algorithm_version: str | None = None


# ---------------------------------------------------------------------------
# Pipeline state
# ---------------------------------------------------------------------------

class PipelineStageState(BaseModel):
    """State tracker for a single pipeline stage."""
    current_stage: PipelineStage
    started_at: datetime
    completed_at: datetime | None = None
    status: PipelineStatus = PipelineStatus.RUNNING


# ---------------------------------------------------------------------------
# AnalysisContext — central pipeline state container
# ---------------------------------------------------------------------------

class AnalysisContext(BaseModel):
    """Central shared object passed through all pipeline stages.

    Append-only contract: modules write to their designated fields.
    Prevents pipeline coupling, enables restart capability.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    request: AnalysisRequest
    profile: Any = None  # ValidationProfile — set after profile loading
    file_metadata: FileMetadata | None = None
    raw_trace: RawTraceData | None = None
    canonical_trace: CanonicalTrace | None = None
    preprocessed_trace: PreprocessedTrace | None = None
    preprocessing_report: PreprocessingReport | None = None
    data_quality_report: RunDataQualityReport | None = None
    region_quality_statuses: dict[str, RegionDataQualityStatus] = Field(default_factory=dict)
    ramp_quality_statuses: dict[str, RampDataQualityStatus] = Field(default_factory=dict)
    validation_quality_impact: ValidationDataQualityImpact | None = None
    process_boundaries: ProcessBoundaries | None = None
    resolved_setpoints: ResolvedSetpoints | None = None
    region_list: RegionList | None = None
    classified_trace: ClassifiedTrace | None = None
    valid_ramp_regions: ValidRampRegionList | None = None
    cycles: CycleList | None = None
    metric_set: MetricSet | None = None
    validation_results: ValidationResults | None = None
    overall_validation_status: OverallStatus | None = None
    phase_conformance: PhaseConformanceSummary | None = None
    profile_comparison_results: ProfileComparisonResults | None = None
    visualisation_bundle: VisualisationBundle | None = None
    report_package: ReportPackage | None = None
    audit_log: AuditLog = Field(default_factory=AuditLog)
    run_metadata: RunMetadata | None = None

    # Pipeline state tracking
    pipeline_stages: list[PipelineStageState] = Field(default_factory=list)
    quality_blocked: bool = False
    
    # Adaptive constants (derived from preprocessing)
    adaptive_constants: Any = None

    # Memory management — explicit release flags
    raw_trace_released: bool = False
    canonical_trace_slimmed: bool = False
    preprocessed_trace_released: bool = False
    classified_trace_released: bool = False

    def release_raw_trace(self) -> None:
        """Release RawTraceData after CanonicalTrace is validated (Phase 2)."""
        self.raw_trace = None
        self.raw_trace_released = True

    def release_preprocessed_trace(self) -> None:
        """Release PreprocessedTrace after classification (Phase 4)."""
        self.preprocessed_trace = None
        self.preprocessed_trace_released = True

    def release_classified_trace(self) -> None:
        """Release ClassifiedTrace after report generation (Phase 7)."""
        self.classified_trace = None
        self.classified_trace_released = True


# ---------------------------------------------------------------------------
# Phase conformance summary
# ---------------------------------------------------------------------------

class PhaseConformanceSummary(BaseModel):
    """Per-phase conformance summary computed from ValidationResults.

    A "phase" = one Region or one ValidRampRegion-or-dwell-equivalent,
    matching the reference report's per-phase row structure.
    """
    total_phases: int = 0
    passed_phases: int = 0
    failed_phases: int = 0
    anomaly_phases: int = 0
    conformance_percentage: float = 0.0
    anomaly_phase_ids: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# AnalysisResult — final output
# ---------------------------------------------------------------------------

class AnalysisResult(BaseModel):
    """Final output of the pipeline."""
    status: OverallValidationStatus
    status_reason: str = ""
    canonical_trace: CanonicalTrace | None = None
    data_quality_report: RunDataQualityReport | None = None
    validation_profile_used: Any = None  # ValidationProfile
    inferred_setpoints: ResolvedSetpoints | None = None
    process_boundaries: ProcessBoundaries | None = None
    classified_regions: RegionList | None = None
    valid_ramp_regions: ValidRampRegionList | None = None
    cycles: CycleList | None = None
    metrics: MetricSet | None = None
    validation_results: ValidationResults | None = None
    profile_comparison_results: ProfileComparisonResults | None = None
    visualisation: VisualisationBundle | None = None
    report_package: ReportPackage | None = None
    audit_log: AuditLog | None = None
    phase_conformance: PhaseConformanceSummary | None = None
