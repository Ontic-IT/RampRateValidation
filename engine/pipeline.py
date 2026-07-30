"""Main pipeline orchestration for ramp rate analysis (Phase 8).

Implements strict 15-step execution order with state machine and restart capability.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from config.constants import AuditCategory, AuditSeverity, PipelineStage, PipelineStatus
from models.domain import (
    AnalysisContext,
    AnalysisRequest,
    AnalysisResult,
    AuditEntry,
    PipelineStageState,
)


def run_ramp_rate_analysis(request: AnalysisRequest) -> AnalysisResult:
    """Main entry point for ramp rate analysis pipeline.
    
    Executes all 15 pipeline stages in strict order with state tracking.
    
    Args:
        request: Analysis request with file path, profile, and options
    
    Returns:
        AnalysisResult with complete analysis outputs
    """
    # Initialize context
    context = AnalysisContext(request=request)
    
    # Execute pipeline stages in order
    try:
        context = _execute_pipeline(context)
    except Exception as e:
        context.audit_log.add(AuditEntry(
            timestamp=datetime.now(),
            module_name="pipeline",
            action="run_ramp_rate_analysis",
            decision="CRITICAL_FAILURE",
            reason=f"Pipeline execution failed: {str(e)}",
            severity=AuditSeverity.CRITICAL,
            category=AuditCategory.PIPELINE,
        ))
        raise
    
    # Build final AnalysisResult
    result = _build_analysis_result(context)
    
    return result


def _execute_pipeline(context: AnalysisContext) -> AnalysisContext:
    """Execute all 15 pipeline stages in strict order.
    
    Args:
        context: Analysis context
    
    Returns:
        Updated context after all stages
    """
    # Stage 1: Ingest source file
    context = _execute_stage(
        context,
        PipelineStage.INGESTION,
        _stage_ingest_source_file,
    )
    
    # Stage 2: Normalize to canonical trace
    context = _execute_stage(
        context,
        PipelineStage.NORMALISATION,
        _stage_normalise_to_canonical_trace,
    )
    
    # Check ingest validity gate (set during normalisation)
    if context.ingest_blocked:
        reasons = "; ".join(context.ingest_validity.flags) if context.ingest_validity else ""
        context.audit_log.add(AuditEntry(
            timestamp=datetime.now(),
            module_name="pipeline",
            action="ingest_validity_gate",
            decision="BLOCKED",
            reason=f"Trace is not an analysable ramp-rate test - pipeline stopped: {reasons}",
            severity=AuditSeverity.ERROR,
            category=AuditCategory.PIPELINE,
        ))
        from models.domain import OverallStatus
        from config.constants import OverallValidationStatus
        context.overall_validation_status = OverallStatus(
            status=OverallValidationStatus.INVALID_INPUT,
            reason=f"Ingest validity gate: {reasons}",
        )
        return context

    # Stage 3: Preprocess trace
    context = _execute_stage(
        context,
        PipelineStage.PREPROCESSING,
        _stage_preprocess_trace,
    )
    
    # Stage 4: Assess data quality [GATE]
    context = _execute_stage(
        context,
        PipelineStage.QUALITY,
        _stage_assess_data_quality,
    )
    
    # Check quality gate
    if context.quality_blocked:
        context.audit_log.add(AuditEntry(
            timestamp=datetime.now(),
            module_name="pipeline",
            action="quality_gate",
            decision="BLOCKED",
            reason="Data quality INVALID - pipeline stopped",
            severity=AuditSeverity.ERROR,
            category=AuditCategory.PIPELINE,
        ))
        return context
    
    # Stage 5: Detect process boundaries
    context = _execute_stage(
        context,
        PipelineStage.BOUNDARIES,
        _stage_detect_process_boundaries,
    )
    
    # Stage 6: Infer or resolve setpoints
    context = _execute_stage(
        context,
        PipelineStage.SETPOINTS,
        _stage_infer_or_resolve_setpoints,
    )
    
    # Stage 7: Classify regions
    context = _execute_stage(
        context,
        PipelineStage.CLASSIFICATION,
        _stage_classify_regions,
    )
    
    # Stage 8: Isolate valid ramp regions
    context = _execute_stage(
        context,
        PipelineStage.RAMP_ISOLATION,
        _stage_isolate_valid_ramp_regions,
    )
    
    # Stage 9: Segment cycles
    context = _execute_stage(
        context,
        PipelineStage.CYCLES,
        _stage_segment_cycles,
    )
    
    # Stage 10: Compute metrics
    context = _execute_stage(
        context,
        PipelineStage.METRICS,
        _stage_compute_metrics,
    )
    
    # Stage 11: Validate against profile
    context = _execute_stage(
        context,
        PipelineStage.VALIDATION,
        _stage_validate_against_profile,
    )
    
    # Stage 12: Compare profile consistency
    context = _execute_stage(
        context,
        PipelineStage.COMPARISON,
        _stage_compare_profile_consistency,
    )
    
    # Stage 13: Build visualisation payload
    context = _execute_stage(
        context,
        PipelineStage.VISUALISATION,
        _stage_build_visualisation_payload,
    )
    
    # Stage 14: Generate report payload
    context = _execute_stage(
        context,
        PipelineStage.REPORTING,
        _stage_generate_report_payload,
    )
    
    # Stage 15: Emit analysis result
    context = _execute_stage(
        context,
        PipelineStage.AUDIT,
        _stage_emit_analysis_result,
    )
    
    return context


def _execute_stage(
    context: AnalysisContext,
    stage: PipelineStage,
    stage_func: Any,
) -> AnalysisContext:
    """Execute a single pipeline stage with state tracking.
    
    Args:
        context: Analysis context
        stage: Pipeline stage enum
        stage_func: Function to execute for this stage
    
    Returns:
        Updated context
    """
    # Create stage state
    stage_state = PipelineStageState(
        current_stage=stage,
        started_at=datetime.now(),
        status=PipelineStatus.RUNNING,
    )
    
    context.audit_log.add(AuditEntry(
        timestamp=datetime.now(),
        module_name="pipeline",
        action=f"stage_{stage.value}",
        decision="STARTED",
        reason=f"Starting stage: {stage.value}",
        severity=AuditSeverity.INFO,
        category=AuditCategory.PIPELINE,
    ))
    
    try:
        # Execute stage function
        context = stage_func(context)
        
        # Mark stage complete
        stage_state.completed_at = datetime.now()
        stage_state.status = PipelineStatus.COMPLETED
        
        context.audit_log.add(AuditEntry(
            timestamp=datetime.now(),
            module_name="pipeline",
            action=f"stage_{stage.value}",
            decision="COMPLETED",
            reason=f"Stage completed: {stage.value}",
            severity=AuditSeverity.INFO,
            category=AuditCategory.PIPELINE,
        ))
        
    except Exception as e:
        # Mark stage failed
        stage_state.completed_at = datetime.now()
        stage_state.status = PipelineStatus.FAILED
        
        context.audit_log.add(AuditEntry(
            timestamp=datetime.now(),
            module_name="pipeline",
            action=f"stage_{stage.value}",
            decision="FAILED",
            reason=f"Stage failed: {str(e)}",
            severity=AuditSeverity.ERROR,
            category=AuditCategory.PIPELINE,
        ))
        
        raise
    
    finally:
        # Append stage state to context
        context.pipeline_stages.append(stage_state)
    
    return context


# ---------------------------------------------------------------------------
# Stage implementations
# ---------------------------------------------------------------------------

def _stage_ingest_source_file(context: AnalysisContext) -> AnalysisContext:
    """Stage 1: Ingest source file."""
    from inputs.file_loader import load_trace_file
    
    raw_trace, file_metadata = load_trace_file(
        file_path=context.request.file_path,
        channel=context.request.channel,
        setpoint_channel=context.request.setpoint_channel,
        audit_log=context.audit_log,
    )
    
    context.raw_trace = raw_trace
    context.file_metadata = file_metadata
    
    return context


def _stage_normalise_to_canonical_trace(context: AnalysisContext) -> AnalysisContext:
    """Stage 2: Normalize to canonical trace, then gate on ingest validity."""
    from prereq.normalisation.trace_builder import build_canonical_trace
    from prereq.normalisation.validity_gate import evaluate_ingest_validity

    canonical_trace = build_canonical_trace(
        raw_trace=context.raw_trace,
        file_metadata=context.file_metadata,
        audit_log=context.audit_log,
    )

    context.canonical_trace = canonical_trace

    # Ingest validity gate: fragments and non-ramp captures (e.g. Vib2
    # vibration screens) are flagged/blocked at the door, with reasons.
    context.ingest_validity = evaluate_ingest_validity(
        canonical_trace=canonical_trace,
        file_metadata=context.file_metadata,
        audit_log=context.audit_log,
    )
    context.ingest_blocked = context.ingest_validity.verdict == "INVALID"

    return context


def _stage_preprocess_trace(context: AnalysisContext) -> AnalysisContext:
    """Stage 3: Preprocess trace."""
    from prereq.preprocessing.signal_processor import preprocess_trace
    
    preprocessed_trace, preprocessing_report = preprocess_trace(
        canonical_trace=context.canonical_trace,
        audit_log=context.audit_log,
    )
    
    context.preprocessed_trace = preprocessed_trace
    context.preprocessing_report = preprocessing_report
    
    return context


def _stage_assess_data_quality(context: AnalysisContext) -> AnalysisContext:
    """Stage 4: Assess data quality [GATE]."""
    from cleaners.quality.quality_gating import assess_data_quality
    
    # Calculate process duration from preprocessed trace
    process_duration = 0.0
    total_rows = 0
    if context.preprocessed_trace and context.preprocessed_trace.rows:
        total_rows = len(context.preprocessed_trace.rows)
        if total_rows > 1:
            first_row = context.preprocessed_trace.rows[0]
            last_row = context.preprocessed_trace.rows[-1]
            process_duration = last_row.elapsed_seconds - first_row.elapsed_seconds
    
    quality_report = assess_data_quality(
        preprocessing_report=context.preprocessing_report,
        total_rows=total_rows,
        process_duration_seconds=process_duration,
        max_spike_count=200,  # Increase spike tolerance for real-world data
        min_process_duration_seconds=10.0,  # Lower minimum duration
        audit_log=context.audit_log,
    )
    
    context.data_quality_report = quality_report
    
    # Only block on INVALID, allow WARNING/INCONCLUSIVE to continue
    from config.constants import DataQualityStatus
    if quality_report.overall_status == DataQualityStatus.INVALID:
        # Check if it's only spike count causing INVALID
        if "Spike count" in quality_report.quality_impact_notes and "Process duration" not in quality_report.quality_impact_notes:
            # Don't block for spikes alone - they're handled in preprocessing
            context.quality_blocked = False
        else:
            context.quality_blocked = True
    
    return context


def _stage_detect_process_boundaries(context: AnalysisContext) -> AnalysisContext:
    """Stage 5: Detect process boundaries."""
    from workers.boundary_detection.process_boundaries import detect_process_boundaries
    
    boundaries = detect_process_boundaries(
        trace=context.preprocessed_trace,
        preprocessing_report=context.preprocessing_report,
        audit_log=context.audit_log,
    )
    
    context.process_boundaries = boundaries
    
    return context


def _stage_infer_or_resolve_setpoints(context: AnalysisContext) -> AnalysisContext:
    """Stage 6: Infer or resolve setpoints."""
    from workers.setpoint_resolution.setpoint_inference import resolve_setpoints
    
    # Mode A when the file actually carries a setpoint channel — read the
    # commanded levels directly rather than inferring them from measured
    # temperature (which can miscluster on multi-level programmes).
    setpoint_channel_present = bool(
        context.file_metadata and context.file_metadata.selected_setpoint_channel
    )
    setpoints = resolve_setpoints(
        trace=context.preprocessed_trace,
        preprocessing_report=context.preprocessing_report,
        process_boundaries=context.process_boundaries,
        setpoint_channel_present=setpoint_channel_present,
        audit_log=context.audit_log,
    )
    
    context.resolved_setpoints = setpoints
    
    # Derive adaptive constants now that we have setpoints and boundaries
    from engine.adaptive_constants_derivation import derive_adaptive_constants
    context.adaptive_constants = derive_adaptive_constants(
        preprocessing_report=context.preprocessing_report,
        setpoints=setpoints,
        boundaries=context.process_boundaries,
        trace=context.preprocessed_trace,
    )
    
    return context


def _stage_classify_regions(context: AnalysisContext) -> AnalysisContext:
    """Stage 7: Classify regions."""
    from workers.region_classification.classification_orchestrator import classify_regions
    
    # Note: classify_regions returns (RegionList, ClassifiedTrace)
    regions, classified_trace = classify_regions(
        trace=context.preprocessed_trace,
        setpoints=context.resolved_setpoints,
        boundaries=context.process_boundaries,
        audit_log=context.audit_log,
    )
    
    context.classified_trace = classified_trace
    context.region_list = regions
    
    return context


def _stage_isolate_valid_ramp_regions(context: AnalysisContext) -> AnalysisContext:
    """Stage 8: Isolate valid ramp regions."""
    from workers.ramp_isolation.ramp_extractor import isolate_valid_ramps
    
    valid_ramps = isolate_valid_ramps(
        classified_trace=context.classified_trace,
        regions=context.region_list,
        setpoints=context.resolved_setpoints,
        preprocessing_report=context.preprocessing_report,
        audit_log=context.audit_log,
    )
    
    from models.domain import ValidRampRegionList
    context.valid_ramp_regions = ValidRampRegionList(regions=valid_ramps)
    
    return context


def _stage_segment_cycles(context: AnalysisContext) -> AnalysisContext:
    """Stage 9: Segment cycles."""
    from workers.cycle_segmentation.cycle_detector import detect_cycles
    
    cycles = detect_cycles(
        regions=context.region_list,
        valid_ramps=context.valid_ramp_regions.regions,
        setpoints=context.resolved_setpoints,
        audit_log=context.audit_log,
    )
    
    context.cycles = cycles
    
    return context


def _stage_compute_metrics(context: AnalysisContext) -> AnalysisContext:
    """Stage 10: Compute metrics."""
    from cleaners.metrics.ramp_metrics import compute_ramp_metrics
    from cleaners.metrics.dwell_metrics import compute_dwell_metrics
    from cleaners.metrics.cycle_metrics import compute_cycle_metrics
    from models.domain import MetricSet
    from config.constants import RegionType
    
    # Compute ramp metrics
    ramp_metrics = []
    for ramp in context.valid_ramp_regions.regions:
        metrics = compute_ramp_metrics(
            valid_ramp=ramp,
            classified_trace=context.classified_trace,
            audit_log=context.audit_log,
        )
        ramp_metrics.append(metrics)
    
    # Compute dwell metrics
    dwell_metrics = []
    for region in context.region_list.regions:
        if region.primary_classification in (RegionType.HOT_DWELL, RegionType.COLD_DWELL, RegionType.AMBIENT_START):
            metrics = compute_dwell_metrics(
                region=region,
                classified_trace=context.classified_trace,
                setpoints=context.resolved_setpoints,
                audit_log=context.audit_log,
            )
            dwell_metrics.append(metrics)
    
    # Compute cycle metrics
    cycle_metrics = []
    for cycle in context.cycles.cycles:
        metrics = compute_cycle_metrics(
            cycle=cycle,
            ramp_metrics=ramp_metrics,
            dwell_metrics=dwell_metrics,
            audit_log=context.audit_log,
        )
        cycle_metrics.append(metrics)
    
    context.metric_set = MetricSet(
        ramp_metrics=ramp_metrics,
        dwell_metrics=dwell_metrics,
        cycle_metrics=cycle_metrics,
    )
    
    return context


def _stage_validate_against_profile(context: AnalysisContext) -> AnalysisContext:
    """Stage 11: Validate against profile."""
    from engine.validation.validation_engine import validate_analysis
    from inputs.profile_loader import load_profile
    
    # Load profile if not already loaded
    if not context.profile:
        profile = load_profile(context.request.profile_path)
        context.profile = profile
    
    validation_results, overall_status, phase_conformance = validate_analysis(
        profile=context.profile,
        regions=context.region_list,
        valid_ramps=context.valid_ramp_regions.regions if context.valid_ramp_regions else [],
        ramp_metrics=context.metric_set.ramp_metrics if context.metric_set else [],
        dwell_metrics=context.metric_set.dwell_metrics if context.metric_set else [],
        cycles=context.cycles,
        adaptive_constants=context.adaptive_constants,
        audit_log=context.audit_log,
    )
    
    context.validation_results = validation_results
    context.overall_validation_status = overall_status
    context.phase_conformance = phase_conformance
    
    return context


def _stage_compare_profile_consistency(context: AnalysisContext) -> AnalysisContext:
    """Stage 12: Compare profile consistency."""
    from workers.profile_comparison.profile_comparator import (
        compare_cycle_metric_distribution,
        detect_cycle_to_cycle_drift,
        compute_profile_consistency_score,
    )
    from models.domain import ProfileComparisonResults
    
    if not context.metric_set:
        # Skip if no metrics
        return context
    
    metric_distribution = compare_cycle_metric_distribution(
        metric_set=context.metric_set,
        regions=context.region_list,
        audit_log=context.audit_log,
    )
    
    drift = detect_cycle_to_cycle_drift(
        cycle_metrics=context.metric_set.cycle_metrics if context.metric_set else [],
        audit_log=context.audit_log,
    )
    
    consistency_score = compute_profile_consistency_score(
        metric_set=context.metric_set,
        audit_log=context.audit_log,
    )
    
    context.profile_comparison_results = ProfileComparisonResults(
        comparisons={
            "metric_distribution": metric_distribution,
            "drift_detection": drift,
            "consistency_score": consistency_score,
        }
    )
    
    return context


def _stage_build_visualisation_payload(context: AnalysisContext) -> AnalysisContext:
    """Stage 13: Build visualisation payload."""
    from engine.visualisation.chart_builder import build_complete_visualisation
    from models.domain import VisualisationBundle
    
    if not context.classified_trace or not context.region_list or not context.cycles or not context.validation_results:
        # Skip if missing required data
        return context
    
    # Resolved dwell setpoint tolerance — overshoots are flagged on the chart
    # only when they exceed it.
    setpoint_tolerance_c = None
    if context.profile and getattr(context.profile, "tolerance_resolutions", None):
        for res in context.profile.tolerance_resolutions:
            if res.parameter_name == "dwell_setpoint_deviation":
                setpoint_tolerance_c = res.resolved_value
                break

    chart = build_complete_visualisation(
        classified_trace=context.classified_trace,
        regions=context.region_list,
        cycles=context.cycles,
        validation_results=context.validation_results,
        setpoints=context.resolved_setpoints,
        region_colour_map=context.profile.visualisation_settings.region_colour_map if context.profile else None,
        dwell_metrics=context.metric_set.dwell_metrics if context.metric_set else None,
        setpoint_tolerance_c=setpoint_tolerance_c,
        audit_log=context.audit_log,
    )
    
    context.visualisation_bundle = VisualisationBundle(charts={"main": chart})
    
    return context


def _stage_generate_report_payload(context: AnalysisContext) -> AnalysisContext:
    """Stage 14: Generate report payload."""
    from engine.reporting.report_builder import generate_report_payload
    
    report = generate_report_payload(
        context=context,
        audit_log=context.audit_log,
    )
    
    context.report_package = report
    
    return context


def _stage_emit_analysis_result(context: AnalysisContext) -> AnalysisContext:
    """Stage 15: Emit analysis result."""
    # Final audit entry
    context.audit_log.add(AuditEntry(
        timestamp=datetime.now(),
        module_name="pipeline",
        action="emit_analysis_result",
        decision="COMPLETE",
        reason="Pipeline execution complete",
        severity=AuditSeverity.INFO,
        category=AuditCategory.PIPELINE,
    ))
    
    return context


def _build_analysis_result(context: AnalysisContext) -> AnalysisResult:
    """Build final AnalysisResult from context.
    
    Args:
        context: Completed analysis context
    
    Returns:
        AnalysisResult with all outputs
    """
    # Use overall_validation_status or create default
    from config.constants import OverallValidationStatus
    if context.overall_validation_status:
        status = context.overall_validation_status.status
        status_reason = context.overall_validation_status.reason
    else:
        status = OverallValidationStatus.INCONCLUSIVE
        status_reason = "Pipeline completed but no validation status available"

    # An ingest-validity flag must be visible in the headline status: a
    # "PASS" on a trace the gate says is not a ramp test is a misfire.
    if context.ingest_validity and context.ingest_validity.verdict == "FLAGGED":
        gate_flags = "; ".join(context.ingest_validity.flags)
        if status == OverallValidationStatus.PASS:
            status = OverallValidationStatus.PASS_WITH_WARNINGS
            status_reason = f"{status_reason} | Ingest validity flagged: {gate_flags}"
        elif status == OverallValidationStatus.FAIL and any(
            "not a temperature ramp test" in f for f in context.ingest_validity.flags
        ):
            # A capture that is not a temperature ramp test cannot FAIL a
            # ramp-rate validation — there is no valid verdict to render.
            status = OverallValidationStatus.INCONCLUSIVE
            status_reason = (
                f"Not analysable as a ramp-rate test: {gate_flags} "
                f"(ramp verdict withheld)"
            )
    
    return AnalysisResult(
        status=status,
        status_reason=status_reason,
        canonical_trace=context.canonical_trace,
        ingest_validity=context.ingest_validity,
        data_quality_report=context.data_quality_report,
        validation_profile_used=context.profile,
        inferred_setpoints=context.resolved_setpoints,
        process_boundaries=context.process_boundaries,
        classified_regions=context.region_list,
        valid_ramp_regions=context.valid_ramp_regions,
        cycles=context.cycles,
        metrics=context.metric_set,
        validation_results=context.validation_results,
        profile_comparison_results=context.profile_comparison_results,
        visualisation=context.visualisation_bundle,
        report_package=context.report_package,
        audit_log=context.audit_log,
        phase_conformance=context.phase_conformance,
    )
