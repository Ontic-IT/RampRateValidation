"""Report payload builder (M14).

Constructs all 13 mandatory report sections.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from config.constants import AuditCategory, AuditSeverity
from models.domain import (
    AnalysisContext,
    AuditEntry,
    AuditLog,
    OverallStatus,
    ReportPackage,
    ValidationResults,
)


def generate_report_payload(
    context: AnalysisContext,
    audit_log: AuditLog | None = None,
) -> ReportPackage:
    """Generate the complete report package with all 13 sections.

    Args:
        context: Analysis context with all pipeline outputs
        audit_log: Optional audit log

    Returns:
        ReportPackage with all 13 sections populated
    """
    if audit_log is None:
        audit_log = AuditLog()

    sections: dict[str, Any] = {}

    sections["executive_summary"] = build_executive_summary(context)
    sections["input_file_summary"] = _build_input_file_summary(context)
    sections["data_quality_summary"] = _build_data_quality_summary(context)
    sections["process_boundary_summary"] = _build_process_boundary_summary(context)
    sections["setpoint_inference_summary"] = _build_setpoint_inference_summary(context)
    sections["region_classification_summary"] = _build_region_classification_summary(context)
    sections["ramp_rate_validation_summary"] = _build_ramp_rate_validation_summary(context)
    sections["cycle_level_summary"] = _build_cycle_level_summary(context)
    sections["overshoot_correction_summary"] = _build_overshoot_correction_summary(context)
    sections["profile_consistency_summary"] = _build_profile_consistency_summary(context)
    sections["visualisation"] = _build_visualisation_summary(context)
    sections["audit_trail"] = _build_audit_trail_summary(context)
    sections["algorithm_appendix"] = _build_algorithm_appendix(context)

    audit_log.add(AuditEntry(
        timestamp=datetime.now(),
        module_name="report_builder",
        action="generate_report_payload",
        decision="SUCCESS",
        reason="All 13 report sections generated",
        severity=AuditSeverity.INFO,
        category=AuditCategory.PIPELINE,
    ))

    return ReportPackage(sections=sections)


def build_executive_summary(context: AnalysisContext) -> dict[str, Any]:
    """Build Section 1: Executive Summary (matches reference template format)."""
    overall = context.overall_validation_status
    status = overall.status.value if overall else "UNKNOWN"
    reason = overall.reason if overall else "No validation performed"
    
    # Extract conformance percentage (prominently displayed in template)
    conformance_pct = context.phase_conformance.conformance_percentage if context.phase_conformance else 0.0
    
    # Extract test parameters
    hot_target = context.resolved_setpoints.inferred_hot_setpoint_c if context.resolved_setpoints else None
    cold_target = context.resolved_setpoints.inferred_cold_setpoint_c if context.resolved_setpoints else None
    
    # Extract tolerances used
    dwell_tolerance = None
    ramp_tolerance = None
    if context.profile and hasattr(context.profile, 'tolerance_resolutions'):
        for resolution in context.profile.tolerance_resolutions:
            if 'dwell' in resolution.parameter_name.lower():
                dwell_tolerance = resolution.resolved_value
            elif 'ramp' in resolution.parameter_name.lower():
                ramp_tolerance = resolution.resolved_value
    
    # Fallback to profile defaults if no resolutions
    if not dwell_tolerance and context.profile:
        dwell_tolerance = getattr(context.profile.dwell_requirements, 'allowed_setpoint_deviation_c', None)
    if not ramp_tolerance and context.profile:
        ramp_tolerance = getattr(context.profile.ramp_rate_requirements, 'allowed_deviation_from_target_c_per_min', None)

    # Build anomaly list
    anomalies_detected = []
    if context.phase_conformance and context.region_list and context.metric_set:
        anomaly_ids = set(context.phase_conformance.anomaly_phase_ids)
        dwell_metrics_map = {dm.region_id: dm for dm in context.metric_set.dwell_metrics}
        
        for region in context.region_list.regions:
            if region.region_id in anomaly_ids:
                # Determine target and actual
                target = None
                actual = None
                if "HOT_DWELL" in region.primary_classification.value:
                    target = hot_target
                elif "COLD_DWELL" in region.primary_classification.value:
                    target = cold_target
                
                if region.region_id in dwell_metrics_map:
                    actual = dwell_metrics_map[region.region_id].mean_temperature_c
                
                anomalies_detected.append({
                    "phase_id": region.region_id,
                    "target_c": target,
                    "actual_avg_c": actual,
                })

    return {
        "title": "Test Summary & Observations",
        "conformance_percentage": conformance_pct,
        "overall_status": status,
        "status_reason": reason,
        "file_name": context.file_metadata.source_file_path if context.file_metadata else "",
        "profile_name": context.profile.profile_metadata.profile_name if context.profile else "",
        "test_parameters": {
            "hot_soak_target_c": hot_target,
            "cold_soak_target_c": cold_target,
            "test_duration_hours": (context.data_quality_report.process_duration_seconds / 3600.0) if context.data_quality_report else None,
            "total_phases_identified": len(context.region_list.regions) if context.region_list else 0,
        },
        "tolerances_used": {
            "dwell_tolerance_c": dwell_tolerance,
            "ramp_tolerance_c_per_min": ramp_tolerance,
        },
        "anomalies_detected": {
            "count": len(anomalies_detected),
            "phases_flagged": anomalies_detected,
        },
        "root_cause_assessment": "Manual review required - automated root cause analysis not yet implemented",
        "probable_causes": [
            "Requires domain expert analysis",
            "Consider: refrigerant levels, valve operation, heat exchanger condition",
            "Review auxiliary sensor data if available",
        ],
        "recommendation": f"Test {'CONFORMING' if conformance_pct >= 60.0 else 'NON-CONFORMING'}. {'Chamber requires maintenance before re-test.' if conformance_pct < 60.0 else 'Chamber performance acceptable.'}",
        "cycle_count": len(context.cycles.cycles) if context.cycles else 0,
        "region_count": len(context.region_list.regions) if context.region_list else 0,
        "valid_ramp_count": len(context.valid_ramp_regions.regions) if context.valid_ramp_regions else 0,
        "phase_conformance": context.phase_conformance.model_dump() if context.phase_conformance else {},
    }


def _build_input_file_summary(context: AnalysisContext) -> dict[str, Any]:
    """Build Section 2: Input File Summary."""
    if not context.file_metadata:
        return {"title": "Input File Summary", "note": "Not applicable — no file metadata"}

    return {
        "title": "Input File Summary",
        "source_file_path": context.file_metadata.source_file_path,
        "detected_delimiter": context.file_metadata.detected_delimiter,
        "detected_encoding": context.file_metadata.detected_encoding,
        "detected_timestamp_format": context.file_metadata.detected_timestamp_format,
        "raw_row_count": context.file_metadata.raw_row_count,
        "usable_row_count": context.file_metadata.usable_row_count,
        "selected_temperature_channel": context.file_metadata.selected_temperature_channel,
        "selected_setpoint_channel": context.file_metadata.selected_setpoint_channel,
    }


def _build_data_quality_summary(context: AnalysisContext) -> dict[str, Any]:
    """Build Section 3: Data Quality Summary."""
    if not context.data_quality_report:
        return {"title": "Data Quality Summary", "note": "Not applicable — no quality report"}

    return {
        "title": "Data Quality Summary",
        "overall_status": context.data_quality_report.overall_status.value,
        "missing_data_pct": context.data_quality_report.missing_data_pct,
        "duplicate_timestamp_pct": context.data_quality_report.duplicate_timestamp_pct,
        "spike_count": context.data_quality_report.spike_count,
        "large_gap_count": context.data_quality_report.large_gap_count,
        "process_duration_seconds": context.data_quality_report.process_duration_seconds,
    }


def _build_process_boundary_summary(context: AnalysisContext) -> dict[str, Any]:
    """Build Section 4: Process Boundary Summary."""
    if not context.process_boundaries:
        return {"title": "Process Boundary Summary", "note": "Not applicable — no boundaries detected"}

    return {
        "title": "Process Boundary Summary",
        "ambient_start_index": context.process_boundaries.ambient_start_index,
        "process_start_index": context.process_boundaries.process_start_index,
        "process_end_index": context.process_boundaries.process_end_index,
        "detection_method": context.process_boundaries.detection_method,
    }


def _build_setpoint_inference_summary(context: AnalysisContext) -> dict[str, Any]:
    """Build Section 5: Setpoint/Dwell Inference Summary."""
    if not context.resolved_setpoints:
        return {"title": "Setpoint/Dwell Inference Summary", "note": "Not applicable — no setpoints resolved"}

    return {
        "title": "Setpoint/Dwell Inference Summary",
        "resolution_mode": context.resolved_setpoints.resolution_mode.value,
        "inferred_ambient_c": context.resolved_setpoints.inferred_ambient_c,
        "inferred_hot_setpoint_c": context.resolved_setpoints.inferred_hot_setpoint_c,
        "inferred_cold_setpoint_c": context.resolved_setpoints.inferred_cold_setpoint_c,
        "algorithm_seed_used": context.resolved_setpoints.algorithm_seed_used,
    }


def _build_region_classification_summary(context: AnalysisContext) -> dict[str, Any]:
    """Build Section 6: Phase Analysis (matches reference template format)."""
    if not context.region_list:
        return {"title": "Phase Analysis", "note": "Not applicable — no regions classified"}

    phase_table = build_phase_analysis_table(context)

    return {
        "title": "Phase Analysis - Middle 80% Values",
        "subtitle": "Ramp rates from middle 80% of transition. Dwell temps from middle 80% of soak.",
        "total_phases": len(context.region_list.regions),
        "phases": phase_table,
    }


def build_phase_analysis_table(context: AnalysisContext) -> list[dict[str, Any]]:
    """Build phase-by-phase analysis table matching reference template.
    
    Columns: #, Type, SP Range, Rate (°C/min), Avg Temp (°C), Duration, Max Dev, Status
    """
    if not context.region_list:
        return []
    
    phase_table = []
    anomaly_ids = set(context.phase_conformance.anomaly_phase_ids) if context.phase_conformance else set()
    
    # Build lookup maps
    ramp_metrics_map = {}
    dwell_metrics_map = {}
    if context.metric_set:
        ramp_metrics_map = {rm.region_id: rm for rm in context.metric_set.ramp_metrics}
        dwell_metrics_map = {dm.region_id: dm for dm in context.metric_set.dwell_metrics}
    
    setpoints = context.resolved_setpoints
    
    for idx, region in enumerate(context.region_list.regions, start=1):
        region_id = region.region_id
        region_type = region.primary_classification.value
        
        # Determine setpoint range
        sp_range = "-"
        if "HEATING_RAMP" in region_type:
            if setpoints:
                sp_range = f"{setpoints.inferred_cold_setpoint_c or setpoints.inferred_ambient_c} -> {setpoints.inferred_hot_setpoint_c}"
        elif "COOLING_RAMP" in region_type:
            if setpoints:
                sp_range = f"{setpoints.inferred_hot_setpoint_c} -> {setpoints.inferred_cold_setpoint_c}"
        elif "HOT_DWELL" in region_type:
            if setpoints:
                sp_range = str(setpoints.inferred_hot_setpoint_c)
        elif "COLD_DWELL" in region_type:
            if setpoints:
                sp_range = str(setpoints.inferred_cold_setpoint_c)
        elif "AMBIENT" in region_type:
            if setpoints:
                sp_range = str(setpoints.inferred_ambient_c or 20)
        
        # Extract ramp rate or dwell avg temp
        rate = None
        avg_temp = None
        max_dev = None
        
        if "RAMP" in region_type and region_id in ramp_metrics_map:
            rm = ramp_metrics_map[region_id]
            rate = round(rm.robust_slope_c_per_min, 2) if rm.robust_slope_c_per_min else None
        
        if "DWELL" in region_type and region_id in dwell_metrics_map:
            dm = dwell_metrics_map[region_id]
            avg_temp = round(dm.mean_temperature_c, 1) if dm.mean_temperature_c else None
            max_dev = round(abs(dm.setpoint_deviation_c), 1) if dm.setpoint_deviation_c else None
        
        # Duration in minutes
        duration_min = round(region.duration_seconds / 60.0, 1)
        
        # Status: OK or ANOMALY
        status = "ANOMALY" if region_id in anomaly_ids else "OK"
        
        # Map region type to simplified display type
        display_type = region_type.lower().replace("_", "_")
        if "heating_ramp" in display_type:
            display_type = "ramp_up"
        elif "cooling_ramp" in display_type:
            display_type = "ramp_down"
        elif "hot_dwell" in display_type or "cold_dwell" in display_type:
            display_type = "dwell"
        
        phase_table.append({
            "phase_number": idx,
            "type": display_type,
            "sp_range": sp_range,
            "rate_c_per_min": rate,
            "avg_temp_c": avg_temp,
            "duration_min": duration_min,
            "max_dev_c": max_dev,
            "status": status,
        })
    
    return phase_table


def _build_ramp_rate_validation_summary(context: AnalysisContext) -> dict[str, Any]:
    """Build Section 7: Ramp-Rate Validation Summary."""
    if not context.validation_results:
        return {"title": "Ramp-Rate Validation Summary", "note": "Not applicable — no validation results"}

    results = []
    for vr in context.validation_results.results:
        if "RAMP_RATE" in vr.requirement_id:
            results.append({
                "requirement_id": vr.requirement_id,
                "measured_value": vr.measured_value,
                "threshold_value": vr.threshold_value,
                "result": vr.result.value,
                "reason": vr.reason,
            })

    return {
        "title": "Ramp-Rate Validation Summary",
        "ramp_results": results,
    }


def _build_cycle_level_summary(context: AnalysisContext) -> dict[str, Any]:
    """Build Section 8: Cycle-Level Summary."""
    if not context.cycles:
        return {"title": "Cycle-Level Summary", "note": "Not applicable — no cycles detected"}

    cycles = []
    for cycle in context.cycles.cycles:
        cycles.append({
            "cycle_id": cycle.cycle_id,
            "cycle_number": cycle.cycle_number,
            "status": cycle.status.value,
            "duration_seconds": cycle.duration_seconds,
            "heating_ramp_count": cycle.heating_ramp_count,
            "cooling_ramp_count": cycle.cooling_ramp_count,
            "hot_dwell_count": cycle.hot_dwell_count,
            "cold_dwell_count": cycle.cold_dwell_count,
        })

    return {
        "title": "Cycle-Level Summary",
        "total_cycles": len(context.cycles.cycles),
        "cycles": cycles,
    }


def _build_overshoot_correction_summary(context: AnalysisContext) -> dict[str, Any]:
    """Build Section 9: Overshoot/Correction Summary."""
    if not context.metric_set:
        return {"title": "Overshoot/Correction Summary", "note": "Not applicable — no metrics computed"}

    overshoots = []
    for dm in context.metric_set.dwell_metrics:
        if dm.overshoot_magnitude_c and dm.overshoot_magnitude_c > 0:
            overshoots.append({
                "region_id": dm.region_id,
                "overshoot_magnitude_c": dm.overshoot_magnitude_c,
                "settling_time_seconds": dm.settling_time_seconds,
                "oscillation_count": dm.oscillation_count,
            })

    return {
        "title": "Overshoot/Correction Summary",
        "overshoot_count": len(overshoots),
        "overshoots": overshoots,
    }


def _build_profile_consistency_summary(context: AnalysisContext) -> dict[str, Any]:
    """Build Section 10: Profile Consistency Summary."""
    if not context.profile_comparison_results:
        return {"title": "Profile Consistency Summary", "note": "Not applicable — no comparison performed"}

    return {
        "title": "Profile Consistency Summary",
        "comparisons": context.profile_comparison_results.comparisons,
    }


def _build_visualisation_summary(context: AnalysisContext) -> dict[str, Any]:
    """Build Section 11: Visualisation."""
    if not context.visualisation_bundle:
        return {"title": "Visualisation", "note": "Not applicable — no visualisation generated"}

    return {
        "title": "Visualisation",
        "charts": list(context.visualisation_bundle.charts.keys()),
    }


def _build_audit_trail_summary(context: AnalysisContext) -> dict[str, Any]:
    """Build Section 12: Audit Trail."""
    if not context.audit_log:
        return {"title": "Audit Trail", "note": "Not applicable — no audit log"}

    entries = []
    for entry in context.audit_log.entries:
        entries.append({
            "timestamp": entry.timestamp.isoformat() if entry.timestamp else "",
            "module_name": entry.module_name,
            "action": entry.action,
            "decision": entry.decision,
            "reason": entry.reason,
            "severity": entry.severity.value,
        })

    return {
        "title": "Audit Trail",
        "total_entries": len(entries),
        "entries": entries[:50],  # First 50 entries for brevity
    }


def _build_algorithm_appendix(context: AnalysisContext) -> dict[str, Any]:
    """Build Section 13: Appendix — Algorithm Configuration."""
    if not context.run_metadata:
        return {"title": "Appendix: Algorithm Configuration", "note": "Not applicable — no run metadata"}

    return {
        "title": "Appendix: Algorithm Configuration",
        "algorithm_version": context.run_metadata.algorithm_version,
        "python_version": context.run_metadata.python_version,
        "random_seed": context.run_metadata.random_seed,
        "adaptive_constants_snapshot": context.run_metadata.adaptive_constants_snapshot,
        "classifier_configuration_snapshot": context.run_metadata.classifier_configuration_snapshot,
        "feature_extraction_configuration": context.run_metadata.feature_extraction_configuration,
    }
