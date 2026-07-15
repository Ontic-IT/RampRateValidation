"""Main validation engine (M11).

Orchestrates all validation rules and produces final validation results.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from config.constants import (
    AuditCategory,
    AuditSeverity,
    RampDirection,
    RegionType,
)
from models.domain import (
    AuditEntry,
    AuditLog,
    ClassifiedTrace,
    CycleList,
    DwellMetrics,
    OverallStatus,
    PhaseConformanceSummary,
    RampMetrics,
    Region,
    RegionList,
    ResolvedSetpoints,
    ValidRampRegion,
    ValidationDataQualityImpact,
    ValidationResult,
    ValidationResults,
)
from models.profile import ValidationProfile


def _get_minimum_dwell_seconds(profile: ValidationProfile, region: Region) -> float:
    """Select the correct minimum dwell duration based on region type."""
    if region.primary_classification == RegionType.HOT_DWELL:
        return profile.dwell_requirements.minimum_hot_dwell_seconds
    elif region.primary_classification == RegionType.COLD_DWELL:
        return profile.dwell_requirements.minimum_cold_dwell_seconds
    else:
        return min(
            profile.dwell_requirements.minimum_hot_dwell_seconds,
            profile.dwell_requirements.minimum_cold_dwell_seconds,
        )
from engine.validation.ramp_rules import (
    validate_heating_ramp_rate,
    validate_cooling_ramp_rate,
    validate_data_quality,
)
from engine.validation.dwell_rules import (
    validate_dwell_duration,
    validate_setpoint_deviation,
    validate_overshoot,
    validate_settling_time,
)
from engine.validation.cycle_rules import (
    validate_cycle_count,
    validate_profile_sequence,
)
from engine.validation.aggregation import aggregate_validation_status, compute_phase_conformance
from engine.validation.tolerance_resolver import (
    AdaptiveConstants,
    resolve_tolerance,
)


def validate_analysis(
    profile: ValidationProfile,
    regions: RegionList,
    valid_ramps: list[ValidRampRegion],
    ramp_metrics: list[RampMetrics],
    dwell_metrics: list[DwellMetrics],
    cycles: CycleList,
    quality_impact: ValidationDataQualityImpact | None = None,
    audit_log: AuditLog | None = None,
    adaptive_constants: AdaptiveConstants | None = None,
) -> tuple[ValidationResults, OverallStatus, PhaseConformanceSummary]:
    """Run all validation rules against analysis results.

    Args:
        profile: Validation profile with requirements
        regions: Classified regions
        valid_ramps: Isolated valid ramps
        ramp_metrics: Computed ramp metrics
        dwell_metrics: Computed dwell metrics
        cycles: Detected cycles
        quality_impact: Data quality impact
        audit_log: Optional audit log
        adaptive_constants: Optional adaptive-derived tolerance values

    Returns:
        Tuple of (ValidationResults, OverallStatus)
    """
    if audit_log is None:
        audit_log = AuditLog()

    # Resolve tolerance-bearing parameters before any rule runs
    tolerance_parameters = ["dwell_setpoint_deviation", "ramp_deviation"]
    for param in tolerance_parameters:
        resolution = resolve_tolerance(param, profile, adaptive_constants, audit_log)
        profile.tolerance_resolutions.append(resolution)

    results = []
    
    ramp_metrics_map = {m.region_id: m for m in ramp_metrics}
    dwell_metrics_map = {m.region_id: m for m in dwell_metrics}
    
    for valid_ramp in valid_ramps:
        metrics = ramp_metrics_map.get(valid_ramp.region_id)
        if not metrics:
            continue
        
        if quality_impact:
            quality_status, quality_reason = validate_data_quality(
                quality_impact, "HEATING_RAMP_RATE"
            )
            if quality_status.value == "INCONCLUSIVE":
                continue
        
        if valid_ramp.direction == RampDirection.HEATING:
            result = validate_heating_ramp_rate(
                valid_ramp,
                metrics,
                required_rate_c_per_min=profile.ramp_rate_requirements.required_heating_ramp_rate_c_per_min,
                minimum_sustained_ratio=profile.ramp_rate_requirements.minimum_sustained_ramp_rate_ratio,
                audit_log=audit_log,
            )
            results.append(result)
        
        elif valid_ramp.direction == RampDirection.COOLING:
            result = validate_cooling_ramp_rate(
                valid_ramp,
                metrics,
                required_rate_c_per_min=profile.ramp_rate_requirements.required_cooling_ramp_rate_c_per_min,
                minimum_sustained_ratio=profile.ramp_rate_requirements.minimum_sustained_ramp_rate_ratio,
                audit_log=audit_log,
            )
            results.append(result)
    
    # DWELL validation: duration and setpoint deviation (for conformance)
    # Get resolved setpoint deviation tolerance (adaptive or explicit)
    setpoint_deviation_tolerance = None
    for resolution in profile.tolerance_resolutions:
        if resolution.parameter_name == "dwell_setpoint_deviation":
            setpoint_deviation_tolerance = resolution.resolved_value
            break
    
    # Fallback if tolerance not resolved
    if setpoint_deviation_tolerance is None:
        setpoint_deviation_tolerance = profile.dwell_requirements.allowed_setpoint_deviation_c or 5.0
    
    # Calculate median and std dev of dwell durations across all dwells in trace
    # for median-based duration validation
    dwell_regions = [r for r in regions.regions 
                     if r.primary_classification in (RegionType.HOT_DWELL, RegionType.COLD_DWELL)]
    
    if len(dwell_regions) > 0:
        import numpy as np
        dwell_durations = np.array([r.duration_seconds for r in dwell_regions])
        median_dwell_duration = float(np.median(dwell_durations))
        dwell_duration_std = float(np.std(dwell_durations))
    else:
        # Fallback if no dwells found
        median_dwell_duration = 30.0
        dwell_duration_std = 10.0
    
    for region in regions.regions:
        if region.primary_classification not in (RegionType.HOT_DWELL, RegionType.COLD_DWELL):
            continue
        
        metrics = dwell_metrics_map.get(region.region_id)
        if not metrics:
            continue
        
        if quality_impact:
            quality_status, _ = validate_data_quality(quality_impact, "DWELL_DURATION")
            if quality_status.value == "INCONCLUSIVE":
                continue
        
        # Validate dwell duration using median-based deviation (flag if >2σ from median)
        duration_result = validate_dwell_duration(
            region,
            metrics,
            median_duration_seconds=median_dwell_duration,
            duration_std_dev=dwell_duration_std,
            sigma_threshold=2.0,
            audit_log=audit_log,
        )
        results.append(duration_result)
        
        # Validate setpoint deviation (for conformance - setpoint tracking accuracy)
        deviation_result = validate_setpoint_deviation(
            region,
            metrics,
            maximum_deviation_c=setpoint_deviation_tolerance,
            audit_log=audit_log,
        )
        results.append(deviation_result)
    
    # Cycle count and profile sequence are DESCRIPTIVE, not prescriptive
    # Don't validate - just report what was measured/classified
    
    validation_results = ValidationResults(results=results)

    overall_status = aggregate_validation_status(
        validation_results,
        quality_impact=quality_impact,
        audit_log=audit_log,
    )

    phase_conformance = compute_phase_conformance(validation_results)

    audit_log.add(AuditEntry(
        timestamp=datetime.now(),
        module_name="validation_engine",
        action="validate_analysis_complete",
        decision=overall_status.status.value,
        reason=overall_status.reason,
        thresholds_used={
            "total_results": len(results),
            "total_phases": phase_conformance.total_phases,
            "passed_phases": phase_conformance.passed_phases,
        },
        severity=AuditSeverity.INFO,
        category=AuditCategory.VALIDATION,
    ))

    return validation_results, overall_status, phase_conformance
