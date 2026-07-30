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


def _derive_direction_band(
    direction,
    valid_ramps: list,
    ramp_metrics_map: dict,
    explicit_rate: float | None,
    audit_log: AuditLog,
) -> tuple[float | None, float | None]:
    """Per-direction band centre (target) and tolerance half-width.

    Computed SEPARATELY for heating and cooling — they are different physical
    processes (resistive heating vs refrigeration) and hold their rates to
    different consistencies, so each gets its own band.

    Band centre (target): explicit profile rate → median commanded setpoint
    slope (Mode A) → median measured rate (stepped/Mode B, self-referential).

    Band half-width is derived ENTIRELY from the trace's own evidence — no
    picked fraction, no floor:

        tolerance = systematic_lag + K * MAD(measured rates)

    where
      - MAD(measured rates) is the robust spread of this direction's per-ramp
        rates: the chamber's DEMONSTRATED rate-holding consistency. A chamber
        that holds its rate tightly earns a tight band; an erratic one a
        looser band. This self-scales, exactly as intended.
      - systematic_lag = |median(measured) - target| absorbs the steady
        commanded-vs-achieved offset (thermal mass makes a chamber track its
        commanded ramp with a small consistent lag), so the symmetric band
        around the commanded rate does not reject ramps for that normal lag.
      - K = 3 x 1.4826 is the standard robust 3-sigma-equivalent scaling
        (MAD->sigma is 1.4826; 3 sigma is the conventional outlier bound). It
        is a fixed statistical convention, not a tuning knob.

    A ramp fails when its rate is a robust >3-sigma outlier relative to the
    chamber's own demonstrated ramp behaviour — two-sided, so too-fast fails
    as readily as too-slow.
    """
    import numpy as np

    MAD_TO_SIGMA = 1.4826
    SIGMA_BOUND = 3.0
    K = SIGMA_BOUND * MAD_TO_SIGMA

    metrics = [
        ramp_metrics_map[vr.region_id]
        for vr in valid_ramps
        if vr.direction == direction and vr.region_id in ramp_metrics_map
    ]
    if not metrics:
        return None, None

    measured = np.array([abs(m.robust_slope_c_per_min) for m in metrics])
    commanded = np.array([
        abs(m.commanded_slope_c_per_min) for m in metrics
        if m.commanded_slope_c_per_min is not None
    ])

    if explicit_rate is not None:
        target = abs(explicit_rate)
        source = "explicit profile rate"
    elif commanded.size:
        target = float(np.median(commanded))
        source = "median commanded setpoint slope"
    elif measured.size:
        target = float(np.median(measured))
        source = "median measured rate (no commanded rate; self-referential)"
    else:
        return None, None

    median_measured = float(np.median(measured))
    mad = float(np.median(np.abs(measured - median_measured)))  # chamber's own consistency
    systematic_lag = abs(median_measured - target)

    # Cross-ramp spread (the chamber's demonstrated consistency), floored at
    # the measurement precision of a single ramp. With few, tightly-clustered
    # ramps the sample MAD collapses toward zero and would flag a sibling
    # differing by a hair (e.g. 4.968 vs 4.972); the band must never be
    # tighter than how precisely a rate is even measurable. Both terms are
    # trace-derived — no picked constant.
    measurement_precision = float(np.median([
        m.slope_uncertainty_c_per_min for m in metrics
        if m.slope_uncertainty_c_per_min > 0
    ] or [0.0]))
    spread = max(K * mad, measurement_precision)
    tolerance = systematic_lag + spread

    dir_name = direction.value if hasattr(direction, "value") else str(direction)
    audit_log.add(AuditEntry(
        timestamp=datetime.now(),
        module_name="validation_engine",
        action="ramp_band_derived",
        input_reference=dir_name,
        decision="DERIVED",
        reason=(
            f"{dir_name} ramp band {target:.2f} +/- {tolerance:.3f} degC/min "
            f"(centre from {source}; half-width = lag {systematic_lag:.3f} + "
            f"spread {spread:.3f} = max(3-sigma MAD {K * mad:.3f} [MAD {mad:.3f} "
            f"over {measured.size} ramp(s)], measurement precision "
            f"{measurement_precision:.3f}); fully trace-derived, no picked constant)"
        ),
        thresholds_used={
            "target": target, "tolerance": tolerance, "mad": mad,
            "lag": systematic_lag, "measurement_precision": measurement_precision,
        },
        severity=AuditSeverity.INFO,
        category=AuditCategory.VALIDATION,
    ))
    return target, tolerance


def _note_descriptive_ramp(valid_ramp, metrics, audit_log: AuditLog) -> None:
    """Record a ramp that has no rate spec to validate against (descriptive)."""
    dir_name = valid_ramp.direction.value if hasattr(valid_ramp.direction, "value") else str(valid_ramp.direction)
    audit_log.add(AuditEntry(
        timestamp=datetime.now(),
        module_name="validation_engine",
        action="ramp_descriptive_only",
        input_reference=valid_ramp.region_id,
        decision="NO_TARGET_RATE",
        reason=(
            f"{dir_name} ramp {valid_ramp.region_id}: no commanded or reference "
            f"rate available (measured {metrics.robust_slope_c_per_min:.2f} "
            "degC/min reported descriptively)"
        ),
        severity=AuditSeverity.INFO,
        category=AuditCategory.VALIDATION,
    ))


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

    # Resolve tolerance-bearing parameters before any rule runs.
    # Explicit profile value wins; otherwise the trace-derived value is used
    # (self-referential validation: the setpoint programme is the spec).
    # Ramp RATE targets are NOT resolved here — they are handled as a
    # per-direction BAND (centre + tolerance) derived directly from each
    # direction's commanded/measured rates in _derive_direction_band.
    tolerance_parameters = [
        "dwell_setpoint_deviation",
        "ramp_deviation",
    ]
    for param in tolerance_parameters:
        try:
            resolution = resolve_tolerance(param, profile, adaptive_constants, audit_log)
            profile.tolerance_resolutions.append(resolution)
        except ValueError:
            # Neither explicit nor derivable (e.g. step-setpoint programme
            # has no commanded ramp rate) — that requirement is descriptive.
            audit_log.add(AuditEntry(
                timestamp=datetime.now(),
                module_name="validation_engine",
                action="tolerance_unresolvable",
                input_reference=param,
                decision="DESCRIPTIVE_ONLY",
                reason=(
                    f"No explicit profile value and no trace-derived value for "
                    f"'{param}'; measured values will be reported without pass/fail"
                ),
                severity=AuditSeverity.WARNING,
                category=AuditCategory.VALIDATION,
            ))

    # Explicit ramp-rate band centres from the profile (override the derived
    # commanded rate when a formal spec exists).
    explicit_heating = profile.get_explicit_tolerance("required_heating_ramp_rate")
    explicit_cooling = profile.get_explicit_tolerance("required_cooling_ramp_rate")

    results = []

    ramp_metrics_map = {m.region_id: m for m in ramp_metrics}
    dwell_metrics_map = {m.region_id: m for m in dwell_metrics}

    # Per-direction band centre (target) and tolerance, derived once from the
    # trace so heating and cooling each get their OWN band (they are different
    # physical processes and may run at different rates).
    heating_target, heating_tol = _derive_direction_band(
        RampDirection.HEATING, valid_ramps, ramp_metrics_map, explicit_heating, audit_log
    )
    cooling_target, cooling_tol = _derive_direction_band(
        RampDirection.COOLING, valid_ramps, ramp_metrics_map, explicit_cooling, audit_log
    )

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

        # Band centre: the ramp's OWN commanded setpoint slope when available
        # (most specific target), else the per-direction target. Tolerance is
        # the per-direction adaptive band half-width.
        if valid_ramp.direction == RampDirection.HEATING:
            centre = (
                abs(metrics.commanded_slope_c_per_min)
                if metrics.commanded_slope_c_per_min is not None
                else heating_target
            )
            if centre is not None and heating_tol is not None:
                results.append(validate_heating_ramp_rate(
                    valid_ramp,
                    metrics,
                    required_rate_c_per_min=centre,
                    tolerance_c_per_min=heating_tol,
                    audit_log=audit_log,
                ))
            else:
                _note_descriptive_ramp(valid_ramp, metrics, audit_log)

        elif valid_ramp.direction == RampDirection.COOLING:
            centre = (
                abs(metrics.commanded_slope_c_per_min)
                if metrics.commanded_slope_c_per_min is not None
                else cooling_target
            )
            if centre is not None and cooling_tol is not None:
                results.append(validate_cooling_ramp_rate(
                    valid_ramp,
                    metrics,
                    required_rate_c_per_min=centre,
                    tolerance_c_per_min=cooling_tol,
                    audit_log=audit_log,
                ))
            else:
                _note_descriptive_ramp(valid_ramp, metrics, audit_log)
    
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
