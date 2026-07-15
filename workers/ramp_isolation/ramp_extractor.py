"""Valid ramp extraction and isolation (M08).

Handles:
- Confidence-weighted departure modelling (ramp start)
- Confidence-weighted arrival modelling (ramp end)
- Adaptive ramp envelope extraction
- Dwell tail, overshoot, and correction exclusion
- Reversal tolerance
- Monotonicity confirmation
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np

from config.constants import (
    AuditCategory,
    AuditSeverity,
    RegionType,
    RampDirection,
)
from models.domain import (
    AuditEntry,
    AuditLog,
    ClassifiedTrace,
    PreprocessingReport,
    Region,
    RegionList,
    ResolvedSetpoints,
    ValidRampRegion,
)


def isolate_valid_ramps(
    classified_trace: ClassifiedTrace,
    regions: RegionList,
    setpoints: ResolvedSetpoints,
    preprocessing_report: PreprocessingReport,
    reversal_tolerance_factor: float = 1.5,
    min_ramp_duration_seconds: float = 30.0,
    min_monotonicity_score: float = 0.7,
    audit_log: AuditLog | None = None,
) -> list[ValidRampRegion]:
    """Isolate valid ramp regions from classified trace.
    
    Args:
        classified_trace: Trace with classification labels
        regions: Classified regions
        setpoints: Resolved setpoints
        preprocessing_report: Preprocessing statistics
        reversal_tolerance_factor: Factor for reversal tolerance
        min_ramp_duration_seconds: Minimum ramp duration
        min_monotonicity_score: Minimum monotonicity for valid ramp
        audit_log: Optional audit log
    
    Returns:
        List of ValidRampRegion objects
    """
    if audit_log is None:
        audit_log = AuditLog()
    
    valid_ramps = []
    noise_floor = preprocessing_report.noise_floor_c
    slope_noise_floor = preprocessing_report.slope_noise_floor_c_per_min
    
    for region in regions.regions:
        if region.primary_classification not in (RegionType.HEATING_RAMP, RegionType.COOLING_RAMP):
            continue
        
        is_heating = region.primary_classification == RegionType.HEATING_RAMP
        direction = RampDirection.HEATING if is_heating else RampDirection.COOLING
        
        region_rows = classified_trace.rows[region.start_row:region.end_row + 1]
        
        if len(region_rows) < 3:
            audit_log.add(AuditEntry(
                timestamp=datetime.now(),
                module_name="ramp_extractor",
                action="skip_ramp",
                input_reference=region.region_id,
                decision="EXCLUDED",
                reason="Insufficient rows",
                severity=AuditSeverity.INFO,
                category=AuditCategory.CLASSIFICATION,
            ))
            continue
        
        temperatures = np.array([r.temperature_c_raw for r in region_rows])
        elapsed = np.array([r.elapsed_seconds for r in region_rows])
        slopes = np.array([
            r.rolling_slope_c_per_min if r.rolling_slope_c_per_min is not None else 0.0
            for r in region_rows
        ])
        
        duration = elapsed[-1] - elapsed[0]
        if duration < min_ramp_duration_seconds:
            audit_log.add(AuditEntry(
                timestamp=datetime.now(),
                module_name="ramp_extractor",
                action="skip_ramp",
                input_reference=region.region_id,
                decision="EXCLUDED",
                reason=f"Duration {duration:.1f}s < minimum {min_ramp_duration_seconds}s",
                severity=AuditSeverity.INFO,
                category=AuditCategory.CLASSIFICATION,
            ))
            continue
        
        valid_start, valid_end, exclusion_reasons = _extract_valid_envelope(
            temperatures,
            slopes,
            elapsed,
            is_heating,
            noise_floor,
            slope_noise_floor,
            reversal_tolerance_factor,
        )
        
        valid_rows = list(range(region.start_row + valid_start, region.start_row + valid_end + 1))
        
        if len(valid_rows) < 3:
            audit_log.add(AuditEntry(
                timestamp=datetime.now(),
                module_name="ramp_extractor",
                action="skip_ramp",
                input_reference=region.region_id,
                decision="EXCLUDED",
                reason="Valid envelope too short after exclusions",
                severity=AuditSeverity.INFO,
                category=AuditCategory.CLASSIFICATION,
            ))
            continue
        
        valid_temps = temperatures[valid_start:valid_end + 1]
        valid_elapsed = elapsed[valid_start:valid_end + 1]
        valid_slopes = slopes[valid_start:valid_end + 1]
        
        monotonicity = _compute_monotonicity(valid_slopes, is_heating)
        
        if monotonicity < min_monotonicity_score:
            audit_log.add(AuditEntry(
                timestamp=datetime.now(),
                module_name="ramp_extractor",
                action="skip_ramp",
                input_reference=region.region_id,
                decision="EXCLUDED",
                reason=f"Monotonicity {monotonicity:.2f} < minimum {min_monotonicity_score}",
                severity=AuditSeverity.INFO,
                category=AuditCategory.CLASSIFICATION,
            ))
            continue
        
        reversal_count, stall_duration = _compute_reversals_and_stalls(
            valid_slopes,
            valid_elapsed,
            slope_noise_floor,
            is_heating,
            reversal_tolerance_factor,
            noise_floor,
        )
        
        departure_confidence = region.classification_evidence[0].evidence.get(
            "dwell_departure_confidence", 0.8
        ) if region.classification_evidence else 0.8
        
        arrival_confidence = region.classification_evidence[0].evidence.get(
            "dwell_arrival_confidence", 0.8
        ) if region.classification_evidence else 0.8
        
        valid_ramp = ValidRampRegion(
            region_id=region.region_id,
            source_region=region,
            direction=direction,
            start_row=valid_rows[0],
            end_row=valid_rows[-1],
            start_time=classified_trace.rows[valid_rows[0]].timestamp,
            end_time=classified_trace.rows[valid_rows[-1]].timestamp,
            duration_seconds=valid_elapsed[-1] - valid_elapsed[0],
            start_temperature_c=float(valid_temps[0]),
            end_temperature_c=float(valid_temps[-1]),
            temperature_delta_c=float(valid_temps[-1] - valid_temps[0]),
            included_rows=valid_rows,
            excluded_rows=[
                r for r in range(region.start_row, region.end_row + 1)
                if r not in valid_rows
            ],
            exclusion_reasons=exclusion_reasons,
            monotonicity_score=monotonicity,
            reversal_count=reversal_count,
            stall_duration_seconds=stall_duration,
            departure_confidence=departure_confidence,
            arrival_confidence=arrival_confidence,
            data_quality_flags=[],
        )
        valid_ramps.append(valid_ramp)
        
        audit_log.add(AuditEntry(
            timestamp=datetime.now(),
            module_name="ramp_extractor",
            action="isolate_valid_ramp",
            input_reference=region.region_id,
            output_reference=valid_ramp.region_id,
            decision="SUCCESS",
            reason=f"Valid {direction.value} ramp: {valid_ramp.duration_seconds:.1f}s, ΔT={valid_ramp.temperature_delta_c:.1f}°C",
            severity=AuditSeverity.INFO,
            category=AuditCategory.CLASSIFICATION,
            rows_used=len(valid_rows),
        ))
    
    audit_log.add(AuditEntry(
        timestamp=datetime.now(),
        module_name="ramp_extractor",
        action="isolate_valid_ramps_complete",
        output_reference=f"ValidRampList({len(valid_ramps)} ramps)",
        decision="SUCCESS",
        reason=f"Isolated {len(valid_ramps)} valid ramps from {len(regions.regions)} regions",
        severity=AuditSeverity.INFO,
        category=AuditCategory.CLASSIFICATION,
    ))
    
    return valid_ramps


def _extract_valid_envelope(
    temperatures: np.ndarray,
    slopes: np.ndarray,
    elapsed: np.ndarray,
    is_heating: bool,
    noise_floor: float,
    slope_noise_floor: float,
    reversal_tolerance_factor: float,
) -> tuple[int, int, list[str]]:
    """Extract valid ramp envelope excluding dwell tails and overshoots."""
    n = len(temperatures)
    exclusion_reasons = []
    
    valid_start = 0
    for i in range(n):
        if is_heating:
            if slopes[i] > slope_noise_floor:
                valid_start = i
                break
        else:
            if slopes[i] < -slope_noise_floor:
                valid_start = i
                break
    else:
        valid_start = 0
    
    if valid_start > 0:
        exclusion_reasons.append(f"Excluded {valid_start} rows at start (dwell tail)")
    
    valid_end = n - 1
    for i in range(n - 1, -1, -1):
        if is_heating:
            if slopes[i] > slope_noise_floor:
                valid_end = i
                break
        else:
            if slopes[i] < -slope_noise_floor:
                valid_end = i
                break
    else:
        valid_end = n - 1
    
    if valid_end < n - 1:
        exclusion_reasons.append(f"Excluded {n - 1 - valid_end} rows at end (overshoot/settling)")
    
    if is_heating:
        peak_idx = np.argmax(temperatures[valid_start:valid_end + 1]) + valid_start
        if peak_idx < valid_end:
            post_peak_drop = temperatures[peak_idx] - temperatures[valid_end]
            if post_peak_drop > noise_floor * 3:
                valid_end = peak_idx
                exclusion_reasons.append(f"Excluded post-peak correction (drop={post_peak_drop:.2f}°C)")
    else:
        trough_idx = np.argmin(temperatures[valid_start:valid_end + 1]) + valid_start
        if trough_idx < valid_end:
            post_trough_rise = temperatures[valid_end] - temperatures[trough_idx]
            if post_trough_rise > noise_floor * 3:
                valid_end = trough_idx
                exclusion_reasons.append(f"Excluded post-trough correction (rise={post_trough_rise:.2f}°C)")
    
    return valid_start, valid_end, exclusion_reasons


def _compute_monotonicity(slopes: np.ndarray, is_heating: bool) -> float:
    """Compute monotonicity score for ramp."""
    if len(slopes) == 0:
        return 0.0
    
    if is_heating:
        correct_direction = np.sum(slopes > 0)
    else:
        correct_direction = np.sum(slopes < 0)
    
    return correct_direction / len(slopes)


def _compute_reversals_and_stalls(
    slopes: np.ndarray,
    elapsed: np.ndarray,
    slope_noise_floor: float,
    is_heating: bool,
    reversal_tolerance_factor: float,
    noise_floor: float,
) -> tuple[int, float]:
    """Compute reversal count and stall duration."""
    reversal_count = 0
    stall_duration = 0.0
    
    reversal_threshold = noise_floor * reversal_tolerance_factor
    
    for i in range(len(slopes)):
        if abs(slopes[i]) < slope_noise_floor:
            if i < len(elapsed) - 1:
                stall_duration += elapsed[i + 1] - elapsed[i]
        
        if i > 0:
            if is_heating:
                if slopes[i] < -reversal_threshold and slopes[i - 1] > reversal_threshold:
                    reversal_count += 1
            else:
                if slopes[i] > reversal_threshold and slopes[i - 1] < -reversal_threshold:
                    reversal_count += 1
    
    return reversal_count, stall_duration
