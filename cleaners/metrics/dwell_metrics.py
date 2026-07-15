"""Dwell metrics computation (M10).

Handles:
- Overshoot magnitude, overshoot_duration, settling time, oscillation count
- time_inside_tolerance_band calculation
- Dwell stability metrics
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np

from config.constants import AuditCategory, AuditSeverity, RegionType
from models.domain import (
    AuditEntry,
    AuditLog,
    ClassifiedTrace,
    DwellMetrics,
    Region,
    ResolvedSetpoints,
)


def compute_dwell_metrics(
    region: Region,
    classified_trace: ClassifiedTrace,
    setpoints: ResolvedSetpoints,
    settling_tolerance_band_c: float = 3.0,
    allowed_setpoint_deviation_c: float | None = None,
    audit_log: AuditLog | None = None,
) -> DwellMetrics:
    """Compute all metrics for a dwell region.
    
    Args:
        region: Dwell region
        classified_trace: Trace with classification labels
        setpoints: Resolved setpoints
        settling_tolerance_band_c: Tolerance band for settling
        allowed_setpoint_deviation_c: Allowed deviation from setpoint
        audit_log: Optional audit log
    
    Returns:
        DwellMetrics with all computed values
    """
    if audit_log is None:
        audit_log = AuditLog()
    
    if region.primary_classification not in (
        RegionType.HOT_DWELL, RegionType.COLD_DWELL, RegionType.AMBIENT_START
    ):
        return DwellMetrics(region_id=region.region_id)
    
    rows = classified_trace.rows[region.start_row:region.end_row + 1]
    
    if len(rows) < 2:
        return DwellMetrics(region_id=region.region_id)
    
    temperatures = np.array([r.temperature_c_raw for r in rows])
    elapsed = np.array([r.elapsed_seconds for r in rows])
    
    if region.primary_classification == RegionType.HOT_DWELL:
        target_setpoint = setpoints.inferred_hot_setpoint_c
    elif region.primary_classification == RegionType.COLD_DWELL:
        target_setpoint = setpoints.inferred_cold_setpoint_c
    else:
        target_setpoint = setpoints.inferred_ambient_c
    
    mean_temp = float(np.mean(temperatures))
    temp_std = float(np.std(temperatures))
    temp_range = float(np.max(temperatures) - np.min(temperatures))
    
    setpoint_deviation = abs(mean_temp - target_setpoint) if target_setpoint else 0.0
    
    effective_deviation = allowed_setpoint_deviation_c if allowed_setpoint_deviation_c is not None else 5.0
    
    time_in_band = _compute_time_in_tolerance_band(
        temperatures, elapsed, target_setpoint, effective_deviation
    )
    
    overshoot_mag, overshoot_dur, settling_time = _compute_overshoot_metrics(
        temperatures, elapsed, target_setpoint, settling_tolerance_band_c,
        region.primary_classification
    )
    
    oscillation_count = _compute_oscillation_count(
        temperatures, target_setpoint, settling_tolerance_band_c
    )
    
    stability_score = _compute_stability_score(temp_std, temp_range, settling_tolerance_band_c)
    
    metrics = DwellMetrics(
        region_id=region.region_id,
        target_setpoint_c=target_setpoint,
        mean_temperature_c=mean_temp,
        temperature_std_c=temp_std,
        temperature_range_c=temp_range,
        setpoint_deviation_c=setpoint_deviation,
        time_inside_tolerance_band_seconds=time_in_band,
        time_inside_tolerance_band_pct=time_in_band / region.duration_seconds * 100 if region.duration_seconds > 0 else 0.0,
        overshoot_magnitude_c=overshoot_mag,
        overshoot_duration_seconds=overshoot_dur,
        settling_time_seconds=settling_time,
        oscillation_count=oscillation_count,
        stability_score=stability_score,
    )
    
    audit_log.add(AuditEntry(
        timestamp=datetime.now(),
        module_name="dwell_metrics",
        action="compute_dwell_metrics",
        input_reference=region.region_id,
        output_reference=f"DwellMetrics(mean={mean_temp:.1f}°C)",
        decision="SUCCESS",
        reason=f"Dwell at {target_setpoint:.1f}°C: deviation={setpoint_deviation:.2f}°C, stability={stability_score:.2f}",
        severity=AuditSeverity.INFO,
        category=AuditCategory.METRICS,
        rows_used=len(rows),
    ))
    
    return metrics


def _compute_time_in_tolerance_band(
    temperatures: np.ndarray,
    elapsed: np.ndarray,
    target_setpoint: float | None,
    tolerance: float,
) -> float:
    """Compute time where temperature is within tolerance band of setpoint."""
    if target_setpoint is None or len(temperatures) < 2:
        return 0.0
    
    in_band = np.abs(temperatures - target_setpoint) <= tolerance
    
    time_in_band = 0.0
    for i in range(len(temperatures) - 1):
        if in_band[i]:
            time_in_band += elapsed[i + 1] - elapsed[i]
    
    if in_band[-1] and len(elapsed) > 1:
        time_in_band += elapsed[-1] - elapsed[-2]
    
    return time_in_band


def _compute_overshoot_metrics(
    temperatures: np.ndarray,
    elapsed: np.ndarray,
    target_setpoint: float | None,
    tolerance: float,
    dwell_type: RegionType,
) -> tuple[float, float, float]:
    """Compute overshoot magnitude, duration, and settling time."""
    if target_setpoint is None or len(temperatures) < 2:
        return 0.0, 0.0, 0.0
    
    if dwell_type == RegionType.HOT_DWELL:
        overshoot_mask = temperatures > target_setpoint + tolerance
        peak_idx = np.argmax(temperatures)
        overshoot_mag = max(0.0, float(temperatures[peak_idx] - target_setpoint))
    elif dwell_type == RegionType.COLD_DWELL:
        overshoot_mask = temperatures < target_setpoint - tolerance
        peak_idx = np.argmin(temperatures)
        overshoot_mag = max(0.0, float(target_setpoint - temperatures[peak_idx]))
    else:
        return 0.0, 0.0, 0.0
    
    overshoot_dur = 0.0
    if np.any(overshoot_mask):
        overshoot_indices = np.where(overshoot_mask)[0]
        for i in overshoot_indices:
            if i < len(elapsed) - 1:
                overshoot_dur += elapsed[i + 1] - elapsed[i]
    
    settling_time = 0.0
    in_band = np.abs(temperatures - target_setpoint) <= tolerance
    for i in range(len(in_band)):
        if in_band[i]:
            settling_time = elapsed[i] - elapsed[0]
            break
    
    return overshoot_mag, overshoot_dur, settling_time


def _compute_oscillation_count(
    temperatures: np.ndarray,
    target_setpoint: float | None,
    tolerance: float,
) -> int:
    """Count oscillations around setpoint."""
    if target_setpoint is None or len(temperatures) < 3:
        return 0
    
    deviations = temperatures - target_setpoint
    
    crossings = 0
    for i in range(1, len(deviations)):
        if deviations[i-1] * deviations[i] < 0:
            crossings += 1
    
    return crossings // 2


def _compute_stability_score(
    temp_std: float,
    temp_range: float,
    tolerance: float,
) -> float:
    """Compute dwell stability score (0-1)."""
    std_score = 1.0 - min(1.0, temp_std / tolerance)
    range_score = 1.0 - min(1.0, temp_range / (tolerance * 4))
    
    return (std_score + range_score) / 2.0
