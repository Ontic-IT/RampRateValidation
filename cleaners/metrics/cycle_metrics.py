"""Cycle metrics computation (M10).

Handles:
- Cycle duration and cycle-to-cycle drift
- Aggregate cycle statistics
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np

from config.constants import AuditCategory, AuditSeverity
from models.domain import (
    AuditEntry,
    AuditLog,
    Cycle,
    CycleMetrics,
    RampMetrics,
    DwellMetrics,
)


def compute_cycle_metrics(
    cycle: Cycle,
    ramp_metrics: list[RampMetrics],
    dwell_metrics: list[DwellMetrics],
    audit_log: AuditLog | None = None,
) -> CycleMetrics:
    """Compute aggregate metrics for a cycle.
    
    Args:
        cycle: Cycle to compute metrics for
        ramp_metrics: Metrics for ramps in this cycle
        dwell_metrics: Metrics for dwells in this cycle
        audit_log: Optional audit log
    
    Returns:
        CycleMetrics with aggregate values
    """
    if audit_log is None:
        audit_log = AuditLog()
    
    heating_slopes = [
        m.robust_slope_c_per_min for m in ramp_metrics
        if m.robust_slope_c_per_min > 0
    ]
    cooling_slopes = [
        m.robust_slope_c_per_min for m in ramp_metrics
        if m.robust_slope_c_per_min < 0
    ]
    
    avg_heating_slope = float(np.mean(heating_slopes)) if heating_slopes else 0.0
    avg_cooling_slope = float(np.mean(cooling_slopes)) if cooling_slopes else 0.0
    
    min_heating_slope = float(np.min(heating_slopes)) if heating_slopes else 0.0
    max_cooling_slope = float(np.max(cooling_slopes)) if cooling_slopes else 0.0
    
    jitter_scores = [m.jitter_score for m in ramp_metrics if m.jitter_score is not None]
    avg_jitter = float(np.mean(jitter_scores)) if jitter_scores else 0.0
    
    taper_scores = [m.taper_score for m in ramp_metrics if m.taper_score is not None]
    avg_taper = float(np.mean(taper_scores)) if taper_scores else 0.0
    
    stability_scores = [m.stability_score for m in dwell_metrics if m.stability_score is not None]
    avg_dwell_stability = float(np.mean(stability_scores)) if stability_scores else 0.0
    
    overshoot_mags = [m.overshoot_magnitude_c for m in dwell_metrics if m.overshoot_magnitude_c is not None]
    max_overshoot = float(np.max(overshoot_mags)) if overshoot_mags else 0.0
    
    total_ramp_time = sum(
        r.duration_seconds for r in cycle.valid_ramps
    ) if cycle.valid_ramps else 0.0
    
    total_dwell_time = sum(
        r.duration_seconds for r in cycle.regions
        if r.primary_classification.value in ("HOT_DWELL", "COLD_DWELL")
    )
    
    metrics = CycleMetrics(
        cycle_id=cycle.cycle_id,
        duration_seconds=cycle.duration_seconds,
        average_heating_slope_c_per_min=avg_heating_slope,
        average_cooling_slope_c_per_min=avg_cooling_slope,
        minimum_heating_slope_c_per_min=min_heating_slope,
        maximum_cooling_slope_c_per_min=max_cooling_slope,
        average_jitter_score=avg_jitter,
        average_taper_score=avg_taper,
        average_dwell_stability=avg_dwell_stability,
        maximum_overshoot_c=max_overshoot,
        total_ramp_time_seconds=total_ramp_time,
        total_dwell_time_seconds=total_dwell_time,
        cycle_to_cycle_drift=cycle.cycle_to_cycle_drift,
        heating_ramp_count=cycle.heating_ramp_count,
        cooling_ramp_count=cycle.cooling_ramp_count,
    )
    
    audit_log.add(AuditEntry(
        timestamp=datetime.now(),
        module_name="cycle_metrics",
        action="compute_cycle_metrics",
        input_reference=cycle.cycle_id,
        output_reference=f"CycleMetrics(duration={cycle.duration_seconds:.0f}s)",
        decision="SUCCESS",
        reason=f"Cycle metrics: avg_heat={avg_heating_slope:.2f}, avg_cool={avg_cooling_slope:.2f}",
        severity=AuditSeverity.INFO,
        category=AuditCategory.METRICS,
    ))
    
    return metrics
