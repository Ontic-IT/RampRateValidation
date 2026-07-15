"""Profile comparison and consistency analysis (M12).

Handles:
- Cycle metric distribution comparison (all 10 metrics)
- Cycle-to-cycle drift detection
- Profile consistency scoring
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np

from config.constants import AuditCategory, AuditSeverity
from models.domain import (
    AuditEntry,
    AuditLog,
    CycleMetrics,
    DwellMetrics,
    MetricSet,
    ProfileComparisonResults,
    RampMetrics,
)


METRIC_KEYS = [
    "ramp_shape",
    "ramp_duration",
    "robust_rate",
    "dwell_duration",
    "dwell_median",
    "overshoot_magnitude",
    "settling_time",
    "cycle_duration",
    "jitter_score",
    "taper_score",
]


def compare_cycle_metric_distribution(
    metric_set: MetricSet,
    regions: Any = None,
    audit_log: AuditLog | None = None,
) -> dict[str, dict[str, float]]:
    """Compare cycle metric distributions across all detected cycles.

    Computes mean, std, min, max for each of the 10 key metrics:
    - ramp_shape: robust_slope_c_per_min from ramp metrics
    - ramp_duration: stall_duration_seconds from ramp metrics
    - robust_rate: robust_slope_c_per_min from ramp metrics (same as ramp_shape)
    - dwell_duration: Region.duration_seconds for dwell regions
    - dwell_median: mean_temperature_c from dwell metrics
    - overshoot_magnitude: overshoot_magnitude_c from dwell metrics
    - settling_time: settling_time_seconds from dwell metrics
    - cycle_duration: duration_seconds from cycle metrics
    - jitter_score: jitter_score from ramp metrics
    - taper_score: taper_score from ramp metrics

    Args:
        metric_set: Computed metrics from Phase 5
        regions: RegionList (optional, for dwell_duration extraction)
        audit_log: Optional audit log

    Returns:
        Dict mapping metric_name -> {mean, std, min, max}
    """
    if audit_log is None:
        audit_log = AuditLog()

    comparison: dict[str, dict[str, float]] = {}

    # Extract metric arrays
    ramp_slopes = [m.robust_slope_c_per_min for m in metric_set.ramp_metrics if m.robust_slope_c_per_min != 0.0]
    ramp_durations = [m.stall_duration_seconds for m in metric_set.ramp_metrics]
    dwell_medians = [m.mean_temperature_c or 0.0 for m in metric_set.dwell_metrics]
    overshoots = [m.overshoot_magnitude_c or 0.0 for m in metric_set.dwell_metrics]
    settling_times = [m.settling_time_seconds or 0.0 for m in metric_set.dwell_metrics]
    jitter_scores = [m.jitter_score or 0.0 for m in metric_set.ramp_metrics]
    taper_scores = [m.taper_score or 0.0 for m in metric_set.ramp_metrics]
    cycle_durations = [m.duration_seconds for m in metric_set.cycle_metrics]

    # Dwell duration: extract from regions if available
    dwell_durations = []
    if regions:
        from config.constants import RegionType
        dwell_durations = [
            r.duration_seconds for r in regions.regions
            if r.primary_classification in (RegionType.HOT_DWELL, RegionType.COLD_DWELL)
        ]
    if not dwell_durations:
        # Fallback: use settling_time if regions not provided
        dwell_durations = [m.settling_time_seconds or 0.0 for m in metric_set.dwell_metrics]

    arrays = {
        "ramp_shape": ramp_slopes,
        "ramp_duration": ramp_durations,
        "robust_rate": ramp_slopes,
        "dwell_duration": dwell_durations,
        "dwell_median": dwell_medians,
        "overshoot_magnitude": overshoots,
        "settling_time": settling_times,
        "cycle_duration": cycle_durations,
        "jitter_score": jitter_scores,
        "taper_score": taper_scores,
    }

    for key, values in arrays.items():
        arr = np.array(values) if values else np.array([0.0])
        comparison[key] = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
        }

    audit_log.add(AuditEntry(
        timestamp=datetime.now(),
        module_name="profile_comparison",
        action="compare_cycle_metric_distribution",
        decision="SUCCESS",
        reason=f"Compared {len(metric_set.cycle_metrics)} cycles across {len(METRIC_KEYS)} metrics",
        severity=AuditSeverity.INFO,
        category=AuditCategory.METRICS,
    ))

    return comparison


def detect_cycle_to_cycle_drift(
    cycle_metrics: list[CycleMetrics],
    audit_log: AuditLog | None = None,
) -> dict[str, Any]:
    """Detect drift between consecutive cycles.

    Args:
        cycle_metrics: List of cycle metrics
        audit_log: Optional audit log

    Returns:
        Dict with drift_detected, drift_metrics, max_drift_cycle_id
    """
    if audit_log is None:
        audit_log = AuditLog()

    if len(cycle_metrics) < 2:
        audit_log.add(AuditEntry(
            timestamp=datetime.now(),
            module_name="profile_comparison",
            action="detect_cycle_to_cycle_drift",
            decision="SKIPPED",
            reason="Less than 2 cycles — drift detection requires multiple cycles",
            severity=AuditSeverity.INFO,
            category=AuditCategory.METRICS,
        ))
        return {"drift_detected": False, "drift_metrics": {}, "max_drift_cycle_id": None}

    drift_metrics: dict[str, list[float]] = {
        "heating_slope": [],
        "cooling_slope": [],
        "dwell_stability": [],
        "cycle_duration": [],
    }

    for i in range(1, len(cycle_metrics)):
        prev = cycle_metrics[i - 1]
        curr = cycle_metrics[i]

        if prev.average_heating_slope_c_per_min != 0.0:
            drift = abs(curr.average_heating_slope_c_per_min - prev.average_heating_slope_c_per_min)
            drift_metrics["heating_slope"].append(drift)

        if prev.average_cooling_slope_c_per_min != 0.0:
            drift = abs(curr.average_cooling_slope_c_per_min - prev.average_cooling_slope_c_per_min)
            drift_metrics["cooling_slope"].append(drift)

        drift_metrics["dwell_stability"].append(abs(curr.average_dwell_stability - prev.average_dwell_stability))
        drift_metrics["cycle_duration"].append(abs(curr.duration_seconds - prev.duration_seconds))

    max_drift = 0.0
    max_drift_metric = ""
    for metric, values in drift_metrics.items():
        if values:
            metric_max = max(values)
            if metric_max > max_drift:
                max_drift = metric_max
                max_drift_metric = metric

    drift_detected = max_drift > 0.5

    audit_log.add(AuditEntry(
        timestamp=datetime.now(),
        module_name="profile_comparison",
        action="detect_cycle_to_cycle_drift",
        decision="DRIFT_DETECTED" if drift_detected else "NO_DRIFT",
        reason=f"Max drift {max_drift:.2f} in {max_drift_metric}" if drift_detected else "No significant drift detected",
        severity=AuditSeverity.WARNING if drift_detected else AuditSeverity.INFO,
        category=AuditCategory.METRICS,
    ))

    return {
        "drift_detected": drift_detected,
        "drift_metrics": {k: float(np.mean(v)) if v else 0.0 for k, v in drift_metrics.items()},
        "max_drift_metric": max_drift_metric,
    }


def compute_profile_consistency_score(
    metric_set: MetricSet,
    audit_log: AuditLog | None = None,
) -> float:
    """Compute overall profile consistency score (0-1).

    Higher = more consistent cycles.

    Args:
        metric_set: Computed metrics
        audit_log: Optional audit log

    Returns:
        Consistency score between 0.0 and 1.0
    """
    if audit_log is None:
        audit_log = AuditLog()

    if not metric_set.cycle_metrics:
        audit_log.add(AuditEntry(
            timestamp=datetime.now(),
            module_name="profile_comparison",
            action="compute_profile_consistency_score",
            decision="SKIPPED",
            reason="No cycle metrics available",
            severity=AuditSeverity.INFO,
            category=AuditCategory.METRICS,
        ))
        return 0.0

    # Coefficient of variation for heating slopes (lower = more consistent)
    heating_slopes = [m.average_heating_slope_c_per_min for m in metric_set.cycle_metrics if m.average_heating_slope_c_per_min != 0.0]
    if heating_slopes and len(heating_slopes) > 1:
        cv_heating = np.std(heating_slopes) / np.mean(heating_slopes)
        heating_consistency = max(0.0, 1.0 - cv_heating)
    else:
        heating_consistency = 1.0

    # Coefficient of variation for cycle durations
    durations = [m.duration_seconds for m in metric_set.cycle_metrics if m.duration_seconds > 0]
    if durations and len(durations) > 1:
        cv_duration = np.std(durations) / np.mean(durations)
        duration_consistency = max(0.0, 1.0 - cv_duration)
    else:
        duration_consistency = 1.0

    score = (heating_consistency + duration_consistency) / 2.0

    audit_log.add(AuditEntry(
        timestamp=datetime.now(),
        module_name="profile_comparison",
        action="compute_profile_consistency_score",
        decision="SUCCESS",
        reason=f"Profile consistency score: {score:.2f}",
        thresholds_used={"score": score},
        severity=AuditSeverity.INFO,
        category=AuditCategory.METRICS,
    ))

    return float(score)
