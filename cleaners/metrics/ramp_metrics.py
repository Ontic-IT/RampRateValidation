"""Ramp metrics computation (M10).

Handles:
- Theil-Sen slope calculation (PRIMARY compliance metric)
- Minimum sustained slope calculation (SECONDARY compliance metric)
- Median rolling slope calculation
- Endpoint slope calculation (DIAGNOSTIC ONLY)
- Jitter, taper, and linearity scores
- stall_duration_seconds and slope_MAD calculation
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
from scipy import stats

from config.constants import AuditCategory, AuditSeverity
from models.domain import (
    AuditEntry,
    AuditLog,
    ClassifiedTrace,
    RampMetrics,
    ValidRampRegion,
)


def compute_ramp_metrics(
    valid_ramp: ValidRampRegion,
    classified_trace: ClassifiedTrace,
    sustained_ramp_window_seconds: float = 60.0,
    audit_log: AuditLog | None = None,
) -> RampMetrics:
    """Compute all metrics for a valid ramp region.
    
    Args:
        valid_ramp: Valid ramp region
        classified_trace: Trace with classification labels
        sustained_ramp_window_seconds: Window for sustained slope calculation
        audit_log: Optional audit log
    
    Returns:
        RampMetrics with all computed values
    """
    if audit_log is None:
        audit_log = AuditLog()
    
    rows = [classified_trace.rows[i] for i in valid_ramp.included_rows]
    
    if len(rows) < 3:
        return RampMetrics(
            region_id=valid_ramp.region_id,
            robust_slope_c_per_min=0.0,
            minimum_sustained_slope_c_per_min=0.0,
            median_rolling_slope_c_per_min=0.0,
            endpoint_slope_c_per_min=0.0,
            slope_calculation_method="insufficient_data",
        )
    
    temperatures = np.array([r.temperature_c_raw for r in rows])
    elapsed = np.array([r.elapsed_seconds for r in rows])
    rolling_slopes = np.array([
        r.rolling_slope_c_per_min if r.rolling_slope_c_per_min is not None else 0.0
        for r in rows
    ])
    
    robust_slope = _compute_theil_sen_slope(temperatures, elapsed)
    
    sample_interval = rows[0].sample_interval_seconds if rows else 1.0
    min_sustained = _compute_minimum_sustained_slope(
        temperatures, elapsed, sustained_ramp_window_seconds, sample_interval
    )
    
    median_rolling = float(np.median(rolling_slopes))
    
    endpoint_slope = _compute_endpoint_slope(temperatures, elapsed)
    
    slope_mad = float(stats.median_abs_deviation(rolling_slopes))
    
    jitter_score = _compute_jitter_score(slope_mad, robust_slope)
    
    taper_score = _compute_taper_score(rolling_slopes)
    
    linearity_score = _compute_linearity_score(temperatures, elapsed, robust_slope)
    
    metrics = RampMetrics(
        region_id=valid_ramp.region_id,
        robust_slope_c_per_min=robust_slope,
        minimum_sustained_slope_c_per_min=min_sustained,
        median_rolling_slope_c_per_min=median_rolling,
        endpoint_slope_c_per_min=endpoint_slope,
        slope_MAD=slope_mad,
        jitter_score=jitter_score,
        taper_score=taper_score,
        linearity_score=linearity_score,
        stall_duration_seconds=valid_ramp.stall_duration_seconds,
        reversal_count=valid_ramp.reversal_count,
        monotonicity_score=valid_ramp.monotonicity_score,
        slope_calculation_method="theil_sen",
        sustained_window_seconds_used=sustained_ramp_window_seconds,
    )
    
    audit_log.add(AuditEntry(
        timestamp=datetime.now(),
        module_name="ramp_metrics",
        action="compute_ramp_metrics",
        input_reference=valid_ramp.region_id,
        output_reference=f"RampMetrics(slope={robust_slope:.2f}°C/min)",
        decision="SUCCESS",
        reason=f"Theil-Sen slope={robust_slope:.2f}, min_sustained={min_sustained:.2f}, jitter={jitter_score:.3f}",
        severity=AuditSeverity.INFO,
        category=AuditCategory.METRICS,
        rows_used=len(rows),
    ))
    
    return metrics


def _compute_theil_sen_slope(
    temperatures: np.ndarray,
    elapsed: np.ndarray,
) -> float:
    """Compute Theil-Sen robust slope estimate.
    
    PRIMARY compliance metric per plan specification.
    """
    if len(temperatures) < 2:
        return 0.0
    
    result = stats.theilslopes(temperatures, elapsed)
    slope_per_second = result.slope
    
    return float(slope_per_second * 60.0)


def _compute_minimum_sustained_slope(
    temperatures: np.ndarray,
    elapsed: np.ndarray,
    window_seconds: float,
    sample_interval: float,
) -> float:
    """Compute minimum sustained slope over rolling windows.
    
    SECONDARY compliance metric per plan specification.
    Uses Theil-Sen over each window position.
    """
    if len(temperatures) < 3:
        return 0.0
    
    window_rows = max(3, int(window_seconds / sample_interval))
    
    if window_rows >= len(temperatures):
        return _compute_theil_sen_slope(temperatures, elapsed)
    
    min_slope = float('inf')
    
    for i in range(len(temperatures) - window_rows + 1):
        window_temps = temperatures[i:i + window_rows]
        window_elapsed = elapsed[i:i + window_rows]
        
        slope = _compute_theil_sen_slope(window_temps, window_elapsed)
        
        if abs(slope) < abs(min_slope):
            min_slope = slope
    
    return float(min_slope) if min_slope != float('inf') else 0.0


def _compute_endpoint_slope(
    temperatures: np.ndarray,
    elapsed: np.ndarray,
) -> float:
    """Compute endpoint slope over final 10% of rows.
    
    DIAGNOSTIC ONLY - never used for pass/fail per plan specification.
    """
    if len(temperatures) < 3:
        return 0.0
    
    n_endpoint = max(3, len(temperatures) // 10)
    
    endpoint_temps = temperatures[-n_endpoint:]
    endpoint_elapsed = elapsed[-n_endpoint:]
    
    return _compute_theil_sen_slope(endpoint_temps, endpoint_elapsed)


def _compute_jitter_score(slope_mad: float, robust_slope: float) -> float:
    """Compute jitter score.
    
    Formula: slope_MAD / abs(robust_slope_c_per_min)
    """
    if abs(robust_slope) < 0.01:
        return 1.0
    
    return min(1.0, slope_mad / abs(robust_slope))


def _compute_taper_score(rolling_slopes: np.ndarray) -> float:
    """Compute taper score.
    
    Formula: (first_half_slope - second_half_slope) / abs(first_half_slope)
    """
    if len(rolling_slopes) < 4:
        return 0.0
    
    mid = len(rolling_slopes) // 2
    first_half = rolling_slopes[:mid]
    second_half = rolling_slopes[mid:]
    
    first_half_slope = float(np.median(first_half))
    second_half_slope = float(np.median(second_half))
    
    if abs(first_half_slope) < 0.01:
        return 0.0
    
    return (first_half_slope - second_half_slope) / abs(first_half_slope)


def _compute_linearity_score(
    temperatures: np.ndarray,
    elapsed: np.ndarray,
    robust_slope: float,
) -> float:
    """Compute linearity score (R² of linear fit)."""
    if len(temperatures) < 3:
        return 0.0
    
    slope_per_second = robust_slope / 60.0
    intercept = temperatures[0] - slope_per_second * elapsed[0]
    
    predicted = intercept + slope_per_second * elapsed
    
    ss_res = np.sum((temperatures - predicted) ** 2)
    ss_tot = np.sum((temperatures - np.mean(temperatures)) ** 2)
    
    if ss_tot < 1e-10:
        return 1.0
    
    r_squared = 1.0 - (ss_res / ss_tot)
    
    return max(0.0, min(1.0, r_squared))
