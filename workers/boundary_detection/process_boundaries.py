"""Process boundary detection (M05).

Handles:
- Ambient region detection
- CUSUM/change-point detection for process start
- Process start/end detection
- Recovery tail detection
- Partial cycle edge handling
"""

from __future__ import annotations

import math
from datetime import datetime

import numpy as np

from config.constants import AuditCategory, AuditSeverity
from models.domain import (
    AuditEntry,
    AuditLog,
    PreprocessedTrace,
    PreprocessingReport,
    ProcessBoundaries,
)


def detect_process_boundaries(
    trace: PreprocessedTrace,
    preprocessing_report: PreprocessingReport,
    minimum_region_duration_seconds: float = 10.0,
    audit_log: AuditLog | None = None,
) -> ProcessBoundaries:
    """Detect process boundaries using CUSUM with adaptive threshold.
    
    Args:
        trace: Preprocessed trace with rolling slopes
        preprocessing_report: Preprocessing statistics
        minimum_region_duration_seconds: Minimum duration for region confirmation
        audit_log: Optional audit log
    
    Returns:
        ProcessBoundaries with all detected indices
    """
    if audit_log is None:
        audit_log = AuditLog()
    
    n_rows = len(trace.rows)
    if n_rows == 0:
        return ProcessBoundaries(detection_method="empty_trace")
    
    sample_interval = preprocessing_report.estimated_sample_interval_s
    if sample_interval <= 0:
        sample_interval = 1.0
    
    slopes = np.array([
        r.rolling_slope_c_per_min if r.rolling_slope_c_per_min is not None else 0.0
        for r in trace.rows
    ])
    
    temperatures = np.array([r.temperature_c_raw for r in trace.rows])
    
    ambient_end = _detect_ambient_end(slopes, temperatures, sample_interval, minimum_region_duration_seconds)
    
    process_start = _cusum_process_start(
        slopes,
        ambient_end_index=ambient_end,
        slope_noise_floor=preprocessing_report.slope_noise_floor_c_per_min,
        sample_interval_s=sample_interval,
        minimum_region_duration_s=minimum_region_duration_seconds,
    )
    
    process_end = _detect_process_end(slopes, temperatures, n_rows, sample_interval, minimum_region_duration_seconds)
    
    recovery_start = _detect_recovery_start(slopes, temperatures, process_end, n_rows)
    
    partial_edges = _detect_partial_cycle_edges(slopes, temperatures, process_start, process_end)
    
    usable_start = process_start
    usable_end = process_end if recovery_start is None else recovery_start
    
    boundaries = ProcessBoundaries(
        ambient_start_index=0,
        ambient_end_index=ambient_end,
        process_start_index=process_start,
        process_end_index=process_end,
        commissioned_loop_start_index=process_start,
        commissioned_loop_end_index=process_end,
        recovery_start_index=recovery_start,
        partial_cycle_edge_indices=partial_edges,
        usable_window_row_range=(usable_start, usable_end),
        detection_method="CUSUM_adaptive",
    )
    
    audit_log.add(AuditEntry(
        timestamp=datetime.now(),
        module_name="process_boundaries",
        action="detect_process_boundaries",
        output_reference=f"ProcessBoundaries(start={process_start}, end={process_end})",
        decision="SUCCESS",
        reason=f"Detected process from row {process_start} to {process_end}, ambient ends at {ambient_end}",
        rows_used=n_rows,
        severity=AuditSeverity.INFO,
        category=AuditCategory.PIPELINE,
    ))
    
    return boundaries


def _detect_ambient_end(
    slopes: np.ndarray,
    temperatures: np.ndarray,
    sample_interval: float,
    min_duration: float,
) -> int:
    """Detect end of ambient region (before process starts)."""
    n = len(slopes)
    if n < 10:
        return 0
    
    min_rows = max(5, int(min_duration / sample_interval))
    
    slope_threshold = np.median(np.abs(slopes[:min(100, n)])) * 3.0
    if slope_threshold < 0.1:
        slope_threshold = 0.1
    
    for i in range(min_rows, n):
        window = slopes[i - min_rows:i]
        if np.median(np.abs(window)) > slope_threshold:
            return max(0, i - min_rows)
    
    return min(n // 4, 100)


def _cusum_process_start(
    slope_series: np.ndarray,
    ambient_end_index: int,
    slope_noise_floor: float,
    sample_interval_s: float,
    minimum_region_duration_s: float,
) -> int:
    """CUSUM detection for process start with adaptive threshold.
    
    Implements the plan specification:
    - cusum_drift = ambient_slope_MAD × 1.0
    - cusum_threshold = ambient_slope_MAD × 5.0, bounded
    - Requires consecutive rows above threshold for confirmation
    """
    n = len(slope_series)
    if n == 0:
        return 0
    
    ambient_window = slice(0, max(1, ambient_end_index))
    ambient_slopes = slope_series[ambient_window]
    
    if len(ambient_slopes) > 0:
        ambient_slope_mad = float(np.median(np.abs(ambient_slopes - np.median(ambient_slopes))))
    else:
        ambient_slope_mad = slope_noise_floor / 5.0
    
    if ambient_slope_mad < 0.01:
        ambient_slope_mad = 0.01
    
    cusum_drift = np.median(ambient_slopes) if len(ambient_slopes) > 0 else 0.0
    
    raw_threshold = ambient_slope_mad * 5.0
    min_threshold = slope_noise_floor * 2.0 if slope_noise_floor > 0 else 0.1
    max_threshold = slope_noise_floor * 10.0 if slope_noise_floor > 0 else 5.0
    cusum_threshold = max(min_threshold, min(max_threshold, raw_threshold))
    
    min_confirm_rows = max(3, int(math.ceil(minimum_region_duration_s / sample_interval_s)))
    
    S = 0.0
    consecutive = 0
    candidate_row = None
    
    for i in range(n):
        slope = abs(slope_series[i])
        S = max(0.0, S + slope - abs(cusum_drift))
        
        if S > cusum_threshold:
            if candidate_row is None:
                candidate_row = i
            consecutive += 1
            if consecutive >= min_confirm_rows:
                return candidate_row
        else:
            S = 0.0
            candidate_row = None
            consecutive = 0
    
    return n - 1


def _detect_process_end(
    slopes: np.ndarray,
    temperatures: np.ndarray,
    n_rows: int,
    sample_interval: float,
    min_duration: float,
) -> int:
    """Detect end of process (return to ambient or end of data)."""
    if n_rows < 10:
        return n_rows - 1
    
    min_rows = max(5, int(min_duration / sample_interval))
    
    slope_threshold = np.median(np.abs(slopes)) * 0.5
    if slope_threshold < 0.1:
        slope_threshold = 0.1
    
    for i in range(n_rows - 1, min_rows, -1):
        window = slopes[i - min_rows:i]
        if np.median(np.abs(window)) > slope_threshold:
            return min(i + min_rows, n_rows - 1)
    
    return n_rows - 1


def _detect_recovery_start(
    slopes: np.ndarray,
    temperatures: np.ndarray,
    process_end: int,
    n_rows: int,
) -> int | None:
    """Detect start of recovery tail (if any)."""
    if process_end >= n_rows - 10:
        return None
    
    tail_slopes = slopes[process_end:]
    if len(tail_slopes) < 5:
        return None
    
    if np.median(np.abs(tail_slopes)) < 0.1:
        return process_end
    
    return None


def _detect_partial_cycle_edges(
    slopes: np.ndarray,
    temperatures: np.ndarray,
    process_start: int,
    process_end: int,
) -> list[int]:
    """Detect partial cycle edges (incomplete cycles at boundaries)."""
    edges = []
    
    if process_start > 0:
        pre_process_slopes = slopes[:process_start]
        if len(pre_process_slopes) > 0 and np.max(np.abs(pre_process_slopes)) > 0.5:
            edges.append(0)
    
    if process_end < len(slopes) - 1:
        post_process_slopes = slopes[process_end:]
        if len(post_process_slopes) > 0 and np.max(np.abs(post_process_slopes)) > 0.5:
            edges.append(process_end)
    
    return edges
