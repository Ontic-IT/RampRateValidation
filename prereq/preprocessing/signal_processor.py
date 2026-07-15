"""Signal processing for trace preprocessing (M03).

Handles:
- Sample interval estimation (median of timestamp differences)
- Hampel spike detection (dual-role: quality flags + analysis signal)
- Rolling statistics (MAD, median) with time-based windows
- Derivative calculation and direction of travel
- Missing sample, duplicate timestamp, and gap detection
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np

from config.constants import AuditCategory, AuditSeverity
from models.domain import (
    AuditEntry,
    AuditLog,
    CanonicalTrace,
    CanonicalTraceRow,
    PreprocessedTrace,
    PreprocessingReport,
)


def preprocess_trace(
    canonical_trace: CanonicalTrace,
    rolling_window_seconds: float = 30.0,
    hampel_k: int = 5,
    hampel_n_sigma: float = 3.0,
    audit_log: AuditLog | None = None,
) -> tuple[PreprocessedTrace, PreprocessingReport]:
    """Preprocess canonical trace with signal processing.
    
    Args:
        canonical_trace: Input canonical trace
        rolling_window_seconds: Window size for rolling statistics
        hampel_k: Hampel filter half-window size (samples each side)
        hampel_n_sigma: Hampel filter threshold in MAD units
        audit_log: Optional audit log
    
    Returns:
        Tuple of (PreprocessedTrace, PreprocessingReport)
    """
    if audit_log is None:
        audit_log = AuditLog()
    
    n_rows = len(canonical_trace.rows)
    if n_rows == 0:
        return PreprocessedTrace(rows=[]), PreprocessingReport()
    
    timestamps = np.array([r.elapsed_seconds for r in canonical_trace.rows])
    temperatures = np.array([r.temperature_c_raw for r in canonical_trace.rows])
    
    sample_interval = _estimate_sample_interval(timestamps)
    
    spike_flags, analysis_signal = _hampel_filter(
        temperatures, k=hampel_k, n_sigma=hampel_n_sigma
    )
    
    window_rows = max(3, int(rolling_window_seconds / sample_interval))
    rolling_median = _rolling_median(temperatures, window_rows)
    rolling_mad = _rolling_mad(temperatures, window_rows)
    
    local_slope = _compute_local_slope(temperatures, timestamps)
    rolling_slope = _rolling_median(local_slope, window_rows)
    
    second_derivative = _compute_second_derivative(local_slope, timestamps)
    
    direction = _compute_direction_of_travel(rolling_slope)
    
    gaps, duplicate_ts, out_of_order = _detect_quality_issues(timestamps, sample_interval)
    
    noise_floor = float(np.median(rolling_mad[:min(100, n_rows)])) if n_rows > 0 else 0.0
    slope_noise_floor = (noise_floor / sample_interval) * 60 * 2.0 if sample_interval > 0 else 0.0
    
    gap_density = len(gaps) / max(1, n_rows)
    dropout_density = sum(g[1] - g[0] for g in gaps) / max(1, n_rows)
    irregular_pct = _compute_irregular_sampling_score(timestamps, sample_interval)
    continuity = 1.0 - dropout_density
    
    preprocessed_rows = []
    for i, row in enumerate(canonical_trace.rows):
        flags = list(row.data_quality_flags)
        if spike_flags[i]:
            flags.append("SPIKE")
        if any(g[0] <= i < g[1] for g in gaps):
            flags.append("GAP")
        if i in duplicate_ts:
            flags.append("DUPLICATE_TIMESTAMP")
        if i in out_of_order:
            flags.append("OUT_OF_ORDER")
        
        new_row = CanonicalTraceRow(
            timestamp=row.timestamp,
            elapsed_seconds=row.elapsed_seconds,
            elapsed_minutes=row.elapsed_minutes,
            temperature_c_raw=row.temperature_c_raw,
            temperature_c_analysis_signal=float(analysis_signal[i]),
            setpoint_c=row.setpoint_c,
            channel=row.channel,
            source_row=row.source_row,
            source_file=row.source_file,
            sample_interval_seconds=sample_interval,
            local_slope_c_per_min=float(local_slope[i]) if not np.isnan(local_slope[i]) else None,
            rolling_slope_c_per_min=float(rolling_slope[i]) if not np.isnan(rolling_slope[i]) else None,
            rolling_temperature_median=float(rolling_median[i]) if not np.isnan(rolling_median[i]) else None,
            rolling_temperature_MAD=float(rolling_mad[i]) if not np.isnan(rolling_mad[i]) else None,
            second_derivative=float(second_derivative[i]) if not np.isnan(second_derivative[i]) else None,
            direction_of_travel=direction[i],
            data_quality_flags=flags,
            region_id=row.region_id,
            classification_label=row.classification_label,
        )
        preprocessed_rows.append(new_row)
    
    report = PreprocessingReport(
        estimated_sample_interval_s=sample_interval,
        noise_floor_c=noise_floor,
        slope_noise_floor_c_per_min=slope_noise_floor,
        temperature_MAD_baseline=float(np.median(rolling_mad)) if n_rows > 0 else 0.0,
        rolling_window_seconds_used=rolling_window_seconds,
        detected_spikes=[i for i, f in enumerate(spike_flags) if f],
        detected_gaps=gaps,
        duplicate_timestamps=list(duplicate_ts),
        out_of_order_rows=list(out_of_order),
        gap_density_score=gap_density,
        dropout_density_score=dropout_density,
        irregular_sampling_score=irregular_pct,
        effective_data_continuity_score=continuity,
    )
    
    audit_log.add(AuditEntry(
        timestamp=datetime.now(),
        module_name="signal_processor",
        action="preprocess_trace",
        output_reference=f"PreprocessedTrace({n_rows} rows)",
        decision="SUCCESS",
        reason=f"Preprocessed {n_rows} rows, {len(report.detected_spikes)} spikes, {len(gaps)} gaps",
        rows_used=n_rows,
        severity=AuditSeverity.INFO,
        category=AuditCategory.PIPELINE,
    ))
    
    return PreprocessedTrace(rows=preprocessed_rows), report


def _estimate_sample_interval(timestamps: np.ndarray) -> float:
    """Estimate sample interval as median of consecutive differences."""
    if len(timestamps) < 2:
        return 1.0
    diffs = np.diff(timestamps)
    positive_diffs = diffs[diffs > 0]
    if len(positive_diffs) == 0:
        return 1.0
    return float(np.median(positive_diffs))


def _hampel_filter(
    values: np.ndarray,
    k: int = 5,
    n_sigma: float = 3.0,
) -> tuple[list[bool], np.ndarray]:
    """Apply Hampel filter for spike detection and signal cleaning.
    
    Args:
        values: Input signal
        k: Half-window size (samples each side)
        n_sigma: Threshold in MAD units
    
    Returns:
        Tuple of (spike_flags, cleaned_signal)
    """
    n = len(values)
    spike_flags = [False] * n
    cleaned = values.copy()
    
    for i in range(n):
        start = max(0, i - k)
        end = min(n, i + k + 1)
        window = values[start:end]
        
        median = np.median(window)
        mad = np.median(np.abs(window - median))
        
        if mad < 1e-10:
            mad = 1e-10
        
        threshold = n_sigma * mad * 1.4826
        
        if abs(values[i] - median) > threshold:
            spike_flags[i] = True
            cleaned[i] = median
    
    return spike_flags, cleaned


def _rolling_median(values: np.ndarray, window: int) -> np.ndarray:
    """Compute rolling median with edge handling."""
    n = len(values)
    result = np.full(n, np.nan)
    half = window // 2
    
    for i in range(n):
        start = max(0, i - half)
        end = min(n, i + half + 1)
        result[i] = np.median(values[start:end])
    
    return result


def _rolling_mad(values: np.ndarray, window: int) -> np.ndarray:
    """Compute rolling MAD (Median Absolute Deviation) with edge handling."""
    n = len(values)
    result = np.full(n, np.nan)
    half = window // 2
    
    for i in range(n):
        start = max(0, i - half)
        end = min(n, i + half + 1)
        w = values[start:end]
        median = np.median(w)
        result[i] = np.median(np.abs(w - median))
    
    return result


def _compute_local_slope(temperatures: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
    """Compute local slope in °C/min using central differences."""
    n = len(temperatures)
    slopes = np.full(n, np.nan)
    
    for i in range(1, n - 1):
        dt = timestamps[i + 1] - timestamps[i - 1]
        if dt > 0:
            dT = temperatures[i + 1] - temperatures[i - 1]
            slopes[i] = (dT / dt) * 60.0
    
    if n > 1:
        dt = timestamps[1] - timestamps[0]
        if dt > 0:
            slopes[0] = ((temperatures[1] - temperatures[0]) / dt) * 60.0
        dt = timestamps[-1] - timestamps[-2]
        if dt > 0:
            slopes[-1] = ((temperatures[-1] - temperatures[-2]) / dt) * 60.0
    
    return slopes


def _compute_second_derivative(slopes: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
    """Compute second derivative (rate of change of slope)."""
    n = len(slopes)
    result = np.full(n, np.nan)
    
    for i in range(1, n - 1):
        dt = timestamps[i + 1] - timestamps[i - 1]
        if dt > 0 and not np.isnan(slopes[i + 1]) and not np.isnan(slopes[i - 1]):
            result[i] = (slopes[i + 1] - slopes[i - 1]) / dt
    
    return result


def _compute_direction_of_travel(rolling_slope: np.ndarray) -> list[str | None]:
    """Determine direction of travel from rolling slope."""
    result = []
    for s in rolling_slope:
        if np.isnan(s):
            result.append(None)
        elif s > 0.1:
            result.append("HEATING")
        elif s < -0.1:
            result.append("COOLING")
        else:
            result.append("STABLE")
    return result


def _detect_quality_issues(
    timestamps: np.ndarray,
    sample_interval: float,
) -> tuple[list[tuple[int, int]], set[int], set[int]]:
    """Detect gaps, duplicate timestamps, and out-of-order rows.
    
    Returns:
        Tuple of (gaps, duplicate_indices, out_of_order_indices)
    """
    n = len(timestamps)
    gaps = []
    duplicates = set()
    out_of_order = set()
    
    gap_threshold = sample_interval * 3.0
    
    for i in range(1, n):
        diff = timestamps[i] - timestamps[i - 1]
        
        if diff < 0:
            out_of_order.add(i)
        elif abs(diff) < 0.001:
            duplicates.add(i)
        elif diff > gap_threshold:
            gaps.append((i - 1, i))
    
    return gaps, duplicates, out_of_order


def _compute_irregular_sampling_score(timestamps: np.ndarray, median_interval: float) -> float:
    """Compute percentage of samples with irregular intervals."""
    if len(timestamps) < 2 or median_interval <= 0:
        return 0.0
    
    diffs = np.diff(timestamps)
    tolerance = median_interval * 0.5
    irregular = np.sum(np.abs(diffs - median_interval) > tolerance)
    return float(irregular / len(diffs))
