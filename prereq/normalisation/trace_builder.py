"""Canonical trace builder (M02).

Transforms RawTraceData into CanonicalTrace with all 18 required fields.
Handles timestamp parsing, temperature conversion, and elapsed time derivation.
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
    FileMetadata,
    RawTraceData,
)
from models.errors import InputFormatError
from inputs.parsers import (
    convert_temperature_to_celsius,
    parse_timestamps,
    find_time_column,
)


def build_canonical_trace(
    raw_trace: RawTraceData,
    file_metadata: FileMetadata,
    audit_log: AuditLog | None = None,
) -> CanonicalTrace:
    """Build canonical trace from raw trace data.
    
    Args:
        raw_trace: Raw trace data with all original columns
        file_metadata: Detected file metadata
        audit_log: Optional audit log for recording decisions
    
    Returns:
        CanonicalTrace with all 18 fields populated
    
    Raises:
        InputFormatError: If trace cannot be normalised
    """
    if audit_log is None:
        audit_log = AuditLog()
    
    if raw_trace.row_count == 0:
        raise InputFormatError("Cannot normalise empty trace")
    
    time_col = find_time_column(raw_trace.columns)
    if time_col is None:
        time_col = raw_trace.columns[0]
    
    temp_col = file_metadata.selected_temperature_channel
    sp_col = file_metadata.selected_setpoint_channel
    
    time_values = [str(row.get(time_col, "")) for row in raw_trace.data]
    timestamps, ts_format = parse_timestamps(
        time_values,
        format_hint=file_metadata.detected_timestamp_format,
    )
    
    if not timestamps:
        raise InputFormatError("Failed to parse any timestamps")
    
    temp_values = []
    for row in raw_trace.data:
        val = row.get(temp_col)
        try:
            temp_values.append(float(val) if val is not None else np.nan)
        except (ValueError, TypeError):
            temp_values.append(np.nan)
    
    temp_array = np.array(temp_values, dtype=np.float64)
    
    if file_metadata.detected_temperature_unit != "C":
        temp_array = convert_temperature_to_celsius(
            temp_array,
            file_metadata.detected_temperature_unit,
        )
        audit_log.add(AuditEntry(
            timestamp=datetime.now(),
            module_name="trace_builder",
            action="temperature_conversion",
            decision="CONVERTED",
            reason=f"Converted from {file_metadata.detected_temperature_unit} to Celsius",
            severity=AuditSeverity.INFO,
            category=AuditCategory.PIPELINE,
        ))
    
    setpoint_values = None
    if sp_col:
        setpoint_values = []
        for row in raw_trace.data:
            val = row.get(sp_col)
            try:
                setpoint_values.append(float(val) if val is not None else None)
            except (ValueError, TypeError):
                setpoint_values.append(None)
    
    base_time = timestamps[0]
    elapsed_seconds = []
    for ts in timestamps:
        delta = (ts - base_time).total_seconds()
        elapsed_seconds.append(delta)
    
    elapsed_array = np.array(elapsed_seconds, dtype=np.float64)
    elapsed_minutes = elapsed_array / 60.0
    
    sample_intervals = np.diff(elapsed_array)
    if len(sample_intervals) > 0:
        median_interval = float(np.median(sample_intervals[sample_intervals > 0])) if np.any(sample_intervals > 0) else 1.0
    else:
        median_interval = 1.0
    
    channel_name = temp_col or "unknown"
    
    rows = []
    for i in range(len(timestamps)):
        row = CanonicalTraceRow(
            timestamp=timestamps[i],
            elapsed_seconds=elapsed_seconds[i],
            elapsed_minutes=elapsed_minutes[i],
            temperature_c_raw=float(temp_array[i]) if not np.isnan(temp_array[i]) else 0.0,
            temperature_c_analysis_signal=None,
            setpoint_c=setpoint_values[i] if setpoint_values else None,
            channel=channel_name,
            source_row=i,
            source_file=file_metadata.source_file_path,
            sample_interval_seconds=median_interval,
            local_slope_c_per_min=None,
            rolling_slope_c_per_min=None,
            rolling_temperature_median=None,
            rolling_temperature_MAD=None,
            second_derivative=None,
            direction_of_travel=None,
            data_quality_flags=[],
            region_id=None,
            classification_label=None,
        )
        rows.append(row)
    
    audit_log.add(AuditEntry(
        timestamp=datetime.now(),
        module_name="trace_builder",
        action="canonical_trace_built",
        output_reference=f"CanonicalTrace({len(rows)} rows)",
        decision="SUCCESS",
        reason=f"Built canonical trace with {len(rows)} rows, median interval={median_interval:.3f}s",
        rows_used=len(rows),
        severity=AuditSeverity.INFO,
        category=AuditCategory.PIPELINE,
    ))
    
    return CanonicalTrace(rows=rows)
