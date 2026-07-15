"""File loading and trace ingestion (M01).

Handles multi-format ingestion with automatic detection of:
- Delimiter (CSV, TSV, semicolon, pipe)
- Encoding (UTF-8, UTF-16)
- Header rows
- Timestamp format
- Temperature unit
- Channel columns
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from config.constants import AuditCategory, AuditSeverity
from models.domain import (
    AuditEntry,
    AuditLog,
    FileMetadata,
    RawTraceData,
)
from models.errors import InputFormatError
from inputs.parsers import (
    detect_delimiter,
    detect_encoding,
    detect_header_rows,
    detect_temperature_unit,
    detect_timestamp_format,
    find_temperature_column,
    find_setpoint_column,
    find_time_column,
)


def load_trace_file(
    file_path: str,
    channel: str | None = None,
    setpoint_channel: str | None = None,
    audit_log: AuditLog | None = None,
) -> tuple[RawTraceData, FileMetadata]:
    """Load a trace file using format-agnostic universal ingestion.
    
    Philosophy: Try multiple delimiters and encodings, auto-detect data start,
    preserve all columns. No hardcoded format assumptions.
    
    Args:
        file_path: Path to the trace file
        channel: Optional specific temperature channel to use
        setpoint_channel: Optional specific setpoint channel to use
        audit_log: Optional audit log to record decisions
    
    Returns:
        Tuple of (RawTraceData, FileMetadata)
    
    Raises:
        InputFormatError: If file cannot be parsed or has no usable data
    """
    path = Path(file_path)
    if not path.exists():
        raise InputFormatError(f"File not found: {file_path}")
    
    if audit_log is None:
        audit_log = AuditLog()
    
    # Try multiple ingestion strategies (format-agnostic)
    # Include different skiprows values to handle preambles
    strategies = []
    for sep, enc, name in [
        ("\t", "utf-8", "tab-delimited UTF-8"),
        (",", "utf-8", "comma-delimited UTF-8"),
        (";", "utf-8", "semicolon-delimited UTF-8"),
        ("\t", "latin-1", "tab-delimited Latin-1"),
        (",", "latin-1", "comma-delimited Latin-1"),
    ]:
        for skiprows in [0, 1, 2, 3, 4, 5]:
            strategies.append({
                "sep": sep,
                "encoding": enc,
                "skiprows": skiprows,
                "name": f"{name} (skip {skiprows})"
            })
    
    df = None
    successful_strategy = None
    
    for strategy in strategies:
        try:
            df = pd.read_csv(
                file_path,
                sep=strategy["sep"],
                encoding=strategy["encoding"],
                skiprows=strategy["skiprows"],
                on_bad_lines="skip",
                low_memory=False,
            )
            
            # Check if we got usable data (at least 2 columns)
            if not df.empty and len(df.columns) >= 2:
                successful_strategy = strategy
                break
                
        except Exception:
            continue
    
    if df is None or df.empty:
        raise InputFormatError(
            f"Could not parse {file_path} as tabular data. "
            "Tried tab, comma, semicolon delimiters with multiple encodings."
        )
    
    # Auto-detect where numeric data starts (skip preamble)
    data_start_row = _find_data_start(df)
    preamble_rows = successful_strategy["skiprows"] + data_start_row
    if data_start_row > 0:
        df = df.iloc[data_start_row:].reset_index(drop=True)
    
    # Clean column names (strip whitespace)
    df.columns = [str(col).strip() for col in df.columns]
    
    columns = list(df.columns)
    
    # Content-based column detection (not name-based)
    time_col = _detect_time_column_by_content(df, columns)
    if time_col is None:
        time_col = columns[0] if columns else None
        if time_col is None:
            raise InputFormatError("No time column detected")
    
    timestamp_samples = df[time_col].head(10).astype(str).tolist()
    timestamp_format = detect_timestamp_format(timestamp_samples)
    
    # Temperature column detection
    if channel:
        temp_col = channel
        if temp_col not in columns:
            raise InputFormatError(f"Specified channel '{channel}' not found in columns: {columns}")
    else:
        temp_col = _detect_temperature_column_by_content(df, columns, time_col)
        if temp_col is None:
            raise InputFormatError("No temperature column detected")
    
    temp_unit = detect_temperature_unit(temp_col)
    
    # Setpoint column detection (optional)
    if setpoint_channel:
        sp_col = setpoint_channel
        if sp_col not in columns:
            audit_log.add(AuditEntry(
                timestamp=datetime.now(),
                module_name="file_loader",
                action="setpoint_channel_not_found",
                decision="WARNING",
                reason=f"Specified setpoint channel '{setpoint_channel}' not found",
                severity=AuditSeverity.WARNING,
                category=AuditCategory.QUALITY,
            ))
            sp_col = None
    else:
        sp_col = _detect_setpoint_column_by_content(df, columns, time_col, temp_col)
        if sp_col is None:
            audit_log.add(AuditEntry(
                timestamp=datetime.now(),
                module_name="file_loader",
                action="no_setpoint_channel",
                decision="WARNING",
                reason="No setpoint channel detected - will use Mode B inference",
                severity=AuditSeverity.WARNING,
                category=AuditCategory.QUALITY,
            ))
    
    raw_data = df.to_dict(orient="records")
    
    raw_trace = RawTraceData(
        columns=columns,
        data=raw_data,
        row_count=len(df),
    )
    
    auxiliary_count = len([c for c in columns if c not in (time_col, temp_col, sp_col)])
    
    file_metadata = FileMetadata(
        source_file_path=str(path.absolute()),
        detected_delimiter=successful_strategy["sep"],
        detected_header_rows=1,
        header_row_index=preamble_rows,
        detected_preamble_line_count=preamble_rows,
        detected_encoding=successful_strategy["encoding"],
        detected_timestamp_format=timestamp_format,
        detected_temperature_unit=temp_unit,
        available_channels=[c for c in columns if c != time_col],
        selected_temperature_channel=temp_col,
        selected_setpoint_channel=sp_col,
        raw_row_count=len(df),
        usable_row_count=len(df),
        auxiliary_channel_count=auxiliary_count,
    )
    
    audit_log.add(AuditEntry(
        timestamp=datetime.now(),
        module_name="file_loader",
        action="file_loaded",
        input_reference=str(path.absolute()),
        output_reference=f"RawTraceData({len(df)} rows)",
        decision="SUCCESS",
        reason=f"Loaded using {successful_strategy['name']}, skipped {preamble_rows} preamble rows",
        rows_used=len(df),
        severity=AuditSeverity.INFO,
        category=AuditCategory.PIPELINE,
    ))
    
    return raw_trace, file_metadata


def load_ess_log_file(
    file_path: str,
    audit_log: AuditLog | None = None,
) -> tuple[RawTraceData, FileMetadata]:
    """Load an ESS .log file with scanning header detection.

    Format characteristics:
    - Multi-line preamble before header
    - Header row position is NOT fixed — detected by scanning
    - Tab-delimited
    - Time column is elapsed seconds
    - Setpoint column present, labelled variably (e.g. "Temp SP")
    - 35+ auxiliary channels: pressures, compressor temps, binary states

    Header detection rules:
    - Scan line by line
    - Header line: tab-delimited AND all tokens are non-numeric
    - Following line: tab-delimited AND first token is numeric
    """
    if audit_log is None:
        audit_log = AuditLog()

    path = Path(file_path)
    if not path.exists():
        raise InputFormatError(f"File not found: {file_path}")

    lines = path.read_text(encoding="utf-8").splitlines()

    if not lines:
        raise InputFormatError("File is empty")

    header_index = _scan_for_ess_header(lines)

    if header_index is None:
        raise InputFormatError(
            f"Could not detect header row in {file_path}. "
            "Expected tab-delimited header with non-numeric tokens."
        )

    preamble_count = header_index
    header_line = lines[header_index]
    columns = [c.strip() for c in header_line.split("\t") if c.strip()]

    data: list[dict[str, Any]] = []
    for i in range(header_index + 1, len(lines)):
        line = lines[i].strip()
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) < len(columns):
            continue
        row = {}
        for j, col in enumerate(columns):
            val = parts[j].strip() if j < len(parts) else ""
            try:
                row[col] = float(val.replace(",", "."))
            except ValueError:
                row[col] = val
        data.append(row)

    if not data:
        raise InputFormatError("File contains no data rows")

    raw_trace = RawTraceData(
        columns=columns,
        data=data,
        row_count=len(data),
    )

    # Detect channels
    time_col = _find_ess_time_column(columns)
    temp_col = _find_ess_temperature_column(columns)
    sp_col = _find_ess_setpoint_column(columns)

    if temp_col is None:
        for col in columns:
            if col != time_col:
                sample_vals = [r.get(col) for r in data[:5]]
                if all(isinstance(v, (int, float)) for v in sample_vals if v is not None):
                    temp_col = col
                    break

    if temp_col is None:
        raise InputFormatError("No temperature column detected")

    auxiliary_count = len([c for c in columns if c not in (time_col, temp_col, sp_col)])

    file_metadata = FileMetadata(
        source_file_path=str(path.absolute()),
        detected_delimiter="\t",
        detected_header_rows=1,
        header_row_index=header_index,
        detected_preamble_line_count=preamble_count,
        detected_encoding="utf-8",
        detected_timestamp_format="elapsed_seconds",
        detected_temperature_unit="C",
        available_channels=[c for c in columns if c != time_col],
        selected_temperature_channel=temp_col,
        selected_setpoint_channel=sp_col,
        raw_row_count=len(data),
        usable_row_count=len(data),
        auxiliary_channel_count=auxiliary_count,
    )

    audit_log.add(AuditEntry(
        timestamp=datetime.now(),
        module_name="file_loader",
        action="ess_log_loaded",
        input_reference=str(path.absolute()),
        output_reference=f"RawTraceData({len(data)} rows, {len(columns)} cols)",
        decision="SUCCESS",
        reason=f"Loaded ESS .log: header at line {header_index}, {preamble_count} preamble lines",
        rows_used=len(data),
        severity=AuditSeverity.INFO,
        category=AuditCategory.PIPELINE,
    ))

    return raw_trace, file_metadata


def _scan_for_ess_header(lines: list[str]) -> int | None:
    """Scan for ESS .log header row.

    Header line: tab-delimited AND all tokens are non-numeric.
    Following line: tab-delimited AND first token is numeric.
    """
    for i in range(len(lines) - 1):
        line = lines[i].strip()
        next_line = lines[i + 1].strip()
        if not line or not next_line:
            continue

        parts = line.split("\t")
        next_parts = next_line.split("\t")

        if len(parts) < 2 or len(next_parts) < 2:
            continue

        # Header line: all non-empty tokens non-numeric
        non_empty_parts = [p.strip() for p in parts if p.strip()]
        if not non_empty_parts:
            continue
        all_non_numeric = all(not _is_numeric_string(p) for p in non_empty_parts)

        # Following line: first token numeric
        first_token_numeric = _is_numeric_string(next_parts[0].strip())

        if all_non_numeric and first_token_numeric:
            return i

    return None


def _is_numeric_string(s: str) -> bool:
    """Check if a string represents a numeric value."""
    if not s:
        return False
    try:
        float(s.replace(",", "."))
        return True
    except ValueError:
        return False


def _find_ess_time_column(columns: list[str]) -> str | None:
    """Find elapsed-seconds time column in ESS .log."""
    time_names = ["time", "elapsed", "elapsed_s", "seconds", "elapsed_seconds"]
    for col in columns:
        lower = col.lower().replace(" ", "_")
        if any(name in lower for name in time_names):
            return col
    return columns[0] if columns else None


def _find_ess_temperature_column(columns: list[str]) -> str | None:
    """Find temperature column in ESS .log."""
    temp_names = ["temperature", "temp", "chamber_temp", "chamber_temperature"]
    for col in columns:
        lower = col.lower().replace(" ", "_")
        if any(name in lower for name in temp_names):
            return col
    return None


def _find_ess_setpoint_column(columns: list[str]) -> str | None:
    """Find setpoint column in ESS .log."""
    sp_names = ["setpoint", "sp", "temp_sp", "temperature_sp", "target"]
    for col in columns:
        lower = col.lower().replace(" ", "_")
        if any(name in lower for name in sp_names):
            return col
    return None


def _find_data_start(df: pd.DataFrame) -> int:
    """Find the row where numeric data starts (skip preamble).
    
    Strategy: Find first row where majority of values are numeric.
    """
    for i in range(min(20, len(df))):
        row = df.iloc[i]
        numeric_count = sum(1 for val in row if pd.notna(val) and _is_numeric_value(val))
        total_count = sum(1 for val in row if pd.notna(val))
        
        if total_count > 0 and numeric_count / total_count > 0.5:
            return i
    
    return 0


def _is_numeric_value(val) -> bool:
    """Check if a value is numeric."""
    try:
        float(str(val).replace(",", "."))
        return True
    except (ValueError, AttributeError):
        return False


def _detect_time_column_by_content(df: pd.DataFrame, columns: list[str]) -> str | None:
    """Detect time column by analyzing content, not just name.
    
    Heuristics:
    1. Column name contains time-related keywords (PRIORITY)
    2. OR parseable timestamps
    3. OR monotonically increasing numeric values (LAST RESORT)
    """
    # PRIORITY: Name-based detection first
    time_keywords = ["time", "elapsed", "seconds", "timestamp", "date"]
    for col in columns:
        col_lower = col.lower().replace(" ", "_")
        if any(kw in col_lower for kw in time_keywords):
            return col
    
    # Check for parseable timestamps
    for col in columns:
        try:
            # Try parsing as datetime
            pd.to_datetime(df[col].head(10), errors='raise')
            return col
        except Exception:
            pass
    
    # LAST RESORT: Monotonically increasing numeric values
    for col in columns:
        try:
            values = pd.to_numeric(df[col], errors='coerce')
            
            # Check if monotonically increasing
            if values.notna().sum() > 10:
                diffs = values.diff().dropna()
                if (diffs >= 0).sum() / len(diffs) > 0.99:  # 99% monotonic (stricter)
                    return col
        except Exception:
            pass
    
    return None


def _detect_temperature_column_by_content(
    df: pd.DataFrame, 
    columns: list[str], 
    time_col: str
) -> str | None:
    """Detect temperature column by analyzing content.
    
    Heuristics:
    1. Numeric values in reasonable temperature range (-100 to +200°C)
    2. Has variance (not constant) OR matches temperature keywords
    3. Column name contains temp-related keywords
    """
    candidates = []
    
    for col in columns:
        if col == time_col:
            continue
        
        try:
            values = pd.to_numeric(df[col], errors='coerce')
            valid_values = values.dropna()
            
            if len(valid_values) < 10:
                continue
            
            # Check temperature range
            min_val = valid_values.min()
            max_val = valid_values.max()
            
            if -100 <= min_val <= 200 and -100 <= max_val <= 200:
                # Score by name match
                temp_keywords = ["temp", "temperature", "chamber", "tc", "thermocouple"]
                col_lower = col.lower().replace(" ", "_")
                name_score = sum(1 for kw in temp_keywords if kw in col_lower)
                
                # Check variance (use larger sample if available)
                sample_size = min(1000, len(valid_values))
                variance = valid_values.head(sample_size).var()
                
                # Accept if has variance OR strong name match
                if variance > 0.001 or name_score > 0:
                    candidates.append((col, name_score, variance))
        except Exception:
            continue
    
    if candidates:
        # Sort by name score first, then variance
        candidates.sort(key=lambda x: (x[1], x[2]), reverse=True)
        return candidates[0][0]
    
    # Fallback: if no candidates but we have non-time numeric columns, use first one
    for col in columns:
        if col != time_col:
            try:
                values = pd.to_numeric(df[col], errors='coerce')
                if values.notna().sum() > 10:
                    return col
            except Exception:
                continue
    
    return None


def _detect_setpoint_column_by_content(
    df: pd.DataFrame,
    columns: list[str],
    time_col: str,
    temp_col: str
) -> str | None:
    """Detect setpoint column by analyzing content.
    
    Heuristics:
    1. Numeric values similar to temperature range
    2. Correlated with temperature column
    3. Column name contains setpoint keywords
    """
    if temp_col is None:
        return None
    
    candidates = []
    
    for col in columns:
        if col in (time_col, temp_col):
            continue
        
        try:
            values = pd.to_numeric(df[col], errors='coerce')
            valid_values = values.dropna()
            
            if len(valid_values) < 10:
                continue
            
            # Check if in temperature range
            min_val = valid_values.min()
            max_val = valid_values.max()
            
            if -100 <= min_val <= 200 and -100 <= max_val <= 200:
                # Check correlation with temperature
                temp_values = pd.to_numeric(df[temp_col], errors='coerce')
                correlation = values.corr(temp_values)
                
                if pd.notna(correlation) and abs(correlation) > 0.3:
                    # Score by name match
                    sp_keywords = ["setpoint", "sp", "target", "set_point"]
                    col_lower = col.lower().replace(" ", "_")
                    name_score = sum(1 for kw in sp_keywords if kw in col_lower)
                    
                    candidates.append((col, name_score, abs(correlation)))
        except Exception:
            continue
    
    if candidates:
        # Sort by name score first, then correlation
        candidates.sort(key=lambda x: (x[1], x[2]), reverse=True)
        return candidates[0][0]
    
    return None
