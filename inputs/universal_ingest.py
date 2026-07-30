"""Universal file ingestion - format-agnostic data loading.

Philosophy: Ingest ANY tabular data, regardless of format.
No hardcoded column names, positions, or file format assumptions.
"""

from __future__ import annotations

import pandas as pd
from pathlib import Path
from datetime import datetime

from models.domain import RawTraceData, FileMetadata, AuditLog, AuditEntry
from models.errors import InputFormatError
from config.constants import AuditSeverity, AuditCategory


def universal_ingest(
    file_path: str,
    audit_log: AuditLog | None = None,
) -> tuple[RawTraceData, FileMetadata]:
    """Ingest any tabular data file into RawTraceData.
    
    Format-agnostic approach:
    - Try multiple delimiters (tab, comma, semicolon, pipe)
    - Try multiple encodings (utf-8, latin-1, cp1252)
    - Auto-detect where data starts (skip preamble)
    - Preserve ALL columns
    
    Args:
        file_path: Path to any tabular data file
        audit_log: Optional audit log
    
    Returns:
        Tuple of (RawTraceData, FileMetadata)
    
    Raises:
        InputFormatError: If file cannot be parsed as tabular data
    """
    # Delegate to the adaptive loader so the two ingestion paths cannot
    # drift apart: load_trace_file performs the same structural read plus
    # evidence-based channel assignment.
    from inputs.file_loader import load_trace_file

    return load_trace_file(file_path, audit_log=audit_log)


def _legacy_universal_ingest(
    file_path: str,
    audit_log: AuditLog | None = None,
) -> tuple[RawTraceData, FileMetadata]:
    """Pre-adaptive implementation, retained for reference."""
    if audit_log is None:
        audit_log = AuditLog()

    path = Path(file_path)
    if not path.exists():
        raise InputFormatError(f"File not found: {file_path}")

    # Try multiple ingestion strategies
    strategies = [
        {"sep": "\t", "encoding": "utf-8", "name": "tab-delimited UTF-8"},
        {"sep": ",", "encoding": "utf-8", "name": "comma-delimited UTF-8"},
        {"sep": ";", "encoding": "utf-8", "name": "semicolon-delimited UTF-8"},
        {"sep": "\t", "encoding": "latin-1", "name": "tab-delimited Latin-1"},
        {"sep": ",", "encoding": "latin-1", "name": "comma-delimited Latin-1"},
        {"sep": "\t", "encoding": "cp1252", "name": "tab-delimited CP1252"},
    ]
    
    df = None
    successful_strategy = None
    
    for strategy in strategies:
        try:
            df = pd.read_csv(
                file_path,
                sep=strategy["sep"],
                encoding=strategy["encoding"],
                skip_blank_lines=True,
                on_bad_lines="skip",
                low_memory=False,
            )
            
            # Check if we got usable data
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
    if data_start_row > 0:
        df = df.iloc[data_start_row:].reset_index(drop=True)
    
    # Clean column names (strip whitespace)
    df.columns = [str(col).strip() for col in df.columns]
    
    # Convert to RawTraceData
    columns = df.columns.tolist()
    data = df.to_dict("records")
    
    raw_trace = RawTraceData(
        columns=columns,
        data=data,
        row_count=len(data),
    )
    
    # Build metadata
    file_metadata = FileMetadata(
        source_file_path=str(path.absolute()),
        detected_delimiter=successful_strategy["sep"],
        detected_header_rows=1,
        detected_encoding=successful_strategy["encoding"],
        detected_timestamp_format="unknown",  # Will be detected in normalisation
        detected_temperature_unit="unknown",  # Will be detected in normalisation
        available_channels=columns,
        selected_temperature_channel=None,  # Will be detected in normalisation
        selected_setpoint_channel=None,  # Will be detected in normalisation
        raw_row_count=len(data),
        usable_row_count=len(data),
    )
    
    audit_log.add(AuditEntry(
        timestamp=datetime.now(),
        module_name="universal_ingest",
        action="ingest_file",
        input_reference=str(path.absolute()),
        output_reference=f"RawTraceData({len(data)} rows, {len(columns)} cols)",
        decision="SUCCESS",
        reason=f"Ingested using {successful_strategy['name']}, skipped {data_start_row} preamble rows",
        rows_used=len(data),
        severity=AuditSeverity.INFO,
        category=AuditCategory.PIPELINE,
    ))
    
    return raw_trace, file_metadata


def _find_data_start(df: pd.DataFrame) -> int:
    """Find the row where numeric data starts (skip preamble).
    
    Strategy: Find first row where majority of values are numeric.
    
    Args:
        df: DataFrame to analyze
    
    Returns:
        Row index where data starts (0 if no preamble detected)
    """
    for i in range(min(20, len(df))):  # Check first 20 rows
        row = df.iloc[i]
        
        # Count numeric values
        numeric_count = 0
        total_count = 0
        
        for val in row:
            if pd.notna(val):
                total_count += 1
                try:
                    float(str(val).replace(",", "."))
                    numeric_count += 1
                except (ValueError, AttributeError):
                    pass
        
        # If >50% of values are numeric, this is likely data start
        if total_count > 0 and numeric_count / total_count > 0.5:
            return i
    
    return 0  # No preamble detected
