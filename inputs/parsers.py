"""Parsing utilities for trace file ingestion.

Handles delimiter detection, encoding detection, timestamp parsing,
and temperature unit conversion.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import Literal

import numpy as np
import pandas as pd


def detect_encoding(file_path: str) -> str:
    """Detect file encoding by reading first bytes.
    
    Returns 'utf-8' for most files, 'utf-16' if BOM detected.
    """
    with open(file_path, "rb") as f:
        raw = f.read(4)
    
    if raw.startswith(b"\xff\xfe") or raw.startswith(b"\xfe\xff"):
        return "utf-16"
    if raw.startswith(b"\xef\xbb\xbf"):
        return "utf-8-sig"
    return "utf-8"


def detect_delimiter(file_path: str, encoding: str = "utf-8") -> str:
    """Detect CSV delimiter by analyzing first few lines.
    
    Supports: comma, tab, semicolon, pipe.
    """
    delimiters = [",", "\t", ";", "|"]
    
    with open(file_path, "r", encoding=encoding) as f:
        lines = [f.readline() for _ in range(5)]
    
    sample = "".join(lines)
    
    counts = {d: sample.count(d) for d in delimiters}
    best = max(counts, key=counts.get)
    
    if counts[best] == 0:
        return ","
    
    return best


def detect_header_rows(file_path: str, encoding: str = "utf-8", delimiter: str = ",") -> int:
    """Detect number of non-data header rows to skip before the column header.
    
    Returns the number of rows to skip before the header row.
    Row 0 is typically the column header, so returns 0 in most cases.
    Returns > 0 if there are metadata rows before the column header.
    """
    with open(file_path, "r", encoding=encoding) as f:
        lines = [f.readline() for _ in range(20)]
    
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        parts = line.split(delimiter)
        non_numeric_count = sum(1 for p in parts if p.strip() and not _is_numeric(p.strip()))
        if non_numeric_count >= 2:
            return i
        numeric_count = sum(1 for p in parts if _is_numeric(p.strip()))
        if numeric_count >= 2 and i == 0:
            return 0
    
    return 0


def _is_numeric(s: str) -> bool:
    """Check if string represents a numeric value."""
    if not s:
        return False
    try:
        float(s.replace(",", "."))
        return True
    except ValueError:
        pass
    if _is_excel_serial_date(s):
        return True
    return False


def _is_excel_serial_date(s: str) -> bool:
    """Check if string looks like an Excel serial date (e.g., 45000.5)."""
    try:
        val = float(s)
        return 25569 < val < 100000
    except ValueError:
        return False


TIMESTAMP_PATTERNS = [
    (r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}", "%Y-%m-%dT%H:%M:%S", "ISO 8601"),
    (r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}\.\d+", "%Y-%m-%dT%H:%M:%S.%f", "ISO 8601 with microseconds"),
    (r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", "%Y-%m-%d %H:%M:%S", "ISO 8601 space"),
    (r"\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2}", None, "DD/MM/YYYY or MM/DD/YYYY"),
    (r"\d{2}-\d{2}-\d{4} \d{2}:\d{2}:\d{2}", None, "DD-MM-YYYY or MM-DD-YYYY"),
    (r"\d{2}/\d{2}/\d{4}", None, "DD/MM/YYYY date only"),
    (r"\d+\.\d+", None, "Excel serial date"),
    (r"\d+", None, "Epoch seconds or elapsed"),
]


def detect_timestamp_format(sample_values: list[str]) -> str:
    """Detect timestamp format from sample values.
    
    Returns format string or descriptive name.
    """
    if not sample_values:
        return "unknown"
    
    sample = str(sample_values[0]).strip()
    
    if re.match(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}\.\d+", sample):
        return "ISO 8601 with microseconds"
    if re.match(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}", sample):
        return "ISO 8601"
    if re.match(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", sample):
        return "ISO 8601 space"
    if re.match(r"\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2}", sample):
        first_part = int(sample.split("/")[0])
        if first_part > 12:
            return "DD/MM/YYYY HH:MM:SS"
        return "MM/DD/YYYY HH:MM:SS"
    if re.match(r"\d{2}/\d{2}/\d{4}", sample):
        first_part = int(sample.split("/")[0])
        if first_part > 12:
            return "DD/MM/YYYY"
        return "MM/DD/YYYY"
    if _is_excel_serial_date(sample):
        return "Excel serial date"
    if re.match(r"^\d+$", sample):
        val = int(sample)
        if val > 1000000000:
            return "Epoch seconds"
        return "Elapsed seconds"
    if re.match(r"^\d+\.\d+$", sample):
        val = float(sample)
        if 25569 < val < 100000:
            return "Excel serial date"
        if val > 1000000000:
            return "Epoch seconds"
        return "Elapsed seconds"
    
    return "unknown"


def parse_timestamps(
    values: list[str],
    format_hint: str = "auto",
) -> tuple[list[datetime], str]:
    """Parse timestamp strings to datetime objects.
    
    Returns (list of datetimes, detected format string).
    """
    if not values:
        return [], "unknown"
    
    if format_hint == "auto":
        format_hint = detect_timestamp_format(values)
    
    results = []
    
    for v in values:
        v = str(v).strip()
        dt = _parse_single_timestamp(v, format_hint)
        results.append(dt)
    
    return results, format_hint


def _parse_single_timestamp(value: str, format_hint: str) -> datetime:
    """Parse a single timestamp value."""
    if format_hint == "ISO 8601":
        value = value.replace("T", " ")
        return datetime.strptime(value[:19], "%Y-%m-%d %H:%M:%S")
    
    if format_hint == "ISO 8601 with microseconds":
        value = value.replace("T", " ")
        return datetime.strptime(value, "%Y-%m-%d %H:%M:%S.%f")
    
    if format_hint == "ISO 8601 space":
        return datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
    
    if format_hint == "DD/MM/YYYY HH:MM:SS":
        return datetime.strptime(value, "%d/%m/%Y %H:%M:%S")
    
    if format_hint == "MM/DD/YYYY HH:MM:SS":
        return datetime.strptime(value, "%m/%d/%Y %H:%M:%S")
    
    if format_hint == "DD/MM/YYYY":
        return datetime.strptime(value, "%d/%m/%Y")
    
    if format_hint == "MM/DD/YYYY":
        return datetime.strptime(value, "%m/%d/%Y")
    
    if format_hint == "Excel serial date":
        serial = float(value)
        excel_epoch = datetime(1899, 12, 30)
        return excel_epoch + timedelta(days=serial)
    
    if format_hint == "Epoch seconds":
        return datetime.fromtimestamp(float(value))
    
    if format_hint == "Elapsed seconds":
        return datetime(2000, 1, 1) + timedelta(seconds=float(value))
    
    try:
        return pd.to_datetime(value).to_pydatetime()
    except Exception:
        return datetime(2000, 1, 1)


TEMPERATURE_PATTERNS = {
    "celsius": [r"temp.*\(?\s*[°]?\s*c\s*\)?", r"temperature.*c", r"tc\d*", r"ch\d+"],
    "fahrenheit": [r"temp.*\(?\s*[°]?\s*f\s*\)?", r"temperature.*f"],
    "kelvin": [r"temp.*\(?\s*k\s*\)?", r"temperature.*k"],
}


def detect_temperature_unit(column_name: str) -> Literal["C", "F", "K"]:
    """Detect temperature unit from column name.
    
    Returns 'C', 'F', or 'K'. Defaults to 'C'.
    """
    col_lower = column_name.lower()
    
    for pattern in TEMPERATURE_PATTERNS["fahrenheit"]:
        if re.search(pattern, col_lower):
            return "F"
    
    for pattern in TEMPERATURE_PATTERNS["kelvin"]:
        if re.search(pattern, col_lower):
            return "K"
    
    return "C"


def convert_temperature_to_celsius(
    values: np.ndarray | list[float],
    from_unit: Literal["C", "F", "K"],
) -> np.ndarray:
    """Convert temperature values to Celsius.
    
    Args:
        values: Temperature values in original unit
        from_unit: Source unit ('C', 'F', or 'K')
    
    Returns:
        Temperature values in Celsius
    """
    arr = np.asarray(values, dtype=np.float64)
    
    if from_unit == "C":
        return arr
    elif from_unit == "F":
        return (arr - 32) * 5 / 9
    elif from_unit == "K":
        return arr - 273.15
    else:
        return arr


def find_temperature_column(columns: list[str]) -> str | None:
    """Find the most likely temperature column from available columns."""
    col_lower = {c.lower(): c for c in columns}
    
    priority_patterns = [
        r"^temp$",
        r"^temperature$",
        r"temp.*c",
        r"temperature.*c",
        r"^tc\d*$",
        r"^ch\d+$",
        r"temp",
        r"temperature",
    ]
    
    for pattern in priority_patterns:
        for lower, original in col_lower.items():
            if re.search(pattern, lower):
                return original
    
    return None


def find_setpoint_column(columns: list[str]) -> str | None:
    """Find the most likely setpoint column from available columns."""
    col_lower = {c.lower(): c for c in columns}
    
    priority_patterns = [
        r"setpoint",
        r"set.*point",
        r"sp$",
        r"target",
        r"sv$",
    ]
    
    for pattern in priority_patterns:
        for lower, original in col_lower.items():
            if re.search(pattern, lower):
                return original
    
    return None


def find_time_column(columns: list[str]) -> str | None:
    """Find the most likely time/timestamp column from available columns."""
    col_lower = {c.lower(): c for c in columns}
    
    priority_patterns = [
        r"^time$",
        r"^timestamp$",
        r"^datetime$",
        r"^date.*time$",
        r"^elapsed",
        r"time",
        r"date",
    ]
    
    for pattern in priority_patterns:
        for lower, original in col_lower.items():
            if re.search(pattern, lower):
                return original
    
    return None
