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


# ---------------------------------------------------------------------------
# Evidence-based time decoding (used by the adaptive normalisation path)
# ---------------------------------------------------------------------------

def decode_time_series(
    values: pd.Series,
    kind: str,
    time_values: pd.Series | None = None,
    anchor: datetime | None = None,
    filename_date: datetime | None = None,
) -> tuple[list[datetime], list[str]]:
    """Decode a time channel into datetimes using structural evidence.

    Args:
        values: The primary time column (dates, datetimes, serials or seconds).
        kind: One of datetime|datetime_pair|datetime_objects|excel_serial|
              epoch_seconds|elapsed_seconds (from channel assignment).
        time_values: The time-of-day column when kind == "datetime_pair".
        anchor: Absolute anchor for elapsed_seconds traces.
        filename_date: Independent date evidence used to disambiguate
              day-first vs month-first formats.

    Returns:
        (timestamps, notes) — notes record every disambiguation decision.
    """
    notes: list[str] = []

    if kind == "elapsed_seconds":
        base = anchor or datetime(2000, 1, 1)
        notes.append(
            f"elapsed seconds anchored to {base.isoformat()}"
            + ("" if anchor else " (no anchor evidence found; placeholder epoch)")
        )
        secs = pd.to_numeric(values, errors="coerce")
        return [base + timedelta(seconds=float(s)) for s in secs], notes

    if kind == "epoch_seconds":
        secs = pd.to_numeric(values, errors="coerce")
        return [datetime.fromtimestamp(float(s)) for s in secs], notes

    if kind == "excel_serial":
        serials = pd.to_numeric(values, errors="coerce")
        epoch = datetime(1899, 12, 30)
        return [epoch + timedelta(days=float(s)) for s in serials], notes

    if kind == "datetime_pair":
        if time_values is None:
            raise ValueError("datetime_pair decoding requires the time column")
        combined = _combine_date_time(values, time_values)
        ts = _parse_with_dayfirst_resolution(combined, filename_date, notes)
        return _repair_daymonth_swaps(ts, notes), notes

    if kind == "datetime_objects":
        ts = [v.to_pydatetime() if isinstance(v, pd.Timestamp) else v for v in values]
        return _repair_daymonth_swaps(ts, notes), notes

    # kind == "datetime": full datetime strings
    strings = values.astype(str).str.strip()
    ts = _parse_with_dayfirst_resolution(strings, filename_date, notes)
    return _repair_daymonth_swaps(ts, notes), notes


def _combine_date_time(dates: pd.Series, times: pd.Series) -> pd.Series:
    """Combine a date column and a time column into datetime strings."""
    def date_str(v) -> str:
        if isinstance(v, (pd.Timestamp, datetime)):
            return v.strftime("%m/%d/%Y")  # unambiguous once formatted
        return str(v).strip()

    def time_str(v) -> str:
        if hasattr(v, "strftime") and not isinstance(v, (int, float)):
            return v.strftime("%H:%M:%S")
        return str(v).strip()

    return pd.Series(
        [f"{date_str(d)} {time_str(t)}" for d, t in zip(dates, times)],
        index=dates.index,
    )


def _parse_with_dayfirst_resolution(
    strings: pd.Series,
    filename_date: datetime | None,
    notes: list[str],
) -> list[datetime]:
    """Parse datetime strings, resolving day-first ambiguity by evidence.

    Evidence hierarchy: (1) an interpretation that fails outright is wrong;
    (2) a trace's timestamps are monotonic — the interpretation that keeps
    them so is right; (3) independent filename date evidence; (4) if still
    tied the interpretations agree on every row, so the choice is moot.
    """
    candidates: dict[str, pd.Series | None] = {}
    for label, dayfirst in (("month-first", False), ("day-first", True)):
        parsed = None
        try:
            parsed = pd.to_datetime(strings, dayfirst=dayfirst, errors="raise")
        except (ValueError, TypeError):
            # Excel round-trips can leave MIXED formats in one column
            # (text dates alongside re-typed datetime cells).
            try:
                parsed = pd.to_datetime(strings, dayfirst=dayfirst, format="mixed", errors="raise")
            except (ValueError, TypeError):
                parsed = None
        candidates[label] = parsed

    valid = {k: v for k, v in candidates.items() if v is not None}
    if not valid:
        raise ValueError(f"Could not parse timestamps (sample: {strings.iloc[0]!r})")

    if len(valid) == 2:
        mf, df_ = candidates["month-first"], candidates["day-first"]
        if mf.equals(df_):
            valid = {"month-first": mf}  # unambiguous rows; choice is moot
        else:
            mono = {k: bool(v.is_monotonic_increasing) for k, v in valid.items()}
            if sum(mono.values()) == 1:
                keep = next(k for k, m in mono.items() if m)
                notes.append(f"date order resolved to {keep} by timestamp monotonicity")
                valid = {keep: valid[keep]}
            elif filename_date is not None:
                dist = {
                    k: abs((v.iloc[0].to_pydatetime() - filename_date).total_seconds())
                    for k, v in valid.items()
                }
                keep = min(dist, key=dist.get)
                notes.append(f"date order resolved to {keep} by filename date evidence")
                valid = {keep: valid[keep]}
            else:
                notes.append(
                    "date order ambiguous (both interpretations monotonic, no "
                    "filename evidence); defaulted to month-first"
                )
                valid = {"month-first": valid["month-first"]}

    parsed = next(iter(valid.values()))
    return [v.to_pydatetime() for v in parsed]


def _repair_daymonth_swaps(ts: list[datetime], notes: list[str]) -> list[datetime]:
    """Repair day/month transposition introduced by spreadsheet locales.

    Symptom: a continuous trace whose date field jumps by ~a month at a
    midnight rollover (Feb 2 23:59 -> Mar 2 00:00 instead of Feb 3). The
    repair swaps day/month wherever doing so restores the sampling cadence.
    """
    if len(ts) < 3:
        return ts

    deltas = [(ts[i + 1] - ts[i]).total_seconds() for i in range(len(ts) - 1)]
    positive = sorted(d for d in deltas if d > 0)
    if not positive:
        return ts
    cadence = positive[len(positive) // 2]
    threshold = max(cadence * 100, 3600.0)  # a >100x cadence jump is structural

    repaired = list(ts)
    fixes = 0
    for i in range(len(repaired) - 1):
        gap = (repaired[i + 1] - repaired[i]).total_seconds()
        if abs(gap) <= threshold:
            continue
        nxt = repaired[i + 1]
        if nxt.day > 12:
            continue  # swap impossible
        try:
            swapped = nxt.replace(day=nxt.month, month=nxt.day)
        except ValueError:
            continue
        if abs((swapped - repaired[i]).total_seconds()) <= threshold:
            # Swap forward only while each swap preserves cadence continuity.
            j = i + 1
            while j < len(repaired):
                cur = repaired[j]
                if cur.day > 12:
                    break
                try:
                    candidate = cur.replace(day=cur.month, month=cur.day)
                except ValueError:
                    break
                if abs((candidate - repaired[j - 1]).total_seconds()) > threshold:
                    break
                repaired[j] = candidate
                fixes += 1
                j += 1
    if fixes:
        notes.append(
            f"repaired day/month transposition on {fixes} rows (spreadsheet "
            "locale damage detected via cadence continuity)"
        )
    return repaired


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
