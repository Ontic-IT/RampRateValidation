"""Structural reading of trace files — format detection without interpretation.

Answers ONE question: "where is the data table in this file, and what
surrounds it?" — for text files (csv/tsv/log) and Excel workbooks alike.

Deliberately knows nothing about temperatures, setpoints or time. The
preamble is preserved as metadata (config lines, unit IDs, profile names),
never discarded: it is evidence for later stages and lineage for reporting.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, date, time as dt_time
from io import StringIO
from pathlib import Path

import pandas as pd

from models.errors import InputFormatError

TEXT_DELIMITERS = ["\t", ",", ";", "|"]
EXCEL_SUFFIXES = {".xlsx", ".xlsm", ".xls"}


@dataclass
class StructuredTable:
    """A located data table plus everything structural around it."""

    df: pd.DataFrame
    columns: list[str]
    headerless: bool
    header_row_index: int          # index in source lines/grid; -1 if headerless
    preamble_lines: list[str]      # raw lines above the table (never discarded)
    delimiter: str                 # "\t", ",", ";", "|" or "excel"
    encoding: str
    sheet_name: str | None = None
    dropped_empty_columns: int = 0
    notes: list[str] = field(default_factory=list)


def read_structured(file_path: str) -> StructuredTable:
    """Read any supported trace file into a StructuredTable."""
    path = Path(file_path)
    if not path.exists():
        raise InputFormatError(f"File not found: {file_path}")

    if path.suffix.lower() in EXCEL_SUFFIXES:
        return _read_excel(path)
    return _read_text(path)


# ---------------------------------------------------------------------------
# Text files
# ---------------------------------------------------------------------------

def _read_lines(path: Path) -> tuple[list[str], str]:
    raw = path.read_bytes()
    # An explicit BOM is unambiguous — honour it first.
    if raw[:3] == b"\xef\xbb\xbf":
        return raw.decode("utf-8-sig").splitlines(), "utf-8-sig"
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return raw.decode("utf-16").splitlines(), "utf-16"
    # No BOM: strict UTF-8, then single-byte codecs that (for cp1252's defined
    # slots / latin-1 always) do not fail. UTF-16 is NOT tried without a BOM:
    # it "succeeds" on single-byte data too, collapsing a file into garbage —
    # e.g. a lone 0xB0 degree sign in a header would otherwise divert the whole
    # trace to a bogus UTF-16 decode (one giant line, no data block found).
    for enc in ("utf-8", "cp1252", "latin-1"):
        try:
            return raw.decode(enc).splitlines(), enc
        except (UnicodeDecodeError, UnicodeError):
            continue
    raise InputFormatError(f"Could not decode {path} with any supported encoding")


def _is_numeric_token(s: str) -> bool:
    if not s:
        return False
    try:
        float(s.replace(",", "."))
        return True
    except ValueError:
        return False


def _tokenize(line: str, delim: str) -> list[str]:
    """Split a line and strip trailing empty tokens (trailing delimiters)."""
    parts = [p.strip() for p in line.split(delim)]
    while parts and parts[-1] == "":
        parts.pop()
    return parts


def _choose_delimiter(lines: list[str]) -> str:
    """Pick the delimiter that most consistently splits the data-dense tail."""
    tail = [ln for ln in lines[-60:] if ln.strip()]
    if not tail:
        tail = [ln for ln in lines if ln.strip()][:60]

    best_delim, best_key = ",", (-1, -1.0)
    for d in TEXT_DELIMITERS:
        counts = [len(_tokenize(ln, d)) for ln in tail]
        counts = [c for c in counts if c >= 2]
        if not counts:
            continue
        mode = max(set(counts), key=counts.count)
        consistency = counts.count(mode) / len(counts)
        key = (mode * consistency, consistency)
        if key > best_key:
            best_key, best_delim = key, d
    return best_delim


DATAISH_RE = re.compile(
    r"^(\d{1,4}[-/]\d{1,2}[-/]\d{1,4}([T ]\d{1,2}:\d{2}(:\d{2}(\.\d+)?)?)?"  # date / datetime
    r"|\d{1,2}:\d{2}(:\d{2}(\.\d+)?)?)$"                                      # time of day
)


def _is_dataish_token(s: str) -> bool:
    """Numeric OR date/time-shaped: both are data, not labels."""
    return _is_numeric_token(s) or bool(DATAISH_RE.match(s))


def _line_numeric_fraction(tokens: list[str]) -> float:
    non_empty = [t for t in tokens if t]
    if not non_empty:
        return 0.0
    return sum(1 for t in non_empty if _is_dataish_token(t)) / len(non_empty)


def _find_data_block(token_rows: list[list[str]]) -> tuple[int, int]:
    """Locate (data_start_index, n_cols) — the widest sustained numeric block.

    A data line has >= 2 tokens and a majority of numeric tokens. The block
    starts at the first data line followed by more data lines of the same
    width (single-line runs accepted only for very small files).
    """
    is_data = [
        len(toks) >= 2 and _line_numeric_fraction(toks) >= 0.6
        for toks in token_rows
    ]
    n = len(token_rows)
    min_run = 3 if sum(is_data) >= 3 else 1
    for i in range(n):
        if not is_data[i]:
            continue
        run = 1
        j = i + 1
        while j < n and (is_data[j] or not token_rows[j]) and run < min_run:
            if is_data[j]:
                run += 1
            j += 1
        if run >= min_run:
            widths = [len(token_rows[k]) for k in range(i, min(i + 200, n)) if is_data[k]]
            n_cols = max(set(widths), key=widths.count)
            return i, n_cols
    raise InputFormatError("No numeric data block found in file")


def _find_header(
    token_rows: list[list[str]],
    data_start: int,
    n_cols: int,
) -> int:
    """Scan upward from the data block for its header line. -1 if none."""
    for i in range(data_start - 1, -1, -1):
        toks = token_rows[i]
        if not toks:
            continue
        # The header must plausibly label the data columns and read as text.
        if abs(len(toks) - n_cols) <= 2 and _line_numeric_fraction(toks) < 0.5:
            return i
        # First non-blank line that doesn't qualify ends the search: anything
        # above it is preamble prose, not a column header.
        return -1
    return -1


def _dedupe_names(names: list[str]) -> list[str]:
    """pandas-style mangling of duplicate column names (X1 -> X1.1, X1.2)."""
    seen: dict[str, int] = {}
    out = []
    for name in names:
        if name in seen:
            seen[name] += 1
            out.append(f"{name}.{seen[name]}")
        else:
            seen[name] = 0
            out.append(name)
    return out


def _read_text(path: Path) -> StructuredTable:
    lines, encoding = _read_lines(path)
    if not any(ln.strip() for ln in lines):
        raise InputFormatError(f"File is empty: {path}")

    delim = _choose_delimiter(lines)
    token_rows = [_tokenize(ln, delim) for ln in lines]
    data_start, n_cols = _find_data_block(token_rows)
    header_idx = _find_header(token_rows, data_start, n_cols)

    if header_idx >= 0:
        raw_names = token_rows[header_idx]
        names = [t if t else f"column_{i + 1}" for i, t in enumerate(raw_names)]
        # Header may have fewer tokens than data (trailing unlabelled cols).
        while len(names) < n_cols:
            names.append(f"column_{len(names) + 1}")
        names = _dedupe_names(names[:n_cols])
        preamble = [lines[i] for i in range(header_idx) if lines[i].strip()]
        headerless = False
    else:
        names = [f"column_{i + 1}" for i in range(n_cols)]
        preamble = [lines[i] for i in range(data_start) if lines[i].strip()]
        headerless = True

    body = "\n".join(lines[data_start:])
    df = pd.read_csv(
        StringIO(body),
        sep=delim,
        header=None,
        names=names,
        usecols=range(n_cols),
        skipinitialspace=True,
        skip_blank_lines=True,
        on_bad_lines="skip",
        low_memory=False,
    )

    df, dropped = _drop_empty_columns(df)

    return StructuredTable(
        df=df,
        columns=list(df.columns),
        headerless=headerless,
        header_row_index=header_idx,
        preamble_lines=preamble,
        delimiter=delim,
        encoding=encoding,
        dropped_empty_columns=dropped,
    )


def _drop_empty_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    empty = [c for c in df.columns if df[c].isna().all()]
    if empty:
        df = df.drop(columns=empty)
    return df, len(empty)


# ---------------------------------------------------------------------------
# Excel workbooks
# ---------------------------------------------------------------------------

def _cell_is_dataish(v) -> bool:
    if v is None:
        return False
    if isinstance(v, (int, float)):
        return not pd.isna(v)
    if isinstance(v, (datetime, date, dt_time)):
        return True
    return _is_numeric_token(str(v).strip())


def _sheet_data_score(grid: pd.DataFrame) -> int:
    """Number of rows that look like data rows (>=2 cells, majority dataish)."""
    if grid.empty:
        return 0
    score = 0
    values = grid.values
    for row in values:
        cells = [v for v in row if v is not None and not (isinstance(v, float) and pd.isna(v))]
        if len(cells) < 2:
            continue
        dataish = sum(1 for v in cells if _cell_is_dataish(v))
        if dataish / len(cells) >= 0.6:
            score += 1
    return score


def _read_excel(path: Path) -> StructuredTable:
    try:
        xl = pd.ExcelFile(path)
    except Exception as e:
        raise InputFormatError(f"Could not open workbook {path}: {e}") from e

    # Choose the sheet with the largest data block; a name resembling the
    # file stem is corroborating evidence, used only to break near-ties.
    stem_key = re.sub(r"\W", "", path.stem).lower()
    best: tuple[int, pd.DataFrame, str] | None = None
    for sheet in xl.sheet_names:
        try:
            grid = xl.parse(sheet, header=None)
        except Exception:
            continue
        score = _sheet_data_score(grid)
        if score == 0:
            continue
        name_key = re.sub(r"\W", "", str(sheet)).lower()
        bonus = 1 if (stem_key and name_key and (name_key in stem_key or stem_key in name_key)) else 0
        if best is None or (score + bonus * max(1, score // 20)) > best[0]:
            best = (score + bonus * max(1, score // 20), grid, sheet)

    if best is None:
        raise InputFormatError(f"No sheet in {path} contains a data table")

    _, grid, sheet = best

    # Reuse the text-file logic by treating grid rows as token rows.
    token_rows: list[list[str]] = []
    for row in grid.values:
        toks = ["" if (v is None or (isinstance(v, float) and pd.isna(v))) else str(v).strip() for v in row]
        while toks and toks[-1] == "":
            toks.pop()
        token_rows.append(toks)

    data_start, n_cols = _find_data_block(token_rows)
    header_idx = _find_header(token_rows, data_start, n_cols)

    if header_idx >= 0:
        raw_names = token_rows[header_idx]
        names = [t if t else f"column_{i + 1}" for i, t in enumerate(raw_names)]
        while len(names) < n_cols:
            names.append(f"column_{len(names) + 1}")
        names = _dedupe_names(names[:n_cols])
        preamble = [" ".join(t for t in token_rows[i] if t) for i in range(header_idx) if token_rows[i]]
        headerless = False
    else:
        names = [f"column_{i + 1}" for i in range(n_cols)]
        preamble = [" ".join(t for t in token_rows[i] if t) for i in range(data_start) if token_rows[i]]
        headerless = True

    # Slice the ORIGINAL grid (native cell types preserved: datetimes stay
    # datetimes) rather than the stringified tokens.
    df = grid.iloc[data_start:, :n_cols].copy()
    df.columns = names
    df = df.reset_index(drop=True)
    df = df.dropna(how="all")
    df, dropped = _drop_empty_columns(df)

    return StructuredTable(
        df=df,
        columns=list(df.columns),
        headerless=headerless,
        header_row_index=header_idx,
        preamble_lines=preamble,
        delimiter="excel",
        encoding="binary",
        sheet_name=sheet,
        dropped_empty_columns=dropped,
        notes=[f"selected sheet '{sheet}' of {len(xl.sheet_names)}"],
    )


# ---------------------------------------------------------------------------
# Preamble / filename evidence extraction
# ---------------------------------------------------------------------------

UNIT_ID_RE = re.compile(r"^[A-Za-z]?\d{3,6}L_[A-Za-z0-9]+_?\w*$")
LOG_FILENAME_RE = re.compile(
    r"Log (\d{2})-(\d{2})-(\d{2})\s+(\d{2}) (\d{2}) (\d{2})", re.IGNORECASE
)
CONFIG_DATETIME_RE = re.compile(
    r"\((\d{2})-(\d{2})-(\d{4}) (\d{2}):(\d{2}):(\d{2})\)"
)
DDMMYY_TOKEN_RE = re.compile(r"(?:^|_)(\d{6}|\d{8})(?:_|$|\.)")


def extract_preamble_metadata(preamble_lines: list[str], file_path: str) -> dict[str, str]:
    """Pull lineage evidence out of the preamble and filename.

    Nothing here is required — every key is optional evidence that later
    stages may corroborate against (time anchoring, date disambiguation).
    """
    meta: dict[str, str] = {}
    name = Path(file_path).name

    for i, line in enumerate(preamble_lines):
        stripped = line.strip()
        if not stripped:
            continue
        if UNIT_ID_RE.match(stripped):
            meta["unit_id"] = stripped
        if stripped.lower().startswith("configuration file"):
            meta["configuration_line"] = stripped
            m = CONFIG_DATETIME_RE.search(stripped)
            if m:
                mm, dd, yyyy, hh, mi, ss = m.groups()
                meta["config_datetime"] = f"{yyyy}-{mm}-{dd}T{hh}:{mi}:{ss}"
        if "version" in stripped.lower() and "software_line" not in meta:
            meta["software_line"] = stripped
        if i == 1 and "system" in stripped.lower():
            meta["system_line"] = stripped

    m = LOG_FILENAME_RE.search(name)
    if m:
        dd, mm, yy, hh, mi, ss = m.groups()
        meta["filename_datetime"] = f"20{yy}-{mm}-{dd}T{hh}:{mi}:{ss}"

    m = DDMMYY_TOKEN_RE.search(Path(file_path).stem)
    if m:
        tok = m.group(1)
        if len(tok) == 6:
            meta["filename_date_ddmmyy"] = tok
        else:
            meta["filename_date_ddmmyyyy"] = tok

    if preamble_lines:
        meta["preamble_line_count"] = str(len(preamble_lines))

    return meta


def extract_time_anchor(meta: dict[str, str]) -> datetime | None:
    """Best absolute-time anchor for traces with only relative time."""
    for key in ("filename_datetime", "config_datetime"):
        if key in meta:
            try:
                return datetime.fromisoformat(meta[key])
            except ValueError:
                continue
    for key, fmts in (
        ("filename_date_ddmmyy", "%d%m%y"),
        ("filename_date_ddmmyyyy", "%d%m%Y"),
    ):
        if key in meta:
            try:
                return datetime.strptime(meta[key], fmts)
            except ValueError:
                continue
    return None
