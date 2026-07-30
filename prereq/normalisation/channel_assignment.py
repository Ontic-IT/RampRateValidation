"""Behavioral channel assignment — eliminate, assign, corroborate.

Identifies the (time, setpoint, temperature) channels of a trace from what
the data DOES, using column names only as a corroborating witness:

1. ELIMINATE by structural fact: non-numeric, constant, sentinel-dominated
   and implausibly-ranged columns disqualify themselves. No weights.
2. ASSIGN by relationship: the setpoint is the piecewise-constant command
   signal; the process temperature is the channel that CONVERGES to it on
   dwells (that is what closed-loop control means). Candidate (setpoint,
   temperature) pairs must satisfy the relationship simultaneously.
3. CORROBORATE with names: when behavior leaves more than one consistent
   assignment, a labelled header resolves it; when names contradict
   behavior, behavior wins and a warning is logged; when neither can
   decide, the file is refused with an explanation — never guessed.

Every decision is recorded as human-readable evidence for the audit log.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, date, time as dt_time

import numpy as np
import pandas as pd

from models.errors import InputFormatError

# Physical bounds for a thermal-chamber signal in Celsius. These are not
# tuning knobs: they express "a chamber temperature", generously.
PLAUSIBLE_MIN_C = -150.0
PLAUSIBLE_MAX_C = 500.0

# Closed-loop convergence: at the end of a long dwell the CONTROL sensor
# reads its setpoint to within a small tolerance. Physically motivated,
# not fitted: a control loop that misses by more than this is not the
# control loop.
DWELL_CONVERGENCE_C = 3.0

MIN_ROWS_FOR_BEHAVIOUR = 50
MIN_DWELL_SAMPLES = 10

TIME_NAME_HINTS = ("time", "elapsed", "timestamp", "date", "seconds")
SP_NAME_HINTS = ("setpoint", "set point", "set_point", "target", "sp", "sv")
TEMP_NAME_HINTS = ("temp", "temperature", "proc", "t/c", "tc", "pv", "actual", "chamber")

DATE_RE = re.compile(r"^\d{1,4}[-/]\d{1,2}[-/]\d{1,4}$")
TIME_RE = re.compile(r"^\d{1,2}:\d{2}(:\d{2}(\.\d+)?)?$")
DATETIME_RE = re.compile(r"^\d{1,4}[-/]\d{1,2}[-/]\d{1,4}[T ]\d{1,2}:\d{2}")


@dataclass
class ColumnFacts:
    """Structural facts about one column. Facts, not scores."""

    name: str
    numeric_fraction: float = 0.0
    n_unique: int = 0
    is_constant: bool = True
    dominant_fraction: float = 1.0
    vmin: float = float("nan")
    vmax: float = float("nan")
    span: float = 0.0
    monotonic_nondecreasing: bool = False
    staircase_fraction: float = 0.0     # fraction of zero first-differences
    is_datetime_dtype: bool = False
    looks_like_date: bool = False       # date-only strings
    looks_like_time: bool = False       # time-of-day-only strings
    looks_like_datetime: bool = False   # full datetime strings


@dataclass
class ChannelAssignment:
    """The outcome: which channels serve which role, and why."""

    time_channel: str | None = None
    time_channel_pair: list[str] = field(default_factory=list)
    time_kind: str = "unknown"  # datetime|datetime_pair|datetime_objects|excel_serial|epoch_seconds|elapsed_seconds
    temperature_channel: str | None = None
    setpoint_channel: str | None = None
    evidence: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def compute_column_facts(df: pd.DataFrame, col: str) -> ColumnFacts:
    s = df[col]
    facts = ColumnFacts(name=str(col))

    if pd.api.types.is_datetime64_any_dtype(s):
        facts.is_datetime_dtype = True
        facts.numeric_fraction = 0.0
        valid = s.dropna()
        facts.n_unique = int(valid.nunique())
        facts.is_constant = facts.n_unique <= 1
        facts.monotonic_nondecreasing = bool(valid.is_monotonic_increasing)
        return facts

    nums = _numeric(s)
    valid = nums.dropna()
    total = max(len(s.dropna()), 1)
    facts.numeric_fraction = len(valid) / total

    if facts.numeric_fraction < 0.5:
        # Textual column: classify a spread sample value-by-value. Excel
        # round-trips leave MIXED cell types (text dates alongside real
        # datetime cells), so a >=90% majority decides, not unanimity.
        sample = s.dropna()
        if len(sample) > 0:
            probe = sample.iloc[:: max(1, len(sample) // 20)].head(20)
            kinds = [_classify_temporal_value(v) for v in probe]
            n = len(kinds)
            facts.looks_like_datetime = kinds.count("datetime") / n >= 0.9 or (
                ("datetime" in kinds) and (kinds.count("datetime") + kinds.count("date")) / n >= 0.9
            )
            facts.looks_like_date = (not facts.looks_like_datetime) and kinds.count("date") / n >= 0.9
            facts.looks_like_time = kinds.count("time") / n >= 0.9
        str_sample = sample.astype(str).str.strip()
        facts.n_unique = int(str_sample.nunique()) if len(str_sample) else 0
        facts.is_constant = facts.n_unique <= 1
        return facts

    facts.n_unique = int(valid.nunique())
    facts.is_constant = facts.n_unique <= 1
    if len(valid) > 0:
        counts = valid.value_counts()
        facts.dominant_fraction = float(counts.iloc[0]) / len(valid)
        facts.vmin = float(valid.min())
        facts.vmax = float(valid.max())
        facts.span = facts.vmax - facts.vmin
        diffs = valid.diff().dropna()
        if len(diffs) > 0:
            facts.monotonic_nondecreasing = bool((diffs >= 0).all()) and facts.span > 0
            facts.staircase_fraction = float((diffs == 0).mean())
    return facts


def _classify_temporal_value(v) -> str:
    """Classify one cell as date / time / datetime / other, whatever its type."""
    if isinstance(v, dt_time):
        return "time"
    if isinstance(v, (pd.Timestamp, datetime)):
        return "date" if (v.hour == 0 and v.minute == 0 and v.second == 0) else "datetime"
    if isinstance(v, date):
        return "date"
    s = str(v).strip()
    if DATETIME_RE.match(s):
        return "datetime"
    if DATE_RE.match(s):
        return "date"
    if TIME_RE.match(s):
        return "time"
    return "other"


# ---------------------------------------------------------------------------
# Time identification
# ---------------------------------------------------------------------------

def _name_hint(name: str, hints: tuple[str, ...]) -> bool:
    lowered = name.lower().replace("_", " ")
    tokens = re.split(r"[^a-z/]+", lowered)
    return any(h in lowered if len(h) > 2 else h in tokens for h in hints)


def _sp_named(name: str) -> bool:
    """Setpoint name witness, including the industry SP-suffix convention
    ('ChmTmpSP', 'Temp SP') that token matching cannot see."""
    return _name_hint(str(name), SP_NAME_HINTS) or str(name).lower().replace(" ", "").replace("_", "").endswith("sp")


def identify_time(
    df: pd.DataFrame,
    facts: dict[str, ColumnFacts],
    assignment: ChannelAssignment,
) -> None:
    """Fill time_channel / time_kind on the assignment. Raises if undecidable."""

    # 1. Native datetime columns (Excel workbooks).
    dt_cols = [c for c, f in facts.items() if f.is_datetime_dtype]
    # A date-only datetime column + separate time column is a PAIR, handled below.
    date_like_dt = [c for c in dt_cols if _is_date_only(df[c])]
    full_dt = [c for c in dt_cols if c not in date_like_dt]
    if full_dt:
        chosen = _pick_named(full_dt, TIME_NAME_HINTS, assignment, "time")
        assignment.time_channel = chosen
        assignment.time_kind = "datetime_objects"
        assignment.evidence.append(f"time: '{chosen}' is a native datetime column")
        return

    # 2. Full datetime strings.
    dt_str = [c for c, f in facts.items() if f.looks_like_datetime]
    if dt_str:
        chosen = _pick_named(dt_str, TIME_NAME_HINTS, assignment, "time")
        assignment.time_channel = chosen
        assignment.time_kind = "datetime"
        assignment.evidence.append(f"time: '{chosen}' contains datetime strings")
        return

    # 3. Date column + time column composed as a pair.
    date_cols = [c for c, f in facts.items() if f.looks_like_date] + date_like_dt
    time_cols = [c for c, f in facts.items() if f.looks_like_time]
    if date_cols and time_cols:
        d, t = date_cols[0], time_cols[0]
        assignment.time_channel = d
        assignment.time_channel_pair = [d, t]
        assignment.time_kind = "datetime_pair"
        assignment.evidence.append(
            f"time: composed from date column '{d}' + time column '{t}'"
        )
        if len(date_cols) > 1 or len(time_cols) > 1:
            assignment.warnings.append(
                f"multiple date/time columns present; used '{d}'+'{t}'"
            )
        return

    # 4. Numeric monotonic columns: excel serial, epoch or elapsed seconds.
    mono = [
        c for c, f in facts.items()
        if f.numeric_fraction >= 0.99 and f.monotonic_nondecreasing and f.span > 0
    ]
    if mono:
        chosen = _pick_named(mono, TIME_NAME_HINTS, assignment, "time")
        vals = _numeric(df[chosen]).dropna()
        v0 = float(vals.iloc[0])
        if 25569 < v0 < 100000:
            assignment.time_kind = "excel_serial"
            assignment.evidence.append(
                f"time: '{chosen}' is monotonic numeric in the Excel serial-date range"
            )
        elif v0 > 1_000_000_000:
            assignment.time_kind = "epoch_seconds"
            assignment.evidence.append(f"time: '{chosen}' is monotonic numeric in epoch range")
        else:
            assignment.time_kind = "elapsed_seconds"
            assignment.evidence.append(f"time: '{chosen}' is monotonic numeric (elapsed seconds)")
        assignment.time_channel = chosen
        return

    raise InputFormatError(
        "No time channel found: no datetime column, no date+time pair, and no "
        "monotonic numeric column. Columns seen: "
        + ", ".join(list(facts)[:20])
    )


def _is_date_only(s: pd.Series) -> bool:
    sample = s.dropna().head(20)
    if len(sample) == 0:
        return False
    try:
        return bool(((sample.dt.hour == 0) & (sample.dt.minute == 0) & (sample.dt.second == 0)).all())
    except AttributeError:
        return False


def _pick_named(
    candidates: list[str],
    hints: tuple[str, ...],
    assignment: ChannelAssignment,
    role: str,
) -> str:
    """Among behaviorally equal candidates, let names corroborate."""
    if len(candidates) == 1:
        return candidates[0]
    named = [c for c in candidates if _name_hint(str(c), hints)]
    if len(named) == 1:
        assignment.evidence.append(
            f"{role}: {len(candidates)} behavioral candidates; header name resolved to '{named[0]}'"
        )
        return named[0]
    chosen = (named or candidates)[0]
    assignment.warnings.append(
        f"{role}: {len(candidates)} candidates ({', '.join(map(str, candidates[:6]))}); "
        f"names could not fully resolve, used '{chosen}'"
    )
    return chosen


# ---------------------------------------------------------------------------
# Setpoint / temperature assignment
# ---------------------------------------------------------------------------

def _pool(facts: dict[str, ColumnFacts], exclude: set[str]) -> list[str]:
    """Columns that survive elimination: numeric, moving, plausibly thermal."""
    out = []
    for c, f in facts.items():
        if c in exclude:
            continue
        if f.numeric_fraction < 0.9 or f.is_constant:
            continue
        if f.dominant_fraction > 0.95:
            continue  # sentinel-dominated (e.g. -350.0 disconnected sensors)
        if not (PLAUSIBLE_MIN_C <= f.vmin and f.vmax <= PLAUSIBLE_MAX_C):
            continue
        if f.span < 1.0:
            continue  # jitter around a constant is not a trace signal
        out.append(c)
    return out


def _dwell_segments(sp: np.ndarray, min_len: int) -> list[tuple[int, int, float]]:
    """(start, end, value) runs where the setpoint is constant."""
    segments = []
    start = 0
    for i in range(1, len(sp) + 1):
        if i == len(sp) or sp[i] != sp[start]:
            if i - start >= min_len and not np.isnan(sp[start]):
                segments.append((start, i, float(sp[start])))
            start = i
    return segments


def _dwell_convergence(pv: np.ndarray, dwells: list[tuple[int, int, float]]) -> float:
    """Median |pv - sp| over the settled tail of each dwell."""
    errs = []
    for start, end, value in dwells:
        tail_start = max(start, end - max(MIN_DWELL_SAMPLES, (end - start) // 5))
        tail = pv[tail_start:end]
        tail = tail[~np.isnan(tail)]
        if len(tail) == 0:
            continue
        errs.append(abs(float(np.mean(tail)) - value))
    return float(np.median(errs)) if errs else float("inf")


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    mask = ~(np.isnan(a) | np.isnan(b))
    if mask.sum() < 10:
        return 0.0
    a, b = a[mask], b[mask]
    if np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def assign_setpoint_and_temperature(
    df: pd.DataFrame,
    facts: dict[str, ColumnFacts],
    assignment: ChannelAssignment,
) -> None:
    exclude = set(assignment.time_channel_pair or ([assignment.time_channel] if assignment.time_channel else []))
    pool = _pool(facts, exclude)

    n_rows = len(df)

    # -- Small-data / degenerate path: behavior cannot discriminate, names decide.
    if not pool or n_rows < MIN_ROWS_FOR_BEHAVIOUR:
        if not pool:
            _assign_by_names(df, facts, assignment, pool, reason="no column moves within a thermal range")
            return
        _assign_by_names(df, facts, assignment, pool, reason=f"only {n_rows} rows")
        return

    sp_cands = [c for c in pool if facts[c].staircase_fraction >= 0.5]
    pv_cands = pool

    # Behavioral pairing: which (sp, pv) pairs satisfy closed-loop convergence?
    arrays = {c: _numeric(df[c]).to_numpy(dtype=float) for c in pool}
    valid_pairs: dict[str, list[tuple[str, float]]] = {}
    for sp in sp_cands:
        dwells = _dwell_segments(arrays[sp], MIN_DWELL_SAMPLES)
        if len(dwells) < 2:
            continue
        passers = []
        for pv in pv_cands:
            if pv == sp:
                continue
            # A channel IDENTICAL to the command is another command (a
            # mirror like ManChmTSP), never a measurement: real sensors
            # carry noise and lag, so they never equal the setpoint
            # sample-for-sample.
            both = ~(np.isnan(arrays[pv]) | np.isnan(arrays[sp]))
            if both.any() and (arrays[pv][both] == arrays[sp][both]).mean() > 0.98:
                continue
            conv = _dwell_convergence(arrays[pv], dwells)
            if conv <= DWELL_CONVERGENCE_C and _safe_corr(arrays[pv], arrays[sp]) > 0.5:
                passers.append((pv, conv))
        if passers:
            valid_pairs[sp] = sorted(passers, key=lambda x: x[1])

    if not valid_pairs:
        # No closed-loop pair exists: single-channel extract (e.g. bare
        # thermocouple export). Unique pool column is unambiguous.
        assignment.setpoint_channel = None
        if len(pool) == 1:
            assignment.temperature_channel = pool[0]
            assignment.evidence.append(
                f"temperature: '{pool[0]}' is the only moving thermal-range column; "
                "no setpoint present"
            )
        else:
            _assign_by_names(df, facts, assignment, pool, reason="no closed-loop pair found")
        return

    # Choose the setpoint.
    if len(valid_pairs) == 1:
        sp = next(iter(valid_pairs))
        assignment.evidence.append(
            f"setpoint: '{sp}' is the only staircase column with a converging process channel"
        )
    else:
        named = [c for c in valid_pairs if _sp_named(c)]
        if len(named) == 1:
            sp = named[0]
            assignment.evidence.append(
                f"setpoint: {len(valid_pairs)} behavioral candidates; header name resolved to '{sp}'"
            )
        elif named:
            sp = min(named, key=lambda c: valid_pairs[c][0][1])
            assignment.warnings.append(
                f"setpoint: {len(valid_pairs)} behavioral candidates, {len(named)} "
                f"named as setpoints ({', '.join(map(str, named))}); chose '{sp}' "
                "whose process channel converges tightest"
            )
        else:
            sp = min(valid_pairs, key=lambda c: valid_pairs[c][0][1])
            assignment.warnings.append(
                f"setpoint: {len(valid_pairs)} candidates ({', '.join(valid_pairs)}); "
                f"chose '{sp}' whose process channel converges tightest (no name evidence)"
            )
    assignment.setpoint_channel = str(sp)

    # Choose the process temperature among that setpoint's passers.
    passers = valid_pairs[sp]
    if len(passers) == 1:
        pv, conv = passers[0]
        assignment.evidence.append(
            f"temperature: '{pv}' uniquely converges to '{sp}' on dwells "
            f"(median error {conv:.2f}degC)"
        )
    else:
        pv, conv = _corroborate_pv(sp, passers, assignment, arrays)
    assignment.temperature_channel = str(pv)

    name_says_temp = _name_hint(str(pv), TEMP_NAME_HINTS)
    if not name_says_temp:
        assignment.warnings.append(
            f"temperature: behavior selected '{pv}' although its name does not "
            "suggest a temperature — behavior wins, flagging for review"
        )


CONTROL_PV_HINTS = ("proc", "process", "chamber", "chm", "ctrl", "control", "pv", "act", "actual")


def _corroborate_pv(
    sp: str,
    passers: list[tuple[str, float]],
    assignment: ChannelAssignment,
    arrays: dict[str, np.ndarray],
) -> tuple[str, float]:
    """More than one channel converges: names corroborate, tightest wins otherwise."""
    names = [p for p, _ in passers]

    # Strongest witness: the channel named like the setpoint minus its SP token
    # ("Temp SP" -> "Temp").
    sp_base = re.sub(r"(?i)\b(sp|setpoint|set ?point|target|sv)\b", "", str(sp)).strip(" _-")
    if sp_base:
        exact = [p for p in names if str(p).strip().lower() == sp_base.lower()]
        if len(exact) == 1:
            conv = dict(passers)[exact[0]]
            assignment.evidence.append(
                f"temperature: {len(passers)} channels converge to '{sp}'; "
                f"'{exact[0]}' matches the setpoint's own name — accepted"
            )
            return exact[0], conv

    named = [p for p in names if _name_hint(str(p), TEMP_NAME_HINTS)]
    if len(named) == 1:
        conv = dict(passers)[named[0]]
        assignment.evidence.append(
            f"temperature: {len(passers)} converging channels; header name resolved to '{named[0]}'"
        )
        return named[0], conv

    best_pv, best_conv = passers[0]
    tied = [p for p, c in passers if abs(c - best_conv) < 0.05]
    if len(tied) > 1:
        # Next witness: a name marking the CONTROL-side sensor among the tie.
        ctrl_named = [p for p in tied if _name_hint(str(p), CONTROL_PV_HINTS)]
        if len(ctrl_named) == 1:
            conv = dict(passers)[ctrl_named[0]]
            assignment.evidence.append(
                f"temperature: {len(tied)} channels tie on convergence to '{sp}'; "
                f"'{ctrl_named[0]}' is named as the control-side sensor — accepted"
            )
            return ctrl_named[0], conv

        # Ambiguity only matters if it changes the answer: if the tied
        # channels agree with EACH OTHER, either one is the same trace.
        a, b = arrays[tied[0]], arrays[tied[1]]
        both = ~(np.isnan(a) | np.isnan(b))
        mutual = float(np.median(np.abs(a[both] - b[both]))) if both.any() else float("inf")
        if mutual <= DWELL_CONVERGENCE_C:
            assignment.warnings.append(
                f"temperature: {len(tied)} channels ({', '.join(map(str, tied[:4]))}) "
                f"converge to '{sp}' equally and agree with each other "
                f"(median mutual difference {mutual:.2f}degC); chose '{best_pv}' — "
                "outcome is equivalent for any of them"
            )
            return best_pv, best_conv

        raise InputFormatError(
            f"Ambiguous temperature channel: {', '.join(map(str, tied[:4]))} all "
            f"converge to setpoint '{sp}' equally well but DISAGREE with each "
            f"other (median mutual difference {mutual:.2f}degC), and names do "
            "not disambiguate. Specify the channel explicitly."
        )

    assignment.warnings.append(
        f"temperature: {len(passers)} channels converge to '{sp}' "
        f"({', '.join(f'{p}={c:.2f}degC' for p, c in passers[:5])}); "
        f"chose '{best_pv}' (tightest convergence)"
    )
    return best_pv, best_conv


def _assign_by_names(
    df: pd.DataFrame,
    facts: dict[str, ColumnFacts],
    assignment: ChannelAssignment,
    pool: list[str],
    reason: str,
) -> None:
    """Degenerate path: name witness first, then the only sensible default.

    Name matching considers every numeric column (a constant column named
    'setpoint' in a short extract IS the setpoint — too little data for
    behavior to speak); the no-name fallback stays restricted to the pool.
    """
    exclude = set(assignment.time_channel_pair or ([assignment.time_channel] if assignment.time_channel else []))
    named_pool = [
        c for c, f in facts.items()
        if c not in exclude and f.numeric_fraction >= 0.5
    ]
    temp_named = [c for c in named_pool if _name_hint(str(c), TEMP_NAME_HINTS)]
    sp_named = [c for c in named_pool if _name_hint(str(c), SP_NAME_HINTS)]
    # A column matching both hint sets ("Temp SP") is a setpoint, not a temperature.
    temp_named = [c for c in temp_named if c not in sp_named]

    if temp_named:
        assignment.temperature_channel = str(temp_named[0])
        assignment.evidence.append(
            f"temperature: named '{temp_named[0]}' ({reason}; behavioral tests unavailable)"
        )
    else:
        non_sp = [c for c in pool if c not in sp_named] or [c for c in named_pool if c not in sp_named]
        if not non_sp:
            raise InputFormatError(
                f"No temperature channel identifiable: {reason}, and no column "
                "name suggests a temperature. Specify the channel explicitly."
            )
        fallback = non_sp[0]
        assignment.temperature_channel = str(fallback)
        assignment.warnings.append(
            f"temperature: no behavioral or name evidence ({reason}); "
            f"used sole/first moving column '{fallback}'"
        )

    if sp_named:
        assignment.setpoint_channel = str(sp_named[0])
        assignment.evidence.append(f"setpoint: named '{sp_named[0]}' ({reason})")
    else:
        assignment.setpoint_channel = None


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def assign_channels(df: pd.DataFrame) -> ChannelAssignment:
    """Assign time/setpoint/temperature roles for a structured table."""
    if df.empty or len(df.columns) == 0:
        raise InputFormatError("Cannot assign channels: table is empty")

    facts = {c: compute_column_facts(df, c) for c in df.columns}
    assignment = ChannelAssignment()

    eliminated = [
        c for c, f in facts.items()
        if f.is_constant or (f.numeric_fraction < 0.5 and not (
            f.looks_like_date or f.looks_like_time or f.looks_like_datetime or f.is_datetime_dtype))
    ]
    if eliminated:
        assignment.evidence.append(
            f"eliminated {len(eliminated)} constant/non-signal columns "
            f"({', '.join(map(str, eliminated[:8]))}{'...' if len(eliminated) > 8 else ''})"
        )

    identify_time(df, facts, assignment)
    assign_setpoint_and_temperature(df, facts, assignment)
    return assignment
