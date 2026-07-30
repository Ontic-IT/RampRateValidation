"""Ingest validity gate — is this trace a ramp-rate test at all?

Runs immediately after normalisation, before any analysis. It answers a
question the channel assignment deliberately does not: the assignment finds
whatever control loop the file contains; this gate checks that the loop is
worth analysing as a temperature ramp test.

Three structural checks, each stated as physics rather than tuning:

1. DURATION — a thermal chamber cannot complete a meaningful ramp+dwell
   cycle in under half an hour; a capture measured in seconds is a fragment
   of an aborted logging session.
2. COMMANDED EXCURSION — a ramp test commands a real temperature change.
   A trace whose setpoint moves only a few degrees is a dwell check or a
   different test type entirely (e.g. Vib2 vibration screens, where the
   controlled variable found by assignment spans ~6 units, not a thermal
   range).
3. CADENCE — the sampling clock must be sane: monotonic, with a stable
   interval. Widespread irregularity means the time base cannot be trusted
   for rate calculations.

Verdicts: VALID (proceed), FLAGGED (proceed, loudly), INVALID (block — the
trace is structurally unanalysable). Everything is explained in the report
and the audit log; nothing is silently dropped.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np

from config.constants import AuditCategory, AuditSeverity
from models.domain import (
    AuditEntry,
    AuditLog,
    CanonicalTrace,
    FileMetadata,
    IngestValidityReport,
)

# Structurally unanalysable below these: no rate can be computed from a
# handful of points or less than a minute of data.
MIN_ANALYSABLE_ROWS = 10
MIN_ANALYSABLE_DURATION_S = 60.0

# A ramp+dwell cycle physically takes tens of minutes; shorter captures are
# fragments or non-ramp activities.
MIN_TEST_DURATION_S = 1800.0

# A temperature ramp test commands tens of degrees of change. Below this
# the trace is a dwell/soak or a non-thermal test.
MIN_COMMAND_EXCURSION_C = 10.0

# Cadence: intervals deviating more than 50% from the median are irregular;
# a trustworthy time base has almost none.
CADENCE_DEVIATION_FACTOR = 0.5
MAX_IRREGULAR_FRACTION = 0.05

# Loggers occasionally resync their clock, producing an isolated small
# backwards step (observed in SSC captures: a single ~55s correction). That
# is routine and handled downstream via out_of_order_rows. The time base is
# only structurally untrustworthy when backwards steps are frequent or huge.
ISOLATED_CLOCK_CORRECTIONS_MAX = 3
CLOCK_CORRECTION_MAX_S = 120.0
STRUCTURAL_BACKWARDS_JUMP_S = 3600.0


def evaluate_ingest_validity(
    canonical_trace: CanonicalTrace,
    file_metadata: FileMetadata,
    audit_log: AuditLog | None = None,
) -> IngestValidityReport:
    """Evaluate whether a normalised trace is an analysable ramp-rate test."""
    if audit_log is None:
        audit_log = AuditLog()

    rows = canonical_trace.rows
    report = IngestValidityReport(row_count=len(rows))
    invalid: list[str] = []
    flags: list[str] = []

    # ---- Duration --------------------------------------------------------
    if rows:
        report.duration_seconds = rows[-1].elapsed_seconds - rows[0].elapsed_seconds

    if len(rows) < MIN_ANALYSABLE_ROWS:
        invalid.append(
            f"only {len(rows)} rows — a fragment, not a trace "
            f"(minimum {MIN_ANALYSABLE_ROWS} to compute any rate)"
        )
    elif report.duration_seconds < MIN_ANALYSABLE_DURATION_S:
        invalid.append(
            f"capture spans {report.duration_seconds:.0f}s — an aborted "
            f"logging session, not a test (minimum {MIN_ANALYSABLE_DURATION_S:.0f}s)"
        )
    short_duration = (
        report.duration_seconds >= MIN_ANALYSABLE_DURATION_S
        and report.duration_seconds < MIN_TEST_DURATION_S
        and len(rows) >= MIN_ANALYSABLE_ROWS
    )

    # ---- Commanded excursion ----------------------------------------------
    temps = np.array(
        [r.temperature_c_raw for r in rows], dtype=float
    ) if rows else np.array([])
    if temps.size:
        report.observed_excursion_c = float(np.nanmax(temps) - np.nanmin(temps))

    setpoints = np.array(
        [r.setpoint_c for r in rows if r.setpoint_c is not None], dtype=float
    )
    thermal_excursion = True
    if setpoints.size:
        report.commanded_excursion_c = float(np.nanmax(setpoints) - np.nanmin(setpoints))
        if report.commanded_excursion_c < MIN_COMMAND_EXCURSION_C:
            thermal_excursion = False
            flags.append(
                f"commanded excursion is {report.commanded_excursion_c:.1f}degC "
                f"(setpoint '{file_metadata.selected_setpoint_channel}') — the "
                "controlled variable does not span a thermal ramp range; this "
                "is not a temperature ramp test "
                f"(expected >= {MIN_COMMAND_EXCURSION_C:.0f}degC)"
            )
    elif temps.size and report.observed_excursion_c < MIN_COMMAND_EXCURSION_C:
        thermal_excursion = False
        flags.append(
            f"no setpoint present and observed temperature excursion is only "
            f"{report.observed_excursion_c:.1f}degC — a dwell/soak capture, "
            f"not a ramp test (expected >= {MIN_COMMAND_EXCURSION_C:.0f}degC)"
        )

    # Duration is corroborative: a short capture WITH a genuine thermal
    # excursion is an analysable partial ramp (note only); a short capture
    # WITHOUT one reinforces "not a ramp test" (flag).
    if short_duration:
        message = (
            f"capture spans {report.duration_seconds / 60:.1f} min — too short "
            f"for a ramp+dwell cycle (expected >= {MIN_TEST_DURATION_S / 60:.0f} min)"
        )
        if thermal_excursion:
            report.notes.append(message + "; genuine thermal excursion present, treating as partial ramp capture")
        else:
            flags.append(message)

    # ---- Cadence sanity ----------------------------------------------------
    if len(rows) >= 3:
        elapsed = np.array([r.elapsed_seconds for r in rows], dtype=float)
        intervals = np.diff(elapsed)
        backwards = intervals[intervals < 0]
        if backwards.size:
            worst = float(-backwards.min())
            if (
                backwards.size > ISOLATED_CLOCK_CORRECTIONS_MAX
                or worst > STRUCTURAL_BACKWARDS_JUMP_S
            ):
                invalid.append(
                    f"{backwards.size} timestamps run backwards (worst "
                    f"{worst:.0f}s) — the time base cannot be trusted for "
                    "rate calculations"
                )
            elif worst > CLOCK_CORRECTION_MAX_S:
                flags.append(
                    f"{backwards.size} backwards timestamp step(s) up to "
                    f"{worst:.0f}s — larger than a routine clock correction"
                )
            else:
                report.notes.append(
                    f"{backwards.size} isolated backwards timestamp step(s) "
                    f"(worst {worst:.0f}s) — routine logger clock correction, "
                    "handled downstream"
                )
        positive = intervals[intervals > 0]
        if positive.size:
            median = float(np.median(positive))
            report.median_interval_seconds = median
            irregular = np.abs(intervals - median) > CADENCE_DEVIATION_FACTOR * median
            report.irregular_interval_fraction = float(irregular.mean())
            if report.irregular_interval_fraction > MAX_IRREGULAR_FRACTION:
                flags.append(
                    f"{report.irregular_interval_fraction:.1%} of sampling "
                    f"intervals deviate >50% from the {median:.1f}s median — "
                    "irregular cadence undermines rate computation"
                )
        else:
            invalid.append("all timestamps identical — no time base")

    # ---- Verdict -----------------------------------------------------------
    report.flags = invalid + flags
    if invalid:
        report.verdict = "INVALID"
    elif flags:
        report.verdict = "FLAGGED"
    else:
        report.verdict = "VALID"

    severity = {
        "VALID": AuditSeverity.INFO,
        "FLAGGED": AuditSeverity.WARNING,
        "INVALID": AuditSeverity.ERROR,
    }[report.verdict]
    audit_log.add(AuditEntry(
        timestamp=datetime.now(),
        module_name="validity_gate",
        action="ingest_validity",
        input_reference=file_metadata.source_file_path,
        decision=report.verdict,
        reason=(
            "; ".join(report.flags)
            if report.flags
            else (
                f"trace spans {report.duration_seconds / 60:.0f} min, commanded "
                f"excursion {report.commanded_excursion_c or report.observed_excursion_c:.1f}degC, "
                f"stable {report.median_interval_seconds:.1f}s cadence"
            )
        ),
        rows_used=report.row_count,
        severity=severity,
        category=AuditCategory.QUALITY,
    ))

    return report
