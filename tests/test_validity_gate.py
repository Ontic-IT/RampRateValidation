"""Tests for the ingest validity gate (prereq/normalisation/validity_gate.py)."""

from datetime import datetime, timedelta

import pytest

from models.domain import AuditLog, CanonicalTrace, CanonicalTraceRow, FileMetadata
from prereq.normalisation.validity_gate import evaluate_ingest_validity


def _make_trace(
    n_rows: int,
    interval_s: float = 6.0,
    temp_fn=lambda i: 25.0,
    sp_fn=None,
    elapsed_fn=None,
) -> CanonicalTrace:
    base = datetime(2026, 1, 1)
    rows = []
    for i in range(n_rows):
        elapsed = elapsed_fn(i) if elapsed_fn else i * interval_s
        rows.append(CanonicalTraceRow(
            timestamp=base + timedelta(seconds=elapsed),
            elapsed_seconds=elapsed,
            elapsed_minutes=elapsed / 60.0,
            temperature_c_raw=temp_fn(i),
            setpoint_c=sp_fn(i) if sp_fn else None,
            channel="temp",
            source_row=i,
            source_file="test",
            sample_interval_seconds=interval_s,
        ))
    return CanonicalTrace(rows=rows)


def _meta() -> FileMetadata:
    return FileMetadata(
        source_file_path="test",
        selected_temperature_channel="temp",
        selected_setpoint_channel="sp",
    )


class TestInvalidVerdicts:
    def test_fragment_few_rows_is_invalid(self):
        trace = _make_trace(5)
        report = evaluate_ingest_validity(trace, _meta())
        assert report.verdict == "INVALID"
        assert any("fragment" in f for f in report.flags)

    def test_seconds_long_capture_is_invalid(self):
        trace = _make_trace(20, interval_s=1.0)  # 19 seconds
        report = evaluate_ingest_validity(trace, _meta())
        assert report.verdict == "INVALID"
        assert any("aborted" in f for f in report.flags)

    def test_backwards_time_is_invalid(self):
        trace = _make_trace(
            600,
            elapsed_fn=lambda i: (600 - i) * 6.0 if i > 300 else i * 6.0,
        )
        report = evaluate_ingest_validity(trace, _meta())
        assert report.verdict == "INVALID"
        assert any("backwards" in f for f in report.flags)


class TestFlaggedVerdicts:
    def test_small_commanded_excursion_flagged(self):
        # Vib2-like: controlled variable spans only ~6 units over 20 min
        trace = _make_trace(
            200,
            temp_fn=lambda i: (i % 60) / 10.0,
            sp_fn=lambda i: 6.0 if i > 30 else 0.0,
        )
        report = evaluate_ingest_validity(trace, _meta())
        assert report.verdict == "FLAGGED"
        assert any("not a temperature ramp test" in f for f in report.flags)

    def test_short_duration_with_real_excursion_is_partial_ramp_note(self):
        # 20 minutes but with a genuine thermal excursion: analysable
        # partial ramp — noted, not flagged.
        trace = _make_trace(
            200,
            temp_fn=lambda i: 20.0 + i * 0.3,
            sp_fn=lambda i: 20.0 + i * 0.3,
        )
        report = evaluate_ingest_validity(trace, _meta())
        assert report.verdict == "VALID"
        assert any("partial ramp" in n for n in report.notes)

    def test_no_setpoint_dwell_only_flagged(self):
        trace = _make_trace(2000, temp_fn=lambda i: 25.0 + (i % 3) * 0.1)
        report = evaluate_ingest_validity(trace, _meta())
        assert report.verdict == "FLAGGED"
        assert any("dwell/soak" in f for f in report.flags)

    def test_irregular_cadence_flagged(self):
        trace = _make_trace(
            2000,
            temp_fn=lambda i: 20.0 + (i % 1000) * 0.1,
            sp_fn=lambda i: 20.0 if i < 1000 else 85.0,
            elapsed_fn=lambda i: i * 6.0 + (i % 3) * 2.0,  # jittered but monotonic clock
        )
        report = evaluate_ingest_validity(trace, _meta())
        assert report.verdict == "FLAGGED"
        assert any("cadence" in f for f in report.flags)


class TestClockCorrections:
    def test_isolated_small_backwards_step_is_valid_with_note(self):
        # A single ~55s logger clock resync (as seen in real SSC captures)
        trace = _make_trace(
            2000,
            temp_fn=lambda i: i * 0.05,
            sp_fn=lambda i: i * 0.05,
            elapsed_fn=lambda i: i * 6.0 - (55.0 if i >= 1000 else 0.0),
        )
        report = evaluate_ingest_validity(trace, _meta())
        assert report.verdict == "VALID"
        assert any("clock correction" in n for n in report.notes)

    def test_large_backwards_step_flagged(self):
        trace = _make_trace(
            2000,
            temp_fn=lambda i: i * 0.05,
            sp_fn=lambda i: i * 0.05,
            elapsed_fn=lambda i: i * 6.0 - (600.0 if i >= 1000 else 0.0),
        )
        report = evaluate_ingest_validity(trace, _meta())
        assert report.verdict == "FLAGGED"
        assert any("backwards" in f for f in report.flags)


class TestValidVerdict:
    def test_realistic_ramp_test_is_valid(self):
        # 4 hours, -20..85 commanded, steady 6s cadence
        def sp(i):
            cycle = i % 1200
            return -20.0 if cycle < 600 else 85.0

        def temp(i):
            return sp(i) + 1.5  # tracking with offset

        trace = _make_trace(2400, temp_fn=temp, sp_fn=sp)
        report = evaluate_ingest_validity(trace, _meta())
        assert report.verdict == "VALID"
        assert report.flags == []
        assert report.commanded_excursion_c == pytest.approx(105.0)

    def test_valid_report_metrics_populated(self):
        trace = _make_trace(2400, temp_fn=lambda i: i * 0.05, sp_fn=lambda i: i * 0.05)
        report = evaluate_ingest_validity(trace, _meta())
        assert report.row_count == 2400
        assert report.duration_seconds == pytest.approx(2399 * 6.0)
        assert report.median_interval_seconds == pytest.approx(6.0)
        assert report.irregular_interval_fraction == 0.0

    def test_audit_entry_recorded(self):
        audit = AuditLog()
        trace = _make_trace(2400, temp_fn=lambda i: i * 0.05, sp_fn=lambda i: i * 0.05)
        evaluate_ingest_validity(trace, _meta(), audit_log=audit)
        entries = [e for e in audit.entries if e.module_name == "validity_gate"]
        assert len(entries) == 1
        assert entries[0].decision == "VALID"
