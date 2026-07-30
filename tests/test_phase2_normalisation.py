"""Unit tests for Phase 2: prereq/normalisation/trace_builder.py — canonical trace generation."""

import pytest
from pathlib import Path
from datetime import datetime

from inputs.file_loader import load_trace_file
from prereq.normalisation.trace_builder import build_canonical_trace
from models.domain import (
    AuditLog,
    CanonicalTrace,
    CanonicalTraceRow,
    FileMetadata,
    RawTraceData,
)
from models.errors import InputFormatError


class TestBuildCanonicalTrace:
    @pytest.fixture
    def simple_raw_trace(self):
        return RawTraceData(
            columns=["timestamp", "temperature_c"],
            data=[
                {"timestamp": "2025-01-01T00:00:00", "temperature_c": 25.0},
                {"timestamp": "2025-01-01T00:00:01", "temperature_c": 25.5},
                {"timestamp": "2025-01-01T00:00:02", "temperature_c": 26.0},
            ],
            row_count=3,
        )

    @pytest.fixture
    def simple_metadata(self):
        return FileMetadata(
            source_file_path="/test/file.csv",
            detected_delimiter=",",
            detected_timestamp_format="ISO 8601",
            detected_temperature_unit="C",
            selected_temperature_channel="temperature_c",
            selected_setpoint_channel=None,
        )

    def test_builds_canonical_trace(self, simple_raw_trace, simple_metadata):
        trace = build_canonical_trace(simple_raw_trace, simple_metadata)
        
        assert isinstance(trace, CanonicalTrace)
        assert len(trace.rows) == 3

    def test_all_18_fields_populated(self, simple_raw_trace, simple_metadata):
        trace = build_canonical_trace(simple_raw_trace, simple_metadata)
        row = trace.rows[0]
        
        assert isinstance(row, CanonicalTraceRow)
        assert row.timestamp == datetime(2025, 1, 1, 0, 0, 0)
        assert row.elapsed_seconds == 0.0
        assert row.elapsed_minutes == 0.0
        assert row.temperature_c_raw == 25.0
        assert row.temperature_c_analysis_signal is None
        assert row.setpoint_c is None
        assert row.channel == "temperature_c"
        assert row.source_row == 0
        assert row.source_file == "/test/file.csv"
        assert row.sample_interval_seconds == 1.0
        assert row.local_slope_c_per_min is None
        assert row.rolling_slope_c_per_min is None
        assert row.rolling_temperature_median is None
        assert row.rolling_temperature_MAD is None
        assert row.second_derivative is None
        assert row.direction_of_travel is None
        assert row.data_quality_flags == []
        assert row.region_id is None
        assert row.classification_label is None

    def test_elapsed_time_derivation(self, simple_raw_trace, simple_metadata):
        trace = build_canonical_trace(simple_raw_trace, simple_metadata)
        
        assert trace.rows[0].elapsed_seconds == 0.0
        assert trace.rows[1].elapsed_seconds == 1.0
        assert trace.rows[2].elapsed_seconds == 2.0
        
        assert trace.rows[0].elapsed_minutes == 0.0
        assert abs(trace.rows[1].elapsed_minutes - 1/60) < 0.001

    def test_source_row_mapping_preserved(self, simple_raw_trace, simple_metadata):
        trace = build_canonical_trace(simple_raw_trace, simple_metadata)
        
        for i, row in enumerate(trace.rows):
            assert row.source_row == i

    def test_temperature_c_raw_preserved(self, simple_raw_trace, simple_metadata):
        trace = build_canonical_trace(simple_raw_trace, simple_metadata)
        
        assert trace.rows[0].temperature_c_raw == 25.0
        assert trace.rows[1].temperature_c_raw == 25.5
        assert trace.rows[2].temperature_c_raw == 26.0

    def test_sample_interval_estimation(self, simple_raw_trace, simple_metadata):
        trace = build_canonical_trace(simple_raw_trace, simple_metadata)
        
        assert trace.rows[0].sample_interval_seconds == 1.0

    def test_empty_trace_raises(self, simple_metadata):
        empty_trace = RawTraceData(columns=[], data=[], row_count=0)
        
        with pytest.raises(InputFormatError, match="empty trace"):
            build_canonical_trace(empty_trace, simple_metadata)

    def test_fahrenheit_conversion(self):
        raw_trace = RawTraceData(
            columns=["time", "temp_f"],
            data=[
                {"time": "0", "temp_f": 77.0},
                {"time": "1", "temp_f": 212.0},
            ],
            row_count=2,
        )
        metadata = FileMetadata(
            source_file_path="/test/file.csv",
            detected_timestamp_format="Elapsed seconds",
            detected_temperature_unit="F",
            selected_temperature_channel="temp_f",
        )
        
        trace = build_canonical_trace(raw_trace, metadata)
        
        assert abs(trace.rows[0].temperature_c_raw - 25.0) < 0.1
        assert abs(trace.rows[1].temperature_c_raw - 100.0) < 0.1

    def test_audit_log_records_build(self, simple_raw_trace, simple_metadata):
        audit_log = AuditLog()
        build_canonical_trace(simple_raw_trace, simple_metadata, audit_log)
        
        build_entries = [e for e in audit_log.entries if e.action == "canonical_trace_built"]
        assert len(build_entries) == 1
        assert build_entries[0].decision == "SUCCESS"

    def test_setpoint_preserved_when_present(self):
        raw_trace = RawTraceData(
            columns=["time", "temp", "setpoint"],
            data=[
                {"time": "0", "temp": 25.0, "setpoint": 30.0},
                {"time": "1", "temp": 26.0, "setpoint": 30.0},
            ],
            row_count=2,
        )
        metadata = FileMetadata(
            source_file_path="/test/file.csv",
            detected_timestamp_format="Elapsed seconds",
            detected_temperature_unit="C",
            selected_temperature_channel="temp",
            selected_setpoint_channel="setpoint",
        )
        
        trace = build_canonical_trace(raw_trace, metadata)
        
        assert trace.rows[0].setpoint_c == 30.0
        assert trace.rows[1].setpoint_c == 30.0


class TestNormaliseSyntheticFixtures:
    """Test normalising the 8 Phase 2 synthetic fixtures."""

    def test_normalise_all_8_fixtures(self):
        fixtures = [
            "syn_clean_heating_ramp.csv",
            "syn_noisy_heating_ramp.csv",
            "syn_clean_cooling_ramp.csv",
            "syn_hot_overshoot.csv",
            "syn_cold_overshoot.csv",
            "syn_ramp_taper.csv",
            "syn_ramp_jitter.csv",
            "syn_partial_dwell.csv",
        ]
        
        base_path = Path(__file__).parent / "classification_reference_traces" / "synthetic"
        
        for fixture in fixtures:
            path = base_path / fixture
            if not path.exists():
                pytest.skip(f"Fixture {fixture} not generated yet")
            
            raw_trace, metadata = load_trace_file(str(path))
            trace = build_canonical_trace(raw_trace, metadata)
            
            assert len(trace.rows) > 0, f"{fixture} produced empty trace"
            assert trace.rows[0].temperature_c_raw is not None
            assert trace.rows[0].source_row == 0
            assert trace.rows[-1].source_row == len(trace.rows) - 1

    def test_clean_heating_ramp_values(self):
        path = Path(__file__).parent / "classification_reference_traces" / "synthetic" / "syn_clean_heating_ramp.csv"
        if not path.exists():
            pytest.skip("Fixture not generated")
        
        raw_trace, metadata = load_trace_file(str(path))
        trace = build_canonical_trace(raw_trace, metadata)
        
        assert trace.rows[0].temperature_c_raw == 25.0
        
        max_temp = max(r.temperature_c_raw for r in trace.rows)
        assert max_temp >= 125.0

    def test_noisy_heating_ramp_has_variation(self):
        path = Path(__file__).parent / "classification_reference_traces" / "synthetic" / "syn_noisy_heating_ramp.csv"
        if not path.exists():
            pytest.skip("Fixture not generated")
        
        raw_trace, metadata = load_trace_file(str(path))
        trace = build_canonical_trace(raw_trace, metadata)
        
        temps = [r.temperature_c_raw for r in trace.rows[:60]]
        temp_std = (sum((t - sum(temps)/len(temps))**2 for t in temps) / len(temps)) ** 0.5
        assert temp_std > 0.1
