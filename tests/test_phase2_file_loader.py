"""Unit tests for Phase 2: inputs/file_loader.py — file loading and ingestion."""

import pytest
from pathlib import Path
from datetime import datetime

from inputs.file_loader import load_trace_file
from models.domain import AuditLog, RawTraceData, FileMetadata
from models.errors import InputFormatError


class TestLoadTraceFile:
    @pytest.fixture
    def simple_csv(self, tmp_path):
        f = tmp_path / "simple.csv"
        f.write_text(
            "timestamp,temperature_c,setpoint\n"
            "2025-01-01T00:00:00,25.0,30.0\n"
            "2025-01-01T00:00:01,25.5,30.0\n"
            "2025-01-01T00:00:02,26.0,30.0\n"
        )
        return f

    @pytest.fixture
    def tab_delimited(self, tmp_path):
        f = tmp_path / "tabbed.txt"
        f.write_text(
            "time\ttemp\tsp\n"
            "0\t25.0\t30.0\n"
            "1\t25.5\t30.0\n"
            "2\t26.0\t30.0\n"
        )
        return f

    @pytest.fixture
    def no_setpoint(self, tmp_path):
        f = tmp_path / "no_sp.csv"
        f.write_text(
            "timestamp,temperature\n"
            "2025-01-01T00:00:00,25.0\n"
            "2025-01-01T00:00:01,25.5\n"
        )
        return f

    def test_load_simple_csv(self, simple_csv):
        raw_trace, metadata = load_trace_file(str(simple_csv))
        
        assert isinstance(raw_trace, RawTraceData)
        assert isinstance(metadata, FileMetadata)
        assert raw_trace.row_count == 3
        assert metadata.detected_delimiter == ","
        assert metadata.detected_timestamp_format == "ISO 8601"
        assert metadata.selected_temperature_channel == "temperature_c"
        assert metadata.selected_setpoint_channel == "setpoint"

    def test_load_tab_delimited(self, tab_delimited):
        raw_trace, metadata = load_trace_file(str(tab_delimited))
        
        assert raw_trace.row_count == 3
        assert metadata.detected_delimiter == "\t"
        assert metadata.selected_temperature_channel == "temp"

    def test_no_setpoint_logs_warning(self, no_setpoint):
        audit_log = AuditLog()
        raw_trace, metadata = load_trace_file(str(no_setpoint), audit_log=audit_log)
        
        assert metadata.selected_setpoint_channel is None
        warning_entries = [e for e in audit_log.entries if e.decision == "WARNING"]
        assert len(warning_entries) >= 1

    def test_file_not_found_raises(self):
        with pytest.raises(InputFormatError, match="File not found"):
            load_trace_file("/nonexistent/path.csv")

    def test_preserves_all_columns(self, simple_csv):
        raw_trace, _ = load_trace_file(str(simple_csv))
        
        assert "timestamp" in raw_trace.columns
        assert "temperature_c" in raw_trace.columns
        assert "setpoint" in raw_trace.columns
        assert len(raw_trace.columns) == 3

    def test_preserves_all_rows(self, simple_csv):
        raw_trace, _ = load_trace_file(str(simple_csv))
        
        assert len(raw_trace.data) == 3
        assert raw_trace.data[0]["temperature_c"] == 25.0

    def test_audit_log_records_success(self, simple_csv):
        audit_log = AuditLog()
        load_trace_file(str(simple_csv), audit_log=audit_log)
        
        success_entries = [e for e in audit_log.entries if e.action == "file_loaded"]
        assert len(success_entries) == 1
        assert success_entries[0].decision == "SUCCESS"

    def test_explicit_channel_selection(self, tmp_path):
        f = tmp_path / "multi_channel.csv"
        f.write_text(
            "time,CH1,CH2,CH3\n"
            "0,25.0,26.0,27.0\n"
            "1,25.5,26.5,27.5\n"
        )
        
        _, metadata = load_trace_file(str(f), channel="CH2")
        assert metadata.selected_temperature_channel == "CH2"

    def test_invalid_channel_raises(self, simple_csv):
        with pytest.raises(InputFormatError, match="not found in columns"):
            load_trace_file(str(simple_csv), channel="nonexistent")


class TestLoadSyntheticFixtures:
    """Test loading the 8 Phase 2 synthetic fixtures."""

    @pytest.fixture
    def synthetic_dir(self):
        return Path(__file__).parent.parent / "tests" / "classification_reference_traces" / "synthetic"

    def test_load_clean_heating_ramp(self):
        path = Path(__file__).parent / "classification_reference_traces" / "synthetic" / "syn_clean_heating_ramp.csv"
        if not path.exists():
            pytest.skip("Synthetic fixture not generated yet")
        
        raw_trace, metadata = load_trace_file(str(path))
        assert raw_trace.row_count > 0
        assert metadata.detected_delimiter == ","
        assert metadata.detected_timestamp_format == "ISO 8601"

    def test_load_noisy_heating_ramp(self):
        path = Path(__file__).parent / "classification_reference_traces" / "synthetic" / "syn_noisy_heating_ramp.csv"
        if not path.exists():
            pytest.skip("Synthetic fixture not generated yet")
        
        raw_trace, metadata = load_trace_file(str(path))
        assert raw_trace.row_count > 0

    def test_load_all_8_fixtures(self):
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
            assert raw_trace.row_count > 0, f"{fixture} has no rows"
            assert metadata.selected_temperature_channel is not None, f"{fixture} has no temp channel"
