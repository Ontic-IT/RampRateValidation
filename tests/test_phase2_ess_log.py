"""Unit tests for Phase 2 Retrofit 3: ESS .log ingestion."""

import pytest
from pathlib import Path

from inputs.file_loader import load_ess_log_file
from models.domain import RawTraceData, FileMetadata
from models.errors import InputFormatError


class TestLoadEssLogFile:
    def test_preamble_skipped(self, tmp_path):
        f = tmp_path / "test.log"
        f.write_text(
            "ESS Log File v1.0\n"
            "\n"
            "Device: Chamber A\n"
            "\n"
            "Time\tTemperature\tTemp SP\tPressure\tHeater\n"
            "0\t25.0\t30.0\t101325\tON\n"
            "1\t25.5\t30.0\t101300\tON\n"
            "2\t26.0\t30.0\t101280\tON\n"
        )
        raw_trace, metadata = load_ess_log_file(str(f))

        assert isinstance(raw_trace, RawTraceData)
        assert raw_trace.row_count == 3
        assert metadata.detected_preamble_line_count == 4
        assert metadata.header_row_index == 4

    def test_header_detected_by_scanning(self, tmp_path):
        f = tmp_path / "test.log"
        f.write_text(
            "File: run_2025_01.log\n"
            "\n"
            "Time\tTemperature\tSetpoint\n"
            "0\t25.0\t30.0\n"
            "1\t26.0\t30.0\n"
        )
        raw_trace, metadata = load_ess_log_file(str(f))

        assert metadata.header_row_index == 2
        assert raw_trace.row_count == 2
        assert "Time" in raw_trace.columns
        assert "Temperature" in raw_trace.columns

    def test_tab_delimited_parsing(self, tmp_path):
        f = tmp_path / "test.log"
        f.write_text(
            "Time\tTemp\tSP\tAux1\tAux2\n"
            "0\t25.0\t30.0\t100\t200\n"
            "1\t25.5\t30.0\t101\t201\n"
        )
        raw_trace, metadata = load_ess_log_file(str(f))

        assert raw_trace.data[0]["Temp"] == 25.0
        assert raw_trace.data[0]["SP"] == 30.0
        assert raw_trace.data[0]["Aux1"] == 100.0
        assert raw_trace.data[0]["Aux2"] == 200.0

    def test_preserves_all_columns(self, tmp_path):
        f = tmp_path / "test.log"
        f.write_text(
            "Time\tTemp\tSP\tPressure\tCompressor\tValve\n"
            "0\t25.0\t30.0\t101325\t45.0\tOPEN\n"
        )
        raw_trace, metadata = load_ess_log_file(str(f))

        assert len(raw_trace.columns) == 6
        assert "Pressure" in raw_trace.columns
        assert "Compressor" in raw_trace.columns
        assert "Valve" in raw_trace.columns

    def test_setpoint_detection(self, tmp_path):
        f = tmp_path / "test.log"
        f.write_text(
            "Time\tTemperature\tTemp SP\n"
            "0\t25.0\t30.0\n"
        )
        _, metadata = load_ess_log_file(str(f))

        assert metadata.selected_setpoint_channel == "Temp SP"

    def test_auxiliary_channel_count(self, tmp_path):
        f = tmp_path / "test.log"
        f.write_text(
            "Time\tTemp\tSP\tP1\tP2\tP3\tState1\tState2\n"
            "0\t25.0\t30.0\t1.0\t2.0\t3.0\tON\tOFF\n"
        )
        _, metadata = load_ess_log_file(str(f))

        assert metadata.auxiliary_channel_count == 5  # P1, P2, P3, State1, State2

    def test_no_header_raises(self, tmp_path):
        f = tmp_path / "test.log"
        f.write_text(
            "0\t25.0\t30.0\n"
            "1\t26.0\t30.0\n"
        )
        with pytest.raises(InputFormatError, match="Could not detect header"):
            load_ess_log_file(str(f))

    def test_empty_file_raises(self, tmp_path):
        f = tmp_path / "test.log"
        f.write_text("")
        with pytest.raises(InputFormatError, match="File is empty"):
            load_ess_log_file(str(f))

    def test_file_not_found_raises(self):
        with pytest.raises(InputFormatError, match="File not found"):
            load_ess_log_file("/nonexistent/path.log")

    def test_variably_labelled_setpoint(self, tmp_path):
        f = tmp_path / "test.log"
        f.write_text(
            "Time\tChamber_Temp\tTarget_Temp\n"
            "0\t25.0\t30.0\n"
        )
        _, metadata = load_ess_log_file(str(f))

        assert metadata.selected_setpoint_channel == "Target_Temp"

    def test_35_plus_auxiliary_channels(self, tmp_path):
        """Test that 35+ auxiliary channels are all preserved."""
        headers = ["Time", "Temp", "SP"] + [f"Aux{i}" for i in range(35)]
        data_line = "\t".join(["0", "25.0", "30.0"] + [str(float(i)) for i in range(35)])
        f = tmp_path / "test.log"
        f.write_text("\t".join(headers) + "\n" + data_line + "\n")

        raw_trace, metadata = load_ess_log_file(str(f))

        assert len(raw_trace.columns) == 38  # Time + Temp + SP + 35 aux
        assert metadata.auxiliary_channel_count == 35
        assert raw_trace.row_count == 1

    def test_elapsed_seconds_time_format(self, tmp_path):
        f = tmp_path / "test.log"
        f.write_text(
            "Elapsed\tTemp\tSP\n"
            "0\t25.0\t30.0\n"
            "60\t30.0\t30.0\n"
        )
        _, metadata = load_ess_log_file(str(f))

        assert metadata.detected_timestamp_format == "elapsed_seconds"

    def test_binary_state_values_preserved(self, tmp_path):
        f = tmp_path / "test.log"
        f.write_text(
            "Time\tTemp\tSP\tHeater\tCompressor\n"
            "0\t25.0\t30.0\tON\tOFF\n"
            "1\t25.5\t30.0\tON\tON\n"
        )
        raw_trace, _ = load_ess_log_file(str(f))

        assert raw_trace.data[0]["Heater"] == "ON"
        assert raw_trace.data[0]["Compressor"] == "OFF"
        assert raw_trace.data[1]["Compressor"] == "ON"
