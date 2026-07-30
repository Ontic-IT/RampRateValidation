"""Unit tests for Phase 2: inputs/parsers.py — parsing utilities."""

import pytest
import tempfile
from pathlib import Path
from datetime import datetime

from inputs.parsers import (
    detect_delimiter,
    detect_encoding,
    detect_header_rows,
    detect_temperature_unit,
    detect_timestamp_format,
    parse_timestamps,
    convert_temperature_to_celsius,
    find_temperature_column,
    find_setpoint_column,
    find_time_column,
)


class TestDetectEncoding:
    def test_utf8_default(self, tmp_path):
        f = tmp_path / "test.csv"
        f.write_text("time,temp\n1,25.0\n", encoding="utf-8")
        assert detect_encoding(str(f)) == "utf-8"

    def test_utf8_bom(self, tmp_path):
        f = tmp_path / "test.csv"
        f.write_bytes(b"\xef\xbb\xbftime,temp\n1,25.0\n")
        assert detect_encoding(str(f)) == "utf-8-sig"


class TestDetectDelimiter:
    def test_comma(self, tmp_path):
        f = tmp_path / "test.csv"
        f.write_text("time,temp,setpoint\n1,25.0,30.0\n2,26.0,30.0\n")
        assert detect_delimiter(str(f)) == ","

    def test_tab(self, tmp_path):
        f = tmp_path / "test.csv"
        f.write_text("time\ttemp\tsetpoint\n1\t25.0\t30.0\n2\t26.0\t30.0\n")
        assert detect_delimiter(str(f)) == "\t"

    def test_semicolon(self, tmp_path):
        f = tmp_path / "test.csv"
        f.write_text("time;temp;setpoint\n1;25.0;30.0\n2;26.0;30.0\n")
        assert detect_delimiter(str(f)) == ";"


class TestDetectHeaderRows:
    def test_single_header(self, tmp_path):
        f = tmp_path / "test.csv"
        f.write_text("time,temp\n1,25.0\n2,26.0\n")
        assert detect_header_rows(str(f)) == 0

    def test_no_header(self, tmp_path):
        f = tmp_path / "test.csv"
        f.write_text("1,25.0\n2,26.0\n")
        assert detect_header_rows(str(f)) == 0


class TestDetectTimestampFormat:
    def test_iso8601(self):
        samples = ["2025-01-01T00:00:00", "2025-01-01T00:00:01"]
        assert detect_timestamp_format(samples) == "ISO 8601"

    def test_iso8601_with_microseconds(self):
        samples = ["2025-01-01T00:00:00.123456"]
        assert detect_timestamp_format(samples) == "ISO 8601 with microseconds"

    def test_iso8601_space(self):
        samples = ["2025-01-01 00:00:00"]
        assert detect_timestamp_format(samples) == "ISO 8601 space"

    def test_iso8601_t_separator(self):
        samples = ["2025-01-01T00:00:00"]
        assert detect_timestamp_format(samples) == "ISO 8601"

    def test_dd_mm_yyyy(self):
        samples = ["25/01/2025 10:30:00"]
        assert detect_timestamp_format(samples) == "DD/MM/YYYY HH:MM:SS"

    def test_mm_dd_yyyy(self):
        samples = ["01/25/2025 10:30:00"]
        assert detect_timestamp_format(samples) == "MM/DD/YYYY HH:MM:SS"

    def test_excel_serial(self):
        samples = ["45658.5"]
        assert detect_timestamp_format(samples) == "Excel serial date"

    def test_epoch_seconds(self):
        samples = ["1735689600"]
        assert detect_timestamp_format(samples) == "Epoch seconds"

    def test_elapsed_seconds(self):
        samples = ["0", "1", "2"]
        assert detect_timestamp_format(samples) == "Elapsed seconds"


class TestParseTimestamps:
    def test_iso8601(self):
        values = ["2025-01-01T00:00:00", "2025-01-01T00:00:01"]
        timestamps, fmt = parse_timestamps(values)
        assert len(timestamps) == 2
        assert timestamps[0] == datetime(2025, 1, 1, 0, 0, 0)
        assert fmt == "ISO 8601"

    def test_elapsed_seconds(self):
        values = ["0", "1", "2"]
        timestamps, fmt = parse_timestamps(values, format_hint="Elapsed seconds")
        assert len(timestamps) == 3
        assert (timestamps[1] - timestamps[0]).total_seconds() == 1.0

    def test_excel_serial(self):
        values = ["45658.0"]
        timestamps, fmt = parse_timestamps(values, format_hint="Excel serial date")
        assert len(timestamps) == 1
        assert timestamps[0].year == 2025


class TestDetectTemperatureUnit:
    def test_celsius_explicit(self):
        assert detect_temperature_unit("Temperature (°C)") == "C"
        assert detect_temperature_unit("Temp_C") == "C"

    def test_fahrenheit(self):
        assert detect_temperature_unit("Temperature (°F)") == "F"
        assert detect_temperature_unit("Temp_F") == "F"

    def test_kelvin(self):
        assert detect_temperature_unit("Temperature (K)") == "K"

    def test_default_celsius(self):
        assert detect_temperature_unit("TC1") == "C"
        assert detect_temperature_unit("CH7") == "C"
        assert detect_temperature_unit("unknown") == "C"


class TestConvertTemperatureToCelsius:
    def test_celsius_passthrough(self):
        values = [25.0, 100.0, -40.0]
        result = convert_temperature_to_celsius(values, "C")
        assert list(result) == values

    def test_fahrenheit_to_celsius(self):
        values = [32.0, 212.0]
        result = convert_temperature_to_celsius(values, "F")
        assert abs(result[0] - 0.0) < 0.01
        assert abs(result[1] - 100.0) < 0.01

    def test_kelvin_to_celsius(self):
        values = [273.15, 373.15]
        result = convert_temperature_to_celsius(values, "K")
        assert abs(result[0] - 0.0) < 0.01
        assert abs(result[1] - 100.0) < 0.01


class TestFindColumns:
    def test_find_temperature_column(self):
        assert find_temperature_column(["time", "temp", "setpoint"]) == "temp"
        assert find_temperature_column(["Time", "Temperature", "SP"]) == "Temperature"
        assert find_temperature_column(["elapsed", "TC1", "SP"]) == "TC1"
        assert find_temperature_column(["time", "CH7", "setpoint"]) == "CH7"

    def test_find_setpoint_column(self):
        assert find_setpoint_column(["time", "temp", "setpoint"]) == "setpoint"
        assert find_setpoint_column(["time", "temp", "SP"]) == "SP"
        assert find_setpoint_column(["time", "temp", "target"]) == "target"

    def test_find_time_column(self):
        assert find_time_column(["time", "temp"]) == "time"
        assert find_time_column(["timestamp", "temperature"]) == "timestamp"
        assert find_time_column(["DateTime", "Temp"]) == "DateTime"
        assert find_time_column(["elapsed_seconds", "temp"]) == "elapsed_seconds"
