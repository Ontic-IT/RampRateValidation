"""Unit tests for Phase 1 Retrofit 2: CanonicalTraceRow auxiliary channels."""

import pytest
from datetime import datetime

from models.domain import CanonicalTraceRow, AuxiliaryChannelMetadata


class TestCanonicalTraceRowAuxiliaryChannels:
    def test_default_empty_dict(self):
        row = CanonicalTraceRow(
            timestamp=datetime(2025, 1, 1, 0, 0, 0),
            elapsed_seconds=0.0,
            elapsed_minutes=0.0,
            temperature_c_raw=25.0,
            channel="CH1",
            source_row=0,
            source_file="test.csv",
            sample_interval_seconds=1.0,
        )
        assert row.auxiliary_channels == {}

    def test_with_numeric_auxiliary_channel(self):
        row = CanonicalTraceRow(
            timestamp=datetime(2025, 1, 1, 0, 0, 0),
            elapsed_seconds=0.0,
            elapsed_minutes=0.0,
            temperature_c_raw=25.0,
            channel="CH1",
            source_row=0,
            source_file="test.csv",
            sample_interval_seconds=1.0,
            auxiliary_channels={"pressure_pa": 101325.0},
        )
        assert row.auxiliary_channels["pressure_pa"] == 101325.0

    def test_with_state_auxiliary_channel(self):
        row = CanonicalTraceRow(
            timestamp=datetime(2025, 1, 1, 0, 0, 0),
            elapsed_seconds=0.0,
            elapsed_minutes=0.0,
            temperature_c_raw=25.0,
            channel="CH1",
            source_row=0,
            source_file="test.csv",
            sample_interval_seconds=1.0,
            auxiliary_channels={"heater_state": "ON"},
        )
        assert row.auxiliary_channels["heater_state"] == "ON"

    def test_with_multiple_auxiliary_channels(self):
        row = CanonicalTraceRow(
            timestamp=datetime(2025, 1, 1, 0, 0, 0),
            elapsed_seconds=0.0,
            elapsed_minutes=0.0,
            temperature_c_raw=25.0,
            channel="CH1",
            source_row=0,
            source_file="test.csv",
            sample_interval_seconds=1.0,
            auxiliary_channels={
                "pressure_pa": 101325.0,
                "heater_state": "ON",
                "humidity_pct": 45.0,
            },
        )
        assert len(row.auxiliary_channels) == 3
        assert row.auxiliary_channels["pressure_pa"] == 101325.0
        assert row.auxiliary_channels["heater_state"] == "ON"
        assert row.auxiliary_channels["humidity_pct"] == 45.0

    def test_existing_18_fields_still_work(self):
        row = CanonicalTraceRow(
            timestamp=datetime(2025, 1, 1, 0, 0, 0),
            elapsed_seconds=10.0,
            elapsed_minutes=0.167,
            temperature_c_raw=25.0,
            temperature_c_analysis_signal=25.1,
            setpoint_c=125.0,
            channel="CH1",
            source_row=0,
            source_file="test.csv",
            sample_interval_seconds=1.0,
            local_slope_c_per_min=5.0,
            rolling_slope_c_per_min=4.9,
            rolling_temperature_median=25.0,
            rolling_temperature_MAD=0.1,
            second_derivative=0.0,
            direction_of_travel="UP",
            data_quality_flags=[],
            region_id="R001",
            classification_label="HEATING_RAMP",
            auxiliary_channels={"pressure_pa": 101325.0},
        )
        assert row.temperature_c_raw == 25.0
        assert row.elapsed_seconds == 10.0
        assert row.auxiliary_channels["pressure_pa"] == 101325.0


class TestAuxiliaryChannelMetadata:
    def test_numeric_channel(self):
        meta = AuxiliaryChannelMetadata(
            channel_name="pressure_pa",
            unit="Pa",
            data_type="NUMERIC",
            source_column_index=3,
        )
        assert meta.channel_name == "pressure_pa"
        assert meta.unit == "Pa"
        assert meta.data_type == "NUMERIC"
        assert meta.source_column_index == 3
        assert meta.used_in_root_cause is False

    def test_state_channel(self):
        meta = AuxiliaryChannelMetadata(
            channel_name="heater_state",
            unit=None,
            data_type="STATE",
            source_column_index=4,
            used_in_root_cause=True,
        )
        assert meta.data_type == "STATE"
        assert meta.used_in_root_cause is True

    def test_default_data_type_is_numeric(self):
        meta = AuxiliaryChannelMetadata(
            channel_name="humidity_pct",
            source_column_index=5,
        )
        assert meta.data_type == "NUMERIC"

    def test_invalid_data_type_rejected(self):
        with pytest.raises(ValueError):
            AuxiliaryChannelMetadata(
                channel_name="bad",
                data_type="INVALID",
                source_column_index=0,
            )
