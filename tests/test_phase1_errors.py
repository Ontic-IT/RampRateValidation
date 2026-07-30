"""Unit tests for Phase 1: models/errors.py — error taxonomy."""

import pytest

from models.errors import (
    ClassificationError,
    InputFormatError,
    QualityGateError,
    RampRateError,
    ReportingError,
    TemperatureParseError,
    TimestampParseError,
    ValidationError,
)


class TestErrorTaxonomy:
    """All custom exceptions inherit from RampRateError."""

    def test_ramp_rate_error_is_exception(self):
        assert issubclass(RampRateError, Exception)

    def test_input_format_error_inherits(self):
        assert issubclass(InputFormatError, RampRateError)

    def test_timestamp_parse_error_inherits(self):
        assert issubclass(TimestampParseError, RampRateError)

    def test_temperature_parse_error_inherits(self):
        assert issubclass(TemperatureParseError, RampRateError)

    def test_quality_gate_error_inherits(self):
        assert issubclass(QualityGateError, RampRateError)

    def test_classification_error_inherits(self):
        assert issubclass(ClassificationError, RampRateError)

    def test_validation_error_inherits(self):
        assert issubclass(ValidationError, RampRateError)

    def test_reporting_error_inherits(self):
        assert issubclass(ReportingError, RampRateError)

    def test_all_seven_subtypes_exist(self):
        subtypes = [
            InputFormatError,
            TimestampParseError,
            TemperatureParseError,
            QualityGateError,
            ClassificationError,
            ValidationError,
            ReportingError,
        ]
        assert len(subtypes) == 7
        for st in subtypes:
            assert issubclass(st, RampRateError)

    def test_can_raise_and_catch(self):
        with pytest.raises(RampRateError):
            raise InputFormatError("bad file")

    def test_message_preserved(self):
        try:
            raise TimestampParseError("cannot parse timestamp")
        except RampRateError as e:
            assert str(e) == "cannot parse timestamp"
