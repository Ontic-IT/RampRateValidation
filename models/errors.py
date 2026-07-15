"""Error taxonomy for the Ramp Rate Validation Tool."""


class RampRateError(Exception):
    """Base exception for all ramp rate validation errors."""
    pass


class InputFormatError(RampRateError):
    """File structurally unusable after ingestion (stops pipeline)."""
    pass


class TimestampParseError(RampRateError):
    """Timestamp column cannot be parsed into a valid datetime series."""
    pass


class TemperatureParseError(RampRateError):
    """Temperature column cannot be parsed into a valid numeric series."""
    pass


class QualityGateError(RampRateError):
    """Data quality gate failed — INVALID status stops pipeline."""
    pass


class ClassificationError(RampRateError):
    """Classification pipeline encountered an unrecoverable error."""
    pass


class ValidationError(RampRateError):
    """Validation engine encountered an unrecoverable error."""
    pass


class ReportingError(RampRateError):
    """Report generation encountered an unrecoverable error."""
    pass
