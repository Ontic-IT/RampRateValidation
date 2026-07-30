"""Static constants and enumerations for the Ramp Rate Validation Tool."""

from enum import Enum


class RegionType(str, Enum):
    """All 14 region classification types."""
    HOT_DWELL = "HOT_DWELL"
    COLD_DWELL = "COLD_DWELL"
    AMBIENT_START = "AMBIENT_START"
    HEATING_RAMP = "HEATING_RAMP"
    COOLING_RAMP = "COOLING_RAMP"
    RAMP_JITTER = "RAMP_JITTER"
    RAMP_TAPER = "RAMP_TAPER"
    HOT_OVERSHOOT = "HOT_OVERSHOOT"
    COLD_OVERSHOOT = "COLD_OVERSHOOT"
    HOT_CORRECTION = "HOT_CORRECTION"
    COLD_CORRECTION = "COLD_CORRECTION"
    RECOVERY = "RECOVERY"
    # Trace-intrinsic repeated wiggle around a level. Used when there is NO
    # setpoint to measure against: you can see the temperature oscillating, but
    # cannot call it an overshoot (or say by how much) until a target is known.
    # With a target selected, overshoots are quantified against it.
    OSCILLATION = "OSCILLATION"
    TRANSIENT = "TRANSIENT"
    UNKNOWN = "UNKNOWN"


class ValidationStatus(str, Enum):
    """5 validation result statuses."""
    PASS = "PASS"
    PASS_WITH_WARNINGS = "PASS_WITH_WARNINGS"
    FAIL = "FAIL"
    INCONCLUSIVE = "INCONCLUSIVE"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class OverallValidationStatus(str, Enum):
    """6 overall validation statuses for AnalysisResult."""
    PASS = "PASS"
    PASS_WITH_WARNINGS = "PASS_WITH_WARNINGS"
    FAIL = "FAIL"
    INCONCLUSIVE = "INCONCLUSIVE"
    INVALID_INPUT = "INVALID_INPUT"
    ERROR = "ERROR"


class DataQualityStatus(str, Enum):
    """4 data quality statuses."""
    ACCEPTABLE = "ACCEPTABLE"
    WARNING = "WARNING"
    INCONCLUSIVE = "INCONCLUSIVE"
    INVALID = "INVALID"


class CycleStatus(str, Enum):
    """5 cycle statuses."""
    COMPLETE = "COMPLETE"
    PARTIAL = "PARTIAL"
    ABORTED = "ABORTED"
    AMBIGUOUS = "AMBIGUOUS"
    INVALID = "INVALID"


class AuditSeverity(str, Enum):
    """Audit entry severity levels."""
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class AuditCategory(str, Enum):
    """Audit entry categories."""
    CLASSIFICATION = "CLASSIFICATION"
    VALIDATION = "VALIDATION"
    QUALITY = "QUALITY"
    PROFILE = "PROFILE"
    METRICS = "METRICS"
    PIPELINE = "PIPELINE"
    BOUNDARY_NORMALISATION = "BOUNDARY_NORMALISATION"


class ConfidenceLevel(str, Enum):
    """Classification confidence levels."""
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class RampDirection(str, Enum):
    """Ramp direction types."""
    HEATING = "HEATING"
    COOLING = "COOLING"


class ToleranceSource(str, Enum):
    """Source of tolerance values for validation."""
    EXPLICIT = "EXPLICIT"
    ADAPTIVE = "ADAPTIVE"


class PipelineStage(str, Enum):
    """Pipeline execution stages in strict order."""
    INGESTION = "INGESTION"
    NORMALISATION = "NORMALISATION"
    PREPROCESSING = "PREPROCESSING"
    QUALITY = "QUALITY"
    BOUNDARIES = "BOUNDARIES"
    SETPOINTS = "SETPOINTS"
    CLASSIFICATION = "CLASSIFICATION"
    RAMP_ISOLATION = "RAMP_ISOLATION"
    CYCLES = "CYCLES"
    METRICS = "METRICS"
    VALIDATION = "VALIDATION"
    COMPARISON = "COMPARISON"
    VISUALISATION = "VISUALISATION"
    REPORTING = "REPORTING"
    AUDIT = "AUDIT"
    COMPLETE = "COMPLETE"


class PipelineStatus(str, Enum):
    """Pipeline stage execution statuses."""
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


class Comparator(str, Enum):
    """Validation comparator types."""
    GTE = "GTE"
    LTE = "LTE"
    WITHIN_RANGE = "WITHIN_RANGE"
    EQ = "EQ"
    RANGE = "RANGE"


class FirstExtreme(str, Enum):
    """Expected first extreme in process sequence."""
    HOT_FIRST = "HOT_FIRST"
    COLD_FIRST = "COLD_FIRST"


class AmbiguityHandling(str, Enum):
    """Profile ambiguity handling modes."""
    WARN = "WARN"
    INCONCLUSIVE = "INCONCLUSIVE"
    ALLOW = "ALLOW"


class SetpointResolutionMode(str, Enum):
    """Setpoint resolution modes."""
    MODE_A = "MODE_A"
    MODE_B = "MODE_B"


class GapSignificance(str, Enum):
    """Gap significance levels for context-sensitive gap impact assessment."""
    NEGLIGIBLE = "NEGLIGIBLE"
    MINOR = "MINOR"
    MODERATE = "MODERATE"
    SEVERE = "SEVERE"
