"""Validation engine package (M11)."""

from engine.validation.validation_engine import validate_analysis
from engine.validation.ramp_rules import (
    validate_heating_ramp_rate,
    validate_cooling_ramp_rate,
    validate_minimum_sustained_ramp_rate,
    validate_data_quality,
)
from engine.validation.dwell_rules import (
    validate_dwell_duration,
    validate_setpoint_deviation,
    validate_overshoot,
    validate_settling_time,
)
from engine.validation.cycle_rules import (
    validate_cycle_count,
    validate_profile_sequence,
)
from engine.validation.aggregation import aggregate_validation_status
from engine.validation.tolerance_resolver import (
    AdaptiveConstants,
    resolve_tolerance,
)

__all__ = [
    "validate_analysis",
    "validate_heating_ramp_rate",
    "validate_cooling_ramp_rate",
    "validate_minimum_sustained_ramp_rate",
    "validate_data_quality",
    "validate_dwell_duration",
    "validate_setpoint_deviation",
    "validate_overshoot",
    "validate_settling_time",
    "validate_cycle_count",
    "validate_profile_sequence",
    "aggregate_validation_status",
    "compute_phase_conformance",
    "AdaptiveConstants",
    "resolve_tolerance",
]
