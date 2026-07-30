"""Profile loader for validation profiles."""

from __future__ import annotations

from pathlib import Path
import yaml

from models.profile import ValidationProfile
from models.errors import InputFormatError


def default_self_validation_profile() -> ValidationProfile:
    """Profile used when no external profile is supplied.

    Every requirement is left unspecified, so the tolerance resolver derives
    ALL thresholds from the trace itself: the setpoint programme supplies
    the commanded ramp rates, and the demonstrated dwell tracking supplies
    the deviation tolerances. This is the default mode — an external YAML
    profile is an optional override for when a formal specification exists.
    """
    from datetime import datetime

    return ValidationProfile(
        profile_metadata={
            "profile_name": "Self-validation (trace-derived requirements)",
            "profile_version": "1.0.0",
            "created_date": datetime(2026, 1, 1),
            "description": (
                "No external profile supplied; all thresholds derived from the "
                "trace's own commanded setpoint programme and demonstrated control accuracy"
            ),
            "algorithm_version_required": "1.0.0",
        },
        expected_process_sequence={},
        ramp_rate_requirements={
            "required_heating_ramp_rate_c_per_min": 0.0,  # derive from setpoint programme
            "required_cooling_ramp_rate_c_per_min": 0.0,  # derive from setpoint programme
            "minimum_sustained_ramp_rate_ratio": 0.0,     # engine applies plan default
            "allowed_ramp_deviation_c": None,             # adaptive
            "tolerance_source": "ADAPTIVE",
        },
        dwell_requirements={
            "allowed_setpoint_deviation_c": None,  # adaptive from dwell tracking
            "tolerance_source": "ADAPTIVE",
        },
        setpoint_deviation_requirements={},
        overshoot_requirements={
            "overshoot_warning_threshold_c": 0.0,
            "overshoot_failure_threshold_c": 0.0,
            "max_overshoot_duration_seconds": 0.0,
        },
        settling_requirements={
            "settling_time_limit_seconds": 0.0,
            "settling_tolerance_band_c": 0.0,
        },
    )


def load_profile(profile_path: str | None) -> ValidationProfile:
    """Load and validate a profile YAML file.

    Args:
        profile_path: Path to profile YAML file, or None for self-validation
            (all requirements derived from the trace)

    Returns:
        ValidationProfile object

    Raises:
        InputFormatError: If profile cannot be loaded or validated
    """
    if profile_path is None:
        return default_self_validation_profile()

    path = Path(profile_path)
    if not path.exists():
        raise InputFormatError(f"Profile not found: {profile_path}")
    
    try:
        with open(path, 'r', encoding='utf-8') as f:
            profile_data = yaml.safe_load(f)
    except Exception as e:
        raise InputFormatError(f"Failed to parse profile YAML: {e}")
    
    try:
        profile = ValidationProfile(**profile_data)
    except Exception as e:
        raise InputFormatError(f"Failed to validate profile: {e}")
    
    return profile
