"""Profile loader for validation profiles."""

from __future__ import annotations

from pathlib import Path
import yaml

from models.profile import ValidationProfile
from models.errors import InputFormatError


def load_profile(profile_path: str) -> ValidationProfile:
    """Load and validate a profile YAML file.
    
    Args:
        profile_path: Path to profile YAML file
    
    Returns:
        ValidationProfile object
    
    Raises:
        InputFormatError: If profile cannot be loaded or validated
    """
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
