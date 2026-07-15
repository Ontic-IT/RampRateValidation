"""Schema for per-run derived adaptive constants with bounds and derivation methods.

AdaptiveConstants are derived during Phase 3 (Preprocessing) and Phase 4 (Classification).
They are NEVER used for pass/fail decisions — only for classification aid.

Derivation sequence (executed in order):
1. noise_floor_c — MAD of pre-process ambient window
2. slope_noise_floor_c_per_min — scaled from noise_floor_c
3. stable_slope_threshold — 3× slope noise floor
4. stable_variance_threshold — 2× ambient rolling temperature MAD
5. ramp_slope_threshold — 15% of mean achievable rate (requires ResolvedSetpoints)
6. overshoot_detection_threshold — 4× dwell MAD (requires ResolvedSetpoints + dwell_MAD_values)
7. correction_oscillation_threshold — 2× dwell MAD
8. dwell_cluster_separation_threshold — 10% of setpoint span
9. minimum_region_duration_seconds — 10× sample interval
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class AdaptiveThreshold(BaseModel):
    """Individual adaptive threshold with bounds and derivation tracking."""
    value: float
    minimum_bound: float
    maximum_bound: float
    derivation_method: str
    was_clamped: bool = False
    derived_value: float | None = None


class AdaptiveConstantBounds(BaseModel):
    """Bounds specification for a single adaptive constant."""
    minimum: float
    maximum: float


# Canonical bounds for each adaptive constant
ADAPTIVE_CONSTANT_BOUNDS: dict[str, AdaptiveConstantBounds] = {
    "noise_floor_c": AdaptiveConstantBounds(minimum=0.01, maximum=2.0),
    "slope_noise_floor_c_per_min": AdaptiveConstantBounds(minimum=0.05, maximum=5.0),
    "stable_slope_threshold": AdaptiveConstantBounds(minimum=0.1, maximum=8.0),
    "stable_variance_threshold": AdaptiveConstantBounds(minimum=0.02, maximum=3.0),
    "ramp_slope_threshold": AdaptiveConstantBounds(minimum=0.3, maximum=20.0),
    "overshoot_detection_threshold": AdaptiveConstantBounds(minimum=0.1, maximum=10.0),
    "correction_oscillation_threshold": AdaptiveConstantBounds(minimum=0.05, maximum=5.0),
    "dwell_cluster_separation_threshold": AdaptiveConstantBounds(minimum=0.5, maximum=15.0),
    "minimum_region_duration_seconds": AdaptiveConstantBounds(minimum=5.0, maximum=120.0),
}


class AdaptiveConstants(BaseModel):
    """Per-run derived adaptive constants. NEVER used for pass/fail decisions."""
    noise_floor_c: AdaptiveThreshold
    slope_noise_floor_c_per_min: AdaptiveThreshold
    stable_slope_threshold: AdaptiveThreshold
    stable_variance_threshold: AdaptiveThreshold
    ramp_slope_threshold: AdaptiveThreshold
    overshoot_detection_threshold: AdaptiveThreshold
    correction_oscillation_threshold: AdaptiveThreshold
    dwell_cluster_separation_threshold: AdaptiveThreshold
    minimum_region_duration_seconds: AdaptiveThreshold

    def to_snapshot(self) -> dict[str, float]:
        """Serialize all adaptive constants to a flat dict for RunMetadata."""
        return {
            "noise_floor_c": self.noise_floor_c.value,
            "slope_noise_floor_c_per_min": self.slope_noise_floor_c_per_min.value,
            "stable_slope_threshold": self.stable_slope_threshold.value,
            "stable_variance_threshold": self.stable_variance_threshold.value,
            "ramp_slope_threshold": self.ramp_slope_threshold.value,
            "overshoot_detection_threshold": self.overshoot_detection_threshold.value,
            "correction_oscillation_threshold": self.correction_oscillation_threshold.value,
            "dwell_cluster_separation_threshold": self.dwell_cluster_separation_threshold.value,
            "minimum_region_duration_seconds": self.minimum_region_duration_seconds.value,
        }


def clamp_threshold(
    name: str,
    derived_value: float,
    derivation_method: str,
) -> AdaptiveThreshold:
    """Create an AdaptiveThreshold, clamping to canonical bounds if necessary.

    If the derived value falls outside bounds, was_clamped is set to True
    and derived_value records the original pre-clamp value. The caller
    MUST emit an AuditEntry with action='ADAPTIVE_THRESHOLD_CLAMPED',
    severity=WARNING when was_clamped is True.
    """
    bounds = ADAPTIVE_CONSTANT_BOUNDS[name]
    clamped = max(bounds.minimum, min(bounds.maximum, derived_value))
    was_clamped = clamped != derived_value

    return AdaptiveThreshold(
        value=clamped,
        minimum_bound=bounds.minimum,
        maximum_bound=bounds.maximum,
        derivation_method=derivation_method,
        was_clamped=was_clamped,
        derived_value=derived_value if was_clamped else None,
    )
