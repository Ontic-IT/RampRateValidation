"""Adaptive constants derivation from preprocessing report.

Derives data-driven tolerance values based on trace characteristics.
These are used as fallback when validation profile has no explicit tolerances.
"""

from __future__ import annotations

from models.domain import PreprocessingReport, ResolvedSetpoints, ProcessBoundaries
from engine.validation.tolerance_resolver import AdaptiveConstants


def derive_adaptive_constants(
    preprocessing_report: PreprocessingReport,
    setpoints: ResolvedSetpoints,
    boundaries: ProcessBoundaries,
) -> AdaptiveConstants:
    """Derive adaptive constants from preprocessing report.
    
    Implements the 9 adaptive constant derivation algorithms from the plan.
    
    Args:
        preprocessing_report: Preprocessing statistics
        setpoints: Resolved setpoints
        boundaries: Process boundaries
    
    Returns:
        AdaptiveConstants object with derived values
    """
    sample_interval = preprocessing_report.estimated_sample_interval_s
    
    # 1. noise_floor_c (already in preprocessing_report)
    noise_floor_c = preprocessing_report.noise_floor_c
    noise_floor_c = max(0.01, min(2.0, noise_floor_c))  # Bounds: [0.01, 2.0]
    
    # 2. slope_noise_floor_c_per_min
    # Formula: (noise_floor_c / sample_interval_s) × 60 × 2
    slope_noise_floor_c_per_min = (noise_floor_c / sample_interval) * 60 * 2
    slope_noise_floor_c_per_min = max(0.05, min(5.0, slope_noise_floor_c_per_min))  # Bounds: [0.05, 5.0]
    
    # 3. stable_slope_threshold
    # Formula: slope_noise_floor_c_per_min × 3.0
    stable_slope_threshold = slope_noise_floor_c_per_min * 3.0
    stable_slope_threshold = max(0.1, min(8.0, stable_slope_threshold))  # Bounds: [0.1, 8.0]
    
    # 4. stable_variance_threshold
    # Formula: 2× ambient rolling temperature MAD
    # Use noise_floor_c as proxy for ambient MAD
    stable_variance_threshold = noise_floor_c * 2.0
    stable_variance_threshold = max(0.02, min(3.0, stable_variance_threshold))  # Bounds: [0.02, 3.0]
    
    # 5. ramp_slope_threshold
    # Formula: (hot_setpoint - cold_setpoint) / (duration_seconds / 60) × 0.15
    hot_setpoint = setpoints.inferred_hot_setpoint_c
    cold_setpoint = setpoints.inferred_cold_setpoint_c
    # Calculate duration from indices
    process_duration = (boundaries.process_end_index - boundaries.process_start_index) * sample_interval
    
    if process_duration > 0 and hot_setpoint > cold_setpoint:
        ramp_slope_threshold = ((hot_setpoint - cold_setpoint) / (process_duration / 60)) * 0.15
    else:
        ramp_slope_threshold = 1.0  # Fallback
    ramp_slope_threshold = max(0.3, min(20.0, ramp_slope_threshold))  # Bounds: [0.3, 20.0]
    
    # 6. overshoot_detection_threshold
    # Formula: 4× dwell MAD
    # Use noise_floor_c × 4 as proxy (dwell MAD not available yet)
    overshoot_detection_threshold = noise_floor_c * 4.0
    overshoot_detection_threshold = max(0.1, min(10.0, overshoot_detection_threshold))  # Bounds: [0.1, 10.0]
    
    # 7. correction_oscillation_threshold
    # Formula: 2× dwell MAD
    correction_oscillation_threshold = noise_floor_c * 2.0
    correction_oscillation_threshold = max(0.05, min(5.0, correction_oscillation_threshold))  # Bounds: [0.05, 5.0]
    
    # 8. dwell_cluster_separation_threshold
    # Formula: (hot_setpoint - cold_setpoint) × 0.10
    if hot_setpoint > cold_setpoint:
        dwell_cluster_separation_threshold = (hot_setpoint - cold_setpoint) * 0.10
    else:
        dwell_cluster_separation_threshold = 5.0  # Fallback
    dwell_cluster_separation_threshold = max(0.5, min(15.0, dwell_cluster_separation_threshold))  # Bounds: [0.5, 15.0]
    
    # 9. minimum_region_duration_seconds
    # Formula: sample_interval_s × 10
    minimum_region_duration_seconds = sample_interval * 10
    minimum_region_duration_seconds = max(5.0, min(120.0, minimum_region_duration_seconds))  # Bounds: [5.0, 120.0]
    
    # Build constants dict
    constants = {
        "noise_floor_c": noise_floor_c,
        "slope_noise_floor_c_per_min": slope_noise_floor_c_per_min,
        "stable_slope_threshold": stable_slope_threshold,
        "stable_variance_threshold": stable_variance_threshold,
        "ramp_slope_threshold": ramp_slope_threshold,
        "overshoot_detection_threshold": overshoot_detection_threshold,
        "correction_oscillation_threshold": correction_oscillation_threshold,
        "dwell_cluster_separation_threshold": dwell_cluster_separation_threshold,
        "minimum_region_duration_seconds": minimum_region_duration_seconds,
        # Validation-specific tolerances
        "ramp_deviation": stable_variance_threshold * 1.5,  # Derived from stable variance
        "dwell_setpoint_deviation": stable_variance_threshold * 2.0,  # Derived from stable variance
    }
    
    # Build derivation methods dict
    derivation_methods = {
        "noise_floor_c": "MAD_of_ambient_window",
        "slope_noise_floor_c_per_min": "noise_floor_scaled_to_per_minute",
        "stable_slope_threshold": "3sigma_above_slope_noise",
        "stable_variance_threshold": "2x_ambient_MAD",
        "ramp_slope_threshold": "15pct_mean_achievable_rate",
        "overshoot_detection_threshold": "4x_dwell_MAD",
        "correction_oscillation_threshold": "2x_dwell_MAD",
        "dwell_cluster_separation_threshold": "10pct_setpoint_span",
        "minimum_region_duration_seconds": "10x_sample_interval",
        "ramp_deviation": "1.5x_stable_variance",
        "dwell_setpoint_deviation": "2x_stable_variance",
    }
    
    return AdaptiveConstants(constants, derivation_methods)
