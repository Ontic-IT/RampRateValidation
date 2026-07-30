"""Adaptive constants derivation — the trace explains its own thresholds.

Two families of derived values, both deterministic (documented formula +
bounds, per plan guardrail 23):

1. DETECTION constants (noise floors, slope thresholds) — derived from the
   trace's noise characteristics, used by classification/preprocessing.

2. REQUIREMENT constants — derived from the trace's own COMMANDED programme:
   - required heating/cooling ramp rate = the setpoint's own slope during
     transitions between plateaus (Mode A: the setpoint channel IS the
     specification of what the chamber was told to do);
   - dwell setpoint deviation tolerance = derived from the control loop's
     demonstrated dwell tracking spread (dwell MADs / tracking error), never
     from ambient sensor noise, with a physical floor (no chamber spec is
     tighter than +/-0.5 degC).

An explicit ValidationProfile value always overrides a derived one (the
tolerance resolver enforces precedence).
"""

from __future__ import annotations

import numpy as np

from models.domain import PreprocessingReport, ResolvedSetpoints, ProcessBoundaries
from engine.validation.tolerance_resolver import AdaptiveConstants

# Physical floor/ceiling for a dwell tolerance: no thermal chamber holds
# tighter than +/-0.5 degC, and a dwell allowed to wander >5 degC is not
# holding its setpoint.
DWELL_TOLERANCE_BOUNDS_C = (0.5, 5.0)
RAMP_DEVIATION_BOUNDS_C = (0.5, 7.5)

# Commanded-rate derivation: a setpoint transition is the move between two
# plateaus. Plateaus shorter than this are jitter, transitions smaller than
# this are corrections, not ramps.
MIN_PLATEAU_SAMPLES = 10
MIN_TRANSITION_SPAN_C = 5.0
COMMANDED_RATE_BOUNDS = (0.1, 100.0)

# Measured temperature lags the command slightly even on a healthy chamber;
# requiring measured >= commanded exactly would fail borderline-good traces.
# Margin verified against the corpus of known-passed traces (all must pass).
COMMANDED_RATE_MARGIN = 0.90


def _clamp(value: float, bounds: tuple[float, float]) -> float:
    return max(bounds[0], min(bounds[1], value))


def _derive_commanded_rates(
    elapsed_s: np.ndarray,
    setpoint: np.ndarray,
) -> tuple[float | None, float | None, int]:
    """Derive commanded heating/cooling rates from the setpoint programme.

    Segments the setpoint into plateaus (constant >= MIN_PLATEAU_SAMPLES)
    and transitions between them; each transition's commanded rate is
    delta-level / transition-time. Returns (heating, cooling, n_transitions),
    None per direction when the programme steps instantaneously (a commanded
    rate does not exist for a step change).
    """
    n = len(setpoint)
    if n < 3 * MIN_PLATEAU_SAMPLES:
        return None, None, 0

    # Locate plateaus: runs of identical setpoint values.
    plateaus: list[tuple[int, int, float]] = []  # (start, end_exclusive, value)
    start = 0
    for i in range(1, n + 1):
        if i == n or setpoint[i] != setpoint[start]:
            if i - start >= MIN_PLATEAU_SAMPLES and not np.isnan(setpoint[start]):
                plateaus.append((start, i, float(setpoint[start])))
            start = i

    heating_rates: list[float] = []
    cooling_rates: list[float] = []
    for (_, prev_end, prev_val), (next_start, _, next_val) in zip(plateaus, plateaus[1:]):
        span = next_val - prev_val
        if abs(span) < MIN_TRANSITION_SPAN_C:
            continue
        # Transition runs from the end of one plateau to the start of the next.
        t0 = elapsed_s[max(prev_end - 1, 0)]
        t1 = elapsed_s[min(next_start, n - 1)]
        duration_s = t1 - t0
        if duration_s <= 0:
            continue
        rate = abs(span) / (duration_s / 60.0)
        (heating_rates if span > 0 else cooling_rates).append(rate)

    def _direction_rate(rates: list[float]) -> float | None:
        if not rates:
            return None
        median = float(np.median(rates))
        # A "commanded rate" beyond any real chamber's capability means the
        # programme steps instantaneously — there is no commanded slope.
        if median > COMMANDED_RATE_BOUNDS[1]:
            return None
        return _clamp(median, COMMANDED_RATE_BOUNDS)

    return (
        _direction_rate(heating_rates),
        _direction_rate(cooling_rates),
        len(heating_rates) + len(cooling_rates),
    )


def _derive_dwell_tracking_tolerance(
    elapsed_s: np.ndarray,
    setpoint: np.ndarray,
    temperature: np.ndarray,
    dwell_mads: dict[str, float],
) -> tuple[float, str]:
    """Dwell setpoint-deviation tolerance from demonstrated control accuracy.

    Preferred: the 95th percentile of |temperature - setpoint| across ALL
    dwell-plateau samples (skipping only the first few samples where the
    command has just stepped). This captures the chamber's DEMONSTRATED
    holding accuracy including steady-state offset under load — not just
    the flattering fully-settled tail. Fallback: 4x median dwell MAD. Both
    bounded so quantised data cannot collapse the tolerance to noise level
    and a wandering chamber cannot grant itself a free pass.
    """
    n = len(setpoint)
    errors: list[float] = []
    start = 0
    for i in range(1, n + 1):
        if i == n or setpoint[i] != setpoint[start]:
            if i - start >= MIN_PLATEAU_SAMPLES and not np.isnan(setpoint[start]):
                seg = np.abs(temperature[start + 5:i] - setpoint[start])
                errors.extend(seg[~np.isnan(seg)].tolist())
            start = i

    if errors:
        tolerance = _clamp(float(np.percentile(errors, 95)), DWELL_TOLERANCE_BOUNDS_C)
        return tolerance, "p95_demonstrated_dwell_tracking_error"

    mads = [v for v in (dwell_mads or {}).values() if v and v > 0]
    if mads:
        tolerance = _clamp(4.0 * float(np.median(mads)), DWELL_TOLERANCE_BOUNDS_C)
        return tolerance, "4x_median_dwell_MAD"

    return DWELL_TOLERANCE_BOUNDS_C[0] * 2, "physical_default_no_dwell_evidence"


def derive_adaptive_constants(
    preprocessing_report: PreprocessingReport,
    setpoints: ResolvedSetpoints,
    boundaries: ProcessBoundaries,
    trace=None,
) -> AdaptiveConstants:
    """Derive adaptive constants from the trace.

    Args:
        preprocessing_report: Preprocessing statistics
        setpoints: Resolved setpoints (incl. dwell_MAD_values)
        boundaries: Process boundaries
        trace: Optional PreprocessedTrace — enables requirement derivation
            from the commanded setpoint programme (Mode A)

    Returns:
        AdaptiveConstants with derived values and derivation methods
    """
    sample_interval = preprocessing_report.estimated_sample_interval_s

    # ------- Detection constants (unchanged formulas) -------
    noise_floor_c = preprocessing_report.noise_floor_c
    noise_floor_c = max(0.01, min(2.0, noise_floor_c))

    slope_noise_floor_c_per_min = (noise_floor_c / sample_interval) * 60 * 2
    slope_noise_floor_c_per_min = max(0.05, min(5.0, slope_noise_floor_c_per_min))

    stable_slope_threshold = slope_noise_floor_c_per_min * 3.0
    stable_slope_threshold = max(0.1, min(8.0, stable_slope_threshold))

    stable_variance_threshold = noise_floor_c * 2.0
    stable_variance_threshold = max(0.02, min(3.0, stable_variance_threshold))

    hot_setpoint = setpoints.inferred_hot_setpoint_c
    cold_setpoint = setpoints.inferred_cold_setpoint_c
    process_duration = (boundaries.process_end_index - boundaries.process_start_index) * sample_interval

    if process_duration > 0 and hot_setpoint > cold_setpoint:
        ramp_slope_threshold = ((hot_setpoint - cold_setpoint) / (process_duration / 60)) * 0.15
    else:
        ramp_slope_threshold = 1.0
    ramp_slope_threshold = max(0.3, min(20.0, ramp_slope_threshold))

    dwell_mads = setpoints.dwell_MAD_values or {}
    mad_values = [v for v in dwell_mads.values() if v and v > 0]
    dwell_mad = float(np.median(mad_values)) if mad_values else noise_floor_c

    # Plan-specified formulas, now on the CORRECT quantity (dwell MAD, not
    # ambient noise proxy).
    overshoot_detection_threshold = max(0.1, min(10.0, dwell_mad * 4.0))
    correction_oscillation_threshold = max(0.05, min(5.0, dwell_mad * 2.0))

    if hot_setpoint > cold_setpoint:
        dwell_cluster_separation_threshold = (hot_setpoint - cold_setpoint) * 0.10
    else:
        dwell_cluster_separation_threshold = 5.0
    dwell_cluster_separation_threshold = max(0.5, min(15.0, dwell_cluster_separation_threshold))

    minimum_region_duration_seconds = max(5.0, min(120.0, sample_interval * 10))

    # ------- Requirement constants (from the commanded programme) -------
    elapsed = sp = temp = None
    if trace is not None and getattr(trace, "rows", None):
        rows = trace.rows
        if rows and any(r.setpoint_c is not None for r in rows[:100]):
            elapsed = np.array([r.elapsed_seconds for r in rows], dtype=float)
            sp = np.array(
                [r.setpoint_c if r.setpoint_c is not None else np.nan for r in rows],
                dtype=float,
            )
            temp = np.array([r.temperature_c_raw for r in rows], dtype=float)

    required_heating = required_cooling = None
    n_transitions = 0
    if sp is not None:
        required_heating, required_cooling, n_transitions = _derive_commanded_rates(elapsed, sp)

    if sp is not None:
        dwell_tolerance, dwell_tol_method = _derive_dwell_tracking_tolerance(
            elapsed, sp, temp, dwell_mads
        )
    elif mad_values:
        # No setpoint channel: fall back to dwell MADs from Mode B inference.
        dwell_tolerance = _clamp(4.0 * float(np.median(mad_values)), DWELL_TOLERANCE_BOUNDS_C)
        dwell_tol_method = "4x_median_dwell_MAD_mode_b"
    else:
        # No setpoint channel AND no dwell evidence: ambient noise says
        # nothing about control accuracy, so use the physical default
        # (standard +/-2degC ESS dwell band) rather than a noise-derived value.
        dwell_tolerance = 2.0
        dwell_tol_method = "physical_default_no_dwell_evidence_mode_b"

    ramp_deviation = _clamp(dwell_tolerance * 1.5, RAMP_DEVIATION_BOUNDS_C)

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
        "ramp_deviation": ramp_deviation,
        "dwell_setpoint_deviation": dwell_tolerance,
    }
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
        "ramp_deviation": "1.5x_dwell_tolerance",
        "dwell_setpoint_deviation": dwell_tol_method,
    }

    if required_heating is not None:
        constants["required_heating_ramp_rate"] = required_heating * COMMANDED_RATE_MARGIN
        derivation_methods["required_heating_ramp_rate"] = (
            f"{COMMANDED_RATE_MARGIN:.0%}_of_median_commanded_setpoint_slope"
            f"_({required_heating:.2f}degC_per_min_commanded,_{n_transitions}_transitions)"
        )
    if required_cooling is not None:
        constants["required_cooling_ramp_rate"] = required_cooling * COMMANDED_RATE_MARGIN
        derivation_methods["required_cooling_ramp_rate"] = (
            f"{COMMANDED_RATE_MARGIN:.0%}_of_median_commanded_setpoint_slope"
            f"_({required_cooling:.2f}degC_per_min_commanded,_{n_transitions}_transitions)"
        )

    return AdaptiveConstants(constants, derivation_methods)
