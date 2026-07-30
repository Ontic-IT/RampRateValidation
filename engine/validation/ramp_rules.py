"""Ramp validation rules (M11).

Handles:
- Heating ramp rate validation
- Cooling ramp rate validation
- Minimum sustained ramp rate validation (standalone helper)
- Data quality validation (standalone helper)
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
import uuid

from config.constants import (
    AuditCategory,
    AuditSeverity,
    Comparator,
    DataQualityStatus,
    RampDirection,
    ValidationStatus,
)
from models.domain import (
    AuditEntry,
    AuditLog,
    RampMetrics,
    ValidRampRegion,
    ValidationDataQualityImpact,
    ValidationResult,
)


def _band_result(
    measured: float,
    target: float,
    tolerance: float,
) -> tuple[ValidationStatus, str]:
    """Two-sided band conformance: |measured - target| <= tolerance.

    Ramp-rate conformance is about tracking the COMMANDED trajectory, so both
    a shortfall AND an overshoot are deviations. Ramping much faster than the
    profile commanded is off-profile (product sees a different thermal
    stress), not a free pass — the old one-sided floor let any fast ramp
    through. A within-tolerance-but-fast ramp still passes; only a rate
    outside the band fails, and the reason states which side.
    """
    deviation = measured - target
    if abs(deviation) <= tolerance:
        return (
            ValidationStatus.PASS,
            f"Rate {measured:.2f} within {target:.2f}+/-{tolerance:.2f} degC/min "
            f"(deviation {deviation:+.2f})",
        )
    if deviation > 0:
        return (
            ValidationStatus.FAIL,
            f"Rate {measured:.2f} exceeds commanded {target:.2f} by "
            f"{deviation:.2f} degC/min (> tolerance {tolerance:.2f}) — ramping "
            "faster than the programmed trajectory",
        )
    return (
        ValidationStatus.FAIL,
        f"Rate {measured:.2f} falls short of commanded {target:.2f} by "
        f"{abs(deviation):.2f} degC/min (> tolerance {tolerance:.2f})",
    )


def validate_heating_ramp_rate(
    valid_ramp: ValidRampRegion,
    ramp_metrics: RampMetrics,
    required_rate_c_per_min: float,
    minimum_sustained_ratio: float = 0.8,
    tolerance_c_per_min: float | None = None,
    audit_log: AuditLog | None = None,
) -> ValidationResult:
    """Validate heating ramp rate against the commanded rate as a BAND.

    PASS iff |robust_ramp_slope - target| <= tolerance. `required_rate_c_per_min`
    is the band centre (commanded/target rate); `tolerance_c_per_min` its
    half-width (defaults to 20% of target when not supplied).

    Args:
        valid_ramp: Valid ramp region
        ramp_metrics: Computed ramp metrics
        required_rate_c_per_min: Target (commanded) heating rate — band centre
        minimum_sustained_ratio: Retained for signature stability (unused by band)
        tolerance_c_per_min: Band half-width; default 20% of target
        audit_log: Optional audit log

    Returns:
        ValidationResult with pass/fail decision
    """
    if audit_log is None:
        audit_log = AuditLog()

    if valid_ramp.direction != RampDirection.HEATING:
        return ValidationResult(
            validation_result_id=str(uuid.uuid4()),
            requirement_id="HEATING_RAMP_RATE",
            requirement_description=f"Heating ramp rate within band of {required_rate_c_per_min} °C/min",
            measured_value=0.0,
            threshold_value=required_rate_c_per_min,
            comparator=Comparator.WITHIN_RANGE,
            unit="°C/min",
            method="theil_sen",
            region_id=valid_ramp.region_id,
            included_rows=len(valid_ramp.included_rows),
            result=ValidationStatus.NOT_APPLICABLE,
            reason="Not a heating ramp",
        )

    target = required_rate_c_per_min
    tolerance = tolerance_c_per_min if tolerance_c_per_min is not None else abs(target) * 0.20
    robust_slope = ramp_metrics.robust_slope_c_per_min

    result, reason = _band_result(robust_slope, target, tolerance)

    validation_result = ValidationResult(
        validation_result_id=str(uuid.uuid4()),
        requirement_id="HEATING_RAMP_RATE",
        requirement_description=f"Heating ramp rate = {target:.2f} +/- {tolerance:.2f} °C/min",
        measured_value=robust_slope,
        threshold_value=target,
        comparator=Comparator.WITHIN_RANGE,
        unit="°C/min",
        method="theil_sen",
        region_id=valid_ramp.region_id,
        included_rows=len(valid_ramp.included_rows),
        excluded_regions=valid_ramp.exclusion_reasons,
        result=result,
        reason=reason,
    )

    audit_log.add(AuditEntry(
        timestamp=datetime.now(),
        module_name="ramp_rules",
        action="validate_heating_ramp_rate",
        input_reference=valid_ramp.region_id,
        output_reference=validation_result.validation_result_id,
        decision=result.value,
        reason=reason,
        thresholds_used={"target": target, "tolerance": tolerance},
        severity=AuditSeverity.INFO,
        category=AuditCategory.VALIDATION,
    ))

    return validation_result


def validate_cooling_ramp_rate(
    valid_ramp: ValidRampRegion,
    ramp_metrics: RampMetrics,
    required_rate_c_per_min: float,
    minimum_sustained_ratio: float = 0.8,
    tolerance_c_per_min: float | None = None,
    audit_log: AuditLog | None = None,
) -> ValidationResult:
    """Validate cooling ramp rate against the commanded rate as a BAND.

    Cooling rates are negative; the band is applied to magnitudes. PASS iff
    ||robust_slope| - target| <= tolerance.

    Args:
        valid_ramp: Valid ramp region
        ramp_metrics: Computed ramp metrics
        required_rate_c_per_min: Target (commanded) cooling rate (positive) — band centre
        minimum_sustained_ratio: Retained for signature stability (unused by band)
        tolerance_c_per_min: Band half-width; default 20% of target
        audit_log: Optional audit log

    Returns:
        ValidationResult with pass/fail decision
    """
    if audit_log is None:
        audit_log = AuditLog()

    if valid_ramp.direction != RampDirection.COOLING:
        return ValidationResult(
            validation_result_id=str(uuid.uuid4()),
            requirement_id="COOLING_RAMP_RATE",
            requirement_description=f"Cooling ramp rate within band of {required_rate_c_per_min} °C/min",
            measured_value=0.0,
            threshold_value=required_rate_c_per_min,
            comparator=Comparator.WITHIN_RANGE,
            unit="°C/min",
            method="theil_sen",
            region_id=valid_ramp.region_id,
            included_rows=len(valid_ramp.included_rows),
            result=ValidationStatus.NOT_APPLICABLE,
            reason="Not a cooling ramp",
        )

    target = abs(required_rate_c_per_min)
    tolerance = tolerance_c_per_min if tolerance_c_per_min is not None else abs(target) * 0.20
    robust_slope = abs(ramp_metrics.robust_slope_c_per_min)

    result, reason = _band_result(robust_slope, target, tolerance)

    validation_result = ValidationResult(
        validation_result_id=str(uuid.uuid4()),
        requirement_id="COOLING_RAMP_RATE",
        requirement_description=f"Cooling ramp rate = {target:.2f} +/- {tolerance:.2f} °C/min",
        measured_value=robust_slope,
        threshold_value=target,
        comparator=Comparator.WITHIN_RANGE,
        unit="°C/min",
        method="theil_sen",
        region_id=valid_ramp.region_id,
        included_rows=len(valid_ramp.included_rows),
        excluded_regions=valid_ramp.exclusion_reasons,
        result=result,
        reason=reason,
    )

    audit_log.add(AuditEntry(
        timestamp=datetime.now(),
        module_name="ramp_rules",
        action="validate_cooling_ramp_rate",
        input_reference=valid_ramp.region_id,
        output_reference=validation_result.validation_result_id,
        decision=result.value,
        reason=reason,
        thresholds_used={"target": target, "tolerance": tolerance},
        severity=AuditSeverity.INFO,
        category=AuditCategory.VALIDATION,
    ))

    return validation_result


def validate_minimum_sustained_ramp_rate(
    ramp_metrics: RampMetrics,
    required_rate_c_per_min: float,
    minimum_sustained_ratio: float = 0.8,
) -> tuple[bool, str]:
    """Standalone helper to validate minimum sustained ramp rate.
    
    Formula: minimum_sustained_slope >= required_rate × ratio
    
    Args:
        ramp_metrics: Computed ramp metrics
        required_rate_c_per_min: Required ramp rate
        minimum_sustained_ratio: Ratio threshold
    
    Returns:
        Tuple of (passes, reason)
    """
    threshold = required_rate_c_per_min * minimum_sustained_ratio
    min_sustained = abs(ramp_metrics.minimum_sustained_slope_c_per_min)
    
    passes = min_sustained >= threshold
    
    if passes:
        reason = f"Minimum sustained {min_sustained:.2f} >= threshold {threshold:.2f}"
    else:
        reason = f"Minimum sustained {min_sustained:.2f} < threshold {threshold:.2f}"
    
    return passes, reason


def validate_data_quality(
    quality_impact: ValidationDataQualityImpact,
    requirement_id: str,
) -> tuple[ValidationStatus, str]:
    """Standalone helper to check data quality impact on validation.
    
    Inspects ValidationDataQualityImpact and returns INCONCLUSIVE
    where data quality blocks specific requirements.
    
    Args:
        quality_impact: Data quality impact from Phase 3
        requirement_id: Requirement being validated
    
    Returns:
        Tuple of (status, reason)
    """
    if quality_impact.blocks_pass_fail:
        if "ALL" in quality_impact.affected_requirement_ids or requirement_id in quality_impact.affected_requirement_ids:
            return (
                ValidationStatus.INCONCLUSIVE,
                f"Data quality blocks validation: {quality_impact.reason}"
            )
    
    return ValidationStatus.PASS, "Data quality acceptable"
