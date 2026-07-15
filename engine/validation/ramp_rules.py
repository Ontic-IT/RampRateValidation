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


def validate_heating_ramp_rate(
    valid_ramp: ValidRampRegion,
    ramp_metrics: RampMetrics,
    required_rate_c_per_min: float,
    minimum_sustained_ratio: float = 0.8,
    audit_log: AuditLog | None = None,
) -> ValidationResult:
    """Validate heating ramp rate against profile requirement.
    
    PASS iff:
    - robust_ramp_slope >= required_rate
    - minimum_sustained >= required_rate × ratio
    
    Args:
        valid_ramp: Valid ramp region
        ramp_metrics: Computed ramp metrics
        required_rate_c_per_min: Required heating rate from profile
        minimum_sustained_ratio: Ratio for minimum sustained check
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
            requirement_description=f"Heating ramp rate >= {required_rate_c_per_min} °C/min",
            measured_value=0.0,
            threshold_value=required_rate_c_per_min,
            comparator=Comparator.GTE,
            unit="°C/min",
            method="theil_sen",
            region_id=valid_ramp.region_id,
            included_rows=len(valid_ramp.included_rows),
            result=ValidationStatus.NOT_APPLICABLE,
            reason="Not a heating ramp",
        )
    
    robust_slope = ramp_metrics.robust_slope_c_per_min
    min_sustained = ramp_metrics.minimum_sustained_slope_c_per_min
    sustained_threshold = required_rate_c_per_min * minimum_sustained_ratio
    
    robust_pass = robust_slope >= required_rate_c_per_min
    sustained_pass = min_sustained >= sustained_threshold
    
    if robust_pass and sustained_pass:
        result = ValidationStatus.PASS
        reason = f"Robust slope {robust_slope:.2f} >= {required_rate_c_per_min}, sustained {min_sustained:.2f} >= {sustained_threshold:.2f}"
    elif robust_pass and not sustained_pass:
        result = ValidationStatus.PASS_WITH_WARNINGS
        reason = f"Robust slope {robust_slope:.2f} passes, but sustained {min_sustained:.2f} < {sustained_threshold:.2f}"
    else:
        result = ValidationStatus.FAIL
        reason = f"Robust slope {robust_slope:.2f} < required {required_rate_c_per_min}"
    
    validation_result = ValidationResult(
        validation_result_id=str(uuid.uuid4()),
        requirement_id="HEATING_RAMP_RATE",
        requirement_description=f"Heating ramp rate >= {required_rate_c_per_min} °C/min",
        measured_value=robust_slope,
        threshold_value=required_rate_c_per_min,
        comparator=Comparator.GTE,
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
        thresholds_used={"required": required_rate_c_per_min, "sustained_ratio": minimum_sustained_ratio},
        severity=AuditSeverity.INFO,
        category=AuditCategory.VALIDATION,
    ))
    
    return validation_result


def validate_cooling_ramp_rate(
    valid_ramp: ValidRampRegion,
    ramp_metrics: RampMetrics,
    required_rate_c_per_min: float,
    minimum_sustained_ratio: float = 0.8,
    audit_log: AuditLog | None = None,
) -> ValidationResult:
    """Validate cooling ramp rate against profile requirement.
    
    Note: Cooling rates are negative, so we compare absolute values.
    
    Args:
        valid_ramp: Valid ramp region
        ramp_metrics: Computed ramp metrics
        required_rate_c_per_min: Required cooling rate (positive value)
        minimum_sustained_ratio: Ratio for minimum sustained check
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
            requirement_description=f"Cooling ramp rate >= {required_rate_c_per_min} °C/min",
            measured_value=0.0,
            threshold_value=required_rate_c_per_min,
            comparator=Comparator.GTE,
            unit="°C/min",
            method="theil_sen",
            region_id=valid_ramp.region_id,
            included_rows=len(valid_ramp.included_rows),
            result=ValidationStatus.NOT_APPLICABLE,
            reason="Not a cooling ramp",
        )
    
    robust_slope = abs(ramp_metrics.robust_slope_c_per_min)
    min_sustained = abs(ramp_metrics.minimum_sustained_slope_c_per_min)
    sustained_threshold = required_rate_c_per_min * minimum_sustained_ratio
    
    robust_pass = robust_slope >= required_rate_c_per_min
    sustained_pass = min_sustained >= sustained_threshold
    
    if robust_pass and sustained_pass:
        result = ValidationStatus.PASS
        reason = f"Robust slope {robust_slope:.2f} >= {required_rate_c_per_min}, sustained {min_sustained:.2f} >= {sustained_threshold:.2f}"
    elif robust_pass and not sustained_pass:
        result = ValidationStatus.PASS_WITH_WARNINGS
        reason = f"Robust slope {robust_slope:.2f} passes, but sustained {min_sustained:.2f} < {sustained_threshold:.2f}"
    else:
        result = ValidationStatus.FAIL
        reason = f"Robust slope {robust_slope:.2f} < required {required_rate_c_per_min}"
    
    validation_result = ValidationResult(
        validation_result_id=str(uuid.uuid4()),
        requirement_id="COOLING_RAMP_RATE",
        requirement_description=f"Cooling ramp rate >= {required_rate_c_per_min} °C/min",
        measured_value=robust_slope,
        threshold_value=required_rate_c_per_min,
        comparator=Comparator.GTE,
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
        thresholds_used={"required": required_rate_c_per_min, "sustained_ratio": minimum_sustained_ratio},
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
