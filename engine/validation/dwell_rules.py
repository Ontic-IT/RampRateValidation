"""Dwell validation rules (M11).

Handles:
- Dwell duration validation
- Setpoint deviation validation
- Overshoot validation
- Settling time validation
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
import uuid

from config.constants import (
    AuditCategory,
    AuditSeverity,
    Comparator,
    RegionType,
    ValidationStatus,
)
from models.domain import (
    AuditEntry,
    AuditLog,
    DwellMetrics,
    Region,
    ValidationResult,
)


def validate_dwell_duration(
    region: Region,
    dwell_metrics: DwellMetrics,
    median_duration_seconds: float,
    duration_std_dev: float,
    sigma_threshold: float = 2.0,
    audit_log: AuditLog | None = None,
) -> ValidationResult:
    """Validate dwell duration using median-based statistical deviation.
    
    Flags dwells that deviate >2σ from the trace median duration.
    This identifies anomalous dwell durations relative to the trace's typical behavior.
    
    Args:
        region: Dwell region
        dwell_metrics: Computed dwell metrics
        median_duration_seconds: Median dwell duration across all dwells in trace
        duration_std_dev: Standard deviation of dwell durations in trace
        sigma_threshold: Number of standard deviations for flagging (default: 2.0)
        audit_log: Optional audit log
    
    Returns:
        ValidationResult with pass/fail decision
    """
    if audit_log is None:
        audit_log = AuditLog()
    
    if region.primary_classification not in (
        RegionType.HOT_DWELL, RegionType.COLD_DWELL, RegionType.AMBIENT_START
    ):
        return ValidationResult(
            validation_result_id=str(uuid.uuid4()),
            requirement_id="DWELL_DURATION",
            requirement_description=f"Dwell duration within {sigma_threshold}σ of median",
            measured_value=0.0,
            threshold_value=median_duration_seconds,
            comparator=Comparator.WITHIN_RANGE,
            unit="s",
            method="median_deviation",
            region_id=region.region_id,
            result=ValidationStatus.NOT_APPLICABLE,
            reason="Not a dwell region",
        )
    
    duration = region.duration_seconds
    deviation_from_median = abs(duration - median_duration_seconds)
    max_allowed_deviation = sigma_threshold * duration_std_dev
    
    if deviation_from_median <= max_allowed_deviation:
        result = ValidationStatus.PASS
        reason = f"Duration {duration:.1f}s within {sigma_threshold}σ of median {median_duration_seconds:.1f}s (deviation: {deviation_from_median:.1f}s <= {max_allowed_deviation:.1f}s)"
    else:
        # A statistically unusual duration is an ANOMALY to surface, not a
        # compliance failure — duration outliers say nothing about whether
        # the chamber held its setpoint.
        result = ValidationStatus.PASS_WITH_WARNINGS
        reason = f"Anomalous duration: {duration:.1f}s deviates >{sigma_threshold}σ from median {median_duration_seconds:.1f}s (deviation: {deviation_from_median:.1f}s > {max_allowed_deviation:.1f}s) — flagged for review"
    
    validation_result = ValidationResult(
        validation_result_id=str(uuid.uuid4()),
        requirement_id="DWELL_DURATION",
        requirement_description=f"Dwell duration within {sigma_threshold}σ of median {median_duration_seconds:.1f}s",
        measured_value=duration,
        threshold_value=median_duration_seconds,
        comparator=Comparator.WITHIN_RANGE,
        unit="s",
        method="median_deviation",
        region_id=region.region_id,
        result=result,
        reason=reason,
    )
    
    audit_log.add(AuditEntry(
        timestamp=datetime.now(),
        module_name="dwell_rules",
        action="validate_dwell_duration",
        input_reference=region.region_id,
        output_reference=validation_result.validation_result_id,
        decision=result.value,
        reason=reason,
        thresholds_used={
            "median_duration": median_duration_seconds,
            "std_dev": duration_std_dev,
            "sigma_threshold": sigma_threshold,
            "max_allowed_deviation": max_allowed_deviation,
        },
        severity=AuditSeverity.INFO,
        category=AuditCategory.VALIDATION,
    ))
    
    return validation_result


def validate_setpoint_deviation(
    region: Region,
    dwell_metrics: DwellMetrics,
    maximum_deviation_c: float,
    audit_log: AuditLog | None = None,
) -> ValidationResult:
    """Validate setpoint deviation against profile requirement.
    
    Args:
        region: Dwell region
        dwell_metrics: Computed dwell metrics
        maximum_deviation_c: Maximum allowed deviation from setpoint
        audit_log: Optional audit log
    
    Returns:
        ValidationResult with pass/fail decision
    """
    if audit_log is None:
        audit_log = AuditLog()
    
    if region.primary_classification not in (
        RegionType.HOT_DWELL, RegionType.COLD_DWELL
    ):
        return ValidationResult(
            validation_result_id=str(uuid.uuid4()),
            requirement_id="SETPOINT_DEVIATION",
            requirement_description=f"Setpoint deviation <= {maximum_deviation_c} °C",
            measured_value=0.0,
            threshold_value=maximum_deviation_c,
            comparator=Comparator.LTE,
            unit="°C",
            method="mean_deviation",
            region_id=region.region_id,
            result=ValidationStatus.NOT_APPLICABLE,
            reason="Not a dwell region",
        )
    
    deviation = dwell_metrics.setpoint_deviation_c or 0.0
    
    if deviation <= maximum_deviation_c:
        result = ValidationStatus.PASS
        reason = f"Deviation {deviation:.2f}°C <= allowed {maximum_deviation_c}°C"
    else:
        result = ValidationStatus.FAIL
        reason = f"Deviation {deviation:.2f}°C > allowed {maximum_deviation_c}°C"
    
    validation_result = ValidationResult(
        validation_result_id=str(uuid.uuid4()),
        requirement_id="SETPOINT_DEVIATION",
        requirement_description=f"Setpoint deviation <= {maximum_deviation_c} °C",
        measured_value=deviation,
        threshold_value=maximum_deviation_c,
        comparator=Comparator.LTE,
        unit="°C",
        method="mean_deviation",
        region_id=region.region_id,
        result=result,
        reason=reason,
    )
    
    audit_log.add(AuditEntry(
        timestamp=datetime.now(),
        module_name="dwell_rules",
        action="validate_setpoint_deviation",
        input_reference=region.region_id,
        output_reference=validation_result.validation_result_id,
        decision=result.value,
        reason=reason,
        thresholds_used={"maximum_deviation": maximum_deviation_c},
        severity=AuditSeverity.INFO,
        category=AuditCategory.VALIDATION,
    ))
    
    return validation_result


def validate_overshoot(
    region: Region,
    dwell_metrics: DwellMetrics,
    maximum_overshoot_c: float,
    audit_log: AuditLog | None = None,
) -> ValidationResult:
    """Validate overshoot magnitude against profile requirement.
    
    Args:
        region: Dwell region (overshoot measured at dwell entry)
        dwell_metrics: Computed dwell metrics
        maximum_overshoot_c: Maximum allowed overshoot
        audit_log: Optional audit log
    
    Returns:
        ValidationResult with pass/fail decision
    """
    if audit_log is None:
        audit_log = AuditLog()
    
    if region.primary_classification not in (
        RegionType.HOT_DWELL, RegionType.COLD_DWELL
    ):
        return ValidationResult(
            validation_result_id=str(uuid.uuid4()),
            requirement_id="OVERSHOOT",
            requirement_description=f"Overshoot <= {maximum_overshoot_c} °C",
            measured_value=0.0,
            threshold_value=maximum_overshoot_c,
            comparator=Comparator.LTE,
            unit="°C",
            method="peak_detection",
            region_id=region.region_id,
            result=ValidationStatus.NOT_APPLICABLE,
            reason="Not a dwell region",
        )
    
    overshoot = dwell_metrics.overshoot_magnitude_c or 0.0
    
    if overshoot <= maximum_overshoot_c:
        result = ValidationStatus.PASS
        reason = f"Overshoot {overshoot:.2f}°C <= allowed {maximum_overshoot_c}°C"
    else:
        result = ValidationStatus.FAIL
        reason = f"Overshoot {overshoot:.2f}°C > allowed {maximum_overshoot_c}°C"
    
    validation_result = ValidationResult(
        validation_result_id=str(uuid.uuid4()),
        requirement_id="OVERSHOOT",
        requirement_description=f"Overshoot <= {maximum_overshoot_c} °C",
        measured_value=overshoot,
        threshold_value=maximum_overshoot_c,
        comparator=Comparator.LTE,
        unit="°C",
        method="peak_detection",
        region_id=region.region_id,
        result=result,
        reason=reason,
    )
    
    audit_log.add(AuditEntry(
        timestamp=datetime.now(),
        module_name="dwell_rules",
        action="validate_overshoot",
        input_reference=region.region_id,
        output_reference=validation_result.validation_result_id,
        decision=result.value,
        reason=reason,
        thresholds_used={"maximum_overshoot": maximum_overshoot_c},
        severity=AuditSeverity.INFO,
        category=AuditCategory.VALIDATION,
    ))
    
    return validation_result


def validate_settling_time(
    region: Region,
    dwell_metrics: DwellMetrics,
    maximum_settling_time_seconds: float,
    audit_log: AuditLog | None = None,
) -> ValidationResult:
    """Validate settling time against profile requirement.
    
    Args:
        region: Dwell region
        dwell_metrics: Computed dwell metrics
        maximum_settling_time_seconds: Maximum allowed settling time
        audit_log: Optional audit log
    
    Returns:
        ValidationResult with pass/fail decision
    """
    if audit_log is None:
        audit_log = AuditLog()
    
    if region.primary_classification not in (
        RegionType.HOT_DWELL, RegionType.COLD_DWELL
    ):
        return ValidationResult(
            validation_result_id=str(uuid.uuid4()),
            requirement_id="SETTLING_TIME",
            requirement_description=f"Settling time <= {maximum_settling_time_seconds} s",
            measured_value=0.0,
            threshold_value=maximum_settling_time_seconds,
            comparator=Comparator.LTE,
            unit="s",
            method="tolerance_band_entry",
            region_id=region.region_id,
            result=ValidationStatus.NOT_APPLICABLE,
            reason="Not a dwell region",
        )
    
    settling_time = dwell_metrics.settling_time_seconds or 0.0
    
    if settling_time <= maximum_settling_time_seconds:
        result = ValidationStatus.PASS
        reason = f"Settling time {settling_time:.1f}s <= allowed {maximum_settling_time_seconds}s"
    else:
        result = ValidationStatus.FAIL
        reason = f"Settling time {settling_time:.1f}s > allowed {maximum_settling_time_seconds}s"
    
    validation_result = ValidationResult(
        validation_result_id=str(uuid.uuid4()),
        requirement_id="SETTLING_TIME",
        requirement_description=f"Settling time <= {maximum_settling_time_seconds} s",
        measured_value=settling_time,
        threshold_value=maximum_settling_time_seconds,
        comparator=Comparator.LTE,
        unit="s",
        method="tolerance_band_entry",
        region_id=region.region_id,
        result=result,
        reason=reason,
    )
    
    audit_log.add(AuditEntry(
        timestamp=datetime.now(),
        module_name="dwell_rules",
        action="validate_settling_time",
        input_reference=region.region_id,
        output_reference=validation_result.validation_result_id,
        decision=result.value,
        reason=reason,
        thresholds_used={"maximum_settling_time": maximum_settling_time_seconds},
        severity=AuditSeverity.INFO,
        category=AuditCategory.VALIDATION,
    ))
    
    return validation_result
