"""Cycle validation rules (M11).

Handles:
- Cycle count validation
- Profile sequence validation
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
import uuid

from config.constants import (
    AuditCategory,
    AuditSeverity,
    Comparator,
    CycleStatus,
    RegionType,
    ValidationStatus,
)
from models.domain import (
    AuditEntry,
    AuditLog,
    Cycle,
    CycleList,
    Region,
    ValidationResult,
)


def validate_cycle_count(
    cycles: CycleList,
    minimum_cycles: int,
    maximum_cycles: int | None = None,
    audit_log: AuditLog | None = None,
) -> ValidationResult:
    """Validate cycle count against profile requirement.
    
    Args:
        cycles: Detected cycles
        minimum_cycles: Minimum required cycles
        maximum_cycles: Maximum allowed cycles (optional)
        audit_log: Optional audit log
    
    Returns:
        ValidationResult with pass/fail decision
    """
    if audit_log is None:
        audit_log = AuditLog()
    
    complete_cycles = [c for c in cycles.cycles if c.status == CycleStatus.COMPLETE]
    cycle_count = len(complete_cycles)
    
    passes_min = cycle_count >= minimum_cycles
    passes_max = maximum_cycles is None or cycle_count <= maximum_cycles
    
    if passes_min and passes_max:
        result = ValidationStatus.PASS
        if maximum_cycles:
            reason = f"Cycle count {cycle_count} within range [{minimum_cycles}, {maximum_cycles}]"
        else:
            reason = f"Cycle count {cycle_count} >= minimum {minimum_cycles}"
    elif not passes_min:
        result = ValidationStatus.FAIL
        reason = f"Cycle count {cycle_count} < minimum {minimum_cycles}"
    else:
        result = ValidationStatus.FAIL
        reason = f"Cycle count {cycle_count} > maximum {maximum_cycles}"
    
    validation_result = ValidationResult(
        validation_result_id=str(uuid.uuid4()),
        requirement_id="CYCLE_COUNT",
        requirement_description=f"Cycle count >= {minimum_cycles}" + (f" and <= {maximum_cycles}" if maximum_cycles else ""),
        measured_value=float(cycle_count),
        threshold_value=float(minimum_cycles),
        comparator=Comparator.GTE,
        unit="cycles",
        method="cycle_detection",
        included_rows=sum(c.end_row - c.start_row + 1 for c in complete_cycles),
        result=result,
        reason=reason,
    )
    
    audit_log.add(AuditEntry(
        timestamp=datetime.now(),
        module_name="cycle_rules",
        action="validate_cycle_count",
        output_reference=validation_result.validation_result_id,
        decision=result.value,
        reason=reason,
        thresholds_used={"minimum": minimum_cycles, "maximum": maximum_cycles or -1},
        severity=AuditSeverity.INFO,
        category=AuditCategory.VALIDATION,
    ))
    
    return validation_result


def validate_profile_sequence(
    regions: list[Region],
    expected_sequence: list[str],
    audit_log: AuditLog | None = None,
) -> ValidationResult:
    """Validate region sequence against expected profile sequence.
    
    Args:
        regions: Classified regions
        expected_sequence: Expected sequence of region types (e.g., ["HEATING_RAMP", "HOT_DWELL", "COOLING_RAMP", "COLD_DWELL"])
        audit_log: Optional audit log
    
    Returns:
        ValidationResult with pass/fail decision
    """
    if audit_log is None:
        audit_log = AuditLog()
    
    if not expected_sequence:
        return ValidationResult(
            validation_result_id=str(uuid.uuid4()),
            requirement_id="PROFILE_SEQUENCE",
            requirement_description="Profile sequence validation",
            measured_value=0.0,
            threshold_value=0.0,
            comparator=Comparator.EQ,
            unit="sequence",
            method="sequence_matching",
            result=ValidationStatus.NOT_APPLICABLE,
            reason="No expected sequence defined",
        )
    
    actual_sequence = [r.primary_classification.value for r in regions]
    
    matches, match_details = _check_sequence_match(actual_sequence, expected_sequence)
    
    if matches:
        result = ValidationStatus.PASS
        reason = f"Sequence matches expected pattern: {' -> '.join(expected_sequence)}"
    else:
        result = ValidationStatus.FAIL
        reason = f"Sequence mismatch: {match_details}"
    
    validation_result = ValidationResult(
        validation_result_id=str(uuid.uuid4()),
        requirement_id="PROFILE_SEQUENCE",
        requirement_description=f"Profile sequence: {' -> '.join(expected_sequence)}",
        measured_value=1.0 if matches else 0.0,
        threshold_value=1.0,
        comparator=Comparator.EQ,
        unit="sequence",
        method="sequence_matching",
        included_rows=sum(r.end_row - r.start_row + 1 for r in regions),
        result=result,
        reason=reason,
    )
    
    audit_log.add(AuditEntry(
        timestamp=datetime.now(),
        module_name="cycle_rules",
        action="validate_profile_sequence",
        output_reference=validation_result.validation_result_id,
        decision=result.value,
        reason=reason,
        severity=AuditSeverity.INFO,
        category=AuditCategory.VALIDATION,
    ))
    
    return validation_result


def _check_sequence_match(
    actual: list[str],
    expected: list[str],
) -> tuple[bool, str]:
    """Check if actual sequence contains expected pattern.
    
    Allows for additional regions (transients, corrections) between expected regions.
    """
    if not expected:
        return True, "No expected sequence"
    
    if not actual:
        return False, "No regions detected"
    
    expected_idx = 0
    
    for region_type in actual:
        if expected_idx < len(expected) and region_type == expected[expected_idx]:
            expected_idx += 1
    
    if expected_idx >= len(expected):
        return True, "All expected regions found in order"
    else:
        missing = expected[expected_idx:]
        return False, f"Missing regions: {', '.join(missing)}"
