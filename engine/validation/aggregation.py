"""Validation status aggregation (M11).

Handles:
- Validation status aggregation across all results
- Quality-blocked flag handling
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from config.constants import (
    AuditCategory,
    AuditSeverity,
    OverallValidationStatus,
    ValidationStatus,
)
from models.domain import (
    AuditEntry,
    AuditLog,
    OverallStatus,
    PhaseConformanceSummary,
    ValidationDataQualityImpact,
    ValidationResult,
    ValidationResults,
)


def aggregate_validation_status(
    results: ValidationResults,
    quality_impact: ValidationDataQualityImpact | None = None,
    audit_log: AuditLog | None = None,
) -> OverallStatus:
    """Aggregate all validation results into overall status.
    
    Rules (Conformance-based):
    - If quality_blocked is True, status must be INCONCLUSIVE
    - Compute phase conformance percentage
    - If conformance >= 95%, overall is PASS
    - If conformance < 95%, overall is FAIL
    
    Args:
        results: All validation results
        quality_impact: Data quality impact (may block pass/fail)
        audit_log: Optional audit log
    
    Returns:
        OverallStatus with aggregated decision
    """
    if audit_log is None:
        audit_log = AuditLog()
    
    if quality_impact and quality_impact.blocks_pass_fail:
        status = OverallValidationStatus.INCONCLUSIVE
        reason = f"Data quality blocks pass/fail: {quality_impact.reason}"
        
        audit_log.add(AuditEntry(
            timestamp=datetime.now(),
            module_name="aggregation",
            action="aggregate_validation_status",
            decision=status.value,
            reason=reason,
            severity=AuditSeverity.WARNING,
            category=AuditCategory.VALIDATION,
        ))
        
        return OverallStatus(status=status, reason=reason)
    
    # Compute phase conformance
    phase_conformance = compute_phase_conformance(results)
    conformance_pct = phase_conformance.conformance_percentage
    
    # Use 95% conformance threshold for pass/fail
    CONFORMANCE_THRESHOLD = 95.0
    
    if conformance_pct >= CONFORMANCE_THRESHOLD:
        status = OverallValidationStatus.PASS
        reason = f"Conformance {conformance_pct:.1f}% >= {CONFORMANCE_THRESHOLD}% threshold ({phase_conformance.passed_phases}/{phase_conformance.total_phases} phases)"
    else:
        status = OverallValidationStatus.FAIL
        reason = f"Conformance {conformance_pct:.1f}% < {CONFORMANCE_THRESHOLD}% threshold ({phase_conformance.passed_phases}/{phase_conformance.total_phases} phases)"
    
    audit_log.add(AuditEntry(
        timestamp=datetime.now(),
        module_name="aggregation",
        action="aggregate_validation_status",
        decision=status.value,
        reason=reason,
        thresholds_used={
            "conformance_percentage": conformance_pct,
            "conformance_threshold": CONFORMANCE_THRESHOLD,
            "total_phases": phase_conformance.total_phases,
            "passed_phases": phase_conformance.passed_phases,
            "failed_phases": phase_conformance.failed_phases,
        },
        severity=AuditSeverity.INFO,
        category=AuditCategory.VALIDATION,
    ))
    
    return OverallStatus(status=status, reason=reason)


def compute_phase_conformance(results: ValidationResults) -> PhaseConformanceSummary:
    """Compute per-phase conformance summary from ValidationResults.

    A "phase" = one Region or one ValidRampRegion-or-dwell-equivalent,
    identified by region_id. Each phase's conformance is determined
    by the worst result for that phase across all requirements.

    Args:
        results: All validation results

    Returns:
        PhaseConformanceSummary with aggregated counts
    """
    phase_results: dict[str, list[ValidationResult]] = {}
    for r in results.results:
        if r.result == ValidationStatus.NOT_APPLICABLE:
            continue
        phase_id = r.region_id or r.cycle_id or "global"
        phase_results.setdefault(phase_id, []).append(r)

    total_phases = len(phase_results)
    passed_phases = 0
    failed_phases = 0
    anomaly_phases = 0
    anomaly_phase_ids: list[str] = []

    for phase_id, phase_reqs in phase_results.items():
        statuses = [r.result for r in phase_reqs]
        if ValidationStatus.FAIL in statuses:
            failed_phases += 1
            anomaly_phases += 1
            anomaly_phase_ids.append(phase_id)
        elif ValidationStatus.INCONCLUSIVE in statuses:
            anomaly_phases += 1
            anomaly_phase_ids.append(phase_id)
        elif ValidationStatus.PASS_WITH_WARNINGS in statuses:
            passed_phases += 1
            anomaly_phases += 1
            anomaly_phase_ids.append(phase_id)
        elif ValidationStatus.PASS in statuses:
            passed_phases += 1

    conformance_percentage = (
        (passed_phases / total_phases * 100) if total_phases > 0 else 0.0
    )

    return PhaseConformanceSummary(
        total_phases=total_phases,
        passed_phases=passed_phases,
        failed_phases=failed_phases,
        anomaly_phases=anomaly_phases,
        conformance_percentage=conformance_percentage,
        anomaly_phase_ids=anomaly_phase_ids,
    )
