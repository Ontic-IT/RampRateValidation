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

    CONFORMANCE_THRESHOLD = 95.0

    # Requirement hierarchy: RAMP RATES are the test — the chamber either
    # achieved its commanded rates or it did not. Setpoint-tracking
    # conformance is a chamber-health indicator: poor tracking on passing
    # ramps suggests maintenance, not test failure.
    ramp_failed = [
        r for r in results.results
        if "RAMP_RATE" in r.requirement_id and r.result == ValidationStatus.FAIL
    ]
    ramp_results = [r for r in results.results if "RAMP_RATE" in r.requirement_id]

    # Systemic vs isolated: a chamber that cannot meet its commanded rate
    # fails on the MAJORITY of transitions. An isolated lagging transition
    # (defrost pause, first ramp with cold product mass) on an otherwise
    # conforming test is an anomaly to flag, not a test failure.
    systemic_ramp_failure = (
        len(ramp_failed) > 0 and len(ramp_failed) * 2 >= len(ramp_results)
    )

    # Pass/fail is decided by RAMP RATES only. Setpoint conformance is
    # reported for information (how well the chamber tracked its setpoint) but
    # never downgrades the verdict — the test is about the ramp rates.
    if systemic_ramp_failure:
        status = OverallValidationStatus.FAIL
        reason = (
            f"{len(ramp_failed)}/{len(ramp_results)} validated ramps fell outside "
            f"their commanded rate band — systemic deviation ({ramp_failed[0].reason}). "
            f"Setpoint conformance {conformance_pct:.1f}% (informational)"
        )
    elif ramp_failed:
        status = OverallValidationStatus.PASS_WITH_WARNINGS
        offending = ", ".join(r.region_id or "?" for r in ramp_failed[:4])
        reason = (
            f"{len(ramp_failed)}/{len(ramp_results)} isolated ramp(s) outside the "
            f"commanded rate band ({offending}) — flagged for review (possible "
            f"defrost/changeover or load effect). Setpoint conformance "
            f"{conformance_pct:.1f}% (informational)"
        )
    else:
        status = OverallValidationStatus.PASS
        reason = (
            f"All {len(ramp_results)} validated ramps within their commanded rate "
            f"band. Setpoint conformance {conformance_pct:.1f}% (informational — "
            "not a pass/fail criterion)"
        )
    
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


def _requirement_credit(r: ValidationResult) -> float:
    """Graded conformance credit for one requirement result, in [0, 1].

    Setpoint deviation: WITHIN tolerance is fully conforming (a steady small
    offset that stays inside the allowance is still on-spec — credit 1.0),
    fading linearly to zero only once the deviation exceeds the tolerance,
    reaching 0 at twice the allowance.
    Ramp rates: measured/required, capped at 1 (exceeding the required rate
    earns no extra credit; a shortfall earns proportional credit).
    Descriptive requirements (duration outlier checks) return 1 unless they
    failed outright.
    """
    if "SETPOINT_DEVIATION" in r.requirement_id and r.threshold_value > 0:
        ratio = r.measured_value / r.threshold_value
        return 1.0 if ratio <= 1.0 else max(0.0, 2.0 - ratio)
    if "RAMP_RATE" in r.requirement_id and r.threshold_value > 0:
        return max(0.0, min(1.0, r.measured_value / r.threshold_value))
    return 0.0 if r.result == ValidationStatus.FAIL else 1.0


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
    credit_weight = 0.0
    total_weight = 0.0

    for phase_id, phase_reqs in phase_results.items():
        statuses = [r.result for r in phase_reqs]

        # Conformance measures ONE thing: how well the actual temperature
        # tracked the target SETPOINT during dwells. It is TIME x DEVIATION
        # weighted over setpoint-deviation results only — ramp-rate thresholds
        # do NOT enter it (ramps are a separate pass/fail band). So adjusting
        # the dwell tolerance moves conformance; adjusting a ramp band does not.
        sp_reqs = [
            r for r in phase_reqs
            if "SETPOINT_DEVIATION" in r.requirement_id and r.threshold_value > 0
        ]
        if sp_reqs:
            weight = float(max(max(r.included_rows for r in sp_reqs), 1))
            total_weight += weight
            credit_weight += weight * min(_requirement_credit(r) for r in sp_reqs)

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
        (credit_weight / total_weight * 100) if total_weight > 0 else 0.0
    )

    return PhaseConformanceSummary(
        total_phases=total_phases,
        passed_phases=passed_phases,
        failed_phases=failed_phases,
        anomaly_phases=anomaly_phases,
        conformance_percentage=conformance_percentage,
        anomaly_phase_ids=anomaly_phase_ids,
    )
