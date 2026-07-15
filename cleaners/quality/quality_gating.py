"""Data quality assessment and gating (M04).

Handles:
- Data quality status determination (INVALID/INCONCLUSIVE/WARNING/ACCEPTABLE)
- Quality gating decisions
- Quality impact on validation
"""

from __future__ import annotations

from datetime import datetime

from config.constants import AuditCategory, AuditSeverity, DataQualityStatus
from models.domain import (
    AuditEntry,
    AuditLog,
    PreprocessingReport,
    RunDataQualityReport,
    ValidationDataQualityImpact,
)
from models.errors import QualityGateError


def assess_data_quality(
    preprocessing_report: PreprocessingReport,
    max_missing_data_pct: float = 5.0,
    max_duplicate_timestamp_pct: float = 1.0,
    max_gap_seconds: float = 30.0,
    min_process_duration_seconds: float = 60.0,
    max_spike_count: int = 10,
    total_rows: int = 0,
    process_duration_seconds: float = 0.0,
    audit_log: AuditLog | None = None,
) -> RunDataQualityReport:
    """Assess overall data quality from preprocessing report.
    
    Args:
        preprocessing_report: Output from preprocessing
        max_missing_data_pct: Maximum allowed missing data percentage
        max_duplicate_timestamp_pct: Maximum allowed duplicate timestamp percentage
        max_gap_seconds: Maximum allowed gap duration
        min_process_duration_seconds: Minimum required process duration
        max_spike_count: Maximum allowed spike count
        total_rows: Total row count for percentage calculations
        process_duration_seconds: Total process duration
        audit_log: Optional audit log
    
    Returns:
        RunDataQualityReport with overall status
    """
    if audit_log is None:
        audit_log = AuditLog()
    
    if total_rows == 0:
        total_rows = 1
    
    missing_pct = preprocessing_report.dropout_density_score * 100
    duplicate_pct = (len(preprocessing_report.duplicate_timestamps) / total_rows) * 100
    out_of_order_count = len(preprocessing_report.out_of_order_rows)
    irregular_pct = preprocessing_report.irregular_sampling_score * 100
    large_gap_count = len(preprocessing_report.detected_gaps)
    spike_count = len(preprocessing_report.detected_spikes)
    dropout_count = len(preprocessing_report.detected_gaps)
    
    max_gap_duration = 0.0
    for gap in preprocessing_report.detected_gaps:
        gap_duration = (gap[1] - gap[0]) * preprocessing_report.estimated_sample_interval_s
        max_gap_duration = max(max_gap_duration, gap_duration)
    
    status = DataQualityStatus.ACCEPTABLE
    notes = []
    
    if missing_pct > max_missing_data_pct * 2:
        status = DataQualityStatus.INVALID
        notes.append(f"Missing data {missing_pct:.1f}% exceeds 2× threshold")
    elif missing_pct > max_missing_data_pct:
        if status != DataQualityStatus.INVALID:
            status = DataQualityStatus.WARNING
        notes.append(f"Missing data {missing_pct:.1f}% exceeds threshold")
    
    if duplicate_pct > max_duplicate_timestamp_pct * 2:
        status = DataQualityStatus.INVALID
        notes.append(f"Duplicate timestamps {duplicate_pct:.1f}% exceeds 2× threshold")
    elif duplicate_pct > max_duplicate_timestamp_pct:
        if status not in (DataQualityStatus.INVALID,):
            status = DataQualityStatus.WARNING
        notes.append(f"Duplicate timestamps {duplicate_pct:.1f}% exceeds threshold")
    
    if out_of_order_count > total_rows * 0.1:
        status = DataQualityStatus.INVALID
        notes.append(f"Out-of-order rows {out_of_order_count} exceeds 10%")
    elif out_of_order_count > 0:
        if status not in (DataQualityStatus.INVALID,):
            status = DataQualityStatus.WARNING
        notes.append(f"{out_of_order_count} out-of-order rows detected")
    
    if max_gap_duration > max_gap_seconds * 2:
        if status != DataQualityStatus.INVALID:
            status = DataQualityStatus.INCONCLUSIVE
        notes.append(f"Gap duration {max_gap_duration:.1f}s exceeds 2× threshold")
    elif max_gap_duration > max_gap_seconds:
        if status not in (DataQualityStatus.INVALID, DataQualityStatus.INCONCLUSIVE):
            status = DataQualityStatus.WARNING
        notes.append(f"Gap duration {max_gap_duration:.1f}s exceeds threshold")
    
    if process_duration_seconds < min_process_duration_seconds:
        status = DataQualityStatus.INVALID
        notes.append(f"Process duration {process_duration_seconds:.1f}s below minimum")
    
    if spike_count > max_spike_count * 2:
        if status not in (DataQualityStatus.INVALID,):
            status = DataQualityStatus.INCONCLUSIVE
        notes.append(f"Spike count {spike_count} exceeds 2× threshold")
    elif spike_count > max_spike_count:
        if status not in (DataQualityStatus.INVALID, DataQualityStatus.INCONCLUSIVE):
            status = DataQualityStatus.WARNING
        notes.append(f"Spike count {spike_count} exceeds threshold")
    
    minimum_ramp_data = preprocessing_report.effective_data_continuity_score >= 0.8
    
    report = RunDataQualityReport(
        overall_status=status,
        missing_data_pct=missing_pct,
        duplicate_timestamp_pct=duplicate_pct,
        out_of_order_row_count=out_of_order_count,
        irregular_interval_pct=irregular_pct,
        large_gap_count=large_gap_count,
        spike_count=spike_count,
        dropout_count=dropout_count,
        process_duration_seconds=process_duration_seconds,
        minimum_ramp_data_available=minimum_ramp_data,
        quality_impact_notes="; ".join(notes) if notes else "No quality issues detected",
        gap_density_score=preprocessing_report.gap_density_score,
        dropout_density_score=preprocessing_report.dropout_density_score,
        irregular_sampling_score=preprocessing_report.irregular_sampling_score,
        effective_data_continuity_score=preprocessing_report.effective_data_continuity_score,
    )
    
    audit_log.add(AuditEntry(
        timestamp=datetime.now(),
        module_name="quality_gating",
        action="assess_data_quality",
        decision=status.value,
        reason=report.quality_impact_notes,
        severity=AuditSeverity.WARNING if status != DataQualityStatus.ACCEPTABLE else AuditSeverity.INFO,
        category=AuditCategory.QUALITY,
    ))
    
    return report


def apply_quality_gate(
    quality_report: RunDataQualityReport,
    audit_log: AuditLog | None = None,
) -> tuple[bool, ValidationDataQualityImpact]:
    """Apply quality gate based on data quality report.
    
    Args:
        quality_report: Data quality assessment
        audit_log: Optional audit log
    
    Returns:
        Tuple of (should_continue, quality_impact)
    
    Raises:
        QualityGateError: If status is INVALID
    """
    if audit_log is None:
        audit_log = AuditLog()
    
    status = quality_report.overall_status
    
    if status == DataQualityStatus.INVALID:
        audit_log.add(AuditEntry(
            timestamp=datetime.now(),
            module_name="quality_gating",
            action="quality_gate_failed",
            decision="STOP",
            reason=f"INVALID status: {quality_report.quality_impact_notes}",
            severity=AuditSeverity.ERROR,
            category=AuditCategory.QUALITY,
        ))
        raise QualityGateError(f"Data quality INVALID: {quality_report.quality_impact_notes}")
    
    blocks_pass_fail = status == DataQualityStatus.INCONCLUSIVE
    affected_requirements = []
    
    if blocks_pass_fail:
        affected_requirements = ["ALL"]
        audit_log.add(AuditEntry(
            timestamp=datetime.now(),
            module_name="quality_gating",
            action="quality_gate_inconclusive",
            decision="CONTINUE_BLOCKED",
            reason=f"INCONCLUSIVE status blocks PASS/FAIL: {quality_report.quality_impact_notes}",
            severity=AuditSeverity.WARNING,
            category=AuditCategory.QUALITY,
        ))
    else:
        audit_log.add(AuditEntry(
            timestamp=datetime.now(),
            module_name="quality_gating",
            action="quality_gate_passed",
            decision="CONTINUE",
            reason=f"Quality status {status.value}: {quality_report.quality_impact_notes}",
            severity=AuditSeverity.INFO,
            category=AuditCategory.QUALITY,
        ))
    
    impact = ValidationDataQualityImpact(
        blocks_pass_fail=blocks_pass_fail,
        affected_requirement_ids=affected_requirements,
        reason=quality_report.quality_impact_notes,
    )
    
    return True, impact
