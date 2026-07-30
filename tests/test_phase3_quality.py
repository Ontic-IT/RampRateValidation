"""Unit tests for Phase 3: cleaners/quality/quality_gating.py — data quality assessment."""

import pytest

from config.constants import DataQualityStatus
from cleaners.quality.quality_gating import assess_data_quality, apply_quality_gate
from models.domain import (
    AuditLog,
    PreprocessingReport,
    RunDataQualityReport,
    ValidationDataQualityImpact,
)
from models.errors import QualityGateError


class TestAssessDataQuality:
    @pytest.fixture
    def clean_report(self):
        return PreprocessingReport(
            estimated_sample_interval_s=1.0,
            noise_floor_c=0.05,
            slope_noise_floor_c_per_min=0.3,
            temperature_MAD_baseline=0.05,
            rolling_window_seconds_used=30.0,
            detected_spikes=[],
            detected_gaps=[],
            duplicate_timestamps=[],
            out_of_order_rows=[],
            gap_density_score=0.0,
            dropout_density_score=0.0,
            irregular_sampling_score=0.0,
            effective_data_continuity_score=1.0,
        )

    def test_clean_data_acceptable(self, clean_report):
        result = assess_data_quality(
            clean_report,
            total_rows=1000,
            process_duration_seconds=600.0,
        )
        assert result.overall_status == DataQualityStatus.ACCEPTABLE

    def test_high_missing_data_warning(self, clean_report):
        clean_report.dropout_density_score = 0.06
        result = assess_data_quality(
            clean_report,
            max_missing_data_pct=5.0,
            total_rows=1000,
            process_duration_seconds=600.0,
        )
        assert result.overall_status == DataQualityStatus.WARNING

    def test_very_high_missing_data_invalid(self, clean_report):
        clean_report.dropout_density_score = 0.15
        result = assess_data_quality(
            clean_report,
            max_missing_data_pct=5.0,
            total_rows=1000,
            process_duration_seconds=600.0,
        )
        assert result.overall_status == DataQualityStatus.INVALID

    def test_duplicate_timestamps_warning(self, clean_report):
        clean_report.duplicate_timestamps = list(range(20))
        result = assess_data_quality(
            clean_report,
            max_duplicate_timestamp_pct=1.0,
            total_rows=1000,
            process_duration_seconds=600.0,
        )
        assert result.overall_status == DataQualityStatus.WARNING

    def test_out_of_order_rows_warning(self, clean_report):
        clean_report.out_of_order_rows = [10, 20, 30]
        result = assess_data_quality(
            clean_report,
            total_rows=1000,
            process_duration_seconds=600.0,
        )
        assert result.overall_status == DataQualityStatus.WARNING

    def test_large_gap_inconclusive(self, clean_report):
        clean_report.detected_gaps = [(100, 200)]
        result = assess_data_quality(
            clean_report,
            max_gap_seconds=30.0,
            total_rows=1000,
            process_duration_seconds=600.0,
        )
        assert result.overall_status in (DataQualityStatus.WARNING, DataQualityStatus.INCONCLUSIVE)

    def test_short_process_duration_invalid(self, clean_report):
        result = assess_data_quality(
            clean_report,
            min_process_duration_seconds=60.0,
            total_rows=1000,
            process_duration_seconds=30.0,
        )
        assert result.overall_status == DataQualityStatus.INVALID

    def test_high_spike_count_inconclusive(self, clean_report):
        clean_report.detected_spikes = list(range(25))
        result = assess_data_quality(
            clean_report,
            max_spike_count=10,
            total_rows=1000,
            process_duration_seconds=600.0,
        )
        assert result.overall_status == DataQualityStatus.INCONCLUSIVE

    def test_quality_impact_notes_populated(self, clean_report):
        clean_report.dropout_density_score = 0.06
        result = assess_data_quality(
            clean_report,
            max_missing_data_pct=5.0,
            total_rows=1000,
            process_duration_seconds=600.0,
        )
        assert "Missing data" in result.quality_impact_notes

    def test_audit_log_records_assessment(self, clean_report):
        audit_log = AuditLog()
        assess_data_quality(
            clean_report,
            total_rows=1000,
            process_duration_seconds=600.0,
            audit_log=audit_log,
        )
        entries = [e for e in audit_log.entries if e.action == "assess_data_quality"]
        assert len(entries) == 1


class TestApplyQualityGate:
    def test_acceptable_continues(self):
        report = RunDataQualityReport(overall_status=DataQualityStatus.ACCEPTABLE)
        should_continue, impact = apply_quality_gate(report)
        assert should_continue is True
        assert impact.blocks_pass_fail is False

    def test_warning_continues(self):
        report = RunDataQualityReport(overall_status=DataQualityStatus.WARNING)
        should_continue, impact = apply_quality_gate(report)
        assert should_continue is True
        assert impact.blocks_pass_fail is False

    def test_inconclusive_blocks_pass_fail(self):
        report = RunDataQualityReport(overall_status=DataQualityStatus.INCONCLUSIVE)
        should_continue, impact = apply_quality_gate(report)
        assert should_continue is True
        assert impact.blocks_pass_fail is True
        assert "ALL" in impact.affected_requirement_ids

    def test_invalid_raises_error(self):
        report = RunDataQualityReport(
            overall_status=DataQualityStatus.INVALID,
            quality_impact_notes="Test invalid",
        )
        with pytest.raises(QualityGateError, match="INVALID"):
            apply_quality_gate(report)

    def test_audit_log_records_gate_decision(self):
        report = RunDataQualityReport(overall_status=DataQualityStatus.ACCEPTABLE)
        audit_log = AuditLog()
        apply_quality_gate(report, audit_log=audit_log)
        entries = [e for e in audit_log.entries if "quality_gate" in e.action]
        assert len(entries) == 1
