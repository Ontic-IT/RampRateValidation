"""Unit tests for Phase 7: Reporting (M14)."""

import pytest
from datetime import datetime
from pathlib import Path
import tempfile

from engine.reporting.report_builder import (
    generate_report_payload,
    build_executive_summary,
)
from engine.reporting.pdf_generator import generate_pdf_report
from engine.reporting.excel_generator import generate_excel_report
from models.domain import (
    AnalysisContext,
    AnalysisRequest,
    AuditLog,
    FileMetadata,
    OverallStatus,
    PhaseConformanceSummary,
    ReportPackage,
)
from config.constants import OverallValidationStatus


def _make_minimal_context() -> AnalysisContext:
    """Create minimal AnalysisContext for testing."""
    request = AnalysisRequest(
        file_path="test.csv",
        profile_path="profile.yaml",
        output_dir=".",
    )
    return AnalysisContext(
        request=request,
        file_metadata=FileMetadata(
            source_file_path="test.csv",
            detected_delimiter=",",
            detected_encoding="utf-8",
            raw_row_count=1000,
            usable_row_count=950,
            selected_temperature_channel="TC1",
            selected_setpoint_channel="SP1",
        ),
        overall_validation_status=OverallStatus(
            status=OverallValidationStatus.PASS,
            reason="All requirements passed",
        ),
        phase_conformance=PhaseConformanceSummary(
            total_phases=10,
            passed_phases=10,
            failed_phases=0,
            anomaly_phases=0,
            conformance_percentage=100.0,
            anomaly_phase_ids=[],
        ),
    )


class TestGenerateReportPayload:
    def test_generates_expected_sections(self):
        context = _make_minimal_context()

        report = generate_report_payload(context)

        expected_sections = [
            "executive_summary",
            "ramp_rate_validation_summary",
            "interactive_validation_data",
            "region_classification_summary",
            "dwell_calibration_summary",
            "visualisation",
        ]
        assert len(report.sections) == len(expected_sections)
        for section in expected_sections:
            assert section in report.sections
        # Removed sections must not reappear.
        for removed in (
            "input_file_summary", "data_quality_summary", "process_boundary_summary",
            "setpoint_inference_summary", "overshoot_correction_summary", "audit_trail",
            "cycle_level_summary", "profile_consistency_summary", "algorithm_appendix",
        ):
            assert removed not in report.sections

    def test_each_section_has_title(self):
        context = _make_minimal_context()
        
        report = generate_report_payload(context)
        
        for section_name, section_data in report.sections.items():
            assert "title" in section_data
            assert isinstance(section_data["title"], str)

    def test_not_applicable_sections_have_note(self):
        context = _make_minimal_context()
        # No metrics set, so dwell calibration should carry a note.
        report = generate_report_payload(context)

        assert "note" in report.sections["dwell_calibration_summary"]


class TestBuildExecutiveSummary:
    def test_includes_overall_status(self):
        context = _make_minimal_context()
        
        summary = build_executive_summary(context)
        
        assert summary["overall_status"] == "PASS"
        assert summary["status_reason"] == "All requirements passed"

    def test_includes_file_name(self):
        context = _make_minimal_context()
        
        summary = build_executive_summary(context)
        
        assert summary["file_name"] == "test.csv"

    def test_includes_phase_conformance(self):
        context = _make_minimal_context()
        
        summary = build_executive_summary(context)
        
        assert "phase_conformance" in summary
        pc = summary["phase_conformance"]
        assert pc["total_phases"] == 10
        assert pc["passed_phases"] == 10
        assert pc["conformance_percentage"] == 100.0

    def test_handles_missing_profile(self):
        context = _make_minimal_context()
        context.profile = None
        
        summary = build_executive_summary(context)
        
        assert summary["profile_name"] == ""

    def test_includes_counts(self):
        context = _make_minimal_context()
        
        summary = build_executive_summary(context)
        
        assert "cycle_count" in summary
        assert "region_count" in summary
        assert "valid_ramp_count" in summary


class TestGeneratePDFReport:
    def test_generates_pdf_file(self):
        report = ReportPackage(sections={
            "executive_summary": {
                "title": "Executive Summary",
                "overall_status": "PASS",
            }
        })
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = str(Path(tmpdir) / "report.pdf")
            
            result_path = generate_pdf_report(report, output_path)
            
            # Should return a path (either .pdf or .html fallback)
            assert result_path.endswith(".pdf") or result_path.endswith(".html")
            assert Path(result_path).exists()

    def test_creates_html_fallback_if_weasyprint_missing(self):
        """If WeasyPrint not installed, should create HTML fallback."""
        report = ReportPackage(sections={
            "executive_summary": {
                "title": "Executive Summary",
                "overall_status": "PASS",
            }
        })
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = str(Path(tmpdir) / "report.pdf")
            
            result_path = generate_pdf_report(report, output_path)
            
            # Should exist
            assert Path(result_path).exists()

    def test_html_contains_section_titles(self):
        report = ReportPackage(sections={
            "executive_summary": {
                "title": "Executive Summary",
                "overall_status": "PASS",
            },
            "input_file_summary": {
                "title": "Input File Summary",
                "file_name": "test.csv",
            }
        })
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = str(Path(tmpdir) / "report.pdf")
            result_path = generate_pdf_report(report, output_path)
            
            if result_path.endswith(".html"):
                content = Path(result_path).read_text(encoding="utf-8")
                assert "Executive Summary" in content
                assert "Input File Summary" in content


class TestGenerateExcelReport:
    def test_generates_excel_file(self):
        report = ReportPackage(sections={
            "executive_summary": {
                "title": "Executive Summary",
                "overall_status": "PASS",
            }
        })
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = str(Path(tmpdir) / "report.xlsx")
            
            result_path = generate_excel_report(report, output_path)
            
            # Should return a path (either .xlsx or .csv fallback)
            assert result_path.endswith(".xlsx") or result_path.endswith(".csv")
            assert Path(result_path).exists()

    def test_creates_csv_fallback_if_openpyxl_missing(self):
        """If openpyxl not installed, should create CSV fallback."""
        report = ReportPackage(sections={
            "executive_summary": {
                "title": "Executive Summary",
                "overall_status": "PASS",
            }
        })
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = str(Path(tmpdir) / "report.xlsx")
            
            result_path = generate_excel_report(report, output_path)
            
            # Should exist
            assert Path(result_path).exists()

    def test_csv_fallback_contains_section_titles(self):
        report = ReportPackage(sections={
            "executive_summary": {
                "title": "Executive Summary",
                "overall_status": "PASS",
            }
        })
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = str(Path(tmpdir) / "report.xlsx")
            result_path = generate_excel_report(report, output_path)
            
            if result_path.endswith(".csv"):
                content = Path(result_path).read_text(encoding="utf-8")
                assert "Executive Summary" in content


class TestReportPackageModel:
    def test_can_instantiate_with_sections(self):
        report = ReportPackage(sections={
            "section1": {"title": "Section 1"},
            "section2": {"title": "Section 2"},
        })
        
        assert len(report.sections) == 2
        assert "section1" in report.sections

    def test_sections_can_contain_nested_data(self):
        report = ReportPackage(sections={
            "executive_summary": {
                "title": "Executive Summary",
                "nested_data": {
                    "key1": "value1",
                    "key2": [1, 2, 3],
                }
            }
        })
        
        assert report.sections["executive_summary"]["nested_data"]["key1"] == "value1"
        assert report.sections["executive_summary"]["nested_data"]["key2"] == [1, 2, 3]
