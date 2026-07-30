"""Unit tests for Phase 7: Visualisation (M13)."""

import pytest
from datetime import datetime

from engine.visualisation.chart_builder import (
    build_temperature_trace_chart,
    build_region_overlay,
    build_cycle_boundary_markers,
    build_annotations,
    build_complete_visualisation,
    REGION_COLOURS,
)
from models.domain import (
    AuditLog,
    CanonicalTraceRow,
    ClassifiedTrace,
    Cycle,
    CycleList,
    Region,
    RegionList,
    ValidationResult,
    ValidationResults,
)
from config.constants import Comparator, RegionType, ValidationStatus, CycleStatus


def _make_trace_row(elapsed: float, temp_raw: float, temp_analysis: float | None = None) -> CanonicalTraceRow:
    return CanonicalTraceRow(
        timestamp=datetime(2025, 1, 1, 0, 0, 0),
        elapsed_seconds=elapsed,
        elapsed_minutes=elapsed / 60.0,
        temperature_c_raw=temp_raw,
        temperature_c_analysis_signal=temp_analysis,
        setpoint_c=None,
        channel="TC1",
        source_row=0,
        source_file="test.csv",
        sample_interval_seconds=1.0,
        local_slope_c_per_min=None,
        rolling_slope_c_per_min=None,
        rolling_temperature_median=None,
        rolling_temperature_MAD=None,
        second_derivative=None,
        direction_of_travel=None,
        data_quality_flags=[],
        region_id=None,
        classification_label=None,
    )


def _make_region(region_id: str, classification: RegionType, start: int, end: int) -> Region:
    return Region(
        region_id=region_id,
        start_row=start,
        end_row=end,
        start_time=datetime(2025, 1, 1, 0, 0, 0),
        end_time=datetime(2025, 1, 1, 0, 5, 0),
        duration_seconds=300.0,
        primary_classification=classification,
    )


def _make_cycle(cycle_id: str, cycle_number: int, start: int, end: int) -> Cycle:
    return Cycle(
        cycle_id=cycle_id,
        cycle_number=cycle_number,
        start_row=start,
        end_row=end,
        start_time=datetime(2025, 1, 1, 0, 0, 0),
        end_time=datetime(2025, 1, 1, 0, 20, 0),
        duration_seconds=1200.0,
        region_ids=["R001", "R002"],
        status=CycleStatus.COMPLETE,
        is_complete=True,
    )


def _make_validation_result(
    result_id: str,
    region_id: str | None,
    status: ValidationStatus,
    measured: float = 5.0,
) -> ValidationResult:
    return ValidationResult(
        validation_result_id=result_id,
        requirement_id="REQ_HEATING_RAMP_RATE",
        requirement_description="Heating ramp rate",
        measured_value=measured,
        threshold_value=5.0,
        comparator=Comparator.GTE,
        unit="°C/min",
        method="theil_sen",
        region_id=region_id,
        cycle_id=None,
        included_rows=100,
        result=status,
        reason="Test",
    )


class TestBuildTemperatureTraceChart:
    def test_builds_chart_with_raw_temperature(self):
        trace = ClassifiedTrace(rows=[
            _make_trace_row(0.0, 25.0),
            _make_trace_row(1.0, 26.0),
            _make_trace_row(2.0, 27.0),
        ])
        
        chart = build_temperature_trace_chart(trace)
        
        assert "data" in chart
        assert "layout" in chart
        assert len(chart["data"]) >= 1
        assert chart["data"][0]["name"] == "Actual Temperature"
        assert chart["data"][0]["opacity"] == 1.0

    def test_includes_analysis_signal_if_present(self):
        trace = ClassifiedTrace(rows=[
            _make_trace_row(0.0, 25.0, 25.1),
            _make_trace_row(1.0, 26.0, 26.1),
        ])
        
        chart = build_temperature_trace_chart(trace)
        
        assert len(chart["data"]) == 2
        assert "Analysis Signal" in chart["data"][1]["name"]
        assert "not used for compliance" in chart["data"][1]["name"]
        assert chart["data"][1]["opacity"] == 0.6
        assert chart["data"][1]["line"]["dash"] == "dash"

    def test_raw_has_higher_zorder_than_analysis(self):
        trace = ClassifiedTrace(rows=[
            _make_trace_row(0.0, 25.0, 25.1),
        ])
        
        chart = build_temperature_trace_chart(trace)
        
        raw_zorder = chart["data"][0]["zorder"]
        analysis_zorder = chart["data"][1]["zorder"]
        assert raw_zorder > analysis_zorder


class TestBuildRegionOverlay:
    def test_creates_shapes_for_all_regions(self):
        regions = RegionList(regions=[
            _make_region("R001", RegionType.HEATING_RAMP, 0, 100),
            _make_region("R002", RegionType.HOT_DWELL, 101, 200),
        ])
        
        shapes = build_region_overlay(regions)
        
        assert len(shapes) == 2

    def test_uses_correct_colours_from_plan(self):
        regions = RegionList(regions=[
            _make_region("R001", RegionType.HEATING_RAMP, 0, 100),
            _make_region("R002", RegionType.COLD_DWELL, 101, 200),
        ])
        
        shapes = build_region_overlay(regions)
        
        assert shapes[0]["fillcolor"] == REGION_COLOURS[RegionType.HEATING_RAMP]
        assert shapes[1]["fillcolor"] == REGION_COLOURS[RegionType.COLD_DWELL]

    def test_profile_colour_map_takes_precedence(self):
        regions = RegionList(regions=[
            _make_region("R001", RegionType.HEATING_RAMP, 0, 100),
        ])
        custom_map = {"HEATING_RAMP": "#FF0000"}
        
        shapes = build_region_overlay(regions, region_colour_map=custom_map)
        
        assert shapes[0]["fillcolor"] == "#FF0000"

    def test_fallback_colour_for_unknown(self):
        regions = RegionList(regions=[
            _make_region("R001", RegionType.UNKNOWN, 0, 100),
        ])
        
        shapes = build_region_overlay(regions)
        
        # Should use REGION_COLOURS[UNKNOWN] = #696969
        assert shapes[0]["fillcolor"] == "#696969"


class TestBuildCycleBoundaryMarkers:
    def test_creates_markers_for_cycle_boundaries(self):
        cycles = CycleList(cycles=[
            _make_cycle("C001", 1, 0, 500),
            _make_cycle("C002", 2, 501, 1000),
        ])
        
        markers = build_cycle_boundary_markers(cycles)
        
        # 2 cycles × 2 markers (start + end) = 4 markers
        assert len(markers) == 4

    def test_markers_have_correct_positions(self):
        cycles = CycleList(cycles=[
            _make_cycle("C001", 1, 100, 500),
        ])
        
        markers = build_cycle_boundary_markers(cycles)
        
        assert markers[0]["x"] == [100]  # Start marker
        assert markers[1]["x"] == [500]  # End marker


class TestBuildAnnotations:
    def test_includes_all_gap_g19_fields(self):
        regions = RegionList(regions=[
            _make_region("R001", RegionType.HEATING_RAMP, 0, 100),
        ])
        results = ValidationResults(results=[
            _make_validation_result("VR001", "R001", ValidationStatus.PASS),
        ])
        
        annotations = build_annotations(results, regions)
        
        assert len(annotations) == 1
        ann = annotations[0]
        
        # Gap G19 contract fields
        assert "annotation_id" in ann
        assert "region_id" in ann
        assert "cycle_id" in ann
        assert "validation_result_id" in ann
        assert "source_row_range" in ann  # REQUIRED
        assert "annotation_type" in ann
        assert "text" in ann
        assert "x_position" in ann
        assert "y_position" in ann

    def test_source_row_range_extracted_from_region(self):
        regions = RegionList(regions=[
            _make_region("R001", RegionType.HEATING_RAMP, 50, 150),
        ])
        results = ValidationResults(results=[
            _make_validation_result("VR001", "R001", ValidationStatus.PASS),
        ])
        
        annotations = build_annotations(results, regions)
        
        assert annotations[0]["source_row_range"] == (50, 150)

    def test_source_row_range_defaults_when_no_region(self):
        regions = RegionList(regions=[])
        results = ValidationResults(results=[
            _make_validation_result("VR001", None, ValidationStatus.PASS),
        ])
        
        annotations = build_annotations(results, regions)
        
        assert annotations[0]["source_row_range"] == (0, 0)

    def test_annotation_type_matches_result_status(self):
        regions = RegionList(regions=[
            _make_region("R001", RegionType.HEATING_RAMP, 0, 100),
        ])
        results = ValidationResults(results=[
            _make_validation_result("VR001", "R001", ValidationStatus.PASS),
            _make_validation_result("VR002", "R001", ValidationStatus.FAIL),
            _make_validation_result("VR003", "R001", ValidationStatus.INCONCLUSIVE),
        ])
        
        annotations = build_annotations(results, regions)
        
        assert annotations[0]["annotation_type"] == "PASS"
        assert annotations[1]["annotation_type"] == "FAIL"
        assert annotations[2]["annotation_type"] == "INCONCLUSIVE"

    def test_pass_uses_green_colour(self):
        regions = RegionList(regions=[
            _make_region("R001", RegionType.HEATING_RAMP, 0, 100),
        ])
        results = ValidationResults(results=[
            _make_validation_result("VR001", "R001", ValidationStatus.PASS),
        ])
        
        annotations = build_annotations(results, regions)
        
        assert annotations[0]["arrowcolor"] == "#228B22"
        assert "✓" in annotations[0]["text"]

    def test_fail_uses_red_colour(self):
        regions = RegionList(regions=[
            _make_region("R001", RegionType.HEATING_RAMP, 0, 100),
        ])
        results = ValidationResults(results=[
            _make_validation_result("VR001", "R001", ValidationStatus.FAIL),
        ])
        
        annotations = build_annotations(results, regions)
        
        assert annotations[0]["arrowcolor"] == "#DC143C"
        assert "✗" in annotations[0]["text"]


class TestBuildCompleteVisualisation:
    def test_assembles_all_components(self):
        trace = ClassifiedTrace(rows=[_make_trace_row(0.0, 25.0)])
        regions = RegionList(regions=[_make_region("R001", RegionType.HEATING_RAMP, 0, 100)])
        cycles = CycleList(cycles=[_make_cycle("C001", 1, 0, 500)])
        results = ValidationResults(results=[_make_validation_result("VR001", "R001", ValidationStatus.PASS)])
        
        viz = build_complete_visualisation(trace, regions, cycles, results)
        
        assert "data" in viz
        assert "layout" in viz
        assert "shapes" in viz["layout"]
        assert "annotations" in viz["layout"]
        assert len(viz["layout"]["shapes"]) > 0
        assert len(viz["layout"]["annotations"]) > 0
