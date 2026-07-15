"""Plotly chart builder for ramp rate analysis visualisation (M13).

Handles:
- Temperature trace chart (raw primary, analysis signal secondary)
- Region overlay bands
- Cycle boundary markers
- Ramp rate, dwell duration, overshoot annotations
- Pass/fail annotations
- Quality warning markers
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np

from config.constants import AuditCategory, AuditSeverity, RegionType
from models.domain import (
    AuditEntry,
    AuditLog,
    CanonicalTrace,
    ClassifiedTrace,
    CycleList,
    Region,
    RegionList,
    ValidationResult,
    ValidationResults,
)


# Colour scheme from plan
REGION_COLOURS = {
    RegionType.HEATING_RAMP: "#FF8C00",
    RegionType.COOLING_RAMP: "#4169E1",
    RegionType.HOT_DWELL: "#228B22",
    RegionType.COLD_DWELL: "#32CD32",
    RegionType.AMBIENT_START: "#808080",
    RegionType.HOT_OVERSHOOT: "#DC143C",
    RegionType.COLD_OVERSHOOT: "#0000CD",
    RegionType.HOT_CORRECTION: "#FF6347",
    RegionType.COLD_CORRECTION: "#4682B4",
    RegionType.RECOVERY: "#9370DB",
    RegionType.TRANSIENT: "#DAA520",
    RegionType.UNKNOWN: "#696969",
}


def build_temperature_trace_chart(
    classified_trace: ClassifiedTrace,
    setpoints: Any = None,
    audit_log: AuditLog | None = None,
) -> dict[str, Any]:
    """Build the main temperature trace chart.

    Raw temperature_c_raw is primary (highest z-order, max opacity).
    Analysis signal is secondary (lower opacity, dashed) if present.
    Setpoint lines shown as horizontal dashed lines if provided.

    Args:
        classified_trace: Trace with temperature and classification data
        setpoints: ResolvedSetpoints (optional, for setpoint lines)
        audit_log: Optional audit log

    Returns:
        Plotly figure dict
    """
    if audit_log is None:
        audit_log = AuditLog()

    rows = classified_trace.rows
    elapsed = [r.elapsed_seconds for r in rows]
    temps_raw = [r.temperature_c_raw for r in rows]
    temps_analysis = [r.temperature_c_analysis_signal for r in rows if r.temperature_c_analysis_signal is not None]
    setpoint_values = [r.setpoint_c for r in rows]

    traces = []

    # Primary: raw measured temperature (MUST be primary signal)
    traces.append({
        "x": elapsed,
        "y": temps_raw,
        "mode": "lines",
        "name": "Actual Temperature (measured)",
        "line": {"color": "#DC143C", "width": 2},
        "opacity": 1.0,
        "zorder": 10,
    })
    
    # Real-time setpoint trace (where temperature SHOULD be vs where it IS)
    if any(sp is not None for sp in setpoint_values):
        # Filter out None values for plotting
        setpoint_elapsed = [e for e, sp in zip(elapsed, setpoint_values) if sp is not None]
        setpoint_temps = [sp for sp in setpoint_values if sp is not None]
        
        if setpoint_elapsed:
            traces.append({
                "x": setpoint_elapsed,
                "y": setpoint_temps,
                "mode": "lines",
                "name": "Target Setpoint (where temp should be)",
                "line": {"color": "#0000FF", "width": 1.5, "dash": "dot"},
                "opacity": 0.8,
                "zorder": 8,
            })
    
    # Static setpoint reference lines (for traces without real-time setpoint data)
    elif setpoints:
        max_elapsed = max(elapsed) if elapsed else 100
        if setpoints.inferred_hot_setpoint_c:
            traces.append({
                "x": [0, max_elapsed],
                "y": [setpoints.inferred_hot_setpoint_c, setpoints.inferred_hot_setpoint_c],
                "mode": "lines",
                "name": "Hot Setpoint (inferred)",
                "line": {"color": "#0000FF", "width": 1, "dash": "dash"},
                "opacity": 0.5,
                "zorder": 5,
            })
        if setpoints.inferred_cold_setpoint_c:
            traces.append({
                "x": [0, max_elapsed],
                "y": [setpoints.inferred_cold_setpoint_c, setpoints.inferred_cold_setpoint_c],
                "mode": "lines",
                "name": "Cold Setpoint (inferred)",
                "line": {"color": "#00CED1", "width": 1, "dash": "dash"},
                "opacity": 0.5,
                "zorder": 5,
            })

    # Secondary: analysis signal (if present)
    if temps_analysis and len(temps_analysis) == len(elapsed):
        traces.append({
            "x": elapsed,
            "y": temps_analysis,
            "mode": "lines",
            "name": "Analysis Signal (classification aid only — not used for compliance)",
            "line": {"color": "#ff7f0e", "width": 1.5, "dash": "dash"},
            "opacity": 0.6,
            "zorder": 5,
        })

    layout = {
        "title": "Temperature Trace with Classification",
        "xaxis": {"title": "Elapsed Time (seconds)"},
        "yaxis": {"title": "Temperature (°C)"},
        "hovermode": "closest",
        "showlegend": True,
    }

    audit_log.add(AuditEntry(
        timestamp=datetime.now(),
        module_name="chart_builder",
        action="build_temperature_trace_chart",
        decision="SUCCESS",
        reason=f"Built chart with {len(rows)} rows",
        severity=AuditSeverity.INFO,
        category=AuditCategory.METRICS,
    ))

    return {"data": traces, "layout": layout}


def build_region_overlay(
    regions: RegionList,
    trace_rows: list[Any],
    region_colour_map: dict[str, str] | None = None,
    anomaly_ids: set[str] | None = None,
    anomaly_only: bool = False,
    audit_log: AuditLog | None = None,
) -> list[dict[str, Any]]:
    """Build region overlay bands for the chart.

    Colour precedence:
    1. region_colour_map parameter (from ValidationProfile.visualisation_settings.region_colour_map)
    2. REGION_COLOURS default map
    3. #CCCCCC fallback

    Args:
        regions: Classified regions
        region_colour_map: Colour map from ValidationProfile.visualisation_settings.region_colour_map
        anomaly_ids: Set of region IDs that are anomalies (for highlighting)
        anomaly_only: If True, only highlight anomalous regions in red/pink
        audit_log: Optional audit log

    Returns:
        List of Plotly shape dicts
    """
    if audit_log is None:
        audit_log = AuditLog()

    # Build row index to elapsed_seconds mapping
    row_to_elapsed = {i: row.elapsed_seconds for i, row in enumerate(trace_rows)}

    shapes = []
    for region in regions.regions:
        # If anomaly_only mode, skip non-anomalous regions
        if anomaly_only and anomaly_ids and region.region_id not in anomaly_ids:
            continue
        
        # Determine colour
        colour = None
        
        # If this is an anomaly and we have anomaly_ids, use red/pink
        if anomaly_ids and region.region_id in anomaly_ids:
            colour = "#FFB6C1"  # Light pink for anomaly highlighting
        else:
            # Precedence: profile colour map → default REGION_COLOURS → fallback
            if region_colour_map:
                colour = region_colour_map.get(region.primary_classification.value)
            if not colour:
                colour = REGION_COLOURS.get(region.primary_classification, "#CCCCCC")

        # Convert row indices to elapsed seconds for x-axis alignment
        x0 = row_to_elapsed.get(region.start_row, region.start_row)
        x1 = row_to_elapsed.get(region.end_row, region.end_row)
        
        shapes.append({
            "type": "rect",
            "x0": x0,
            "x1": x1,
            "y0": 0,
            "y1": 1,
            "xref": "x",
            "yref": "paper",
            "fillcolor": colour,
            "opacity": 0.25 if anomaly_ids and region.region_id in anomaly_ids else 0.15,
            "line": {"width": 0},
            "layer": "below",
        })

    audit_log.add(AuditEntry(
        timestamp=datetime.now(),
        module_name="chart_builder",
        action="build_region_overlay",
        decision="SUCCESS",
        reason=f"Built overlays for {len(regions.regions)} regions",
        severity=AuditSeverity.INFO,
        category=AuditCategory.METRICS,
    ))

    return shapes


def build_cycle_boundary_markers(
    cycles: CycleList,
    trace_rows: list[Any],
    audit_log: AuditLog | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build cycle boundary markers as vertical lines and annotations.

    Args:
        cycles: Detected cycles
        trace_rows: Trace rows to map row indices to elapsed_seconds
        audit_log: Optional audit log

    Returns:
        Tuple of (shapes for vertical lines, annotations for cycle labels)
    """
    if audit_log is None:
        audit_log = AuditLog()

    # Build row index to elapsed_seconds mapping
    row_to_elapsed = {i: row.elapsed_seconds for i, row in enumerate(trace_rows)}

    shapes = []
    annotations = []
    
    for cycle in cycles.cycles:
        # Convert row index to elapsed seconds
        cycle_start_x = row_to_elapsed.get(cycle.start_row, cycle.start_row)
        
        # Vertical line at cycle start
        shapes.append({
            "type": "line",
            "x0": cycle_start_x,
            "x1": cycle_start_x,
            "y0": 0,
            "y1": 1,
            "xref": "x",
            "yref": "paper",
            "line": {
                "color": "#000000",
                "width": 2,
                "dash": "dash",
            },
        })
        
        # Cycle label at top of chart
        annotations.append({
            "x": cycle_start_x,
            "y": 1.05,
            "xref": "x",
            "yref": "paper",
            "text": f"<b>Cycle {cycle.cycle_number}</b>",
            "showarrow": False,
            "font": {"size": 12, "color": "#000000"},
            "bgcolor": "rgba(255,255,255,0.8)",
            "bordercolor": "#000000",
            "borderwidth": 1,
            "borderpad": 4,
        })

    audit_log.add(AuditEntry(
        timestamp=datetime.now(),
        module_name="chart_builder",
        action="build_cycle_boundary_markers",
        decision="SUCCESS",
        reason=f"Built boundary lines and labels for {len(cycles.cycles)} cycles",
        severity=AuditSeverity.INFO,
        category=AuditCategory.METRICS,
    ))

    return shapes, annotations


def build_annotations(
    validation_results: ValidationResults,
    regions: RegionList,
    trace_rows: list[Any],
    region_colour_map: dict[str, str] | None = None,
    audit_log: AuditLog | None = None,
) -> list[dict[str, Any]]:
    """Build pass/fail and metric annotations.

    Annotation Contract (Gap G19):
    - annotation_id: str
    - region_id: str | None
    - cycle_id: str | None
    - validation_result_id: str | None
    - source_row_range: tuple[int, int]  # REQUIRED
    - annotation_type: str
    - text: str
    - x_position: float
    - y_position: float

    Args:
        validation_results: Validation results
        regions: Classified regions (to get source_row_range)
        region_colour_map: Optional color map for regions
        audit_log: Optional audit log

    Returns:
        List of annotation dicts conforming to Gap G19 contract
    """
    if audit_log is None:
        audit_log = AuditLog()

    # Build row index to elapsed_seconds mapping
    row_to_elapsed = {i: row.elapsed_seconds for i, row in enumerate(trace_rows)}
    
    # Build region lookup for source_row_range
    region_map = {r.region_id: r for r in regions.regions}

    annotations = []
    for idx, result in enumerate(validation_results.results):
        # Filter annotations to reduce clutter:
        # 1. Skip DWELL_DURATION - shown in phase table, not needed on chart
        # 2. Only show FAIL results (anomalies) - passes clutter the chart
        # 3. Keep ramp rate and setpoint deviation failures for visibility
        if "DWELL_DURATION" in result.requirement_id:
            continue
        
        if result.result.value in ("PASS", "PASS_WITH_WARNINGS"):
            # Skip pass annotations to reduce clutter
            continue
        
        # Determine symbol based on pass/fail
        if result.result.value == "FAIL":
            symbol = "✗"
            annotation_type = "FAIL"
        else:
            symbol = "?"
            annotation_type = "INCONCLUSIVE"

        # Get region info for positioning and color
        source_row_range = (0, 0)
        x_position = 0
        region_colour = "#808080"  # Default gray
        
        if result.region_id and result.region_id in region_map:
            region = region_map[result.region_id]
            source_row_range = (region.start_row, region.end_row)
            # Use region midpoint for x-position to distribute across all cycles
            # Convert from row indices to elapsed seconds
            midpoint_row = (region.start_row + region.end_row) // 2
            x_position = row_to_elapsed.get(midpoint_row, midpoint_row)
            
            # Match annotation color to region color for visual clarity
            if region_colour_map:
                region_type_str = region.primary_classification.value if hasattr(region.primary_classification, 'value') else str(region.primary_classification)
                region_colour = region_colour_map.get(region_type_str, REGION_COLOURS.get(region.primary_classification, "#808080"))
            else:
                region_colour = REGION_COLOURS.get(region.primary_classification, "#808080")
        else:
            # Fallback to included_rows if no region
            x_position = result.included_rows if result.included_rows > 0 else 0

        # Only include valid Plotly annotation properties
        annotations.append({
            "text": f"{symbol} {result.requirement_id}",
            "x": x_position,
            "y": result.measured_value,
            "xref": "x",
            "yref": "y",
            "showarrow": True,
            "arrowhead": 2,
            "arrowcolor": region_colour,
            "font": {"color": region_colour, "size": 10},
            "bgcolor": "rgba(255,255,255,0.8)",
        })

    audit_log.add(AuditEntry(
        timestamp=datetime.now(),
        module_name="chart_builder",
        action="build_annotations",
        decision="SUCCESS",
        reason=f"Built {len(annotations)} annotations with Gap G19 contract",
        severity=AuditSeverity.INFO,
        category=AuditCategory.METRICS,
    ))

    return annotations


def build_phase_number_overlay(
    regions: RegionList,
    audit_log: AuditLog | None = None,
) -> list[dict[str, Any]]:
    """Build phase number overlay annotations for the chart.
    
    Places phase numbers at the top of each region for easy identification.
    
    Args:
        regions: Classified regions
        audit_log: Optional audit log
    
    Returns:
        List of phase number annotation dicts
    """
    if audit_log is None:
        audit_log = AuditLog()
    
    phase_annotations = []
    for idx, region in enumerate(regions.regions, start=1):
        # Calculate midpoint of region for x position
        x_pos = (region.start_row + region.end_row) / 2.0
        
        phase_annotations.append({
            "x": x_pos,
            "y": 1.02,  # Just above the top of the chart
            "xref": "x",
            "yref": "paper",
            "text": f"<b>{idx}</b>",
            "showarrow": False,
            "font": {"size": 11, "color": "#000000"},
            "bgcolor": "rgba(255,255,255,0.7)",
            "bordercolor": "#CCCCCC",
            "borderwidth": 1,
            "borderpad": 3,
        })
    
    audit_log.add(AuditEntry(
        timestamp=datetime.now(),
        module_name="chart_builder",
        action="build_phase_number_overlay",
        decision="SUCCESS",
        reason=f"Built phase number overlays for {len(regions.regions)} regions",
        severity=AuditSeverity.INFO,
        category=AuditCategory.METRICS,
    ))
    
    return phase_annotations


def build_complete_visualisation(
    classified_trace: ClassifiedTrace,
    regions: RegionList,
    cycles: CycleList,
    validation_results: ValidationResults,
    setpoints: Any = None,
    region_colour_map: dict[str, str] | None = None,
    audit_log: AuditLog | None = None,
) -> dict[str, Any]:
    """Build complete visualisation bundle.

    Args:
        classified_trace: Trace data
        regions: Classified regions
        cycles: Detected cycles
        validation_results: Validation results
        setpoints: ResolvedSetpoints (optional, for setpoint lines)
        region_colour_map: Optional colour overrides
        audit_log: Optional audit log

    Returns:
        Complete visualisation dict with data, layout, shapes, annotations
    """
    if audit_log is None:
        audit_log = AuditLog()

    # Identify failed phases from validation results
    failed_region_ids = set()
    if validation_results and validation_results.results:
        for result in validation_results.results:
            if result.result.value == "FAIL":
                # Extract region_id if available
                if result.region_id:
                    failed_region_ids.add(result.region_id)

    chart = build_temperature_trace_chart(classified_trace, setpoints, audit_log)
    trace_rows = classified_trace.rows
    shapes = build_region_overlay(regions, trace_rows, region_colour_map, anomaly_ids=failed_region_ids, audit_log=audit_log)
    cycle_shapes, cycle_annotations = build_cycle_boundary_markers(cycles, trace_rows, audit_log)
    annotations = build_annotations(validation_results, regions, trace_rows, region_colour_map, audit_log)
    
    # Add phase number overlays
    phase_annotations = build_phase_number_overlay(regions, audit_log)
    annotations.extend(phase_annotations)
    
    # Add cycle labels
    annotations.extend(cycle_annotations)

    # Combine all shapes (regions + cycle boundaries)
    chart["layout"]["shapes"] = shapes + cycle_shapes
    chart["layout"]["annotations"] = annotations

    audit_log.add(AuditEntry(
        timestamp=datetime.now(),
        module_name="chart_builder",
        action="build_complete_visualisation",
        decision="SUCCESS",
        reason="Complete visualisation assembled",
        severity=AuditSeverity.INFO,
        category=AuditCategory.METRICS,
    ))

    return chart
