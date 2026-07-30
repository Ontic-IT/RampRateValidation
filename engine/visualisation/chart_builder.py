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
        # Drag draws a box that zooms into that region (box-zoom); the toolbar
        # Pan button switches to panning, and the scroll wheel also zooms.
        "dragmode": "zoom",
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

        # Anchor the annotation AT the event: x at the offending sample, y at
        # the actual trace temperature there — never at the measured metric
        # value (which is a deviation in degC, not a chart coordinate).
        anchor_row = None
        region_colour = "#808080"  # Default gray

        if result.region_id and result.region_id in region_map:
            region = region_map[result.region_id]
            start = max(0, min(region.start_row, len(trace_rows) - 1))
            end = max(0, min(region.end_row, len(trace_rows) - 1))

            # For setpoint-deviation failures, point at the WORST sample in
            # the region; otherwise anchor at the region midpoint.
            anchor_row = (start + end) // 2
            if "SETPOINT_DEVIATION" in result.requirement_id and end > start:
                worst_dev = -1.0
                for i in range(start, end + 1):
                    row = trace_rows[i]
                    if row.setpoint_c is not None:
                        dev = abs(row.temperature_c_raw - row.setpoint_c)
                        if dev > worst_dev:
                            worst_dev = dev
                            anchor_row = i

            # Match annotation color to region color for visual clarity
            if region_colour_map:
                region_type_str = region.primary_classification.value if hasattr(region.primary_classification, 'value') else str(region.primary_classification)
                region_colour = region_colour_map.get(region_type_str, REGION_COLOURS.get(region.primary_classification, "#808080"))
            else:
                region_colour = REGION_COLOURS.get(region.primary_classification, "#808080")

        if anchor_row is None:
            anchor_row = min(max(result.included_rows, 0), len(trace_rows) - 1)

        anchor = trace_rows[anchor_row]
        x_position = anchor.elapsed_seconds
        y_position = anchor.temperature_c_raw

        # Stagger arrow offsets so clustered annotations remain readable.
        stagger = (len(annotations) % 3) * 22

        annotations.append({
            "text": (
                f"{symbol} {result.requirement_id}<br>"
                f"{result.measured_value:.2f} vs allowed {result.threshold_value:.2f} {result.unit}"
            ),
            "x": x_position,
            "y": y_position,
            "xref": "x",
            "yref": "y",
            "showarrow": True,
            "arrowhead": 2,
            "arrowcolor": region_colour,
            "ax": 0,
            "ay": -(40 + stagger),
            "font": {"color": region_colour, "size": 10},
            "bgcolor": "rgba(255,255,255,0.85)",
            "bordercolor": region_colour,
            "borderwidth": 1,
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


REGION_LABELS = {
    RegionType.HEATING_RAMP: "HEATING RAMP",
    RegionType.COOLING_RAMP: "COOLING RAMP",
    RegionType.HOT_DWELL: "HOT DWELL",
    RegionType.COLD_DWELL: "COLD DWELL",
    RegionType.AMBIENT_START: "AMBIENT",
    RegionType.HOT_OVERSHOOT: "HOT OVERSHOOT",
    RegionType.COLD_OVERSHOOT: "COLD OVERSHOOT",
    RegionType.HOT_CORRECTION: "HOT CORRECTION",
    RegionType.COLD_CORRECTION: "COLD CORRECTION",
    RegionType.RECOVERY: "RECOVERY",
    RegionType.TRANSIENT: "TRANSIENT",
    RegionType.UNKNOWN: "UNKNOWN",
}

# A label only fits a region wide enough to read it against; narrower
# regions stay identified by band colour + the chart legend + hover.
# (Narrow bands render the label rotated — see build_region_label_overlay.)
MIN_LABEL_WIDTH_FRACTION = 0.01
MIN_PHASE_NUMBER_WIDTH_FRACTION = 0.008

# Region classifications that are events of concern — always called out with
# an arrow annotation pointing at the region, regardless of band width.
EVENT_REGION_TYPES = {
    RegionType.HOT_OVERSHOOT,
    RegionType.COLD_OVERSHOOT,
    RegionType.HOT_CORRECTION,
    RegionType.COLD_CORRECTION,
}

# Call-out gates: an event is only worth an arrow if it is real, not noise.
OVERSHOOT_CALLOUT_MIN_C = 1.0        # excursion beyond setpoint worth flagging
OSCILLATION_MIN_CROSSINGS = 6        # sustained ringing, not a couple of wiggles
OSCILLATION_MIN_RANGE_C = 3.0        # ringing must have real amplitude


def build_phase_number_overlay(
    regions: RegionList,
    trace_rows: list[Any],
    audit_log: AuditLog | None = None,
) -> list[dict[str, Any]]:
    """Build phase number overlay annotations for the chart.

    Places phase numbers (matching the phase table) above each region wide
    enough to carry a label without overlap. X positions are in elapsed
    seconds, matching the chart axis.
    """
    if audit_log is None:
        audit_log = AuditLog()

    row_to_elapsed = {i: row.elapsed_seconds for i, row in enumerate(trace_rows)}
    span = trace_rows[-1].elapsed_seconds - trace_rows[0].elapsed_seconds if trace_rows else 1.0
    min_width = span * MIN_PHASE_NUMBER_WIDTH_FRACTION

    phase_annotations = []
    labelled = 0
    for idx, region in enumerate(regions.regions, start=1):
        x0 = row_to_elapsed.get(region.start_row, region.start_row)
        x1 = row_to_elapsed.get(region.end_row, region.end_row)
        if (x1 - x0) < min_width:
            continue  # numbered in the phase table; too narrow to label here
        labelled += 1

        phase_annotations.append({
            "x": (x0 + x1) / 2.0,
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
        reason=f"Labelled {labelled} of {len(regions.regions)} regions with phase numbers (width filter)",
        severity=AuditSeverity.INFO,
        category=AuditCategory.METRICS,
    ))

    return phase_annotations


def build_region_label_overlay(
    regions: RegionList,
    trace_rows: list[Any],
    audit_log: AuditLog | None = None,
) -> list[dict[str, Any]]:
    """Classification labels drawn INSIDE each region band.

    This is the point of the tool: the chart must say what each region IS.
    Regions wide enough get their classification written across the band;
    narrower regions are covered by the colour legend.
    """
    if audit_log is None:
        audit_log = AuditLog()

    row_to_elapsed = {i: row.elapsed_seconds for i, row in enumerate(trace_rows)}
    span = trace_rows[-1].elapsed_seconds - trace_rows[0].elapsed_seconds if trace_rows else 1.0
    min_width = span * MIN_LABEL_WIDTH_FRACTION

    labels = []
    for region in regions.regions:
        x0 = row_to_elapsed.get(region.start_row, region.start_row)
        x1 = row_to_elapsed.get(region.end_row, region.end_row)
        if (x1 - x0) < min_width:
            continue

        classification = region.primary_classification
        text = REGION_LABELS.get(classification, str(classification))
        colour = REGION_COLOURS.get(classification, "#555555")

        labels.append({
            "x": (x0 + x1) / 2.0,
            "y": 0.03,
            "xref": "x",
            "yref": "paper",
            "text": f"<b>{text}</b>",
            "showarrow": False,
            "textangle": -90 if (x1 - x0) < span * 0.05 else 0,
            "font": {"size": 9, "color": colour},
            "bgcolor": "rgba(255,255,255,0.6)",
        })

    audit_log.add(AuditEntry(
        timestamp=datetime.now(),
        module_name="chart_builder",
        action="build_region_label_overlay",
        decision="SUCCESS",
        reason=f"Built {len(labels)} region classification labels",
        severity=AuditSeverity.INFO,
        category=AuditCategory.METRICS,
    ))

    return labels


def build_region_legend_traces(
    regions: RegionList,
    region_colour_map: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Legend entries mapping band colours to classification names.

    Plotly shapes carry no legend, so each classification present in the
    trace gets an invisible marker trace whose legend swatch shows the band
    colour.
    """
    seen: dict[str, str] = {}
    for region in regions.regions:
        classification = region.primary_classification
        name = REGION_LABELS.get(classification, str(classification))
        if name in seen:
            continue
        colour = None
        if region_colour_map:
            key = classification.value if hasattr(classification, "value") else str(classification)
            colour = region_colour_map.get(key)
        seen[name] = colour or REGION_COLOURS.get(classification, "#CCCCCC")

    return [
        {
            "x": [None],
            "y": [None],
            "mode": "markers",
            "marker": {"size": 10, "color": colour, "symbol": "square", "opacity": 0.5},
            "name": f"Region: {name}",
            "showlegend": True,
            "hoverinfo": "skip",
        }
        for name, colour in seen.items()
    ]


def build_event_callouts(
    regions: RegionList,
    trace_rows: list[Any],
    dwell_metrics: list[Any] | None = None,
    setpoint_tolerance_c: float | None = None,
    audit_log: AuditLog | None = None,
) -> list[dict[str, Any]]:
    """Arrow call-outs for events of concern — distinct from region colours.

    Region bands/colours show WHAT every region is; these call-outs draw an
    arrow to the specific events a reviewer must notice: overshoots,
    corrections, and oscillatory dwells. They appear only when such an event
    is present (a clean run has none), pointing at the region in question.

    An overshoot is only flagged when it exceeds the SETPOINT TOLERANCE — an
    excursion the chamber holds within tolerance is not worth a call-out.
    """
    if audit_log is None:
        audit_log = AuditLog()

    overshoot_threshold = (
        setpoint_tolerance_c if setpoint_tolerance_c and setpoint_tolerance_c > 0
        else OVERSHOOT_CALLOUT_MIN_C
    )

    metrics_by_region = {m.region_id: m for m in (dwell_metrics or [])}
    callouts: list[dict[str, Any]] = []

    def _anchor(region):
        start = max(0, min(region.start_row, len(trace_rows) - 1))
        end = max(0, min(region.end_row, len(trace_rows) - 1))
        return start, end

    for region in regions.regions:
        classification = region.primary_classification
        start, end = _anchor(region)
        seg = trace_rows[start:end + 1]
        if not seg:
            continue
        temps = [r.temperature_c_raw for r in seg]
        setpoints_seg = [r.setpoint_c for r in seg if r.setpoint_c is not None]
        seg_setpoint = float(np.median(setpoints_seg)) if setpoints_seg else None
        dm = metrics_by_region.get(region.region_id)

        label = None
        colour = "#B00020"
        # Overshoot / correction regions: point at the peak/trough.
        if classification in (RegionType.HOT_OVERSHOOT, RegionType.COLD_OVERSHOOT):
            is_hot = classification == RegionType.HOT_OVERSHOOT
            peak_i = int(np.argmax(temps)) if is_hot else int(np.argmin(temps))
            anchor_row = start + peak_i
            # Excursion beyond the commanded setpoint at the peak — overshoot
            # regions are not dwells, so derive it from the row setpoints.
            if dm and dm.overshoot_magnitude_c:
                mag = dm.overshoot_magnitude_c
            elif seg_setpoint is not None:
                mag = abs(temps[peak_i] - seg_setpoint)
            else:
                mag = abs(temps[peak_i] - float(np.median(temps)))
            if mag <= overshoot_threshold:
                continue  # within setpoint tolerance — not worth a call-out
            label = f"⚠ {'HOT' if is_hot else 'COLD'} OVERSHOOT +{mag:.1f}°C"
            colour = REGION_COLOURS.get(classification, "#DC143C")
        elif classification in (RegionType.HOT_CORRECTION, RegionType.COLD_CORRECTION):
            anchor_row = (start + end) // 2
            label = f"CORRECTION ({'hot' if classification == RegionType.HOT_CORRECTION else 'cold'})"
            colour = REGION_COLOURS.get(classification, "#FF6347")
        # Events at a CONTROLLED dwell (hot/cold) only — the ambient soak is
        # not actively held, so its noise crossings are not oscillations of
        # concern.
        elif classification in (RegionType.HOT_DWELL, RegionType.COLD_DWELL) and dm:
            if dm.overshoot_magnitude_c and dm.overshoot_magnitude_c > overshoot_threshold:
                is_hot = classification == RegionType.HOT_DWELL
                peak_i = int(np.argmax(temps)) if is_hot else int(np.argmin(temps))
                anchor_row = start + peak_i
                rs = getattr(dm, "overshoot_recovery_seconds", None) or 0
                recov = (f", recovered in {rs/60:.0f} min" if rs >= 60 else f", recovered in {rs:.0f}s") if rs else ""
                label = f"⚠ OVERSHOOT +{dm.overshoot_magnitude_c:.1f}°C{recov}"
                colour = "#DC143C"
            elif (
                dm.oscillation_count and dm.oscillation_count >= OSCILLATION_MIN_CROSSINGS
                and (dm.temperature_range_c or 0.0) >= OSCILLATION_MIN_RANGE_C
            ):
                anchor_row = (start + end) // 2
                label = f"OSCILLATION ({dm.oscillation_count} crossings, ±{(dm.temperature_range_c or 0)/2:.1f}°C)"
                colour = "#9370DB"
            else:
                continue
        else:
            continue

        anchor_row = max(0, min(anchor_row, len(trace_rows) - 1))
        anchor = trace_rows[anchor_row]
        callouts.append({
            "x": anchor.elapsed_seconds,
            "y": anchor.temperature_c_raw,
            "xref": "x",
            "yref": "y",
            "text": f"<b>{label}</b>",
            "showarrow": True,
            "arrowhead": 2,
            "arrowsize": 1.2,
            "arrowwidth": 2,
            "arrowcolor": colour,
            "ax": 0,
            "ay": -45,
            "font": {"color": colour, "size": 11},
            "bgcolor": "rgba(255,255,255,0.9)",
            "bordercolor": colour,
            "borderwidth": 1,
            "borderpad": 3,
        })

    audit_log.add(AuditEntry(
        timestamp=datetime.now(),
        module_name="chart_builder",
        action="build_event_callouts",
        decision="SUCCESS",
        reason=f"Built {len(callouts)} event call-out(s) (overshoot/correction/oscillation)",
        severity=AuditSeverity.INFO,
        category=AuditCategory.METRICS,
    ))
    return callouts


def build_complete_visualisation(
    classified_trace: ClassifiedTrace,
    regions: RegionList,
    cycles: CycleList,
    validation_results: ValidationResults,
    setpoints: Any = None,
    region_colour_map: dict[str, str] | None = None,
    dwell_metrics: list[Any] | None = None,
    setpoint_tolerance_c: float | None = None,
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

    # Setpoint-relative overlays (setpoint-deviation call-outs, overshoot
    # markers) only make sense when there IS a setpoint to deviate from. With
    # no setpoint channel (Mode B) the target is undefined until the reader
    # picks one in the report, so these are suppressed server-side and instead
    # drawn client-side against the chosen target.
    from config.constants import SetpointResolutionMode
    setpoint_available = bool(
        setpoints is not None
        and getattr(setpoints, "resolution_mode", None) == SetpointResolutionMode.MODE_A
    )

    shapes = build_region_overlay(regions, trace_rows, region_colour_map, anomaly_ids=failed_region_ids, audit_log=audit_log)
    cycle_shapes, cycle_annotations = build_cycle_boundary_markers(cycles, trace_rows, audit_log)
    annotations = []
    if setpoint_available:
        annotations = build_annotations(validation_results, regions, trace_rows, region_colour_map, audit_log)

    # Add phase number overlays and region classification labels
    annotations.extend(build_phase_number_overlay(regions, trace_rows, audit_log))
    annotations.extend(build_region_label_overlay(regions, trace_rows, audit_log))

    # Add arrow call-outs for events of concern (overshoot/correction/oscillation)
    if setpoint_available:
        annotations.extend(build_event_callouts(regions, trace_rows, dwell_metrics, setpoint_tolerance_c, audit_log))

    # Add cycle labels
    annotations.extend(cycle_annotations)

    # Legend entries for the region band colours (shapes carry no legend)
    chart["data"] = chart["data"] + build_region_legend_traces(regions, region_colour_map)

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
