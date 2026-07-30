"""Report payload builder (M14).

Constructs all 13 mandatory report sections.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from config.constants import AuditCategory, AuditSeverity, RegionType
from models.domain import (
    AnalysisContext,
    AuditEntry,
    AuditLog,
    OverallStatus,
    ReportPackage,
    ValidationResults,
)


def generate_report_payload(
    context: AnalysisContext,
    audit_log: AuditLog | None = None,
) -> ReportPackage:
    """Generate the complete report package with all 13 sections.

    Args:
        context: Analysis context with all pipeline outputs
        audit_log: Optional audit log

    Returns:
        ReportPackage with all 13 sections populated
    """
    if audit_log is None:
        audit_log = AuditLog()

    sections: dict[str, Any] = {}

    # Section order = render order. Headline results lead (test summary with
    # conformance, then ramp-rate validation directly under it); supporting
    # detail follows. Input-file / data-quality / boundary / setpoint-inference
    # sections are intentionally omitted — their figures already appear in the
    # test-parameters block — and the audit trail is not part of the final
    # report.
    sections["executive_summary"] = build_executive_summary(context)
    sections["ramp_rate_validation_summary"] = _build_ramp_rate_validation_summary(context)
    sections["interactive_validation_data"] = _build_interactive_validation_data(context)
    sections["region_classification_summary"] = _build_region_classification_summary(context)
    # Overshoot-recovery summary: how long dwell overshoots take to return to
    # setpoint (calibration-band tables and the overshoot distribution were
    # removed — the recovery-time trend is what matters).
    sections["dwell_calibration_summary"] = _build_dwell_calibration_summary(context)
    sections["visualisation"] = _build_visualisation_summary(context)

    audit_log.add(AuditEntry(
        timestamp=datetime.now(),
        module_name="report_builder",
        action="generate_report_payload",
        decision="SUCCESS",
        reason=f"{len(sections)} report sections generated",
        severity=AuditSeverity.INFO,
        category=AuditCategory.PIPELINE,
    ))

    return ReportPackage(sections=sections)


def _trace_cycle_spans(rows) -> list[list[float]]:
    """Cycle [start, end] spans derived purely from the temperature trace.

    Used when the data has no setpoint channel, so the dwell-anchored cycle
    detector cannot be trusted. A cycle is one full thermal excursion between
    the hot and cold extremes. Hysteresis bands (25% in from each extreme)
    debounce control-loop wiggle; a cycle boundary falls each time the trace
    returns to the extreme it started from having visited the other.
    """
    temps = [r.temperature_c_raw for r in rows]
    if len(temps) < 4:
        return []
    lo, hi = min(temps), max(temps)
    span = hi - lo
    if span < 5.0:  # essentially isothermal — not a cycling run
        return []
    hot_band = hi - 0.25 * span   # "at hot" once above this
    cold_band = lo + 0.25 * span  # "at cold" once below this

    # Sequence of extreme visits: list of (row_index, 'hot'|'cold').
    visits: list[tuple[int, str]] = []
    state = None
    for i, t in enumerate(temps):
        here = "hot" if t >= hot_band else ("cold" if t <= cold_band else None)
        if here and here != state:
            visits.append((i, here))
            state = here
    if len(visits) < 2:
        return []

    # One cycle spans from one visit of a given kind to the next visit of the
    # SAME kind (a full there-and-back). Anchor on whichever extreme the trace
    # reaches first.
    first_kind = visits[0][1]
    anchor_rows = [i for i, k in visits if k == first_kind]
    if len(anchor_rows) < 2:
        # Only one full excursion present → single cycle across the trace.
        return [[round(rows[0].elapsed_seconds, 1), round(rows[-1].elapsed_seconds, 1)]]

    spans: list[list[float]] = []
    for a, b in zip(anchor_rows, anchor_rows[1:]):
        spans.append([round(rows[a].elapsed_seconds, 1), round(rows[b].elapsed_seconds, 1)])
    # Extend the last span to the end of the trace (final return-to-ambient tail).
    if spans:
        spans[-1][1] = round(rows[-1].elapsed_seconds, 1)
        # And let the first span reach back to the trace start (lead-in soak).
        spans[0][0] = round(rows[0].elapsed_seconds, 1)
    return spans


def _inferred_setpoint_trajectory(rows, context) -> list[list[float]]:
    """A stepped setpoint reconstructed from the trace's own held levels.

    When there is no commanded setpoint AND no catalog profile applies, the
    reader can still validate against what the chamber was evidently TRYING to
    hold: the levels the trace settles at. We snap each sample to the nearest
    held level when it is dwelling, and step to the destination level while it
    is slewing (a step-controlled chamber commands the target, then ramps to
    it). This is the trace's own evidence, not an external target.
    """
    temps = [r.temperature_c_raw for r in rows]
    if len(temps) < 4:
        return []
    lo, hi = min(temps), max(temps)
    span = hi - lo
    if span < 5.0:
        return []

    # Held levels = the temperatures the chamber actually HOLDS, not the trace
    # extremes. The extremes capture brief overshoot peaks (e.g. a 58 C spike
    # on a dwell that settles at 54 C); using them as the setpoint would score
    # the whole dwell as off-target. So take the levels from the dwell regions'
    # own mean hold temperatures, clustering repeats of the same level.
    import numpy as _np
    hold_temps: list[float] = []
    ms = getattr(context, "metric_set", None)
    if ms and getattr(ms, "dwell_metrics", None):
        hold_temps = [d.mean_temperature_c for d in ms.dwell_metrics if d.mean_temperature_c is not None]
    if not hold_temps:
        # Fallback: modes of the temperature histogram — the most-dwelt values.
        hist, edges = _np.histogram(temps, bins=40)
        centres = (edges[:-1] + edges[1:]) / 2.0
        peak_bins = [i for i in range(len(hist)) if hist[i] >= 0.3 * hist.max()]
        hold_temps = [float(centres[i]) for i in peak_bins] or [hi, lo]

    # Cluster hold temperatures that are within 8% of span of each other and
    # take each cluster's median as a level.
    tol_cluster = max(1.0, 0.08 * span)
    levels: list[float] = []
    for t in sorted(hold_temps):
        if levels and abs(t - levels[-1]) <= tol_cluster:
            levels[-1] = (levels[-1] + t) / 2.0
        else:
            levels.append(t)
    # Include the ambient start level if the trace clearly begins away from any
    # held level (the lead-in soak).
    ambient = temps[0]
    if levels and all(abs(ambient - lv) > 0.1 * span for lv in levels):
        levels.append(ambient)
    levels = sorted(set(round(lv, 1) for lv in levels)) or [round(lo, 1), round(hi, 1)]

    band = 0.12 * span
    # In-band level per sample (None while slewing between levels).
    in_band: list[float | None] = []
    for t in temps:
        best = min(levels, key=lambda lv: abs(lv - t))
        in_band.append(best if abs(best - t) <= band else None)

    # Fill slewing gaps with the DESTINATION level (next settled level).
    committed: list[float] = [0.0] * len(temps)
    # first committed = first settled level, else nearest to start temp
    nxt = next((v for v in in_band if v is not None), min(levels, key=lambda lv: abs(lv - temps[0])))
    last = nxt
    # forward pass carrying the next known destination
    dest_after = [None] * len(temps)
    nd = None
    for i in range(len(temps) - 1, -1, -1):
        if in_band[i] is not None:
            nd = in_band[i]
        dest_after[i] = nd
    for i, v in enumerate(in_band):
        if v is not None:
            last = v
            committed[i] = v
        else:
            committed[i] = dest_after[i] if dest_after[i] is not None else last

    step = max(1, len(rows) // 600)
    return [
        [round(rows[i].elapsed_seconds, 1), round(committed[i], 2)]
        for i in range(0, len(rows), step)
    ]


def _build_interactive_validation_data(context: AnalysisContext) -> dict[str, Any]:
    """Machine-readable per-phase data for the report's threshold panel.

    Carries every phase's measured values and weights so the HTML report can
    re-run validation client-side when the reader adjusts a threshold. Not
    rendered as a table — the HTML generator turns it into the interactive
    panel beside the chart.
    """
    data: dict[str, Any] = {"title": "Interactive Validation Data", "thresholds": {}, "bands": {}, "phases": []}

    # --- Profile-overlay support (for traces without a setpoint channel) ---
    # Whether the ingested data carried its own setpoint. When it did not, the
    # reader picks a target profile from the catalog and it is overlaid client
    # side. We embed: the catalog cycle-units, the trace's actual temperature
    # (downsampled), its detected cycle count, and its duration.
    no_setpoint = not (context.file_metadata and context.file_metadata.selected_setpoint_channel)
    data["no_setpoint"] = no_setpoint
    data["n_cycles"] = len(context.cycles.cycles) if context.cycles else 0

    try:
        from inputs.profile_catalog import load_catalog_units
        data["catalog"] = load_catalog_units()
    except Exception:
        data["catalog"] = []

    if context.canonical_trace and context.canonical_trace.rows:
        rows = context.canonical_trace.rows
        data["trace_duration_s"] = round(rows[-1].elapsed_seconds - rows[0].elapsed_seconds, 1)
        # Each detected cycle's [start, end] elapsed seconds — the JS anchors
        # one profile cycle to each so the overlay stays in sync with the
        # trace's actual cycles while keeping the profile's native rates.
        if context.cycles and context.cycles.cycles:
            data["cycle_spans"] = [
                [round(rows[max(0, min(c.start_row, len(rows) - 1))].elapsed_seconds, 1),
                 round(rows[max(0, min(c.end_row, len(rows) - 1))].elapsed_seconds, 1)]
                for c in context.cycles.cycles
            ]
        # No setpoint channel means the dwell-anchored cycle detector is
        # unreliable (plateaus get read as overshoot/correction against an
        # inferred setpoint, collapsing the cycle count). For a profile to
        # tile correctly we need the trace's OWN cycle count, so derive spans
        # directly from the temperature excursions — this needs no setpoint.
        if no_setpoint:
            spans = _trace_cycle_spans(rows)
            if spans:
                data["cycle_spans"] = spans
                data["n_cycles"] = len(spans)
            # An inferred stepped setpoint from the trace's own held levels —
            # offered as a target when no catalog profile applies.
            infer = _inferred_setpoint_trajectory(rows, context)
            if infer:
                data["inferred_setpoint"] = infer
        # Downsample the actual temperature to ~600 points for client-side
        # conformance recomputation against an overlaid profile.
        step = max(1, len(rows) // 600)
        data["trace_series"] = [
            [round(rows[i].elapsed_seconds, 1), round(rows[i].temperature_c_raw, 2)]
            for i in range(0, len(rows), step)
        ]
        # Dwell-region time spans so the client can evaluate conformance and
        # overshoot PER REGION (not per sample) and place worst-deviation
        # markers within each dwell.
        if context.region_list:
            dwell_spans = []
            for rg in context.region_list.regions:
                cls = getattr(rg.primary_classification, "value", str(rg.primary_classification))
                if "DWELL" in cls:
                    s = rows[max(0, min(rg.start_row, len(rows) - 1))].elapsed_seconds
                    e = rows[max(0, min(rg.end_row, len(rows) - 1))].elapsed_seconds
                    dwell_spans.append([round(s, 1), round(e, 1)])
            data["dwell_regions"] = dwell_spans

    if context.profile and getattr(context.profile, "tolerance_resolutions", None):
        for res in context.profile.tolerance_resolutions:
            data["thresholds"][res.parameter_name] = {
                "value": round(res.resolved_value, 3),
                "source": res.source,
                "derivation": res.derivation_method or "explicit profile value",
            }

    # Per-direction ramp bands (centre + tolerance), read back from the audit
    # trail where the engine recorded them.
    if context.audit_log:
        for e in context.audit_log.entries:
            if e.action == "ramp_band_derived" and e.thresholds_used:
                target = e.thresholds_used.get("target", 0.0)
                tolerance = e.thresholds_used.get("tolerance", 0.0)
                # Band width is exposed to the reader as a PERCENTAGE of the
                # target rate (e.g. ±20% of 5°C/min), which is how a spec is
                # usually stated, rather than an absolute °C/min.
                tolerance_pct = (tolerance / target * 100.0) if target else 0.0
                data["bands"][e.input_reference.lower()] = {
                    "target": round(target, 3),
                    "tolerance": round(tolerance, 3),
                    "tolerance_pct": round(tolerance_pct, 1),
                    "derivation": e.reason,
                }

    if not context.validation_results:
        return data

    # Per-ramp band centre (the ramp's own commanded slope) for the panel.
    target_by_region = {}
    for vr in context.validation_results.results:
        if "RAMP_RATE" in vr.requirement_id:
            target_by_region[vr.region_id] = round(vr.threshold_value, 3)

    for vr in context.validation_results.results:
        if vr.result.value == "NOT_APPLICABLE":
            continue
        if "SETPOINT_DEVIATION" in vr.requirement_id:
            data["phases"].append({
                "id": vr.region_id,
                "kind": "dwell",
                "measured": round(vr.measured_value, 3),
                "weight": max(vr.included_rows, 1),
            })
        elif "HEATING_RAMP_RATE" in vr.requirement_id:
            data["phases"].append({
                "id": vr.region_id,
                "kind": "heating_ramp",
                "measured": round(vr.measured_value, 3),
                "target": target_by_region.get(vr.region_id),
                "weight": max(vr.included_rows, 1),
            })
        elif "COOLING_RAMP_RATE" in vr.requirement_id:
            data["phases"].append({
                "id": vr.region_id,
                "kind": "cooling_ramp",
                "measured": round(vr.measured_value, 3),
                "target": target_by_region.get(vr.region_id),
                "weight": max(vr.included_rows, 1),
            })

    return data


def _summarise_tolerance_resolutions(context: AnalysisContext) -> dict[str, Any] | None:
    """Requirements/tolerances table: value, source, and derivation.

    Makes the self-referential validation legible: every threshold states
    whether it came from an explicit profile or was derived from the trace's
    own commanded programme / demonstrated control accuracy.
    """
    if not context.profile or not getattr(context.profile, "tolerance_resolutions", None):
        return None
    out: dict[str, Any] = {}
    for res in context.profile.tolerance_resolutions:
        unit = "°C/min" if "ramp_rate" in res.parameter_name else "°C"
        out[res.parameter_name] = {
            "value": round(res.resolved_value, 3),
            "unit": unit,
            "source": res.source,
            "derivation": res.derivation_method or "explicit profile value",
        }
    return out


def build_executive_summary(context: AnalysisContext) -> dict[str, Any]:
    """Build Section 1: Executive Summary (matches reference template format)."""
    overall = context.overall_validation_status
    status = overall.status.value if overall else "UNKNOWN"
    reason = overall.reason if overall else "No validation performed"
    
    # Conformance is only meaningful against a real setpoint. With no setpoint
    # channel it is undefined until the reader picks a target in the report, so
    # report it as N/A rather than a number computed against a self-derived
    # target.
    no_setpoint = not (context.file_metadata and context.file_metadata.selected_setpoint_channel)
    conformance_num = context.phase_conformance.conformance_percentage if context.phase_conformance else 0.0
    # Wrap the figure in a span the report JS can update live once a target is
    # picked (id rr-exec-conf), so this matches the panel rather than showing a
    # stale server number.
    if no_setpoint:
        conformance_pct = '<span id="rr-exec-conf">N/A — select a target profile</span>'
        # Do NOT cite a stale conformance % in the reason (it confused 31% vs
        # the panel's live value); conformance is determined on target select.
        reason = ("Ramp rates validated from the trace. No setpoint channel — setpoint "
                  "conformance is pending until a target profile is selected in the report.")
    else:
        conformance_pct = f'<span id="rr-exec-conf">{conformance_num:.1f}%</span>'

    # Extract test parameters
    hot_target = context.resolved_setpoints.inferred_hot_setpoint_c if context.resolved_setpoints else None
    cold_target = context.resolved_setpoints.inferred_cold_setpoint_c if context.resolved_setpoints else None

    return {
        "title": "Test Summary & Observations",
        "conformance_percentage": conformance_pct,
        "overall_status": status,
        "status_reason": reason,
        "file_name": context.file_metadata.source_file_path if context.file_metadata else "",
        "profile_name": context.profile.profile_metadata.profile_name if context.profile else "",
        "test_parameters": {
            "hot_soak_target_c": hot_target,
            "cold_soak_target_c": cold_target,
            "test_duration_hours": (context.data_quality_report.process_duration_seconds / 3600.0) if context.data_quality_report else None,
            "total_phases_identified": len(context.region_list.regions) if context.region_list else 0,
        },
        "root_cause_assessment": "Manual review required - automated root cause analysis not yet implemented",
        "probable_causes": [
            "Requires domain expert analysis",
            "Consider: refrigerant levels, valve operation, heat exchanger condition",
            "Review auxiliary sensor data if available",
        ],
        "recommendation": (
            "Ramp rates validated from the trace; conformance is pending — select a target "
            "profile in the report to assess it."
            if no_setpoint
            else f"Test {'CONFORMING' if conformance_num >= 60.0 else 'NON-CONFORMING'}. "
                 f"{'Chamber requires maintenance before re-test.' if conformance_num < 60.0 else 'Chamber performance acceptable.'}"
        ),
        "cycle_count": len(context.cycles.cycles) if context.cycles else 0,
        "region_count": len(context.region_list.regions) if context.region_list else 0,
        "valid_ramp_count": len(context.valid_ramp_regions.regions) if context.valid_ramp_regions else 0,
        "phase_conformance": context.phase_conformance.model_dump() if context.phase_conformance else {},
    }


def _build_input_file_summary(context: AnalysisContext) -> dict[str, Any]:
    """Build Section 2: Input File Summary."""
    if not context.file_metadata:
        return {"title": "Input File Summary", "note": "Not applicable — no file metadata"}

    return {
        "title": "Input File Summary",
        "source_file_path": context.file_metadata.source_file_path,
        "detected_delimiter": context.file_metadata.detected_delimiter,
        "detected_encoding": context.file_metadata.detected_encoding,
        "detected_timestamp_format": context.file_metadata.detected_timestamp_format,
        "raw_row_count": context.file_metadata.raw_row_count,
        "usable_row_count": context.file_metadata.usable_row_count,
        "selected_temperature_channel": context.file_metadata.selected_temperature_channel,
        "selected_setpoint_channel": context.file_metadata.selected_setpoint_channel,
    }


def _build_data_quality_summary(context: AnalysisContext) -> dict[str, Any]:
    """Build Section 3: Data Quality Summary."""
    if not context.data_quality_report:
        return {"title": "Data Quality Summary", "note": "Not applicable — no quality report"}

    return {
        "title": "Data Quality Summary",
        "overall_status": context.data_quality_report.overall_status.value,
        "missing_data_pct": context.data_quality_report.missing_data_pct,
        "duplicate_timestamp_pct": context.data_quality_report.duplicate_timestamp_pct,
        "spike_count": context.data_quality_report.spike_count,
        "large_gap_count": context.data_quality_report.large_gap_count,
        "process_duration_seconds": context.data_quality_report.process_duration_seconds,
    }


def _build_process_boundary_summary(context: AnalysisContext) -> dict[str, Any]:
    """Build Section 4: Process Boundary Summary."""
    if not context.process_boundaries:
        return {"title": "Process Boundary Summary", "note": "Not applicable — no boundaries detected"}

    return {
        "title": "Process Boundary Summary",
        "ambient_start_index": context.process_boundaries.ambient_start_index,
        "process_start_index": context.process_boundaries.process_start_index,
        "process_end_index": context.process_boundaries.process_end_index,
        "detection_method": context.process_boundaries.detection_method,
    }


def _build_setpoint_inference_summary(context: AnalysisContext) -> dict[str, Any]:
    """Build Section 5: Setpoint/Dwell Inference Summary."""
    if not context.resolved_setpoints:
        return {"title": "Setpoint/Dwell Inference Summary", "note": "Not applicable — no setpoints resolved"}

    return {
        "title": "Setpoint/Dwell Inference Summary",
        "resolution_mode": context.resolved_setpoints.resolution_mode.value,
        "inferred_ambient_c": context.resolved_setpoints.inferred_ambient_c,
        "inferred_hot_setpoint_c": context.resolved_setpoints.inferred_hot_setpoint_c,
        "inferred_cold_setpoint_c": context.resolved_setpoints.inferred_cold_setpoint_c,
        "algorithm_seed_used": context.resolved_setpoints.algorithm_seed_used,
    }


def _build_region_classification_summary(context: AnalysisContext) -> dict[str, Any]:
    """Build Section 6: Phase Analysis (matches reference template format)."""
    if not context.region_list:
        return {"title": "Phase Analysis", "note": "Not applicable — no regions classified"}

    phase_table = build_phase_analysis_table(context)

    return {
        "title": "Phase Analysis",
        "subtitle": (
            "Ramp rates: Theil-Sen robust slope over the isolated ramp envelope "
            "(dwell tails and overshoot/correction excluded). Dwell temps: mean "
            "over the soak region."
        ),
        "total_phases": len(context.region_list.regions),
        "phases": phase_table,
    }


def build_phase_analysis_table(context: AnalysisContext) -> list[dict[str, Any]]:
    """Build phase-by-phase analysis table matching reference template.
    
    Columns: #, Type, SP Range, Rate (°C/min), Avg Temp (°C), Duration, Max Dev, Status
    """
    if not context.region_list:
        return []
    
    phase_table = []
    anomaly_ids = set(context.phase_conformance.anomaly_phase_ids) if context.phase_conformance else set()
    # With no setpoint channel, dwell "setpoint-deviation" anomalies are
    # measured against a self-derived target and are meaningless (they flagged
    # dwells 50+ C "off" a bogus inferred setpoint). Only ramp-rate anomalies
    # are trace-intrinsic and valid, so suppress dwell anomalies in Mode B.
    no_setpoint = not (context.file_metadata and context.file_metadata.selected_setpoint_channel)
    
    # Build lookup maps
    ramp_metrics_map = {}
    dwell_metrics_map = {}
    if context.metric_set:
        ramp_metrics_map = {rm.region_id: rm for rm in context.metric_set.ramp_metrics}
        dwell_metrics_map = {dm.region_id: dm for dm in context.metric_set.dwell_metrics}
    
    setpoints = context.resolved_setpoints
    
    for idx, region in enumerate(context.region_list.regions, start=1):
        region_id = region.region_id
        region_type = region.primary_classification.value
        
        # Determine setpoint range
        sp_range = "-"
        if "HEATING_RAMP" in region_type:
            if setpoints:
                sp_range = f"{setpoints.inferred_cold_setpoint_c or setpoints.inferred_ambient_c} -> {setpoints.inferred_hot_setpoint_c}"
        elif "COOLING_RAMP" in region_type:
            if setpoints:
                sp_range = f"{setpoints.inferred_hot_setpoint_c} -> {setpoints.inferred_cold_setpoint_c}"
        elif "HOT_DWELL" in region_type:
            if setpoints:
                sp_range = str(setpoints.inferred_hot_setpoint_c)
        elif "COLD_DWELL" in region_type:
            if setpoints:
                sp_range = str(setpoints.inferred_cold_setpoint_c)
        elif "AMBIENT" in region_type:
            if setpoints:
                sp_range = str(setpoints.inferred_ambient_c or 20)
        
        # Extract ramp rate or dwell avg temp
        rate = None
        avg_temp = None
        max_dev = None
        
        if "RAMP" in region_type and region_id in ramp_metrics_map:
            rm = ramp_metrics_map[region_id]
            rate = round(rm.robust_slope_c_per_min, 2) if rm.robust_slope_c_per_min else None
        
        if "DWELL" in region_type and region_id in dwell_metrics_map:
            dm = dwell_metrics_map[region_id]
            avg_temp = round(dm.mean_temperature_c, 1) if dm.mean_temperature_c else None
            max_dev = round(abs(dm.setpoint_deviation_c), 1) if dm.setpoint_deviation_c else None
        
        # Duration in minutes
        duration_min = round(region.duration_seconds / 60.0, 1)
        
        # Status: OK or ANOMALY. In Mode B only ramp regions can be anomalies
        # (ramp rate is trace-intrinsic); dwell/other setpoint-relative
        # anomalies are undefined without a target.
        is_anom = region_id in anomaly_ids
        if no_setpoint and "RAMP" not in region_type:
            is_anom = False
        status = "ANOMALY" if is_anom else "OK"
        
        # Map region type to simplified display type
        display_type = region_type.lower().replace("_", "_")
        if "heating_ramp" in display_type:
            display_type = "ramp_up"
        elif "cooling_ramp" in display_type:
            display_type = "ramp_down"
        elif "hot_dwell" in display_type or "cold_dwell" in display_type:
            display_type = "dwell"
        
        phase_table.append({
            "phase_number": idx,
            "type": display_type,
            "sp_range": sp_range,
            "rate_c_per_min": rate,
            "avg_temp_c": avg_temp,
            "duration_min": duration_min,
            "max_dev_c": max_dev,
            "status": status,
        })
    
    return phase_table


def _build_ramp_rate_validation_summary(context: AnalysisContext) -> dict[str, Any]:
    """Build Section 7: Ramp-Rate Validation Summary."""
    if not context.validation_results:
        return {"title": "Ramp-Rate Validation Summary", "note": "Not applicable — no validation results"}

    results = []
    for vr in context.validation_results.results:
        if "RAMP_RATE" in vr.requirement_id:
            # For a ramp-rate band the threshold_value is the target (band
            # centre); deviation is how far the measured rate sits from it
            # (signed: + = ramping faster than target, - = slower).
            deviation = None
            if vr.measured_value is not None and vr.threshold_value is not None:
                deviation = round(vr.measured_value - vr.threshold_value, 2)
            results.append({
                "requirement_id": vr.requirement_id,
                "measured_value": vr.measured_value,
                "threshold_value": vr.threshold_value,
                "deviation_c_per_min": deviation,
                "result": vr.result.value,
                "reason": vr.reason,
            })

    # Aggregate the spread of deviations across all ramps. The plain (signed)
    # mean is deliberately NOT used — opposite-sign deviations cancel and hide
    # real scatter. Both aggregates below measure MAGNITUDE:
    #   mean_abs = mean(|deviation|)      — typical distance off target
    #   rms      = sqrt(mean(deviation^2)) — variance-based spread (penalises
    #                                        large misses more; this is the
    #                                        "total variance" figure)
    import math as _math
    devs = [r["deviation_c_per_min"] for r in results if r["deviation_c_per_min"] is not None]
    deviation_summary = None
    if devs:
        deviation_summary = {
            "n_ramps": len(devs),
            "mean_abs_deviation_c_per_min": round(sum(abs(d) for d in devs) / len(devs), 3),
            "rms_deviation_c_per_min": round(_math.sqrt(sum(d * d for d in devs) / len(devs)), 3),
            "max_abs_deviation_c_per_min": round(max(abs(d) for d in devs), 2),
        }

    return {
        "title": "Ramp-Rate Validation Summary",
        "ramp_results": results,
        "deviation_summary": deviation_summary,
    }


def _build_cycle_level_summary(context: AnalysisContext) -> dict[str, Any]:
    """Build Section 8: Cycle-Level Summary."""
    if not context.cycles:
        return {"title": "Cycle-Level Summary", "note": "Not applicable — no cycles detected"}

    cycles = []
    for cycle in context.cycles.cycles:
        cycles.append({
            "cycle_id": cycle.cycle_id,
            "cycle_number": cycle.cycle_number,
            "status": cycle.status.value,
            "duration_seconds": cycle.duration_seconds,
            "heating_ramp_count": cycle.heating_ramp_count,
            "cooling_ramp_count": cycle.cooling_ramp_count,
            "hot_dwell_count": cycle.hot_dwell_count,
            "cold_dwell_count": cycle.cold_dwell_count,
        })

    return {
        "title": "Cycle-Level Summary",
        "total_cycles": len(context.cycles.cycles),
        "cycles": cycles,
    }


def _region_overshoot(trace_rows, region, tolerance: float):
    """Magnitude and recovery time for an overshoot/correction region.

    Returns (magnitude_beyond_setpoint, setpoint, recovery_seconds), or None
    when the excursion does not exceed the tolerance. The setpoint is the
    region's own commanded value; recovery extends forward through the trace
    (past the region end) until the temperature returns within tolerance —
    the same "first out → first back in" measure the dwell metrics use.
    """
    import numpy as np

    start, end = region.start_row, region.end_row
    seg = trace_rows[start:end + 1]
    if not seg:
        return None
    sps = [r.setpoint_c for r in seg if r.setpoint_c is not None]
    if not sps:
        return None
    setpoint = float(np.median(sps))
    temps = np.array([r.temperature_c_raw for r in seg])

    is_hot = region.primary_classification in (RegionType.HOT_OVERSHOOT, RegionType.HOT_CORRECTION)
    if is_hot:
        mag = float(np.max(temps)) - setpoint
        out_of_band = temps > setpoint + tolerance
    else:
        mag = setpoint - float(np.min(temps))
        out_of_band = temps < setpoint - tolerance
    if mag <= tolerance or not out_of_band.any():
        return None

    # Recovery: first out-of-band sample in the region → first sample (here or
    # later in the trace) back within tolerance of this setpoint.
    first_out = start + int(np.where(out_of_band)[0][0])
    recovery = float(trace_rows[-1].elapsed_seconds - trace_rows[first_out].elapsed_seconds)
    for j in range(first_out, len(trace_rows)):
        if abs(trace_rows[j].temperature_c_raw - setpoint) <= tolerance:
            recovery = float(trace_rows[j].elapsed_seconds - trace_rows[first_out].elapsed_seconds)
            break
    return mag, setpoint, recovery


def _build_dwell_calibration_summary(context: AnalysisContext) -> dict[str, Any]:
    """Overshoot recovery & stability.

    Reports, for every overshoot that breaks the setpoint tolerance, HOW LONG
    the temperature took to return within tolerance (the recovery time), and
    whether overshoot magnitude, oscillation and recovery time shrink over the
    run (a chamber settling in — a good sign). Per-level calibration bands and
    the overshoot distribution were removed: recovery time is the headline."""
    import numpy as np

    # Overshoot is setpoint-relative — undefined with no setpoint channel.
    # Until the reader picks a target in the report, overshoots are assessed
    # client-side against that target, so this static section defers.
    no_setpoint = not (context.file_metadata and context.file_metadata.selected_setpoint_channel)
    if no_setpoint:
        return {
            "title": "Overshoot Recovery & Stability",
            "note": "No setpoint channel — overshoot is undefined until a target profile "
                    "is selected in the report; it is then assessed live against that target.",
        }

    if not context.metric_set or not context.metric_set.dwell_metrics:
        return {"title": "Overshoot Recovery & Stability", "note": "No dwell metrics computed"}

    # Order dwells chronologically (by region start row) so trends read
    # first-to-last across the run.
    order = {}
    if context.region_list:
        order = {r.region_id: r.start_row for r in context.region_list.regions}
    dwells = sorted(
        context.metric_set.dwell_metrics,
        key=lambda d: order.get(d.region_id, 0),
    )

    # Resolved setpoint tolerance — an overshoot only counts when it exceeds it.
    setpoint_tolerance = None
    if context.profile and getattr(context.profile, "tolerance_resolutions", None):
        for res in context.profile.tolerance_resolutions:
            if res.parameter_name == "dwell_setpoint_deviation":
                setpoint_tolerance = res.resolved_value
                break

    def _fmt_recovery(recovery_s: float) -> str:
        return f"{recovery_s / 60.0:.1f} min" if recovery_s >= 60 else f"{recovery_s:.0f} s"

    # Overshoots that break the setpoint tolerance — each with how long it
    # took the temperature to return within tolerance (the recovery time).
    # Sourced from BOTH the dwell metrics AND separate overshoot/correction
    # regions, so the report table always agrees with the chart call-outs.
    flagged = []
    recovery_values: list[float] = []
    for d in dwells:
        mag = d.overshoot_magnitude_c or 0.0
        if setpoint_tolerance and mag > setpoint_tolerance:
            recovery_s = d.overshoot_recovery_seconds or 0.0
            recovery_values.append(recovery_s)
            flagged.append({
                "region_id": d.region_id,
                "setpoint_c": round(d.target_setpoint_c, 1) if d.target_setpoint_c is not None else None,
                "overshoot_c": round(mag, 2),
                "beyond_tolerance_c": round(mag - setpoint_tolerance, 2),
                "recovery_time": _fmt_recovery(recovery_s),
            })

    # Separate overshoot / correction regions (the classifier split these out
    # of their dwell, so they are not in dwell_metrics). Compute each one's
    # magnitude and recovery time from the raw trace.
    if setpoint_tolerance and context.canonical_trace and context.region_list:
        trace_rows = context.canonical_trace.rows
        for region in context.region_list.regions:
            if region.primary_classification not in (
                RegionType.HOT_OVERSHOOT, RegionType.COLD_OVERSHOOT,
                RegionType.HOT_CORRECTION, RegionType.COLD_CORRECTION,
            ):
                continue
            res = _region_overshoot(trace_rows, region, setpoint_tolerance)
            if res is None:
                continue
            mag, sp, recovery_s = res
            recovery_values.append(recovery_s)
            flagged.append({
                "region_id": region.region_id,
                "setpoint_c": round(sp, 1),
                "overshoot_c": round(mag, 2),
                "beyond_tolerance_c": round(mag - setpoint_tolerance, 2),
                "recovery_time": _fmt_recovery(recovery_s),
            })

    # Sort flagged overshoots chronologically for readability.
    flagged.sort(key=lambda f: order.get(f["region_id"], 0))

    # Headline recovery statistic: how long overshoots take to return to
    # setpoint, in aggregate.
    recovery_summary = None
    if recovery_values:
        recovery_summary = {
            "setpoint_tolerance_c": round(setpoint_tolerance, 2) if setpoint_tolerance else None,
            "n_overshoots": len(recovery_values),
            "median_recovery": _fmt_recovery(float(np.median(recovery_values))),
            "worst_recovery": _fmt_recovery(float(np.max(recovery_values))),
        }
    elif setpoint_tolerance:
        recovery_summary = {"setpoint_tolerance_c": round(setpoint_tolerance, 2), "n_overshoots": 0}

    # Over-run trend: do overshoot magnitude, oscillation count and recovery
    # time fall from the first half of the run to the second? (stabilising).
    def _trend(values: list[float]) -> dict[str, Any] | None:
        vals = [v for v in values if v is not None]
        if len(vals) < 4:
            return None
        mid = len(vals) // 2
        first = float(np.mean(vals[:mid]))
        second = float(np.mean(vals[mid:]))
        if first <= 0:
            direction = "stable"
        elif second < first * 0.9:
            direction = "reducing over run (improving — chamber stabilising)"
        elif second > first * 1.1:
            direction = "increasing over run (worsening — worth review)"
        else:
            direction = "stable across run"
        return {"first_half_mean": round(first, 2), "second_half_mean": round(second, 2), "assessment": direction}

    trends = {
        "overshoot_magnitude_c": _trend([d.overshoot_magnitude_c or 0.0 for d in dwells]),
        "oscillation_count": _trend([float(d.oscillation_count) for d in dwells]),
        "recovery_seconds": _trend([d.overshoot_recovery_seconds or 0.0 for d in dwells]),
    }

    return {
        "title": "Overshoot Recovery & Stability",
        "recovery_summary": recovery_summary,
        "flagged_overshoots": flagged,
        "flagged_overshoot_count": len(flagged),
        "trends": {k: v for k, v in trends.items() if v is not None},
    }


def _build_profile_consistency_summary(context: AnalysisContext) -> dict[str, Any]:
    """Build Section 10: Profile Consistency Summary."""
    if not context.profile_comparison_results:
        return {"title": "Profile Consistency Summary", "note": "Not applicable — no comparison performed"}

    return {
        "title": "Profile Consistency Summary",
        "comparisons": context.profile_comparison_results.comparisons,
    }


def _build_visualisation_summary(context: AnalysisContext) -> dict[str, Any]:
    """Build Section 11: Visualisation."""
    if not context.visualisation_bundle:
        return {"title": "Visualisation", "note": "Not applicable — no visualisation generated"}

    return {
        "title": "Visualisation",
        "charts": list(context.visualisation_bundle.charts.keys()),
    }


def _build_audit_trail_summary(context: AnalysisContext) -> dict[str, Any]:
    """Build Section 12: Audit Trail."""
    if not context.audit_log:
        return {"title": "Audit Trail", "note": "Not applicable — no audit log"}

    entries = []
    for entry in context.audit_log.entries:
        entries.append({
            "timestamp": entry.timestamp.isoformat() if entry.timestamp else "",
            "module_name": entry.module_name,
            "action": entry.action,
            "decision": entry.decision,
            "reason": entry.reason,
            "severity": entry.severity.value,
        })

    return {
        "title": "Audit Trail",
        "total_entries": len(entries),
        "entries": entries[:50],  # First 50 entries for brevity
    }


def _build_algorithm_appendix(context: AnalysisContext) -> dict[str, Any]:
    """Build Section 13: Appendix — Algorithm Configuration."""
    if not context.run_metadata:
        return {"title": "Appendix: Algorithm Configuration", "note": "Not applicable — no run metadata"}

    return {
        "title": "Appendix: Algorithm Configuration",
        "algorithm_version": context.run_metadata.algorithm_version,
        "python_version": context.run_metadata.python_version,
        "random_seed": context.run_metadata.random_seed,
        "adaptive_constants_snapshot": context.run_metadata.adaptive_constants_snapshot,
        "classifier_configuration_snapshot": context.run_metadata.classifier_configuration_snapshot,
        "feature_extraction_configuration": context.run_metadata.feature_extraction_configuration,
    }
