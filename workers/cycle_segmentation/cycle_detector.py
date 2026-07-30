"""Cycle detection and segmentation (M09).

Handles:
- Cycle sequence detection and region assignment
- Cycle completeness evaluation
- Cycle duration and cycle-to-cycle drift
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np

from config.constants import (
    AuditCategory,
    AuditSeverity,
    CycleStatus,
    RegionType,
)
from models.domain import (
    AuditEntry,
    AuditLog,
    Cycle,
    CycleList,
    Region,
    RegionList,
    ResolvedSetpoints,
    ValidRampRegion,
)


def detect_cycles(
    regions: RegionList,
    valid_ramps: list[ValidRampRegion],
    setpoints: ResolvedSetpoints,
    min_cycle_regions: int = 2,
    audit_log: AuditLog | None = None,
) -> CycleList:
    """Detect and segment thermal cycles from classified regions.
    
    A cycle is defined as a sequence of regions that form a complete
    thermal profile (e.g., ambient -> hot dwell -> cold dwell -> ambient,
    or hot dwell -> cold dwell -> hot dwell).
    
    Args:
        regions: Classified regions
        valid_ramps: Isolated valid ramps
        setpoints: Resolved setpoints
        min_cycle_regions: Minimum regions to form a cycle
        audit_log: Optional audit log
    
    Returns:
        CycleList with detected cycles
    """
    if audit_log is None:
        audit_log = AuditLog()
    
    if not regions.regions:
        return CycleList(cycles=[])
    
    ramp_map = {r.region_id: r for r in valid_ramps}
    
    cycle_boundaries = _find_cycle_boundaries(regions.regions, setpoints)
    
    cycles = []
    for i, (start_idx, end_idx) in enumerate(cycle_boundaries):
        cycle_regions = regions.regions[start_idx:end_idx + 1]
        
        if len(cycle_regions) < min_cycle_regions:
            continue
        
        cycle_ramps = [
            ramp_map[r.region_id] for r in cycle_regions
            if r.region_id in ramp_map
        ]
        
        status, completeness_reason = _evaluate_cycle_completeness(
            cycle_regions, setpoints
        )
        
        start_time = cycle_regions[0].start_time
        end_time = cycle_regions[-1].end_time
        duration = (end_time - start_time).total_seconds()
        
        # Count only SUBSTANTIAL ramps — those traversing a meaningful part of
        # the hot-cold span. On noisy traces the return-to-ambient can shatter
        # into many cooling slivers; those are not separate ramps and must not
        # inflate the per-cycle count (which should be ~1 heating + 1 cooling).
        span_full = abs(
            (setpoints.inferred_hot_setpoint_c or 0.0)
            - (setpoints.inferred_cold_setpoint_c or 0.0)
        )
        min_ramp_span = span_full * 0.3 if span_full > 0 else 0.0
        heating_ramps = [
            r for r in cycle_ramps
            if r.direction.value == "HEATING" and abs(r.temperature_delta_c) >= min_ramp_span
        ]
        cooling_ramps = [
            r for r in cycle_ramps
            if r.direction.value == "COOLING" and abs(r.temperature_delta_c) >= min_ramp_span
        ]
        
        cycle = Cycle(
            cycle_id=f"C{i:04d}",
            cycle_number=i + 1,
            start_row=cycle_regions[0].start_row,
            end_row=cycle_regions[-1].end_row,
            start_time=start_time,
            end_time=end_time,
            duration_seconds=duration,
            region_ids=[r.region_id for r in cycle_regions],
            regions=cycle_regions,
            valid_ramps=cycle_ramps,
            heating_ramp_count=len(heating_ramps),
            cooling_ramp_count=len(cooling_ramps),
            hot_dwell_count=sum(1 for r in cycle_regions if r.primary_classification == RegionType.HOT_DWELL),
            cold_dwell_count=sum(1 for r in cycle_regions if r.primary_classification == RegionType.COLD_DWELL),
            status=status,
            completeness_reason=completeness_reason,
            is_complete=status == CycleStatus.COMPLETE,
        )
        cycles.append(cycle)
        
        audit_log.add(AuditEntry(
            timestamp=datetime.now(),
            module_name="cycle_detector",
            action="detect_cycle",
            input_reference=f"regions {start_idx}-{end_idx}",
            output_reference=cycle.cycle_id,
            decision=status.value,
            reason=completeness_reason,
            severity=AuditSeverity.INFO,
            category=AuditCategory.CLASSIFICATION,
            rows_used=cycle.end_row - cycle.start_row + 1,
        ))
    
    if len(cycles) >= 2:
        _compute_cycle_drift(cycles)
    
    audit_log.add(AuditEntry(
        timestamp=datetime.now(),
        module_name="cycle_detector",
        action="detect_cycles_complete",
        output_reference=f"CycleList({len(cycles)} cycles)",
        decision="SUCCESS",
        reason=f"Detected {len(cycles)} cycles from {len(regions.regions)} regions",
        severity=AuditSeverity.INFO,
        category=AuditCategory.CLASSIFICATION,
    ))
    
    return CycleList(cycles=cycles)


def _find_cycle_boundaries(
    regions: list[Region],
    setpoints: ResolvedSetpoints,
) -> list[tuple[int, int]]:
    """Find cycle boundaries.

    A cycle is one complete thermal excursion away from ambient and back:
    it contains ONE heating ramp and ONE cooling ramp (with their dwells),
    in EITHER order — the tool must not assume heat-first or cold-first,
    because different tests run them in different orders.

    A new cycle begins when a ramp direction REPEATS: once the current cycle
    has both a heating and a cooling ramp, the next ramp of an
    already-seen direction is the start of the next excursion. Leading and
    trailing non-ramp regions (the ambient soak at the start and the
    return-to-ambient at the end) attach to the first and last cycle. So the
    heating ramp is always inside its cycle — which is why it is now counted.
    """
    if not regions:
        return []

    n = len(regions)

    # Anchor on SUBSTANTIAL DWELLS, not ramps. Dwells (one hot, one cold per
    # cycle) are cleanly classified even when ramps fragment into slivers, so
    # counting cycles by dwells is robust. But a noisy start can leave tiny
    # dwell fragments (a 0-1 min "dwell" next to an overshoot); those are not
    # real soaks and must not each spawn a cycle. So only dwells lasting a
    # meaningful fraction of the typical soak count as cycle anchors.
    dwell_durations = [
        r.duration_seconds for r in regions
        if r.primary_classification in (RegionType.HOT_DWELL, RegionType.COLD_DWELL)
    ]
    min_dwell_seconds = (float(np.median(dwell_durations)) * 0.2) if dwell_durations else 0.0

    # A new cycle begins once the current one has both a hot and a cold dwell
    # and the next dwell of an already-seen type appears; the cut is placed at
    # the ramp that starts that next excursion (right after the previous dwell).
    cuts = [0]
    seen: set[str] = set()
    prev_dwell_idx = -1
    for i, region in enumerate(regions):
        rt = region.primary_classification
        kind = (
            "H" if rt == RegionType.HOT_DWELL
            else "C" if rt == RegionType.COLD_DWELL
            else None
        )
        if kind is None or region.duration_seconds < min_dwell_seconds:
            continue
        if kind in seen:
            cut = prev_dwell_idx + 1
            if cut > cuts[-1]:
                cuts.append(cut)
            seen = {kind}
        else:
            seen.add(kind)
        prev_dwell_idx = i

    boundaries = [
        (cuts[k], (cuts[k + 1] - 1) if k + 1 < len(cuts) else n - 1)
        for k in range(len(cuts))
    ]

    # A trailing group that does not itself contain BOTH ramp directions is
    # the return-to-ambient tail of the previous cycle — merge it back.
    if len(boundaries) >= 2:
        s, e = boundaries[-1]
        types = {regions[j].primary_classification for j in range(s, e + 1)}
        if not (RegionType.HEATING_RAMP in types and RegionType.COOLING_RAMP in types):
            ps, _ = boundaries[-2]
            boundaries[-2] = (ps, e)
            boundaries.pop()

    return boundaries


def _is_cycle_transition(prev_region: Region, curr_region: Region) -> bool:
    """Check if transition between regions indicates new cycle.
    
    This is intentionally conservative - we don't try to detect cycle boundaries.
    Instead, we let _find_cycle_boundaries use a state machine to track thermal excursions.
    """
    # AMBIENT_START always begins a new cycle
    if curr_region.primary_classification == RegionType.AMBIENT_START:
        return True
    
    return False


def _evaluate_cycle_completeness(
    regions: list[Region],
    setpoints: ResolvedSetpoints,
) -> tuple[CycleStatus, str]:
    """Evaluate whether a cycle is complete."""
    region_types = [r.primary_classification for r in regions]
    
    has_heating = RegionType.HEATING_RAMP in region_types
    has_cooling = RegionType.COOLING_RAMP in region_types
    has_hot_dwell = RegionType.HOT_DWELL in region_types
    has_cold_dwell = RegionType.COLD_DWELL in region_types
    
    if has_heating and has_cooling and (has_hot_dwell or has_cold_dwell):
        return CycleStatus.COMPLETE, "Full thermal cycle with ramps and dwells"
    
    if has_heating and has_hot_dwell:
        return CycleStatus.PARTIAL, "Heating cycle only (no cooling)"
    
    if has_cooling and has_cold_dwell:
        return CycleStatus.PARTIAL, "Cooling cycle only (no heating)"
    
    if has_heating or has_cooling:
        return CycleStatus.PARTIAL, "Ramp without corresponding dwell"
    
    return CycleStatus.INVALID, "No valid ramps detected"


def _compute_cycle_drift(cycles: list[Cycle]) -> None:
    """Compute cycle-to-cycle duration drift."""
    for i in range(1, len(cycles)):
        prev_duration = cycles[i - 1].duration_seconds
        curr_duration = cycles[i].duration_seconds
        
        if prev_duration > 0:
            drift = (curr_duration - prev_duration) / prev_duration
            cycles[i].cycle_to_cycle_drift = drift
