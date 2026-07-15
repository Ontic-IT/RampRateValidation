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
        
        heating_ramps = [r for r in cycle_ramps if r.direction.value == "HEATING"]
        cooling_ramps = [r for r in cycle_ramps if r.direction.value == "COOLING"]
        
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
    """Find cycle boundaries by tracking complete thermal excursions.
    
    Domain knowledge: Traces ALWAYS start on COLD ramp down first.
    
    A complete cycle = HOT ramp → HOT dwell → COLD ramp → COLD dwell
    - COLD dwell ends the current cycle
    - Next HOT dwell starts a new cycle
    - Consecutive HOT_DWELL regions without COLD_DWELL are part of the SAME cycle
    
    Pattern: [initial COLD] → [HOT→COLD cycle 1] → [HOT→COLD cycle 2] → ...
    """
    if not regions:
        return []
    
    boundaries = []
    current_start = 0
    in_cycle = False  # Track if we're currently in a HOT→COLD cycle
    
    for i, region in enumerate(regions):
        region_type = region.primary_classification
        
        # HOT dwell starts a new cycle (if not already in one)
        if region_type == RegionType.HOT_DWELL:
            if not in_cycle:
                # Start new cycle at this HOT dwell
                current_start = i
                in_cycle = True
        
        # COLD dwell ends the current cycle (if we're in one)
        elif region_type == RegionType.COLD_DWELL:
            if in_cycle:
                # End current cycle at this COLD dwell
                boundaries.append((current_start, i))
                in_cycle = False
    
    # Add final incomplete cycle if we're still in one
    if in_cycle and current_start < len(regions):
        boundaries.append((current_start, len(regions) - 1))
    
    # Fallback: if no boundaries found, treat entire trace as one cycle
    if not boundaries and regions:
        boundaries = [(0, len(regions) - 1)]
    
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
