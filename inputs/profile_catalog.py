"""Profile catalog — parse chamber program (.pgm) files into setpoint programmes.

A .pgm is a controller step-list: each step names a target temperature and a
duration to reach/hold it, plus an optional loop (jump back to an earlier step
N times). Simulating the steps yields the commanded setpoint-vs-time
trajectory — the target the achieved trace is measured against when the
ingested data carries no setpoint channel of its own.

Line format (whitespace/comma separated, columns after the target are aux
channels the tool ignores):
    <step> <target_c> ... <HH:MM:SS duration> <next_step> <loop_count> ...
A step whose target differs from the previous is a ramp; an equal target is a
dwell. A `next_step` pointing backwards with loop_count>0 repeats a block.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

CATALOG_DIR = Path(__file__).resolve().parent.parent / "config" / "profiles"
_DURATION_RE = re.compile(r"^\d{1,2}:\d{2}:\d{2}$")
_MAX_STEPS = 100000  # safety bound on loop expansion


@dataclass
class ProfileSegment:
    """One commanded segment of a setpoint programme."""
    kind: str          # "ramp" | "dwell"
    from_c: float
    to_c: float
    duration_s: float

    @property
    def rate_c_per_min(self) -> float:
        if self.duration_s <= 0:
            return 0.0
        return abs(self.to_c - self.from_c) / (self.duration_s / 60.0)


@dataclass
class SetpointProgramme:
    """Fully expanded commanded setpoint trajectory from a .pgm."""
    name: str
    segments: list[ProfileSegment] = field(default_factory=list)
    source_file: str = ""

    def points(self) -> list[tuple[float, float]]:
        """(elapsed_seconds, setpoint_c) vertices of the trajectory."""
        if not self.segments:
            return []
        pts = [(0.0, self.segments[0].from_c)]
        t = 0.0
        for seg in self.segments:
            t += seg.duration_s
            pts.append((t, seg.to_c))
        return pts

    @property
    def total_duration_s(self) -> float:
        return sum(s.duration_s for s in self.segments)


def _parse_duration(token: str) -> float:
    h, m, s = (int(x) for x in token.split(":"))
    return h * 3600 + m * 60 + s


def _parse_pgm_steps(text: str) -> dict[int, tuple[float, float, int, int]]:
    """Parse raw steps: {step_num: (target_c, duration_s, next_step, loop_count)}."""
    steps: dict[int, tuple[float, float, int, int]] = {}
    # These controller files pad fields with control bytes (NUL, STX, ...).
    # Replace every control character except newlines/tabs with a space so the
    # numeric fields tokenise cleanly.
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", " ", text)
    for line in text.splitlines():
        # Tokenise on commas and whitespace; keep non-empty tokens.
        tokens = [t for t in re.split(r"[,\s]+", line.strip()) if t]
        if len(tokens) < 4:
            continue
        # First token must be an integer step number.
        try:
            step_num = int(tokens[0])
        except ValueError:
            continue
        # Duration is the sole HH:MM:SS token.
        dur_idx = next((i for i, t in enumerate(tokens) if _DURATION_RE.match(t)), None)
        if dur_idx is None:
            continue
        try:
            target_c = float(tokens[1])
            duration_s = _parse_duration(tokens[dur_idx])
            next_step = int(tokens[dur_idx + 1])
            loop_count = int(tokens[dur_idx + 2])
        except (ValueError, IndexError):
            continue
        steps[step_num] = (target_c, duration_s, next_step, loop_count)
    return steps


def parse_pgm(path: str | Path, name: str | None = None) -> SetpointProgramme:
    """Parse a .pgm file into a fully expanded SetpointProgramme."""
    path = Path(path)
    text = path.read_text(encoding="latin-1", errors="replace")
    steps = _parse_pgm_steps(text)
    programme = SetpointProgramme(name=name or path.stem.strip(), source_file=str(path))
    if not steps:
        return programme

    ordered = sorted(steps)
    start = ordered[0]
    # Step 0 (if present) sets the starting temperature; execution begins at 1.
    current_temp = steps[start][0]
    exec_step = start + 1 if (start + 1) in steps else start
    loop_taken: dict[int, int] = {}
    budget = _MAX_STEPS

    while exec_step in steps and budget > 0:
        budget -= 1
        target, duration, next_step, loop_count = steps[exec_step]
        kind = "dwell" if abs(target - current_temp) < 1e-6 else "ramp"
        if duration > 0:
            programme.segments.append(
                ProfileSegment(kind=kind, from_c=current_temp, to_c=target, duration_s=duration)
            )
        current_temp = target

        # Loop: a backward jump with a positive loop count repeats the block.
        if loop_count > 0 and next_step <= exec_step:
            taken = loop_taken.get(exec_step, 0)
            if taken < loop_count:
                loop_taken[exec_step] = taken + 1
                exec_step = next_step
                continue
            else:
                loop_taken[exec_step] = 0
                exec_step = exec_step + 1
                continue
        # Normal advance: follow next_step, or fall through; 0 means end.
        if next_step and next_step != exec_step:
            exec_step = next_step
        else:
            exec_step = exec_step + 1
        if exec_step == start:  # jumped back to the init step → stop
            break

    return programme


def _held_levels(segments: list[ProfileSegment], min_hold_s: float = 60.0) -> list[float]:
    """Distinct temperatures the programme dwells at (rounded, sorted)."""
    levels: dict[int, float] = {}
    for s in segments:
        if s.kind == "dwell" and s.duration_s >= min_hold_s:
            levels.setdefault(round(s.to_c), s.to_c)
    return [levels[k] for k in sorted(levels)]


def extract_cycle_unit(programme: SetpointProgramme) -> dict:
    """One representative thermal cycle + ambient, for tiling onto a trace.

    A cycle is the programme's repeating unit: the segments spanning from one
    cold-dwell arrival to the next (a full cold→hot→cold period). The ambient
    level is the near-room level the programme starts/ends at. The cycle keeps
    the programme's NATIVE ramp rates and dwell durations so tiling does not
    distort them.
    """
    segs = programme.segments
    levels = _held_levels(segs)
    if not levels:
        return {}

    hot_c = max(levels)
    cold_c = min(levels)
    # Ambient = a held level near room temperature (15-30C) if present, else
    # the programme's very first temperature.
    room = [lv for lv in levels if 10 <= lv <= 32]
    ambient_c = room[0] if room else (segs[0].from_c if segs else 20.0)

    # Cold-dwell GROUP starts: a cold-extreme dwell not immediately preceded by
    # another cold dwell (the programme often splits a soak into sub-steps).
    band = max(2.0, abs(hot_c - cold_c) * 0.05)
    cold_dwell_idx = [
        i for i, s in enumerate(segs)
        if s.kind == "dwell" and abs(s.to_c - cold_c) <= band
    ]
    group_starts = [
        idx for j, idx in enumerate(cold_dwell_idx)
        if j == 0 or cold_dwell_idx[j - 1] != idx - 1
    ]

    cycle_segs: list[ProfileSegment]
    if len(group_starts) >= 2:
        # One full period: from the start of one cold soak to the start of the
        # next (cold soak → heat → hot soak → cool → next cold soak).
        cycle_segs = segs[group_starts[0]:group_starts[1]]
    else:
        cycle_segs = segs

    heat = [s.rate_c_per_min for s in cycle_segs if s.kind == "ramp" and s.to_c > s.from_c]
    cool = [s.rate_c_per_min for s in cycle_segs if s.kind == "ramp" and s.to_c < s.from_c]

    import statistics as _st
    return {
        "name": programme.name,
        "ambient_c": round(ambient_c, 1),
        "hot_c": round(hot_c, 1),
        "cold_c": round(cold_c, 1),
        "heat_rate_c_per_min": round(_st.median(heat), 2) if heat else None,
        "cool_rate_c_per_min": round(_st.median(cool), 2) if cool else None,
        "cycle_duration_s": round(sum(s.duration_s for s in cycle_segs), 0),
        # One cycle's segments as (from, to, seconds) — the JS tiles these.
        "cycle_segments": [
            {"from_c": round(s.from_c, 2), "to_c": round(s.to_c, 2), "duration_s": round(s.duration_s, 0)}
            for s in cycle_segs
        ],
        "levels": [round(lv, 1) for lv in levels],
    }


def load_catalog(catalog_dir: str | Path = CATALOG_DIR) -> list[SetpointProgramme]:
    """Load every .pgm profile under catalog_dir/*/ (and the top level)."""
    catalog_dir = Path(catalog_dir)
    programmes: list[SetpointProgramme] = []
    for pgm in sorted(catalog_dir.glob("**/*.pgm")):
        try:
            prog = parse_pgm(pgm, name=pgm.parent.name if pgm.parent != catalog_dir else pgm.stem)
            if prog.segments:
                programmes.append(prog)
        except Exception:
            continue
    return programmes


def _unit_from_spec(spec: dict, name: str) -> dict | None:
    """Build a cycle-unit dict from a declarative .profile.json spec.

    For product lines whose target is defined by a written spec (a PDF/data
    sheet) rather than a controller .pgm — e.g. a burn-in cycle stated as
    "3 h at -40 C, 5 h at +55 C, >= 5 C/min". The spec declares the one cycle
    directly; segments may give an explicit duration_s or a rate_c_per_min
    (from which duration is computed). Same output shape as extract_cycle_unit.
    """
    raw = spec.get("cycle_segments") or spec.get("cycle") or []
    if not raw:
        return None
    segs: list[dict] = []
    prev = raw[0].get("from_c", spec.get("ambient_c", 20.0))
    for s in raw:
        to_c = float(s["to_c"])
        from_c = float(s.get("from_c", prev))
        if "duration_s" in s:
            dur = float(s["duration_s"])
        else:
            rate = float(s.get("rate_c_per_min", 0) or 0)
            dur = abs(to_c - from_c) / rate * 60.0 if rate > 0 else 0.0
        segs.append({"from_c": round(from_c, 2), "to_c": round(to_c, 2), "duration_s": round(dur, 0)})
        prev = to_c

    tos = [s["to_c"] for s in segs]
    hot_c = spec.get("hot_c", max(tos))
    cold_c = spec.get("cold_c", min(tos))
    heat = [abs(s["to_c"] - s["from_c"]) / (s["duration_s"] / 60.0)
            for s in segs if s["to_c"] > s["from_c"] and s["duration_s"] > 0]
    cool = [abs(s["to_c"] - s["from_c"]) / (s["duration_s"] / 60.0)
            for s in segs if s["to_c"] < s["from_c"] and s["duration_s"] > 0]
    import statistics as _st
    return {
        "name": spec.get("name", name),
        "ambient_c": round(float(spec.get("ambient_c", 20.0)), 1),
        "hot_c": round(float(hot_c), 1),
        "cold_c": round(float(cold_c), 1),
        "heat_rate_c_per_min": round(spec["heat_rate_c_per_min"], 2) if "heat_rate_c_per_min" in spec
            else (round(_st.median(heat), 2) if heat else None),
        "cool_rate_c_per_min": round(spec["cool_rate_c_per_min"], 2) if "cool_rate_c_per_min" in spec
            else (round(_st.median(cool), 2) if cool else None),
        "cycle_duration_s": round(sum(s["duration_s"] for s in segs), 0),
        "cycle_segments": segs,
        "levels": spec.get("levels") or sorted({round(t, 1) for t in tos}),
        "source": spec.get("source", ""),
    }


def load_catalog_units(catalog_dir: str | Path = CATALOG_DIR) -> list[dict]:
    """Every catalog profile as a tileable cycle-unit dict.

    Combines two sources, each named by its product-line folder:
      * `*.pgm`          — controller programmes, parsed and reduced to a cycle.
      * `*.profile.json` — declarative specs for product lines defined by a
                           data sheet rather than a controller file.
    """
    catalog_dir = Path(catalog_dir)
    units: list[dict] = []
    seen: set[str] = set()
    for prog in load_catalog(catalog_dir):
        unit = extract_cycle_unit(prog)
        if unit and unit.get("cycle_segments"):
            units.append(unit)
            seen.add(unit["name"])
    import json as _json
    for spec_file in sorted(catalog_dir.glob("**/*.profile.json")):
        try:
            spec = _json.loads(spec_file.read_text(encoding="utf-8"))
        except Exception:
            continue
        name = spec.get("name") or (spec_file.parent.name if spec_file.parent != catalog_dir else spec_file.stem)
        if name in seen:
            continue
        unit = _unit_from_spec(spec, name)
        if unit and unit.get("cycle_segments"):
            units.append(unit)
            seen.add(name)
    units.sort(key=lambda u: u["name"])
    return units
