"""PDF report generator (M14).

Generates PDF reports from report payload using WeasyPrint.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from config.constants import AuditCategory, AuditSeverity
from models.domain import AuditEntry, AuditLog, ReportPackage


def generate_pdf_report(
    report_package: ReportPackage,
    output_path: str,
    visualisation_bundle: Any = None,
    audit_log: AuditLog | None = None,
) -> str:
    """Generate PDF report from report payload.

    Args:
        report_package: Complete report with all 13 sections
        output_path: Path to write PDF file
        visualisation_bundle: Optional visualisation bundle with Plotly charts
        audit_log: Optional audit log

    Returns:
        Path to generated PDF file
    """
    if audit_log is None:
        audit_log = AuditLog()

    # Always generate HTML with interactive charts
    html_content = _build_html(report_package, visualisation_bundle)
    html_path = str(Path(output_path).with_suffix(".html"))
    Path(html_path).write_text(html_content, encoding="utf-8")
    
    audit_log.add(AuditEntry(
        timestamp=datetime.now(),
        module_name="pdf_generator",
        action="generate_html_report",
        decision="SUCCESS",
        reason=f"HTML report with interactive charts written to {html_path}",
        severity=AuditSeverity.INFO,
        category=AuditCategory.PIPELINE,
    ))

    try:
        from weasyprint import HTML
        # Generate static PDF (charts will be static images)
        HTML(string=html_content).write_pdf(output_path)
        
        audit_log.add(AuditEntry(
            timestamp=datetime.now(),
            module_name="pdf_generator",
            action="generate_pdf_report",
            decision="SUCCESS",
            reason=f"PDF report written to {output_path}",
            severity=AuditSeverity.INFO,
            category=AuditCategory.PIPELINE,
        ))
    except ImportError:
        audit_log.add(AuditEntry(
            timestamp=datetime.now(),
            module_name="pdf_generator",
            action="generate_pdf_report",
            decision="WARNING",
            reason="WeasyPrint not installed — PDF not generated, use HTML instead",
            severity=AuditSeverity.WARNING,
            category=AuditCategory.PIPELINE,
        ))

    return output_path


def _build_html(report_package: ReportPackage, visualisation_bundle: Any = None) -> str:
    """Build HTML string from report sections with embedded Plotly charts."""
    sections = []
    
    # Add visualisation section first if available
    if visualisation_bundle and hasattr(visualisation_bundle, 'charts'):
        sections.append("<h2>Temperature vs Setpoint Visualization</h2>")
        for chart_name, chart_data in visualisation_bundle.charts.items():
            if chart_data:
                # Convert Plotly chart to HTML
                try:
                    import plotly.graph_objects as go
                    if isinstance(chart_data, dict):
                        fig = go.Figure(chart_data)
                        chart_html = fig.to_html(
                            include_plotlyjs='cdn',
                            div_id=f'chart_{chart_name}',
                            config={
                                'scrollZoom': True,          # wheel zooms
                                'displaylogo': False,
                                # Lasso and box-select removed — not useful here.
                                'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
                            },
                        )
                        sections.append(chart_html)
                except Exception as e:
                    sections.append(f"<p>Chart rendering error: {e}</p>")
    
    # Interactive threshold panel sits directly under the trace: shows the
    # thresholds in force (value + where each came from) and lets the reader
    # adjust them, re-running validation client-side in real time.
    interactive_data = report_package.sections.get("interactive_validation_data")
    if interactive_data and interactive_data.get("phases"):
        sections.append(_build_threshold_panel_html(interactive_data))

    # Add other report sections
    for section_name, section_data in report_package.sections.items():
        if section_name in ("visualisation", "interactive_validation_data"):
            continue  # rendered with/beside the chart, not as tables
        title = section_data.get("title", section_name)
        sections.append(f"<h2>{title}</h2>")
        sections.append(_dict_to_html(section_data))

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Ramp Rate Validation Report</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 40px; }}
h1 {{ color: #333; }}
h2 {{ color: #555; border-bottom: 1px solid #ccc; padding-bottom: 5px; margin-top: 30px; }}
table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
th {{ background-color: #f2f2f2; }}
tr:nth-child(even) {{ background-color: #f9f9f9; }}
.plotly-graph-div {{ margin: 20px 0; }}
</style>
</head>
<body>
<h1>Ramp Rate Validation Report</h1>
{''.join(sections)}
</body>
</html>"""


def _build_threshold_panel_html(data: dict[str, Any]) -> str:
    """Adjustable-threshold panel with client-side revalidation.

    Shows the thresholds in force (value + provenance) directly under the
    trace, and re-runs the validation maths in the browser as the reader
    adjusts them. The recorded verdict remains the audited one computed with
    the derived thresholds; the panel is for exploration.
    """
    import json as _json

    thresholds = data.get("thresholds", {})
    bands = data.get("bands", {})
    phases = data.get("phases", [])

    dwell = thresholds.get("dwell_setpoint_deviation", {"value": 2.0, "source": "n/a", "derivation": "n/a"})
    heat = bands.get("heating", {"target": 0.0, "tolerance": 0.0, "tolerance_pct": 20.0, "derivation": "n/a"})
    cool = bands.get("cooling", {"target": 0.0, "tolerance": 0.0, "tolerance_pct": 20.0, "derivation": "n/a"})

    no_setpoint = bool(data.get("no_setpoint"))
    catalog = data.get("catalog", [])
    inferred_setpoint = data.get("inferred_setpoint", [])
    # Per-ramp rows so the ramp-validation table re-renders client-side when
    # the reader changes the target rates (measured stays; target/deviation/
    # result/footer all move).
    ramp_rows = [
        {
            "region_id": p.get("id"),
            "direction": "heating" if p.get("kind") == "heating_ramp" else "cooling",
            "measured": p.get("measured"),
        }
        for p in phases if p.get("kind") in ("heating_ramp", "cooling_ramp")
    ]
    payload = _json.dumps({
        "phases": phases,
        "ramp_rows": ramp_rows,
        "no_setpoint": no_setpoint,
        "catalog": catalog,
        "inferred_setpoint": inferred_setpoint,
        "cycle_spans": data.get("cycle_spans", []),
        "dwell_regions": data.get("dwell_regions", []),
        "trace_series": data.get("trace_series", []),
        "trace_duration_s": data.get("trace_duration_s", 0.0),
        "n_cycles": data.get("n_cycles", 0),
    })

    # Profile picker: a dropdown of catalog profiles next to the target rates.
    # When the data carries no setpoint channel, the reader must pick one for
    # conformance to mean anything; when it does, the picker is an optional
    # overlay for comparison against a reference programme.
    _opts = "".join(
        f"<option value=\"{i}\">{c.get('name','profile')} "
        f"(hot {c.get('hot_c','?')}&deg;, cold {c.get('cold_c','?')}&deg;, "
        f"{c.get('heat_rate_c_per_min','?')}/{c.get('cool_rate_c_per_min','?')} &deg;/min)</option>"
        for i, c in enumerate(catalog)
    )
    inferred_opt = ""
    if inferred_setpoint:
        inferred_opt = ("<option value=\"inferred\">Inferred setpoint "
                        "(recommended for lack of profile)</option>")
    profile_picker = f"""
    <div class=\"rr-ctrl\">
      <b>Target profile</b> <span style=\"color:#777;font-weight:normal;\">(catalog)</span><br>
      <select id=\"rr-profile\" style=\"font-size:14px; max-width:290px;\">
        <option value=\"-1\">— none (use data setpoint) —</option>{inferred_opt}{_opts}
      </select>
      <button id=\"rr-add-profile\" type=\"button\" title=\"Add your own profile (.pgm)\"
        style=\"font-size:16px; font-weight:bold; width:28px; height:28px; margin-left:4px;
        vertical-align:middle; cursor:pointer; border:1px solid #60A8D6; background:#EAF3FA; color:#2471A3; border-radius:4px;\">+</button>
      <input type=\"file\" id=\"rr-profile-file\" accept=\".pgm,.txt\" style=\"display:none;\">
    </div>"""

    banner = ""
    if no_setpoint:
        banner = ("<div id=\"rr-banner\" style=\"border:1px solid #B7770D; background:#FEF6E7; color:#7D5A08; "
                  "padding:10px 14px; margin:8px 0; border-radius:4px; font-weight:bold;\">"
                  "&#9888; This dataset has no setpoint channel &mdash; only achieved temperature was recorded. "
                  "Conformance and ramp-rate targets are undefined until you select a target profile below.</div>")

    def _num(key, val, step):
        return (f"<input type=\"number\" id=\"rr-{key}\" value=\"{val}\" step=\"{step}\" "
                f"min=\"0\" style=\"width:80px; font-size:14px;\">")

    controls = f"""
    <div class=\"rr-ctrl\">
      <b>Dwell setpoint tolerance (°C)</b><br>{_num('dwell', dwell['value'], 0.1)}
    </div>
    <div class=\"rr-ctrl\">
      <b>Heating rate (°C/min)</b><br>
      target {_num('heatT', heat['target'], 0.1)} &plusmn; {_num('heatPct', heat.get('tolerance_pct', 20.0), 1)}%
    </div>
    <div class=\"rr-ctrl\">
      <b>Cooling rate (°C/min)</b><br>
      target {_num('coolT', cool['target'], 0.1)} &plusmn; {_num('coolPct', cool.get('tolerance_pct', 20.0), 1)}%
    </div>"""

    return f"""
<h2>Validation Thresholds (adjustable)</h2>
{banner}
<div id=\"rr-panel\" style=\"border:1px solid #ccc; padding:14px; background:#fafafa;\">
  <div style=\"display:flex; gap:28px; flex-wrap:wrap;\">{profile_picker}{controls}
    <div class=\"rr-ctrl\" style=\"min-width:300px;\">
      <b>Live result</b> <span style=\"font-weight:normal; color:#777;\">(verdict = ramp rates only)</span><br>
      <span id=\"rr-verdict\" style=\"font-size:17px; font-weight:bold;\"></span><br>
      Ramps within rate band: <b id=\"rr-ramps\"></b><br>
      <span style=\"color:#777;\">Setpoint conformance: <b id=\"rr-conf\"></b> &nbsp; Dwells in tolerance: <b id=\"rr-dwells\"></b></span>
    </div>
  </div>
</div>
<style>.rr-ctrl {{ margin-bottom: 6px; }} .rr-src {{ color:#777; font-size:11px; max-width:250px; }}</style>
<script>
(function() {{
  const DATA = {payload};
  const el = id => document.getElementById(id);
  function credit_dwell(measured, tol) {{
    if (tol <= 0) return measured <= 0 ? 1.0 : 0.0;
    // WITHIN tolerance = fully conforming (a steady small offset that stays
    // inside the setpoint band is still on-spec). Credit only tapers once the
    // deviation exceeds the tolerance, reaching 0 at twice the tolerance.
    if (measured <= tol) return 1.0;
    return Math.max(0, 1.0 - (measured - tol) / tol);
  }}
  function credit_band(measured, target, tol) {{
    if (tol <= 0) return measured === target ? 1.0 : 0.0;
    // Full credit at centre, 0.5 at the band edge, 0 at 1.5x tolerance out.
    return Math.max(0, Math.min(1, 1.5 - Math.abs(measured - target) / tol));
  }}

  // ---- Profile overlay (catalog target when data carries no setpoint) ----
  let profileTraceIdx = null;   // Plotly trace index of the overlaid line
  let overshootTraceIdx = null; // Plotly trace index of the overshoot markers
  let devTraceIdx = null;       // Plotly trace index of worst-deviation markers
  let profileSetpoint = null;   // [[t_s, sp_c], ...] tiled setpoint trajectory

  // Per dwell-region evaluation against the selected setpoint: a region is
  // "in tolerance" if its MEDIAN deviation is within tol (the region as a
  // whole, not each sample). Also returns the worst-deviation point per
  // region for chart markers.
  function dwellRegionStats(tol) {{
    const ts = DATA.trace_series || [], regs = DATA.dwell_regions || [];
    const worst = []; let inTol = 0, total = 0;
    for (const span of regs) {{
      const s = span[0], e = span[1];
      let devs = [], wAbs = -1, wPt = null;
      for (const [t, temp] of ts) {{
        if (t < s || t > e) continue;
        const sp = spAt(t); if (sp === null) continue;
        const d = temp - sp; devs.push(Math.abs(d));
        if (Math.abs(d) > wAbs) {{ wAbs = Math.abs(d); wPt = {{t: t, temp: temp, dev: d}}; }}
      }}
      if (!devs.length) continue;
      devs.sort((a, b) => a - b);
      total++;
      if (devs[devs.length >> 1] <= tol) inTol++;
      if (wPt && wAbs > tol) worst.push(wPt);
    }}
    return {{inTol: inTol, total: total, worst: worst}};
  }}

  // Oscillation detection in dwell regions. A dwell that merely drifts is not
  // oscillating; a genuine oscillation crosses the setpoint REPEATEDLY and
  // REGULARLY. So: count sign-changes of the deviation (crossings), require
  // several, then confirm the crossings are evenly spaced (low coefficient of
  // variation of the intervals = periodic) with a meaningful amplitude.
  function detectOscillations(tol) {{
    const ts = DATA.trace_series || [], regs = DATA.dwell_regions || [];
    const found = [];
    for (const span of regs) {{
      const s = span[0], e = span[1];
      const seg = ts.filter(p => p[0] >= s && p[0] <= e);
      if (seg.length < 8) continue;
      const dev = seg.map(p => p[1] - spAt(p[0]));
      const cross = [];   // times where deviation changes sign (crosses setpoint)
      let prev = 0;
      for (let i = 0; i < dev.length; i++) {{
        const sgn = dev[i] > 0.15 * tol ? 1 : (dev[i] < -0.15 * tol ? -1 : 0);
        if (sgn !== 0) {{ if (prev !== 0 && sgn !== prev) cross.push(seg[i][0]); prev = sgn; }}
      }}
      if (cross.length < 3) continue;   // need repeated crossings
      const iv = [];
      for (let i = 1; i < cross.length; i++) iv.push(cross[i] - cross[i - 1]);
      const mean = iv.reduce((a, b) => a + b, 0) / iv.length;
      const sd = Math.sqrt(iv.reduce((a, b) => a + (b - mean) * (b - mean), 0) / iv.length);
      const cv = mean > 0 ? sd / mean : 99;     // regularity: low = periodic
      const amp = Math.max.apply(null, dev.map(Math.abs));
      if (cv < 0.6 && amp > tol) {{
        found.push({{start: s, end: e, mid: (s + e) / 2, crossings: cross.length, period_s: mean * 2, amplitude: amp}});
      }}
    }}
    return found;
  }}

  function renderDeviationMarkers(worst) {{
    if (!window.Plotly) return;
    const gd = document.getElementById('chart_main');
    if (!gd || !gd.data) return;
    if (devTraceIdx !== null) {{ try {{ Plotly.deleteTraces(gd, devTraceIdx); }} catch (e) {{}} devTraceIdx = null; }}
    if (!worst || !worst.length) return;
    Plotly.addTraces(gd, {{
      x: worst.map(w => w.t), y: worst.map(w => w.temp),
      name: 'Worst dwell deviation', mode: 'markers',
      marker: {{symbol: 'x', size: 9, color: '#8E44AD', line: {{color: 'white', width: 1}}}},
      text: worst.map(w => (w.dev >= 0 ? '+' : '') + w.dev.toFixed(2) + '&deg;C from setpoint'),
      hoverinfo: 'text',
    }});
    devTraceIdx = gd.data.length - 1;
  }}

  // Overshoot excursions against the SELECTED setpoint. An overshoot only
  // means something once a target exists, so this runs only when one is
  // picked. Logic: for each setpoint level, once the trace has ARRIVED within
  // tolerance (armed), any later departure beyond the band is an excursion;
  // recovery = time from leaving the band to returning within it. Slew time
  // before first arrival is transit, not overshoot, so it is excluded.
  function computeOvershoots(tol) {{
    const ts = DATA.trace_series || [];
    if (!ts.length || !profileSetpoint || tol <= 0) return [];
    const events = [];
    let armed = false, curSp = null, oshSide = 0;  // +1 = overshoot ABOVE sp, -1 = BELOW
    let inExc = false, excStart = 0, excPeak = 0, excDir = 0, excCount = 0;
    const flush = (tEnd) => {{
      // Keep only REAL overshoots: sustained (>=2 samples) and clearly past
      // the band — filters out single-sample noise blips.
      if (inExc && excCount >= 2 && excPeak > tol) {{
        events.push({{t: excStart, mag: excPeak, recovery: tEnd - excStart, dir: excDir}});
      }}
      inExc = false; excCount = 0;
    }};
    for (const [t, temp] of ts) {{
      const sp = spAt(t);
      if (sp === null) continue;
      if (curSp === null) {{ curSp = sp; }}
      else if (Math.abs(sp - curSp) > 1e-6) {{
        // Setpoint stepped: an overshoot is going PAST the new level in the
        // step's direction (up-step → past = above; down-step → past = below).
        oshSide = sp > curSp ? 1 : -1;
        curSp = sp; armed = false; flush(t);
      }}
      const dev = temp - sp;
      if (!armed) {{ if (Math.abs(dev) <= tol) armed = true; else continue; }}
      // ONLY the overshoot side counts (temperature past the target). Slow
      // approach, droop, or undershoot on the other side is NOT an overshoot.
      const past = (oshSide > 0 && dev > tol) || (oshSide < 0 && dev < -tol);
      if (past) {{
        if (!inExc) {{ inExc = true; excStart = t; excPeak = Math.abs(dev); excDir = oshSide; excCount = 1; }}
        else {{ excCount++; if (Math.abs(dev) > excPeak) excPeak = Math.abs(dev); }}
      }} else if (inExc) {{
        flush(t);
      }}
    }}
    if (inExc) flush(ts[ts.length - 1][0]);
    return events;
  }}

  function _fmtSecs(s) {{ return s >= 60 ? (s / 60).toFixed(1) + ' min' : Math.round(s) + ' s'; }}

  // Re-render the Ramp-Rate Validation Summary table live from the current
  // target rates. Measured slopes are fixed (trace-derived); target,
  // deviation, result and the aggregate footer all update.
  function renderRampTable(hT, hTol, cT, cTol) {{
    const host = el('rr-ramp-live');
    if (!host || !DATA.ramp_rows || !DATA.ramp_rows.length) return;
    const th = 'padding:8px; text-align:center;';
    let body = '', devs = [];
    for (const r of DATA.ramp_rows) {{
      const heating = r.direction === 'heating';
      const T = heating ? hT : cT, tol = heating ? hTol : cTol;
      const dev = r.measured - T;
      devs.push(dev);
      const pass = Math.abs(dev) <= tol;
      const reason = pass
        ? 'within ' + T.toFixed(2) + ' \\u00b1 ' + tol.toFixed(2) + ' \\u00b0C/min'
        : (dev > 0 ? 'exceeds target by ' : 'falls short of target by ') + Math.abs(dev).toFixed(2) + ' \\u00b0C/min';
      const rs = pass ? '' : 'background-color:#FFB6C1;';
      body += '<tr style="' + rs + '">'
        + '<td style="' + th + '">' + (heating ? 'HEATING_RAMP_RATE' : 'COOLING_RAMP_RATE') + '</td>'
        + '<td style="' + th + '">' + r.measured.toFixed(2) + '</td>'
        + '<td style="' + th + '">' + T.toFixed(2) + '</td>'
        + '<td style="' + th + '">' + (dev >= 0 ? '+' : '') + dev.toFixed(2) + '</td>'
        + '<td style="' + th + '">' + (pass ? 'PASS' : 'FAIL') + '</td>'
        + '<td style="' + th + '">' + reason + '</td></tr>';
    }}
    const n = devs.length;
    const meanAbs = n ? devs.reduce((a, d) => a + Math.abs(d), 0) / n : 0;
    const rms = n ? Math.sqrt(devs.reduce((a, d) => a + d * d, 0) / n) : 0;
    const head = '<tr style="background-color:#4472C4; color:white;">'
      + ['Requirement', 'Measured (\\u00b0C/min)', 'Target (\\u00b0C/min)', 'Deviation (\\u00b0C/min)', 'Result', 'Reason']
        .map(h => '<th style="' + th + '">' + h + '</th>').join('') + '</tr>';
    const foot = '<tfoot><tr style="border-top:2px solid #4472C4; background:#EEF2FB;">'
      + '<td style="padding:8px; font-weight:bold;">Deviation across ' + n + ' ramps</td><td></td><td></td>'
      + '<td style="' + th + '"><div><b>\\u00b1' + meanAbs.toFixed(3) + '</b> mean abs</div>'
      + '<div><b>' + rms.toFixed(3) + '</b> RMS</div></td><td></td><td></td></tr></tfoot>';
    host.innerHTML = '<table style="width:100%; border-collapse:collapse; margin:14px 0;">'
      + '<thead>' + head + '</thead><tbody>' + body + '</tbody>' + foot + '</table>'
      + '<div style="color:#777; font-size:11px; max-width:640px;">Aggregate deviation is magnitude '
      + '(opposite-sign misses do not cancel). <b>Mean abs</b> = average distance off target; '
      + '<b>RMS</b> = root-mean-square (weights larger misses more). Recomputed live from the target rates above.</div>';
  }}

  function renderOvershoots(events, oscillations) {{
    // Overshoot + oscillation findings rendered INTO the Overshoot Recovery &
    // Stability report section (not the thresholds panel). Recomputes whenever
    // the tolerance or target changes.
    const host = el('rr-overshoot-report');
    if (host) {{
      const td = 'padding:4px 12px;';
      if (!profileSetpoint) {{
        host.innerHTML = '<p style=\"color:#777;\">Select a target profile (or the inferred setpoint) '
          + 'in the Validation Thresholds panel to assess overshoots against it.</p>';
      }} else {{
        let htmlOut = '';
        if (!events.length) {{
          htmlOut += '<p>No overshoots beyond the setpoint tolerance.</p>';
        }} else {{
          const ordered = events.slice().sort((a, b) => a.t - b.t);
          let rows = '';
          ordered.forEach((e, i) => {{
            const mag = (e.dir > 0 ? '+' : '\\u2212') + e.mag.toFixed(2);
            rows += '<tr><td style=\"' + td + '\">' + (i + 1) + '</td>'
                  + '<td style=\"' + td + '\">' + (e.t / 60.0).toFixed(1) + ' min</td>'
                  + '<td style=\"' + td + '\">' + mag + ' &deg;C</td>'
                  + '<td style=\"' + td + '\">' + _fmtSecs(e.recovery) + '</td></tr>';
          }});
          htmlOut += '<p>' + events.length + ' overshoot(s) beyond the setpoint tolerance. '
            + 'Recovery = time from crossing the band to returning within tolerance of the setpoint.</p>'
            + '<table style=\"border-collapse:collapse; font-size:13px;\">'
            + '<tr style=\"background:#4472C4; color:white;\"><th style=\"' + td + '\">#</th>'
            + '<th style=\"' + td + '\">At</th><th style=\"' + td + '\">Overshoot</th>'
            + '<th style=\"' + td + '\">Recovery</th></tr>' + rows + '</table>';
        }}
        // Oscillation findings (dwell regions crossing the setpoint regularly).
        const osc = oscillations || [];
        if (osc.length) {{
          let orows = '';
          osc.forEach((o, i) => {{
            orows += '<tr><td style=\"' + td + '\">' + (i + 1) + '</td>'
                   + '<td style=\"' + td + '\">' + (o.start / 60.0).toFixed(1) + '\\u2013' + (o.end / 60.0).toFixed(1) + ' min</td>'
                   + '<td style=\"' + td + '\">' + o.crossings + '</td>'
                   + '<td style=\"' + td + '\">' + _fmtSecs(o.period_s) + '</td>'
                   + '<td style=\"' + td + '\">&plusmn;' + o.amplitude.toFixed(2) + ' &deg;C</td></tr>';
          }});
          htmlOut += '<p style=\"margin-top:10px;\"><b>Oscillation</b> detected in ' + osc.length
            + ' dwell region(s) \\u2014 the temperature crosses the setpoint repeatedly and regularly '
            + '(confirmed periodic, not drift).</p>'
            + '<table style=\"border-collapse:collapse; font-size:13px;\">'
            + '<tr style=\"background:#8E44AD; color:white;\"><th style=\"' + td + '\">#</th>'
            + '<th style=\"' + td + '\">Window</th><th style=\"' + td + '\">Crossings</th>'
            + '<th style=\"' + td + '\">~Period</th><th style=\"' + td + '\">Amplitude</th></tr>'
            + orows + '</table>';
        }}
        host.innerHTML = htmlOut;
      }}
    }}
    // Chart flags.
    if (!window.Plotly) return;
    const gd = document.getElementById('chart_main');
    if (!gd || !gd.data) return;
    if (overshootTraceIdx !== null) {{ try {{ Plotly.deleteTraces(gd, overshootTraceIdx); }} catch (e) {{}} overshootTraceIdx = null; }}
    if (!events.length) return;
    Plotly.addTraces(gd, {{
      x: events.map(e => e.t),   // elapsed SECONDS (chart x-axis units)
      y: events.map(e => spAt(e.t) + e.dir * e.mag),
      name: 'Overshoot', mode: 'markers',
      marker: {{symbol: 'triangle-up', size: 11, color: '#C0392B', line: {{color: 'white', width: 1}}}},
      text: events.map(e => (e.dir > 0 ? '+' : '-') + e.mag.toFixed(1) + '&deg;C, recovery ' + _fmtSecs(e.recovery)),
      hoverinfo: 'text',
    }});
    overshootTraceIdx = gd.data.length - 1;
  }}

  // Tile a catalog cycle-unit across the trace's detected cycles at NATIVE
  // rate, anchored to each cycle's [start,end] span, bracketed by ambient.
  function buildProfileSetpoint(prof) {{
    const segs = prof.cycle_segments || [];
    if (!segs.length) return null;
    const amb = prof.ambient_c;
    const dur = DATA.trace_duration_s || 0;
    let spans = DATA.cycle_spans || [];
    // No cycle spans detected (e.g. single-cycle trace): tile once from 0.
    if (!spans.length) spans = [[0, dur]];
    const pts = [];
    const push = (t, v) => {{ if (!pts.length || pts[pts.length-1][0] < t) pts.push([t, v]); else pts.push([t + 1e-3, v]); }};
    push(0, amb);
    push(spans[0][0], amb);                       // hold ambient into first cycle
    for (const [s, e] of spans) {{
      let t = s;
      push(t, segs[0].from_c);
      for (const sg of segs) {{ t += sg.duration_s; push(t, sg.to_c); }}
      if (t < e) push(e, segs[segs.length - 1].to_c);   // hold last level to span end
    }}
    push(spans[spans.length - 1][1], amb);        // return to ambient after last cycle
    push(dur, amb);
    return pts;
  }}

  // Linear interpolation of the tiled setpoint at time t (seconds).
  function spAt(t) {{
    const p = profileSetpoint;
    if (!p || !p.length) return null;
    if (t <= p[0][0]) return p[0][1];
    if (t >= p[p.length-1][0]) return p[p.length-1][1];
    // binary search
    let lo = 0, hi = p.length - 1;
    while (hi - lo > 1) {{ const m = (lo + hi) >> 1; if (p[m][0] <= t) lo = m; else hi = m; }}
    const [t0, v0] = p[lo], [t1, v1] = p[hi];
    return t1 === t0 ? v0 : v0 + (v1 - v0) * (t - t0) / (t1 - t0);
  }}

  // Conformance of the achieved trace against the tiled profile setpoint:
  // time-weighted, deviation-graded credit within the dwell tolerance band.
  function profileConformance(dTol) {{
    const ts = DATA.trace_series || [];
    if (!ts.length || !profileSetpoint) return null;
    let creditW = 0, n = 0, inTol = 0;
    for (const [t, temp] of ts) {{
      const sp = spAt(t);
      if (sp === null) continue;
      const dev = Math.abs(temp - sp);
      creditW += credit_dwell(dev, dTol);
      if (dTol <= 0 || dev <= dTol) inTol++;
      n++;
    }}
    return n ? {{conf: creditW / n * 100, inTol, total: n}} : null;
  }}

  function overlayProfile() {{
    if (!window.Plotly) return;
    const gd = document.getElementById('chart_main');
    if (!gd || !gd.data) return;
    if (profileTraceIdx !== null) {{ try {{ Plotly.deleteTraces(gd, profileTraceIdx); }} catch (e) {{}} profileTraceIdx = null; }}
    if (!profileSetpoint) return;
    Plotly.addTraces(gd, {{
      x: profileSetpoint.map(p => p[0]),   // chart x-axis is elapsed SECONDS
      y: profileSetpoint.map(p => p[1]),
      name: 'Target profile (' + (el('rr-profile').selectedOptions[0].text.split(' (')[0]) + ')',
      mode: 'lines', line: {{color: '#8e44ad', width: 2, dash: 'dash'}},
    }});
    profileTraceIdx = gd.data.length - 1;
  }}

  function onProfileChange() {{
    const raw = el('rr-profile').value;
    const b = el('rr-banner');
    // Inferred setpoint: the trace's own held levels, no tiling, ramp-band
    // centres stay trace-derived.
    if (raw === 'inferred') {{
      profileSetpoint = (DATA.inferred_setpoint || []).slice();
      overlayProfile();
      if (b) {{ b.style.borderColor = '#2471A3'; b.style.background = '#EBF3FA'; b.style.color = '#1B4F72';
        b.innerHTML = '&#8505; Validating against the <b>inferred setpoint</b> &mdash; reconstructed from the levels the trace itself settles at (no external profile).'; }}
      recompute();
      return;
    }}
    const idx = parseInt(raw, 10);
    if (idx < 0 || !DATA.catalog[idx]) {{
      profileSetpoint = null;
      overlayProfile();
      if (b && DATA.no_setpoint) {{ b.style.borderColor = '#B7770D'; b.style.background = '#FEF6E7'; b.style.color = '#7D5A08';
        b.innerHTML = '&#9888; This dataset has no setpoint channel &mdash; only achieved temperature was recorded. Conformance and ramp-rate targets are undefined until you select a target profile below.'; }}
      recompute();
      return;
    }}
    const prof = DATA.catalog[idx];
    profileSetpoint = buildProfileSetpoint(prof);
    // Adopt the profile's native rates as the ramp-band centres.
    if (prof.heat_rate_c_per_min != null) el('rr-heatT').value = prof.heat_rate_c_per_min;
    if (prof.cool_rate_c_per_min != null) el('rr-coolT').value = prof.cool_rate_c_per_min;
    overlayProfile();
    if (b) {{ b.style.borderColor = '#1E8449'; b.style.background = '#EAF7EF'; b.style.color = '#1E6B3B';
      b.innerHTML = '&#10003; Target profile <b>' + prof.name + '</b> applied &mdash; conformance and ramp-rate targets below are computed against it.'; }}
    recompute();
  }}
  function recompute() {{
    const dTol = parseFloat(el('rr-dwell').value) || 0;
    const hT = parseFloat(el('rr-heatT').value) || 0, hPct = parseFloat(el('rr-heatPct').value) || 0;
    const cT = parseFloat(el('rr-coolT').value) || 0, cPct = parseFloat(el('rr-coolPct').value) || 0;
    // Band half-width = percentage of the target rate.
    const hTol = hT * hPct / 100.0, cTol = cT * cPct / 100.0;
    // Conformance: SETPOINT TRACKING ONLY (dwell temperature vs setpoint).
    // Ramp bands do not enter it.
    let creditW = 0, totalW = 0, rampIn = 0, rampTotal = 0, dwellPass = 0, dwellTotal = 0, rampOut = 0;
    for (const p of DATA.phases) {{
      const w = p.weight || 1;
      if (p.kind === 'dwell') {{
        dwellTotal++;
        creditW += w * credit_dwell(p.measured, dTol);
        totalW += w;
        if (dTol <= 0 || p.measured <= dTol) dwellPass++;
      }} else {{
        rampTotal++;
        const T = p.kind === 'heating_ramp' ? hT : cT;
        const tol = p.kind === 'heating_ramp' ? hTol : cTol;
        if (Math.abs(p.measured - T) <= tol) rampIn++; else rampOut++;
      }}
    }}
    let conf = totalW > 0 ? (creditW / totalW * 100) : 0;
    // Conformance needs a REFERENCE to measure against. It is meaningful only
    // when the data carried its own setpoint, OR the reader has picked a
    // target (a catalog profile / custom .pgm / the inferred setpoint). With
    // no setpoint and nothing selected there is nothing to conform TO, so we
    // show N/A rather than a number computed against a self-derived target.
    const hasReference = (!DATA.no_setpoint) || (profileSetpoint != null);
    // When a target is selected it defines the setpoint everywhere, so
    // conformance is measured against it over the whole trace. "Dwells in
    // tolerance" counts whole DWELL REGIONS (median deviation within tol),
    // not individual samples.
    if (profileSetpoint) {{
      const pc = profileConformance(dTol);
      if (pc) conf = pc.conf;
      const ds = dwellRegionStats(dTol);
      dwellPass = ds.inTol; dwellTotal = ds.total;
      renderDeviationMarkers(ds.worst);
    }} else {{
      renderDeviationMarkers([]);
    }}
    // Verdict is driven by RAMP RATES only; conformance is informational.
    let verdict, colour;
    if (rampOut > 0 && rampOut * 2 >= rampTotal) {{ verdict = 'FAIL (systemic ramp deviation)'; colour = '#C0392B'; }}
    else if (rampOut > 0) {{ verdict = 'PASS WITH WARNINGS (isolated off-band ramp)'; colour = '#B7770D'; }}
    else {{ verdict = 'PASS'; colour = '#1E8449'; }}
    el('rr-verdict').textContent = verdict;
    el('rr-verdict').style.color = colour;
    const ec = el('rr-exec-conf');  // conformance figure in Test Summary section
    if (hasReference) {{
      el('rr-conf').textContent = conf.toFixed(1) + '%';
      el('rr-dwells').textContent = dwellPass + '/' + dwellTotal;
      if (ec) ec.textContent = conf.toFixed(1) + '%';
    }} else {{
      el('rr-conf').textContent = 'N/A — select a target';
      el('rr-dwells').textContent = '—';
      if (ec) ec.textContent = 'N/A — select a target profile';
    }}
    el('rr-ramps').textContent = rampIn + '/' + rampTotal;
    // Overshoot + oscillation vs the selected setpoint (only with a target).
    renderOvershoots(
      profileSetpoint ? computeOvershoots(dTol) : [],
      profileSetpoint ? detectOscillations(dTol) : []
    );
    // Live-update the Ramp-Rate Validation Summary table with the new targets.
    renderRampTable(hT, hTol, cT, cTol);
  }}
  // ---- Client-side .pgm parse (mirror of inputs/profile_catalog.py) ----
  function _median(a) {{
    if (!a.length) return null;
    const b = [...a].sort((x, y) => x - y); const m = b.length >> 1;
    return b.length % 2 ? b[m] : (b[m - 1] + b[m]) / 2;
  }}
  function cycleUnitFromSegments(segs, name) {{
    const levelMap = {{}};
    for (const s of segs) if (s.kind === 'dwell' && s.duration_s >= 60) {{
      const k = Math.round(s.to_c); if (!(k in levelMap)) levelMap[k] = s.to_c;
    }}
    const levels = Object.keys(levelMap).map(Number).sort((a, b) => a - b).map(k => levelMap[k]);
    if (!levels.length) return null;
    const hot = Math.max(...levels), cold = Math.min(...levels);
    const room = levels.filter(lv => lv >= 10 && lv <= 32);
    const ambient = room.length ? room[0] : (segs.length ? segs[0].from_c : 20);
    const band = Math.max(2.0, Math.abs(hot - cold) * 0.05);
    const coldIdx = [];
    segs.forEach((s, i) => {{ if (s.kind === 'dwell' && Math.abs(s.to_c - cold) <= band) coldIdx.push(i); }});
    const groupStarts = coldIdx.filter((idx, j) => j === 0 || coldIdx[j - 1] !== idx - 1);
    let cyc = segs;
    if (groupStarts.length >= 2) cyc = segs.slice(groupStarts[0], groupStarts[1]);
    const heat = cyc.filter(s => s.kind === 'ramp' && s.to_c > s.from_c).map(s => Math.abs(s.to_c - s.from_c) / (s.duration_s / 60));
    const cool = cyc.filter(s => s.kind === 'ramp' && s.to_c < s.from_c).map(s => Math.abs(s.to_c - s.from_c) / (s.duration_s / 60));
    const r1 = v => Math.round(v * 10) / 10, r2 = v => Math.round(v * 100) / 100;
    return {{
      name: name, ambient_c: r1(ambient), hot_c: r1(hot), cold_c: r1(cold),
      heat_rate_c_per_min: heat.length ? r2(_median(heat)) : null,
      cool_rate_c_per_min: cool.length ? r2(_median(cool)) : null,
      cycle_duration_s: Math.round(cyc.reduce((a, s) => a + s.duration_s, 0)),
      cycle_segments: cyc.map(s => ({{from_c: r2(s.from_c), to_c: r2(s.to_c), duration_s: Math.round(s.duration_s)}})),
      levels: levels.map(r1),
    }};
  }}
  function parsePgmUnit(text, name) {{
    text = text.replace(/[\\x00-\\x08\\x0b\\x0c\\x0e-\\x1f]/g, ' ');
    const durRe = /^\\d{{1,2}}:\\d{{2}}:\\d{{2}}$/;
    const steps = {{}};
    for (const line of text.split(/\\r?\\n/)) {{
      const toks = line.trim().split(/[,\\s]+/).filter(t => t.length);
      if (toks.length < 4) continue;
      const stepNum = parseInt(toks[0], 10);
      if (!Number.isInteger(stepNum) || String(stepNum) !== toks[0]) continue;
      let durIdx = -1;
      for (let i = 0; i < toks.length; i++) {{ if (durRe.test(toks[i])) {{ durIdx = i; break; }} }}
      if (durIdx < 0) continue;
      const target = parseFloat(toks[1]);
      const p = toks[durIdx].split(':').map(Number);
      const dur = p[0] * 3600 + p[1] * 60 + p[2];
      const nextStep = parseInt(toks[durIdx + 1], 10);
      const loopCount = parseInt(toks[durIdx + 2], 10);
      if ([target, nextStep, loopCount].some(v => Number.isNaN(v))) continue;
      steps[stepNum] = [target, dur, nextStep, loopCount];
    }}
    const keys = Object.keys(steps).map(Number).sort((a, b) => a - b);
    if (!keys.length) return null;
    const start = keys[0];
    let curTemp = steps[start][0];
    let execStep = (start + 1) in steps ? start + 1 : start;
    const loopTaken = {{}}; let budget = 100000; const segs = [];
    while ((execStep in steps) && budget-- > 0) {{
      const [target, dur, nextStep, loopCount] = steps[execStep];
      const kind = Math.abs(target - curTemp) < 1e-6 ? 'dwell' : 'ramp';
      if (dur > 0) segs.push({{kind, from_c: curTemp, to_c: target, duration_s: dur}});
      curTemp = target;
      if (loopCount > 0 && nextStep <= execStep) {{
        const taken = loopTaken[execStep] || 0;
        if (taken < loopCount) {{ loopTaken[execStep] = taken + 1; execStep = nextStep; continue; }}
        loopTaken[execStep] = 0; execStep = execStep + 1; continue;
      }}
      if (nextStep && nextStep !== execStep) execStep = nextStep; else execStep = execStep + 1;
      if (execStep === start) break;
    }}
    return cycleUnitFromSegments(segs, name);
  }}

  for (const id of ['rr-dwell','rr-heatT','rr-heatPct','rr-coolT','rr-coolPct']) el(id).addEventListener('input', recompute);
  const sel = el('rr-profile');
  if (sel) sel.addEventListener('change', onProfileChange);

  // "+" adds the reader's own .pgm to the catalog, parsed in the browser.
  const addBtn = el('rr-add-profile'), fileInput = el('rr-profile-file');
  if (addBtn && fileInput && sel) {{
    addBtn.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', ev => {{
      const f = ev.target.files[0]; if (!f) return;
      const reader = new FileReader();
      reader.onload = () => {{
        const name = f.name.replace(/\\.[^.]+$/, '') + ' (custom)';
        let unit = null;
        try {{ unit = parsePgmUnit(String(reader.result), name); }} catch (e) {{ unit = null; }}
        if (!unit || !unit.cycle_segments || !unit.cycle_segments.length) {{
          alert('Could not parse a chamber programme (.pgm) from that file.'); return;
        }}
        DATA.catalog.push(unit);
        const idx = DATA.catalog.length - 1;
        const opt = document.createElement('option');
        opt.value = String(idx);
        opt.text = unit.name + ' (hot ' + unit.hot_c + '\\u00b0, cold ' + unit.cold_c + '\\u00b0, '
          + unit.heat_rate_c_per_min + '/' + unit.cool_rate_c_per_min + ' \\u00b0/min)';
        sel.appendChild(opt); sel.value = String(idx);
        onProfileChange();
      }};
      reader.readAsText(f);
    }});
  }}
  recompute();
  // The ramp table sits later in the document than this script, so re-run
  // once the whole DOM is present to populate it (and the overshoot chart).
  if (document.readyState === 'complete') recompute();
  else window.addEventListener('load', recompute);
}})();
</script>
"""


def _build_phase_table_html(data: dict[str, Any]) -> str:
    """Build HTML table for phase analysis with proper formatting and anomaly highlighting."""
    phases = data.get("phases", [])
    if not phases:
        return "<p>No phases to display</p>"
    
    # Build table header
    html = ['<table style="width:100%; border-collapse: collapse; margin: 20px 0;">']
    html.append('<thead><tr style="background-color: #4472C4; color: white;">')
    html.append('<th style="padding: 8px; text-align: center;">#</th>')
    html.append('<th style="padding: 8px; text-align: center;">Type</th>')
    html.append('<th style="padding: 8px; text-align: center;">SP Range</th>')
    html.append('<th style="padding: 8px; text-align: center;">Rate (°C/min)</th>')
    html.append('<th style="padding: 8px; text-align: center;">Avg Temp (°C)</th>')
    html.append('<th style="padding: 8px; text-align: center;">Duration</th>')
    html.append('<th style="padding: 8px; text-align: center;">Max Dev</th>')
    html.append('<th style="padding: 8px; text-align: center;">Status</th>')
    html.append('</tr></thead>')
    
    # Build table body
    html.append('<tbody>')
    for phase in phases:
        status = phase.get("status", "OK")
        # Highlight anomaly rows with pink/red background
        row_style = 'background-color: #FFB6C1;' if status == "ANOMALY" else ''
        
        html.append(f'<tr style="{row_style}">')
        html.append(f'<td style="padding: 8px; text-align: center;">{phase.get("phase_number", "-")}</td>')
        html.append(f'<td style="padding: 8px; text-align: center;">{phase.get("type", "-")}</td>')
        html.append(f'<td style="padding: 8px; text-align: center;">{phase.get("sp_range", "-")}</td>')
        
        # Rate - only show for ramps
        rate = phase.get("rate_c_per_min")
        html.append(f'<td style="padding: 8px; text-align: center;">{rate if rate is not None else "-"}</td>')
        
        # Avg Temp - only show for dwells
        avg_temp = phase.get("avg_temp_c")
        html.append(f'<td style="padding: 8px; text-align: center;">{avg_temp if avg_temp is not None else "-"}</td>')
        
        # Duration in minutes
        duration = phase.get("duration_min", 0)
        html.append(f'<td style="padding: 8px; text-align: center;">{duration:.1f} min</td>')
        
        # Max deviation
        max_dev = phase.get("max_dev_c")
        html.append(f'<td style="padding: 8px; text-align: center;">{max_dev if max_dev is not None else "-"}</td>')
        
        # Status
        html.append(f'<td style="padding: 8px; text-align: center; font-weight: bold;">{status}</td>')
        html.append('</tr>')
    
    html.append('</tbody></table>')
    
    # Add summary info
    if "subtitle" in data:
        html.insert(0, f'<p style="font-style: italic; color: #666;">{data["subtitle"]}</p>')
    if "total_phases" in data:
        html.append(f'<p><strong>Total Phases:</strong> {data["total_phases"]}</p>')
    
    return ''.join(html)


_RAMP_COL_LABELS = {
    "requirement_id": "Requirement",
    "measured_value": "Measured (°C/min)",
    "threshold_value": "Target (°C/min)",
    "deviation_c_per_min": "Deviation (°C/min)",
    "result": "Result",
    "reason": "Reason",
}


def _build_ramp_results_table_html(data: dict[str, Any]) -> str:
    """Ramp-rate validation table with a deviation-summary footer.

    The footer sits under the Deviation column and reports how far the ramps
    sit from target IN AGGREGATE — as magnitudes, so opposite-sign deviations
    do not cancel.
    """
    results = data.get("ramp_results", [])
    if not results:
        return "<p>No ramp-rate results to display.</p>"
    cols = list(results[0].keys())

    def _fmt(key, val):
        if val is None:
            return "-"
        if key in ("measured_value", "threshold_value") and isinstance(val, (int, float)):
            return f"{val:.2f}"
        if key == "deviation_c_per_min" and isinstance(val, (int, float)):
            return f"{val:+.2f}"  # keep the sign visible
        return str(val)

    # Wrapped in a live container so the panel JS can re-render it when the
    # reader changes the heating/cooling target rates (measured stays; target,
    # deviation, result and the aggregate footer all move).
    html = ['<div id="rr-ramp-live">']
    html.append('<table style="width:100%; border-collapse:collapse; margin:14px 0;">')
    html.append('<thead><tr style="background-color:#4472C4; color:white;">')
    for c in cols:
        html.append(f'<th style="padding:8px; text-align:center;">{_RAMP_COL_LABELS.get(c, c)}</th>')
    html.append("</tr></thead><tbody>")
    for row in results:
        rowstyle = "background-color:#FFB6C1;" if str(row.get("result")) not in ("PASS", "PASS_WITH_WARNINGS") else ""
        html.append(f'<tr style="{rowstyle}">')
        for c in cols:
            html.append(f'<td style="padding:8px; text-align:center;">{_fmt(c, row.get(c))}</td>')
        html.append("</tr>")
    html.append("</tbody>")

    # Footer: aggregate deviation, aligned under the Deviation column.
    summary = data.get("deviation_summary")
    if summary and "deviation_c_per_min" in cols:
        dev_idx = cols.index("deviation_c_per_min")
        mean_abs = summary.get("mean_abs_deviation_c_per_min")
        rms = summary.get("rms_deviation_c_per_min")
        n = summary.get("n_ramps")
        cell = (
            f'<div><b>&plusmn;{mean_abs:.3f}</b> mean abs</div>'
            f'<div><b>{rms:.3f}</b> RMS</div>'
        )
        html.append('<tfoot><tr style="border-top:2px solid #4472C4; background:#EEF2FB;">')
        for i, c in enumerate(cols):
            if i == 0:
                html.append(f'<td style="padding:8px; font-weight:bold;">Deviation across {n} ramps</td>')
            elif i == dev_idx:
                html.append(f'<td style="padding:8px; text-align:center;">{cell}</td>')
            else:
                html.append('<td></td>')
        html.append("</tr></tfoot>")
    html.append("</table>")
    html.append(
        '<div style="color:#777; font-size:11px; max-width:640px;">'
        "Aggregate deviation is reported as magnitude so opposite-sign misses do not cancel. "
        "<b>Mean abs</b> = average distance each ramp sits from its target rate. "
        "<b>RMS</b> = root-mean-square deviation (variance-based; weights larger misses more heavily).</div>"
    )
    html.append("</div>")
    return "".join(html)


def _dict_to_html(data: dict[str, Any], indent: int = 0) -> str:
    """Convert a dict to HTML table rows with special handling for phase analysis table."""
    # Special handling for phase analysis table
    if "phases" in data and isinstance(data.get("phases"), list):
        return _build_phase_table_html(data)
    # Ramp-rate validation table with a deviation-summary footer.
    if "ramp_results" in data and isinstance(data.get("ramp_results"), list) and data.get("ramp_results"):
        return _build_ramp_results_table_html(data)
    # Overshoot Recovery & Stability: wrap in a live container so the panel JS
    # can fill it with per-overshoot recovery times against the selected target.
    if data.get("title") == "Overshoot Recovery & Stability":
        inner = _dict_to_html({k: v for k, v in data.items() if k != "title"})
        return f'<div id="rr-overshoot-report">{inner}</div>'

    rows = []
    for key, value in data.items():
        if isinstance(value, dict):
            rows.append(f"<tr><td><strong>{key}</strong></td><td>{_dict_to_html(value)}</td></tr>")
        elif isinstance(value, list):
            if value and isinstance(value[0], dict):
                # List of dicts → table
                sub_headers = list(value[0].keys())
                sub_rows = []
                for item in value:
                    sub_rows.append("<tr>" + "".join(f"<td>{item.get(h, '')}</td>" for h in sub_headers) + "</tr>")
                table = f"<table><tr>{''.join(f'<th>{h}</th>' for h in sub_headers)}</tr>{''.join(sub_rows)}</table>"
                rows.append(f"<tr><td><strong>{key}</strong></td><td>{table}</td></tr>")
            else:
                rows.append(f"<tr><td><strong>{key}</strong></td><td>{', '.join(str(v) for v in value)}</td></tr>")
        else:
            rows.append(f"<tr><td><strong>{key}</strong></td><td>{value}</td></tr>")

    return f"<table>{''.join(rows)}</table>"
