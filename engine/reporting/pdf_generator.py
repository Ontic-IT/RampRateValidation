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
                        chart_html = fig.to_html(include_plotlyjs='cdn', div_id=f'chart_{chart_name}')
                        sections.append(chart_html)
                except Exception as e:
                    sections.append(f"<p>Chart rendering error: {e}</p>")
    
    # Add other report sections
    for section_name, section_data in report_package.sections.items():
        if section_name == "visualisation":  # Skip the placeholder visualisation section
            continue
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


def _dict_to_html(data: dict[str, Any], indent: int = 0) -> str:
    """Convert a dict to HTML table rows with special handling for phase analysis table."""
    # Special handling for phase analysis table
    if "phases" in data and isinstance(data.get("phases"), list):
        return _build_phase_table_html(data)
    
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
