"""Command-line interface for ramp rate validation tool (Phase 8)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from engine.pipeline import run_ramp_rate_analysis
from engine.reporting.pdf_generator import generate_pdf_report
from engine.reporting.excel_generator import generate_excel_report
from models.domain import AnalysisRequest


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Ramp Rate Validation Tool - Thermal chamber test analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic analysis
  ramp-rate-tool analyse --file data.csv --profile profile.yaml --output results/
  
  # Specify channels
  ramp-rate-tool analyse --file data.csv --profile profile.yaml --output results/ \\
      --channel TC1 --setpoint-channel SP1
  
  # Export formats
  ramp-rate-tool analyse --file data.csv --profile profile.yaml --output results/ \\
      --pdf --excel --json
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Analyse command
    analyse_parser = subparsers.add_parser("analyse", help="Run ramp rate analysis")
    analyse_parser.add_argument(
        "--file",
        required=True,
        help="Path to input CSV/log file"
    )
    analyse_parser.add_argument(
        "--profile",
        required=True,
        help="Path to validation profile YAML"
    )
    analyse_parser.add_argument(
        "--output",
        required=True,
        help="Output directory for results"
    )
    analyse_parser.add_argument(
        "--channel",
        default=None,
        help="Temperature channel name (auto-detected if not specified)"
    )
    analyse_parser.add_argument(
        "--setpoint-channel",
        default=None,
        help="Setpoint channel name (optional)"
    )
    analyse_parser.add_argument(
        "--pdf",
        action="store_true",
        help="Generate PDF report"
    )
    analyse_parser.add_argument(
        "--excel",
        action="store_true",
        help="Generate Excel report"
    )
    analyse_parser.add_argument(
        "--json",
        action="store_true",
        help="Export results as JSON"
    )
    analyse_parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    if args.command == "analyse":
        run_analyse(args)


def run_analyse(args):
    """Run analysis command."""
    print(f"Ramp Rate Validation Tool")
    print(f"=" * 60)
    print(f"Input file: {args.file}")
    print(f"Profile: {args.profile}")
    print(f"Output: {args.output}")
    print(f"=" * 60)
    
    # Validate inputs
    if not Path(args.file).exists():
        print(f"ERROR: Input file not found: {args.file}")
        sys.exit(1)
    
    if not Path(args.profile).exists():
        print(f"ERROR: Profile file not found: {args.profile}")
        sys.exit(1)
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Build request
    request = AnalysisRequest(
        file_path=args.file,
        profile_path=args.profile,
        output_dir=str(output_dir),
        channel=args.channel,
        setpoint_channel=args.setpoint_channel,
    )
    
    # Run analysis
    print("\nRunning analysis pipeline...")
    try:
        result = run_ramp_rate_analysis(request)
    except Exception as e:
        print(f"\nERROR: Analysis failed: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)
    
    print("✓ Analysis complete")
    
    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    
    if result.overall_validation_status:
        print(f"Overall Status: {result.overall_validation_status.status.value}")
        print(f"Reason: {result.overall_validation_status.reason}")
    
    if result.phase_conformance:
        print(f"\nConformance: {result.phase_conformance.conformance_percentage:.1f}%")
        print(f"Phases Passed: {result.phase_conformance.passed_phases}/{result.phase_conformance.total_phases}")
        if result.phase_conformance.anomaly_phases > 0:
            print(f"Anomalies: {result.phase_conformance.anomaly_phases} phases flagged")
    
    if result.cycles:
        print(f"\nCycles Detected: {len(result.cycles.cycles)}")
    
    if result.region_list:
        print(f"Regions Classified: {len(result.region_list.regions)}")
    
    # Export outputs
    print("\n" + "=" * 60)
    print("EXPORTING OUTPUTS")
    print("=" * 60)
    
    # Always export JSON
    json_path = output_dir / "analysis_result.json"
    print(f"Writing JSON: {json_path}")
    _export_json(result, json_path)
    
    # PDF report
    if args.pdf or not (args.excel or args.json):  # Default to PDF if no format specified
        if result.report_package:
            pdf_path = output_dir / "report.pdf"
            print(f"Writing PDF: {pdf_path}")
            try:
                generate_pdf_report(result.report_package, str(pdf_path), result.audit_log)
            except Exception as e:
                print(f"WARNING: PDF generation failed: {str(e)}")
    
    # Excel report
    if args.excel:
        if result.report_package:
            excel_path = output_dir / "report.xlsx"
            print(f"Writing Excel: {excel_path}")
            try:
                generate_excel_report(result.report_package, str(excel_path), result.audit_log)
            except Exception as e:
                print(f"WARNING: Excel generation failed: {str(e)}")
    
    # Audit log
    audit_path = output_dir / "audit_log.json"
    print(f"Writing audit log: {audit_path}")
    _export_audit_log(result.audit_log, audit_path)
    
    print("\n" + "=" * 60)
    print("✓ Analysis complete - outputs written to:", output_dir)
    print("=" * 60)


def _export_json(result, output_path: Path):
    """Export AnalysisResult as JSON."""
    # Convert to dict
    result_dict = result.model_dump(mode="json")
    
    # Write JSON
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result_dict, f, indent=2, default=str)


def _export_audit_log(audit_log, output_path: Path):
    """Export audit log as JSON."""
    entries = []
    for entry in audit_log.entries:
        entries.append({
            "timestamp": entry.timestamp.isoformat() if entry.timestamp else None,
            "module_name": entry.module_name,
            "action": entry.action,
            "decision": entry.decision,
            "reason": entry.reason,
            "severity": entry.severity.value,
            "category": entry.category.value,
        })
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"entries": entries, "total": len(entries)}, f, indent=2)


if __name__ == "__main__":
    main()
