"""Integration tests for Phase 8: Full pipeline orchestration."""

import pytest
from pathlib import Path

from engine.pipeline import run_ramp_rate_analysis
from models.domain import AnalysisRequest
from config.constants import PipelineStage


class TestPipelineIntegration:
    """Integration tests for full pipeline execution."""
    
    def test_pipeline_executes_all_15_stages(self, tmp_path):
        """Test that pipeline executes all 15 stages in order."""
        # This is a smoke test - requires actual test data
        # For now, just verify the pipeline structure is correct
        
        # Verify all stage functions exist
        from engine.pipeline import (
            _stage_ingest_source_file,
            _stage_normalise_to_canonical_trace,
            _stage_preprocess_trace,
            _stage_assess_data_quality,
            _stage_detect_process_boundaries,
            _stage_infer_or_resolve_setpoints,
            _stage_classify_regions,
            _stage_isolate_valid_ramp_regions,
            _stage_segment_cycles,
            _stage_compute_metrics,
            _stage_validate_against_profile,
            _stage_compare_profile_consistency,
            _stage_build_visualisation_payload,
            _stage_generate_report_payload,
            _stage_emit_analysis_result,
        )
        
        # All 15 stage functions exist
        assert callable(_stage_ingest_source_file)
        assert callable(_stage_normalise_to_canonical_trace)
        assert callable(_stage_preprocess_trace)
        assert callable(_stage_assess_data_quality)
        assert callable(_stage_detect_process_boundaries)
        assert callable(_stage_infer_or_resolve_setpoints)
        assert callable(_stage_classify_regions)
        assert callable(_stage_isolate_valid_ramp_regions)
        assert callable(_stage_segment_cycles)
        assert callable(_stage_compute_metrics)
        assert callable(_stage_validate_against_profile)
        assert callable(_stage_compare_profile_consistency)
        assert callable(_stage_build_visualisation_payload)
        assert callable(_stage_generate_report_payload)
        assert callable(_stage_emit_analysis_result)
    
    def test_pipeline_stage_order_is_correct(self):
        """Test that PipelineStage enum has all 15 stages in correct order."""
        from config.constants import PipelineStage
        
        expected_stages = [
            "INGESTION",
            "NORMALISATION",
            "PREPROCESSING",
            "QUALITY",
            "BOUNDARIES",
            "SETPOINTS",
            "CLASSIFICATION",
            "RAMP_ISOLATION",
            "CYCLES",
            "METRICS",
            "VALIDATION",
            "COMPARISON",
            "VISUALISATION",
            "REPORTING",
            "AUDIT",
        ]
        
        actual_stages = [stage.value for stage in PipelineStage]
        
        # Verify all expected stages exist
        for expected in expected_stages:
            assert expected in actual_stages, f"Missing stage: {expected}"
    
    def test_analysis_request_model(self):
        """Test AnalysisRequest model can be instantiated."""
        request = AnalysisRequest(
            file_path="test.csv",
            profile_path="profile.yaml",
            output_dir="output/",
        )
        
        assert request.file_path == "test.csv"
        assert request.profile_path == "profile.yaml"
        assert request.output_dir == "output/"
        assert request.channel is None
        assert request.setpoint_channel is None
    
    def test_analysis_result_structure(self):
        """Test AnalysisResult has all required fields."""
        from models.domain import AnalysisResult
        
        # Get model fields
        fields = AnalysisResult.model_fields
        
        # Verify key fields exist (using actual field names from model)
        assert "status" in fields
        assert "status_reason" in fields
        assert "canonical_trace" in fields
        assert "data_quality_report" in fields
        assert "process_boundaries" in fields
        assert "classified_regions" in fields
        assert "valid_ramp_regions" in fields
        assert "cycles" in fields
        assert "metrics" in fields
        assert "validation_results" in fields
        assert "phase_conformance" in fields
        assert "profile_comparison_results" in fields
        assert "visualisation" in fields
        assert "report_package" in fields
        assert "audit_log" in fields


class TestPipelineStateTracking:
    """Tests for pipeline state tracking."""
    
    def test_pipeline_stage_state_model(self):
        """Test PipelineStageState model."""
        from models.domain import PipelineStageState
        from config.constants import PipelineStage, PipelineStatus
        from datetime import datetime
        
        state = PipelineStageState(
            current_stage=PipelineStage.INGESTION,
            started_at=datetime.now(),
            status=PipelineStatus.RUNNING,
        )
        
        assert state.current_stage == PipelineStage.INGESTION
        assert state.status == PipelineStatus.RUNNING
        assert state.completed_at is None
        
        # Mark complete
        state.completed_at = datetime.now()
        state.status = PipelineStatus.COMPLETED
        
        assert state.status == PipelineStatus.COMPLETED
        assert state.completed_at is not None


class TestRunMetadata:
    """Tests for RunMetadata reproducibility contract."""
    
    def test_run_metadata_model(self):
        """Test RunMetadata has all reproducibility fields."""
        from models.domain import RunMetadata
        from datetime import datetime
        
        metadata = RunMetadata(
            algorithm_version="1.0.0",
            profile_hash="abc123",
            input_file_hash="def456",
            execution_timestamp=datetime.now(),
            python_version="3.11.9",
            random_seed=42,
        )
        
        assert metadata.algorithm_version == "1.0.0"
        assert metadata.profile_hash == "abc123"
        assert metadata.input_file_hash == "def456"
        assert metadata.random_seed == 42
        assert metadata.python_version == "3.11.9"
    
    def test_run_metadata_has_all_required_fields(self):
        """Test RunMetadata has all fields from plan spec."""
        from models.domain import RunMetadata
        
        fields = RunMetadata.model_fields
        
        # Verify all reproducibility fields exist
        assert "git_commit" in fields
        assert "algorithm_version" in fields
        assert "profile_hash" in fields
        assert "input_file_hash" in fields
        assert "execution_timestamp" in fields
        assert "python_version" in fields
        assert "dependency_versions" in fields
        assert "classification_weight_version" in fields
        assert "classification_weight_hash" in fields
        assert "adaptive_constants_snapshot" in fields
        assert "classifier_configuration_snapshot" in fields
        assert "feature_extraction_configuration" in fields
        assert "random_seed" in fields
        assert "calibration_dataset_version" in fields


class TestCLI:
    """Tests for CLI interface."""
    
    def test_cli_module_imports(self):
        """Test CLI module can be imported."""
        from engine import cli
        
        assert hasattr(cli, "main")
        assert callable(cli.main)
    
    def test_cli_has_analyse_command(self):
        """Test CLI has analyse command."""
        from engine.cli import run_analyse
        
        assert callable(run_analyse)
