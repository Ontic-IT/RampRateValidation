"""ValidationProfile model with 12 required sub-objects."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from config.constants import AmbiguityHandling, FirstExtreme


class ProfileMetadata(BaseModel):
    """Profile identification and version metadata."""
    profile_name: str
    profile_version: str
    created_date: datetime
    description: str
    algorithm_version_required: str


class ExpectedProcessSequence(BaseModel):
    """Expected process sequence configuration."""
    first_extreme: FirstExtreme = FirstExtreme.HOT_FIRST
    expected_end_condition: str = ""
    expected_region_sequence: list[str] = Field(default_factory=list)
    hot_cycle_count_min: int | None = None
    hot_cycle_count_max: int | None = None


class RampRateRequirements(BaseModel):
    """Ramp rate compliance requirements."""
    required_heating_ramp_rate_c_per_min: float
    required_cooling_ramp_rate_c_per_min: float
    minimum_sustained_ramp_rate_ratio: float = 0.8
    sustained_ramp_window_seconds: float = 60.0
    minimum_valid_ramp_duration_seconds: float = 30.0
    allowed_ramp_deviation_c: float | None = None
    tolerance_source: Literal['EXPLICIT', 'ADAPTIVE'] = 'EXPLICIT'


class ToleranceResolution(BaseModel):
    """Audit-grade record of which tolerance value was used and why.

    Never computes the unused alternative; records only the chosen value.
    """
    parameter_name: str
    resolved_value: float
    source: Literal['EXPLICIT_USER', 'EXPLICIT_PROFILE', 'ADAPTIVE_DERIVED']
    explicit_value_provided: float | None = None
    adaptive_value_skipped: bool = True
    derivation_method: str | None = None


class DwellRequirements(BaseModel):
    """Dwell compliance requirements."""
    minimum_hot_dwell_seconds: float = 0.0  # 0.0 = report measured duration from classified dwells
    minimum_cold_dwell_seconds: float = 0.0  # 0.0 = report measured duration from classified dwells
    allowed_setpoint_deviation_c: float | None = None
    dwell_stability_window_seconds: float = 30.0
    tolerance_source: Literal['EXPLICIT', 'ADAPTIVE'] = 'EXPLICIT'


class SetpointDeviationRequirements(BaseModel):
    """Setpoint deviation requirements."""
    allowed_setpoint_deviation_c: float | None = None
    setpoint_deviation_warning_c: float | None = None


class OvershootRequirements(BaseModel):
    """Overshoot requirements."""
    overshoot_warning_threshold_c: float
    overshoot_failure_threshold_c: float
    max_overshoot_duration_seconds: float


class SettlingRequirements(BaseModel):
    """Settling time requirements."""
    settling_time_limit_seconds: float
    settling_tolerance_band_c: float


class DataQualityRequirements(BaseModel):
    """Data quality requirements."""
    max_missing_data_pct: float = 5.0
    max_duplicate_timestamp_pct: float = 1.0
    max_gap_seconds: float = 30.0
    min_process_duration_seconds: float = 60.0
    max_spike_count: int = 10


class ClassificationSettings(BaseModel):
    """Classification tuning settings."""
    slope_threshold_min_c_per_min: float = 0.1
    slope_threshold_max_c_per_min: float = 50.0
    MAD_threshold_min_c: float = 0.01
    MAD_threshold_max_c: float = 10.0
    min_region_duration_seconds: float = 5.0

    # Evidence fusion settings
    secondary_classification_threshold: float = 0.5
    ambiguity_margin_threshold: float = 0.1
    high_confidence_threshold: float = 0.8
    medium_confidence_threshold: float = 0.6
    ambiguity_handling: AmbiguityHandling = AmbiguityHandling.WARN


class CycleRules(BaseModel):
    """Cycle validation rules."""
    allow_partial_cycle_validation: bool = False
    minimum_complete_cycles_required: int = 1
    cycle_sequence_strict: bool = True


class VisualisationSettings(BaseModel):
    """Visualisation configuration."""
    show_analysis_signal: bool = True
    region_colour_map: dict[str, str] = Field(default_factory=dict)
    annotation_font_size: int = 10
    export_format: str = "html"


class ReportingSettings(BaseModel):
    """Reporting configuration."""
    include_audit_trail: bool = True
    include_algorithm_appendix: bool = True
    export_pdf: bool = True
    export_excel: bool = True
    report_title: str = "Ramp Rate Validation Report"


class ValidationProfile(BaseModel):
    """Complete validation profile with 12 required sub-objects."""
    profile_metadata: ProfileMetadata
    expected_process_sequence: ExpectedProcessSequence
    ramp_rate_requirements: RampRateRequirements
    dwell_requirements: DwellRequirements
    setpoint_deviation_requirements: SetpointDeviationRequirements
    overshoot_requirements: OvershootRequirements
    settling_requirements: SettlingRequirements
    data_quality_requirements: DataQualityRequirements = Field(
        default_factory=DataQualityRequirements
    )
    classification_settings: ClassificationSettings = Field(
        default_factory=ClassificationSettings
    )
    cycle_rules: CycleRules = Field(default_factory=CycleRules)
    visualisation_settings: VisualisationSettings = Field(
        default_factory=VisualisationSettings
    )
    reporting_settings: ReportingSettings = Field(
        default_factory=ReportingSettings
    )
    tolerance_resolutions: list[ToleranceResolution] = Field(default_factory=list)

    def get_explicit_tolerance(self, parameter_name: str) -> float | None:
        """Return the explicit tolerance value for a parameter, or None.

        Resolution order:
        - dwell_setpoint_deviation: dwell_requirements first, then setpoint_deviation_requirements
        - ramp_deviation: ramp_rate_requirements
        """
        if parameter_name == "dwell_setpoint_deviation":
            if self.dwell_requirements.allowed_setpoint_deviation_c is not None:
                return self.dwell_requirements.allowed_setpoint_deviation_c
            return self.setpoint_deviation_requirements.allowed_setpoint_deviation_c
        elif parameter_name == "ramp_deviation":
            return self.ramp_rate_requirements.allowed_ramp_deviation_c
        return None
