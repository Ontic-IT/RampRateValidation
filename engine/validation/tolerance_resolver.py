"""Tolerance precedence resolver (Retrofit 4).

Sits in front of validation rules. For each tolerance-bearing parameter:
- Explicit profile value → EXPLICIT_PROFILE (adaptive NEVER runs)
- No explicit value → adaptive derivation → ADAPTIVE_DERIVED

Guardrail: adaptive derivation is never invoked when explicit value exists.
"""

from __future__ import annotations

from datetime import datetime

from config.constants import AuditCategory, AuditSeverity
from models.domain import AuditEntry, AuditLog
from models.profile import ValidationProfile, ToleranceResolution


class AdaptiveConstants:
    """Container for adaptively-derived tolerance values.

    Maps parameter_name -> derived_value and derivation_method.
    """

    def __init__(
        self,
        constants: dict[str, float] | None = None,
        derivation_methods: dict[str, str] | None = None,
    ):
        self._constants = constants or {}
        self._derivation_methods = derivation_methods or {}

    def get(self, parameter_name: str) -> float | None:
        return self._constants.get(parameter_name)

    def get_derivation_method(self, parameter_name: str) -> str | None:
        return self._derivation_methods.get(parameter_name)


def resolve_tolerance(
    parameter_name: str,
    profile: ValidationProfile,
    adaptive_constants: AdaptiveConstants | None = None,
    audit_log: AuditLog | None = None,
) -> ToleranceResolution:
    """Resolve tolerance value with explicit → adaptive precedence.

    Args:
        parameter_name: e.g. "dwell_setpoint_deviation", "ramp_deviation"
        profile: Validation profile
        adaptive_constants: Adaptive-derived values (only used if no explicit)
        audit_log: Optional audit log

    Returns:
        ToleranceResolution documenting the chosen value and source.
    """
    if audit_log is None:
        audit_log = AuditLog()

    explicit_value = profile.get_explicit_tolerance(parameter_name)

    if explicit_value is not None:
        resolution = ToleranceResolution(
            parameter_name=parameter_name,
            resolved_value=explicit_value,
            source="EXPLICIT_PROFILE",
            explicit_value_provided=explicit_value,
            adaptive_value_skipped=True,
            derivation_method=None,
        )
        reason = f"Explicit profile value {explicit_value} used for {parameter_name}"
    else:
        if adaptive_constants is None:
            raise ValueError(
                f"No explicit tolerance for '{parameter_name}' and "
                "adaptive_constants not provided"
            )
        derived_value = adaptive_constants.get(parameter_name)
        if derived_value is None:
            raise ValueError(
                f"No explicit or adaptive tolerance for '{parameter_name}'"
            )
        derivation_method = adaptive_constants.get_derivation_method(parameter_name)
        resolution = ToleranceResolution(
            parameter_name=parameter_name,
            resolved_value=derived_value,
            source="ADAPTIVE_DERIVED",
            explicit_value_provided=None,
            adaptive_value_skipped=False,
            derivation_method=derivation_method,
        )
        reason = f"Adaptive value {derived_value} derived for {parameter_name}"
        if derivation_method:
            reason += f" via {derivation_method}"

    audit_log.add(AuditEntry(
        timestamp=datetime.now(),
        module_name="tolerance_resolver",
        action="TOLERANCE_RESOLVED",
        input_reference=parameter_name,
        output_reference=resolution.source,
        decision=resolution.source,
        reason=reason,
        thresholds_used={
            "resolved_value": resolution.resolved_value,
            "explicit_value": explicit_value if explicit_value is not None else -1,
        },
        severity=AuditSeverity.INFO,
        category=AuditCategory.VALIDATION,
    ))

    return resolution
