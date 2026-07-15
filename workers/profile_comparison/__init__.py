"""Profile comparison package (M12) — cycle metric distribution and consistency."""

from workers.profile_comparison.profile_comparator import (
    compare_cycle_metric_distribution,
    detect_cycle_to_cycle_drift,
    compute_profile_consistency_score,
)

__all__ = [
    "compare_cycle_metric_distribution",
    "detect_cycle_to_cycle_drift",
    "compute_profile_consistency_score",
]
