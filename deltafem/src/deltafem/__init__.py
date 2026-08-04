"""DeltaFEM-LLM CPU reference package."""

from .core import (
    CorrectionPolicy,
    IncrementalLinearOperator,
    LinearCostEstimate,
    WeightedResidualLedger,
    choose_active_indices,
    estimate_linear_cost,
)
from .experiment import run_phase0

__all__ = [
    "CorrectionPolicy",
    "IncrementalLinearOperator",
    "LinearCostEstimate",
    "WeightedResidualLedger",
    "choose_active_indices",
    "estimate_linear_cost",
    "run_phase0",
]
