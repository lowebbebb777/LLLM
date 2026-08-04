"""DeltaFEM-LLM reference package."""

from .core import (
    CorrectionPolicy,
    IncrementalLinearOperator,
    LinearCostEstimate,
    WeightedResidualLedger,
    choose_active_indices,
    estimate_linear_cost,
)
from .d1_metrics import DeltaMetricConfig, align_activation_pair, analyze_activation_delta
from .d1_probe import (
    PromptPair,
    ToyCausalLM,
    ToyTokenizer,
    capture_activations,
    compare_snapshots,
    load_prompt_pairs,
    run_prompt_edit_probe,
    run_token_step_probe,
    summarize_rows,
    write_probe_results,
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
    "DeltaMetricConfig",
    "align_activation_pair",
    "analyze_activation_delta",
    "PromptPair",
    "ToyCausalLM",
    "ToyTokenizer",
    "capture_activations",
    "compare_snapshots",
    "load_prompt_pairs",
    "run_prompt_edit_probe",
    "run_token_step_probe",
    "summarize_rows",
    "write_probe_results",
]
