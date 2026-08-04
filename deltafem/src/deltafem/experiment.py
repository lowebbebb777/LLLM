"""Synthetic Phase-0 experiment for DeltaFEM-LLM."""

from __future__ import annotations

from dataclasses import asdict
from typing import Iterable

import numpy as np

from .core import IncrementalLinearOperator, estimate_linear_cost


def run_phase0(
    *,
    seed: int = 7,
    input_features: int = 256,
    output_features: int = 256,
    steps: int = 32,
    changed_fractions: Iterable[float] = (0.01, 0.05, 0.10, 0.25, 0.50, 1.0),
    reanchor_interval: int = 16,
    management_fraction_of_dense: float = 0.02,
) -> dict:
    if steps <= 0:
        raise ValueError("steps must be positive")
    rng = np.random.default_rng(seed)
    matrix = rng.normal(0.0, 1.0 / np.sqrt(input_features), size=(output_features, input_features))
    initial_input = rng.normal(size=input_features)
    dense_flops = float(2 * input_features * output_features)
    management_flops = management_fraction_of_dense * dense_flops

    cases = []
    for fraction in changed_fractions:
        if not 0.0 < fraction <= 1.0:
            raise ValueError("changed fractions must be in (0, 1]")
        changed = max(1, min(input_features, int(round(input_features * fraction))))
        operator = IncrementalLinearOperator(matrix, initial_input)
        max_abs_error = 0.0
        max_relative_error = 0.0

        for step in range(steps):
            indices = np.sort(rng.choice(input_features, size=changed, replace=False))
            delta = rng.normal(scale=0.01, size=changed)
            operator.apply_local_delta(indices, delta)
            exact = operator.exact_output()
            incremental = operator.current_output()
            max_abs_error = max(max_abs_error, float(np.max(np.abs(exact - incremental))))
            relative = float(
                np.linalg.norm(exact - incremental)
                / max(float(np.linalg.norm(exact)), 1e-12)
            )
            max_relative_error = max(max_relative_error, relative)
            if (step + 1) % reanchor_interval == 0:
                operator.reanchor()

        estimate = estimate_linear_cost(
            input_features=input_features,
            output_features=output_features,
            changed_features=changed,
            reanchor_interval=reanchor_interval,
            management_flops=management_flops,
        )
        case = asdict(estimate)
        case.update(
            {
                "requested_changed_fraction": fraction,
                "changed_features": changed,
                "max_abs_error": max_abs_error,
                "max_relative_error": max_relative_error,
            }
        )
        cases.append(case)

    return {
        "experiment": "deltafem_phase0_linear_reference",
        "interpretation": (
            "Algebraic correctness and FLOP screening only; no Transformer or GPU speedup claim."
        ),
        "seed": seed,
        "input_features": input_features,
        "output_features": output_features,
        "steps": steps,
        "reanchor_interval": reanchor_interval,
        "management_fraction_of_dense": management_fraction_of_dense,
        "cases": cases,
    }
