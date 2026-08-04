"""CPU reference primitives for FEM-inspired incremental LLM inference.

This module intentionally implements a small, auditable linear prototype. It
is not a claim of Transformer speedup. The goal is to test the algebraic and
cost conditions that must hold before custom GPU kernels are justified.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

Array = np.ndarray


class WeightedResidualLedger:
    """Represent a state as an anchor plus a compensated sum of weighted deltas."""

    def __init__(self, anchor: Array):
        anchor_array = np.asarray(anchor, dtype=np.float64)
        if anchor_array.ndim != 1:
            raise ValueError("anchor must be a one-dimensional vector")
        self._anchor = anchor_array.copy()
        self._increment = np.zeros_like(self._anchor)
        self._compensation = np.zeros_like(self._anchor)
        self._steps = 0

    @property
    def size(self) -> int:
        return int(self._anchor.size)

    @property
    def steps(self) -> int:
        return self._steps

    @property
    def anchor(self) -> Array:
        return self._anchor.copy()

    def materialize(self) -> Array:
        return self._anchor + self._increment

    def append(
        self,
        delta: Array,
        *,
        weight: float = 1.0,
        indices: Optional[Sequence[int]] = None,
    ) -> None:
        if not np.isfinite(weight):
            raise ValueError("weight must be finite")

        if indices is None:
            contribution = np.asarray(delta, dtype=np.float64)
            if contribution.shape != self._anchor.shape:
                raise ValueError("global delta shape must match anchor shape")
            target_indices: object = slice(None)
        else:
            index_array = np.asarray(indices, dtype=np.int64)
            if index_array.ndim != 1:
                raise ValueError("indices must be one-dimensional")
            if index_array.size == 0:
                self._steps += 1
                return
            if np.any(index_array < 0) or np.any(index_array >= self.size):
                raise IndexError("delta index is out of range")
            if np.unique(index_array).size != index_array.size:
                raise ValueError("indices must not contain duplicates")
            contribution = np.asarray(delta, dtype=np.float64)
            if contribution.shape != (index_array.size,):
                raise ValueError("local delta length must match indices length")
            target_indices = index_array

        weighted = weight * contribution
        previous = self._increment[target_indices]
        correction = weighted - self._compensation[target_indices]
        updated = previous + correction
        self._compensation[target_indices] = (updated - previous) - correction
        self._increment[target_indices] = updated
        self._steps += 1

    def residual(self, full_state: Array) -> Array:
        full = np.asarray(full_state, dtype=np.float64)
        if full.shape != self._anchor.shape:
            raise ValueError("full_state shape must match anchor shape")
        return full - self.materialize()

    def relative_residual_norm(self, full_state: Array, *, eps: float = 1e-12) -> float:
        full = np.asarray(full_state, dtype=np.float64)
        numerator = np.linalg.norm(self.residual(full))
        denominator = max(float(np.linalg.norm(full)), eps)
        return float(numerator / denominator)

    def reanchor(self, full_state: Optional[Array] = None) -> None:
        if full_state is None:
            new_anchor = self.materialize()
        else:
            new_anchor = np.asarray(full_state, dtype=np.float64)
            if new_anchor.shape != self._anchor.shape:
                raise ValueError("full_state shape must match anchor shape")
            new_anchor = new_anchor.copy()

        self._anchor = new_anchor
        self._increment.fill(0.0)
        self._compensation.fill(0.0)
        self._steps = 0


class IncrementalLinearOperator:
    """Exact incremental reference for y = W x with sparse input changes."""

    def __init__(self, matrix: Array, initial_input: Array):
        matrix_array = np.asarray(matrix, dtype=np.float64)
        input_array = np.asarray(initial_input, dtype=np.float64)
        if matrix_array.ndim != 2:
            raise ValueError("matrix must be two-dimensional")
        if input_array.ndim != 1:
            raise ValueError("initial_input must be one-dimensional")
        if matrix_array.shape[1] != input_array.size:
            raise ValueError("matrix input dimension must match initial_input")

        self.matrix = matrix_array.copy()
        self.input_ledger = WeightedResidualLedger(input_array)
        self.output_ledger = WeightedResidualLedger(self.matrix @ input_array)

    @property
    def input_size(self) -> int:
        return int(self.matrix.shape[1])

    @property
    def output_size(self) -> int:
        return int(self.matrix.shape[0])

    @property
    def steps(self) -> int:
        return self.input_ledger.steps

    def current_input(self) -> Array:
        return self.input_ledger.materialize()

    def current_output(self) -> Array:
        return self.output_ledger.materialize()

    def apply_local_delta(
        self,
        indices: Sequence[int],
        delta: Array,
        *,
        weight: float = 1.0,
    ) -> Array:
        index_array = np.asarray(indices, dtype=np.int64)
        local_delta = np.asarray(delta, dtype=np.float64)
        if index_array.ndim != 1:
            raise ValueError("indices must be one-dimensional")
        if local_delta.shape != (index_array.size,):
            raise ValueError("delta length must match indices length")

        self.input_ledger.append(local_delta, weight=weight, indices=index_array)
        output_delta = self.matrix[:, index_array] @ (weight * local_delta)
        self.output_ledger.append(output_delta)
        return output_delta

    def exact_output(self) -> Array:
        return self.matrix @ self.current_input()

    def output_residual(self) -> Array:
        return self.exact_output() - self.current_output()

    def relative_output_residual_norm(self, *, eps: float = 1e-12) -> float:
        exact = self.exact_output()
        numerator = np.linalg.norm(exact - self.current_output())
        denominator = max(float(np.linalg.norm(exact)), eps)
        return float(numerator / denominator)

    def reanchor(self) -> None:
        verified_input = self.current_input()
        verified_output = self.matrix @ verified_input
        self.input_ledger.reanchor(verified_input)
        self.output_ledger.reanchor(verified_output)


@dataclass(frozen=True)
class CorrectionPolicy:
    relative_residual_tolerance: float = 1e-8
    max_increment_steps: int = 32

    def __post_init__(self) -> None:
        if self.relative_residual_tolerance < 0.0:
            raise ValueError("relative_residual_tolerance must be non-negative")
        if self.max_increment_steps <= 0:
            raise ValueError("max_increment_steps must be positive")

    def should_correct(self, *, relative_residual: float, steps: int) -> bool:
        if not np.isfinite(relative_residual):
            return True
        return (
            relative_residual > self.relative_residual_tolerance
            or steps >= self.max_increment_steps
        )


@dataclass(frozen=True)
class LinearCostEstimate:
    dense_flops: float
    incremental_flops: float
    amortized_flops: float
    theoretical_speedup: float
    changed_fraction: float


def estimate_linear_cost(
    *,
    input_features: int,
    output_features: int,
    changed_features: int,
    reanchor_interval: int,
    management_flops: float = 0.0,
) -> LinearCostEstimate:
    if input_features <= 0 or output_features <= 0:
        raise ValueError("feature dimensions must be positive")
    if changed_features < 0 or changed_features > input_features:
        raise ValueError("changed_features must be in [0, input_features]")
    if reanchor_interval <= 0:
        raise ValueError("reanchor_interval must be positive")
    if management_flops < 0.0:
        raise ValueError("management_flops must be non-negative")

    dense_flops = float(2 * input_features * output_features)
    incremental_flops = float(2 * changed_features * output_features) + management_flops
    amortized_flops = incremental_flops + dense_flops / reanchor_interval
    speedup = dense_flops / amortized_flops if amortized_flops > 0.0 else float("inf")

    return LinearCostEstimate(
        dense_flops=dense_flops,
        incremental_flops=incremental_flops,
        amortized_flops=amortized_flops,
        theoretical_speedup=speedup,
        changed_fraction=changed_features / input_features,
    )


def choose_active_indices(delta: Array, *, energy_fraction: float) -> Array:
    vector = np.asarray(delta, dtype=np.float64)
    if vector.ndim != 1:
        raise ValueError("delta must be one-dimensional")
    if not 0.0 < energy_fraction <= 1.0:
        raise ValueError("energy_fraction must be in (0, 1]")
    if vector.size == 0:
        return np.array([], dtype=np.int64)

    energy = np.square(vector)
    total = float(np.sum(energy))
    if total == 0.0:
        return np.array([], dtype=np.int64)

    order = np.argsort(energy)[::-1]
    cumulative = np.cumsum(energy[order])
    count = int(np.searchsorted(cumulative, energy_fraction * total, side="left") + 1)
    return np.sort(order[:count]).astype(np.int64)
