"""Metrics for Phase D1 activation-delta screening.

D1 does not alter model execution. It records pairs of activations and measures
whether their differences are sparse, block-local, or low-rank enough to
justify later incremental kernels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch


@dataclass(frozen=True)
class DeltaMetricConfig:
    energy_targets: tuple[float, ...] = (0.90, 0.95, 0.99)
    block_size: int = 64
    absolute_threshold: float = 1e-8
    relative_threshold: float = 1e-3
    max_svd_rows: int = 2048

    def __post_init__(self) -> None:
        if not self.energy_targets:
            raise ValueError("energy_targets must not be empty")
        if any(not 0.0 < target <= 1.0 for target in self.energy_targets):
            raise ValueError("energy_targets must be in (0, 1]")
        if self.block_size <= 0:
            raise ValueError("block_size must be positive")
        if self.absolute_threshold < 0.0 or self.relative_threshold < 0.0:
            raise ValueError("thresholds must be non-negative")
        if self.max_svd_rows <= 0:
            raise ValueError("max_svd_rows must be positive")


def _crop_sequence(tensor: torch.Tensor, length: int, alignment: str) -> torch.Tensor:
    if tensor.ndim < 2:
        return tensor
    if alignment == "prefix":
        return tensor.narrow(-2, 0, length)
    if alignment == "suffix":
        return tensor.narrow(-2, tensor.shape[-2] - length, length)
    raise ValueError("alignment must be 'prefix' or 'suffix'")


def align_activation_pair(
    base: torch.Tensor,
    edited: torch.Tensor,
    *,
    view: str = "last_token",
    alignment: str = "suffix",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return same-shaped float CPU tensors suitable for delta analysis.

    The last dimension is treated as the feature dimension. For attention KV
    tensors, the penultimate dimension is the token axis, which also matches
    standard hidden-state tensors.
    """

    left = base.detach().float().cpu()
    right = edited.detach().float().cpu()
    if left.ndim == 0 or right.ndim == 0:
        return left.reshape(1, 1), right.reshape(1, 1)
    if left.shape[-1] != right.shape[-1]:
        raise ValueError("feature dimensions differ between activation pairs")

    if view == "last_token" and left.ndim >= 2 and right.ndim >= 2:
        left = left.select(-2, left.shape[-2] - 1).unsqueeze(-2)
        right = right.select(-2, right.shape[-2] - 1).unsqueeze(-2)
    elif view == "aligned_sequence" and left.ndim >= 2 and right.ndim >= 2:
        common = min(left.shape[-2], right.shape[-2])
        left = _crop_sequence(left, common, alignment)
        right = _crop_sequence(right, common, alignment)
    elif view != "last_token" and view != "aligned_sequence":
        raise ValueError("view must be 'last_token' or 'aligned_sequence'")

    left_matrix = left.reshape(-1, left.shape[-1])
    right_matrix = right.reshape(-1, right.shape[-1])
    rows = min(left_matrix.shape[0], right_matrix.shape[0])
    if alignment == "prefix":
        return left_matrix[:rows], right_matrix[:rows]
    return left_matrix[-rows:], right_matrix[-rows:]


def _minimum_active_fraction(energy: np.ndarray, target: float) -> float:
    if energy.size == 0:
        return 0.0
    total = float(energy.sum())
    if total <= 0.0:
        return 0.0
    order = np.sort(energy)[::-1]
    count = int(np.searchsorted(np.cumsum(order), target * total, side="left") + 1)
    return count / energy.size


def _rank_metrics(matrix: np.ndarray, targets: Iterable[float], max_rows: int) -> dict[str, float | int]:
    if matrix.size == 0 or not np.any(matrix):
        result: dict[str, float | int] = {"stable_rank": 0.0, "entropy_effective_rank": 0.0}
        for target in targets:
            result[f"svd_rank_{int(round(target * 100))}"] = 0
        return result

    sampled = matrix
    if matrix.shape[0] > max_rows:
        indices = np.linspace(0, matrix.shape[0] - 1, max_rows, dtype=np.int64)
        sampled = matrix[indices]
    singular_values = np.linalg.svd(sampled, compute_uv=False)
    squared = np.square(singular_values)
    total = float(squared.sum())
    spectral = float(squared[0]) if squared.size else 0.0
    probabilities = squared / total if total > 0.0 else np.zeros_like(squared)
    nonzero = probabilities[probabilities > 0.0]
    entropy_rank = float(np.exp(-np.sum(nonzero * np.log(nonzero)))) if nonzero.size else 0.0
    result = {
        "stable_rank": float(total / spectral) if spectral > 0.0 else 0.0,
        "entropy_effective_rank": entropy_rank,
    }
    cumulative = np.cumsum(squared)
    for target in targets:
        count = int(np.searchsorted(cumulative, target * total, side="left") + 1) if total > 0.0 else 0
        result[f"svd_rank_{int(round(target * 100))}"] = count
    return result


def analyze_activation_delta(
    base: torch.Tensor,
    edited: torch.Tensor,
    *,
    config: DeltaMetricConfig | None = None,
    view: str = "last_token",
    alignment: str = "suffix",
) -> dict[str, float | int | list[int]]:
    cfg = config or DeltaMetricConfig()
    left, right = align_activation_pair(base, edited, view=view, alignment=alignment)
    delta = (right - left).numpy().astype(np.float64, copy=False)
    flat = delta.reshape(-1)
    absolute = np.abs(flat)
    energy = np.square(flat)
    max_abs = float(absolute.max()) if absolute.size else 0.0
    threshold = max(cfg.absolute_threshold, cfg.relative_threshold * max_abs)
    changed_fraction = float(np.mean(absolute > threshold)) if absolute.size else 0.0

    block_count = int(np.ceil(flat.size / cfg.block_size)) if flat.size else 0
    if block_count:
        padded = np.pad(energy, (0, block_count * cfg.block_size - energy.size))
        block_energy = padded.reshape(block_count, cfg.block_size).sum(axis=1)
    else:
        block_energy = np.array([], dtype=np.float64)

    metrics: dict[str, float | int | list[int]] = {
        "shape": list(delta.shape),
        "rows": int(delta.shape[0]),
        "features": int(delta.shape[1]),
        "elements": int(flat.size),
        "delta_l2": float(np.linalg.norm(flat)),
        "base_l2": float(np.linalg.norm(left.numpy())),
        "relative_delta_l2": float(np.linalg.norm(flat) / max(float(np.linalg.norm(left.numpy())), 1e-12)),
        "max_abs_delta": max_abs,
        "activity_threshold": float(threshold),
        "changed_fraction": changed_fraction,
        "block_size": cfg.block_size,
        "blocks": block_count,
    }
    for target in cfg.energy_targets:
        suffix = int(round(target * 100))
        metrics[f"active_fraction_{suffix}"] = _minimum_active_fraction(energy, target)
        metrics[f"active_block_fraction_{suffix}"] = _minimum_active_fraction(block_energy, target)
    metrics.update(_rank_metrics(delta, cfg.energy_targets, cfg.max_svd_rows))
    return metrics
