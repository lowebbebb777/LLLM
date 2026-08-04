from __future__ import annotations

import math

import torch

from deltafem.d1_metrics import DeltaMetricConfig, align_activation_pair, analyze_activation_delta


def test_last_token_alignment_handles_different_sequence_lengths() -> None:
    left = torch.arange(24, dtype=torch.float32).reshape(1, 3, 8)
    right = torch.arange(40, dtype=torch.float32).reshape(1, 5, 8)
    aligned_left, aligned_right = align_activation_pair(left, right, view="last_token")
    assert aligned_left.shape == aligned_right.shape == (1, 8)
    assert torch.equal(aligned_left[0], left[0, -1])
    assert torch.equal(aligned_right[0], right[0, -1])


def test_single_active_value_has_small_energy_support() -> None:
    left = torch.zeros(1, 1, 10)
    right = left.clone()
    right[0, 0, 3] = 2.0
    metrics = analyze_activation_delta(
        left,
        right,
        config=DeltaMetricConfig(block_size=2, relative_threshold=0.0),
    )
    assert math.isclose(float(metrics["active_fraction_90"]), 0.1)
    assert math.isclose(float(metrics["active_block_fraction_90"]), 0.2)
    assert math.isclose(float(metrics["changed_fraction"]), 0.1)


def test_rank_metrics_detect_rank_one_delta() -> None:
    base = torch.zeros(1, 4, 3)
    vector = torch.tensor([1.0, 2.0, 3.0])
    edited = base + vector
    metrics = analyze_activation_delta(base, edited, view="aligned_sequence")
    assert math.isclose(float(metrics["stable_rank"]), 1.0, rel_tol=1e-6)
    assert int(metrics["svd_rank_90"]) == 1
