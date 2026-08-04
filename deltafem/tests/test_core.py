from __future__ import annotations

import unittest

import numpy as np

from deltafem import (
    CorrectionPolicy,
    IncrementalLinearOperator,
    WeightedResidualLedger,
    choose_active_indices,
    estimate_linear_cost,
    run_phase0,
)


class WeightedResidualLedgerTests(unittest.TestCase):
    def test_weighted_global_and_local_deltas_match_manual_sum(self) -> None:
        ledger = WeightedResidualLedger(np.array([1.0, 2.0, 3.0, 4.0]))
        ledger.append(np.array([0.5, -1.0, 2.0, 0.0]), weight=0.25)
        ledger.append(np.array([3.0, -2.0]), indices=[0, 3], weight=0.5)

        expected = np.array([1.0, 2.0, 3.0, 4.0])
        expected += 0.25 * np.array([0.5, -1.0, 2.0, 0.0])
        expected[[0, 3]] += 0.5 * np.array([3.0, -2.0])
        np.testing.assert_allclose(ledger.materialize(), expected, rtol=0.0, atol=1e-14)
        self.assertEqual(ledger.steps, 2)

    def test_reanchor_resets_increment_history(self) -> None:
        ledger = WeightedResidualLedger(np.zeros(3))
        ledger.append(np.ones(3))
        ledger.reanchor()
        np.testing.assert_allclose(ledger.anchor, np.ones(3))
        np.testing.assert_allclose(ledger.materialize(), np.ones(3))
        self.assertEqual(ledger.steps, 0)

    def test_full_state_correction_removes_residual(self) -> None:
        ledger = WeightedResidualLedger(np.zeros(2))
        ledger.append(np.array([1.0, 1.0]))
        verified = np.array([1.0, 1.25])
        self.assertGreater(ledger.relative_residual_norm(verified), 0.0)
        ledger.reanchor(verified)
        self.assertEqual(ledger.relative_residual_norm(verified), 0.0)


class IncrementalLinearOperatorTests(unittest.TestCase):
    def test_sparse_delta_update_is_algebraically_exact(self) -> None:
        rng = np.random.default_rng(3)
        matrix = rng.normal(size=(12, 16))
        initial = rng.normal(size=16)
        operator = IncrementalLinearOperator(matrix, initial)

        for _ in range(20):
            indices = np.sort(rng.choice(16, size=4, replace=False))
            delta = rng.normal(scale=0.05, size=4)
            operator.apply_local_delta(indices, delta, weight=0.7)
            np.testing.assert_allclose(
                operator.current_output(), operator.exact_output(), rtol=1e-12, atol=1e-12
            )

    def test_reanchor_preserves_state(self) -> None:
        matrix = np.arange(24, dtype=np.float64).reshape(4, 6)
        operator = IncrementalLinearOperator(matrix, np.ones(6))
        operator.apply_local_delta([1, 5], np.array([0.25, -0.5]))
        before_input = operator.current_input()
        before_output = operator.current_output()
        operator.reanchor()
        np.testing.assert_allclose(operator.current_input(), before_input)
        np.testing.assert_allclose(operator.current_output(), before_output)
        self.assertEqual(operator.steps, 0)


class PolicyAndCostTests(unittest.TestCase):
    def test_policy_triggers_on_error_or_step_limit(self) -> None:
        policy = CorrectionPolicy(relative_residual_tolerance=1e-3, max_increment_steps=8)
        self.assertFalse(policy.should_correct(relative_residual=1e-4, steps=7))
        self.assertTrue(policy.should_correct(relative_residual=2e-3, steps=1))
        self.assertTrue(policy.should_correct(relative_residual=0.0, steps=8))
        self.assertTrue(policy.should_correct(relative_residual=float("nan"), steps=1))

    def test_sparse_case_has_theoretical_headroom(self) -> None:
        estimate = estimate_linear_cost(
            input_features=1024,
            output_features=1024,
            changed_features=32,
            reanchor_interval=32,
            management_flops=0.02 * (2 * 1024 * 1024),
        )
        self.assertGreater(estimate.theoretical_speedup, 5.0)

    def test_dense_change_has_no_theoretical_benefit(self) -> None:
        estimate = estimate_linear_cost(
            input_features=256,
            output_features=256,
            changed_features=256,
            reanchor_interval=16,
            management_flops=0.0,
        )
        self.assertLess(estimate.theoretical_speedup, 1.0)

    def test_active_index_selection_retains_requested_energy(self) -> None:
        delta = np.array([4.0, 3.0, 0.1, 0.1])
        indices = choose_active_indices(delta, energy_fraction=0.90)
        retained = float(np.sum(np.square(delta[indices])))
        total = float(np.sum(np.square(delta)))
        self.assertGreaterEqual(retained / total, 0.90)
        self.assertEqual(indices.tolist(), [0, 1])


class Phase0ExperimentTests(unittest.TestCase):
    def test_phase0_report_is_deterministic_and_exact(self) -> None:
        first = run_phase0(
            seed=11,
            input_features=32,
            output_features=24,
            steps=8,
            changed_fractions=(0.125, 1.0),
            reanchor_interval=4,
        )
        second = run_phase0(
            seed=11,
            input_features=32,
            output_features=24,
            steps=8,
            changed_fractions=(0.125, 1.0),
            reanchor_interval=4,
        )
        self.assertEqual(first, second)
        for case in first["cases"]:
            self.assertLess(case["max_relative_error"], 1e-12)


if __name__ == "__main__":
    unittest.main()
