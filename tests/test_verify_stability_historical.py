"""Regression tests for historical controls with legitimate platform roundoff."""
import copy
import unittest
from scripts.verify_stability_report import historical_full_control


class HistoricalControlTests(unittest.TestCase):
    def setUp(self):
        self.sources = {('1', 'maest', 'unweighted')}
        self.methods = ['raw', 'platt']
        self.historical = [{'seed': '1', 'representation': 'maest', 'weighting': 'unweighted',
            'fraction': '1.0', 'method': method, 'macro_brier': '0.2', 'macro_log_loss': '0.6'}
            for method in self.methods]
        self.current = {(0, '1', 'maest', 'unweighted', 1.0, method):
            {'macro_brier': .2, 'macro_log_loss': .6} for method in self.methods}
        self.target = self.current[(0, '1', 'maest', 'unweighted', 1.0, 'platt')]

    def verify(self, claimed=0.0):
        return historical_full_control(self.current, self.historical, self.sources, self.methods, 1e-8, claimed)

    def test_nonzero_difference_within_frozen_limit_is_recomputed_and_accepted(self):
        self.target['macro_log_loss'] += 2e-9
        actual = abs(self.target['macro_log_loss'] - .6)
        self.assertGreater(actual, 1e-12)
        self.assertEqual(self.verify(claimed=actual), actual)

    def test_frozen_limit_is_enforced_even_when_receipt_agrees(self):
        self.target['macro_brier'] += 2e-8
        actual = abs(self.target['macro_brier'] - .2)
        with self.assertRaisesRegex(ValueError, 'exceeds frozen tolerance'):
            self.verify(claimed=actual)

    def test_dishonest_zero_or_exaggerated_control_is_rejected(self):
        self.target['macro_log_loss'] += 2e-9
        for claimed in [0.0, 8e-9, float('nan'), -1.0]:
            with self.subTest(claimed=claimed), self.assertRaisesRegex(ValueError, 'does not match'):
                self.verify(claimed=claimed)

    def test_missing_or_duplicate_baseline_endpoint_cannot_hide_largest_difference(self):
        original = copy.deepcopy(self.historical)
        for changed in [original[:1], original + [original[0]]]:
            self.historical = changed
            with self.subTest(rows=len(changed)), self.assertRaisesRegex(ValueError, 'coverage|Duplicate'):
                self.verify()

    def test_nonfinite_historical_metric_is_rejected(self):
        self.historical[0]['macro_log_loss'] = 'nan'
        with self.assertRaisesRegex(ValueError, 'Nonfinite'):
            self.verify()

    def test_partial_budget_does_not_enter_full_endpoint_control(self):
        self.historical.append({**self.historical[0], 'fraction': '0.25', 'macro_log_loss': '5.0'})
        self.assertEqual(self.verify(), 0.0)


if __name__ == '__main__':
    unittest.main()
