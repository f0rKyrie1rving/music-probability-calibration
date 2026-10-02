"""Independent numerical checks for the portable bounded logistic optimizer."""
import math
import unittest

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from music_calibration_portable.solver import solve


def objective(theta, design, y, penalty=0., center=None):
    logits = design @ theta
    loss = np.where(y == 1, np.logaddexp(0, -logits), np.logaddexp(0, logits)).mean()
    difference = theta - (np.zeros_like(theta) if center is None else center)
    return float(loss + penalty * np.dot(difference, difference))


class PortableSolverTests(unittest.TestCase):
    def test_temperature_matches_analytic_binomial_optimum(self):
        result = solve(np.ones((6, 1)), np.array([0, 0, 1, 1, 1, 1]), [1.], [(0.001, 100.)])
        self.assertTrue(result['success'], result)
        self.assertAlmostEqual(result['params'][0], math.log(2), places=11)
        self.assertLessEqual(result['projected_gradient'], 2e-12)

    def test_exact_stationary_start_is_not_an_optimizer_failure(self):
        z = np.repeat([-math.log(3), math.log(3)], 4)
        y = np.array([0, 0, 0, 1, 0, 1, 1, 1])
        result = solve(np.column_stack([z, np.ones(len(z))]), y, [1., 0.], [(0.001, 100.), (-20., 20.)])
        self.assertTrue(result['success'], result)
        np.testing.assert_allclose(result['params'], [1., 0.], atol=1e-12, rtol=0)

    def test_separation_does_not_stop_early_on_a_small_gradient(self):
        design = np.array([[-3.], [-1.], [1.], [3.]])
        y = np.array([0, 0, 1, 1])
        result = solve(design, y, [1.], [(0.001, 100.)])
        self.assertTrue(result['success'], result)
        self.assertEqual(result['params'], [100.])
        self.assertEqual(result['projected_gradient'], 0.)
        self.assertLess(result['objective'], 1e-40)

    def test_strongly_saturated_separation_keeps_log_derivative_sign(self):
        result = solve(np.array([[-100.], [100.]]), np.array([0, 1]), [10.], [(0.001, 100.)])
        self.assertTrue(result['success'], result)
        self.assertEqual(result['params'], [100.])
        self.assertGreater(result['coordinate_sweeps'], 0)

    def test_lower_bound_has_correct_kkt_sign(self):
        z = np.array([-4., -2., 2., 4.])
        result = solve(z[:, None], np.array([1, 1, 0, 0]), [1.], [(0.001, 100.)])
        self.assertTrue(result['success'], result)
        self.assertEqual(result['params'], [0.001])
        self.assertGreater(result['gradient'][0], 0.)
        self.assertEqual(result['projected_gradient'], 0.)

    def test_flat_slope_retains_initial_and_intercept_is_identifiable(self):
        design = np.column_stack([np.zeros(6), np.ones(6)])
        result = solve(design, np.array([0, 0, 1, 1, 1, 1]), [1., 0.], [(0.001, 100.), (-20., 20.)])
        self.assertTrue(result['success'], result)
        self.assertEqual(result['params'][0], 1.)
        self.assertAlmostEqual(result['params'][1], math.log(2), places=10)

    def test_rank_deficient_columns_are_deterministic_under_row_permutation(self):
        design = np.ones((6, 2))
        y = np.array([0, 0, 1, 1, 1, 1])
        args = ([1., 0.], [(0.001, 100.), (-20., 20.)])
        first = solve(design, y, *args)
        order = [4, 0, 3, 2, 5, 1]
        second = solve(design[order], y[order], *args)
        self.assertTrue(first['success'], first)
        self.assertEqual(first['params'], second['params'])
        self.assertAlmostEqual(sum(first['params']), math.log(2), places=10)

    def test_ridge_objective_and_independent_gradient_are_unchanged(self):
        z = np.array([-5., -2., -.3, .2, 1., 4.])
        design = np.column_stack([z, np.ones(len(z))])
        y = np.array([0, 1, 0, 1, 0, 1])
        center = np.array([1., 0.])
        result = solve(design, y, center, [(0.001, 100.), (-20., 20.)], center, .1)
        theta = np.asarray(result['params'])
        independent_gradient = design.T @ (expit(design @ theta) - y) / len(y) + .2 * (theta - center)
        self.assertTrue(result['success'], result)
        self.assertLess(np.max(np.abs(independent_gradient)), 3e-12)
        self.assertAlmostEqual(result['objective'], objective(theta, design, y, .1, center), places=14)

    def test_beta_fit_matches_independent_constrained_optimization(self):
        rng = np.random.default_rng(3871)
        z = rng.normal(size=120) * 2
        design = np.column_stack([-np.logaddexp(0., -z), np.logaddexp(0., z), np.ones(len(z))])
        y = rng.binomial(1, expit(.7 * z + .3))
        initial = np.array([1., 1., 0.])
        bounds = [(0.001, 100.), (0.001, 100.), (-20., 20.)]
        result = solve(design, y, initial, bounds)
        reference = minimize(objective, initial, args=(design, y), method='SLSQP', bounds=bounds,
                             options={'ftol': 1e-13, 'maxiter': 2000})
        self.assertTrue(result['success'], result)
        self.assertTrue(reference.success, reference.message)
        self.assertLessEqual(result['objective'], reference.fun + 2e-12)
        self.assertLess(abs(result['objective'] - reference.fun), 1e-10)
        np.testing.assert_allclose(expit(design @ result['params']), expit(design @ reference.x), atol=1e-6, rtol=0)

    def test_iteration_exhaustion_does_not_claim_success(self):
        z = np.array([-3., -2., -1., 0., 1., 2., 3.])
        result = solve(np.column_stack([z, np.ones(len(z))]), np.array([0, 1, 0, 0, 1, 0, 1]),
                       [10., 5.], [(0.001, 100.), (-20., 20.)], max_iterations=1)
        self.assertFalse(result['success'])
        self.assertIn('not reached', result['message'])
        self.assertTrue(np.isfinite(result['params']).all())

    def test_invalid_or_misaligned_inputs_raise(self):
        for y in [np.array([0, np.nan]), np.array([0, .5]), np.array([0])]:
            with self.subTest(y=y), self.assertRaises(ValueError):
                solve(np.ones((2, 1)), y, [1.], [(0.001, 100.)])
        with self.assertRaises(ValueError):
            solve(np.ones((2, 1)), [0, 1], [101.], [(0.001, 100.)])
        with self.assertRaises(ValueError):
            solve(np.ones((2, 1)), [0, 1], [1.], [(0.001, 100.)], penalty=-.1)


if __name__ == '__main__':
    unittest.main()
