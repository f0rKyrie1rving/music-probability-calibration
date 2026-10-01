import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from music_calibration.calibrators import _objective, fit, predict, select_ridge


class CalibrationTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(731)
        self.z = rng.normal(size=180)
        self.y = rng.binomial(1, expit(0.7 * self.z - 1.0))
        self.folds = np.arange(len(self.z)) % 3

    def test_raw_is_exact_identity_and_json_roundtrip(self):
        z = np.array([-1000.0, -40.0, -1.0, 0.0, 0.1, 40.0, 1000.0])
        record = fit(self.z, self.y, "raw")
        np.testing.assert_array_equal(predict(z, json.loads(json.dumps(record))), expit(z))

    def test_all_maps_are_monotone_finite_at_extremes(self):
        z = np.r_[-1000.0, np.linspace(-15.0, 15.0, 201), 1000.0]
        for method in ("platt", "temperature", "beta", "isotonic", "ridge_platt"):
            with self.subTest(method=method):
                record = fit(self.z, self.y, method, penalty=0.1 if method == "ridge_platt" else 0)
                self.assertFalse(record["fallback"])
                record = json.loads(json.dumps(record, allow_nan=False))
                p = predict(z, record)
                self.assertTrue(np.isfinite(p).all())
                self.assertTrue(np.all((p >= 0) & (p <= 1)))
                self.assertTrue(np.all(np.diff(p) >= 0))

    def test_isotonic_clips_to_observed_endpoints(self):
        record = fit(np.array([-2., -1., 1., 2.]), np.array([0, 1, 0, 1]), "isotonic")
        np.testing.assert_array_equal(predict([-1e3, 1e3], record), [0, 1])
        constant = fit(np.zeros(4), [0, 1, 0, 1], "isotonic")
        np.testing.assert_array_equal(predict([-1000, 0, 1000], constant), [0.5, 0.5, 0.5])

    def test_single_class_fit_falls_back_exactly(self):
        for method in ("platt", "temperature", "beta", "isotonic", "ridge_platt"):
            record = fit(self.z, np.zeros(len(self.z)), method)
            self.assertTrue(record["fallback"])
            self.assertFalse(record["success"])
            self.assertEqual(record["reason"], "single_class_calibration")
            np.testing.assert_array_equal(predict(self.z, record), expit(self.z))

    def test_nonconvergence_is_counted_but_exceptions_raise(self):
        failed = SimpleNamespace(x=np.array([1., 0.]), fun=1., success=False,
                                 status=2, message="test nonconvergence", nit=3)
        with patch("music_calibration.calibrators.minimize", return_value=failed):
            record = fit(self.z, self.y, "platt")
            self.assertEqual(record["reason"], "optimizer_nonconvergence")
            np.testing.assert_array_equal(predict(self.z, record), expit(self.z))
        with patch("music_calibration.calibrators.minimize", side_effect=RuntimeError("programming failure")):
            with self.assertRaises(RuntimeError):
                fit(self.z, self.y, "platt")

    def test_invalid_inputs_raise(self):
        for z, y in [([], []), ([1., np.nan], [0, 1]), ([1., 2.], [0]),
                     ([1., 2.], [0, 0.5]), ([[1., 2.]], [[0, 1]])]:
            with self.subTest(z=z, y=y), self.assertRaises(ValueError):
                fit(z, y, "platt")
        for penalty in [-1, np.nan, np.inf, True, "0.1"]:
            with self.subTest(penalty=penalty), self.assertRaises(ValueError):
                fit(self.z, self.y, "ridge_platt", penalty=penalty)
        with self.assertRaises(ValueError):
            fit(self.z, self.y, "unknown")
        with self.assertRaises(ValueError):
            fit(self.z, self.y, "platt", penalty=1)
        with self.assertRaises(ValueError):
            predict([np.inf], {"method": "raw"})
        for folds in [np.zeros(180, dtype=int), self.folds.astype(float), self.folds[:-1]]:
            with self.assertRaises(ValueError):
                select_ridge(self.z, self.y, folds)
        for penalties in [[], [0., 0.], [-1.]]:
            with self.assertRaises(ValueError):
                select_ridge(self.z, self.y, self.folds, penalties=penalties)

    def test_objective_gradients_and_penalty_definition(self):
        cases = [("platt", np.array([1.3, -.4]), 0),
                 ("temperature", np.array([1.3]), 0),
                 ("beta", np.array([1.3, .8, -.4]), 0),
                 ("ridge_platt", np.array([1.3, -.4]), .1)]
        for method, theta, penalty in cases:
            with self.subTest(method=method):
                value, grad = _objective(theta, self.z, self.y, method, penalty)
                numerical = np.empty(len(theta))
                for j in range(len(theta)):
                    delta = np.zeros(len(theta)); delta[j] = 1e-6
                    numerical[j] = (_objective(theta + delta, self.z, self.y, method, penalty)[0]
                                    - _objective(theta - delta, self.z, self.y, method, penalty)[0]) / 2e-6
                np.testing.assert_allclose(grad, numerical, atol=1e-8)
                if method == "ridge_platt":
                    base = _objective(theta, self.z, self.y, "platt", 0)[0]
                    self.assertAlmostEqual(value - base, penalty * ((theta[0] - 1) ** 2 + theta[1] ** 2))

    def test_strong_penalty_keeps_map_near_identity(self):
        weak = fit(self.z, self.y, "ridge_platt", penalty=0)
        strong = fit(self.z, self.y, "ridge_platt", penalty=100)
        deviation = lambda r: (r["slope"] - 1) ** 2 + r["intercept"] ** 2
        self.assertLess(deviation(strong), deviation(weak) / 100)

    def test_platt_matches_legacy_formula_and_zero_ridge(self):
        def original(theta):
            u = theta[0] * self.z + theta[1]
            residual = expit(u) - self.y
            return float(np.mean(np.logaddexp(0, u) - self.y * u)), np.array([
                np.mean(residual * self.z), np.mean(residual)])
        result = minimize(original, np.array([1., 0.]), jac=True, method="L-BFGS-B",
                          bounds=[(.001, 100.), (-20., 20.)],
                          options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-8})
        fitted = fit(self.z, self.y, "platt")
        np.testing.assert_allclose(predict(self.z, fitted), expit(result.x[0] * self.z + result.x[1]), atol=1e-14, rtol=0)
        np.testing.assert_array_equal(predict(self.z, fitted), predict(self.z, fit(self.z, self.y, "ridge_platt", penalty=0)))

    def test_ridge_single_class_folds_counted_and_identity_tie_wins(self):
        y = np.zeros(len(self.z))
        result = select_ridge(self.z, y, self.folds, penalties=[0., .1])
        self.assertEqual(result["selected_penalty"], "identity")
        self.assertEqual(result["inner_fallback_count"], 6)
        np.testing.assert_array_equal(predict(self.z, result), expit(self.z))
        json.dumps(result, allow_nan=False)

    def test_selection_uses_oof_labels_and_saves_reconstructible_scores(self):
        z = np.zeros(120)
        folds = np.arange(120) % 3
        balanced = np.tile([0, 1], 60)
        low_prevalence = (np.arange(120) % 4 == 0).astype(int)
        identity = select_ridge(z, balanced, folds)
        shifted = select_ridge(z, low_prevalence, folds)
        self.assertEqual(identity["selected_penalty"], "identity")
        self.assertEqual(shifted["selected_penalty"], 0.)
        self.assertLess(np.mean(predict(z, shifted)), .3)
        for row in shifted["oof_scores"]:
            key = "identity" if row["penalty"] == "identity" else str(row["penalty"])
            p = np.asarray(shifted["oof_predictions"][key])
            self.assertAlmostEqual(np.mean((p - low_prevalence) ** 2), row["brier"])
        self.assertFalse(shifted["selection"]["uses_evaluation_data"])

    def test_tie_between_finite_penalties_chooses_strongest(self):
        z = np.zeros(12)
        y = np.tile([0, 0, 0, 1], 3)
        folds = np.arange(12) % 3
        same_fit = {"method": "ridge_platt", "effective_method": "ridge_platt",
                    "slope": 1., "intercept": -np.log(3.), "fallback": False}
        with patch("music_calibration.calibrators.fit", return_value=same_fit):
            result = select_ridge(z, y, folds, penalties=[0., .1, 1.])
        self.assertEqual(result["selected_penalty"], 1.)


if __name__ == "__main__":
    unittest.main()
