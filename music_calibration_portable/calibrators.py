"""Monotone binary probability maps and calibration-only ridge selection.

These are standard post-hoc methods. A recorded identity fallback is part of the
fitted policy, not evidence that an unsuccessful fit was a successful calibrator.
"""

from numbers import Real

import numpy as np
from .solver import solve
from scipy.special import expit
from sklearn.isotonic import IsotonicRegression


METHODS = frozenset({"raw", "platt", "temperature", "beta", "isotonic", "ridge_platt"})
SLOPE_BOUNDS = (0.001, 100.0)
INTERCEPT_BOUNDS = (-20.0, 20.0)
SOLVER_GRADIENT_TOLERANCE = 2e-12
SOLVER_PARAMETER_TOLERANCE = 2e-12


def _logits(z):
    z = np.asarray(z, dtype=np.float64)
    if z.ndim != 1 or not np.isfinite(z).all():
        raise ValueError("Expected a finite one-dimensional logit array")
    return z


def _inputs(z, y):
    z = _logits(z)
    y = np.asarray(y, dtype=np.float64)
    if not len(z) or y.shape != z.shape or not np.isin(y, [0.0, 1.0]).all():
        raise ValueError("Expected nonempty, aligned logits and binary labels")
    return z, y


def _penalty(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError("Penalty must be a finite nonnegative number")
    value = float(value)
    if not np.isfinite(value) or value < 0:
        raise ValueError("Penalty must be a finite nonnegative number")
    return value


def _identity(method="raw", *, reason=None, penalty=0.0):
    return {
        "method": method,
        "effective_method": "raw",
        "slope": 1.0,
        "intercept": 0.0,
        "success": reason is None,
        "fallback": reason is not None,
        "reason": reason,
        "penalty": float(penalty),
        "bound_hit": False,
    }


def _objective(theta, z, y, method, penalty):
    if method == "beta":
        log_p = -np.logaddexp(0.0, -z)
        negative_log_one_minus_p = np.logaddexp(0.0, z)
        u = theta[0] * log_p + theta[1] * negative_log_one_minus_p + theta[2]
        residual = expit(u) - y
        gradient = np.array([
            np.mean(residual * log_p),
            np.mean(residual * negative_log_one_minus_p),
            np.mean(residual),
        ])
    else:
        a = theta[0]
        b = 0.0 if method == "temperature" else theta[1]
        u = a * z + b
        residual = expit(u) - y
        gradient = [np.mean(residual * z)]
        if method != "temperature":
            gradient.append(np.mean(residual))
        gradient = np.array(gradient)
    value = np.mean(np.logaddexp(0.0, u) - y * u)
    if method == "ridge_platt" and penalty:
        value += penalty * ((theta[0] - 1.0) ** 2 + theta[1] ** 2)
        gradient += 2.0 * penalty * np.array([theta[0] - 1.0, theta[1]])
    return float(value), gradient


def fit(z, y, method, penalty=0.0):
    """Fit one label, returning a JSON-serializable prediction record.

    Invalid inputs raise. Missing class support or explicit optimizer failure
    return a documented raw-probability fallback; exceptions are not swallowed.
    """
    z, y = _inputs(z, y)
    if method not in METHODS:
        raise ValueError(f"Unknown calibration method: {method}")
    penalty = _penalty(penalty)
    if method != "ridge_platt" and penalty != 0.0:
        raise ValueError("A penalty applies only to ridge_platt")
    if method == "raw":
        return _identity()
    if len(np.unique(y)) != 2:
        return _identity(method, reason="single_class_calibration", penalty=penalty)
    if method == "isotonic":
        model = IsotonicRegression(increasing=True, out_of_bounds="clip")
        model.fit(z, y)
        return {
            "method": method, "effective_method": method,
            "success": True, "fallback": False, "reason": None,
            "x_thresholds": model.X_thresholds_.tolist(),
            "y_thresholds": model.y_thresholds_.tolist(),
            "penalty": 0.0, "bound_hit": False,
        }

    if method == "beta":
        initial = np.array([1.0, 1.0, 0.0])
        bounds = [SLOPE_BOUNDS, SLOPE_BOUNDS, INTERCEPT_BOUNDS]
    elif method == "temperature":
        initial = np.array([1.0])
        bounds = [SLOPE_BOUNDS]
    else:
        initial = np.array([1.0, 0.0])
        bounds = [SLOPE_BOUNDS, INTERCEPT_BOUNDS]
    if method == "beta":
        design = np.column_stack((-np.logaddexp(0.0, -z), np.logaddexp(0.0, z), np.ones(len(z))))
    elif method == "temperature":
        design = z[:, None]
    else:
        design = np.column_stack((z, np.ones(len(z))))
    result = solve(design, y, initial, bounds,
        penalty_center=initial, penalty=penalty if method == "ridge_platt" else 0.0,
        gradient_tolerance=SOLVER_GRADIENT_TOLERANCE,
        parameter_tolerance=SOLVER_PARAMETER_TOLERANCE)
    parameters = np.asarray(result["params"], dtype=float)
    finite = bool(np.isfinite(result["objective"]) and np.isfinite(parameters).all())
    diagnostics = {
        "status": 0 if result["success"] else 1, "message": result["message"],
        "iterations": int(result["iterations"]),
        "objective": float(result["objective"]),
        "initial_objective": float(result["initial_objective"]),
        "optimizer_success": bool(result["success"]),
        "solver": result["solver"],
        "gradient": [float(value) for value in result["gradient"]],
        "projected_gradient": float(result["projected_gradient"]),
        "convex_gap_bound": float(result["convex_gap_bound"]),
        "relative_parameter_step": float(result["relative_parameter_step"]),
        "coordinate_sweeps": int(result["coordinate_sweeps"]),
        "backtracks": int(result["backtracks"]),
    }
    if not result["success"] or not finite:
        reason = "optimizer_nonconvergence" if not result["success"] else "nonfinite_fit"
        return {**_identity(method, reason=reason, penalty=penalty), **diagnostics}
    record = {
        "method": method, "effective_method": method,
        "success": True, "fallback": False, "reason": None,
        "penalty": penalty,
        "calibration_log_loss": _objective(parameters, z, y, method, 0.0)[0],
        "bound_hit": bool(any(
            np.isclose(value, lo, atol=1e-7) or np.isclose(value, hi, atol=1e-7)
            for value, (lo, hi) in zip(parameters, bounds)
        )),
        **diagnostics,
    }
    if method == "beta":
        record.update(a=float(parameters[0]), b=float(parameters[1]), intercept=float(parameters[2]))
    else:
        slope = float(parameters[0])
        record.update(slope=slope, intercept=0.0 if method == "temperature" else float(parameters[1]))
        if method == "temperature":
            record["temperature"] = 1.0 / slope
    return record


def predict(z, record):
    """Apply a saved record; outputs remain independent binary probabilities."""
    z = _logits(z)
    method = record.get("effective_method", record.get("method"))
    if method not in METHODS:
        raise ValueError(f"Unknown prediction method: {method}")
    if method == "raw":
        return expit(z)
    if method == "isotonic":
        x = np.asarray(record["x_thresholds"], dtype=float)
        y = np.asarray(record["y_thresholds"], dtype=float)
        if (x.ndim != 1 or not len(x) or x.shape != y.shape or
                not np.isfinite(x).all() or not np.isfinite(y).all() or
                np.any(np.diff(x) <= 0) or np.any(np.diff(y) < 0) or
                np.any((y < 0) | (y > 1))):
            raise ValueError("Invalid isotonic thresholds")
        return np.interp(z, x, y)
    if method == "beta":
        a, b, c = float(record["a"]), float(record["b"]), float(record["intercept"])
        if not np.isfinite([a, b, c]).all() or a < 0 or b < 0:
            raise ValueError("Invalid beta parameters")
        u = -a * np.logaddexp(0.0, -z) + b * np.logaddexp(0.0, z) + c
    else:
        a, b = float(record["slope"]), float(record["intercept"])
        if not np.isfinite([a, b]).all() or a < 0:
            raise ValueError("Invalid sigmoid parameters")
        u = a * z + b
    probabilities = expit(u)
    if not np.isfinite(probabilities).all():
        raise ValueError("Nonfinite probabilities from saved calibration map")
    return probabilities


def select_ridge(z, y, folds, penalties=(0.0, 0.001, 0.01, 0.1, 1.0), tie_tolerance=1e-12):
    """Select shrinkage solely from out-of-fold calibration Brier loss.

    The caller groups artists before assigning fold IDs. Identity wins numerical
    ties; otherwise the strongest tied penalty wins. Evaluation data are absent.
    """
    z, y = _inputs(z, y)
    folds = np.asarray(folds)
    if (folds.shape != z.shape or not np.issubdtype(folds.dtype, np.integer) or
            np.any(folds < 0) or len(np.unique(folds)) < 2):
        raise ValueError("Expected aligned integer fold IDs with at least two folds")
    penalties = [_penalty(value) for value in penalties]
    if not penalties or len(set(penalties)) != len(penalties):
        raise ValueError("Penalty grid must be nonempty and contain unique values")
    tie_tolerance = _penalty(tie_tolerance)
    raw = expit(z)
    scores = [{"penalty": "identity", "brier": float(np.mean((raw - y) ** 2)),
               "inner_fallback_count": 0, "fold_fits": []}]
    oof_predictions = {"identity": raw.tolist()}
    for penalty in penalties:
        oof = np.empty(len(z), dtype=float)
        fold_fits = []
        for fold in np.unique(folds):
            held_out = folds == fold
            fitted = fit(z[~held_out], y[~held_out], "ridge_platt", penalty=penalty)
            oof[held_out] = predict(z[held_out], fitted)
            fold_fits.append({"fold": int(fold), "train_n": int((~held_out).sum()),
                              "validation_n": int(held_out.sum()), "fit": fitted})
        scores.append({
            "penalty": penalty, "brier": float(np.mean((oof - y) ** 2)),
            "inner_fallback_count": sum(int(row["fit"]["fallback"]) for row in fold_fits),
            "fold_fits": fold_fits,
        })
        oof_predictions[str(penalty)] = oof.tolist()
    minimum = min(row["brier"] for row in scores)
    tied = [row for row in scores if row["brier"] <= minimum + tie_tolerance]
    selected = max(tied, key=lambda row: float("inf") if row["penalty"] == "identity" else row["penalty"])
    penalty = selected["penalty"]
    if penalty == "identity":
        record = _identity("ridge_platt")
    else:
        record = fit(z, y, "ridge_platt", penalty=penalty)
    record.update({
        "selected_penalty": penalty,
        "oof_scores": scores,
        "oof_predictions": oof_predictions,
        "inner_fallback_count": sum(row["inner_fallback_count"] for row in scores),
        "selected_inner_fallback_count": selected["inner_fallback_count"],
        "final_fit_fallback": record["fallback"],
        "selection": {
            "criterion": "track_weighted_oof_brier",
            "tie_rule": "identity_then_strongest_penalty",
            "tie_tolerance": tie_tolerance,
            "fold_count": int(len(np.unique(folds))),
            "minimum_oof_brier": minimum,
            "selected_oof_brier": selected["brier"],
            "uses_evaluation_data": False,
        },
    })
    return record
