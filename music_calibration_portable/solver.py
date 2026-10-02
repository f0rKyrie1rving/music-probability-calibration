"""Small deterministic box-constrained logistic solver with explicit KKT checks.

The objective is mean binary logistic NLL plus ``penalty * ||theta-center||²``.
Only the numerical solver changes: neither the objective nor its box is relaxed.
Scalar compensated sums and a fixed-order Cholesky solve avoid BLAS reduction and
line-search status differences. Newton steps are polished with exact coordinate
brackets when curvature is singular or a floating-point line search stalls.
"""
import math

import numpy as np
from scipy.special import expit


def _sum(values):
    return math.fsum(float(value) for value in values)


def _linear(design, theta):
    value = np.zeros(len(design), dtype=np.float64)
    for column in range(design.shape[1]):
        value += design[:, column] * theta[column]
    return value


def _evaluate(design, y, theta, center, penalty):
    logits = _linear(design, theta)
    signs = 1.0 - 2.0 * y
    loss = np.logaddexp(0.0, signs * logits)
    residual = signs * expit(signs * logits)
    curvature = expit(logits) * expit(-logits)
    n, dimension = design.shape
    difference = theta - center
    objective = _sum(loss) / n + penalty * _sum(difference * difference)
    gradient = np.array([_sum(residual * design[:, j]) / n + 2 * penalty * difference[j]
                         for j in range(dimension)], dtype=np.float64)
    hessian = np.empty((dimension, dimension), dtype=np.float64)
    for row in range(dimension):
        for column in range(row + 1):
            value = _sum(curvature * design[:, row] * design[:, column]) / n
            if row == column:
                value += 2 * penalty
            hessian[row, column] = hessian[column, row] = value
    return objective, gradient, hessian


def _projected(gradient, theta, lower, upper):
    result = gradient.copy()
    result[(theta <= lower) & (gradient >= 0)] = 0.0
    result[(theta >= upper) & (gradient <= 0)] = 0.0
    return result


def _cholesky_solve(matrix, rhs):
    """Fixed-order arithmetic, returning None for numerically singular curvature."""
    n = len(rhs)
    result = np.zeros((n, n), dtype=np.float64)
    scale = max(float(matrix[i, i]) for i in range(n))
    if scale <= 0:
        return None
    for row in range(n):
        for column in range(row + 1):
            value = float(matrix[row, column]) - math.fsum(
                float(result[row, k] * result[column, k]) for k in range(column))
            if row == column:
                if value <= scale * 1e-14:
                    return None
                result[row, column] = math.sqrt(value)
            else:
                result[row, column] = value / result[column, column]
    intermediate = np.empty(n, dtype=np.float64)
    answer = np.empty(n, dtype=np.float64)
    for row in range(n):
        intermediate[row] = (rhs[row] - math.fsum(
            float(result[row, k] * intermediate[k]) for k in range(row))) / result[row, row]
    for row in reversed(range(n)):
        answer[row] = (intermediate[row] - math.fsum(
            float(result[k, row] * answer[k]) for k in range(row + 1, n))) / result[row, row]
    return answer


def _newton_direction(gradient, hessian, theta, lower, upper):
    projected = _projected(gradient, theta, lower, upper)
    free = [j for j in range(len(theta)) if not (
        (theta[j] <= lower[j] and gradient[j] >= 0) or
        (theta[j] >= upper[j] and gradient[j] <= 0))]
    direction = np.zeros(len(theta), dtype=np.float64)
    while free:
        solution = _cholesky_solve(hessian[np.ix_(free, free)], -gradient[free])
        if solution is None:
            return None
        direction[:] = 0.0
        direction[free] = solution
        outward = [j for j in free if (theta[j] <= lower[j] and direction[j] < 0)
                   or (theta[j] >= upper[j] and direction[j] > 0)]
        if not outward:
            if _sum(gradient * direction) >= 0 and np.any(projected != 0):
                return None
            return direction
        free = [j for j in free if j not in outward]
    return direction if not np.any(projected) else None


def _coordinate_sign(design, y, theta, center, penalty, column):
    """Derivative sign remains usable even when logistic residuals underflow."""
    logits = _linear(design, theta)
    signs = (1.0 - 2.0 * y) * np.sign(design[:, column])
    nonzero = signs != 0
    if not np.any(nonzero):
        return float(np.sign(theta[column] - center[column])) if penalty else 0.0
    logs = (np.log(np.abs(design[nonzero, column])) - math.log(len(y))
            - np.logaddexp(0.0, -(1.0 - 2.0 * y[nonzero]) * logits[nonzero]))
    log_values = list(logs)
    sign_values = list(signs[nonzero])
    prior = 2 * penalty * (theta[column] - center[column])
    if prior:
        log_values.append(math.log(abs(prior)))
        sign_values.append(math.copysign(1.0, prior))
    shift = max(log_values)
    return math.fsum(float(sign) * math.exp(float(value) - shift)
                     for sign, value in zip(sign_values, log_values))


def _coordinate_sweep(design, y, theta, lower, upper, center, penalty, parameter_tolerance):
    result = theta.copy()
    for column in range(len(theta)):
        trial = result.copy()
        trial[column] = lower[column]
        low_sign = _coordinate_sign(design, y, trial, center, penalty, column)
        trial[column] = upper[column]
        high_sign = _coordinate_sign(design, y, trial, center, penalty, column)
        if low_sign == 0 and high_sign == 0:
            continue  # A flat coordinate retains its current, deterministic value.
        if low_sign >= 0:
            result[column] = lower[column]
            continue
        if high_sign <= 0:
            result[column] = upper[column]
            continue
        lo, hi = float(lower[column]), float(upper[column])
        for _ in range(80):
            midpoint = lo + (hi - lo) / 2
            trial[column] = midpoint
            derivative = _coordinate_sign(design, y, trial, center, penalty, column)
            if derivative < 0:
                lo = midpoint
            elif derivative > 0:
                hi = midpoint
            else:
                lo = hi = midpoint
            if hi - lo <= parameter_tolerance * (1 + abs(midpoint)):
                break
        result[column] = lo + (hi - lo) / 2
    return result


def solve(design, y, initial, bounds, penalty_center=None, penalty=0.0,
          *, max_iterations=300, gradient_tolerance=2e-12, parameter_tolerance=2e-12):
    """Return JSON-ready parameters and an explicit constrained-stationarity certificate.

    Success requires both the box KKT residual and a negligible Newton/coordinate
    step. A small gradient alone is insufficient on nearly separated data. A
    failed certificate is returned as a failure, never silently accepted.
    """
    design = np.asarray(design, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    theta = np.asarray(initial, dtype=np.float64).copy()
    bounds = np.asarray(bounds, dtype=np.float64)
    if (design.ndim != 2 or design.shape[1] not in (1, 2, 3) or not len(design)
            or y.shape != (len(design),) or theta.shape != (design.shape[1],)
            or bounds.shape != (design.shape[1], 2) or not np.isfinite(design).all()
            or not np.isfinite(theta).all() or not np.isfinite(bounds).all()
            or not np.isin(y, [0.0, 1.0]).all()):
        raise ValueError('Expected finite aligned data, 1-3 parameters, and binary targets')
    if (not np.isfinite(penalty) or penalty < 0 or type(max_iterations) is not int
            or max_iterations < 1 or not np.isfinite(gradient_tolerance) or gradient_tolerance <= 0
            or not np.isfinite(parameter_tolerance) or parameter_tolerance <= 0):
        raise ValueError('Invalid solver tolerances, iteration count, or penalty')
    lower, upper = bounds[:, 0], bounds[:, 1]
    if np.any(lower >= upper) or np.any(theta < lower) or np.any(theta > upper):
        raise ValueError('Initial parameters must lie inside nonempty finite bounds')
    center = np.zeros_like(theta) if penalty_center is None else np.asarray(penalty_center, dtype=np.float64)
    if center.shape != theta.shape or not np.isfinite(center).all():
        raise ValueError('Penalty center must match the finite parameter vector')
    initial_objective = _evaluate(design, y, theta, center, penalty)[0]
    coordinate_sweeps, backtracks, success = 0, 0, False
    last_step = float('inf')
    for iteration in range(max_iterations):
        objective, gradient, hessian = _evaluate(design, y, theta, center, penalty)
        projected = _projected(gradient, theta, lower, upper)
        kkt = float(np.max(np.abs(projected)))
        direction = _newton_direction(gradient, hessian, theta, lower, upper)
        if direction is not None:
            last_step = float(np.max(np.abs(np.clip(theta + direction, lower, upper) - theta) / (1 + np.abs(theta))))
            if kkt <= gradient_tolerance and last_step <= parameter_tolerance:
                success = True
                break
        candidate = None
        if direction is not None and np.any(direction):
            alpha = 1.0
            for j, delta in enumerate(direction):
                if delta > 0:
                    alpha = min(alpha, (upper[j] - theta[j]) / delta)
                elif delta < 0:
                    alpha = min(alpha, (lower[j] - theta[j]) / delta)
            descent = _sum(gradient * direction)
            for _ in range(55):
                trial = np.clip(theta + alpha * direction, lower, upper)
                value, trial_gradient, _ = _evaluate(design, y, trial, center, penalty)
                trial_kkt = float(np.max(np.abs(_projected(trial_gradient, trial, lower, upper))))
                rounding = 4 * np.finfo(float).eps * max(1.0, abs(objective))
                if (value <= objective + 1e-4 * alpha * descent
                        or (abs(value - objective) <= rounding and trial_kkt < kkt)):
                    candidate = trial
                    break
                alpha *= .5
                backtracks += 1
        if candidate is None or np.array_equal(candidate, theta):
            candidate = _coordinate_sweep(design, y, theta, lower, upper, center, penalty, parameter_tolerance / 4)
            coordinate_sweeps += 1
            last_step = float(np.max(np.abs(candidate - theta) / (1 + np.abs(theta))))
            if kkt <= gradient_tolerance and last_step <= parameter_tolerance:
                theta = candidate
                success = True
                break
        theta = candidate
    objective, gradient, _ = _evaluate(design, y, theta, center, penalty)
    projected = _projected(gradient, theta, lower, upper)
    kkt = float(np.max(np.abs(projected)))
    success = bool(success and kkt <= gradient_tolerance and np.isfinite(objective)
                   and objective <= initial_objective + 4 * np.finfo(float).eps * max(1.0, abs(initial_objective)))
    gap = _sum(np.maximum(gradient * (theta - lower), gradient * (theta - upper)))
    return {'params': theta.tolist(), 'success': success, 'objective': float(objective),
            'initial_objective': float(initial_objective), 'projected_gradient': kkt,
            'gradient': gradient.tolist(), 'convex_gap_bound': max(0.0, float(gap)),
            'iterations': iteration + 1, 'coordinate_sweeps': coordinate_sweeps,
            'backtracks': backtracks, 'relative_parameter_step': float(last_step),
            'gradient_tolerance': gradient_tolerance, 'parameter_tolerance': parameter_tolerance,
            'message': 'KKT and parameter-step certified' if success else 'Convergence certificate not reached',
            'solver': 'bounded-logistic-newton-v1'}
