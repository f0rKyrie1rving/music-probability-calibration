# Numerical portability of the calibration benchmark

Execution protocol established on 2026-10-02, before the six complete v2 runs.
This is a numerical reproducibility study of already observed development data.
It is neither a new calibration method nor a fresh-data confirmation. The v1
Windows/Linux discrepancies were known when this work was designed.

## Question and preserved evidence

Can the same bounded calibration objectives produce the same probability maps,
ridge choices and fallback decisions across macOS, Linux and Windows when solved
with explicit, deterministic numerical convergence checks?

The v1 implementation used L-BFGS-B. Its platform-dependent nonconvergence in a
small number of calibration folds changed fallback predictions, OOF Brier scores
and, in some cases, selected ridge strengths. Those results and their failed
strict comparisons remain part of the research record. Preserve the entire v1
package, protocols, frozen runs and reports. Implement v2 in the separate
`music_calibration_portable` package and save results in separate directories.

V2 changes only the numerical solver and the diagnostics needed to verify it.
The data, heads, outer splits, nested artist budgets, OOF fold assignment,
objectives, parameter bounds, initialization, method grid and selection rule are
unchanged. Do not choose a solver setting to improve evaluation scores, reproduce
a favorable v1 conclusion, or change which methods are reported. Calibration-only
regression fixtures and convergence scans may be used before freezing v2 to check
the numerical implementation; they do not provide an independent evaluation.

## Fixed data and six runs

Inherit the input/exposure/label limitations from `budget_v1.md`: 120 source cases
cover 20 historical artist-disjoint outer splits, three frozen representations
and two weighting settings. The 1,206 songs and 469 artists have already been
observed. Source-derived binary genre tags are imperfect reference labels;
artist aliases and encoder-pretraining overlap remain unknown. No new recordings,
embeddings, labels or test cohort are acquired here. Carry forward the existing
future-confirmation exclusion ledger.

Run the original budget seed 2026100101 and the five previously declared seeds
2026100201 through 2026100205, named `budget_original`, then `draw_01` through
`draw_05`. Retain inner seed 2026100102, three artist OOF folds and budgets 25%,
50% and 100%. Between child configs, only `budget_seed` changes. All six configs
and all track/artist/fold plans must be frozen before any complete run is fitted.

This requires 6 x 120 x 3 = **2,160 case executions**, representing **1,560
distinct source/subset cases**: six partial-budget draws at each of two budgets,
plus one full-budget endpoint for each source case. The six full-budget copies
are identical-input negative controls, not six independent observations. Their
saved calibrator, prediction and metric hashes must match within each OS. Count
the endpoint once per source case in any aggregate across draws. Report the
original draw separately from the five-draw stability summary.

The same fixed fold-assignment procedure is reused. Changing the selected subset
can change which artists fall into each OOF fold; this study does not isolate
subset randomness from the selection procedure. Evaluation labels never enter
fitting, ridge selection or any numerical convergence decision.

## Unchanged statistical methods

Keep all seven methods: raw, fit-count prior offset, temperature, logit-input
Platt, monotone beta, isotonic and identity-centered ridge Platt. Preserve the
exact identity candidate. Fit four labels independently without normalization.

Temperature minimizes unweighted mean binary NLL with slope in [0.001, 100].
Platt uses slope in [0.001, 100] and intercept in [-20, 20]. Ridge Platt adds
lambda times `((slope - 1)^2 + intercept^2)` to that mean loss. Beta minimizes
the same NLL for `a log(p) - b log(1-p) + c`, with a and b in [0.001, 100],
c in [-20, 20], and log terms computed stably from the input logit. Initial
parameters remain the identity map. Isotonic fitting and out-of-range clipping,
raw sigmoid and fit-count prior offset remain unchanged.

For each label, three-fold artist OOF Brier selects among lambda 0, .001, .01,
.1, 1 and exact identity. Within the unchanged 1e-12 score tie tolerance, prefer
identity, then the strongest penalty. Refit the chosen finite penalty on the
whole selected calibration subset. This policy does not guarantee improved
evaluation loss. All seven methods and all cases remain in the reports.

## Numerical solver and diagnostics

Use `bounded-logistic-newton-v1`: box-constrained Newton updates, stable logistic
loss/residuals, analytic gradient and Hessian, compensated scalar summation and a
fixed-order Cholesky solve. If curvature is numerically singular or a line search
stalls, polish with fixed-order coordinate derivative-sign bisection. The loss is
mean NLL plus the original penalty; no additional penalty, altered box, data
scaling, label perturbation or relaxed statistical objective is introduced.

Freeze these constants before the six full runs:

- Maximum 300 outer iterations.
- Box-projected gradient infinity norm at most 2e-12 and relative proposed
  parameter step `max(abs(step) / (1 + abs(theta)))` at most 2e-12.
- Armijo constant 1e-4; at most 55 line-search attempts with step halving.
- Cholesky pivot floor 1e-14 times the largest Hessian diagonal.
- At most 80 bisection steps per coordinate; bracket tolerance is one quarter
  of the relative parameter tolerance.
- A loss change within `4 * float64_epsilon * max(1, abs(current_loss))` may
  be accepted only when the projected-gradient residual decreases. Final loss
  must not exceed its initial value by more than the same roundoff allowance.

Success requires finite parameters/objective, feasible bounds, the gradient and
parameter-step checks, and the initial-to-final objective check. Save the objective,
initial objective, gradient, projected-gradient norm, relative step, iteration
and backtrack counts, coordinate-sweep count and convex box-gap diagnostic.
For a convex differentiable objective, the first-order box gap bounds objective
suboptimality in exact arithmetic; the saved value uses floating-point derivatives
and is a diagnostic, not an interval-certified mathematical proof.

Poorly identified or rank-deficient designs need not determine unique coefficients.
Therefore coefficient equality and matching iteration counts are diagnostic only.
Each fit must satisfy its own convergence and feasibility checks, reconstruct its
saved probabilities, and pass the probability-level comparison below. Do not infer
parameter uniqueness or universal cross-platform determinism from this corpus.

## Failures and controls

Retain the explicit identity fallback for a single-class calibration target or
numerical nonconvergence, with its reason saved at every inner and final fit.
Data, programming and nonfinite-input errors abort. No redraw, skipped case,
hidden retry with different settings or alternative seed is allowed.

For **v2 acceptance**, the number of numerical-failure fallbacks must be **zero**
over all six complete runs, including unselected ridge candidates. A failed fit
is evidence that v2 has not met its acceptance criterion; recording a fallback
does not make it acceptable. Single-class fallbacks are permitted only when their
locations, flags, reasons and counts match exactly across systems. This stricter
v2 acceptance condition does not rewrite the historical v1 failure policy.

For each paired weighting condition, electronic, pop and rock use identical
source logits/targets and therefore must have exactly matching evaluation
predictions for every method; their ridge OOF probabilities and selected choices
must also match exactly. Full-budget raw and prior-offset predictions must still
match the saved historical controls within absolute 1e-8. Save historical Platt
and temperature differences as diagnostics: a new minimizer is allowed to reach a
different fit, so matching an old numerical failure is not an acceptance target.

## Three-system portability acceptance

Generate the complete six-run v2 reference on macOS arm64 first. Capture exact OS,
architecture, Python, dependency and input/source versions. Independently execute
the same six frozen plans on Linux and Windows with the declared dependency
versions. All three systems must pass the independent reconstruction checks and
the following comparisons against the complete macOS reference:

- Every saved evaluation probability for all seven methods and four labels:
  absolute difference at most **1e-8**, with zero relative tolerance.
- Every OOF prediction for every ridge candidate, including identity and
  unselected penalties, plus each candidate's OOF Brier: the same **1e-8** bound.
- Exact experimental identities, track/target/fold alignment, selected penalties,
  fallback locations/reasons/flags/counts, and report discrete fields. There is
  no exception for unselected-candidate fallback drift.
- All seven numerical report files for each run (`summary.json`,
  `summary_cells.csv`, `case_metrics.csv`, `label_metrics.csv`,
  `ridge_selections.csv`, `example_reliability.csv`, `fallbacks.json`): exact
  schemas and identities; all continuous measurements within **1e-8**. This
  includes AP and ECE, even though ties and bin boundaries can be sensitive.

Do not average away mismatches, compare only the winning candidate, relax the
tolerance after seeing results, or call portability successful with a failed
required system. Record the largest observed errors, exact-field differences,
failed gates and system receipts. The supported claim is restricted to these
data, plans, dependency versions and tested environments, not all future inputs
or all hardware. If a frozen implementation fails, preserve it; a further solver
revision requires another versioned freeze and complete rerun.

## Freeze and reporting

Hash the data bundle; parent and child protocols; all six child configs/plans;
portable package; execution driver; and comparison/verification sources before
complete fitting. Verify current files against that freeze before execution and
reporting. Save results separately from v1. Interrupted execution may resume only
the same frozen configuration; completed cases cannot silently be replaced.

Report v2-versus-v1 prediction/selection/fallback differences as a numerical
implementation diagnostic alongside the three-system v2 comparisons. Do not
force v2 to reproduce the v1 selected model or its scores. Retain all unfavorable
changes. Descriptive research summaries keep the original Brier/AP/reliability
definitions and paired contrasts. Overlapping outer splits and repeated songs
do not become independent samples, and no population confidence interval is
constructed from their spread. This task makes no application/model upgrade,
fresh-data accuracy claim, or publication novelty claim.

## References for the numerical checks

- [SciPy L-BFGS-B documentation](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html)
  describes the previous optimizer's stopping criteria.
- [Boyd and Vandenberghe, Convex Optimization](https://stanford.edu/~boyd/cvxbook/)
  gives the convex first-order optimality framework behind the gap diagnostic.
- Calibration-method references remain those in `budget_v1.md`; this solver
  replacement does not change the statistical methods' prior-art status.
