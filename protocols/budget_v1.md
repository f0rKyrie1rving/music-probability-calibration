# Calibration with limited artist data

Protocol established on 2026-10-01 before fitting the additional methods below.
This is a prospective execution plan for a retrospective/exploratory benchmark,
not a preregistration or a new unseen-data confirmation. Historical calibration
results motivated the question and were already known.

## Question and relation to previous work

Does identity-centered, cross-validated regularization reduce unnecessary changes
when calibration data are limited, especially for unweighted classifiers whose
raw probabilities previously performed reasonably well? Compare it with flexible
beta and isotonic maps, and with simpler existing controls. No method is assumed
to win, and this is an empirical comparison of existing ideas, not a new algorithm.

Earlier music experiments compared raw, weight offset, intercept, temperature,
sigmoid and OOF-selected blends. They also studied calibration budgets in five
historical splits. This experiment adds nonlinear methods and identity-centered
parameter regularization to the three-representation, two-weighting design.

## Fixed inputs and exposure

Use all 120 completed runs from `calibration_extension/20260927_v2` in the original
Music Classification repository: 20 artist-isolated splits, three representations
(Discogs-MAEST, MERT-v0, MERT-v1), two ambient-weighting settings. Scalers and heads
were already fitted on approximately 60% of the 1,206-song / 469-artist development
pool; calibration and evaluation each use approximately 20%, with disjoint artist
IDs within a split. Encoders, classifiers and original splits stay fixed here.

The standalone input bundle contains IDs, artist groups, binary source-derived
labels, logits, original control predictions and training-count offsets. Verify
source run receipts, hashes, target alignment and disjointness before export.
Do not distribute audio, pretrained weights or full embedding caches in this bundle.
The original metadata terms/attribution remain applicable; the software license
does not relicense third-party material.

All these songs have been observed in earlier studies. The 239-, 266-, 204- and
372-song later cohorts are also previously observed and are not new test sets.
Carry forward the latest ledger excluding 2,835 tracks and 1,296 artist IDs for
future confirmation. Artist aliases and encoder pretraining overlap remain unknown.

## Budgets, pairing and selection

Within each of the 20 calibration artist pools, sort artist IDs and use a seeded
random permutation (NumPy SeedSequence: outer seed and budget seed). Take nested
ceil(25%), ceil(50%) and 100% artist prefixes, keeping every song of each selected
artist. This is one draw per outer split, not an estimate of within-split budget
sampling variability. Use identical subsets across representations and weightings.
No target balancing or redraw is allowed. Save every ID and split before fitting.

Assign each selected artist to one of three OOF folds via a second seeded
permutation (outer seed, inner seed, budget index), followed by round-robin fold
assignment. All songs of an artist remain together. Ridge strength is selected
separately for each label from these calibration folds; no evaluation inputs enter
fitting or selection. Averaged OOF Brier is track-weighted, matching the main metric.
Thus there are 20 x 3 x 2 x 3 = 360 dataset/budget cases.

## Seven methods

- `raw`: sigmoid of the original head logit, no calibration data.
- `prior_offset`: in the ambient-balanced condition only, add
  log(n_positive / n_negative) from classifier-fit labels to the ambient logit;
  other labels and the unweighted condition are exact raw controls. This idealized
  class-weight correction need not reconstruct an unweighted regularized model.
- `temperature`: sigmoid(a z), a in [0.001, 100], unweighted mean calibration NLL.
- `platt`: sigmoid(a z + b), a in [0.001, 100], b in [-20, 20], same loss and
  optimizer settings as the historical sigmoid baseline. The map acts on logits
  and includes identity; do not transfer the beta paper's probability-input
  logistic-family identity limitation to this logit-input implementation.
- `beta`: sigmoid(a log(p) - b log(1-p) + c), p = sigmoid(z), a and b in
  [0.001, 100], c in [-20, 20]; compute log terms stably from z. This explicitly
  bounded monotone implementation may differ from other beta-calibration software.
- `isotonic`: nondecreasing least-squares mapping of z, clipped at calibration
  endpoint values outside the fitting range. Its plateaus may change ranking ties.
- `ridge_platt`: minimize mean NLL + lambda * ((a-1)^2 + b^2), same bounds as
  Platt. Three-fold artist OOF Brier selects among lambda in {0, .001, .01, .1, 1}
  and exact identity. Within 1e-12 of the best choose the strongest penalty, with
  identity first. Refit on the full selected calibration subset. This policy has
  no guarantee of improving evaluation loss; it is not described as "safe".

Parametric fits use L-BFGS-B, maxiter 2000, ftol 1e-12, gtol 1e-8.
All labels are independent binary outputs; never normalize across the four labels.

## Failures and reporting

For a single-class calibration target or explicit optimizer nonconvergence, use
identity for that label and log the reason, including failures within OOF selection.
Report policy-wide scores, successful-fit counts, selected strengths and fallback
counts. Never hide data errors or programming errors behind fallback. Do not
replace failed cases with alternative splits, methods, random seeds or datasets.

Primary: mean binary Brier across four labels, track-weighted within each split.
Secondary: per-label Brier, binary log loss (p clipped to [1e-15,1-1e-15] for scoring),
AP, and fixed 5/10-bin reliability tables (counts, mean prediction and label fraction).
Brier combines discrimination and calibration; lower Brier alone does not prove
every probability range is calibrated. Reliability bins are descriptive and noisy.

Report every representation x weighting x budget x method cell: mean/median/min/max
paired Brier changes versus raw, and wins/ties/losses at tolerance 1e-12. Compare
ridge with ordinary Platt as a second prespecified descriptive contrast. Separate
weighted and unweighted results. Do not treat overlapping splits as independent,
construct population confidence intervals from their spread, perform split-level
t-tests, pool repeated predictions as independent songs, or select a deployment
method based on the evaluation scores. Negative and null results remain in reports.

## Reproducibility gates

Freeze source, protocol, config, input checksums and all budget/fold IDs before fitting.
At full budget, reproduce historical raw/prior-offset/Platt/temperature probabilities
(maximum difference 1e-8); verify all three unaffected-label weighting controls.
Keep numerical reconstruction separate from fitting: verify every saved probability
from saved parameters, every Brier/log-loss result and every summary from saved rows.
Run tests for artist isolation, nested budgets, leakage-safe selection, calibration
monotonicity/extreme values, exact identity, failure records and tamper detection.
Reproduction of old predictions is an implementation control, not new evidence.

## Primary references

- Kull, Silva Filho and Flach (2017), [Beta calibration](https://proceedings.mlr.press/v54/kull17a.html).
- Caplin, Martin and Marx (2022), [Calibrating for Class Weights by Modeling Machine Learning](https://arxiv.org/abs/2205.04613).
- Ye et al. (2022), [Uncertainty Calibration for Deep Audio Classifiers](https://arxiv.org/abs/2206.13071).

These works establish that calibration and class-weight distortion are prior art.
The research contribution sought here is a reproducible, qualified empirical result.
