# How sensitive are calibration results to which artists are selected?

Execution plan established on 2026-10-02, after the complete budget_v1 results
were known and before fitting these additional draws. This is an exploratory
robustness extension on the same exposed development data, not confirmation on
new songs. It changes neither the application nor the frozen budget_v1 engine.

## Question

The first benchmark used one selected artist subset per outer split and budget.
Its 20 overlapping outer splits did not measure variation across different
subsets of the same calibration pool. Check whether method comparisons change
when we select five different subsets, retaining unfavorable results and failures.
No best seed, method, penalty grid, threshold or publication claim is selected.

## Fixed design

Use all 120 existing source cases: 20 outer splits, three representations, two
training-weight conditions. Each original calibration pool has 94 artists.
Retain all seven methods, the existing artist budget fractions 25%, 50%, 100%,
the optimizer settings, ridge candidates, inner seed 2026100102 and three-fold
OOF procedure. Classifier-fit and evaluation rows, logits and labels do not change.

The five new budget seeds are exactly 2026100201 through 2026100205. For each
child run, change only budget_seed in the original budget_v1 configuration.
The original budget seed 2026100101 is a historical reference and is not included
as a sixth draw. The unchanged engine permutes artists using the outer seed and
budget seed, taking nested prefixes of 24, 47 and 94 artists. Every selected
artist contributes all their songs. The same subsets and folds apply across
representations and weighting conditions for a given outer split and draw.

Holding the inner seed fixed preserves the assignment procedure, not necessarily
each artist's fold membership: changing the selected artists changes the input
to the existing inner-fold permutation. Therefore this experiment measures
subset and resulting selection stability under a fixed fold rule. It does not
separate a pure sample-composition effect from the resulting fold changes.

Five complete child runs execute 5 x 360 = 1,800 cases. Their 100% subsets and
folds are identical, making 120 x (5 x 2 + 1) = 1,320 distinct source/subset cases.
Before fitting, check that the five partial-budget subsets differ within every
outer split; do not redraw if this fails. The repeated full-budget predictions,
calibrator records and metrics must have identical hashes. Full-budget Brier and
log-loss values must also reproduce the original report within 1e-8, allowing
small cross-platform numerical differences. This implementation-control tolerance
is separate from the descriptive win/tie/loss threshold of 1e-12. Within the five
new runs on one platform, the repeated full-budget file hashes must match exactly.

The repeated full-budget point is a numerical control. Aggregate it once per
outer split, never as five independent observations. The original 25%/50%
results remain separately available in reports/budget_v1 and are not pooled into
the new estimates. No new audio, labeling, encoder fitting or head fitting occurs.

## Freeze and execution

Freeze this script, protocol, study configuration, all original engine sources,
the original protocol/report artifacts, the score-bundle manifest and all five
child plans before any child fit. Each child uses the existing freeze, run,
summarize and independent verify functions. No monkeypatching or changed engine
code is allowed. Modified sources, receipts or completed results stop execution.
All original budget_v1 artifacts remain unchanged.

Each child keeps the original explicit identity fallback for single-class targets
and recorded optimizer nonconvergence. Programming errors, corruption or an
incomplete case stop and retain that run. Do not silently redraw or omit a child.
An interrupted child may be recovered with the separately documented recovery
workflow; this study script never deletes or silently replaces a failed case.

## Outcomes and descriptive summaries

Primary: macro binary Brier minus raw, with the original track weighting within
each evaluation set. The second contrast is Brier minus ordinary Platt. Report
every method and label; child outputs retain log loss, AP and reliability tables.

For each outer split x representation x weight x partial budget x method,
describe the five paired draw results by mean, median, minimum, maximum and
range. Count improvements, ties and deteriorations relative to raw using an
absolute tolerance of 1e-12. A sign flip requires at least one improvement and
one deterioration; a tie alone is not a flip. Repeat the contrast against Platt.

For each of the 18 representation x weighting x budget conditions and seven
methods, average the 20 outer-split means equally. Report the distribution of
within-split ranges, numbers of splits with sign flips, and numbers for which
all draws improve or all deteriorate. The full-budget endpoint uses one draw.
Report ridge strength changes, identity choices and fallback counts separately;
these are diagnostics, not a rule for selecting a new deployed method.

Do not treat 100 partial-budget evaluations as 100 independent datasets or the
five draws as new independent songs. No population confidence interval, t-test,
unbiased future success rate or minimum adequate sample size is inferred.
Five draws provide a bounded sensitivity check, not an exhaustive variability
estimate. Brier is a probability-quality score, not classification accuracy or
proof of reliability within every probability bin.

All 1,206 songs and 469 artist IDs remain previously exposed. Source-tag noise,
sample selection, artist aliases and encoder pretraining overlap remain limits.
The exclusion ledger of 2,835 tracks and 1,296 artists continues to apply to any
future genuinely new project-disjoint confirmation.
