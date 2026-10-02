# Numerical-v1 to numerical-v2 changes

The new solver changes some fitted probability maps and three selected label calibrators. It does **not** uniformly improve evaluation Brier. This diagnostic compares saved macOS v1 and v2 runs on the same six previously exposed data plans; cross-platform reproduction is a separate requirement and is not established by this comparison.

The [machine-readable record](legacy_difference.json) retains both favorable and unfavorable changes. Runs are matched by their configured budget seed, with identical calibration plans, evaluation IDs/artists/targets, source-case hashes and score-bundle identity. Completed-artifact hashes are checked, and macro Brier is recomputed from every saved evaluation probability. No fitting or new-song evaluation occurs in this audit.

All six plans contribute 2,160 executions and 14,563,584 probability comparisons. Of these probabilities, 547,266 differ by more than `1e-8`. There are three selected-label changes and 46 changed inner-fallback locations; every changed fallback goes from v1 `optimizer_nonconvergence` to a successful v2 fit. The 1,560 unique source/subset conditions retain all six partial-budget draws, while the repeated full-budget endpoint is counted once per source case after exact prediction and policy hashes are checked. Songs and outer splits overlap.

Mean paired macro-Brier changes across those 1,560 conditions are below. A negative change means lower error on this exposed evaluation data. Small numerical changes are reported without claiming practical improvement.

| Method | Mean Brier change, v2 minus v1 |
|---|---:|
| Raw | 0 |
| Prior offset | 0 |
| Isotonic | 0 |
| Temperature | +4.63048e-11 |
| Ordinary Platt | -7.32615e-11 |
| Beta | -1.37719e-6 |
| Ridge Platt | **+3.52030e-6** |

Every execution appears in the changed-case record because it retains probability changes above `1e-8`, Brier changes above `1e-12`, and any selection or fallback change. This does not mean that all changes are large. Beta case-level Brier changes range from -0.00102503 to +0.000299724; the largest Ridge deterioration is +0.00272175633.

## Why the three selections changed

All three changes occur at the 25% calibration budget.

- **Budget seed 2026100101; outer seed 2026092809; MERT-v1, unweighted, ambient:** v1 selected penalty 1.0, while v2 selects identity. The v1 penalty-1.0 fit in inner fold 0 returned status 2, `ABNORMAL`, and used the prescribed identity fallback. V2 fits that fold successfully, with recorded projected-gradient residual `2.22e-16`. The candidate's OOF Brier changes from 0.1505029356 to 0.1509340398, above the unchanged identity score 0.1505259692. Evaluation macro Brier increases by 0.0000481995.
- **Budget seed 2026100205; outer seed 2026092818; MERT-v0, rock, under both weighting conditions:** v1 selected penalty 0.0; v2 selects identity. In inner fold 0, the 42 training songs, including four positives, are separable. V1 stopped at slope 29.0587 and intercept 13.5410, with training loss about `1.25e-8`. V2 reaches the existing upper bounds, slope 100 and intercept 20, with training loss about `2.41e-13` and a recorded zero projected gradient. The unpenalized candidate's OOF Brier rises from 0.0937867989 to 0.1437796681, above identity at 0.0942005113. This is a consequential change in held-out predictions despite the smaller training objective; it is not roundoff. The two weighting conditions share this rock-label behavior and are not independent examples. Their evaluation macro Brier increases by 0.00272175622 and 0.00272175633, respectively.

The largest individual probability change, 0.404575164, occurs in those two rock cases. For `track_1317901`, the final Ridge score changes from 0.877550445 to the identity score 0.472975281. The observed source target is 1. These retained unfavorable outcomes illustrate why more reliable numerical optimization must not be described as guaranteed predictive improvement.

The frozen v1 results remain unchanged. This audit does not select a new method using evaluation performance, certify source tags as human truth, or establish generalization to unseen songs. Cross-platform v2 acceptance must independently compare all evaluation and candidate OOF probabilities, selections and fallback decisions under its strict gate.

Run the README's historical-v1 and numerical-v2 reproduction commands first,
then compare their default output directories without overwriting this published
reference. The published comparison used macOS; historical v1 runs on other
systems can differ for the portability reasons documented in this project.

```sh
python scripts/audit_numerical_change.py \
  --legacy-budget runs/reproduction \
  --legacy-stability runs/stability_reproduction \
  --portable runs/portable_reproduction \
  --out runs/numerical_change_audit.json
```
