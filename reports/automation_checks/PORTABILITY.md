# Numerical portability findings

The published reference runs were generated on macOS arm64. The new checks
separate successful execution and independent arithmetic verification from
reproducing the same fitted policy on another platform. **Cross-platform
reproduction is not fully established. The failing comparison gates remain
enabled; their tolerances have not been widened.**

The tested fitting engine, score bundle, plans and dependency versions are
unchanged. The CI observations below belong to commit
`12c1a85e452c2d85af8ff2fbb20c67016f4c7382`.

| Study | Platform | Execution and numerical reconstruction | Comparison to published macOS results |
|---|---|---|---|
| Original 360 cases | macOS | Passed | Passed |
| Original 360 cases | Linux | Passed | Passed with four explicitly recorded unselected-fit diagnostic differences |
| Original 360 cases | Windows | Passed | Failed: three selected calibrators differ |
| Five draws / 1,800 cases | macOS | Passed | Passed |
| Five draws / 1,800 cases | Linux | Passed for all five child runs | Failed: one selected calibrator differs |

The original-study CI is [run 36973137078](https://github.com/f0rKyrie1rving/music-probability-calibration/actions/runs/36973137078).
The five-draw CI is [run 36973137034](https://github.com/f0rKyrie1rving/music-probability-calibration/actions/runs/36973137034).
The latter also exposed a separate audit-script error: its historical-control
comparison assumed zero error. The corrected checker independently recomputes
that value against the hash-bound original table, retains the declared `1e-8`
historical limit, and verifies the saved scalar within `1e-12`. Its successful
recheck of the downloaded Linux report is saved alongside the original failure.

## Observed differences

- The original Linux benchmark's four diagnostic differences affect unsuccessful
  **unselected** inner fits only. Selected penalties and selected/final fallbacks
  match; the largest metric difference is `3.48e-10`. The optional diagnostic
  exception requires complete row inventories and consistent totals, and records
  every difference. The default comparator remains strict.
- The Windows benchmark changes 3 of 1,440 per-label ridge choices, affecting
  3 of 2,520 method-case rows. The maximum macro-Brier difference is `4.82e-5`
  for an individual case and `2.41e-6` for a summary cell. The archive does not
  contain individual Windows optimizer records, so exact fold/status messages
  are not established.
- The Linux five-draw study changes one of 7,200 per-label ridge choices:
  draw 5, split `2026092819`, MERT-v1, unweighted, 50% budget, ambient. The
  selected map changes from identity to penalty 1. The affected summary mean
  macro-Brier differs by `1.619e-6`. Saved fit records show that one penalty-1
  inner fold returned `ABNORMAL` on Linux and triggered the prescribed identity
  fallback. Its changed out-of-fold score changed the winner; the final refit
  succeeded. The lower-level library/hardware cause has not been established.

All 18 ridge-versus-raw and ridge-versus-Platt **mean comparison directions**
remain unchanged in these checks. That does not excuse the reproduction
failures or establish equivalence on other hardware. The numeric and selection
differences are retained, including those that do not change the broad narrative.

## Evidence and interpretation

- [Downloaded CI artifact provenance](ci_artifact_provenance.json)
- [Linux original-study regression](linux_budget_regression.json)
- [Windows original-study diagnosis](windows_budget_diagnosis.json)
- [Complete Linux five-draw differences](linux_stability_portability.json)
- [Linux five-draw fit diagnosis](linux_stability_fit_diagnosis.json)
- [Corrected independent Linux aggregation audit](linux_stability_aggregation_check.json)
- [Original local aggregation receipt, retained unchanged](initial_mac_aggregation_check.json)

The new portability auditor reports `matched`, `not_reproduced`, or evidence
`error`; it never overrides the strict regression gate. Future numerical-solver
changes need a separate frozen procedure and new results. The historical fits
and the released music application have not been modified to conceal this issue.
