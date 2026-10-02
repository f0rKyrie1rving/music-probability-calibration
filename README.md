# Music Probability Calibration

**Can probability calibration avoid unnecessary changes when only a small
calibration set is available?** This research benchmark studies that question
using frozen music-classifier scores, artist-disjoint fitting/calibration/evaluation
partitions, and explicit controls for training class weights.

This is the separate research project following
[Music Classification v1.2](https://github.com/f0rKyrie1rving/music-classification/releases/tag/v1.2.0).
It does not change the released application. The studies are exploratory and use
previously observed MTG-Jamendo data; they are not new independent test sets.

The current implementation is **numerical v2**. It addresses platform-dependent
optimizer failures found in the original benchmark while preserving the original
statistical objectives and all historical results. This is a numerical
reproducibility improvement, not a claim of better music classification.

## What is compared

Seven probability treatments: raw scores, analytical class-weight offset,
temperature scaling, Platt scaling, beta calibration, isotonic regression, and
identity-centered Platt regularization selected by artist-grouped cross-validation.
They estimate four separate label probabilities: electronic, pop, ambient and rock.

Each budget draw contains **360 cases**: 20 historical artist splits × three audio
representations × two ambient-weight settings × three calibration-artist budgets
(25%, 50%, 100%). Every method uses the same evaluation rows within each case.
Within each source case, classifiers stay fixed. Numerical v2 repeats the original
draw and five declared follow-up draws: **2,160 case executions**, covering
**1,560 distinct source/subset conditions**. The full-budget endpoint is identical
across draws, serves as a numerical control, and counts once in pooled summaries.

The regularized method can select exact identity (retain the original score), but
has no guarantee against deterioration. Beta and isotonic are existing methods;
this project does not claim to invent probability calibration.

## Results and protocol

- [Numerical-v2 validation status and evidence](reports/numerical_v2/PORTABILITY.md)
- [Chinese explanation](reports/numerical_v2/SUMMARY_ZH.md) and
  [retained v1-to-v2 differences](reports/numerical_v2/LEGACY_DIFFERENCES.md)
- [Numerical-v2 protocol](protocols/numerical_v2.md),
  [child configuration](protocols/numerical_v2.json) and
  [six-run acceptance plan](protocols/numerical_study_v2.json)
- [Data attribution and exposure boundary](DATA.md)

**Validation status:** [the complete three-system run passed](https://github.com/f0rKyrie1rving/music-probability-calibration/actions/runs/36978254282).
Windows, Linux and macOS each passed 166 tests and all 2,160 case executions.
Every selected penalty and fallback decision matched; the largest probability
difference was `1.84e-13`, below the fixed `1e-8` limit. The linked validation
record preserves the exact environments, source hashes and comparison receipts.

The original evidence remains available unchanged:

- Budget v1: [full report](reports/budget_v1/REPORT.md),
  [Chinese explanation](reports/budget_v1/SUMMARY_ZH.md),
  [protocol](protocols/budget_v1.md),
  [independent verification](reports/budget_v1/verification.json) and
  [original clean-environment reproduction](reports/budget_v1/reproduction.json).
- Five-draw stability v1: [full report](reports/stability_v1/REPORT.md),
  [Chinese explanation](reports/stability_v1/SUMMARY_ZH.md) and
  [protocol](protocols/stability_v1.md). It studies subset and calibration-selection
  variability under a fixed fold-assignment rule, using the same observed data.
- [Historical v1 platform failures and diagnosis](reports/automation_checks/PORTABILITY.md):
  Windows and Linux optimizer failures changed some ridge choices. These failures
  motivated v2 and are preserved, not reclassified as acceptable roundoff.

![Historical budget-v1 calibration comparison](reports/budget_v1/budget_brier.png)

This figure shows **historical budget-v1 results**. Negative Brier changes mean
lower probability error. The curves average 20 overlapping splits of one dataset;
they are descriptive, not confidence intervals or independent replications. Lower
Brier alone does not establish calibration in every probability interval, nor
does it establish better genre recognition.

## Reproduce from this repository

The roughly 10 MB score bundle is included. This score-level experiment needs
neither audio downloads nor the original application's private feature caches,
PyTorch, pretrained encoder weights or a GPU. Full reproduction from raw audio
is a separate task requiring the original manifests, models and feature pipeline.

Use Python 3.13 and the pinned dependencies. Create and activate an environment
(on Windows, activate with `.venv\Scripts\Activate.ps1` in PowerShell):

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
python -m unittest discover -s tests -v
python scripts/reproduce_portable.py
```

The final command freezes **all six plans before fitting**, executes 2,160 cases,
writes six reports, and independently verifies the saved calculations. Defaults
are `runs/portable_reproduction` and `reports/portable_reproduction`. For a separate
copy, pass `--out runs/portable_02 --report reports/portable_02`.

Rerunning the command reuses verified completed cases. Interrupted cases owned
by the wrapper are archived with a recovery record before being refitted from
their frozen plans. Failed or malformed cases stop execution and remain available
for investigation. Locks protect the run and report directories; changed frozen
source, data or numerical environments are rejected. Do not run the lower-level
engine concurrently against these directories, because it does not use the
wrapper's locks. Existing historical runs are not adopted as v2 runs.

V2 uses bounded Newton steps with deterministic summation and coordinate
polishing. The mean logistic loss, identity-centered penalty, parameter bounds,
initialization and ridge grid match v1. A fit must pass explicit projected-gradient
and parameter-step convergence checks; an independent verifier reconstructs its
objective, gradient and constrained first-order conditions (KKT), as well as
probabilities, metrics and inner-fold selection. These are floating-point checks,
not a proof of external scientific validity. Coefficients can be nonunique in
poorly identified fits, so acceptance focuses on their probability maps and
selection decisions, not identical optimizer iteration counts.

Every numerical fit must converge, including unselected inner candidates.
Single-class calibration folds may use the recorded identity fallback. All six
full-budget copies must match exactly within a system, as must the three labels
unaffected by the paired ambient-weighting change.

The primary CI workflow runs all six plans on **macOS, Linux and Windows**. It
compares every evaluation probability and every candidate OOF probability/score
against the macOS reference at absolute tolerance **1e-8**, with zero relative
tolerance. Selected penalties and every inner/final fallback flag, reason and
count must match exactly, including unselected candidates. All 42 numerical
report files are checked, including AP and ECE. There are no diagnostic exceptions
in v2. Passing this test supports the recorded environments and data, not every
future machine or input.

To compare your six reports with the compact published reference:

```bash
python scripts/compare_portable_runs.py --reference reports/numerical_v2 --actual reports/portable_reproduction --out runs/portable_reproduction/report_comparison.json --reports-only
```

`--reports-only` checks reported tables; it does **not** compare every prediction.
The full cross-system check omits this option and uses the prediction/OOF
signatures saved in the CI artifacts. Both sides must contain those signatures.

All fit parameters, OOF scores, fallback records, predictions and per-label metrics
are saved under the run directory. Reports include 5/10-bin reliability tables
and signatures for full comparison; the repository keeps compact reported tables
and figures. The example reliability split is fixed in advance.

### Historical v1 reproduction

The original package and commands remain available to reproduce the historical
record, including its known portability failures:

```bash
python scripts/reproduce.py
python scripts/stability.py all --out runs/stability_reproduction --report reports/stability_reproduction
python scripts/verify_stability_report.py --report reports/stability_reproduction --base-config protocols/budget_v1.json --manifest data/legacy_scores/manifest.json
```

The first command runs 360 cases with defaults `runs/reproduction` and
`reports/reproduction`. The next two execute and verify the five-draw v1 study
(1,800 executions, 1,320 distinct conditions). That historical study stops on a
partially written case and preserves its evidence. Its separate workflow is
manually triggered; it is not the current v2 acceptance workflow. The original
lower-level CLI remains available through `python -m music_calibration --help`.

## Data provenance

The 120 source runs cover 1,206 tracks / 469 artist IDs. The exporter validated
the original frozen inputs, row identities, final broad-label mapping, feature
hashes and saved head outputs before producing this standalone bundle.
Its exact source is [commit 92bc9fb](https://github.com/f0rKyrie1rving/music-classification/tree/92bc9fb8090f1c8c18c63b6dcea25ae04030b9de),
run `outputs/calibration_extension/20260927_v2`. The original development and
selection history remains part of the interpretation; moving files does not
make previously inspected evaluation data unseen.

With the complete original local workspace, the export can be reconstructed:

```bash
python scripts/export_legacy.py --legacy-root /path/to/music-classification --out data/local_reexport
```

No import or execution of historical training runners is needed. The latest
future-confirmation exclusion ledger covers 2,835 tracks and 1,296 artist IDs.
It is preserved alongside per-track source/artist/license attribution.

## Limitations and next study

This is same-source, source-tag-based evidence. Missing tags, artist aliases,
encoder pretraining overlap, selected sample composition and previously inspected
results limit generalization. The original `budget_v1` uses one nested budget draw
per split; the five-draw follow-up describes subset and selection variability
under the fixed fold rule without population confidence intervals. Numerical v2
reuses those same six plans. Improving numerical reproducibility adds no new
independent evidence about genre labels or real-world calibration accuracy.

In historical budget v1, beta calibration reached the declared bounds in 648 of
1,440 final per-label fits. These fits were retained; the comparison concerns
this bounded monotone implementation, not every possible beta-calibration
implementation. The report's `successful_labels` counts valid outputs, including
unfitted raw controls and deliberate identity choices, rather than only optimized
fits.

A subsequent confirmatory experiment must first freeze its method and success
criteria, then acquire genuinely new project-disjoint data. A second data source
or independently annotated labels would strengthen external validity. No paper
acceptance, novel algorithm or universally reliable probability guarantee is claimed.

Project-owned code: [MIT](LICENSE). Bundled source-derived metadata/data:
[separate terms and attribution](DATA.md). [AI assistance](AI_ASSISTANCE.md) is
documented separately from scientific results.
