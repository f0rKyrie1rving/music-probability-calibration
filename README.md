# Music Probability Calibration

**Can probability calibration avoid unnecessary changes when only a small
calibration set is available?** This research benchmark studies that question
using frozen music-classifier scores, artist-disjoint fitting/calibration/evaluation
partitions, and explicit controls for training class weights.

This is the separate research project following
[Music Classification v1.2](https://github.com/f0rKyrie1rving/music-classification/releases/tag/v1.2.0).
It does not change the released application. The first study is exploratory and
uses previously observed MTG-Jamendo data; it is not a new independent test set.

## What is compared

Seven probability treatments: raw scores, analytical class-weight offset,
temperature scaling, Platt scaling, beta calibration, isotonic regression, and
identity-centered Platt regularization selected by artist-grouped cross-validation.
They estimate four separate label probabilities: electronic, pop, ambient and rock.

The fixed design contains **360 cases**: 20 historical artist splits × three audio
representations × two ambient-weight settings × three calibration-artist budgets
(25%, 50%, 100%). Every method uses the same evaluation rows within each case.
Within each source case, classifiers stay fixed. The new comparisons isolate calibration choices rather
than differences in audio-model training or decision thresholds.

The regularized method can select exact identity (retain the original score), but
has no guarantee against deterioration. Beta and isotonic are existing methods;
this project does not claim to invent probability calibration.

## Results and protocol

- [Full report and all comparison tables](reports/budget_v1/REPORT.md)
- [Chinese explanation](reports/budget_v1/SUMMARY_ZH.md)
- [Frozen study protocol](protocols/budget_v1.md) and [configuration](protocols/budget_v1.json)
- [Numerical verification](reports/budget_v1/verification.json)
- [Clean-environment reproduction](reports/budget_v1/reproduction.json): all 360 cases
  rerun; seven numeric reports reproduced byte-for-byte; 38 tests passed.
- [Data attribution and exposure boundary](DATA.md)

![Calibration budget comparison](reports/budget_v1/budget_brier.png)

Negative Brier changes mean lower probability error. The curves average 20
overlapping splits of one dataset; they are descriptive, not confidence intervals
or independent replications. Lower Brier alone does not establish calibration in
every probability interval, nor does it establish better genre recognition.

## Reproduce from this repository

The roughly 10 MB score bundle is included. This score-level experiment needs
neither audio downloads nor the original application's private feature caches,
PyTorch, pretrained encoder weights or a GPU. Full reproduction from raw audio
is a separate task requiring the original manifests, models and feature pipeline.

Use Python 3.13 for the pinned environment (Python 3.12+ is supported by the source):

```bash
python3 -m venv .venv
# Windows: use .venv\Scripts\python.exe instead of .venv/bin/python
.venv/bin/python -m pip install -r requirements-lock.txt
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m music_calibration audit --data data/legacy_scores
.venv/bin/python -m music_calibration freeze --data data/legacy_scores --out runs/reproduction_01
.venv/bin/python -m music_calibration run --out runs/reproduction_01
.venv/bin/python -m music_calibration summarize --out runs/reproduction_01 --report reports/reproduction_01
.venv/bin/python -m music_calibration verify --out runs/reproduction_01
```

Choose a new directory for a new execution. Freezing refuses to overwrite old runs;
resuming accepts only completed cases with matching receipts. The verifier reads
saved parameters and reconstructs probabilities/metrics independently of the fitting
functions, including inner-fold ridge selection. This verifies numerical consistency,
not external scientific validity.

All model parameters, OOF scores, fallback records, prediction arrays, per-label
metrics and 5/10-bin reliability tables are saved under `runs/`. Compact reported
tables and figures are committed under `reports/`. The first historical split is
declared in advance for readable example reliability tables; no best split is chosen.

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
results limit generalization. One nested budget draw per split does not estimate
within-split sampling variability. Single-class or failed calibrator fits use
explicitly reported identity fallback; unexpected errors abort.

Beta calibration reached the declared parameter bounds in 648 of 1,440 final
per-label fits. These fits were retained; the comparison concerns this bounded
monotone implementation, not every possible beta-calibration implementation.
The report's `successful_labels` counts valid outputs, including unfitted raw
controls and deliberate identity choices, rather than only optimized fits.

A subsequent confirmatory experiment must first freeze its method and success
criteria, then acquire genuinely new project-disjoint data. A second data source
or independently annotated labels would strengthen external validity. No paper
acceptance, novel algorithm or universally reliable probability guarantee is claimed.

Project-owned code: [MIT](LICENSE). Bundled source-derived metadata/data:
[separate terms and attribution](DATA.md). [AI assistance](AI_ASSISTANCE.md) is
documented separately from scientific results.
