# Fresh confirmation v1: results and public evidence

The protocol was frozen locally before acquisition on 3 October 2026. All 780
selected tracks were retained: 143 tracks / 94 artists for calibration and
637 tracks / 400 artists for evaluation. These artists are outside the audited
project history, but the source remains MTG-Jamendo and the labels are source
tags, not adjudicated listening judgments.

For 20 fixed unweighted MAEST research heads, five prespecified subset draws and
24/47-artist budgets, identity-centered ridge Platt reduced mean four-label Brier
loss relative to ordinary Platt by **0.00704987**, with a two-sided paired
artist-cluster 95% percentile interval **[−0.00836099, −0.00574796]**. The coverage
gate passed, satisfying the prespecified primary criterion. These 200 repeated
conditions are not independent datasets. The interval conditions on the fixed
heads and calibration pool/draws.

The important limitation is visible in the same comparison: raw **0.12503**,
ridge Platt **0.12706**, ordinary Platt **0.13411**. Regularization improved on
ordinary Platt but did not surpass the raw-score control on this mean.

- [Primary result](primary.json), [summary table](summary_cells.csv),
  [exploratory secondary contrasts](secondary_contrasts.json), and
  [coverage](coverage.json).
- [Original independent numerical audit](independent_verification.json),
  [selection audit](selection_verification.json),
  [acquisition audit](acquisition_verification.json), and
  [extraction receipt](extraction_receipt.json).
- [Protocol and export boundary](../../research/fresh_confirmation_v1/README.md),
  [portable data](../../data/fresh_confirmation_v1/README.md),
  [preprint](../../paper/README.md).

[The public-bundle verification receipt](public_verification.json) records a
clean-directory local check on 6 October 2026 (China time): all **440** conditions
were refitted, **7,847,840** evaluation probabilities and **1,760** ridge OOF
selections were checked, and every refitted evaluation/OOF probability and
per-case metric matched the saved records exactly in the recorded environment.
The independent elementary probability formula differed by at most `1.67e-16`.
The primary/secondary bootstrap intervals and summary tables were reproduced.
Four targeted integrity/leakage tests also pass. These are local numerical
checks, not new data or another external confirmation.

Run the public reconstruction from the repository root:

```bash
python scripts/verify_fresh_confirmation.py
```

Add `--refit --out runs/fresh_confirmation_refit.json` to refit every condition
from the included calibration scores. This needs the pinned Python dependencies,
not audio, encoder weights or the original application's local caches. The
published fresh-bundle validation is local; the older cross-platform benchmark
validation must not be described as a new three-system run of this cohort.

The JSON/CSV numerical results and historical audit receipts retain original
bytes; large `case_metrics.json` and `actual_calibration_subsets.json` are stored
as `.gz` with unchanged decompressed content. `completed.json` retains the
original study hashes, including those uncompressed filenames. Use the
[public manifest](../../data/fresh_confirmation_v1/manifest.json) for exported-file
integrity. Original local freeze/protocol files were not rewritten. Hash integrity
and saved chronology do not establish external preregistration, clean encoder
pretraining exposure, human label accuracy or absence of all aliases/near-duplicates.
