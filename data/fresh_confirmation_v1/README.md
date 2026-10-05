# Fresh confirmation: portable scores and fitted records

This bundle reconstructs an already completed study without the original
application, audio, pretrained weights, research heads or feature cache.
Use Python 3.13 and the repository's pinned requirements, then run from the
repository root:

```bash
python scripts/verify_fresh_confirmation.py
python scripts/verify_fresh_confirmation.py --refit --out runs/fresh_confirmation_refit.json
```

The first command applies the saved parameters and independently reconstructs
ridge OOF choices and the primary bootstrap; it also checks every published
per-case metric and summary. The second additionally fits all 440 conditions
again from the calibration scores and frozen folds. It compares evaluation/OOF
probabilities and selected penalties, rather than requiring identical optimizer
iteration counts. Neither command selects new songs or changes the protocol.

| File | Contents |
|---|---|
| `calibration_scores.npz` | 143 track IDs, 94 artist IDs, four labels, logits for 20 fixed MAEST heads × two weight settings, and fit-only class offsets |
| `evaluation_inputs.npz` | 637 track IDs, 400 artist IDs and corresponding logits/offsets; no labels |
| `evaluation_labels.npz` | Aligned evaluation IDs and four proxy labels |
| `calibration_plans.json` | Five artist-subset draws, budgets of 24/47/94 artists and fixed grouped inner folds |
| `fitted_calibrators.json.gz` | All 440 case records and final/OOF fits, consolidated losslessly from original JSON; per-case original hashes retained |
| `selected_manifest.json` | Source tags, roles, credited names/titles and per-track license/source attribution |
| `exclusions.json` | Historical track/artist boundary, including three conservative artist-name collisions |
| `future_exclusions.json` | Updated 4,770-track / 1,852-artist boundary, including unselected examined candidates |
| `manifest.json` | Original versus exported hashes, transformations, code hashes and reproducibility scope |

NPZ arrays load with `allow_pickle=False`; gzip JSON is ordinary UTF-8 JSON after
decompression. Source-derived data/metadata and derived scores use
**CC BY-NC-SA 4.0 for non-commercial research**; see [DATA.md](../../DATA.md).

Evaluation labels are public now that the study has finished. The original
run separated them from fitting and recorded when the report first read them;
this was procedural separation, not third-party blinded custody. Reproduction
checks the published computations, not the original acquisition chronology or
raw audio/feature provenance. See [public export notes](../../research/fresh_confirmation_v1/README.md).
