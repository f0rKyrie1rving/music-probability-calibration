# Data, attribution and prior exposure

The source is the [MTG-Jamendo Dataset](https://github.com/MTG/mtg-jamendo-dataset),
by Dmitry Bogdanov, Minz Won, Philip Tovstogan, Alastair Porter and Xavier Serra
(Music Technology Group). See their 2019 work, *The MTG-Jamendo Dataset for
Automatic Music Tagging*, Machine Learning for Music Discovery Workshop, ICML.
Upstream copyright notice: Copyright 2019–2023 Music Technology Group.

The upstream repository states that metadata is available under
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/), while audio
has per-recording Creative Commons licenses. It restricts dataset use to
non-commercial research and academic purposes. Those upstream terms continue
to apply; the MIT software license in this project does not replace them.

The `data/legacy_scores` bundle is provided for non-commercial research under
CC BY-NC-SA 4.0 for its source-derived labels/metadata and the project-owned
derived score tables packaged with them. Original credits and individual track
licenses are retained in `provenance/track_attribution.json`. No source audio,
large feature cache, pretrained weights or fitted classifier weights are included.

Changes to source material: four broad labels were derived from the source tags
using the historical frozen ontology; artist/track IDs were aligned to historical
splits; numeric model logits and reference probabilities were added. Broad labels
can differ from earlier exact-tag labels in the original project. All exported
labels are checked against the actual saved calibration/evaluation arrays.

## Contents and integrity

- `manifest.json`: every score case, SHA-256 and source commit/run identity.
- `cases/*.npz`: non-pickle arrays for classifier-fit IDs/groups/targets,
  calibration and evaluation IDs/groups/targets/logits, fit-only weight offsets,
  and raw/Platt/temperature/offset historical reference probabilities.
- `provenance/export_audit.json`: feature provenance, reconstructed-logit checks,
  target mapping, source status and control-prediction checks.
- `provenance/legacy_*`: historical frozen protocol, configuration, split records
  and numerical verification.
- `provenance/future_exclusions.json`: the historical 2,835-track / 1,296-artist
  boundary, including previously examined eligible sampling frames.

Hash checks detect changes relative to these records; they do not establish the
truth of genre tags or prove that no external data leakage exists.

## Historical exposure

All 1,206 development tracks / 469 artist IDs in this bundle have been used in
previous experiments. Twenty splits overlap. MAEST/MERT representations share
the same songs; they are not three independent datasets. Earlier 239-, 266-,
204- and 372-song evaluation cohorts have also been observed in the project.
They are not included as fresh confirmation samples here. Artist aliases and
pretraining exposure cannot currently be completely audited.

Future study selection must start from the latest exclusion ledger and a new
protocol. This project reports exploratory numerical comparisons, not human
adjudication of disputed musical labels.

## Fresh confirmation bundle (3 October 2026)

`data/fresh_confirmation_v1` contains the calibration scores/labels for 143
tracks, evaluation scores and separately stored labels for 637 tracks, frozen
artist-grouped fold plans, and losslessly compressed fitted calibrator records.
The 780 tracks span 494 artists with no calibration/evaluation artist overlap.
`selected_manifest.json` retains source IDs, credited titles/artists, tags,
per-track licensing and attribution. Its relative audio paths describe the
original run; those recordings are not distributed here.

This source-derived metadata/label/score bundle is supplied under the same
**CC BY-NC-SA 4.0, non-commercial research** terms described above. Per-recording
licenses and original attribution remain applicable. The project MIT software
license does not relicense the dataset. Numeric scores and fitted calibration
parameters were added; labels were derived through the frozen four-label
ontology. No audio, pretrained encoder weights, research classifier weights,
feature arrays or private request caches are part of the public bundle.

The new `future_exclusions.json` covers **4,770 tracks / 1,852 artist IDs**,
including the entire examined eligible frame and name-collision exclusions.
Future selection must use this ledger, rather than the older historical boundary.
The cohort is unseen in documented project use, but still comes from a previously
known dataset with unadjudicated tags. Unknown aliases, near-duplicates and
upstream encoder pretraining exposure remain limitations.

[The export manifest](data/fresh_confirmation_v1/manifest.json) links exact or
losslessly compressed public artifacts to their original hashes. Public config
and freeze copies remove workstation path prefixes and are explicitly identified
as derivatives. The original frozen local records remain unchanged. Hashes check
integrity relative to the published record; they are not external time stamps or
independent proof of source-label truth.
