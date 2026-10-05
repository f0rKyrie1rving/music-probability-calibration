# Prospective confirmation of limited-data music probability calibration

Protocol version 1, 3 October 2026. Freeze this document, its configuration, every selected track/artist, calibration subset/fold plans, historical model hashes, numerical source and runtime before new cohort audio acquisition. This follows the already observed exploratory results; it is a prospective confirmation, not a retrospective preregistration claim. No remote preregistration is claimed.

## Question and scope

Primary: with 24 or 47 calibration artists, does identity-centered ridge Platt have lower four-label Brier loss than ordinary Platt for the fixed, unweighted MAEST heads on new project-disjoint artists? Retain raw and analytical class-weight correction controls. Full-budget results, weighted heads, other calibration methods, per-label scores and subset sensitivity are secondary. A positive primary result never implies better probabilities than raw, universal music generalization, or a novel algorithm.

The source is the pinned official MTG-Jamendo metadata commit cafd8e20c265ed84f1e61f1c875327971f43a62f. The dataset has been used previously, but the selected artists/audio have not participated in documented project experiments. Historical song exposure is audited locally before freezing. Source metadata was read to assess eligibility and artist names; no new prediction or outcome comparison was inspected. Pretrained encoder training exposure and undiscovered artist aliases remain unknown. Official genre tags mapped through the existing four-label ontology are proxy labels; no listening-based relabeling or model-generated truth is used.

## Frame and label-independent sampling

Exclude all 2,835 track IDs and 1,296 artist IDs in the previously frozen ledger. Also conservatively exclude artist_341615, artist_358635 and artist_369169 (23 tracks across the metadata): normalized credited names collide with historical artists. Matching names can represent different people; the conservative exclusion is recorded rather than called confirmed identity.

Use archives 20 through 39 inclusive, the first pre-declared block of previously unused archives with enough eligible artists. Eligibility requires source genre metadata, at least 30 seconds, an official audio checksum and a Creative Commons BY/BY-SA/BY-NC/BY-NC-SA recording license. The audited frame contains 1,912 tracks from 553 artists. All genres, including songs outside the four focus labels, remain eligible. There are no positive-label quotas. This frame is not the earlier coverage-enriched development distribution.

SHA-256 order all artists by fixed sample seed, then take the first 94 for calibration and next 400 for evaluation. Within each artist, independently hash-order tracks and take at most two. IDs, ordering and roles are frozen before archive requests. Selection never depends on labels, classifier scores or analysis results. The full eligible frame and known aliases are added to future exclusions, including unselected and failed rows. No replacements or outcome-based sample-size extensions are permitted.

## Planning sample size

Use 400 evaluation artists with at most two tracks each. Historical numerical-v2 predictions provide only a planning approximation: for the unweighted primary contrast after averaging five draws and the two partial budgets, the artist-cluster track-ratio influence SD across the 20 historical splits has a 90th percentile of about 0.01734 and a maximum of about 0.01949. At 400 artists, normal-approximation 95% half-widths are roughly 0.00170 and 0.00191; at 320, roughly 0.00190 and 0.00214. Use 0.02 as a rounded conservative planning SD (half-width 0.00196 at 400). These calculations are not guarantees of power or new-cohort precision: distribution, tagging, effect size and missingness can differ. Do not use newly observed results to increase the sample.

## Acquisition, integrity and deduplication

Only the first 30 seconds are used. Exact HTTP 206 requests must match requested byte ranges and archive sizes, with bounded responses and durable request receipts. Preserve every retrieved TAR header and its hash. Persistent sessions and at most eight archive/download workers are allowed. Each request has three attempts (connect 8 seconds, read 25 seconds); at most three phase invocations allow documented transport recovery. Reuse only hash-verified completed checkpoints. Interrupted attempts and retries remain in the record; outcome-dependent retries, alternate-song replacements and deleted failure logs are prohibited.

Download at most 1,310,720 encoded bytes per track using the existing ID3-skip and bounded-prefix approach, decode 30 seconds to mono 22,050-Hz PCM16. Report partial-prefix hashes accurately; do not call them verified full MP3 hashes unless the entire indexed member was downloaded and matched the official checksum. All audio retains individual license/source attribution and is not committed or redistributed.

Before scoring, compare canonical hashes of the first 30 seconds of decoded audio (mono, resampled to 16 kHz and quantized to PCM16 under the frozen routine). If any selected artist has an excerpt identical to an available historical excerpt or an excerpt under a different selected artist, exclude every implicated fresh artist. Within one artist, keep only the first selected track among identical excerpts. No replacement follows. This detects identical quantized excerpts, not every cover, re-encoding, near-duplicate or unknown pretraining occurrence. Save the historical inventory, mismatches, unavailable historical files and exact exclusion reasons.

Acquisition and integrity decisions are made without probabilities or outcome inspection. Record every failure and duplicate. A process interruption can resume transport and extraction under the fixed bounded rules; completed outputs that fail their recorded hash are corruption and stop the run.

## Frozen classifiers and features

Use all twenty existing research MAEST head seeds 2026092800 through 2026092819, each with unweighted and ambient-balanced versions. No head is selected by its earlier performance and none is retrained. Application v1.2 classifiers are not substituted. Use existing saved mean/scale/coefficient/intercept arrays and derive analytical class-weight offsets from their original fit counts only. Snapshot all 40 heads and hash their provenance.

MAEST extraction uses the same locally pinned weights, layer-seven CLS/DIST/signal-mean concatenation (2,304 dimensions), first-30-second preprocessing and CPU runtime as the historical experiments. Re-extract historical control track_0001100 before new extraction and require maximum float32 feature difference <=1e-5. Save per-track features and hashes; use identical feature rows for both head weightings. The three labels unaffected by ambient weighting must produce identical logits/probabilities where the method and subset match. Failing source/runtime/control integrity stops rather than becoming a model outcome.

## Calibration and evaluation separation

Five fixed seeds 2026100301 through 2026100305 order the same 94 calibration artists. Each produces nested budgets of 24, 47 and 94 artists; three artist-grouped inner folds use seed 2026100311 and the frozen rule. Fit every calibrator only on its calibration subset, and select ridge penalties only from its artist-grouped out-of-fold predictions. Missing rows are intersected with the original plan, never replaced or redrawn; planned and actual counts are reported. Full-budget inputs and fold plans are identical across draws and counted once in summaries.

Use the unchanged numerical-v2 implementations of raw, analytical prior offset, temperature, ordinary Platt, beta, isotonic and ridge Platt. Ridge candidates are lambda = 0, .001, .01, .1, 1 plus exact identity; ties within 1e-12 prefer identity, then stronger regularization. Preserve the old parameter bounds and explicit numerical convergence tests. Single-class calibration/inner folds use the recorded identity fallback. Any numerical nonconvergence or nonfinite data aborts this confirmatory run for investigation; do not silently switch implementations or hide failed candidates.

Export calibration labels/scores separately from evaluation input scores. All saved calibrators and evaluation predictions must be finalized and hash-bound before the report opens evaluation labels. The evaluation labels are not cryptographically inaccessible to the experimenter; this is a code and protocol separation with an auditable chronology, not a claim of third-party blinded custody.

## Primary estimand and statistical analysis

For each evaluation track, compute ridge-minus-ordinary-Platt squared-error difference averaged over four labels, all twenty unweighted heads, all five calibration draws, and the two partial budgets (24 and 47 planned artists). Average those per-track differences over evaluation tracks. This is an average of losses, NOT a probability ensemble. Lower is better. Heads and draws are fixed repeated conditions, not independent experimental samples.

Use 10,000 paired bootstrap draws of evaluation artists with seed 2026100321. Each resampled artist contributes all of that artist's observed tracks; divide summed losses by resampled track count. Use identical resamples for compared methods. Report a two-sided percentile 95% interval. Primary support requires adequate coverage and its upper endpoint strictly below zero. If it includes zero, report inconclusive; if it excludes zero above zero, report evidence of worse performance. Preserve the point estimate and all uncertainty regardless of direction. The interval is conditional on the fixed classifiers and calibration cohort/plans, and does not cover uncertainty from sampling a new calibration population, model training, missingness, source labels or upstream pretraining.

There is one primary comparison. Secondary contrasts (ridge versus raw/offset, weighted results, each budget, labels, logarithmic loss, average precision, ECE and reliability tables) are reported together; any intervals are unadjusted pointwise exploratory descriptions. Do not declare them additional confirmatory discoveries. Five subset draws describe sensitivity inside one new calibration pool; do not turn five draws, twenty heads or multiple budgets into independent replications. Full-budget repeats count once and act as integrity controls.

## Coverage and interpretation

Require at least 80% of selected calibration artists overall; at each planned budget require >=ceil(.8 * planned_artists) usable artists. Evaluation must retain at least 320 of the 400 selected artists and 80% of selected tracks. For each of the four labels, require at least 20 evaluation artists with at least one positive track and 20 artists whose evaluated tracks are all negative. If coverage fails, retain scores but label the study incomplete_evidence; no resampling or claim that the primary benefit was confirmed follows. Report attrition by role/reason and observed class counts after predictions are frozen.

After outcomes are known, do not amend this protocol to achieve success. Corrections to code or integrity failures must be dated, documented and separated from the original attempt. Report deviations and whether fresh confirmatory status was compromised. Numerical reproduction of this run is not another external validation. Any later second-source study requires its own new protocol, harmonized label meaning and data boundary.

## Outputs

Preserve protocol/configuration, selection/future exclusions, source/encoder/head hashes, transport logs, audio/feature receipts, plans, raw scores, per-label fit/OOF records, predictions, paired comparisons, artist-cluster uncertainty, coverage, reliability diagnostics and an independent verification receipt. Publish only shareable code/metadata/derived scores under applicable licenses. Update the bilingual manuscript to distinguish historical exploratory evidence from this prospective, same-source confirmation. Do not alter the released desktop application.
