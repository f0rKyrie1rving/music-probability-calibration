# Fresh confirmation protocol and public export

The [protocol](protocol.md) is an exact copy of the document frozen locally on
3 October 2026 before acquisition of the new cohort. It is not a retroactive
claim of registry preregistration or independent timestamp attestation.
The original freeze SHA-256 is
`2af3f1b7222d39f3ac50a4b2d03f52dd78935e865e1a0bafa2e9c565b690bb88`.

The 6 October 2026 public export retains numerical data and evidence while
removing workstation-specific dependencies:

- `protocol.md`, calibration plans, score arrays and public-safe report/receipt
  files preserve their original bytes. Large report JSON is gzip-compressed with
  the decompressed bytes unchanged.
- `config.json` removes only the original `legacy_root` and `calibration_repo`
  workstation paths. Statistical settings are unchanged.
- `freeze.public.json` replaces absolute external-file prefixes with
  `<research-repository>` and `<application-repository>`. **It is a derivative,
  not the original freeze**, and is not expected to hash to the original digest.
- `fit_started.json` and `fit_completed.json` are exact historical receipts.
  Their hashes refer to the original private-run layout. The portable verifier
  uses the public export manifest, not a false claim that all original external
  files are distributed here.
- Fitted case metadata and parameters are consolidated in the compressed score
  bundle with per-case original hashes. These are existing fitted records, not
  newly selected or changed model parameters.

The [export manifest](../../data/fresh_confirmation_v1/manifest.json) records every
transformation and both original/exported hashes. Original frozen local records
have not been edited. The repository's unchanged `music_calibration_portable`
source must match the numerical source hashes in the public manifest.

The [verification command](../../scripts/verify_fresh_confirmation.py) applies all
saved calibrators, checks independent elementary probability formulas and ridge
OOF selection, reconstructs every per-case metric and summary, and regenerates
the paired artist-bootstrap primary and secondary intervals. `--refit` additionally
fits all 440 conditions from the scores/plans and compares predicted/OOF values
at `1e-8` absolute tolerance with identical ridge selections.

This is **score-level reproduction**. Audio, pretrained encoder/research-head
weights, per-track features, original archive caches and network logs are not
published. Therefore this command cannot re-extract logits from audio, repeat
historical acoustic deduplication, or independently establish the original
chronology. The published acquisition, selection and original independent-audit
receipts describe checks performed on that complete local run. They do not turn
this numerical reconstruction into a second external scientific replication.

[Results](../../reports/fresh_confirmation_v1/README.md) retain the same-source,
proxy-label and conditional-uncertainty limitations. The method beats ordinary
Platt in its primary contrast but does not beat raw probabilities on that mean.
