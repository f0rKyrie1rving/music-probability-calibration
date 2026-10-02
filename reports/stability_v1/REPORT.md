# Calibration subset and selection stability

Five new calibration-artist draws reuse the unchanged budget_v1 engine and all 120 source cases.
The study executes 1,800 cases, covering 1,320 distinct source/subset conditions.
Full-budget outputs are repeated numerical controls and are aggregated once per outer split.
The original historical partial-budget draw is excluded from these summaries.

![Mean Brier changes across repeated calibration subsets](budget_brier.png)

Curves average five partial-budget draws within each of 20 overlapping outer splits.
The common full-budget endpoint is counted once. These curves are not confidence intervals.

All five child runs passed independent numerical verification. This is not new-data confirmation.
Within-split ranges and sign changes describe five draws under a fixed fold-assignment procedure.
Selected artists and resulting fold memberships vary together; their effects are not separated.
Twenty outer splits overlap, and no confidence interval or independent-replicate test is constructed.

## Ridge calibration compared with raw and ordinary Platt

| Representation | Weighting | Budget | Mean change vs raw | Mean change vs Platt | Raw sign-flip splits / 20 |
|---|---|---:|---:|---:|---:|
| maest | ambient_balanced | 25% | -0.002458 | -0.003280 | 15 |
| maest | ambient_balanced | 50% | -0.006109 | -0.001874 | 2 |
| maest | ambient_balanced | 100% | -0.007849 | -0.000908 | 0 |
| maest | unweighted | 25% | +0.003186 | -0.005787 | 7 |
| maest | unweighted | 50% | +0.001061 | -0.002864 | 13 |
| maest | unweighted | 100% | +0.000351 | -0.000885 | 0 |
| mert_v0 | ambient_balanced | 25% | -0.005981 | -0.002321 | 10 |
| mert_v0 | ambient_balanced | 50% | -0.010035 | -0.000925 | 1 |
| mert_v0 | ambient_balanced | 100% | -0.011809 | -0.000120 | 0 |
| mert_v0 | unweighted | 25% | +0.003152 | -0.005005 | 13 |
| mert_v0 | unweighted | 50% | +0.000578 | -0.002134 | 18 |
| mert_v0 | unweighted | 100% | -0.000252 | -0.000355 | 0 |
| mert_v1 | ambient_balanced | 25% | -0.006653 | -0.002770 | 11 |
| mert_v1 | ambient_balanced | 50% | -0.010351 | -0.000873 | 0 |
| mert_v1 | ambient_balanced | 100% | -0.012297 | -0.000293 | 0 |
| mert_v1 | unweighted | 25% | +0.003223 | -0.005159 | 12 |
| mert_v1 | unweighted | 50% | +0.000734 | -0.002070 | 17 |
| mert_v1 | unweighted | 100% | -0.000270 | -0.000543 | 0 |

Negative Brier changes are improvements. They are not accuracy changes or proof of reliability in every probability bin.
The full-budget sign-flip count is zero by construction because its endpoint is identical across draws.
All seven methods, individual draw outcomes and within-split ranges are retained in the CSV tables.
Ridge choices and explicit fallbacks are recorded separately. No best-performing draw is selected.

Missing source tags, historical selection, artist aliases and unknown encoder-pretraining exposure remain limitations.
The released music application and original budget_v1 experiment are unchanged.
