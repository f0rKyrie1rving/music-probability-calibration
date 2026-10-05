# Probability calibration with limited artist data in music tagging

**Retrospective and prospective evidence on regularization and numerical reproducibility**

Chenglin Song / 宋承麟 · **Preprint v1 · 6 October 2026 · Not peer reviewed**

The English and Chinese manuscripts report the same study. They are two language
versions of one research output, not separate papers. This is a public preprint,
not an accepted journal article. No journal publication or DOI is claimed.

## Read or download

| Language | PDF | Editable Word document |
|---|---|---|
| English | [English PDF](English_Probability_Calibration.pdf) | [English Word](English_Probability_Calibration.docx) |
| 中文 | [中文 PDF](Chinese_Probability_Calibration.pdf) | [中文 Word](Chinese_Probability_Calibration.docx) |

These files are versioned in the repository and do not use an expiring GitHub
Actions artifact link. The `paper-v1` tag identifies this publication snapshot.

## What the study finds

The historical benchmark compares seven probability treatments using preserved
scores from 1,206 tracks. A later, locally frozen confirmation uses 780 tracks
from artist IDs absent from audited project history: 143 calibration tracks and
637 evaluation tracks. Both stages use the same music source and noisy source
tags; they are not independently adjudicated labels or a cross-source test.

On the prespecified partial-budget comparison, identity-centered ridge Platt
reduces mean Brier loss relative to ordinary Platt by **0.007050** (paired
evaluation-artist bootstrap 95% interval **[−0.008361, −0.005748]**). However, the
average Brier losses are **0.12503 for raw output, 0.12706 for ridge Platt and
0.13411 for ordinary Platt**. Lower is better. Regularization helps relative to
ordinary calibration here, but does not establish an advantage over retaining
the raw output. Brier loss measures overall probability quality, not calibration
alone. No new calibration algorithm is claimed.

The earlier numerical benchmark was reproduced across macOS, Linux and Windows.
The new 780-track confirmation and its public refit receipt were checked locally;
the historical three-system result must not be presented as a three-system
reproduction of this new cohort.

## Evidence and reproduction

- [Fresh protocol and original-freeze mapping](../research/fresh_confirmation_v1/README.md)
- [Fresh numerical results and verification receipt](../reports/fresh_confirmation_v1/README.md)
- [Score inputs, fitted records and attribution](../data/fresh_confirmation_v1/README.md)
- [Historical numerical validation](../reports/numerical_v2/PORTABILITY.md)
- [Reference sources and release corrections](reference_verification.md)
- [Document export and checks](release_validation.json) and [SHA-256 checksums](SHA256SUMS)

From the repository root, using Python 3.13 and `requirements-lock.txt`:

```bash
python scripts/verify_fresh_confirmation.py
python scripts/verify_fresh_confirmation.py --refit --out runs/fresh_confirmation_refit.json
```

The second command refits all 440 distinct conditions from the public calibration
scores and frozen folds. It checks the saved evaluation predictions, internal
selection decisions and reported comparisons. This release does not redistribute
audio, feature caches, pretrained encoders or complete private execution folders;
it does not rerun original audio acquisition or feature extraction. Original
chronology receipts document a local protocol freeze, not external preregistration.
The fixed heads and repeated draws do not create independent new datasets.

## Citation and responsibility

Suggested citation:

> Song, C. (2026). *Probability calibration with limited artist data in music tagging:
> Retrospective and prospective evidence on regularization and numerical reproducibility*.
> Preprint, version 1. https://github.com/f0rKyrie1rving/music-probability-calibration/tree/paper-v1/paper

See the repository [CITATION.cff](../CITATION.cff) for machine-readable citation
metadata. Generative AI assisted with literature retrieval, experimental design,
implementation, verification and bilingual drafting. The manuscripts and
[AI_ASSISTANCE.md](../AI_ASSISTANCE.md) disclose that assistance. Chenglin Song
is responsible for the interpretation, references and any future submission
declarations. Results and limitations should be reported together.

The manuscripts are copyright © 2026 Chenglin Song and are made publicly
available for reading and citation; no separate blanket reuse license is granted
for the manuscript text or figures. Repository software is MIT-licensed.
Source-derived data, metadata and scores have separate non-commercial terms and
attribution requirements in [DATA.md](../DATA.md); the code license does not
override them.
