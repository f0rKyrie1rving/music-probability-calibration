# Reference release audit — 2026-10-06

## Platt book chapter: resolved

Replace the previous hybrid reference (1999 preprint long title combined with 2000 book publication data) with the formal published chapter citation:

Platt JC. Probabilities for SV machines. In: Smola AJ, Bartlett PL, Schölkopf B, Schuurmans D, editors. *Advances in Large-Margin Classifiers*. Cambridge, MA: MIT Press; 2000. p. 61–74. https://doi.org/10.7551/mitpress/1113.003.0008

The publisher-deposited Crossref record was retrieved successfully with curl on 2026-10-06:
https://api.crossref.org/works/10.7551/mitpress/1113.003.0008

Relevant returned fields:

```json
{
  "publisher": "The MIT Press",
  "type": "book-chapter",
  "title": ["Probabilities for SV Machines"],
  "author": [{"given": "John C.", "family": "Platt"}],
  "container-title": ["Advances in Large-Margin Classifiers"],
  "page": "61-74",
  "published-print": {"date-parts": [[2000, 9, 29]]},
  "DOI": "10.7551/mitpress/1113.003.0008",
  "ISBN": ["9780262283977"],
  "resource": {"primary": {"URL": "https://direct.mit.edu/books/book/2787/chapter/75444/Probabilities-for-SV-Machines"}}
}
```

The MIT Press book page independently confirms the editors, publisher, and 2000-09-29 publication date:
https://mitpress.mit.edu/9780262194488/advances-in-large-margin-classifiers/

The MIT Press chapter DOI search record independently confirms the short chapter title, author, and year. Its full chapter PDF was not accessible through the web tool; the metadata/pagination check does not claim a new full-text reading. No need to omit the page range now that the publisher-deposited DOI record explicitly verifies it.

## Das et al.: keep the cited version explicit

Das S, Dasgupta N, Dutta P. Geometric calibration and neutral zones for uncertainty-aware multi-class classification. arXiv:2511.20960v1; 2025. https://arxiv.org/abs/2511.20960v1

The version-specific abstract page confirms Soumojit Das, Nairanjana Dasgupta, Prashanta Dutta and v1 submission date 2025-11-26. Version 2 exists (2025-11-29); linking the fixed v1 prevents a version mismatch.

The v1 full text identifies itself as a working paper/preprint. Section 2.2.2, equation (7), explicitly contains the penalty `lambda_1 ||A-I||_F^2 + lambda_2 ||b||_2^2`, alongside positive-definiteness/trace constraints. It supports the narrow statement that shrinkage toward an identity map already appears in prior calibration work; it is not evidence that our independently fitted binary formulation is identical.

Full-text primary source: https://arxiv.org/html/2511.20960v1#S2.SS2.SSS2

These corrections are incorporated into both language versions of preprint v1. The numerical results are unchanged.


## Full bibliography record

[references.json](references.json) lists all 16 entries in manuscript order, with primary-source links, the specific claim each supports and the verification scope. The original bibliography checks date to 3 October 2026; the release audit above was performed on 6 October 2026. Two cited works (Das et al. and Caplin et al.) are explicitly identified as preprints. A valid citation is not a guarantee of novelty, peer review or agreement with this study.
