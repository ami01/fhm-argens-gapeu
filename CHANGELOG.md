# Changelog

## 2026-10-12 — Article published
The article was published in *Geomatics, Natural Hazards and Risk*,
17(1), 2707816 (online 2026-09-30),
https://doi.org/10.1080/19475705.2026.2707816, under the final title
**"Machine learning surrogates for flash flood hazard mapping:
CSI-optimized feature selection and spatial transferability"**.

### Changed
- README: final title, journal reference, paper DOI badge, and a full
  citation with BibTeX.
- CITATION.cff: `preferred-citation` now gives volume, issue, article
  number, and DOI. The article DOI is also listed under `identifiers`.
- No changes to code or data.

## v2.0 — 2026 (Geomatics, Natural Hazards and Risk revision)

Revised release accompanying the manuscript retitled
**"Machine Learning Surrogates for Flash Flood Hazard Mapping:
CSI-Optimised Feature Selection and Spatial Transferability"**,
submitted to *Geomatics, Natural Hazards and Risk* (Taylor & Francis).

### Changed
- README updated with the revised title, target journal, author
  affiliation, method summary, and citation.
- `models/README.md`: filled in the run commands and documented the
  CSI-optimised feature selection and the calibration step.

### Added
- `utils/calibration.py`: computes Brier score and AUROC and draws the
  reliability diagrams for RF, XGB, and NNet on the Gapeau basin.
- `data/calibration/`: balanced Gapeau calibration samples
  (10,000 flood / 10,000 non-flood pixels, numpy seed=42) for RF, XGB,
  and NNet, plus a README describing the columns and reference grid.
- `figures/Fig_Reliability_Diagrams.{pdf,png}`: reliability-diagram panel
  as published in the revised manuscript.

### Reproduced metrics (extrapolation, Gapeau)
| Model | AUROC | Brier |
|-------|-------|-------|
| Random Forest | 0.9722 | 0.0714 |
| XGBoost | 0.9761 | 0.0648 |
| Neural Network | 0.9738 | 0.0655 |

## v1.0 — 2026-02-24 (initial submission)
- Initial code and data snapshot (archived as
  [10.5281/zenodo.18758068](https://doi.org/10.5281/zenodo.18758068)).
