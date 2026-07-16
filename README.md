# Flood Hazard Mapping — Argens & Gapeau Basins

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18758068.svg)](https://doi.org/10.5281/zenodo.18758068)

## Description
Code and datasets accompanying the manuscript:

**"Machine Learning Surrogates for Flash Flood Hazard Mapping:
CSI-Optimised Feature Selection and Spatial Transferability"**
submitted to *Geomatics, Natural Hazards and Risk* (Taylor & Francis).

**Authors:** Aman Arora, Pierre Nicolle, Olivier Payrastre
**Affiliation:** Univ Gustave Eiffel, GERS-LEE, F-44344 Bouguenais, France

The study evaluates three machine-learning models — Random Forest (RF),
eXtreme Gradient Boosting (XGB), and a Feed-Forward Neural Network (NNet) —
as efficient surrogates for high-resolution 2D hydrodynamic simulation.
It addresses two questions: whether ML can retrieve the deterministic
flood-hazard extent from probabilistic susceptibility maps, and how well
the models transfer to an unseen catchment. Models are trained on the
Argens basin and tested on the Gapeau basin against a 5 m FLOODOS 2D
simulation of the 1000-year flood in southeastern France.

## Repository Structure

    fhm-argens-gapeu/
    |-- data/
    |   |-- raw/               Input datasets and inventory
    |   |-- processed/         Normalised training data
    |   |-- calibration/       Balanced Gapeau samples for reliability analysis
    |-- models/
    |   |-- random_forest/     RF model (Python)
    |   |-- xgboost/           XGB model (Python)
    |   |-- neural_network/    NNet model (R)
    |-- utils/                 Shared utility functions (Python)
    |   |-- feature_selection.py   CSI-optimised forward feature selection
    |   |-- calibration.py         Brier score + reliability diagrams
    |-- figures/               Output figures
    |-- results/               Output results per model

## Method summary

A performance-driven feature selection algorithm — a greedy forward search
in which candidate features are retained by their incremental contribution
to the Critical Success Index (CSI) of the retrieved flood extent, rather
than to threshold-independent accuracy metrics — reduces 21 candidate
factors to four physically relevant predictors: peak discharge, Height
Above Nearest Drainage, Distance to Stream, and local river slope. Model
probability outputs are thresholded at the CSI-maximising cut-off to
delineate the binary hazard extent, and calibration of those probabilities
is assessed with Brier scores and reliability diagrams.

## Requirements

### Python (RF, XGB, feature selection, calibration)
    pip install -r requirements.txt

### R (Neural Network)
    install.packages(c("nnet", "raster", "caret"))

## Quick Start

### Random Forest
    cd models/random_forest/
    python 01_training.py
    python 02_prediction.py
    python 03_validation.py
    python 04_extrapolation.py
    python 05_extrapolation_validation.py
    python 06_csi_evaluation.py

### XGBoost
    cd models/xgboost/
    python 01_training.py
    ...

### Neural Network (R)
    cd models/neural_network/
    Rscript 01_training.R
    ...

### Calibration / reliability analysis
    python -m utils.calibration
    # reads data/calibration/*.csv, prints Brier + AUROC per model,
    # and writes figures/Fig_Reliability_Diagrams.pdf

## License
MIT License — see LICENSE file.

## Citation
Arora, A., Nicolle, P., and Payrastre, O.: Machine Learning Surrogates
for Flash Flood Hazard Mapping: CSI-Optimised Feature Selection and
Spatial Transferability, Geomatics, Natural Hazards and Risk, 2026.
Software archived at https://doi.org/10.5281/zenodo.18758068
