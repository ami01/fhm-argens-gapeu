# Flood Hazard Mapping — Argens & Gapeau Basins

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18758068.svg)](https://doi.org/10.5281/zenodo.18758068)

## Description
Code and datasets accompanying the manuscript:
**"Evaluation of Machine Learning Approaches and Their 
Extrapolation to Unseen Areas for Regional Scale Flood 
Hazard Mapping"** submitted to Natural Hazards and Earth 
System Sciences (NHESS).

**Authors:** Aman Arora, Pierre Nicolle, Olivier Payrastre

## Repository Structure

    fhm-argens-gapeu/
    |-- data/              Input datasets
    |-- models/
    |   |-- random_forest/     RF model (Python)
    |   |-- xgboost/           XGB model (Python)
    |   |-- neural_network/    NNet model (R)
    |-- utils/             Shared utility functions (Python)
    |-- figures/           Output figures
    |-- results/           Output results per model

## Requirements

### Python (RF, XGB)
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

## License
MIT License - see LICENSE file.

## Citation
Arora, A., Nicolle, P., and Payrastre, O.: Evaluation of Machine 
Learning Approaches and Their Extrapolation to Unseen Areas for 
Regional Scale Flood Hazard Mapping, Nat. Hazards Earth Syst. Sci., 
https://doi.org/10.5281/zenodo.18758068, 2025.
