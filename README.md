# Parkinson's Research

## Overview

This repository contains an applied machine learning research project for early Parkinson's disease detection from voice and acoustic features. The work was developed as an undergraduate research effort and later written up as a research outcome paper.

The core problem is straightforward: Parkinson's disease is difficult to identify early because symptoms can be subtle, and voice changes are often one of the earliest measurable signals. This project explores whether machine learning can detect Parkinson's disease from UCI voice data with enough reliability to support early screening.

The paper associated with this project reports a lightweight ensemble approach built from ML and CNN components and evaluates it on the UCI Parkinson's dataset. The reported outcome is strong on the available benchmark data, but the repository also includes later experiments that focus on leakage control, class imbalance, and threshold calibration.

## Research Problem

The project addresses this question:

How well can acoustic and speech-derived features be used to distinguish healthy subjects from Parkinson's patients in a small clinical-style dataset, while keeping the model simple enough for screening use?

That problem matters because traditional diagnosis can be delayed, and automatic screening may help prioritize patients for follow-up. The challenge is that the available dataset is small and imbalanced, so model performance can look better or worse depending on how the data is split.

## Research Outcome

The paper included in this repository's references reports a lightweight ensemble approach for early Parkinson's detection using the UCI dataset. According to the paper, the final ensemble achieved 99.47% accuracy and 99.50% F1-score on the reported benchmark setup.

That result is useful as a research outcome, but it should not be read as proof of production readiness. The dataset is small, the benchmark is single-source, and there is no external clinical validation in this repository.

## Current Project Status

This project is good enough to publish publicly as an academic portfolio project, but it should be framed honestly:

- it is a research prototype, not a diagnostic product
- it demonstrates end-to-end data preparation, model training, and evaluation
- it is limited by database size and by the absence of a large external validation set
- the safest claim is that it shows promising screening-oriented modeling on a small dataset

## What Is In The Repo

- [src/experiments/run_experiment.py](src/experiments/run_experiment.py) is the primary entry point containing the complete modern experiment, including group-aware splitting, candidate model selection, threshold calibration, and figure export to the reports directory.
- [src/pipeline/data_preprocessing.py](src/pipeline/data_preprocessing.py) contains the data cleaning, patient ID extraction, and data division logic.
- [tests/test_pipeline.py](tests/test_pipeline.py) contains unit tests validating the model threshold logic and preprocessing pipeline.
- [src/Research_Components/main_new.py](src/Research_Components/main_new.py) contains the legacy standalone modern experiment.
- [src/V004/main.py](src/V004/main.py) contains the legacy simpler cost-sensitive XGBoost baseline.
- [src/V003](src/V003) contains an earlier legacy version of the same experimental line.
- [practice](practice) contains learning and practice scripts.
- [archive/Discarded_Programs](archive/Discarded_Programs) stores older or replaced experiments.
- [docs/WorkPlan.md](docs/WorkPlan.md) keeps the project timeline and references.

## Industry-Style Folder Layout

The repository uses a cleaner top-level layout that is closer to a standard ML project:

- `data/raw` for original source data (e.g. `parkinsons.data`)
- `data/processed` for cleaned and derived datasets (e.g. `Parkinsons_cleaned.csv`, `Parkinsons_status.csv`, `Parkinsons_groups.csv`)
- `models` for saved trained artifacts
- `notebooks` for exploratory analysis and draft experiments
- `reports/figures` for plots and exported images
- `reports/tables` for metric tables and result summaries
- `src/pipeline` for reusable training and evaluation code
- `src/experiments` for runnable experiment entry points
- `src/utils` for shared helpers
- `tests` for automated checks

Legacy folders are still kept for traceability while the project is being organized:

- `src/Research_Components`
- `src/V003`
- `src/V004`
- `src/Main-Sections`

## Data Files

The current scripts use these core files:

- `data/raw/parkinsons.data` (original dataset)
- `data/processed/Parkinsons_cleaned.csv` (features only)
- `data/processed/Parkinsons_status.csv` (labels only)
- `data/processed/Parkinsons_groups.csv` (patient group IDs for group-aware splitting)

The newer pipeline loads data relative to the script location, which makes it less sensitive to the current working directory.

## Method Summary

The main experimental flow is:

1. Load cleaned acoustic features and binary labels.
2. Split by patient group so the same patient does not leak into both train and test.
3. Train a cost-sensitive classifier or an ensemble variant.
4. Tune the classification threshold from validation or out-of-fold probabilities.
5. Evaluate final performance on the held-out test set.
6. Export a confusion matrix and feature-importance chart to `reports/figures/`.

The newer code path in [src/experiments/run_experiment.py](src/experiments/run_experiment.py) uses grouped cross-validation, threshold calibration, and a minimum threshold floor to avoid an overly permissive Parkinson's prediction rule.

## Environment Setup

Install the core Python packages from the requirements file:

```bash
pip install -r requirements.txt
```

## How To Run

1. Run the data preprocessing script to prepare the features:
   ```bash
   python src/pipeline/data_preprocessing.py
   ```

2. Run the main modern experiment:
   ```bash
   python src/experiments/run_experiment.py
   ```

3. Run the automated tests:
   ```bash
   python -m unittest tests/test_pipeline.py
   ```

These scripts print training statistics, threshold selection details, final classification metrics, and save generated figures to `reports/figures/`.

## Outputs

The repository produces or reports:

- a final confusion matrix image in `reports/figures/FINAL_Winning_Matrix.png`
- a feature-importance chart in `reports/figures/Feature_Importance.png`
- accuracy, F1-score, ROC-AUC, sensitivity, specificity, and balanced accuracy
- a written research summary in the project documentation

## Honest Limitation Statement

This project should not be presented as a full clinical success. The main reason is the database size: the sample is too small to support strong generalization claims, and the benchmark is not broad enough to claim production-ready performance.

If you publish this publicly, describe it as an undergraduate research prototype or an academic screening study. Do not describe it as a finished medical diagnostic system.

## Good Next Steps

If you want to strengthen the project further, the next high-value steps are:

1. add a larger and more diverse dataset
2. run external validation on a second cohort
3. remove legacy experiment duplication once the final pipeline is stable
4. export a final results table into `reports/tables`