# Parkinson's Research

This repository contains a student research workflow for early Parkinson's disease detection from acoustic and clinical feature data. The project explores multiple supervised machine learning pipelines, with a focus on handling class imbalance, group-aware splitting, and threshold tuning for medical screening.

The latest experimental pipeline is in [src/Research_Components/main_new.py](src/Research_Components/main_new.py). Earlier baselines and comparison scripts are kept in [src/V003](src/V003) and [src/V004](src/V004), and older discarded work is preserved in [archive/Discarded_Programs](archive/Discarded_Programs).

## Project Goal

The goal is to compare lightweight classifiers for Parkinson's disease screening and evaluate whether the models can separate healthy and Parkinson's cases using the available dataset.

The current implementation emphasizes:

- group-aware splitting to reduce patient leakage between train and test sets
- class-imbalance handling with weighted XGBoost and SMOTE-based variants
- threshold calibration to prioritize medical sensitivity and specificity
- feature-importance reporting for basic model interpretability

## Important Limitation

This project is not a full clinical success case. The main limitation is the database size: the dataset is relatively small, which limits how far the models can generalize and makes performance sensitive to sample composition, class imbalance, and train/test split choice.

In practical terms, the results should be treated as exploratory research rather than a finished diagnostic system. More patient data, stronger external validation, and repeated cross-validation on larger cohorts would be needed before claiming robust real-world performance.

## Repository Structure

- `src/Research_Components/` contains the most complete experimental pipeline, data-cleaning scripts, and generated CSV outputs.
- `src/V003/` and `src/V004/` contain earlier model versions for comparison.
- `src/Main-Sections/` contains helper scripts for preprocessing, scaling, splitting, and model training.
- `datasets/parkinsons/` stores dataset metadata files.
- `docs/WorkPlan.md` records the project timeline and research notes.
- `practice/` contains learning and experimentation scripts for pandas and scikit-learn.
- `archive/Discarded_Programs/` stores older or replaced scripts.
- `assets/` is used for generated figures and supporting material.

## Main Data Files

The main scripts expect these CSV files to be available in the project workspace:

- `Parkinsons_cleaned.csv`
- `Parkinsons_status.csv`
- `Parkinsons_groups.csv`

In the newer pipeline, the data is loaded relative to the script location so it is less dependent on the current working directory.

## Method Summary

The current research pipeline typically follows these steps:

1. Load the cleaned feature matrix and labels.
2. Split the data using patient groups so the same patient does not appear in both training and testing.
3. Train a cost-sensitive classifier, usually XGBoost.
4. Search for a decision threshold that improves the medical trade-off between sensitivity and specificity.
5. Evaluate the final model on the held-out test set.
6. Export a confusion matrix and feature-importance chart.

## Environment Setup

Install the core Python packages used across the project:

```bash
pip install pandas numpy scikit-learn seaborn matplotlib xgboost imbalanced-learn
```

Depending on the script you run, you may also need Jupyter or additional plotting tools.

## How to Run

To run the most complete experimental pipeline:

```bash
python src/Research_Components/main_new.py
```

To run the simpler specificity-focused version:

```bash
python src/V004/main.py
```

These scripts print training statistics, threshold selection details, final classification metrics, and save generated figures into the script directory.

## Outputs

The main pipeline can generate:

- a final confusion matrix image
- a feature-importance chart
- console metrics such as accuracy, F1-score, ROC-AUC, sensitivity, and specificity

## Notes For Readers

This project is useful for comparing screening-oriented machine learning strategies, but it should not be interpreted as a medical device or a production-grade diagnostic tool. The dataset size, feature availability, and class distribution all limit the confidence of the results.

If you extend the project, the most useful next steps are to add more data, test external validation sets, and compare against simpler baseline models before making stronger claims.